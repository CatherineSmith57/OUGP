"""Hidden-state diagnostics for graph/parameter pruning interactions.

This script trains one OUGP model per seed, freezes each epoch's model and
memory snapshot, then probes dense, graph-only, parameter-only, full OUGP, and
no-cross masks without optimizer or memory updates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ougp_matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_case_study import (  # noqa: E402
    VARIANTS,
    evaluate_metric,
    set_seed,
    target_keep_at,
    temperature_at,
)
from ougp.data import load_graph_dataset  # noqa: E402
from ougp.model import OUGPConfig, OUGPGCN  # noqa: E402


MODES = ("dense", "graph_only", "param_only", "ougp_full", "ougp_no_cross")
RAW_EPOCHS_DEFAULT = (0, 10, 25, 50, 100, 150, 199)
PLOT_METRICS = (
    "l2_norm",
    "std",
    "node_cosine_mean",
    "effective_rank",
    "hidden_grad_norm",
    "delta_cosine_gp",
    "interaction_ratio",
)
MODE_STYLES = {
    "dense": {"color": "black", "linestyle": "-", "linewidth": 1.8, "zorder": 3},
    "graph_only": {"color": "tab:blue", "linestyle": "--", "linewidth": 1.8, "zorder": 4},
    "param_only": {"color": "tab:orange", "linestyle": "-.", "linewidth": 1.8, "zorder": 5},
    "ougp_no_cross": {"color": "tab:green", "linestyle": ":", "linewidth": 2.0, "zorder": 6},
    "ougp_full": {"color": "tab:red", "linestyle": "-", "linewidth": 2.8, "zorder": 8},
}


def replace_cfg(cfg: OUGPConfig, **updates) -> OUGPConfig:
    values = asdict(cfg)
    values.update(updates)
    return OUGPConfig(**values)


def tensor_digest(tensor: torch.Tensor) -> str:
    arr = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def model_fingerprint(model: OUGPGCN) -> dict[str, str]:
    return {name: tensor_digest(param) for name, param in model.state_dict().items()}


def assert_fingerprint_equal(before: dict[str, str], after: dict[str, str], context: str) -> None:
    changed = [key for key, value in before.items() if after.get(key) != value]
    if changed:
        raise RuntimeError(f"{context} changed model state unexpectedly: {changed[:8]}")


def build_cfg(args: argparse.Namespace, dataset, seed: int) -> OUGPConfig:
    cfg_kwargs = dict(
        in_dim=dataset.num_features,
        hidden_dim=args.hidden_dim,
        out_dim=dataset.num_classes,
        num_nodes=dataset.num_nodes,
        num_edges=dataset.edge_index.size(1),
        num_gnn_layers=args.num_gnn_layers,
        memory_rank=args.memory_rank,
        graph_target_keep=1.0 - args.graph_sparsity,
        param_target_keep=1.0 - args.param_sparsity,
        graph_gamma=args.graph_gamma,
        param_gamma=args.param_gamma,
        graph_score_scale_decay=args.graph_score_scale_decay,
        graph_score_scale_min=args.graph_score_scale_min,
        graph_score_scale_max=args.graph_score_scale_max,
        graph_correction_clip=args.graph_correction_clip,
        param_score_scale_decay=args.param_score_scale_decay,
        param_score_scale_min=args.param_score_scale_min,
        param_score_scale_max=args.param_score_scale_max,
        param_correction_clip=args.param_correction_clip,
        cross_gamma=args.cross_gamma,
        write_beta=args.write_beta,
        write_lambda=args.write_lambda,
        event_gamma=args.event_gamma,
        event_beta=args.event_beta,
        event_decay=args.event_decay,
        event_top_k=args.event_top_k,
        recall_gamma=args.recall_gamma,
        recall_beta=args.recall_beta,
        recall_decay=args.recall_decay,
        recall_top_k=args.recall_top_k,
        hard_masks=args.hard_eval,
        use_steering_memory=args.use_steering_memory,
        steer_gamma=args.steer_gamma,
        steer_beta=args.steer_beta,
        steer_lambda=args.steer_lambda,
        backbone=args.backbone,
        budget_target=args.budget_target,
        memory_write_mode=args.memory_write_mode,
        graph_memory_granularity=args.graph_memory_granularity,
        graph_memory_layout=args.graph_memory_layout,
        use_graph_full_branch=args.use_graph_full_branch,
        use_graph_grad_branch=args.use_graph_grad_branch,
        use_graph_branch_gates=args.use_graph_branch_gates,
        param_memory_layout=args.param_memory_layout,
        graph_score_init=args.graph_score_init,
        param_score_init=args.param_score_init,
        freeze_pruning_scores=args.freeze_pruning_scores,
        seed=seed,
    )
    cfg_kwargs.update(VARIANTS["ougp"])
    return OUGPConfig(**cfg_kwargs)


def mask_snapshot(model: OUGPGCN, temperature: float) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    original_cfg = model.cfg
    original_last = (
        model.last_graph_score,
        model.last_param_score,
        model.last_graph_mask,
        model.last_param_mask,
    )
    model.eval()
    with torch.no_grad():
        model.cfg = replace_cfg(original_cfg, use_cross=True)
        graph_mask, param_mask, _ = model.masks(temperature)
        graph_mask = graph_mask.detach().clone()
        param_mask = param_mask.detach().clone()

        model.cfg = replace_cfg(original_cfg, use_cross=False)
        no_cross_graph, no_cross_param, _ = model.masks(temperature)
        no_cross_graph = no_cross_graph.detach().clone()
        no_cross_param = no_cross_param.detach().clone()

    model.cfg = original_cfg
    (
        model.last_graph_score,
        model.last_param_score,
        model.last_graph_mask,
        model.last_param_mask,
    ) = original_last
    ones_graph = torch.ones_like(graph_mask)
    ones_param = torch.ones_like(param_mask)
    return {
        "dense": (ones_graph, ones_param),
        "graph_only": (graph_mask, ones_param),
        "param_only": (ones_graph, param_mask),
        "ougp_full": (graph_mask, param_mask),
        "ougp_no_cross": (no_cross_graph, no_cross_param),
    }


def rank_ordinal(x: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(x)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(x.numel(), device=x.device, dtype=torch.float32)
    return ranks


def safe_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    if a.numel() == 0 or b.numel() == 0:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if float(denom.item()) <= 1e-12:
        return 0.0
    return float((a @ b / denom).item())


def sampled_node_cosine(x: torch.Tensor, max_nodes: int) -> float:
    if x.ndim != 2 or x.size(0) <= 1:
        return 0.0
    h = x.detach().float()
    if max_nodes > 0 and h.size(0) > max_nodes:
        ids = torch.linspace(0, h.size(0) - 1, max_nodes, device=h.device).long()
        h = h[ids]
    h = F.normalize(h, dim=1, eps=1e-12)
    sim = h @ h.t()
    n = sim.size(0)
    return float((sim.sum() - sim.diag().sum()).item() / max(1, n * (n - 1)))


def effective_rank(x: torch.Tensor) -> float:
    if x.ndim != 2 or min(x.shape) == 0:
        return 0.0
    h = x.detach().float()
    try:
        s = torch.linalg.svdvals(h)
    except RuntimeError:
        s = torch.linalg.svdvals(h.cpu()).to(h.device)
    p = s / s.sum().clamp_min(1e-12)
    entropy = -(p * torch.log(p.clamp_min(1e-12))).sum()
    return float(torch.exp(entropy).item())


def tensor_stats(x: torch.Tensor, grad: torch.Tensor | None, node_cosine_sample: int) -> dict[str, float]:
    h = x.detach().float()
    flat = h.flatten()
    mean = flat.mean()
    centered = flat - mean
    std = flat.std(unbiased=False)
    q = torch.quantile(flat, torch.tensor([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99], device=flat.device))
    skew = (centered.pow(3).mean() / std.clamp_min(1e-12).pow(3)).item()
    kurt = (centered.pow(4).mean() / std.clamp_min(1e-12).pow(4)).item()
    node_norm = h.norm(dim=1) if h.ndim == 2 else flat.abs()
    channel_norm = h.norm(dim=0) if h.ndim == 2 else flat.abs()
    out = {
        "mean": float(mean.item()),
        "median": float(q[3].item()),
        "std": float(std.item()),
        "variance": float(flat.var(unbiased=False).item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "q01": float(q[0].item()),
        "q05": float(q[1].item()),
        "q25": float(q[2].item()),
        "q75": float(q[4].item()),
        "q95": float(q[5].item()),
        "q99": float(q[6].item()),
        "l1_norm": float(flat.abs().sum().item()),
        "l2_norm": float(flat.norm().item()),
        "zero_ratio": float((flat == 0).float().mean().item()),
        "positive_ratio": float((flat > 0).float().mean().item()),
        "negative_ratio": float((flat < 0).float().mean().item()),
        "skewness": float(skew),
        "kurtosis": float(kurt),
        "node_norm_mean": float(node_norm.mean().item()),
        "node_norm_median": float(node_norm.median().item()),
        "node_norm_std": float(node_norm.std(unbiased=False).item()),
        "channel_norm_mean": float(channel_norm.mean().item()),
        "channel_norm_median": float(channel_norm.median().item()),
        "channel_norm_std": float(channel_norm.std(unbiased=False).item()),
        "node_cosine_mean": sampled_node_cosine(h, node_cosine_sample),
        "effective_rank": effective_rank(h),
    }
    if grad is None:
        out.update({"hidden_grad_norm": 0.0, "channel_utility_mean": 0.0, "channel_utility_std": 0.0})
    else:
        g = grad.detach().float()
        utility = (h * g).abs()
        if utility.ndim == 2:
            utility = utility.mean(dim=0)
        else:
            utility = utility.flatten()
        out.update(
            {
                "hidden_grad_norm": float(g.norm().item()),
                "channel_utility_mean": float(utility.mean().item()),
                "channel_utility_std": float(utility.std(unbiased=False).item()),
            }
        )
    return out


def pair_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    af = a.detach().float().flatten()
    bf = b.detach().float().flatten()
    return {
        "delta_cosine_gp": safe_corr(af, bf),
        "delta_pearson_gp": safe_corr(af, bf),
        "delta_spearman_gp": safe_corr(rank_ordinal(af), rank_ordinal(bf)),
        "delta_g_norm": float(af.norm().item()),
        "delta_p_norm": float(bf.norm().item()),
    }


def run_diagnostic_epoch(
    model: OUGPGCN,
    x: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    task_type: str,
    temperature: float,
    seed: int,
    epoch: int,
    args: argparse.Namespace,
    raw_dir: Path,
) -> tuple[list[dict[str, float | int | str]], dict[str, torch.Tensor]]:
    before = model_fingerprint(model)
    was_training = model.training
    model.eval()
    model.zero_grad(set_to_none=True)
    masks = mask_snapshot(model, temperature)
    rows: list[dict[str, float | int | str]] = []
    tensors_by_mode: dict[str, dict[tuple[int, str], torch.Tensor]] = {}
    raw_payload: dict[str, dict[str, torch.Tensor]] = {}

    for mode in MODES:
        model.zero_grad(set_to_none=True)
        logits, _, hidden_states = model(
            x,
            temperature=temperature,
            return_hidden_states=True,
            fixed_masks=masks[mode],
            retain_hidden_grad=True,
        )
        loss = (
            F.binary_cross_entropy_with_logits(logits[train_mask], y[train_mask])
            if task_type == "multilabel"
            else F.cross_entropy(logits[train_mask], y[train_mask])
        )
        loss.backward()
        tensors_by_mode[mode] = {}
        raw_payload[mode] = {}
        for item in hidden_states:
            layer = int(item["layer"])
            kind = str(item["kind"])
            tensor = item["tensor"]
            if not isinstance(tensor, torch.Tensor):
                continue
            grad = tensor.grad
            tensors_by_mode[mode][(layer, kind)] = tensor.detach()
            if epoch in args.raw_epochs:
                raw_payload[mode][f"layer{layer}_{kind}"] = tensor.detach().cpu()
            row: dict[str, float | int | str] = {
                "seed": seed,
                "epoch": epoch,
                "mode": mode,
                "layer": layer,
                "kind": kind,
                "train_loss_probe": float(loss.detach().item()),
            }
            row.update(tensor_stats(tensor, grad, args.node_cosine_sample))
            rows.append(row)
        model.zero_grad(set_to_none=True)

    keys = sorted(set(tensors_by_mode["dense"]).intersection(*[set(tensors_by_mode[m]) for m in MODES]))
    for layer, kind in keys:
        dense = tensors_by_mode["dense"][(layer, kind)]
        graph = tensors_by_mode["graph_only"][(layer, kind)]
        param = tensors_by_mode["param_only"][(layer, kind)]
        full = tensors_by_mode["ougp_full"][(layer, kind)]
        dg = graph - dense
        dp = param - dense
        residual = full - graph - param + dense
        row = {
            "seed": seed,
            "epoch": epoch,
            "mode": "interaction",
            "layer": layer,
            "kind": kind,
            "residual_fro_norm": float(residual.norm().item()),
            "interaction_ratio": float(residual.norm().item() / (dg.norm().item() + dp.norm().item() + 1e-12)),
        }
        row.update(pair_metrics(dg, dp))
        rows.append(row)

    if epoch in args.raw_epochs:
        raw_path = raw_dir / f"seed{seed:02d}_epoch{epoch:03d}_hidden.pt"
        torch.save(raw_payload, raw_path)

    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()
    after = model_fingerprint(model)
    assert_fingerprint_equal(before, after, f"diagnostic seed={seed} epoch={epoch}")
    return rows, {mode: masks[mode][0].detach().cpu() for mode in MODES}


def train_seed(args: argparse.Namespace, dataset, seed: int, out_dir: Path) -> dict[str, float | int | str]:
    set_seed(seed)
    device = torch.device(args.device)
    x = dataset.x.to(device)
    y = dataset.y.to(device)
    edge_index = dataset.edge_index.to(device)
    train_mask = dataset.train_mask.to(device)
    val_mask = dataset.val_mask.to(device)
    test_mask = dataset.test_mask.to(device)

    model = OUGPGCN(build_cfg(args, dataset, seed), edge_index=edge_index, x=x).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    seed_dir = out_dir / f"seed{seed}"
    raw_dir = seed_dir / "raw_hidden"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str]] = []
    history: list[dict[str, float | int | str]] = []
    best_val = -1.0
    best_test = -1.0
    best_epoch = -1
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        keep_g = target_keep_at(epoch, args.epochs, args.warmup_epochs, 1.0 - args.graph_sparsity)
        keep_p = target_keep_at(epoch, args.epochs, args.warmup_epochs, 1.0 - args.param_sparsity)
        model.cfg = replace_cfg(model.cfg, graph_target_keep=keep_g, param_target_keep=keep_p)
        temperature = temperature_at(epoch, args.epochs, args.temp_start, args.temp_end)

        optimizer.zero_grad(set_to_none=True)
        logits, mask_stats = model(x, temperature=temperature)
        loss = (
            F.binary_cross_entropy_with_logits(logits[train_mask], y[train_mask])
            if dataset.task_type == "multilabel"
            else F.cross_entropy(logits[train_mask], y[train_mask])
        )
        total_loss = loss + args.sparsity_lambda * model.regularization()
        if args.budget_lambda != 0.0:
            total_loss = total_loss + args.budget_lambda * model.resource_regularization()
        total_loss.backward()
        memory_stats = model.write_memories()
        optimizer.step()
        model.zero_grad(set_to_none=True)

        model.eval()
        with torch.no_grad():
            eval_logits, eval_mask_stats = model(x, temperature=max(args.temp_end, temperature))
            val_acc = evaluate_metric(eval_logits, y, val_mask, dataset.task_type)
            test_acc = evaluate_metric(eval_logits, y, test_mask, dataset.task_type)
            train_acc = evaluate_metric(eval_logits, y, train_mask, dataset.task_type)
        if val_acc > best_val:
            best_val = val_acc
            best_test = test_acc
            best_epoch = epoch

        diag_rows, _ = run_diagnostic_epoch(
            model,
            x,
            y,
            train_mask,
            dataset.task_type,
            max(args.temp_end, temperature),
            seed,
            epoch,
            args,
            raw_dir,
        )
        rows.extend(diag_rows)
        history.append(
            {
                "seed": seed,
                "epoch": epoch,
                "loss": float(total_loss.detach().item()),
                "train_acc": train_acc,
                "val_acc": val_acc,
                "test_acc": test_acc,
                "best_val": best_val,
                "best_test": best_test,
                **mask_stats,
                **eval_mask_stats,
                **memory_stats,
            }
        )
        if args.verbose and (epoch % args.log_every == 0 or epoch == args.epochs - 1):
            print(
                f"seed={seed} epoch={epoch:03d} loss={float(total_loss.detach().item()):.4f} "
                f"val={val_acc:.4f} test={test_acc:.4f}",
                flush=True,
            )

    write_csv(seed_dir / f"hidden_diagnostic_seed{seed}.csv", rows)
    write_csv(seed_dir / f"training_history_seed{seed}.csv", history)
    result = {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_acc": best_val,
        "best_test_acc": best_test,
        "final_val_acc": history[-1]["val_acc"],
        "final_test_acc": history[-1]["test_acc"],
        "runtime_sec": time.time() - start_time,
    }
    (seed_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(paths: Iterable[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def aggregate_rows(rows: list[dict[str, str]], out_dir: Path) -> list[dict[str, float | int | str]]:
    group_keys = ("epoch", "mode", "layer", "kind")
    numeric_keys = sorted(
        key
        for key in {k for row in rows for k in row}
        if key not in group_keys and key != "seed" and any(row.get(key, "") not in {"", None} for row in rows)
    )
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in group_keys), []).append(row)
    out: list[dict[str, float | int | str]] = []
    for key, group in sorted(grouped.items(), key=lambda item: (int(item[0][0]), item[0][1], int(item[0][2]), item[0][3])):
        agg: dict[str, float | int | str] = {
            "epoch": int(key[0]),
            "mode": key[1],
            "layer": int(key[2]),
            "kind": key[3],
        }
        for metric in numeric_keys:
            values = []
            for row in group:
                text = row.get(metric, "")
                if text == "":
                    continue
                try:
                    value = float(text)
                except ValueError:
                    continue
                if math.isfinite(value):
                    values.append(value)
            if values:
                arr = np.asarray(values, dtype=np.float64)
                agg[f"{metric}_mean"] = float(arr.mean())
                agg[f"{metric}_std"] = float(arr.std())
                agg[f"{metric}_median"] = float(np.median(arr))
        out.append(agg)
    write_csv(out_dir / "hidden_diagnostic_5seed_aggregate.csv", out)
    return out


def series_from_aggregate(
    aggregate: list[dict[str, float | int | str]],
    metric: str,
    mode: str,
    kind: str = "activation",
    layer: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = [
        row
        for row in aggregate
        if row["mode"] == mode and row["kind"] == kind and f"{metric}_mean" in row and f"{metric}_std" in row
    ]
    if layer is not None:
        selected = [row for row in selected if int(row["layer"]) == layer]
    by_epoch: dict[int, list[tuple[float, float]]] = {}
    for row in selected:
        by_epoch.setdefault(int(row["epoch"]), []).append((float(row[f"{metric}_mean"]), float(row[f"{metric}_std"])))
    epochs, means, stds = [], [], []
    for epoch in sorted(by_epoch):
        vals = np.asarray([v[0] for v in by_epoch[epoch]], dtype=np.float64)
        sds = np.asarray([v[1] for v in by_epoch[epoch]], dtype=np.float64)
        epochs.append(epoch)
        means.append(float(vals.mean()))
        stds.append(float(sds.mean()))
    return np.asarray(epochs), np.asarray(means), np.asarray(stds)


def plot_mode_comparison_by_layer(
    aggregate: list[dict[str, float | int | str]],
    plot_dir: Path,
    metric: str,
    kind: str = "activation",
) -> None:
    modes = ("dense", "graph_only", "param_only", "ougp_no_cross", "ougp_full")
    layers = sorted({int(row["layer"]) for row in aggregate if row["mode"] in modes and row["kind"] == kind})
    for layer in layers:
        plt.figure(figsize=(9, 5))
        plotted = False
        for mode in modes:
            x, y, ystd = series_from_aggregate(aggregate, metric, mode, kind=kind, layer=layer)
            if x.size == 0:
                continue
            plotted = True
            style = MODE_STYLES.get(mode, {})
            plt.plot(x, y, label=mode, **style)
            plt.fill_between(x, y - ystd, y + ystd, alpha=0.12)
        if not plotted:
            plt.close()
            continue
        plt.xlabel("epoch")
        plt.ylabel(metric)
        plt.title(f"{metric} layer {layer} {kind} (mode comparison)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"{metric}_layer{layer}_{kind}_mode_compare.png", dpi=160)
        plt.close()


def plot_interaction_by_layer(
    aggregate: list[dict[str, float | int | str]],
    plot_dir: Path,
    metric: str,
    kind: str = "activation",
) -> None:
    layers = sorted({int(row["layer"]) for row in aggregate if row["mode"] == "interaction" and row["kind"] == kind})
    plt.figure(figsize=(9, 5))
    plotted = False
    for layer in layers:
        x, y, ystd = series_from_aggregate(aggregate, metric, "interaction", kind=kind, layer=layer)
        if x.size == 0:
            continue
        plotted = True
        plt.plot(x, y, label=f"layer{layer}")
        plt.fill_between(x, y - ystd, y + ystd, alpha=0.12)
    if not plotted:
        plt.close()
        return
    plt.xlabel("epoch")
    plt.ylabel(metric)
    plt.title(f"{metric} by layer ({kind})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / f"{metric}_{kind}_layer_compare.png", dpi=160)
    plt.close()


def plot_aggregate(aggregate: list[dict[str, float | int | str]], out_dir: Path) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for metric in PLOT_METRICS:
        plt.figure(figsize=(9, 5))
        modes = ("interaction",) if metric in {"delta_cosine_gp", "interaction_ratio"} else MODES
        kind = "activation"
        for mode in modes:
            x, y, ystd = series_from_aggregate(aggregate, metric, mode, kind=kind)
            if x.size == 0:
                continue
            plt.plot(x, y, label=mode)
            plt.fill_between(x, y - ystd, y + ystd, alpha=0.18)
        plt.xlabel("epoch")
        plt.ylabel(metric)
        plt.title(f"{metric} (5-seed mean +/- std)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"{metric}_5seed.png", dpi=160)
        plt.close()
        if metric in {"delta_cosine_gp", "interaction_ratio"}:
            plot_interaction_by_layer(aggregate, plot_dir, metric)
        else:
            plot_mode_comparison_by_layer(aggregate, plot_dir, metric)


def select_representative(results: list[dict[str, float | int | str]]) -> int:
    vals = np.asarray([float(row["best_val_acc"]) for row in results], dtype=np.float64)
    median = float(np.median(vals))
    idx = int(np.argmin(np.abs(vals - median)))
    return int(results[idx]["seed"])


def plot_representative(seed: int, out_dir: Path) -> None:
    seed_csv = out_dir / f"seed{seed}" / f"hidden_diagnostic_seed{seed}.csv"
    rows = read_csv_rows([seed_csv])
    rep_agg = aggregate_rows(rows, out_dir / f"seed{seed}")
    plot_dir = out_dir / "plots" / f"representative_seed{seed}"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for metric in PLOT_METRICS:
        modes = ("interaction",) if metric in {"delta_cosine_gp", "interaction_ratio"} else MODES
        for zoom in (False, True):
            plt.figure(figsize=(9, 5))
            for mode in modes:
                x, y, _ = series_from_aggregate(rep_agg, metric, mode)
                if x.size == 0:
                    continue
                if zoom:
                    keep = x >= max(0, int(x.max()) - 19)
                    x, y = x[keep], y[keep]
                plt.plot(x, y, label=mode)
            plt.xlabel("epoch")
            plt.ylabel(metric)
            suffix = "last20" if zoom else "full"
            plt.title(f"{metric} representative seed {seed} ({suffix})")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_dir / f"{metric}_{suffix}.png", dpi=160)
            plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="cora", choices=["cora"])
    parser.add_argument("--data-root", default="data/raw/planetoid")
    parser.add_argument("--out-dir", default="experiments/exp063_cora_hidden_state_diagnostic")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--backbone", choices=["gcn"], default="gcn")
    parser.add_argument("--num-gnn-layers", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--memory-rank", type=int, default=8)
    parser.add_argument("--graph-sparsity", type=float, default=0.30)
    parser.add_argument("--param-sparsity", type=float, default=0.30)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--sparsity-lambda", type=float, default=0.08)
    parser.add_argument("--budget-lambda", type=float, default=0.0)
    parser.add_argument("--budget-target", type=float, default=0.70)
    parser.add_argument("--graph-gamma", type=float, default=0.35)
    parser.add_argument("--param-gamma", type=float, default=0.35)
    parser.add_argument("--graph-score-scale-decay", type=float, default=0.95)
    parser.add_argument("--graph-score-scale-min", type=float, default=0.02)
    parser.add_argument("--graph-score-scale-max", type=float, default=0.50)
    parser.add_argument("--graph-correction-clip", type=float, default=2.0)
    parser.add_argument("--param-score-scale-decay", type=float, default=0.95)
    parser.add_argument("--param-score-scale-min", type=float, default=0.02)
    parser.add_argument("--param-score-scale-max", type=float, default=0.50)
    parser.add_argument("--param-correction-clip", type=float, default=2.0)
    parser.add_argument("--cross-gamma", type=float, default=0.20)
    parser.add_argument("--write-beta", type=float, default=0.12)
    parser.add_argument("--write-lambda", type=float, default=0.98)
    parser.add_argument("--event-gamma", type=float, default=0.0)
    parser.add_argument("--event-beta", type=float, default=0.10)
    parser.add_argument("--event-decay", type=float, default=0.95)
    parser.add_argument("--event-top-k", type=int, default=2000)
    parser.add_argument("--recall-gamma", type=float, default=0.0)
    parser.add_argument("--recall-beta", type=float, default=0.10)
    parser.add_argument("--recall-decay", type=float, default=0.95)
    parser.add_argument("--recall-top-k", type=int, default=2000)
    parser.add_argument("--use-steering-memory", action="store_true")
    parser.add_argument("--steer-gamma", type=float, default=0.0)
    parser.add_argument("--steer-beta", type=float, default=0.10)
    parser.add_argument("--steer-lambda", type=float, default=0.95)
    parser.add_argument("--memory-write-mode", choices=["residual", "feature", "none"], default="residual")
    parser.add_argument("--graph-memory-granularity", choices=["edge", "subgraph"], default="edge")
    parser.add_argument("--graph-memory-layout", choices=["single", "multi"], default="multi")
    parser.add_argument("--param-memory-layout", choices=["single", "multi"], default="multi")
    parser.add_argument("--use-graph-full-branch", action="store_true", default=True)
    parser.add_argument("--use-graph-grad-branch", action="store_true", default=True)
    parser.add_argument("--use-graph-branch-gates", action="store_true", default=True)
    parser.add_argument("--graph-score-init", choices=["constant", "random", "degree", "similarity", "topofeat"], default="topofeat")
    parser.add_argument("--param-score-init", choices=["constant", "random", "magnitude"], default="magnitude")
    parser.add_argument("--freeze-pruning-scores", action="store_true")
    parser.add_argument("--temp-start", type=float, default=2.0)
    parser.add_argument("--temp-end", type=float, default=0.5)
    parser.add_argument("--hard-eval", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--raw-epochs", nargs="+", type=int, default=list(RAW_EPOCHS_DEFAULT))
    parser.add_argument("--node-cosine-sample", type=int, default=512)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "command.txt").write_text(
        " ".join(shlex.quote(part) for part in [sys.executable, "scripts/run_hidden_state_diagnostic.py", *sys.argv[1:]])
        + "\n",
        encoding="utf-8",
    )
    dataset = load_graph_dataset(args.data_root, args.dataset)
    results = [train_seed(args, dataset, seed, out_dir) for seed in args.seeds]
    write_csv(out_dir / "seed_results.csv", results)
    all_rows = read_csv_rows(out_dir.glob("seed*/hidden_diagnostic_seed*.csv"))
    aggregate = aggregate_rows(all_rows, out_dir)
    plot_aggregate(aggregate, out_dir)
    rep_seed = select_representative(results)
    plot_representative(rep_seed, out_dir)
    manifest = {
        "args": vars(args),
        "results": results,
        "representative_seed": rep_seed,
        "outputs": {
            "seed_csv_pattern": str(out_dir / "seed*/hidden_diagnostic_seed*.csv"),
            "aggregate_csv": str(out_dir / "hidden_diagnostic_5seed_aggregate.csv"),
            "plots": str(out_dir / "plots"),
            "raw_hidden_pattern": str(out_dir / "seed*/raw_hidden/*.pt"),
        },
        "diagnostic_safety": "Each diagnostic epoch checks the model state_dict fingerprint before/after probing.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote hidden-state diagnostic outputs to {out_dir}")
    print(f"Representative seed: {rep_seed}")


if __name__ == "__main__":
    main()
