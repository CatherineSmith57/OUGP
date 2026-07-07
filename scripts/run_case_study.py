"""Run the first OUGP citation-network case study.

The default is intentionally small enough for CPU sanity runs:

    PYTHONPATH=src python scripts/run_case_study.py --dataset cora --epochs 80 --seeds 0

It saves per-run JSON plus aggregate CSV/Markdown files under
`results/ougp_case_study/`.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shlex
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from ougp.data import CitationGraph, load_graph_dataset
from ougp.model import OUGPConfig, OUGPGCN
from ougp.trace import PruningTraceRecorder


BACKBONES = ("gcn", "sage", "gat", "deepgcn")
GRAPH_SCORE_INITS = ("constant", "random", "degree", "similarity")
PARAM_SCORE_INITS = ("constant", "random", "magnitude")

VARIANTS = {
    "dense": dict(use_graph_pruning=False, use_param_pruning=False, use_memory=False, use_cross=False),
    "graph_only": dict(use_graph_pruning=True, use_param_pruning=False, use_memory=True, use_cross=False),
    "param_only": dict(use_graph_pruning=False, use_param_pruning=True, use_memory=True, use_cross=False),
    "dual_static": dict(use_graph_pruning=True, use_param_pruning=True, use_memory=False, use_cross=False),
    "random_static": dict(
        use_graph_pruning=True,
        use_param_pruning=True,
        use_memory=False,
        use_cross=False,
        graph_score_init="random",
        param_score_init="random",
        freeze_pruning_scores=True,
    ),
    "degree_magnitude_static": dict(
        use_graph_pruning=True,
        use_param_pruning=True,
        use_memory=False,
        use_cross=False,
        graph_score_init="degree",
        param_score_init="magnitude",
        freeze_pruning_scores=True,
    ),
    "similarity_magnitude_static": dict(
        use_graph_pruning=True,
        use_param_pruning=True,
        use_memory=False,
        use_cross=False,
        graph_score_init="similarity",
        param_score_init="magnitude",
        freeze_pruning_scores=True,
    ),
    "ougp_no_cross": dict(use_graph_pruning=True, use_param_pruning=True, use_memory=True, use_cross=False),
    "ougp": dict(use_graph_pruning=True, use_param_pruning=True, use_memory=True, use_cross=True),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def accuracy(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred[mask] == y[mask]).float().mean().item())


@torch.no_grad()
def evaluate_metric(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, task_type: str) -> float:
    if task_type == "multilabel":
        y_true = y[mask].detach().cpu().numpy()
        y_score = torch.sigmoid(logits[mask]).detach().cpu().numpy()
        return float(roc_auc_score(y_true, y_score, average="macro"))
    return accuracy(logits, y, mask)


def temperature_at(epoch: int, epochs: int, start: float, end: float) -> float:
    if epochs <= 1:
        return end
    ratio = epoch / (epochs - 1)
    return start * (end / start) ** ratio


def target_keep_at(epoch: int, epochs: int, warmup_epochs: int, target_keep: float) -> float:
    if epoch < warmup_epochs:
        return 1.0
    span = max(1, epochs - warmup_epochs)
    ratio = min(1.0, (epoch - warmup_epochs + 1) / span)
    return 1.0 - ratio * (1.0 - target_keep)


@torch.no_grad()
def recall_diagnostics(
    model: OUGPGCN,
    x: torch.Tensor,
    y: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    task_type: str,
    temperature: float,
    effective_eps: float,
) -> dict[str, float]:
    """Compare evaluation masks with recall disabled vs. enabled."""

    original_cfg = model.cfg
    original_recall_gamma = float(original_cfg.recall_gamma)

    model.cfg = OUGPConfig(**{**asdict(original_cfg), "recall_gamma": 0.0})
    logits_before, before_mask_stats = model(x, temperature=temperature)
    before_param_mask = (
        model.last_param_mask.detach().float().clone()
        if model.last_param_mask is not None
        else torch.ones(original_cfg.hidden_dim, device=x.device)
    )
    val_before = evaluate_metric(logits_before, y, val_mask, task_type)
    test_before = evaluate_metric(logits_before, y, test_mask, task_type)

    model.cfg = original_cfg
    logits_after, after_mask_stats = model(x, temperature=temperature)
    after_param_mask = (
        model.last_param_mask.detach().float().clone()
        if model.last_param_mask is not None
        else torch.ones(original_cfg.hidden_dim, device=x.device)
    )
    val_after = evaluate_metric(logits_after, y, val_mask, task_type)
    test_after = evaluate_metric(logits_after, y, test_mask, task_type)

    recall_bias = model.param_memory.recall_bias.detach().float().to(after_param_mask.device)
    if recall_bias.numel() != after_param_mask.numel() or original_recall_gamma == 0.0:
        recalled = torch.zeros_like(after_param_mask, dtype=torch.bool)
    else:
        recalled = recall_bias > 0
    effective_recovered = recalled & (after_param_mask > before_param_mask + effective_eps)

    return {
        "recall_diag_param_sparsity": float(1.0 - after_mask_stats["param_keep"]),
        "recall_diag_dropped_parameters": float((1.0 - after_param_mask).sum().item()),
        "recall_diag_hard_dropped_parameters": float((after_param_mask < 0.5).sum().item()),
        "recall_diag_recalled_parameters": float(recalled.sum().item()),
        "recall_diag_effective_recovered_parameters": float(effective_recovered.sum().item()),
        "recall_diag_val_acc_before_recall": val_before,
        "recall_diag_val_acc_after_recall": val_after,
        "recall_diag_test_acc_before_recall": test_before,
        "recall_diag_test_acc_after_recall": test_after,
        "recall_diag_param_keep_before_recall": float(before_mask_stats["param_keep"]),
        "recall_diag_param_keep_after_recall": float(after_mask_stats["param_keep"]),
    }


def run_one(args: argparse.Namespace, dataset, variant: str, seed: int, out_dir: Path) -> dict[str, float | int | str]:
    set_seed(seed)
    device = torch.device(args.device)
    data = dataset
    x = data.x.to(device)
    y = data.y.to(device)
    edge_index = data.edge_index.to(device)
    train_mask = data.train_mask.to(device)
    val_mask = data.val_mask.to(device)
    test_mask = data.test_mask.to(device)

    cfg_kwargs = dict(
        in_dim=data.num_features,
        hidden_dim=args.hidden_dim,
        out_dim=data.num_classes,
        num_nodes=data.num_nodes,
        num_edges=edge_index.size(1),
        num_gnn_layers=args.num_gnn_layers,
        memory_rank=args.memory_rank,
        graph_target_keep=1.0 - args.graph_sparsity,
        param_target_keep=1.0 - args.param_sparsity,
        graph_gamma=args.graph_gamma,
        param_gamma=args.param_gamma,
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
        use_steering_memory=args.use_steering_memory,
        steer_gamma=args.steer_gamma,
        steer_beta=args.steer_beta,
        steer_lambda=args.steer_lambda,
        hard_masks=args.hard_eval,
        backbone=args.backbone,
        budget_target=args.budget_target,
        memory_write_mode=args.memory_write_mode,
        graph_score_init=args.graph_score_init,
        param_score_init=args.param_score_init,
        freeze_pruning_scores=args.freeze_pruning_scores,
        seed=seed,
    )
    cfg_kwargs.update(VARIANTS[variant])
    model = OUGPGCN(OUGPConfig(**cfg_kwargs), edge_index=edge_index, x=x).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    trace_recorder = (
        PruningTraceRecorder(out_dir, args.dataset, variant, seed)
        if args.trace_pruning and VARIANTS[variant]["use_graph_pruning"]
        else None
    )

    best_val = -1.0
    best_test = -1.0
    best_epoch = -1
    history = []
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        keep_g = target_keep_at(epoch, args.epochs, args.warmup_epochs, 1.0 - args.graph_sparsity)
        keep_w = target_keep_at(epoch, args.epochs, args.warmup_epochs, 1.0 - args.param_sparsity)
        model.cfg = OUGPConfig(**{**asdict(model.cfg), "graph_target_keep": keep_g, "param_target_keep": keep_w})
        temperature = temperature_at(epoch, args.epochs, args.temp_start, args.temp_end)

        optimizer.zero_grad(set_to_none=True)
        logits, mask_stats = model(x, temperature=temperature)
        if data.task_type == "multilabel":
            task_loss = F.binary_cross_entropy_with_logits(logits[train_mask], y[train_mask])
        else:
            task_loss = F.cross_entropy(logits[train_mask], y[train_mask])
        loss = task_loss + args.sparsity_lambda * model.regularization()
        if args.budget_lambda != 0.0:
            loss = loss + args.budget_lambda * model.resource_regularization()
        loss.backward()
        memory_stats = model.write_memories()
        trace_stats = {}
        if trace_recorder is not None and epoch % args.trace_every == 0:
            graph_events = model.graph_trace_snapshot(
                dataset=args.dataset,
                variant=variant,
                seed=seed,
                epoch=epoch,
                top_k=args.trace_top_k,
            )
            trace_recorder.record_graph_events(graph_events)
            trace_stats["trace_graph_events"] = len(graph_events)
        optimizer.step()
        churn_stats = model.churn()

        model.eval()
        with torch.no_grad():
            eval_logits, eval_mask_stats = model(x, temperature=max(args.temp_end, temperature))
            eval_resource_stats = model.resource_stats()
            val_acc = evaluate_metric(eval_logits, y, val_mask, data.task_type)
            test_acc = evaluate_metric(eval_logits, y, test_mask, data.task_type)
            train_acc = evaluate_metric(eval_logits, y, train_mask, data.task_type)
            if val_acc > best_val:
                best_val = val_acc
                best_test = test_acc
                best_epoch = epoch

        diagnostic_stats = {}
        should_log_epoch = args.verbose and (epoch % args.log_every == 0 or epoch == args.epochs - 1)
        if args.recall_diagnostics and should_log_epoch:
            model.eval()
            diagnostic_stats = recall_diagnostics(
                model,
                x,
                y,
                val_mask,
                test_mask,
                data.task_type,
                max(args.temp_end, temperature),
                args.recall_effect_eps,
            )
            val_acc = diagnostic_stats["recall_diag_val_acc_after_recall"]
            test_acc = diagnostic_stats["recall_diag_test_acc_after_recall"]

        row = {
            "epoch": epoch,
            "loss": float(loss.item()),
            "task_loss": float(task_loss.item()),
            "train_acc": train_acc,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "best_val": best_val,
            "best_test": best_test,
            "temperature": temperature,
            **mask_stats,
            **eval_mask_stats,
            **eval_resource_stats,
            **churn_stats,
            **memory_stats,
            **trace_stats,
            **diagnostic_stats,
        }
        history.append(row)
        if should_log_epoch:
            message = (
                f"{variant}/seed{seed} epoch={epoch:03d} "
                f"loss={loss.item():.4f} val={val_acc:.4f} test={test_acc:.4f} "
                f"g_keep={eval_mask_stats['graph_keep']:.3f} p_keep={eval_mask_stats['param_keep']:.3f}"
            )
            if args.score_diagnostics:
                message += (
                    f" param_logits_mean={eval_mask_stats['param_logits_mean']:.6f}"
                    f" param_logits_std={eval_mask_stats['param_logits_std']:.6f}"
                    f" param_memory_correction_mean={eval_mask_stats['param_memory_correction_mean']:.6f}"
                    f" param_memory_correction_std={eval_mask_stats['param_memory_correction_std']:.6f}"
                    f" param_memory_unit_correction_std={eval_mask_stats['param_memory_unit_correction_std']:.6f}"
                    f" param_memory_raw_correction_mean={eval_mask_stats['param_memory_raw_correction_mean']:.6f}"
                    f" param_memory_raw_correction_std={eval_mask_stats['param_memory_raw_correction_std']:.6f}"
                    f" recall_correction_mean={eval_mask_stats['recall_correction_mean']:.6f}"
                    f" recall_correction_std={eval_mask_stats['recall_correction_std']:.6f}"
                    f" recall_unit_correction_std={eval_mask_stats['recall_unit_correction_std']:.6f}"
                    f" param_score_scale={eval_mask_stats['param_score_scale']:.6f}"
                    f" param_memory_score_delta_std={eval_mask_stats['param_memory_score_delta_std']:.6f}"
                    f" recall_score_delta_std={eval_mask_stats['recall_score_delta_std']:.6f}"
                )
            if diagnostic_stats:
                message += (
                    f" param_sparsity={diagnostic_stats['recall_diag_param_sparsity']:.3f}"
                    f" dropped_params={diagnostic_stats['recall_diag_dropped_parameters']:.1f}"
                    f" recalled_params={diagnostic_stats['recall_diag_recalled_parameters']:.0f}"
                    f" effective_recovered={diagnostic_stats['recall_diag_effective_recovered_parameters']:.0f}"
                    f" param_churn={churn_stats.get('param_churn', 0.0):.4f}"
                    f" acc_before_recall={diagnostic_stats['recall_diag_val_acc_before_recall']:.4f}"
                    f" acc_after_recall={diagnostic_stats['recall_diag_val_acc_after_recall']:.4f}"
                )
            print(message, flush=True)

    final = history[-1]
    result = {
        "dataset": args.dataset,
        "variant": variant,
        "seed": seed,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_val_acc": best_val,
        "best_test_acc": best_test,
        "final_train_acc": final["train_acc"],
        "final_val_acc": final["val_acc"],
        "final_test_acc": final["test_acc"],
        "graph_keep": final["graph_keep"],
        "graph_sparsity": 1.0 - final["graph_keep"],
        "param_keep": final["param_keep"],
        "param_sparsity": 1.0 - final["param_keep"],
        "graph_churn": final.get("graph_churn", 0.0),
        "param_churn": final.get("param_churn", 0.0),
        "message_cost_ratio": final.get("message_cost_ratio", 1.0),
        "message_cost_reduction": final.get("message_cost_reduction", 0.0),
        "parameter_cost_ratio": final.get("parameter_cost_ratio", 1.0),
        "parameter_cost_reduction": final.get("parameter_cost_reduction", 0.0),
        "dense_message_cost": final.get("dense_message_cost", 0.0),
        "effective_message_cost": final.get("effective_message_cost", 0.0),
        "dense_parameter_count": final.get("dense_parameter_count", 0.0),
        "effective_parameter_count": final.get("effective_parameter_count", 0.0),
        "memory_state_items": final.get("memory_state_items", 0.0),
        "memory_overhead_vs_dense_params": final.get("memory_overhead_vs_dense_params", 0.0),
        "graph_memory_state_norm": float(model.graph_memory.state.norm().item()),
        "param_memory_state_norm": float(model.param_memory.state.norm().item()),
        "graph_recall_bias_norm": float(model.graph_memory.recall_bias.norm().item()),
        "param_recall_bias_norm": float(model.param_memory.recall_bias.norm().item()),
        "steering_memory_state_norm": float(model.steering_memory.state.norm().item()),
        "runtime_sec": time.time() - start_time,
        "num_nodes": data.num_nodes,
        "num_edges": edge_index.size(1),
        "original_num_nodes": int(getattr(args, "original_num_nodes", data.num_nodes)),
        "original_num_edges": int(getattr(args, "original_num_edges", edge_index.size(1))),
        "node_sample_size": int(args.node_sample_size),
        "node_sample_mode": args.node_sample_mode,
        "edge_sample_size": int(args.edge_sample_size),
        "hidden_dim": args.hidden_dim,
        "num_gnn_layers": args.num_gnn_layers,
        "memory_rank": args.memory_rank,
        "memory_write_mode": args.memory_write_mode,
        "graph_score_init": cfg_kwargs["graph_score_init"],
        "param_score_init": cfg_kwargs["param_score_init"],
        "freeze_pruning_scores": bool(cfg_kwargs["freeze_pruning_scores"]),
        "backbone": args.backbone,
        "param_memory_slots": int(getattr(model.param_memory, "num_channels", 1)),
        "task_type": data.task_type,
        "metric_name": data.metric_name,
    }

    run_path = out_dir / f"{args.dataset}_{variant}_seed{seed}.json"
    run_path.write_text(json.dumps({"result": result, "history": history, "config": cfg_kwargs}, indent=2))
    return result


def aggregate(results: list[dict[str, float | int | str]], out_dir: Path, dataset: str) -> None:
    csv_path = out_dir / f"{dataset}_summary.csv"
    fieldnames = list(results[0].keys())
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    by_variant: dict[str, list[dict[str, float | int | str]]] = {}
    for row in results:
        by_variant.setdefault(str(row["variant"]), []).append(row)

    lines = [
        "# OUGP Case Study Summary",
        "",
        f"Dataset: `{dataset}`",
        f"Backbone: `{results[0].get('backbone', 'gcn')}`",
        "",
        "| Variant | Best Test Acc | Graph Sparsity | Param Sparsity | Message Cost Red. | Param Cost Red. | Graph Churn | Param Churn |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant, rows in sorted(by_variant.items()):
        def mean(key: str) -> float:
            return float(np.mean([float(r[key]) for r in rows]))

        def std(key: str) -> float:
            return float(np.std([float(r[key]) for r in rows]))

        lines.append(
            f"| {variant} | {mean('best_test_acc'):.4f} +/- {std('best_test_acc'):.4f} | "
            f"{mean('graph_sparsity'):.3f} | {mean('param_sparsity'):.3f} | "
            f"{mean('message_cost_reduction'):.3f} | {mean('parameter_cost_reduction'):.3f} | "
            f"{mean('graph_churn'):.3f} | {mean('param_churn'):.3f} |"
        )
    (out_dir / f"{dataset}_summary.md").write_text("\n".join(lines) + "\n")


def write_run_manifest(args: argparse.Namespace, results: list[dict[str, float | int | str]], out_dir: Path) -> None:
    command = " ".join(shlex.quote(part) for part in [sys.executable, "scripts/run_case_study.py", *sys.argv[1:]])
    (out_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    manifest = {
        "command": command,
        "args": vars(args),
        "result_files": {
            "summary_csv": str(out_dir / f"{args.dataset}_summary.csv"),
            "summary_md": str(out_dir / f"{args.dataset}_summary.md"),
            "run_json_pattern": str(out_dir / f"{args.dataset}_*_seed*.json"),
        },
        "final_metrics": results,
        "log_note": "Per-epoch metrics are stored in each run JSON under the `history` key.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def maybe_sample_edges(dataset: CitationGraph, sample_size: int, seed: int) -> CitationGraph:
    if sample_size <= 0 or sample_size >= dataset.edge_index.size(1):
        return dataset
    generator = torch.Generator()
    generator.manual_seed(seed)
    edge_ids = torch.randperm(dataset.edge_index.size(1), generator=generator)[:sample_size]
    sampled_edge_index = dataset.edge_index[:, edge_ids].contiguous()
    return replace(dataset, edge_index=sampled_edge_index)


def _sample_from_mask(mask: torch.Tensor, max_items: int, generator: torch.Generator) -> torch.Tensor:
    indices = mask.cpu().nonzero(as_tuple=False).flatten()
    if max_items <= 0 or indices.numel() == 0:
        return indices[:0]
    order = torch.randperm(indices.numel(), generator=generator)[: min(max_items, indices.numel())]
    return indices[order]


def _split_seed_nodes(dataset: CitationGraph, sample_size: int, generator: torch.Generator) -> torch.Tensor:
    forced_budget = max(1, min(256, sample_size // 10))
    forced_parts = [
        _sample_from_mask(dataset.train_mask, forced_budget, generator),
        _sample_from_mask(dataset.val_mask, forced_budget, generator),
        _sample_from_mask(dataset.test_mask, forced_budget, generator),
    ]
    forced = torch.unique(torch.cat(forced_parts)) if forced_parts else torch.empty(0, dtype=torch.long)
    if forced.numel() > sample_size:
        forced = forced[torch.randperm(forced.numel(), generator=generator)[:sample_size]]
    return forced


def _fill_random_nodes(keep: torch.Tensor, remaining: int, generator: torch.Generator) -> int:
    if remaining <= 0:
        return 0
    candidates = torch.randperm(keep.numel(), generator=generator)
    added = 0
    for node in candidates.tolist():
        if added >= remaining:
            break
        if not bool(keep[node].item()):
            keep[node] = True
            added += 1
    return added


def _build_undirected_adjacency(edge_index: torch.Tensor, num_nodes: int) -> list[list[int]]:
    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    row, col = edge_index.cpu()
    for src, dst in zip(row.tolist(), col.tolist()):
        if 0 <= src < num_nodes and 0 <= dst < num_nodes:
            adjacency[src].append(dst)
            if src != dst:
                adjacency[dst].append(src)
    return adjacency


def _fill_frontier_nodes(dataset: CitationGraph, keep: torch.Tensor, remaining: int, generator: torch.Generator) -> int:
    if remaining <= 0:
        return 0
    adjacency = _build_undirected_adjacency(dataset.edge_index, dataset.num_nodes)
    frontier = keep.nonzero(as_tuple=False).flatten().tolist()
    added = 0
    while added < remaining and frontier:
        candidate_seen: dict[int, None] = {}
        for node in frontier:
            for neighbor in adjacency[node]:
                if not bool(keep[neighbor].item()):
                    candidate_seen.setdefault(neighbor, None)
        if not candidate_seen:
            break
        candidates = torch.tensor(list(candidate_seen.keys()), dtype=torch.long)
        order = torch.randperm(candidates.numel(), generator=generator)
        selected = candidates[order[: min(remaining - added, candidates.numel())]]
        keep[selected] = True
        added += int(selected.numel())
        frontier = selected.tolist()
    return added


def maybe_sample_nodes(dataset: CitationGraph, sample_size: int, seed: int, mode: str = "random") -> CitationGraph:
    """Build a node-sampled induced subgraph while preserving split coverage.

    This is a static subgraph smoke path for large-graph validation. It is not
    full mini-batch neighbor sampling, but it keeps node features, labels, split
    masks, and edge endpoints consistent after remapping to local node ids.
    """

    if sample_size <= 0 or sample_size >= dataset.num_nodes:
        return dataset
    if sample_size < 3:
        raise ValueError("--node-sample-size must be at least 3 when enabled.")
    if mode not in {"random", "frontier"}:
        raise ValueError("--node-sample-mode must be one of: random, frontier.")

    generator = torch.Generator()
    generator.manual_seed(seed)

    forced = _split_seed_nodes(dataset, sample_size, generator)
    keep = torch.zeros(dataset.num_nodes, dtype=torch.bool)
    keep[forced] = True
    remaining = sample_size - int(keep.sum().item())
    if remaining > 0:
        if mode == "frontier":
            remaining -= _fill_frontier_nodes(dataset, keep, remaining, generator)
        if remaining > 0:
            _fill_random_nodes(keep, remaining, generator)

    node_ids = keep.nonzero(as_tuple=False).flatten()
    old_to_new = torch.full((dataset.num_nodes,), -1, dtype=torch.long)
    old_to_new[node_ids] = torch.arange(node_ids.numel(), dtype=torch.long)

    row, col = dataset.edge_index.cpu()
    edge_keep = keep[row] & keep[col]
    sampled_edge_index = old_to_new[dataset.edge_index.cpu()[:, edge_keep]].contiguous()

    sampled = replace(
        dataset,
        x=dataset.x[node_ids].contiguous(),
        y=dataset.y[node_ids].contiguous(),
        edge_index=sampled_edge_index,
        train_mask=dataset.train_mask[node_ids].contiguous(),
        val_mask=dataset.val_mask[node_ids].contiguous(),
        test_mask=dataset.test_mask[node_ids].contiguous(),
    )
    if sampled.train_mask.sum() == 0 or sampled.val_mask.sum() == 0 or sampled.test_mask.sum() == 0:
        raise RuntimeError(
            "Node sampling produced an empty train/val/test split. "
            "Increase --node-sample-size or use a different --node-sample-seed."
        )
    if sampled.edge_index.numel() == 0:
        raise RuntimeError(
            "Node sampling produced a subgraph with no edges. "
            "Increase --node-sample-size or use a different --node-sample-seed."
        )
    return sampled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="cora",
        choices=["cora", "citeseer", "pubmed", "photo", "ogbn-arxiv", "ogbn-products", "ogbn-proteins"],
    )
    parser.add_argument("--data-root", default="data/raw/planetoid")
    parser.add_argument("--out-dir", default="experiments/manual_case_study")
    parser.add_argument("--node-sample-size", type=int, default=0)
    parser.add_argument("--node-sample-seed", type=int, default=0)
    parser.add_argument("--node-sample-mode", choices=["random", "frontier"], default="random")
    parser.add_argument("--edge-sample-size", type=int, default=0)
    parser.add_argument("--edge-sample-seed", type=int, default=0)
    parser.add_argument("--variants", nargs="+", default=["dense", "dual_static", "ougp_no_cross", "ougp"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--backbone", choices=BACKBONES, default="gcn")
    parser.add_argument("--num-gnn-layers", type=int, default=2)
    parser.add_argument("--memory-rank", type=int, default=8)
    parser.add_argument("--memory-write-mode", choices=["residual", "feature", "none"], default="residual")
    parser.add_argument("--graph-score-init", choices=GRAPH_SCORE_INITS, default="constant")
    parser.add_argument("--param-score-init", choices=PARAM_SCORE_INITS, default="constant")
    parser.add_argument("--freeze-pruning-scores", action="store_true")
    parser.add_argument("--graph-sparsity", type=float, default=0.30)
    parser.add_argument("--param-sparsity", type=float, default=0.30)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--sparsity-lambda", type=float, default=0.08)
    parser.add_argument("--budget-lambda", type=float, default=0.0)
    parser.add_argument("--budget-target", type=float, default=0.70)
    parser.add_argument("--graph-gamma", type=float, default=0.35)
    parser.add_argument("--param-gamma", type=float, default=0.35)
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
    parser.add_argument("--temp-start", type=float, default=2.0)
    parser.add_argument("--temp-end", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-gpus", type=int, default=4)
    parser.add_argument("--hard-eval", action="store_true")
    parser.add_argument("--trace-pruning", action="store_true")
    parser.add_argument("--trace-every", type=int, default=10)
    parser.add_argument("--trace-top-k", type=int, default=200)
    parser.add_argument("--recall-diagnostics", action="store_true")
    parser.add_argument("--recall-effect-eps", type=float, default=1e-4)
    parser.add_argument("--score-diagnostics", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args()


def validate_gpu_budget(args: argparse.Namespace) -> None:
    if args.trace_every <= 0:
        raise ValueError("--trace-every must be a positive integer.")
    if args.trace_top_k < 0:
        raise ValueError("--trace-top-k must be non-negative.")
    if args.event_top_k < 0:
        raise ValueError("--event-top-k must be non-negative.")
    if args.recall_top_k < 0:
        raise ValueError("--recall-top-k must be non-negative.")
    if args.edge_sample_size < 0:
        raise ValueError("--edge-sample-size must be non-negative.")
    if args.node_sample_size < 0:
        raise ValueError("--node-sample-size must be non-negative.")
    if args.budget_lambda < 0:
        raise ValueError("--budget-lambda must be non-negative.")
    if not 0.0 < args.budget_target <= 1.0:
        raise ValueError("--budget-target must be in (0, 1].")
    if args.num_gnn_layers < 2:
        raise ValueError("--num-gnn-layers must be at least 2.")
    if args.backbone != "deepgcn" and args.num_gnn_layers != 2:
        raise ValueError("--num-gnn-layers is currently only configurable with --backbone deepgcn.")
    if args.backbone == "deepgcn" and args.num_gnn_layers < 3:
        raise ValueError("--backbone deepgcn requires --num-gnn-layers >= 3.")
    if args.max_gpus > 4:
        raise ValueError("This workspace policy allows at most 4 GPUs per run.")
    if args.device.startswith("cuda"):
        visible = torch.cuda.device_count()
        if visible > args.max_gpus:
            raise RuntimeError(
                f"{visible} CUDA devices are visible, but this run is capped at {args.max_gpus}. "
                f"Set CUDA_VISIBLE_DEVICES to at most {args.max_gpus} devices."
            )


def main() -> None:
    args = parse_args()
    validate_gpu_budget(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_graph_dataset(args.data_root, args.dataset)
    args.original_num_nodes = int(dataset.num_nodes)
    args.original_num_edges = int(dataset.edge_index.size(1))
    dataset = maybe_sample_nodes(dataset, args.node_sample_size, args.node_sample_seed, args.node_sample_mode)
    dataset = maybe_sample_edges(dataset, args.edge_sample_size, args.edge_sample_seed)
    results = []
    for variant in args.variants:
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}; choose from {sorted(VARIANTS)}")
        for seed in args.seeds:
            results.append(run_one(args, dataset, variant, seed, out_dir))
    aggregate(results, out_dir, args.dataset)
    write_run_manifest(args, results, out_dir)
    print(f"Wrote {len(results)} runs to {out_dir}")
    print((out_dir / f"{args.dataset}_summary.md").read_text())


if __name__ == "__main__":
    main()
