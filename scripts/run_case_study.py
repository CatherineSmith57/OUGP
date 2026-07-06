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
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from ougp.data import load_graph_dataset
from ougp.model import OUGPConfig, OUGPGCN
from ougp.trace import PruningTraceRecorder


VARIANTS = {
    "dense": dict(use_graph_pruning=False, use_param_pruning=False, use_memory=False, use_cross=False),
    "graph_only": dict(use_graph_pruning=True, use_param_pruning=False, use_memory=True, use_cross=False),
    "param_only": dict(use_graph_pruning=False, use_param_pruning=True, use_memory=True, use_cross=False),
    "dual_static": dict(use_graph_pruning=True, use_param_pruning=True, use_memory=False, use_cross=False),
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
        memory_rank=args.memory_rank,
        graph_target_keep=1.0 - args.graph_sparsity,
        param_target_keep=1.0 - args.param_sparsity,
        graph_gamma=args.graph_gamma,
        param_gamma=args.param_gamma,
        cross_gamma=args.cross_gamma,
        write_beta=args.write_beta,
        write_lambda=args.write_lambda,
        event_gamma=args.event_gamma,
        event_beta=args.event_beta,
        event_decay=args.event_decay,
        event_top_k=args.event_top_k,
        hard_masks=args.hard_eval,
        seed=seed,
        **VARIANTS[variant],
    )
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
            val_acc = evaluate_metric(eval_logits, y, val_mask, data.task_type)
            test_acc = evaluate_metric(eval_logits, y, test_mask, data.task_type)
            train_acc = evaluate_metric(eval_logits, y, train_mask, data.task_type)
            if val_acc > best_val:
                best_val = val_acc
                best_test = test_acc
                best_epoch = epoch

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
            **churn_stats,
            **memory_stats,
            **trace_stats,
        }
        history.append(row)
        if args.verbose and (epoch % args.log_every == 0 or epoch == args.epochs - 1):
            print(
                f"{variant}/seed{seed} epoch={epoch:03d} "
                f"loss={loss.item():.4f} val={val_acc:.4f} test={test_acc:.4f} "
                f"g_keep={eval_mask_stats['graph_keep']:.3f} p_keep={eval_mask_stats['param_keep']:.3f}",
                flush=True,
            )

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
        "graph_memory_state_norm": float(model.graph_memory.state.norm().item()),
        "param_memory_state_norm": float(model.param_memory.state.norm().item()),
        "runtime_sec": time.time() - start_time,
        "num_nodes": data.num_nodes,
        "num_edges": edge_index.size(1),
        "hidden_dim": args.hidden_dim,
        "memory_rank": args.memory_rank,
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
        "",
        "| Variant | Best Test Acc | Graph Sparsity | Param Sparsity | Graph Churn | Param Churn |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant, rows in sorted(by_variant.items()):
        def mean(key: str) -> float:
            return float(np.mean([float(r[key]) for r in rows]))

        def std(key: str) -> float:
            return float(np.std([float(r[key]) for r in rows]))

        lines.append(
            f"| {variant} | {mean('best_test_acc'):.4f} +/- {std('best_test_acc'):.4f} | "
            f"{mean('graph_sparsity'):.3f} | {mean('param_sparsity'):.3f} | "
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="cora",
        choices=["cora", "citeseer", "pubmed", "photo", "ogbn-arxiv", "ogbn-products", "ogbn-proteins"],
    )
    parser.add_argument("--data-root", default="data/raw/planetoid")
    parser.add_argument("--out-dir", default="experiments/manual_case_study")
    parser.add_argument("--variants", nargs="+", default=["dense", "dual_static", "ougp_no_cross", "ougp"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--memory-rank", type=int, default=8)
    parser.add_argument("--graph-sparsity", type=float, default=0.30)
    parser.add_argument("--param-sparsity", type=float, default=0.30)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--sparsity-lambda", type=float, default=0.08)
    parser.add_argument("--graph-gamma", type=float, default=0.35)
    parser.add_argument("--param-gamma", type=float, default=0.35)
    parser.add_argument("--cross-gamma", type=float, default=0.20)
    parser.add_argument("--write-beta", type=float, default=0.12)
    parser.add_argument("--write-lambda", type=float, default=0.98)
    parser.add_argument("--event-gamma", type=float, default=0.0)
    parser.add_argument("--event-beta", type=float, default=0.10)
    parser.add_argument("--event-decay", type=float, default=0.95)
    parser.add_argument("--event-top-k", type=int, default=2000)
    parser.add_argument("--temp-start", type=float, default=2.0)
    parser.add_argument("--temp-end", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-gpus", type=int, default=4)
    parser.add_argument("--hard-eval", action="store_true")
    parser.add_argument("--trace-pruning", action="store_true")
    parser.add_argument("--trace-every", type=int, default=10)
    parser.add_argument("--trace-top-k", type=int, default=200)
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
