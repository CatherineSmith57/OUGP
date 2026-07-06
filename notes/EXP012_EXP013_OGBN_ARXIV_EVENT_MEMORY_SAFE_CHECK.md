# EXP012/EXP013: ogbn-arxiv Event-Memory Safe Check

日期：2026-07-06

## 目标

按照大图路线，先验证新版 event-memory 代码没有破坏大图安全路径，再确认 full-batch graph pruning 是否仍然 OOM。

本轮不直接跑 `ogbn-products` / `ogbn-proteins`，也不直接跑完整 full-batch OUGP。

## 环境

- conda env: `tianjiaying`
- GPU: `CUDA_VISIBLE_DEVICES=1`
- max GPUs: `1`
- event feedback: disabled (`event_gamma=0.0`)
- trace pruning: disabled

## EXP012: ogbn-arxiv safe regression

配置：

```text
configs/ogbn_arxiv_safe_regression_event_memory_v1.json
```

输出：

```text
experiments/exp012_ogbn_arxiv_safe_regression_event_memory/
results/tables/ogbn_arxiv_exp012_safe_regression_event_memory_summary.csv
results/tables/ogbn_arxiv_exp012_safe_regression_event_memory_summary.md
```

设置：

- dataset: `ogbn-arxiv`
- variants: `dense`, `param_only`
- seeds: `0, 1, 2`
- epochs: `80`
- hidden dim: `64`
- memory rank: `8`
- param sparsity: `0.30`

结果：

| Variant | Best Test Acc | Graph Sparsity | Param Sparsity | Graph Churn | Param Churn |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense | 0.6051 +/- 0.0017 | 0.000 | 0.000 | 0.000 | 0.000 |
| param_only | 0.5943 +/- 0.0017 | 0.000 | 0.300 | 0.000 | 0.005 |

结论：

- 新版 event-memory 代码没有破坏 `dense` / `param_only` 大图运行。
- 30% parameter sparsity 仍带来轻微准确率下降。

## EXP013: ogbn-arxiv graph pruning smoke

配置：

```text
configs/ogbn_arxiv_graph_pruning_smoke_event_memory_v1.json
```

输出：

```text
experiments/exp013_ogbn_arxiv_graph_pruning_smoke_event_memory/run.log
```

设置：

- dataset: `ogbn-arxiv`
- variant: `graph_only`
- seed: `0`
- epochs: `1`
- hidden dim: `64`
- event gamma: `0.0`
- trace pruning: disabled

结果：

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 106.83 GiB.
```

失败发生在：

```text
loss.backward()
```

结论：

- full-batch graph pruning 在 `ogbn-arxiv` 上仍然不可行。
- OOM 与 event feedback 是否开启无关，因为本轮 `event_gamma=0.0` 且 trace pruning disabled。
- 根因仍是 full-batch GCN + learnable edge mask 在百万边级别 backward 的内存需求过高。

## 当前结论

大图路线应停止 full-batch 完整 OUGP，转向 sampled / mini-batch OUGP。

可以继续保留的证据：

1. `ogbn-arxiv` dense / param-only 能跑。
2. 30% parameter pruning 在大图上有小幅性能下降。
3. graph-pruning full-batch backward 需要约 106.83 GiB，当前 24 GiB GPU 无法承载。

## 下一步

实现：

```text
scripts/run_ogb_sampled_case_study.py
```

目标：

- 用 neighbor sampling / mini-batch 替代 full-batch GCN。
- graph mask 只作用于 sampled subgraph / batch edges。
- event memory 以 batch edge event 或 hash/cache 形式更新。
- trace recorder 记录 batch 内剪枝事件。

