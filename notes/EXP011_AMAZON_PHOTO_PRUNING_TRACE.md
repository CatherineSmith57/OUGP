# EXP011: Amazon Photo Pruning Trace Verification

日期：2026-07-06

## 目标

验证新的两层记录设计是否能在 Amazon Photo 上正常工作：

```text
OnlinePruningMemory  -> 训练时低秩统计记忆，继续影响 mask score
PruningTraceRecorder -> 实验分析记录器，只记录剪枝事件，不影响训练
```

本实验重点检查：

- 是否能记录 graph pruning 的具体边位置。
- 是否能记录被剪边的连接节点。
- 是否能记录连接节点的重要程度。
- trace 开启后是否保持原有训练结果大致一致。

## 环境

- conda env: `tianjiaying`
- GPU: `CUDA_VISIBLE_DEVICES=0`
- device: `cuda`
- max GPUs: `1`

## 设置

- dataset: `photo`
- variants: `dense`, `graph_only`, `param_only`, `dual_static`, `ougp_no_cross`, `ougp`
- seeds: `0, 1, 2`
- epochs: `120`
- warmup epochs: `15`
- hidden dim: `64`
- memory rank: `16`
- graph sparsity: `0.30`
- param sparsity: `0.30`
- graph gamma / param gamma: `2.0`
- write beta: `0.25`
- trace pruning: enabled
- trace every: `10`
- trace top k: `200`

配置文件：

```text
configs/photo_pruning_trace_v1.json
```

## 输出

完整实验目录：

```text
experiments/exp011_photo_gpu_pruning_trace/
```

整理表：

```text
results/tables/photo_exp011_gpu_pruning_trace_summary.csv
results/tables/photo_exp011_gpu_pruning_trace_summary.md
```

剪枝事件记录：

```text
experiments/exp011_photo_gpu_pruning_trace/pruning_trace/
```

本轮生成了 12 个 graph-pruning trace CSV：

```text
graph_only: 3 seeds
dual_static: 3 seeds
ougp_no_cross: 3 seeds
ougp: 3 seeds
```

每个 CSV 包含 2000 条 pruning events 加表头。

## Trace 字段

每条 graph pruning event 包含：

```text
dataset
variant
seed
epoch
edge_id
src_node
dst_node
prev_mask
current_mask
mask_delta
graph_score
graph_utility
src_degree
dst_degree
src_feature_norm
dst_feature_norm
src_node_importance
dst_node_importance
edge_importance
graph_keep
param_keep
```

其中：

- `edge_id`：被记录的边编号。
- `src_node` / `dst_node`：这条边连接的两个节点。
- `mask_delta`：这次记录时该边 mask 下降幅度。
- `graph_utility`：来自 edge logit gradient 和 graph mask 的 utility proxy。
- `src_node_importance` / `dst_node_importance`：由 degree、feature norm、incident edge utility 归一化组合得到。
- `edge_importance`：结合 graph utility 和两端节点重要性的边重要性。

注意：当前 trace 只记录 graph pruning events，尚未记录 parameter/channel pruning events。

## 结果

| Variant | Best Test Acc | Graph Sparsity | Param Sparsity | Graph Churn | Param Churn |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense | 0.8124 +/- 0.0161 | 0.000 | 0.000 | 0.000 | 0.000 |
| dual_static | 0.7159 +/- 0.0139 | 0.300 | 0.300 | 0.003 | 0.003 |
| graph_only | 0.8275 +/- 0.0102 | 0.300 | 0.000 | 0.010 | 0.000 |
| ougp | 0.7490 +/- 0.0136 | 0.300 | 0.300 | 0.011 | 0.003 |
| ougp_no_cross | 0.7514 +/- 0.0137 | 0.300 | 0.300 | 0.011 | 0.003 |
| param_only | 0.7116 +/- 0.0131 | 0.000 | 0.300 | 0.000 | 0.003 |

这个结果和 EXP006 的 Amazon Photo 主实验基本一致，说明 trace 记录没有明显改变训练行为。

## 当前结论

新的两层设计已经生效：

1. `OnlinePruningMemory` 仍然负责训练中的低秩 memory correction。
2. `PruningTraceRecorder` 可以记录每次 graph pruning 的边位置、连接节点、mask 变化、节点重要性和边重要性。
3. 记录器默认关闭，只有加 `--trace-pruning` 时才写 CSV，因此不会影响普通实验。
4. 当前已经可以分析“被剪掉的边是否更多连接低重要性节点”。

## 下一步

建议下一步做两个小分析：

1. 统计被剪边的 `edge_importance` 分布，和随机边作对比。
2. 扩展 parameter/channel pruning trace，记录每次被压低的 hidden channel、channel utility、两层 weight norm。
