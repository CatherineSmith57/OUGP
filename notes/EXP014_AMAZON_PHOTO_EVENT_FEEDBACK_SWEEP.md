# EXP014 Amazon Photo Event Feedback Sweep

## 目的

验证新加入 `OnlinePruningMemory.event_bias` 后，剪枝事件记忆是否能改善 Amazon Photo 上的图剪枝稳定性或分类性能。

这次实验不是改源码硬跑，而是通过命令行参数打开/调节：

```text
event_gamma = 0.00 / 0.05 / 0.10 / 0.20
```

其中 `event_gamma=0.00` 是旧行为对照：event memory 仍会写入日志和 buffer，但不会加回 `graph_score`，因此不会影响模型训练决策。

## 实验设置

| 项目 | 设置 |
| --- | --- |
| Dataset | Amazon Photo |
| Backbone | 2-layer GCN |
| Variants | `graph_only`, `ougp` |
| Seeds | 0, 1, 2 |
| Epochs | 120 |
| Hidden dim | 64 |
| Memory rank | 16 |
| Graph sparsity | 30% |
| Parameter sparsity | 30% |
| GPU | 1 GPU, `tianjiaying` env |
| Trace | enabled, every 10 epochs, top 200 events |

配置文件：

```text
configs/photo_event_feedback_sweep_v1.json
```

启动脚本：

```text
scripts/run_exp014_photo_event_feedback_sweep.sh
```

原始输出：

```text
experiments/exp014_photo_event_feedback_sweep/
```

汇总表：

```text
results/tables/photo_exp014_event_feedback_sweep_summary.csv
results/tables/photo_exp014_event_feedback_sweep_summary.md
```

## 结果

| event_gamma | Variant | Best Test Acc | Graph Sparsity | Param Sparsity | Graph Churn | Event Bias Norm |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0.00 | graph_only | 0.8272 +/- 0.0120 | 0.300 | 0.000 | 0.018 | 2.317 |
| 0.00 | ougp | 0.7504 +/- 0.0121 | 0.300 | 0.300 | 0.010 | 2.355 |
| 0.05 | graph_only | 0.8276 +/- 0.0113 | 0.300 | 0.000 | 0.011 | 1.989 |
| 0.05 | ougp | 0.7516 +/- 0.0157 | 0.300 | 0.300 | 0.011 | 2.105 |
| 0.10 | graph_only | 0.8242 +/- 0.0089 | 0.300 | 0.000 | 0.004 | 2.136 |
| 0.10 | ougp | 0.7492 +/- 0.0126 | 0.300 | 0.300 | 0.007 | 2.144 |
| 0.20 | graph_only | 0.8243 +/- 0.0154 | 0.300 | 0.000 | 0.014 | 2.029 |
| 0.20 | ougp | 0.7490 +/- 0.0139 | 0.300 | 0.300 | 0.010 | 1.851 |

## 观察

1. `event_gamma=0.05` 对 OUGP 有轻微提升：`0.7504 -> 0.7516`，但幅度很小，低于随机种子波动范围。
2. `event_gamma=0.10` 和 `0.20` 没有进一步改善，反而略低。
3. `graph_only` 仍明显优于同时做 graph pruning + parameter pruning 的 OUGP。
4. 这说明 Amazon Photo 当前主要瓶颈仍然是 parameter pruning 对表达能力的伤害；event feedback 只能调整图边剪枝分数，不能直接修复参数剪枝带来的通道容量损失。
5. `event_bias` 的 norm 非零，说明事件记忆确实被写入；但只有 `event_gamma > 0` 时才会参与 `graph_score`。

## 当前结论

在 Amazon Photo + 2-layer GCN + 30% graph sparsity + 30% parameter sparsity 设置下，当前 event feedback 设计没有带来明显准确率提升。

更准确的表述是：

```text
剪枝事件记忆已经接入 OnlinePruningMemory，并且可以影响 graph_score；
但在 Amazon Photo 上，它无法解决 OUGP 主要由 parameter pruning 引起的性能下降。
```

## 下一步建议

1. 先不要在大图上继续 full-batch graph pruning，因为 `ogbn-arxiv` 已经证明会 OOM。
2. 若继续改模型，应优先处理 parameter pruning，例如降低参数剪枝率、延迟参数剪枝、或给参数剪枝增加恢复机制。
3. 若要验证大图，需要实现 sampled / mini-batch OUGP，而不是继续 full-batch GCN。
