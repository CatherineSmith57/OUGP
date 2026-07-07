# EXP028: Resource Metrics Smoke

## 目的

补齐 idea 中要求的计算成本相关指标，使每次实验不仅记录 accuracy / sparsity / churn，也记录 message passing cost、parameter cost 和 online memory overhead。

这组是 Cora 上的短 smoke，只验证指标是否正确落盘，不作为性能结论。

## 设置

```text
dataset: Cora
backbone: GCN
variants: dense, OUGP
seed: 0
epochs: 3
warmup epochs: 1
hidden dim: 16
memory rank: 4
graph sparsity: 20%
parameter sparsity: 20%
budget lambda: 0.0
device: CPU
```

实验目录：

```text
experiments/smoke_resource_metrics/
```

## 新增指标

```text
dense_message_cost
effective_message_cost
message_cost_ratio
message_cost_reduction
dense_parameter_count
effective_parameter_count
parameter_cost_ratio
parameter_cost_reduction
memory_state_items
memory_overhead_vs_dense_params
```

## 结果

| Variant | Best Test Acc | Graph Sparsity | Param Sparsity | Message Cost Red. | Param Cost Red. | Graph Churn | Param Churn |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | 0.5240 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| ougp | 0.5210 | 0.200 | 0.200 | 0.161 | 0.200 | 0.100 | 0.100 |

## 观察

```text
新增 cost 指标已进入 per-run JSON、summary CSV、summary Markdown 和 manifest。
OUGP 在 20% graph sparsity + 20% parameter sparsity 下，近似 message cost reduction 为 16.1%，parameter cost reduction 为 20.0%。
message cost reduction 小于 graph sparsity，是因为 GCN/GAT 分支包含 self-loop，self-loop 不受 graph pruning 影响。
```

## 限制

```text
当前是近似 cost，不是硬件实测 latency。
budget regularization 已有入口，但默认关闭；正式实验需要单独 sweep --budget-lambda。
```

## Budget Regularization 入口验证

额外跑了一个极短 smoke，确认非零 `--budget-lambda` 可以正常训练和落盘：

```text
实验目录: experiments/smoke_budget_regularization/
dataset: Cora
variant: OUGP
epochs: 2
budget lambda: 0.01
budget target: 0.80
结果: 跑通，并记录 message/parameter cost reduction
```
