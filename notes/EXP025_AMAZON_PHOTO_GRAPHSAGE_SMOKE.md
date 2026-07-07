# EXP025: Amazon Photo GraphSAGE Smoke

## 目的

验证 `--backbone sage` 在 Amazon Photo 这种中等规模图上能否跑通 OUGP 的完整训练流程。

这不是正式性能实验。20 epoch 对 Amazon Photo + GraphSAGE 明显不够，dense baseline 也没有充分收敛。

## 设置

```text
dataset: Amazon Photo
backbone: GraphSAGE
variants: dense, ougp
seed: 0
epochs: 20
warmup epochs: 5
hidden dim: 64
memory rank: 8
graph sparsity: 30%
parameter sparsity: 30%
device: CPU
```

实验目录：

```text
experiments/exp025_photo_graphsage_validation/
```

## 结果

| Variant | Best Test Acc | Graph Sparsity | Param Sparsity | Graph Churn | Param Churn |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense | 0.3399 | 0.000 | 0.000 | 0.000 | 0.000 |
| ougp | 0.2570 | 0.300 | 0.300 | 0.020 | 0.020 |

## 解释

```text
这组实验只能说明 GraphSAGE + OUGP 在 Amazon Photo 上可以运行并落盘。
不能说明 GraphSAGE 版本的 OUGP 效果差，因为 dense baseline 自身也明显没有收敛。
正式比较需要使用更长 epoch、多 seed，最好用 GPU 跑。
```
