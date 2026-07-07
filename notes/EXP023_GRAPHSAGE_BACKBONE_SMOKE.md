# EXP023: GraphSAGE Backbone Smoke

## 目的

验证当前 OUGP 不再只绑定 2-layer GCN，而是可以在同一套 graph mask、channel mask 和 online memory 机制下切换到 GraphSAGE。

## 改动

新增 `--backbone` 参数：

```text
--backbone gcn
--backbone sage
```

GraphSAGE 分支使用：

```text
neighbor mean aggregation
self linear + neighbor linear
hidden channel mask
shared graph/channel online memory
```

## Smoke 设置

```text
dataset: Cora
backbone: sage
variants: dense, ougp
seeds: 0
epochs: 4
graph sparsity: 20%
parameter sparsity: 20%
device: CPU
```

实验目录：

```text
experiments/smoke_sage_backbone/
```

## 结果

| Variant | Best Test Acc | Graph Sparsity | Param Sparsity | Graph Churn | Param Churn |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense | 0.6640 | 0.000 | 0.000 | 0.000 | 0.000 |
| ougp | 0.6610 | 0.200 | 0.200 | 0.067 | 0.067 |

## 结论

这不是正式性能结论，只是功能验证：

```text
GraphSAGE backbone 已经能跑通；
OUGP mask/memory/write 路径在 GraphSAGE 下有效；
实验参数、日志、JSON history、summary 都已落盘。
```

下一步正式验证需要：

```text
Cora / CiteSeer / PubMed 多 seed
Amazon Photo 多 seed
与 GCN 在同等剪枝率下比较
再继续扩展 GAT 或大图 mini-batch 版本
```
