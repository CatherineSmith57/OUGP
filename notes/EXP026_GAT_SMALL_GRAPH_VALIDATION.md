# EXP026: GAT Small-Graph Validation

## 目的

在当前 OUGP 已支持 `--backbone gat` 后，验证同一套 online graph/channel pruning memory 是否能在 Cora、CiteSeer、PubMed 上跑通。

这组实验是单 seed、20 epoch 的轻量验证，用于检查方法流程和趋势，不作为最终论文结果。

## 设置

```text
backbone: GAT
variants: dense, ougp
seeds: 0
epochs: 20
warmup epochs: 5
hidden dim: 32
memory rank: 8
graph sparsity: 30%
parameter sparsity: 30%
device: CPU
```

实验目录：

```text
experiments/exp026_gat_small_graph_validation/
```

## 结果

| Dataset | Dense Best Test Acc | OUGP Best Test Acc | Graph Sparsity | Param Sparsity | Graph Churn | Param Churn |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cora | 0.7780 | 0.7680 | 0.300 | 0.300 | 0.020 | 0.020 |
| CiteSeer | 0.5120 | 0.5250 | 0.300 | 0.300 | 0.020 | 0.020 |
| PubMed | 0.7220 | 0.7240 | 0.300 | 0.300 | 0.020 | 0.020 |

## 观察

```text
GAT + OUGP 的完整训练、mask 生成、memory write、结果落盘都已跑通。
OUGP 能稳定实现 30% graph sparsity + 30% parameter sparsity。
mask churn 保持在 0.020 左右，没有出现高频震荡。
Cora 轻微下降；CiteSeer 和 PubMed 在该轻量设置下略高于 dense，但还需要多 seed 正式验证。
```

## 下一步

```text
1. 对 GAT 做多 seed 正式验证。
2. 在 Amazon Photo 上验证 GAT + OUGP。
3. 对比 GCN / GraphSAGE / GAT 三种 backbone 下 memory correction 和 churn。
4. 继续补大图 mini-batch 或 sampled-subgraph 版本。
```
