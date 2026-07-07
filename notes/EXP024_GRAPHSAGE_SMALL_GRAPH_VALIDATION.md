# EXP024: GraphSAGE Small-Graph Validation

## 目的

在当前 OUGP 已支持 `--backbone sage` 后，验证同一套 online graph/channel pruning memory 是否能在 Cora、CiteSeer、PubMed 上跑通。

这组实验是单 seed、20 epoch 的轻量验证，用于检查方法流程和趋势，不作为最终论文结果。

## 设置

```text
backbone: GraphSAGE
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
experiments/exp024_sage_small_graph_validation/
```

## 结果

| Dataset | Dense Best Test Acc | OUGP Best Test Acc | Graph Sparsity | Param Sparsity | Graph Churn | Param Churn |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cora | 0.7030 | 0.6820 | 0.300 | 0.300 | 0.020 | 0.020 |
| CiteSeer | 0.4740 | 0.4390 | 0.300 | 0.300 | 0.020 | 0.020 |
| PubMed | 0.7390 | 0.7390 | 0.300 | 0.300 | 0.020 | 0.020 |

## 观察

```text
GraphSAGE + OUGP 的完整训练、mask 生成、memory write、结果落盘都已跑通。
OUGP 能稳定实现 30% graph sparsity + 30% parameter sparsity。
mask churn 保持在 0.020 左右，没有出现之前 channel memory 未对齐时的强震荡。
PubMed 上几乎不掉点；Cora 和 CiteSeer 有性能下降，需要后续多 seed 和参数调节。
```

## 下一步

```text
1. 对 GraphSAGE 做多 seed 正式验证。
2. 在 Amazon Photo 上验证 GraphSAGE + OUGP。
3. 对比 GCN vs GraphSAGE 下 channel memory 是否同样稳定。
4. 扩展 GAT 或 mini-batch 大图版本。
```
