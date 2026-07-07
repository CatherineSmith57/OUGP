# EXP031: DeeperGCN Backbone Smoke

## 目的

补齐 idea 中的 DeeperGCN backbone，使 OUGP 不只支持 2-layer GCN / GraphSAGE / GAT，也能在更深的 GCN-style residual network 上运行。

这组是 Cora 上的 4 epoch smoke，只验证 backbone、mask/memory 路径、cost 指标和结果落盘，不作为性能结论。

## 设置

```text
dataset: Cora
backbone: deepgcn
num_gnn_layers: 4
variants: dense, OUGP
seed: 0
epochs: 4
warmup epochs: 1
hidden dim: 16
memory rank: 4
graph sparsity: 20%
parameter sparsity: 20%
device: CPU
```

实验目录：

```text
experiments/exp031_deepgcn_backbone_smoke/
```

## 结果

| Variant | Best Test Acc | Graph Sparsity | Param Sparsity | Message Cost Red. | Param Cost Red. | Graph Churn | Param Churn |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | 0.3160 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| ougp | 0.3130 | 0.200 | 0.200 | 0.165 | 0.203 | 0.067 | 0.067 |

## 观察

```text
deepgcn + OUGP 的 forward/backward/memory write 已跑通。
OUGP 能稳定达到 20% graph sparsity + 20% parameter sparsity。
resource metrics 能根据 num_gnn_layers 记录更深模型的 message/parameter cost reduction。
```

## 限制

```text
这是 4 epoch smoke，不能解释 DeeperGCN 的最终性能。
当前 deepgcn 是轻量 residual hidden-block 版本，还不是完整 DeeperGCN 论文里的全部工程组件。
```

## 下一步

```text
1. 对 deepgcn 做 Cora / CiteSeer / PubMed 多 seed 验证。
2. 对比 GCN / GraphSAGE / GAT / deepgcn 四种 backbone。
3. 检查深层网络下 memory write mode 和 mask churn 是否更敏感。
```
