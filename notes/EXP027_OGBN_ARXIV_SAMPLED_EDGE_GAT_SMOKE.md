# EXP027: ogbn-arxiv Sampled-Edge GAT Smoke

## 目的

验证当前 `--edge-sample-size` 入口能否在大图 ogbn-arxiv 上跑通 OUGP，并与新接入的 `--backbone gat` 共用同一套 graph/channel online memory。

这不是正式大图实验。它只抽取一部分边做流程验证，不等同于完整 mini-batch/subgraph training，也不能作为论文性能结论。

## 设置

```text
dataset: ogbn-arxiv
backbone: GAT
variant: OUGP
seed: 0
epochs: 3
warmup epochs: 1
hidden dim: 16
memory rank: 4
edge sample size: 20000
graph sparsity: 20%
parameter sparsity: 20%
device: CPU
```

实验目录：

```text
experiments/exp027_ogbn_arxiv_sampled_edge_gat_smoke/
```

## 结果

| Variant | Best Test Acc | Graph Sparsity | Param Sparsity | Graph Churn | Param Churn |
| --- | ---: | ---: | ---: | ---: | ---: |
| ougp | 0.2289 | 0.200 | 0.200 | 0.100 | 0.100 |

记录的规模信息：

```text
original_num_edges: 1166243
edge_sample_size: 20000
```

## 观察

```text
ogbn-arxiv 数据加载成功。
sampled-edge 图构造成功。
GAT + OUGP 在大图节点规模下可以 forward/backward/write memory。
最终达到 20% graph sparsity + 20% parameter sparsity。
```

## 限制

```text
这是 sampled-edge smoke，不是正式大图训练。
目前仍然是 full-node forward，只是抽了边；真正的大图方案还需要 sampled-subgraph / mini-batch。
```

## 下一步

```text
1. 做 ogbn-arxiv sampled-edge 多 seed/多 backbone smoke。
2. 设计正式 sampled-subgraph mini-batch trainer。
3. 引入 FLOPs / latency / memory overhead 指标。
```
