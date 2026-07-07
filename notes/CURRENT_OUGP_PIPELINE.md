# Current OUGP Pipeline

## 目标

当前 OUGP 的目标是在 GNN 训练过程中同时做两类剪枝：

```text
1. graph pruning: 剪图边
2. parameter pruning: 剪 hidden channel / 参数通道
```

同时引入 online memory，让模型根据历史训练信号动态调整剪枝决策。

当前实验主模型默认是：

```text
2-layer GCN
```

同时已经接入一个 GraphSAGE 分支：

```text
--backbone gcn    # 默认
--backbone sage   # 已支持 smoke 验证
--backbone gat    # 已支持小图验证
--backbone deepgcn --num-gnn-layers N  # 已支持 smoke 验证，N >= 3
```

为了给大图验证做过渡，runner 还支持 sampled-edge smoke：

```text
--edge-sample-size N
```

它会在加载完整数据集后抽取 `N` 条边来构造当前实验图，并在结果中记录：

```text
original_num_edges
edge_sample_size
```

注意：这只是大图流程验证入口，不等同于正式 mini-batch/subgraph training。

现在还新增了 node-sampled induced subgraph smoke：

```text
--node-sample-size N
--node-sample-seed S
--node-sample-mode random|frontier
```

它会在加载完整数据集后抽取节点，保留这些节点之间的 induced edges，并把原图节点 id 重映射成局部节点 id。结果中会记录：

```text
original_num_nodes
original_num_edges
node_sample_size
num_nodes
num_edges
```

注意：这比 sampled-edge 更接近大图 subgraph 路线，但仍然是静态子图 smoke，不等同于正式 mini-batch neighbor/subgraph training。

其中：

```text
random:
  保留 train/val/test 种子覆盖后，随机补齐节点。

frontier:
  保留 train/val/test 种子覆盖后，从这些种子沿邻居扩展，再补齐节点。
```

`frontier` 模式通常会得到更密的 induced subgraph，更适合做大图 smoke。

## 总体流程

```text
输入图数据
  ↓
构造 graph context 和 parameter context
  ↓
online memory read
  ↓
生成 graph_score 和 param_score
  ↓
budgeted sigmoid 生成 graph_mask 和 param_mask
  ↓
用 mask 跑 2-layer GNN forward
  ↓
计算 loss 并 backward
  ↓
根据 gradient utility 写入 online memory
  ↓
optimizer 更新模型参数
```

## Step 1: 输入

每个数据集包含：

```text
x: node features
edge_index: graph edges
y: labels
train / val / test mask
```

当前主要实验数据集包括：

```text
Cora
CiteSeer
PubMed
Amazon Photo
```

## Step 2: 动态设置目标剪枝率

每个 epoch 会根据 warmup schedule 设置当前目标 keep rate：

```text
graph_target_keep: 从 1.0 逐渐下降到 0.7
param_target_keep: 从 1.0 逐渐下降到 0.7
```

对应最终目标：

```text
graph sparsity = 30%
parameter sparsity = 30%
```

## Step 3: 构造 context

### Graph context

graph memory 是 edge-level 的，每条边都有一个 context。

大致包含：

```text
src node feature projection
dst node feature projection
src degree
dst degree
current parameter keep
```

它用来判断：

```text
哪些边更应该保留或剪掉
```

### Parameter context

parameter memory 是 channel-level 的，每个 hidden channel 有一个 context。

大致包含：

```text
lin1 channel norm
lin2 channel norm
channel id
current channel keep probability
current graph keep
bias
```

它用来判断：

```text
哪些 hidden channel 更应该保留或剪掉
```

## Step 4: Online memory read

### Graph memory

graph memory 输出：

```text
graph_corr: 每条边的 memory correction
```

### Channel-specific parameter memory

当前 parameter memory 已经改成 channel-specific memory。

它的 memory state 形状是：

```text
[num_channels, memory_rank, memory_rank]
```

含义是：

```text
每个 hidden channel 有自己的 memory slot
```

memory read 输出：

```text
raw_param_corr: 每个 channel 的原始 memory correction
```

## Step 5: 统一 parameter score 尺度

当前版本里，`param_logits` 是 parameter pruning 的主坐标系。

为了避免 memory correction 和 recall correction 尺度不一致，现在统一到 `param_logits` 的动态尺度。

流程是：

```text
raw memory correction
  ↓
去均值
  ↓
标准化为 unit correction
  ↓
clip 限幅
  ↓
乘 param_logits.std() 的 EMA scale
```

当前 parameter score 公式：

```python
param_score =
    param_logits
  + param_gamma * param_logit_scale_ema * normalized_channel_memory
  + recall_gamma * param_logit_scale_ema * normalized_recall
```

这样做的目的：

```text
param_logits、channel memory、recall correction 都在同一个 score 尺度里参与排序
```

## Step 6: 生成 pruning score

### Graph score

```python
graph_score =
    edge_logits
  + graph_gamma * graph_corr
  + event_gamma * graph_event_bias
  + recall_gamma * graph_recall_bias
```

### Parameter score

```python
param_score =
    param_logits
  + param_gamma * scaled_channel_memory
  + recall_gamma * scaled_param_recall
```

## Step 7: Budgeted sigmoid 生成 mask

OUGP 使用 fixed-budget pruning。

也就是说，mask 不是自由变稀疏，而是尽量满足目标 keep rate：

```text
mean(graph_mask) ≈ graph_target_keep
mean(param_mask) ≈ param_target_keep
```

最终得到：

```text
graph_mask: 每条边的保留/剪枝权重
param_mask: 每个 hidden channel 的保留/剪枝权重
```

## Step 8: Backbone forward

### GCN

GCN forward 大致是：

```text
输入 node features
  ↓
使用 graph_mask 后的图做 graph convolution
  ↓
Linear layer 1
  ↓
ReLU
  ↓
乘 param_mask，剪 hidden channel
  ↓
Dropout
  ↓
第二次 graph convolution
  ↓
Linear layer 2
  ↓
输出 logits
```

其中：

```text
graph_mask 控制图结构
param_mask 控制 hidden representation 的通道
```

### GraphSAGE

GraphSAGE 分支复用同一套 mask 和 online memory：

```text
输入 node features
  ↓
使用 graph_mask 后的邻居 mean aggregation
  ↓
self linear + neighbor linear
  ↓
ReLU
  ↓
乘 param_mask，剪 hidden channel
  ↓
Dropout
  ↓
第二层 self linear + neighbor linear
  ↓
输出 logits
```

GraphSAGE 的 parameter context 仍然是 channel-level，但是通道 norm 会综合 self/neighbor 两组线性层。

### GAT

GAT 分支目前是轻量单头版本，复用同一套 OUGP mask 和 memory：

```text
输入 node features
  ↓
Linear projection
  ↓
使用 graph_mask 加权 edge attention
  ↓
attention aggregation
  ↓
ReLU
  ↓
乘 param_mask，剪 hidden channel
  ↓
Dropout
  ↓
第二层 attention aggregation
  ↓
输出 logits
```

GAT 的 parameter context 仍以 hidden channel 为单位，并把第一层 attention 参数的通道强度纳入 channel norm。

### DeeperGCN

DeeperGCN 分支目前是 residual hidden-block 版本：

```text
输入 node features
  ↓
GCN aggregation + Linear(in -> hidden)
  ↓
ReLU + param_mask
  ↓
重复 N-2 个 hidden residual block:
    GCN aggregation
    Linear(hidden -> hidden)
    residual add
    ReLU + param_mask
  ↓
最后一层 GCN aggregation + Linear(hidden -> out)
  ↓
输出 logits
```

它复用同一套 graph mask、channel mask、online memory 和 cost 统计。  
`num_gnn_layers` 当前只对 `deepgcn` 生效。

## Step 9: Loss 和 backward

训练 loss 是：

```text
task_loss + sparsity regularization
```

然后执行 backward，得到：

```text
edge_logits.grad
param_logits.grad
```

这些梯度会作为 online memory 写入信号的一部分。

## Step 10: Online memory write

### Graph utility

```python
graph_utility = abs(edge_logits.grad * last_graph_mask)
```

用于更新：

```text
graph memory
graph event memory
graph recall memory
```

### Parameter utility

```python
param_utility = abs(param_logits.grad * last_param_mask)
```

用于更新：

```text
channel-specific parameter memory
parameter recall memory
```

其中 channel-specific parameter memory 是逐 channel 写入的：

```text
channel i 的 utility 只更新 channel i 的 memory slot
```

### Memory write 消融模式

当前 runner 支持：

```text
--memory-write-mode residual
--memory-write-mode feature
--memory-write-mode none
```

含义是：

```text
residual: 主方法。写入 utility prediction residual 调制后的 value。
feature: 只写 context feature value，不使用 residual utility 调制。
none: 只读不写。跳过 graph/parameter/event/recall/steering memory state 更新。
```

这个入口对应 idea 里的关键消融：

```text
state 只读不写
state 只写特征，不写 residual utility
gated delta-rule residual write
```

## Step 11: Optimizer step

最后 optimizer 更新：

```text
GCN weights
GraphSAGE weights
edge_logits
param_logits
memory projection / read heads 等可学习参数
```

memory state 本身不是普通 optimizer 参数，而是通过 memory write 手动更新。

## 当前重要指标

实验时主要看：

```text
best_test_acc
graph_sparsity
param_sparsity
message_cost_ratio
message_cost_reduction
parameter_cost_ratio
parameter_cost_reduction
memory_overhead_vs_dense_params
graph_churn
param_churn
param_logits_std
param_score_scale
param_memory_raw_correction_std
param_memory_correction_std
recall_correction_std
```

其中：

```text
param_churn
```

非常关键。它表示 parameter mask 每轮变化幅度。  
如果太大，说明 channel pruning 决策不稳定。

## Step 12: 资源成本估计

当前版本已经记录近似资源指标：

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

这些指标是估算值，用于跨 sparsity / backbone / dataset 比较趋势：

```text
message cost: 近似 message passing 中边聚合相关计算量
parameter count: 近似结构化 hidden channel 剪枝后的参数量
memory overhead: online memory state + event/recall bias 的状态规模
```

runner 还支持可选 budget regularization：

```text
--budget-lambda
--budget-target
```

默认 `--budget-lambda 0.0`，所以不会改变旧实验行为。  
目前这还不是正式 latency benchmark，也不是硬件实测 FLOPs。

## Step 13: 静态剪枝 Baseline

当前 runner 支持静态 score 初始化和冻结：

```text
--graph-score-init constant|random|degree|similarity
--param-score-init constant|random|magnitude
--freeze-pruning-scores
```

已经内置的静态 baseline variants：

```text
random_static
degree_magnitude_static
similarity_magnitude_static
```

含义是：

```text
random_static:
  graph score = random
  parameter score = random
  pruning scores frozen

degree_magnitude_static:
  graph score = src/dst degree
  parameter score = initial channel weight magnitude
  pruning scores frozen

similarity_magnitude_static:
  graph score = feature cosine similarity
  parameter score = initial channel weight magnitude
  pruning scores frozen
```

这些 baseline 用来对比 OUGP 的 online residual memory 是否优于普通静态剪枝准则。

## 当前 Amazon Photo 结果

统一尺度后的当前最好结果：

```text
Best Test Acc: 0.7550 ± 0.0149
Graph sparsity: 30%
Parameter sparsity: 30%
Param churn: 0.003
```

说明：

```text
channel-specific memory 已经生效；
scale alignment 缓解了 mask 震荡；
但 parameter pruning 仍然没有追上 graph_only baseline。
```

## 当前结论

当前 OUGP pipeline 的关键经验是：

```text
online memory 不能只记录全局剪枝率；
需要记录 edge/channel 级别的信息；
memory feedback 还必须和 pruning logits 统一尺度；
否则 memory 越有效，越可能导致 mask instability。
```

## 与 idea 设计的当前差距

已经实现：

```text
统一 graph pruning + channel parameter pruning
online read-steer-write memory
channel-specific parameter memory
recall/event memory
parameter score scale alignment
GCN / GraphSAGE / GAT / DeeperGCN 四个 backbone 入口
近似 message/parameter cost 指标
random / degree / similarity / magnitude 静态剪枝 baseline
小图到 Amazon Photo 的实验记录
```

仍未完整实现：

```text
大图 mini-batch / sampled graph pruning
Subgraph-State Write 和 Multi-State Write
真实 FLOPs / latency benchmark
更完整的 baseline 矩阵和消融
```

已新增但仍属于 smoke 级别：

```text
sampled-edge large-graph入口：--edge-sample-size
node-sampled induced subgraph入口：--node-sample-size, --node-sample-mode random|frontier
```

最新 smoke 验证：

```text
实验目录: experiments/smoke_sage_backbone
数据集: Cora
backbone: GraphSAGE
variants: dense, ougp
结果: OUGP 跑通，并达到 20% graph sparsity + 20% parameter sparsity
```

最新 DeeperGCN smoke 验证：

```text
实验目录: experiments/exp031_deepgcn_backbone_smoke
数据集: Cora
backbone: deepgcn
num_gnn_layers: 4
variants: dense, ougp
结果: OUGP 跑通，并达到 20% graph sparsity + 20% parameter sparsity
```

最新大图 smoke 验证：

```text
实验目录: experiments/exp027_ogbn_arxiv_sampled_edge_gat_smoke
数据集: ogbn-arxiv
backbone: GAT
edge sample size: 20000 / original edges: 1166243
结果: OUGP 跑通，并达到 20% graph sparsity + 20% parameter sparsity
注意: sampled-edge smoke，不是正式 mini-batch 大图性能结论
```

最新 node-subgraph smoke 验证：

```text
实验目录: experiments/exp032_ogbn_arxiv_node_subgraph_smoke
数据集: ogbn-arxiv
backbone: GCN
node sample size: 5000 / original nodes: 169343
sampled edges: 1247 / original edges: 1166243
结果: OUGP 跑通，并达到 20% graph sparsity + 20% parameter sparsity
注意: node-sampled induced subgraph smoke，不是正式 mini-batch 大图性能结论
```

最新 frontier subgraph smoke 验证：

```text
实验目录: experiments/exp033_ogbn_arxiv_frontier_subgraph_smoke
数据集: ogbn-arxiv
backbone: GCN
node sample mode: frontier
node sample size: 5000 / original nodes: 169343
sampled edges: 24160 / original edges: 1166243
对比: random node-subgraph 同样 5000 节点只有 1247 条 induced edges
结果: OUGP 跑通，并达到 20% graph sparsity + 20% parameter sparsity
注意: frontier subgraph smoke 仍不是正式 mini-batch 大图性能结论
```
