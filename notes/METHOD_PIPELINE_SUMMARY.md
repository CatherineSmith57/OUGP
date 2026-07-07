# OUGP Current Method Pipeline Summary

## 1. 方法目标

当前 OUGP 的目标是：

```text
在 GNN 训练过程中，同时进行
1. graph pruning
2. parameter pruning

并且用 online memory 动态修正剪枝决策。
```

这里的 parameter pruning 目前不是剪单个 weight，而是剪 hidden channel，这样更接近结构化剪枝，也更有真实加速意义。

## 2. 当前主模型

当前最成熟、最稳定的主版本是：

```text
2-layer GCN
+ unified graph pruning + channel pruning
+ channel-specific parameter memory
+ recall / event memory
+ param-score scale alignment
```

在代码层面，已经支持：

```text
GCN
GraphSAGE
GAT
DeeperGCN
```

但目前最主要的实验结论仍然来自：

```text
2-layer GCN
```

## 3. 整体流程

当前 pipeline 可以概括成：

```text
输入图数据
  -> 构造 graph context / parameter context
  -> online memory read
  -> 生成 graph score / param score
  -> budgeted mask 生成
  -> masked GNN forward
  -> backward 得到 utility signal
  -> online memory write
  -> optimizer update
```

## 4. 输入与剪枝对象

输入数据包括：

```text
x           节点特征
edge_index  图边
y           标签
train/val/test mask
```

当前两个主要剪枝对象是：

```text
graph unit:
  edge

parameter unit:
  hidden channel
```

也就是说：

```text
graph pruning 控制哪些边保留
parameter pruning 控制哪些 hidden channel 保留
```

## 5. Graph 分支

### 5.1 Graph context

每条边都会构造一个 graph context，当前大致包含：

```text
src node feature projection
dst node feature projection
src degree
dst degree
current parameter keep
```

这个 context 的作用是让模型知道：

```text
当前这条边在当前训练状态下，是否更值得保留
```

### 5.2 Graph memory read

graph memory 会输出：

```text
graph_corr
```

它表示：

```text
历史 memory 对当前 edge score 的修正
```

### 5.3 Graph score

当前 graph score 形式大致是：

```python
graph_score =
    edge_logits
  + graph_gamma * graph_corr
  + event_gamma * graph_event_bias
  + recall_gamma * graph_recall_bias
```

其中：

```text
edge_logits        当前可学习基础分数
graph_corr         graph memory 的历史修正
event_bias         被剪枝事件的累计影响
recall_bias        恢复/保护相关的累计影响
```

## 6. Parameter 分支

### 6.1 为什么要改成 channel-specific memory

早期版本的 parameter memory 更像一个全局 bias，不能区分不同 channel。后来改成了：

```text
每个 hidden channel 一个独立 memory slot
```

也就是 memory state 形状变成：

```text
[num_channels, memory_rank, memory_rank]
```

这一步的意义是：

```text
memory 终于能对不同 channel 给出不同修正
```

### 6.2 Parameter context

每个 hidden channel 会构造一个 parameter context，当前大致包含：

```text
lin1 channel norm
lin2 channel norm
channel id
current channel keep probability
current graph keep
bias
```

它的作用是让模型知道：

```text
当前这个 hidden channel 是否更重要
```

### 6.3 Parameter memory read

parameter memory 读出：

```text
raw_param_corr
```

它表示：

```text
历史 memory 对当前 channel pruning score 的原始修正
```

## 7. Scale Alignment

这是当前 method 里非常关键的一步。

前面虽然做了 channel-specific memory，但是如果 memory correction 的数值尺度和 `param_logits` 不一致，就会出现：

```text
memory 明明有信息，但加到 param score 后要么几乎没影响，
要么影响过强，导致 param mask 震荡。
```

所以当前版本把：

```text
channel memory correction
recall correction
param_logits
```

统一到了同一个动态尺度里。

当前 parameter score 的核心形式是：

```python
param_score =
    param_logits
  + param_gamma * param_logit_scale_ema * normalized_channel_memory
  + recall_gamma * param_logit_scale_ema * normalized_recall
```

这个设计的含义是：

```text
param_logits 是主坐标系
memory 和 recall 只是对这个主坐标系做稳定修正
```

这一步解决了：

```text
channel memory 生效后 param mask 容易剧烈震荡的问题
```

## 8. Mask 生成

当前 OUGP 不是让 mask 自由稀疏，而是使用 fixed-budget pruning：

```text
mean(graph_mask) ≈ graph_target_keep
mean(param_mask) ≈ param_target_keep
```

默认主实验目标通常是：

```text
graph sparsity = 30%
parameter sparsity = 30%
```

也就是 keep rate 大致从 warmup 期间的 `1.0` 逐步下降到 `0.7`。

## 9. Forward

以当前主版本 2-layer GCN 为例：

```text
输入特征
  -> graph convolution
  -> linear / hidden transform
  -> ReLU
  -> 乘 param_mask，剪 hidden channel
  -> dropout
  -> 第二层图传播与输出
```

因此：

```text
graph_mask 控制 message passing 结构
param_mask 控制 hidden representation 通道容量
```

## 10. Utility 与 Memory Write

forward/backward 之后，当前实现会从梯度中构造 utility signal。

### 10.1 Graph utility

当前近似形式：

```python
graph_utility = abs(edge_logits.grad * last_graph_mask)
```

### 10.2 Parameter utility

当前近似形式：

```python
param_utility = abs(param_logits.grad * last_param_mask)
```

这些 utility 会写回 memory，用来更新历史状态。

支持的写入模式包括：

```text
residual   主方法，写 utility residual 调制后的值
feature    只写 feature value，不写 residual utility
none       只读不写
```

这对应了我们的方法消融：

```text
memory 是否真的在写
写 residual 是否比只写 feature 更好
```

## 11. Recall / Event Memory

除了主 graph/parameter memory 之外，现在还接入了：

```text
event memory
recall memory
```

它们的作用更偏向：

```text
记录曾经剪掉的结构/通道的历史影响
在后续剪枝时对 score 做保护或修正
```

目前它们已经进入训练流程，但从现有实验看，真正最关键的提升还是：

```text
channel-specific memory + scale alignment
```

## 12. 当前最核心的进展

如果只抓最重要的 method 变化，可以概括成三点：

### 12.1 从全局 parameter memory 到 channel-specific memory

原来 memory 近似是一个 global bias。  
现在每个 channel 都有自己的 memory slot，所以 parameter pruning 真正有了 channel-level discrimination。

### 12.2 从不统一尺度到 scale-aligned parameter score

现在 channel memory correction、recall correction、`param_logits` 都放到了同一坐标系里，所以 memory 不再轻易把系统推到震荡状态。

### 12.3 从只做小图到开始建立大图入口

当前大图路线已经有三层入口：

```text
1. sampled-edge smoke
2. node-sampled induced subgraph
3. frontier node-subgraph
```

其中 frontier 模式比纯随机抽节点更接近真实局部图结构。

## 13. 当前验证到哪里了

### 13.1 小图与中图

已经在这些数据集上完成主版本验证：

```text
Cora
CiteSeer
PubMed
Amazon Photo
```

### 13.2 Backbone

已经支持并做过 smoke / validation：

```text
GCN
GraphSAGE
GAT
DeeperGCN
```

### 13.3 大图

已经开始验证：

```text
ogbn-arxiv
```

当前还属于：

```text
subgraph smoke / feasibility stage
```

不是正式的 mini-batch 大图训练结论。

## 14. 当前方法的阶段性结论

现在这版 OUGP 可以总结为：

```text
一个统一的 graph pruning + channel pruning 框架
+ 用 online memory 记录历史剪枝反馈
+ graph 分支修正 edge pruning
+ parameter 分支修正 channel pruning
+ 通过 channel-specific memory 提供参数级区分能力
+ 通过 scale alignment 保证修正信号稳定进入 param score
+ 已经从小图验证推进到大图 subgraph smoke
```

## 15. 目前还没完全做到的部分

和最初 idea 相比，当前还没完整做到的主要是：

```text
真正的 mini-batch / neighbor-sampled 大图训练
Subgraph-State Write
Multi-State Write
更完整的真实 latency / FLOPs benchmark
```

所以当前版本更准确的定位是：

```text
核心机制已经成型；
小图和中图已经稳定；
大图路线已经打通入口；
但离最终完整论文版 method 还有最后一段工程和实验要补。
```

