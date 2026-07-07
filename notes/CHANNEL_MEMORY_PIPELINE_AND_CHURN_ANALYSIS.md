# Channel Memory Pipeline and Churn Analysis

## 一句话结论

Channel-specific memory 已经实现了“每个 channel 有自己的记忆”，但当前 pipeline 里 memory correction 被标准化后直接加到 parameter pruning score 上。如果 `param_gamma` 过大，memory 会主导剪枝排序，导致 parameter mask 每轮大幅变化，也就是我们看到的过度震荡。

这不是“特殊化无效”，而是“特殊化后反馈变强了，但还没有控制反馈强度”。

## 为什么特殊化后反而更容易震荡？

### 1. 原来的问题是 memory 太像 global bias

旧版 parameter memory 把所有 channel 的信息写进同一个全局 state。这样输出容易接近：

```text
param_corr = [14.8, 14.8, 14.8, ...]
```

这种 correction 对 fixed-budget pruning 几乎没有排序作用。因为我们的 mask 使用固定稀疏预算，核心看的是 channel 之间的相对排序，而不是整体加了多少常数。

### 2. 新版特殊化后，memory 终于能区分 channel

现在 `ChannelPruningMemory` 的 state 是：

```text
[num_channels, memory_rank, memory_rank]
```

也就是每个 hidden channel 都有自己的 memory slot。读取时：

```text
channel i 的 context 只读 channel i 的 memory state
```

写入时：

```text
channel i 的 utility 只更新 channel i 的 memory state
```

所以它确实能产生 channel-level correction。

### 3. 但 correction 被标准化到了很强的尺度

当前代码里，parameter memory 读出来后会做：

```python
param_corr = raw_param_corr - raw_param_corr.mean()
param_corr = param_corr / param_corr.std()
```

这样做的好处是：去掉全局 bias，只保留 channel 之间的相对差异。

但副作用是：无论 raw correction 原本多小，都会被放大到接近：

```text
param_memory_correction_std ~= 1
```

然后 pruning score 里用：

```python
param_score = param_logits + param_gamma * param_corr + recall_gamma * recall_corr
```

如果 `param_gamma=2.0`，memory 对 score 的扰动就是大约 `2.0` 的量级。

而 Amazon Photo 实验里，训练后 `param_logits.std` 大约只有 `0.15 ~ 0.18`。也就是说，memory correction 比模型自己学出来的 channel score 差异大很多，parameter mask 就会被 memory 主导。

### 4. fixed-budget pruning 对排序变化很敏感

我们的 `budgeted_sigmoid()` 会根据 score 的 quantile 设 threshold。也就是说，它不是问“这个 channel 绝对分数高不高”，而是问：

```text
哪些 channel 排名前 70%？
哪些 channel 排名后 30%？
```

所以只要 memory correction 改变了 channel 排序，mask 就会变。  
当 correction 每轮都比较强时，mask 就容易来回换，表现为 `param_churn` 变大。

### 5. parameter utility 本身也比较噪

parameter memory 的写入信号来自：

```python
param_utility = abs(param_logits.grad * last_param_mask)
```

这表示“当前这一轮里，某个 channel 的 pruning logit 对 loss 的梯度影响”。这个信号有意义，但它是 per-epoch 的局部梯度信号，不一定稳定代表长期重要性。

所以当我们给每个 channel 单独记忆后，它确实更敏感了；但如果不加平滑/限幅/温和权重，敏感性就会变成震荡。

## 当前训练 pipeline

下面是当前 OUGP 的大致流程。

### Step 1: 进入一个 epoch

训练脚本先根据 epoch 设置当前目标 sparsity：

```text
graph_target_keep = 从 1.0 逐步 warmup 到 0.7
param_target_keep = 从 1.0 逐步 warmup 到 0.7
temperature       = 从 temp_start 退火到 temp_end
```

也就是前期少剪，后期达到 30% graph sparsity 和 30% parameter sparsity。

### Step 2: 构造 graph context 和 parameter context

Graph context 是 edge-level 的：

```text
[src node feature projection,
 dst node feature projection,
 src degree,
 dst degree,
 current parameter keep]
```

Parameter context 是 channel-level 的：

```text
[lin1 channel norm,
 lin2 channel norm,
 channel id,
 current channel keep probability,
 current graph keep,
 bias]
```

所以 graph memory 以 edge 为单位，parameter memory 以 hidden channel 为单位。

### Step 3: memory read

Graph memory 输出：

```text
graph_corr: 每条边的修正分数
```

Parameter channel memory 输出：

```text
raw_param_corr: 每个 channel 的原始修正分数
```

然后当前代码会把 parameter correction 做：

```text
zero-mean + std normalization + clamp
```

得到：

```text
param_corr
```

### Step 4: 生成 pruning score

当前 graph score：

```python
graph_score =
    edge_logits
  + graph_gamma * graph_corr
  + event_gamma * graph_event_bias
  + recall_gamma * graph_recall_bias
```

当前 parameter score：

```python
param_score =
    param_logits
  + param_gamma * param_corr
  + recall_gamma * param_recall_bias
```

其中 `param_gamma` 决定 channel memory 对参数剪枝的影响强度。

### Step 5: 生成 mask

用 fixed-budget pruning：

```text
graph_score -> budgeted_sigmoid -> graph_mask
param_score -> budgeted_sigmoid -> param_mask
```

它会尽量让：

```text
mean(graph_mask) ~= graph_target_keep
mean(param_mask) ~= param_target_keep
```

所以最终稀疏率稳定在 30%，但具体剪哪些边/哪些 channel 由 score 排序决定。

### Step 6: GCN forward

当前模型是 2-layer GCN：

```text
graph_mask 控制图边
param_mask 控制 hidden channel
```

大致是：

```text
X
-> graph convolution with pruned graph
-> Linear layer 1
-> ReLU
-> multiply param_mask
-> dropout
-> graph convolution
-> Linear layer 2
-> logits
```

所以 parameter mask 实际影响的是 hidden representation 的 channel。

### Step 7: loss backward

训练 loss 是：

```text
task_loss + sparsity regularization
```

然后反向传播，得到：

```text
edge_logits.grad
param_logits.grad
```

### Step 8: memory write

memory 根据梯度 utility 更新：

```python
graph_utility = abs(edge_logits.grad * last_graph_mask)
param_utility = abs(param_logits.grad * last_param_mask)
```

Graph memory 用 `graph_utility` 写入 graph memory、event memory 和 graph recall memory。

Parameter memory 用 `param_utility` 写入 channel-specific memory 和 parameter recall memory。

### Step 9: optimizer step

最后 optimizer 更新普通可学习参数：

```text
GCN weights
edge_logits
param_logits
memory projection/read/write heads
```

memory 的 state 本身是 buffer，不是普通 optimizer 参数；它通过 `write()` 手动更新。

## 实验证据

EXP020 的 `param_gamma` sweep 说明了这个机制：

| param_gamma | Best Test Acc | Param Churn | 解释 |
| ---: | ---: | ---: | --- |
| 0.1 | 0.7495 +/- 0.0134 | 0.005 | 稳定，接近旧 OUGP |
| 0.2 | 0.7480 +/- 0.0169 | 0.022 | 稍强，但还可控 |
| 0.5 | 0.7153 +/- 0.0161 | 0.125 | 反馈过强，mask 大幅震荡 |
| 1.0 | 0.7295 +/- 0.0314 | 0.119 | 反馈过强，seed 方差变大 |
| 2.0 | 0.7281 +/- 0.0170 | 0.114 | EXP019 设置，反馈过强 |

这说明：channel-specific memory 的确影响了 pruning decision。否则调 `param_gamma` 不会让 `param_churn` 从 `0.005` 变到 `0.12` 左右。

## 现在应该怎么理解这个结果？

不要把它理解成：

```text
channel memory 没用
```

更准确的理解是：

```text
channel memory 已经能区分 channel；
但 correction 的尺度和 param_gamma 还没有校准好；
过大的 correction 会让 parameter mask 频繁变化，从而破坏 GCN 表示学习。
```

也就是说，我们已经从第一个问题：

```text
memory 没有 channel-level discrimination
```

推进到了第二个问题：

```text
有 discrimination，但 feedback controller 太激进
```

## 下一步建议

优先把 `param_gamma=0.1` 作为当前稳定默认设置。

接下来不要继续增大 `param_gamma`，而应该测试更温和的 correction 进入方式：

1. 只做 mean-centering，不强制 `std=1`。
2. 给 parameter correction 加 EMA 平滑。
3. 对 `param_corr` 做更小的 clamp，例如 `[-1, 1]`。
4. 让 `param_gamma` warmup，从 0 慢慢升到 0.1。
5. 把 recall 和 channel memory 分开调，避免两个反馈同时推同一批 channel。

当前最重要的指标不只是 accuracy，还要同时看：

```text
param_memory_raw_correction_std
param_memory_correction_std
param_logits_std
param_churn
best_test_acc
```

理想状态是：

```text
param_churn 不要太高；
param_corr 有区分度；
accuracy 至少不低于旧 OUGP；
最好逐步靠近 graph_only。
```
