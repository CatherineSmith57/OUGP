# Amazon Photo 中参数剪枝问题分析：一个可能的通病

日期：2026-07-06

## 一句话结论

Amazon Photo 上的问题不是 OUGP 不能做稀疏化，而是 **parameter-side pruning 对模型表达能力的伤害明显大于 graph pruning**。这很可能不是单个实验的偶然现象，而是图神经网络里“结构剪枝”和“参数/通道剪枝”不对称的一类通病。

需要谨慎表述：

> 目前我们已经观察到 Amazon Photo 上 parameter pruning 的稳定伤害；它“可能是通病”，但还需要在更多数据集、模型和剪枝策略上验证。

## 背景

当前模型是 `OUGPGCN`，也就是 **两层 GCN + graph mask + hidden-channel/parameter-side mask + online memory**。

相关代码：

```text
src/ougp/model.py
scripts/run_case_study.py
```

当前 `parameter sparsity` 更准确地说是 **hidden channel sparsity / parameter-side sparsity**：

- 它不是逐元素剪掉每个 weight。
- 它是在 GCN hidden dimension 上乘一个 `param_mask`。
- 代码位置是 `h = F.relu(h) * param_mask`。
- 因此它会直接减少 hidden representation 的有效维度，也就是减少模型表达能力。

所以它和 graph pruning 的性质不同：

| 剪枝类型 | 剪掉什么 | 主要影响 |
| --- | --- | --- |
| graph pruning | 边 / message passing | 可能减少噪声边、减少传播成本 |
| parameter pruning | hidden channel / 参数侧表达能力 | 可能压缩模型容量、损失特征组合能力 |

## 现象证据

### EXP006：30% / 30% 双剪枝

Amazon Photo 3-seed 结果：

| Variant | Best Test Acc | Graph Sparsity | Param Sparsity |
| --- | ---: | ---: | ---: |
| dense | 0.8125 +/- 0.0161 | 0.000 | 0.000 |
| graph_only | 0.8273 +/- 0.0095 | 0.300 | 0.000 |
| param_only | 0.7121 +/- 0.0124 | 0.000 | 0.300 |
| dual_static | 0.7161 +/- 0.0137 | 0.300 | 0.300 |
| ougp_no_cross | 0.7513 +/- 0.0146 | 0.300 | 0.300 |
| ougp | 0.7509 +/- 0.0136 | 0.300 | 0.300 |

观察：

- `graph_only` 比 dense 更好：`0.8273` vs `0.8125`。
- `param_only` 明显变差：`0.7121`。
- `ougp` 比 `dual_static` 好：`0.7509` vs `0.7161`，说明 online memory 能补救一部分损失。
- 但 `ougp` 仍明显低于 `graph_only`，说明主要瓶颈来自 parameter-side pruning。

### EXP010：parameter sparsity sweep

固定 graph sparsity = 30%，改变 parameter sparsity：

| Target Param Sparsity | param_only | dual_static | ougp | graph_only |
| ---: | ---: | ---: | ---: | ---: |
| 0.05 | 0.8059 | 0.8061 | 0.8188 | 0.8273 |
| 0.10 | 0.7924 | 0.7973 | 0.8131 | 0.8281 |
| 0.20 | 0.7617 | 0.7644 | 0.7895 | 0.8245 |
| 0.30 | 0.7126 | 0.7157 | 0.7503 | 0.8263 |

观察：

1. `param_only` 随 parameter sparsity 增大近似单调下降。
2. `dual_static` 和 `param_only` 非常接近，说明双剪枝里的主要伤害来自参数侧，而不是 graph 侧。
3. `ougp` 始终高于 `dual_static`，说明 online memory 有缓解作用。
4. `ougp` 始终低于 `graph_only`，说明 memory 还不能完全补回参数容量损失。

## 为什么会出现这种问题

### 1. 参数剪枝直接削弱表示能力

GCN 的 hidden channel 负责把节点特征投影到可分类的表示空间。当前 parameter mask 作用在 hidden representation 上：

```text
h = ReLU(GCN_1(x))
h = h * param_mask
```

当 parameter sparsity = 30% 时，相当于让一部分 hidden channels 长期弱化或失效。对于 Amazon Photo 这种特征维度较高、类别信号可能依赖多个特征组合的数据集，这会直接减少可用表达维度。

### 2. graph pruning 可能是去噪，parameter pruning 更像容量压缩

Amazon Photo 是商品共购图。共购边不一定全部有利于分类，有些边可能只是弱相关或噪声相关。

因此：

- 剪掉一部分边，可能让消息传播更干净。
- 剪掉一部分 hidden channel，则可能让模型无法表达必要的类别差异。

这解释了为什么 `graph_only` 可以提升准确率，而 `param_only` 会下降。

### 3. 当前 schedule 对 parameter mask 可能偏激进

当前训练设置：

- epochs: 120
- warmup epochs: 15
- warmup 后逐步达到目标 sparsity

对 graph mask 来说，这可能可以看成逐步去掉冗余边；但对 parameter mask 来说，模型还没充分学稳时就开始压缩 hidden channels，后期即使 memory 尝试修正，也可能已经失去部分表达路径。

### 4. 当前 regularization 只约束达到目标稀疏率，不保证被剪的是冗余通道

代码里的 sparsity loss 主要让 mask 的平均保留率接近目标：

```text
loss = task_loss + sparsity_lambda * regularization
```

这能保证达到稀疏率，但不能保证剪掉的一定是“无用 channel”。如果数据集本身需要较多特征通道，强行达到 20%-30% 参数稀疏率就会伤准确率。

### 5. online memory 能缓解，但不能创造被剪掉的容量

OUGP 的 online memory 可以调整 mask score，让某些重要边或通道更容易被保留。因此它比 `dual_static` 好。

但如果目标稀疏率本身太高，memory 只能在有限预算里重新分配保留位置，不能凭空增加 hidden capacity。所以在 30% parameter sparsity 下，`ougp` 仍然低于 `graph_only`。

## 为什么怀疑这是一个通病

这个现象可能具有普遍性，因为它来自 GNN 中两种资源的本质差异：

1. 图边通常包含冗余或噪声，剪边可能等价于结构正则化。
2. hidden channels 是模型表达空间，剪通道更容易造成容量不足。
3. 中小型 GCN 本身参数量不大，可剪冗余可能有限。
4. 节点特征强、类别细粒度的数据集更可能依赖完整特征变换。
5. 同样的 sparsity 数值不代表同样的风险：30% edge sparsity 和 30% channel sparsity 的伤害机制完全不同。

因此，不能简单假设：

> graph sparsity 可行，所以 parameter sparsity 也同样可行。

更合理的假设是：

> 在 GNN 中，parameter-side pruning 需要更温和的稀疏率、更慢的 schedule、更强的恢复机制，或者更细粒度的剪枝策略；否则它会成为双剪枝系统的主要性能瓶颈。

## 对 OUGP 论文叙事的影响

### 可以说的

可以把 Amazon Photo 写成一个有价值的 case study：

> Amazon Photo reveals an asymmetric pruning behavior: graph pruning improves or preserves accuracy, while parameter-side pruning introduces a sharp accuracy-efficiency trade-off. OUGP partially mitigates this degradation compared with static dual pruning, suggesting the value of online memory under constrained pruning budgets.

中文意思：

> Amazon Photo 暴露了图剪枝和参数剪枝的不对称性：图剪枝可以保准确率甚至提升准确率，而参数剪枝存在明显 accuracy-efficiency trade-off。OUGP 相比静态双剪枝能缓解损失，说明 online memory 在受限剪枝预算下有价值。

### 不能过度说的

当前不能说：

- OUGP 在 Amazon Photo 上准确率最好。
- cross-level context 已经被证明有效。
- 30% parameter sparsity 在 Amazon Photo 上是无损压缩。
- 这个问题已经被证明是所有 GNN / 所有数据集上的通病。

## 下一步验证：怎么判断是不是通病

### 实验 A：跨数据集验证

在已有数据集上做相同 parameter sparsity sweep：

```text
Cora
CiteSeer
PubMed
Amazon Photo
ogbn-arxiv 的 param-only 可行部分
```

重点看：

- `param_only` 是否随 sparsity 增大单调下降。
- `graph_only` 是否比 `param_only` 更稳。
- `ougp` 是否稳定高于 `dual_static`。

如果多个数据集都出现类似趋势，才可以更有信心说这是通病。

### 实验 B：只改变 parameter sparsity，不改变 graph sparsity

推荐网格：

```text
0.00, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30
```

目标是找到 Amazon Photo 的折中点。当前看起来 5%-10% 可能还能接受，20%-30% 明显偏强。

### 实验 C：调 schedule

固定 parameter sparsity = 0.10 或 0.20，比较：

```text
warmup_epochs: 15, 30, 60
temp_end: 0.5, 0.8, 1.0
sparsity_lambda: 0.02, 0.04, 0.08
```

如果更慢 schedule / 更弱正则能缓解伤害，说明问题不是“参数剪枝绝对不行”，而是当前训练策略太硬。

### 实验 D：换剪枝粒度

当前是 hidden-channel mask。可以比较：

- channel-level pruning
- layer-wise pruning
- unstructured weight pruning
- input feature dimension pruning

如果 channel-level 最伤，说明问题来自 hidden representation bottleneck。

### 实验 E：换模型容量

固定 sparsity，改变 hidden_dim：

```text
hidden_dim: 32, 64, 128, 256
```

如果 hidden_dim 越大，parameter pruning 伤害越小，说明当前问题是“容量不够剪”，不是 OUGP 思路本身错误。

## 当前最值得做的下一步

优先级最高的是：

1. Amazon Photo 上固定 `param_sparsity=0.10`，调低 `sparsity_lambda`。
2. Amazon Photo 上做更细低稀疏率 sweep：`0.00/0.025/0.05/0.075/0.10`。
3. 在 Cora/CiteSeer/PubMed 上复用同样 sweep，检查是否也存在 parameter pruning 单调伤害。

如果这些实验成立，论文可以形成一个更稳的论点：

> Unified graph-parameter pruning is not simply about applying equal sparsity to both sides. Graph and parameter pruning have asymmetric failure modes; online memory helps allocate pruning budgets adaptively, but parameter-side sparsity must be controlled more carefully.

## 备注：效率优势和当前代码实现的区别

30% graph sparsity 和 30% parameter sparsity 在概念上是明确的效率/存储优势。

但当前实验代码主要是在训练时使用 mask 验证稀疏化行为。要在实际 wall-clock speed 或存储文件大小上得到完整收益，还需要进一步实现：

- 真正删除被剪边后的 sparse graph 存储。
- 真正压缩 hidden channels / weight matrices。
- 使用支持稀疏或压缩结构的推理实现。

所以当前结果更适合写成：

> The method achieves the target sparsity and indicates potential computation/storage savings.

而不是直接声称：

> 当前代码已经在实际运行时间上减少了 30%。

