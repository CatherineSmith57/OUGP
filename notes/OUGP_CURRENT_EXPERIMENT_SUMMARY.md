# OUGP 当前实验结果总结

日期：2026-07-06

## 结论先行

目前实验说明：

1. **当前所有主实验都基于 two-layer GCN backbone**：也就是说，目前验证的是 `GCN + OUGP mask/memory`，还没有在 GraphSAGE、GAT 或其他 GNN backbone 上验证。
2. **OUGP 可以稳定实现目标稀疏率**：在 Cora / CiteSeer / PubMed / Amazon Photo 上，当前 GCN 实现都能达到约 `30% graph sparsity` 和 `30% parameter sparsity`。
3. **稀疏化本身是明确优势**：剪掉 30% graph edge/message 和 30% hidden channel/parameter-side capacity，意味着潜在计算、存储和部署成本下降。
4. **OUGP 的准确率优势还不稳定**：在 Cora / CiteSeer / PubMed 上，OUGP 与 dense / ablation 差距很小；在 Amazon Photo 上，OUGP 明显优于静态双剪枝，但仍低于 `graph_only`。
5. **当前最大问题是 parameter pruning**：Amazon Photo 和 ogbn-arxiv 都显示，30% parameter-side sparsity 会伤害准确率。
6. **大图扩展还没解决**：当前 full-batch graph-pruning 实现在 ogbn-arxiv 上 backward OOM，需要 mini-batch / neighbor sampling 版本。

所以目前最诚实的判断是：

> 在 two-layer GCN backbone 上，OUGP 已经证明了“统一图/参数剪枝流程可运行、可达到目标稀疏率，并且 online memory 能在部分场景缓解静态双剪枝损失”。但它还没有证明“完整 OUGP 在准确率上稳定优于 dense、graph-only 或所有消融项”，也还没有证明该现象能泛化到其他 GNN backbone。

## 主结果表

说明：

- `Dense`：普通 two-layer GCN，不剪枝。
- `Best Single Pruning`：`graph_only` 和 `param_only` 中更好的一个。
- `Dual Static`：同时剪 graph 和 parameter，但不开 online memory。
- `OUGP`：完整方法，graph pruning + parameter pruning + online memory + cross context。
- `OUGP vs Dense`：完整 OUGP 相对 dense 的准确率变化。
- `OUGP vs Dual`：完整 OUGP 相对静态双剪枝的变化，主要看 online memory 是否有帮助。

| Dataset | Setting | Dense | Best Single Pruning | Dual Static | OUGP | OUGP vs Dense | OUGP vs Dual | Main Takeaway |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Cora | 30% graph / 30% param, 3 seeds, GPU | 0.8083 +/- 0.0049 | param_only 0.8147 +/- 0.0073 | 0.8090 +/- 0.0114 | 0.8060 +/- 0.0088 | -0.0023 | -0.0030 | OUGP 能达到稀疏率，但准确率不优于消融项 |
| CiteSeer | 30% graph / 30% param, 3 seeds, CPU | 0.7160 +/- 0.0042 | param_only 0.7153 +/- 0.0009 | 0.7150 +/- 0.0036 | 0.7157 +/- 0.0025 | -0.0003 | +0.0007 | 各方法非常接近，OUGP 没有明显优势也没有明显崩 |
| PubMed | 30% graph / 30% param, 3 seeds, CPU | 0.7900 +/- 0.0016 | graph_only 0.7927 +/- 0.0019 | 0.7890 +/- 0.0024 | 0.7890 +/- 0.0024 | -0.0010 | +0.0000 | graph_only 略好，OUGP 与 dual_static 基本一样 |
| Amazon Photo | 30% graph / 30% param, 3 seeds, GPU | 0.8125 +/- 0.0161 | graph_only 0.8273 +/- 0.0095 | 0.7161 +/- 0.0137 | 0.7509 +/- 0.0136 | -0.0616 | +0.0348 | OUGP 能缓解双剪枝损失，但 parameter pruning 伤害很大 |
| ogbn-arxiv | dense vs param_only, 3 seeds, GPU | 0.6051 +/- 0.0017 | param_only 0.5959 +/- 0.0011 | N/A | N/A | N/A | N/A | param pruning 轻微伤害；完整 graph-pruning OUGP full-batch OOM |

## Amazon Photo 参数剪枝 sweep

这组实验专门检查 Amazon Photo 上 parameter pruning 为什么伤害大。固定 graph sparsity = 30%，扫描 parameter sparsity。

| Target Param Sparsity | param_only | dual_static | ougp | graph_only |
| ---: | ---: | ---: | ---: | ---: |
| 0.05 | 0.8059 | 0.8061 | 0.8188 | 0.8273 |
| 0.10 | 0.7924 | 0.7973 | 0.8131 | 0.8281 |
| 0.20 | 0.7617 | 0.7644 | 0.7895 | 0.8245 |
| 0.30 | 0.7126 | 0.7157 | 0.7503 | 0.8263 |

核心现象：

- `param_only` 随 parameter sparsity 增大明显下降：`0.8059 -> 0.7126`。
- `dual_static` 与 `param_only` 很接近，说明双剪枝里的主要伤害来自 parameter-side pruning。
- `ougp` 始终比 `dual_static` 高，说明 online memory 确实能缓解损失。
- `ougp` 仍低于 `graph_only`，说明 memory 还不能完全弥补参数/通道容量损失。

## 实验范围

当前实验范围要明确写成：

```text
Backbone: two-layer GCN
Method: GCN + OUGP graph mask / parameter mask / online memory
Datasets: Cora, CiteSeer, PubMed, Amazon Photo, partial ogbn-arxiv
```

因此现在的结论只应表述为：

> OUGP has been preliminarily validated on a two-layer GCN backbone.

不能表述为：

> OUGP has been validated across different GNN architectures.

如果之后要把结论推广到 GNN 方法层面，需要补 GraphSAGE / GAT / GCNII 等 backbone 实验。

## OUGP 的优势

### 1. 在 GCN 上能稳定达到双稀疏目标

当前实验中，`GCN + OUGP` 在多个数据集上都能稳定达到：

```text
graph_sparsity ~= 0.30
param_sparsity ~= 0.30
```

这说明方法实现不是 toy demo，训练流程、mask 预算控制、结果记录都已经跑通。

### 2. 稀疏化本身有实际价值

即使准确率没有全面超过 dense，30% graph sparsity 和 30% parameter sparsity 仍然是优势：

- graph sparsity：潜在减少 message passing 的边数和图存储。
- parameter sparsity：潜在减少 hidden channel / 参数侧计算和模型容量。
- 对部署场景，稀疏化可以转化为更低内存、更低通信、更低推理成本。

需要注意：当前代码主要验证 mask 稀疏行为。要真正得到 wall-clock 加速和文件大小压缩，还需要后续实现真实压缩/稀疏推理。

### 3. Online memory 在 Amazon Photo 上有明确补救作用

Amazon Photo 30% / 30% 下：

```text
dual_static = 0.7161
ougp        = 0.7509
```

OUGP 比静态双剪枝高约 `+0.0348`。这说明 online memory 不是完全无效，它能在参数剪枝造成严重伤害时补回一部分性能。

这个结果可以作为论文里的一个重要 preliminary finding：

> On a two-layer GCN backbone, online memory can mitigate the degradation of static dual pruning under constrained pruning budgets.

### 4. 暴露了一个有研究价值的问题

Amazon Photo 和 ogbn-arxiv 都显示 parameter-side pruning 比 graph pruning 更危险。这不是坏事，反而可能形成论文动机：

> 统一剪 graph 和 parameter 不能简单使用相同稀疏率，因为二者的 failure mode 不同。graph pruning 可能去噪，parameter pruning 可能损失表达能力。

这能引出后续方法改进：自适应预算、分侧 schedule、不同正则强度、capacity-aware pruning。

## OUGP 的不足

### 1. 完整 OUGP 目前没有稳定超过 dense / graph_only

在 Cora、CiteSeer、PubMed 上，OUGP 与 dense 很接近，但没有明显胜出。

在 Amazon Photo 上，`graph_only` 是最好结果：

```text
graph_only = 0.8273
ougp       = 0.7509
dense      = 0.8125
```

所以现在不能说：

> OUGP 在准确率上全面优于 baseline。

更准确的说法是：

> OUGP 在保持双稀疏目标的同时，部分场景能缓解静态双剪枝损失，但准确率优势尚不稳定。

### 2. Cross-level context 贡献还不明显

`ougp` 和 `ougp_no_cross` 在多个实验里非常接近。

例如 Amazon Photo：

```text
ougp_no_cross = 0.7513
ougp          = 0.7509
```

这说明当前 cross context 还不能作为强 claim。后续需要：

- 改 cross 机制。
- 或者找到它真正有效的数据集/稀疏压力区间。
- 或者在论文中弱化 cross claim，把主要贡献放在 online memory 和 failure-mode analysis 上。

### 3. Parameter pruning 是当前主要瓶颈

Amazon Photo 上，30% parameter sparsity 明显伤害准确率：

```text
dense      = 0.8125
param_only = 0.7121
```

ogbn-arxiv 上也有轻微下降：

```text
dense      = 0.6051
param_only = 0.5959
```

这说明当前 parameter-side pruning 可能过强，或者 hidden-channel pruning 粒度不合适。

### 4. 大图 full-batch 实现不可扩展

ogbn-arxiv 的 graph-pruning backward 失败：

```text
CUDA OOM: tried to allocate 106.83 GiB
```

这说明当前 full-batch OUGP 不能直接跑 `ogbn-products` / `ogbn-proteins` 的完整 graph-pruning 实验。大图需要 mini-batch / neighbor sampling 实现。

### 5. 当前实验还不足以支撑论文主 claim

当前结果适合写 preliminary case study，但还不够支撑完整论文主结论。缺口包括：

- seed 数还可以更多。
- 需要更多数据集上的 parameter sparsity sweep。
- 需要更多 GNN backbone，例如 GraphSAGE / GAT，证明不是 GCN 特例。
- 需要真实效率指标，如 FLOPs、edge count、parameter count、实际推理时间。
- 需要更强 baseline，如标准 GCN、GraphSAGE、GAT 或已有 pruning 方法。

## 当前可以写进汇报的话

比较稳妥的中文表述：

> 目前我们先在 two-layer GCN backbone 上验证 OUGP。实验已经在 Cora、CiteSeer、PubMed 和 Amazon Photo 这类小中规模图上跑通，并能稳定达到 30% graph sparsity 和 30% parameter sparsity；但在 ogbn-arxiv 这类大图上，当前 full-batch GCN + learnable graph mask 的实现会在 graph-pruning backward 阶段 OOM，因此大图完整 OUGP 需要 mini-batch / neighbor sampling 版本。结果显示，统一剪枝本身具备明确的效率/存储潜力；但完整 OUGP 在准确率上还没有稳定超过 dense GCN 或 graph-only baseline。Amazon Photo 暴露了一个关键 failure mode：graph pruning 可能带来去噪收益，而 parameter-side pruning 会显著压缩 GCN 的 hidden-channel 表达能力。OUGP 的 online memory 能缓解静态双剪枝损失，但还不足以完全抵消高参数稀疏率带来的伤害。

英文草稿：

> On a two-layer GCN backbone, OUGP consistently achieves the target graph and parameter sparsity across small and medium-scale graph benchmarks. While the current results do not yet show a stable accuracy advantage over dense GCN or graph-only baselines, Amazon Photo reveals an important asymmetric pruning behavior: graph pruning can improve generalization, whereas parameter-side pruning introduces a strong accuracy-efficiency trade-off by reducing hidden-channel capacity. OUGP partially mitigates the degradation of static dual pruning, suggesting that online memory is useful under constrained pruning budgets.

## 下一步建议

优先级从高到低：

1. **Amazon Photo 低参数稀疏率精扫**

```text
param_sparsity = 0.00, 0.025, 0.05, 0.075, 0.10
```

目标：找到准确率和参数稀疏的折中点。

2. **调低 sparsity_lambda**

固定 `param_sparsity=0.10` 或 `0.20`，尝试：

```text
sparsity_lambda = 0.02, 0.04, 0.08
```

目标：判断参数剪枝伤害是否来自正则太强。

3. **跨数据集做 parameter sparsity sweep**

在 Cora / CiteSeer / PubMed / Amazon Photo 上统一扫描 parameter sparsity。目标是验证“parameter pruning failure mode 是否是通病”。

4. **实现 sampled OUGP**

为 OGB 大图实现 mini-batch / neighbor sampling 版本，否则完整 graph-pruning OUGP 无法扩展到大图。

5. **补其他 GNN backbone**

至少补：

```text
GraphSAGE
GAT
```

目标：判断 OUGP 的优势/不足是 GCN 特例，还是更一般的 GNN pruning 现象。

6. **补真实效率指标**

除了 accuracy 和 sparsity，还应统计：

```text
remaining edges
remaining hidden channels
parameter count
estimated FLOPs
inference time
GPU memory
```

这样才能把“稀疏率优势”进一步转化成论文里的“计算/存储优势”。
