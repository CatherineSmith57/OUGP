# EXP010: Amazon Photo Parameter Sparsity Sweep

日期：2026-07-06

## 目标

在 EXP006 中，Amazon Photo 上出现了一个很关键的现象：

- `graph_only` 在 30% graph sparsity 下优于 dense。
- `param_only` 在 30% parameter sparsity 下明显伤害准确率。
- `ougp` 能比 `dual_static` 好，说明 online memory 可能在补救双剪枝损失，但没有超过 `dense` / `graph_only`。

本实验专门回答：

> 参数剪枝在 Amazon Photo 上为什么伤害明显？伤害是否随 parameter sparsity 增大而变强？

同时纠正一个表述问题：

> 30% graph sparsity 和 30% parameter sparsity 本身就是效率/存储优势。之前说“没有明显优势”只应理解为：当前 OUGP 在准确率或相对消融项的算法贡献上还没有稳定胜出，而不是说稀疏化没有价值。

## 设置

- dataset: `photo`
- device: `cuda`
- GPU: `CUDA_VISIBLE_DEVICES=1`
- conda env: `tianjiaying`
- variants: `dense`, `graph_only`, `param_only`, `dual_static`, `ougp_no_cross`, `ougp`
- seeds: `0, 1, 2`
- epochs: `120`
- warmup epochs: `15`
- hidden dim: `64`
- memory rank: `16`
- graph sparsity: `0.30`
- parameter sparsity sweep: `0.05, 0.10, 0.20, 0.30`
- graph gamma / param gamma: `2.0`
- write beta: `0.25`
- max GPUs: `1`

配置文件：

```text
configs/photo_param_sparsity_sweep_v1.json
```

完整实验输出：

```text
experiments/exp010_photo_param_sparsity_sweep/
├── s005/
├── s010/
├── s020/
└── s030/
```

整理表：

```text
results/tables/photo_exp010_param_sparsity_sweep_summary.csv
results/tables/photo_exp010_param_sparsity_sweep_summary.md
```

## 命令模板

每个 sparsity 档位单独运行一次，例如 10%：

```bash
cd /home/shizitong/tianjiaying/research/ougp
CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 PYTHONPATH=src \
  /home/shizitong/miniconda3/envs/tianjiaying/bin/python scripts/run_case_study.py \
  --dataset photo \
  --epochs 120 \
  --warmup-epochs 15 \
  --hidden-dim 64 \
  --memory-rank 16 \
  --variants dense graph_only param_only dual_static ougp_no_cross ougp \
  --seeds 0 1 2 \
  --out-dir experiments/exp010_photo_param_sparsity_sweep/s010 \
  --graph-sparsity 0.30 \
  --param-sparsity 0.10 \
  --graph-gamma 2.0 \
  --param-gamma 2.0 \
  --write-beta 0.25 \
  --device cuda \
  --max-gpus 1 \
  --verbose \
  --log-every 20
```

## 结果总表

| Target Param Sparsity | Variant | Best Test Acc | Delta vs Dense | Delta vs Graph-only |
| ---: | --- | ---: | ---: | ---: |
| 0.05 | dense | 0.8124 +/- 0.0158 | 0.0000 | -0.0149 |
| 0.05 | graph_only | 0.8273 +/- 0.0100 | +0.0149 | 0.0000 |
| 0.05 | param_only | 0.8059 +/- 0.0181 | -0.0065 | -0.0214 |
| 0.05 | dual_static | 0.8061 +/- 0.0197 | -0.0063 | -0.0212 |
| 0.05 | ougp_no_cross | 0.8188 +/- 0.0178 | +0.0064 | -0.0084 |
| 0.05 | ougp | 0.8188 +/- 0.0161 | +0.0064 | -0.0084 |
| 0.10 | dense | 0.8124 +/- 0.0160 | 0.0000 | -0.0157 |
| 0.10 | graph_only | 0.8281 +/- 0.0085 | +0.0157 | 0.0000 |
| 0.10 | param_only | 0.7924 +/- 0.0212 | -0.0199 | -0.0356 |
| 0.10 | dual_static | 0.7973 +/- 0.0180 | -0.0151 | -0.0308 |
| 0.10 | ougp_no_cross | 0.8112 +/- 0.0186 | -0.0011 | -0.0168 |
| 0.10 | ougp | 0.8131 +/- 0.0157 | +0.0007 | -0.0150 |
| 0.20 | dense | 0.8123 +/- 0.0160 | 0.0000 | -0.0122 |
| 0.20 | graph_only | 0.8245 +/- 0.0161 | +0.0122 | 0.0000 |
| 0.20 | param_only | 0.7617 +/- 0.0167 | -0.0507 | -0.0629 |
| 0.20 | dual_static | 0.7644 +/- 0.0170 | -0.0479 | -0.0601 |
| 0.20 | ougp_no_cross | 0.7903 +/- 0.0173 | -0.0221 | -0.0343 |
| 0.20 | ougp | 0.7895 +/- 0.0163 | -0.0228 | -0.0350 |
| 0.30 | dense | 0.8123 +/- 0.0160 | 0.0000 | -0.0139 |
| 0.30 | graph_only | 0.8263 +/- 0.0092 | +0.0139 | 0.0000 |
| 0.30 | param_only | 0.7126 +/- 0.0123 | -0.0997 | -0.1136 |
| 0.30 | dual_static | 0.7157 +/- 0.0138 | -0.0966 | -0.1105 |
| 0.30 | ougp_no_cross | 0.7506 +/- 0.0136 | -0.0617 | -0.0757 |
| 0.30 | ougp | 0.7503 +/- 0.0119 | -0.0620 | -0.0760 |

## 关键观察

1. 参数剪枝伤害随稀疏率增大而单调变强。

`param_only` 从 5% 的 `0.8059` 下降到 30% 的 `0.7126`。这说明 Amazon Photo 上的性能下降不是偶然噪声，而是和 parameter sparsity 强度直接相关。

2. graph pruning 和 parameter pruning 的作用方向不同。

`graph_only` 在各档位附近稳定在 `0.8245-0.8281`，高于 dense 的约 `0.8123`。这说明 Amazon Photo 可能存在有害边或冗余边，剪图边可以改善泛化；但剪参数会削弱模型表达能力。

3. OUGP 的 online memory 有补救作用，但不足以完全抵消高参数稀疏率的伤害。

在 30% parameter sparsity 下：

- `dual_static`: `0.7157`
- `ougp`: `0.7503`

OUGP 比静态双剪枝高约 `+0.0345`，说明 memory/update 机制确实有价值。但它仍低于 dense 和 graph-only，所以不能宣称“完整 OUGP 在 Amazon Photo 上准确率最好”。

4. cross-level context 目前贡献不明显。

`ougp` 和 `ougp_no_cross` 在所有档位非常接近，有时甚至略低。这说明当前 cross 机制还没有稳定带来额外收益，后续如果要写成论文贡献，需要继续改机制或做更细的消融。

## 当前结论

可以支持的结论：

> Amazon Photo 上，30% graph sparsity 本身是有效的：它减少图计算并且准确率不降反升。parameter sparsity 则存在明显 accuracy-efficiency trade-off：5% 几乎可接受，10% 勉强，20% 和 30% 明显伤害性能。OUGP 的 online memory 能缓解双剪枝带来的性能下降，但还没有超过 graph-only baseline。

不应该过度声称的结论：

> 不应说 OUGP 在 Amazon Photo 上全面优于 dense 或 graph-only；目前更准确的说法是，OUGP 在双剪枝设置下优于静态双剪枝，说明动态记忆机制有补救价值。

## 为什么参数剪枝在 Amazon Photo 上伤害明显

目前最可能的解释是：

1. Amazon Photo 的节点特征维度较高，类别信号可能更依赖特征通道组合。
2. 当前 parameter pruning 剪的是模型参数/通道表达能力，不只是去掉冗余计算；当剪到 20%-30% 时，模型容量不够。
3. 图剪枝可以去掉噪声边，相当于结构正则化；参数剪枝则更像压缩模型容量，两者不是同一种收益。
4. 当前 warmup 和 sparsity schedule 可能对 parameter mask 太激进，导致模型还没学稳就逐步失去表达通道。

## 下一步

推荐优先做两个小实验：

1. 固定 parameter sparsity = 10%，调低 `sparsity_lambda`，看是否能让 OUGP 接近 graph-only。
2. 固定 graph sparsity = 30%，只对 `ougp` / `ougp_no_cross` 做 parameter sparsity 更细网格：`0.00, 0.025, 0.05, 0.075, 0.10`，找 accuracy-efficiency 折中点。
