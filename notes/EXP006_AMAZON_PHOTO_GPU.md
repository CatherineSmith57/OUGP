# EXP006: Amazon Photo GPU Case Study

日期：2026-07-06

## 目标

在 Cora / CiteSeer / PubMed 之后，加入 Amazon Photo，检查 OUGP 在更偏商品共购网络的数据集上是否有效。

本轮用空闲 GPU 2 跑完：

- dataset: `photo`
- GPU: `CUDA_VISIBLE_DEVICES=2`
- device: `cuda`
- variants: `dense`, `graph_only`, `param_only`, `dual_static`, `ougp_no_cross`, `ougp`
- seeds: `0, 1, 2`
- epochs: `120`
- warmup epochs: `15`
- hidden dim: `64`
- memory rank: `16`
- graph sparsity: `0.30`
- param sparsity: `0.30`
- graph gamma / param gamma: `2.0`
- write beta: `0.25`

## 数据

Amazon Photo 使用 `gnn-benchmark` 的公开 NPZ 文件：

```text
data/raw/amazon/photo/raw/amazon_electronics_photo.npz
```

数据规模：

```text
nodes: 7650
directed edges after symmetrization: 238162
features: 745
classes: 8
split: 765 train / 765 val / 6120 test
```

说明：这个 loader 使用固定 `10% / 10% / 80%` train / val / test split。Amazon Photo 没有复用 Planetoid 固定 split，所以和 Cora/CiteSeer/PubMed 的 split 设定不同，写论文时需要明确说明。

## 代码改动

为本实验新增了轻量 Amazon Photo loader，避免引入 PyG 等重依赖：

```text
src/ougp/data.py
scripts/run_case_study.py
```

新增支持：

```bash
--dataset photo
```

## 命令

```bash
cd /home/shizitong/tianjiaying/research/ougp
CUDA_VISIBLE_DEVICES=2 /home/shizitong/miniconda3/bin/conda run -n tianjiaying env PYTHONPATH=src python scripts/run_case_study.py \
  --dataset photo \
  --epochs 120 \
  --warmup-epochs 15 \
  --hidden-dim 64 \
  --memory-rank 16 \
  --variants dense graph_only param_only dual_static ougp_no_cross ougp \
  --seeds 0 1 2 \
  --out-dir experiments/exp006_photo_gpu_multiseed \
  --graph-gamma 2.0 \
  --param-gamma 2.0 \
  --write-beta 0.25 \
  --device cuda \
  --max-gpus 1 \
  --verbose \
  --log-every 20
```

## 输出位置

完整记录：

```text
experiments/exp006_photo_gpu_multiseed/
```

整理表：

```text
results/tables/photo_exp006_gpu_multiseed_summary.csv
results/tables/photo_exp006_gpu_multiseed_summary.md
```

## 结果

| Variant | Best Test Acc | Graph Sparsity | Param Sparsity | Graph Churn | Param Churn |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense | 0.8125 +/- 0.0161 | 0.000 | 0.000 | 0.000 | 0.000 |
| dual_static | 0.7161 +/- 0.0137 | 0.300 | 0.300 | 0.003 | 0.003 |
| graph_only | 0.8273 +/- 0.0095 | 0.300 | 0.000 | 0.015 | 0.000 |
| ougp | 0.7509 +/- 0.0136 | 0.300 | 0.300 | 0.022 | 0.003 |
| ougp_no_cross | 0.7513 +/- 0.0146 | 0.300 | 0.300 | 0.014 | 0.003 |
| param_only | 0.7121 +/- 0.0124 | 0.000 | 0.300 | 0.000 | 0.003 |

## 初步解读

这次结果很明确：

- `graph_only` 是当前最好结果，超过 dense：`0.8273` vs `0.8125`。
- `param_only` 明显变差：`0.7121`。
- `dual_static` 也明显变差：`0.7161`。
- 完整 `ougp` 比 `dual_static` 好很多：`0.7509` vs `0.7161`，说明 online memory 可能缓解了双剪枝的一部分伤害。
- 但是完整 `ougp` 仍然明显低于 `dense` 和 `graph_only`，所以不能说 OUGP 在 Amazon Photo 上已经有效胜出。
- `ougp` 和 `ougp_no_cross` 几乎一样，cross-level context 仍没有显示稳定收益。

当前最诚实的结论：

> On Amazon Photo, graph pruning alone improves accuracy, but parameter pruning is harmful under the current setting. OUGP partially recovers from the dual-pruning degradation, yet does not outperform dense or graph-only baselines.

## 下一步建议

Amazon Photo 暗示当前问题可能不在 graph pruning，而在 parameter/channel pruning 太强或太早。建议下一步：

- 做 Photo 的 parameter sparsity sweep：`0.05, 0.10, 0.20, 0.30`。
- 固定 graph sparsity 30%，降低 parameter sparsity，看 OUGP 是否能接近 `graph_only`。
- 单独调低 `sparsity_lambda`，避免 parameter mask 训练过度影响分类。
- 做 `graph_only` 的高稀疏率 sweep，确认 Amazon Photo 是否天然受益于 graph sparsification。
