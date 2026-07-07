# EXP030: Static Baseline Smoke

## 目的

补齐 idea 中的 baseline 支持，使 OUGP 可以和普通静态 graph/parameter pruning 准则对比。

新增静态 baseline：

```text
random_static
degree_magnitude_static
similarity_magnitude_static
```

这组是 Cora 上的 4 epoch smoke，只验证 baseline variants、score 初始化、冻结和结果落盘，不作为性能结论。

## 设置

```text
dataset: Cora
backbone: GCN
variants:
  random_static
  degree_magnitude_static
  similarity_magnitude_static
  ougp
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
experiments/exp030_static_baseline_smoke/
```

## Baseline 定义

| Variant | Graph Score Init | Param Score Init | Frozen Scores |
| --- | --- | --- | --- |
| random_static | random | random | yes |
| degree_magnitude_static | degree | magnitude | yes |
| similarity_magnitude_static | similarity | magnitude | yes |
| ougp | constant | constant | no |

## 结果

| Variant | Best Test Acc | Graph Sparsity | Param Sparsity | Message Cost Red. | Param Cost Red. | Graph Churn | Param Churn |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| random_static | 0.5220 | 0.200 | 0.200 | 0.161 | 0.200 | 0.067 | 0.101 |
| degree_magnitude_static | 0.5180 | 0.200 | 0.200 | 0.161 | 0.200 | 0.067 | 0.073 |
| similarity_magnitude_static | 0.5200 | 0.200 | 0.200 | 0.161 | 0.200 | 0.067 | 0.073 |
| ougp | 0.5190 | 0.200 | 0.200 | 0.161 | 0.200 | 0.067 | 0.067 |

## 观察

```text
三个静态 baseline 都能正常运行并落盘。
static baseline 的 graph/param score init 和 freeze_pruning_scores 已写入 JSON/CSV/manifest。
4 epoch 太短，accuracy 差异不能解释方法优劣。
```

## 后续正式实验

```text
1. 在 Cora / CiteSeer / PubMed 上跑多 seed。
2. 在 Amazon Photo 上跑多 seed。
3. 对比 static baseline、dual_static、OUGP no-cross、OUGP。
4. 报告 accuracy、sparsity、cost reduction、churn 和 memory overhead。
```
