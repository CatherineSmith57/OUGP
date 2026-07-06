# EXP003/EXP004: CiteSeer and PubMed Cross-Dataset Case Studies

日期：2026-07-06

## 目标

在 Cora multi-seed case study 之后，继续跑 CiteSeer 和 PubMed，检查当前 OUGP 现象是否跨数据集存在。

本轮保持和 Cora `exp002_cora_gpu_multiseed` 尽量一致的设置：

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

## 运行环境说明

Conda 环境：

```text
tianjiaying
```

本轮没有使用 `atma`。

GPU preflight 时，0-2 号 GPU 被大显存任务占用，3-7 号 GPU 被其他用户任务占用，虽然显存只占约 844 MiB，但 GPU utilization 较高。为了不和别人的训练硬挤，本轮 CiteSeer / PubMed 使用 CPU 跑完。

因此这两个 case study 的作用是：

- 用 `tianjiaying` 环境验证跨数据集结果。
- 作为论文前的初步证据。
- 后续如果 GPU 空出来，可以用相同参数复跑 GPU 版本。

## 数据下载记录

第一次 CiteSeer 启动失败，原因是脚本内 `urllib` 下载 GitHub raw 文件超时。

失败日志：

```text
experiments/exp003_citeseer_cpu_multiseed/run_failed_download_timeout.log
```

之后已用 `curl` 带重试下载 CiteSeer / PubMed Planetoid raw 文件到：

```text
data/raw/planetoid/citeseer/raw/
data/raw/planetoid/pubmed/raw/
```

## EXP003: CiteSeer

完整记录：

```text
experiments/exp003_citeseer_cpu_multiseed/
```

整理表：

```text
results/tables/citeseer_exp003_cpu_multiseed_summary.csv
results/tables/citeseer_exp003_cpu_multiseed_summary.md
```

命令摘要：

```bash
/home/shizitong/miniconda3/bin/conda run -n tianjiaying env PYTHONPATH=src python scripts/run_case_study.py \
  --dataset citeseer \
  --epochs 120 \
  --warmup-epochs 15 \
  --hidden-dim 64 \
  --memory-rank 16 \
  --variants dense graph_only param_only dual_static ougp_no_cross ougp \
  --seeds 0 1 2 \
  --out-dir experiments/exp003_citeseer_cpu_multiseed \
  --graph-gamma 2.0 \
  --param-gamma 2.0 \
  --write-beta 0.25 \
  --device cpu
```

结果：

| Variant | Best Test Acc | Graph Sparsity | Param Sparsity | Graph Churn | Param Churn |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense | 0.7160 +/- 0.0042 | 0.000 | 0.000 | 0.000 | 0.000 |
| dual_static | 0.7150 +/- 0.0036 | 0.300 | 0.300 | 0.003 | 0.003 |
| graph_only | 0.7143 +/- 0.0009 | 0.300 | 0.000 | 0.009 | 0.000 |
| ougp | 0.7157 +/- 0.0025 | 0.300 | 0.300 | 0.011 | 0.003 |
| ougp_no_cross | 0.7160 +/- 0.0022 | 0.300 | 0.300 | 0.013 | 0.003 |
| param_only | 0.7153 +/- 0.0009 | 0.000 | 0.300 | 0.000 | 0.003 |

## EXP004: PubMed

完整记录：

```text
experiments/exp004_pubmed_cpu_multiseed/
```

整理表：

```text
results/tables/pubmed_exp004_cpu_multiseed_summary.csv
results/tables/pubmed_exp004_cpu_multiseed_summary.md
```

命令摘要：

```bash
/home/shizitong/miniconda3/bin/conda run -n tianjiaying env PYTHONPATH=src python scripts/run_case_study.py \
  --dataset pubmed \
  --epochs 120 \
  --warmup-epochs 15 \
  --hidden-dim 64 \
  --memory-rank 16 \
  --variants dense graph_only param_only dual_static ougp_no_cross ougp \
  --seeds 0 1 2 \
  --out-dir experiments/exp004_pubmed_cpu_multiseed \
  --graph-gamma 2.0 \
  --param-gamma 2.0 \
  --write-beta 0.25 \
  --device cpu
```

结果：

| Variant | Best Test Acc | Graph Sparsity | Param Sparsity | Graph Churn | Param Churn |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense | 0.7900 +/- 0.0016 | 0.000 | 0.000 | 0.000 | 0.000 |
| dual_static | 0.7890 +/- 0.0024 | 0.300 | 0.300 | 0.003 | 0.003 |
| graph_only | 0.7927 +/- 0.0019 | 0.300 | 0.000 | 0.005 | 0.000 |
| ougp | 0.7890 +/- 0.0024 | 0.300 | 0.300 | 0.008 | 0.003 |
| ougp_no_cross | 0.7900 +/- 0.0036 | 0.300 | 0.300 | 0.006 | 0.003 |
| param_only | 0.7913 +/- 0.0017 | 0.000 | 0.300 | 0.000 | 0.003 |

## 跨数据集初步解读

当前三组数据集的共同现象：

- OUGP 训练流程在 Cora / CiteSeer / PubMed 上都能跑通。
- 30% graph sparsity 和 30% parameter sparsity 都能稳定达到。
- 剪枝后准确率整体没有崩，说明当前实现作为 pruning case study 是可用的。

但当前还不能支持强 claim：

- `ougp` 没有稳定超过 `dual_static`。
- `ougp` 没有稳定超过 `ougp_no_cross`。
- cross-level context 的收益还不清楚。
- 在 PubMed 上，`graph_only` 和 `param_only` 反而略好于完整 OUGP。

当前最诚实的论文表述应该是：

> Across Cora, CiteSeer, and PubMed, OUGP can perform joint graph/parameter pruning while preserving accuracy under a 30% sparsity target, but the current online memory and cross-level coupling components do not yet show a consistent advantage over static or single-level pruning baselines.

## 下一步建议

不要继续只在 30% sparsity 上微调 Cora。更应该做：

- 更高稀疏率：50% / 50%，甚至 70% / 70%，看压力下 OUGP memory 是否出现优势。
- 超参扫描：`write_beta`, `graph_gamma`, `param_gamma`, `memory_rank`。
- 更明确的 memory 消融：no write, no read, EMA utility, residual delta write。
- 如果 GPU 空闲，优先用相同参数复跑 PubMed GPU 版本，用来确认 CPU/GPU 数值一致性和加速后续 sweep。
