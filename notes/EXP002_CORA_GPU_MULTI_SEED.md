# EXP002: Cora GPU Multi-Seed Case Study

日期：2026-07-06

## 目标

在修复 `tianjiaying` GPU 环境后，用 1 张 GPU 跑一个比初始 case study 更稳的小规模进阶版本：

- 从单 seed 扩展到 3 个 seed：`0, 1, 2`
- epoch 从 `80` 增加到 `120`
- hidden dim 从 `32` 增加到 `64`
- memory rank 从 `8` 增加到 `16`
- 保持 Cora 和 30% graph / parameter sparsity，不一次性把实验变得太重

说明：这个脚本是全图 GCN 训练，Cora 图很小，所以没有 batch size 参数。这里“调大一点点”对应的是 epoch、hidden dim、memory rank 和 seed 数。

## 环境与 GPU

- Conda 环境：`tianjiaying`
- PyTorch：`2.7.1+cu118`
- CUDA build：`11.8`
- GPU：`CUDA_VISIBLE_DEVICES=3`
- 实际可见 GPU 数：`1`
- 运行限制：`--max-gpus 1`

## 命令

```bash
cd /home/shizitong/tianjiaying/research/ougp
CUDA_VISIBLE_DEVICES=3 /home/shizitong/miniconda3/bin/conda run -n tianjiaying env PYTHONPATH=src python scripts/run_case_study.py \
  --dataset cora \
  --epochs 120 \
  --warmup-epochs 15 \
  --hidden-dim 64 \
  --memory-rank 16 \
  --variants dense graph_only param_only dual_static ougp_no_cross ougp \
  --seeds 0 1 2 \
  --out-dir experiments/exp002_cora_gpu_multiseed \
  --graph-gamma 2.0 \
  --param-gamma 2.0 \
  --write-beta 0.25 \
  --device cuda \
  --max-gpus 1 \
  --verbose \
  --log-every 20
```

## 输出位置

完整实验记录：

```text
experiments/exp002_cora_gpu_multiseed/
```

其中：

- `command.txt`：完整命令
- `manifest.json`：参数、输出路径、最终指标
- `run.log`：训练日志
- `run_failed_missing_scipy.log`：第一次启动失败日志，原因是 `tianjiaying` 缺 `scipy`
- `cora_*_seed*.json`：每个 variant/seed 的完整 history
- `cora_summary.csv` / `cora_summary.md`：本次实验汇总

论文整理表：

```text
results/tables/cora_exp002_gpu_multiseed_summary.csv
results/tables/cora_exp002_gpu_multiseed_summary.md
```

## 结果

| Variant | Best Test Acc | Graph Sparsity | Param Sparsity | Graph Churn | Param Churn |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense | 0.8083 +/- 0.0049 | 0.000 | 0.000 | 0.000 | 0.000 |
| dual_static | 0.8090 +/- 0.0114 | 0.300 | 0.300 | 0.003 | 0.003 |
| graph_only | 0.8083 +/- 0.0084 | 0.300 | 0.000 | 0.011 | 0.000 |
| ougp | 0.8060 +/- 0.0088 | 0.300 | 0.300 | 0.007 | 0.003 |
| ougp_no_cross | 0.8070 +/- 0.0083 | 0.300 | 0.300 | 0.008 | 0.003 |
| param_only | 0.8147 +/- 0.0073 | 0.000 | 0.300 | 0.000 | 0.003 |

每个 seed 的 best test acc：

| Variant | Seed 0 | Seed 1 | Seed 2 |
| --- | ---: | ---: | ---: |
| dense | 0.8090 | 0.8020 | 0.8140 |
| dual_static | 0.8090 | 0.7950 | 0.8230 |
| graph_only | 0.8110 | 0.7970 | 0.8170 |
| ougp | 0.8090 | 0.7940 | 0.8150 |
| ougp_no_cross | 0.8090 | 0.7960 | 0.8160 |
| param_only | 0.8090 | 0.8100 | 0.8250 |

## 初步解读

这次实验支持：

- `tianjiaying` 环境已经可以稳定用 GPU 跑 OUGP。
- 多 seed 结果已经比单 seed 更可靠。
- 在 30% sparsity 下，剪枝方法整体没有明显崩掉，准确率都在 0.806 到 0.815 左右。
- 当前最好的平均结果是 `param_only`，说明这个设置下参数剪枝比图剪枝更有帮助。

这次实验还不支持：

- 不能声称完整 `ougp` 已经优于所有 baseline。
- 不能声称 cross-level context 已经带来明确提升，因为 `ougp` 仍略低于 `ougp_no_cross` 和 `dual_static`。

下一步建议：

- 固定当前 GPU 环境后，跑 CiteSeer / PubMed 做跨数据集检查。
- 做更高 sparsity，例如 50% / 50%，观察 OUGP memory 是否在压力更大时体现优势。
- 单独扫 `write_beta`、`graph_gamma`、`param_gamma`，因为现在 OUGP 的优势可能被超参压住了。
