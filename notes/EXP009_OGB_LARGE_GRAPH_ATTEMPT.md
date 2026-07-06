# EXP009: OGB Large Graph Attempt

日期：2026-07-06

## 目标

按 `idea.md` 中的大图设置，依次尝试：

- `ogbn-arxiv`
- `ogbn-products`
- `ogbn-proteins`

目标是检查当前 full-batch OUGP 实现能否扩展到 OGB 大图。

## 环境

- Conda: `tianjiaying`
- GPU: `CUDA_VISIBLE_DEVICES=2`
- PyTorch: `2.7.1+cu118`
- 新增依赖：`ogb==1.3.6`

本轮没有使用 `atma`。

## 代码改动

新增 OGB loader：

```text
src/ougp/data.py
```

新增 runner 支持：

```text
--dataset ogbn-arxiv
--dataset ogbn-products
--dataset ogbn-proteins
```

同时修复了 `ogb==1.3.6` 在 PyTorch 2.7 下加载预处理缓存时的兼容问题：

```text
torch.load(..., weights_only=False)
```

这个兼容只在 OGB loader 内部临时生效。

## OGB 数据规模

来自 OGB 官方 node property prediction 文档：

| Dataset | Nodes | Edges | Task |
| --- | ---: | ---: | --- |
| `ogbn-arxiv` | 169,343 | 1,166,243 | multiclass classification |
| `ogbn-products` | 2,449,029 | 61,859,140 | multiclass classification |
| `ogbn-proteins` | 132,534 | 39,561,252 | 112 binary classification tasks, ROC-AUC |

## EXP007: ogbn-arxiv smoke

Smoke 跑通：

```text
experiments/exp007_ogbn_arxiv_gpu_smoke/
```

结果说明：`ogbn-arxiv` 数据能加载，dense / OUGP 1 epoch 前向和基础训练入口能启动。

## EXP008: ogbn-arxiv full OUGP attempt

尝试命令：

```bash
CUDA_VISIBLE_DEVICES=2 /home/shizitong/miniconda3/bin/conda run -n tianjiaying env PYTHONPATH=src python scripts/run_case_study.py \
  --dataset ogbn-arxiv \
  --epochs 80 \
  --warmup-epochs 10 \
  --hidden-dim 64 \
  --memory-rank 8 \
  --variants dense graph_only param_only dual_static ougp_no_cross ougp \
  --seeds 0 1 2 \
  --out-dir experiments/exp008_ogbn_arxiv_gpu_multiseed \
  --graph-gamma 2.0 \
  --param-gamma 2.0 \
  --write-beta 0.25 \
  --device cuda \
  --max-gpus 1
```

失败记录：

```text
experiments/exp008_ogbn_arxiv_gpu_multiseed/run_failed_graph_pruning_oom.log
```

关键报错：

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 106.83 GiB.
```

失败发生在 `graph_only/seed0` 的 `loss.backward()`。

解释：

- dense GCN 可以跑。
- 一旦 graph mask 进入 sparse message passing 的可学习 edge weight，backward 会在百万边级别产生极大的内存需求。
- 这不是 GPU 2 不空，而是当前 full-batch graph pruning 实现不适合直接扩展到 OGB 大图。

## EXP009: ogbn-arxiv safe param-only run

为了保留可用的大图证据，继续跑了不含 graph pruning 的安全对照：

```text
experiments/exp009_ogbn_arxiv_gpu_param_only/
results/tables/ogbn_arxiv_exp009_gpu_param_only_summary.csv
```

结果：

| Variant | Best Test Acc | Graph Sparsity | Param Sparsity | Graph Churn | Param Churn |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense | 0.6051 +/- 0.0017 | 0.000 | 0.000 | 0.000 | 0.000 |
| param_only | 0.5959 +/- 0.0011 | 0.000 | 0.300 | 0.000 | 0.009 |

解读：

- `ogbn-arxiv` 上，当前参数剪枝会带来轻微性能下降。
- 这只是 parameter pruning evidence，不能替代完整 OUGP 结果。

## 为什么没有直接跑 ogbn-products / ogbn-proteins

`ogbn-arxiv` 只有约 116 万边，graph pruning backward 已经需要约 106 GiB。

相比之下：

- `ogbn-products` 有约 6186 万边。
- `ogbn-proteins` 有约 3956 万边，而且还是 multi-label ROC-AUC 任务。

所以在当前 full-batch OUGP 代码下，直接跑完整 `graph_only / dual_static / ougp` 几乎必然 OOM，并且会浪费 GPU 时间。

这不是数据集不能做，而是方法实现需要换成大图版本。

## 下一步需要的正确方案

要真正跑 `ogbn-products` 和 `ogbn-proteins`，需要先改实验实现：

- 从 full-batch GCN 改成 mini-batch / neighbor sampling。
- graph mask 不能对全部边一次性建可学习 dense 向量。
- OPM memory 需要按 batch edge / sampled subgraph 更新，而不是全图边一次性读写。
- `ogbn-proteins` 需要继续使用 BCE loss + ROC-AUC，而不是 multiclass accuracy。

推荐下一步：

1. 先实现 `scripts/run_ogb_sampled_case_study.py`。
2. 在 `ogbn-arxiv` 上确认 sampled 版本和 full-batch dense 大致一致。
3. 再跑 `ogbn-products` / `ogbn-proteins` 的 sampled OUGP。

## 当前可写进汇报的结论

> The current full-batch OUGP implementation scales to medium graphs such as Amazon Photo, but not to OGB-scale graph-pruning experiments. On ogbn-arxiv, dense and parameter-only runs complete, while graph-pruning variants fail during backward with a 106.83 GiB allocation request. This indicates that OGB-scale validation requires a mini-batch or neighbor-sampling implementation of OUGP.
