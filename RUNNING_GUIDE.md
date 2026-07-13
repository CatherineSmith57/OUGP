# OUGP 运行指南

这份文档面向第一次拿到仓库、准备在 GPU / H200 服务器上复现实验的人。本文只使用仓库中已经存在的代码和脚本，不要求上传数据集到 GitHub。

本次希望在 H200 上优先运行 EXP065 大图 full-graph 任务，EXP066 sampled-subgraph 作为 OOM 时的备选方案。

## 1. 环境创建

进入项目目录：

```bash
cd /path/to/OUGP
```

用仓库里的 `environment.yml` 创建 conda 环境：

```bash
conda env create -f environment.yml
conda activate ougp
```

该环境会安装：

- Python 3.10
- PyTorch
- NumPy / SciPy
- scikit-learn
- pytest
- ogb

手动运行 Python 脚本时需要设置：

```bash
export PYTHONPATH=src
```

仓库里的主要 shell 脚本会自动设置 `PYTHONPATH=src`。

## 2. 数据集位置与下载

数据集不上传 GitHub。第一次运行时，如果服务器可以联网，代码会自动下载数据。

默认数据根目录是：

```text
data/raw/
```

具体位置：

```text
data/raw/planetoid/cora/
data/raw/planetoid/citeseer/
data/raw/planetoid/pubmed/
data/raw/amazon/photo/
data/raw/ogb/ogbn_arxiv/
data/raw/ogb/ogbn_products/
data/raw/ogb/ogbn_proteins/
```

说明：

- Cora / CiteSeer / PubMed 通过 Planetoid raw files 自动下载。
- Amazon Photo 会下载 `amazon_electronics_photo.npz`。
- ogbn-arxiv / ogbn-products / ogbn-proteins 通过 `ogb.nodeproppred.NodePropPredDataset` 自动下载。
- 脚本默认传入 `--data-root data/raw/planetoid`；对于 OGBN，`src/ougp/data.py` 会自动把根目录重定向到 `data/raw/ogb/`。

如果 H200 服务器不能联网，需要先把对应数据预先放到上面的目录。

## 3. 当前实验脚本

### `scripts/run_case_study.py`

通用训练入口。支持：

- 数据集：`cora`、`citeseer`、`pubmed`、`photo`、`ogbn-arxiv`、`ogbn-products`、`ogbn-proteins`
- backbone：`gcn`、`sage`、`gat`、`deepgcn`
- variants：`dense`、`ougp` 等
- random / frontier node subgraph sampling
- OUGP + LHCU 参数，例如 `--use-hidden-coupling`

### `scripts/run_hidden_state_diagnostic.py`

hidden-state diagnostic，用于分析 graph pruning 和 parameter pruning 对 hidden state 的扰动关系。不是大图主实验入口。

### `scripts/run_exp064_full_graph_hidden_coupling_validation.sh`

小/中图 full graph 验证：

- 数据集：Cora、CiteSeer、PubMed、Amazon Photo
- backbone：4-layer GCN
- variants：Dense 和 OUGP+LHCU
- seeds：0, 1, 2
- epochs：200
- 输出目录：`experiments/exp064_full_graph_hidden_coupling_validation`
- 日志目录：`experiments/exp064_full_graph_hidden_coupling_validation/logs`

### `scripts/run_exp065_full_graph_backbone_queue_wait_gpu.sh`

full-graph backbone 队列实验：

- Stage 1：ogbn-arxiv / ogbn-products / ogbn-proteins 上跑 4-layer GCN
- Stage 2：Cora / CiteSeer / PubMed / Amazon Photo / OGBN 三个大图上跑 GraphSAGE、GAT、DeeperGCN
- variants：Dense 和 OUGP+LHCU
- seeds：0, 1, 2, 3
- epochs：200
- 输出目录：`experiments/exp065_full_graph_backbone_queue`
- 状态文件：`experiments/exp065_full_graph_backbone_queue/status.tsv`

注意：OGBN full graph 的 OUGP 版本很容易 OOM，建议只在 H200 上尝试。

### `scripts/run_exp066_ogbn_sampled_lhcu_backbones_wait_gpu.sh`

EXP065 OOM 时的 sampled-subgraph 备选实验：

- 数据集：ogbn-arxiv、ogbn-products、ogbn-proteins
- backbone：GCN、GraphSAGE、GAT、DeeperGCN
- 采样方式：random subgraph 和 frontier subgraph
- variants：Dense 和 OUGP+LHCU
- seeds：0, 1, 2, 3
- epochs：200
- 输出目录：`experiments/exp066_ogbn_sampled_lhcu_backbones`
- launcher 日志：`experiments/exp066_ogbn_sampled_lhcu_backbones/launcher.log`
- 状态文件：`experiments/exp066_ogbn_sampled_lhcu_backbones/status.tsv`

注意：这个脚本当前写死了：

```bash
ROOT="/home/shizitong/tianjiaying/research/ougp"
PYTHON_BIN="${PYTHON_BIN:-/home/shizitong/miniconda3/envs/tianjiaying/bin/python}"
```

如果 H200 服务器上的仓库路径不同，需要先把 `ROOT` 改成实际项目目录；也可以通过环境变量覆盖 `PYTHON_BIN`：

```bash
PYTHON_BIN="$(which python)" bash scripts/run_exp066_ogbn_sampled_lhcu_backbones_wait_gpu.sh
```

## 4. 普通 GPU 与 H200 任务划分

普通 GPU 建议先跑：

- Cora smoke test
- `run_exp064_full_graph_hidden_coupling_validation.sh`
- Cora hidden-state diagnostic

H200 推荐跑：

- `run_exp065_full_graph_backbone_queue_wait_gpu.sh`
  - 这是本次优先运行的大图 full-graph 任务。
  - OGBN full graph + OUGP 在 24GB GPU 上已知容易 OOM，所以优先放到 H200。
- `run_exp066_ogbn_sampled_lhcu_backbones_wait_gpu.sh`
  - 这是 EXP065 仍然 OOM 时的备选方案。
  - random / frontier subgraph 都会跑。
  - 覆盖 OGBN 三个大图和四种 backbone。

## 5. 推荐运行顺序

### Step 1：环境检查

```bash
cd /path/to/OUGP
conda activate ougp
export PYTHONPATH=src
python -m py_compile src/ougp/model.py src/ougp/data.py scripts/run_case_study.py
```

### Step 2：先跑 Cora smoke test

```bash
cd /path/to/OUGP
conda activate ougp
export PYTHONPATH=src
python scripts/run_case_study.py \
  --dataset cora \
  --out-dir experiments/manual_smoke \
  --variants dense ougp \
  --seeds 0 \
  --epochs 5 \
  --warmup-epochs 1 \
  --backbone gcn \
  --num-gnn-layers 4 \
  --graph-memory-layout multi \
  --param-memory-layout multi \
  --graph-score-init topofeat \
  --param-score-init magnitude \
  --use-hidden-coupling \
  --hidden-coupling-mix-graph 0.2 \
  --hidden-coupling-mix-param 0.2
```

输出目录：

```text
experiments/manual_smoke/
```

### Step 3：ogbn-arxiv 单 seed full-graph smoke

先用 ogbn-arxiv、GCN、full graph 跑一个短实验，确认 OGBN 数据下载、GPU 和 OUGP+LHCU 都正常：

```bash
cd /path/to/OUGP
conda activate ougp
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python scripts/run_case_study.py \
  --dataset ogbn-arxiv \
  --data-root data/raw/planetoid \
  --out-dir experiments/manual_ogbn_arxiv_gcn_fullgraph_seed0 \
  --variants dense ougp \
  --seeds 0 \
  --epochs 20 \
  --warmup-epochs 2 \
  --hidden-dim 32 \
  --memory-rank 8 \
  --graph-sparsity 0.30 \
  --param-sparsity 0.30 \
  --backbone gcn \
  --num-gnn-layers 4 \
  --graph-memory-layout multi \
  --param-memory-layout multi \
  --graph-score-init topofeat \
  --param-score-init magnitude \
  --use-hidden-coupling \
  --hidden-coupling-mix-graph 0.2 \
  --hidden-coupling-mix-param 0.2 \
  --device cuda \
  --verbose \
  --log-every 10
```

输出目录：

```text
experiments/manual_ogbn_arxiv_gcn_fullgraph_seed0/
```

### Step 4：H200 优先任务，EXP065 大图 full-graph

```bash
cd /path/to/OUGP
conda activate ougp
screen -dmS ougp_exp065_fullgraph_queue bash -lc 'cd /path/to/OUGP && conda activate ougp && PYTHON="$(which python)" bash scripts/run_exp065_full_graph_backbone_queue_wait_gpu.sh'
```

日志位置：

```text
experiments/exp065_full_graph_backbone_queue/logs/
```

状态文件：

```text
experiments/exp065_full_graph_backbone_queue/status.tsv
```

结果目录：

```text
experiments/exp065_full_graph_backbone_queue/
```

### Step 5：如果 EXP065 仍然 OOM，再运行 EXP066 sampled-subgraph

如果仓库路径不是 `/home/shizitong/tianjiaying/research/ougp`，先修改 `scripts/run_exp066_ogbn_sampled_lhcu_backbones_wait_gpu.sh` 里的 `ROOT`。

然后用 screen 后台启动：

```bash
cd /path/to/OUGP
conda activate ougp
screen -dmS ougp_exp066_ogbn_sampled_lhcu bash -lc 'cd /path/to/OUGP && conda activate ougp && PYTHON_BIN="$(which python)" bash scripts/run_exp066_ogbn_sampled_lhcu_backbones_wait_gpu.sh'
```

日志位置：

```text
experiments/exp066_ogbn_sampled_lhcu_backbones/launcher.log
experiments/exp066_ogbn_sampled_lhcu_backbones/<dataset>/<backbone>/<route>/run.log
```

结果目录：

```text
experiments/exp066_ogbn_sampled_lhcu_backbones/
```

### 可选：小/中图 full graph 验证

```bash
cd /path/to/OUGP
conda activate ougp
GPU_ID=0 bash scripts/run_exp064_full_graph_hidden_coupling_validation.sh
```

日志位置：

```text
experiments/exp064_full_graph_hidden_coupling_validation/logs/
```

结果目录：

```text
experiments/exp064_full_graph_hidden_coupling_validation/
```

## 6. 如何判断实验是否在运行

查看 GPU：

```bash
nvidia-smi
```

查看 screen：

```bash
screen -ls
```

进入某个 screen：

```bash
screen -r ougp_exp066_ogbn_sampled_lhcu
```

从 screen 里退出但不停止任务：

```text
Ctrl-a d
```

### 查看 EXP065 full-graph 任务

进入 EXP065 的 screen：

```bash
screen -r ougp_exp065_fullgraph_queue
```


查看 EXP066 launcher 日志：

```bash
tail -f experiments/exp066_ogbn_sampled_lhcu_backbones/launcher.log
```

查看 EXP066 状态：

```bash
cat experiments/exp066_ogbn_sampled_lhcu_backbones/status.tsv
```

查看某个子任务日志：

```bash
tail -f experiments/exp066_ogbn_sampled_lhcu_backbones/ogbn_arxiv/gcn/random_subgraph/run.log
```

查找已经生成的 summary：

```bash
find experiments/exp066_ogbn_sampled_lhcu_backbones -name '*summary.md' -o -name '*summary.csv'
```

判断是否结束：

- `screen -ls` 里对应 session 消失，通常说明脚本已经退出。
- `launcher.log` 末尾出现 `EXP066 launcher finished.`，说明 EXP066 全部队列结束。
- `status.tsv` 中每个任务都有 `complete` 或 `failed_xxx`。
- 每个结果目录下出现 `*_summary.csv` / `*_summary.md`。

## 7. ogbn-proteins 指标提醒

`ogbn-proteins` 是 multilabel 任务，仓库里对应的 `metric_name` 是：

```text
rocauc
```

因此：

- ogbn-proteins 看的是 ROC-AUC，不是 accuracy。
- 选择最佳 epoch 时应使用 validation ROC-AUC。
- 不能根据 test ROC-AUC 选择最佳 epoch。
- test ROC-AUC 只用于最终报告。

## 8. 常见输出文件

每个实验目录通常会包含：

```text
command.txt
manifest.json
*_seed*.json
*_summary.csv
*_summary.md
run.log
```

大队列脚本还会包含：

```text
launcher.log
status.tsv
```

其中：

- `*_seed*.json`：单个 seed 的配置、训练历史和最终指标。
- `*_summary.csv` / `*_summary.md`：多 seed 汇总结果。
- `run.log`：单个任务日志。
- `launcher.log`：队列启动和调度日志。
- `status.tsv`：每个任务完成或失败状态。
