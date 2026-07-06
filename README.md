# OUGP

Online Unified Graph and Parameter Pruning for centralized GNNs.

## Layout

```text
research/ougp/
├── README.md
├── environment.yml
├── .gitignore
├── src/ougp/           # core package: data loader, model, OUGP modules
├── scripts/            # runnable experiment entrypoints
├── configs/            # reproducible parameter presets
├── data/
│   ├── raw/            # Planetoid raw files
│   └── processed/      # reserved for processed datasets
├── experiments/        # full records for each run
├── results/
│   ├── figures/
│   └── tables/         # cleaned summary tables
├── notes/              # idea, trackers, interpretation
├── tests/
└── third_party/
```

## Environment

Current working local environment:

```bash
/home/shizitong/miniconda3/envs/tianjiaying/bin/python
```

The `tianjiaying` environment has been repaired for CUDA PyTorch and is configured with `PYTHONNOUSERSITE=1` to avoid reading `~/.local` user packages. Use `environment.yml` for a clean environment specification.

## GPU Limit

Any run must expose at most 4 GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ...
```

The runner also refuses CUDA runs when more than `--max-gpus 4` devices are visible.

## Smoke Test

Run from `research/ougp/`:

```bash
PYTHONPATH=src /home/shizitong/miniconda3/envs/tianjiaying/bin/python scripts/run_case_study.py \
  --dataset cora \
  --epochs 1 \
  --warmup-epochs 1 \
  --variants dense ougp \
  --seeds 0 \
  --out-dir experiments/manual_smoke
```

## First Case Study

Run from `research/ougp/`:

```bash
PYTHONPATH=src /home/shizitong/miniconda3/envs/tianjiaying/bin/python scripts/run_case_study.py \
  --dataset cora \
  --epochs 80 \
  --warmup-epochs 10 \
  --variants dense graph_only param_only dual_static ougp_no_cross ougp \
  --seeds 0 \
  --out-dir experiments/exp001_cora_case_study_initial/primary_v2 \
  --graph-gamma 2.0 \
  --param-gamma 2.0 \
  --write-beta 0.25 \
  --verbose \
  --log-every 20
```

Each experiment directory should contain:

- `command.txt` - exact command
- `manifest.json` - arguments, output paths, and final metrics
- `*_seed*.json` - config, per-epoch history, and final result
- `*_summary.csv` / `*_summary.md` - final metrics table

Cleaned summary tables are copied to `results/tables/`.

## How To Read Experiments

If you are learning how to interpret experiments, start here:

```text
notes/EXPERIMENT_GUIDE_ZH.md
```

GPU diagnosis and recommended run plan:

```text
notes/GPU_DIAGNOSIS_AND_FIX_PLAN.md
```

Latest GPU multi-seed Cora run:

```text
notes/EXP002_CORA_GPU_MULTI_SEED.md
experiments/exp002_cora_gpu_multiseed/
results/tables/cora_exp002_gpu_multiseed_summary.csv
```

Latest Amazon Photo GPU run:

```text
notes/EXP006_AMAZON_PHOTO_GPU.md
experiments/exp006_photo_gpu_multiseed/
results/tables/photo_exp006_gpu_multiseed_summary.csv
configs/photo_case_study_v1.json
```

OGB large-graph feasibility attempt:

```text
notes/EXP009_OGB_LARGE_GRAPH_ATTEMPT.md
experiments/exp009_ogbn_arxiv_gpu_param_only/
results/tables/ogbn_arxiv_exp009_gpu_param_only_summary.csv
configs/ogbn_arxiv_param_only_v1.json
```
