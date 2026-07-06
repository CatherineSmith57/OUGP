#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
PYTHON_BIN="${PYTHON_BIN:-/home/shizitong/miniconda3/envs/tianjiaying/bin/python}"
BASE_DIR="experiments/exp014_photo_event_feedback_sweep"

mkdir -p "${BASE_DIR}"

for gamma in 0.00 0.05 0.10 0.20; do
  out_dir="${BASE_DIR}/gamma_${gamma//./_}"
  mkdir -p "${out_dir}"
  echo "Running Amazon Photo event feedback sweep: event_gamma=${gamma}, gpu=${GPU_ID}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH=src "${PYTHON_BIN}" scripts/run_case_study.py \
    --dataset photo \
    --data-root data/raw/planetoid \
    --out-dir "${out_dir}" \
    --variants graph_only ougp \
    --seeds 0 1 2 \
    --epochs 120 \
    --warmup-epochs 15 \
    --hidden-dim 64 \
    --memory-rank 16 \
    --graph-sparsity 0.30 \
    --param-sparsity 0.30 \
    --graph-gamma 2.0 \
    --param-gamma 2.0 \
    --write-beta 0.25 \
    --event-gamma "${gamma}" \
    --event-beta 0.10 \
    --event-decay 0.95 \
    --event-top-k 2000 \
    --trace-pruning \
    --trace-every 10 \
    --trace-top-k 200 \
    --device cuda \
    --max-gpus 1 \
    --verbose 2>&1 | tee "${out_dir}/run.log"
done
