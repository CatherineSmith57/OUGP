#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-7}"
PYTHON_BIN="${PYTHON_BIN:-/home/shizitong/miniconda3/envs/tianjiaying/bin/python}"
BASE_DIR="experiments/exp015_photo_recall_steering_sweep"
COMMON_ARGS=(
  --dataset photo
  --data-root data/raw/planetoid
  --seeds 0 1 2
  --epochs 120
  --warmup-epochs 15
  --hidden-dim 64
  --memory-rank 16
  --graph-sparsity 0.30
  --param-sparsity 0.30
  --graph-gamma 2.0
  --param-gamma 2.0
  --write-beta 0.25
  --device cuda
  --max-gpus 1
  --verbose
)

mkdir -p "${BASE_DIR}"

run_case() {
  local case_name="$1"
  shift
  local out_dir="${BASE_DIR}/${case_name}"
  mkdir -p "${out_dir}"
  echo "Running EXP015 case=${case_name} gpu=${GPU_ID}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH=src "${PYTHON_BIN}" scripts/run_case_study.py \
    --out-dir "${out_dir}" \
    "${COMMON_ARGS[@]}" \
    "$@" 2>&1 | tee "${out_dir}/run.log"
}

run_case baseline \
  --variants graph_only ougp

run_case recall_only \
  --variants ougp \
  --recall-gamma 0.20 \
  --recall-beta 0.10 \
  --recall-decay 0.95 \
  --recall-top-k 2000

run_case steering_only \
  --variants ougp \
  --use-steering-memory \
  --steer-gamma 0.10 \
  --steer-beta 0.20 \
  --steer-lambda 0.90

run_case recall_steering \
  --variants ougp \
  --recall-gamma 0.20 \
  --recall-beta 0.10 \
  --recall-decay 0.95 \
  --recall-top-k 2000 \
  --use-steering-memory \
  --steer-gamma 0.10 \
  --steer-beta 0.20 \
  --steer-lambda 0.90
