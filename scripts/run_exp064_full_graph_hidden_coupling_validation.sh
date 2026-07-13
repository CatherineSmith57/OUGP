#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-/home/shizitong/miniconda3/envs/tianjiaying/bin/python}"
GPU_ID="${GPU_ID:-4}"
OUT_ROOT="${OUT_ROOT:-experiments/exp064_full_graph_hidden_coupling_validation}"
LOG_DIR="$OUT_ROOT/logs"
mkdir -p "$LOG_DIR"

export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES="$GPU_ID"

COMMON_ARGS=(
  --variants dense ougp
  --seeds 0 1 2
  --epochs 200
  --warmup-epochs 10
  --backbone gcn
  --num-gnn-layers 4
  --graph-sparsity 0.30
  --param-sparsity 0.30
  --graph-memory-layout multi
  --param-memory-layout multi
  --graph-score-init topofeat
  --param-score-init magnitude
  --use-hidden-coupling
  --hidden-coupling-mix-graph 0.2
  --hidden-coupling-mix-param 0.2
  --device cuda
  --verbose
  --log-every 20
)

for dataset in cora citeseer pubmed photo; do
  echo "===== exp064 full graph hidden coupling: ${dataset} ====="
  "$PYTHON" scripts/run_case_study.py \
    --dataset "$dataset" \
    --out-dir "$OUT_ROOT/$dataset" \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "$LOG_DIR/${dataset}.log"
done

echo "===== exp064 done ====="
