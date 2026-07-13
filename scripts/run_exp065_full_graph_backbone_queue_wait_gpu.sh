#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-/home/shizitong/miniconda3/envs/tianjiaying/bin/python}"
OUT_ROOT="${OUT_ROOT:-experiments/exp065_full_graph_backbone_queue}"
LOG_DIR="$OUT_ROOT/logs"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
GPU_MEM_THRESHOLD_MIB="${GPU_MEM_THRESHOLD_MIB:-500}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-300}"
mkdir -p "$LOG_DIR"

export PYTHONPATH=src

COMMON_ARGS=(
  --variants dense ougp
  --seeds 0 1 2 3
  --epochs 200
  --warmup-epochs 10
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

STATUS_FILE="$OUT_ROOT/status.tsv"
printf "time\tstage\tdataset\tbackbone\tgpu\tstatus\tlog\n" > "$STATUS_FILE"

PIDS=()
PID_GPUS=()

is_gpu_active() {
  local gpu="$1"
  for active in "${PID_GPUS[@]:-}"; do
    [[ "$active" == "$gpu" ]] && return 0
  done
  return 1
}

cleanup_finished() {
  local new_pids=()
  local new_gpus=()
  local idx pid gpu
  for idx in "${!PIDS[@]}"; do
    pid="${PIDS[$idx]}"
    gpu="${PID_GPUS[$idx]}"
    if kill -0 "$pid" 2>/dev/null; then
      new_pids+=("$pid")
      new_gpus+=("$gpu")
    else
      wait "$pid" >/dev/null 2>&1 || true
    fi
  done
  PIDS=("${new_pids[@]}")
  PID_GPUS=("${new_gpus[@]}")
}

find_free_gpu() {
  local line gpu mem
  while IFS=, read -r gpu mem; do
    gpu="${gpu// /}"
    mem="${mem// MiB/}"
    mem="${mem// /}"
    [[ -z "$gpu" || -z "$mem" ]] && continue
    if (( mem <= GPU_MEM_THRESHOLD_MIB )) && ! is_gpu_active "$gpu"; then
      printf "%s\n" "$gpu"
      return 0
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null)
  return 1
}

wait_for_slot_and_gpu() {
  local gpu=""
  while true; do
    cleanup_finished
    if (( ${#PIDS[@]} < MAX_PARALLEL )); then
      gpu="$(find_free_gpu || true)"
      if [[ -n "$gpu" ]]; then
        printf "%s\n" "$gpu"
        return 0
      fi
    fi
    echo "[$(date '+%F %T')] waiting for free GPU; active_jobs=${#PIDS[@]}; threshold=${GPU_MEM_THRESHOLD_MIB}MiB" >&2
    sleep "$GPU_POLL_SECONDS"
  done
}

layers_for_backbone() {
  local backbone="$1"
  if [[ "$backbone" == "gcn" || "$backbone" == "deepgcn" ]]; then
    printf "4\n"
  else
    printf "2\n"
  fi
}

launch_job() {
  local stage="$1"
  local dataset="$2"
  local backbone="$3"
  local gpu="$4"
  local layers
  layers="$(layers_for_backbone "$backbone")"
  local out_dir="$OUT_ROOT/$stage/${dataset}_${backbone}"
  local log_file="$LOG_DIR/${stage}_${dataset}_${backbone}.log"
  mkdir -p "$out_dir"
  echo "[$(date '+%F %T')] launch stage=${stage} dataset=${dataset} backbone=${backbone} gpu=${gpu} layers=${layers}"
  (
    set -o pipefail
    {
      echo "stage=${stage}"
      echo "dataset=${dataset}"
      echo "backbone=${backbone}"
      echo "gpu=${gpu}"
      echo "layers=${layers}"
      CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/run_case_study.py \
        --dataset "$dataset" \
        --out-dir "$out_dir" \
        --backbone "$backbone" \
        --num-gnn-layers "$layers" \
        "${COMMON_ARGS[@]}"
    } 2>&1 | tee "$log_file"
    status="${PIPESTATUS[0]}"
    if [[ "$status" -eq 0 ]]; then
      printf "%s\t%s\t%s\t%s\t%s\tcomplete\t%s\n" "$(date '+%F %T')" "$stage" "$dataset" "$backbone" "$gpu" "$log_file" >> "$STATUS_FILE"
    else
      printf "%s\t%s\t%s\t%s\t%s\tfailed_%s\t%s\n" "$(date '+%F %T')" "$stage" "$dataset" "$backbone" "$gpu" "$status" "$log_file" >> "$STATUS_FILE"
    fi
    exit "$status"
  ) &
  PIDS+=("$!")
  PID_GPUS+=("$gpu")
}

run_stage() {
  local stage="$1"
  shift
  local job dataset backbone gpu
  for job in "$@"; do
    dataset="${job%%:*}"
    backbone="${job##*:}"
    gpu="$(wait_for_slot_and_gpu)"
    launch_job "$stage" "$dataset" "$backbone" "$gpu"
  done
  while (( ${#PIDS[@]} > 0 )); do
    cleanup_finished
    if (( ${#PIDS[@]} > 0 )); then
      echo "[$(date '+%F %T')] waiting for stage=${stage}; active_jobs=${#PIDS[@]}"
      sleep "$GPU_POLL_SECONDS"
    fi
  done
}

STAGE1_JOBS=(
  "ogbn-arxiv:gcn"
  "ogbn-products:gcn"
  "ogbn-proteins:gcn"
)

STAGE2_JOBS=()
for dataset in cora citeseer pubmed photo ogbn-arxiv ogbn-products ogbn-proteins; do
  for backbone in sage gat deepgcn; do
    STAGE2_JOBS+=("${dataset}:${backbone}")
  done
done

echo "===== EXP065 stage1: OGBN full graph GCN-4-layer ====="
run_stage "stage1_ogbn_gcn4" "${STAGE1_JOBS[@]}"

echo "===== EXP065 stage2: all remaining backbones on all requested graphs ====="
run_stage "stage2_other_backbones" "${STAGE2_JOBS[@]}"

echo "===== EXP065 queue done ====="
