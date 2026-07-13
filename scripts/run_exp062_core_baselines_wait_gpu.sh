#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/shizitong/tianjiaying/research/ougp"
PYTHON_BIN="/home/shizitong/miniconda3/envs/tianjiaying/bin/python"
EXP_ROOT="experiments/exp062_large_core_baseline_sample300e"
DATA_ROOT="data/raw/planetoid"

FREE_MEM_MIB="${FREE_MEM_MIB:-1000}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
POLL_SECONDS="${POLL_SECONDS:-120}"
EPOCHS="${EPOCHS:-300}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-30}"
HIDDEN_DIM="${HIDDEN_DIM:-32}"
MEMORY_RANK="${MEMORY_RANK:-8}"
GRAPH_SPARSITY="${GRAPH_SPARSITY:-0.30}"
PARAM_SPARSITY="${PARAM_SPARSITY:-0.30}"
DRY_RUN="${DRY_RUN:-0}"

DATASETS_STR="${DATASETS_STR:-ogbn-arxiv ogbn-products ogbn-proteins}"
BACKBONES_STR="${BACKBONES_STR:-gcn sage gat deepgcn}"
ROUTES_STR="${ROUTES_STR:-random_subgraph frontier_subgraph}"
VARIANTS_STR="${VARIANTS_STR:-dense dual_static dropedge_static neuralsparse_dual_dynamic dspar_dual_static degree_gradient_static degree_grasp_static degree_lottery_static degree_rigl_dynamic serial_degree_magnitude_static ace_eagles_unified_dynamic ougp}"
SEEDS_STR="${SEEDS_STR:-0 1 2 3 4 5 6 7 8 9}"

read -r -a DATASETS <<< "$DATASETS_STR"
read -r -a BACKBONES <<< "$BACKBONES_STR"
read -r -a ROUTES <<< "$ROUTES_STR"
read -r -a VARIANTS <<< "$VARIANTS_STR"
read -r -a SEEDS <<< "$SEEDS_STR"

cd "$ROOT"
mkdir -p "$EXP_ROOT"
exec > >(tee -a "$EXP_ROOT/launcher.log") 2>&1

gpu_used_table() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits
}

running_jobs=()

gpu_is_reserved() {
  local gpu="$1"
  local entry
  local entry_gpu
  for entry in "${running_jobs[@]:-}"; do
    entry_gpu="${entry#*:}"
    if [[ "$entry_gpu" == "$gpu" ]]; then
      return 0
    fi
  done
  return 1
}

find_free_gpu() {
  gpu_used_table | while IFS=',' read -r raw_idx raw_used; do
    idx="$(echo "$raw_idx" | tr -d ' ')"
    used="$(echo "$raw_used" | tr -d ' ')"
    if [[ -n "$idx" && -n "$used" && "$used" -le "$FREE_MEM_MIB" ]] && ! gpu_is_reserved "$idx"; then
      echo "$idx"
      return 0
    fi
  done
}

reap_finished_jobs() {
  local alive=()
  local entry
  local pid
  for entry in "${running_jobs[@]:-}"; do
    pid="${entry%%:*}"
    if kill -0 "$pid" 2>/dev/null; then
      alive+=("$entry")
    fi
  done
  running_jobs=("${alive[@]:-}")
}

wait_for_slot() {
  while true; do
    reap_finished_jobs
    if [[ "${#running_jobs[@]}" -lt "$MAX_PARALLEL" ]]; then
      return 0
    fi
    sleep "$POLL_SECONDS"
  done
}

wait_for_gpu() {
  local gpu
  while true; do
    gpu="$(find_free_gpu || true)"
    if [[ -n "$gpu" ]]; then
      echo "$gpu"
      return 0
    fi
    echo "[$(date '+%F %T')] No clean GPU under ${FREE_MEM_MIB} MiB; waiting ${POLL_SECONDS}s..." >&2
    sleep "$POLL_SECONDS"
  done
}

sample_size_for_dataset() {
  local dataset="$1"
  case "$dataset" in
    ogbn-proteins) echo "${PROTEINS_NODE_SAMPLE_SIZE:-20000}" ;;
    ogbn-products) echo "${PRODUCTS_NODE_SAMPLE_SIZE:-5000}" ;;
    ogbn-arxiv) echo "${ARXIV_NODE_SAMPLE_SIZE:-5000}" ;;
    *) echo "${NODE_SAMPLE_SIZE:-5000}" ;;
  esac
}

mode_for_route() {
  local route="$1"
  if [[ "$route" == "frontier_subgraph" ]]; then
    echo "frontier"
  else
    echo "random"
  fi
}

launch_job() {
  local dataset="$1"
  local backbone="$2"
  local route="$3"
  local gpu="$4"
  local sample_size
  local mode_arg
  local dataset_key
  local out_dir
  local log_file
  local layers_args=()

  sample_size="$(sample_size_for_dataset "$dataset")"
  mode_arg="$(mode_for_route "$route")"
  dataset_key="${dataset//-/_}"
  out_dir="$EXP_ROOT/$dataset_key/$backbone/$route"
  log_file="$out_dir/run.log"

  if [[ "$backbone" == "deepgcn" ]]; then
    layers_args=(--num-gnn-layers 4)
  fi

  mkdir -p "$out_dir"
  cat > "$out_dir/command.txt" <<CMD
CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src $PYTHON_BIN scripts/run_case_study.py --dataset $dataset --data-root $DATA_ROOT --out-dir $out_dir --variants ${VARIANTS[*]} --seeds ${SEEDS[*]} --epochs $EPOCHS --warmup-epochs $WARMUP_EPOCHS --hidden-dim $HIDDEN_DIM --memory-rank $MEMORY_RANK --graph-sparsity $GRAPH_SPARSITY --param-sparsity $PARAM_SPARSITY --backbone $backbone ${layers_args[*]} --node-sample-size $sample_size --node-sample-seed 0 --node-sample-mode $mode_arg --graph-memory-granularity subgraph --graph-memory-layout multi --param-memory-layout multi --device cuda --verbose --log-every 100
CMD

  echo "[$(date '+%F %T')] Launching $dataset $backbone $route on GPU $gpu"
  (
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src "$PYTHON_BIN" scripts/run_case_study.py \
      --dataset "$dataset" \
      --data-root "$DATA_ROOT" \
      --out-dir "$out_dir" \
      --variants "${VARIANTS[@]}" \
      --seeds "${SEEDS[@]}" \
      --epochs "$EPOCHS" \
      --warmup-epochs "$WARMUP_EPOCHS" \
      --hidden-dim "$HIDDEN_DIM" \
      --memory-rank "$MEMORY_RANK" \
      --graph-sparsity "$GRAPH_SPARSITY" \
      --param-sparsity "$PARAM_SPARSITY" \
      --backbone "$backbone" \
      "${layers_args[@]}" \
      --node-sample-size "$sample_size" \
      --node-sample-seed 0 \
      --node-sample-mode "$mode_arg" \
      --graph-memory-granularity subgraph \
      --graph-memory-layout multi \
      --param-memory-layout multi \
      --device cuda \
      --verbose \
      --log-every 100
  ) > "$log_file" 2>&1 &
  running_jobs+=("$!:$gpu")
}

echo "[$(date '+%F %T')] EXP062 launcher started."
echo "FREE_MEM_MIB=$FREE_MEM_MIB MAX_PARALLEL=$MAX_PARALLEL POLL_SECONDS=$POLL_SECONDS"
echo "DATASETS=${DATASETS[*]}"
echo "BACKBONES=${BACKBONES[*]}"
echo "ROUTES=${ROUTES[*]}"
echo "VARIANTS=${VARIANTS[*]}"
echo "DRY_RUN=$DRY_RUN"

planned_jobs=0

for dataset in "${DATASETS[@]}"; do
  dataset_key="${dataset//-/_}"
  for backbone in "${BACKBONES[@]}"; do
    for route in "${ROUTES[@]}"; do
      out_dir="$EXP_ROOT/$dataset_key/$backbone/$route"
      summary_csv="$out_dir/${dataset}_summary.csv"
      if [[ -f "$summary_csv" ]]; then
        echo "[$(date '+%F %T')] Skip existing result: $summary_csv"
        continue
      fi
      planned_jobs=$((planned_jobs + 1))
      if [[ "$DRY_RUN" == "1" ]]; then
        echo "[$(date '+%F %T')] DRY-RUN would launch: dataset=$dataset backbone=$backbone route=$route out_dir=$out_dir variants=${VARIANTS[*]}"
        continue
      fi
      wait_for_slot
      gpu="$(wait_for_gpu)"
      launch_job "$dataset" "$backbone" "$route" "$gpu"
      sleep 10
    done
  done
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[$(date '+%F %T')] DRY-RUN finished. Planned jobs: $planned_jobs"
  exit 0
fi

echo "[$(date '+%F %T')] All jobs launched; waiting for completion."
while [[ "${#running_jobs[@]}" -gt 0 ]]; do
  reap_finished_jobs
  sleep "$POLL_SECONDS"
done
echo "[$(date '+%F %T')] EXP062 launcher finished."
