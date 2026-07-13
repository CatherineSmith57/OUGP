#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/shizitong/tianjiaying/research/ougp"
PYTHON_BIN="/home/shizitong/miniconda3/envs/tianjiaying/bin/python"
EXP_ROOT="experiments/exp058_ogbn_proteins_backbone_sample300e"
DATASET="ogbn-proteins"
DATA_ROOT="data/raw/planetoid"

FREE_MEM_MIB="${FREE_MEM_MIB:-1000}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
POLL_SECONDS="${POLL_SECONDS:-120}"
NODE_SAMPLE_SIZE="${NODE_SAMPLE_SIZE:-20000}"

BACKBONE_ROUTES=(
  "gcn random_subgraph"
  "gcn frontier_subgraph"
  "sage random_subgraph"
  "sage frontier_subgraph"
  "gat random_subgraph"
  "gat frontier_subgraph"
  "deepgcn random_subgraph"
  "deepgcn frontier_subgraph"
)

cd "$ROOT"
mkdir -p "$EXP_ROOT"
exec > >(tee -a "$EXP_ROOT/launcher.log") 2>&1

gpu_used_table() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits
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

launch_job() {
  local backbone="$1"
  local route="$2"
  local gpu="$3"
  local out_dir="$EXP_ROOT/$backbone/$route"
  local log_file="$out_dir/run.log"
  local mode_arg="random"
  local layers_args=()

  if [[ "$route" == "frontier_subgraph" ]]; then
    mode_arg="frontier"
  fi
  if [[ "$backbone" == "deepgcn" ]]; then
    layers_args=(--num-gnn-layers 4)
  fi

  mkdir -p "$out_dir"
  cat > "$out_dir/command.txt" <<CMD
CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src $PYTHON_BIN scripts/run_case_study.py --dataset $DATASET --data-root $DATA_ROOT --out-dir $out_dir --variants dense random_static degree_magnitude_static similarity_magnitude_static dual_static ougp --seeds 0 1 2 3 4 5 6 7 8 9 --epochs 300 --warmup-epochs 30 --hidden-dim 32 --memory-rank 8 --graph-sparsity 0.30 --param-sparsity 0.30 --backbone $backbone ${layers_args[*]} --node-sample-size $NODE_SAMPLE_SIZE --node-sample-seed 0 --node-sample-mode $mode_arg --graph-memory-granularity subgraph --graph-memory-layout multi --param-memory-layout multi --device cuda --verbose --log-every 100
CMD

  echo "[$(date '+%F %T')] Launching $DATASET $backbone $route on GPU $gpu"
  (
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src "$PYTHON_BIN" scripts/run_case_study.py \
      --dataset "$DATASET" \
      --data-root "$DATA_ROOT" \
      --out-dir "$out_dir" \
      --variants dense random_static degree_magnitude_static similarity_magnitude_static dual_static ougp \
      --seeds 0 1 2 3 4 5 6 7 8 9 \
      --epochs 300 \
      --warmup-epochs 30 \
      --hidden-dim 32 \
      --memory-rank 8 \
      --graph-sparsity 0.30 \
      --param-sparsity 0.30 \
      --backbone "$backbone" \
      "${layers_args[@]}" \
      --node-sample-size "$NODE_SAMPLE_SIZE" \
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

echo "[$(date '+%F %T')] EXP058 launcher started."
echo "FREE_MEM_MIB=$FREE_MEM_MIB MAX_PARALLEL=$MAX_PARALLEL POLL_SECONDS=$POLL_SECONDS NODE_SAMPLE_SIZE=$NODE_SAMPLE_SIZE"

for item in "${BACKBONE_ROUTES[@]}"; do
  read -r backbone route <<< "$item"
  out_dir="$EXP_ROOT/$backbone/$route"
  summary_csv="$out_dir/ogbn-proteins_summary.csv"
  if [[ -f "$summary_csv" ]]; then
    echo "[$(date '+%F %T')] Skip existing result: $summary_csv"
    continue
  fi
  wait_for_slot
  gpu="$(wait_for_gpu)"
  launch_job "$backbone" "$route" "$gpu"
  sleep 10
done

echo "[$(date '+%F %T')] All jobs launched; waiting for completion."
while [[ "${#running_jobs[@]}" -gt 0 ]]; do
  reap_finished_jobs
  sleep "$POLL_SECONDS"
done
echo "[$(date '+%F %T')] EXP058 launcher finished."
