#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/shizitong/tianjiaying/research/ougp"
PYTHON_BIN="${PYTHON_BIN:-/home/shizitong/miniconda3/envs/tianjiaying/bin/python}"
EXP_ROOT="${EXP_ROOT:-experiments/exp066_ogbn_sampled_lhcu_backbones}"
DATA_ROOT="${DATA_ROOT:-data/raw/planetoid}"

FREE_MEM_MIB="${FREE_MEM_MIB:-1000}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
POLL_SECONDS="${POLL_SECONDS:-120}"
EPOCHS="${EPOCHS:-200}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-20}"
HIDDEN_DIM="${HIDDEN_DIM:-32}"
MEMORY_RANK="${MEMORY_RANK:-8}"
GRAPH_SPARSITY="${GRAPH_SPARSITY:-0.30}"
PARAM_SPARSITY="${PARAM_SPARSITY:-0.30}"
HIDDEN_COUPLING_MIX_GRAPH="${HIDDEN_COUPLING_MIX_GRAPH:-0.2}"
HIDDEN_COUPLING_MIX_PARAM="${HIDDEN_COUPLING_MIX_PARAM:-0.2}"

DATASETS_STR="${DATASETS_STR:-ogbn-arxiv ogbn-products ogbn-proteins}"
BACKBONES_STR="${BACKBONES_STR:-gcn sage gat deepgcn}"
ROUTES_STR="${ROUTES_STR:-random_subgraph frontier_subgraph}"
SEEDS_STR="${SEEDS_STR:-0 1 2 3}"
VARIANTS_STR="${VARIANTS_STR:-dense ougp}"

read -r -a DATASETS <<< "$DATASETS_STR"
read -r -a BACKBONES <<< "$BACKBONES_STR"
read -r -a ROUTES <<< "$ROUTES_STR"
read -r -a SEEDS <<< "$SEEDS_STR"
read -r -a VARIANTS <<< "$VARIANTS_STR"

cd "$ROOT"
mkdir -p "$EXP_ROOT"
exec > >(tee -a "$EXP_ROOT/launcher.log") 2>&1

STATUS_FILE="$EXP_ROOT/status.tsv"
if [[ ! -f "$STATUS_FILE" ]]; then
  printf "time\tdataset\tbackbone\troute\tgpu\tstatus\tout_dir\tlog\n" > "$STATUS_FILE"
fi

running_jobs=()

gpu_used_table() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits
}

gpu_is_reserved() {
  local gpu="$1"
  local entry entry_gpu
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
  if [[ "${#running_jobs[@]}" -eq 0 ]]; then
    return 0
  fi
  local alive=()
  local entry pid
  for entry in "${running_jobs[@]}"; do
    pid="${entry%%:*}"
    if kill -0 "$pid" 2>/dev/null; then
      alive+=("$entry")
    else
      wait "$pid" || true
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

layers_args_for_backbone() {
  local backbone="$1"
  if [[ "$backbone" == "gcn" || "$backbone" == "deepgcn" ]]; then
    echo "--num-gnn-layers 4"
  else
    echo ""
  fi
}

launch_job() {
  local dataset="$1"
  local backbone="$2"
  local route="$3"
  local gpu="$4"
  local sample_size mode_arg dataset_key out_dir log_file
  local layers_args=()

  sample_size="$(sample_size_for_dataset "$dataset")"
  mode_arg="$(mode_for_route "$route")"
  dataset_key="${dataset//-/_}"
  out_dir="$EXP_ROOT/$dataset_key/$backbone/$route"
  log_file="$out_dir/run.log"

  if [[ "$backbone" == "gcn" || "$backbone" == "deepgcn" ]]; then
    layers_args=(--num-gnn-layers 4)
  fi

  mkdir -p "$out_dir"
  cat > "$out_dir/command.txt" <<CMD
CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src $PYTHON_BIN scripts/run_case_study.py --dataset $dataset --data-root $DATA_ROOT --out-dir $out_dir --variants ${VARIANTS[*]} --seeds ${SEEDS[*]} --epochs $EPOCHS --warmup-epochs $WARMUP_EPOCHS --hidden-dim $HIDDEN_DIM --memory-rank $MEMORY_RANK --graph-sparsity $GRAPH_SPARSITY --param-sparsity $PARAM_SPARSITY --backbone $backbone ${layers_args[*]} --node-sample-size $sample_size --node-sample-seed 0 --node-sample-mode $mode_arg --graph-memory-granularity subgraph --graph-memory-layout multi --param-memory-layout multi --graph-score-init topofeat --param-score-init magnitude --use-hidden-coupling --hidden-coupling-mix-graph $HIDDEN_COUPLING_MIX_GRAPH --hidden-coupling-mix-param $HIDDEN_COUPLING_MIX_PARAM --device cuda --verbose --log-every 50
CMD

  echo "[$(date '+%F %T')] Launching dataset=$dataset backbone=$backbone route=$route gpu=$gpu sample_size=$sample_size"
  (
    set +e
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
      --graph-score-init topofeat \
      --param-score-init magnitude \
      --use-hidden-coupling \
      --hidden-coupling-mix-graph "$HIDDEN_COUPLING_MIX_GRAPH" \
      --hidden-coupling-mix-param "$HIDDEN_COUPLING_MIX_PARAM" \
      --device cuda \
      --verbose \
      --log-every 50 \
      > "$log_file" 2>&1
    status="$?"
    if [[ "$status" -eq 0 ]]; then
      printf "%s\t%s\t%s\t%s\t%s\tcomplete\t%s\t%s\n" "$(date '+%F %T')" "$dataset" "$backbone" "$route" "$gpu" "$out_dir" "$log_file" >> "$STATUS_FILE"
    else
      printf "%s\t%s\t%s\t%s\t%s\tfailed_%s\t%s\t%s\n" "$(date '+%F %T')" "$dataset" "$backbone" "$route" "$gpu" "$status" "$out_dir" "$log_file" >> "$STATUS_FILE"
    fi
    exit "$status"
  ) &
  running_jobs+=("$!:$gpu")
}

echo "[$(date '+%F %T')] EXP066 launcher started."
echo "FREE_MEM_MIB=$FREE_MEM_MIB MAX_PARALLEL=$MAX_PARALLEL POLL_SECONDS=$POLL_SECONDS"
echo "DATASETS=${DATASETS[*]}"
echo "BACKBONES=${BACKBONES[*]}"
echo "ROUTES=${ROUTES[*]}"
echo "VARIANTS=${VARIANTS[*]}"
echo "SEEDS=${SEEDS[*]} EPOCHS=$EPOCHS"

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
      wait_for_slot
      gpu="$(wait_for_gpu)"
      launch_job "$dataset" "$backbone" "$route" "$gpu"
      sleep 10
    done
  done
done

echo "[$(date '+%F %T')] All jobs launched; waiting for completion."
while [[ "${#running_jobs[@]}" -gt 0 ]]; do
  reap_finished_jobs
  sleep "$POLL_SECONDS"
done
echo "[$(date '+%F %T')] EXP066 launcher finished."
