# OUGP

Online Unified Graph and Parameter Pruning for centralized GNNs.

This repository contains the current OUGP implementation with **Layer-wise Hidden Coupling Utility (LHCU)**. OUGP jointly prunes graph edges and hidden channels, while LHCU measures how graph pruning and parameter pruning perturb hidden states and writes that coupling signal into the existing graph/parameter memory update.

## What Is Included

```text
src/ougp/
  data.py      # dataset loaders and subgraph sampling inputs
  model.py     # GNN backbones, OUGP memory, LHCU, pruning masks
  trace.py     # pruning trace utilities

scripts/
  run_case_study.py
  run_hidden_state_diagnostic.py
  run_exp064_full_graph_hidden_coupling_validation.sh
  run_exp065_full_graph_backbone_queue_wait_gpu.sh
  run_exp066_ogbn_sampled_lhcu_backbones_wait_gpu.sh

configs/       # older reproducible presets
data/          # dataset cache; raw data is not committed
tests/         # regression tests for memory, recall, and sampling
```

Experiment outputs under `experiments/` and summary tables under `results/` are local artifacts and are not required to run the code.

## Environment

Create the environment from the provided file:

```bash
conda env create -f environment.yml
conda activate ougp
```

The code expects PyTorch, NumPy, SciPy, scikit-learn, pytest, and OGB. The `environment.yml` includes the Python packages needed by the experiment scripts.

When running Python commands manually, use:

```bash
export PYTHONPATH=src
```

The launcher scripts set `PYTHONPATH=src` themselves.

## Datasets

Raw datasets are **not committed**. They are downloaded on demand:

- Cora / CiteSeer / PubMed: Planetoid raw files from the public Planetoid repository.
- Amazon Photo: public `amazon_electronics_photo.npz` from gnn-benchmark.
- ogbn-arxiv / ogbn-products / ogbn-proteins: downloaded through `ogb.nodeproppred.NodePropPredDataset`.

Default data root:

```text
data/raw/planetoid
```

For OGBN, the loader redirects this to:

```text
data/raw/ogb/
```

The first run needs network access. If the server cannot access the internet, pre-populate the corresponding raw dataset directories under `data/raw/`.

## Quick Smoke Test

Run from the repository root:

```bash
PYTHONPATH=src python scripts/run_case_study.py \
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

## Main Experiment Entrypoints

### Full-Graph Small/Medium Graph Validation

Runs Cora, CiteSeer, PubMed, and Amazon Photo with 4-layer GCN, comparing Dense and OUGP+LHCU:

```bash
bash scripts/run_exp064_full_graph_hidden_coupling_validation.sh
```

### Full-Graph Backbone Validation

Runs full-graph validation across GraphSAGE, GAT, and DeeperGCN. OGBN full-graph OUGP is expected to hit CUDA OOM on 24GB GPUs; see the note below.

```bash
bash scripts/run_exp065_full_graph_backbone_queue_wait_gpu.sh
```

### OGBN Sampled-Subgraph Validation

Recommended large-graph validation. Runs ogbn-arxiv, ogbn-products, and ogbn-proteins with GCN, GraphSAGE, GAT, and DeeperGCN on both random and frontier subgraphs:

```bash
bash scripts/run_exp066_ogbn_sampled_lhcu_backbones_wait_gpu.sh
```

The script waits for free GPUs and runs up to four jobs in parallel by default.

## Hidden-State Diagnostic

To inspect graph/parameter hidden-state coupling:

```bash
PYTHONPATH=src python scripts/run_hidden_state_diagnostic.py \
  --dataset cora \
  --out-dir experiments/exp063_cora_hidden_state_diagnostic \
  --seeds 0 1 2 3 4 \
  --epochs 200 \
  --backbone gcn \
  --num-gnn-layers 4
```

The diagnostic compares dense, graph-only, parameter-only, full OUGP, and no-cross masks under the same model snapshot.

## Known Practical Limits

Full-graph OUGP on OGBN datasets can exceed 24GB GPU memory:

- ogbn-arxiv full-graph OUGP backward may require more than 100GB.
- ogbn-products full edge similarity initialization is too large for 24GB GPUs.
- ogbn-proteins full-graph memory read/correction can also OOM.

For large graphs, use sampled subgraph experiments (`random` or `frontier`) unless graph correction and memory read/write are changed to chunked mini-batch execution.

## Testing

Recommended checks:

```bash
PYTHONPATH=src python -m pytest tests/test_memory_recall_steering.py tests/test_subgraph_sampling.py
python -m py_compile src/ougp/model.py src/ougp/data.py scripts/run_case_study.py scripts/run_hidden_state_diagnostic.py
bash -n scripts/run_exp064_full_graph_hidden_coupling_validation.sh
bash -n scripts/run_exp065_full_graph_backbone_queue_wait_gpu.sh
bash -n scripts/run_exp066_ogbn_sampled_lhcu_backbones_wait_gpu.sh
```

## Output Format

Each run writes:

- `command.txt`: command used for the run
- `manifest.json`: arguments and final result metadata
- `*_seed*.json`: per-seed config, epoch history, and final metrics
- `*_summary.csv` / `*_summary.md`: aggregate result table

Large launcher scripts also write `launcher.log` and `status.tsv`.
