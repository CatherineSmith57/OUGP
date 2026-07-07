# EXP032: Node-Sampled Induced Subgraph Smoke

## Purpose

This experiment adds a static node-sampled induced subgraph route to the case-study runner.

It is closer to the original OUGP idea than sampled-edge smoke because it keeps node features, labels, train/val/test masks, and local edge endpoints consistent after sampling nodes.

Important limitation:

```text
This is still not full mini-batch neighbor/subgraph training.
It is a static sampled-subgraph smoke path for large-graph feasibility checks.
```

## Code Change

Runner:

```text
scripts/run_case_study.py
```

New arguments:

```text
--node-sample-size
--node-sample-seed
```

Sampling order:

```text
load full dataset
  -> optionally sample nodes and build induced subgraph
  -> optionally sample edges
  -> run OUGP
```

The result JSON now records:

```text
original_num_nodes
original_num_edges
node_sample_size
edge_sample_size
num_nodes
num_edges
```

## Verification

Unit tests:

```text
PYTHONPATH=src /home/shizitong/miniconda3/envs/tianjiaying/bin/python -m pytest tests/test_subgraph_sampling.py -q
```

Result:

```text
2 passed
```

Regression tests:

```text
PYTHONPATH=src /home/shizitong/miniconda3/envs/tianjiaying/bin/python -m pytest tests/test_memory_recall_steering.py -q
```

Result:

```text
11 passed
```

## Smoke 1: Cora Node Subgraph

Directory:

```text
experiments/exp032_cora_node_subgraph_smoke/
```

Command:

```text
PYTHONPATH=src /home/shizitong/miniconda3/envs/tianjiaying/bin/python scripts/run_case_study.py \
  --dataset cora \
  --data-root data/raw/planetoid \
  --out-dir experiments/exp032_cora_node_subgraph_smoke \
  --variants ougp \
  --seeds 0 \
  --epochs 2 \
  --warmup-epochs 1 \
  --hidden-dim 16 \
  --memory-rank 4 \
  --graph-sparsity 0.20 \
  --param-sparsity 0.20 \
  --backbone gcn \
  --node-sample-size 1000 \
  --device cpu \
  --verbose \
  --log-every 1
```

Key result:

| Dataset | Original Nodes | Sampled Nodes | Original Edges | Sampled Edges | Best Test Acc | Graph Sparsity | Param Sparsity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cora | 2708 | 1000 | 10556 | 1502 | 0.4875 | 0.200 | 0.200 |

## Smoke 2: ogbn-arxiv Node Subgraph

Directory:

```text
experiments/exp032_ogbn_arxiv_node_subgraph_smoke/
```

GPU:

```text
CUDA_VISIBLE_DEVICES=0
tianjiaying conda environment
1 RTX 3090 visible to the run
```

Command:

```text
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /home/shizitong/miniconda3/envs/tianjiaying/bin/python scripts/run_case_study.py \
  --dataset ogbn-arxiv \
  --data-root data/raw/planetoid \
  --out-dir experiments/exp032_ogbn_arxiv_node_subgraph_smoke \
  --variants ougp \
  --seeds 0 \
  --epochs 2 \
  --warmup-epochs 1 \
  --hidden-dim 16 \
  --memory-rank 4 \
  --graph-sparsity 0.20 \
  --param-sparsity 0.20 \
  --backbone gcn \
  --node-sample-size 5000 \
  --device cuda \
  --verbose \
  --log-every 1
```

Key result:

| Dataset | Original Nodes | Sampled Nodes | Original Edges | Sampled Edges | Best Test Acc | Graph Sparsity | Param Sparsity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ogbn-arxiv | 169343 | 5000 | 1166243 | 1247 | 0.0456 | 0.200 | 0.200 |

## Interpretation

This experiment verifies:

```text
node-sampled induced subgraph path works;
node ids are remapped correctly;
train/val/test masks are preserved after sampling;
OUGP can run read -> mask -> forward/backward -> memory write on the sampled large graph;
all run parameters, logs, JSON, and summary tables are saved.
```

It does not prove:

```text
large-graph performance;
true scalable mini-batch training;
final ogbn-arxiv accuracy.
```

The ogbn-arxiv sampled graph has only 1247 induced edges after random node sampling, so accuracy is expected to be poor in a 2-epoch smoke run. The next method step should be neighbor-aware or split-aware subgraph sampling, then true mini-batch/subgraph training.

## Outputs

Tables:

```text
results/tables/exp032_cora_node_subgraph_smoke_summary.csv
results/tables/exp032_cora_node_subgraph_smoke_summary.md
results/tables/exp032_ogbn_arxiv_node_subgraph_smoke_summary.csv
results/tables/exp032_ogbn_arxiv_node_subgraph_smoke_summary.md
```

