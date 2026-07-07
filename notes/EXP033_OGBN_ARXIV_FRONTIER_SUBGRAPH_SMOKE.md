# EXP033: ogbn-arxiv Frontier Subgraph Smoke

## Purpose

EXP032 showed that random node-sampled induced subgraphs can be too sparse on large graphs.

This experiment adds and verifies a neighbor-aware frontier sampling mode:

```text
--node-sample-mode frontier
```

The goal is to move the large-graph validation route closer to the original OUGP idea of sampled subgraph training.

Important limitation:

```text
This is still a static sampled-subgraph smoke run.
It is not full mini-batch neighbor sampling yet.
```

## Code Change

Runner:

```text
scripts/run_case_study.py
```

New argument:

```text
--node-sample-mode random|frontier
```

Sampling behavior:

```text
random:
  preserve train/val/test seed coverage
  fill remaining sampled nodes uniformly at random

frontier:
  preserve train/val/test seed coverage
  expand from those seeds through graph neighbors
  if frontier expansion is insufficient, fill the rest uniformly at random
```

The result JSON records:

```text
node_sample_mode
```

## Verification

Unit tests:

```text
PYTHONPATH=src /home/shizitong/miniconda3/envs/tianjiaying/bin/python -m pytest tests/test_subgraph_sampling.py -q
```

Result:

```text
4 passed
```

Regression tests:

```text
PYTHONPATH=src /home/shizitong/miniconda3/envs/tianjiaying/bin/python -m pytest tests/test_memory_recall_steering.py -q
```

Result:

```text
11 passed
```

## GPU Preflight

GPU status before the run:

```text
GPU 0: 10 MiB / 24576 MiB, 0% util
GPU 5: 10 MiB / 24576 MiB, 0% util
GPU 6: 10 MiB / 24576 MiB, 0% util
GPU 7: 10 MiB / 24576 MiB, 0% util
```

The run used:

```text
CUDA_VISIBLE_DEVICES=0
tianjiaying conda environment
1 RTX 3090 visible to the run
```

This respects the workspace cap of at most 4 GPUs.

## Command

```text
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /home/shizitong/miniconda3/envs/tianjiaying/bin/python scripts/run_case_study.py \
  --dataset ogbn-arxiv \
  --data-root data/raw/planetoid \
  --out-dir experiments/exp033_ogbn_arxiv_frontier_subgraph_smoke \
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
  --node-sample-mode frontier \
  --device cuda \
  --verbose \
  --log-every 1
```

## Result

| Dataset | Mode | Original Nodes | Sampled Nodes | Original Edges | Sampled Edges | Best Test Acc | Graph Sparsity | Param Sparsity | Message Cost Red. | Param Cost Red. |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ogbn-arxiv | random | 169343 | 5000 | 1166243 | 1247 | 0.0456 | 0.200 | 0.200 | 0.061 | 0.200 |
| ogbn-arxiv | frontier | 169343 | 5000 | 1166243 | 24160 | 0.1664 | 0.200 | 0.200 | 0.184 | 0.200 |

## Interpretation

Frontier sampling is a better large-graph smoke route than random node sampling:

```text
same sampled node count: 5000
random induced edges: 1247
frontier induced edges: 24160
```

This means the sampled graph retains much more local message-passing structure.

The accuracy is still not a final performance result because this is only a 2-epoch smoke run. The useful conclusion is:

```text
OUGP can now run on a denser ogbn-arxiv sampled subgraph with graph pruning, parameter pruning, memory read/write, and recorded resource metrics.
```

## Next Step

The next large-graph implementation step should be true mini-batch/subgraph training:

```text
sample a fresh frontier/neighbor subgraph per epoch or step
maintain OUGP memory across sampled subgraphs
write subgraph-level utility residuals into memory
compare Edge-State Write vs Subgraph-State Write
```

That would directly target the original idea's Subgraph-State Write design.

## Outputs

```text
experiments/exp033_ogbn_arxiv_frontier_subgraph_smoke/
results/tables/exp033_ogbn_arxiv_frontier_subgraph_smoke_summary.csv
results/tables/exp033_ogbn_arxiv_frontier_subgraph_smoke_summary.md
```

