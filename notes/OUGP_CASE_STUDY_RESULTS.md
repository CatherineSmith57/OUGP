# OUGP Case Study Results

Date: 2026-07-05

## Goal

Run the first paper-oriented case study for `idea.md`: centralized GNN pruning with graph-level and parameter-level masks, online pruning memory, residual utility writes, and cross-level context.

## Implementation

- Dataset: Cora with the standard Planetoid split.
- Backbone: 2-layer GCN implemented with `torch.sparse.mm`.
- Method code:
  - `ougp/data.py`: Planetoid raw loader.
  - `ougp/model.py`: OUGP GCN, graph mask, hidden-channel mask, online pruning memories.
  - `ougp/run_case_study.py`: experiment runner, variants, JSON/CSV/Markdown result output.
- Environment used: `/home/shizitong/miniconda3/envs/atma/bin/python`, CPU.
- GPU constraint: future runs must expose at most 4 GPUs, e.g. `CUDA_VISIBLE_DEVICES=0,1,2,3`.

## Variants

- `dense`: dense GCN.
- `graph_only`: graph pruning only.
- `param_only`: structured hidden-channel pruning only.
- `dual_static`: graph + parameter pruning without online memory.
- `ougp_no_cross`: online memory without cross-level context.
- `ougp`: online memory with graph-parameter context.

## Command

```bash
cd /home/shizitong/tianjiaying/research/ougp
PYTHONPATH=src /home/shizitong/miniconda3/envs/atma/bin/python scripts/run_case_study.py \
  --dataset cora \
  --epochs 80 \
  --warmup-epochs 10 \
  --variants dense graph_only param_only dual_static ougp_no_cross ougp \
  --seeds 0 \
  --out-dir experiments/exp001_cora_case_study_initial/primary_v2 \
  --graph-gamma 2.0 \
  --param-gamma 2.0 \
  --write-beta 0.25 \
  --verbose \
  --log-every 20
```

## Results

| Variant | Best Test Acc | Graph Sparsity | Param Sparsity | Graph Churn | Param Churn |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense | 0.8030 | 0.000 | 0.000 | 0.000 | 0.000 |
| dual_static | 0.8120 | 0.300 | 0.300 | 0.004 | 0.004 |
| graph_only | 0.8120 | 0.300 | 0.000 | 0.008 | 0.000 |
| ougp | 0.8090 | 0.300 | 0.300 | 0.009 | 0.004 |
| ougp_no_cross | 0.8080 | 0.300 | 0.300 | 0.009 | 0.004 |
| param_only | 0.8060 | 0.000 | 0.300 | 0.000 | 0.004 |

Raw files:

- `research/ougp/experiments/exp001_cora_case_study_initial/primary_v2/cora_summary.csv`
- `research/ougp/experiments/exp001_cora_case_study_initial/primary_v2/cora_summary.md`
- `research/ougp/experiments/exp001_cora_case_study_initial/primary_v2/cora_*_seed0.json`
- `research/ougp/results/tables/cora_case_study_v2_summary.csv`

## Interpretation

Supported by this first case study:

- The OUGP training loop is executable end-to-end on a real citation graph.
- The current implementation can jointly prune graph edges and hidden channels to about 30% sparsity each.
- Joint pruning remains close to dense GCN accuracy on Cora under this light sparsity setting.

Not yet supported:

- Online residual memory is not yet clearly better than static dual pruning.
- Cross-level context currently gives only a tiny improvement over `ougp_no_cross` on one seed.
- The churn metric is now implemented as soft-mask movement, but more seeds and harder sparsity budgets are needed before claiming stability improvements.

Next recommended experiments:

- Run 3-5 seeds on Cora for the current variant set.
- Add PubMed/CiteSeer to test whether the OPM signal appears beyond Cora.
- Add a stricter high-sparsity setting such as 50%/50%, with matched realized sparsity.
- Add explicit ablations for state read-only, no residual write, EMA utility, and rank `r in {4,8,16,32}`.
