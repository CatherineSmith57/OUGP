# EXP031 DeeperGCN Backbone Smoke

| Dataset | Backbone | Layers | Variant | Best Test Acc | Graph Sparsity | Param Sparsity | Message Cost Red. | Param Cost Red. | Graph Churn | Param Churn |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cora | deepgcn | 4 | dense | 0.3160 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Cora | deepgcn | 4 | ougp | 0.3130 | 0.200 | 0.200 | 0.165 | 0.203 | 0.067 | 0.067 |

Seed: `0`; epochs: `4`. This is a DeeperGCN backbone smoke run, not a performance comparison.
