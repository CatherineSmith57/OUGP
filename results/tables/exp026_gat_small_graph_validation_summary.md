# EXP026 GAT Small-Graph Validation

| Dataset | Variant | Best Test Acc | Graph Sparsity | Param Sparsity | Graph Churn | Param Churn |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Cora | dense | 0.7780 | 0.000 | 0.000 | 0.000 | 0.000 |
| Cora | ougp | 0.7680 | 0.300 | 0.300 | 0.020 | 0.020 |
| CiteSeer | dense | 0.5120 | 0.000 | 0.000 | 0.000 | 0.000 |
| CiteSeer | ougp | 0.5250 | 0.300 | 0.300 | 0.020 | 0.020 |
| PubMed | dense | 0.7220 | 0.000 | 0.000 | 0.000 | 0.000 |
| PubMed | ougp | 0.7240 | 0.300 | 0.300 | 0.020 | 0.020 |

Backbone: `gat`; seed: `0`; epochs: `20`.
