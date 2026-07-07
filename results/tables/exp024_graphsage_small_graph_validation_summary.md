# EXP024 GraphSAGE Small-Graph Validation

| Dataset | Variant | Best Test Acc | Graph Sparsity | Param Sparsity | Graph Churn | Param Churn |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Cora | dense | 0.7030 | 0.000 | 0.000 | 0.000 | 0.000 |
| Cora | ougp | 0.6820 | 0.300 | 0.300 | 0.020 | 0.020 |
| CiteSeer | dense | 0.4740 | 0.000 | 0.000 | 0.000 | 0.000 |
| CiteSeer | ougp | 0.4390 | 0.300 | 0.300 | 0.020 | 0.020 |
| PubMed | dense | 0.7390 | 0.000 | 0.000 | 0.000 | 0.000 |
| PubMed | ougp | 0.7390 | 0.300 | 0.300 | 0.020 | 0.020 |

Backbone: `sage`; seed: `0`; epochs: `20`.
