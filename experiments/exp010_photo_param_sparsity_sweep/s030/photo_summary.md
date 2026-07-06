# OUGP Case Study Summary

Dataset: `photo`

| Variant | Best Test Acc | Graph Sparsity | Param Sparsity | Graph Churn | Param Churn |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense | 0.8123 +/- 0.0160 | 0.000 | 0.000 | 0.000 | 0.000 |
| dual_static | 0.7157 +/- 0.0138 | 0.300 | 0.300 | 0.003 | 0.003 |
| graph_only | 0.8263 +/- 0.0092 | 0.300 | 0.000 | 0.014 | 0.000 |
| ougp | 0.7503 +/- 0.0119 | 0.300 | 0.300 | 0.014 | 0.003 |
| ougp_no_cross | 0.7506 +/- 0.0136 | 0.300 | 0.300 | 0.015 | 0.003 |
| param_only | 0.7126 +/- 0.0123 | 0.000 | 0.300 | 0.000 | 0.003 |
