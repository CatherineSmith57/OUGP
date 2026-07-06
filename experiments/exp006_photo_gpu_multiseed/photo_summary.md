# OUGP Case Study Summary

Dataset: `photo`

| Variant | Best Test Acc | Graph Sparsity | Param Sparsity | Graph Churn | Param Churn |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense | 0.8125 +/- 0.0161 | 0.000 | 0.000 | 0.000 | 0.000 |
| dual_static | 0.7161 +/- 0.0137 | 0.300 | 0.300 | 0.003 | 0.003 |
| graph_only | 0.8273 +/- 0.0095 | 0.300 | 0.000 | 0.015 | 0.000 |
| ougp | 0.7509 +/- 0.0136 | 0.300 | 0.300 | 0.022 | 0.003 |
| ougp_no_cross | 0.7513 +/- 0.0146 | 0.300 | 0.300 | 0.014 | 0.003 |
| param_only | 0.7121 +/- 0.0124 | 0.000 | 0.300 | 0.000 | 0.003 |
