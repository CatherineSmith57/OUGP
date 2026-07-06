# OUGP Case Study Summary

Dataset: `photo`

| Variant | Best Test Acc | Graph Sparsity | Param Sparsity | Graph Churn | Param Churn |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense | 0.8124 +/- 0.0160 | 0.000 | 0.000 | 0.000 | 0.000 |
| dual_static | 0.7973 +/- 0.0180 | 0.300 | 0.100 | 0.003 | 0.001 |
| graph_only | 0.8281 +/- 0.0085 | 0.300 | 0.000 | 0.015 | 0.000 |
| ougp | 0.8131 +/- 0.0157 | 0.300 | 0.100 | 0.014 | 0.001 |
| ougp_no_cross | 0.8112 +/- 0.0186 | 0.300 | 0.100 | 0.018 | 0.001 |
| param_only | 0.7924 +/- 0.0212 | 0.000 | 0.100 | 0.000 | 0.001 |
