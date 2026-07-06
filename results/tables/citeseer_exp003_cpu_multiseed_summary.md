# OUGP Case Study Summary

Dataset: `citeseer`

| Variant | Best Test Acc | Graph Sparsity | Param Sparsity | Graph Churn | Param Churn |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense | 0.7160 +/- 0.0042 | 0.000 | 0.000 | 0.000 | 0.000 |
| dual_static | 0.7150 +/- 0.0036 | 0.300 | 0.300 | 0.003 | 0.003 |
| graph_only | 0.7143 +/- 0.0009 | 0.300 | 0.000 | 0.009 | 0.000 |
| ougp | 0.7157 +/- 0.0025 | 0.300 | 0.300 | 0.011 | 0.003 |
| ougp_no_cross | 0.7160 +/- 0.0022 | 0.300 | 0.300 | 0.013 | 0.003 |
| param_only | 0.7153 +/- 0.0009 | 0.000 | 0.300 | 0.000 | 0.003 |
