# OUGP Case Study Summary

Dataset: `photo`

| Variant | Best Test Acc | Graph Sparsity | Param Sparsity | Graph Churn | Param Churn |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense | 0.8124 +/- 0.0161 | 0.000 | 0.000 | 0.000 | 0.000 |
| dual_static | 0.7159 +/- 0.0139 | 0.300 | 0.300 | 0.003 | 0.003 |
| graph_only | 0.8275 +/- 0.0102 | 0.300 | 0.000 | 0.010 | 0.000 |
| ougp | 0.7490 +/- 0.0136 | 0.300 | 0.300 | 0.011 | 0.003 |
| ougp_no_cross | 0.7514 +/- 0.0137 | 0.300 | 0.300 | 0.011 | 0.003 |
| param_only | 0.7116 +/- 0.0131 | 0.000 | 0.300 | 0.000 | 0.003 |
