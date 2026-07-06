# OUGP Case Study Summary

Dataset: `photo`

| Variant | Best Test Acc | Graph Sparsity | Param Sparsity | Graph Churn | Param Churn |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense | 0.8123 +/- 0.0160 | 0.000 | 0.000 | 0.000 | 0.000 |
| dual_static | 0.7644 +/- 0.0170 | 0.300 | 0.200 | 0.003 | 0.002 |
| graph_only | 0.8245 +/- 0.0161 | 0.300 | 0.000 | 0.019 | 0.000 |
| ougp | 0.7895 +/- 0.0163 | 0.300 | 0.200 | 0.014 | 0.002 |
| ougp_no_cross | 0.7903 +/- 0.0173 | 0.300 | 0.200 | 0.011 | 0.002 |
| param_only | 0.7617 +/- 0.0167 | 0.000 | 0.200 | 0.000 | 0.002 |
