# OUGP Case Study Summary

Dataset: `photo`

| Variant | Best Test Acc | Graph Sparsity | Param Sparsity | Graph Churn | Param Churn |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense | 0.8124 +/- 0.0158 | 0.000 | 0.000 | 0.000 | 0.000 |
| dual_static | 0.8061 +/- 0.0197 | 0.300 | 0.050 | 0.003 | 0.000 |
| graph_only | 0.8273 +/- 0.0100 | 0.300 | 0.000 | 0.015 | 0.000 |
| ougp | 0.8188 +/- 0.0161 | 0.300 | 0.050 | 0.022 | 0.000 |
| ougp_no_cross | 0.8188 +/- 0.0178 | 0.300 | 0.050 | 0.017 | 0.001 |
| param_only | 0.8059 +/- 0.0181 | 0.000 | 0.050 | 0.000 | 0.000 |
