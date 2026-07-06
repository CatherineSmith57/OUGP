# OUGP Case Study Summary

Dataset: `cora`

| Variant | Best Test Acc | Graph Sparsity | Param Sparsity | Graph Churn | Param Churn |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense | 0.8083 +/- 0.0049 | 0.000 | 0.000 | 0.000 | 0.000 |
| dual_static | 0.8090 +/- 0.0114 | 0.300 | 0.300 | 0.003 | 0.003 |
| graph_only | 0.8083 +/- 0.0084 | 0.300 | 0.000 | 0.011 | 0.000 |
| ougp | 0.8060 +/- 0.0088 | 0.300 | 0.300 | 0.007 | 0.003 |
| ougp_no_cross | 0.8070 +/- 0.0083 | 0.300 | 0.300 | 0.008 | 0.003 |
| param_only | 0.8147 +/- 0.0073 | 0.000 | 0.300 | 0.000 | 0.003 |
