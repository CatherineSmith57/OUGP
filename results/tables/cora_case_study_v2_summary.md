# OUGP Case Study Summary

Dataset: `cora`

| Variant | Best Test Acc | Graph Sparsity | Param Sparsity | Graph Churn | Param Churn |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense | 0.8030 +/- 0.0000 | 0.000 | 0.000 | 0.000 | 0.000 |
| dual_static | 0.8120 +/- 0.0000 | 0.300 | 0.300 | 0.004 | 0.004 |
| graph_only | 0.8120 +/- 0.0000 | 0.300 | 0.000 | 0.008 | 0.000 |
| ougp | 0.8090 +/- 0.0000 | 0.300 | 0.300 | 0.009 | 0.004 |
| ougp_no_cross | 0.8080 +/- 0.0000 | 0.300 | 0.300 | 0.009 | 0.004 |
| param_only | 0.8060 +/- 0.0000 | 0.000 | 0.300 | 0.000 | 0.004 |
