# OUGP Case Study Summary

Dataset: `pubmed`

| Variant | Best Test Acc | Graph Sparsity | Param Sparsity | Graph Churn | Param Churn |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense | 0.7900 +/- 0.0016 | 0.000 | 0.000 | 0.000 | 0.000 |
| dual_static | 0.7890 +/- 0.0024 | 0.300 | 0.300 | 0.003 | 0.003 |
| graph_only | 0.7927 +/- 0.0019 | 0.300 | 0.000 | 0.005 | 0.000 |
| ougp | 0.7890 +/- 0.0024 | 0.300 | 0.300 | 0.008 | 0.003 |
| ougp_no_cross | 0.7900 +/- 0.0036 | 0.300 | 0.300 | 0.006 | 0.003 |
| param_only | 0.7913 +/- 0.0017 | 0.000 | 0.300 | 0.000 | 0.003 |
