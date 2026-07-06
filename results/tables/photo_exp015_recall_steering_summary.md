# EXP015 Amazon Photo Recall + Steering Sweep

| Setting | Variant | Best Test Acc | Delta vs OUGP Baseline | Graph Sparsity | Param Sparsity | Graph Churn | Param Churn |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | graph_only | 0.8261 +/- 0.0106 | +0.0763 | 0.300 | 0.000 | 0.010 | 0.000 |
| baseline | ougp | 0.7498 +/- 0.0109 | 0.0000 | 0.300 | 0.300 | 0.013 | 0.003 |
| recall_only | ougp | 0.7497 +/- 0.0134 | -0.0001 | 0.300 | 0.300 | 0.019 | 0.003 |
| steering_only | ougp | 0.6386 +/- 0.0577 | -0.1112 | 0.300 | 0.300 | 0.011 | 0.003 |
| recall_steering | ougp | 0.6308 +/- 0.0474 | -0.1190 | 0.300 | 0.300 | 0.012 | 0.006 |

Conclusion: recall memory did not recover Amazon Photo accuracy at 30% parameter sparsity, and the first steering MLP setting hurt accuracy substantially.
