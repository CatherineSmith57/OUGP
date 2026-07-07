# EXP021 Amazon Photo Param-Logit Scaled Memory

Dataset: Amazon Photo

Model: 2-layer GCN with channel-specific parameter memory.

Setting: 30% graph sparsity + 30% parameter sparsity, 3 seeds, 120 epochs.

| Setting | Best Test Acc | Graph Sparsity | Param Sparsity | Graph Churn | Param Churn | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Old OUGP global parameter memory | 0.7503 +/- 0.0119 | 0.300 | 0.300 | 0.014 | 0.003 | Old baseline; parameter memory was weak/global |
| Channel memory, unscaled, gamma=0.2 | 0.7480 +/- 0.0169 | 0.300 | 0.300 | 0.015 | 0.022 | Active but not scale-aligned |
| Channel memory, unscaled, gamma=2.0 | 0.7281 +/- 0.0170 | 0.300 | 0.300 | 0.012 | 0.114 | Too strong; parameter mask churn is high |
| Channel memory, param-logit scaled, gamma=0.2 | 0.7550 +/- 0.0149 | 0.300 | 0.300 | 0.015 | 0.003 | Current best; stable and slightly above old OUGP |

Main finding: after mapping channel memory and recall correction into the `param_logits` scale, the parameter mask no longer oscillates. The method keeps channel-level memory active while restoring low parameter churn.
