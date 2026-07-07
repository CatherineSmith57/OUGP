# EXP020 Amazon Photo Param Gamma Sweep

Dataset: Amazon Photo

Model: 2-layer GCN with channel-specific parameter memory.

Setting: 30% graph sparsity + 30% parameter sparsity, 3 seeds, 120 epochs.

| param_gamma | Best Test Acc | Graph Sparsity | Param Sparsity | Graph Churn | Param Churn | Interpretation |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.1 | 0.7495 +/- 0.0134 | 0.300 | 0.300 | 0.008 | 0.005 | Stable; close to old OUGP baseline |
| 0.2 | 0.7480 +/- 0.0169 | 0.300 | 0.300 | 0.015 | 0.022 | Still stable; mild channel-level feedback |
| 0.5 | 0.7153 +/- 0.0161 | 0.300 | 0.300 | 0.008 | 0.125 | Too strong; parameter mask churn becomes large |
| 1.0 | 0.7295 +/- 0.0314 | 0.300 | 0.300 | 0.021 | 0.119 | Too strong; unstable across seeds |
| 2.0 | 0.7281 +/- 0.0170 | 0.300 | 0.300 | 0.012 | 0.114 | Previous exp019 setting; too strong |

Main finding: channel-specific memory is active, but `param_gamma` must be small. The useful range is around `0.1` to `0.2`; larger values make the parameter mask change too aggressively.
