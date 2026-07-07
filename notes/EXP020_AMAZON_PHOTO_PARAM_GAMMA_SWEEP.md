# EXP020 Amazon Photo Param Gamma Sweep

## Purpose

After adding channel-specific parameter memory, EXP019 showed that memory was no longer a global bias, but `param_gamma=2.0` made parameter-mask churn too large. This experiment sweeps `param_gamma` to find a gentler feedback strength.

## Setup

- Dataset: Amazon Photo
- Model: 2-layer GCN
- Method: OUGP with channel-specific parameter memory
- Seeds: 0, 1, 2
- Epochs: 120
- Graph sparsity: 30%
- Parameter sparsity: 30%
- Graph gamma: 2.0
- Recall gamma: 0.20
- Swept parameter gamma: 0.1, 0.2, 0.5, 1.0
- EXP019 provides the `param_gamma=2.0` comparison.

## Results

| param_gamma | Best Test Acc | Param Churn | Conclusion |
| ---: | ---: | ---: | --- |
| 0.1 | 0.7495 +/- 0.0134 | 0.005 | Best stable setting in this sweep |
| 0.2 | 0.7480 +/- 0.0169 | 0.022 | Still reasonable, stronger feedback |
| 0.5 | 0.7153 +/- 0.0161 | 0.125 | Too much feedback, mask becomes unstable |
| 1.0 | 0.7295 +/- 0.0314 | 0.119 | Too much feedback, seed variance increases |
| 2.0 | 0.7281 +/- 0.0170 | 0.114 | Too much feedback, from EXP019 |

## Interpretation

The channel-specific memory implementation works: changing `param_gamma` changes parameter-mask churn and accuracy. The problem in EXP019 was not that memory failed, but that feedback strength was too large.

The best current setting is `param_gamma=0.1`. It keeps 30% graph sparsity and 30% parameter sparsity while keeping parameter churn low. It is close to the old OUGP baseline, but does not clearly exceed it yet.

## Next Step

Use `param_gamma=0.1` as the stable default for channel-specific memory. The next useful sweep is not a larger `param_gamma`; it should test weaker correction normalization, for example mean-centering without forcing correction std to 1.
