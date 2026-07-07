# EXP021 Amazon Photo Param-Score Scaled Memory

## Purpose

The previous channel-specific memory version solved the "global bias" problem, but its correction was not in the same scale as `param_logits`. This experiment tests a scale-aligned version:

```python
param_score =
    param_logits
  + param_gamma * param_logit_scale_ema * normalized_channel_memory
  + recall_gamma * param_logit_scale_ema * normalized_recall
```

The key idea is that `param_logits` is the main coordinate system for parameter pruning. Memory and recall should be expressed in the same units before being added to the score.

## Setup

- Dataset: Amazon Photo
- Model: 2-layer GCN
- Seeds: 0, 1, 2
- Epochs: 120
- Graph sparsity: 30%
- Parameter sparsity: 30%
- Graph gamma: 2.0
- Parameter gamma: 0.2
- Recall gamma: 0.20
- Parameter score scale: EMA of `param_logits.std()`
- Scale clamp: `[0.02, 0.50]`
- Correction clip: `[-2, 2]`

## Result

| Metric | Value |
| --- | ---: |
| Best test accuracy | 0.7550 +/- 0.0149 |
| Graph sparsity | 0.300 |
| Parameter sparsity | 0.300 |
| Graph churn | 0.015 |
| Parameter churn | 0.003 |

Per-seed best test accuracy:

| Seed | Best Test Acc |
| ---: | ---: |
| 0 | 0.7709 |
| 1 | 0.7588 |
| 2 | 0.7351 |

## Interpretation

This is the first Amazon Photo result where channel-specific memory is active and parameter churn remains low. Compared with the unscaled channel memory run, the improvement is mainly in stability:

| Setting | Best Test Acc | Param Churn |
| --- | ---: | ---: |
| Channel memory, unscaled, gamma=2.0 | 0.7281 +/- 0.0170 | 0.114 |
| Channel memory, unscaled, gamma=0.2 | 0.7480 +/- 0.0169 | 0.022 |
| Channel memory, param-logit scaled, gamma=0.2 | 0.7550 +/- 0.0149 | 0.003 |

The scale-aligned version supports the hypothesis that the earlier failure was not caused by channel memory itself, but by mixing correction signals with incompatible scales.

## Current Conclusion

The current best Amazon Photo setting is:

```text
channel-specific parameter memory
+ param-logit scale alignment
+ param_gamma = 0.2
+ recall_gamma = 0.20
```

It reaches 30% graph sparsity and 30% parameter sparsity while keeping parameter-mask churn close to the old stable baseline.
