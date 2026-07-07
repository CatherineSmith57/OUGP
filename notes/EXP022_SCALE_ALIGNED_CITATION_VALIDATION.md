# EXP022 Scale-Aligned Channel Memory on Citation Datasets

## Purpose

EXP021 showed that scale-aligning channel memory and recall correction to the `param_logits` scale stabilized Amazon Photo. This experiment checks whether the same mechanism is also stable on smaller citation datasets.

## Setup

- Datasets: Cora, CiteSeer, PubMed
- Model: 2-layer GCN
- Method: OUGP with channel-specific parameter memory
- Scale alignment: channel memory and recall correction are scaled by EMA of `param_logits.std()`
- Seeds: 0, 1, 2
- Epochs: 120
- Graph sparsity: 30%
- Parameter sparsity: 30%
- Graph gamma: 2.0
- Parameter gamma: 0.2
- Recall gamma: 0.20

## Results

| Dataset | Best Test Acc | Graph Sparsity | Param Sparsity | Graph Churn | Param Churn |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cora | 0.8083 +/- 0.0106 | 0.300 | 0.300 | 0.011 | 0.003 |
| CiteSeer | 0.7143 +/- 0.0052 | 0.300 | 0.300 | 0.006 | 0.003 |
| PubMed | 0.7897 +/- 0.0009 | 0.300 | 0.300 | 0.005 | 0.003 |

## Comparison With Old OUGP

| Dataset | Old OUGP | Scale-Aligned Channel Memory | Change |
| --- | ---: | ---: | ---: |
| Cora | 0.8060 | 0.8083 | +0.0023 |
| CiteSeer | 0.7160 | 0.7143 | -0.0017 |
| PubMed | 0.7890 | 0.7897 | +0.0007 |

These differences are small. The key result is not a large accuracy gain, but stability: parameter churn remains around `0.003` on all three datasets.

## Interpretation

The scale-aligned channel memory does not damage the citation-network experiments. This supports the current design:

```text
channel-specific memory gives parameter-level discrimination
scale alignment prevents memory feedback from dominating param_logits
```

The mechanism is therefore stable on Cora, CiteSeer, PubMed, and Amazon Photo under the current 2-layer GCN setting.

## Current Conclusion

Scale-aligned channel memory is a safer replacement for the earlier unscaled channel memory. It avoids the parameter-mask instability observed when memory correction and `param_logits` used incompatible scales.
