# EXP022 Scale-Aligned Channel Memory on Citation Datasets

Model: 2-layer GCN with scale-aligned channel-specific parameter memory.

Setting: 30% graph sparsity + 30% parameter sparsity, 3 seeds, 120 epochs.

| Dataset | Best Test Acc | Old OUGP Best Test Acc | Graph Sparsity | Param Sparsity | Graph Churn | Param Churn | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Cora | 0.8083 +/- 0.0106 | 0.8060 | 0.300 | 0.300 | 0.011 | 0.003 | Stable; matches old OUGP level |
| CiteSeer | 0.7143 +/- 0.0052 | 0.7160 | 0.300 | 0.300 | 0.006 | 0.003 | Stable; close to old OUGP |
| PubMed | 0.7897 +/- 0.0009 | 0.7890 | 0.300 | 0.300 | 0.005 | 0.003 | Stable; matches old OUGP level |

Main finding: scale-aligned channel memory is stable on Cora, CiteSeer, and PubMed. It does not introduce the high parameter-mask churn observed in the unscaled Amazon Photo experiment.
