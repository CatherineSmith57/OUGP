# EXP029 Memory Write Mode Smoke

| Dataset | Backbone | Variant | Write Mode | Best Test Acc | Graph Sparsity | Param Sparsity | Message Cost Red. | Param Cost Red. | Graph Memory Norm | Param Memory Norm |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cora | gcn | ougp | residual | 0.5190 | 0.200 | 0.200 | 0.161 | 0.200 | 0.0020 | 0.2018 |
| Cora | gcn | ougp | feature | 0.5180 | 0.200 | 0.200 | 0.161 | 0.200 | 0.1068 | 0.7829 |
| Cora | gcn | ougp | none | 0.5180 | 0.200 | 0.200 | 0.161 | 0.200 | 0.0000 | 0.0000 |

Seed: `0`; epochs: `4`. This is a write-mode smoke run, not a performance comparison.
