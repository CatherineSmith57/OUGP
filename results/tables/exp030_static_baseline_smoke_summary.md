# EXP030 Static Baseline Smoke

| Dataset | Backbone | Variant | Best Test Acc | Graph Sparsity | Param Sparsity | Message Cost Red. | Param Cost Red. | Graph Churn | Param Churn | Graph Init | Param Init | Frozen |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| Cora | gcn | random_static | 0.5220 | 0.200 | 0.200 | 0.161 | 0.200 | 0.067 | 0.101 | random | random | yes |
| Cora | gcn | degree_magnitude_static | 0.5180 | 0.200 | 0.200 | 0.161 | 0.200 | 0.067 | 0.073 | degree | magnitude | yes |
| Cora | gcn | similarity_magnitude_static | 0.5200 | 0.200 | 0.200 | 0.161 | 0.200 | 0.067 | 0.073 | similarity | magnitude | yes |
| Cora | gcn | ougp | 0.5190 | 0.200 | 0.200 | 0.161 | 0.200 | 0.067 | 0.067 | constant | constant | no |

Seed: `0`; epochs: `4`. This is a static-baseline smoke run, not a performance comparison.
