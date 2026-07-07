# OUGP Case Study Summary

Dataset: `cora`
Backbone: `gcn`

| Variant | Best Test Acc | Graph Sparsity | Param Sparsity | Message Cost Red. | Param Cost Red. | Graph Churn | Param Churn |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| degree_magnitude_static | 0.5180 +/- 0.0000 | 0.200 | 0.200 | 0.161 | 0.200 | 0.067 | 0.073 |
| ougp | 0.5190 +/- 0.0000 | 0.200 | 0.200 | 0.161 | 0.200 | 0.067 | 0.067 |
| random_static | 0.5220 +/- 0.0000 | 0.200 | 0.200 | 0.161 | 0.200 | 0.067 | 0.101 |
| similarity_magnitude_static | 0.5200 +/- 0.0000 | 0.200 | 0.200 | 0.161 | 0.200 | 0.067 | 0.073 |
