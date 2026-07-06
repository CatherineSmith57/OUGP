# OUGP Experiments

This directory stores full experiment records. Keep raw per-run JSON, command files, manifests, logs, and summary files here.

- `exp001_cora_case_study_initial/`
  - `primary_v2/`: main Cora case study.
  - `initial_v1/`: earlier Cora case-study run.
- `exp002_cora_high_sparsity_probe/s60/`
  - 60% graph / 60% parameter sparsity pressure probe.
- `exp003_cora_gamma_probe/gamma_g2/`
  - memory and cross-strength probe.
- `smoke/`
  - short runs for path and training-loop verification.

Cleaned tables copied for paper use live in `../results/tables/`.
