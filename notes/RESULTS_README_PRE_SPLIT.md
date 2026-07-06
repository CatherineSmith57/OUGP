# OUGP Results

Each case-study run directory should contain:

- `command.txt` - exact command used for the run.
- `manifest.json` - arguments, result paths, and final metrics.
- `*_seed*.json` - config, per-epoch history, and final result for one variant/seed.
- `*_summary.csv` - machine-readable final metrics.
- `*_summary.md` - compact human-readable table.

Existing result directories:

- `ougp_case_study_v2/` - primary Cora case study.
- `ougp_case_study_s60/` - 60%/60% sparsity pressure probe.
- `ougp_gamma_probe_g2/` - memory/cross strength probe.
- `ougp_case_study_smoke*/` - short smoke-test outputs.

GPU policy: every future run must expose at most 4 GPUs.

