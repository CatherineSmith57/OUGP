# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| R001 | M0 | demo sanity | gated delta-memory | synthetic seen keys | empty MSE, post-write MSE | MUST | DONE | empty 0.9227, post-write 0.7571 |
| R002 | M1 | capacity sweep | key_dim 8/16/32/64 | synthetic seen keys | post-write MSE | MUST | DONE | `results/capacity_sweep.csv`; n=64 gated reduction improves from -0.007 to 0.288 |
| R003 | M1 | pressure sweep | n_items 32/64/128 | synthetic seen keys | MSE reduction | MUST | DONE | covered by capacity script |
| R004 | M2 | gate ablation | gated vs always-write | synthetic seen keys | MSE, state norm | MUST | DONE | always-write is worse in most small-capacity settings |
| R005 | M3 | delayed retrieval | A then distractor B then query A | synthetic delayed keys | delayed recall MSE | MUST | TODO | next script after demo |
| R006 | M4 | tiny steering | predictor with/without memory readout | synthetic delayed task | accuracy or MSE | NICE | TODO | only after M0-M3 |
| R007 | M5 | OUGP case study | Cora + GCN: dense, graph-only, param-only, dual-static, OUGP no-cross, OUGP | Planetoid split | test accuracy, graph/parameter sparsity, soft-mask churn | MUST | DONE | `results/ougp_case_study_v2/cora_summary.csv`; OUGP is runnable at 30%/30% sparsity but current memory gain is inconclusive |
