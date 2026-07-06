# EXP015 Amazon Photo Recall + Steering Memory

## 目的

验证最新版 recoverable pruning memory 是否能缓解 Amazon Photo 上 30% parameter pruning 导致的准确率下降。

本轮测试三种新机制：

1. `recall_only`: 对被剪掉但 gradient utility 仍高的边/参数通道增加 recall bias，让它们后续有机会被补回来。
2. `steering_only`: 使用 delta-mem 风格的 `MemorySteeringMLP`，从 online state 读出 hidden correction，直接加到 GCN hidden representation。
3. `recall_steering`: 同时打开 recall memory 和 steering MLP。

## 设置

| 项目 | 设置 |
| --- | --- |
| Dataset | Amazon Photo |
| Backbone | 2-layer GCN |
| Seeds | 0, 1, 2 |
| Epochs | 120 |
| Graph sparsity | 30% |
| Parameter sparsity | 30% |
| Hidden dim | 64 |
| Memory rank | 16 |
| recall_gamma | 0.20 |
| recall_top_k | 2000 |
| steer_gamma | 0.10 |
| steer_beta / steer_lambda | 0.20 / 0.90 |
| GPU | 1 GPU, `tianjiaying` env |

配置文件：

```text
configs/photo_recall_steering_sweep_v1.json
```

启动脚本：

```text
scripts/run_exp015_photo_recall_steering_sweep.sh
```

原始输出：

```text
experiments/exp015_photo_recall_steering_sweep/
```

汇总表：

```text
results/tables/photo_exp015_recall_steering_summary.csv
results/tables/photo_exp015_recall_steering_summary.md
```

## 结果

| Setting | Variant | Best Test Acc | Delta vs OUGP Baseline |
| --- | --- | ---: | ---: |
| baseline | graph_only | 0.8261 +/- 0.0106 | +0.0763 |
| baseline | ougp | 0.7498 +/- 0.0109 | 0.0000 |
| recall_only | ougp | 0.7497 +/- 0.0134 | -0.0001 |
| steering_only | ougp | 0.6386 +/- 0.0577 | -0.1112 |
| recall_steering | ougp | 0.6308 +/- 0.0474 | -0.1190 |

## 观察

1. `recall_only` 基本没有改善 OUGP：`0.7498 -> 0.7497`。
2. `steering_only` 明显伤害准确率：`0.7498 -> 0.6386`。
3. `recall_steering` 也明显下降：`0.7498 -> 0.6308`。
4. `graph_only` 仍然明显高于 OUGP，说明 Amazon Photo 当前主要问题还是 parameter pruning 伤害表达能力。
5. JSON 中 `graph_recall_bias_norm`、`param_recall_bias_norm`、`steering_memory_state_norm` 非零，说明机制不是没接上，而是当前更新规则/强度没有带来有效恢复。

## 当前结论

Amazon Photo 上，30% parameter pruning 之后目前不能保证 accuracy。

本轮 recoverable memory 的第一版没有解决该问题：

```text
recall memory 能记录和写入恢复偏置，但没有拉回准确率；
steering MLP 会直接改变 hidden representation，但当前梯度均值写入规则过强或方向不稳，导致性能下降。
```

## 下一步建议

1. 先不要继续扩大 steering MLP 实验；当前 `steer_gamma=0.10` 太激进。
2. 下一轮应先调小 steering 强度，例如 `steer_gamma=0.005 / 0.01 / 0.02`。
3. recall memory 可以保留，但需要重点改 parameter-channel recovery，而不是只依赖 mask-drop 信号。
4. 更稳妥的方向是 parameter pruning schedule：降低剪枝率、延迟参数剪枝、或逐步从 5%/10% 开始恢复。
