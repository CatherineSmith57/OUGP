# EXP029: Memory Write Mode Smoke

## 目的

补齐 idea 中的关键消融入口，比较 Online Pruning Memory 的不同写入方式：

```text
residual: 主方法，写入 utility prediction residual 调制后的 value
feature: 只写 context feature value，不使用 residual utility
none: 只读不写，跳过 online memory state 更新
```

这组是 Cora 上的 4 epoch smoke，只验证消融入口和状态行为，不作为性能结论。

## 设置

```text
dataset: Cora
backbone: GCN
variant: OUGP
seed: 0
epochs: 4
warmup epochs: 1
hidden dim: 16
memory rank: 4
graph sparsity: 20%
parameter sparsity: 20%
device: CPU
```

实验目录：

```text
experiments/exp029_memory_write_mode_smoke/
```

## 结果

| Write Mode | Best Test Acc | Graph Sparsity | Param Sparsity | Message Cost Red. | Param Cost Red. | Graph Memory Norm | Param Memory Norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| residual | 0.5190 | 0.200 | 0.200 | 0.161 | 0.200 | 0.0020 | 0.2018 |
| feature | 0.5180 | 0.200 | 0.200 | 0.161 | 0.200 | 0.1068 | 0.7829 |
| none | 0.5180 | 0.200 | 0.200 | 0.161 | 0.200 | 0.0000 | 0.0000 |

## 观察

```text
三种 memory_write_mode 都能正常训练并落盘。
none 模式下 graph/parameter memory state norm 保持 0，说明只读不写真正生效。
feature 模式下 state norm 明显大于 residual，说明只写特征会更快累积 state，但不一定更有用。
4 epoch 太短，accuracy 差异不能解释方法优劣。
```

## 后续正式消融

```text
Cora / CiteSeer / PubMed: 多 seed
Amazon Photo: 多 seed
对比 residual vs feature vs none 的 accuracy、churn、state norm、cost reduction
如果 residual 明显优于 feature/none，才能支持“residual utility write 有贡献”的论文论点。
```
