# OUGP 实验阅读与复现小指南

这份指南给“还在学习阶段”的自己用：每次跑完实验后，知道去哪里看、先看什么、每个参数和指标是什么意思，以及怎样判断结果能不能写进论文。

## 1. 先记住目录分工

在 `research/ougp/` 里：

```text
src/ougp/        # 方法和模型代码
scripts/         # 跑实验的入口脚本
configs/         # 重要实验参数备份
data/raw/        # 原始数据
experiments/     # 每一次实验的完整记录
results/tables/  # 整理后的最终表格
notes/           # 想法、实验解释、tracker
```

看结果时优先顺序：

1. 先看 `results/tables/*.csv` 或 `*.md`，这是整理后的最终表格。
2. 再看对应的 `experiments/.../command.txt`，确认实验怎么跑的。
3. 再看 `experiments/.../*_seed*.json`，里面有每个 epoch 的历史曲线和最终指标。

## 1.1 当前用的是什么模型

当前实现的 backbone 是 **两层 GCN**，代码类名是：

```text
src/ougp/model.py
OUGPGCN
```

可以把它理解成：

```text
输入节点特征 x
  -> 第一次 GCN 消息传播
  -> Linear(in_dim, hidden_dim)
  -> ReLU
  -> parameter/channel mask
  -> dropout
  -> 第二次 GCN 消息传播
  -> Linear(hidden_dim, num_classes)
  -> 分类 logits
```

它和普通 GCN 的区别是多了两类可学习 mask：

| 组件 | 作用 | 对应实验指标 |
| --- | --- | --- |
| `graph_mask` | 给每条边/message 一个保留权重，用来做 graph sparsity | `graph_keep`, `graph_sparsity` |
| `param_mask` | 给 hidden channel 一个保留权重，用来做 parameter/channel sparsity | `param_keep`, `param_sparsity` |
| `graph_memory` | 根据边的 context 动态修正 graph mask score | `graph_memory_state_norm` |
| `param_memory` | 根据通道/参数 context 动态修正 param mask score | `param_memory_state_norm` |
| `cross context` | graph 和 parameter 两侧互相提供当前保留率信息 | `ougp` vs `ougp_no_cross` |

所以论文里可以写：

> We use a two-layer GCN as the backbone and augment it with online graph-edge and hidden-channel pruning masks.

注意：当前的 `parameter sparsity` 更准确地说是 **hidden channel sparsity / parameter-side sparsity**。它不是把每一个矩阵元素单独剪掉，而是在 hidden dimension 上做通道级 mask，因此会直接影响模型表达能力。

## 1.2 代码是怎么运行的

入口脚本是：

```text
scripts/run_case_study.py
```

一次实验的执行流程是：

1. 解析命令行参数，例如数据集、seed、epoch、稀疏率、variant、输出目录。
2. 调用 `load_graph_dataset(...)` 读取图数据，得到：
   - `x`: 节点特征
   - `y`: 标签
   - `edge_index`: 图边
   - `train_mask` / `val_mask` / `test_mask`: 数据划分
3. 对每个 `variant` 和每个 `seed` 依次训练一个模型。
4. 每个模型都是 `OUGPGCN`，只是开关不同：
   - `dense`: 不剪图、不剪参数、不开 memory
   - `graph_only`: 只开 graph pruning
   - `param_only`: 只开 parameter pruning
   - `dual_static`: 同时剪图和参数，但不开 memory
   - `ougp_no_cross`: 开 memory，但不开 cross context
   - `ougp`: 完整版本
5. 每个 epoch 内部：
   - 根据 warmup 和目标稀疏率计算当前 `graph_keep` / `param_keep`
   - 前向传播得到 logits 和 mask 统计
   - 用训练集计算 loss
   - 加上 sparsity regularization
   - 反向传播更新模型参数和 mask logits
   - 写入 online memory
   - 在 train/val/test 上评估
6. 训练结束后写出：
   - 每个 run 的 `*_seed*.json`
   - 汇总表 `*_summary.csv`
   - 人类可读表 `*_summary.md`
   - 完整命令 `command.txt`
   - 实验 manifest `manifest.json`

读代码时，建议按这个顺序看：

```text
scripts/run_case_study.py      # 实验主循环
src/ougp/data.py               # 数据怎么加载
src/ougp/model.py              # GCN、mask、memory 怎么实现
```

## 1.3 剪枝事件记录：两层 memory 设计

现在实现分成两层：

```text
OnlinePruningMemory      # 训练时使用的低秩统计记忆，可接收 event bias 影响 mask score
PruningTraceRecorder     # 实验分析用的事件记录器，负责写出剪枝事件 CSV
```

相关代码：

```text
src/ougp/model.py        # OnlinePruningMemory + OUGPGCN.graph_trace_snapshot()
src/ougp/trace.py        # PruningTraceRecorder
scripts/run_case_study.py
```

默认不记录剪枝事件。如果想记录每次 graph pruning 位置和连接节点的重要程度，运行时加：

```bash
--trace-pruning \
--trace-every 10 \
--trace-top-k 200
```

输出位置：

```text
experiments/<exp_name>/pruning_trace/
```

每条 graph pruning event 会记录：

```text
epoch, edge_id, src_node, dst_node,
prev_mask, current_mask, mask_delta,
graph_score, graph_utility,
src_degree, dst_degree,
src_feature_norm, dst_feature_norm,
src_node_importance, dst_node_importance,
edge_importance,
graph_keep, param_keep
```

注意：`PruningTraceRecorder` 只是记录，不参与训练。这样可以先分析“剪了哪些边、这些边连接的节点是否重要”，之后再决定要不要把 event memory 反馈进模型。

如果想让历史剪枝事件真正参与训练，需要额外打开 event feedback：

```bash
--event-gamma 0.2 \
--event-beta 0.1 \
--event-decay 0.9 \
--event-top-k 2000
```

这时流程变成：

```text
剪枝事件 + 节点/边重要性
  -> 生成 event_delta
  -> 写入 OnlinePruningMemory.event_bias
  -> 下一轮 graph_score += event_gamma * event_bias
  -> 影响 graph_mask
```

如果 `--event-gamma 0.0`，event memory 仍可被更新和记录，但不会改变 `graph_score`，因此不会改变模型行为。

## 2. 当前主实验在哪里

当前主 case study 是 Cora + GCN。优先看最新的 GPU 多 seed 实验：

```text
experiments/exp002_cora_gpu_multiseed/
```

整理后的表格：

```text
results/tables/cora_exp002_gpu_multiseed_summary.csv
results/tables/cora_exp002_gpu_multiseed_summary.md
```

对应中文记录：

```text
notes/EXP002_CORA_GPU_MULTI_SEED.md
```

历史上的第一版单 seed case study 是：

```text
experiments/exp001_cora_case_study_initial/primary_v2/
```

它的整理表格是：

```text
results/tables/cora_case_study_v2_summary.csv
results/tables/cora_case_study_v2_summary.md
```

完整记录包括：

- `command.txt`：当时完整命令。
- `cora_summary.csv`：每个方法一行的最终指标。
- `cora_summary.md`：方便人看的表格。
- `cora_ougp_seed0.json`：OUGP 这一组的完整记录。
- 其他 `cora_*_seed0.json`：其他 baseline / ablation 的完整记录。

## 3. 怎么复现实验

进入项目目录：

```bash
cd /home/shizitong/tianjiaying/research/ougp
```

当前只能使用自己的 `tianjiaying` 环境，不要使用 `atma`。`atma` 是别人/其他项目的环境，不应该污染或依赖。

跑一个最短 smoke test，确认代码和数据没坏：

```bash
PYTHONPATH=src /home/shizitong/miniconda3/envs/tianjiaying/bin/python scripts/run_case_study.py \
  --dataset cora \
  --epochs 1 \
  --warmup-epochs 1 \
  --variants dense ougp \
  --seeds 0 \
  --out-dir experiments/manual_smoke
```

复现当前主实验，建议显式指定一张空闲 GPU，例如 GPU 3：

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src /home/shizitong/miniconda3/envs/tianjiaying/bin/python scripts/run_case_study.py \
  --dataset cora \
  --epochs 120 \
  --warmup-epochs 15 \
  --hidden-dim 64 \
  --memory-rank 16 \
  --variants dense graph_only param_only dual_static ougp_no_cross ougp \
  --seeds 0 1 2 \
  --out-dir experiments/exp002_cora_gpu_multiseed \
  --graph-gamma 2.0 \
  --param-gamma 2.0 \
  --write-beta 0.25 \
  --device cuda \
  --max-gpus 1 \
  --verbose \
  --log-every 20
```

注意：任何 GPU 实验最多只能暴露 4 张卡，例如：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ...
```

当前推荐优先用 1 张 GPU 跑 Cora。Cora 很小，多卡通常不会明显加速，反而会增加并行复杂度。

## 4. 每个 variant 是什么意思

主表里的 `variant` 表示对比方法：

| Variant | 含义 | 你应该用它回答什么问题 |
| --- | --- | --- |
| `dense` | 不剪图、不剪参数的普通 GCN | 剪枝前的基础性能是多少 |
| `graph_only` | 只剪 graph edge/message passing | 只减少图传播成本会怎样 |
| `param_only` | 只剪 hidden channel/参数维度 | 只减少模型参数会怎样 |
| `dual_static` | 同时剪 graph 和 parameter，但不用 online memory | 普通“双剪枝” baseline |
| `ougp_no_cross` | 有 online memory，但 graph/parameter 不做 cross context | online memory 本身有没有用 |
| `ougp` | 完整方法：online memory + graph/parameter cross context | 我们的方法是否比 ablation 更好 |

读论文结果时，不要只看 `ougp` 和 `dense`，更要看：

- `ougp` vs `dual_static`：证明 online memory 是否有贡献。
- `ougp` vs `ougp_no_cross`：证明 cross-level coupling 是否有贡献。
- `graph_only` / `param_only` vs `dual_static`：证明统一剪枝是否有必要。

## 5. summary.csv 里的指标怎么看

当前主表字段包括：

| 字段 | 中文解释 | 怎么判断 |
| --- | --- | --- |
| `dataset` | 数据集，比如 Cora | 当前只跑了 Cora |
| `variant` | 方法名字 | 看是哪一个 baseline / ablation |
| `seed` | 随机种子 | 当前主实验有 seed 0/1/2，作为 case study 可以，论文主结果最好更多 seed |
| `epochs` | 训练轮数 | 当前主实验是 120 |
| `best_epoch` | 验证集最好的 epoch | 说明最好模型出现在第几轮 |
| `best_val_acc` | 验证集最好准确率 | 用它选择模型 |
| `best_test_acc` | 在 best val 对应模型上的测试准确率 | 主性能指标，优先看它 |
| `final_train_acc` | 最后一轮训练集准确率 | 太高但 val/test 不高说明可能过拟合 |
| `final_val_acc` | 最后一轮验证准确率 | 看训练最后是否退化 |
| `final_test_acc` | 最后一轮测试准确率 | 辅助看稳定性 |
| `graph_keep` | 保留多少 graph edge/message | 0.70 表示保留 70% |
| `graph_sparsity` | 图稀疏率 | 0.30 表示剪掉 30% |
| `param_keep` | 保留多少 hidden channel | 0.70 表示保留 70% |
| `param_sparsity` | 参数/通道稀疏率 | 0.30 表示剪掉 30% |
| `graph_churn` | graph mask 的变化幅度 | 越低说明 mask 越稳定 |
| `param_churn` | parameter mask 的变化幅度 | 越低说明 channel mask 越稳定 |
| `graph_memory_state_norm` | graph online memory 的状态范数 | 大于 0 表示 memory 在写入 |
| `param_memory_state_norm` | parameter online memory 的状态范数 | 大于 0 表示 memory 在写入 |
| `runtime_sec` | 运行时间 | 只作成本参考 |
| `num_nodes` / `num_edges` | 图规模 | Cora 是 2708 节点、10556 边 |
| `hidden_dim` | GCN hidden 维度 | 当前主实验是 64 |
| `memory_rank` | OPM memory rank | 当前主实验是 16 |

最重要的三个指标：

1. `best_test_acc`：性能有没有保持住或提升。
2. `graph_sparsity` / `param_sparsity`：到底剪掉了多少。
3. `graph_churn` / `param_churn`：mask 是否稳定。

## 6. 怎么读当前结果

当前最新 GPU 多 seed 实验表格简化如下：

| Variant | Best Test Acc | Graph Sparsity | Param Sparsity |
| --- | ---: | ---: | ---: |
| dense | 0.8083 +/- 0.0049 | 0.000 | 0.000 |
| graph_only | 0.8083 +/- 0.0084 | 0.300 | 0.000 |
| param_only | 0.8147 +/- 0.0073 | 0.000 | 0.300 |
| dual_static | 0.8090 +/- 0.0114 | 0.300 | 0.300 |
| ougp_no_cross | 0.8070 +/- 0.0083 | 0.300 | 0.300 |
| ougp | 0.8060 +/- 0.0088 | 0.300 | 0.300 |

可以说：

- 代码已经跑通真实 Cora 图，不是 toy demo。
- OUGP 能同时做到约 30% graph sparsity 和 30% parameter sparsity。
- 在这个轻量设置下，双剪枝后准确率没有崩。
- 当前 Cora 多 seed 结果已经足够作为 preliminary case study / feasibility study。

不能说：

- 不能说 OUGP 已经明显超过所有 baseline。
- 不能说 online memory 已经被充分证明有效。
- 不能说 cross-level context 已经带来明确提升。
- 不能把 Cora 一个数据集当作论文的完整主实验。

当前最诚实的结论是：

> Cora 已经跑够了“证明流程可运行”的 case study，但还不够支撑论文主 claim。当前结果显示 OUGP 能稳定完成统一图/参数剪枝，但 online memory 和 cross-level context 的优势还不明显，需要换数据集、更高稀疏率和超参/消融继续验证。

### 6.1 Cora 现在算跑够了吗

分两种标准看：

- 如果目标是做一次 case study / feasibility study：够了。现在已经有 Cora、6 个 variant、3 个 seed、GPU 运行记录、完整日志和最终指标。
- 如果目标是写论文主实验：不够。因为目前只有一个数据集，而且 `ougp` 没有明显超过 `dual_static` / `ougp_no_cross`，还不能支撑“online memory 明确有效”这个核心 claim。

所以接下来不建议继续在 Cora 30% sparsity 上反复小修小补。更值得做的是：

- 跑 CiteSeer / PubMed，看现象是否跨数据集存在。
- 跑更高 sparsity，例如 50% / 50%，看压力更大时 OUGP memory 是否有优势。
- 做关键消融和超参扫描，例如 `write_beta`、`graph_gamma`、`param_gamma`、`memory_rank`。

## 7. 看单个 JSON 文件

例如：

```text
experiments/exp001_cora_case_study_initial/primary_v2/cora_ougp_seed0.json
```

里面有三块：

```json
{
  "result": {...},
  "history": [...],
  "config": {...}
}
```

怎么看：

- `result`：最终摘要，和 summary.csv 基本一致。
- `history`：每个 epoch 的训练记录，可以用来画曲线。
- `config`：这个 variant 的模型开关，比如是否使用 graph pruning、parameter pruning、memory、cross。

`history` 里常看的字段：

- `epoch`：第几轮。
- `loss` / `task_loss`：训练 loss。
- `train_acc` / `val_acc` / `test_acc`：每轮准确率。
- `graph_keep` / `param_keep`：每轮保留率。
- `temperature`：soft mask 温度，逐渐下降。
- `graph_memory_state_norm` / `param_memory_state_norm`：memory 是否在更新。

如果要画图，最适合画：

- epoch vs `val_acc`
- epoch vs `test_acc`
- epoch vs `graph_keep`
- epoch vs `param_keep`
- epoch vs `graph_churn`

## 8. 参数怎么看

命令里的关键参数：

| 参数 | 含义 |
| --- | --- |
| `--dataset cora` | 使用 Cora 数据集 |
| `--epochs 120` | 训练 120 轮 |
| `--warmup-epochs 15` | 前 15 轮先不正式剪枝 |
| `--variants ...` | 要跑哪些方法 |
| `--seeds 0 1 2` | 随机种子，当前主实验有 3 个 seed |
| `--graph-sparsity 0.30` | 目标 graph 剪掉 30%，默认就是 0.30 |
| `--param-sparsity 0.30` | 目标 parameter/channel 剪掉 30%，默认就是 0.30 |
| `--memory-rank 16` | Online Pruning Memory 的 rank |
| `--graph-gamma 2.0` | graph memory readout 对 graph score 的影响强度 |
| `--param-gamma 2.0` | parameter memory readout 对 parameter score 的影响强度 |
| `--write-beta 0.25` | memory 写入强度 |
| `--hidden-dim 64` | GCN hidden dim |

学习阶段建议一次只改一个变量。例如：

- 只改 `--seeds 0 1 2`
- 或只改 `--graph-sparsity 0.5 --param-sparsity 0.5`
- 或只改 `--memory-rank 4/8/16`

不要一次改很多参数，否则不知道结果变化来自哪里。

## 9. 一个实验是否“有用”的判断顺序

每次新实验跑完，按这个顺序看：

1. 有没有跑完？
   - 看有没有 `summary.csv`。
   - 看有没有每个 variant 的 `*_seed*.json`。
2. 目标稀疏率达到了吗？
   - 看 `graph_sparsity` 和 `param_sparsity`。
3. 准确率有没有崩？
   - 看 `best_test_acc`。
4. 我们的方法有没有比关键 ablation 好？
   - `ougp` vs `dual_static`
   - `ougp` vs `ougp_no_cross`
5. memory 是否真的在用？
   - 看 `graph_memory_state_norm`、`param_memory_state_norm` 是否大于 0。
6. 是否稳定？
   - 看 `graph_churn`、`param_churn`。
7. 结果能不能写论文？
   - 至少需要多 seed。
   - 最好有多个数据集。
   - 必须有关键消融。

## 10. 下一步最推荐跑什么

最推荐的下一步不是马上写论文，而是补证据：

### 10.1 多 seed

这一步已经完成，最新记录见：

```text
experiments/exp002_cora_gpu_multiseed/
notes/EXP002_CORA_GPU_MULTI_SEED.md
```

复现命令：

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src /home/shizitong/miniconda3/envs/tianjiaying/bin/python scripts/run_case_study.py \
  --dataset cora \
  --epochs 120 \
  --warmup-epochs 15 \
  --hidden-dim 64 \
  --memory-rank 16 \
  --variants dense graph_only param_only dual_static ougp_no_cross ougp \
  --seeds 0 1 2 \
  --out-dir experiments/exp002_cora_gpu_multiseed \
  --graph-gamma 2.0 \
  --param-gamma 2.0 \
  --write-beta 0.25 \
  --device cuda \
  --max-gpus 1
```

### 10.2 更高稀疏率

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src /home/shizitong/miniconda3/envs/tianjiaying/bin/python scripts/run_case_study.py \
  --dataset cora \
  --epochs 120 \
  --warmup-epochs 15 \
  --variants dense dual_static ougp_no_cross ougp \
  --seeds 0 1 2 \
  --graph-sparsity 0.5 \
  --param-sparsity 0.5 \
  --out-dir experiments/exp005_cora_s50_multiseed \
  --graph-gamma 2.0 \
  --param-gamma 2.0 \
  --write-beta 0.25 \
  --device cuda \
  --max-gpus 1
```

### 10.3 PubMed / CiteSeer

这一步已经完成，记录见：

```text
notes/EXP003_EXP004_CITESEER_PUBMED_CROSS_DATASET.md
experiments/exp003_citeseer_cpu_multiseed/
experiments/exp004_pubmed_cpu_multiseed/
```

当时没有空闲 GPU，所以使用 `tianjiaying` 环境的 CPU 跑完。以后 GPU 空出来，可以用相同参数把 `--device cpu` 改成 `--device cuda`，并加上 `CUDA_VISIBLE_DEVICES=<空闲GPU>` 复跑。

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src /home/shizitong/miniconda3/envs/tianjiaying/bin/python scripts/run_case_study.py \
  --dataset citeseer \
  --epochs 120 \
  --warmup-epochs 15 \
  --variants dense dual_static ougp_no_cross ougp \
  --seeds 0 1 2 \
  --out-dir experiments/exp006_citeseer_case_study \
  --graph-gamma 2.0 \
  --param-gamma 2.0 \
  --write-beta 0.25 \
  --device cuda \
  --max-gpus 1
```

如果下载新数据遇到网络问题，需要允许联网下载 Planetoid raw 文件。

### 10.4 Amazon Photo

Amazon Photo 已经完成 GPU case study，记录见：

```text
notes/EXP006_AMAZON_PHOTO_GPU.md
experiments/exp006_photo_gpu_multiseed/
results/tables/photo_exp006_gpu_multiseed_summary.csv
configs/photo_case_study_v1.json
```

这次结果说明：`graph_only` 在 Amazon Photo 上最好，完整 `ougp` 没有超过 dense / graph-only，但比 `dual_static` 好，说明 OUGP 可能缓解了双剪枝伤害，却还没有真正胜出。

之后又完成了 parameter sparsity sweep，记录见：

```text
notes/EXP010_AMAZON_PHOTO_PARAM_SPARSITY_SWEEP.md
experiments/exp010_photo_param_sparsity_sweep/
results/tables/photo_exp010_param_sparsity_sweep_summary.csv
configs/photo_param_sparsity_sweep_v1.json
```

这次 sweep 的核心结论是：

- 30% graph sparsity 本身是明确的效率/存储优势。
- `graph_only` 在 Amazon Photo 上准确率不降反升，说明剪边可能有去噪作用。
- parameter sparsity 从 5% 增加到 30% 时，`param_only` 准确率从约 `0.806` 降到约 `0.713`，说明参数/通道剪枝会明显削弱模型表达能力。
- `ougp` 比 `dual_static` 好，说明 online memory 能缓解双剪枝损失；但还没有超过 `graph_only`，所以不能说完整 OUGP 在 Amazon Photo 上准确率最好。

### 10.5 OGB 大图

OGB 大图已经做了 feasibility attempt，记录见：

```text
notes/EXP009_OGB_LARGE_GRAPH_ATTEMPT.md
experiments/exp009_ogbn_arxiv_gpu_param_only/
results/tables/ogbn_arxiv_exp009_gpu_param_only_summary.csv
configs/ogbn_arxiv_param_only_v1.json
```

当前结论：

- `ogbn-arxiv` 的 dense / param-only 可以在 GPU 上跑完。
- `ogbn-arxiv` 的 graph pruning 在 backward 时 OOM，需要约 106.83 GiB。
- 所以当前 full-batch OUGP 不能直接跑 `ogbn-products` / `ogbn-proteins` 的完整 graph-pruning 实验。
- 正确下一步是实现 sampled / mini-batch OUGP，而不是硬跑全图。

## 11. 写论文时怎么组织这部分

可以先写成这样：

- **Case Study Setup**
  - Dataset: Cora
  - Backbone: 2-layer GCN
  - Compared variants: dense, graph-only, param-only, dual-static, OUGP no-cross, OUGP
  - Metrics: accuracy, graph sparsity, parameter sparsity, mask churn

- **Preliminary Finding**
  - OUGP runs end-to-end and achieves matched graph/parameter sparsity.
  - Accuracy remains close to dense.
  - Current online memory advantage is inconclusive on Cora multi-seed results.

- **Next Validation**
  - Multi-seed.
  - PubMed/CiteSeer.
  - EMA vs gated delta-rule.
  - memory rank ablation.

这样写比较诚实，也更像科研过程。
