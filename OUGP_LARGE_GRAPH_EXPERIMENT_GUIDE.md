# OUGP 大图实验指南（EXP074–EXP076）

> 目标：在另一台服务器上复现小中图 EXP071–EXP073 的三类实验，但改为 OGB 大图，并同时记录理论节省与真实硬件开销。  
> 本指南供 Codex 直接读取和执行。**先检查现有代码和参数，不要凭空新增不兼容的命令行参数。**

---

## 1. 实验目标

在以下大图上测试：

- `ogbn-arxiv`
- `ogbn-products`
- `ogbn-proteins`

对比方法：

- `dense`
- `ougp_without_lhcm`
- `ougp_lhcm`

主要回答三个问题：

1. OUGP 在大图上能否保持性能并达到目标 graph/parameter sparsity？
2. 理论计算量下降是否能转化为真实的时间、吞吐率和显存收益？
3. LHCM 在不同 sparsity 和 coupling strength 下是否稳定、是否值得其额外开销？

---

## 2. 先阅读和复用现有实现

开始修改前，先检查：

```text
scripts/run_exp071_small_medium_hardware_theory_gcn4.sh
scripts/run_exp072_small_medium_sparsity_sweep_gcn4.sh
scripts/run_exp073_small_medium_lhcm_mix_sweep_gcn4.sh
scripts/summarize_small_medium_experiment.py
scripts/run_exp071_072_073_queue_wait_gpu.sh
```

同时检查项目中已有的大图入口和采样实验，例如：

```text
scripts/run_exp066_ogbn_sampled_lhcu*
scripts/run_case_study.py
src/ougp/model.py
```

要求：

- 尽量复用现有 sampled-subgraph / neighbor-sampling pipeline；
- 不要把 `ogbn-products`、`ogbn-proteins` 强行改成 full-batch；
- 不要改变 OUGP、LHCM 的核心定义；
- 保留原有输出格式，便于和小中图结果统一汇总；
- 先运行 `python scripts/run_case_study.py --help`，确认真实参数名。

---

## 3. 大图训练原则

### 3.1 数据划分

使用 OGB 官方 split：

- `ogbn-arxiv`：Accuracy
- `ogbn-products`：Accuracy
- `ogbn-proteins`：ROC-AUC

不要重新随机划分数据。

### 3.2 训练方式

优先使用当前项目已经验证过的大图采样方式：

- neighbor sampling；
- random subgraph；
- frontier subgraph；
- 或现有 `sampled-edge` 路径。

同一数据集的所有 variant 必须保持完全一致的：

- backbone；
- 层数；
- hidden dimension；
- batch/subgraph size；
- fanout；
- sampler；
- sampler seed；
- epoch 数；
- optimizer；
- learning rate；
- graph/parameter sparsity；
- GPU 型号；
- dataloader workers。

### 3.3 Backbone

默认优先：

```text
ogbn-arxiv    → 当前已验证的大图 backbone
ogbn-products → GraphSAGE / 当前最稳定的 sampled backbone
ogbn-proteins → GraphSAGE、GAT 或 DeeperGCN 中已验证最稳定者
```

不要为了统一名称而强行使用不适合大图的 full-batch GCN。  
若当前代码支持 4-layer sampled GCN，可额外保留 GCN4；否则使用统一的 4-layer GraphSAGE，并在配置文件中明确记录。

---

## 4. EXP074：大图硬件与理论节省主表

建议脚本：

```text
scripts/run_exp074_large_hardware_theory.sh
```

输出目录：

```text
experiments/exp074_large_hardware_theory/
```

数据集：

```text
ogbn-arxiv
ogbn-products
ogbn-proteins
```

比较：

```text
dense
ougp_without_lhcm
ougp_lhcm
```

建议正式配置：

```text
seeds: 0,1,2
graph sparsity: 0.30
parameter sparsity: 0.30
```

若大图成本过高，可先做：

```text
ogbn-arxiv：3 seeds
ogbn-products：3 seeds
ogbn-proteins：1 seed smoke，确认后补 3 seeds
```

### 必须记录的指标

#### 模型效果

- validation metric
- test metric
- best epoch
- train loss
- convergence epoch

#### 剪枝结果

- target graph sparsity
- actual graph sparsity
- target parameter sparsity
- actual parameter sparsity
- graph mask churn
- parameter mask churn
- graph keep ratio
- parameter keep ratio

#### 理论资源节省

- dense message cost
- effective message cost
- message cost reduction
- dense parameter count
- effective parameter count
- parameter cost reduction
- estimated FLOPs reduction
- memory state overhead
- memory overhead vs dense parameters

#### 真实硬件指标

- epoch wall-clock time
- total training time
- train step time
- validation time
- inference latency
- throughput：nodes/s、edges/s 或 batches/s
- peak GPU allocated memory
- peak GPU reserved memory
- CPU RSS（可选）
- dataloader/sampling time
- host-to-device transfer time
- mask generation time
- memory read time
- memory write time
- LHCM/crossing time
- total policy overhead
- policy overhead ratio

### 计时要求

GPU 计时必须使用：

```python
torch.cuda.synchronize()
start = time.perf_counter()
# measured operation
torch.cuda.synchronize()
elapsed = time.perf_counter() - start
```

显存测量前：

```python
torch.cuda.reset_peak_memory_stats()
```

测量后记录：

```python
torch.cuda.max_memory_allocated()
torch.cuda.max_memory_reserved()
```

推理 latency：

- 先 warm up 10–20 batches；
- 正式测量至少 50 batches；
- 报告 mean、median、p95；
- dense 与 OUGP 必须使用相同 batch/subgraph；
- 不把首次数据加载、编译和缓存时间混入正式 latency。

注意：

> estimated reduction 与真实 speedup 必须分开报告。  
> mask 达到 30% sparsity，不代表 PyTorch 稠密 kernel 会自动获得 30% 加速。

---

## 5. EXP075：大图 Sparsity Sweep

建议脚本：

```text
scripts/run_exp075_large_sparsity_sweep.sh
```

输出目录：

```text
experiments/exp075_large_sparsity_sweep/
```

比较：

```text
ougp_without_lhcm
ougp_lhcm
```

Sparsity：

```text
0.10
0.30
0.50
0.70
```

graph 与 parameter 先使用相同 sparsity。  
如 70% 在大图上明显崩溃，仍然保留结果，不要自动删掉。

建议：

```text
每个数据集至少 3 seeds
资源不足时先 1 seed 全 sweep，再对关键点补 3 seeds
```

重点分析：

- performance–sparsity trade-off；
- actual sparsity 是否达到 target；
- theoretical reduction；
- real speedup；
- GPU memory；
- mask churn；
- LHCM 是否在高 sparsity 下改善性能或稳定性；
- 70% sparsity 是否出现训练失败、梯度异常或显存异常。

输出主表：

| dataset | method | sparsity | val/test | actual graph sparsity | actual param sparsity | estimated reduction | epoch time | latency | throughput | peak GPU memory | policy overhead |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

---

## 6. EXP076：大图 LHCM Strength Sweep

建议脚本：

```text
scripts/run_exp076_large_lhcm_mix_sweep.sh
```

输出目录：

```text
experiments/exp076_large_lhcm_mix_sweep/
```

`hidden_coupling_mix`：

```text
0.00
0.10
0.30
0.50
1.00
```

默认 sparsity：

```text
graph sparsity: 0.30
parameter sparsity: 0.30
```

第一轮建议：

```text
ogbn-arxiv
ogbn-products
ogbn-proteins
```

若成本太高：

```text
先在 ogbn-arxiv + ogbn-products 完成全部 sweep；
ogbn-proteins 先跑 0.00 / 0.30 / 1.00。
```

重点记录：

- val/test metric；
- convergence；
- graph/parameter sparsity；
- graph/parameter churn；
- LHCM correction norm；
- coupling gate/mix；
- policy overhead；
- epoch time；
- peak memory；
- inference latency；
- throughput。

目标是判断：

```text
LHCM strength 太低 → 与 no-LHCM 几乎相同
LHCM strength 适中 → 性能/稳定性改善
LHCM strength 太高 → correction 过强、churn 增大或性能下降
```

---

## 7. 公平性与可复现性

每次运行保存完整配置：

```text
dataset
split
backbone
num_layers
hidden_dim
sampler
fanout
batch_size / subgraph_size
num_workers
epochs
warmup_epochs
seed
graph_sparsity
parameter_sparsity
hidden_coupling_mix
temperature schedule
optimizer
learning_rate
GPU name
CUDA version
PyTorch version
git commit
```

同一组比较必须：

- 使用相同 GPU；
- 使用相同采样 batch；
- 固定 sampler seed；
- 不同时运行其他高负载任务；
- 不在一个 variant 使用缓存而另一个不使用；
- 不把排队等待时间计入训练总时间；
- 失败任务记录原因，不静默跳过。

---

## 8. Smoke Test

正式运行前，先做：

```text
dataset: ogbn-arxiv
epochs: 2–3
seed: 0
variants: dense / ougp_without_lhcm / ougp_lhcm
```

检查：

- 数据集能加载；
- sampled forward/backward 正常；
- 三种 variant 使用同一 sampler 配置；
- actual sparsity 正确；
- timing 不为 0；
- GPU memory 正确记录；
- policy overhead 能拆分；
- 输出无 NaN/Inf；
- summary 脚本能读取结果。

Smoke test 通过后，再启动正式实验。

---

## 9. 脚本与队列

建议创建：

```text
scripts/run_exp074_large_hardware_theory.sh
scripts/run_exp075_large_sparsity_sweep.sh
scripts/run_exp076_large_lhcm_mix_sweep.sh
scripts/summarize_large_experiment.py
scripts/run_exp074_075_076_queue_wait_gpu.sh
```

队列逻辑：

```text
等待满足显存阈值的 GPU
        ↓
运行 EXP074
        ↓
确认 EXP074 退出码
        ↓
运行 EXP075
        ↓
确认 EXP075 退出码
        ↓
运行 EXP076
        ↓
统一汇总
```

注意修复空 PID 问题：

```bash
if [[ -n "${pid:-}" ]]; then
    wait "$pid"
fi
```

并使用数组保存有效 PID，不要把空字符串传给 `wait`。

建议 screen：

```bash
screen -S ougp_exp074_075_076
bash scripts/run_exp074_075_076_queue_wait_gpu.sh
```

挂起：

```text
Ctrl+A，然后按 D
```

查看：

```bash
screen -r ougp_exp074_075_076
```

---

## 10. 输出目录建议

```text
experiments/
├── exp074_large_hardware_theory/
│   ├── config/
│   ├── raw/
│   ├── logs/
│   ├── tables/
│   └── summary.md
├── exp075_large_sparsity_sweep/
├── exp076_large_lhcm_mix_sweep/
└── exp074_075_076_queue.log
```

每个实验至少输出：

```text
results.csv
hardware_metrics.csv
per_epoch_metrics.csv
summary.csv
summary.md
launcher.log
```

---

## 11. 最终汇总表

### 主表 A：效果与稀疏率

| dataset | method | metric | actual graph sparsity | actual param sparsity | best epoch |
|---|---|---:|---:|---:|---:|

### 主表 B：理论与真实节省

| dataset | method | estimated message reduction | estimated param reduction | epoch speedup | inference speedup | throughput gain | peak memory reduction | policy overhead |
|---|---|---:|---:|---:|---:|---:|---:|---:|

### 主表 C：LHCM Sweep

| dataset | hidden coupling mix | metric | churn | correction norm | epoch time | peak memory |
|---|---:|---:|---:|---:|---:|---:|

---

## 12. 验收要求

完成后先报告：

1. 修改/新增了哪些文件；
2. 每个大图实际使用什么 backbone 和 sampler；
3. 完整运行命令；
4. 每个实验预计耗时；
5. 每个实验预计显存；
6. 理论 reduction 与真实 speedup 是否分开；
7. policy overhead 是否单独计时；
8. smoke test 是否通过；
9. 是否存在 OOM、NaN、空 PID、日志中断；
10. 汇总表和日志位置。

不要在 smoke test 未通过时直接启动全部正式实验。
