# Idea: OUGP - Online Unified Graph and Parameter Pruning for GNNs

## 1. 核心问题

现有 graph 背景下的剪枝方法通常把剪枝看成一个静态选择问题：先根据某个准则估计边、节点、通道或参数的重要性，再生成稀疏子图或稀疏模型。这个范式有两个明显限制。

第一，图结构和 GNN 参数的重要性会随训练过程变化。早期看起来冗余的边或参数，可能在表示逐渐成形后变得重要；早期看起来重要的结构，也可能只是局部噪声或过拟合信号。

第二，graph-level pruning 和 parameter-level pruning 往往分开设计。只剪图会降低 message passing 成本，但模型仍可能冗余；只剪参数会降低模型计算与存储，但图传播开销仍然存在。EAGLES 的启发是：图结构和模型参数应当被统一考虑；delta-mem 的启发是：重要性判断不应只来自当前瞬时观测，而应由一个随训练动态更新的 online state 来持续修正。

本 idea 的核心目标是：

> 为集中式 GNN 设计一个在线统一剪枝框架，用固定容量的 Online Pruning Memory 持续吸收训练过程中的 graph/parameter pruning utility residual，并通过 read-steer-write 机制动态生成 graph mask 和 parameter mask。

暂定命名：**OUGP: Online Unified Graph and Parameter Pruning**。

## 2. Online 的准确定义

这里的 `online` 不是在线微调 backbone，也不是必须处理动态图新节点到来的狭义 online learning，而是：

> 在训练过程中，每个 pruning interval 都持续更新一个固定容量的 **Online Pruning Memory (OPM)**。OPM 记忆的是 graph units 和 parameter units 的历史 pruning utility residual，并在下一次剪枝时修正 mask 决策。

因此 OPM 不是存历史 mask，也不是存完整图结构，而是一个低维 utility estimator。它要解决的是静态重要性估计的漂移问题：当前梯度、attention 或 message norm 只反映瞬时信号，OPM 则吸收过去多轮剪枝后的误差反馈，使剪枝轨迹更稳定。

## 3. 第一性原理

graph pruning 的本质不是“删边”，而是在保留任务相关信息流的同时减少 message passing 负载。parameter pruning 的本质不是“删权重”，而是在保留表达能力的同时减少参数冗余。两者共享同一个底层问题：

> 在训练过程中，如何估计某个可剪单元对未来模型性能和计算成本的边际贡献？

这个边际贡献不是静态属性，而是与当前表示、训练阶段、邻域结构和参数状态共同相关。delta-mem 的 read-steer-write 可以抽象为：先读历史状态，修正当前决策，再把当前观测中旧状态无法预测的 residual 写回状态。迁移到剪枝中，就是维护一个 pruning utility memory。

## 4. 方法总览

OUGP 包含四个模块：

1. **Graph Online Pruning Memory**：记忆不同局部结构模式下的边、节点或子图 pruning utility residual。
2. **Parameter Online Pruning Memory**：记忆不同层、通道、head、hidden dimension 或参数块的 pruning utility residual。
3. **Unified Read-Steer-Write Mask Generator**：读取 OPM 修正 graph mask 和 parameter mask。
4. **Cross-level Coupling Objective**：显式建模 graph pruning 与 parameter pruning 的相互影响，并在任务性能、图稀疏率、参数稀疏率、稳定性和真实计算成本之间联合优化。

## 5. Online Pruning Memory 设计

给定图 `G=(V,E,X)` 和 GNN `f_theta`，维护两类状态：

```text
S_G in R^{r x r}              # graph pruning memory
S_W^{(l)} in R^{r x r}        # 第 l 层 parameter pruning memory
```

`r` 是很小的 memory rank，例如 8、16 或 32。关键点是不要为每条边或每个参数维护大状态，否则 online 机制会失去轻量意义。状态只负责记忆低维关联：

```text
S_G: context of edge/subgraph pattern -> predicted graph pruning utility
S_W: context of parameter block/channel -> predicted parameter pruning utility
```

`S q` 输出的不是最终 mask，而是对当前基础剪枝分数的历史残差修正。最终 mask 仍由当前准则、OPM readout 和稀疏预算共同决定。

### 5.1 Graph 单元表示

对边 `e_ij` 或一跳子图构造剪枝上下文：

```text
c^G_ij = MLP_G([h_i, h_j, x_i, x_j, deg_i, deg_j, edge_attr_ij, a_ij, p_ctx])
```

其中 `h_i,h_j` 是当前 GNN 表示，`a_ij` 可以是 attention、message norm、梯度敏感度或结构准则得分。`p_ctx` 是当前 parameter sparsity context，例如当前层稀疏率、通道 mask 均值或 feature transform capacity。加入 `p_ctx` 是为了让 graph pruning 感知参数剪枝已经改变了表示能力。

再投影到 memory 空间：

```text
q^G_ij = norm(tanh(W^G_q c^G_ij))
k^G_ij = norm(tanh(W^G_k c^G_ij))
v^G_ij = W^G_v c^G_ij
```

### 5.2 Parameter 单元表示

参数剪枝不建议从单个 weight 开始，否则粒度太细且难以真实加速。更合理的粒度是 channel、hidden dimension、attention head、MLP neuron 或 GNN layer block。

对第 `l` 层第 `u` 个参数组构造上下文：

```text
c^W_{l,u} = MLP_W([norm(W_{l,u}), norm(grad_{l,u}), activation_{l,u}, layer_id, sparsity_l, g_ctx])
```

其中 `g_ctx` 是当前图稀疏上下文，例如图 mask 均值、message passing FLOPs、节点表示漂移或 subgraph sparsity。加入 `g_ctx` 是为了让 parameter pruning 感知 graph pruning 已经改变了输入消息分布。

再得到：

```text
q^W_{l,u}, k^W_{l,u}, v^W_{l,u}
```

## 6. Read-Steer-Write 剪枝机制

### 6.1 Read：读取历史剪枝信号

对图单元：

```text
r^G_ij = S_G q^G_ij
```

对参数单元：

```text
r^W_{l,u} = S_W^{(l)} q^W_{l,u}
```

`r` 是历史 state 对当前单元的 utility 修正，不是最终重要性。

### 6.2 Steer：修正当前 mask 生成

先用普通准则得到基础分数：

```text
b^G_ij = phi_G(c^G_ij)        # message norm / gradient / spectral / attention score
b^W_{l,u} = phi_W(c^W_{l,u})  # magnitude / gradient / activation-aware score
```

再用 online readout 修正：

```text
s^G_ij = b^G_ij + gamma_G * w_G^T r^G_ij
s^W_{l,u} = b^W_{l,u} + gamma_W * w_W^T r^W_{l,u}
```

mask 使用 hard concrete、Gumbel-sigmoid 或 STE：

```text
m^G_ij = HardConcrete(s^G_ij, tau_G)
m^W_{l,u} = HardConcrete(s^W_{l,u}, tau_W)
```

训练时使用 soft mask 保持可恢复性；评估或部署时再硬化为离散稀疏结构。

### 6.3 Write：用 utility prediction error 更新 OPM

关键是定义“当前 state 的预测错了多少”。剪枝 utility 可以用损失变化与资源收益的比值表示：

```text
u^G_ij = - Delta L_ij / (Delta MessageFLOPs_ij + eps)
u^W_{l,u} = - Delta L_{l,u} / (Delta ParamFLOPs_{l,u} + eps)
```

实际实现不能对每条边或每个通道做昂贵的 leave-one-out。第一版建议使用一阶近似：

```text
u^G_ij ≈ |dL / dm^G_ij * m^G_ij| / (Delta MessageFLOPs_ij + eps)
u^W_{l,u} ≈ |dL / dm^W_{l,u} * m^W_{l,u}| / (Delta ParamFLOPs_{l,u} + eps)
```

OPM 对 utility 的预测为：

```text
hat_u^G_ij = w_u^T (S_G k^G_ij)
hat_u^W_{l,u} = w_u^T (S_W^{(l)} k^W_{l,u})
```

真正写入的是 utility prediction error：

```text
e^G_ij = u^G_ij - stopgrad(hat_u^G_ij)
e^W_{l,u} = u^W_{l,u} - stopgrad(hat_u^W_{l,u})
```

状态更新借鉴 delta-mem 的 gated delta rule：

```text
S_G <- Diag(lambda_G) S_G
       + Diag(beta_G) (v^G_ij(e^G_ij) - S_G k^G_ij) (k^G_ij)^T

S_W^{(l)} <- Diag(lambda_W) S_W^{(l)}
       + Diag(beta_W) (v^W_{l,u}(e^W_{l,u}) - S_W^{(l)} k^W_{l,u}) (k^W_{l,u})^T
```

`v(e)` 表示由 residual utility 调制后的目标写入值，例如：

```text
v_tilde = v * stopgrad(normalize(e))
```

这样 OPM 写入的是“重要性反馈残差”，而不是普通特征累计。`lambda` 和 `beta` 必须由 sigmoid 或 clipped sigmoid 约束到 `[0,1]`，避免长期递推造成状态爆炸。

## 7. Graph-Parameter 交互耦合

统一剪枝不能只是并列放两个模块。graph pruning 会改变邻域消息分布，从而改变参数通道的重要性；parameter pruning 会改变特征变换能力，从而改变边和邻居的贡献。因此 OUGP 使用 cross-read 机制：

```text
r^G_ij = S_G q^G_ij + A_GW pool_l(S_W^{(l)} q^W_l)
r^W_{l,u} = S_W^{(l)} q^W_{l,u} + A_WG pool_e(S_G q^G_e)
```

更 KISS 的实现可以写成：

```text
s^G_ij = b^G_ij + gamma_G * w_G^T r^G_ij + xi_G * current_parameter_sparsity
s^W_{l,u} = b^W_{l,u} + gamma_W * w_W^T r^W_{l,u} + xi_W * current_graph_sparsity
```

这部分是相对普通双剪枝方法的关键区别：graph mask 与 parameter mask 不是两条独立分支，而是通过 online utility residual 和当前稀疏上下文互相修正。

## 8. 写入粒度设计

delta-mem 提醒我们，online state 的写入粒度很重要。集中式 graph pruning 可以设计三种粒度：

1. **Edge-State Write**：每个 sampled edge 都写入，最细粒度，但噪声大。
2. **Subgraph-State Write**：对 mini-batch subgraph 或 ego-graph 聚合后写入，更稳定。
3. **Multi-State Write**：维护多个并行状态，例如 topology state、feature state、gradient state、parameter state，分别记忆不同类型的剪枝证据。

推荐主方法使用 Subgraph-State Write + Multi-State Write：

```text
S_G = {S_topo, S_feat, S_grad}
S_W = {S_layer, S_channel}
```

这样能避免单一 state 同时记忆结构同质性、特征相似性和梯度敏感性时发生互相覆盖。

## 9. 训练目标

总体目标：

```text
L = L_task
  + alpha_G * R_sparsity(m_G, rho_G)
  + alpha_W * R_sparsity(m_W, rho_W)
  + eta * R_stability(m_G, m_W)
  + zeta * R_budget(FLOPs, Params)
```

其中：

```text
R_sparsity = |mean(m) - target_sparsity|
R_stability = ||m_t - m_{t-1}||_1
R_budget = normalized_message_passing_cost + normalized_parameter_cost
```

稳定性项不是为了保守，而是避免 online mask 在相邻 epoch 间剧烈震荡。真正的可恢复性来自 soft mask、delayed hardening 和 residual state 更新。

## 10. 训练流程

```text
Input: G, labels, GNN f_theta, target graph sparsity rho_G, target parameter sparsity rho_W
Initialize theta, S_G, {S_W^{(l)}}

Warm-up: train dense or lightly sparse GNN for several epochs

for epoch = 1 ... T:
    sample mini-batch subgraph B

    # Read
    build graph contexts c^G and parameter contexts c^W
    read r^G from S_G, read r^W from S_W

    # Cross-level steer
    generate graph mask m_G and parameter mask m_W
    run masked GNN forward/backward

    # Feedback
    estimate pruning utility u_G, u_W by first-order approximation
    compute utility prediction error e_G, e_W

    # Write
    update S_G and S_W by gated delta rule

    # Anneal
    gradually harden mask temperature and increase sparsity budget

Output: sparse graph G_s, sparse GNN theta_s
```

## 11. 预期创新点

1. **从静态剪枝到 Online Pruning Memory**：不是每轮重新估计重要性，而是维护一个固定容量的 OPM 持续记忆历史 pruning utility residual。
2. **统一 graph-level 与 parameter-level pruning**：同时减少 message passing 成本和模型参数冗余，并通过 cross-read 显式建模二者耦合。
3. **残差写入而非特征累积**：state 更新写入的是当前剪枝反馈中未被旧 state 预测到的部分。
4. **写入粒度可控**：从 edge/token 类比到 subgraph/segment，从单状态到 multi-state，形成 graph 场景特有的 online pruning 设计空间。
5. **可恢复剪枝**：训练阶段使用 soft gate，避免一次性硬剪造成不可逆错误。

## 12. 实验设计

### 12.1 数据集

小中型图：

- Cora
- Citeseer
- Pubmed
- Amazon Photo

大图：

- ogbn-arxiv
- ogbn-products
- ogbn-proteins

### 12.2 Backbone

- GCN
- GraphSAGE
- GAT
- DeeperGCN

### 12.3 Baseline

graph sparsification:

- random edge pruning
- degree-based pruning
- similarity-based pruning
- DropEdge
- learnable adjacency pruning / NeuralSparse 类方法
- DSpar 类方法

parameter pruning:

- magnitude pruning
- gradient pruning
- SNIP / GraSP 类训练前或早期剪枝
- dynamic sparse training / RigL 类方法
- lottery-ticket style pruning

unified pruning:

- 串行 graph pruning + parameter pruning
- ACE-GLT / EAGLES 中可迁移到集中式的 dual sparsification 思路

### 12.4 指标

- Accuracy / ROC-AUC
- graph sparsity
- parameter sparsity
- training FLOPs
- inference FLOPs
- wall-clock latency
- mask churn rate
- state overhead
- pruning recovery cost

其中 `mask churn rate = ||m_t - m_{t-1}||_0 / |m|`，用于证明 online state 是否带来更稳定的剪枝轨迹。

## 13. 关键消融

1. 无 online state，只用当前准则。
2. 只做 graph pruning。
3. 只做 parameter pruning。
4. graph 和 parameter 分开训练，不共享统一目标。
5. state 只读不写。
6. state 只写特征，不写 residual utility。
7. Edge-State Write vs Subgraph-State Write vs Multi-State Write。
8. 不同 state rank：`r=4,8,16,32`。
9. 不同 hardening schedule。
10. 不做 graph-parameter cross-read。
11. gated delta-rule vs EMA utility estimator vs RNN/MLP dynamic scorer。

## 14. 可能风险与修正

### 风险 1：online state 是否只是复杂化的动态打分器

修正：必须证明 residual write 有贡献。实验中比较：

- 当前打分器
- EMA 打分器
- RNN/MLP 动态打分器
- gated delta-rule state

如果 gated delta-rule 不能明显优于 EMA，则创新不足。

### 风险 2：剪枝反馈 `u` 估计代价过高

修正：不要对每条边做真实 leave-one-out。使用一阶梯度敏感度、message norm 变化、mini-batch loss residual 或 sampled perturbation。

### 风险 3：graph mask 和 parameter mask 相互干扰

修正：加入交替 warm-up：

1. 前若干 epoch 只学习 state，不硬剪。
2. 中期先轻度 graph pruning。
3. 后期逐步增加 parameter sparsity。
4. 最后联合微调。

### 风险 4：真实加速不足

修正：graph 剪枝优先删边/邻居，parameter 剪枝优先结构化粒度，如 channel、head、hidden dimension，而不是任意非结构化 weight。

## 15. 一句话版本

OUGP 不是把 LLM memory 直接搬到 GNN，而是借鉴 delta-mem 的在线残差记忆机制，提出 graph 背景下的 Online Pruning Memory：用固定容量 state 记忆历史 pruning utility residual，在每个训练阶段读取 state 来修正图结构和模型参数的剪枝决策，再用 utility prediction error 写回 state，从而实现 graph-level 和 parameter-level 的统一、动态、可恢复剪枝。
