# Sparse Transformer 方法说明（端到端）

> 历史入口说明：本文件记录旧流程，不能直接作为修订后论文的方法。
> 旧流程存在全记录标准化、重复历史 Q 及阶段开关不严格的问题。
> 新增实验请使用 `run_revision.py`，以 `docs/REVISION_DEPLOYMENT.md`
> 的新协议、实际运行配置和保存的预处理参数为准。旧消融与探测结果
> 不能因为图形重绘就当成已经按新协议计算。

本文档说明本项目从模型训练到可解释性分析的完整计算流程，并明确哪些指标是"真实消融值"，哪些是"启发式"。

## 1. 数据与任务定义

代码位置：`src/flood_transformer/data.py`

- 从 `Forcing/<station>.csv` 与 `Streamflow/<station>.csv` 构建样本。
- 目标列统一命名为 `target_q`。
- 特征标准化：`x_norm = (x - mean) / std`。
- 目标标准化：`y_norm = (y - mean) / std`。
- 构建滑动窗口：输入 `[seq_len, F]`，预测未来 `pred_horizon`。
- 数据集切分：train/val/test（默认 0.7/0.15/0.15）。

## 2. 模型结构

代码位置：`src/flood_transformer/model.py`

模型：`ExplainableSparseTransformer`

- 输入门控：`input_mask_logits`（可学习输入节点开关）
- 多层 Transformer Block：每层包含
- 稀疏注意力：`SparseMultiHeadSelfAttention`（含 `head_mask_logits`）
- 稀疏 MLP：`SparseMLP`（含 `neuron_mask_logits`）
- 输出：最后时间步回归头

门控机制：

- 前向使用 `STE` 二值掩码（`ste_binary_mask`）
- 训练可学习 logits，前向 hard、反向保留梯度

## 3. 训练流程（3 阶段）

代码位置：`src/flood_transformer/train.py`

阶段如下：

- dense：正常训练（不施加权重稀疏）
- sparse：施加权重 Top-K 稀疏（随步数退火）
- mask：在 sparse 基础上加 mask L1 正则

关键计算：

- 任务损失：回归为 MSE
- 权重稀疏率退火：`update_sparsity(...)`
- 权重 Top-K：`apply_global_weight_sparsity(...)`
- mask 正则：
- head/neuron 掩码 L1：`lambda_mask_l1`
- input 掩码 L1：`lambda_input_mask_l1`

## 4. 最小电路提取

代码位置：

- `src/flood_transformer/model.py::prune_circuit`
- `src/flood_transformer/explain.py::prune_circuit`

方法：

- 对 input/head/neuron 的 sigmoid 概率按阈值二值化
- 输出 `circuit_structure.json`
- `active_inputs`
- `active_heads`
- `active_neurons`

这一步给出"结构保留"，不直接代表因果强弱。

## 5. 节点重要性（真实消融）

代码位置：`src/flood_transformer/explain.py`

### 5.1 输入节点消融

函数：`input_ablation_test`

- 将单个输入门控替换为输入门控均值
- 比较 MSE 变化：`delta_mse = mse_ablate - mse_base`

### 5.2 Head 消融

函数：`head_ablation_test`

- 逐层逐 head 置 0
- 记录 `delta_mse`

### 5.3 Neuron 消融

函数：`mean_ablation(ablate="neuron")`

- 将神经元 gate 替换为该层 gate 均值
- 记录 `delta_mse`

输出：

- `head_importance_ranking.csv`
- `neuron_importance_ranking.csv`
- `node_importance_ranking.csv`

说明：这些是"真实消融"得到的节点级指标。

## 6. 边重要性

代码位置：`src/flood_transformer/train.py` 与 `src/flood_transformer/explain.py`

### 6.1 候选边生成（内部边全集 + 可选截断）

函数：`_build_candidate_edges`

- 构造内部边候选：`H(l)->N(l)` 与 `N(l)->H(l+1)`
- 启发式分数：`sqrt(imp(src) * imp(dst))`（仅用于截断排序）
- `edge_ablation_topk=0` 时评估全部内部边（当前默认）
- `edge_ablation_topk>0` 时仅评估 Top-K

### 6.2 真实边消融

函数：`edge_ablation_test`

对候选边 `(s,t)` 计算四个 MSE：

- `base`
- `src`：仅消融源节点
- `dst`：仅消融目标节点
- `both`：同时消融源与目标

定义：

- `delta_both = both - base`
- `interaction = both - src - dst + base`

输出：`edge_importance_ranking.csv`

### 6.3 关键结论

- `edge_importance_ranking.csv` 是真实值（节点联合消融得到）。
- 默认配置 `edge_ablation_topk=0`：覆盖全部内部边。
- 当 `edge_ablation_topk>0`：仅覆盖 Top-K 候选边。

当前默认配置已改为：

- `edge_ablation_topk=0`（`run_experiment.py` / `run_all_stations.py` 默认）
- 即 `run all` 直接产出全内部边真实消融结果，无需再跑重算脚本

## 7. Faithfulness（关键节点保真曲线）

代码位置：`src/flood_transformer/explain.py::faithfulness_k_sweep`

方法：

- 按节点重要性排序，保留 Top-K 节点，其他节点关闭
- 测试性能随 K 的变化（MSE/NSE）

输出：`faithfulness_curve.csv`

作用：评估"最小关键子电路"能否保持原模型性能。

## 8. 因果分析

代码位置：`src/flood_transformer/explain.py`

### 8.1 降雨干预

函数：`causal_intervention_test`

- `replace_mean`：降雨序列替换为均值
- `shuffle_keep_total`：打乱时间顺序但保持总量
- 观察 MSE 增量

输出：`causal_intervention_results.json`

### 8.2 变量交互（核心）

函数：`variable_interaction_test`

定义：

- `Δ_i = MSE(do(x_i)) - MSE(base)`
- `Δ_ij = MSE(do(x_i,x_j)) - MSE(base)`
- `I_ij = Δ_ij - Δ_i - Δ_j`

解释：

- `I_ij > 0`：协同效应
- `I_ij < 0`：冗余/替代效应

输出：

- `variable_interactions.json`
- `variable_interactions_ranking.csv`

### 8.3 IN -> 节点桥接（真实交互）

函数：`in_node_interaction_test`

定义：

- `I(i,j) = MSE(do(i),ablate(j)) - MSE(do(i)) - MSE(ablate(j)) + MSE(base)`

其中：

- `i` 是输入变量
- `j` 是目标节点（训练流程默认取 L0 active head/neuron，可通过 `in_node_bridge_max_nodes` 截断）

输出：

- `in_node_interactions.json`
- `in_node_interactions_ranking.csv`

## 9. 探针分析（Probe）

代码位置：`src/flood_transformer/explain.py::linear_probe`

方法：

- 用最后层隐藏态拟合线性回归
- 预测物理代理量：`cum_rain`, `api`, `dpdt`
- 指标：`R^2`

输出：`physical_probe_results.json`

作用：判断隐藏表示是否编码了可解释的物理信息。

## 10. 注意力统计

代码位置：`src/flood_transformer/explain.py::get_attention_statistics`

方法：

- 对最后一层注意力取最后 query 对历史 lag 的关注
- 比较洪水样本（高分位）与正常样本均值差异

输出：`attention_statistics.json`

## 11. 可视化层的分数融合逻辑（非常重要）

代码位置：`streamlit_circuit_viz.py`

- 真实边：读取 `edge_importance_ranking*.csv` 的 `interaction`
- 启发式边：由节点分数推导
- 融合：`merge_edge_importance(real, heuristic, real_weight)`
- 还存在按层排序增强 `_layer_rank_fade`

因此图上边色不等于"纯真实边值"。

## 12. 现在新增的离线重算脚本

### 12.1 单站重算

- `scripts/recompute_real_edge_importance.py`

用途：

- 加载已有 `sparse_transformer.pt`（不重训）
- 对可评估内部边做真实消融
- 输出 `edge_importance_ranking_real.csv`

### 12.2 全站批量重算

- `scripts/recompute_real_edge_importance_all_stations.py`

用途：

- 扫描 `output_root` 下所有站点
- 批量生成各站点 `edge_importance_ranking_real.csv`
- 生成汇总报告 `real_edge_recompute_report.csv`
- 支持多进程并行：`--num_workers`（默认 4）

### 12.3 Streamlit 读取优先级

代码位置：`streamlit_circuit_viz.py::_load_edge_importance_table`

优先读取：

1. `edge_importance_ranking_real_full.csv`
2. `edge_importance_ranking_real.csv`
3. `edge_importance_ranking.csv`

## 13. 关于“相关性分析”当前状态

代码中没有单独实现 Pearson/Spearman 相关系数模块。

当前可用的"关联"分析主要是：

- 变量交互矩阵 `interaction_matrix`（因果交互）
- Probe 的 `R^2`（表示可读性）
- 注意力差异（洪水 vs 非洪水）

若你需要严格相关性分析（如 `corr(feature, pred)`、`corr(single_delta, input_share)`），建议新增独立分析脚本。

## 14. 哪些是"真实值"，哪些是"启发式"（总表）

- 真实值：
- 输入/Head/Neuron 消融 `delta_mse`
- 候选边的 `interaction` 与 `delta_both`
- 因果干预 `single_delta / interaction_matrix`
- Faithfulness 曲线
- Probe `R^2`

- 启发式：
- 候选边生成分数 `sqrt(imp(src)*imp(dst))`
- 可视化里对非真实边的补分与融合
- 可视化按层排序增强后的颜色强度

## 15. 推荐解释口径（避免冲突）

- 回答"哪个输入主导"：优先用 `single_delta`
- 回答"内部哪些单元关键"：用节点/边真实消融
- 回答"结构-因果是否一致"：需要额外计算 `IN -> L0` 的真实桥接量（当前内部边脚本不覆盖）
