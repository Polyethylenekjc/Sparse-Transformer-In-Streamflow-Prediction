# Explainable Sparse Transformer for Flood Prediction

本项目实现一个**可解释稀疏 Transformer 洪水预测模型**，目标不仅是预测流量，还用于分析模型是否学习到水文学机制。

## 功能概览

- 时间序列 Transformer（多头注意力 + 残差 + MLP）
- 权重 Top-K 稀疏与稀疏率退火 `update_sparsity`
- MLP 输出 Top-K 激活稀疏
- 可学习电路 mask（input feature + attention head + MLP neuron）
- 电路提取与消融分析：
  - `prune_circuit`
  - `mean_ablation`
  - `head_ablation_test`
- 可解释性接口：
  - `get_attention_statistics`
  - `extract_hidden_states`
  - `linear_probe`
  - `causal_intervention_test`

## 目录结构

- `src/flood_transformer/config.py`: 实验参数配置
- `src/flood_transformer/sparsity.py`: 稀疏函数与 STE
- `edge_importance_ranking.csv`: Top-K 连线真实边消融重要性（interaction）
- `variable_interactions.json`: 输入变量两两交互矩阵（真实干预）
- `variable_interactions_ranking.csv`: 变量对交互强度排序
- `node_importance_ranking.csv`: 节点重要性统一排序（head+neuron）
- `faithfulness_curve.csv`: 保留 Top-K 节点时的性能曲线
- `cherry_samples.json`: 高流量/低流量代表样本及变量时序
## 安装

```bash
pip install -r requirements.txt
```

## 运行

```bash
python run_experiment.py \
  --station_id 01013500 \
  --forcing_dir Processed/Forcing \
  --streamflow_dir Processed/Streamflow \
  --task regression \
  --device cpu \
  --epochs_dense 5 \
  --epochs_sparse 5 \
  --epochs_mask 5 \
  --debug \
  --log_every 20 \
  --sparsity_anneal_mode power_law \
  --sparsity_anneal_exponent 2 \
  --weight_topk_mode neuronwise \
  --minimum_alive_per_neuron 2 \
  --circuit_threshold 0.5 \
  --edge_ablation_topk 30

如果你的数据在 `data/Forcing` 与 `data/Streamflow`，可直接改为：

```bash
python run_experiment.py \
  --station_id 01013500 \
  --forcing_dir data/Forcing \
  --streamflow_dir data/Streamflow \
  --task regression \
  --device cpu
```
```

## 输出结果

默认输出到 `outputs/`：

- `head_importance_ranking.csv`: attention head 重要性排序
- `neuron_importance_ranking.csv`: MLP neuron 重要性排序
- `circuit_graph.png`: 最小电路结构图
- `attention_statistics.png`: 洪水前后注意力分布
- `physical_probe_results.json`: 与物理变量相关性（线性探针）
- `causal_intervention_results.json`: 因果干预测试结果
- `summary.json`: 全部输出文件索引

## 可交互电路可视化

运行：

```bash
streamlit run streamlit_circuit_viz.py
```

页面支持：

- 交互电路图（输入变量简写 -> heads -> neurons -> 输出流量）
- Attention Head / MLP Neuron 重要性交互排行
- 洪水前后 attention 分布曲线
- 因果干预与线性探针结果查看
- 输入变量交互矩阵与 Top 变量对交互排序
- Figure4 风格关键路径图（Top-K 节点 + 非活跃灰化 + 层边界虚线 + faithfulness 曲线）

默认读取 `outputs/`，可在左侧栏切换输出目录。

### 训练调试输出

- 使用 `--debug` 开启 step 级调试信息（loss、mask loss、sparsity、val loss）。
- 默认启用 Rich 进度条（按阶段显示 dense/sparse/mask）。
- 如需纯文本日志可加 `--no_rich`。

### 模型性能门槛（NSE）

- 回归任务会自动计算 `NSE`（Nash-Sutcliffe Efficiency）。
- 可通过 `--nse_threshold` 设置达标阈值（默认 `0.5`）。
- 结果会写入 `outputs/summary.json`，并在 Streamlit 页面中显示 `NSE Pass`。

## 说明

- 数据列名会自动匹配（`P/T/PET/Q_up/Soil/Q` 等候选名）。
- 若真实数据不满足列名或结构要求，代码将自动回退到合成数据，以保证流程可运行。
- 借鉴 `openai/circuit_sparsity` 思路：支持 `cosine/linear/power_law` 稀疏退火，且可启用 neuron-wise Top-K 并约束每个神经元最小存活连接数。


### 输入变量稀疏裁剪

现在输入特征也会参与可学习 mask，并可在电路提取时按阈值裁剪：

- `--lambda_input_mask_l1`：输入节点稀疏正则强度（越大越稀疏）。
- `--input_threshold`：提取最小电路时输入节点激活阈值（越大保留越少）。

推荐调参（从保守到激进）：

1. 先固定 `--input_threshold 0.5`，将 `--lambda_input_mask_l1` 从 `1e-5 -> 1e-4 -> 5e-4` 递增。
2. 观察 `outputs/node_importance_ranking.csv` 里 `node_type=input` 的数量变化与 `summary.json` 的 NSE。
3. 若性能下降明显（NSE 下降 > 0.03），减小 `--lambda_input_mask_l1` 或将 `--input_threshold` 降到 `0.4`。
4. 若输入仍过多，保持 L1 不变，把 `--input_threshold` 提升到 `0.55~0.7` 做后处理裁剪。

