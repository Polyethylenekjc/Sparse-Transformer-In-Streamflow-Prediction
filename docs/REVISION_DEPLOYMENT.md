# 最小修改实验：部署与执行

本分支只准备实验代码和配置；**未运行真实流域训练**。论文目标是水文相关表征的存在性和模型内计算作用，不增加精度排行榜。D 盘原工作区及 Word 文件没有在本轮源码任务中改动。

## 新服务器部署

```bash
git clone --branch codex/minimal-revision-experiments-a4 https://github.com/Polyethylenekjc/Sparse-Transformer-In-Streamflow-Prediction.git
cd Sparse-Transformer-In-Streamflow-Prediction
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-revision.txt
```

Windows 可用 `py -3.11 -m venv .venv`、`.venv\Scripts\Activate.ps1` 和相同 pip 命令。已有 Conda 环境可直接 `conda activate pytorch`；也提供 `environment-revision.yml`。CUDA 服务器请根据服务器驱动安装适合的 PyTorch 包，再安装其余依赖；无需安装 Streamlit、Optuna 或 R 来运行新增实验。版本范围便于部署，运行时版本会保存；本轮小测试使用 Python 3.13 / PyTorch 2.12.0，依赖完整版本见验证记录。

数据目录应包含 `Forcing/<8位流域ID>.csv`、`Streamflow/<ID>.csv`，每日 `date_utc` 日期及数值变量。历史输出和检查点不上传；已有 Git 数据可直接使用，新增数据可独立同步后通过 `--data-root` 指定。默认配置用相对路径，不依赖原作者的 `/mnt/d` 路径。

## 一键入口与成本控制

```bash
# 默认只冻结抽样和列出任务，不训练
bash start_revision.sh
# 在服务器准备好后，显式启动全套必要实验
bash start_revision.sh run
# 指定外部数据和结果目录
bash start_revision.sh prepare --data-root /srv/camels --output-root /srv/results/revision_v1
bash start_revision.sh run --data-root /srv/camels --output-root /srv/results/revision_v1
# 限制本次任务数；仍沿用已经固定的抽样名单
python run_revision.py run --max-jobs 1
# 单独运行已经抽中的流域，不能临时添加“成功案例”
python run_revision.py run --station 01013500 --only steh
python run_revision.py summarize
```

Windows：`./start_revision.ps1` 只准备；`./start_revision.ps1 -Action run` 启动。可指定 `-Python`、`-Config`、`-DataRoot`、`-OutputRoot`。

默认简单随机抽取 **12 个流域 × 3 个种子 × 3 个模型设置 = 108 个任务**：含 Q STEH、无 Q STEH、含 Q LSTM。LSTM 每任务 4 个验证集候选，实际训练次数高于任务数。每个 STEH 有 10/10/10 个阶段 epoch；LSTM 每候选 30 epoch。这是可调整的预设预算，不是已经根据验证结果选出的最佳配置。先审阅 `configs/revision_minimal.json` 再冻结方案。不会默认运行全部 521 流域、全边消融、长窗口、RF 重训或 C01–C05 扩展。

`manifest.json` 保存总体名单、无放回抽样名单、种子、判据、任务、配置/源码/样本文件哈希。不能根据测试表现换样本。更改数据、代码或配置后必须使用新的输出目录；不会静默复用旧结果。`summary.json` 是完成标志，重复命令跳过已完成任务。训练完成后先保存 `trained.pt`，解释阶段中断可从它续跑；训练阶段中断会重新训练该任务，不承诺逐 batch 断点恢复。失败记录保留为 `failure.json`，失败流域不从总体或样本中删除。准备阶段只做文件和列结构筛选，完整数据质量检查在加载时进行。

## 实验与任务对应

|清单项目|新增实现|主要输出|
|---|---|---|
|E01 固定抽样|`prepare`，无放回随机抽样，预设判据|manifest.json|
|E02 持续性参照|最后一次已知 Q；固定岭参数的 15 阶 AR|baseline_predictions.csv、baselines.json|
|E03 无 Q 重训|移除所有径流副本，随机初始化训练|steh_no_q/、preprocessing.json|
|E04 无 Q 解释|输入消融、验证集选前三交互、未过滤探测|interpretation.json、probes.json|
|E05 回路作用|完整、仅核心、去核心、20 次结构匹配随机控制|circuit_summary.csv|
|E06 种子稳定性|同名输入、交互类别和功能差异；不匹配神经元编号|seed_stability.csv|
|E07 LSTM|同日期/含 Q 输入，4 个候选仅按验证 MSE 选择|lstm_tuning.json、summary.csv|
|A01/A04|未过滤探测 R²、测试原单位 NSE/KGE/RMSE/MAE/bias|probe_summary_unfiltered.csv、summary.csv|
|A02/A03/A07|图片排版由作者自行处理，新增重绘代码已回滚|沿用原图与原绘图脚本|
|C06 条件窗口|`optional_window_lengths` 默认为空，可显式改为 `[7]`|独立 L7/ 输出；不默认开展|

无 Q 模型从头训练，不能由含 Q 检查点删列替代。持续性/AR 仍需要 Q，属于竞争解释参照，不能作为输入公平的无 Q 基线。不同模型都使用共同的有效目标日期；样本需具有完整历史 Q，Q 仅用于配对日期资格和有 Q 基线，不会进入无 Q 网络。长窗口/泛化/RF 扩展不默认执行。

## 数据与论文解释必须同步的更正

旧加载器把 `Forcing` 的 `streamflow_cfs` 与 `Streamflow` 的 `target_q` 同时作为输入，并在划分前用全记录拟合标准化。这些现有结果不能通过补写方法说明修复。新入口使用 `revision/data.py`，只留一份 Q；无 Q 移除全部 Q 类列，训练期拟合标准化，按目标日期划分，缺失日期补成缺测，舍弃跨缺口窗口，不将非连续日期压成连续序列。旧入口和检查点保留用于追溯，**不要用新标准化参数评价旧检查点**。旧 Q–Q 交互可能反映重复输入，需用新结果重新解释。

新入口逐流域、逐种子独立初始化模型，不共享权重。默认 3 层 encoder block、128 维表示、4 个注意力头、256 维前馈隐层、学习位置编码、pre-LayerNorm 残差；最终 LayerNorm 后取最后时间步的线性回归头，**没有 decoder**。激活 Top-K 作用于前馈输出的 128 维，不是 256 维前馈隐层。STE 门前向 `sigmoid >=0.5`，反向保留 sigmoid 梯度；L1 对门概率求和，第三阶段更新权重和门。权重投影逐矩阵进行，MLP 用 neuronwise 保留最少连接，不能写成整个模型合并排序的全局 Top-K。旧入口的门和激活稀疏开关未严格按三阶段关闭；新入口显式切换，新增方法图只描述新协议。实际旧实验配置仍应从逐站记录恢复，不把默认值写成全部既有流域都使用的设置。

提供的数据曾经插值，`imputation_report.csv` 只有数量和最大缺口，不能恢复哪些日期是原始观测。没有逐日标记时输出会明确写 `provided_CSV_may_include_imputation`，不能声称完成 A06。获得原始标记后设置 `observation_mask_dir`：每站 CSV 必须有日期和 `q_observed` 数值 0/1。验证/测试会排除非原始观测目标日；可设置 `exclude_imputed_q_history=true` 同时排除历史 Q 插值窗口。训练数据允许保留已提供的插值，标准化仍只拟合训练期；原插值算法是否使用未来信息需要单独审核。本代码不会伪造逐日观测标记。

## 回路检验与统计范围

核心只按验证集单位消融影响选择，默认每层先按权重强度选最多 32 个神经元候选，再验证排名；所有具有效非零输入/输出连接且门开启的头均纳入候选。核心保留每组候选的 25%，完整模型保留全部训练门。`max_neuron_candidates_per_layer=0` 可以做所有活跃神经元消融，成本明显增加，默认不做。候选覆盖率输出，避免把部分搜索称为全局最小回路。

随机控制精确匹配层、单位类型和数量，并在有非零连接的单位中近似匹配输入/输出权重度数；记录度数偏差与核心重叠。保留输入门、位置编码、偏置、残差和输出头。**这是单位掩码功能检验，不是只保留一张闭合边图，也不是严格的全图路径匹配**；控制与核心完全相同时不会记作核心胜出。如果期刊要求逐边拓扑或严格路径匹配，需要进一步实现，不能把本轮近似控制写成已经满足该要求。

测试推理分析默认固定抽取 512 个目标日，验证消融用固定 256 个目标日，不用随机重复日期夸大样本量；完整预测指标仍评分全部有效测试日。输入消融的零指训练均值，交互是组合消融的非加性 MSE，不证明物理因果。探测器在训练期隐状态拟合、测试期评分，保留负 R²、常数目标和拟合失败；额外附低成本输入历史探测对照，未默认增加随机网络探测。CUM/API/CHG 使用原单位降雨和 SI 的公式，参数明确保存在配置；旧 `dpdt` 采用另一计算方式，不将它直接改名成新的 CHG。

默认功能通过条件：核心损失不超过 0.05 NSE、去核心损失至少 0.02 NSE、核心严格优于至少 80% 随机控制。这些是供修改方案预先审阅的操作判据，不是来自已有结果的科学定理；应在训练前决定，不能看完测试结果再调阈值。

同一流域所有预设种子均通过才记一次流域通过。全部样本完成后才输出流域比例与通过反演超几何分布计算的精确有限总体 95% 区间；附描述性 Wilson 二项近似作为参照。它只适用于本次无放回简单随机抽样、已冻结判据及合格数据总体。多个种子/日期/重复抽样均不能增加独立流域数，也不能据此断言整个模型、全部流域或真实水文机制成立。未完成或失败必须报告，不能只统计成功任务。

## 图片处理

2026年10月12日按作者要求撤回全部新增图片及绘图样式更新。保留原图片和原绘图脚本；作者自行处理图片字号和排版。本分支仅提供实验准备代码，不提供替代论文原图的重绘结果。

## 验证

```bash
python -m unittest discover -s tests -v
# 只审核抽样数据的日期、Q 副本和特征；不训练
python -m revision.audit --output-root outputs/revision_data_audit
```

仅 140 天合成序列、8 维 STEH、1 个 epoch/阶段、2 个小 LSTM 候选，验证无 Q 列移除、训练期缩放、日历缺口、公共测试日期、显式启动、续跑和检查点重载。它不验证论文结论，不替代实际服务器 CUDA 检查或新增流域实验。真实实验可在服务器上先用 `--max-jobs 1` 验证部署。
