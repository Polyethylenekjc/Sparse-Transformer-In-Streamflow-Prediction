# 图源、纠错与 A4 出图记录

新统一入口：`revision/figures.py`；版式：`revision/plot_style.py`。原 `scripts/` 中忽略的 Python/R 文件已纳入版本控制，保留为历史实现。新入口不用旧图中的“反事实误差合格”判据。

|稿件原图|原图源或现有文件|新图与处理|
|---|---|---|
|Fig.1 方法概览|Fig1 MainStruct.png；未找到矢量生成器|Fig1_revision_workflow.pdf：可复现的新协议示意，不伪称旧实验实际流程|
|Fig.2 三阶段|Fig2 ThreeStep.png；旧 Stage II 标签错误|Fig2_revision_training.pdf：Dense / Sparse / Circuit gating 标签与新增实现一致|
|Fig.3 表现和地理分布|plot_nse_hist_taylor.R、streamlit_us_map.py、stations.csv|Fig3_performance_and_size.pdf、Fig3_basin_locations.pdf；高程直接读 forcing_elev_m，单位 m，不复用不明颜色数值|
|Fig.4 密集示例/保留曲线|streamlit_circuit_viz.py、circuit_structure.json、variable_interactions.json、faithfulness_curve.csv|Fig4_example_01013500/03504000/01439500.pdf；三站分别独立出图，旧保留曲线明确标 legacy；输入里的两份 Q 分别标 Q (forcing)/Q (target)|
|Fig.5 探测|physical_probe_results.json|Fig5_probe_scores_unfiltered.pdf；全部现有 R²、负值不滤除、过大负值使用标明的对称对数轴；旧 dpdt 保留“legacy rain change”身份|
|Fig.6 交互总体|plot_interaction_analysis_3x2.R、plot_structural_chord.R|Fig6_interaction_counts.pdf：用类别计数替代难读的圆形总结；前 12 类加 Other，完整计数 CSV 单独保存|
|Fig.7 扰动|physical_sensitivity_analysis.py、plot_counterfactual_sensitivity.R、station_table.csv|Fig7_model_sensitivity_*.pdf：相对基准预测总和百分比；不把与原观测的误差变化当物理正确性验证|
|Fig.8 RF|run_random_forest_models.py、plot_steh_rf_figure_4x2.R、performance.csv|Fig8_RF_available_rows.pdf：明确 available rows，不当成稿件选定 327 个流域；需先提供同一子集才能直接替换对应原图|

当前示例保留稿件顺序：01013500 高 NSE；03504000 较低 NSE≈0.622；01439500 中等 NSE≈0.737。拆图后不沿用旧 b/c/d/e 面板编号；原 Word 的对应段落、图注和引用需要在作者选择最终图后同步。新图没有自动替换 Word 图片。

旧输出仍使用原标准化和重复 Q 输入；重绘改善排版和表述，不修复这些科学问题，也不代表新协议的结果。尤其旧 Q–Q 类别需重新评价。没有找到源文件的完整网络插图未进行像素涂改；提供新代码生成的示意和已有数值的简洁视图，细粒度拓扑图保留原件供追溯。

PNG 为本地 600 dpi 预览，矢量 PDF 是论文导入优先格式。所有图严格保持 170 mm 页面宽度，没有使用会改变物理尺寸的自动紧裁剪。单栏使用时应在 plot_style.py 重新按栏宽生成，不能直接缩小双栏成图。

极端负 NSE 和负 R² 没有删除；分布/散点使用明确标出的对称对数轴。NSE 地理色标以 0 分界分别缩放负值区间和 0–1 区间，以免少数极差模型把其余点压成同一种颜色；图注应说明此分段色标。完整数值随本地重绘 CSV 保留。
