"""Write an evidence assessment of the inspected Word manuscript; no document editing."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--manuscript-text',type=Path,required=True)
    p.add_argument('--supplement-text',type=Path,required=True)
    args=p.parse_args();out=args.report
    audit=json.loads((out/'audit.json').read_text(encoding='utf-8'))
    control=json.loads((out/'caliper-audit.json').read_text(encoding='utf-8'))
    assert audit['status']==control['status']=='PASS'
    paired=pd.read_csv(out/'paired_primary.csv',dtype={'station':str})
    diag=pd.read_csv(out/'convergence_diagnostics.csv');diag=diag[diag.seed==42]
    stab=pd.read_csv(out/'stability_basin.csv',dtype={'station':str})
    assoc=pd.read_csv(out/'node_count_associations.csv')
    probes=pd.read_csv(out/'probe_summary.csv')
    runs=pd.read_csv(out/'all_runs.csv')
    native_criterion=((runs.nse-runs.core_nse)<=.05)&((runs.nse-runs.removed_nse)>=.02)&(runs.random_win_fraction>=.8)
    flips=int((native_criterion!=runs.criterion_pass).sum())
    numerical={'models_checked':len(runs),'classification_changes_when_native_full_nse_is_reference':flips,
               'native_binary_forward_difference':audit['native_binary_forward_difference'],
               'scope':'Sensitivity only; original binary intervention criterion remains primary.'}
    (out/'native-binary-criterion-sensitivity.json').write_text(json.dumps(numerical,indent=2),encoding='utf-8')
    skill_context=[]
    for q in [True,False]:
        group=runs[(runs.seed==42)&(runs.with_q==q)]
        passed=group[group.criterion_pass]
        skill_context.append({'with_q':q,'basins':len(group),'circuit_passes':len(passed),
            'passes_with_negative_native_nse':int((passed.nse<0).sum()),
            'passes_with_native_nse_above_half':int((passed.nse>.5).sum()),
            'all_native_nse_above_half':int((group.nse>.5).sum()),
            'scope':'Post-hoc descriptive predictive-skill context only; does not redefine the frozen criterion or add confirmation inference.'})
    pd.DataFrame(skill_context).to_csv(out/'circuit_predictive_skill_context.csv',index=False)
    med=lambda key:paired[key].median()
    gt=int((paired.steh_with_q_nse>.5).sum())
    text=f'''# 实验完成情况与论文写作证据评估（2026-10-14）

## 判断

实验记录可以支撑一篇聚焦“输入衍生水文信息的可解码性与部分功能子回路”的论文。**当前 Word 稿件还不能直接据此投稿**：数值、图表和方法描述仍混有旧实验；结构完全收敛、唯一物理回路、普遍优于 LSTM、STEH 特异 RF 迁移增益均没有获得本轮证据支持。实验计算核验通过与科学结论成立是两个不同判断。本评估采用 academic-research-skills 的统计和图表审查标准，不是正式多视角同行评审或录用保证。

评估对象为实际 Word 的可读文本快照：Manuscript_1st.docx 与 Supplementary Information_1st.docx。它们没有被本脚本修改。本报告和新图表完成不等于正文、补充材料、审稿回复已经同步完成。

## 已完成的实验与核验

- LSTM：全部 521 个站点各自训练一个新模型，seed42，15 天窗口。共用搜索空间与一套最终参数；24 个 TPE 候选在 6 个开发站点上进行验证，144 次调参拟合。调参与最终训练共 35.83 分钟。没有增加 LSTM 多种子、多窗口。
- STEH：521×2 个输入设置×seed42，加 12 个预先随机选定的新流域×2 设置×seed43/44，共 1,090 个模型；所有训练完成。权重逐站点独立，归一化逐站点只由训练期拟合，共享一套最终超参数。
- 完整、仅核心、去核心及原定最近度匹配对照：1,090 个结果完成。原先 3 个失败来自用二值干预前向的原始单位 RMSE/MAE 与原生门控前向混作同一参考，已改为分别验证两套预测与其指标；原始失败记录和数值差异均保留，检查点与科学判据未改变。
- 两套预设结构匹配宽度、每模型每宽度 20 个对照：1,090 个模型、43,600 组逐日对照预测完成并独立重评分。向量化推理的旧运行只完成 1,016/1,090 个核验，其 74 个失败保留；最终结果将所有对照改为逐个参考推理，保留站点间并行，没有放宽数值容差或选择较好测试结果。
- 原生预测、二值干预预测、控制预测、时间划分、数据来源散列、选择身份、有限总体区间及功能稳定性均有审计记录。探测 R² 只能核查保存的分数、数量和状态，未保存的探测预测/表示无法独立重算，不能称为全面独立复现。

## 可以写进论文的结果

| 结果 | 本轮证据 | 写作边界 |
| --- | --- | --- |
| 预测背景 | STEH with-Q 的 NSE 中位数 {med('steh_with_q_nse'):.6f}；LSTM {med('lstm_nse'):.6f}；no-Q {med('steh_no_q_nse'):.6f} | LSTM 为匹配的逐流域本地基线，不是跨流域联合模型的最优水平 |
| 逐站比较 | with-Q 在 {audit['primary_with_q_beats_lstm']}/521 站点胜过 LSTM；NSE 差值中位数 {med('with_q_minus_lstm'):.6f} | 均值受极端值影响；不能写全面优越或最佳精度 |
| 固定回路判据 | with-Q {audit['primary_passes']['True']}/521（{100*audit['primary_passes']['True']/521:.1f}%），no-Q {audit['primary_passes']['False']}/521（{100*audit['primary_passes']['False']/521:.1f}%） | 三项联合条件为核心损失≤0.05、移除损失≥0.02、胜率≥0.8；不可将单项满足算作总通过 |
| 预测能力计数 | with-Q 共 {gt}/521 个 NSE>0.5 | 当前 Word 的 327 应更新；不可仍使用旧筛选样本统计 |
| 探测 | CUM/API 输入历史线性参照 R² 接近 1；隐藏表示可解码但更低；CHG 保留负分数 | 目标由输入构造，说明信息保留，不证明独立土壤水分、积雪状态或新发现物理机制 |
| 稳定性 | 12 个确认流域，3 个固定种子；报告流域内排序、双精度 CKA、完整/核心预测相关性 | 72 个种子对不是 72 个独立流域；不同神经元编号/结构不相同不自动意味着功能不同 |

所有 521 个流域均保留，负 NSE、极端值、未定义胜率和未通过回路都进入主表。总体描述无需对这 521 个已枚举流域虚构抽样区间。原定确认样本区间针对 N=503 的合格总体；两套匹配宽度的确认区间针对排除开发流域后的 N=492，四个 98.75% 边际区间仅在该四项族内具有至少 95% 联合覆盖。它们不推断所有初始化。

## 结构收敛与功能差异：需要明确报告的问题

'''
    for q in [True,False]:
        d=diag[diag.with_q==q];s=stab[stab.with_q==q]
        text+=f"- {'with-Q' if q else 'no-Q'}：掩码末三轮门控翻转率中位数 {100*d.mask_tail_gate_churn.median():.2f}%；入选检查点中概率距 0.5 小于 0.02 的门控比例中位数 {100*d.mask_selected_near_threshold_fraction.median():.2f}%。末期翻转率超过 1% 的站点有 {int((d.mask_tail_gate_churn>.01).sum())}/521；1% 仅为描述标记，不是事后科学通过阈值。确认流域的输入作用排序相关均值 {s.input_rank_mean.mean():.3f}，FP64 CKA 均值 {s.hidden_cka_mean.mean():.3f}。\n"
    text+='\n对照匹配敏感性（每设置 521 个主要模型）：\n\n'
    for row in control['population_summary']:
        text+=f"- {'with-Q' if row['with_q'] else 'no-Q'}，预设匹配宽度 {row['caliper']:g}：联合判据通过 {row['caliper_passes']}/521（原定最近度对照 {row['nearest_degree_passes']}/521）。\n"
    text+='\n确认样本的三个种子联合通过（条件总体 N=492）：\n\n'
    for row in control['confirmation']:
        lo,hi=row['success_fraction_bounds']
        text+=f"- {'with-Q' if row['with_q'] else 'no-Q'}，宽度 {row['caliper']:g}：{row['sample_successes']}/12；98.75% 边际区间 [{lo:.3f}, {hi:.3f}]。\n"
    numerical_difference=audit['native_binary_forward_difference']
    text+=f"\n原生/二值前向差异：{numerical_difference['models_exceeding_pointwise_tolerance']} 个模型存在超出原逐日容差的预测；最大训练缩放单位差异 {numerical_difference['maximum_train_scaled_prediction_difference']:.6g}，最大 NSE 差异 {numerical_difference['maximum_absolute_nse_difference']:.6g}。改用原生完整模型 NSE 作为三项判据参考时，分类改变数为 {flips}/{len(runs)}。两套逐日预测均保留，原定干预分析不被替换。\n"
    text+='\n**通过相对回路判据不等于预测具有足够技能。**\n\n'
    for row in skill_context:
        text+=f"- {'with-Q' if row['with_q'] else 'no-Q'}：通过模型中有 {row['passes_with_negative_native_nse']} 个完整模型 NSE<0；同时达到 NSE>0.5 且通过回路判据的为 {row['passes_with_native_nse_above_half']}/{row['basins']}（NSE>0.5 的总数 {row['all_native_nse_above_half']}）。\n"
    text+='\n上述 NSE=0.5 分层仅为事后描述性预测背景，不修改原判据，也不新增确认推断。原定确认中，no-Q 的 02312200 三种子均通过相对回路判据，但完整模型三种子平均 NSE 约为 −3.554；不能将它当作具有良好预测技能的稳定水文回路。确认成功还须结合完整模型技能、作用大小和失败机制解读。\n'
    text+='''
这些数值不支持“门控完全稳定收敛”。有限训练预算下以验证损失选择检查点是合理操作，但性能选择与结构收敛不能混为一谈。图 3 同时呈现功能稳定性和门控变化，图 5 呈现固定三阶段损失分布。回路相似度不应只比较未经匹配的神经元编号；CKA 较高也不证明存在同一个因果图。

已实施的改进为：验证集选核心取代失败的固定 25% 核心；增加两套预设结构对照敏感性；保留退化与重复对照；修正数值参考、用逐个推理保证对照一致；以 FP64 中心化检查 CKA；用流域作为确认单位。它们提高核验与统计有效性，并没有使所有站点都变成稳定成功结果。

若论文必须坚持“结构充分收敛”的强主张，后续应只在开发/验证数据上预设门控学习率退火、阈值裕量/离散化策略及更长训练预算，用验证性能与末期翻转联合选择；需要重新冻结协议后统一重跑受影响实验。本轮没有根据测试回路通过率反复调整方法，也没有把未运行的稳定化训练写成完成。

## 对当前 Word 四个研究问题的逐项判断

| 研究问题/任务 | 已有新证据 | 仍需处理 |
| --- | --- | --- |
| P24 结构与性能；A03 | 全 521 流域连续关系，图 8 与 node_count_associations.csv | 只能写相关；验证保真预算可机械关联核心大小与性能。旧 NSE 分组、45.77/56.35 节点均值与 p 值不可沿用 |
| P25 表征存在；A01/C01 | 未过滤的 6,540 条探测记录、输入参照，图 7 | CUM/API 可由输入线性重构；无全站随机未训练表示对照。若只保留可解码性，明确这一较弱范围 |
| P26 计算行为；E04/E05/E06 | 输入单特征消融、核心/去核心/结构对照、跨种子功能比较 | 本轮未补完整成对交互或旧情境扰动。P–Q/Q–Q/TQ 相关性标签不能冒充内部成对回路交互。保留旧扰动必须恢复输出并重算相对原模型响应；否则删减相关结果与强措辞 |
| P27 RF 实用应用；C05/T21 | 旧稿有同流域 RF 示例 | 本轮未重新核验 RF 的 0.760→0.797、327 个筛选样本、10/14/15 天有效历史与选择范围；不能当作新协议结果。建议移到明确标注旧协议的探索性补充，或删除核心贡献中的 RF 结论；保留特异增益则仍需匹配通用特征控制 |

E01（协议/抽样）、E02（持久性与 AR）、E03（无 Q 从头训练）、E05（核心干预）、E07（调参 LSTM）已取得数值证据；E04 与 E06 的输入/核心部分已做，成对交互部分仍未做，不能整项宣称全部完成。T17–T20 的方法事实现可补写，但当前 Word 尚未应用。A05 的本轮归一化、划分、调参及核心选择已核验，旧稿 RF/插值来源不能借此追认；A06 缺原始观测标记，不能做“仅原始目标日”重评分。T26 审稿回复未完成。

## 稀疏度与性能的连续关系

'''
    for row in assoc.to_dict('records'):
        text+=f"- {'with-Q' if row['with_q'] else 'no-Q'}，{row['unit_measure']} 与 NSE 的 Spearman 相关 = {row['spearman_with_nse']:.3f}。\n"
    text+='''
不同输入设置关联方向不同，旧稿“更小回路普遍带来更强泛化”的推论不成立。全体流域描述不依赖人为以 NSE=0.5 分组；不使用组间显著性代替效应大小。应同时解释预测任务、输入差异和核心选择程序。

## 投稿前必须同步的文稿内容

1. 用 publication_model_table.csv、publication_circuit_table.csv 与确认区间替换旧数值，正文重点为含/无 Q 回路与失败范围，LSTM 和完整控制放补充材料。删去“尚未计算 no-Q/Persistence”的旧句。
2. 方法注明每站点独立模型、全站共用超参数、逐站 train-only 缩放、70/15/15 目标日切分、15 天历史下一日预测、验证调参和验证选核心。输入已提供的 CSV 可能含填补，不能声称获得了完整原始观测审核。
3. 架构对应 encoder-only Transformer、最后时间位置读出；没有实现的 decoder 不应出现在图注。写真实 Python 3.11.17、PyTorch 2.12.0+cu126、RTX3090，以及 dense/sparse/mask 各 10 轮、AdamW、动态稀疏投影和验证检查点选择。
4. 90% 稀疏度针对指定权重矩阵，并非全部参数；位置嵌入仍占参数。零值权重仍用稠密算子，不据此声称 90% 推理加速。门控为直通二值选择，权重支持可恢复，不是永久删除边。
5. 图表使用新协议结果与六项追踪字段；图 1/2/3/4/5/6/7/8 各有 PDF/SVG/600dpi PNG，原稿图 4/6/7 不能只换图号而保留旧结果。示例采用公开的中位性能选择规则，08190500 极端例不截断。
6. 全文范围限定为短期配置下的模型内部计算；不外推季节蓄水、未见流域迁移、独立物理状态或因果机制。CKA 是表示相似性指标，其解释边界见 [Kornblith 等原文](https://proceedings.mlr.press/v97/kornblith19a.html)；独立水文状态研究可对照 [Lees 等 2022](https://hess.copernicus.org/articles/26/3079/2022/)。
7. KGE 与 NSE 的基准不同，不将 KGE=0 自动写成技能界限，见 [Knoben 等 2019](https://hess.copernicus.org/articles/23/4323/2019/)。
8. 引文还需要逐条核验。已确认 P327 的 Apak 等 2026 流量论文配错 DOI：应为 [Scientific Reports 10.1038/s41598-026-40713-1](https://www.nature.com/articles/s41598-026-40713-1)；原 DOI s41467-026-70523-y 对应 DNA 甲基化方法论文，见 [出版方原文](https://www.nature.com/articles/s41467-026-70523-y.pdf)。该发现不等于其余参考文献已全部审核。

## 可用于改稿的英文段落（需与正文和补充材料同步）

**Methods.** We trained separate models for each of the 521 catchments, with shared hyperparameters but catchment-specific weights and training-period normalization. The primary comparison used one seed (42) and a 15-day input history to predict next-day discharge. Twelve additional catchments were selected before confirmation analysis and evaluated with three fixed seeds. Hyperparameter selection and circuit extraction used validation data; test data were used for final evaluation. The LSTM was a tuned, per-catchment baseline and was not evaluated across multiple seeds or input windows.

'''
    text+=f'''**Results.** The median test NSE was {med('steh_with_q_nse'):.3f} for STEH with discharge history and {med('lstm_nse'):.3f} for the matched LSTM baseline. STEH achieved a higher NSE in {audit['primary_with_q_beats_lstm']} of 521 catchments; the median paired difference was {med('with_q_minus_lstm'):.4f}. Under the fixed joint circuit criterion, {audit['primary_passes']['True']} with-Q and {audit['primary_passes']['False']} no-Q models passed. These results identify a subset of models with functionally supported extracted subcircuits rather than a universally successful or uniquely determined circuit architecture.

**Discussion.** Input-derived rainfall proxies were decodable from hidden representations, but the input-history controls could reconstruct cumulative and antecedent rainfall nearly exactly. The probes therefore indicate information retention rather than independent validation of physical states. Gate changes remained appreciable under the finite training budget. Representation similarity and functional ablation provide complementary evidence, but neither establishes complete structural convergence or a unique causal hydrological mechanism.

## 审计与交付入口

- audit.json：主实验计算核验及各项科学失败计数。
- caliper-audit.json：43,600 个保存对照的独立重评分、源检查点散列与条件确认区间。
- 实验核验与论文展示.md、publication_model_table.csv、publication_circuit_table.csv：结果报告与可排版表。
- STEH实验核验.xlsx：全站配对、回路、收敛、功能稳定性和控制敏感性工作簿。
- trace.json、caliper-trace.json、source_manifest.json：数据→变换→图表→结论→限制追踪。
- 本报告是当前证据的范围判断；未修改 Word、未补写逐条回复，未把暂缓实验标为完成。
'''
    target=out/'实验完成情况与论文写作证据评估_2026-10-14.md'
    target.write_text(text,encoding='utf-8')
    manifest={'assessment_scope':'Experiment evidence review of actual Word snapshots; no manuscript editing or formal reviewer panel.',
              'sources':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [args.manuscript_text,args.supplement_text,out/'audit.json',out/'caliper-audit.json',out/'paired_primary.csv',out/'all_runs.csv',out/'confirmation_basin.csv',out/'convergence_diagnostics.csv',out/'stability_basin.csv',out/'node_count_associations.csv']],
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'output':target.name,'output_sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
    (out/'paper-readiness-source-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(target)


if __name__=='__main__':main()
