"""Independently audit shared-parameter LSTM records and make scientific figures."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from revision.common import metrics,digest,clean

def choose_global(rows, stations):
    """Independently reconstruct the complete validation-only candidate ranking."""
    panels = {}
    for row in rows:
        panel = panels.setdefault(row['trial'], {})
        assert row['station'] not in panel
        panel[row['station']] = row['validation_metrics']['nse']
    assert panels and all(set(panel) == set(stations) for panel in panels.values())
    scores = {trial: float(np.mean([panel[s] for s in stations]))
              for trial, panel in panels.items()}
    assert np.isfinite(list(scores.values())).all()
    return max(scores, key=lambda trial: (scores[trial], -trial)), scores

M=['nse','kge','rmse','mae','bias']
PALETTE=['#0077BB','#EE7733','#009988']
LABELS={'lstm':'LSTM','persistence':'Persistence','ar':'AR ridge'}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,o):p.write_text(json.dumps(clean(o),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def compare(actual,expected):
    if expected is None:assert actual is None
    else:np.testing.assert_allclose(actual,expected,rtol=3e-6,atol=3e-5)

def verify_prediction(folder,kind,record):
    p=pd.read_csv(folder/(kind+'_predictions.csv'))
    values=metrics(p.predicted,p.observed)
    for m in M:compare(values[m],record[m])
    return p,values

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference',type=Path)
    args=p.parse_args();src,out=args.input,args.output;out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((src/'tuning_protocol.json').read_text())
    decision=json.loads((src/'selected_hyperparameters.json').read_text())
    complete=json.loads((src/'completion.json').read_text())
    assert complete['status']=='PASS' and complete['final_models']==len(protocol['final_stations'])
    assert decision['selection_data']=='validation only' and not decision['test_evaluated_for_selection']
    hp=decision['hyperparameters'];sources=[]
    def trace_file(f):sources.append({'path':f.relative_to(src).as_posix(),'sha256':sha(f)})
    search=[]
    for file in sorted(src.glob('search/trial_*/*/trial_result.json')):
        r=json.loads(file.read_text());log=pd.read_csv(file.parent/'training_log.csv')
        assert r['seed']==protocol['seed'] and len(log)==protocol['training_config']['epochs_lstm']
        verify_prediction(file.parent,'validation',r['validation_metrics'])
        np.testing.assert_allclose(log.val_mse.min(),r['validation_mse'],rtol=2e-6,atol=1e-7)
        search.append(r)
        for f in [file,file.parent/'training_log.csv',file.parent/'validation_predictions.csv']:trace_file(f)
    winner,scores=choose_global(search,protocol['tuning_stations'])
    assert winner==decision['selected_trial']
    np.testing.assert_allclose(scores[winner],decision['mean_validation_nse'])
    assert len(scores)==decision['completed_trials']
    tuning=pd.DataFrame([{'trial':r['trial'],'station':r['station'],'seed':r['seed'],**r['hyperparameters'],
                         'validation_nse':r['validation_metrics']['nse'],'validation_kge':r['validation_metrics']['kge'],
                         'validation_mse':r['validation_mse'],'best_epoch':r['best_epoch'],'seconds':r['seconds'],
                         'selected':r['trial']==winner} for r in search])
    candidate=pd.DataFrame([{'trial':t,'mean_validation_nse':v,**json.loads((src/'search'/f'trial_{t:03d}'/'hyperparameters.json').read_text()),'selected':t==winner} for t,v in sorted(scores.items())])
    rows=[];logs=[];seen=set()
    for folder in sorted((src/'final/models').iterdir()):
        if not folder.is_dir():continue
        r=json.loads((folder/'summary.json').read_text());station=r['station'];seen.add(station)
        assert r['seed']==protocol['seed'] and r['hyperparameters']==hp and r['selected_trial']==winner
        assert r['protocol_sha256']==digest(protocol)
        final=pd.read_csv(folder/'predictions.csv');baseline=pd.read_csv(folder/'baseline_predictions.csv')
        assert final.target_date.equals(baseline.target_date)
        np.testing.assert_allclose(final.observed,baseline.observed,rtol=1e-6,atol=1e-4)
        assert digest(final.target_date.tolist())==r['test_dates_sha256']
        result=metrics(final.predicted,final.observed)
        assert np.isfinite(final[['observed','predicted']]).all().all()
        for m in M:compare(result[m],r['metrics'][m])
        meta=json.loads((folder/'preprocessing.json').read_text())
        assert meta['q_source_column']=='streamflow_cfs' and meta['feature_names'].count('target_q')==1
        log=pd.read_csv(folder/'training_log.csv');assert len(log)==protocol['training_config']['epochs_lstm']
        logs.append(log.assign(station=station))
        trial=json.loads((folder/'trial_result.json').read_text());np.testing.assert_allclose(log.val_mse.min(),trial['validation_mse'],rtol=2e-6,atol=1e-7)
        rows.append({'station':station,'seed':protocol['seed'],'model':'lstm',**result,**hp,
                     'selected_trial':winner,'best_validation_mse':trial['validation_mse'],
                     'best_epoch':trial['best_epoch'],'total_seconds':trial['seconds'],
                     'test_start':final.target_date.iloc[0],'test_end':final.target_date.iloc[-1]})
        base_record=json.loads((folder/'baselines.json').read_text())
        for model in ['persistence','ar']:
            b=metrics(baseline[model],baseline.observed)
            for m in M:compare(b[m],base_record['metrics'][model][m])
            rows.append({'station':station,'model':model,**b})
        for name in ['summary.json','predictions.csv','baseline_predictions.csv','baselines.json','preprocessing.json','run_config.json','training_log.csv','trial_result.json']:trace_file(folder/name)
    assert seen==set(protocol['final_stations'])
    raw=pd.DataFrame(rows);assert np.isfinite(raw[['rmse','mae','bias']]).all().all()
    lstm=raw[raw.model=='lstm'].copy();lstm['n']=lstm.n.astype(int)
    pivot=raw.pivot(index='station',columns='model',values='nse')
    basins=lstm[['station','n','nse','kge','rmse','mae','bias']].copy().set_index('station')
    basins['persistence_nse']=pivot.persistence;basins['ar_nse']=pivot.ar;basins=basins.reset_index()
    summaries=[]
    for model in ['lstm','persistence','ar']:
        df=raw[raw.model==model];row={'model':LABELS[model],'basins':len(df)}
        for m in M:row[m+'_mean']=float(df[m].mean());row[m+'_sd']=float(df[m].std(ddof=1))
        row['nse_defined']=int(df.nse.notna().sum());row['kge_defined']=int(df.kge.notna().sum())
        row['nse_median']=float(df.nse.median());row['kge_median']=float(df.kge.median());summaries.append(row)
    summary=pd.DataFrame(summaries);training=pd.concat(logs,ignore_index=True)
    tables={'model_summary':summary,'per_basin':basins,'seed_metrics':lstm,'candidate_trials':tuning,
            'global_candidates':candidate,'final_training_log':training}
    for name,df in tables.items():df.to_csv(out/(name+'.csv'),index=False)
    reference_ids=[]
    if args.reference:
        ref=json.loads(args.reference.read_text());reference_ids=ref['sample']
        # Predefined comparison subset: dates must match STEH's fixed manifest outputs.
        matched=basins[basins.station.isin(reference_ids)]
        assert len(matched)==len(reference_ids)
        matched.to_csv(out/'steh_comparison_basins.csv',index=False)
    n=len(basins);win_p=int((pivot.lstm>pivot.persistence).sum());win_ar=int((pivot.lstm>pivot.ar).sum())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.labelsize':9,'axes.titlesize':9,
                         'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    traces=[]
    def save(fig,ident,files,caption,limits):
        for ext in ['png','pdf','svg']:fig.savefig(out/(ident+'.'+ext),dpi=600,facecolor='white')
        plt.close(fig)
        traces.append({'artifact_id':ident,'source_data':[{'path':f,'sha256':sha(out/f)} for f in files],
                       'transformation':{'script':'experiments/report_global_lstm.py','sha256':sha(Path(__file__))},
                       'caption_claim':caption,'supported_manuscript_claims':[],'limitations':limits})
    fig,axes=plt.subplots(1,2,figsize=(175/25.4,86/25.4),layout='constrained')
    nonlinear=False
    for ax,m in zip(axes,['nse','kge']):
        for i,model in enumerate(['lstm','persistence','ar']):
            values=np.sort(raw.loc[raw.model==model,m].dropna().to_numpy());quantile=np.arange(1,len(values)+1)/len(values)
            ax.step(values,quantile,where='post',color=PALETTE[i],ls=['-','--',':'][i],lw=1.3,label=LABELS[model]+f' (n={len(values)})')
        if raw[m].min() < -3:ax.set_xscale('asinh',linear_width=1);nonlinear=True
        if raw[m].min() < -3:
            ax.set_xticks([-100,-10,-1,0,1],labels=['-100','-10','-1','0','1'])
            ax.minorticks_off()
        ax.set(xlabel='Test '+m.upper()+' (dimensionless'+('; asinh scale' if raw[m].min() < -3 else '')+')',
               ylabel='Cumulative fraction of defined station scores',ylim=(0,1.02),title=f'{n} station models; seed {protocol["seed"]}')
        ax.grid(color='#dddddd',lw=.5)
    axes[1].legend(frameon=False,loc='upper left')
    save(fig,'fig01_test_distribution',['per_basin.csv','seed_metrics.csv','model_summary.csv'],
         f'Empirical distributions of test NSE and KGE across all {n} stations. Each station contributes one freshly trained, seed-{protocol["seed"]} LSTM with the same globally selected parameters. Q-history references use identical target dates. All negative results are included.'+(' An asinh horizontal scale preserves extreme negative tails.' if nonlinear else ''),
         ['One seed per station; training-seed sensitivity is not estimated.','Baselines use only Q history, whereas LSTM also uses meteorological inputs.','Undefined scores remain in station tables; each CDF denominator counts only its defined scores.'])
    fig,ax=plt.subplots(figsize=(175/25.4,72/25.4),layout='constrained')
    x=candidate.trial.to_numpy()+1;y=candidate.mean_validation_nse.to_numpy()
    ax.plot(x,np.maximum.accumulate(y),color=PALETTE[0],lw=1.3,label='Best observed validation score')
    ax.scatter(x,y,color=PALETTE[1],marker='o',s=18,label='Candidate score')
    ax.scatter([winner+1],[scores[winner]],color=PALETTE[2],marker='*',s=70,label='Selected shared parameters',zorder=4)
    ax.set(xlabel='Candidate evaluation order',ylabel='Mean validation NSE (dimensionless)',
           title=f'Bayesian search on {len(protocol["tuning_stations"])} fixed random stations')
    ax.grid(color='#dddddd',lw=.5);ax.legend(frameon=False,loc='lower right')
    save(fig,'fig02_bayesian_search',['global_candidates.csv','candidate_trials.csv'],
         f'{len(candidate)} global configurations evaluated by the unweighted mean validation NSE on the same {len(protocol["tuning_stations"])} randomly sampled stations. The first four candidates reproduce the initial grid; subsequent candidates use the seeded TPE sampler. The selected star marks the best observed validation configuration.',
         ['Best within the evaluated search budget, not a guaranteed global optimum.','Two candidates are proposed per batch before either batch score is returned.','No test result enters this search.'])
    curve=training.groupby('epoch')[['train_mse','val_mse']].agg(['median',lambda s:s.quantile(.25),lambda s:s.quantile(.75)])
    envelopes=pd.DataFrame({'epoch':np.arange(1,protocol['training_config']['epochs_lstm']+1)})
    fig,axes=plt.subplots(1,2,figsize=(175/25.4,78/25.4),layout='constrained')
    for ax,m,label in zip(axes,['train_mse','val_mse'],['Training','Validation']):
        median,low,high=[curve[m].iloc[:,i].to_numpy() for i in range(3)]
        ax.plot(envelopes.epoch,median,color=PALETTE[0],lw=1.3,label='Median across stations')
        ax.fill_between(envelopes.epoch,low,high,color=PALETTE[0],alpha=.2,label='Interquartile range')
        ax.set(xlabel='Epoch',ylabel='MSE in standardized Q (dimensionless)',title=label,ylim=(0,None))
        ax.grid(color='#dddddd',lw=.5)
        for name,vals in zip(['median','q25','q75'],[median,low,high]):envelopes[m+'_'+name]=vals
    axes[1].legend(frameon=False)
    envelopes.to_csv(out/'learning_curve_envelope.csv',index=False)
    save(fig,'fig03_final_learning',['final_training_log.csv','learning_curve_envelope.csv'],
         f'Final fresh-model learning curves for {n} stations sharing one configuration and seed {protocol["seed"]}. Lines show the median and shading the interquartile range across stations. Q scaling uses each station\'s training period only.',
         ['The band is descriptive between-station spread, not a confidence interval.','Validation curves condition on the preceding global parameter search.'])
    summary[['model','basins','nse_mean','nse_sd','nse_median','kge_mean','kge_sd','kge_median']].to_latex(out/'table01_summary.tex',index=False,
                float_format='%.3f',caption=f'Descriptive mean and sample SD across {n} stations, one seed each.',label='tab:global-lstm',position='tb')
    if reference_ids:
        matched[['station','nse','kge','persistence_nse','ar_nse']].to_latex(out/'table02_steh_subset.tex',index=False,float_format='%.3f',
                 caption='The prespecified 12-station STEH comparison subset; shared-parameter LSTM, seed 42.',label='tab:lstm-steh-subset',position='tb')
    traces.append({'artifact_id':'tables','source_data':[{'path':f,'sha256':sha(out/f)} for f in ['model_summary.csv','per_basin.csv','global_candidates.csv']],
                   'transformation':{'script':'experiments/report_global_lstm.py','sha256':sha(Path(__file__))},
                   'caption_claim':'Complete station results and validation-only global parameter selection.',
                   'supported_manuscript_claims':[],'limitations':['Raw cfs errors combine stations with different discharge scales.']})
    dump(out/'figure_table_trace.json',traces)
    for f in [src/'tuning_protocol.json',src/'selected_hyperparameters.json',src/'completion.json']:trace_file(f)
    dump(out/'source_manifest.json',{'source_root':str(src.resolve()),'files':sources})
    audit={'status':'PASS','stations':n,'final_models':n,'seed':protocol['seed'],'candidate_configurations':len(candidate),
           'candidate_station_fits':len(tuning),'shared_hyperparameters':hp,'winning_trial':winner,
           'mean_validation_nse':scores[winner],'negative_test_nse_stations':int((lstm.nse<0).sum()),
           'wins_vs_persistence':win_p,'wins_vs_ar':win_ar,'metrics_independently_recomputed':True,
           'dates_paired':True,'test_used_for_selection':False,'models':summaries,'timing':json.loads((src/'timing.json').read_text())}
    dump(out/'audit.json',audit)
    dump(out/'workbook_data.json',{'summary':summary.to_dict('records'),'basins':basins.to_dict('records'),
                                  'runs':lstm.to_dict('records'),'trials':tuning.to_dict('records'),
                                  'candidates':candidate.to_dict('records'),'hyperparameters':hp,'seed':protocol['seed'],
                                  'comparison':matched.to_dict('records') if reference_ids else basins.head(12).to_dict('records')})
    lines=['# 共享参数 LSTM 基线实验','',
           f"已完成 {n} 个站点各一个独立模型，全部使用同一套超参数、单一种子 {protocol['seed']} 和 15 天输入窗口。各站点从头重新初始化训练，不复用贝叶斯搜索阶段模型。",'',
           f"贝叶斯优化评估了 {len(candidate)} 套共享配置。固定调参站点由 521 个合格站点以采样种子 {protocol['sampling_seed']} 随机抽取：{', '.join(protocol['tuning_stations'])}。目标为这 {len(protocol['tuning_stations'])} 站点的平均验证 NSE，每个站点等权；测试集不参与任何选参。",'',
           f"最佳候选索引（从 0 开始）：{winner}；平均验证 NSE：{scores[winner]:.6f}。参数：`{json.dumps(hp)}`。这是给定搜索预算内的验证集最优结果，不能保证全局最优。",'',
           f"训练和选参总耗时 {audit['timing']['total_seconds']/60:.1f} 分钟。所有候选每站点训练 30 轮，按最低验证 MSE 保存轮次；最终模型亦训练 30 轮。数据顺序划分 70%/15%/15%，标准化仅拟合训练期；批量大小 128，AdamW、梯度裁剪 1，FP32 和确定性算法。LSTM 的架构、学习率、层间 dropout、weight decay 参与搜索，数据协议与 STEH 一致。",'',
           '| 模型 | NSE 均值 ± SD | NSE 中位数 | KGE 均值 ± SD | KGE 中位数 | 站点数 |','|---|---:|---:|---:|---:|---:|']
    for r in summaries:lines.append(f"| {r['model']} | {r['nse_mean']:.3f} ± {r['nse_sd']:.3f} | {r['nse_median']:.3f} | {r['kge_mean']:.3f} ± {r['kge_sd']:.3f} | {r['kge_median']:.3f} | {n} |")
    lines += ['',f"LSTM 在 {win_p}/{n} 个站点的 NSE 高于持续性基线，在 {win_ar}/{n} 个站点高于 AR；负 NSE 站点 {audit['negative_test_nse_stations']} 个，全部保留。SD 是站点间散布，不是多种子误差或置信区间。",'',
              '全部指标已从原始逐日 CSV 独立重算，并核对候选的完整调参站点面板、共享参数、种子和预测日期。RMSE、MAE、bias 单位为 cfs（ft³/s），bias 为预测减观测。', '']
    for t in traces[:3]:lines += [f"![{t['artifact_id']}]({t['artifact_id']}.png)",'',t['caption_claim'],'']
    lines += ['提供的 CSV 可能包含此前填补值，未提供逐日原始观测标志，不能称为仅原始观测评估。基线 AR 和持续性仅使用 Q 历史，LSTM 还使用气象强迫。一个种子不估计初始化敏感性；当前不作显著性或全总体优越性结论。', '',
              '原始数据位于仓库 data/Forcing 和 data/Streamflow；raw/ 保存全部候选的验证预测、训练曲线、模型配置和最终逐日预测。最终模型检查点单独归档；SHA256 和来源索引随结果保存。', '',
              '复现：`python experiments/global_lstm.py --scope all --workers 14`；绘图：`python experiments/report_global_lstm.py --input experiments/lstm_global_20261012/raw --output experiments/lstm_global_20261012/report --reference experiments/lstm_global_20261012/reference_manifest.json`。', '',
              '按照 [academic-research-skills visualization_agent](https://github.com/Imbad0202/academic-research-skills/blob/main/academic-paper/agents/visualization_agent.md) 生成色盲友好配色、600 dpi PNG 与矢量 PDF/SVG，并记录图表六项来源追踪；skill 版本 d1e51716e53fc371727ea6164dd709fc8ea28c97。', '',
              'TPE 搜索参考 [Optuna 4.9.0 官方文档](https://optuna.readthedocs.io/en/v4.9.0/reference/samplers/generated/optuna.samplers.TPESampler.html)。']
    lines += ['', '当目标方差或相关性无法定义时，NSE/KGE 按未定义保留为空值；汇总均值和 CDF 仅使用对应指标有定义的站点，model_summary.csv 记录其数量。不将未定义值替换成 0，也不删除对应站点记录。']
    (out/'实验报告.md').write_text('\n'.join(lines),encoding='utf-8')
    (out/'figures.tex').write_text('\n'.join('\\begin{figure}[tb]\n\\centering\n\\includegraphics[width=\\textwidth]{'+t['artifact_id']+'.pdf}\n\\caption{'+t['caption_claim']+'}\n\\end{figure}' for t in traces[:3]),encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
