"""Audit the frozen STEH population experiment and produce claim-linked figures.

This script reads records only. It never trains models or selects circuits.
"""
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
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from revision.common import clean, digest, metrics, finite_population_interval
from experiments.stability_numerics import linear_cka_fp64,correlation_fp64

METRICS = ['nse', 'kge', 'rmse', 'mae', 'bias']
COLORS = ['#0077BB', '#EE7733', '#009988', '#CC3311']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2,
                               allow_nan=False), encoding='utf-8')


def folder(root, job):
    return root/job['station']/f"seed_{job['seed']}"/'L15'/(
        'steh_with_q' if job['with_q'] else 'steh_no_q')


def same(actual, expected):
    if expected is None:
        assert actual is None
    else:
        np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-5)


def describe(frame, column):
    s = pd.to_numeric(frame[column], errors='coerce')
    assert np.isfinite(s.dropna()).all()
    return {'defined': int(s.notna().sum()), 'mean': float(s.mean()),
            'median': float(s.median()), 'sd_between_basins': float(s.std()),
            'q25': float(s.quantile(.25)), 'q75': float(s.quantile(.75)),
            'min': float(s.min()), 'max': float(s.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['training', 'circuits', 'lstm', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=ROOT/'data')
    args = parser.parse_args()
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    plan = read(args.training/'manifest.json')
    completion = read(args.training/'training_completion.json')
    circuit_completion = read(args.circuits/'completion.json')
    protocol = read(args.circuits/'protocol.json')
    assert completion['status'] == circuit_completion['status'] == 'PASS'
    assert completion['completed'] == circuit_completion['completed'] == len(plan['jobs'])
    assert protocol['script_sha256'] == plan['refinement_script_sha256']
    assert protocol['source_manifest_sha256'] == digest(plan)
    lstm_protocol = read(args.lstm/'tuning_protocol.json')
    assert set(plan['population']) == set(lstm_protocol['final_stations'])
    for relative, expected in plan['data_sha256'].items():
        assert sha(args.data_root/relative) == expected, relative
    assert plan['primary_seed'] == 42
    records, curves, diagnostics, sources, probe_records = [], [], [], [], []

    def source(path, kind):
        sources.append({'root': kind, 'path': path.relative_to(getattr(args, kind)).as_posix(),
                        'sha256': sha(path)})

    for job in plan['jobs']:
        f, c = folder(args.training, job), folder(args.circuits, job)
        trained, result = read(f/'summary.json'), read(c/'result.json')
        assert trained['job'] == job and result['status'] == 'complete'
        assert trained['run_sha256'] == digest({'job': job, 'manifest': plan})
        assert result['identity_sha256'] == digest({'job': job, 'protocol': protocol})
        assert trained['selected_hyperparameters'] == plan['selected_hyperparameters']
        meta = read(f/'preprocessing.json')
        assert meta['source_sha256'] == lstm_protocol['basin_source_hashes'][job['station']]
        assert ('target_q' in meta['feature_names']) == job['with_q']
        assert read(f/'validation_selection.json')['test_used_for_selection'] is False
        frozen = read(f/'validation_selection.json')
        reproduced = read(c/'validation_selection.json')
        assert frozen['core_units'] == reproduced['core_units']
        assert frozen['validation_dates_sha256'] == reproduced['validation_dates_sha256']
        assert frozen['method_script_sha256'] == protocol['script_sha256']
        prediction = pd.read_csv(f/'predictions.csv')
        interventions = pd.read_csv(c/'predictions.csv')
        assert np.isfinite(interventions[['observed','full','core','removed']]).all().all()
        assert not prediction.target_date.duplicated().any()
        assert prediction.target_date.tolist() == interventions.target_date.tolist()
        np.testing.assert_allclose(prediction.observed, interventions.observed)
        normalized_full=(interventions.full.to_numpy()-meta['q_mean'])/meta['q_std']
        normalized_trained=(prediction.predicted.to_numpy()-meta['q_mean'])/meta['q_std']
        normalized_parity_max_abs=float(abs(normalized_trained-normalized_full).max())
        # Native STE arithmetic can differ from exact binary gates in FP32;
        # Top-K boundaries can amplify this. Audit both metric sets separately,
        # disclose pointwise differences rather than silently widening tolerance.
        parity_exceedances=int((abs(normalized_trained-normalized_full)>
                                3e-6+3e-5*abs(normalized_full)).sum())
        assert digest(prediction.target_date.tolist()) == trained['test_dates_sha256'] == result['test_dates_sha256']
        for variant in ['full', 'core', 'removed']:
            recomputed = metrics(interventions[variant], interventions.observed)
            for name in METRICS:
                same(recomputed[name], result['metrics'][variant][name])
                if variant == 'full':same(metrics(prediction.predicted,prediction.observed)[name],trained['metrics'][name])
        same(result['metrics']['full']['nse'],trained['metrics']['nse'])
        controls = read(c/'random_controls.json')
        assert len(controls) == protocol['random_controls']
        distinct = [r for r in controls if not r['identical_core'] and r['nse'] is not None]
        assert len(distinct) == result['distinct_control_count']
        win = float(np.mean([result['metrics']['core']['nse'] > r['nse'] for r in distinct])) if distinct else None
        same(win, result['random_win_fraction'])
        full, core, removed = [result['metrics'][v]['nse'] for v in ['full', 'core', 'removed']]
        same(full-core, result['core_delta_nse']); same(full-removed, result['remove_delta_nse'])
        criterion = bool(full-core <= .05 and full-removed >= .02 and win is not None and win >= .8)
        assert criterion == result['pass']
        log = pd.read_csv(f/'training_log.csv')
        assert log.groupby('phase').size().to_dict() == {'dense':10, 'sparse':10, 'mask':10}
        assert log.groupby('phase').selected_phase_checkpoint.sum().eq(1).all()
        for phase, group in log.groupby('phase'):
            best = group[group.selected_phase_checkpoint.astype(bool)].iloc[0]
            same(best.val_mse, group.val_mse.min())
        mask_log = log[log.phase == 'mask']; tail = mask_log.tail(3)
        selected_mask=mask_log[mask_log.selected_phase_checkpoint.astype(bool)].iloc[0]
        diagnostics.append(job|{
            'mask_tail_gate_churn':tail.gate_epoch_churn.mean(),
            'mask_tail_weight_churn':tail.weight_epoch_churn.mean(),
            'mask_final_near_threshold_fraction':mask_log.iloc[-1].gate_near_threshold_fraction,
            'mask_selected_near_threshold_fraction':selected_mask.gate_near_threshold_fraction,
            'mask_selected_weight_nonzero_fraction':selected_mask.weight_nonzero_fraction,
            'mask_selected_epoch':int(mask_log[mask_log.selected_phase_checkpoint.astype(bool)].epoch.iloc[0]),
            'mask_best_validation_mse':mask_log.val_mse.min(),
            'mask_final_validation_mse':mask_log.iloc[-1].val_mse,
            'training_seconds':log.seconds.sum(),
            'job_seconds':read(f/'performance.json')['total_seconds']})
        curves.append(log.assign(station=job['station'], seed=job['seed'], with_q=job['with_q']))
        records.append(job|{name:trained['metrics'][name] for name in METRICS}|
                       {'test_days':len(prediction), 'core_nse':core, 'removed_nse':removed,
                        'core_delta_nse':full-core, 'remove_delta_nse':full-removed,
                        'random_win_fraction':win, 'distinct_control_count':len(distinct),
                        'criterion_pass':criterion,
                        'negative_prediction_days':int((prediction.predicted<0).sum()),
                        'full_prediction_parity_max_abs_train_scaled':normalized_parity_max_abs,
                        'full_prediction_parity_exceeding_standard_tolerance_days':parity_exceedances,
                        'binary_full_nse':result['metrics']['full']['nse'],
                        'binary_minus_native_full_nse':result['metrics']['full']['nse']-trained['metrics']['nse'],
                        'test_observed_mean_cfs':float(prediction.observed.mean()),
                        'test_observed_std_cfs':float(prediction.observed.std(ddof=0)),
                        'train_observed_mean_cfs':meta['q_mean'],
                        'train_observed_std_cfs':meta['q_std'],
                        **result['validation_search']})
        probe=read(f/'probes.json')
        assert probe['status']=='computed' and len(probe['results'])==6
        assert {(r['target'],r['representation']) for r in probe['results']}=={
            (t,k) for t in ['CUM','API','CHG'] for k in ['trained_hidden','input_history']}
        for row in probe['results']:
            assert row['r2'] is None or np.isfinite(row['r2'])
            if row['status']=='failed':assert 'reason' in row and row['r2'] is None
            else:
                assert row['n_train']==min(plan['config']['probe_train_samples'],log.training_samples.iloc[0])
                assert row['n_test']==min(plan['config']['analysis_test_samples'],len(prediction))
            probe_records.append(job|row)
        if job['seed'] == plan['primary_seed']:
            lf = args.lstm/'final'/'models'/job['station']
            lp = pd.read_csv(lf/'predictions.csv')
            assert prediction.target_date.tolist() == lp.target_date.tolist()
            np.testing.assert_allclose(prediction.observed, lp.observed)
            for name in ['summary.json', 'predictions.csv']: source(lf/name, 'lstm')
        for name in ['summary.json','predictions.csv','training_log.csv','performance.json',
                     'run_config.json','preprocessing.json','validation_selection.json','probes.json']:
            source(f/name, 'training')
        for name in ['result.json','predictions.csv','random_controls.json',
                     'validation_selection.json','feature_effects.json','validation_hidden.npy']:
            source(c/name, 'circuits')

    all_runs = pd.DataFrame(records); diag = pd.DataFrame(diagnostics)
    primary = all_runs[all_runs.seed == 42].copy()
    assert len(primary) == 2*len(plan['population'])
    assert not primary.duplicated(['station','with_q']).any()
    paired = primary.pivot(index='station', columns='with_q', values='nse')
    paired.columns = ['steh_no_q_nse', 'steh_with_q_nse']
    kge = primary.pivot(index='station', columns='with_q', values='kge')
    paired['steh_with_q_kge'] = kge[True]
    paired['steh_no_q_kge'] = kge[False]
    for station in paired.index:
        lf = args.lstm/'final'/'models'/station
        p = pd.read_csv(lf/'predictions.csv')
        lm = metrics(p.predicted,p.observed)
        paired.loc[station, 'lstm_nse'] = lm['nse']
        paired.loc[station, 'lstm_kge'] = lm['kge']
        reference=pd.read_csv(lf/'baseline_predictions.csv')
        assert reference.target_date.tolist()==p.target_date.tolist()
        np.testing.assert_allclose(reference.observed,p.observed)
        saved_reference=read(lf/'baselines.json')['metrics']
        for name in ['persistence','ar']:
            score=metrics(reference[name],reference.observed)
            for key in METRICS:same(score[key],saved_reference[name][key])
            paired.loc[station,name+'_nse']=score['nse']
            paired.loc[station,name+'_kge']=score['kge']
        for name in ['baseline_predictions.csv','baselines.json']:source(lf/name,'lstm')
    paired['with_q_minus_lstm'] = paired.steh_with_q_nse-paired.lstm_nse
    paired['with_q_minus_no_q'] = paired.steh_with_q_nse-paired.steh_no_q_nse
    paired = paired.reset_index()
    stability = pd.read_csv(args.circuits/'functional_stability.csv', dtype={'station':str})
    expected_pairs = 2*len(plan['confirmation_stations'])*3
    assert len(stability) == expected_pairs
    assert set(stability.station) == set(plan['confirmation_stations'])
    stability_rows=[];pair_keys=set()
    for row in stability.to_dict('records'):
        key=(row['station'],row['with_q'],tuple(sorted([int(row['seed_a']),int(row['seed_b'])])))
        assert key not in pair_keys;pair_keys.add(key)
        assert set(key[2])<=set([42,43,44]) and len(set(key[2]))==2
        folders=[folder(args.circuits,{'station':row['station'],'with_q':row['with_q'],'seed':int(row[s])})
                 for s in ['seed_a','seed_b']]
        selections=[read(f/'validation_selection.json') for f in folders]
        assert selections[0]['validation_dates_sha256']==selections[1]['validation_dates_sha256']
        a,b=[np.load(f/'validation_hidden.npy',allow_pickle=False) for f in folders]
        assert a.shape==b.shape==(protocol['validation_days'],plan['config']['d_model'])
        x,y=a-a.mean(axis=0),b-b.mean(axis=0)
        denominator=np.linalg.norm(x.T@x,'fro')*np.linalg.norm(y.T@y,'fro')
        fp32=float(np.linalg.norm(x.T@y,'fro')**2/denominator) if denominator else None
        recorded=row['validation_hidden_linear_cka']
        if pd.isna(recorded):assert fp32 is None
        else:same(fp32,recorded)
        effects=[pd.DataFrame(read(f/'feature_effects.json')).set_index('feature') for f in folders]
        rank_a=effects[0].validation_delta_nse.rank()
        rank_b=effects[1].loc[rank_a.index].validation_delta_nse.rank()
        rank=correlation_fp64(rank_a,rank_b)
        if pd.isna(row['validation_input_effect_spearman']):assert rank is None
        else:same(rank,row['validation_input_effect_spearman'])
        pa,pb=[pd.read_csv(f/'predictions.csv') for f in folders]
        assert pa.target_date.tolist()==pb.target_date.tolist()
        stability_rows.append(row|{'validation_hidden_linear_cka_fp64':linear_cka_fp64(a,b),
                                  'full_prediction_correlation':correlation_fp64(pa.full,pb.full),
                                  'core_prediction_correlation':correlation_fp64(pa.core,pb.core)})
    stability=pd.DataFrame(stability_rows)
    stability_basin = stability.groupby(['station','with_q'],as_index=False).agg(
        input_rank_mean=('validation_input_effect_spearman','mean'),
        input_rank_min=('validation_input_effect_spearman','min'),
        hidden_cka_mean=('validation_hidden_linear_cka_fp64','mean'),
        hidden_cka_min=('validation_hidden_linear_cka_fp64','min'),
        hidden_cka_original_mean=('validation_hidden_linear_cka','mean'),
        full_prediction_correlation_mean=('full_prediction_correlation','mean'),
        core_prediction_correlation_mean=('core_prediction_correlation','mean'))
    confirm = all_runs[all_runs.station.isin(plan['confirmation_stations'])].copy()
    assert len(confirm) == 6*len(plan['confirmation_stations'])
    confirm_basin = confirm.groupby(['station','with_q'],as_index=False).agg(
        all_three_seeds_pass=('criterion_pass','all'), seeds_passing=('criterion_pass','sum'),
        nse_mean=('nse','mean'), nse_sd_fixed_seeds=('nse','std'),
        core_loss_max=('core_delta_nse','max'))
    confirmation = []
    for q in [True,False]:
        frame = confirm_basin[confirm_basin.with_q == q]; k = int(frame.all_three_seeds_pass.sum())
        confirmation.append({'with_q':q, 'basins':len(frame), 'all_three_seeds_pass':k,
                             'finite_population_interval':finite_population_interval(k,len(frame),len(plan['confirmation_population'])),
                             'population':len(plan['confirmation_population']),
                             'outcome':'all three prespecified seed circuits satisfy fixed criterion'})
    summary = []
    for name in ['steh_with_q_nse','steh_no_q_nse','lstm_nse','with_q_minus_lstm','with_q_minus_no_q',
                 'steh_with_q_kge','steh_no_q_kge','lstm_kge',
                 'persistence_nse','ar_nse','persistence_kge','ar_kge']:
        summary.append({'measure':name, **describe(paired,name)})
    summary.extend({'measure':f"core_{q}_loss", **describe(primary[primary.with_q == q],'core_delta_nse')}
                   for q in [True,False])
    probes=pd.DataFrame(probe_records);probe_summary=[]
    for (q,target,representation),group in probes[probes.seed==42].groupby(['with_q','target','representation']):
        probe_summary.append({'with_q':bool(q),'target':target,'representation':representation,
                              'basins':len(group),'negative_r2':int((group.r2<0).sum()),
                              'failed_or_undefined':int(group.r2.isna().sum()),**describe(group,'r2')})
    quality = {'status':'PASS', 'trained_models':len(records), 'population_basins':len(paired),
               'confirmation_basins':len(plan['confirmation_stations']),
               'primary_passes':{str(q):int(primary[primary.with_q == q].criterion_pass.sum()) for q in [True,False]},
               'primary_with_q_beats_lstm':int((paired.with_q_minus_lstm>0).sum()),
               'primary_with_q_beats_persistence':int((paired.steh_with_q_nse>paired.persistence_nse).sum()),
               'primary_with_q_beats_ar':int((paired.steh_with_q_nse>paired.ar_nse).sum()),
               'probe_summary':probe_summary,
               'probe_audit_scope':'Frozen unfiltered probe scores imported and count/status/source checked; hidden/sample forecasts not saved, so R2 was not independently recomputed.',
               'stability_numeric_audit':{
                   'pairs_checked':len(stability),'robust_cka_undefined':int(stability.validation_hidden_linear_cka_fp64.isna().sum()),
                   'max_fp32_fp64_cka_difference':float((stability.validation_hidden_linear_cka_fp64-stability.validation_hidden_linear_cka).abs().max()),
                   'numerical_helper_sha256':sha(ROOT/'experiments/stability_numerics.py'),
                   'method':'Recompute original FP32 CKA and input ranks; center FP64 after first-row translation for robust CKA; constant representations remain undefined.'},
               'native_binary_forward_difference':{
                   'models_exceeding_pointwise_tolerance':int((all_runs.full_prediction_parity_exceeding_standard_tolerance_days>0).sum()),
                   'maximum_train_scaled_prediction_difference':float(all_runs.full_prediction_parity_max_abs_train_scaled.max()),
                   'maximum_absolute_nse_difference':float(all_runs.binary_minus_native_full_nse.abs().max()),
                   'pointwise_reference_tolerance':'3e-6 + 3e-5*abs(train-scaled binary prediction); differences disclosed, not removed',
                   'metric_audit':'Native and binary predictions audited against their own saved metrics; full NSE agreement checked separately.'},
               'primary_failure_components':{
                   str(q):{'core_loss_above_005':int((primary[primary.with_q==q].core_delta_nse>.05).sum()),
                           'removal_loss_below_002':int((primary[primary.with_q==q].remove_delta_nse<.02).sum()),
                           'win_below_08':int((primary[primary.with_q==q].random_win_fraction<.8).sum()),
                           'undefined_control_win':int(primary[primary.with_q==q].random_win_fraction.isna().sum())}
                   for q in [True,False]},
               'summary':summary, 'confirmation':confirmation,
               'scope':'Census descriptions of these 521 basins; no sampling confidence interval for the census. '
                       'SRS confirmation interval targets the 503 eligible basins and three fixed seeds only.',
               'limits':['Computational integrity PASS does not establish scientific validity.',
                         'LSTM has one seed and receives Q history; no-Q STEH is an input ablation.',
                         'Original-observation flags are unavailable; supplied CSV targets may contain imputation.',
                         'Core retains embedding, residuals, biases, input gates and readout; not an isolated closed graph.',
                         'Control draws exclude the core itself but may repeat each other; neither overlap nor win fraction establishes a significance level.',
                         'Individual control forecasts were not stored; audit reconstructs wins from saved control NSE scores.',
                         'Near-threshold gates and churn describe finite-epoch sensitivity; not proof of convergence.']}
    quality['limits'].extend([
        'Persistence and AR use Q history and are references for with-Q models; not input-matched competitors for no-Q.',
        'Per-basin single-seed LSTM is the requested matched local baseline; this comparison does not establish superiority over pooled regional or ensemble LSTMs.',
        'KGE=0 is not an inherent skill threshold; interpret KGE with explicit reference models.'])
    quality['limits'].append('Primary forecasts use native STE gates; interventions use exact binary overrides. FP32 gate arithmetic and Top-K can produce small differences; all are disclosed separately.')
    quality['limits'].append('CUM/API/CHG are input-derived rainfall proxies. Probe R2 measures decodability; not evidence of independent physical states or causality. Negative values, undefined scores and failures retained.')
    tables = {'all_runs':all_runs,'paired_primary':paired,'convergence_diagnostics':diag,
              'phase_training_curves':pd.concat(curves,ignore_index=True),
              'stability_basin':stability_basin,'confirmation_basin':confirm_basin,
              'functional_stability_audited':stability,
              'model_summary':pd.DataFrame(summary),'probe_records':probes,
              'probe_summary':pd.DataFrame(probe_summary)}
    associations=[]
    for q in [True,False]:
        group=primary[primary.with_q==q]
        for units in ['active_units','core_units']:
            associations.append({'with_q':q,'unit_measure':units,'basins':len(group),
                'spearman_with_nse':group[units].corr(group.nse,method='spearman'),
                'scope':'Descriptive census association; validation-based selection can mechanically couple core size and skill.'})
    tables['node_count_associations']=pd.DataFrame(associations)
    model_table=[]
    for label,prefix in [('STEH with Q','steh_with_q'),('STEH no Q','steh_no_q'),
                         ('LSTM','lstm'),('Persistence','persistence'),('AR','ar')]:
        for metric in ['nse','kge']:
            values=paired[prefix+'_'+metric]
            model_table.append({'Model':label,'Metric':metric.upper(),'Defined':int(values.notna().sum()),
                                'Mean':values.mean(),'SD':values.std(),'Median':values.median()})
    tables['publication_model_table']=pd.DataFrame(model_table)
    circuit_table=[]
    for q in [True,False]:
        group=primary[primary.with_q==q]
        circuit_table.append({'Input':'With Q' if q else 'No Q','Basins':len(group),
                              'Criterion passes':int(group.criterion_pass.sum()),
                              'Median core loss':group.core_delta_nse.median(),
                              'Median removal loss':group.remove_delta_nse.median(),
                              'Undefined control win':int(group.random_win_fraction.isna().sum())})
    tables['publication_circuit_table']=pd.DataFrame(circuit_table)
    for name,frame in tables.items():frame.to_csv(out/(name+'.csv'),index=False)
    for name in ['publication_model_table','publication_circuit_table']:
        tables[name].to_latex(out/(name+'.tex'),index=False,float_format=lambda value:f'{value:.3f}',
                             escape=True,caption=('Descriptive results under the frozen protocol. All basins retained; '
                                                  'SD is between basins. Q-history references apply to with-Q forecasts.'
                                                  if name=='publication_model_table' else
                                                  'Original nearest-degree control criterion. Draws excluding the core may repeat.'),
                             label='tab:'+name)
    dump(out/'audit.json',quality);dump(out/'source_manifest.json',sources)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.labelsize':9,
                         'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'svg.fonttype':'none'})
    traces = []

    def plain_ticks(ax, small=False):
        lo,hi=ax.get_xlim()
        ticks=[-1000000,-100000,-10000,-1000,-100,-10,-1,0,1,10,100,1000,10000,100000,1000000]
        if small:ticks=[-1000000,-10000,-100,-1,0,1,100,10000,1000000]
        ticks=sorted(t for t in ticks if lo<=t<=hi)
        ax.set_xticks(ticks,[f'{t:g}' for t in ticks])

    def plain_y_ticks(ax):
        lo,hi=ax.get_ylim()
        ticks=[-1000000,-100000,-10000,-1000,-100,-10,-1,0,1,10,100,1000,10000,100000,1000000]
        ticks=[t for t in ticks if lo<=t<=hi]
        ax.set_yticks(ticks,[f'{t:g}' for t in ticks])

    def save(fig, ident, files, claim, limits):
        fig.tight_layout(pad=.9)
        for ext in ['png','pdf','svg']:fig.savefig(out/(ident+'.'+ext),dpi=600)
        plt.close(fig)
        traces.append({'artifact_id':ident,'source_data':[{'path':name,'sha256':sha(out/name)} for name in files],
                       'transformation':{'script':'experiments/report_steh_population.py','sha256':sha(Path(__file__)),
                                         'numerical_helper_sha256':sha(ROOT/'experiments/stability_numerics.py')},
                       'caption_claim':claim,'supported_manuscript_claims':[claim], 'limitations':limits})

    fig,axes = plt.subplots(1,2,figsize=(175/25.4,73/25.4))
    for index,(col,label) in enumerate([('steh_with_q_nse','STEH with Q'),('lstm_nse','LSTM'),('steh_no_q_nse','STEH no Q')]):
        v=np.sort(paired[col].dropna());axes[0].plot(v,np.arange(1,len(v)+1)/len(v),
                    color=COLORS[index],linestyle=['-','--',':'][index],label=label)
    axes[0].set_xscale('asinh',linear_width=1);axes[0].set_xlabel('Test NSE (asinh scale)');axes[0].set_ylabel('Fraction of basins');axes[0].legend(loc='upper left')
    plain_ticks(axes[0])
    d=np.sort(paired.with_q_minus_lstm.dropna());axes[1].plot(d,np.arange(1,len(d)+1)/len(d),color=COLORS[0]);axes[1].axvline(0,color='#666666',ls='--',lw=.8)
    axes[1].set_xscale('asinh',linear_width=.1);axes[1].set_xlabel('Paired NSE: STEH with Q - LSTM');axes[1].set_ylabel('Fraction of basins')
    plain_ticks(axes[1],True)
    save(fig,'fig1_paired_prediction',['paired_primary.csv'],
         'Complete basin distributions and within-basin test differences under the frozen protocol.',quality['limits'][:3]+[quality['limits'][8]])
    fig,axes=plt.subplots(1,3,figsize=(175/25.4,66/25.4))
    for q,color,style in [(True,COLORS[0],'-'),(False,COLORS[1],'--')]:
        frame=primary[primary.with_q==q];label='With Q' if q else 'No Q'
        for ax,col in zip(axes,['core_delta_nse','remove_delta_nse','random_win_fraction']):
            v=np.sort(frame[col].dropna());ax.plot(v,np.arange(1,len(v)+1)/len(v),color=color,ls=style,label=label)
    for ax,threshold,label in zip(axes,[.05,.02,.8],['Full - core NSE','Full - removed NSE','Win vs nonidentical draws']):
        ax.axvline(threshold,color='#666666',ls=':',lw=.8);ax.set_xlabel(label);ax.set_ylabel('Fraction of defined basins')
    for ax in axes[:2]:
        ax.set_xscale('asinh',linear_width=.1);plain_ticks(ax,True)
    axes[0].legend(loc='best')
    save(fig,'fig2_circuit_validity',['all_runs.csv'],'Separate sufficiency, necessity and structural-control evidence; failures retained.',quality['limits'][3:5])
    fig,axes=plt.subplots(1,2,figsize=(175/25.4,65/25.4))
    for q,color,marker in [(True,COLORS[0],'o'),(False,COLORS[1],'^')]:
        f=stability_basin[stability_basin.with_q==q];axes[0].scatter(f.input_rank_mean,f.hidden_cka_mean,
                color=color,marker=marker,s=18,label='With Q' if q else 'No Q')
        f=diag[(diag.seed==42)&(diag.with_q==q)];v=np.sort(f.mask_tail_gate_churn.dropna())
        axes[1].plot(v,np.arange(1,len(v)+1)/len(v),color=color,ls='-' if q else '--')
    axes[0].set_xlabel('Mean input-effect rank correlation');axes[0].set_ylabel('Mean hidden linear CKA');axes[0].legend(loc='best')
    axes[1].set_xlabel('Gate churn: final three mask epochs');axes[1].set_ylabel('Fraction of primary basins')
    save(fig,'fig3_functional_stability',['stability_basin.csv','functional_stability_audited.csv','convergence_diagnostics.csv'],
         'Initialization sensitivity is assessed within basins using function-level evidence; epoch churn reports finite-budget stability.',
         ['12 prespecified new basins, three fixed seeds; sample size is 12, not 36.',quality['limits'][6],
          'CKA centered in FP64 after first-row translation; undefined constant representations are excluded from plotting and counted explicitly.'])
    training=tables['phase_training_curves'];training=training[training.seed==42]
    fig,axes=plt.subplots(1,3,figsize=(175/25.4,66/25.4))
    for ax,phase,title in zip(axes,['dense','sparse','mask'],['I: dense','II: dynamic sparsity','III: learned gates']):
        for q,color in [(True,COLORS[0]),(False,COLORS[1])]:
            group=training[(training.with_q==q)&(training.phase==phase)]
            for name,style in [('train_mse','-'),('val_mse','--')]:
                values=group.groupby('epoch')[name];median=values.median();low=values.quantile(.25);high=values.quantile(.75)
                ax.plot(median.index,median,color=color,ls=style,
                        label=f"{'With Q' if q else 'No Q'} {'train' if name=='train_mse' else 'val'}")
                ax.fill_between(median.index,low,high,color=color,alpha=.08)
        ax.set_title(title);ax.set_xlabel('Epoch within phase');ax.set_ylabel('Train-scaled target MSE')
        ax.set_xticks([1,5,10])
    axes[0].legend(loc='best',fontsize=8)
    save(fig,'fig5_phase_training',['phase_training_curves.csv'],
         'Median and interquartile training/validation loss across 521 primary basins under the fixed three-phase budget.',
         ['Bands are between-basin IQR, not confidence intervals.',
          'Stage-II training uses dynamic sparsity; validation evaluates the final sparsity budget.',
          'Finite loss curves do not prove structural convergence.'])
    median=float(paired.steh_with_q_nse.median())
    ranked=paired.assign(distance_to_median=(paired.steh_with_q_nse-median).abs()).sort_values(['distance_to_median','station'])
    representative=str(ranked.iloc[0].station)
    illustrated=[representative,'08190500'];hydrographs=[]
    fig,axes=plt.subplots(2,1,figsize=(175/25.4,100/25.4))
    for ax,station in zip(axes,illustrated):
        job={'station':station,'seed':42,'with_q':True};p=pd.read_csv(folder(args.training,job)/'predictions.csv')
        l=pd.read_csv(args.lstm/'final'/'models'/station/'predictions.csv')
        no=pd.read_csv(folder(args.training,job|{'with_q':False})/'predictions.csv')
        assert p.target_date.tolist()==l.target_date.tolist()==no.target_date.tolist()
        table=pd.DataFrame({'station':station,'target_date':p.target_date,'observed':p.observed,
                            'steh_with_q':p.predicted,'lstm':l.predicted,'steh_no_q':no.predicted})
        hydrographs.append(table);dates=pd.to_datetime(p.target_date)
        for col,color,style,label in [('observed','#222222','-','Observed'),('steh_with_q',COLORS[0],'-','STEH with Q'),
                                     ('lstm',COLORS[1],'--','LSTM'),('steh_no_q',COLORS[2],':','STEH no Q')]:
            ax.plot(dates,table[col],color=color,ls=style,lw=.65,label=label)
        ax.set_yscale('asinh',linear_width=1);plain_y_ticks(ax);ax.set_ylabel('Discharge (cfs; asinh scale)')
        ax.set_title(f"{station}: {'closest to median primary STEH NSE' if station==representative else 'previously identified LSTM extreme'}")
    axes[0].legend(loc='upper right',ncols=4,fontsize=8);axes[1].set_xlabel('Test target date')
    pd.concat(hydrographs,ignore_index=True).to_csv(out/'illustrative_hydrographs.csv',index=False)
    save(fig,'fig6_illustrative_hydrographs',['illustrative_hydrographs.csv','paired_primary.csv'],
         'Full test hydrographs at a disclosed median-performance example and the retained LSTM extreme case.',
         ['Illustrations are descriptive, not a random sample or evidence of universal superiority.',
          'Median example selected after performance aggregation; tie broken by basin ID.',
          'Station08190500 was identified before final STEH analysis; no test-driven retraining or clipping.'])
    dump(out/'illustration_selection.json',{'representative_basin':representative,
         'rule':'closest to median primary with-Q STEH NSE; lexicographic basin-ID tie break',
         'preidentified_lstm_extreme':'08190500','selection_role':'descriptive illustrations only'})
    fig,axes=plt.subplots(1,3,figsize=(175/25.4,68/25.4))
    for ax,target in zip(axes,['CUM','API','CHG']):
        for q,color in [(True,COLORS[0]),(False,COLORS[1])]:
            for representation,style,label in [('trained_hidden','-','hidden'),('input_history','--','input')]:
                group=probes[(probes.seed==42)&(probes.with_q==q)&(probes.target==target)&(probes.representation==representation)]
                values=np.sort(group.r2.dropna());ax.plot(values,np.arange(1,len(values)+1)/len(values),
                    color=color,ls=style,label=f"{'With Q' if q else 'No Q'} {label}")
        ax.set_xscale('asinh',linear_width=1);plain_ticks(ax)
        ax.set_title(target);ax.set_xlabel('Test proxy R² (asinh scale)');ax.set_ylabel('Fraction of defined basins')
        ax.axvline(0,color='#666666',ls=':',lw=.8)
    axes[0].legend(loc='best',fontsize=8)
    save(fig,'fig7_proxy_decodability',['probe_records.csv','probe_summary.csv'],
         'Unfiltered input-derived proxy decodability compared between trained hidden states and input history.',
         [quality['limits'][-1],quality['probe_audit_scope'],
          'Probes fit on a fixed training subset and evaluate a fixed held-out test subset; not all daily forecasts.'])
    fig,axes=plt.subplots(1,2,figsize=(175/25.4,72/25.4))
    for ax,units,title in zip(axes,['active_units','core_units'],['Active gated units','Validation-selected core units']):
        for q,color,marker in [(True,COLORS[0],'o'),(False,COLORS[1],'^')]:
            group=primary[primary.with_q==q]
            rho=group[units].corr(group.nse,method='spearman')
            ax.scatter(group[units],group.nse,color=color,marker=marker,s=9,alpha=.32,
                       label=f"{'With Q' if q else 'No Q'}: rank r={rho:.3f}",rasterized=True)
        ax.set_xlabel(title);ax.set_ylabel('Native test NSE (asinh scale)')
        ax.set_yscale('asinh',linear_width=1)
        plain_y_ticks(ax)
        ax.legend(loc='lower left',fontsize=8)
    save(fig,'fig8_structure_performance',['all_runs.csv','node_count_associations.csv'],
         'Continuous descriptive associations of retained unit counts with skill across all 521 basins per setting.',
         ['No causal effect of compactness is established.',
          'Validation-based core selection can mechanically couple core size and performance.',
          'All negative and extreme NSE values retained; no NSE=0.5 grouping or outlier exclusion.'])
    for name in tables:
        traces.append({'artifact_id':name,'source_data':'source_manifest.json',
                       'transformation':{'script':'experiments/report_steh_population.py','sha256':sha(Path(__file__))},
                       'caption_claim':f'Audited {name} records under the frozen protocol.',
                       'supported_manuscript_claims':[], 'limitations':quality['limits']})
    dump(out/'trace.json',traces)
    dump(out/'workbook_data.json',{'quality':quality,**{name:frame.to_dict('records') for name,frame in tables.items() if name!='phase_training_curves'}})
    report = '\n'.join(['# STEH 实验核验与论文展示判断',
        f"训练和回路分析均完成 {len(records)} 个任务；全站点主要结果每站点每设置使用 seed42。",
        f"主要回路固定判据通过数：with-Q {quality['primary_passes']['True']}／{len(paired)}，no-Q {quality['primary_passes']['False']}／{len(paired)}。",
        f"STEH with-Q 的 NSE 高于同站点 LSTM：{quality['primary_with_q_beats_lstm']}／{len(paired)}。",
        f"with-Q 相对持久性／AR参考的NSE胜出站点数：{quality['primary_with_q_beats_persistence']}／{quality['primary_with_q_beats_ar']}（总计{len(paired)}）。",
        '所有极端值、负 NSE 和回路失败均保留。平均值、标准差、中位数及四分位数同时列于 model_summary.csv。',
        '全站点是当前 521 个流域的总体描述，不用流域间标准差冒充多种子不确定性。',
        '12 个确认流域在 503 个未参与开发或调参的流域中预先简单随机抽样。',
        '固定成功事件为一个流域三个预设种子都通过原定回路判据，区间不推断所有可能初始化。',
        '不同流域回路不同可对应水文异质性；同一流域的初始化差异用输入作用排序、线性 CKA 和干预保真度共同判断。',
        '不能以隐藏神经元编号重叠、少量随机对照胜率或计算核验 PASS 单独证明普遍存在唯一、闭合、因果回路。',
        '若固定判据通过率有限，应将论文结论限定为局部可提取的忠实回路，报告失败机制；不按测试结果反复调整阈值。',
        '门控近阈值比例和末期翻转比例单独报告。有限轮次的损失下降不等于结构已经收敛。',
        '跨种子输入作用排序和原始CKA独立重算；另保留双精度中心化的CKA及完整/核心预测相关性。常数表示的CKA保持未定义，不当作稳定证据。',
        '主预测采用原生直通门控，回路干预采用精确二值门控；两种前向的逐日差异和NSE差异单独核验并保留。',
        '物理探测完整保留负R²、未定义及失败记录，并以输入历史为参考。CUM/API/CHG由降雨输入构造，只能支持可解码性，不能证明独立物理状态或因果回路。',
        '探测汇总核查记录数量、状态和来源；未保存的隐藏表示及探测预测无法独立重算R²，该限制明确保留。',
        '逐流域单种子LSTM用于当前匹配的本地训练基线；不据此宣称超过跨流域联合训练或集成LSTM。',
        'KGE=0不具有NSE=0的固有基准含义；应结合明示的持久性和AR参考模型解释。',
        'KGE解释来源：https://hess.copernicus.org/articles/23/4323/2019/',
        '本地与跨流域LSTM比较来源：https://hess.copernicus.org/articles/28/4187/2024/',
        'CKA 方法来源：https://proceedings.mlr.press/v97/kornblith19a.html',
        '原始观测标记缺失，不能宣称评估目标全是未经填补的原始观测。'])
    (out/'实验核验与论文展示.md').write_text(report,encoding='utf-8')
    print(json.dumps(clean(quality),ensure_ascii=True))


if __name__ == '__main__':
    main()
