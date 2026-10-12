"""Audit every stored caliper-control forecast and report both fixed calipers."""
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
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from revision.common import metrics,digest,clean,finite_population_interval


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(clean(v),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['input','circuits','output']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();root=args.input;out=args.output;out.mkdir(parents=True,exist_ok=True)
    protocol=read(root/'protocol.json');completion=read(root/'completion.json')
    assert completion['status']=='PASS' and completion['completed']==len(protocol['jobs'])
    assert protocol['calipers']==[1.,2.] and protocol['controls_per_caliper']==20
    sources=[];rows=[]
    for job in protocol['jobs']:
        relative=Path(job['station'])/f"seed_{job['seed']}"/'L15'/('steh_with_q' if job['with_q'] else 'steh_no_q')
        f=root/relative;result=read(f/'result.json');design=read(f/'control_design.json')
        controls=read(f/'control_membership.json');prior=read(args.circuits/relative/'result.json')
        assert result['identity_sha256']==digest({'job':job,'protocol':protocol})
        assert sha(f/'control_design.json')==result['control_design_sha256']
        assert design['selection_data']=='graph degrees and RNG only; no test performance'
        assert design['core_source']==read(args.circuits/relative/'validation_selection.json')['core_units']
        with np.load(f/'control_predictions.npz',allow_pickle=False) as raw:
            dates=raw['target_date'].tolist();target=raw['observed'];predictions=raw['controls']
            full_nse=metrics(raw['full'],target)['nse'];core_nse=metrics(raw['core'],target)['nse']
            assert predictions.shape==(40,len(target)) and np.isfinite(predictions).all()
        assert digest(dates)==result['test_dates_sha256']==prior['test_dates_sha256']
        np.testing.assert_allclose(full_nse,prior['metrics']['full']['nse'],rtol=3e-6,atol=3e-5)
        np.testing.assert_allclose(core_nse,prior['metrics']['core']['nse'],rtol=3e-6,atol=3e-5)
        offset=0
        for index,bank in enumerate(controls['banks']):
            assert len(bank)==20;frozen=design['banks'][index]
            for old,new in zip(frozen,bank):
                assert old['members']==new['members'] and old['mask_sha256']==new['mask_sha256']==digest(new['members'])
                for group,core_ids in design['core_source'].items():
                    assert len(new['members'][group])==len(core_ids)
                    assert len(set(new['members'][group]))==len(core_ids)
                overlap=sum(len(set(v)&set(design['core_source'][k])) for k,v in new['members'].items())
                size=sum(len(v) for v in design['core_source'].values())
                assert new['identical_core']==(overlap==size)
                np.testing.assert_allclose(new['core_overlap_fraction'],overlap/max(1,size))
                actual=metrics(predictions[offset],target)['nse'];offset+=1
                np.testing.assert_allclose(actual,new['nse'],rtol=3e-6,atol=3e-5)
            valid=[r for r in bank if not r['identical_core'] and r['nse'] is not None]
            unique={r['mask_sha256']:r for r in valid}
            win=float(np.mean([core_nse>r['nse'] for r in valid])) if valid else None
            recorded=result['caliper_results'][index]
            assert recorded['nonidentical_draws']==len(valid) and recorded['unique_control_masks']==len(unique)
            if win is None:assert recorded['random_win_fraction'] is None
            else:np.testing.assert_allclose(win,recorded['random_win_fraction'])
            scientific=bool(prior['core_delta_nse']<=.05 and prior['remove_delta_nse']>=.02 and win is not None and win>=.8)
            assert scientific==recorded['criterion_pass']
            rows.append(job|recorded|{'nearest_degree_pass':prior['pass'],
                         'nearest_degree_win_fraction':prior['random_win_fraction'],
                         'mean_core_overlap':float(np.mean([r['core_overlap_fraction'] for r in bank])),
                         'mean_standardized_degree_distance':float(np.mean([r['mean_standardized_degree_distance'] for r in bank]))})
        for name in ['result.json','control_design.json','control_membership.json','control_predictions.npz']:
            sources.append({'path':(relative/name).as_posix(),'sha256':sha(f/name)})
    frame=pd.DataFrame(rows);assert len(frame)==2*len(protocol['jobs'])
    frame.to_csv(out/'caliper_audited_runs.csv',index=False)
    primary=frame[frame.seed==42];assert len(primary)==2084
    summary=[]
    for (q,c),g in primary.groupby(['with_q','caliper']):
        summary.append({'with_q':bool(q),'caliper':float(c),'basins':len(g),
                        'nearest_degree_passes':int(g.nearest_degree_pass.sum()),
                        'caliper_passes':int(g.criterion_pass.sum()),
                        'undefined_control_win':int(g.random_win_fraction.isna().sum()),
                        'median_unique_control_masks':float(g.unique_control_masks.median()),
                        'median_core_overlap':float(g.mean_core_overlap.median())})
    summary=pd.DataFrame(summary);summary.to_csv(out/'caliper_population_summary.csv',index=False)
    confirmed=frame[frame.station.isin(protocol['confirmation_stations'])]
    basin=confirmed.groupby(['station','with_q','caliper'],as_index=False).agg(
        all_three_seeds_pass=('criterion_pass','all'),minimum_seed_win=('random_win_fraction','min'))
    assert len(basin)==48
    basin.to_csv(out/'caliper_confirmation_basins.csv',index=False)
    intervals=[]
    for (q,c),g in basin.groupby(['with_q','caliper']):
        k=int(g.all_three_seeds_pass.sum());bounds=finite_population_interval(k,12,
            len(protocol['conditional_confirmation_population']),confidence=protocol['confidence'])
        intervals.append({'with_q':bool(q),'caliper':float(c),'sample_successes':k,'sample_basins':12,
                          'population':len(protocol['conditional_confirmation_population']),**bounds})
    assert read(root/'confirmation.json')['results']==[
        {'with_q':r['with_q'],'caliper':r['caliper'],'successes':r['sample_successes'],'basins':12,
         'interval':{k:r[k] for k in ['confidence','success_count_bounds','success_fraction_bounds']}}
        for r in intervals]
    audit={'status':'PASS','models':len(protocol['jobs']),'stored_control_forecasts':40*len(protocol['jobs']),
           'population_summary':summary.to_dict('records'),'confirmation':intervals,
           'limits':['Sensitivity analysis developed on the fixed first12 early-review basins; original protocol retained.',
                     'Both calipers reported; no selection based on test results.',
                     'Conditional SRS confirmation targets 492 eligible basins and three fixed seeds, not all initializations.',
                     'Four Bonferroni-adjusted marginal 98.75% intervals have at least 95% joint coverage.',
                     'Control membership, overlaps, duplicates and standardized degree distances are disclosed.',
                     'Matching is approximate; residual/input/readout pathways remain. No causal or unique closed-circuit proof.']}
    dump(out/'caliper-audit.json',audit);dump(out/'caliper-source-manifest.json',sources)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'legend.fontsize':8,
                         'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(175/25.4,70/25.4))
    for q,color in [(True,'#0077BB'),(False,'#EE7733')]:
        for c,style in [(1.,'-'),(2.,'--')]:
            g=primary[(primary.with_q==q)&(primary.caliper==c)];v=np.sort(g.random_win_fraction.dropna())
            axes[0].plot(v,np.arange(1,len(v)+1)/len(v),color=color,ls=style,
                         label=f"{'With Q' if q else 'No Q'}, caliper {c:g}")
    axes[0].axvline(.8,color='#666666',ls=':',lw=.8);axes[0].set_xlabel('Win fraction vs nonidentical controls');axes[0].set_ylabel('Fraction of defined primary basins');axes[0].legend(loc='upper left')
    for i,r in enumerate(intervals):
        point=r['sample_successes']/12;lower,upper=r['success_fraction_bounds']
        axes[1].errorbar(point,i,xerr=[[point-lower],[upper-point]],color='#0077BB' if r['with_q'] else '#EE7733',
                         fmt='o' if r['caliper']==1 else '^',capsize=3,ms=4)
    axes[1].set_yticks(range(4),[f"{'With Q' if r['with_q'] else 'No Q'}, caliper {r['caliper']:g}" for r in intervals])
    axes[1].set_xlim(-.03,1.03);axes[1].set_xlabel('All-three-seeds pass fraction (N=492)')
    fig.tight_layout(pad=.9)
    for ext in ['png','pdf','svg']:fig.savefig(out/('fig4_control_sensitivity.'+ext),dpi=600)
    plt.close(fig)
    trace={'artifact_id':'fig4_control_sensitivity','source_data':[
        {'path':'caliper_audited_runs.csv','sha256':sha(out/'caliper_audited_runs.csv')},
        {'path':'caliper-audit.json','sha256':sha(out/'caliper-audit.json')}],
        'transformation':{'script':'experiments/audit_steh_controls.py','sha256':sha(Path(__file__))},
        'caption_claim':'Control conclusions are assessed under both fixed structural calipers with basin-level seed confirmation.',
        'supported_manuscript_claims':['Disclose sensitivity to structural-control matching.'],
        'limitations':audit['limits']}
    dump(out/'caliper-trace.json',trace);print(json.dumps(clean(audit),ensure_ascii=True))


if __name__=='__main__':main()
