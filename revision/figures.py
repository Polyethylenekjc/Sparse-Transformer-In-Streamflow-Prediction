"""Redraw supplied results without model inference or R dependencies."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import TwoSlopeNorm
from .plot_style import style, canvas, save, BLUE, ORANGE, GREEN, GRAY
from .common import write_json

LABELS = {"dayl_s":"DAYL","prcp_mm_day":"PRCP","srad_w_m2":"SRAD","swe_mm":"SWE",
          "tmax_c":"TMAX","tmin_c":"TMIN","vp_pa":"VP","streamflow_cfs":"Q (forcing)","target_q":"Q (target)"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def title(ax, label, text):
    ax.set_title(f"({label}) {text}",loc="left",fontweight="bold",pad=8)


def signed_bins(values, linear=1., count=25):
    values=np.asarray(values,float)
    lo,hi=float(values.min()),float(values.max())
    edges=list(np.linspace(-linear,linear,count))
    if lo < -linear:
        edges.extend(-np.geomspace(linear,abs(lo)*(1+1e-9),14))
    if hi > linear:
        edges.extend(np.geomspace(linear,hi*(1+1e-9),14))
    return np.unique(edges)


def schematic(out, registry):
    fig, axes = canvas(112)
    ax=axes[0,0]
    ax.set(xlim=(0,1),ylim=(0,1)); ax.axis("off")
    boxes=[(.04,.68,.27,.24,"Dated daily inputs\n15-day history\nWith-Q / no-Q"),
           (.365,.68,.27,.24,"Chronological split\nTrain-only scaling\nNext-day prediction"),
           (.69,.68,.27,.24,"Sparse Transformer\nValidation-selected gates\nFrozen test evaluation"),
           (.04,.15,.27,.31,"Prediction context\nPersistence and AR\nOne LSTM architecture"),
           (.365,.15,.27,.31,"Representation probes\nCUM / API / CHG\nUnfiltered test scores"),
           (.69,.15,.27,.31,"Computational role\nCore / removal / random\nBasin and seed summaries")]
    for x,y,w,h,text in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.012",facecolor="#EDF3F6",edgecolor=BLUE,lw=.8))
        ax.text(x+w/2,y+h/2,text,ha="center",va="center",linespacing=1.7)
    for a,b in ((.32,.355),(.645,.68)):
        ax.annotate("",xy=(b,.80),xytext=(a,.80),arrowprops=dict(arrowstyle="->",color=BLUE))
    for x in (.175,.50,.825):
        ax.annotate("",xy=(x,.48),xytext=(x,.66),arrowprops=dict(arrowstyle="->",color=BLUE))
    ax.text(.5,.02,"Evidence concerns decodability and model computation within the tested protocol.",ha="center",va="bottom",fontsize=8)
    save(fig,out,"Fig1_revision_workflow",registry)
    fig,axes=canvas(94)
    ax=axes[0,0]; ax.set(xlim=(0,1),ylim=(0,1)); ax.axis("off")
    texts=["Stage I: dense training\nAll gates open\nWeight sparsity: off\nActivation sparsity: off\nValidation checkpoint",
           "Stage II: sparse training\nWeight Top-K projection\nActivation Top-K annealing\nAll gates open",
           "Stage III: circuit gating\nInput / head / neuron gates\nWeights and gates updated\nFixed sparsity budget"]
    for i,t in enumerate(texts):
        x=.04+i*.325
        ax.add_patch(FancyBboxPatch((x,.32),.27,.55,boxstyle="round,pad=.012",facecolor=["#EDF3F6","#F9F0E4","#ECF4EF"][i],edgecolor=[BLUE,ORANGE,GREEN][i],lw=.8))
        ax.text(x+.135,.595,t,ha="center",va="center",linespacing=1.8)
        if i<2:
            ax.annotate("",xy=(x+.315,.595),xytext=(x+.285,.595),arrowprops=dict(arrowstyle="->",color=GRAY))
    ax.text(.5,.14,"Projection is reapplied after every sparse optimizer update.\nEach phase restores its best validation checkpoint.",ha="center",va="center",linespacing=1.6,fontsize=8)
    save(fig,out,"Fig2_revision_training",registry)


def collect(root):
    records,probes=[],[]
    for station in sorted(p for p in root.iterdir() if p.is_dir() and p.name.isdigit()):
        summary=read(station/"summary.json")
        if not summary:
            continue
        circuit=read(station/"circuit_structure.json") or {}
        interaction=read(station/"variable_interactions.json") or {}
        record={"station_id":station.name,"nse":summary.get("metrics",{}).get("nse"),
                "active_units":sum(sum(x) for k in ("active_heads","active_neurons") for x in circuit.get(k,[])) if circuit else np.nan}
        ranked=interaction.get("ranked_pairs") or interaction.get("ranking") or []
        # Older files sometimes store only the matrix. Reconstruct the defined
        # interaction as paired delta minus the two single deltas.
        if not ranked and interaction.get("pair_delta_matrix"):
            names=interaction["feature_names"]; singles=interaction["single_delta"]
            m=np.asarray(interaction["pair_delta_matrix"])
            ranked=[{"var_i":names[i],"var_j":names[j],"interaction":float(m[i,j]-singles[names[i]]-singles[names[j]])} for i in range(len(names)) for j in range(i+1,len(names))]
        if ranked:
            top=max(ranked,key=lambda r:abs(r["interaction"]))
            record["dominant_pair"]=" + ".join(LABELS.get(top[k],top[k]) for k in ("var_i","var_j"))
        records.append(record)
        p=read(station/"physical_probe_results.json") or {}
        for name, value in p.items():
            if isinstance(value,dict):
                probes.append({"station_id":station.name,"target":name,"r2":value.get("r2"),"status":"saved" if value.get("r2") is not None else "missing"})
    return pd.DataFrame(records),pd.DataFrame(probes)


def population_figures(records, probes, stations_csv, out, registry):
    records.to_csv(out/"legacy_station_summary.csv",index=False)
    probes.to_csv(out/"legacy_probe_scores_unfiltered.csv",index=False)
    if len(records):
        fig,axes=canvas(88,1,2)
        a,b=axes[0]
        values=records.nse.dropna().to_numpy()
        a.hist(values,bins=signed_bins(values),color=BLUE,edgecolor="white",lw=.4)
        a.set_xscale("symlog",linthresh=1)
        a.set_xlim(min(values.min()*1.05,-1),1.05)
        a.set(xlabel="Saved NSE (symmetric-log bins)",ylabel="Basin count"); title(a,"a",f"Available basins (n={len(records)})")
        b.scatter(records.active_units,records.nse,s=8,alpha=.5,color=BLUE,rasterized=True)
        b.set_yscale("symlog",linthresh=1)
        b.set_ylim(min(values.min()*1.05,-1),1.05)
        b.set(xlabel="Active heads + neurons",ylabel="NSE (symmetric log)"); title(b,"b","Sparsity–performance association")
        save(fig,out,"Fig3_performance_and_size",registry)
        if stations_csv and Path(stations_csv).exists():
            meta=pd.read_csv(stations_csv,dtype={"gauge_id":str,"station_id":str})
            key="gauge_id" if "gauge_id" in meta else "station_id"
            meta[key]=meta[key].str.zfill(8)
            for column in ("lat","lon","forcing_elev_m"):
                if column in meta:
                    meta[column]=pd.to_numeric(meta[column],errors="coerce")
            merged=records.merge(meta,left_on="station_id",right_on=key)
            if all(c in merged for c in ("lat","lon","forcing_elev_m")):
                fig,axes=canvas(88,1,2)
                for ax,col,label,text in zip(axes[0],["nse","forcing_elev_m"],["a","b"],["Saved test NSE","Forcing elevation (m)"]):
                    norm=TwoSlopeNorm(vmin=min(float(merged[col].min()),-1e-6),vcenter=0,vmax=1) if col=="nse" else None
                    im=ax.scatter(merged.lon,merged.lat,c=merged[col],s=5,cmap="viridis",norm=norm,rasterized=True)
                    ax.set(xlabel="Longitude (°)",ylabel="Latitude (°)",xlim=(-127,-65),ylim=(24,51))
                    title(ax,label,text)
                    bar=fig.colorbar(im,ax=ax,shrink=.72)
                    if col=="nse":
                        bar.set_ticks([v for v in [-500,-100,0,.5,1] if norm.vmin<=v<=norm.vmax])
                save(fig,out,"Fig3_basin_locations",registry)
                merged.to_csv(out/"legacy_basin_map_values.csv",index=False)
    if len(probes):
        targets=sorted(probes.target.unique())
        fig,axes=canvas(110,1,len(targets))
        rng=np.random.default_rng(42)
        for ax,name,i in zip(axes[0],targets,range(len(targets))):
            values=pd.to_numeric(probes.loc[probes.target==name,"r2"],errors="coerce").dropna().to_numpy()
            ax.scatter(rng.uniform(-.16,.16,len(values)),values,s=5,alpha=.35,color=BLUE,rasterized=True)
            ax.axhline(0,color=GRAY,lw=.7)
            ax.set(xticks=[],ylabel="Saved probe R²",xlim=(-.3,.3))
            if len(values) and values.min() < -2:
                ax.set_yscale("symlog",linthresh=1)
                ax.set_ylim(values.min()*1.2,max(1.1,float(values.max())*1.1))
                ticks=[v for v in [-1e12,-1e9,-1e6,-1e3,-1,0,1] if ax.get_ylim()[0] <= v <= ax.get_ylim()[1]]
                ax.set_yticks(ticks)
                ax.set_ylabel("R² (symmetric log)")
            title(ax,chr(97+i),{"cum_rain":"Cumulative rain","api":"API","dpdt":"Rain change\n(legacy)"}.get(name,name))
            ax.text(.5,.03,f"n={len(values)}\nnegative={np.sum(values<0)}",transform=ax.transAxes,ha="center",va="bottom",fontsize=8,bbox=dict(fc="white",ec="none",alpha=.85))
        save(fig,out,"Fig5_probe_scores_unfiltered",registry)
    if "dominant_pair" in records:
        counts=records.dominant_pair.fillna("Unavailable").value_counts()
        all_counts=counts.copy()
        if len(counts)>12:
            counts=pd.concat([counts.iloc[:12],pd.Series({"Other categories":int(counts.iloc[12:].sum())})])
        fig,axes=canvas(max(85,10*len(counts)+25))
        ax=axes[0,0]; ax.barh(counts.index[::-1],counts.to_numpy()[::-1],color=BLUE)
        ax.set(xlabel="Basin count"); title(ax,"a","Dominant saved interaction categories")
        save(fig,out,"Fig6_interaction_counts",registry)
        all_counts.rename("count").to_csv(out/"legacy_interaction_counts.csv")


def examples(root,out,registry,ids):
    for station in ids:
        folder=root/station
        interaction=read(folder/"variable_interactions.json")
        summary=read(folder/"summary.json")
        if not interaction or not summary:
            continue
        names=interaction["feature_names"]
        singles=interaction["single_delta"]
        pair=np.asarray(interaction["pair_delta_matrix"])
        values=pair-np.array([singles[n] for n in names])[:,None]-np.array([singles[n] for n in names])[None,:]
        np.fill_diagonal(values,0)
        fig,axes=canvas(106,1,2)
        a,b=axes[0]
        limit=max(float(abs(values).max()),1e-12)
        im=a.imshow(values,cmap="RdBu_r",vmin=-limit,vmax=limit)
        labels=[LABELS.get(n,n) for n in names]
        a.set(xticks=range(len(names)),yticks=range(len(names)),xticklabels=labels,yticklabels=labels)
        a.tick_params(axis="x",rotation=65)
        title(a,"a",f"{station}; NSE={summary['metrics']['nse']:.3f}")
        fig.colorbar(im,ax=a,orientation="horizontal",shrink=.8,pad=.02,label="Interaction (ΔMSE combination)")
        faith=folder/"faithfulness_curve.csv"
        if faith.exists():
            f=pd.read_csv(faith); b.plot(f.k,f.nse,color=BLUE)
            b.axhline(summary["metrics"]["nse"],color=ORANGE,ls="--",label="Saved full model")
            b.set(xlabel="Retained ranked units (legacy k)",ylabel="Saved test NSE"); b.legend()
        title(b,"b","Legacy retention curve")
        save(fig,out,f"Fig4_example_{station}",registry)


def sensitivity(path,out,registry):
    if not path or not Path(path).exists():
        return
    frame=pd.read_csv(path,dtype={"station_id":str})
    required={"scenario_kind","scenario_value","base_volume","volume","delta_peak_time"}
    if not required.issubset(frame):
        raise ValueError(f"Sensitivity table missing {required-set(frame)}")
    frame["response_percent"]=np.where(abs(frame.base_volume)>1e-12,100*(frame.volume-frame.base_volume)/frame.base_volume,np.nan)
    frame.to_csv(out/"sensitivity_model_relative.csv",index=False)
    kinds=list(frame.scenario_kind.drop_duplicates())
    for page in range(0,len(kinds),4):
        subset=kinds[page:page+4]
        fig,axes=canvas(155,2,2)
        for i,(ax,kind) in enumerate(zip(axes.flat,subset)):
            data=frame[frame.scenario_kind==kind]
            groups=list(data.scenario_value.astype(str).drop_duplicates())
            arrays=[data.loc[data.scenario_value.astype(str)==g,"response_percent"].dropna().to_numpy() for g in groups]
            if any(len(v) for v in arrays):
                ax.boxplot(arrays,positions=range(len(groups)),widths=.55,showfliers=True,medianprops=dict(color=ORANGE),boxprops=dict(color=BLUE),whiskerprops=dict(color=BLUE),flierprops=dict(marker=".",markersize=2))
            ax.axhline(0,color=GRAY,ls="--",lw=.7)
            ax.set(xticks=range(len(groups)),xticklabels=groups,xlabel="Perturbation setting",ylabel="Change in prediction sum (%)")
            ax.tick_params(axis="x",rotation=25)
            title(ax,chr(97+i),str(kind).replace("_"," "))
        for ax in list(axes.flat)[len(subset):]:
            ax.axis("off")
        save(fig,out,f"Fig7_model_sensitivity_{page//4+1}",registry)


def rf(path,out,registry):
    if not path or not Path(path).exists():
        return
    df=pd.read_csv(path,dtype={"station_id":str})
    if not {"nse_base","nse_mod"}.issubset(df):
        raise ValueError("RF table needs nse_base,nse_mod")
    valid=df[["nse_base","nse_mod"]].dropna()
    fig,axes=canvas(88,1,2)
    a,b=axes[0]
    a.scatter(valid.nse_base,valid.nse_mod,s=9,alpha=.55,color=BLUE,rasterized=True)
    lo=min(valid.min().min(),0); a.plot([lo,1],[lo,1],color=GRAY,ls="--")
    a.set_xscale("symlog",linthresh=1); a.set_yscale("symlog",linthresh=1)
    a.set_xlim(min(lo*1.05,-1),1.05); a.set_ylim(min(lo*1.05,-1),1.05)
    a.set(xlabel="Baseline NSE (symmetric log)",ylabel="Guided NSE (symmetric log)"); title(a,"a",f"Available RF rows (n={len(valid)})")
    delta=valid.nse_mod-valid.nse_base
    b.hist(delta,bins=signed_bins(delta,linear=.1),color=BLUE,edgecolor="white"); b.axvline(0,color=GRAY,ls="--")
    b.set_xscale("symlog",linthresh=.1)
    b.set_xlim(min(float(delta.min())*1.05,-.1),max(float(delta.max())*1.05,.1))
    b.set(xlabel="ΔNSE (symmetric-log bins)",ylabel="Basin count"); title(b,"b","Same-basin performance change")
    save(fig,out,"Fig8_RF_available_rows",registry)
    write_json(out/"rf_available_summary.json",{"n":len(valid),"mean_delta_nse":float(delta.mean()),"improved_fraction":float((delta>0).mean()),"limits":"All available CSV rows; this does not reproduce the manuscript's selected 327-basin subset unless that exact subset CSV is supplied."})


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--legacy-root",type=Path)
    p.add_argument("--stations-csv")
    p.add_argument("--sensitivity-csv")
    p.add_argument("--rf-csv")
    p.add_argument("--out",type=Path,default=Path("outputs/revision_figures"))
    p.add_argument("--examples",nargs="+",default=["01013500","03504000","01439500"])
    a=p.parse_args(argv); a.out.mkdir(parents=True,exist_ok=True)
    font=style(); registry=[]
    schematic(a.out,registry)
    if a.legacy_root:
        records,probes=collect(a.legacy_root)
        population_figures(records,probes,a.stations_csv,a.out,registry)
        examples(a.legacy_root,a.out,registry,a.examples)
    sensitivity(a.sensitivity_csv,a.out,registry)
    rf(a.rf_csv,a.out,registry)
    write_json(a.out/"figure_manifest.json",{"font":font,"figures":registry,"sources":vars(a)|{"legacy_root":str(a.legacy_root),"out":str(a.out)},
       "limits":"Redrawn legacy results retain their original scientific limitations; no new training or inference. Main-text replacement requires matching captions and subset identities. Basemap panels display basin coordinates, not region polygons."})
    print(f"Exported {len(registry)} figures: {a.out}")


if __name__=="__main__":
    main()
