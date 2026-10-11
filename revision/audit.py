"""Audit sampled data without training, model inference or performance filtering."""
import argparse
from pathlib import Path
import numpy as np
from .runner import config, prepare
from .data import load_basin, read_daily, is_q
from .common import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",default="configs/revision_minimal.json")
    p.add_argument("--data-root")
    p.add_argument("--output-root",default="outputs/revision_data_audit")
    a=p.parse_args(); cfg=config(a.config,a.data_root,a.output_root)
    manifest=prepare(cfg); rows=[]
    for station in manifest["sample"]:
        with_q,meta=load_basin(cfg["data_root"],station,cfg|{"with_q":True})
        no_q,no_meta=load_basin(cfg["data_root"],station,cfg|{"with_q":False})
        for split in with_q:
            np.testing.assert_array_equal(with_q[split]["dates"],no_q[split]["dates"])
            np.testing.assert_array_equal(with_q[split]["y"],no_q[split]["y"])
        forcing=read_daily(Path(cfg["data_root"])/"Forcing"/f"{station}.csv")
        flow=read_daily(Path(cfg["data_root"])/"Streamflow"/f"{station}.csv")
        target=next(c for c in flow if is_q(c))
        overlap=[]
        for col in forcing:
            if is_q(col):
                dates=forcing.index.intersection(flow.index)
                u=forcing.loc[dates,col].to_numpy(float); v=flow.loc[dates,target].to_numpy(float)
                valid=np.isfinite(u)&np.isfinite(v)
                overlap.append({"forcing_column":col,"paired_finite_days":int(valid.sum()),
                                "maximum_absolute_difference":float(abs(u[valid]-v[valid]).max()) if valid.any() else None})
        rows.append({"station":station,"counts":meta["counts"],"with_q_features":meta["feature_names"],
                     "no_q_features":no_meta["feature_names"],"paired_target_dates_equal":True,
                     "legacy_Q_copy_check":overlap,"target_status":meta["target_status"]})
    write_json(Path(cfg["output_root"])/"data_audit.json",rows)
    print(f"Audited {len(rows)} sampled basins; no model training or inference.")


if __name__=="__main__":
    main()
