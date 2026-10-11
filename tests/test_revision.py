"""Small synthetic checks only; no CAMELS training or result assertions."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from revision.data import load_basin
from revision.common import wilson, metrics, finite_population_interval
from revision.runner import main

ROOT=Path(__file__).resolve().parents[1]


def fixture(root):
    for name in ("Forcing","Streamflow"):
        (root/name).mkdir(parents=True)
    dates=pd.date_range("2000-01-01",periods=140)
    t=np.arange(len(dates))
    q=10+np.sin(t/3)*2+t*.1
    pd.DataFrame({"date_utc":dates,"prcp_mm_day":(t%7).astype(float),"tmax_c":12+np.sin(t/4),
                  "streamflow_cfs":q,"year":2000}).to_csv(root/"Forcing/01013500.csv",index=False)
    pd.DataFrame({"date_utc":dates,"streamflow_cfs":q,"qc_flag":"A"}).to_csv(root/"Streamflow/01013500.csv",index=False)


def small_config(root):
    cfg=json.loads((ROOT/"configs/revision_minimal.json").read_text())
    cfg.update(data_root=str(root),output_root=str(root/"results"),sample_size=1,seeds=[42],
               seq_len=4,d_model=8,n_heads=2,num_layers=1,d_ff=8,batch_size=16,
               epochs_dense=1,epochs_sparse=1,epochs_mask=1,epochs_lstm=1,torch_threads=1,device="cpu",
               analysis_val_samples=8,analysis_test_samples=8,probe_train_samples=32,
               max_neuron_candidates_per_layer=4,random_controls=3,
               lstm_trials=[{"hidden_size":4,"num_layers":1,"learning_rate":.001},
                            {"hidden_size":8,"num_layers":1,"learning_rate":.001}])
    return cfg


class RevisionChecks(unittest.TestCase):
    def test_no_q_scaler_and_calendar_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); fixture(root); cfg=small_config(root)|{"with_q":True}
            paired,meta=load_basin(root,"01013500",cfg)
            self.assertEqual(meta["feature_names"],["prcp_mm_day","tmax_c","target_q"])
            no_q,no_meta=load_basin(root,"01013500",cfg|{"with_q":False})
            self.assertNotIn("target_q",no_meta["feature_names"])
            for name in paired:
                np.testing.assert_array_equal(paired[name]["dates"],no_q[name]["dates"])
            forcing=pd.read_csv(root/"Forcing/01013500.csv")
            forcing.loc[110:,"tmax_c"]=1e8
            forcing.to_csv(root/"Forcing/01013500.csv",index=False)
            _,changed=load_basin(root,"01013500",cfg)
            np.testing.assert_allclose(meta["x_mean"],changed["x_mean"])
            forcing=forcing.drop(index=20)
            forcing.to_csv(root/"Forcing/01013500.csv",index=False)
            gap,_=load_basin(root,"01013500",cfg)
            # Missing day 21 cannot be compressed into a consecutive window.
            forbidden=pd.date_range("2000-01-22",periods=4).strftime("%Y-%m-%d")
            self.assertTrue(set(forbidden).isdisjoint(set(gap["train"]["dates"])))

    def test_three_jobs_resume_and_checkpoint_flags(self):
        import torch
        from revision.models import transformer,phase_switch,predict
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); fixture(root); cfg=small_config(root)
            path=root/"config.json"; path.write_text(json.dumps(cfg))
            with contextlib.redirect_stdout(io.StringIO()):
                main(["prepare","--config",str(path)])
            self.assertFalse(list((root/"results").rglob("trained.pt")))
            with contextlib.redirect_stdout(io.StringIO()):
                main(["run","--config",str(path),"--fail-fast"])
            summaries=list((root/"results").rglob("summary.json"))
            self.assertEqual(len(summaries),3)
            fingerprints=[p.read_bytes() for p in summaries]
            with contextlib.redirect_stdout(io.StringIO()) as stream:
                main(["run","--config",str(path),"--fail-fast"])
            self.assertEqual(stream.getvalue().count("SKIP completed"),3)
            self.assertEqual(fingerprints,[p.read_bytes() for p in summaries])
            dates={json.loads(p.read_text())["test_dates_sha256"] for p in summaries}
            self.assertEqual(len(dates),1)
            checkpoint=next(p for p in (root/"results").rglob("trained.pt") if p.parent.name=="steh_no_q")
            state=torch.load(checkpoint,map_location="cpu",weights_only=False)
            local=state["config"]; splits,meta=load_basin(root,"01013500",local)
            model=transformer(len(meta["feature_names"]),local); model.load_state_dict(state["state_dict"])
            phase_switch(model,"mask",local["target_activation_sparsity"])
            expected=pd.read_csv(checkpoint.parent/"predictions.csv").predicted.to_numpy()
            actual=predict(model,splits["test"],local)*meta["q_std"]+meta["q_mean"]
            np.testing.assert_allclose(expected,actual,rtol=1e-6,atol=1e-6)
            from revision.analysis import binary_overrides
            pred=predict(model,splits["test"],local)
            np.testing.assert_allclose(pred,predict(model,splits["test"],local,binary_overrides(model,"cpu")),rtol=1e-6,atol=1e-6)
            estimates=json.loads((root/"results/sample_estimates.json").read_text())
            self.assertEqual(estimates[0]["sample_n"],1)

    def test_metrics_and_interval(self):
        self.assertAlmostEqual(metrics([1,2,3],[1,2,3])["nse"],1.)
        self.assertAlmostEqual(metrics([1,2,3],[1,2,3])["kge"],1.)
        self.assertGreater(wilson(12,12)[0],.7)
        self.assertEqual(wilson(0,0),[None,None])
        self.assertEqual(finite_population_interval(2,3,3)["success_fraction_bounds"],[2/3,2/3])
        interval=finite_population_interval(12,12,521)
        self.assertGreater(interval["success_fraction_bounds"][0],.7)
        self.assertLess(interval["success_fraction_bounds"][0],1.)
        self.assertEqual(interval["success_fraction_bounds"][1],1.)


if __name__=="__main__":
    unittest.main()
