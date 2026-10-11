"""No-argument/default command only prepares a plan. Training is explicit."""
import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import sys
import traceback
import numpy as np
import pandas as pd
from .common import digest, metrics, wilson, finite_population_interval, write_json
from .data import load_basin, read_daily, is_q

ROOT = Path(__file__).resolve().parents[1]


def config(path, data_root=None, output_root=None):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    if data_root:
        cfg["data_root"] = data_root
    if output_root:
        cfg["output_root"] = output_root
    if not 0 < cfg["core_fraction"] < 1:
        raise ValueError("core_fraction must lie strictly between zero and one")
    if cfg["seq_len"] < 2 or cfg["pred_horizon"] < 1 or cfg["random_controls"] < 1:
        raise ValueError("Invalid sequence, horizon or control count")
    if cfg["sample_size"] < 1 or not cfg["seeds"] or len(set(cfg["seeds"])) != len(cfg["seeds"]):
        raise ValueError("Require positive sample_size and unique seeds")
    if not 0 < cfg["train_ratio"] < 1 or not 0 < cfg["val_ratio"] < 1-cfg["train_ratio"]:
        raise ValueError("Invalid split ratios")
    return cfg


def prepare(cfg):
    root, out = Path(cfg["data_root"]), Path(cfg["output_root"])
    f = {p.stem for p in (root/"Forcing").glob("*.csv")}
    q = {p.stem for p in (root/"Streamflow").glob("*.csv")}
    population, excluded = [], []
    for station in sorted(f & q):
        try:
            a = pd.read_csv(root/"Forcing"/f"{station}.csv", nrows=2)
            b = pd.read_csv(root/"Streamflow"/f"{station}.csv", nrows=2)
            if not any("date" in c.lower() for c in a) or not any("date" in c.lower() for c in b):
                raise ValueError("missing date column")
            if not any(is_q(c) for c in b):
                raise ValueError("missing observed-Q column")
            population.append(station)
        except (ValueError, OSError, pd.errors.ParserError) as e:
            excluded.append({"station": station, "reason": str(e)})
    if len(population) < cfg["sample_size"]:
        raise ValueError(f"Only {len(population)} structurally eligible basins; sample_size={cfg['sample_size']}. Set a smaller sample explicitly.")
    selected = sorted(np.random.default_rng(cfg["sample_seed"]).choice(population, cfg["sample_size"], replace=False).tolist())
    hashes = {f"{folder}/{station}.csv": hashlib.sha256((root/folder/f"{station}.csv").read_bytes()).hexdigest()
              for station in selected for folder in ("Forcing", "Streamflow")}
    if cfg.get("observation_mask_dir"):
        hashes.update({f"observation_mask/{station}.csv": hashlib.sha256((Path(cfg["observation_mask_dir"])/f"{station}.csv").read_bytes()).hexdigest() for station in selected})
    source_files = list((ROOT/"revision").glob("*.py")) + [ROOT/"src/flood_transformer/model.py", ROOT/"src/flood_transformer/sparsity.py"]
    code_hash = digest({p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files})
    windows = [cfg["seq_len"]] + [w for w in cfg.get("optional_window_lengths", []) if w != cfg["seq_len"]]
    jobs = [{"station": station, "seed": seed, "with_q": with_q, "model": model, "seq_len": window}
            for station in selected for seed in cfg["seeds"] for window in windows
            for model, with_q in (("steh", True), ("steh", False), ("lstm", True))]
    manifest = {"config": cfg, "config_sha256": digest(cfg), "code_sha256": code_hash,
                "population": population, "population_sha256": digest(population), "excluded": excluded,
                "sample": selected, "data_sha256": hashes, "jobs": jobs,
                "selection": "simple random sampling without replacement; before reading model performance",
                "criteria": {k:cfg[k] for k in ("core_max_delta_nse","remove_min_delta_nse","random_min_win_fraction")},
                "scope": "Eligible local paired CSV population, not automatically all 521 CAMELS basins"}
    path = out/"manifest.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Existing manifest differs in config, code or data. Use a new output_root to preserve frozen results.")
    write_json(path, manifest)
    write_json(out/"config_resolved.json", cfg)
    print(f"PLAN ONLY: {len(population)} eligible basins; selected {selected}; {len(jobs)} jobs, {len(cfg['lstm_trials'])} validation trials per LSTM job. No training started.")
    return manifest


def baselines(splits, meta, cfg, out):
    from sklearn.linear_model import Ridge
    a, v, t = splits["train"], splits["val"], splits["test"]
    mean, std = meta["q_mean"], meta["q_std"]
    xtrain = (a["q_history"]-mean)/std
    xval, xtest = (v["q_history"]-mean)/std, (t["q_history"]-mean)/std
    fitted = Ridge(alpha=cfg["ar_ridge"]).fit(xtrain, a["y"])
    p = {"persistence": t["q_history"][:,-1], "ar": fitted.predict(xtest)*std+mean}
    target = t["y"]*std+mean
    pd.DataFrame({"target_date": t["dates"], "observed": target, **p}).to_csv(out/"baseline_predictions.csv", index=False)
    write_json(out/"baselines.json", {"metrics": {k: metrics(v,target) for k,v in p.items()},
                "AR": {"alpha": cfg["ar_ridge"], "coef": fitted.coef_, "intercept": fitted.intercept_,
                       "val_metrics": metrics(fitted.predict(xval)*std+mean, v["y"]*std+mean)},
                "scope": "Q-history references for with-Q; not input-matched no-Q competitors"})


def job_path(out, job):
    return out/job["station"]/f"seed_{job['seed']}"/f"L{job['seq_len']}"/f"{job['model']}_{'with_q' if job['with_q'] else 'no_q'}"


def run(cfg, manifest, args):
    # Import PyTorch only after prepare, so deployment plans and figures work
    # without a training runtime and can be inspected on a CPU laptop.
    import torch
    from .models import fit, transformer, LSTM, seed_all, predict
    from .analysis import interpret, probes
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = cfg["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    torch.set_num_threads(cfg["torch_threads"])
    out = Path(cfg["output_root"])
    jobs = [j for j in manifest["jobs"] if (not args.station or j["station"] == args.station)
            and (not args.only or j["model"] == args.only)]
    if args.max_jobs is not None:
        jobs = jobs[:args.max_jobs]
    if not jobs:
        raise ValueError("No jobs match requested filters")
    errors = []
    for j in jobs:
        folder = job_path(out, j)
        folder.mkdir(parents=True, exist_ok=True)
        stamp = digest({"job": j, "manifest": manifest})
        summary_path = folder/"summary.json"
        if summary_path.exists():
            old = json.loads(summary_path.read_text(encoding="utf-8"))
            if old.get("run_sha256") != stamp:
                raise RuntimeError(f"Incompatible completed job: {folder}")
            print(f"SKIP completed {folder}", flush=True)
            continue
        try:
            print(f"RUN {j}", flush=True)
            local = {**cfg, **j, "device": device}
            seed_all(j["seed"])
            versions={"python": platform.python_version(), "torch": torch.__version__,
                      "cuda": torch.version.cuda, "device": device}
            training_versions=versions
            splits, meta = load_basin(cfg["data_root"], j["station"], local)
            write_json(folder/"preprocessing.json", meta)
            write_json(folder/"run_config.json", local)
            baselines(splits, meta, local, folder)
            completed = folder/"trained.pt"
            log, tuning = [], []
            if j["model"] == "steh":
                model = transformer(len(meta["feature_names"]), local)
                trial = None
            else:
                trial = None
                model = None
            if completed.exists():
                checkpoint = torch.load(completed, map_location="cpu", weights_only=False)
                if checkpoint["run_sha256"] != stamp:
                    raise RuntimeError("Interrupted checkpoint does not match the frozen run")
                trial = checkpoint.get("lstm_trial")
                training_versions=checkpoint.get("training_versions",{})
                if j["model"] == "lstm":
                    model = LSTM(len(meta["feature_names"]), trial["hidden_size"], trial["num_layers"], cfg["dropout"])
                model.load_state_dict(checkpoint["state_dict"])
                model.to(device)
                if j["model"] == "steh":
                    from .models import phase_switch
                    phase_switch(model, "mask", cfg["target_activation_sparsity"])
            elif j["model"] == "steh":
                model, log = fit(model, splits, local)
            else:
                best, best_state, best_log = float("inf"), None, None
                for index, candidate in enumerate(cfg["lstm_trials"]):
                    seed_all(j["seed"])
                    candidate_model = LSTM(len(meta["feature_names"]), candidate["hidden_size"], candidate["num_layers"], cfg["dropout"])
                    candidate_model, candidate_log = fit(candidate_model, splits, local, "lstm", candidate)
                    loss = float(np.mean((predict(candidate_model, splits["val"], local)-splits["val"]["y"])**2))
                    tuning.append({"trial": index, **candidate, "validation_mse": loss})
                    if loss < best:
                        best, trial = loss, candidate
                        best_state = {k:v.detach().cpu().clone() for k,v in candidate_model.state_dict().items()}
                        best_log = candidate_log
                if best_state is None:
                    raise RuntimeError("All LSTM validation trials failed")
                model = LSTM(len(meta["feature_names"]), trial["hidden_size"], trial["num_layers"], cfg["dropout"]).to(device)
                model.load_state_dict(best_state)
                log = best_log
            if not completed.exists():
                temporary = folder/"trained.tmp"
                torch.save({"state_dict": model.state_dict(), "config": local, "preprocessing": meta,
                            "run_sha256": stamp, "lstm_trial": trial, "training_versions": versions}, temporary)
                temporary.replace(completed)
                pd.DataFrame(log).to_csv(folder/"training_log.csv", index=False)
                if tuning:
                    write_json(folder/"lstm_tuning.json", tuning)
            p = predict(model, splits["test"], local)*meta["q_std"]+meta["q_mean"]
            y = splits["test"]["y"]*meta["q_std"]+meta["q_mean"]
            pd.DataFrame({"target_date": splits["test"]["dates"], "observed": y, "predicted": p}).to_csv(folder/"predictions.csv", index=False)
            if j["model"] == "steh":
                write_json(folder/"interpretation.json", interpret(model, splits, meta, local))
                write_json(folder/"probes.json", probes(model, splits, meta, local))
            write_json(summary_path, {"job": j, "run_sha256": stamp, "metrics": metrics(p,y),
                       "target_status": meta["target_status"], "training_versions": training_versions, "inference_versions": versions,
                       "lstm_trial": trial, "test_dates_sha256": digest(splits["test"]["dates"].tolist())})
        except Exception as e:
            failure = {"job": j, "error": str(e), "traceback": traceback.format_exc()}
            write_json(folder/"failure.json", failure)
            errors.append(failure)
            print(f"FAILED {j}: {e}", file=sys.stderr, flush=True)
            if args.fail_fast:
                raise
    summarize(cfg, manifest)
    if errors:
        raise RuntimeError(f"{len(errors)} job(s) failed. They remain in the sample; see failure.json. Re-run to retry.")


def summarize(cfg, manifest):
    out = Path(cfg["output_root"])
    rows, effects, probes_rows = [], [], []
    for j in manifest["jobs"]:
        folder = job_path(out,j)
        path = folder/"summary.json"
        if not path.exists():
            rows.append({**j, "status": "failed" if (folder/"failure.json").exists() else "pending"})
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        rows.append({**j, "status": "complete", **value["metrics"], "target_status": value["target_status"]})
        if j["model"] == "steh":
            analysis = json.loads((folder/"interpretation.json").read_text(encoding="utf-8"))
            effects.append({**j, **analysis["circuit"]})
            for row in json.loads((folder/"probes.json").read_text(encoding="utf-8"))["results"]:
                probes_rows.append({**j, **row})
    pd.DataFrame(rows).to_csv(out/"summary.csv", index=False)
    pd.DataFrame(effects).to_csv(out/"circuit_summary.csv", index=False)
    pd.DataFrame(probes_rows).to_csv(out/"probe_summary_unfiltered.csv", index=False)
    stability = []
    for station in manifest["sample"]:
        for with_q in (True,False):
            for window in sorted({j["seq_len"] for j in manifest["jobs"]}):
                entries = []
                for seed in cfg["seeds"]:
                    p = job_path(out,{"station":station,"seed":seed,"model":"steh","with_q":with_q,"seq_len":window})/"interpretation.json"
                    if p.exists():
                        entries.append((seed,json.loads(p.read_text(encoding="utf-8"))))
                for index,(sa,a) in enumerate(entries):
                    for sb,b in entries[index+1:]:
                        aa = {r["feature"] for r in a["inputs"] if r["active"]}
                        bb = {r["feature"] for r in b["inputs"] if r["active"]}
                        ma = {r["feature"]:r["test_delta_nse"] for r in a["inputs"]}
                        mb = {r["feature"]:r["test_delta_nse"] for r in b["inputs"]}
                        names = sorted(ma.keys() & mb.keys())
                        ra, rb = pd.Series([ma[n] for n in names]).rank().to_numpy(), pd.Series([mb[n] for n in names]).rank().to_numpy()
                        corr = float(np.corrcoef(ra,rb)[0,1]) if np.std(ra)>0 and np.std(rb)>0 else None
                        pa = {tuple(sorted(r["features"])) for r in a["interactions"]}
                        pb = {tuple(sorted(r["features"])) for r in b["interactions"]}
                        stability.append({"station":station,"with_q":with_q,"seq_len":window,"seed_a":sa,"seed_b":sb,
                           "input_jaccard":len(aa&bb)/len(aa|bb) if aa|bb else None,
                           "input_effect_spearman":corr,"top_pair_jaccard":len(pa&pb)/len(pa|pb) if pa|pb else None,
                           "core_delta_nse_a":a["circuit"]["core_delta_nse"],"core_delta_nse_b":b["circuit"]["core_delta_nse"]})
    pd.DataFrame(stability).to_csv(out/"seed_stability.csv", index=False)
    estimates = []
    for with_q in (True,False):
        # Primary estimate only for the prespecified 15-day (or configured) window.
        success, failed, pending = [], [], []
        for station in manifest["sample"]:
            entries = [r for r in effects if r["station"] == station and r["with_q"] == with_q and r["seq_len"] == cfg["seq_len"]]
            if len(entries) != len(cfg["seeds"]):
                pending.append(station)
            elif all(r["pass"] for r in entries):
                success.append(station)
            else:
                failed.append(station)
        total = len(manifest["sample"])
        estimates.append({"with_q":with_q,"success":success,"criterion_failures":failed,"incomplete":pending,
                          "independent_unit":"basin; all prespecified seeds must pass",
                          "population_n":len(manifest["population"]),"sample_n":total,
                          "success_fraction":len(success)/total if not pending else None,
                          "exact_finite_population_95":finite_population_interval(len(success),total,len(manifest["population"])) if not pending else None,
                          "wilson_95_descriptive_binomial_approximation":wilson(len(success),total) if not pending else [None,None],
                          "limits":"Exact interval inverts the hypergeometric distribution for SRS without replacement; Wilson is a descriptive binomial approximation. No interval until all sampled basins finish. Applies to this eligible population and fixed criterion, not proof that every model or basin satisfies it."})
    write_json(out/"sample_estimates.json", estimates)
    print(f"Summaries written to {out}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare","run","summarize"], nargs="?", default="prepare")
    parser.add_argument("--config", default=str(ROOT/"configs/revision_minimal.json"))
    parser.add_argument("--data-root")
    parser.add_argument("--output-root")
    parser.add_argument("--station", help="Run only an already sampled basin")
    parser.add_argument("--only", choices=["steh","lstm"])
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args(argv)
    if args.max_jobs is not None and args.max_jobs < 1:
        parser.error("--max-jobs must be positive")
    cfg = config(args.config,args.data_root,args.output_root)
    manifest = prepare(cfg)
    if args.command == "run":
        run(cfg, manifest, args)
    elif args.command == "summarize":
        summarize(cfg,manifest)


if __name__ == "__main__":
    main()
