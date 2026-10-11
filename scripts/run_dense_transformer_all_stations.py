from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from typing import Callable, Dict, List, Optional, Set

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.flood_transformer.config import ExperimentConfig
from src.flood_transformer.data import load_data
from src.flood_transformer.model import ExplainableSparseTransformer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model: ExplainableSparseTransformer, loader, device: torch.device) -> float:
    model.eval()
    losses = []
    loss_fn = nn.MSELoss()
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x, return_intermediates=False).pred
        losses.append(float(loss_fn(pred, y).item()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def collect_preds_targets(model: ExplainableSparseTransformer, loader, device: torch.device):
    model.eval()
    preds = []
    ys = []
    for x, y in loader:
        x = x.to(device)
        out = model(x, return_intermediates=False)
        preds.append(out.pred.detach().cpu().numpy())
        ys.append(y.numpy())
    if not preds:
        return np.array([]), np.array([])
    return np.concatenate(preds), np.concatenate(ys)


def compute_loader_metrics(model: ExplainableSparseTransformer, loader, device: torch.device) -> Dict[str, float]:
    pred, y = collect_preds_targets(model, loader, device)
    return regression_metrics(pred, y)


def regression_metrics(pred: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    if len(pred) == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "nse": float("nan")}
    err = pred - y
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    denom = float(np.sum((y - y.mean()) ** 2))
    nse = float(1.0 - np.sum(err ** 2) / (denom + 1e-12))
    return {"rmse": rmse, "mae": mae, "nse": nse}


def list_station_ids(forcing_dir: str, streamflow_dir: str, require_streamflow: bool = True) -> List[str]:
    forcing_ids: Set[str] = {
        os.path.splitext(f)[0]
        for f in os.listdir(forcing_dir)
        if f.endswith(".csv")
    }
    if not require_streamflow:
        return sorted(forcing_ids)

    if not os.path.exists(streamflow_dir):
        return sorted(forcing_ids)

    stream_ids: Set[str] = {
        os.path.splitext(f)[0]
        for f in os.listdir(streamflow_dir)
        if f.endswith(".csv")
    }
    return sorted(forcing_ids.intersection(stream_ids))


def train_dense_model(
    config: ExperimentConfig,
    epochs: int,
    prune_callback: Optional[Callable[[ExplainableSparseTransformer, object, int], None]] = None,
):
    data = load_data(config)
    device = torch.device(config.device)

    model = ExplainableSparseTransformer(
        input_dim=len(data.feature_names),
        d_model=config.d_model,
        n_heads=config.n_heads,
        num_layers=config.num_layers,
        d_ff=config.d_ff,
        dropout=config.dropout,
        task_type="regression",
        enable_activation_sparsity=False,
        activation_sparsity_ratio=0.0,
        enable_learnable_masks=False,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    train_log = []

    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses = []
        for x, y in data.train_loader:
            x = x.to(device)
            y = y.to(device)

            out = model(x, return_intermediates=False)
            loss = loss_fn(out.pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(float(loss.item()))

        train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        val_loss = evaluate(model, data.val_loader, device)
        train_log.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if prune_callback is not None:
            prune_callback(model, data, epoch)

        if np.isfinite(val_loss) and val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, data, best_val, train_log


def train_one_station(config: ExperimentConfig, epochs: int, save_model_name: str = "transformer_dense.pt") -> Dict[str, object]:
    model, data, best_val, train_log = train_dense_model(config, epochs)
    device = torch.device(config.device)

    test_loss = evaluate(model, data.test_loader, device)
    metrics = compute_loader_metrics(model, data.test_loader, device)

    os.makedirs(config.output_dir, exist_ok=True)
    model_path = os.path.join(config.output_dir, save_model_name)
    torch.save(model.state_dict(), model_path)

    train_log_path = os.path.join(config.output_dir, "dense_train_log.csv")
    pd.DataFrame(train_log).to_csv(train_log_path, index=False)

    summary = {
        "station_id": config.station_id,
        "model_type": "dense_transformer",
        "best_val_loss": best_val,
        "test_loss": test_loss,
        "metrics": metrics,
        "feature_count": len(data.feature_names),
        "artifacts": {
            "model": model_path,
            "train_log": train_log_path,
        },
    }

    summary_path = os.path.join(config.output_dir, "dense_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return {
        "station_id": str(config.station_id),
        "status": "ok",
        "best_val_loss": float(best_val),
        "test_loss": float(test_loss),
        "rmse": float(metrics["rmse"]),
        "mae": float(metrics["mae"]),
        "nse": float(metrics["nse"]),
        "feature_count": int(len(data.feature_names)),
        "model_path": model_path,
        "summary_path": summary_path,
        "epochs": int(epochs),
        "seq_len": int(config.seq_len),
        "pred_horizon": int(config.pred_horizon),
        "d_model": int(config.d_model),
        "n_heads": int(config.n_heads),
        "num_layers": int(config.num_layers),
        "d_ff": int(config.d_ff),
        "dropout": float(config.dropout),
        "batch_size": int(config.batch_size),
        "learning_rate": float(config.learning_rate),
        "weight_decay": float(config.weight_decay),
        "train_ratio": float(config.train_ratio),
        "val_ratio": float(config.val_ratio),
        "seed": int(config.seed),
        "enable_weight_sparsity": False,
        "enable_activation_sparsity": False,
        "enable_learnable_masks": False,
        "epochs_sparse": 0,
        "epochs_mask": 0,
        "target_weight_sparsity": float(config.target_weight_sparsity),
        "target_activation_sparsity": float(config.target_activation_sparsity),
        "lambda_mask_l1": float(config.lambda_mask_l1),
        "lambda_input_mask_l1": float(config.lambda_input_mask_l1),
    }


def build_config(
    station_id: str,
    forcing_dir: str,
    streamflow_dir: str,
    output_dir: str,
    device: str,
    seq_len: int,
    pred_horizon: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    d_model: int,
    n_heads: int,
    num_layers: int,
    d_ff: int,
    dropout: float,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    epochs_dense: int,
    target_weight_sparsity: float,
    target_activation_sparsity: float,
    lambda_mask_l1: float,
    lambda_input_mask_l1: float,
) -> ExperimentConfig:
    return ExperimentConfig(
        station_id=station_id,
        forcing_dir=forcing_dir,
        streamflow_dir=streamflow_dir,
        output_dir=output_dir,
        device=device,
        seq_len=seq_len,
        pred_horizon=pred_horizon,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        epochs_dense=epochs_dense,
        epochs_sparse=0,
        epochs_mask=0,
        enable_weight_sparsity=False,
        enable_activation_sparsity=False,
        enable_learnable_masks=False,
        target_weight_sparsity=target_weight_sparsity,
        target_activation_sparsity=target_activation_sparsity,
        lambda_mask_l1=lambda_mask_l1,
        lambda_input_mask_l1=lambda_input_mask_l1,
    )


def run_dense_trial(
    config: ExperimentConfig,
    epochs: int,
    prune_callback: Optional[Callable[[ExplainableSparseTransformer, object, int], None]] = None,
) -> float:
    model, data, _, _ = train_dense_model(config, epochs, prune_callback=prune_callback)
    device = torch.device(config.device)
    metrics = compute_loader_metrics(model, data.val_loader, device)
    val_nse = float(metrics.get("nse", float("nan")))
    if not np.isfinite(val_nse):
        return -1e9
    return val_nse


def run_bayes_opt(args, station_id: str, n_trials: int, timeout: Optional[int]) -> Dict[str, object]:
    try:
        import optuna
    except Exception as exc:
        raise RuntimeError("Optuna is required for Bayesian optimization. Please install optuna.") from exc

    def objective(trial: "optuna.trial.Trial") -> float:
        lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        d_model_candidates = [v for v in range(16, 257, 16) if v % max(1, args.n_heads) == 0]
        d_model = trial.suggest_categorical("d_model", d_model_candidates)
        num_layers = trial.suggest_int("num_layers", 1, 16)
        target_weight_sparsity = trial.suggest_float("target_weight_sparsity", 0.7, 0.95)
        target_activation_sparsity = trial.suggest_float("target_activation_sparsity", 0.6, 0.9)
        lambda_mask_l1 = trial.suggest_float("lambda_mask_l1", 1e-5, 1e-3, log=True)
        lambda_input_mask_l1 = trial.suggest_float("lambda_input_mask_l1", 1e-5, 1e-3, log=True)

        config = build_config(
            station_id=station_id,
            forcing_dir=args.forcing_dir,
            streamflow_dir=args.streamflow_dir,
            output_dir=os.path.join(args.output_root, station_id),
            device=args.device,
            seq_len=args.seq_len,
            pred_horizon=args.pred_horizon,
            batch_size=batch_size,
            learning_rate=lr,
            weight_decay=args.weight_decay,
            d_model=d_model,
            n_heads=args.n_heads,
            num_layers=num_layers,
            d_ff=args.d_ff,
            dropout=args.dropout,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed + trial.number,
            epochs_dense=args.epochs,
            target_weight_sparsity=target_weight_sparsity,
            target_activation_sparsity=target_activation_sparsity,
            lambda_mask_l1=lambda_mask_l1,
            lambda_input_mask_l1=lambda_input_mask_l1,
        )

        prune_callback = None
        if args.bo_prune:
            def prune_callback(model, data, epoch):
                device = torch.device(config.device)
                metrics = compute_loader_metrics(model, data.val_loader, device)
                val_nse = float(metrics.get("nse", float("nan")))
                trial.report(val_nse, step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        return run_dense_trial(config, epochs=args.epochs, prune_callback=prune_callback)

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=args.bo_prune_warmup) if args.bo_prune else None
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    output_dir = os.path.join(args.output_root, "bayes_opt", station_id)
    os.makedirs(output_dir, exist_ok=True)
    trials_path = os.path.join(output_dir, "dense_transformer_bo_trials.csv")
    study.trials_dataframe().to_csv(trials_path, index=False)

    summary = {
        "station_id": station_id,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "trials_csv": trials_path,
    }
    summary_path = os.path.join(output_dir, "dense_transformer_bo_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    summary["summary_path"] = summary_path
    return summary


def run_station_job(job: Dict[str, object]) -> Dict[str, object]:
    station_id = str(job["station_id"])
    epochs = int(job["epochs"])

    config = build_config(
        station_id=station_id,
        forcing_dir=str(job["forcing_dir"]),
        streamflow_dir=str(job["streamflow_dir"]),
        output_dir=str(job["output_dir"]),
        device=str(job["device"]),
        seq_len=int(job["seq_len"]),
        pred_horizon=int(job["pred_horizon"]),
        batch_size=int(job["batch_size"]),
        learning_rate=float(job["lr"]),
        weight_decay=float(job["weight_decay"]),
        d_model=int(job["d_model"]),
        n_heads=int(job["n_heads"]),
        num_layers=int(job["num_layers"]),
        d_ff=int(job["d_ff"]),
        dropout=float(job["dropout"]),
        train_ratio=float(job["train_ratio"]),
        val_ratio=float(job["val_ratio"]),
        seed=int(job["seed"]),
        epochs_dense=epochs,
        target_weight_sparsity=float(job["target_weight_sparsity"]),
        target_activation_sparsity=float(job["target_activation_sparsity"]),
        lambda_mask_l1=float(job["lambda_mask_l1"]),
        lambda_input_mask_l1=float(job["lambda_input_mask_l1"]),
    )

    try:
        return train_one_station(config=config, epochs=epochs)
    except Exception as e:
        return {
            "station_id": station_id,
            "status": "failed",
            "error": str(e),
            "epochs": epochs,
            "seq_len": int(job["seq_len"]),
            "pred_horizon": int(job["pred_horizon"]),
            "d_model": int(job["d_model"]),
            "n_heads": int(job["n_heads"]),
            "num_layers": int(job["num_layers"]),
            "d_ff": int(job["d_ff"]),
            "dropout": float(job["dropout"]),
            "batch_size": int(job["batch_size"]),
            "learning_rate": float(job["lr"]),
            "weight_decay": float(job["weight_decay"]),
            "train_ratio": float(job["train_ratio"]),
            "val_ratio": float(job["val_ratio"]),
            "seed": int(job["seed"]),
            "enable_weight_sparsity": False,
            "enable_activation_sparsity": False,
            "enable_learnable_masks": False,
            "epochs_sparse": 0,
            "epochs_mask": 0,
            "target_weight_sparsity": float(job["target_weight_sparsity"]),
            "target_activation_sparsity": float(job["target_activation_sparsity"]),
            "lambda_mask_l1": float(job["lambda_mask_l1"]),
            "lambda_input_mask_l1": float(job["lambda_input_mask_l1"]),
        }


def build_args():
    p = argparse.ArgumentParser(description="Run dense Transformer training for all stations (no sparse/mask).")
    p.add_argument("--forcing_dir", type=str, default="data/Forcing")
    p.add_argument("--streamflow_dir", type=str, default="data/Streamflow")
    p.add_argument("--output_root", type=str, default="outputs/transformer_dense")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seq_len", type=int, default=15)
    p.add_argument("--pred_horizon", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--workers", type=int, default=4, help="Number of parallel station workers")
    p.add_argument("--progress-heartbeat-sec", type=float, default=30.0, help="Heartbeat interval for parallel mode")

    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--d_ff", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--target_weight_sparsity", type=float, default=0.9)
    p.add_argument("--target_activation_sparsity", type=float, default=0.8)
    p.add_argument("--lambda_mask_l1", type=float, default=1e-4)
    p.add_argument("--lambda_input_mask_l1", type=float, default=1e-4)

    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--max_stations", type=int, default=0, help=">0 means only first N stations")
    p.add_argument("--require_streamflow", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--station_id", type=str, default="", help="Run single station id if set")

    p.add_argument("--bayes_opt", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--bo_trials", type=int, default=30)
    p.add_argument("--bo_timeout", type=int, default=0, help="Seconds; 0 disables timeout")
    p.add_argument("--bo_prune", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--bo_prune_warmup", type=int, default=2)
    p.add_argument("--bo_only", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--bo_per_station", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--bo_trials_per_station", type=int, default=5)

    p.add_argument(
        "--params_csv_name",
        type=str,
        default="dense_transformer_station_params.csv",
        help="Per-station parameter summary CSV file name under output_root",
    )
    return p.parse_args()


def main():
    args = build_args()
    set_seed(args.seed)

    if not os.path.exists(args.forcing_dir):
        raise FileNotFoundError(f"forcing_dir not found: {args.forcing_dir}")

    station_ids = list_station_ids(
        forcing_dir=args.forcing_dir,
        streamflow_dir=args.streamflow_dir,
        require_streamflow=args.require_streamflow,
    )

    if args.station_id:
        station_ids = [s for s in station_ids if s == args.station_id]
    if args.max_stations > 0:
        station_ids = station_ids[: args.max_stations]

    if len(station_ids) == 0:
        raise RuntimeError("No stations found to run.")

    os.makedirs(args.output_root, exist_ok=True)

    timeout = None if int(args.bo_timeout) <= 0 else int(args.bo_timeout)
    if args.bayes_opt and args.bo_per_station:
        for station_id in station_ids:
            run_bayes_opt(args, station_id, n_trials=int(args.bo_trials_per_station), timeout=timeout)
        if args.bo_only:
            return
    elif args.bayes_opt:
        bo_station_id = args.station_id or station_ids[0]
        bo_summary = run_bayes_opt(args, bo_station_id, n_trials=int(args.bo_trials), timeout=timeout)
        best_params = bo_summary.get("best_params", {})
        args.lr = float(best_params.get("learning_rate", args.lr))
        args.batch_size = int(best_params.get("batch_size", args.batch_size))
        args.d_model = int(best_params.get("d_model", args.d_model))
        args.num_layers = int(best_params.get("num_layers", args.num_layers))
        args.target_weight_sparsity = float(best_params.get("target_weight_sparsity", args.target_weight_sparsity))
        args.target_activation_sparsity = float(best_params.get("target_activation_sparsity", args.target_activation_sparsity))
        args.lambda_mask_l1 = float(best_params.get("lambda_mask_l1", args.lambda_mask_l1))
        args.lambda_input_mask_l1 = float(best_params.get("lambda_input_mask_l1", args.lambda_input_mask_l1))
        if args.bo_only:
            return

    jobs: List[Dict[str, object]] = []
    for station_id in station_ids:
        jobs.append(
            {
                "station_id": station_id,
                "forcing_dir": args.forcing_dir,
                "streamflow_dir": args.streamflow_dir,
                "output_dir": os.path.join(args.output_root, station_id),
                "device": args.device,
                "epochs": int(args.epochs),
                "seq_len": int(args.seq_len),
                "pred_horizon": int(args.pred_horizon),
                "batch_size": int(args.batch_size),
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "d_model": int(args.d_model),
                "n_heads": int(args.n_heads),
                "num_layers": int(args.num_layers),
                "d_ff": int(args.d_ff),
                "dropout": float(args.dropout),
                "train_ratio": float(args.train_ratio),
                "val_ratio": float(args.val_ratio),
                "seed": int(args.seed),
                "target_weight_sparsity": float(args.target_weight_sparsity),
                "target_activation_sparsity": float(args.target_activation_sparsity),
                "lambda_mask_l1": float(args.lambda_mask_l1),
                "lambda_input_mask_l1": float(args.lambda_input_mask_l1),
            }
        )

    all_rows: List[Dict[str, object]] = []
    workers = max(1, int(args.workers))
    if workers == 1:
        for i, job in enumerate(jobs, start=1):
            print(f"[{i}/{len(jobs)}] training station {job['station_id']} -> {job['output_dir']}", flush=True)
            all_rows.append(run_station_job(job))
    else:
        print(f"Parallel mode enabled: workers={workers}", flush=True)
        heartbeat_sec = max(1.0, float(args.progress_heartbeat_sec))
        start_ts = time.time()
        ex = ProcessPoolExecutor(max_workers=workers)
        interrupted = False
        try:
            futures = [ex.submit(run_station_job, job) for job in jobs]
            total = len(futures)
            pending = set(futures)
            completed = 0

            while pending:
                done, pending = wait(pending, timeout=heartbeat_sec, return_when=FIRST_COMPLETED)
                if not done:
                    elapsed_min = (time.time() - start_ts) / 60.0
                    print(
                        f"[dense] heartbeat: {completed}/{total} completed, {len(pending)} in-flight, elapsed={elapsed_min:.1f} min",
                        flush=True,
                    )
                    continue

                for fut in done:
                    all_rows.append(fut.result())
                    completed += 1

                if completed == total or completed % max(1, total // 100) == 0:
                    print(f"[dense] progress: {completed}/{total} ({completed / total:.1%})", flush=True)
        except KeyboardInterrupt:
            interrupted = True
            print("\nInterrupted. Cancelling pending dense Transformer jobs...", flush=True)
            ex.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            if not interrupted:
                ex.shutdown(wait=True)

    summary_csv = os.path.join(args.output_root, args.params_csv_name)
    pd.DataFrame(all_rows).to_csv(summary_csv, index=False)

    summary_json = os.path.join(args.output_root, "dense_transformer_run_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)

    ok = sum(1 for r in all_rows if r.get("status") == "ok")
    failed = len(all_rows) - ok
    print(json.dumps({
        "total": len(all_rows),
        "ok": ok,
        "failed": failed,
        "params_csv": summary_csv,
        "summary_json": summary_json,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
