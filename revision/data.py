"""New protocol: split by target date, preserve calendar gaps, fit scalers on train only.

Legacy checkpoints must continue to use their original preprocessing. This loader
is deliberately separate so that a new scaler cannot silently change old results.
"""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd


def is_q(name):
    name = name.lower().replace("-", "_")
    return name in {"q", "flow", "target_q", "runoff", "q_obs", "q_input"} or any(
        t in name for t in ("streamflow", "runoff", "discharge", "upstream_q", "q_up"))


def read_daily(path):
    df = pd.read_csv(path)
    date_col = next((c for c in df if "date" in c.lower()), None)
    if date_col is None:
        raise ValueError(f"A dated daily CSV is required: {path}")
    dates = pd.to_datetime(df.pop(date_col), errors="raise")
    df.index = dates
    if df.index.has_duplicates:
        raise ValueError(f"Duplicate dates: {path}")
    return df.sort_index()


def load_basin(root, station, cfg):
    root = Path(root)
    forcing_path = root / "Forcing" / f"{station}.csv"
    q_path = root / "Streamflow" / f"{station}.csv"
    forcing, flow = read_daily(forcing_path), read_daily(q_path)
    q_col = next((c for c in flow if is_q(c)), None)
    if q_col is None:
        raise ValueError(f"Cannot identify observed streamflow in {q_path}")
    # Remove ALL Q copies from forcing, including CAMELS streamflow_cfs.
    # With-Q has exactly one history channel, appended from the target source.
    numeric = forcing.select_dtypes(include="number")
    excluded = {"year", "month", "day", "doy", "gauge_id", "station_id", "qc_flag"}
    names = [c for c in numeric if not is_q(c) and c.lower() not in excluded
             and not c.lower().startswith("unnamed:")]
    if not names:
        raise ValueError("No meteorological forcing features")
    start, end = max(forcing.index.min(), flow.index.min()), min(forcing.index.max(), flow.index.max())
    calendar = pd.date_range(start, end, freq="D")
    frame = forcing[names].reindex(calendar).replace([np.inf, -np.inf], np.nan)
    q = pd.to_numeric(flow[q_col], errors="coerce").reindex(calendar).to_numpy(dtype=float, copy=True)
    # Negative observed discharge is a missing-value sentinel, never a target.
    q[~np.isfinite(q) | (q < 0)] = np.nan
    with_q = cfg["with_q"]
    if with_q:
        frame["target_q"] = q
    values = frame.to_numpy(float)
    n = len(calendar)
    train_end, val_end = int(n * cfg["train_ratio"]), int(n * (cfg["train_ratio"] + cfg["val_ratio"]))
    if not 0 < train_end < val_end < n:
        raise ValueError("Invalid chronological split")
    mean, std = np.nanmean(values[:train_end], axis=0), np.nanstd(values[:train_end], axis=0)
    qmean, qstd = np.nanmean(q[:train_end]), np.nanstd(q[:train_end])
    if not np.isfinite(mean).all() or not np.isfinite(qmean):
        raise ValueError("Training period has an entirely missing feature or target")
    std = np.maximum(std, 1e-6)
    qstd = max(float(qstd), 1e-6)
    scaled = (values - mean) / std
    seq, horizon = cfg["seq_len"], cfg["pred_horizon"]
    masks = {"train": [], "val": [], "test": []}
    xs, ys, histories, target_indices = [], [], [], []
    # Optional ORIGINAL-observation flags; the aggregate imputation report
    # cannot identify individual filled days and is not accepted as a mask.
    mask_root = cfg.get("observation_mask_dir")
    observed = np.ones(n, bool)
    if mask_root:
        mask = read_daily(Path(mask_root) / f"{station}.csv")
        if "q_observed" not in mask or not mask.q_observed.dropna().isin([0, 1]).all():
            raise ValueError("Observation mask must contain numeric q_observed=0/1")
        observed = mask.q_observed.reindex(calendar).fillna(0).to_numpy() == 1
    for t in range(seq + horizon - 1, n):
        stop = t - horizon + 1
        begin = stop - seq
        x = scaled[begin:stop]
        # Same target dates for with-Q, no-Q, LSTM and Q-history references.
        # Q availability defines paired eligibility; no-Q never receives Q values.
        if not np.isfinite(x).all() or not np.isfinite(q[t]) or not np.isfinite(q[begin:stop]).all():
            continue
        split = "train" if t < train_end else "val" if t < val_end else "test"
        if split != "train" and not observed[t]:
            continue
        if split != "train" and cfg.get("exclude_imputed_q_history", False):
            if not mask_root:
                raise ValueError("exclude_imputed_q_history requires original observation masks")
            if not observed[begin:stop].all():
                continue
        idx = len(xs)
        xs.append(x.astype(np.float32))
        ys.append((q[t] - qmean) / qstd)
        histories.append(q[begin:stop].copy())
        target_indices.append(t)
        masks[split].append(idx)
    if any(not ids for ids in masks.values()):
        raise ValueError(f"Empty split in {station}; inspect gaps, dates and observation flags")
    xs, ys, histories = np.array(xs), np.array(ys, np.float32), np.array(histories)
    dates = calendar[np.array(target_indices)].strftime("%Y-%m-%d").to_numpy()
    splits = {k: {"x": xs[ids], "y": ys[ids], "q_history": histories[ids], "dates": dates[ids]}
              for k, ids in masks.items()}
    metadata = {
        "station_id": station, "feature_names": list(frame), "q_source_column": q_col,
        "x_mean": mean.tolist(), "x_std": std.tolist(), "q_mean": float(qmean), "q_std": qstd,
        "train_end_exclusive": str(calendar[train_end].date()),
        "validation_end_exclusive": str(calendar[val_end].date()),
        "counts": {k: len(v["y"]) for k, v in splits.items()},
        "observation_mask_supplied": bool(mask_root),
        "target_status": "original_observations_only" if mask_root else "provided_CSV_may_include_imputation",
        "source_sha256": {str(p.name) + ":" + p.parent.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (forcing_path, q_path)},
    }
    if mask_root:
        p = Path(mask_root) / f"{station}.csv"
        metadata["observation_mask_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    return splits, metadata
