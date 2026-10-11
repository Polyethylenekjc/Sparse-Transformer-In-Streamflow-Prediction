import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats

try:
    from rich.progress import track
except Exception:
    track = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.flood_transformer.model import ExplainableSparseTransformer


RAIN_TOKENS = ("prcp", "precip", "rain", "rainfall", "p_")
TEMP_TOKENS = ("temp", "tmin", "tmax", "tmean")
EXPECTED_VOLUME_RELAX_RATIO = 0.01
EXPECTED_MSE_RELAX_RATIO = 0.02


def _preferred_output_root() -> str:
    candidates = [
        os.environ.get("SPARSE_TRANSFORMER_OUTPUT_ROOT", "").strip(),
        "/mnt/d/Store/Sparse_Transformer",
        "/mnt/d/Store/Sparse-Transformer",
        "outputs",
    ]
    for p in candidates:
        if p and os.path.isdir(p):
            return p
    return "outputs"


def _safe_read_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _station_dirs(output_root: str) -> List[Path]:
    root = Path(output_root)
    if not root.exists():
        return []
    out = []
    for p in root.iterdir():
        if p.is_dir() and p.name.isdigit():
            out.append(p)
    return sorted(out, key=lambda x: x.name)


def _norm_name(name: str) -> str:
    return str(name).strip().lower()


def _is_rain(name: str) -> bool:
    n = _norm_name(name)
    return any(tok in n for tok in RAIN_TOKENS)


def _is_temp(name: str) -> bool:
    n = _norm_name(name)
    return any(tok in n for tok in TEMP_TOKENS)


def _find_feature_idx(feature_names: List[str], pred) -> Optional[int]:
    for i, n in enumerate(feature_names):
        if pred(n):
            return i
    return None


def _find_temp_indices(feature_names: List[str]) -> List[int]:
    idx = []
    for i, n in enumerate(feature_names):
        if _is_temp(n):
            idx.append(i)
    return idx


def _is_streamflow_history(name: str) -> bool:
    n = _norm_name(name)
    if n in {"q", "flow", "streamflow", "streamflow_cfs", "q_input", "q_obs", "runoff"}:
        return True
    return ("q_" in n) or ("streamflow" in n) or ("runoff" in n)


def _parse_state_dict(path: Path) -> Optional[Dict[str, torch.Tensor]]:
    try:
        sd = torch.load(path, map_location="cpu")
    except Exception:
        return None
    if isinstance(sd, dict):
        return sd
    return None


def _infer_model_hparams(state_dict: Dict[str, torch.Tensor], input_dim: int):
    d_model = int(state_dict["input_proj.weight"].shape[0])
    d_ff = int(state_dict["layers.0.mlp.fc1.weight"].shape[0])

    layer_ids = set()
    for k in state_dict.keys():
        if k.startswith("layers."):
            parts = k.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                layer_ids.add(int(parts[1]))
    num_layers = max(layer_ids) + 1 if layer_ids else 1

    n_heads = int(state_dict["layers.0.attn.head_mask_logits"].shape[0])

    return {
        "input_dim": input_dim,
        "d_model": d_model,
        "n_heads": n_heads,
        "num_layers": num_layers,
        "d_ff": d_ff,
    }


def _build_model_from_state_dict(state_dict: Dict[str, torch.Tensor], input_dim: int) -> ExplainableSparseTransformer:
    hp = _infer_model_hparams(state_dict, input_dim)
    model = ExplainableSparseTransformer(
        input_dim=hp["input_dim"],
        d_model=hp["d_model"],
        n_heads=hp["n_heads"],
        num_layers=hp["num_layers"],
        d_ff=hp["d_ff"],
        dropout=0.0,
        task_type="regression",
        enable_activation_sparsity=True,
        activation_sparsity_ratio=0.8,
        enable_learnable_masks=True,
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def _resolve_device(device_arg: str) -> torch.device:
    req = str(device_arg).strip().lower()
    if req == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("CUDA requested but not available; falling back to CPU")
        return torch.device("cpu")
    return torch.device("cpu")


def _predict(model: ExplainableSparseTransformer, x: np.ndarray, device: torch.device, batch_size: int = 512) -> np.ndarray:
    outs: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = torch.tensor(x[i : i + batch_size], dtype=torch.float32, device=device)
            out = model(xb, return_intermediates=False)
            outs.append(out.pred.detach().cpu().numpy())
    return np.concatenate(outs, axis=0)


def _load_test_arrays(station_dir: Path):
    path = station_dir / "test_arrays.npz"
    if not path.exists():
        return None
    arr = np.load(path, allow_pickle=True)
    if not {"x", "y"}.issubset(set(arr.files)):
        return None
    x = np.asarray(arr["x"], dtype=np.float32)
    y = np.asarray(arr["y"], dtype=np.float32)
    return x, y


def _maybe_parse_date(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        lc = c.lower()
        if "date" in lc or "time" in lc:
            out = df.copy()
            out[c] = pd.to_datetime(out[c], errors="coerce")
            if out[c].notna().mean() > 0.8:
                return out.sort_values(c).reset_index(drop=True)
    return df.reset_index(drop=True)


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    for c in df.columns:
        low = c.lower()
        for cand in candidates:
            if len(cand) >= 3 and cand.lower() in low:
                return c
    return None


def _load_scalers_from_raw(
    station_id: str,
    feature_names: List[str],
    forcing_dir: str,
    streamflow_dir: str,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    fpath = Path(forcing_dir) / f"{station_id}.csv"
    qpath = Path(streamflow_dir) / f"{station_id}.csv"

    if not fpath.exists() or not qpath.exists():
        raise FileNotFoundError("forcing/streamflow file not found")

    df_f = _maybe_parse_date(pd.read_csv(fpath))
    df_q = _maybe_parse_date(pd.read_csv(qpath))

    q_col = _find_col(df_q, ["Q", "streamflow", "flow", "streamflow_cfs"])
    if q_col is None:
        numeric = [c for c in df_q.columns if pd.api.types.is_numeric_dtype(df_q[c])]
        if not numeric:
            raise ValueError("streamflow file has no numeric column")
        q_col = numeric[0]

    target_col = "target_q"
    df_q = df_q.rename(columns={q_col: target_col})

    date_f = _find_col(df_f, ["date", "time"])
    date_q = _find_col(df_q, ["date", "time"])
    if date_f is not None and date_q is not None:
        merged = df_f.merge(df_q, left_on=date_f, right_on=date_q, how="inner", suffixes=("", "_q"))
    else:
        n = min(len(df_f), len(df_q))
        merged = pd.concat([df_f.iloc[:n].reset_index(drop=True), df_q.iloc[:n].reset_index(drop=True)], axis=1)

    keep = list(dict.fromkeys(feature_names + [target_col]))
    missing = [c for c in keep if c not in merged.columns]
    if missing:
        raise ValueError(f"missing columns in merged data: {missing}")

    clean = merged[keep].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    feat = clean[feature_names].to_numpy(dtype=np.float32)
    q = clean[target_col].to_numpy(dtype=np.float32)

    x_mean = feat.mean(axis=0)
    x_std = feat.std(axis=0) + 1e-6
    q_mean = float(q.mean())
    q_std = float(q.std() + 1e-6)
    return x_mean, x_std, q_mean, q_std


def _get_scalers_with_fallback(
    station_id: str,
    feature_names: List[str],
    forcing_dir: str,
    streamflow_dir: str,
    x: np.ndarray,
    y: np.ndarray,
):
    try:
        x_mean, x_std, q_mean, q_std = _load_scalers_from_raw(station_id, feature_names, forcing_dir, streamflow_dir)
    except Exception:
        # Fallback keeps relative deltas meaningful even when raw forcing is unavailable.
        x_mean = np.zeros(x.shape[2], dtype=np.float32)
        x_std = np.ones(x.shape[2], dtype=np.float32)
        q_mean = 0.0
        q_std = 1.0
    if len(x_mean) != x.shape[2]:
        x_mean = np.zeros(x.shape[2], dtype=np.float32)
        x_std = np.ones(x.shape[2], dtype=np.float32)
    return x_mean.astype(np.float32), x_std.astype(np.float32), float(q_mean), float(q_std)


def _denorm_x(x_norm: np.ndarray, x_mean: np.ndarray, x_std: np.ndarray) -> np.ndarray:
    return x_norm * x_std[None, None, :] + x_mean[None, None, :]


def _renorm_x(x_phys: np.ndarray, x_mean: np.ndarray, x_std: np.ndarray) -> np.ndarray:
    return (x_phys - x_mean[None, None, :]) / x_std[None, None, :]


def _series_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = float(np.mean((y_pred - y_true) ** 2))
    peak_t = int(np.argmax(y_pred)) if len(y_pred) > 0 else -1
    volume = float(np.sum(y_pred))
    return {
        "mse": mse,
        "peak_time_idx": peak_t,
        "volume": volume,
    }


def _rain_pattern_weights(seq_len: int, mode: str) -> np.ndarray:
    idx = np.arange(seq_len, dtype=np.float32)
    if mode == "front_loaded":
        w = np.exp(-idx / max(1.0, seq_len / 5.0))
    elif mode == "back_loaded":
        rev = (seq_len - 1) - idx
        w = np.exp(-rev / max(1.0, seq_len / 5.0))
    elif mode == "middle_loaded":
        c = (seq_len - 1) / 2.0
        sigma = max(1.0, seq_len / 6.0)
        w = np.exp(-0.5 * ((idx - c) / sigma) ** 2)
    elif mode == "uniform":
        w = np.ones(seq_len, dtype=np.float32)
    else:
        raise ValueError(f"unknown mode: {mode}")
    s = float(w.sum())
    if s <= 0:
        return np.ones(seq_len, dtype=np.float32) / float(seq_len)
    return w / s


def _apply_intervention(
    x_base: np.ndarray,
    feature_names: List[str],
    x_mean: np.ndarray,
    x_std: np.ndarray,
    scenario: str,
    value: float,
) -> np.ndarray:
    x_phys = _denorm_x(x_base, x_mean, x_std)
    rain_idx = _find_feature_idx(feature_names, _is_rain)
    temp_idx = _find_temp_indices(feature_names)

    if scenario.startswith("rain_scale"):
        if rain_idx is None:
            return x_base.copy()
        factor = float(value)
        rain = np.maximum(0.0, x_phys[:, :, rain_idx])
        x_phys[:, :, rain_idx] = rain * factor
        return _renorm_x(x_phys, x_mean, x_std).astype(np.float32)

    if scenario.startswith("rain_shift"):
        if rain_idx is None:
            return x_base.copy()
        shift_days = int(value)
        rain = np.maximum(0.0, x_phys[:, :, rain_idx])
        rain_new = np.zeros_like(rain)
        seq_len = rain.shape[1]
        if abs(shift_days) < seq_len:
            if shift_days > 0:
                rain_new[:, shift_days:] = rain[:, : seq_len - shift_days]
            elif shift_days < 0:
                s = -shift_days
                rain_new[:, : seq_len - s] = rain[:, s:]
            else:
                rain_new = rain.copy()
        x_phys[:, :, rain_idx] = rain_new
        return _renorm_x(x_phys, x_mean, x_std).astype(np.float32)

    if scenario.startswith("streamflow_scaling") or scenario.startswith("runoff_scaling"):
        streamflow_idx = _find_feature_idx(feature_names, _is_streamflow_history)
        if streamflow_idx is None:
            return x_base.copy()
        factor = float(value)
        streamflow_hist = x_phys[:, :, streamflow_idx]
        x_phys[:, :, streamflow_idx] = streamflow_hist * factor
        return _renorm_x(x_phys, x_mean, x_std).astype(np.float32)

    if scenario.startswith("temp_plus"):
        if not temp_idx:
            return x_base.copy()
        delta_c = float(value)
        for ti in temp_idx:
            x_phys[:, :, ti] = x_phys[:, :, ti] + delta_c
        return _renorm_x(x_phys, x_mean, x_std).astype(np.float32)

    if scenario.startswith("rain_pattern"):
        if rain_idx is None:
            return x_base.copy()
        mode = str(value)
        w = _rain_pattern_weights(x_phys.shape[1], mode=mode)
        rain = np.maximum(0.0, x_phys[:, :, rain_idx])
        total = rain.sum(axis=1, keepdims=True)
        x_phys[:, :, rain_idx] = total * w[None, :]
        return _renorm_x(x_phys, x_mean, x_std).astype(np.float32)

    return x_base.copy()


def _build_scenarios() -> List[Tuple[str, object]]:
    out: List[Tuple[str, object]] = []

    for f in [0.8, 0.9, 1.1, 1.2]:
        out.append(("rain_scale", f))

    for f in [0.1, 0.5, 0.9, 1.1, 1.5, 1.9]:
        out.append(("streamflow_scaling", f))

    for dt in [3.0, 5.0, 7.0]:
        out.append(("temp_plus", dt))

    for mode in ["front_loaded", "middle_loaded", "back_loaded", "uniform"]:
        out.append(("rain_pattern", mode))

    return out


def _scenario_name(kind: str, value) -> str:
    if kind == "rain_scale":
        return f"rain_scale_{value:.2f}".replace(".", "p")
    if kind == "streamflow_scaling":
        return f"streamflow_scaling_{value:.2f}x".replace(".", "p")
    if kind == "temp_plus":
        return f"temp_plus_{value:.1f}c".replace(".", "p")
    return f"rain_pattern_{value}"


def run_station_counterfactual(
    station_dir: Path,
    forcing_dir: str,
    streamflow_dir: str,
    device: torch.device,
) -> Optional[pd.DataFrame]:
    sid = station_dir.name

    arrays = _load_test_arrays(station_dir)
    if arrays is None:
        return None
    x_base, y = arrays

    summary = _safe_read_json(station_dir / "summary.json") or {}
    station_nse = float((summary.get("metrics") or {}).get("nse", np.nan))
    nse_group = "NSE<0.5" if np.isfinite(station_nse) and station_nse < 0.5 else "NSE>0.5"
    feature_names = summary.get("feature_names", [])
    if not isinstance(feature_names, list) or len(feature_names) != x_base.shape[2]:
        feature_names = [f"feature_{i}" for i in range(x_base.shape[2])]

    sd = _parse_state_dict(station_dir / "sparse_transformer.pt")
    if sd is None:
        return None

    model = _build_model_from_state_dict(sd, input_dim=x_base.shape[2])
    model = model.to(device)
    pred_base = _predict(model, x_base, device=device)

    x_mean, x_std, q_mean, q_std = _get_scalers_with_fallback(
        station_id=sid,
        feature_names=feature_names,
        forcing_dir=forcing_dir,
        streamflow_dir=streamflow_dir,
        x=x_base,
        y=y,
    )

    y_raw = y * q_std + q_mean
    p_base_raw = pred_base * q_std + q_mean
    base_m = _series_metrics(y_raw, p_base_raw)

    rows = []
    scenarios = _build_scenarios()
    for kind, value in scenarios:
        x_cf = _apply_intervention(x_base, feature_names, x_mean, x_std, kind, value)
        p_cf = _predict(model, x_cf, device=device)
        p_cf_raw = p_cf * q_std + q_mean
        m = _series_metrics(y_raw, p_cf_raw)

        row = {
            "station_id": sid,
            "station_nse": station_nse,
            "nse_group": nse_group,
            "scenario": _scenario_name(kind, value),
            "scenario_kind": kind,
            "scenario_value": str(value),
            "base_mse": base_m["mse"],
            "mse": m["mse"],
            "delta_mse": m["mse"] - base_m["mse"],
            "base_peak_time_idx": base_m["peak_time_idx"],
            "peak_time_idx": m["peak_time_idx"],
            "delta_peak_time": int(m["peak_time_idx"] - base_m["peak_time_idx"]),
            "base_volume": base_m["volume"],
            "volume": m["volume"],
            "delta_volume": m["volume"] - base_m["volume"],
        }

        expected_behavior = False
        expected_rule = ""
        if kind == "rain_scale":
            factor = float(value)
            expected_rule = "rain_scale_sign_volume"
            vol_tol = EXPECTED_VOLUME_RELAX_RATIO * max(abs(float(row["base_volume"])), 1.0)
            if factor > 1.0:
                expected_behavior = row["delta_volume"] > -vol_tol
            elif factor < 1.0:
                expected_behavior = row["delta_volume"] < vol_tol
            else:
                expected_behavior = abs(row["delta_volume"]) <= vol_tol
        elif kind == "streamflow_scaling":
            factor = float(value)
            vol_tol = EXPECTED_VOLUME_RELAX_RATIO * max(abs(float(row["base_volume"])), 1.0)
            expected_rule = "streamflow_scaling_volume_direction"
            if factor > 1.0:
                expected_behavior = row["delta_volume"] > -vol_tol
            elif factor < 1.0:
                expected_behavior = row["delta_volume"] < vol_tol
            else:
                expected_behavior = abs(row["delta_volume"]) <= vol_tol
        elif kind == "temp_plus":
            mse_tol = EXPECTED_MSE_RELAX_RATIO * max(abs(float(row["base_mse"])), 1.0)
            expected_rule = f"temp_perturbation_delta_mse_gt_-{EXPECTED_MSE_RELAX_RATIO:.2f}|base|"
            expected_behavior = row["delta_mse"] > -mse_tol
        elif kind == "rain_pattern":
            mse_tol = EXPECTED_MSE_RELAX_RATIO * max(abs(float(row["base_mse"])), 1.0)
            expected_rule = f"rain_pattern_delta_mse_gt_-{EXPECTED_MSE_RELAX_RATIO:.2f}|base|"
            expected_behavior = row["delta_mse"] > -mse_tol

        row["expected_behavior"] = bool(expected_behavior)
        row["expected_rule"] = expected_rule

        if kind == "temp_plus":
            if row["delta_volume"] > 0:
                row["temp_conflict_direction"] = "snowmelt_dominant"
            elif row["delta_volume"] < 0:
                row["temp_conflict_direction"] = "evap_dominant"
            else:
                row["temp_conflict_direction"] = "balanced"
            row["temp_conflict_strength"] = abs(float(row["delta_volume"]))
        else:
            row["temp_conflict_direction"] = ""
            row["temp_conflict_strength"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def _bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    if valid.sum() == 0:
        return q

    pv = p[valid]
    m = pv.size
    order = np.argsort(pv)
    ranks = np.arange(1, m + 1, dtype=float)
    q_sorted = pv[order] * (m / ranks)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)

    idx_valid = np.where(valid)[0]
    q[idx_valid[order]] = q_sorted
    return q


def _bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=42):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    n = v.size
    samples = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        s = v[rng.integers(0, n, n)]
        samples[i] = np.median(s)

    lo = np.percentile(samples, 100 * (alpha / 2.0))
    hi = np.percentile(samples, 100 * (1.0 - alpha / 2.0))
    return float(lo), float(hi)


def _one_sample_test(values, label: str, alternative: str, n_boot=2000, alpha=0.05, seed=42):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = int(v.size)

    out = {
        "metric": label,
        "n": n,
        "mean": float(np.mean(v)) if n else float("nan"),
        "median": float(np.median(v)) if n else float("nan"),
        "positive_ratio": float((v > 0).mean()) if n else float("nan"),
        "wilcoxon_stat": float("nan"),
        "p_value": float("nan"),
        "ci_median_low": float("nan"),
        "ci_median_high": float("nan"),
    }

    if n == 0:
        return out

    nz = v[np.abs(v) > 0]
    if nz.size > 0:
        try:
            w = stats.wilcoxon(nz, zero_method="wilcox", alternative=alternative)
            out["wilcoxon_stat"] = float(w.statistic)
            out["p_value"] = float(w.pvalue)
        except Exception:
            pass

    ci_lo, ci_hi = _bootstrap_ci(v, n_boot=n_boot, alpha=alpha, seed=seed)
    out["ci_median_low"] = ci_lo
    out["ci_median_high"] = ci_hi
    return out


def summarize_counterfactual(station_df: pd.DataFrame, n_boot=2000, alpha=0.05, seed=42):
    rows = []

    for sc, grp in station_df.groupby("scenario", sort=True):
        kind = str(grp["scenario_kind"].iloc[0])

        rows.append(
            _one_sample_test(
                grp["delta_mse"].to_numpy(),
                label=f"{sc}__delta_mse_gt_0",
                alternative="greater",
                n_boot=n_boot,
                alpha=alpha,
                seed=seed,
            )
        )

        alt_peak = "two-sided"
        rows.append(
            _one_sample_test(
                grp["delta_peak_time"].to_numpy(),
                label=f"{sc}__delta_peak_time",
                alternative=alt_peak,
                n_boot=n_boot,
                alpha=alpha,
                seed=seed,
            )
        )

        rows.append(
            _one_sample_test(
                grp["delta_volume"].to_numpy(),
                label=f"{sc}__delta_volume",
                alternative="two-sided",
                n_boot=n_boot,
                alpha=alpha,
                seed=seed,
            )
        )

    summary = pd.DataFrame(rows)
    summary["q_value_bh"] = _bh_fdr(summary["p_value"].to_numpy())

    temp_rows = station_df[station_df["scenario_kind"] == "temp_plus"]
    extra = {
        "n_stations": int(station_df["station_id"].nunique()),
        "n_records": int(len(station_df)),
        "n_temp_records": int(len(temp_rows)),
        "temp_snowmelt_dominant_ratio": float((temp_rows["temp_conflict_direction"] == "snowmelt_dominant").mean()) if len(temp_rows) else float("nan"),
        "temp_evap_dominant_ratio": float((temp_rows["temp_conflict_direction"] == "evap_dominant").mean()) if len(temp_rows) else float("nan"),
    }
    return summary, extra


def _run_r_plotter(
    station_csv: Path,
    out_dir: Path,
    plot_script: Path,
    rscript_bin: str,
    prefix: str,
) -> Tuple[bool, str]:
    if not plot_script.exists():
        return False, f"plot script not found: {plot_script}"

    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        rscript_bin,
        str(plot_script),
        "--station_csv",
        str(station_csv),
        "--out_dir",
        str(out_dir),
        "--prefix",
        str(prefix),
    ]
    try:
        p = subprocess.run(cmd, check=True, capture_output=True, text=True)
        msg = (p.stdout or "").strip()
        return True, msg
    except FileNotFoundError:
        return False, f"Rscript not found: {rscript_bin}"
    except subprocess.CalledProcessError as e:
        detail = (e.stdout or "") + "\n" + (e.stderr or "")
        return False, detail.strip()


def _safe_read_station_table(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if len(df) < 1:
        return None
    if "station_id" not in df.columns:
        return None
    return df


def _expected_scenario_count() -> int:
    return len(_build_scenarios())


def _completed_station_ids(station_df: pd.DataFrame) -> set[str]:
    if station_df is None or station_df.empty:
        return set()
    need_cols = {"station_id", "scenario"}
    if not need_cols.issubset(set(station_df.columns)):
        return set()

    expected = _expected_scenario_count()
    counts = station_df.groupby("station_id")["scenario"].nunique(dropna=True)
    done = counts[counts >= expected].index.astype(str)
    return set(done.tolist())


def _merge_station_tables(base_df: Optional[pd.DataFrame], new_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if base_df is None or base_df.empty:
        merged = new_df.copy() if (new_df is not None) else pd.DataFrame()
    elif new_df is None or new_df.empty:
        merged = base_df.copy()
    else:
        merged = pd.concat([base_df, new_df], ignore_index=True)

    if merged.empty:
        return merged

    if {"station_id", "scenario"}.issubset(set(merged.columns)):
        merged = merged.drop_duplicates(subset=["station_id", "scenario"], keep="last")
    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Counterfactual physical sensitivity analysis over station outputs"
    )
    parser.add_argument("--output-root", type=str, default=None, help="outputs root; auto-detect mnt path if omitted")
    parser.add_argument("--forcing-dir", type=str, default="data/Forcing")
    parser.add_argument("--streamflow-dir", type=str, default="data/Streamflow")
    parser.add_argument(
        "--station-out",
        type=str,
        default=None,
        help="station-level output table",
    )
    parser.add_argument(
        "--summary-out",
        type=str,
        default=None,
        help="hypothesis-level summary output table",
    )
    parser.add_argument(
        "--extra-out",
        type=str,
        default=None,
        help="extra summary JSON",
    )
    parser.add_argument("--n-boot", type=int, default=2000, help="bootstrap samples")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="debug: only run first N stations")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="inference device")
    parser.add_argument("--no-rich", action="store_true", help="disable rich progress bar")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="resume from existing station table and skip completed stations",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=20,
        help="write station table checkpoint every N successful stations (0 to disable)",
    )
    parser.add_argument(
        "--auto-plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically generate publication-style figures via R",
    )
    parser.add_argument(
        "--plot-script",
        type=str,
        default="scripts/plot_counterfactual_sensitivity.R",
        help="R plotting script path",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="",
        help="Output directory for figures; default is <station_out_parent>/figures",
    )
    parser.add_argument(
        "--plot-prefix",
        type=str,
        default="Fig_counterfactual",
        help="Figure filename prefix",
    )
    parser.add_argument(
        "--rscript-bin",
        type=str,
        default="Rscript",
        help="Rscript executable name or path",
    )
    args = parser.parse_args()

    output_root = args.output_root or _preferred_output_root()
    station_out_path = args.station_out or os.path.join(output_root, "physical_sensitivity_station_table.csv")
    summary_out_path = args.summary_out or os.path.join(output_root, "physical_sensitivity_summary.csv")
    extra_out_path = args.extra_out or os.path.join(output_root, "physical_sensitivity_extra.json")

    station_out = Path(station_out_path)
    summary_out = Path(summary_out_path)
    extra_out = Path(extra_out_path)
    station_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    extra_out.parent.mkdir(parents=True, exist_ok=True)

    stations = _station_dirs(output_root)
    if args.limit and args.limit > 0:
        stations = stations[: int(args.limit)]

    if not stations:
        raise SystemExit(f"No station directories found under: {output_root}")

    existing_df = _safe_read_station_table(station_out) if args.resume else None
    done_ids = _completed_station_ids(existing_df) if existing_df is not None else set()
    pending = [sd for sd in stations if sd.name not in done_ids]

    print(f"Discovered stations: {len(stations)}")
    print(f"Completed in cache: {len(done_ids)}")
    print(f"Pending stations: {len(pending)}")

    device = _resolve_device(args.device)
    print(f"Using device: {device}")

    use_rich = (track is not None) and (not args.no_rich)

    buffered_rows: List[Dict[str, object]] = []
    failed_rows: List[Dict[str, str]] = []
    success_count = 0

    def flush_checkpoint_if_needed(force: bool = False):
        nonlocal existing_df, buffered_rows
        if not buffered_rows:
            return
        if (not force) and args.checkpoint_every > 0 and (len(buffered_rows) < args.checkpoint_every):
            return
        df_new = pd.DataFrame.from_records(buffered_rows)
        existing_df = _merge_station_tables(existing_df, df_new)
        existing_df.to_csv(station_out, index=False)
        buffered_rows = []

    if pending:
        iterator = track(pending, description="Running station counterfactuals") if use_rich else pending
        for sd in iterator:
            try:
                df = run_station_counterfactual(
                    sd,
                    args.forcing_dir,
                    args.streamflow_dir,
                    device=device,
                )
            except Exception as e:
                failed_rows.append({
                    "station_id": sd.name,
                    "error": f"{type(e).__name__}: {e}",
                })
                continue

            if df is None or df.empty:
                failed_rows.append({
                    "station_id": sd.name,
                    "error": "empty result",
                })
                continue

            buffered_rows.extend(df.to_dict(orient="records"))
            success_count += 1
            flush_checkpoint_if_needed(force=False)

    flush_checkpoint_if_needed(force=True)

    if existing_df is None or existing_df.empty:
        raise SystemExit("No station produced valid counterfactual results")

    station_df = existing_df.copy()
    summary_df, extra = summarize_counterfactual(
        station_df,
        n_boot=args.n_boot,
        alpha=args.alpha,
        seed=args.seed,
    )

    station_df.to_csv(station_out, index=False)
    summary_df.to_csv(summary_out, index=False)

    if failed_rows:
        fail_out = station_out.parent / "physical_sensitivity_failed_stations.csv"
        pd.DataFrame(failed_rows).to_csv(fail_out, index=False)
        extra["failed_station_count"] = int(len(failed_rows))
        extra["failed_station_file"] = str(fail_out)
    extra["success_station_count_this_run"] = int(success_count)
    extra["resume_used"] = bool(args.resume)
    extra["workers"] = 1
    extra["device"] = str(device)

    with open(extra_out, "w", encoding="utf-8") as f:
        json.dump(extra, f, ensure_ascii=False, indent=2)

    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(summary_df.to_string(index=False))

    print(f"Using output root: {output_root}")
    print(f"Wrote station-level table: {station_out}")
    print(f"Wrote hypothesis summary: {summary_out}")
    print(f"Wrote extra summary json: {extra_out}")
    if failed_rows:
        print(f"Failed stations: {len(failed_rows)} (see {station_out.parent / 'physical_sensitivity_failed_stations.csv'})")

    if args.auto_plot:
        default_plot_dir = station_out.parent / "figures"
        plot_dir = Path(args.plot_dir) if args.plot_dir else default_plot_dir
        plot_script = Path(args.plot_script)
        ok, msg = _run_r_plotter(
            station_csv=station_out,
            out_dir=plot_dir,
            plot_script=plot_script,
            rscript_bin=args.rscript_bin,
            prefix=args.plot_prefix,
        )
        if ok:
            print(f"Wrote figures to: {plot_dir}")
            if msg:
                print(msg)
        else:
            print("Figure generation skipped or failed.")
            if msg:
                print(msg)


if __name__ == "__main__":
    main()
