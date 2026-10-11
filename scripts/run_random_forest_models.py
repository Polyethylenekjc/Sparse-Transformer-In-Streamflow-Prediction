import os
import argparse
import time
import json
from functools import lru_cache
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

try:
    from rich.progress import track
except Exception:
    track = None


def safe_corr(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.size == 0 or b.size == 0:
        return 0.0
    if np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return 0.0
    return np.corrcoef(a, b)[0, 1]


def identify_regime(P, T, Q, SRD=None):
    if len(Q) >= 2:
        r_qq = safe_corr(Q[:-1], Q[1:])
    else:
        r_qq = 0.0
    r_pq = safe_corr(P, Q)
    r_tq = safe_corr(T, Q)

    scores = {
        "P-Q": r_pq,
        "Q-Q": r_qq,
        "T-Q": r_tq,
    }
    regime = max(scores, key=scores.get)
    return regime, scores


def nse(obs, sim):
    obs = np.asarray(obs)
    sim = np.asarray(sim)
    mask = ~np.isnan(obs) & ~np.isnan(sim)
    if mask.sum() == 0:
        return np.nan
    obs = obs[mask]
    sim = sim[mask]
    denom = np.sum((obs - np.mean(obs)) ** 2)
    if denom == 0:
        return np.nan
    return 1.0 - np.sum((obs - sim) ** 2) / denom


def mse(obs, sim):
    obs = np.asarray(obs)
    sim = np.asarray(sim)
    mask = ~np.isnan(obs) & ~np.isnan(sim)
    if mask.sum() == 0:
        return np.nan
    return np.mean((obs[mask] - sim[mask]) ** 2)


def kge(obs, sim):
    obs = np.asarray(obs)
    sim = np.asarray(sim)
    mask = ~np.isnan(obs) & ~np.isnan(sim)
    if mask.sum() == 0:
        return np.nan
    obs = obs[mask]
    sim = sim[mask]
    if np.std(obs) == 0 or np.std(sim) == 0:
        return np.nan
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim) / np.std(obs)
    beta = np.mean(sim) / np.mean(obs)
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def lognse(obs, sim, eps=1e-3):
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(sim)
    if mask.sum() == 0:
        return np.nan
    obs = obs[mask]
    sim = sim[mask]
    obs_t = np.log(np.maximum(obs, 0.0) + eps)
    sim_t = np.log(np.maximum(sim, 0.0) + eps)
    return nse(obs_t, sim_t)


def _score_by_objective(obs, sim, objective):
    if objective == "kge":
        score = kge(obs, sim)
        return score, score
    if objective == "mse":
        score = mse(obs, sim)
        if np.isnan(score):
            return np.nan, -1e9
        return score, -score
    if objective == "lognse":
        score = lognse(obs, sim)
        return score, score if np.isfinite(score) else -1e9
    if objective == "nse_kge":
        s_nse = nse(obs, sim)
        s_kge = kge(obs, sim)
        if not np.isfinite(s_nse) or not np.isfinite(s_kge):
            return np.nan, -1e9
        score = 0.5 * s_nse + 0.5 * s_kge
        return score, score

    score = nse(obs, sim)
    if np.isnan(score):
        return np.nan, -1e9
    return score, score


def _series_metric(obs, sim, warmup=0):
    warmup = int(max(0, warmup))
    if len(obs) <= warmup:
        return {"nse": np.nan, "mse": np.nan, "kge": np.nan}
    o = np.asarray(obs, dtype=float)[warmup:]
    s = np.asarray(sim, dtype=float)[warmup:]
    return {
        "nse": nse(o, s),
        "mse": mse(o, s),
        "kge": kge(o, s),
    }


def _zscore(arr):
    arr = np.asarray(arr, dtype=float)
    m = np.nanmean(arr)
    s = np.nanstd(arr)
    if np.isnan(m):
        return arr
    if s == 0 or np.isnan(s):
        return arr - m
    return (arr - m) / s


def _rolling_sum(arr, t, window):
    left = max(0, t - window + 1)
    return float(np.sum(arr[left : t + 1]))


def _rolling_mean(arr, t, window):
    left = max(0, t - window + 1)
    return float(np.mean(arr[left : t + 1]))


def _build_supervised_features(P, T, SRD, Q, lags, variant, regime):
    p = np.asarray(P, dtype=float)
    t = np.asarray(T, dtype=float)
    srd = np.asarray(SRD, dtype=float)
    q = np.asarray(Q, dtype=float)

    n = len(q)
    rows = []
    y = []
    t_index = []

    for cur in range(lags, n):
        feat = []

        for k in range(1, lags + 1):
            feat.append(float(p[cur - k]))
            feat.append(float(t[cur - k]))
            feat.append(float(srd[cur - k]))
            feat.append(float(q[cur - k]))

        feat.extend(
            [
                _rolling_sum(p, cur - 1, 3),
                _rolling_sum(p, cur - 1, 7),
                _rolling_mean(t, cur - 1, 7),
                _rolling_mean(srd, cur - 1, 7),
                _rolling_mean(q, cur - 1, 3),
                float(q[cur - 1] - q[cur - 2]) if cur - 2 >= 0 else 0.0,
            ]
        )

        if variant == "modified":
            p3 = _rolling_sum(p, cur - 1, 3)
            p7 = _rolling_sum(p, cur - 1, 7)
            p14 = _rolling_sum(p, cur - 1, 14)
            q3 = _rolling_mean(q, cur - 1, 3)
            q7 = _rolling_mean(q, cur - 1, 7)
            t7 = _rolling_mean(t, cur - 1, 7)
            t_pos7 = float(max(0.0, t7))
            t_neg7 = float(max(0.0, -t7))
            srd7 = _rolling_mean(srd, cur - 1, 7)

            feat.extend(
                [
                    p3,
                    p7,
                    p14,
                    q3,
                    q7,
                    float(q3 - q7),
                    t7,
                    t_pos7,
                    t_neg7,
                    srd7,
                ]
            )

            feat.extend(
                [
                    1.0 if regime == "P-Q" else 0.0,
                    1.0 if regime == "Q-Q" else 0.0,
                    1.0 if regime == "T-Q" else 0.0,
                ]
            )

        rows.append(feat)
        y.append(float(q[cur]))
        t_index.append(cur)

    return np.asarray(rows, dtype=float), np.asarray(y, dtype=float), np.asarray(t_index, dtype=int)


def _decode_rf_params(raw):
    p = dict(raw)
    if int(p.get("max_depth", 0)) <= 0:
        p["max_depth"] = None
    else:
        p["max_depth"] = int(p["max_depth"])
    p["n_estimators"] = int(p["n_estimators"])
    p["min_samples_split"] = int(p["min_samples_split"])
    p["min_samples_leaf"] = int(p["min_samples_leaf"])
    p["max_features"] = float(p["max_features"])
    p["bootstrap"] = bool(int(p["bootstrap"]))
    p["random_state"] = int(p.get("random_state", 42))
    p["n_jobs"] = 1
    return p


def _encode_params(raw):
    return np.asarray(
        [
            float(raw["max_depth"]),
            float(raw["min_samples_split"]),
            float(raw["min_samples_leaf"]),
            float(raw["max_features"]),
        ],
        dtype=float,
    )


def _sample_rf_params_bayes_space(rng, seed, n_estimators):
    return {
        "n_estimators": int(n_estimators),
        "max_depth": int(rng.integers(2, 9)),
        "min_samples_split": int(rng.integers(2, 9)),
        "min_samples_leaf": int(rng.integers(1, 5)),
        "max_features": float(rng.uniform(0.5, 0.8)),
        "bootstrap": 1,
        "random_state": int(seed),
    }


def calibrate_rf_bayes(
    X_train,
    y_train,
    X_val,
    y_val,
    objective="mse",
    variant="baseline",
    n_trials=40,
    seed=42,
    n_estimators=60,
    early_stop_patience=8,
    min_improvement=1e-4,
    eval_stride=1,
    max_seconds=0.0,
):
    _ = variant
    rng = np.random.default_rng(seed)
    best_score = -1e18
    best_params = None
    no_improve = 0
    trials_run = 0
    stopped_early = False
    stop_reason = "completed"
    t0 = time.time()

    trials = int(max(1, n_trials))
    patience = int(max(1, early_stop_patience))
    stride = int(max(1, eval_stride))
    min_delta = float(max(0.0, min_improvement))
    time_budget = float(max(0.0, max_seconds))

    history_x = []
    history_y = []
    init_trials = int(max(4, min(8, trials // 2 if trials > 8 else 4)))
    seen = set()

    for i in range(trials):
        if time_budget > 0 and (time.time() - t0) >= time_budget:
            stopped_early = True
            stop_reason = "time_budget"
            break

        if i < init_trials or len(history_x) < 4:
            raw = _sample_rf_params_bayes_space(rng, seed=seed + i, n_estimators=n_estimators)
        else:
            kernel = (
                ConstantKernel(1.0, (1e-3, 1e3))
                * Matern(length_scale=np.ones(4), nu=2.5)
                + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e-2))
            )
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True,
                random_state=seed,
                n_restarts_optimizer=1,
            )
            x_train_gp = np.asarray(history_x, dtype=float)
            y_train_gp = np.asarray(history_y, dtype=float)
            gp.fit(x_train_gp, y_train_gp)

            cand = [
                _sample_rf_params_bayes_space(rng, seed=seed + i + j + 1, n_estimators=n_estimators)
                for j in range(128)
            ]
            x_cand = np.asarray([_encode_params(c) for c in cand], dtype=float)
            mu, std = gp.predict(x_cand, return_std=True)
            std = np.maximum(std, 1e-9)
            improv = mu - (np.max(y_train_gp) + min_delta)
            z = improv / std
            ei = improv * stats.norm.cdf(z) + std * stats.norm.pdf(z)
            best_idx = int(np.nanargmax(ei)) if len(ei) > 0 else 0
            raw = cand[best_idx]

        key = (
            int(raw["max_depth"]),
            int(raw["min_samples_split"]),
            int(raw["min_samples_leaf"]),
            round(float(raw["max_features"]), 4),
        )
        if key in seen:
            raw = _sample_rf_params_bayes_space(rng, seed=seed + 1000 + i, n_estimators=n_estimators)
            key = (
                int(raw["max_depth"]),
                int(raw["min_samples_split"]),
                int(raw["min_samples_leaf"]),
                round(float(raw["max_features"]), 4),
            )
        seen.add(key)

        params = _decode_rf_params(raw)

        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        _, fitness = _score_by_objective(y_val, pred, objective)
        history_x.append(_encode_params(raw))
        history_y.append(float(fitness) if np.isfinite(fitness) else -1e9)
        trials_run += 1

        if np.isfinite(fitness) and fitness > (best_score + min_delta):
            best_score = float(fitness)
            best_params = raw
            no_improve = 0
        else:
            no_improve += 1

        if (trials_run % stride == 0) and (no_improve >= patience):
            stopped_early = True
            stop_reason = "no_improvement"
            break

    if best_params is None:
        best_params = {
            "n_estimators": int(n_estimators),
            "max_depth": 4,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": 0.6,
            "bootstrap": 1,
            "random_state": seed,
        }
        best_score = np.nan

    return {
        "best_params": best_params,
        "best_score": float(best_score),
        "trials_run": int(trials_run),
        "trials_planned": int(trials),
        "stopped_early": bool(stopped_early),
        "stop_reason": str(stop_reason),
        "elapsed_seconds": float(time.time() - t0),
    }


def fit_predict_rf_series(
    P,
    T,
    SRD,
    Q_obs,
    train_end,
    val_end,
    lags,
    variant,
    regime,
    calibrate,
    objective,
    n_trials,
    warmup_cal,
    seed,
    early_stop_patience,
    min_improvement,
    calib_eval_stride,
    calib_max_seconds,
    rf_n_estimators,
):
    X, y, idx = _build_supervised_features(P, T, SRD, Q_obs, lags=lags, variant=variant, regime=regime)
    if len(y) < 80:
        return None

    train_mask = idx < int(train_end)
    val_mask = (idx >= int(train_end)) & (idx < int(val_end))
    test_mask = idx >= int(val_end)
    if train_mask.sum() < 60 or val_mask.sum() < 20 or test_mask.sum() < 20:
        return None

    X_train = X[train_mask]
    y_train = y[train_mask]

    if calibrate:
        X_sub = X_train
        y_sub = y_train
        X_val = X[val_mask]
        y_val = y[val_mask]

        if len(y_sub) < 20 or len(y_val) < 10:
            X_sub = X_train
            y_sub = y_train
            X_val = X_train
            y_val = y_train

        calib_info = calibrate_rf_bayes(
            X_train=X_sub,
            y_train=y_sub,
            X_val=X_val,
            y_val=y_val,
            objective=objective,
            variant=variant,
            n_trials=n_trials,
            seed=seed,
            n_estimators=rf_n_estimators,
            early_stop_patience=early_stop_patience,
            min_improvement=min_improvement,
            eval_stride=calib_eval_stride,
            max_seconds=calib_max_seconds,
        )
        best_params_raw = calib_info["best_params"]
        calib_score = calib_info["best_score"]
    else:
        calib_score = np.nan
        calib_info = {
            "trials_run": 0,
            "trials_planned": int(max(1, n_trials)),
            "stopped_early": False,
            "stop_reason": "disabled",
            "elapsed_seconds": 0.0,
        }
        best_params_raw = {
            "n_estimators": 60 if variant == "modified" else 60,
            "max_depth": 8 if variant == "modified" else 4,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": 0.5 if variant == "modified" else 0.5,
            "bootstrap": 1,
            "random_state": seed,
        }

    params = _decode_rf_params(best_params_raw)
    model = RandomForestRegressor(**params)

    if variant == "modified" and regime == "P-Q":
        base_idx = 4 * int(lags)
        p7_idx = base_idx + 1
        weights = 1.0 + 0.35 * (X_train[:, p7_idx] > np.percentile(X_train[:, p7_idx], 75)).astype(float)
        model.fit(X_train, y_train, sample_weight=weights)
    elif variant == "modified" and regime == "T-Q":
        base_idx = 4 * int(lags)
        t_pos_idx = base_idx + 7
        weights = 1.0 + 0.25 * (X_train[:, t_pos_idx] > np.percentile(X_train[:, t_pos_idx], 75)).astype(float)
        model.fit(X_train, y_train, sample_weight=weights)
    else:
        model.fit(X_train, y_train)

    pred_all = model.predict(X)
    pred_series = np.full(len(Q_obs), np.nan, dtype=float)
    pred_series[idx] = pred_all

    train_target = Q_obs[:train_end]
    val_target = Q_obs[train_end:val_end]
    test_target = Q_obs[val_end:]
    train_pred = pred_series[:train_end]
    val_pred = pred_series[train_end:val_end]
    test_pred = pred_series[val_end:]

    warmup_eff = int(max(0, min(int(warmup_cal), len(train_target) - 1)))
    train_metrics = _series_metric(train_target, train_pred, warmup=warmup_eff)
    val_metrics = _series_metric(val_target, val_pred, warmup=0)
    test_metrics = _series_metric(test_target, test_pred, warmup=0)

    return {
        "pred_series": pred_series,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "params_raw": best_params_raw,
        "calib_score": calib_score,
        "calib_info": calib_info,
        "n_train_eff": int(train_mask.sum()),
        "n_val_eff": int(val_mask.sum()),
        "n_test_eff": int(test_mask.sum()),
    }


def load_station_data(forcing_path, streamflow_path):
    df_f = pd.read_csv(forcing_path, parse_dates=["date_utc"])
    df_q = pd.read_csv(streamflow_path, parse_dates=["date_utc"])

    required_f = ["date_utc", "prcp_mm_day", "tmax_c", "tmin_c", "srad_w_m2"]
    required_q = ["date_utc", "streamflow_cfs"]
    if not all(c in df_f.columns for c in required_f):
        raise ValueError("Forcing file missing required columns")
    if not all(c in df_q.columns for c in required_q):
        raise ValueError("Streamflow file missing required columns")

    df_f = df_f[required_f].copy()
    df_q = df_q[required_q].copy()

    merged = pd.merge(df_f, df_q, on="date_utc", how="inner")
    merged = merged.sort_values("date_utc").reset_index(drop=True)

    merged["streamflow_cfs"] = pd.to_numeric(merged["streamflow_cfs"], errors="coerce")
    merged = merged[np.isfinite(merged["streamflow_cfs"])].reset_index(drop=True)
    return merged


@lru_cache(maxsize=1)
def _load_station_area_km2_map():
    stations_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "stations.csv"))
    if not os.path.exists(stations_path):
        return {}

    st = pd.read_csv(stations_path)
    if "gauge_id" not in st.columns:
        return {}

    st["gauge_id"] = st["gauge_id"].astype(str).str.strip().str.zfill(8)
    area_km2 = pd.to_numeric(st.get("drainage_area_km2"), errors="coerce")
    area_m2 = pd.to_numeric(st.get("forcing_area_m2"), errors="coerce")
    area_km2_alt = area_m2 / 1e6

    final_area = area_km2.copy()
    missing = ~np.isfinite(final_area)
    final_area.loc[missing] = area_km2_alt.loc[missing]

    out = {}
    for gid, area in zip(st["gauge_id"], final_area):
        if np.isfinite(area) and float(area) > 0:
            out[str(gid)] = float(area)
    return out


def _streamflow_cfs_to_mm_day(q_cfs, area_km2):
    return np.asarray(q_cfs, dtype=float) * 2.446575546 / float(area_km2)


def save_station_timeseries(output_root, station_id, dates, q_obs, q_base, q_mod, calibrated=False):
    station_dir = os.path.join(output_root, "random_forest_station_series")
    os.makedirs(station_dir, exist_ok=True)
    suffix = "_tuned" if calibrated else ""
    out_path = os.path.join(station_dir, f"{station_id}_rf_series{suffix}.csv")

    ts_df = pd.DataFrame(
        {
            "date_utc": pd.to_datetime(dates),
            "q_obs_mm_day": q_obs,
            "q_rf_baseline_mm_day": q_base,
            "q_rf_modified_mm_day": q_mod,
        }
    )
    ts_df.to_csv(out_path, index=False)
    return out_path


def _load_sparse_transformer_good_stations(outputs_dir, nse_threshold=0.5):
    threshold = float(nse_threshold)
    csv_path = os.path.join(outputs_dir, "sparse_transformer_station_metrics.csv")
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if "station_id" in df.columns and "nse" in df.columns:
                sid = df["station_id"].astype(str).str.strip().str.zfill(8)
                nse_vals = pd.to_numeric(df["nse"], errors="coerce")
                mask = np.isfinite(nse_vals) & (nse_vals > threshold)
                return set(sid[mask].tolist())
        except Exception:
            pass

    good = set()
    try:
        for name in os.listdir(outputs_dir):
            d = os.path.join(outputs_dir, name)
            if not (os.path.isdir(d) and str(name).isdigit()):
                continue
            summary_path = os.path.join(d, "summary.json")
            if not os.path.exists(summary_path):
                continue
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    s = json.load(f)
                nse_val = float(s.get("metrics", {}).get("nse", np.nan))
                if np.isfinite(nse_val) and nse_val > threshold:
                    good.add(str(name).zfill(8))
            except Exception:
                continue
    except Exception:
        return set()

    return good


def process_station(
    fname,
    forcing_dir,
    streamflow_dir,
    outputs_dir,
    params_default,
    calibrate,
    objective,
    population,
    generations,
    seed,
    early_stop_patience,
    min_improvement,
    normalize_inputs=False,
    train_ratio=0.7,
    warmup_days=365,
    calib_eval_stride=1,
    calib_max_seconds=0.0,
    bo_trials=18,
    rf_n_estimators=60,
):
    _ = params_default
    _ = train_ratio

    station_id = os.path.splitext(fname)[0]
    fpath = os.path.join(forcing_dir, fname)
    qpath = os.path.join(streamflow_dir, fname)
    if not os.path.exists(qpath):
        return None

    try:
        df = load_station_data(fpath, qpath)
    except Exception:
        return None

    df = df.sort_values("date_utc").reset_index(drop=True)
    P = df["prcp_mm_day"].astype(float).fillna(0.0).values
    T = (df["tmax_c"].astype(float).fillna(np.nan).values + df["tmin_c"].astype(float).fillna(np.nan).values) / 2.0
    SRD = df["srad_w_m2"].astype(float).fillna(0.0).values
    Q_obs = df["streamflow_cfs"].astype(float).values

    area_map = _load_station_area_km2_map()
    area_km2 = area_map.get(str(station_id).zfill(8))
    if area_km2 is None or not np.isfinite(area_km2) or area_km2 <= 0:
        return None
    Q_obs = _streamflow_cfs_to_mm_day(Q_obs, area_km2)

    if len(Q_obs) < 140:
        return None

    if np.any(~np.isfinite(T)):
        t_ser = pd.Series(T)
        T = t_ser.interpolate(limit_direction="both").fillna(t_ser.mean()).to_numpy(dtype=float)

    if normalize_inputs:
        P = _zscore(P)
        T = _zscore(T)
        SRD = _zscore(SRD)
        Q_obs = _zscore(Q_obs)

    n = len(Q_obs)
    train_end = int(round(n * 0.70))
    val_end = int(round(n * 0.85))
    train_end = max(80, min(n - 60, train_end))
    val_end = max(train_end + 20, min(n - 20, val_end))
    warmup_cal = int(max(0, min(int(warmup_days), train_end - 20)))

    regime, regime_scores = identify_regime(P[:train_end], T[:train_end], Q_obs[:train_end], SRD[:train_end])

    n_trials = int(max(6, min(60, int(bo_trials))))

    base = fit_predict_rf_series(
        P=P,
        T=T,
        SRD=SRD,
        Q_obs=Q_obs,
        train_end=train_end,
        val_end=val_end,
        lags=10,
        variant="baseline",
        regime=regime,
        calibrate=calibrate,
        objective=objective,
        n_trials=n_trials,
        warmup_cal=warmup_cal,
        seed=seed,
        early_stop_patience=early_stop_patience,
        min_improvement=min_improvement,
        calib_eval_stride=calib_eval_stride,
        calib_max_seconds=calib_max_seconds,
        rf_n_estimators=rf_n_estimators,
    )

    mod = fit_predict_rf_series(
        P=P,
        T=T,
        SRD=SRD,
        Q_obs=Q_obs,
        train_end=train_end,
        val_end=val_end,
        lags=10,
        variant="modified",
        regime=regime,
        calibrate=calibrate,
        objective=objective,
        n_trials=n_trials,
        warmup_cal=warmup_cal,
        seed=seed + 1,
        early_stop_patience=early_stop_patience,
        min_improvement=min_improvement,
        calib_eval_stride=calib_eval_stride,
        calib_max_seconds=calib_max_seconds,
        rf_n_estimators=rf_n_estimators,
    )

    if base is None or mod is None:
        return None

    save_station_timeseries(
        output_root=outputs_dir,
        station_id=station_id,
        dates=df["date_utc"].values,
        q_obs=Q_obs,
        q_base=base["pred_series"],
        q_mod=mod["pred_series"],
        calibrated=calibrate,
    )

    p_base = base["params_raw"]
    p_mod = mod["params_raw"]

    return {
        "station_id": station_id,
        "n_obs": int(n),
        "n_train": int(train_end),
        "n_val": int(val_end - train_end),
        "n_test": int(n - val_end),
        "n_train_eff_base": int(base["n_train_eff"]),
        "n_val_eff_base": int(base["n_val_eff"]),
        "n_test_eff_base": int(base["n_test_eff"]),
        "n_train_eff_mod": int(mod["n_train_eff"]),
        "n_val_eff_mod": int(mod["n_val_eff"]),
        "n_test_eff_mod": int(mod["n_test_eff"]),
        "regime": regime,
        "regime_score_pq": float(regime_scores.get("P-Q", np.nan)),
        "regime_score_qq": float(regime_scores.get("Q-Q", np.nan)),
        "regime_score_tq": float(regime_scores.get("T-Q", np.nan)),
        "calibrated": bool(calibrate),
        "zscore_applied": bool(normalize_inputs),
        "q_obs_unit": "mm_day" if not normalize_inputs else "zscore(mm_day)",
        "area_km2": float(area_km2),
        "calib_objective": objective if calibrate else "none",
        "calib_method": "bayesian_optimization" if calibrate else "none",
        "warmup_cal": int(warmup_cal),
        "calib_score_base": float(base["calib_score"]) if np.isfinite(base["calib_score"]) else np.nan,
        "calib_score_mod": float(mod["calib_score"]) if np.isfinite(mod["calib_score"]) else np.nan,
        "calib_trials_run_base": int(base.get("calib_info", {}).get("trials_run", 0)),
        "calib_trials_planned_base": int(base.get("calib_info", {}).get("trials_planned", n_trials)),
        "calib_stopped_early_base": bool(base.get("calib_info", {}).get("stopped_early", False)),
        "calib_stop_reason_base": str(base.get("calib_info", {}).get("stop_reason", "")),
        "calib_elapsed_seconds_base": float(base.get("calib_info", {}).get("elapsed_seconds", 0.0)),
        "calib_trials_run_mod": int(mod.get("calib_info", {}).get("trials_run", 0)),
        "calib_trials_planned_mod": int(mod.get("calib_info", {}).get("trials_planned", n_trials)),
        "calib_stopped_early_mod": bool(mod.get("calib_info", {}).get("stopped_early", False)),
        "calib_stop_reason_mod": str(mod.get("calib_info", {}).get("stop_reason", "")),
        "calib_elapsed_seconds_mod": float(mod.get("calib_info", {}).get("elapsed_seconds", 0.0)),
        "n_estimators_base": int(p_base["n_estimators"]),
        "max_depth_base": int(p_base["max_depth"]),
        "min_samples_split_base": int(p_base["min_samples_split"]),
        "min_samples_leaf_base": int(p_base["min_samples_leaf"]),
        "max_features_base": float(p_base["max_features"]),
        "bootstrap_base": int(p_base["bootstrap"]),
        "n_estimators_mod": int(p_mod["n_estimators"]),
        "max_depth_mod": int(p_mod["max_depth"]),
        "min_samples_split_mod": int(p_mod["min_samples_split"]),
        "min_samples_leaf_mod": int(p_mod["min_samples_leaf"]),
        "max_features_mod": float(p_mod["max_features"]),
        "bootstrap_mod": int(p_mod["bootstrap"]),
        "lags_base": 10,
        "lags_mod": 10,
        "nse_base": float(base["test_metrics"]["nse"]) if np.isfinite(base["test_metrics"]["nse"]) else np.nan,
        "mse_base": float(base["test_metrics"]["mse"]) if np.isfinite(base["test_metrics"]["mse"]) else np.nan,
        "kge_base": float(base["test_metrics"]["kge"]) if np.isfinite(base["test_metrics"]["kge"]) else np.nan,
        "nse_mod": float(mod["test_metrics"]["nse"]) if np.isfinite(mod["test_metrics"]["nse"]) else np.nan,
        "mse_mod": float(mod["test_metrics"]["mse"]) if np.isfinite(mod["test_metrics"]["mse"]) else np.nan,
        "kge_mod": float(mod["test_metrics"]["kge"]) if np.isfinite(mod["test_metrics"]["kge"]) else np.nan,
        "train_nse_base": float(base["train_metrics"]["nse"]) if np.isfinite(base["train_metrics"]["nse"]) else np.nan,
        "train_mse_base": float(base["train_metrics"]["mse"]) if np.isfinite(base["train_metrics"]["mse"]) else np.nan,
        "train_kge_base": float(base["train_metrics"]["kge"]) if np.isfinite(base["train_metrics"]["kge"]) else np.nan,
        "train_nse_mod": float(mod["train_metrics"]["nse"]) if np.isfinite(mod["train_metrics"]["nse"]) else np.nan,
        "train_mse_mod": float(mod["train_metrics"]["mse"]) if np.isfinite(mod["train_metrics"]["mse"]) else np.nan,
        "train_kge_mod": float(mod["train_metrics"]["kge"]) if np.isfinite(mod["train_metrics"]["kge"]) else np.nan,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Random Forest baseline/modified with optional hyperparameter search.")
    parser.add_argument("--station", type=str, default=None, help="Only run one station id, e.g., 01013500")
    parser.add_argument("--max-stations", "--max_stations", dest="max_stations", type=int, default=0, help=">0 means only run first N stations after sorting")
    parser.add_argument("--objective", type=str, default="nse_kge", choices=["mse", "nse", "kge", "lognse", "nse_kge"], help="Calibration objective")
    parser.add_argument("--calibrate", action="store_true", help="Enable Bayesian optimization for RF hyperparameters")
    parser.add_argument("--bo-trials", type=int, default=18, help="Bayesian optimization total trials")
    parser.add_argument("--rf-n-estimators", type=int, default=60, help="Fixed RF tree count during BO to reduce training cost")
    parser.add_argument("--population", type=int, default=6, help="Legacy compatibility only")
    parser.add_argument("--generations", type=int, default=12, help="Legacy compatibility only")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers across stations")
    parser.add_argument(
        "--progress-heartbeat-sec",
        type=float,
        default=30.0,
        help="Seconds between heartbeat logs when no station finishes in parallel mode",
    )
    parser.add_argument("--zscore", dest="zscore", action="store_true", default=True, help="Apply z-score normalization to all model input series and streamflow target per station (default: enabled)")
    parser.add_argument("--no-zscore", dest="zscore", action="store_false", help="Disable z-score normalization")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Legacy compatibility; split is fixed to train/val/test = 0.70/0.15/0.15")
    parser.add_argument("--warmup-days", type=int, default=365, help="Warmup steps ignored in calibration metrics")
    parser.add_argument("--early-stop-patience", type=int, default=8, help="Legacy compatibility")
    parser.add_argument("--min-improvement", type=float, default=1e-4, help="Legacy compatibility")
    parser.add_argument("--calib-eval-stride", type=int, default=1, help="Legacy compatibility")
    parser.add_argument("--calib-max-seconds", type=float, default=0.0, help="Legacy compatibility")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--sparse-nse-threshold",
        type=float,
        default=0.5,
        help="Only use stations with Sparse Transformer NSE above this threshold when summarizing RF NSE",
    )
    parser.add_argument(
        "--sparse-output-root",
        type=str,
        default="",
        help="Sparse Transformer output root for NSE subset filter (default: /mnt/d/Store/Sparse_Transformer if exists, otherwise local outputs)",
    )
    args = parser.parse_args()

    forcing_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "Forcing"))
    streamflow_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "Streamflow"))
    outputs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
    preferred_sparse_root = "/mnt/d/Store/Sparse_Transformer"
    if args.sparse_output_root and str(args.sparse_output_root).strip():
        sparse_output_root = os.path.abspath(os.path.expanduser(str(args.sparse_output_root).strip()))
    elif os.path.isdir(preferred_sparse_root):
        sparse_output_root = preferred_sparse_root
    else:
        sparse_output_root = outputs_dir

    files = sorted([f for f in os.listdir(forcing_dir) if f.endswith(".csv")])
    if args.station:
        target = f"{args.station}.csv"
        files = [f for f in files if f == target]
        if not files:
            raise FileNotFoundError(f"Station file not found in Forcing: {target}")
    elif args.max_stations and args.max_stations > 0:
        files = files[: int(args.max_stations)]

    print(f"Found {len(files)} stations in Forcing folder.")

    out = []
    workers = max(1, int(args.workers))
    desc = "Tuning stations" if args.calibrate else "Running stations"

    if workers == 1:
        file_iter = files
        if track is not None:
            file_iter = track(files, description=desc)
        for fname in file_iter:
            row = process_station(
                fname=fname,
                forcing_dir=forcing_dir,
                streamflow_dir=streamflow_dir,
                outputs_dir=outputs_dir,
                params_default={},
                calibrate=args.calibrate,
                objective=args.objective,
                population=args.population,
                generations=args.generations,
                seed=args.seed,
                early_stop_patience=args.early_stop_patience,
                min_improvement=args.min_improvement,
                normalize_inputs=args.zscore,
                train_ratio=args.train_ratio,
                warmup_days=args.warmup_days,
                calib_eval_stride=args.calib_eval_stride,
                calib_max_seconds=args.calib_max_seconds,
                bo_trials=args.bo_trials,
                rf_n_estimators=args.rf_n_estimators,
            )
            if row is not None:
                out.append(row)
    else:
        ex = ProcessPoolExecutor(max_workers=workers)
        interrupted = False
        try:
            futures = [
                ex.submit(
                    process_station,
                    fname,
                    forcing_dir,
                    streamflow_dir,
                    outputs_dir,
                    {},
                    args.calibrate,
                    args.objective,
                    args.population,
                    args.generations,
                    args.seed,
                    args.early_stop_patience,
                    args.min_improvement,
                    args.zscore,
                    args.train_ratio,
                    args.warmup_days,
                    args.calib_eval_stride,
                    args.calib_max_seconds,
                    args.bo_trials,
                    args.rf_n_estimators,
                )
                for fname in files
            ]

            total = len(futures)
            heartbeat_sec = max(1.0, float(args.progress_heartbeat_sec))
            pending = set(futures)
            completed = 0
            start_ts = time.time()

            while pending:
                done, pending = wait(pending, timeout=heartbeat_sec, return_when=FIRST_COMPLETED)
                if not done:
                    elapsed_min = (time.time() - start_ts) / 60.0
                    print(
                        f"[rf] heartbeat: {completed}/{total} completed, {len(pending)} in-flight, elapsed={elapsed_min:.1f} min",
                        flush=True,
                    )
                    continue

                for fut in done:
                    row = fut.result()
                    completed += 1
                    if row is not None:
                        out.append(row)

                if completed == total or completed % max(1, total // 100) == 0:
                    print(f"[rf] progress: {completed}/{total} ({completed / total:.1%})", flush=True)
        except KeyboardInterrupt:
            interrupted = True
            print("\nInterrupted. Cancelling pending Random Forest jobs...", flush=True)
            ex.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            if not interrupted:
                ex.shutdown(wait=True)

    results = pd.DataFrame(out)
    suffix = "_tuned" if args.calibrate else ""
    station_suffix = f"_{args.station}" if args.station else ""
    results_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "outputs", f"random_forest_model_performance{suffix}{station_suffix}.csv")
    )
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    results.to_csv(results_path, index=False)

    print("\nSummary across stations:")
    print(f"Stations processed: {len(results)}")
    if len(results) > 0:
        if "station_id" in results.columns:
            results["station_id"] = results["station_id"].astype(str).str.strip().str.zfill(8)

        sparse_good = _load_sparse_transformer_good_stations(
            outputs_dir=sparse_output_root,
            nse_threshold=args.sparse_nse_threshold,
        )

        if len(sparse_good) > 0 and "station_id" in results.columns:
            stat_df = results[results["station_id"].isin(sparse_good)].copy()
            print(
                f"NSE stats subset: Sparse Transformer NSE > {args.sparse_nse_threshold:.2f} "
                f"({len(stat_df)}/{len(results)} stations matched, source={sparse_output_root})"
            )
        else:
            stat_df = results.copy()
            print(
                "Warning: No Sparse Transformer station metrics found or no stations passed threshold; "
                f"source={sparse_output_root}; falling back to all processed stations for summary stats."
            )

        if len(stat_df) > 0:
            print(f"Mean NSE (baseline): {stat_df['nse_base'].mean():.4f}")
            print(f"Mean NSE (modified): {stat_df['nse_mod'].mean():.4f}")
        else:
            print("Mean NSE (baseline): nan")
            print("Mean NSE (modified): nan")

        regime_counts = stat_df["regime"].value_counts(dropna=False) if len(stat_df) > 0 else pd.Series(dtype=float)
        print("Regime dominance: "
              f"P-Q={int(regime_counts.get('P-Q', 0))}, "
              f"Q-Q={int(regime_counts.get('Q-Q', 0))}, "
              f"T-Q={int(regime_counts.get('T-Q', 0))}")

        mask = stat_df["nse_base"].notna() & stat_df["nse_mod"].notna() if len(stat_df) > 0 else pd.Series([], dtype=bool)
        if mask.sum() >= 5:
            tstat, pval = stats.ttest_rel(stat_df.loc[mask, "nse_base"], stat_df.loc[mask, "nse_mod"])
            mean_diff = (stat_df.loc[mask, "nse_mod"] - stat_df.loc[mask, "nse_base"]).mean()
            print(
                f"Paired t-test (NSE mod - NSE base): t={tstat:.4f}, p={pval:.4e}, mean diff={mean_diff:.4f}, n={mask.sum()}"
            )
        else:
            print("Not enough paired NSE values for t-test.")

    print(f"Saved results to: {results_path}")


if __name__ == "__main__":
    main()
