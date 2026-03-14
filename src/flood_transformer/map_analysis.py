from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


def _safe_read_json(path: str) -> Dict[str, Any] | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _safe_read_csv(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _abbr_feature(name: str) -> str:
    low = name.lower()
    if low == "target_q":
        return "Qtar"
    if "streamflow" in low or low in ["q", "flow", "streamflow_cfs"]:
        return "Q"
    if "prcp" in low or "precip" in low or "rain" in low:
        return "P"
    if "tmax" in low:
        return "Tmax"
    if "tmin" in low:
        return "Tmin"
    if "pet" in low:
        return "PET"
    if "temp" in low or low == "t":
        return "T"
    if "dayl" in low:
        return "DayL"
    if "srad" in low:
        return "SRAD"
    if "swe" in low:
        return "SWE"
    if low == "vp_pa" or low.startswith("vp"):
        return "VP"
    if "soil" in low:
        return "Soil"
    parts = [p for p in name.replace("-", "_").split("_") if p]
    if not parts:
        return name[:5]
    return "".join([p[0].upper() for p in parts])[:5]


def discover_station_ids(output_root: str) -> List[str]:
    if not os.path.isdir(output_root):
        return []
    out: List[str] = []
    for name in sorted(os.listdir(output_root)):
        p = os.path.join(output_root, name)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "summary.json")):
            out.append(name)
    return out


def _extract_top_input_features(node_df: pd.DataFrame | None, top_n: int = 5) -> Dict[str, float]:
    if node_df is None or len(node_df) == 0:
        return {}
    needed = {"node_type", "name", "importance"}
    if not needed.issubset(set(node_df.columns)):
        return {}

    x = node_df.copy()
    x = x[x["node_type"].astype(str).str.lower() == "input"]
    if len(x) == 0:
        return {}

    x["abbr"] = x["name"].astype(str).map(_abbr_feature)
    x["importance"] = pd.to_numeric(x["importance"], errors="coerce").fillna(0.0)
    x = x[x["abbr"] != "Qtar"]
    if len(x) == 0:
        return {}

    grp = x.groupby("abbr", as_index=False)["importance"].sum().sort_values("importance", ascending=False)
    grp = grp.head(max(1, int(top_n)))
    return {str(r["abbr"]): float(r["importance"]) for _, r in grp.iterrows()}


def _extract_struct_features(node_df: pd.DataFrame | None) -> Dict[str, float]:
    if node_df is None or len(node_df) == 0:
        return {}
    needed = {"node_type", "layer", "importance"}
    if not needed.issubset(set(node_df.columns)):
        return {}

    x = node_df.copy()
    x["node_type"] = x["node_type"].astype(str).str.lower()
    x["layer"] = pd.to_numeric(x["layer"], errors="coerce")
    x["importance"] = pd.to_numeric(x["importance"], errors="coerce").fillna(0.0)

    out: Dict[str, float] = {}
    for node_type in ["head", "neuron"]:
        y = x[x["node_type"] == node_type]
        if len(y) == 0:
            continue
        out[f"struct_sum_{node_type}"] = float(y["importance"].sum())
        out[f"struct_max_{node_type}"] = float(y["importance"].max())
        layer_grp = y.groupby("layer", as_index=False)["importance"].sum()
        for _, r in layer_grp.iterrows():
            l = int(r["layer"])
            out[f"struct_{node_type}_layer_{l}"] = float(r["importance"])

    all_struct = x[x["node_type"].isin(["head", "neuron"])]
    if len(all_struct) > 0:
        out["struct_total"] = float(all_struct["importance"].sum())

    return out


def _extract_probe_features(probe: Dict[str, Any] | None) -> Dict[str, float]:
    if probe is None:
        return {}
    out: Dict[str, float] = {}
    for key in ["cum_rain", "api", "dpdt"]:
        obj = probe.get(key)
        if isinstance(obj, dict):
            r2 = obj.get("r2")
            if r2 is not None:
                out[f"probe_r2_{key}"] = float(r2)
            bias = obj.get("bias")
            if bias is not None:
                out[f"probe_bias_{key}"] = float(bias)

    r2_cols = [c for c in out.keys() if c.startswith("probe_r2_")]
    if r2_cols:
        out["probe_r2_mean"] = float(np.mean([out[c] for c in r2_cols]))
    return out


def _extract_top_pair_interaction(var_rank_df: pd.DataFrame | None) -> Dict[str, float]:
    if var_rank_df is None or len(var_rank_df) == 0:
        return {}
    needed = {"var_i", "var_j", "interaction"}
    if not needed.issubset(set(var_rank_df.columns)):
        return {}

    x = var_rank_df.copy()
    x["interaction"] = pd.to_numeric(x["interaction"], errors="coerce").fillna(0.0).abs()
    if len(x) == 0:
        return {}

    row = x.sort_values("interaction", ascending=False).iloc[0]
    vi = _abbr_feature(str(row["var_i"]))
    vj = _abbr_feature(str(row["var_j"]))
    return {
        "top_pair_interaction_score": float(row["interaction"]),
        "top_pair_interaction_name": f"{vi}x{vj}",
    }


def load_station_feature_table(
    output_root: str,
    stations_csv: str,
    camels_root: str | None = None,
    top_input_n: int = 5,
) -> pd.DataFrame:
    station_ids = discover_station_ids(output_root)
    rows: List[Dict[str, Any]] = []

    for sid in station_ids:
        d = os.path.join(output_root, sid)
        summary = _safe_read_json(os.path.join(d, "summary.json")) or {}
        node_df = _safe_read_csv(os.path.join(d, "node_importance_ranking.csv"))
        probe = _safe_read_json(os.path.join(d, "physical_probe_results.json")) or {}
        var_rank_df = _safe_read_csv(os.path.join(d, "variable_interactions_ranking.csv"))

        metrics = summary.get("metrics", {}) if isinstance(summary, dict) else {}

        row: Dict[str, Any] = {
            "station_id": str(sid),
            "nse": float(metrics.get("nse")) if metrics.get("nse") is not None else np.nan,
            "rmse": float(metrics.get("rmse")) if metrics.get("rmse") is not None else np.nan,
            "mae": float(metrics.get("mae")) if metrics.get("mae") is not None else np.nan,
            "nse_pass": bool(metrics.get("nse_pass")) if metrics.get("nse_pass") is not None else None,
        }

        top_inputs = _extract_top_input_features(node_df, top_n=top_input_n)
        for k, v in top_inputs.items():
            row[f"top_factor_{k}"] = float(v)
        if top_inputs:
            row["top_factor_name"] = max(top_inputs.items(), key=lambda kv: kv[1])[0]
            row["top_factor_score"] = float(max(top_inputs.values()))

        struct = _extract_struct_features(node_df)
        row.update(struct)

        probes = _extract_probe_features(probe)
        row.update(probes)

        pair = _extract_top_pair_interaction(var_rank_df)
        row.update(pair)

        rows.append(row)

    feat_df = pd.DataFrame(rows)
    if len(feat_df) == 0:
        return feat_df

    stations_df = pd.read_csv(stations_csv)
    # Standardize station IDs to 8-digit USGS gauge IDs to avoid leading-zero join issues.
    stations_df["gauge_id"] = stations_df["gauge_id"].astype(str).str.strip().str.zfill(8)
    feat_df["station_id"] = feat_df["station_id"].astype(str).str.strip().str.zfill(8)
    merged = feat_df.merge(
        stations_df,
        left_on="station_id",
        right_on="gauge_id",
        how="left",
    )

    if camels_root:
        camels_df = load_camels_attributes(camels_root)
        if len(camels_df) > 0:
            merged = merged.merge(camels_df, on="gauge_id", how="left")

    return merged


def _find_camels_files(camels_root: str) -> List[str]:
    files: List[str] = []
    if not os.path.isdir(camels_root):
        return files
    for root, _, names in os.walk(camels_root):
        for name in names:
            low = name.lower()
            if not (low.endswith(".txt") or low.endswith(".csv")):
                continue
            if "camels" in low or "attr" in low:
                files.append(os.path.join(root, name))
    return sorted(files)


def _detect_station_id_col(columns: Sequence[str]) -> str | None:
    for c in columns:
        low = c.lower()
        if low in ["gauge_id", "gage_id", "station_id", "gaugeid", "id"]:
            return c
    for c in columns:
        if "gauge" in c.lower() or "gage" in c.lower():
            return c
    return None


def load_camels_attributes(camels_root: str) -> pd.DataFrame:
    files = _find_camels_files(camels_root)
    if not files:
        return pd.DataFrame(columns=["gauge_id"])

    merged: pd.DataFrame | None = None
    for p in files:
        local = None
        for sep in [";", ",", "\t"]:
            try:
                local = pd.read_csv(p, sep=sep)
                if local is not None and len(local.columns) > 1:
                    break
            except Exception:
                local = None
        if local is None or len(local.columns) == 0:
            continue

        id_col = _detect_station_id_col(list(local.columns))
        if id_col is None:
            continue

        local = local.rename(columns={id_col: "gauge_id"})
        local["gauge_id"] = local["gauge_id"].astype(str).str.strip().str.zfill(8)

        for c in list(local.columns):
            if c == "gauge_id":
                continue
            try:
                local[c] = pd.to_numeric(local[c])
            except Exception:
                # Keep non-numeric columns (e.g., climate timing category) as-is.
                pass

        if merged is None:
            merged = local
        else:
            keep_cols = ["gauge_id"] + [c for c in local.columns if c not in set(merged.columns)]
            merged = merged.merge(local[keep_cols], on="gauge_id", how="outer")

    if merged is None:
        return pd.DataFrame(columns=["gauge_id"])

    return merged


@dataclass
class ClusterResult:
    labels: np.ndarray
    k: int
    score: float


def _build_feature_matrix(df: pd.DataFrame, cols: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    x = df.loc[:, list(cols)].apply(pd.to_numeric, errors="coerce")
    keep = x.notna().sum(axis=1) >= max(1, int(0.8 * x.shape[1]))
    x = x.loc[keep].copy()
    x = x.fillna(x.median(numeric_only=True))
    arr = x.to_numpy(dtype=float)

    # z-score normalization for clustering comparability.
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    arr = (arr - mean) / std
    return keep.to_numpy(), arr


def run_hierarchical_clustering(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    k: int,
    linkage_method: str = "ward",
    metric: str = "euclidean",
) -> ClusterResult:
    if k < 2:
        raise ValueError("k must be >= 2")
    if len(feature_cols) == 0:
        raise ValueError("feature_cols is empty")

    keep_mask, arr = _build_feature_matrix(df, feature_cols)
    if arr.shape[0] < k:
        raise ValueError("not enough valid stations for selected k")

    from scipy.cluster.hierarchy import fcluster, linkage
    from sklearn.metrics import silhouette_score

    z = linkage(arr, method=linkage_method, metric=metric)
    labels = fcluster(z, t=k, criterion="maxclust").astype(int)

    score = float("nan")
    if len(np.unique(labels)) > 1 and arr.shape[0] > len(np.unique(labels)):
        score = float(silhouette_score(arr, labels, metric=metric))

    full_labels = np.full(shape=(len(df),), fill_value=-1, dtype=int)
    full_labels[keep_mask] = labels
    return ClusterResult(labels=full_labels, k=k, score=score)


def suggest_feature_sets(df: pd.DataFrame) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}

    top_factor_cols = [c for c in df.columns if c.startswith("top_factor_") and c not in ["top_factor_name"]]
    struct_cols = [c for c in df.columns if c.startswith("struct_")]
    probe_cols = [c for c in df.columns if c.startswith("probe_r2_")]

    out["top_factor"] = sorted(top_factor_cols)
    out["top_struct_factor"] = sorted(struct_cols)
    out["probe"] = sorted(probe_cols)
    return out


def cluster_geo_climate_association(
    df: pd.DataFrame,
    cluster_col: str,
    candidate_cols: Iterable[str],
) -> pd.DataFrame:
    from scipy.stats import chi2_contingency, kruskal

    rows: List[Dict[str, Any]] = []
    work = df[df[cluster_col] > 0].copy()
    if len(work) == 0:
        return pd.DataFrame(columns=["variable", "test", "p_value", "effect"])

    for c in candidate_cols:
        if c not in work.columns:
            continue
        s = work[c]
        if s.notna().sum() < 10:
            continue

        if pd.api.types.is_numeric_dtype(s):
            groups = [g[c].dropna().to_numpy() for _, g in work.groupby(cluster_col)]
            groups = [g for g in groups if len(g) > 0]
            if len(groups) < 2:
                continue
            stat, p = kruskal(*groups)
            rows.append(
                {
                    "variable": c,
                    "test": "kruskal",
                    "p_value": float(p),
                    "effect": float(stat),
                }
            )
        else:
            tab = pd.crosstab(work[cluster_col], work[c])
            if tab.shape[0] < 2 or tab.shape[1] < 2:
                continue
            chi2, p, _, _ = chi2_contingency(tab)
            rows.append(
                {
                    "variable": c,
                    "test": "chi2",
                    "p_value": float(p),
                    "effect": float(chi2),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["variable", "test", "p_value", "effect"])
    return pd.DataFrame(rows).sort_values("p_value")
