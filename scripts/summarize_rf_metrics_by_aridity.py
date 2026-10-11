import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.flood_transformer.map_analysis import load_camels_attributes


def _pad_station_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.zfill(8)


def _detect_aridity_column(camels_df: pd.DataFrame) -> str:
    if "aridity" in camels_df.columns:
        return "aridity"

    candidates = [
        c
        for c in camels_df.columns
        if c.lower().startswith("aridity") or "aridity" in c.lower()
    ]
    if not candidates:
        raise ValueError("No aridity-related column found in CAMELS attributes.")
    return candidates[0]


def _resolve_sparse_root(cli_path: str) -> str:
    if cli_path and str(cli_path).strip():
        return os.path.abspath(os.path.expanduser(str(cli_path).strip()))

    preferred = "/mnt/d/Store/Sparse_Transformer"
    if os.path.isdir(preferred):
        return preferred
    return os.path.abspath(os.path.join(REPO_ROOT, "outputs"))


def _load_sparse_good_stations(sparse_root: str, nse_threshold: float) -> set[str]:
    threshold = float(nse_threshold)
    metrics_csv = os.path.join(sparse_root, "sparse_transformer_station_metrics.csv")
    if os.path.exists(metrics_csv):
        try:
            df = pd.read_csv(metrics_csv)
            if "station_id" in df.columns and "nse" in df.columns:
                sid = _pad_station_id(df["station_id"])
                nse = pd.to_numeric(df["nse"], errors="coerce")
                return set(sid[(np.isfinite(nse)) & (nse > threshold)].tolist())
        except Exception:
            pass

    good: set[str] = set()
    for name in os.listdir(sparse_root):
        d = os.path.join(sparse_root, name)
        if not (os.path.isdir(d) and str(name).isdigit()):
            continue
        summary_path = os.path.join(d, "summary.json")
        if not os.path.exists(summary_path):
            continue
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                s = json.load(f)
            nse_val = float((s or {}).get("metrics", {}).get("nse", np.nan))
            if np.isfinite(nse_val) and nse_val > threshold:
                good.add(str(name).zfill(8))
        except Exception:
            continue

    return good


def _ensure_mae(df: pd.DataFrame, mse_col: str, mae_col: str) -> pd.Series:
    if mae_col in df.columns:
        return pd.to_numeric(df[mae_col], errors="coerce")

    mse = pd.to_numeric(df[mse_col], errors="coerce")
    return np.sqrt(np.clip(mse, 0.0, None)) * np.sqrt(2.0 / np.pi)


def _summarize_metrics(df: pd.DataFrame, model: str) -> Dict[str, float]:
    if model == "baseline":
        mse = pd.to_numeric(df["mse_base"], errors="coerce")
        mae = _ensure_mae(df, "mse_base", "mae_base")
        nse = pd.to_numeric(df["nse_base"], errors="coerce")
        kge = pd.to_numeric(df["kge_base"], errors="coerce")
    else:
        mse = pd.to_numeric(df["mse_mod"], errors="coerce")
        mae = _ensure_mae(df, "mse_mod", "mae_mod")
        nse = pd.to_numeric(df["nse_mod"], errors="coerce")
        kge = pd.to_numeric(df["kge_mod"], errors="coerce")

    return {
        "MSE": float(mse.mean(skipna=True)),
        "MAE": float(mae.mean(skipna=True)),
        "NSE": float(nse.mean(skipna=True)),
        "KGE": float(kge.mean(skipna=True)),
    }


def build_table(rf_df: pd.DataFrame) -> pd.DataFrame:
    group_order: List[str] = ["Arid", "Humid", "Sub-humid", "All Stations"]
    model_order: List[str] = ["baseline", "modified"]

    rows = []
    for model in model_order:
        for grp in group_order:
            if grp == "All Stations":
                sub = rf_df
            else:
                sub = rf_df[rf_df["aridity_class"] == grp]

            vals = _summarize_metrics(sub, model=model)
            rows.append(
                {
                    "model": model,
                    "group": grp,
                    "MSE": vals["MSE"],
                    "MAE": vals["MAE"],
                    "NSE": vals["NSE"],
                    "KGE": vals["KGE"],
                }
            )

    out = pd.DataFrame(rows).set_index(["model", "group"])
    return out[["MSE", "MAE", "NSE", "KGE"]]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize RF metrics by aridity class (Arid/Humid/Sub-humid/All Stations)."
    )
    parser.add_argument(
        "--rf-csv",
        type=str,
        default="outputs/random_forest_model_performance.csv",
        help="Random forest performance CSV path.",
    )
    parser.add_argument(
        "--camels-root",
        type=str,
        default="/home/fifth/WorkSpace/CAMELS/Data/Raw",
        help="CAMELS raw attribute root path.",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="outputs/figures/rf_metrics_by_aridity_table.csv",
        help="Output CSV path for the aggregated table.",
    )
    parser.add_argument(
        "--sparse-output-root",
        type=str,
        default="",
        help="Sparse Transformer output root for station filtering.",
    )
    parser.add_argument(
        "--sparse-nse-threshold",
        type=float,
        default=0.5,
        help="Only keep stations with Sparse Transformer NSE above this threshold.",
    )
    args = parser.parse_args()

    rf_csv = os.path.abspath(os.path.expanduser(args.rf_csv))
    camels_root = os.path.abspath(os.path.expanduser(args.camels_root))
    out_csv = os.path.abspath(os.path.expanduser(args.out_csv))
    sparse_root = _resolve_sparse_root(args.sparse_output_root)

    if not os.path.exists(rf_csv):
        raise FileNotFoundError(f"RF CSV not found: {rf_csv}")
    if not os.path.isdir(camels_root):
        raise FileNotFoundError(f"CAMELS root not found: {camels_root}")
    if not os.path.isdir(sparse_root):
        raise FileNotFoundError(f"Sparse Transformer output root not found: {sparse_root}")

    rf_df = pd.read_csv(rf_csv)
    required = ["station_id", "mse_base", "nse_base", "kge_base", "mse_mod", "nse_mod", "kge_mod"]
    missing = [c for c in required if c not in rf_df.columns]
    if missing:
        raise ValueError(f"RF CSV missing required columns: {missing}")

    rf_df = rf_df.copy()
    rf_df["station_id"] = _pad_station_id(rf_df["station_id"])

    sparse_good = _load_sparse_good_stations(
        sparse_root=sparse_root,
        nse_threshold=float(args.sparse_nse_threshold),
    )
    if not sparse_good:
        raise RuntimeError(
            f"No stations satisfy Sparse Transformer NSE > {args.sparse_nse_threshold:.2f} under {sparse_root}"
        )
    rf_df = rf_df[rf_df["station_id"].isin(sparse_good)].copy()
    if len(rf_df) == 0:
        raise RuntimeError("No overlapping stations between RF results and Sparse Transformer NSE-qualified set.")

    camels_df = load_camels_attributes(camels_root)
    if len(camels_df) == 0:
        raise RuntimeError(f"No CAMELS attributes loaded from: {camels_root}")

    aridity_col = _detect_aridity_column(camels_df)
    aridity_df = camels_df[["gauge_id", aridity_col]].copy()
    aridity_df["gauge_id"] = _pad_station_id(aridity_df["gauge_id"])
    aridity_df["aridity"] = pd.to_numeric(aridity_df[aridity_col], errors="coerce")

    merged = rf_df.merge(aridity_df[["gauge_id", "aridity"]], left_on="station_id", right_on="gauge_id", how="left")
    merged["aridity_class"] = pd.cut(
        merged["aridity"],
        bins=[-1, 0.65, 1.0, 10],
        labels=["Humid", "Sub-humid", "Arid"],
    )

    table = build_table(merged)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    table.to_csv(out_csv)

    with pd.option_context("display.max_rows", 50, "display.max_columns", None, "display.width", 160):
        print(table)

    print(f"\nAridity source column: {aridity_col}")
    print(f"Sparse source: {sparse_root}")
    print(f"Sparse filter: NSE > {args.sparse_nse_threshold:.2f}")
    print(f"Qualified stations used: {len(rf_df)}")
    print(f"Output table saved to: {out_csv}")


if __name__ == "__main__":
    main()
