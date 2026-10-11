import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import stats


def load_sparse_good_stations(sparse_output_root: str, nse_threshold: float) -> set[str]:
    threshold = float(nse_threshold)
    csv_path = os.path.join(sparse_output_root, "sparse_transformer_station_metrics.csv")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if "station_id" in df.columns and "nse" in df.columns:
            sid = df["station_id"].astype(str).str.strip().str.zfill(8)
            nse_vals = pd.to_numeric(df["nse"], errors="coerce")
            mask = np.isfinite(nse_vals) & (nse_vals > threshold)
            return set(sid[mask].tolist())

    good = set()
    for name in os.listdir(sparse_output_root):
        d = os.path.join(sparse_output_root, name)
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

    return good


def resolve_default_sparse_root(local_outputs: str) -> str:
    preferred = "/mnt/d/Store/Sparse_Transformer"
    if os.path.isdir(preferred):
        return preferred
    return local_outputs


def main():
    parser = argparse.ArgumentParser(description="Summarize RF NSE on Sparse-Transformer-good stations without retraining")
    parser.add_argument(
        "--rf-results",
        type=str,
        default="",
        help="Path to RF result CSV (default: outputs/random_forest_model_performance.csv)",
    )
    parser.add_argument(
        "--sparse-output-root",
        type=str,
        default="",
        help="Sparse Transformer output root (default: /mnt/d/Store/Sparse_Transformer if exists, else outputs)",
    )
    parser.add_argument(
        "--sparse-nse-threshold",
        type=float,
        default=0.5,
        help="Keep stations with Sparse Transformer NSE > threshold",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="",
        help="Optional path to write filtered station table",
    )
    args = parser.parse_args()

    local_outputs = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
    rf_results = (
        os.path.abspath(os.path.expanduser(args.rf_results))
        if args.rf_results.strip()
        else os.path.join(local_outputs, "random_forest_model_performance.csv")
    )
    sparse_output_root = (
        os.path.abspath(os.path.expanduser(args.sparse_output_root))
        if args.sparse_output_root.strip()
        else resolve_default_sparse_root(local_outputs)
    )

    if not os.path.exists(rf_results):
        raise FileNotFoundError(f"RF result CSV not found: {rf_results}")
    if not os.path.isdir(sparse_output_root):
        raise FileNotFoundError(f"Sparse Transformer output root not found: {sparse_output_root}")

    rf_df = pd.read_csv(rf_results)
    required_cols = {"station_id", "nse_base", "nse_mod"}
    missing = required_cols - set(rf_df.columns)
    if missing:
        raise ValueError(f"RF result CSV missing columns: {sorted(missing)}")

    rf_df = rf_df.copy()
    rf_df["station_id"] = rf_df["station_id"].astype(str).str.strip().str.zfill(8)

    sparse_good = load_sparse_good_stations(
        sparse_output_root=sparse_output_root,
        nse_threshold=float(args.sparse_nse_threshold),
    )
    if not sparse_good:
        raise RuntimeError(
            f"No stations passed Sparse Transformer NSE > {args.sparse_nse_threshold:.2f} under {sparse_output_root}"
        )

    stat_df = rf_df[rf_df["station_id"].isin(sparse_good)].copy()

    print(f"RF results source: {rf_results}")
    print(f"Sparse source: {sparse_output_root}")
    print(f"Subset rule: Sparse Transformer NSE > {args.sparse_nse_threshold:.2f}")
    print(f"Stations matched: {len(stat_df)}/{len(rf_df)}")

    if len(stat_df) == 0:
        print("No matched stations in RF results.")
        return

    mean_base = float(pd.to_numeric(stat_df["nse_base"], errors="coerce").mean())
    mean_mod = float(pd.to_numeric(stat_df["nse_mod"], errors="coerce").mean())
    print(f"Mean NSE (baseline): {mean_base:.4f}")
    print(f"Mean NSE (modified): {mean_mod:.4f}")

    mask = stat_df["nse_base"].notna() & stat_df["nse_mod"].notna()
    if int(mask.sum()) >= 5:
        tstat, pval = stats.ttest_rel(stat_df.loc[mask, "nse_base"], stat_df.loc[mask, "nse_mod"])
        mean_diff = (stat_df.loc[mask, "nse_mod"] - stat_df.loc[mask, "nse_base"]).mean()
        print(f"Paired t-test (NSE mod - NSE base): t={tstat:.4f}, p={pval:.4e}, mean diff={mean_diff:.4f}, n={int(mask.sum())}")
    else:
        print(f"Not enough paired values for t-test (n={int(mask.sum())}).")

    if "regime" in stat_df.columns:
        regime_counts = stat_df["regime"].value_counts(dropna=False)
        print(
            "Regime dominance: "
            f"P-Q={int(regime_counts.get('P-Q', 0))}, "
            f"Q-Q={int(regime_counts.get('Q-Q', 0))}, "
            f"T-Q={int(regime_counts.get('T-Q', 0))}"
        )

    if args.out_csv.strip():
        out_csv = os.path.abspath(os.path.expanduser(args.out_csv))
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        stat_df.to_csv(out_csv, index=False)
        print(f"Wrote filtered station table: {out_csv}")


if __name__ == "__main__":
    main()
