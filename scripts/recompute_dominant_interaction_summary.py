#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Ensure `src` is importable even when running from outside repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.flood_transformer.map_analysis import load_station_feature_table


INTERACTION_ORDER = ["Q*Qtar", "P*Q", "Tmax*Q", "Tmin*Q", "SRAD*Q", "Other"]


def _normalize_name(x: object) -> str:
    if x is None:
        return ""
    return str(x).strip().upper()


def _classify_interaction_pair(name: object) -> str:
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return "Other"
    raw = str(name).strip().replace("*", "x")

    has_qtar = "Qtar" in raw
    has_q = "Q" in raw
    has_p = "P" in raw
    has_tmax = "Tmax" in raw
    has_tmin = "Tmin" in raw
    has_srad = "SRAD" in raw

    if has_qtar and has_q:
        return "Q*Qtar"
    if has_p and has_q and (not has_qtar) and (not has_tmax) and (not has_tmin) and (not has_srad):
        return "P*Q"
    if has_tmax and has_q and (not has_qtar):
        return "Tmax*Q"
    if has_tmin and has_q and (not has_qtar):
        return "Tmin*Q"
    if has_srad and has_q and (not has_qtar):
        return "SRAD*Q"
    return "Other"


def _fallback_merge_station_climate(feat_df: pd.DataFrame, workspace_root: str) -> pd.DataFrame:
    fallback_path = os.path.join(workspace_root, "data", "streamflow_station_name_lat_lon.csv")
    if not os.path.exists(fallback_path):
        return feat_df

    fb = pd.read_csv(fallback_path)
    required = {"gauge_name", "aridity", "mean_prcp_mm_day"}
    if not required.issubset(set(fb.columns)):
        return feat_df

    fb = fb.copy()
    fb["_gauge_name_norm"] = fb["gauge_name"].map(_normalize_name)

    out = feat_df.copy()
    if "gauge_name" in out.columns:
        out["_gauge_name_norm"] = out["gauge_name"].map(_normalize_name)
        use_cols = [c for c in ["_gauge_name_norm", "aridity", "mean_prcp_mm_day"] if c in fb.columns]
        out = out.merge(fb[use_cols], on="_gauge_name_norm", how="left", suffixes=("", "_fb"))
        out = out.drop(columns=["_gauge_name_norm"], errors="ignore")

        if "aridity" in out.columns and "aridity_fb" in out.columns:
            out["aridity"] = pd.to_numeric(out["aridity"], errors="coerce").fillna(pd.to_numeric(out["aridity_fb"], errors="coerce"))
            out = out.drop(columns=["aridity_fb"], errors="ignore")
        elif "aridity_fb" in out.columns:
            out["aridity"] = pd.to_numeric(out["aridity_fb"], errors="coerce")
            out = out.drop(columns=["aridity_fb"], errors="ignore")

        if "mean_prcp_mm_day_fb" in out.columns:
            out["mean_prcp_mm_day"] = pd.to_numeric(out["mean_prcp_mm_day_fb"], errors="coerce")
            out = out.drop(columns=["mean_prcp_mm_day_fb"], errors="ignore")

    return out


def _compute_geo_classes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def _col_as_numeric(name: str) -> pd.Series:
        if name in out.columns:
            return pd.to_numeric(out[name], errors="coerce")
        return pd.Series(np.nan, index=out.index, dtype=float)

    out["forcing_elev_m"] = _col_as_numeric("forcing_elev_m")
    out["lon"] = _col_as_numeric("lon")
    out["aridity"] = _col_as_numeric("aridity")

    out["aridity_class"] = pd.cut(
        out["aridity"],
        bins=[-1, 0.65, 1.0, 10],
        labels=["Humid", "Sub-humid", "Arid"],
    )

    frac_snow = _col_as_numeric("frac_snow")
    if frac_snow.notna().any():
        out["snow_regime"] = pd.cut(
            frac_snow,
            bins=[-0.1, 0.1, 0.3, 1.0],
            labels=["Rain-dominant", "Mixed", "Snow-influenced"],
        )
    else:
        out["snow_regime"] = pd.NA

    precip_col = None
    for c in ["p_mean", "mean_prcp_mm_day", "prcp_mean", "precip_mean"]:
        if c in out.columns and pd.to_numeric(out[c], errors="coerce").notna().any():
            precip_col = c
            break

    if precip_col is not None:
        p = pd.to_numeric(out[precip_col], errors="coerce")
        valid = p.notna()
        out["precipitation_class"] = pd.NA
        if valid.sum() >= 3:
            # Use ranked qcut to avoid duplicated-edge failures when many values are tied.
            p_rank = p[valid].rank(method="first")
            out.loc[valid, "precipitation_class"] = pd.qcut(
                p_rank,
                q=3,
                labels=["Low Precip", "Medium Precip", "High Precip"],
            )
    else:
        out["precipitation_class"] = pd.NA

    def climate_type(row: pd.Series) -> str:
        elev = float(row.get("forcing_elev_m")) if pd.notna(row.get("forcing_elev_m")) else 0.0
        lon = float(row.get("lon")) if pd.notna(row.get("lon")) else 0.0
        arid = float(row.get("aridity")) if pd.notna(row.get("aridity")) else 1.0
        if elev >= 1500:
            return "Plateau/Mountain"
        if lon <= -120:
            return "Oceanic"
        if arid >= 1.5:
            return "Arid Continental"
        return "Humid Continental"

    out["climate_type"] = out.apply(climate_type, axis=1)
    return out


def _summarize_sector(df: pd.DataFrame, sector_col: str, sector_name: str) -> pd.DataFrame:
    use = df[["interaction_group", sector_col]].dropna().copy()
    if use.empty:
        return pd.DataFrame(columns=["sector", "interaction_group", "station_type", "count", "group_total", "percent_within_group"])

    use["interaction_group"] = use["interaction_group"].astype(str)
    use[sector_col] = use[sector_col].astype(str)

    interaction_levels = [x for x in INTERACTION_ORDER if x in set(use["interaction_group"].unique())]
    remaining = sorted([x for x in use["interaction_group"].unique() if x not in interaction_levels])
    interaction_levels.extend(remaining)

    station_levels = list(dict.fromkeys(use[sector_col].tolist()))

    full_index = pd.MultiIndex.from_product([interaction_levels, station_levels], names=["interaction_group", "station_type"])
    cnt = (
        use.groupby(["interaction_group", sector_col], as_index=False)
        .size()
        .rename(columns={"size": "count", sector_col: "station_type"})
        .set_index(["interaction_group", "station_type"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    totals = cnt.groupby("interaction_group", as_index=False)["count"].sum().rename(columns={"count": "group_total"})
    out = cnt.merge(totals, on="interaction_group", how="left")
    out["percent_within_group"] = np.where(
        out["group_total"] > 0,
        np.round(out["count"] * 100.0 / out["group_total"], 2),
        0.0,
    )
    out.insert(0, "sector", sector_name)
    return out


def recompute_summary(output_root: str, stations_csv: str, camels_root: Optional[str], workspace_root: str) -> pd.DataFrame:
    if not camels_root:
        auto_camels = os.path.join(os.path.expanduser("~"), "WorkSpace", "CAMELS", "Data", "Raw")
        if os.path.isdir(auto_camels):
            camels_root = auto_camels

    feat_df = load_station_feature_table(
        output_root=output_root,
        stations_csv=stations_csv,
        camels_root=camels_root,
        top_input_n=8,
    )
    if feat_df.empty:
        raise RuntimeError(f"No station outputs found under: {output_root}")

    feat_df = _fallback_merge_station_climate(feat_df, workspace_root)
    feat_df = _compute_geo_classes(feat_df)

    feat_df["interaction_group"] = feat_df.get("top_pair_interaction_name", pd.Series(index=feat_df.index)).map(_classify_interaction_pair)
    feat_df = feat_df[feat_df["interaction_group"].notna()].copy()

    sectors = [
        ("aridity_class", "aridity_class"),
        ("precipitation_class", "precipitation_class"),
        ("climate_type", "climate_type"),
        ("snow_regime", "snow_regime"),
    ]

    parts = []
    for col, name in sectors:
        if col in feat_df.columns:
            part = _summarize_sector(feat_df, col, name)
            if not part.empty:
                parts.append(part)

    if not parts:
        raise RuntimeError("No valid sector columns available to summarize.")

    out = pd.concat(parts, axis=0, ignore_index=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute dominant interaction summary using streamlit_us_map/map_analysis feature logic")
    parser.add_argument("--output-root", default="/mnt/d/Store/Sparse_Transformer")
    parser.add_argument("--stations-csv", default="data/stations.csv")
    default_camels = os.path.join(os.path.expanduser("~"), "WorkSpace", "CAMELS", "Data", "Raw")
    parser.add_argument("--camels-root", default=default_camels if os.path.isdir(default_camels) else None)
    parser.add_argument("--workspace-root", default=os.getcwd())
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--also-write-workspace-copy", action="store_true")
    args = parser.parse_args()

    out_csv = args.out_csv or os.path.join(args.output_root, "circular_charts", "dominant_interaction_summary.csv")

    summary = recompute_summary(
        output_root=args.output_root,
        stations_csv=args.stations_csv,
        camels_root=args.camels_root,
        workspace_root=args.workspace_root,
    )

    try:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        summary.to_csv(out_csv, index=False)
        print(f"Wrote summary CSV: {out_csv}")
    except PermissionError as e:
        print(f"Warning: cannot write to {out_csv}: {e}")
    print(f"Rows: {len(summary)}")

    if args.also_write_workspace_copy:
        ws_csv = os.path.join(args.workspace_root, "dominant_interaction_summary.csv")
        summary.to_csv(ws_csv, index=False)
        print(f"Wrote workspace copy: {ws_csv}")


if __name__ == "__main__":
    main()
