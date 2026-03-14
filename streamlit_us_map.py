from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.flood_transformer.map_analysis import (
    cluster_geo_climate_association,
    load_station_feature_table,
    run_hierarchical_clustering,
    suggest_feature_sets,
)


st.set_page_config(page_title="US Station Cluster Map", layout="wide")
st.title("CAMELS US: Explainability Cluster Map")


@st.cache_data(show_spinner=False)
def _load_features(output_root: str, stations_csv: str, camels_root: str | None) -> pd.DataFrame:
    return load_station_feature_table(
        output_root=output_root,
        stations_csv=stations_csv,
        camels_root=camels_root,
        top_input_n=8,
    )


with st.sidebar:
    st.header("Data")
    output_root = st.text_input("Output root", value="outputs")
    stations_csv = st.text_input("Stations CSV", value="data/stations.csv")
    camels_root = st.text_input("CAMELS attribute root", value="")
    camels_root = camels_root.strip() or None

    st.header("Clustering")
    target_mode = st.selectbox(
        "Target object",
        options=["top_factor", "top_struct_factor", "probe"],
        index=0,
    )
    k = st.slider("Number of clusters (k)", min_value=2, max_value=12, value=4, step=1)
    linkage_method = st.selectbox("Linkage", options=["ward", "average", "complete"], index=0)
    metric = st.selectbox("Distance metric", options=["euclidean", "cosine"], index=0)


if not os.path.exists(stations_csv):
    st.error(f"stations csv not found: {stations_csv}")
    st.stop()

if not os.path.isdir(output_root):
    st.error(f"output root not found: {output_root}")
    st.stop()

if camels_root and (not os.path.isdir(camels_root)):
    st.warning("CAMELS root does not exist in current workspace runtime. Continue with station+output features only.")
    camels_root = None

with st.spinner("Loading station features..."):
    feat_df = _load_features(output_root, stations_csv, camels_root)

if len(feat_df) == 0:
    st.warning("No station outputs found (missing summary.json in outputs/*).")
    st.stop()

feature_sets = suggest_feature_sets(feat_df)
selected_cols: List[str] = feature_sets.get(target_mode, [])
if len(selected_cols) == 0:
    st.error(f"No features available for target mode: {target_mode}")
    st.stop()

try:
    clustering = run_hierarchical_clustering(
        df=feat_df,
        feature_cols=selected_cols,
        k=k,
        linkage_method=linkage_method,
        metric=metric,
    )
except Exception as e:
    st.error(f"Clustering failed: {e}")
    st.stop()

plot_df = feat_df.copy()
cluster_col = f"cluster_{target_mode}"
plot_df[cluster_col] = clustering.labels
plot_df["cluster_label"] = plot_df[cluster_col].map(lambda x: "invalid" if x <= 0 else f"C{x}")

st.caption(
    f"Stations: {len(plot_df)} | mode: {target_mode} | silhouette: {clustering.score:.3f}"
)

cols_top = st.columns(4)
with cols_top[0]:
    nse_min = st.number_input("Min NSE", value=0.0, step=0.05)
with cols_top[1]:
    huc_list = sorted([str(x) for x in plot_df["huc_02"].dropna().unique()]) if "huc_02" in plot_df.columns else []
    sel_huc = st.multiselect("HUC-02", options=huc_list, default=[])
with cols_top[2]:
    color_mode = st.selectbox("Color by", options=["cluster", "nse", "probe_r2_mean", "top_factor_score"], index=0)
with cols_top[3]:
    size_mode = st.selectbox("Size by", options=["fixed", "top_factor_score", "struct_total", "probe_r2_mean"], index=0)

mask = pd.Series(True, index=plot_df.index)
if "nse" in plot_df.columns:
    mask &= pd.to_numeric(plot_df["nse"], errors="coerce").fillna(-999) >= float(nse_min)
if sel_huc and "huc_02" in plot_df.columns:
    mask &= plot_df["huc_02"].astype(str).isin(sel_huc)
plot_df = plot_df.loc[mask].copy()

if len(plot_df) == 0:
    st.warning("No stations after filtering.")
    st.stop()

plot_df["lat"] = pd.to_numeric(plot_df.get("lat"), errors="coerce")
plot_df["lon"] = pd.to_numeric(plot_df.get("lon"), errors="coerce")
plot_df = plot_df.dropna(subset=["lat", "lon"])

if len(plot_df) == 0:
    st.warning("No valid lat/lon after merge with stations.csv.")
    st.stop()

if color_mode == "cluster":
    color_col = "cluster_label"
    color_continuous_scale = None
else:
    color_col = color_mode
    color_continuous_scale = "Viridis"

if size_mode == "fixed" or size_mode not in plot_df.columns:
    plot_df["marker_size"] = 8.0
else:
    z = pd.to_numeric(plot_df[size_mode], errors="coerce")
    z = z.fillna(z.median())
    zmin, zmax = float(z.min()), float(z.max())
    if zmax <= zmin:
        plot_df["marker_size"] = 8.0
    else:
        plot_df["marker_size"] = 6.0 + 12.0 * (z - zmin) / (zmax - zmin)

hover_cols = [
    "station_id",
    "gauge_name",
    "huc_02",
    "nse",
    "top_factor_name",
    "top_factor_score",
    "probe_r2_mean",
    cluster_col,
]
hover_cols = [c for c in hover_cols if c in plot_df.columns]

fig = px.scatter_geo(
    plot_df,
    lat="lat",
    lon="lon",
    color=color_col,
    size="marker_size",
    size_max=20,
    hover_name="station_id",
    hover_data=hover_cols,
    scope="usa",
    title="US Stations clustered by explainability features",
    color_continuous_scale=color_continuous_scale,
)
fig.update_layout(height=650, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)

st.subheader("Cluster Summary")
summary_cols = [cluster_col]
for c in ["nse", "top_factor_score", "struct_total", "probe_r2_mean"]:
    if c in plot_df.columns:
        summary_cols.append(c)
cluster_summary = plot_df[summary_cols].groupby(cluster_col, as_index=False).mean(numeric_only=True)
cluster_counts = plot_df.groupby(cluster_col).size().reset_index(name="count")
cluster_summary = cluster_summary.merge(cluster_counts, on=cluster_col, how="left").sort_values(cluster_col)
st.dataframe(cluster_summary, use_container_width=True)

st.subheader("Station Detail")
default_station = str(plot_df.iloc[0]["station_id"])
station_id = st.selectbox("Station", options=sorted(plot_df["station_id"].astype(str).unique()), index=0)
detail = plot_df[plot_df["station_id"].astype(str) == str(station_id)].head(1)
if len(detail) > 0:
    display_cols = [
        "station_id",
        "gauge_name",
        "huc_02",
        "lat",
        "lon",
        "nse",
        "top_factor_name",
        "top_factor_score",
        "probe_r2_cum_rain",
        "probe_r2_api",
        "probe_r2_dpdt",
        cluster_col,
    ]
    display_cols = [c for c in display_cols if c in detail.columns]
    st.dataframe(detail[display_cols], use_container_width=True)

st.subheader("Geo/Climate Association Test")
num_candidates = [
    "lat",
    "lon",
    "drainage_area_km2",
    "forcing_elev_m",
    "probe_r2_mean",
]
num_candidates.extend([c for c in plot_df.columns if c.startswith("p_mean") or c.startswith("pet_mean") or c.startswith("aridity")])
num_candidates = [c for c in num_candidates if c in plot_df.columns]

if len(num_candidates) == 0 and "huc_02" not in plot_df.columns:
    st.info("No geo/climate variables available for significance test.")
else:
    test_cols = list(dict.fromkeys(num_candidates + (["huc_02"] if "huc_02" in plot_df.columns else [])))
    test_df = cluster_geo_climate_association(plot_df, cluster_col, test_cols)
    if len(test_df) == 0:
        st.info("No valid test results (insufficient data per cluster/variable).")
    else:
        st.dataframe(test_df, use_container_width=True)

st.download_button(
    "Download station table CSV",
    data=plot_df.to_csv(index=False).encode("utf-8"),
    file_name=f"station_cluster_table_{target_mode}.csv",
    mime="text/csv",
)
