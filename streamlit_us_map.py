from __future__ import annotations

import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from src.flood_transformer.map_analysis import (
    cluster_geo_climate_association,
    load_station_feature_table,
    run_hierarchical_clustering,
    suggest_feature_sets,
)


st.set_page_config(page_title="US Station Cluster Map", layout="wide")
st.title("CAMELS US: Explainability Cluster Map")


def _configure_plotly_font_template() -> None:
    axis_style = dict(
        title_font=dict(family="Times New Roman", size=16),
        tickfont=dict(family="Times New Roman", size=12),
    )
    pio.templates["times_new_roman_axis"] = go.layout.Template(
        layout=go.Layout(
            font=dict(family="Times New Roman", size=13, color="#222222"),
            xaxis=axis_style,
            yaxis=axis_style,
            yaxis2=axis_style,
        )
    )
    pio.templates.default = "plotly+times_new_roman_axis"


_configure_plotly_font_template()


def _trace_label(i: int, trace: Any) -> str:
    tname = str(getattr(trace, "name", "") or "").strip()
    if tname:
        return f"{i}: {tname}"
    ttype = str(getattr(trace, "type", "trace"))
    return f"{i}: {ttype}"


def _apply_export_style(
    fig: go.Figure,
    title_text: str,
    x_axis_title: str,
    y_axis_title: str,
    y2_axis_title: str,
    font_color: str,
    title_size: int,
    axis_title_size: int,
    tick_size: int,
    legend_size: int,
    legend_title: str,
    show_legend: bool,
    legend_pos: str,
    keep_trace_indices: set[int],
    renamed_trace_names: Dict[int, str],
    width_px: int,
    height_px: int,
) -> go.Figure:
    preview = go.Figure(fig)

    legend_cfg = {
        "top-right": dict(x=1.0, y=1.0, xanchor="right", yanchor="top", orientation="v"),
        "top-left": dict(x=0.0, y=1.0, xanchor="left", yanchor="top", orientation="v"),
        "bottom-right": dict(x=1.0, y=0.0, xanchor="right", yanchor="bottom", orientation="v"),
        "bottom-left": dict(x=0.0, y=0.0, xanchor="left", yanchor="bottom", orientation="v"),
        "top-horizontal": dict(x=0.5, y=1.05, xanchor="center", yanchor="bottom", orientation="h"),
        "bottom-horizontal": dict(x=0.5, y=-0.2, xanchor="center", yanchor="top", orientation="h"),
    }
    legend_dict = legend_cfg.get(legend_pos, legend_cfg["top-right"])

    for i, tr in enumerate(preview.data):
        tr.name = renamed_trace_names.get(i, tr.name)
        tr.visible = True if i in keep_trace_indices else False
        tr.showlegend = bool(show_legend and (i in keep_trace_indices))

    preview.update_layout(
        title=title_text,
        width=int(width_px),
        height=int(height_px),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family="Times New Roman", color=font_color),
        title_font=dict(family="Times New Roman", size=title_size, color=font_color),
        legend=dict(
            title=dict(text=legend_title, font=dict(family="Times New Roman", size=legend_size, color=font_color)),
            font=dict(family="Times New Roman", size=legend_size, color=font_color),
            **legend_dict,
        ),
        xaxis=dict(
            title=dict(text=x_axis_title, font=dict(family="Times New Roman", size=axis_title_size, color=font_color)),
            tickfont=dict(family="Times New Roman", size=tick_size, color=font_color),
            showgrid=True,
            gridcolor="rgba(170,170,170,0.55)",
            zeroline=True,
            zerolinecolor="rgba(140,140,140,0.85)",
        ),
        yaxis=dict(
            title=dict(text=y_axis_title, font=dict(family="Times New Roman", size=axis_title_size, color=font_color)),
            tickfont=dict(family="Times New Roman", size=tick_size, color=font_color),
            showgrid=True,
            gridcolor="rgba(170,170,170,0.55)",
            zeroline=True,
            zerolinecolor="rgba(140,140,140,0.85)",
        ),
        yaxis2=dict(
            title=dict(text=y2_axis_title, font=dict(family="Times New Roman", size=axis_title_size, color=font_color)),
            tickfont=dict(family="Times New Roman", size=tick_size, color=font_color),
            showgrid=True,
            gridcolor="rgba(170,170,170,0.55)",
            zeroline=True,
            zerolinecolor="rgba(140,140,140,0.85)",
        ),
    )
    return preview


def _default_chrome_path_wsl() -> str:
    candidates = [
        "/usr/bin/google-chrome",
        "/usr/bin/chromium-browser",
        "/usr/bin/chromium",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return ""


def _configure_kaleido_chrome(chrome_path: str) -> None:
    if not chrome_path:
        return
    os.environ["BROWSER_PATH"] = chrome_path
    os.environ["CHROME_PATH"] = chrome_path
    os.environ["PLOTLY_CHROME_PATH"] = chrome_path


def _show_export_studio(fig: go.Figure, chart_key: str, default_file_name: str) -> None:
    pop = st.popover(f"Open Export Studio: {default_file_name}", use_container_width=False)
    with pop:
        st.caption("弹出调试面板：修改参数后点击 Update Preview，避免主界面频繁重渲染。")

        traces = list(fig.data)
        trace_options = [_trace_label(i, tr) for i, tr in enumerate(traces)]
        all_idx = set(range(len(traces)))
        layout_json = fig.to_plotly_json().get("layout", {})

        default_title = str(getattr(fig.layout.title, "text", "") or "")
        default_x = str(getattr(fig.layout.xaxis.title, "text", "") or "")
        default_y = str(getattr(fig.layout.yaxis.title, "text", "") or "")
        default_y2 = str(layout_json.get("yaxis2", {}).get("title", {}).get("text", "") or "")
        default_legend_title = str(getattr(fig.layout.legend.title, "text", "") or "")
        base_width = int(getattr(fig.layout, "width", None) or 1400)
        base_height = int(getattr(fig.layout, "height", None) or 800)
        default_chrome = st.session_state.get(f"{chart_key}_chrome_path", _default_chrome_path_wsl())

        with st.form(key=f"{chart_key}_export_form", clear_on_submit=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                out_format = st.selectbox("Format", options=["svg", "pdf", "png", "jpg", "webp"], index=0, key=f"{chart_key}_fmt")
                dpi = st.number_input("DPI", min_value=72, max_value=1200, value=300, step=12, key=f"{chart_key}_dpi")
                width = st.number_input("Width (px)", min_value=320, max_value=6000, value=base_width, step=20, key=f"{chart_key}_w")
                height = st.number_input("Height (px)", min_value=240, max_value=6000, value=base_height, step=20, key=f"{chart_key}_h")
            with c2:
                title_text = st.text_input("Figure Title", value=default_title, key=f"{chart_key}_title")
                x_axis_title = st.text_input("X Axis Name", value=default_x, key=f"{chart_key}_x")
                y_axis_title = st.text_input("Y Axis Name", value=default_y, key=f"{chart_key}_y")
                y2_axis_title = st.text_input("Y2 Axis Name", value=default_y2, key=f"{chart_key}_y2")
                legend_title = st.text_input("Legend Title", value=default_legend_title, key=f"{chart_key}_legend_title")
            with c3:
                font_color = st.color_picker("Font Color", value="#222222", key=f"{chart_key}_font_color")
                title_size = st.slider("Title Font Size", 8, 64, 22, key=f"{chart_key}_title_size")
                axis_title_size = st.slider("Axis Title Font Size", 8, 48, 16, key=f"{chart_key}_axis_title_size")
                tick_size = st.slider("Axis Tick Font Size", 6, 36, 12, key=f"{chart_key}_tick_size")
                legend_size = st.slider("Legend Font Size", 6, 36, 12, key=f"{chart_key}_legend_size")

            st.markdown("Kaleido / Chrome")
            chrome_path = st.text_input(
                "Chrome executable path (Linux/WSL)",
                value=default_chrome,
                key=f"{chart_key}_chrome_path",
            )

            c4, c5 = st.columns(2)
            with c4:
                show_legend = st.checkbox("Show Legend", value=True, key=f"{chart_key}_show_legend")
                legend_pos = st.selectbox(
                    "Legend Position",
                    options=["top-right", "top-left", "bottom-right", "bottom-left", "top-horizontal", "bottom-horizontal"],
                    index=0,
                    key=f"{chart_key}_legend_pos",
                )
            with c5:
                keep_labels = st.multiselect(
                    "Keep Traces (Legend/Series)",
                    options=trace_options,
                    default=trace_options,
                    key=f"{chart_key}_keep_traces",
                )

            renamed_trace_names: Dict[int, str] = {}
            if traces:
                st.markdown("Trace Names")
                for i, tr in enumerate(traces):
                    default_name = str(getattr(tr, "name", "") or f"trace_{i}")
                    renamed_trace_names[i] = st.text_input(
                        f"Trace {i} name",
                        value=default_name,
                        key=f"{chart_key}_trace_name_{i}",
                    )

            submit = st.form_submit_button("Update Preview")

        preview_state_key = f"{chart_key}_preview_fig"
        preview_format_key = f"{chart_key}_preview_fmt"
        preview_bytes_key = f"{chart_key}_preview_bytes"

        if submit or (preview_state_key not in st.session_state):
            keep_trace_indices = set()
            for i, label in enumerate(trace_options):
                if label in keep_labels:
                    keep_trace_indices.add(i)
            if len(keep_trace_indices) == 0 and len(trace_options) > 0:
                keep_trace_indices = set(all_idx)

            preview = _apply_export_style(
                fig=fig,
                title_text=title_text,
                x_axis_title=x_axis_title,
                y_axis_title=y_axis_title,
                y2_axis_title=y2_axis_title,
                font_color=font_color,
                title_size=int(title_size),
                axis_title_size=int(axis_title_size),
                tick_size=int(tick_size),
                legend_size=int(legend_size),
                legend_title=legend_title,
                show_legend=show_legend,
                legend_pos=legend_pos,
                keep_trace_indices=keep_trace_indices,
                renamed_trace_names=renamed_trace_names,
                width_px=int(width),
                height_px=int(height),
            )
            st.session_state[preview_state_key] = preview

            scale = float(dpi) / 96.0
            if scale <= 0:
                scale = 1.0
            try:
                _configure_kaleido_chrome(str(chrome_path).strip())
                image_bytes = preview.to_image(
                    format=out_format,
                    width=int(width),
                    height=int(height),
                    scale=scale,
                )
                st.session_state[preview_bytes_key] = image_bytes
                st.session_state[preview_format_key] = out_format
            except Exception as e:
                st.session_state[preview_bytes_key] = None
                st.session_state[preview_format_key] = out_format
                st.error(f"Export failed. Details: {e}")
                st.info(
                    "请在 WSL 环境安装 Linux Chrome/Chromium，并在此填写其可执行路径（例如 /usr/bin/google-chrome）。"
                )

        preview_to_show = st.session_state.get(preview_state_key)
        if preview_to_show is not None:
            image_bytes = st.session_state.get(preview_bytes_key)
            image_fmt = str(st.session_state.get(preview_format_key, "svg"))
            c_download, c_hint = st.columns([1, 2])
            with c_download:
                if image_bytes:
                    st.download_button(
                        "Download Figure",
                        data=image_bytes,
                        file_name=f"{default_file_name}.{image_fmt}",
                        mime="application/octet-stream",
                        key=f"{chart_key}_download",
                    )
            with c_hint:
                st.caption("After clicking Update Preview, download appears here.")

            st.plotly_chart(preview_to_show, use_container_width=False, key=f"{chart_key}_preview")


def _plotly_chart_with_export(fig: go.Figure, chart_key: str, default_file_name: str, use_container_width: bool = True) -> None:
    st.plotly_chart(fig, use_container_width=use_container_width, key=f"{chart_key}_chart")
    _show_export_studio(fig, chart_key=chart_key, default_file_name=default_file_name)


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
_plotly_chart_with_export(fig, chart_key="us_cluster_map", default_file_name="us_cluster_map")

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
