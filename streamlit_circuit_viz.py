from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.flood_transformer.map_analysis import (
    cluster_geo_climate_association,
    load_station_feature_table,
    run_hierarchical_clustering,
    suggest_feature_sets,
)


st.set_page_config(page_title="Flood Circuit Visualizer", layout="wide")


def _safe_read_json(path: str):
    mtime = _file_mtime(path)
    return _safe_read_json_cached(path, mtime)


def _safe_read_csv(path: str):
    mtime = _file_mtime(path)
    return _safe_read_csv_cached(path, mtime)


def _load_edge_importance_table(output_dir: str) -> Tuple[pd.DataFrame | None, str]:
    # Prefer fully recomputed real-edge files.
    candidates = [
        "edge_importance_ranking_real_full.csv",
        "edge_importance_ranking_real.csv",
        "edge_importance_ranking.csv",
    ]
    for name in candidates:
        p = os.path.join(output_dir, name)
        if os.path.exists(p):
            return _safe_read_csv(p), name
    return None, ""


def _load_in_node_records(output_dir: str) -> Tuple[pd.DataFrame | None, str]:
    bundle_path = os.path.join(output_dir, "real_importance_bundle.json")
    if os.path.exists(bundle_path):
        b = _safe_read_json(bundle_path)
        if isinstance(b, dict) and isinstance(b.get("in_node_interactions"), dict):
            rec = b.get("in_node_interactions", {}).get("records", [])
            if isinstance(rec, list) and len(rec) > 0:
                return pd.DataFrame(rec), "real_importance_bundle.json"

    candidates = [
        "in_node_interactions_ranking.csv",
        "in_node_interactions.json",
    ]
    for name in candidates:
        p = os.path.join(output_dir, name)
        if not os.path.exists(p):
            continue
        if name.endswith(".csv"):
            df = _safe_read_csv(p)
            if df is not None and len(df) > 0:
                return df, name
        else:
            j = _safe_read_json(p)
            if isinstance(j, dict) and isinstance(j.get("records"), list) and len(j.get("records", [])) > 0:
                return pd.DataFrame(j.get("records", [])), name
    return None, ""


def _build_in_edge_real_importance(
    in_node_df: pd.DataFrame | None,
    edges: List[Tuple[str, str]],
) -> Dict[Tuple[str, str], float]:
    if in_node_df is None or len(in_node_df) == 0:
        return {}
    if not {"feature", "node_id"}.issubset(set(in_node_df.columns)):
        return {}

    val_col = "abs_interaction" if "abs_interaction" in in_node_df.columns else "interaction" if "interaction" in in_node_df.columns else None
    if val_col is None:
        return {}

    edge_set = set(edges)
    out: Dict[Tuple[str, str], float] = {}
    for _, r in in_node_df.iterrows():
        feat = str(r.get("feature", ""))
        nid = str(r.get("node_id", ""))
        src = f"IN:{_abbr_feature(feat)}"
        e = (src, nid)
        if e not in edge_set:
            continue
        try:
            v = abs(float(r.get(val_col, 0.0)))
        except Exception:
            v = 0.0
        out[e] = max(out.get(e, 0.0), v)

    if len(out) == 0:
        return {}
    vmax = max(out.values())
    if vmax > 0:
        for k in list(out.keys()):
            out[k] = float(out[k] / vmax)
    return out


def _inject_in_real_edges(
    edge_imp: Dict[Tuple[str, str], float],
    in_real_imp: Dict[Tuple[str, str], float],
    in_real_weight: float = 1.0,
) -> Dict[Tuple[str, str], float]:
    if len(in_real_imp) == 0:
        return edge_imp
    w = float(max(0.0, min(1.0, in_real_weight)))
    out = dict(edge_imp)
    for e, rv in in_real_imp.items():
        hv = float(out.get(e, 0.0))
        out[e] = float(w * rv + (1.0 - w) * hv)
    return out


def _file_mtime(path: str) -> float | None:
    if not os.path.exists(path):
        return None
    try:
        return float(os.path.getmtime(path))
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _safe_read_json_cached(path: str, mtime: float | None):
    del mtime
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def _safe_read_csv_cached(path: str, mtime: float | None):
    del mtime
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _save_plot_png(fig: go.Figure, station_id: str, image_name: str, out_dir: str = "pic", scale: int = 3) -> str:
    os.makedirs(out_dir, exist_ok=True)
    safe_station = station_id.strip() if station_id else "unknown"
    out_path = os.path.join(out_dir, f"{safe_station}_{image_name}.png")

    export_fig = go.Figure(fig)
    export_fig.update_layout(title=None)
    export_fig.write_image(out_path, format="png", scale=scale)
    return out_path


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
    if "up" in low and "q" in low:
        return "Qup"
    if low in ["q", "flow", "streamflow"]:
        return "Q"
    parts = [p for p in name.replace("-", "_").split("_") if p]
    if not parts:
        return name[:4]
    return "".join([p[0].upper() for p in parts])[:5]


def _safe_positive(x: float) -> float:
    try:
        return float(max(0.0, x))
    except Exception:
        return 0.0


def build_circuit_edges(
    circuit: Dict,
    max_neurons_per_layer: int = 24,
    input_features: List[str] | None = None,
) -> Tuple[List[str], List[Tuple[str, str]], Dict[str, str]]:
    display_map: Dict[str, str] = {}
    input_features = input_features or []
    input_nodes = sorted([f"IN:{_abbr_feature(f)}" for f in input_features])
    active_inputs = circuit.get("active_inputs", [])
    if input_features and len(active_inputs) == len(input_features):
        kept = []
        for i, feat in enumerate(input_features):
            if float(active_inputs[i]) > 0:
                kept.append(f"IN:{_abbr_feature(feat)}")
        if kept:
            input_nodes = sorted(kept)
    if not input_nodes:
        input_nodes = ["IN:P", "IN:T", "IN:PET"]

    nodes = input_nodes.copy()
    edges: List[Tuple[str, str]] = []

    active_heads = circuit.get("active_heads", [])
    active_neurons = circuit.get("active_neurons", [])

    for n in input_nodes:
        display_map[n] = n

    last_nodes = input_nodes
    for l_idx, (h_mask, n_mask) in enumerate(zip(active_heads, active_neurons)):
        h_idx = [i for i, v in enumerate(h_mask) if v > 0]
        n_idx = [i for i, v in enumerate(n_mask) if v > 0][:max_neurons_per_layer]

        h_nodes = [f"L{l_idx}.H{i}" for i in h_idx]
        n_nodes = [f"L{l_idx}.N{i}" for i in n_idx]

        for h in h_nodes:
            display_map[h] = h.replace(".", "-")
        for n in n_nodes:
            display_map[n] = n.replace(".", "-")

        nodes.extend(h_nodes)
        nodes.extend(n_nodes)

        if h_nodes:
            for src in last_nodes:
                for dst in h_nodes:
                    edges.append((src, dst))

            for h in h_nodes:
                for n in n_nodes:
                    edges.append((h, n))

            last_nodes = n_nodes if n_nodes else h_nodes
        elif n_nodes:
            # 当该层没有激活 head 时，使用旁路连接保持层间连续
            for src in last_nodes:
                for dst in n_nodes:
                    edges.append((src, dst))
            last_nodes = n_nodes
        else:
            # 该层完全被剪空：保持 last_nodes 不变，后续层继续连接
            continue

    nodes.append("OUT:Q")
    display_map["OUT:Q"] = "OUT:Q"
    for src in last_nodes:
        edges.append((src, "OUT:Q"))

    nodes = list(dict.fromkeys(nodes))
    return nodes, edges, display_map


def build_node_importance(
    nodes: List[str],
    head_df: pd.DataFrame | None,
    neuron_df: pd.DataFrame | None,
) -> Dict[str, float]:
    imp: Dict[str, float] = {n: 0.0 for n in nodes}

    if head_df is not None and len(head_df) > 0:
        for _, r in head_df.iterrows():
            key = f"L{int(r['layer'])}.H{int(r['head'])}"
            imp[key] = max(imp.get(key, 0.0), _safe_positive(float(r.get("delta_mse", 0.0))))

    if neuron_df is not None and len(neuron_df) > 0:
        for _, r in neuron_df.iterrows():
            key = f"L{int(r['layer'])}.N{int(r['index'])}"
            imp[key] = max(imp.get(key, 0.0), _safe_positive(float(r.get("delta_mse", 0.0))))

    max_imp = max(imp.values()) if imp else 0.0
    if max_imp > 0:
        for k in imp:
            imp[k] = imp[k] / max_imp

    # 输入节点没有直接消融分数：用第一层 head 的平均重要性近似
    first_layer_heads = [v for k, v in imp.items() if k.startswith("L0.H")]
    approx_input = float(np.mean(first_layer_heads)) if first_layer_heads else 0.2
    for n in nodes:
        if n.startswith("IN:"):
            imp[n] = max(imp.get(n, 0.0), approx_input)
        if n.startswith("OUT:"):
            imp[n] = max(imp.get(n, 0.0), 1.0)

    return imp


def build_edge_importance(
    edges: List[Tuple[str, str]],
    node_importance: Dict[str, float],
) -> Dict[Tuple[str, str], float]:
    edge_imp: Dict[Tuple[str, str], float] = {}
    for s, t in edges:
        sv = float(node_importance.get(s, 0.0))
        tv = float(node_importance.get(t, 0.0))
        # 连线重要性：源节点与目标节点的重要性几何平均
        edge_imp[(s, t)] = float(np.sqrt(max(0.0, sv * tv)))
    return edge_imp


def build_edge_importance_from_real(
    edges: List[Tuple[str, str]],
    edge_df: pd.DataFrame | None,
) -> Dict[Tuple[str, str], float]:
    if edge_df is None or len(edge_df) == 0:
        return {}
    if not {"src_node", "dst_node", "interaction"}.issubset(set(edge_df.columns)):
        return {}

    valid_edges = set(edges)
    m = float(edge_df["interaction"].abs().max()) if len(edge_df) > 0 else 0.0
    m = m if m > 0 else 1.0

    edge_imp: Dict[Tuple[str, str], float] = {}
    for _, r in edge_df.iterrows():
        s = str(r["src_node"])
        t = str(r["dst_node"])
        if (s, t) not in valid_edges:
            continue
        score = float(abs(float(r["interaction"])) / m)
        edge_imp[(s, t)] = score
    return edge_imp


def merge_edge_importance(
    edges: List[Tuple[str, str]],
    real_imp: Dict[Tuple[str, str], float],
    heuristic_imp: Dict[Tuple[str, str], float],
    real_weight: float = 0.8,
) -> Dict[Tuple[str, str], float]:
    """融合真实边消融分数与启发式分数，避免仅少数边有颜色。"""
    w = float(max(0.0, min(1.0, real_weight)))
    out: Dict[Tuple[str, str], float] = {}
    for e in edges:
        rv = float(real_imp.get(e, 0.0))
        hv = float(heuristic_imp.get(e, 0.0))
        if e in real_imp:
            out[e] = w * rv + (1.0 - w) * hv
        else:
            out[e] = hv
    return out


def _normalize_scores(scores: List[float], mode: str) -> List[float]:
    if len(scores) == 0:
        return []
    arr = np.asarray(scores, dtype=float)

    if mode == "rank":
        order = np.argsort(arr)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(arr), dtype=float)
        if len(arr) == 1:
            return [1.0]
        return (ranks / (len(arr) - 1)).tolist()

    if mode == "quantile":
        q10, q90 = np.quantile(arr, 0.10), np.quantile(arr, 0.90)
        if abs(q90 - q10) < 1e-12:
            return np.ones_like(arr).tolist()
        z = np.clip((arr - q10) / (q90 - q10), 0.0, 1.0)
        return z.tolist()

    # linear
    vmin, vmax = float(arr.min()), float(arr.max())
    if abs(vmax - vmin) < 1e-12:
        return np.ones_like(arr).tolist()
    return ((arr - vmin) / (vmax - vmin)).tolist()


def _enhance_mode_key(label: str) -> str:
    if label in ["按层排序渐变（新）", "排序增强"]:
        return "rank"
    if label == "分位数增强":
        return "quantile"
    return "linear"


def _edge_layer_id(src: str, dst: str) -> int:
    # 连线归属：优先使用目标层，便于表达“进入某层的重要边”
    return _node_visual_layer(dst)


def _node_visual_layer(node_name: str) -> int:
    if node_name.startswith("IN:"):
        return -1
    if node_name.startswith("OUT:"):
        return 10_000
    if node_name.startswith("L") and "." in node_name:
        try:
            block = int(node_name.split(".")[0][1:])
        except Exception:
            block = 0
        if ".H" in node_name:
            return block * 2
        if ".N" in node_name:
            return block * 2 + 1
        return block * 2
    return 0


def _layer_rank_fade(scores: List[float], top_ratio: float = 0.2, sharpness: float = 5.0) -> List[float]:
    if len(scores) == 0:
        return []
    arr = np.asarray(scores, dtype=float)
    n = len(arr)
    if n == 1:
        return [1.0]

    top_ratio = float(max(0.05, min(0.8, top_ratio)))
    order = np.argsort(-arr)
    out = np.zeros(n, dtype=float)

    for rank_pos, idx in enumerate(order):
        r = rank_pos / (n - 1)
        if r <= top_ratio:
            t = r / max(top_ratio, 1e-6)
            v = 1.0 - 0.15 * t
        else:
            t = (r - top_ratio) / max(1.0 - top_ratio, 1e-6)
            v = 0.85 * np.exp(-sharpness * t)
        out[idx] = float(max(0.02, min(1.0, v)))
    return out.tolist()


def _score_to_blue_gray(score: float, alpha_min: float = 0.38, alpha_max: float = 0.92) -> str:
    score = float(max(0.0, min(1.0, score)))
    c_hi = np.array([26.0, 86.0, 219.0], dtype=float)   # 深蓝
    c_mid = np.array([120.0, 180.0, 245.0], dtype=float)  # 浅蓝
    c_lo = np.array([220.0, 224.0, 230.0], dtype=float)   # 淡灰

    if score >= 0.5:
        t = (score - 0.5) / 0.5
        rgb = c_mid + (c_hi - c_mid) * t
    else:
        t = score / 0.5
        rgb = c_lo + (c_mid - c_lo) * t

    r, g, b = [int(max(0, min(255, round(v)))) for v in rgb.tolist()]
    alpha = alpha_min + (alpha_max - alpha_min) * (score ** 1.6)
    return f"rgba({r},{g},{b},{alpha:.3f})"


def _contribution_shares(nodes: List[str], node_contribution: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    # relative_to_output: share of eventual flow that reaches OUT (bounded in [0, 1])
    positive_vals = {n: max(0.0, float(node_contribution.get(n, 0.0))) for n in nodes}
    out_nodes = [n for n in nodes if n.startswith("OUT:")]
    out_ref = max([positive_vals.get(n, 0.0) for n in out_nodes], default=0.0)
    if out_ref <= 0:
        out_ref = 1.0
    relative_to_output = {n: float(v / out_ref) for n, v in positive_vals.items()}

    # input share: sums to 1 over input nodes only
    input_nodes = [n for n in nodes if n.startswith("IN:")]
    input_total = float(sum(positive_vals.get(n, 0.0) for n in input_nodes))
    input_share = {n: 0.0 for n in nodes}
    if input_total > 0:
        for n in input_nodes:
            input_share[n] = float(positive_vals.get(n, 0.0) / input_total)

    # layer share: sums to 1 within the same visual layer
    by_layer: Dict[int, List[str]] = {}
    for n in nodes:
        by_layer.setdefault(_node_visual_layer(n), []).append(n)

    layer_share: Dict[str, float] = {n: 0.0 for n in nodes}
    for _, layer_nodes in by_layer.items():
        layer_total = float(sum(positive_vals.get(n, 0.0) for n in layer_nodes))
        if layer_total <= 0:
            continue
        for n in layer_nodes:
            layer_share[n] = float(positive_vals.get(n, 0.0) / layer_total)

    return relative_to_output, layer_share, input_share


def _input_contribution_override_from_var_interaction(var_interaction: Dict | None) -> Dict[str, float]:
    if not isinstance(var_interaction, dict):
        return {}
    single = var_interaction.get("single_delta")
    if not isinstance(single, dict):
        return {}

    raw: Dict[str, float] = {}
    for k, v in single.items():
        try:
            abbr = _abbr_feature(str(k))
            if abbr == "Qtar":
                continue
            node_id = f"IN:{abbr}"
            raw[node_id] = raw.get(node_id, 0.0) + max(0.0, float(v))
        except Exception:
            continue

    total = float(sum(raw.values()))
    if total <= 0:
        return {}
    return {k: float(v / total) for k, v in raw.items()}


def _apply_input_contribution_override(
    nodes: List[str],
    node_contribution: Dict[str, float],
    input_override_share: Dict[str, float],
) -> Dict[str, float]:
    if not input_override_share:
        return dict(node_contribution)

    out = dict(node_contribution)
    input_nodes = [n for n in nodes if n.startswith("IN:")]
    if not input_nodes:
        return out

    # Keep input layer scale comparable to hidden layer magnitudes.
    non_input_vals = [max(0.0, float(out.get(n, 0.0))) for n in nodes if not n.startswith("IN:")]
    target_sum = float(np.percentile(non_input_vals, 75)) if len(non_input_vals) > 0 else 1.0
    target_sum = max(target_sum, 1e-6)

    for n in input_nodes:
        share = float(input_override_share.get(n, 0.0))
        out[n] = float(share * target_sum)
    return out


def build_circuit_figure(
    nodes: List[str],
    edges: List[Tuple[str, str]],
    display_map: Dict[str, str],
    edge_importance: Dict[Tuple[str, str], float] | None = None,
    node_contribution: Dict[str, float] | None = None,
    input_share_override: Dict[str, float] | None = None,
    color_mode: str = "layer",
    enhance_mode: str = "rank",
    top_ratio: float = 0.2,
) -> go.Figure:
    layer_map = {}
    for n in nodes:
        layer_map[n] = _node_visual_layer(n)

    x_pos = {}
    y_pos = {}
    by_layer: Dict[int, List[str]] = {}
    for n in nodes:
        by_layer.setdefault(layer_map[n], []).append(n)

    sorted_layers = sorted(by_layer.keys())
    for lx, layer in enumerate(sorted_layers):
        layer_nodes = sorted(by_layer[layer])
        for i, n in enumerate(layer_nodes):
            x_pos[n] = lx
            y_pos[n] = -i

    node_x = [x_pos[n] for n in nodes]
    node_y = [y_pos[n] for n in nodes]
    node_contribution = node_contribution or {}
    node_contrib_vals = [float(node_contribution.get(n, 0.0)) for n in nodes]
    _, layer_share, input_share = _contribution_shares(nodes, node_contribution)
    if input_share_override:
        for n in nodes:
            if n.startswith("IN:"):
                input_share[n] = float(max(0.0, input_share_override.get(n, 0.0)))

    node_color = []
    for n in nodes:
        if n.startswith("IN:"):
            node_color.append("#9467bd")
        elif ".H" in n:
            node_color.append("#1f77b4")
        elif ".N" in n:
            node_color.append("#2ca02c")
        elif n.startswith("OUT:"):
            node_color.append("#ff7f0e")
        else:
            node_color.append("#7f7f7f")

    fig = go.Figure()
    edge_importance = edge_importance or {}

    # 按层排序 + 非线性衰减：前 top_ratio 明显，其余快速变灰
    layer_scores: Dict[int, List[float]] = {}
    for (a, b) in edges:
        layer = _edge_layer_id(a, b)
        layer_scores.setdefault(layer, []).append(float(edge_importance.get((a, b), 0.0)))

    layer_fade_map: Dict[int, Dict[Tuple[str, str], float]] = {}
    for l, vals in layer_scores.items():
        this_edges = [e for e in edges if _edge_layer_id(e[0], e[1]) == l]
        fade_vals = _layer_rank_fade(vals, top_ratio=top_ratio)
        layer_fade_map[l] = {e: float(v) for e, v in zip(this_edges, fade_vals)}

    for (a, b) in edges:
        raw_score = float(edge_importance.get((a, b), 0.0))
        layer = _edge_layer_id(a, b)
        score = float(layer_fade_map.get(layer, {}).get((a, b), 0.0))

        score = float(max(0.0, min(1.0, score)))
        width = 1.0 + 4.5 * (score ** 1.2)
        color = _score_to_blue_gray(score)
        fig.add_trace(
            go.Scatter(
                x=[x_pos[a], x_pos[b]],
                y=[y_pos[a], y_pos[b]],
                mode="lines",
                line=dict(width=width, color=color),
                hovertemplate=(
                    f"edge: {display_map.get(a,a)} -> {display_map.get(b,b)}"
                    f"<br>importance_raw={raw_score:.4f}"
                    f"<br>importance_norm={score:.4f}<extra></extra>"
                ),
                showlegend=False,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            marker=dict(size=14, color=node_color, line=dict(width=1, color="black")),
            text=[display_map.get(n, n) for n in nodes],
            customdata=[[v, layer_share.get(n, 0.0) * 100.0, input_share.get(n, 0.0) * 100.0] for n, v in zip(nodes, node_contrib_vals)],
            textposition="top center",
            hovertemplate=(
                "node=%{text}"
                "<br>contribution_score=%{customdata[0]:.4f}"
                "<br>layer_share=%{customdata[1]:.2f}%"
                "<br>input_share(if IN)=%{customdata[2]:.2f}%"
                "<extra></extra>"
            ),
            name="nodes",
        )
    )
    fig.update_layout(
        title="Pruned Circuit Graph",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=650,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def ranking_bar(df: pd.DataFrame, label_col: str, value_col: str, title: str, topn: int) -> go.Figure:
    if df is None or len(df) == 0:
        return go.Figure()
    d = df.sort_values(value_col, ascending=False).head(topn)
    fig = go.Figure(
        go.Bar(
            x=d[value_col],
            y=d[label_col],
            orientation="h",
        )
    )
    fig.update_layout(title=title, yaxis=dict(autorange="reversed"), height=480)
    return fig


def attention_figure(attn_stats: Dict) -> go.Figure:
    if not attn_stats:
        return go.Figure()
    flood = np.asarray(attn_stats.get("flood_attention", []))
    normal = np.asarray(attn_stats.get("normal_attention", []))
    diff = np.asarray(attn_stats.get("attention_diff", []))
    x = np.arange(len(flood))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=flood, mode="lines", name="flood"))
    fig.add_trace(go.Scatter(x=x, y=normal, mode="lines", name="normal"))
    fig.add_trace(go.Scatter(x=x, y=diff, mode="lines", name="flood-normal"))
    fig.update_layout(title="Attention Statistics (last query)", xaxis_title="Lag", yaxis_title="Attention")
    return fig


def _layout_positions(nodes: List[str]) -> Tuple[Dict[str, float], Dict[str, float], List[int]]:
    layer_map = {n: _node_visual_layer(n) for n in nodes}
    by_layer: Dict[int, List[str]] = {}
    for n in nodes:
        by_layer.setdefault(layer_map[n], []).append(n)

    sorted_layers = sorted(by_layer.keys())
    x_pos: Dict[str, float] = {}
    y_pos: Dict[str, float] = {}
    for lx, layer in enumerate(sorted_layers):
        layer_nodes = sorted(by_layer[layer])
        for i, n in enumerate(layer_nodes):
            x_pos[n] = float(lx)
            y_pos[n] = float(-i)

    return x_pos, y_pos, sorted_layers


def build_figure4_circuit(
    nodes: List[str],
    edges: List[Tuple[str, str]],
    display_map: Dict[str, str],
    edge_importance: Dict[Tuple[str, str], float],
    active_node_ids: set[str],
    show_inactive: bool,
    node_contribution: Dict[str, float] | None = None,
    input_share_override: Dict[str, float] | None = None,
    enhance_mode: str = "rank",
    top_ratio: float = 0.2,
    focus_node_ids: set[str] | None = None,
) -> go.Figure:
    x_pos, y_pos, sorted_layers = _layout_positions(nodes)

    active_edges = [(s, t) for (s, t) in edges if (s in active_node_ids and t in active_node_ids)]
    inactive_edges = [e for e in edges if e not in active_edges]
    focus_node_ids = set(focus_node_ids or set())

    fig = go.Figure()

    # inactive edges (greyed out)
    if show_inactive:
        for (a, b) in inactive_edges:
            fig.add_trace(
                go.Scatter(
                    x=[x_pos[a], x_pos[b]],
                    y=[y_pos[a], y_pos[b]],
                    mode="lines",
                    line=dict(width=1.0, color="rgba(160,160,160,0.25)"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    def _group_scores(group_edges: List[Tuple[str, str]]) -> Dict[Tuple[str, str], float]:
        by_layer: Dict[int, List[Tuple[str, str]]] = {}
        for e in group_edges:
            by_layer.setdefault(_edge_layer_id(e[0], e[1]), []).append(e)
        out: Dict[Tuple[str, str], float] = {}
        for _, layer_edges in by_layer.items():
            vals = [float(edge_importance.get(e, 0.0)) for e in layer_edges]
            fades = _layer_rank_fade(vals, top_ratio=top_ratio)
            for e, s in zip(layer_edges, fades):
                out[e] = float(s)
        return out

    def _edge_color(base_rgb: Tuple[int, int, int], s: float, alpha_lo: float = 0.35, alpha_hi: float = 0.95) -> str:
        s = float(max(0.0, min(1.0, s)))
        low = np.array([220.0, 224.0, 230.0], dtype=float)
        high = np.array([float(base_rgb[0]), float(base_rgb[1]), float(base_rgb[2])], dtype=float)
        rgb = low + (high - low) * (s ** 0.9)
        r, g, b = [int(max(0, min(255, round(v)))) for v in rgb.tolist()]
        alpha = alpha_lo + (alpha_hi - alpha_lo) * (s ** 1.3)
        return f"rgba({r},{g},{b},{alpha:.3f})"

    if focus_node_ids:
        palette: List[Tuple[int, int, int]] = [
            (230, 80, 150),   # pink
            (60, 170, 90),    # green
            (88, 120, 235),   # blue
            (242, 146, 56),   # orange
            (152, 86, 198),   # purple
            (210, 110, 70),   # brown
        ]

        edge_style: Dict[Tuple[str, str], Tuple[float, Tuple[int, int, int], str, str]] = {}
        covered_edges: Set[Tuple[str, str]] = set()

        focus_order = sorted(list(focus_node_ids))
        for i, node in enumerate(focus_order):
            base_rgb = palette[i % len(palette)]

            left_edges_node = [(a, b) for (a, b) in active_edges if b == node and a != node]
            right_edges_node = [(a, b) for (a, b) in active_edges if a == node and b != node]

            left_vals = [float(edge_importance.get(e, 0.0)) for e in left_edges_node]
            right_vals = [float(edge_importance.get(e, 0.0)) for e in right_edges_node]
            left_fades = _layer_rank_fade(left_vals, top_ratio=top_ratio) if left_vals else []
            right_fades = _layer_rank_fade(right_vals, top_ratio=top_ratio) if right_vals else []

            for e, s in zip(left_edges_node, left_fades):
                covered_edges.add(e)
                prev = edge_style.get(e)
                if (prev is None) or (s > prev[0]):
                    edge_style[e] = (float(s), base_rgb, node, "left")

            for e, s in zip(right_edges_node, right_fades):
                covered_edges.add(e)
                prev = edge_style.get(e)
                if (prev is None) or (s > prev[0]):
                    edge_style[e] = (float(s), base_rgb, node, "right")

        for (a, b), (s, base_rgb, owner, side) in edge_style.items():
            raw = float(edge_importance.get((a, b), 0.0))
            s = float(max(0.0, min(1.0, s)))
            color = _edge_color(base_rgb, s)
            fig.add_trace(
                go.Scatter(
                    x=[x_pos[a], x_pos[b]],
                    y=[y_pos[a], y_pos[b]],
                    mode="lines",
                    line=dict(width=1.0 + 4.5 * (s ** 1.2), color=color),
                    hovertemplate=(
                        f"edge: {display_map.get(a,a)} -> {display_map.get(b,b)}"
                        f"<br>importance={raw:.4f}"
                        f"<br>focus={display_map.get(owner, owner)}"
                        f"<br>side={side}<extra></extra>"
                    ),
                    showlegend=False,
                )
            )

        other_edges = [(a, b) for (a, b) in active_edges if (a, b) not in covered_edges]
        other_vals = [float(edge_importance.get(e, 0.0)) for e in other_edges]
        other_norm = _normalize_scores(other_vals, mode="rank") if len(other_vals) > 0 else []
        other_norm_map = {e: float(v) for e, v in zip(other_edges, other_norm)}

        for (a, b) in other_edges:
            raw = float(edge_importance.get((a, b), 0.0))
            s = float(max(0.0, min(1.0, other_norm_map.get((a, b), 0.0))))
            fig.add_trace(
                go.Scatter(
                    x=[x_pos[a], x_pos[b]],
                    y=[y_pos[a], y_pos[b]],
                    mode="lines",
                    line=dict(width=1.0 + 3.8 * (s ** 1.1), color=_score_to_blue_gray(s, alpha_min=0.35, alpha_max=0.90)),
                    hovertemplate=(
                        f"edge: {display_map.get(a,a)} -> {display_map.get(b,b)}"
                        f"<br>importance={raw:.4f}"
                        f"<br>group=auto-global-blue<extra></extra>"
                    ),
                    showlegend=False,
                )
            )
    else:
        by_layer_edges: Dict[int, List[Tuple[str, str]]] = {}
        for e in active_edges:
            by_layer_edges.setdefault(_edge_layer_id(e[0], e[1]), []).append(e)

        active_norm_map: Dict[Tuple[str, str], float] = {}
        for _, layer_edges in by_layer_edges.items():
            vals = [float(edge_importance.get(e, 0.0)) for e in layer_edges]
            fades = _layer_rank_fade(vals, top_ratio=top_ratio)
            for e, s in zip(layer_edges, fades):
                active_norm_map[e] = float(s)

        for (a, b) in active_edges:
            raw = float(edge_importance.get((a, b), 0.0))
            s = float(active_norm_map.get((a, b), 0.0))
            s = float(max(0.0, min(1.0, s)))
            color = _score_to_blue_gray(s)
            fig.add_trace(
                go.Scatter(
                    x=[x_pos[a], x_pos[b]],
                    y=[y_pos[a], y_pos[b]],
                    mode="lines",
                    line=dict(width=1.0 + 4.5 * (s ** 1.2), color=color),
                    hovertemplate=(
                        f"edge: {display_map.get(a,a)} -> {display_map.get(b,b)}"
                        f"<br>importance={raw:.4f}<extra></extra>"
                    ),
                    showlegend=False,
                )
            )

    # nodes
    node_x = [x_pos[n] for n in nodes]
    node_y = [y_pos[n] for n in nodes]
    node_text = [display_map.get(n, n) for n in nodes]
    node_contribution = node_contribution or {}
    node_contrib_vals = [float(node_contribution.get(n, 0.0)) for n in nodes]
    _, layer_share, input_share = _contribution_shares(nodes, node_contribution)
    if input_share_override:
        for n in nodes:
            if n.startswith("IN:"):
                input_share[n] = float(max(0.0, input_share_override.get(n, 0.0)))
    node_color = []
    node_size = []
    for n in nodes:
        is_active = n in active_node_ids
        if n.startswith("IN:"):
            base = "rgba(148,103,189,1.0)" if is_active else "rgba(148,103,189,0.25)"
        elif ".H" in n:
            base = "rgba(31,119,180,1.0)" if is_active else "rgba(31,119,180,0.20)"
        elif ".N" in n:
            base = "rgba(44,160,44,1.0)" if is_active else "rgba(44,160,44,0.20)"
        elif n.startswith("OUT:"):
            base = "rgba(255,127,14,1.0)" if is_active else "rgba(255,127,14,0.25)"
        else:
            base = "rgba(120,120,120,0.3)"
        node_color.append(base)
        node_size.append(14 if is_active else 10)

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            marker=dict(size=node_size, color=node_color, line=dict(width=1, color="black")),
            text=node_text,
            customdata=[[v, layer_share.get(n, 0.0) * 100.0, input_share.get(n, 0.0) * 100.0] for n, v in zip(nodes, node_contrib_vals)],
            textposition="top center",
            hovertemplate=(
                "node=%{text}"
                "<br>contribution_score=%{customdata[0]:.4f}"
                "<br>layer_share=%{customdata[1]:.2f}%"
                "<br>input_share(if IN)=%{customdata[2]:.2f}%"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )

    # dashed layer boundaries (vertical)
    if sorted_layers:
        for i in range(len(sorted_layers) - 1):
            xline = i + 0.5
            fig.add_shape(
                type="line",
                x0=xline,
                x1=xline,
                y0=min(node_y) - 1,
                y1=max(node_y) + 1,
                line=dict(color="rgba(80,80,80,0.35)", width=1, dash="dash"),
            )

    fig.update_layout(
        title="Figure4-style Sparse Circuit",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=720,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def faithfulness_figure(df: pd.DataFrame | None) -> go.Figure:
    if df is None or len(df) == 0 or "k" not in df.columns:
        return go.Figure()
    d = df.sort_values("k")
    fig = go.Figure()
    if "nse" in d.columns:
        fig.add_trace(go.Scatter(x=d["k"], y=d["nse"], mode="lines+markers", name="NSE"))
    if "mse" in d.columns:
        fig.add_trace(go.Scatter(x=d["k"], y=d["mse"], mode="lines+markers", name="MSE", yaxis="y2"))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", title="MSE"))
    fig.update_layout(title="Faithfulness @ K", xaxis_title="Kept Nodes (K)", yaxis_title="NSE", height=360)
    return fig


def _scale_signed_array(values: np.ndarray, mode: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    if mode == "raw":
        return arr
    if mode == "signed_log":
        return np.sign(arr) * np.log1p(np.abs(arr))
    if mode == "zscore":
        m = float(np.mean(arr))
        s = float(np.std(arr)) + 1e-12
        return (arr - m) / s
    if mode == "robust":
        med = float(np.median(arr))
        q1, q3 = np.quantile(arr, 0.25), np.quantile(arr, 0.75)
        iqr = float(q3 - q1) + 1e-12
        return (arr - med) / iqr
    if mode == "abs_norm":
        m = float(np.max(np.abs(arr))) + 1e-12
        return arr / m
    return arr


def _scale_label(mode: str) -> str:
    mapping = {
        "raw": "原始值",
        "signed_log": "符号对数",
        "zscore": "Z-Score",
        "robust": "稳健缩放(IQR)",
        "abs_norm": "绝对值归一化",
    }
    return mapping.get(mode, mode)


def variable_mechanism_sankey(
    var_interaction: Dict | None,
    top_pairs: int = 6,
    scale_mode: str = "signed_log",
) -> go.Figure:
    if not var_interaction or not isinstance(var_interaction.get("single_delta"), dict):
        return go.Figure()

    # 过滤掉目标流量变量 Qtar（使用缩写判断以覆盖不同原始命名）
    def _is_qtar_name(n: str) -> bool:
        try:
            return _abbr_feature(str(n)) == "Qtar"
        except Exception:
            return False

    single_raw = {k: float(v) for k, v in var_interaction.get("single_delta", {}).items()}
    # 删除目标流量变量 Qtar
    single = {k: v for k, v in single_raw.items() if not _is_qtar_name(k)}
    pairs = var_interaction.get("pair_ranking", []) or []

    single_keys = list(single.keys())
    single_vals = np.asarray([single[k] for k in single_keys], dtype=float)
    single_scaled_vals = _scale_signed_array(single_vals, scale_mode)
    single_scaled = {k: float(v) for k, v in zip(single_keys, single_scaled_vals)}

    # 构建节点：变量、交互对、输出，所有展示均为缩写
    pair_rows = sorted(pairs, key=lambda d: abs(float(d.get("interaction", 0.0))), reverse=True)[:top_pairs]
    # 过滤掉包含目标流量变量 Qtar 的 pair
    pair_rows = [r for r in pair_rows if not (_is_qtar_name(r.get("var_i")) or _is_qtar_name(r.get("var_j")))]
    pair_inter_vals = np.asarray([float(r.get("interaction", 0.0)) for r in pair_rows], dtype=float)
    pair_inter_scaled = _scale_signed_array(pair_inter_vals, scale_mode)

    # 原始变量名 -> 缩写 映射
    abbr_map = {n: _abbr_feature(n) for n in single_keys}
    var_nodes = [abbr_map[n] for n in single_keys]
    pair_nodes = [f"{_abbr_feature(str(r['var_i']))}×{_abbr_feature(str(r['var_j']))}" for r in pair_rows]
    out_node = "Q"

    labels = var_nodes + pair_nodes + [out_node]
    # 建立从原始名到索引的映射（用于 src/dst 构建）
    idx = {}
    for i, n in enumerate(single_keys):
        idx[n] = i
    base_offset = len(single_keys)
    for j, r in enumerate(pair_rows):
        idx[f"PAIR:{j}"] = base_offset + j
    idx[out_node] = len(labels) - 1

    src, dst, val, color = [], [], [], []

    # 变量 -> 输出（单变量影响） — 使用更显色但更透明的连线颜色
    for vname, dv in single_scaled.items():
        src.append(idx[vname])
        dst.append(idx[out_node])
        val.append(abs(dv) + 1e-6)
        color.append("rgba(100,100,180,0.45)")

    # 变量 -> 交互项 -> 输出
    for pair_idx, (row, pnode, inter_scaled) in enumerate(zip(pair_rows, pair_nodes, pair_inter_scaled)):
        vi = str(row["var_i"])
        vj = str(row["var_j"])
        inter = float(row.get("interaction", 0.0))
        w = abs(float(inter_scaled)) + 1e-6
        # 正负交互颜色，降低透明度以避免背景过重
        c = "rgba(214,39,40,0.45)" if inter >= 0 else "rgba(31,119,180,0.45)"

        # pair 节点使用 PAIR: 索引映射
        src.extend([idx[vi], idx[vj], idx[f"PAIR:{pair_idx}"]])
        dst.extend([idx[f"PAIR:{pair_idx}"], idx[f"PAIR:{pair_idx}"], idx[out_node]])
        val.extend([w * 0.5, w * 0.5, w])
        color.extend([c, c, c])

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    label=labels,
                    pad=16,
                    thickness=16,
                    color=["rgba(140,107,230,0.60)"] * len(var_nodes)
                    + ["rgba(44,192,76,0.60)"] * len(pair_nodes)
                    + ["rgba(255,159,64,0.60)"],
                ),
                link=dict(source=src, target=dst, value=val, color=color),
            )
        ]
    )
    # 使用更清新的字体与较浅的文字颜色
    fig.update_layout(
        title=f"变量→交互→流量 机制链路图（{_scale_label(scale_mode)}）",
        height=520,
        font=dict(family="Helvetica Neue, Arial, sans-serif", color="#444444"),
    )
    return fig


def project_input_to_l0_neurons(
    nodes: List[str],
    edges: List[Tuple[str, str]],
    edge_importance: Dict[Tuple[str, str], float],
) -> Tuple[List[str], List[Tuple[str, str]], Dict[Tuple[str, str], float], pd.DataFrame]:
    input_nodes = sorted([n for n in nodes if n.startswith("IN:")])
    l0_head_nodes = sorted([n for n in nodes if n.startswith("L0.H")])
    l0_neuron_nodes = sorted([n for n in nodes if n.startswith("L0.N")])

    edge_set = set(edges)
    in2h: Dict[Tuple[str, str], float] = {}
    h2n: Dict[Tuple[str, str], float] = {}
    direct: Dict[Tuple[str, str], float] = {}

    for i in input_nodes:
        for h in l0_head_nodes:
            if (i, h) in edge_set:
                in2h[(i, h)] = float(edge_importance.get((i, h), 0.0))
        for n in l0_neuron_nodes:
            if (i, n) in edge_set:
                direct[(i, n)] = float(edge_importance.get((i, n), 0.0))

    for h in l0_head_nodes:
        for n in l0_neuron_nodes:
            if (h, n) in edge_set:
                h2n[(h, n)] = float(edge_importance.get((h, n), 0.0))

    projected_nodes = input_nodes + l0_neuron_nodes
    projected_edges: List[Tuple[str, str]] = []
    projected_imp: Dict[Tuple[str, str], float] = {}
    rows: List[Dict[str, float | str]] = []

    for i in input_nodes:
        for n in l0_neuron_nodes:
            via_attention = 0.0
            for h in l0_head_nodes:
                via_attention += float(in2h.get((i, h), 0.0)) * float(h2n.get((h, n), 0.0))
            direct_score = float(direct.get((i, n), 0.0))
            total = float(via_attention + direct_score)

            if total > 0.0:
                projected_edges.append((i, n))
                projected_imp[(i, n)] = total

            rows.append(
                {
                    "input": i,
                    "l0_neuron": n,
                    "weight": total,
                    "via_attention": via_attention,
                    "direct": direct_score,
                }
            )

    if len(projected_imp) > 0:
        vmax = max(projected_imp.values())
        if vmax > 0:
            for e in list(projected_imp.keys()):
                projected_imp[e] = float(projected_imp[e] / vmax)

    return projected_nodes, projected_edges, projected_imp, pd.DataFrame(rows)


def compute_structural_synergy_l0(
    projection_df: pd.DataFrame,
    node_contribution: Dict[str, float],
) -> Dict[str, object]:
    """Compute variable structural synergy via shared L0 neurons.

    S(i,j) = sum_n c_n * w(i,n) * w(j,n)
    where c_n is propagated contribution score of L0 neuron n,
    and w(i,n) is projected IN->L0 edge weight.
    """
    if projection_df is None or len(projection_df) == 0:
        return {
            "input_nodes": [],
            "input_labels": [],
            "synergy_matrix": [],
            "single_structural": [],
            "pair_ranking": [],
        }

    need_cols = {"input", "l0_neuron", "weight"}
    if not need_cols.issubset(set(projection_df.columns)):
        return {
            "input_nodes": [],
            "input_labels": [],
            "synergy_matrix": [],
            "single_structural": [],
            "pair_ranking": [],
        }

    df = projection_df.copy()
    df["input"] = df["input"].astype(str)
    df["l0_neuron"] = df["l0_neuron"].astype(str)
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)
    df = df[df["weight"] > 0.0]
    if len(df) == 0:
        return {
            "input_nodes": [],
            "input_labels": [],
            "synergy_matrix": [],
            "single_structural": [],
            "pair_ranking": [],
        }

    # Filter target variable from structural view.
    input_nodes = sorted([n for n in df["input"].unique().tolist() if n != "IN:Qtar"])
    if len(input_nodes) == 0:
        return {
            "input_nodes": [],
            "input_labels": [],
            "synergy_matrix": [],
            "single_structural": [],
            "pair_ranking": [],
        }

    idx = {n: i for i, n in enumerate(input_nodes)}
    mat = np.zeros((len(input_nodes), len(input_nodes)), dtype=float)

    l0_contrib = {
        str(k): max(0.0, float(v))
        for k, v in (node_contribution or {}).items()
        if str(k).startswith("L0.N")
    }

    # Pre-group by L0 neuron for efficient pairwise accumulation.
    by_neuron: Dict[str, List[Tuple[str, float]]] = {}
    for _, r in df.iterrows():
        n = str(r["l0_neuron"])
        i = str(r["input"])
        if i not in idx:
            continue
        by_neuron.setdefault(n, []).append((i, float(r["weight"])))

    for n, pairs in by_neuron.items():
        c = float(l0_contrib.get(n, 0.0))
        if c <= 0.0:
            continue
        for i_name, wi in pairs:
            ii = idx[i_name]
            for j_name, wj in pairs:
                jj = idx[j_name]
                mat[ii, jj] += c * wi * wj

    labels = [n.replace("IN:", "") for n in input_nodes]

    single_rows = []
    for i, name in enumerate(input_nodes):
        single_rows.append(
            {
                "input": name,
                "abbr": labels[i],
                "structural_single": float(mat[i, i]),
            }
        )
    single_rows = sorted(single_rows, key=lambda d: float(d["structural_single"]), reverse=True)

    pair_rows = []
    for i in range(len(input_nodes)):
        for j in range(i + 1, len(input_nodes)):
            pair_rows.append(
                {
                    "var_i": input_nodes[i],
                    "var_j": input_nodes[j],
                    "var_i_abbr": labels[i],
                    "var_j_abbr": labels[j],
                    "synergy": float(mat[i, j]),
                }
            )
    pair_rows = sorted(pair_rows, key=lambda d: float(d["synergy"]), reverse=True)

    return {
        "input_nodes": input_nodes,
        "input_labels": labels,
        "synergy_matrix": mat.tolist(),
        "single_structural": single_rows,
        "pair_ranking": pair_rows,
    }


def collapse_attention_heads(
    nodes: List[str],
    edges: List[Tuple[str, str]],
    edge_importance: Dict[Tuple[str, str], float],
) -> Tuple[List[str], List[Tuple[str, str]], Dict[Tuple[str, str], float]]:
    head_nodes = {n for n in nodes if ".H" in n}
    kept_nodes = [n for n in nodes if n not in head_nodes]

    in_map: Dict[str, List[str]] = {}
    out_map: Dict[str, List[str]] = {}
    for s, t in edges:
        out_map.setdefault(s, []).append(t)
        in_map.setdefault(t, []).append(s)

    merged: Dict[Tuple[str, str], float] = {}

    # 保留原本不经过 head 的边
    for s, t in edges:
        if s in head_nodes or t in head_nodes:
            continue
        merged[(s, t)] = merged.get((s, t), 0.0) + float(edge_importance.get((s, t), 0.0))

    # 将 p->head->q 折叠为 p->q，避免后续层被省略
    for h in head_nodes:
        preds = [p for p in in_map.get(h, []) if p != h and p not in head_nodes]
        succs = [q for q in out_map.get(h, []) if q != h and q not in head_nodes]
        for p in preds:
            w_ph = float(edge_importance.get((p, h), 0.0))
            if w_ph <= 0:
                continue
            for q in succs:
                w_hq = float(edge_importance.get((h, q), 0.0))
                if w_hq <= 0:
                    continue
                merged[(p, q)] = merged.get((p, q), 0.0) + (w_ph * w_hq)

    merged_edges = list(merged.keys())
    if len(merged) > 0:
        vmax = max(merged.values())
        if vmax > 0:
            for e in merged_edges:
                merged[e] = float(merged[e] / vmax)

    return kept_nodes, merged_edges, merged


def _derive_node_importance_from_edges(
    nodes: List[str],
    edges: List[Tuple[str, str]],
    edge_importance: Dict[Tuple[str, str], float],
    base_node_importance: Dict[str, float] | None = None,
) -> Dict[str, float]:
    base_node_importance = base_node_importance or {}
    imp = {n: float(base_node_importance.get(n, 0.0)) for n in nodes}
    for s, t in edges:
        w = float(edge_importance.get((s, t), 0.0))
        imp[s] = max(imp.get(s, 0.0), w)
        imp[t] = max(imp.get(t, 0.0), w)
    vmax = max(imp.values()) if imp else 0.0
    if vmax > 0:
        for k in list(imp.keys()):
            imp[k] = float(imp[k] / vmax)
    return imp


def _compute_node_contribution_to_output(
    nodes: List[str],
    edges: List[Tuple[str, str]],
    edge_importance: Dict[Tuple[str, str], float],
) -> Dict[str, float]:
    """Estimate node contribution score to OUT by weighted backward propagation.

    OUT nodes are seeded as 1.0, then each upstream node accumulates weighted
    contributions from its outgoing edges. Scores are unconstrained and may
    exceed 1.0.
    """
    if not nodes:
        return {}

    succs: Dict[str, List[Tuple[str, float]]] = {n: [] for n in nodes}
    for s, t in edges:
        w = max(0.0, float(edge_importance.get((s, t), 0.0)))
        if s in succs:
            succs[s].append((t, w))

    contrib: Dict[str, float] = {n: 0.0 for n in nodes}
    for n in nodes:
        if n.startswith("OUT:"):
            contrib[n] = 1.0

    ordered = sorted(nodes, key=_node_visual_layer, reverse=True)
    for n in ordered:
        if n.startswith("OUT:"):
            continue
        outs = succs.get(n, [])
        if not outs:
            continue
        score = 0.0
        for t, w in outs:
            score += w * float(contrib.get(t, 0.0))
        contrib[n] = float(score)

    return contrib


def _expand_with_neighbors(selected: Set[str], edges: List[Tuple[str, str]]) -> Set[str]:
    if not selected:
        return set()
    preds: Dict[str, Set[str]] = {}
    succs: Dict[str, Set[str]] = {}
    for s, t in edges:
        preds.setdefault(t, set()).add(s)
        succs.setdefault(s, set()).add(t)

    out = set(selected)

    # 仅向上游追溯到 IN：不在中途反向扩展
    up_visited = set(selected)
    up_queue = list(selected)
    while up_queue:
        cur = up_queue.pop(0)
        cur_layer = _node_visual_layer(cur)
        for p in preds.get(cur, set()):
            if _node_visual_layer(p) == cur_layer:
                continue
            if p in up_visited:
                continue
            up_visited.add(p)
            out.add(p)
            if not p.startswith("IN:"):
                up_queue.append(p)

    # 仅向下游延伸到 OUT：不在中途反向扩展
    down_visited = set(selected)
    down_queue = list(selected)
    while down_queue:
        cur = down_queue.pop(0)
        cur_layer = _node_visual_layer(cur)
        for s in succs.get(cur, set()):
            if _node_visual_layer(s) == cur_layer:
                continue
            if s in down_visited:
                continue
            down_visited.add(s)
            out.add(s)
            if not s.startswith("OUT:"):
                down_queue.append(s)
    return out


def build_fig4_picker(
    nodes: List[str],
    edges: List[Tuple[str, str]],
    display_map: Dict[str, str],
) -> go.Figure:
    x_pos, y_pos, _ = _layout_positions(nodes)
    fig = go.Figure()

    for s, t in edges:
        fig.add_trace(
            go.Scatter(
                x=[x_pos[s], x_pos[t]],
                y=[y_pos[s], y_pos[t]],
                mode="lines",
                line=dict(width=1.0, color="rgba(160,160,160,0.25)"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    node_x = [x_pos[n] for n in nodes]
    node_y = [y_pos[n] for n in nodes]
    node_text = [display_map.get(n, n) for n in nodes]
    node_color = []
    for n in nodes:
        if n.startswith("IN:"):
            node_color.append("rgba(148,103,189,0.90)")
        elif ".N" in n:
            node_color.append("rgba(44,160,44,0.90)")
        elif ".H" in n:
            node_color.append("rgba(31,119,180,0.90)")
        elif n.startswith("OUT:"):
            node_color.append("rgba(255,127,14,0.90)")
        else:
            node_color.append("rgba(120,120,120,0.90)")

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            marker=dict(size=13, color=node_color, line=dict(width=1, color="black")),
            text=node_text,
            textposition="top center",
            customdata=nodes,
            hovertemplate="node=%{customdata}<extra></extra>",
            showlegend=False,
        )
    )

    fig.update_layout(
        title="点击选择节点（自动包含上下游一跳）",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=480,
        margin=dict(l=20, r=20, t=50, b=20),
        clickmode="event+select",
        dragmode="select",
    )
    return fig


def _extract_points_from_plot_state(state: Any) -> List[dict]:
    if state is None:
        return []
    if isinstance(state, dict):
        if isinstance(state.get("selection"), dict):
            return list(state.get("selection", {}).get("points", []) or [])
        return list(state.get("points", []) or [])

    sel = getattr(state, "selection", None)
    if sel is None:
        return []
    if isinstance(sel, dict):
        return list(sel.get("points", []) or [])

    points = getattr(sel, "points", None)
    if points is None:
        return []
    try:
        return list(points)
    except Exception:
        return []


def _extract_node_ids_from_points(points: List[dict], valid_nodes: Set[str]) -> Set[str]:
    out: Set[str] = set()
    for p in points:
        if not isinstance(p, dict):
            continue
        cd = p.get("customdata")
        node_id = None
        if isinstance(cd, str):
            node_id = cd
        elif isinstance(cd, (list, tuple)) and len(cd) > 0 and isinstance(cd[0], str):
            node_id = cd[0]
        if isinstance(node_id, str) and node_id in valid_nodes:
            out.add(node_id)
    return out


@st.cache_data(show_spinner=False)
def _discover_station_dirs(output_root: str) -> List[str]:
    if not os.path.isdir(output_root):
        return []
    station_dirs: List[str] = []
    for name in sorted(os.listdir(output_root)):
        p = os.path.join(output_root, name)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "summary.json")):
            station_dirs.append(name)
    return station_dirs


def _station_signatures(output_root: str, station_dirs: List[str]) -> Tuple[Tuple[str, float, float], ...]:
    sigs: List[Tuple[str, float, float]] = []
    for sid in station_dirs:
        s_m = _file_mtime(os.path.join(output_root, sid, "summary.json"))
        v_m = _file_mtime(os.path.join(output_root, sid, "variable_interactions.json"))
        sigs.append((sid, float(s_m or -1.0), float(v_m or -1.0)))
    return tuple(sigs)


def _persistent_cache_dir() -> str:
    d = os.path.join(os.getcwd(), ".streamlit_cache")
    os.makedirs(d, exist_ok=True)
    return d


def _overview_cache_file(output_root: str) -> str:
    safe = output_root.replace("/", "_").replace("\\", "_").replace(":", "_")
    return os.path.join(_persistent_cache_dir(), f"station_overview_{safe}.json")


def _load_overview_from_disk_cache(output_root: str) -> Tuple[pd.DataFrame | None, float | None]:
    path = _overview_cache_file(output_root)
    if not os.path.exists(path):
        return None, None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        rows = payload.get("rows", [])
        created_at = payload.get("created_at")
        if not isinstance(rows, list):
            return None, None
        return pd.DataFrame(rows), float(created_at) if created_at is not None else None
    except Exception:
        return None, None


def _save_overview_to_disk_cache(output_root: str, df: pd.DataFrame) -> None:
    path = _overview_cache_file(output_root)
    payload = {
        "output_root": output_root,
        "created_at": time.time(),
        "rows": df.to_dict(orient="records"),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def _build_and_cache_overview(output_root: str) -> pd.DataFrame:
    station_dirs = _discover_station_dirs(output_root)
    signatures = _station_signatures(output_root, station_dirs)
    overview = _build_station_overview(output_root, signatures)
    if len(overview) > 0:
        _save_overview_to_disk_cache(output_root, overview)
    return overview


def _top_struct_pair_from_in_node(in_node_df: pd.DataFrame | None) -> Tuple[str | None, float | None]:
    if in_node_df is None or len(in_node_df) == 0:
        return None, None
    if not {"feature", "node_id"}.issubset(set(in_node_df.columns)):
        return None, None

    val_col = "abs_interaction" if "abs_interaction" in in_node_df.columns else "interaction" if "interaction" in in_node_df.columns else None
    if val_col is None:
        return None, None

    df = in_node_df.copy()
    df["feature"] = df["feature"].astype(str)
    df["node_id"] = df["node_id"].astype(str)
    df["score"] = pd.to_numeric(df[val_col], errors="coerce").fillna(0.0).abs()
    df = df[df["score"] > 0.0]
    if len(df) == 0:
        return None, None

    pair_scores: Dict[Tuple[str, str], float] = {}
    for _, g in df.groupby("node_id"):
        gg = g.sort_values("score", ascending=False)
        feats = []
        for _, r in gg.iterrows():
            ab = _abbr_feature(str(r["feature"]))
            if ab == "Qtar":
                continue
            feats.append((ab, float(r["score"])))
        dedup: Dict[str, float] = {}
        for f, s in feats:
            dedup[f] = max(dedup.get(f, 0.0), s)
        items = sorted(dedup.items(), key=lambda kv: kv[1], reverse=True)
        if len(items) < 2:
            continue
        (f1, s1), (f2, s2) = items[0], items[1]
        key = tuple(sorted([f1, f2]))
        pair_scores[key] = pair_scores.get(key, 0.0) + float(min(s1, s2))

    if not pair_scores:
        return None, None
    pair, sc = max(pair_scores.items(), key=lambda kv: kv[1])
    return f"{pair[0]}×{pair[1]}", float(sc)


@st.cache_data(show_spinner=False)
def _build_station_overview(output_root: str, signatures: Tuple[Tuple[str, float, float], ...]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for sid, _, _ in signatures:
        station_dir = os.path.join(output_root, sid)
        summary = _safe_read_json(os.path.join(station_dir, "summary.json")) or {}
        metrics = summary.get("metrics", {}) if isinstance(summary, dict) else {}
        feature_names = summary.get("feature_names", []) if isinstance(summary, dict) else []

        top_factor = None
        top_factor_delta = None
        var_interaction = _safe_read_json(os.path.join(station_dir, "variable_interactions.json"))
        if isinstance(var_interaction, dict) and isinstance(var_interaction.get("single_delta"), dict):
            items = [
                (str(k), float(v))
                for k, v in var_interaction.get("single_delta", {}).items()
                if _abbr_feature(str(k)) != "Qtar"
            ]
            if items:
                top_name, top_val = max(items, key=lambda kv: abs(float(kv[1])))
                top_factor = _abbr_feature(top_name)
                top_factor_delta = float(top_val)

        top_factor_struct = None
        top_factor_struct_score = None
        top_pair_struct = None
        top_pair_struct_score = None
        in_node_df, _ = _load_in_node_records(station_dir)
        if in_node_df is not None and len(in_node_df) > 0 and "feature" in in_node_df.columns:
            val_col = "abs_interaction" if "abs_interaction" in in_node_df.columns else "interaction" if "interaction" in in_node_df.columns else None
            if val_col is not None:
                tmp = in_node_df.copy()
                tmp["feature"] = tmp["feature"].astype(str)
                tmp["abbr"] = tmp["feature"].map(_abbr_feature)
                tmp["score"] = pd.to_numeric(tmp[val_col], errors="coerce").fillna(0.0).abs()
                tmp = tmp[tmp["abbr"] != "Qtar"]
                if len(tmp) > 0:
                    grp = tmp.groupby("abbr", as_index=False)["score"].sum().sort_values("score", ascending=False)
                    if len(grp) > 0:
                        top_factor_struct = str(grp.iloc[0]["abbr"])
                        top_factor_struct_score = float(grp.iloc[0]["score"])

            top_pair_struct, top_pair_struct_score = _top_struct_pair_from_in_node(in_node_df)

        probe = _safe_read_json(os.path.join(station_dir, "physical_probe_results.json")) or {}
        probe_r2_cum_rain = float(probe.get("cum_rain", {}).get("r2")) if isinstance(probe.get("cum_rain"), dict) and probe.get("cum_rain", {}).get("r2") is not None else np.nan
        probe_r2_api = float(probe.get("api", {}).get("r2")) if isinstance(probe.get("api"), dict) and probe.get("api", {}).get("r2") is not None else np.nan
        probe_r2_dpdt = float(probe.get("dpdt", {}).get("r2")) if isinstance(probe.get("dpdt"), dict) and probe.get("dpdt", {}).get("r2") is not None else np.nan
        probe_vals = [v for v in [probe_r2_cum_rain, probe_r2_api, probe_r2_dpdt] if pd.notna(v)]
        probe_r2_mean = float(np.mean(probe_vals)) if len(probe_vals) > 0 else np.nan

        total_active_nodes = np.nan
        active_input_count = np.nan
        active_head_count = np.nan
        active_neuron_count = np.nan
        circuit = _safe_read_json(os.path.join(station_dir, "circuit_structure.json"))
        if isinstance(circuit, dict):
            try:
                ai = int(np.sum(np.asarray(circuit.get("active_inputs", []), dtype=float) > 0))
                ah = int(sum(int(np.sum(np.asarray(x, dtype=float) > 0)) for x in circuit.get("active_heads", [])))
                an = int(sum(int(np.sum(np.asarray(x, dtype=float) > 0)) for x in circuit.get("active_neurons", [])))
                active_input_count = ai
                active_head_count = ah
                active_neuron_count = an
                total_active_nodes = int(ai + ah + an)
            except Exception:
                pass

        rows.append(
            {
                "station_id": sid,
                "nse": float(metrics.get("nse")) if metrics.get("nse") is not None else np.nan,
                "nse_pass": bool(metrics.get("nse_pass")) if metrics.get("nse_pass") is not None else None,
                "rmse": float(metrics.get("rmse")) if metrics.get("rmse") is not None else np.nan,
                "mae": float(metrics.get("mae")) if metrics.get("mae") is not None else np.nan,
                "feature_count": int(len(feature_names)) if isinstance(feature_names, list) else 0,
                "top_factor": top_factor,
                "top_factor_delta": top_factor_delta,
                "top_factor_struct": top_factor_struct,
                "top_factor_struct_score": top_factor_struct_score,
                "top_pair_struct": top_pair_struct,
                "top_pair_struct_score": top_pair_struct_score,
                "probe_r2_cum_rain": probe_r2_cum_rain,
                "probe_r2_api": probe_r2_api,
                "probe_r2_dpdt": probe_r2_dpdt,
                "probe_r2_mean": probe_r2_mean,
                "active_input_count": active_input_count,
                "active_head_count": active_head_count,
                "active_neuron_count": active_neuron_count,
                "total_active_nodes": total_active_nodes,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "station_id", "nse", "nse_pass", "rmse", "mae", "feature_count",
                "top_factor", "top_factor_delta",
                "top_factor_struct", "top_factor_struct_score",
                "top_pair_struct", "top_pair_struct_score",
                "probe_r2_cum_rain", "probe_r2_api", "probe_r2_dpdt", "probe_r2_mean",
                "active_input_count", "active_head_count", "active_neuron_count", "total_active_nodes",
            ]
        )
    return pd.DataFrame(rows).sort_values("station_id")


def _render_station_overview(output_root: str):
    st.subheader("站点总览与筛选")

    required_cols_with_default: Dict[str, Any] = {
        "station_id": "",
        "nse": np.nan,
        "nse_pass": None,
        "rmse": np.nan,
        "mae": np.nan,
        "feature_count": 0,
        "top_factor": None,
        "top_factor_delta": np.nan,
        "top_factor_struct": None,
        "top_factor_struct_score": np.nan,
        "top_pair_struct": None,
        "top_pair_struct_score": np.nan,
        "probe_r2_cum_rain": np.nan,
        "probe_r2_api": np.nan,
        "probe_r2_dpdt": np.nan,
        "probe_r2_mean": np.nan,
        "active_input_count": np.nan,
        "active_head_count": np.nan,
        "active_neuron_count": np.nan,
        "total_active_nodes": np.nan,
    }

    c_cache1, c_cache2, c_cache3 = st.columns([1.5, 1.2, 3.3])
    with c_cache1:
        use_disk_cache = st.checkbox("优先使用磁盘缓存", value=True)
    with c_cache2:
        rebuild_clicked = st.button("重建索引缓存")

    if rebuild_clicked:
        st.cache_data.clear()
        overview = _build_and_cache_overview(output_root)
        cache_time = time.time()
        cache_source = "rebuilt"
    else:
        cached_df, cache_time = _load_overview_from_disk_cache(output_root) if use_disk_cache else (None, None)
        if cached_df is not None and len(cached_df) > 0:
            overview = cached_df
            cache_source = "disk"
        else:
            overview = _build_and_cache_overview(output_root)
            cache_time = time.time() if len(overview) > 0 else None
            cache_source = "fresh"

    if len(overview) == 0:
        st.info("站点索引为空。")
        return

    # Backward compatibility: old disk caches may not contain newly added columns.
    for c, default_v in required_cols_with_default.items():
        if c not in overview.columns:
            overview[c] = default_v

    with c_cache3:
        if cache_time is not None:
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(cache_time))
            src_text = "磁盘缓存" if cache_source == "disk" else "重建缓存" if cache_source == "rebuilt" else "新扫描"
            st.caption(f"当前索引来源: {src_text} | 更新时间: {ts}")
        else:
            st.caption("当前索引来源: 新扫描")

    # Row 1: core filters
    f1, f2, f3, f4, f5 = st.columns([2.0, 2.2, 1.2, 1.6, 1.6])
    with f1:
        query = st.text_input("搜索站点ID", value="", key="overview_query")
    with f3:
        pass_filter = st.selectbox("NSE通过", options=["全部", "通过", "未通过"], index=0)
    with f4:
        factor_options = sorted([x for x in overview["top_factor"].dropna().unique().tolist() if str(x).strip()])
        factor_filter = st.multiselect("因果主要因素", options=factor_options)
    with f5:
        struct_factor_options = sorted([x for x in overview["top_factor_struct"].dropna().unique().tolist() if str(x).strip()])
        struct_factor_filter = st.multiselect("结构主要因素", options=struct_factor_options)

    # Row 2: visualization clipping controls
    cclip1, cclip2, cclip3, cclip4, cclip5, cclip6 = st.columns([1.3, 1.0, 1.0, 1.3, 1.0, 1.0])
    with cclip1:
        clip_nse_for_viz = st.checkbox("裁剪NSE可视化", value=True, help="仅影响总览筛选与图表显示，不修改原始NSE")
    with cclip2:
        nse_floor = st.number_input("NSE下限", value=-5.0, step=0.5)
    with cclip3:
        nse_ceiling = st.number_input("NSE上限", value=1.0, step=0.1)
    with cclip4:
        clip_probe_for_viz = st.checkbox("裁剪Probe可视化", value=True, help="仅影响probe相关图表显示，不修改原始R²")
    with cclip5:
        probe_floor = st.number_input("Probe下限", value=-1.0, step=0.5)
    with cclip6:
        probe_ceiling = st.number_input("Probe上限", value=1.0, step=0.1)

    # Build viz columns
    overview["nse_viz"] = pd.to_numeric(overview["nse"], errors="coerce")
    if clip_nse_for_viz:
        lo_clip = float(min(nse_floor, nse_ceiling))
        hi_clip = float(max(nse_floor, nse_ceiling))
        overview["nse_viz"] = overview["nse_viz"].clip(lower=lo_clip, upper=hi_clip)

    for col in ["probe_r2_cum_rain", "probe_r2_api", "probe_r2_dpdt", "probe_r2_mean"]:
        overview[f"{col}_viz"] = pd.to_numeric(overview[col], errors="coerce")
        if clip_probe_for_viz:
            lo_p = float(min(probe_floor, probe_ceiling))
            hi_p = float(max(probe_floor, probe_ceiling))
            s = overview[f"{col}_viz"]
            overview[f"{col}_viz"] = s.where(s.between(lo_p, hi_p), np.nan)

    with f2:
        nse_series = overview["nse_viz"].dropna()
        if len(nse_series) > 0:
            lo = float(np.floor(nse_series.min() * 100.0) / 100.0)
            hi = float(np.ceil(nse_series.max() * 100.0) / 100.0)
            nse_range = st.slider("NSE 区间", min_value=lo, max_value=hi, value=(lo, hi), step=0.01)
        else:
            nse_range = (-1e9, 1e9)

    mask = np.ones(len(overview), dtype=bool)
    if query.strip():
        mask &= overview["station_id"].astype(str).str.contains(query.strip(), case=False, regex=False)
    mask &= overview["nse_viz"].fillna(-1e9).between(float(nse_range[0]), float(nse_range[1]))
    if pass_filter == "通过":
        mask &= overview["nse_pass"].fillna(False)
    elif pass_filter == "未通过":
        mask &= ~overview["nse_pass"].fillna(False)
    if factor_filter:
        mask &= overview["top_factor"].isin(factor_filter)
    if struct_factor_filter:
        mask &= overview["top_factor_struct"].isin(struct_factor_filter)

    filtered = overview.loc[mask].copy()
    filtered = filtered.sort_values(["nse", "station_id"], ascending=[False, True])
    st.caption(f"当前筛选结果: {len(filtered)} / {len(overview)} 个站点")

    display_cols = [
        "station_id", "nse", "nse_pass", "rmse", "mae", "feature_count",
        "top_factor", "top_factor_delta", "top_factor_struct", "top_factor_struct_score",
        "probe_r2_mean", "total_active_nodes",
    ]
    st.dataframe(filtered[display_cols], use_container_width=True, height=460)

    st.markdown("### 总览可视化")
    g1, g2 = st.columns(2)
    with g1:
        causal_counts = filtered["top_factor"].dropna().astype(str)
        if len(causal_counts) > 0:
            vc = causal_counts.value_counts().head(12)
            fig = go.Figure(go.Bar(x=vc.index.tolist(), y=vc.values.tolist(), marker_color="#5b8ff9"))
            fig.update_layout(title="因果主要影响因素（站点计数）", height=320, xaxis_title="变量", yaxis_title="站点数")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("暂无可用的因果主要影响因素数据。")
    with g2:
        struct_counts = filtered["top_factor_struct"].dropna().astype(str)
        if len(struct_counts) > 0:
            vc = struct_counts.value_counts().head(12)
            fig = go.Figure(go.Bar(x=vc.index.tolist(), y=vc.values.tolist(), marker_color="#36cfc9"))
            fig.update_layout(title="结构主要影响因素（站点计数）", height=320, xaxis_title="变量", yaxis_title="站点数")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("暂无可用的结构主要影响因素数据。")

    g3, g4 = st.columns(2)
    with g3:
        pair_df = filtered[["top_pair_struct", "top_pair_struct_score"]].dropna().copy()
        if len(pair_df) > 0:
            agg = (
                pair_df.groupby("top_pair_struct", as_index=False)
                .agg(count=("top_pair_struct", "count"), mean_score=("top_pair_struct_score", "mean"))
            )
            agg["combo_score"] = agg["count"] * agg["mean_score"]
            agg = agg.sort_values("combo_score", ascending=False).head(12)
            fig = go.Figure(go.Bar(x=agg["top_pair_struct"], y=agg["combo_score"], marker_color="#73d13d"))
            fig.update_layout(title="结构变量组合（出现多且贡献大）", height=320, xaxis_title="变量组合", yaxis_title="count × mean(score)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("暂无可用的结构变量组合数据。")
    with g4:
        p = filtered[["probe_r2_cum_rain_viz", "probe_r2_api_viz", "probe_r2_dpdt_viz"]].copy()
        vals = [
            float(p["probe_r2_cum_rain_viz"].mean()) if p["probe_r2_cum_rain_viz"].notna().any() else np.nan,
            float(p["probe_r2_api_viz"].mean()) if p["probe_r2_api_viz"].notna().any() else np.nan,
            float(p["probe_r2_dpdt_viz"].mean()) if p["probe_r2_dpdt_viz"].notna().any() else np.nan,
        ]
        if any(pd.notna(v) for v in vals):
            fig = go.Figure(go.Bar(x=["cum_rain", "api", "dpdt"], y=vals, marker_color=["#9254de", "#597ef7", "#fa8c16"]))
            fig.update_layout(title="探针情况（平均 R²）", height=320, xaxis_title="物理代理变量", yaxis_title="Mean R²")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("暂无 probe R² 数据。")

    g5, g6 = st.columns(2)
    with g5:
        rel_df = filtered[["total_active_nodes", "nse_viz", "station_id"]].dropna().copy()
        if len(rel_df) > 0:
            x = pd.to_numeric(rel_df["total_active_nodes"], errors="coerce")
            y = pd.to_numeric(rel_df["nse_viz"], errors="coerce")
            rho = float(x.corr(y, method="spearman")) if len(rel_df) > 1 else np.nan
            fit_a = np.nan
            fit_b = np.nan
            fit_r2 = np.nan
            fig = go.Figure(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    marker=dict(size=9, color="#1677ff", opacity=0.75),
                    text=rel_df["station_id"],
                    hovertemplate="station=%{text}<br>active_nodes=%{x}<br>NSE=%{y:.4f}<extra></extra>",
                )
            )
            if len(rel_df) >= 2:
                try:
                    coef = np.polyfit(x.to_numpy(dtype=float), y.to_numpy(dtype=float), 1)
                    fit_a = float(coef[0])
                    fit_b = float(coef[1])
                    xx = np.linspace(float(np.min(x)), float(np.max(x)), 100)
                    yy = coef[0] * xx + coef[1]
                    fig.add_trace(
                        go.Scatter(
                            x=xx,
                            y=yy,
                            mode="lines",
                            line=dict(color="#cf1322", width=2),
                            name="fit",
                            hovertemplate="linear_fit<extra></extra>",
                            showlegend=False,
                        )
                    )
                    y_hat = coef[0] * x.to_numpy(dtype=float) + coef[1]
                    ss_res = float(np.sum((y.to_numpy(dtype=float) - y_hat) ** 2))
                    ss_tot = float(np.sum((y.to_numpy(dtype=float) - float(np.mean(y.to_numpy(dtype=float)))) ** 2) + 1e-12)
                    fit_r2 = float(1.0 - ss_res / ss_tot)
                except Exception:
                    pass
            fig.update_layout(title="结构节点数量 vs 模型性能", height=340, xaxis_title="Total Active Nodes", yaxis_title="NSE")
            if pd.notna(rho) or pd.notna(fit_a):
                eq_text = ""
                if pd.notna(fit_a) and pd.notna(fit_b):
                    sign = "+" if fit_b >= 0 else "-"
                    eq_text = f"<br>fit: y={fit_a:.4f}x {sign} {abs(fit_b):.4f}"
                r2_text = f"<br>R²={fit_r2:.3f}" if pd.notna(fit_r2) else ""
                fig.add_annotation(
                    xref="paper",
                    yref="paper",
                    x=0.02,
                    y=0.02,
                    text=(f"Spearman ρ={rho:.3f}" if pd.notna(rho) else "") + eq_text + r2_text,
                    showarrow=False,
                    font=dict(size=12, color="#333333"),
                    bgcolor="rgba(255,255,255,0.65)",
                    align="left",
                )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("暂无结构节点数量与性能关系数据。")
    with g6:
        rel_df = filtered[["probe_r2_mean_viz", "nse_viz", "station_id"]].dropna().copy()
        if len(rel_df) > 0:
            x = pd.to_numeric(rel_df["probe_r2_mean_viz"], errors="coerce")
            y = pd.to_numeric(rel_df["nse_viz"], errors="coerce")
            fig = go.Figure(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    marker=dict(size=9, color="#13c2c2", opacity=0.75),
                    text=rel_df["station_id"],
                    hovertemplate="station=%{text}<br>probe_r2_mean=%{x:.4f}<br>NSE=%{y:.4f}<extra></extra>",
                )
            )
            fig.update_layout(title="探针可解码性 vs 模型性能", height=340, xaxis_title="Probe Mean R²", yaxis_title="NSE")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("暂无探针与性能关系数据。")

    if len(filtered) == 0:
        st.info("当前筛选条件下没有可选站点。")
        return

    station_options = filtered["station_id"].astype(str).tolist()
    default_sid = st.session_state.get("viz_station_id")
    default_idx = station_options.index(default_sid) if default_sid in station_options else 0
    selected_sid = st.selectbox("选择站点进入详情", options=station_options, index=default_idx)

    c_open, c_refresh = st.columns([1, 1])
    with c_open:
        if st.button("进入站点详情", type="primary"):
            st.session_state["viz_station_id"] = selected_sid
            st.session_state["viz_view"] = "detail"
            st.rerun()
    with c_refresh:
        if st.button("刷新内存缓存"):
            st.cache_data.clear()
            st.rerun()


def _render_station_detail(output_root: str, station_id: str):
    output_dir = os.path.join(output_root, station_id)
    if not os.path.isdir(output_dir):
        st.error(f"站点目录不存在: {output_dir}")
        return

    top_heads = st.sidebar.slider("显示 top heads", 5, 100, 30)
    top_neurons = st.sidebar.slider("显示 top neurons", 10, 200, 60)
    max_neurons_graph = st.sidebar.slider("电路图每层最大神经元", 5, 80, 24)
    if "collapse_attention_view" not in st.session_state:
        st.session_state["collapse_attention_view"] = False
    if "fig4_manual_nodes" not in st.session_state:
        st.session_state["fig4_manual_nodes"] = []
    if "fig4_clicked_nodes" not in st.session_state:
        st.session_state["fig4_clicked_nodes"] = []
    if "fig4_focus_nodes" not in st.session_state:
        st.session_state["fig4_focus_nodes"] = []
    if "fig4_selected_nodes" not in st.session_state:
        st.session_state["fig4_selected_nodes"] = []
    if st.sidebar.button("切换：省略 Attention 层"):
        st.session_state["collapse_attention_view"] = not st.session_state["collapse_attention_view"]

    collapse_attention_view = bool(st.session_state["collapse_attention_view"])
    st.sidebar.caption(f"当前电路视图：{'省略Attention头（保留后续层）' if collapse_attention_view else '完整电路'}")
    edge_top_ratio = st.sidebar.slider("每层明显连线占比", 0.05, 0.50, 0.20, 0.05)
    real_edge_weight = st.sidebar.slider("真实边分数权重", 0.0, 1.0, 0.80, 0.05)
    fig4_topk = st.sidebar.slider("Figure4 显示 Top-K 节点", 4, 64, 12)
    fig4_show_inactive = st.sidebar.checkbox("Figure4 显示非活跃部分(灰色)", value=True)

    c_back, c_path = st.columns([1, 3])
    with c_back:
        if st.button("返回站点总览"):
            st.session_state["viz_view"] = "stations"
            st.rerun()
    with c_path:
        st.caption(f"当前站点: {station_id} | 结果目录: {output_dir}")

    summary = _safe_read_json(os.path.join(output_dir, "summary.json"))
    if summary and isinstance(summary.get("metrics"), dict):
        metrics = summary.get("metrics", {})
        m1, m2, m3 = st.columns(3)
        m1.metric("NSE", f"{float(metrics.get('nse')):.4f}" if metrics.get("nse") is not None else "NA")
        m2.metric("RMSE", f"{float(metrics.get('rmse')):.4f}" if metrics.get("rmse") is not None else "NA")
        m3.metric("MAE", f"{float(metrics.get('mae')):.4f}" if metrics.get("mae") is not None else "NA")

    panel = st.radio(
        "站点详情视图",
        options=["电路图", "Head/Neuron 排名", "Attention", "因果与探针", "变量交互", "Figure4风格"],
        horizontal=True,
        key=f"station_panel_{station_id}",
    )

    if panel == "电路图":
        circuit = _safe_read_json(os.path.join(output_dir, "circuit_structure.json"))
        var_interaction = _safe_read_json(os.path.join(output_dir, "variable_interactions.json"))
        in_node_df, in_node_src = _load_in_node_records(output_dir)
        if circuit is None:
            st.warning("未找到 circuit_structure.json，请先运行训练生成结果。")
            return

        head_df = _safe_read_csv(os.path.join(output_dir, "head_importance_ranking.csv"))
        neuron_df = _safe_read_csv(os.path.join(output_dir, "neuron_importance_ranking.csv"))
        edge_df, edge_src_name = _load_edge_importance_table(output_dir)
        if edge_src_name:
            st.caption(f"当前边重要性来源: {edge_src_name}")

        feature_names = summary.get("feature_names", []) if summary else []
        nodes, edges, display_map = build_circuit_edges(
            circuit,
            max_neurons_per_layer=max_neurons_graph,
            input_features=feature_names,
        )

        node_imp = build_node_importance(nodes, head_df, neuron_df)
        heuristic_edge_imp = build_edge_importance(edges, node_imp)
        edge_imp_real = build_edge_importance_from_real(edges, edge_df)
        edge_imp = merge_edge_importance(
            edges,
            real_imp=edge_imp_real,
            heuristic_imp=heuristic_edge_imp,
            real_weight=real_edge_weight,
        )
        in_real_imp = _build_in_edge_real_importance(in_node_df, edges)
        edge_imp = _inject_in_real_edges(edge_imp, in_real_imp, in_real_weight=1.0)
        causal_input_share = _input_contribution_override_from_var_interaction(var_interaction)

        if collapse_attention_view:
            p_df = project_input_to_l0_neurons(nodes, edges, edge_imp)[3]
            c_nodes, c_edges, c_imp = collapse_attention_heads(nodes, edges, edge_imp)
            c_node_imp = _compute_node_contribution_to_output(c_nodes, c_edges, c_imp)
            fig = build_circuit_figure(
                c_nodes,
                c_edges,
                display_map,
                edge_importance=c_imp,
                node_contribution=c_node_imp,
                input_share_override=causal_input_share,
                top_ratio=edge_top_ratio,
            )
            shown_nodes, shown_edges = c_nodes, c_edges
        else:
            p_df = None
            node_contrib = _compute_node_contribution_to_output(nodes, edges, edge_imp)
            fig = build_circuit_figure(
                nodes,
                edges,
                display_map,
                edge_importance=edge_imp,
                node_contribution=node_contrib,
                input_share_override=causal_input_share,
                top_ratio=edge_top_ratio,
            )
            shown_nodes, shown_edges = nodes, edges

        st.plotly_chart(fig, use_container_width=True)
        if st.button("导出电路图PNG(300DPI)", key="export_circuit_png"):
            try:
                p = _save_plot_png(fig, station_id=station_id, image_name="circuit")
                st.success(f"已保存: {p}")
            except Exception as e:
                st.error(f"导出失败: {e}")

        st.caption(
            f"节点数: {len(shown_nodes)} | 边数: {len(shown_edges)} | 输入简写: {', '.join([_abbr_feature(x) for x in feature_names])} | "
            f"每层明显连线占比: {edge_top_ratio:.2f} | 真实边权重: {real_edge_weight:.2f}"
        )
        if in_node_src:
            st.caption(f"IN->节点真实交互来源: {in_node_src} | 已注入 IN->L0 边: {len(in_real_imp)}")

        if in_node_df is not None and len(in_node_df) > 0:
            show_cols = [c for c in ["feature", "node_id", "interaction", "abs_interaction", "mse_i", "mse_j", "mse_ij"] if c in set(in_node_df.columns)]
            if len(show_cols) > 0:
                st.markdown("**IN->节点真实交互（Top）**")
                st.dataframe(
                    in_node_df.sort_values("abs_interaction", ascending=False).head(30) if "abs_interaction" in in_node_df.columns else in_node_df.head(30),
                    use_container_width=True,
                    height=260,
                )

        if collapse_attention_view:
            st.markdown("**输入变量 -> 第一层节点（L0 Neuron）影响权重**")
            if p_df is not None and len(p_df) > 0:
                mat = p_df.pivot(index="input", columns="l0_neuron", values="weight").fillna(0.0)
                hfig = go.Figure(
                    data=go.Heatmap(
                        z=mat.values,
                        x=list(mat.columns),
                        y=list(mat.index),
                        colorscale="Blues",
                        zmin=0.0,
                    )
                )
                hfig.update_layout(height=340, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(hfig, use_container_width=True)
                st.dataframe(
                    p_df.sort_values("weight", ascending=False).head(30),
                    use_container_width=True,
                    height=240,
                )
            else:
                st.info("当前没有可展示的输入->L0节点投影权重。")

        if edge_df is not None and len(edge_df) > 0 and {"src_node", "dst_node", "interaction"}.issubset(set(edge_df.columns)):
            top_show = edge_df.sort_values("interaction", ascending=False).head(20).copy()
            top_show["src"] = top_show["src_node"].map(lambda x: display_map.get(str(x), str(x)))
            top_show["dst"] = top_show["dst_node"].map(lambda x: display_map.get(str(x), str(x)))
            st.markdown("**Top 连线重要性（真实边消融 interaction）**")
            st.dataframe(
                top_show[["src", "dst", "interaction", "delta_both", "heuristic_score"]],
                use_container_width=True,
                height=260,
            )
        else:
            topk = min(20, len(edge_imp))
            if topk > 0:
                edge_items = sorted(edge_imp.items(), key=lambda kv: kv[1], reverse=True)[:topk]
                edge_tbl = pd.DataFrame(
                    [
                        {
                            "src": display_map.get(s, s),
                            "dst": display_map.get(t, t),
                            "edge_importance": v,
                        }
                        for (s, t), v in edge_items
                    ]
                )
                st.markdown("**Top 连线重要性（启发式）**")
                st.dataframe(edge_tbl, use_container_width=True, height=260)

    elif panel == "Head/Neuron 排名":
        head_df = _safe_read_csv(os.path.join(output_dir, "head_importance_ranking.csv"))
        neuron_df = _safe_read_csv(os.path.join(output_dir, "neuron_importance_ranking.csv"))
        c1, c2 = st.columns(2)
        with c1:
            if head_df is not None and len(head_df) > 0:
                d = head_df.copy()
                d["label"] = d.apply(lambda r: f"L{int(r['layer'])}-H{int(r['head'])}", axis=1)
                st.plotly_chart(
                    ranking_bar(d, "label", "delta_mse", "Attention Head Importance", top_heads),
                    use_container_width=True,
                )
                st.dataframe(d.head(top_heads), use_container_width=True, height=300)
            else:
                st.info("未找到 head_importance_ranking.csv")

        with c2:
            if neuron_df is not None and len(neuron_df) > 0:
                d = neuron_df.copy()
                d["label"] = d.apply(lambda r: f"L{int(r['layer'])}-N{int(r['index'])}", axis=1)
                st.plotly_chart(
                    ranking_bar(d, "label", "delta_mse", "MLP Neuron Importance", top_neurons),
                    use_container_width=True,
                )
                st.dataframe(d.head(top_neurons), use_container_width=True, height=300)
            else:
                st.info("未找到 neuron_importance_ranking.csv")

    elif panel == "Attention":
        attn_stats = _safe_read_json(os.path.join(output_dir, "attention_statistics.json"))
        if attn_stats:
            st.plotly_chart(attention_figure(attn_stats), use_container_width=True)
            st.json({"flood_threshold_y": attn_stats.get("flood_threshold_y")})
        else:
            st.info("未找到 attention_statistics.json")

    elif panel == "因果与探针":
        causal = _safe_read_json(os.path.join(output_dir, "causal_intervention_results.json"))
        probe = _safe_read_json(os.path.join(output_dir, "physical_probe_results.json"))
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Causal Intervention")
            if causal:
                st.json(causal)
            else:
                st.info("未找到 causal_intervention_results.json")

        with c2:
            st.subheader("Linear Probe")
            if probe:
                st.json(probe)
            else:
                st.info("未找到 physical_probe_results.json")

    elif panel == "变量交互":
        st.subheader("输入变量相互作用（真实干预，原始值）")
        var_interaction = _safe_read_json(os.path.join(output_dir, "variable_interactions.json"))
        var_interaction_rank = _safe_read_csv(os.path.join(output_dir, "variable_interactions_ranking.csv"))

        if var_interaction and "interaction_matrix" in var_interaction and "feature_names" in var_interaction:
            names_all = list(var_interaction["feature_names"]) or []
            mat = np.asarray(var_interaction["interaction_matrix"], dtype=float)
            keep_idxs = [i for i, n in enumerate(names_all) if _abbr_feature(n) != "Qtar"]
            if len(keep_idxs) == 0:
                st.info("交互矩阵仅包含流量变量或为空，已被过滤。")
            else:
                names = [names_all[i] for i in keep_idxs]
                mat_sub = mat[np.ix_(keep_idxs, keep_idxs)]

                abbr_labels = [_abbr_feature(n) for n in names]
                fig = go.Figure(
                    data=go.Heatmap(
                        z=mat_sub,
                        x=abbr_labels,
                        y=abbr_labels,
                        colorscale="RdYlBu",
                        zmid=0.0,
                        colorbar=dict(title="Interaction (Raw)"),
                    )
                )
                fig.update_layout(
                    title="Pairwise Interaction Matrix (Raw)",
                    height=520,
                    font=dict(family="Helvetica Neue, Arial, sans-serif", color="#444444"),
                )
                st.plotly_chart(fig, use_container_width=True)

            if isinstance(var_interaction.get("single_delta"), dict):
                single_raw = var_interaction.get("single_delta", {}) or {}
                single_items = [(k, float(v)) for k, v in single_raw.items() if _abbr_feature(k) != "Qtar"]
                if single_items:
                    single_df = pd.DataFrame(
                        [{"variable": k, "delta_single": v, "abbr": _abbr_feature(k)} for k, v in single_items]
                    ).sort_values("delta_single", ascending=False)
                    st.markdown("**单变量干预敏感性（原始 Δ_i）**")
                    st.dataframe(single_df, use_container_width=True, height=220)
                else:
                    st.info("单变量干预结果仅包含流量变量或为空，已被过滤。")
        else:
            st.info("未找到 variable_interactions.json，请重新训练后生成。")

        if var_interaction_rank is not None and len(var_interaction_rank) > 0:
            show = var_interaction_rank.copy()
            mask_keep = ~(
                show["var_i"].astype(str).map(lambda x: _abbr_feature(x) == "Qtar")
                | show["var_j"].astype(str).map(lambda x: _abbr_feature(x) == "Qtar")
            )
            show = show[mask_keep].copy()
            if len(show) == 0:
                st.info("Top 变量对中不包含非流量变量，已被过滤。")
            else:
                show["abs_interaction"] = show["interaction"].abs()
                show = show.sort_values("abs_interaction", ascending=False).head(20)
                show["var_i_abbr"] = show["var_i"].astype(str).map(lambda x: _abbr_feature(x))
                show["var_j_abbr"] = show["var_j"].astype(str).map(lambda x: _abbr_feature(x))
                st.markdown("**Top 变量对交互（按 |interaction|，原始值）**")
                st.dataframe(
                    show[["var_i_abbr", "var_j_abbr", "interaction", "delta_pair", "delta_i", "delta_j"]],
                    use_container_width=True,
                    height=280,
                )

        st.markdown("**变量-交互-流量机制链路（原始值）**")
        sk = variable_mechanism_sankey(var_interaction, top_pairs=8, scale_mode="raw")
        if len(sk.data) > 0:
            st.plotly_chart(sk, use_container_width=True)
            st.caption("红色链路：正协同交互（interaction>0）；蓝色链路：负交互/替代关系（interaction<0）。")
        else:
            st.info("缺少 variable_interactions.json，无法绘制机制链路图。")

        st.markdown("---")
        st.subheader("结构强度协作（省略 Attention，基于 L0 节点）")
        st.caption("定义: S(i,j)=Σ_n c_n*w(i,n)*w(j,n)，其中 c_n 为 L0 节点传播贡献分，w(i,n) 为 IN->L0 投影权重。")

        circuit = _safe_read_json(os.path.join(output_dir, "circuit_structure.json"))
        in_node_df, _ = _load_in_node_records(output_dir)
        head_df = _safe_read_csv(os.path.join(output_dir, "head_importance_ranking.csv"))
        neuron_df = _safe_read_csv(os.path.join(output_dir, "neuron_importance_ranking.csv"))
        edge_df, _ = _load_edge_importance_table(output_dir)

        if circuit is None:
            st.info("缺少 circuit_structure.json，无法计算结构强度协作。")
        else:
            feature_names = summary.get("feature_names", []) if summary else []
            s_nodes, s_edges, _ = build_circuit_edges(
                circuit,
                max_neurons_per_layer=max_neurons_graph,
                input_features=feature_names,
            )
            s_node_imp = build_node_importance(s_nodes, head_df, neuron_df)
            s_heuristic_edge = build_edge_importance(s_edges, s_node_imp)
            s_real_edge = build_edge_importance_from_real(s_edges, edge_df)
            s_edge_imp = merge_edge_importance(
                s_edges,
                real_imp=s_real_edge,
                heuristic_imp=s_heuristic_edge,
                real_weight=real_edge_weight,
            )
            s_in_real = _build_in_edge_real_importance(in_node_df, s_edges)
            s_edge_imp = _inject_in_real_edges(s_edge_imp, s_in_real, in_real_weight=1.0)

            c_nodes, c_edges, c_imp = collapse_attention_heads(s_nodes, s_edges, s_edge_imp)
            c_node_contrib = _compute_node_contribution_to_output(c_nodes, c_edges, c_imp)
            _, _, _, proj_df = project_input_to_l0_neurons(s_nodes, s_edges, s_edge_imp)
            struct = compute_structural_synergy_l0(proj_df, c_node_contrib)

            mat = np.asarray(struct.get("synergy_matrix", []), dtype=float)
            labels = list(struct.get("input_labels", []))

            if mat.size == 0 or len(labels) == 0:
                st.info("当前结构图中缺少有效的 IN->L0 投影边，无法计算结构协作。")
            else:
                fig_s = go.Figure(
                    data=go.Heatmap(
                        z=mat,
                        x=labels,
                        y=labels,
                        colorscale="YlGnBu",
                        colorbar=dict(title="Structural Synergy"),
                    )
                )
                fig_s.update_layout(
                    title="Pairwise Structural Synergy Matrix (L0)",
                    height=520,
                    font=dict(family="Helvetica Neue, Arial, sans-serif", color="#444444"),
                )
                st.plotly_chart(fig_s, use_container_width=True)

                s_single = pd.DataFrame(struct.get("single_structural", []))
                if len(s_single) > 0:
                    st.markdown("**单变量结构影响度（对角项 S(i,i)）**")
                    st.dataframe(s_single.head(20), use_container_width=True, height=220)

                s_pairs = pd.DataFrame(struct.get("pair_ranking", []))
                if len(s_pairs) > 0:
                    st.markdown("**Top 变量对结构协作（按 synergy）**")
                    st.dataframe(
                        s_pairs.head(20)[["var_i_abbr", "var_j_abbr", "synergy"]],
                        use_container_width=True,
                        height=280,
                    )

    else:
        circuit = _safe_read_json(os.path.join(output_dir, "circuit_structure.json"))
        var_interaction = _safe_read_json(os.path.join(output_dir, "variable_interactions.json"))
        in_node_df, _ = _load_in_node_records(output_dir)
        head_df = _safe_read_csv(os.path.join(output_dir, "head_importance_ranking.csv"))
        neuron_df = _safe_read_csv(os.path.join(output_dir, "neuron_importance_ranking.csv"))
        edge_df, _ = _load_edge_importance_table(output_dir)
        node_df = _safe_read_csv(os.path.join(output_dir, "node_importance_ranking.csv"))
        faith_df = _safe_read_csv(os.path.join(output_dir, "faithfulness_curve.csv"))
        cherry = _safe_read_json(os.path.join(output_dir, "cherry_samples.json"))

        st.subheader("Figure 4 风格：最小关键路径 + Faithfulness")
        if circuit is None:
            st.info("未找到 circuit_structure.json")
            return

        feature_names = summary.get("feature_names", []) if summary else []
        nodes, edges, display_map = build_circuit_edges(
            circuit,
            max_neurons_per_layer=max_neurons_graph,
            input_features=feature_names,
        )

        node_imp = build_node_importance(nodes, head_df, neuron_df)
        edge_imp_real = build_edge_importance_from_real(edges, edge_df)
        heuristic_edge_imp = build_edge_importance(edges, node_imp)
        edge_imp = merge_edge_importance(
            edges,
            real_imp=edge_imp_real,
            heuristic_imp=heuristic_edge_imp,
            real_weight=real_edge_weight,
        )
        in_real_imp_full = _build_in_edge_real_importance(in_node_df, edges)
        edge_imp = _inject_in_real_edges(edge_imp, in_real_imp_full, in_real_weight=1.0)

        if collapse_attention_view:
            fig4_nodes, fig4_edges, fig4_edge_imp = collapse_attention_heads(nodes, edges, edge_imp)
        else:
            fig4_nodes, fig4_edges, fig4_edge_imp = nodes, edges, edge_imp

        fig4_node_imp = _compute_node_contribution_to_output(fig4_nodes, fig4_edges, fig4_edge_imp)

        st.caption(f"Figure4 视图模式：{'省略 Attention 头（保留后续层）' if collapse_attention_view else '完整电路'}")

        picker_fig = build_fig4_picker(fig4_nodes, fig4_edges, display_map)
        pick_state = st.plotly_chart(
            picker_fig,
            use_container_width=True,
            key="fig4_picker_chart",
            on_select="rerun",
            selection_mode=("points", "box", "lasso"),
        )

        valid_set = set(fig4_nodes)
        points = _extract_points_from_plot_state(pick_state)
        if len(points) == 0:
            points = _extract_points_from_plot_state(st.session_state.get("fig4_picker_chart"))
        picked_nodes = _extract_node_ids_from_points(points, valid_set)
        if picked_nodes:
            merged_click = set(st.session_state.get("fig4_clicked_nodes", [])) | set(picked_nodes)
            st.session_state["fig4_clicked_nodes"] = sorted(merged_click)
        else:
            picked_nodes = set(st.session_state.get("fig4_clicked_nodes", []))

        picked_nodes = set(st.session_state.get("fig4_clicked_nodes", []))

        if picked_nodes:
            picked_show = ", ".join([display_map.get(n, n) for n in sorted(picked_nodes)])
            st.caption(f"已捕获点击节点：{picked_show}")
        else:
            st.caption("已捕获点击节点：暂无（请直接点击节点圆点，或框选节点）")

        if st.button("生成所选节点 Figure4", key="fig4_generate_from_click"):
            expanded = _expand_with_neighbors(picked_nodes, fig4_edges)
            st.session_state["fig4_manual_nodes"] = sorted(expanded)
            st.session_state["fig4_selected_nodes"] = sorted(expanded)
            st.session_state["fig4_focus_nodes"] = sorted(picked_nodes)

        if st.button("重置点击捕获", key="fig4_reset_click_capture"):
            st.session_state["fig4_clicked_nodes"] = []

        if st.button("清空自定义节点", key="fig4_clear_manual"):
            st.session_state["fig4_manual_nodes"] = []
            st.session_state["fig4_selected_nodes"] = []
            st.session_state["fig4_clicked_nodes"] = []
            st.session_state["fig4_focus_nodes"] = []

        node_options = sorted(fig4_nodes)
        default_selected = []
        if node_df is not None and len(node_df) > 0 and "node_id" in node_df.columns:
            ranked_default = node_df.sort_values("importance", ascending=False)["node_id"].astype(str).tolist()
            default_selected = [n for n in ranked_default if n in set(node_options)][:fig4_topk]
        if len(default_selected) == 0:
            ranked_nodes = sorted(fig4_node_imp.items(), key=lambda kv: kv[1], reverse=True)
            default_selected = [k for k, _ in ranked_nodes[:fig4_topk]]

        if len(st.session_state.get("fig4_selected_nodes", [])) == 0:
            seed = [n for n in st.session_state.get("fig4_manual_nodes", []) if n in set(node_options)] or default_selected
            st.session_state["fig4_selected_nodes"] = list(seed)

        selected_nodes = st.multiselect(
            "自定义节点（可与点击选择联用；留空则使用 Top-K 自动节点）",
            options=node_options,
            format_func=lambda n: f"{display_map.get(n, n)} ({n})",
            key="fig4_selected_nodes",
        )

        if selected_nodes:
            active_nodes = set(selected_nodes)
        elif node_df is not None and len(node_df) > 0 and "node_id" in node_df.columns:
            active_nodes = set(
                [
                    n
                    for n in node_df.sort_values("importance", ascending=False)["node_id"].astype(str).tolist()
                    if n in set(fig4_nodes)
                ][:fig4_topk]
            )
        else:
            ranked_nodes = sorted(fig4_node_imp.items(), key=lambda kv: kv[1], reverse=True)
            active_nodes = set([k for k, _ in ranked_nodes[:fig4_topk]])

        active_nodes.update([n for n in fig4_nodes if n.startswith("IN:") or n.startswith("OUT:")])

        focus_nodes = set(st.session_state.get("fig4_focus_nodes", []))
        if not focus_nodes and selected_nodes:
            focus_nodes = set(selected_nodes)
        focus_nodes = set([n for n in focus_nodes if n in active_nodes])

        fig = build_figure4_circuit(
            fig4_nodes,
            fig4_edges,
            display_map,
            edge_importance=fig4_edge_imp,
            active_node_ids=active_nodes,
            show_inactive=fig4_show_inactive,
            node_contribution=fig4_node_imp,
            input_share_override=_input_contribution_override_from_var_interaction(var_interaction),
            top_ratio=edge_top_ratio,
            focus_node_ids=focus_nodes if focus_nodes else None,
        )
        st.plotly_chart(fig, use_container_width=True)
        if st.button("导出Figure4 PNG(300DPI)", key="export_fig4_png"):
            try:
                p = _save_plot_png(fig, station_id=station_id, image_name="fig4")
                st.success(f"已保存: {p}")
            except Exception as e:
                st.error(f"导出失败: {e}")

        c1, c2 = st.columns([1, 1])
        with c1:
            st.plotly_chart(faithfulness_figure(faith_df), use_container_width=True)
        with c2:
            if node_df is not None and len(node_df) > 0:
                st.markdown("**Top-K 关键节点**")
                st.dataframe(node_df.sort_values("importance", ascending=False).head(fig4_topk), use_container_width=True, height=320)
            else:
                st.info("未找到 node_importance_ranking.csv")

        st.markdown("**Cherry-picked 样本（高流量/低流量）**")
        if cherry and isinstance(cherry, dict):
            high = cherry.get("high_flow_samples", []) or []
            low = cherry.get("low_flow_samples", []) or []
            pool = [("high", i, s) for i, s in enumerate(high)] + [("low", i, s) for i, s in enumerate(low)]

            if pool:
                labels = [f"{grp}-{idx} (target={s['target']:.3f}, pred={s['pred']:.3f})" for grp, idx, s in pool]
                pick = st.selectbox("选择样本", options=list(range(len(labels))), format_func=lambda i: labels[i])
                sel = pool[pick][2]

                if isinstance(sel.get("feature_series"), dict) and len(sel["feature_series"]) > 0:
                    feat_names = list(sel["feature_series"].keys())
                    feat = st.selectbox("选择变量", options=feat_names)
                    y = np.asarray(sel["feature_series"][feat], dtype=float)
                    x = np.arange(len(y))
                    sf = go.Figure(go.Scatter(x=x, y=y, mode="lines", name=feat))
                    sf.update_layout(title=f"Sample Feature Trace: {feat}", xaxis_title="Time Lag", yaxis_title="Value", height=280)
                    st.plotly_chart(sf, use_container_width=True)
                    st.json({"target": sel.get("target"), "pred": sel.get("pred"), "error": sel.get("error")})
            else:
                st.info("cherry_samples.json 为空")
        else:
            st.info("未找到 cherry_samples.json，请重新训练生成。")


@st.cache_data(show_spinner=False)
def _load_cluster_features(output_root: str, stations_csv: str, camels_root: str | None) -> pd.DataFrame:
    return load_station_feature_table(
        output_root=output_root,
        stations_csv=stations_csv,
        camels_root=camels_root,
        top_input_n=8,
    )


def _render_cluster_map(output_root: str, stations_csv: str, camels_root: str | None):
    st.subheader("CAMELS 主导变量空间分布（非聚类）")

    if not os.path.exists(stations_csv):
        st.error(f"站点文件不存在: {stations_csv}")
        return
    if not os.path.isdir(output_root):
        st.error(f"结果目录不存在: {output_root}")
        return

    if camels_root and (not os.path.isdir(camels_root)):
        st.warning("CAMELS 属性目录不可访问，当前仅使用站点+解释结果特征。")
        camels_root = None

    with st.spinner("加载站点特征..."):
        plot_df = _load_cluster_features(output_root, stations_csv, camels_root)
    if len(plot_df) == 0:
        st.info("未发现有效站点输出（outputs/*/summary.json）。")
        return

    plot_df = plot_df.copy()
    plot_df["lat"] = pd.to_numeric(plot_df.get("lat"), errors="coerce")
    plot_df["lon"] = pd.to_numeric(plot_df.get("lon"), errors="coerce")
    plot_df["forcing_elev_m"] = pd.to_numeric(plot_df.get("forcing_elev_m"), errors="coerce")
    if "aridity" in plot_df.columns:
        plot_df["aridity"] = pd.to_numeric(plot_df.get("aridity"), errors="coerce")
    if "frac_snow" in plot_df.columns:
        plot_df["frac_snow"] = pd.to_numeric(plot_df.get("frac_snow"), errors="coerce")

    plot_df["lat_band"] = pd.cut(plot_df["lat"], bins=[-90, 35, 42, 90], labels=["South", "Mid", "North"])
    plot_df["elev_band"] = pd.cut(plot_df["forcing_elev_m"], bins=[-1, 300, 1000, 9000], labels=["Low", "Middle", "High"])
    if "aridity" in plot_df.columns and pd.to_numeric(plot_df["aridity"], errors="coerce").notna().any():
        plot_df["aridity_class"] = pd.cut(plot_df["aridity"], bins=[-1, 0.65, 1.0, 10], labels=["Humid", "Sub-humid", "Arid"])
    else:
        plot_df["aridity_class"] = "unknown"
    if "frac_snow" in plot_df.columns and pd.to_numeric(plot_df["frac_snow"], errors="coerce").notna().any():
        plot_df["snow_regime"] = pd.cut(plot_df["frac_snow"], bins=[-0.1, 0.1, 0.3, 1.0], labels=["Rain-dominant", "Mixed", "Snow-influenced"])
    else:
        plot_df["snow_regime"] = "unknown"

    if "top_pair_interaction_name" in plot_df.columns:
        tpn = plot_df["top_pair_interaction_name"].fillna("unknown").astype(str)
        plot_df["struct_top_factor_name"] = tpn.map(lambda s: s.split("x")[0] if "x" in s else s)
    else:
        plot_df["struct_top_factor_name"] = "unknown"

    def agg_landcover(lc):
        if pd.isna(lc) or str(lc).lower() in ["nan", "unknown"]: return "Unknown"
        s = str(lc).lower()
        if "forest" in s or "wood" in s: return "Forest"
        if "shrub" in s or "grass" in s or "savanna" in s: return "Shrub/Grass"
        if "crop" in s or "agri" in s: return "Crop"
        if "urban" in s or "built" in s: return "Urban"
        if "barren" in s or "sparse" in s or "snow" in s or "ice" in s or "water" in s: return "Barren/Water/Snow"
        return "Other"

    def get_climate_type(row):
        elev = float(row.get("forcing_elev_m", 0) if pd.notna(row.get("forcing_elev_m")) else 0)
        lon = float(row.get("lon", 0) if pd.notna(row.get("lon")) else 0)
        arid = float(row.get("aridity", 1.0) if pd.notna(row.get("aridity")) else 1.0)
        if elev >= 1500:
            return "高原/山地 (Plateau/Mountain)"
        elif lon <= -120:
            return "海洋性气候 (Oceanic)"
        elif arid >= 1.5:
            return "干旱大陆性 (Arid Continental)"
        else:
            return "湿润大陆性/季风 (Humid Continental/Subtropical)"

    if "dom_land_cover" in plot_df.columns:
        plot_df["dom_land_cover_agg"] = plot_df["dom_land_cover"].apply(agg_landcover)
    else:
        plot_df["dom_land_cover_agg"] = "unknown"

    plot_df["geo_climate_type"] = plot_df.apply(get_climate_type, axis=1)

    st.caption(f"当前输出目录已加载站点数: {len(plot_df)}")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        dominant_mode = st.selectbox(
            "主导变量维度",
            options=["top_factor_name", "struct_top_factor_name", "top_pair_interaction_name", "probe_r2_mean"],
            format_func=lambda x: {
                "top_factor_name": "Top Factor（变量名）",
                "struct_top_factor_name": "Struct Top Factor（变量名）",
                "top_pair_interaction_name": "Struct 交互对（变量名）",
                "probe_r2_mean": "Probe 均值水平",
            }.get(x, x),
            index=0,
        )
    with c2:
        nse_min = st.number_input("最小 NSE", value=0.0, step=0.05)
    with c3:
        huc_list = sorted([str(x) for x in plot_df["huc_02"].dropna().unique()]) if "huc_02" in plot_df.columns else []
        sel_huc = st.multiselect("HUC-02", options=huc_list, default=[])
    with c4:
        size_mode = st.selectbox("点大小", options=["fixed", "nse", "top_factor_score", "probe_r2_mean"], index=0)

    if dominant_mode == "probe_r2_mean":
        v = pd.to_numeric(plot_df.get("probe_r2_mean"), errors="coerce")
        if v.notna().sum() > 0:
            try:
                plot_df["dominant_label"] = pd.qcut(v, q=4, labels=["Probe-Low", "Probe-MidLow", "Probe-MidHigh", "Probe-High"], duplicates="drop")
            except Exception:
                plot_df["dominant_label"] = "Probe-unknown"
        else:
            plot_df["dominant_label"] = "Probe-unknown"
    else:
        if dominant_mode in plot_df.columns:
            plot_df["dominant_label"] = plot_df[dominant_mode].fillna("unknown").astype(str)
        else:
            plot_df["dominant_label"] = "unknown"

    f1, f2, f3 = st.columns(3)
    with f1:
        color_mode = st.selectbox(
            "颜色",
            options=[
                "dominant_label",
                "nse",
                "probe_r2_mean",
                "aridity_class",
                "lat_band",
                "elev_band",
                "snow_regime",
                "dom_land_cover",
                "dom_land_cover_agg",
                "geo_climate_type",
                "huc_02",
            ],
            index=0,
        )
    with f2:
        show_name_labels = st.checkbox("地图显示名称标签", value=False)
    with f3:
        show_climate_band = st.checkbox("显示气候带参考线", value=False)

    label_options = [x for x in ["station_id", "dominant_label", "top_factor_name", "struct_top_factor_name", "top_pair_interaction_name"] if x in plot_df.columns]
    label_field = st.selectbox("标签字段", options=label_options, index=0) if len(label_options) > 0 else "station_id"

    mask = pd.Series(True, index=plot_df.index)
    if "nse" in plot_df.columns:
        mask &= pd.to_numeric(plot_df["nse"], errors="coerce").fillna(-999) >= float(nse_min)
    if sel_huc and "huc_02" in plot_df.columns:
        mask &= plot_df["huc_02"].astype(str).isin(sel_huc)
    plot_df = plot_df.loc[mask].copy()

    plot_df = plot_df.dropna(subset=["lat", "lon"])
    if len(plot_df) == 0:
        st.info("筛选后无有效经纬度站点。")
        return

    if size_mode == "fixed" or size_mode not in plot_df.columns:
        plot_df["marker_size"] = 8.0
    else:
        z = pd.to_numeric(plot_df[size_mode], errors="coerce")
        z = z.fillna(z.median())
        zmin, zmax = float(z.min()), float(z.max())
        plot_df["marker_size"] = 8.0 if zmax <= zmin else 6.0 + 12.0 * (z - zmin) / (zmax - zmin)

    color_is_numeric = color_mode in ["nse", "probe_r2_mean"]
    text_arg = None
    if show_name_labels and len(plot_df) <= 250 and label_field in plot_df.columns:
        text_arg = label_field
    elif show_name_labels and len(plot_df) > 250:
        st.info("当前点位较多，已自动关闭文本标签以避免重叠。可先筛选后再开启。")

    common_kwargs = dict(
        data_frame=plot_df,
        lat="lat",
        lon="lon",
        color=color_mode,
        size="marker_size",
        size_max=20,
        text=text_arg,
        hover_name="station_id",
        hover_data=[c for c in ["dominant_label", "top_factor_name", "struct_top_factor_name", "top_pair_interaction_name", "nse", "huc_02"] if c in plot_df.columns],
        scope="usa",
        title="US Stations: 主导变量空间分布",
    )
    if color_is_numeric:
        fig = px.scatter_geo(**common_kwargs, color_continuous_scale="Viridis")
    else:
        fig = px.scatter_geo(**common_kwargs)

    if text_arg is not None:
        fig.update_traces(textposition="top center", textfont_size=9)

    if show_climate_band:
        lon_min = float(plot_df["lon"].min())
        lon_max = float(plot_df["lon"].max())
        lon_seq = np.linspace(lon_min, lon_max, 160)
        for lat_line, label in [(35.0, "South/Mid"), (42.0, "Mid/North")]:
            fig.add_trace(
                go.Scattergeo(
                    lon=lon_seq,
                    lat=np.full_like(lon_seq, lat_line),
                    mode="lines",
                    line=dict(color="rgba(80,80,80,0.55)", width=1, dash="dash"),
                    name=f"气候带参考线 {label}",
                    showlegend=True,
                    hovertemplate=f"lat={lat_line:.1f}<extra>{label}</extra>",
                )
            )

    fig.update_layout(height=650, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 主导变量分布")
    vc = plot_df["dominant_label"].astype(str).value_counts().head(25).reset_index()
    vc.columns = ["dominant_label", "count"]
    fig_vc = px.bar(vc, x="dominant_label", y="count", title="主导变量类别计数")
    fig_vc.update_layout(height=320, xaxis_title="主导变量", yaxis_title="站点数")
    st.plotly_chart(fig_vc, use_container_width=True)

    st.markdown("### 主导变量 × 气候构成占比")
    comp_c1, comp_c2 = st.columns(2)
    with comp_c1:
        comp_dominant_mode = st.selectbox(
            "分析变量",
            options=["top_factor_name", "struct_top_factor_name", "top_pair_interaction_name"],
            format_func=lambda x: {
                "top_factor_name": "Top Factor（变量名）",
                "struct_top_factor_name": "Struct Top Factor（变量名）",
                "top_pair_interaction_name": "Struct 交互对（变量名）",
            }.get(x, x),
            index=0,
            key="comp_dominant_mode",
        )
    with comp_c2:
        climate_mode = st.selectbox(
            "气候指标",
            options=[c for c in ["aridity_class", "elev_band", "lat_band", "snow_regime", "dom_land_cover", "dom_land_cover_agg", "geo_climate_type"] if c in plot_df.columns],
            index=0,
        )
        
    use_col = comp_dominant_mode
    if use_col not in plot_df.columns:
        plot_df[use_col] = "unknown"
        
    use = plot_df[[use_col, climate_mode]].dropna().copy()
    if len(use) == 0:
        st.info("当前筛选下无可用数据。")
    else:
        tab = pd.crosstab(use[use_col].astype(str), use[climate_mode].astype(str), normalize="index")
        # Keep top dominant labels for readability.
        top_labels = plot_df[use_col].astype(str).value_counts().head(15).index.tolist()
        tab = tab.loc[[x for x in tab.index.tolist() if x in top_labels]]
        comp = tab.reset_index().melt(id_vars=[use_col], var_name="climate_class", value_name="ratio")
        b = px.bar(
            comp,
            x=use_col,
            y="ratio",
            color="climate_class",
            barmode="stack",
            title=f"不同主导变量类别的气候构成占比",
        )
        b.update_layout(height=420, yaxis_tickformat=".0%", xaxis_title="主导变量")
        st.plotly_chart(b, use_container_width=True)

    st.markdown("### 相关性与线性拟合分析")
    
    tab1, tab2 = st.tabs(["分类变量相关性检验 (卡方检验)", "连续变量线性拟合 (OLS)"])
    
    with tab1:
        st.write(f"检验 **{use_col}** 与 **{climate_mode}** 之间是否相互关联。")
        try:
            from scipy.stats import chi2_contingency
            raw_tab = pd.crosstab(use[use_col].astype(str), use[climate_mode].astype(str))
            if len(use) > 10 and raw_tab.size > 1 and min(raw_tab.shape) > 1:
                chi2, p, dof, ex = chi2_contingency(raw_tab)
                
                n = raw_tab.sum().sum()
                min_dim = min(raw_tab.shape) - 1
                cramer_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
                
                st.metric(label="Pearson 卡方检验 P-value", value=f"{p:.4e}", help="P值越小(如<0.05)，越能说明两个变量之间存在关联")
                
                if p < 0.05:
                    st.success(f"**结论认为有关联:** P-value < 0.05，在显著性水平0.05下，可以认定 **{use_col}** 和 **{climate_mode}** 之间存在显著关联！\n\n(关联强度 Cramer's V = {cramer_v:.3f}，值越接近1说明关联越强)")
                else:
                    st.warning(f"**未发现明显关联:** P-value >= 0.05，目前证据**无法**证明两者的分布有显著相关性。 (关联强度 Cramer's V: {cramer_v:.3f})")
                
                with st.expander("查看原始列联表"):
                    st.dataframe(raw_tab)
            else:
                st.info("数据样本不够或只有单一类别，无法完成检验。")
        except ImportError:
            st.warning("如需卡方检验请安装 scipy 库。")

    with tab2:
        num_cols = plot_df.select_dtypes(include=[np.number]).columns.tolist()
        num_cols = [c for c in num_cols if plot_df[c].nunique() > 1]
        
        st.write("利用线性回归(OLS)分析**连续维度**变量间的关系，这常用于验证数值指标间的线性趋势。")
        c1, c2 = st.columns(2)
        with c1:
            x_opts = [c for c in ["aridity", "frac_snow", "forcing_elev_m", "lat", "lon", "p_mean"] if c in num_cols] + num_cols
            # Deduplicate while preserving order
            x_opts = list(dict.fromkeys(x_opts))
            x_ax = st.selectbox("X 轴 (如连续气象、地理维度)", options=x_opts, index=0)
        with c2:
            y_opts = [c for c in ["top_factor_score", "nse", "probe_r2_mean"] if c in num_cols] + [c for c in num_cols if "top_factor" in c] + num_cols
            y_opts = list(dict.fromkeys(y_opts))
            y_ax = st.selectbox("Y 轴 (如模型指标、变量注意力得分)", options=y_opts, index=0)
            
        fit_use = plot_df[[x_ax, y_ax]].dropna()
        if len(fit_use) > 2:
            try:
                from scipy.stats import pearsonr
                corr, p_corr = pearsonr(fit_use[x_ax], fit_use[y_ax])
                
                if p_corr < 0.05:
                    st.success(f"**结论认为有关联:** Pearson相关系数 **R = {corr:.3f}** (P值: {p_corr:.4e})。变量存在显著线性关系！")
                else:
                    st.warning(f"**未发现明显线性关联:** Pearson相关系数 **R = {corr:.3f}** (P值: {p_corr:.4e})。说明线性趋势不显著。")
                    
                color_opt = climate_mode if climate_mode in plot_df.columns else None
                fig_scatter = px.scatter(
                    plot_df, x=x_ax, y=y_ax, 
                    hover_name="station_id",
                    color=color_opt,
                    trendline="ols",
                    title=f"`{y_ax}` 与 `{x_ax}` 的线性拟合图"
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            except Exception as e:
                st.error(f"绘图时出错: {e}")
        else:
            st.info("缺少足够的同时满足X和Y的有数值的数据点。")


def main():
    st.title("可交互电路可视化")

    workspace_root = os.getcwd()
    local_default = os.path.join(workspace_root, "outputs")
    preferred_output_root = "/mnt/d/Store/Sparse_Transformer"
    local_stations_default = os.path.join(workspace_root, "data", "stations.csv")
    local_camels_default = "/home/fifth/WorkSpace/CAMELS/Data/Raw"
    if "viz_output_root" not in st.session_state:
        st.session_state["viz_output_root"] = preferred_output_root if os.path.isdir(preferred_output_root) else local_default
    if "viz_stations_csv" not in st.session_state:
        st.session_state["viz_stations_csv"] = local_stations_default
    if "viz_camels_root" not in st.session_state:
        st.session_state["viz_camels_root"] = local_camels_default
    if "viz_view" not in st.session_state:
        st.session_state["viz_view"] = "stations"

    st.sidebar.markdown("### 全局设置")
    output_root = st.sidebar.text_input(
        "结果根目录",
        value=st.session_state.get("viz_output_root", preferred_output_root if os.path.isdir(preferred_output_root) else local_default),
    )
    stations_csv = st.sidebar.text_input("站点元数据CSV", value=st.session_state.get("viz_stations_csv", local_stations_default))
    camels_root = st.sidebar.text_input("CAMELS属性目录", value=st.session_state.get("viz_camels_root", local_camels_default))
    page_mode = st.sidebar.radio("页面", options=["站点总览", "聚类地图"], index=0)
    st.session_state["viz_output_root"] = output_root
    st.session_state["viz_stations_csv"] = stations_csv
    st.session_state["viz_camels_root"] = camels_root

    view_mode = st.session_state.get("viz_view", "stations")
    if page_mode == "聚类地图":
        _render_cluster_map(output_root, stations_csv, camels_root.strip() or None)
    elif view_mode == "detail":
        station_id = st.session_state.get("viz_station_id")
        if not station_id:
            st.session_state["viz_view"] = "stations"
            st.rerun()
        _render_station_detail(output_root, station_id)
    else:
        _render_station_overview(output_root)


if __name__ == "__main__":
    main()
