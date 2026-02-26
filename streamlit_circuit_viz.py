from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="Flood Circuit Visualizer", layout="wide")


def _safe_read_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_read_csv(path: str):
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


def build_circuit_figure(
    nodes: List[str],
    edges: List[Tuple[str, str]],
    display_map: Dict[str, str],
    edge_importance: Dict[Tuple[str, str], float] | None = None,
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
            textposition="top center",
            hovertemplate="node=%{text}<extra></extra>",
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
            textposition="top center",
            hovertemplate="node=%{text}<extra></extra>",
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

    single = {k: float(v) for k, v in var_interaction.get("single_delta", {}).items()}
    pairs = var_interaction.get("pair_ranking", []) or []

    single_keys = list(single.keys())
    single_vals = np.asarray([single[k] for k in single_keys], dtype=float)
    single_scaled_vals = _scale_signed_array(single_vals, scale_mode)
    single_scaled = {k: float(v) for k, v in zip(single_keys, single_scaled_vals)}

    # 构建节点：变量、交互对、输出
    var_nodes = list(single.keys())
    pair_rows = sorted(pairs, key=lambda d: abs(float(d.get("interaction", 0.0))), reverse=True)[:top_pairs]
    pair_inter_vals = np.asarray([float(r.get("interaction", 0.0)) for r in pair_rows], dtype=float)
    pair_inter_scaled = _scale_signed_array(pair_inter_vals, scale_mode)
    pair_nodes = [f"{r['var_i']}×{r['var_j']}" for r in pair_rows]
    out_node = "Q (streamflow)"

    labels = var_nodes + pair_nodes + [out_node]
    idx = {n: i for i, n in enumerate(labels)}

    src, dst, val, color = [], [], [], []

    # 变量 -> 输出（单变量影响）
    for vname, dv in single_scaled.items():
        src.append(idx[vname])
        dst.append(idx[out_node])
        val.append(abs(dv) + 1e-6)
        color.append("rgba(80,80,80,0.45)")

    # 变量 -> 交互项 -> 输出
    for row, pnode, inter_scaled in zip(pair_rows, pair_nodes, pair_inter_scaled):
        vi = str(row["var_i"])
        vj = str(row["var_j"])
        inter = float(row.get("interaction", 0.0))
        w = abs(float(inter_scaled)) + 1e-6
        c = "rgba(214,39,40,0.65)" if inter >= 0 else "rgba(31,119,180,0.65)"

        src.extend([idx[vi], idx[vj], idx[pnode]])
        dst.extend([idx[pnode], idx[pnode], idx[out_node]])
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
                    color=["#9467bd"] * len(var_nodes)
                    + ["#2ca02c"] * len(pair_nodes)
                    + ["#ff7f0e"],
                ),
                link=dict(source=src, target=dst, value=val, color=color),
            )
        ]
    )
    fig.update_layout(title=f"变量→交互→流量 机制链路图（{_scale_label(scale_mode)}）", height=520)
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


def main():
    st.title("可交互电路可视化")

    output_root = "outputs"
    station_dirs: List[str] = []
    if os.path.exists(output_root) and os.path.isdir(output_root):
        for name in sorted(os.listdir(output_root)):
            p = os.path.join(output_root, name)
            if os.path.isdir(p) and os.path.exists(os.path.join(p, "summary.json")):
                station_dirs.append(name)

    if station_dirs:
        query = st.sidebar.text_input("搜索站点ID", value="")
        filtered = [s for s in station_dirs if query.strip().lower() in s.lower()] if query.strip() else station_dirs
        if len(filtered) == 0:
            st.sidebar.warning("未匹配到站点ID，请修改搜索条件")
            filtered = station_dirs

        if "viz_station_id" not in st.session_state or st.session_state.get("viz_station_id") not in filtered:
            st.session_state["viz_station_id"] = filtered[0]
        station_id = st.sidebar.selectbox("站点", options=filtered, key="viz_station_id")
        output_dir = os.path.join(output_root, station_id)
        st.sidebar.caption(f"当前结果目录: {output_dir}")
    else:
        station_id = os.path.basename(os.path.normpath(output_root)) or "unknown"
        output_dir = output_root
        st.sidebar.info("未找到 outputs/站点ID 结果目录，将读取 outputs 根目录")
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

    summary = _safe_read_json(os.path.join(output_dir, "summary.json"))
    if summary:
        st.sidebar.success("summary.json 已加载")

    head_df = _safe_read_csv(os.path.join(output_dir, "head_importance_ranking.csv"))
    neuron_df = _safe_read_csv(os.path.join(output_dir, "neuron_importance_ranking.csv"))
    edge_df = _safe_read_csv(os.path.join(output_dir, "edge_importance_ranking.csv"))
    node_df = _safe_read_csv(os.path.join(output_dir, "node_importance_ranking.csv"))
    faith_df = _safe_read_csv(os.path.join(output_dir, "faithfulness_curve.csv"))
    cherry = _safe_read_json(os.path.join(output_dir, "cherry_samples.json"))
    var_interaction = _safe_read_json(os.path.join(output_dir, "variable_interactions.json"))
    var_interaction_rank = _safe_read_csv(os.path.join(output_dir, "variable_interactions_ranking.csv"))
    causal = _safe_read_json(os.path.join(output_dir, "causal_intervention_results.json"))
    probe = _safe_read_json(os.path.join(output_dir, "physical_probe_results.json"))
    circuit = _safe_read_json(os.path.join(output_dir, "circuit_structure.json"))
    attn_stats = _safe_read_json(os.path.join(output_dir, "attention_statistics.json"))

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["电路图", "Head/Neuron 排名", "Attention", "因果与探针", "变量交互", "Figure4风格"])

    with tab1:
        if circuit is None:
            st.warning("未找到 circuit_structure.json，请先运行训练生成结果。")
        else:
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

            if collapse_attention_view:
                p_df = project_input_to_l0_neurons(nodes, edges, edge_imp)[3]
                c_nodes, c_edges, c_imp = collapse_attention_heads(nodes, edges, edge_imp)
                fig = build_circuit_figure(
                    c_nodes,
                    c_edges,
                    display_map,
                    edge_importance=c_imp,
                    top_ratio=edge_top_ratio,
                )
                shown_nodes, shown_edges = c_nodes, c_edges
            else:
                p_df = None
                fig = build_circuit_figure(
                    nodes,
                    edges,
                    display_map,
                    edge_importance=edge_imp,
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

            if collapse_attention_view:
                st.markdown("**输入变量 → 第一层节点（L0 Neuron）影响权重**")
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
                    st.info("当前没有可展示的输入→L0节点投影权重。")

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

    with tab2:
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

    with tab3:
        if attn_stats:
            st.plotly_chart(attention_figure(attn_stats), use_container_width=True)
            st.json({"flood_threshold_y": attn_stats.get("flood_threshold_y")})
        else:
            st.info("未找到 attention_statistics.json")

    with tab4:
        if summary and isinstance(summary.get("metrics"), dict):
            metrics = summary["metrics"]
            nse = metrics.get("nse")
            nse_thr = metrics.get("nse_threshold")
            nse_pass = metrics.get("nse_pass")

            c0, c1, c2 = st.columns(3)
            c0.metric("NSE", f"{nse:.4f}" if nse is not None else "NA")
            c1.metric("NSE Threshold", f"{nse_thr:.2f}" if nse_thr is not None else "NA")
            c2.metric("NSE Pass", "✅" if nse_pass else "❌")

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

    with tab5:
        st.subheader("输入变量相互作用（真实干预）")
        scale_label = st.selectbox(
            "交互值缩放方式",
            options=["符号对数", "原始值", "Z-Score", "稳健缩放(IQR)", "绝对值归一化"],
            index=0,
            key="interaction_scale_mode",
        )
        scale_mode_map = {
            "符号对数": "signed_log",
            "原始值": "raw",
            "Z-Score": "zscore",
            "稳健缩放(IQR)": "robust",
            "绝对值归一化": "abs_norm",
        }
        scale_mode = scale_mode_map.get(scale_label, "signed_log")

        if var_interaction and "interaction_matrix" in var_interaction and "feature_names" in var_interaction:
            names = var_interaction["feature_names"]
            mat = np.asarray(var_interaction["interaction_matrix"], dtype=float)
            mat_scaled = _scale_signed_array(mat, scale_mode)

            fig = go.Figure(
                data=go.Heatmap(
                    z=mat_scaled,
                    x=names,
                    y=names,
                    colorscale="RdBu",
                    zmid=0.0,
                    colorbar=dict(title=f"Interaction ({_scale_label(scale_mode)})"),
                )
            )
            fig.update_layout(title=f"Pairwise Interaction Matrix ({_scale_label(scale_mode)})", height=520)
            st.plotly_chart(fig, use_container_width=True)

            if isinstance(var_interaction.get("single_delta"), dict):
                single_df = pd.DataFrame(
                    [{"variable": k, "delta_single": v} for k, v in var_interaction["single_delta"].items()]
                ).sort_values("delta_single", ascending=False)
                single_df["delta_single_scaled"] = _scale_signed_array(single_df["delta_single"].to_numpy(dtype=float), scale_mode)
                st.markdown("**单变量干预敏感性（Δ_i）**")
                st.dataframe(single_df, use_container_width=True, height=220)
        else:
            st.info("未找到 variable_interactions.json，请重新训练后生成。")

        if var_interaction_rank is not None and len(var_interaction_rank) > 0:
            show = var_interaction_rank.copy()
            show["abs_interaction"] = show["interaction"].abs()
            show = show.sort_values("abs_interaction", ascending=False).head(20)
            show["interaction_scaled"] = _scale_signed_array(show["interaction"].to_numpy(dtype=float), scale_mode)
            st.markdown("**Top 变量对交互（按 |interaction|）**")
            st.dataframe(
                show[["var_i", "var_j", "interaction", "interaction_scaled", "delta_pair", "delta_i", "delta_j"]],
                use_container_width=True,
                height=280,
            )

        st.markdown("**变量-交互-流量机制链路**")
        sk = variable_mechanism_sankey(var_interaction, top_pairs=8, scale_mode=scale_mode)
        if len(sk.data) > 0:
            st.plotly_chart(sk, use_container_width=True)
            st.caption("红色链路：正协同交互（interaction>0）；蓝色链路：负交互/替代关系（interaction<0）。")
        else:
            st.info("缺少 variable_interactions.json，无法绘制机制链路图。")

    with tab6:
        st.subheader("Figure 4 风格：最小关键路径 + Faithfulness")
        if circuit is None:
            st.info("未找到 circuit_structure.json")
        else:
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

            if collapse_attention_view:
                fig4_nodes, fig4_edges, fig4_edge_imp = collapse_attention_heads(nodes, edges, edge_imp)
            else:
                fig4_nodes, fig4_edges, fig4_edge_imp = nodes, edges, edge_imp

            fig4_node_imp = _derive_node_importance_from_edges(
                fig4_nodes,
                fig4_edges,
                fig4_edge_imp,
                base_node_importance={n: float(node_imp.get(n, 0.0)) for n in fig4_nodes},
            )

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
                ranked_default = (
                    node_df.sort_values("importance", ascending=False)["node_id"].astype(str).tolist()
                )
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
                    [n for n in node_df.sort_values("importance", ascending=False)["node_id"].astype(str).tolist() if n in set(fig4_nodes)][:fig4_topk]
                )
            else:
                ranked_nodes = sorted(fig4_node_imp.items(), key=lambda kv: kv[1], reverse=True)
                active_nodes = set([k for k, _ in ranked_nodes[:fig4_topk]])

            # ensure IO nodes remain visible
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


if __name__ == "__main__":
    main()
