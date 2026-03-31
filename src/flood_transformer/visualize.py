from __future__ import annotations

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_head_ranking(records: List[Dict[str, float]], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(records)
    df = df.sort_values("delta_mse", ascending=False).reset_index(drop=True)
    path = os.path.join(output_dir, "head_importance_ranking.csv")
    df.to_csv(path, index=False)

    topn = min(20, len(df))
    if topn > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        labels = [f"L{int(r['layer'])}-H{int(r['head'])}" for _, r in df.iloc[:topn].iterrows()]
        ax.barh(labels[::-1], df.iloc[:topn]["delta_mse"].values[::-1])
        ax.set_title("Top Attention Heads Importance (delta MSE)")
        ax.set_xlabel("delta MSE")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "head_importance_top20.png"), dpi=180)
        plt.close(fig)

    return path


def save_neuron_ranking(records: List[Dict[str, float]], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(records)
    df = df.sort_values("delta_mse", ascending=False).reset_index(drop=True)
    path = os.path.join(output_dir, "neuron_importance_ranking.csv")
    df.to_csv(path, index=False)

    topn = min(30, len(df))
    if topn > 0:
        fig, ax = plt.subplots(figsize=(10, 8))
        labels = [f"L{int(r['layer'])}-N{int(r['index'])}" for _, r in df.iloc[:topn].iterrows()]
        ax.barh(labels[::-1], df.iloc[:topn]["delta_mse"].values[::-1])
        ax.set_title("Top MLP Neurons Importance (delta MSE)")
        ax.set_xlabel("delta MSE")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "neuron_importance_top30.png"), dpi=180)
        plt.close(fig)

    return path


def save_attention_statistics(attn_stats: Dict[str, np.ndarray | float], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    flood = np.array(attn_stats["flood_attention"])
    normal = np.array(attn_stats["normal_attention"])
    diff = np.array(attn_stats["attention_diff"])

    x = np.arange(len(flood))
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(x, flood)
    axes[0].set_title("Flood Attention (last query)")

    axes[1].plot(x, normal)
    axes[1].set_title("Normal Attention (last query)")

    axes[2].plot(x, diff)
    axes[2].set_title("Flood - Normal Attention")
    axes[2].set_xlabel("Lag index")

    plt.tight_layout()
    path = os.path.join(output_dir, "attention_statistics.png")
    plt.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_circuit_graph(circuit: Dict[str, List], output_dir: str) -> str:
    """将最小电路可视化为简图。"""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "circuit_graph.png")

    try:
        import networkx as nx

        g = nx.DiGraph()
        g.add_node("Input")

        active_heads = circuit["active_heads"]
        active_neurons = circuit["active_neurons"]

        last_nodes = ["Input"]
        for l, (h_mask, n_mask) in enumerate(zip(active_heads, active_neurons)):
            h_idx = np.where(np.asarray(h_mask) > 0)[0]
            n_idx = np.where(np.asarray(n_mask) > 0)[0]

            head_nodes = [f"L{l}_H{int(i)}" for i in h_idx]
            neuron_nodes = [f"L{l}_N{int(i)}" for i in n_idx[:20]]  # 过多神经元时做截断显示

            for node in head_nodes + neuron_nodes:
                g.add_node(node)

            for src in last_nodes:
                for dst in head_nodes:
                    g.add_edge(src, dst)
            for h in head_nodes:
                for n in neuron_nodes:
                    g.add_edge(h, n)

            last_nodes = neuron_nodes if neuron_nodes else head_nodes

        g.add_node("Output")
        for src in last_nodes:
            g.add_edge(src, "Output")

        plt.figure(figsize=(16, 10))
        pos = nx.spring_layout(g, seed=42)
        nx.draw_networkx(g, pos=pos, node_size=400, font_size=7, arrows=True)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()
        return path
    except Exception:
        # 回退：至少输出一个占位图
        plt.figure(figsize=(6, 3))
        plt.text(0.05, 0.6, "Circuit graph rendering requires networkx.", fontsize=12)
        plt.text(0.05, 0.3, "Install: pip install networkx", fontsize=10)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        return path
