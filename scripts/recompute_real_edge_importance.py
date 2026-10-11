from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from src.flood_transformer.config import ExperimentConfig
from src.flood_transformer.data import load_data
from src.flood_transformer.explain import edge_ablation_test, in_node_interaction_test
from src.flood_transformer.model import ExplainableSparseTransformer


def _infer_model_dims_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int, int, int]:
    input_dim = int(state_dict["input_proj.weight"].shape[1])
    d_model = int(state_dict["input_proj.weight"].shape[0])

    layer_ids = set()
    for k in state_dict.keys():
        if k.startswith("layers."):
            try:
                layer_ids.add(int(k.split(".")[1]))
            except Exception:
                continue
    num_layers = (max(layer_ids) + 1) if layer_ids else 0

    n_heads = int(state_dict["layers.0.attn.head_mask_logits"].numel())
    d_ff = int(state_dict["layers.0.mlp.fc1.weight"].shape[0])
    return input_dim, d_model, n_heads, d_ff


def _load_circuit_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_all_internal_edges(circuit: Dict) -> List[Tuple[str, str, float]]:
    """Build all evaluable internal edges in current sparse circuit.

    Notes:
    - IN->L0.H edges are NOT included because current edge_ablation_test
      only supports node ids in format L{layer}.H{idx} / L{layer}.N{idx}.
    """
    active_heads = circuit.get("active_heads", [])
    active_neurons = circuit.get("active_neurons", [])

    candidates: List[Tuple[str, str, float]] = []

    # Intra-layer: H(l) -> N(l)
    for l_idx, (h_mask, n_mask) in enumerate(zip(active_heads, active_neurons)):
        h_idx = [i for i, v in enumerate(h_mask) if float(v) > 0.0]
        n_idx = [i for i, v in enumerate(n_mask) if float(v) > 0.0]
        for h in h_idx:
            s = f"L{l_idx}.H{h}"
            for n in n_idx:
                t = f"L{l_idx}.N{n}"
                candidates.append((s, t, 0.0))

    # Inter-layer: N(l) -> H(l+1)
    for l_idx in range(max(0, len(active_heads) - 1)):
        n_idx = [i for i, v in enumerate(active_neurons[l_idx]) if float(v) > 0.0]
        h_next = [i for i, v in enumerate(active_heads[l_idx + 1]) if float(v) > 0.0]
        for n in n_idx:
            s = f"L{l_idx}.N{n}"
            for h in h_next:
                t = f"L{l_idx + 1}.H{h}"
                candidates.append((s, t, 0.0))

    # Dedup while preserving order
    seen = set()
    dedup: List[Tuple[str, str, float]] = []
    for s, t, sc in candidates:
        if (s, t) in seen:
            continue
        seen.add((s, t))
        dedup.append((s, t, sc))
    return dedup


def _maybe_limit_edges(
    edges: List[Tuple[str, str, float]],
    max_edges: int,
    output_dir: str,
) -> List[Tuple[str, str, float]]:
    if max_edges <= 0 or len(edges) <= max_edges:
        return edges

    node_rank_path = os.path.join(output_dir, "node_importance_ranking.csv")
    if not os.path.exists(node_rank_path):
        return edges[:max_edges]

    df = pd.read_csv(node_rank_path)
    if not {"node_id", "importance"}.issubset(set(df.columns)):
        return edges[:max_edges]

    node_imp = {
        str(r["node_id"]): max(0.0, float(r["importance"]))
        for _, r in df.iterrows()
    }

    scored = []
    for s, t, _ in edges:
        score = float(np.sqrt(max(0.0, node_imp.get(s, 0.0) * node_imp.get(t, 0.0))))
        scored.append((s, t, score))

    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:max_edges]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Recompute REAL edge importance without retraining.")
    p.add_argument("--station_id", required=True, help="Station ID, e.g. 04056500")
    p.add_argument("--output_root", default="/mnt/d/Store/Sparse_Transformer", help="Root dir containing station outputs")
    p.add_argument("--forcing_dir", default="data/Forcing")
    p.add_argument("--streamflow_dir", default="data/Streamflow")
    p.add_argument("--device", default="cpu")

    # Must match training/data preprocessing settings if you changed them.
    p.add_argument("--seq_len", type=int, default=15)
    p.add_argument("--pred_horizon", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--include_q_input", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--circuit_threshold", type=float, default=0.5, help="Threshold used by binary overrides in edge ablation")
    p.add_argument("--max_edges", type=int, default=0, help="0 means all evaluable edges; >0 limits edges for speed")
    p.add_argument("--out_csv", default="edge_importance_ranking_real.csv")
    p.add_argument("--compute_in_node_bridge", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--in_node_bridge_max_nodes", type=int, default=32, help="<=0 means no truncation")
    p.add_argument("--in_node_out_json", default="in_node_interactions.json")
    p.add_argument("--in_node_out_csv", default="in_node_interactions_ranking.csv")
    p.add_argument("--bundle_out_json", default="real_importance_bundle.json")
    return p.parse_args()


def recompute_real_edge_importance_for_station(
    station_id: str,
    output_root: str = "/mnt/d/Store/Sparse_Transformer",
    forcing_dir: str = "data/Forcing",
    streamflow_dir: str = "data/Streamflow",
    device_name: str = "cpu",
    seq_len: int = 15,
    pred_horizon: int = 1,
    batch_size: int = 128,
    include_q_input: bool = True,
    circuit_threshold: float = 0.5,
    max_edges: int = 0,
    out_csv: str = "edge_importance_ranking_real.csv",
    compute_in_node_bridge: bool = True,
    in_node_bridge_max_nodes: int = 32,
    in_node_out_json: str = "in_node_interactions.json",
    in_node_out_csv: str = "in_node_interactions_ranking.csv",
    bundle_out_json: str = "real_importance_bundle.json",
) -> str:
    station_dir = os.path.join(output_root, station_id)

    model_path = os.path.join(station_dir, "sparse_transformer.pt")
    circuit_path = os.path.join(station_dir, "circuit_structure.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(circuit_path):
        raise FileNotFoundError(f"Circuit json not found: {circuit_path}")

    device = torch.device(device_name)

    state = torch.load(model_path, map_location=device)
    input_dim, d_model, n_heads, d_ff = _infer_model_dims_from_state_dict(state)
    num_layers = len({int(k.split(".")[1]) for k in state.keys() if k.startswith("layers.")})

    cfg = ExperimentConfig(
        station_id=station_id,
        forcing_dir=forcing_dir,
        streamflow_dir=streamflow_dir,
        device=device_name,
        seq_len=seq_len,
        pred_horizon=pred_horizon,
        batch_size=batch_size,
        include_streamflow_history_input=include_q_input,
        output_dir=station_dir,
    )

    data = load_data(cfg)

    model = ExplainableSparseTransformer(
        input_dim=input_dim,
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=cfg.dropout,
        task_type=cfg.task_type,
        enable_activation_sparsity=cfg.enable_activation_sparsity,
        activation_sparsity_ratio=cfg.target_activation_sparsity,
        enable_learnable_masks=cfg.enable_learnable_masks,
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    circuit = _load_circuit_json(circuit_path)
    all_edges = _build_all_internal_edges(circuit)
    eval_edges = _maybe_limit_edges(all_edges, max_edges, station_dir)

    print(f"[info] station={station_id}")
    print(f"[info] all_internal_edges={len(all_edges)}")
    print(f"[info] evaluating_edges={len(eval_edges)}")

    # Real edge ablation: no retraining, only test-set forward passes with overrides.
    edge_records = edge_ablation_test(
        model,
        data.test_loader,
        device,
        candidate_edges=eval_edges,
        threshold=circuit_threshold,
    )

    out_path = os.path.join(station_dir, out_csv)
    pd.DataFrame(edge_records).to_csv(out_path, index=False)
    print(f"[done] saved real edge ranking: {out_path}")

    bridge = None

    if compute_in_node_bridge:
        l0_targets: List[str] = []
        active_heads = circuit.get("active_heads", [])
        active_neurons = circuit.get("active_neurons", [])
        if len(active_heads) > 0:
            h0 = np.asarray(active_heads[0])
            l0_targets.extend([f"L0.H{i}" for i in np.where(h0 > 0)[0].tolist()])
        if len(active_neurons) > 0:
            n0 = np.asarray(active_neurons[0])
            l0_targets.extend([f"L0.N{i}" for i in np.where(n0 > 0)[0].tolist()])

        if in_node_bridge_max_nodes > 0:
            l0_targets = l0_targets[:in_node_bridge_max_nodes]

        print(f"[info] station={station_id} in_node_targets={len(l0_targets)}")
        bridge = in_node_interaction_test(
            model,
            data.test_loader,
            device,
            feature_names=data.feature_names,
            target_node_ids=l0_targets,
            mode="replace_mean",
            threshold=circuit_threshold,
        )

        bridge_json_path = os.path.join(station_dir, in_node_out_json)
        with open(bridge_json_path, "w", encoding="utf-8") as f:
            json.dump(bridge, f, ensure_ascii=False, indent=2)

        bridge_csv_path = os.path.join(station_dir, in_node_out_csv)
        pd.DataFrame(bridge.get("records", [])).to_csv(bridge_csv_path, index=False)
        print(f"[done] saved IN->node bridge json: {bridge_json_path}")
        print(f"[done] saved IN->node bridge csv: {bridge_csv_path}")

    bundle_path = os.path.join(station_dir, bundle_out_json)
    with open(bundle_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "edge_importance": edge_records,
                "in_node_interactions": bridge,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[done] saved unified bundle: {bundle_path}")

    return out_path


def main() -> None:
    args = parse_args()
    recompute_real_edge_importance_for_station(
        station_id=args.station_id,
        output_root=args.output_root,
        forcing_dir=args.forcing_dir,
        streamflow_dir=args.streamflow_dir,
        device_name=args.device,
        seq_len=args.seq_len,
        pred_horizon=args.pred_horizon,
        batch_size=args.batch_size,
        include_q_input=args.include_q_input,
        circuit_threshold=args.circuit_threshold,
        max_edges=args.max_edges,
        out_csv=args.out_csv,
        compute_in_node_bridge=args.compute_in_node_bridge,
        in_node_bridge_max_nodes=args.in_node_bridge_max_nodes,
        in_node_out_json=args.in_node_out_json,
        in_node_out_csv=args.in_node_out_csv,
        bundle_out_json=args.bundle_out_json,
    )


if __name__ == "__main__":
    main()
