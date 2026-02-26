from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .model import ExplainableSparseTransformer


@torch.no_grad()
def extract_hidden_states(
    model: ExplainableSparseTransformer,
    data_loader: DataLoader,
    device: torch.device,
) -> List[np.ndarray]:
    """提取所有层 hidden states。

    返回:
        list[np.ndarray]，长度为 num_layers+1，
        每个元素形状为 [N, L, D]
    """
    model.eval()
    all_layers: Optional[List[List[np.ndarray]]] = None

    for x, _ in data_loader:
        x = x.to(device)
        out = model(x, return_intermediates=True)
        hidden = [h.detach().cpu().numpy() for h in out.hidden_states]

        if all_layers is None:
            all_layers = [[] for _ in range(len(hidden))]
        for i, h in enumerate(hidden):
            all_layers[i].append(h)

    assert all_layers is not None
    return [np.concatenate(parts, axis=0) for parts in all_layers]


@torch.no_grad()
def _collect_preds(
    model: ExplainableSparseTransformer,
    data_loader: DataLoader,
    device: torch.device,
    intervention: Optional[str] = None,
    rainfall_index: Optional[int] = None,
    input_mask_override: Optional[torch.Tensor] = None,
    head_mask_overrides: Optional[Dict[int, torch.Tensor]] = None,
    neuron_mask_overrides: Optional[Dict[int, torch.Tensor]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[torch.Tensor]]:
    model.eval()
    preds, ys = [], []
    all_attn = []

    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)

        if intervention is not None and rainfall_index is not None:
            x_mod = x.clone()
            rain = x_mod[:, :, rainfall_index]

            if intervention == "replace_mean":
                rain_mean = rain.mean(dim=1, keepdim=True)
                x_mod[:, :, rainfall_index] = rain_mean
            elif intervention == "shuffle_keep_total":
                # 仅打乱时间顺序，总和天然保持不变
                bsz, seq = rain.shape
                idx = torch.stack([torch.randperm(seq, device=x.device) for _ in range(bsz)], dim=0)
                shuffled = rain.gather(1, idx)
                x_mod[:, :, rainfall_index] = shuffled
            x = x_mod

        out = model(
            x,
            input_mask_override=input_mask_override,
            head_mask_overrides=head_mask_overrides,
            neuron_mask_overrides=neuron_mask_overrides,
            return_intermediates=True,
        )
        preds.append(out.pred.detach().cpu().numpy())
        ys.append(y.detach().cpu().numpy())

        # 记录最后一层注意力，供统计使用
        if out.attention_maps:
            all_attn.append(out.attention_maps[-1].detach().cpu())

    pred_np = np.concatenate(preds)
    y_np = np.concatenate(ys)
    return pred_np, y_np, all_attn


def linear_probe(hidden_states: np.ndarray, physical_variable: np.ndarray) -> Dict[str, np.ndarray | float]:
    """线性探针：用隐藏态预测物理变量。

    参数:
        hidden_states: [N, L, D] 或 [N, D]
        physical_variable: [N]
    """
    if hidden_states.ndim == 3:
        x = hidden_states[:, -1, :]
    else:
        x = hidden_states

    y = physical_variable.reshape(-1, 1)
    n = x.shape[0]
    n_train = int(n * 0.8)

    x_train, x_test = x[:n_train], x[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    x_aug_train = np.concatenate([x_train, np.ones((len(x_train), 1))], axis=1)
    x_aug_test = np.concatenate([x_test, np.ones((len(x_test), 1))], axis=1)

    w, *_ = np.linalg.lstsq(x_aug_train, y_train, rcond=None)
    pred = x_aug_test @ w

    ss_res = float(np.sum((y_test - pred) ** 2))
    ss_tot = float(np.sum((y_test - y_test.mean()) ** 2) + 1e-12)
    r2 = 1.0 - ss_res / ss_tot

    return {
        "r2": r2,
        "coef": w[:-1, 0],
        "bias": float(w[-1, 0]),
    }


@torch.no_grad()
def get_attention_statistics(
    model: ExplainableSparseTransformer,
    data_loader: DataLoader,
    device: torch.device,
    flood_quantile: float = 0.9,
) -> Dict[str, np.ndarray | float]:
    """统计洪水前后注意力分布。"""
    model.eval()
    attn_last_query = []
    y_list = []

    for x, y in data_loader:
        x = x.to(device)
        out = model(x, return_intermediates=True)
        if not out.attention_maps:
            continue

        # 最后一层 attention: [B, H, L, L]
        attn = out.attention_maps[-1].detach().cpu().numpy()
        # 取最后一个 query 对所有历史 key 的关注
        attn = attn[:, :, -1, :]  # [B, H, L]
        attn = attn.mean(axis=1)  # [B, L]

        attn_last_query.append(attn)
        y_list.append(y.numpy())

    attn_all = np.concatenate(attn_last_query, axis=0)
    y_all = np.concatenate(y_list, axis=0)

    flood_thr = np.quantile(y_all, flood_quantile)
    flood_mask = y_all >= flood_thr

    flood_attn = attn_all[flood_mask].mean(axis=0) if flood_mask.any() else np.zeros(attn_all.shape[1])
    normal_attn = attn_all[~flood_mask].mean(axis=0) if (~flood_mask).any() else np.zeros(attn_all.shape[1])

    return {
        "flood_threshold_y": float(flood_thr),
        "flood_attention": flood_attn,
        "normal_attention": normal_attn,
        "attention_diff": flood_attn - normal_attn,
    }


@torch.no_grad()
def head_ablation_test(
    model: ExplainableSparseTransformer,
    data_loader: DataLoader,
    device: torch.device,
) -> List[Dict[str, float]]:
    """逐层逐 head 消融，评估性能下降（MSE increase）。"""
    model.eval()

    base_pred, base_y, _ = _collect_preds(model, data_loader, device)
    base_mse = float(np.mean((base_pred - base_y) ** 2))

    results = []
    for l_idx, layer in enumerate(model.layers):
        n_heads = layer.attn.n_heads
        for h_idx in range(n_heads):
            overrides = {i: torch.ones(layer.attn.n_heads, device=device) for i, layer in enumerate(model.layers)}
            overrides[l_idx][h_idx] = 0.0

            preds = []
            ys = []
            for x, y in data_loader:
                x = x.to(device)
                out = model(x, head_mask_overrides=overrides, return_intermediates=False)
                preds.append(out.pred.detach().cpu().numpy())
                ys.append(y.numpy())

            p = np.concatenate(preds)
            yy = np.concatenate(ys)
            mse = float(np.mean((p - yy) ** 2))
            results.append(
                {
                    "layer": l_idx,
                    "head": h_idx,
                    "mse": mse,
                    "delta_mse": mse - base_mse,
                }
            )

    results.sort(key=lambda d: d["delta_mse"], reverse=True)
    return results


@torch.no_grad()
def mean_ablation(
    model: ExplainableSparseTransformer,
    data_loader: DataLoader,
    device: torch.device,
    ablate: str = "neuron",
) -> List[Dict[str, float]]:
    """均值消融测试。

    ablate='head': 将某个 head gate 替换为该层 head gate 均值
    ablate='neuron': 将某个 neuron gate 替换为该层 neuron gate 均值
    """
    model.eval()

    base_pred, base_y, _ = _collect_preds(model, data_loader, device)
    base_mse = float(np.mean((base_pred - base_y) ** 2))

    records = []
    for l_idx, layer in enumerate(model.layers):
        if ablate == "head":
            probs = torch.sigmoid(layer.attn.head_mask_logits).detach().to(device)
            mean_val = float(probs.mean().item())
            for i in range(len(probs)):
                overrides = {j: torch.sigmoid(ly.attn.head_mask_logits).detach().to(device) for j, ly in enumerate(model.layers)}
                overrides[l_idx] = overrides[l_idx].clone()
                overrides[l_idx][i] = mean_val

                preds = []
                ys = []
                for x, y in data_loader:
                    x = x.to(device)
                    out = model(x, head_mask_overrides=overrides, return_intermediates=False)
                    preds.append(out.pred.detach().cpu().numpy())
                    ys.append(y.numpy())
                mse = float(np.mean((np.concatenate(preds) - np.concatenate(ys)) ** 2))
                records.append({"layer": l_idx, "index": i, "delta_mse": mse - base_mse})
        else:
            probs = torch.sigmoid(layer.mlp.neuron_mask_logits).detach().to(device)
            mean_val = float(probs.mean().item())
            for i in range(len(probs)):
                overrides = {j: torch.sigmoid(ly.mlp.neuron_mask_logits).detach().to(device) for j, ly in enumerate(model.layers)}
                overrides[l_idx] = overrides[l_idx].clone()
                overrides[l_idx][i] = mean_val

                preds = []
                ys = []
                for x, y in data_loader:
                    x = x.to(device)
                    out = model(x, neuron_mask_overrides=overrides, return_intermediates=False)
                    preds.append(out.pred.detach().cpu().numpy())
                    ys.append(y.numpy())
                mse = float(np.mean((np.concatenate(preds) - np.concatenate(ys)) ** 2))
                records.append({"layer": l_idx, "index": i, "delta_mse": mse - base_mse})

    records.sort(key=lambda d: d["delta_mse"], reverse=True)
    return records


@torch.no_grad()
def input_ablation_test(
    model: ExplainableSparseTransformer,
    data_loader: DataLoader,
    device: torch.device,
    feature_names: List[str],
) -> List[Dict[str, float | str]]:
    """输入节点均值消融：将单个输入 gate 替换为输入 gate 均值。"""
    model.eval()

    base_pred, base_y, _ = _collect_preds(model, data_loader, device)
    base_mse = float(np.mean((base_pred - base_y) ** 2))

    probs = torch.sigmoid(model.input_mask_logits).detach().to(device)
    mean_val = float(probs.mean().item())

    records: List[Dict[str, float | str]] = []
    for i in range(len(probs)):
        override = probs.clone()
        override[i] = mean_val
        preds, ys, _ = _collect_preds(
            model,
            data_loader,
            device,
            input_mask_override=override,
        )
        mse = float(np.mean((preds - ys) ** 2))
        records.append(
            {
                "feature_idx": i,
                "feature": feature_names[i] if i < len(feature_names) else f"feature_{i}",
                "delta_mse": mse - base_mse,
            }
        )

    records.sort(key=lambda d: float(d["delta_mse"]), reverse=True)
    return records


def prune_circuit(
    model: ExplainableSparseTransformer,
    threshold: float = 0.5,
    input_threshold: float | None = None,
) -> Dict[str, List[torch.Tensor] | torch.Tensor]:
    """提取最小电路（关键 input / head / neuron）。"""
    return model.prune_circuit(threshold=threshold, input_threshold=input_threshold)


@torch.no_grad()
def causal_intervention_test(
    model: ExplainableSparseTransformer,
    data_loader: DataLoader,
    device: torch.device,
    feature_names: List[str],
) -> Dict[str, float]:
    """因果干预：
    1) 用均值替换降雨序列
    2) 保持总雨量不变，打乱降雨时间顺序
    观察预测变化
    """
    rain_idx = None
    for i, n in enumerate(feature_names):
        low = n.lower()
        if any(k in low for k in ["p", "prcp", "precip", "rain"]):
            rain_idx = i
            break

    if rain_idx is None:
        return {
            "baseline_mse": float("nan"),
            "replace_mean_mse": float("nan"),
            "shuffle_keep_total_mse": float("nan"),
            "delta_replace_mean": float("nan"),
            "delta_shuffle": float("nan"),
        }

    base_pred, base_y, _ = _collect_preds(model, data_loader, device)
    mean_pred, mean_y, _ = _collect_preds(model, data_loader, device, intervention="replace_mean", rainfall_index=rain_idx)
    shuf_pred, shuf_y, _ = _collect_preds(model, data_loader, device, intervention="shuffle_keep_total", rainfall_index=rain_idx)

    base_mse = float(np.mean((base_pred - base_y) ** 2))
    mean_mse = float(np.mean((mean_pred - mean_y) ** 2))
    shuf_mse = float(np.mean((shuf_pred - shuf_y) ** 2))

    return {
        "baseline_mse": base_mse,
        "replace_mean_mse": mean_mse,
        "shuffle_keep_total_mse": shuf_mse,
        "delta_replace_mean": mean_mse - base_mse,
        "delta_shuffle": shuf_mse - base_mse,
    }


@torch.no_grad()
def _collect_preds_feature_interventions(
    model: ExplainableSparseTransformer,
    data_loader: DataLoader,
    device: torch.device,
    feature_indices: List[int],
    mode: str = "replace_mean",
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, ys = [], []

    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)
        x_mod = x.clone()

        for feat_idx in feature_indices:
            series = x_mod[:, :, feat_idx]
            if mode == "replace_mean":
                mean_val = series.mean(dim=1, keepdim=True)
                x_mod[:, :, feat_idx] = mean_val
            elif mode == "shuffle_keep_total":
                bsz, seq = series.shape
                perm = torch.stack([torch.randperm(seq, device=x.device) for _ in range(bsz)], dim=0)
                x_mod[:, :, feat_idx] = series.gather(1, perm)
            else:
                raise ValueError(f"unknown intervention mode: {mode}")

        out = model(x_mod, return_intermediates=False)
        preds.append(out.pred.detach().cpu().numpy())
        ys.append(y.detach().cpu().numpy())

    return np.concatenate(preds), np.concatenate(ys)


@torch.no_grad()
def variable_interaction_test(
    model: ExplainableSparseTransformer,
    data_loader: DataLoader,
    device: torch.device,
    feature_names: List[str],
    mode: str = "replace_mean",
) -> Dict[str, object]:
    """变量交互因果分析。

    记:
      Δ_i   = MSE(do(x_i)) - MSE(base)
      Δ_ij  = MSE(do(x_i,x_j)) - MSE(base)
      I_ij  = Δ_ij - Δ_i - Δ_j

    I_ij > 0 表示协同效应（同时干预造成的损失高于线性叠加）
    I_ij < 0 表示冗余/替代效应
    """
    model.eval()

    base_pred, base_y, _ = _collect_preds(model, data_loader, device)
    base_mse = float(np.mean((base_pred - base_y) ** 2))

    n_feat = len(feature_names)
    single_delta = np.zeros(n_feat, dtype=np.float64)

    for i in range(n_feat):
        p, y = _collect_preds_feature_interventions(
            model,
            data_loader,
            device,
            feature_indices=[i],
            mode=mode,
        )
        mse = float(np.mean((p - y) ** 2))
        single_delta[i] = mse - base_mse

    pair_delta = np.zeros((n_feat, n_feat), dtype=np.float64)
    interaction = np.zeros((n_feat, n_feat), dtype=np.float64)
    pair_records: List[Dict[str, float | str]] = []

    for i in range(n_feat):
        for j in range(i + 1, n_feat):
            p, y = _collect_preds_feature_interventions(
                model,
                data_loader,
                device,
                feature_indices=[i, j],
                mode=mode,
            )
            mse = float(np.mean((p - y) ** 2))
            d_ij = mse - base_mse
            inter = d_ij - single_delta[i] - single_delta[j]

            pair_delta[i, j] = pair_delta[j, i] = d_ij
            interaction[i, j] = interaction[j, i] = inter

            pair_records.append(
                {
                    "var_i": feature_names[i],
                    "var_j": feature_names[j],
                    "delta_pair": d_ij,
                    "delta_i": float(single_delta[i]),
                    "delta_j": float(single_delta[j]),
                    "interaction": inter,
                }
            )

    pair_records.sort(key=lambda d: abs(float(d["interaction"])), reverse=True)

    return {
        "mode": mode,
        "baseline_mse": base_mse,
        "feature_names": feature_names,
        "single_delta": {feature_names[i]: float(single_delta[i]) for i in range(n_feat)},
        "pair_delta_matrix": pair_delta.tolist(),
        "interaction_matrix": interaction.tolist(),
        "pair_ranking": pair_records,
    }


def _parse_node(node_id: str) -> Optional[Tuple[str, int, int]]:
    if node_id.startswith("IN:"):
        return ("input", -1, -1)
    if not node_id.startswith("L") or "." not in node_id:
        return None
    left, right = node_id.split(".", 1)
    try:
        layer = int(left[1:])
    except Exception:
        return None

    if right.startswith("H"):
        return ("head", layer, int(right[1:]))
    if right.startswith("N"):
        return ("neuron", layer, int(right[1:]))
    return None


def parse_node_id(node_id: str) -> Optional[Tuple[str, int, int]]:
    """Public wrapper for node id parsing: L{layer}.H{idx} / L{layer}.N{idx}."""
    return _parse_node(node_id)


def _default_binary_overrides(
    model: ExplainableSparseTransformer,
    device: torch.device,
    threshold: float = 0.5,
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    input_overrides = (torch.sigmoid(model.input_mask_logits).detach().to(device) > threshold).float()
    head_overrides: Dict[int, torch.Tensor] = {}
    neuron_overrides: Dict[int, torch.Tensor] = {}
    for l_idx, layer in enumerate(model.layers):
        hp = (torch.sigmoid(layer.attn.head_mask_logits).detach().to(device) > threshold).float()
        np_ = (torch.sigmoid(layer.mlp.neuron_mask_logits).detach().to(device) > threshold).float()
        head_overrides[l_idx] = hp
        neuron_overrides[l_idx] = np_
    return input_overrides, head_overrides, neuron_overrides


def default_binary_overrides(
    model: ExplainableSparseTransformer,
    device: torch.device,
    threshold: float = 0.5,
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    return _default_binary_overrides(model, device, threshold)


def _eval_mse(
    model: ExplainableSparseTransformer,
    data_loader: DataLoader,
    device: torch.device,
    input_override: Optional[torch.Tensor] = None,
    head_overrides: Optional[Dict[int, torch.Tensor]] = None,
    neuron_overrides: Optional[Dict[int, torch.Tensor]] = None,
) -> float:
    pred, y, _ = _collect_preds(
        model,
        data_loader,
        device,
        input_mask_override=input_override,
        head_mask_overrides=head_overrides,
        neuron_mask_overrides=neuron_overrides,
    )
    return float(np.mean((pred - y) ** 2))


def build_overrides_from_active_nodes(
    model: ExplainableSparseTransformer,
    device: torch.device,
    active_node_ids: List[str],
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """从节点集合构建 hard overrides：仅保留 active_node_ids，其余置零。"""
    input_overrides = torch.zeros_like(model.input_mask_logits, device=device)
    head_overrides: Dict[int, torch.Tensor] = {}
    neuron_overrides: Dict[int, torch.Tensor] = {}

    for l_idx, layer in enumerate(model.layers):
        head_overrides[l_idx] = torch.zeros(layer.attn.n_heads, device=device)
        neuron_overrides[l_idx] = torch.zeros(layer.mlp.fc1.out_features, device=device)

    for node_id in active_node_ids:
        if node_id.startswith("IN:"):
            feat_name = node_id[3:]
            if feat_name.startswith("F") and feat_name[1:].isdigit():
                fi = int(feat_name[1:])
                if 0 <= fi < len(input_overrides):
                    input_overrides[fi] = 1.0
            continue

        parsed = _parse_node(node_id)
        if parsed is None:
            continue
        ntype, layer, idx = parsed
        if ntype == "head" and layer in head_overrides and idx < len(head_overrides[layer]):
            head_overrides[layer][idx] = 1.0
        if ntype == "neuron" and layer in neuron_overrides and idx < len(neuron_overrides[layer]):
            neuron_overrides[layer][idx] = 1.0

    return input_overrides, head_overrides, neuron_overrides


def faithfulness_k_sweep(
    model: ExplainableSparseTransformer,
    data_loader: DataLoader,
    device: torch.device,
    node_ranking: List[Dict[str, float | str]],
    k_values: List[int],
) -> List[Dict[str, float]]:
    """评估保留 Top-K 节点时的性能变化，用于最小电路 faithful 曲线。"""
    model.eval()
    base_pred, base_y, _ = _collect_preds(model, data_loader, device)
    base_mse = float(np.mean((base_pred - base_y) ** 2))
    denom = float(np.sum((base_y - base_y.mean()) ** 2) + 1e-12)
    base_nse = float(1.0 - np.sum((base_pred - base_y) ** 2) / denom)

    ranked_nodes = [str(r["node_id"]) for r in node_ranking]
    total = len(ranked_nodes)

    records: List[Dict[str, float]] = []
    for k in sorted(set([max(1, int(x)) for x in k_values])):
        kk = min(k, total)
        keep = ranked_nodes[:kk]
        in_ovr, h_ovr, n_ovr = build_overrides_from_active_nodes(model, device, keep)

        pred, y, _ = _collect_preds(
            model,
            data_loader,
            device,
            input_mask_override=in_ovr,
            head_mask_overrides=h_ovr,
            neuron_mask_overrides=n_ovr,
        )
        mse = float(np.mean((pred - y) ** 2))
        nse = float(1.0 - np.sum((pred - y) ** 2) / (float(np.sum((y - y.mean()) ** 2)) + 1e-12))

        records.append(
            {
                "k": float(kk),
                "mse": mse,
                "delta_mse": float(mse - base_mse),
                "nse": nse,
                "delta_nse": float(nse - base_nse),
            }
        )

    records.append({"k": float(total), "mse": base_mse, "delta_mse": 0.0, "nse": base_nse, "delta_nse": 0.0})
    records = sorted(records, key=lambda r: r["k"])
    return records


def edge_ablation_test(
    model: ExplainableSparseTransformer,
    data_loader: DataLoader,
    device: torch.device,
    candidate_edges: List[Tuple[str, str, float]],
    threshold: float = 0.5,
) -> List[Dict[str, float | str]]:
    """真实边消融：对每条候选边做四种干预并计算交互效应。

    对边 e=(s,t)：
      - base: 不额外消融
      - src: 仅消融源节点
      - dst: 仅消融目标节点
      - both: 同时消融源和目标

    interaction = both - src - dst + base
    """
    model.eval()

    base_inputs, base_heads, base_neurons = _default_binary_overrides(model, device, threshold=threshold)

    def clone_overrides():
        return (
            base_inputs.clone(),
            {k: v.clone() for k, v in base_heads.items()},
            {k: v.clone() for k, v in base_neurons.items()},
        )

    def ablate_node(node_id: str, inputs: torch.Tensor, heads: Dict[int, torch.Tensor], neurons: Dict[int, torch.Tensor]) -> bool:
        if node_id.startswith("IN:"):
            feat_name = node_id[3:]
            if feat_name.startswith("F") and feat_name[1:].isdigit():
                fi = int(feat_name[1:])
                if 0 <= fi < len(inputs):
                    inputs[fi] = 0.0
                    return True
            return False

        parsed = _parse_node(node_id)
        if parsed is None:
            return False
        ntype, layer, idx = parsed
        if ntype == "head":
            if layer not in heads or idx >= len(heads[layer]):
                return False
            heads[layer][idx] = 0.0
            return True
        if ntype == "neuron":
            if layer not in neurons or idx >= len(neurons[layer]):
                return False
            neurons[layer][idx] = 0.0
            return True
        return False

    cache: Dict[Tuple[str, ...], float] = {}

    def mse_with_ablations(nodes_to_ablate: List[str]) -> float:
        key = tuple(sorted(nodes_to_ablate))
        if key in cache:
            return cache[key]

        inputs, heads, neurons = clone_overrides()
        for nid in nodes_to_ablate:
            ablate_node(nid, inputs, heads, neurons)

        mse = _eval_mse(model, data_loader, device, input_override=inputs, head_overrides=heads, neuron_overrides=neurons)
        cache[key] = mse
        return mse

    base_mse = mse_with_ablations([])

    records: List[Dict[str, float | str]] = []
    for src, dst, heuristic in candidate_edges:
        src_ok = (src.startswith("IN:")) or (_parse_node(src) is not None)
        dst_ok = (dst.startswith("IN:")) or (_parse_node(dst) is not None)
        if not (src_ok and dst_ok):
            continue

        src_mse = mse_with_ablations([src])
        dst_mse = mse_with_ablations([dst])
        both_mse = mse_with_ablations([src, dst])

        interaction = float(both_mse - src_mse - dst_mse + base_mse)
        delta_both = float(both_mse - base_mse)

        records.append(
            {
                "src_node": src,
                "dst_node": dst,
                "base_mse": base_mse,
                "src_mse": src_mse,
                "dst_mse": dst_mse,
                "both_mse": both_mse,
                "delta_both": delta_both,
                "interaction": interaction,
                "heuristic_score": float(heuristic),
            }
        )

    records.sort(key=lambda r: float(r["interaction"]), reverse=True)
    return records
