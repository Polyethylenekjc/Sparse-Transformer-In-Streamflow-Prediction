from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW

try:
    from rich.console import Console
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
except Exception:
    Console = None
    Progress = None

from .config import ExperimentConfig
from .data import DataBundle, load_data
from .explain import (
    causal_intervention_test,
    edge_ablation_test,
    extract_hidden_states,
    get_attention_statistics,
    head_ablation_test,
    linear_probe,
    mean_ablation,
    prune_circuit,
    variable_interaction_test,
    faithfulness_k_sweep,
)
from .model import ExplainableSparseTransformer
from .sparsity import update_sparsity
from .visualize import (
    save_attention_statistics,
    save_circuit_graph,
    save_head_ranking,
    save_neuron_ranking,
)


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _get_console():
    if Console is None:
        return None
    return Console()


def _task_loss(task_type: str, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if task_type == "classification":
        return nn.BCEWithLogitsLoss()(pred, y)
    return nn.MSELoss()(pred, y)


@torch.no_grad()
def _evaluate(model: ExplainableSparseTransformer, loader, device: torch.device, task_type: str) -> float:
    model.eval()
    losses = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        out = model(x, return_intermediates=False)
        loss = _task_loss(task_type, out.pred, y)
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def _collect_preds_targets(
    model: ExplainableSparseTransformer,
    loader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, ys = [], []
    for x, y in loader:
        x = x.to(device)
        out = model(x, return_intermediates=False)
        preds.append(out.pred.detach().cpu().numpy())
        ys.append(y.numpy())
    if not preds:
        return np.array([]), np.array([])
    return np.concatenate(preds), np.concatenate(ys)


def _regression_metrics(pred: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    if len(pred) == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "nse": float("nan")}

    err = pred - y
    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))

    denom = float(np.sum((y - y.mean()) ** 2))
    nse = float(1.0 - np.sum(err**2) / (denom + 1e-12))
    return {"rmse": rmse, "mae": mae, "nse": nse}


def _train_phase(
    model: ExplainableSparseTransformer,
    loader,
    val_loader,
    optimizer,
    device: torch.device,
    task_type: str,
    total_steps: int,
    start_step: int,
    apply_weight_sparse: bool,
    target_weight_sparsity: float,
    warmup_ratio: float,
    sparsity_anneal_mode: str,
    sparsity_anneal_exponent: int,
    weight_topk_mode: str,
    minimum_alive_per_neuron: int,
    lambda_mask_l1: float,
    phase_name: str,
    epoch_idx: int,
    total_epochs: int,
    debug: bool,
    log_every: int,
    progress: Any = None,
    task_id: int | None = None,
    console: Any = None,
) -> Tuple[int, List[Dict[str, float]]]:
    step = start_step
    history = []

    model.train()
    for batch_idx, (x, y) in enumerate(loader, start=1):
        step += 1
        x = x.to(device)
        y = y.to(device)

        if apply_weight_sparse:
            curr_sparsity = update_sparsity(
                step=step,
                total_steps=total_steps,
                target_sparsity=target_weight_sparsity,
                warmup_ratio=warmup_ratio,
                mode=sparsity_anneal_mode,
                exponent=sparsity_anneal_exponent,
            )
            model.apply_global_weight_sparsity(
                curr_sparsity,
                topk_mode=weight_topk_mode,
                minimum_alive_per_neuron=minimum_alive_per_neuron,
            )
        else:
            curr_sparsity = 0.0

        out = model(x, return_intermediates=False)
        task_loss = _task_loss(task_type, out.pred, y)

        mask_loss = model.mask_regularization() * lambda_mask_l1 if lambda_mask_l1 > 0 else torch.tensor(0.0, device=device)
        loss = task_loss + mask_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history.append(
            {
                "step": step,
                "loss": float(loss.item()),
                "task_loss": float(task_loss.item()),
                "mask_loss": float(mask_loss.item()),
                "weight_sparsity": float(curr_sparsity),
                "weight_topk_mode": weight_topk_mode,
            }
        )

        if progress is not None and task_id is not None:
            progress.advance(task_id, 1)

        if debug and (batch_idx == 1 or batch_idx % max(1, log_every) == 0 or batch_idx == len(loader)):
            msg = (
                f"[{phase_name}] epoch {epoch_idx}/{total_epochs} "
                f"batch {batch_idx}/{len(loader)} step {step} "
                f"loss={float(loss.item()):.6f} task={float(task_loss.item()):.6f} "
                f"mask={float(mask_loss.item()):.6f} sparsity={float(curr_sparsity):.4f}"
            )
            if console is not None:
                console.log(msg)
            else:
                print(msg)

    val_loss = _evaluate(model, val_loader, device, task_type)
    history.append({"step": step, "val_loss": val_loss})
    if debug:
        msg = f"[{phase_name}] epoch {epoch_idx}/{total_epochs} val_loss={val_loss:.6f}"
        if console is not None:
            console.log(msg)
        else:
            print(msg)
    return step, history


def _collect_physical_variables(test_loader, feature_names: List[str]) -> Dict[str, np.ndarray]:
    x_all = []
    for x, _ in test_loader:
        x_all.append(x.numpy())
    x_all = np.concatenate(x_all, axis=0)  # [N, L, F]

    rain_idx = 0
    soil_idx = None
    for i, f in enumerate(feature_names):
        low = f.lower()
        if any(k in low for k in ["p", "rain", "prcp", "precip"]):
            rain_idx = i
        if "soil" in low:
            soil_idx = i

    rain = x_all[:, :, rain_idx]
    cum_rain = rain.sum(axis=1)

    if soil_idx is not None:
        api = x_all[:, :, soil_idx].mean(axis=1)
    else:
        # 退化版 API：指数衰减累计降雨
        L = rain.shape[1]
        decay = np.exp(-np.arange(L)[::-1] / 10.0)
        api = (rain * decay[None, :]).sum(axis=1)

    dpdt = np.diff(rain, axis=1).mean(axis=1)

    return {
        "cum_rain": cum_rain,
        "api": api,
        "dpdt": dpdt,
    }


@torch.no_grad()
def _collect_test_arrays(model: ExplainableSparseTransformer, loader, device: torch.device):
    model.eval()
    xs, ys, ps = [], [], []
    for x, y in loader:
        out = model(x.to(device), return_intermediates=False)
        xs.append(x.numpy())
        ys.append(y.numpy())
        ps.append(out.pred.detach().cpu().numpy())
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ps)


def _build_node_ranking(
    head_importance: List[Dict[str, float]],
    neuron_importance: List[Dict[str, float]],
) -> List[Dict[str, float | str]]:
    rows: List[Dict[str, float | str]] = []
    for r in head_importance:
        node_id = f"L{int(r['layer'])}.H{int(r['head'])}"
        rows.append(
            {
                "node_id": node_id,
                "node_type": "head",
                "layer": int(r["layer"]),
                "index": int(r["head"]),
                "importance": float(max(0.0, r.get("delta_mse", 0.0))),
            }
        )

    for r in neuron_importance:
        node_id = f"L{int(r['layer'])}.N{int(r['index'])}"
        rows.append(
            {
                "node_id": node_id,
                "node_type": "neuron",
                "layer": int(r["layer"]),
                "index": int(r["index"]),
                "importance": float(max(0.0, r.get("delta_mse", 0.0))),
            }
        )

    rows.sort(key=lambda d: float(d["importance"]), reverse=True)
    return rows


def _extract_cherry_samples(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    p_arr: np.ndarray,
    feature_names: List[str],
    n_each: int = 3,
) -> Dict[str, object]:
    hi_idx = np.argsort(y_arr)[-n_each:][::-1].tolist()
    lo_idx = np.argsort(y_arr)[:n_each].tolist()

    def _pack(idx: int) -> Dict[str, object]:
        x = x_arr[idx]  # [L, F]
        feature_series = {feature_names[f]: x[:, f].tolist() for f in range(min(x.shape[1], len(feature_names)))}
        return {
            "sample_index": int(idx),
            "target": float(y_arr[idx]),
            "pred": float(p_arr[idx]),
            "error": float(p_arr[idx] - y_arr[idx]),
            "feature_series": feature_series,
        }

    return {
        "high_flow_samples": [_pack(i) for i in hi_idx],
        "low_flow_samples": [_pack(i) for i in lo_idx],
    }


def _build_node_importance_maps(
    head_importance: List[Dict[str, float]],
    neuron_importance: List[Dict[str, float]],
) -> Dict[str, float]:
    node_imp: Dict[str, float] = {}
    for r in head_importance:
        key = f"L{int(r['layer'])}.H{int(r['head'])}"
        node_imp[key] = max(node_imp.get(key, 0.0), max(0.0, float(r.get("delta_mse", 0.0))))
    for r in neuron_importance:
        key = f"L{int(r['layer'])}.N{int(r['index'])}"
        node_imp[key] = max(node_imp.get(key, 0.0), max(0.0, float(r.get("delta_mse", 0.0))))

    m = max(node_imp.values()) if node_imp else 0.0
    if m > 0:
        for k in list(node_imp.keys()):
            node_imp[k] /= m
    return node_imp


def _build_candidate_edges(
    circuit: Dict[str, List[torch.Tensor]],
    node_imp: Dict[str, float],
    top_k: int,
) -> List[Tuple[str, str, float]]:
    active_heads = circuit["active_heads"]
    active_neurons = circuit["active_neurons"]

    candidates: List[Tuple[str, str, float]] = []

    # 同层: head -> neuron
    for l_idx, (h_mask, n_mask) in enumerate(zip(active_heads, active_neurons)):
        h_idx = np.where(np.asarray(h_mask.cpu()) > 0)[0].tolist()
        n_idx = np.where(np.asarray(n_mask.cpu()) > 0)[0].tolist()
        for h in h_idx:
            s = f"L{l_idx}.H{h}"
            for n in n_idx:
                t = f"L{l_idx}.N{n}"
                score = float(np.sqrt(max(0.0, node_imp.get(s, 0.0) * node_imp.get(t, 0.0))))
                candidates.append((s, t, score))

    # 跨层: neuron(l) -> head(l+1)
    for l_idx in range(len(active_heads) - 1):
        n_idx = np.where(np.asarray(active_neurons[l_idx].cpu()) > 0)[0].tolist()
        h_next = np.where(np.asarray(active_heads[l_idx + 1].cpu()) > 0)[0].tolist()
        for n in n_idx:
            s = f"L{l_idx}.N{n}"
            for h in h_next:
                t = f"L{l_idx + 1}.H{h}"
                score = float(np.sqrt(max(0.0, node_imp.get(s, 0.0) * node_imp.get(t, 0.0))))
                candidates.append((s, t, score))

    candidates.sort(key=lambda x: x[2], reverse=True)

    dedup = []
    seen = set()
    for s, t, sc in candidates:
        if (s, t) in seen:
            continue
        seen.add((s, t))
        dedup.append((s, t, sc))
        if len(dedup) >= max(1, top_k):
            break
    return dedup


def run_full_experiment(config: ExperimentConfig) -> Dict[str, str]:
    """五阶段流程：
    1) 训练 dense Transformer
    2) 加入权重稀疏退火
    3) 训练可学习 mask（电路学习）
    4) 提取最小电路
    5) 因果干预测试
    """
    os.makedirs(config.output_dir, exist_ok=True)
    _set_seed(config.seed)

    device = torch.device(config.device)
    data: DataBundle = load_data(config)

    model = ExplainableSparseTransformer(
        input_dim=len(data.feature_names),
        d_model=config.d_model,
        n_heads=config.n_heads,
        num_layers=config.num_layers,
        d_ff=config.d_ff,
        dropout=config.dropout,
        task_type=config.task_type,
        enable_activation_sparsity=config.enable_activation_sparsity,
        activation_sparsity_ratio=config.target_activation_sparsity,
        enable_learnable_masks=config.enable_learnable_masks,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    total_steps = (config.epochs_dense + config.epochs_sparse + config.epochs_mask) * len(data.train_loader)
    global_step = 0
    logs = []

    console = _get_console() if config.use_rich_progress else None
    use_rich = (Progress is not None) and config.use_rich_progress

    if config.debug:
        boot_msg = (
            f"Training start | device={config.device} | train_batches={len(data.train_loader)} "
            f"val_batches={len(data.val_loader)} test_batches={len(data.test_loader)}"
        )
        if console is not None:
            console.log(boot_msg)
        else:
            print(boot_msg)

    def _run_epochs(phase_name: str, n_epochs: int, apply_sparse: bool, lambda_l1: float):
        nonlocal global_step, logs
        if n_epochs <= 0:
            return

        if use_rich:
            with Progress(
                TextColumn("[bold cyan]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                for e in range(1, n_epochs + 1):
                    task_id = progress.add_task(f"{phase_name} epoch {e}/{n_epochs}", total=len(data.train_loader))
                    global_step, hist = _train_phase(
                        model=model,
                        loader=data.train_loader,
                        val_loader=data.val_loader,
                        optimizer=optimizer,
                        device=device,
                        task_type=config.task_type,
                        total_steps=total_steps,
                        start_step=global_step,
                        apply_weight_sparse=apply_sparse,
                        target_weight_sparsity=config.target_weight_sparsity if apply_sparse else 0.0,
                        warmup_ratio=config.sparsity_warmup_ratio,
                        sparsity_anneal_mode=config.sparsity_anneal_mode,
                        sparsity_anneal_exponent=config.sparsity_anneal_exponent,
                        weight_topk_mode=config.weight_topk_mode,
                        minimum_alive_per_neuron=config.minimum_alive_per_neuron,
                        lambda_mask_l1=lambda_l1,
                        phase_name=phase_name,
                        epoch_idx=e,
                        total_epochs=n_epochs,
                        debug=config.debug,
                        log_every=config.log_every,
                        progress=progress,
                        task_id=task_id,
                        console=console,
                    )
                    logs.extend(hist)
        else:
            for e in range(1, n_epochs + 1):
                global_step, hist = _train_phase(
                    model=model,
                    loader=data.train_loader,
                    val_loader=data.val_loader,
                    optimizer=optimizer,
                    device=device,
                    task_type=config.task_type,
                    total_steps=total_steps,
                    start_step=global_step,
                    apply_weight_sparse=apply_sparse,
                    target_weight_sparsity=config.target_weight_sparsity if apply_sparse else 0.0,
                    warmup_ratio=config.sparsity_warmup_ratio,
                    sparsity_anneal_mode=config.sparsity_anneal_mode,
                    sparsity_anneal_exponent=config.sparsity_anneal_exponent,
                    weight_topk_mode=config.weight_topk_mode,
                    minimum_alive_per_neuron=config.minimum_alive_per_neuron,
                    lambda_mask_l1=lambda_l1,
                    phase_name=phase_name,
                    epoch_idx=e,
                    total_epochs=n_epochs,
                    debug=config.debug,
                    log_every=config.log_every,
                    progress=None,
                    task_id=None,
                    console=console,
                )
                logs.extend(hist)

    _run_epochs("dense", config.epochs_dense, apply_sparse=False, lambda_l1=0.0)
    _run_epochs("sparse", config.epochs_sparse, apply_sparse=config.enable_weight_sparsity, lambda_l1=0.0)
    _run_epochs("mask", config.epochs_mask, apply_sparse=config.enable_weight_sparsity, lambda_l1=config.lambda_mask_l1)

    # 基础评估
    test_loss = _evaluate(model, data.test_loader, device, config.task_type)
    metrics = {}
    if config.task_type == "regression":
        pred_test, y_test = _collect_preds_targets(model, data.test_loader, device)
        metrics = _regression_metrics(pred_test, y_test)
        metrics["nse_threshold"] = float(config.nse_threshold)
        metrics["nse_pass"] = bool(metrics["nse"] >= config.nse_threshold)

        if config.debug:
            msg = (
                f"[eval] RMSE={metrics['rmse']:.6f} MAE={metrics['mae']:.6f} "
                f"NSE={metrics['nse']:.6f} threshold={config.nse_threshold:.3f} pass={metrics['nse_pass']}"
            )
            if console is not None:
                console.log(msg)
            else:
                print(msg)

    # 4) 提取电路 + 消融
    circuit = prune_circuit(model, threshold=config.circuit_threshold)
    head_importance = head_ablation_test(model, data.test_loader, device)
    neuron_importance = mean_ablation(model, data.test_loader, device, ablate="neuron")

    node_imp = _build_node_importance_maps(head_importance, neuron_importance)
    candidate_edges = _build_candidate_edges(
        circuit,
        node_imp=node_imp,
        top_k=config.edge_ablation_topk,
    )
    edge_importance = edge_ablation_test(
        model,
        data.test_loader,
        device,
        candidate_edges=candidate_edges,
        threshold=config.circuit_threshold,
    )

    node_ranking = _build_node_ranking(head_importance, neuron_importance)
    k_values = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
    faithfulness = faithfulness_k_sweep(
        model,
        data.test_loader,
        device,
        node_ranking=node_ranking,
        k_values=k_values,
    )

    # 5) 可解释性接口
    attn_stats = get_attention_statistics(model, data.test_loader, device)
    hidden_states = extract_hidden_states(model, data.test_loader, device)
    phys = _collect_physical_variables(data.test_loader, data.feature_names)

    probe_results = {
        "cum_rain": linear_probe(hidden_states[-1], phys["cum_rain"]),
        "api": linear_probe(hidden_states[-1], phys["api"]),
        "dpdt": linear_probe(hidden_states[-1], phys["dpdt"]),
    }

    causal = causal_intervention_test(model, data.test_loader, device, data.feature_names)
    var_interaction = variable_interaction_test(
        model,
        data.test_loader,
        device,
        feature_names=data.feature_names,
        mode="replace_mean",
    )

    # 输出结果
    log_path = os.path.join(config.output_dir, "train_log.csv")
    pd.DataFrame(logs).to_csv(log_path, index=False)

    head_rank_path = save_head_ranking(head_importance, config.output_dir)
    neuron_rank_path = save_neuron_ranking(neuron_importance, config.output_dir)

    edge_rank_path = os.path.join(config.output_dir, "edge_importance_ranking.csv")
    pd.DataFrame(edge_importance).to_csv(edge_rank_path, index=False)

    node_rank_path = os.path.join(config.output_dir, "node_importance_ranking.csv")
    pd.DataFrame(node_ranking).to_csv(node_rank_path, index=False)

    faith_path = os.path.join(config.output_dir, "faithfulness_curve.csv")
    pd.DataFrame(faithfulness).to_csv(faith_path, index=False)

    attn_fig_path = save_attention_statistics(attn_stats, config.output_dir)
    circuit_fig_path = save_circuit_graph(circuit, config.output_dir)

    circuit_json_path = os.path.join(config.output_dir, "circuit_structure.json")
    with open(circuit_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "active_heads": [x.tolist() for x in circuit["active_heads"]],
                "active_neurons": [x.tolist() for x in circuit["active_neurons"]],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    attn_json_path = os.path.join(config.output_dir, "attention_statistics.json")
    with open(attn_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "flood_threshold_y": float(attn_stats["flood_threshold_y"]),
                "flood_attention": np.asarray(attn_stats["flood_attention"]).tolist(),
                "normal_attention": np.asarray(attn_stats["normal_attention"]).tolist(),
                "attention_diff": np.asarray(attn_stats["attention_diff"]).tolist(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    probe_path = os.path.join(config.output_dir, "physical_probe_results.json")
    with open(probe_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                k: {"r2": float(v["r2"]), "bias": float(v["bias"])}
                for k, v in probe_results.items()
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    causal_path = os.path.join(config.output_dir, "causal_intervention_results.json")
    with open(causal_path, "w", encoding="utf-8") as f:
        json.dump(causal, f, ensure_ascii=False, indent=2)

    var_interaction_json = os.path.join(config.output_dir, "variable_interactions.json")
    with open(var_interaction_json, "w", encoding="utf-8") as f:
        json.dump(var_interaction, f, ensure_ascii=False, indent=2)

    var_interaction_csv = os.path.join(config.output_dir, "variable_interactions_ranking.csv")
    pd.DataFrame(var_interaction.get("pair_ranking", [])).to_csv(var_interaction_csv, index=False)

    model_path = os.path.join(config.output_dir, "sparse_transformer.pt")
    torch.save(model.state_dict(), model_path)

    x_arr, y_arr, p_arr = _collect_test_arrays(model, data.test_loader, device)
    cherry = _extract_cherry_samples(x_arr, y_arr, p_arr, data.feature_names, n_each=3)
    cherry_path = os.path.join(config.output_dir, "cherry_samples.json")
    with open(cherry_path, "w", encoding="utf-8") as f:
        json.dump(cherry, f, ensure_ascii=False, indent=2)

    summary_path = os.path.join(config.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "test_loss": test_loss,
                "metrics": metrics,
                "flood_threshold": data.flood_threshold,
                "feature_names": data.feature_names,
                "data_source": data.data_source,
                "outputs": {
                    "log": log_path,
                    "head_ranking": head_rank_path,
                    "neuron_ranking": neuron_rank_path,
                    "edge_ranking": edge_rank_path,
                                        "node_ranking": node_rank_path,
                                        "faithfulness": faith_path,
                    "attention_fig": attn_fig_path,
                    "circuit_fig": circuit_fig_path,
                    "probe": probe_path,
                    "causal": causal_path,
                    "variable_interaction": var_interaction_json,
                    "variable_interaction_ranking": var_interaction_csv,
                    "circuit_json": circuit_json_path,
                    "attention_json": attn_json_path,
                                        "cherry_samples": cherry_path,
                        "node_ranking": node_rank_path,
                        "faithfulness": faith_path,
                    "model": model_path,
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "summary": summary_path,
        "model": model_path,
        "head_ranking": head_rank_path,
        "neuron_ranking": neuron_rank_path,
        "edge_ranking": edge_rank_path,
        "circuit_fig": circuit_fig_path,
        "attention_fig": attn_fig_path,
        "probe": probe_path,
        "causal": causal_path,
        "variable_interaction": var_interaction_json,
        "variable_interaction_ranking": var_interaction_csv,
        "circuit_json": circuit_json_path,
        "attention_json": attn_json_path,
        "cherry_samples": cherry_path,
    }
