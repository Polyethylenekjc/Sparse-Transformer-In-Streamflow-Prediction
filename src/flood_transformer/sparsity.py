from __future__ import annotations

import math
from typing import Optional

import torch


def update_sparsity(
    step: int,
    total_steps: int,
    target_sparsity: float,
    warmup_ratio: float = 0.2,
    mode: str = "cosine",
    exponent: int = 2,
) -> float:
    """按训练步数退火稀疏率。

    设计：
    - 先 warmup 保持 dense（稀疏率为 0）
    - 后续采用余弦退火平滑过渡到 target_sparsity
    """
    if total_steps <= 0:
        return target_sparsity

    step = max(0, min(step, total_steps))
    warmup_steps = int(total_steps * warmup_ratio)
    if step <= warmup_steps:
        return 0.0

    progress = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
    progress = float(max(0.0, min(1.0, progress)))

    if mode == "linear":
        scale = progress
    elif mode == "power_law":
        scale = progress**max(1, int(exponent))
    else:
        scale = 0.5 * (1.0 - math.cos(math.pi * progress))

    return float(target_sparsity * scale)


def apply_topk_sparsity(weight: torch.Tensor, sparsity_ratio: float) -> torch.Tensor:
    """对任意张量施加 Top-K 稀疏：保留绝对值最大的 K 个元素，其余置零。"""
    if sparsity_ratio <= 0:
        return weight
    if sparsity_ratio >= 1:
        return torch.zeros_like(weight)

    flat = weight.reshape(-1)
    total = flat.numel()
    k = max(1, int((1.0 - sparsity_ratio) * total))

    if k >= total:
        return weight

    _, topk_idx = torch.topk(flat.abs(), k=k, largest=True, sorted=False)
    mask = torch.zeros_like(flat)
    mask[topk_idx] = 1.0
    mask = mask.reshape_as(weight)
    return weight * mask


def ste_binary_mask(mask_logits: torch.Tensor) -> torch.Tensor:
    """Sigmoid + Straight-Through Estimator.

    前向使用 hard 二值化，反向保留 sigmoid 梯度。
    """
    probs = torch.sigmoid(mask_logits)
    hard = (probs >= 0.5).float()
    return hard + probs - probs.detach()


def topk_activation(x: torch.Tensor, sparsity_ratio: float) -> torch.Tensor:
    """在 batch 维保持不变的前提下，对最后一维做逐样本 Top-K 激活稀疏。"""
    if sparsity_ratio <= 0:
        return x
    if sparsity_ratio >= 1:
        return torch.zeros_like(x)

    last_dim = x.shape[-1]
    k = max(1, int((1.0 - sparsity_ratio) * last_dim))
    if k >= last_dim:
        return x

    values, _ = torch.topk(x.abs(), k=k, dim=-1, largest=True, sorted=True)
    thresh = values[..., -1:].expand_as(x)
    mask = (x.abs() >= thresh).float()
    return x * mask


def apply_topk_sparsity_neuronwise(
    weight: torch.Tensor,
    sparsity_ratio: float,
    neuron_dim: int,
    minimum_alive_per_neuron: int = 0,
) -> torch.Tensor:
    """按神经元维度做 Top-K 稀疏。

    设 weight 为二维，neuron_dim 表示每个神经元所在维度：
    - neuron_dim=0: 每一行是一个神经元
    - neuron_dim=1: 每一列是一个神经元

    该函数模仿 circuit_sparsity 的核心思路：
    - 全局目标稀疏率
    - 可额外保证每个神经元至少保留 minimum_alive_per_neuron 条连接
    """
    if weight.dim() != 2:
        return apply_topk_sparsity(weight, sparsity_ratio)
    if sparsity_ratio <= 0:
        return weight
    if sparsity_ratio >= 1:
        return torch.zeros_like(weight)

    total = weight.numel()
    keep_total = max(1, int((1.0 - sparsity_ratio) * total))

    flat = weight.abs().flatten()
    forced_keep_mask = torch.zeros_like(flat, dtype=torch.bool)

    if minimum_alive_per_neuron > 0:
        if neuron_dim == 0:
            n_neurons = weight.shape[0]
            per_len = weight.shape[1]
            max_alive = min(per_len, minimum_alive_per_neuron)
            for r in range(n_neurons):
                row = weight[r].abs()
                _, idx = torch.topk(row, k=max_alive, largest=True, sorted=False)
                flat_idx = r * per_len + idx
                forced_keep_mask[flat_idx] = True
        else:
            n_neurons = weight.shape[1]
            per_len = weight.shape[0]
            max_alive = min(per_len, minimum_alive_per_neuron)
            for c in range(n_neurons):
                col = weight[:, c].abs()
                _, idx = torch.topk(col, k=max_alive, largest=True, sorted=False)
                flat_idx = idx * weight.shape[1] + c
                forced_keep_mask[flat_idx] = True

    forced_count = int(forced_keep_mask.sum().item())
    remaining = max(0, keep_total - forced_count)

    score = flat.clone()
    score[forced_keep_mask] = -1.0

    keep_mask = forced_keep_mask.clone()
    if remaining > 0:
        _, idx = torch.topk(score, k=remaining, largest=True, sorted=False)
        keep_mask[idx] = True

    return weight * keep_mask.reshape_as(weight).to(weight.dtype)
