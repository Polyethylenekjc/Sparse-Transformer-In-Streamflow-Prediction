from __future__ import annotations

import math
from typing import Optional

import torch


def compute_topk_threshold(weight: torch.Tensor, sparsity_ratio: float) -> Optional[torch.Tensor]:
    """Compute magnitude threshold from target sparsity ratio.

    Definition: keep ratio = (1 - sparsity_ratio); threshold is the k-th largest absolute value, where
    k = max(1, int((1 - sparsity_ratio) * N)), where N is the total number of elements.
    """
    if sparsity_ratio <= 0 or sparsity_ratio >= 1:
        return None

    flat = weight.reshape(-1).abs()
    total = flat.numel()
    k = max(1, int((1.0 - sparsity_ratio) * total))
    if k >= total:
        return None

    topk_vals, _ = torch.topk(flat, k=k, largest=True, sorted=True)
    return topk_vals[-1]


def update_sparsity(
    step: int,
    total_steps: int,
    target_sparsity: float,
    warmup_ratio: float = 0.2,
    mode: str = "cosine",
    exponent: int = 2,
) -> float:
    """Anneal sparsity ratio by training step.

    Design:
    - Keep dense during warmup (sparsity ratio = 0)
    - Then use cosine annealing to smoothly reach target_sparsity
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
    """Apply Top-K sparsity to any tensor: keep K largest absolute values, zero out the rest."""
    if sparsity_ratio <= 0:
        return weight
    if sparsity_ratio >= 1:
        return torch.zeros_like(weight)

    threshold = compute_topk_threshold(weight, sparsity_ratio)
    if threshold is None:
        return weight

    mask = (weight.abs() >= threshold).to(weight.dtype)
    return weight * mask


def ste_binary_mask(mask_logits: torch.Tensor) -> torch.Tensor:
    """Sigmoid + Straight-Through Estimator.

    Use hard binarization in forward pass and keep sigmoid gradient in backward pass.
    """
    probs = torch.sigmoid(mask_logits)
    hard = (probs >= 0.5).float()
    return hard + probs - probs.detach()


def topk_activation(x: torch.Tensor, sparsity_ratio: float) -> torch.Tensor:
    """Apply per-sample Top-K activation sparsity on the last dimension while preserving batch dimension."""
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
    """Apply Top-K sparsity along neuron dimension.

    Assume weight is 2D; neuron_dim indicates which axis indexes neurons:
    - neuron_dim=0: each row is one neuron
    - neuron_dim=1: each column is one neuron

    This function follows the core idea of circuit_sparsity:
    - global target sparsity ratio
    - can additionally enforce at least minimum_alive_per_neuron connections per neuron
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
