from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sparsity import apply_topk_sparsity, apply_topk_sparsity_neuronwise, ste_binary_mask, topk_activation


@dataclass
class ModelOutputs:
    pred: torch.Tensor
    attention_maps: List[torch.Tensor]
    hidden_states: List[torch.Tensor]


class SparseMultiHeadSelfAttention(nn.Module):
    """Explainable MHA: exposes attention and applies learnable gating masks per head."""

    def __init__(self, d_model: int, n_heads: int, dropout: float, enable_learnable_masks: bool = True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.enable_learnable_masks = enable_learnable_masks

        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.head_mask_logits = nn.Parameter(torch.zeros(n_heads))

    def _get_head_gate(self, override_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if override_mask is not None:
            return override_mask
        if self.enable_learnable_masks:
            return ste_binary_mask(self.head_mask_logits)
        return torch.ones_like(self.head_mask_logits)

    def forward(self, x: torch.Tensor, head_mask_override: Optional[torch.Tensor] = None):
        bsz, seq_len, _ = x.shape

        qkv = self.qkv(x)
        qkv = qkv.view(bsz, seq_len, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, L, Dh]

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, v)  # [B, H, L, Dh]
        head_gate = self._get_head_gate(head_mask_override).view(1, self.n_heads, 1, 1)
        context = context * head_gate

        context = context.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        out = self.out_proj(context)
        return out, attn


class SparseMLP(nn.Module):
    """Explainable MLP: learnable gate per hidden neuron with Top-K output activation sparsity."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float,
        activation_sparsity_ratio: float = 0.0,
        enable_activation_sparsity: bool = False,
        enable_learnable_masks: bool = True,
    ):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        self.enable_activation_sparsity = enable_activation_sparsity
        self.activation_sparsity_ratio = activation_sparsity_ratio
        self.enable_learnable_masks = enable_learnable_masks

        self.neuron_mask_logits = nn.Parameter(torch.zeros(d_ff))

    def _get_neuron_gate(self, override_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if override_mask is not None:
            return override_mask
        if self.enable_learnable_masks:
            return ste_binary_mask(self.neuron_mask_logits)
        return torch.ones_like(self.neuron_mask_logits)

    def forward(self, x: torch.Tensor, neuron_mask_override: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden = F.gelu(self.fc1(x))
        neuron_gate = self._get_neuron_gate(neuron_mask_override).view(1, 1, -1)
        hidden = hidden * neuron_gate

        out = self.fc2(hidden)
        out = self.dropout(out)

        if self.enable_activation_sparsity:
            out = topk_activation(out, self.activation_sparsity_ratio)
        return out


class SparseTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        activation_sparsity_ratio: float,
        enable_activation_sparsity: bool,
        enable_learnable_masks: bool,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.attn = SparseMultiHeadSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            enable_learnable_masks=enable_learnable_masks,
        )
        self.mlp = SparseMLP(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation_sparsity_ratio=activation_sparsity_ratio,
            enable_activation_sparsity=enable_activation_sparsity,
            enable_learnable_masks=enable_learnable_masks,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        head_mask_override: Optional[torch.Tensor] = None,
        neuron_mask_override: Optional[torch.Tensor] = None,
    ):
        residual = x
        attn_out, attn_map = self.attn(self.ln1(x), head_mask_override=head_mask_override)
        x = residual + self.dropout(attn_out)

        residual = x
        mlp_out = self.mlp(self.ln2(x), neuron_mask_override=neuron_mask_override)
        x = residual + self.dropout(mlp_out)
        return x, attn_map


class ExplainableSparseTransformer(nn.Module):
    """Time-series transformer for flood prediction.

    Features:
    - Multi-head attention + residual + MLP
    - Weight sparsity (apply_global_weight_sparsity is called externally during training)
    - Activation sparsity (MLP Top-K output)
    - input/head/neuron learnable circuit masks
    - Exports attention maps and hidden states
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        d_ff: int,
        dropout: float,
        task_type: str = "regression",
        enable_activation_sparsity: bool = False,
        activation_sparsity_ratio: float = 0.0,
        enable_learnable_masks: bool = True,
    ):
        super().__init__()
        self.task_type = task_type
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.enable_learnable_masks = enable_learnable_masks

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 2048, d_model) * 0.02)
        self.input_mask_logits = nn.Parameter(torch.zeros(input_dim))

        self.layers = nn.ModuleList(
            [
                SparseTransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation_sparsity_ratio=activation_sparsity_ratio,
                    enable_activation_sparsity=enable_activation_sparsity,
                    enable_learnable_masks=enable_learnable_masks,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def _get_input_gate(self, override_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if override_mask is not None:
            return override_mask
        if self.enable_learnable_masks:
            return ste_binary_mask(self.input_mask_logits)
        return torch.ones_like(self.input_mask_logits)

    def forward(
        self,
        x: torch.Tensor,
        input_mask_override: Optional[torch.Tensor] = None,
        head_mask_overrides: Optional[Dict[int, torch.Tensor]] = None,
        neuron_mask_overrides: Optional[Dict[int, torch.Tensor]] = None,
        return_intermediates: bool = True,
    ) -> ModelOutputs:
        seq_len = x.shape[1]
        input_gate = self._get_input_gate(input_mask_override).view(1, 1, -1)
        x = x * input_gate
        x = self.input_proj(x) + self.pos_embed[:, :seq_len, :]

        attention_maps: List[torch.Tensor] = []
        hidden_states: List[torch.Tensor] = [x]

        for layer_idx, layer in enumerate(self.layers):
            head_ovr = None if head_mask_overrides is None else head_mask_overrides.get(layer_idx)
            neuron_ovr = None if neuron_mask_overrides is None else neuron_mask_overrides.get(layer_idx)
            x, attn = layer(x, head_mask_override=head_ovr, neuron_mask_override=neuron_ovr)
            if return_intermediates:
                attention_maps.append(attn)
                hidden_states.append(x)

        x = self.final_ln(x)
        cls = x[:, -1, :]  # Use the last timestep for prediction.
        pred = self.head(cls).squeeze(-1)

        return ModelOutputs(pred=pred, attention_maps=attention_maps, hidden_states=hidden_states)

    def apply_global_weight_sparsity(
        self,
        sparsity_ratio: float,
        topk_mode: str = "global",
        minimum_alive_per_neuron: int = 0,
    ):
        """Apply Top-K sparsity to weight parameters (excluding bias and mask logits).

        - global: global Top-K across parameters
        - neuronwise: neuron-dimension Top-K for MLP weights, with a minimum number of alive connections
        """
        with torch.no_grad():
            for name, param in self.named_parameters():
                if "bias" in name or "mask_logits" in name or "pos_embed" in name:
                    continue
                if param.dim() < 2:
                    continue

                if topk_mode == "neuronwise" and "layers" in name and "mlp" in name and param.dim() == 2:
                    if name.endswith("mlp.fc1.weight"):
                        sparse_param = apply_topk_sparsity_neuronwise(
                            param,
                            sparsity_ratio=sparsity_ratio,
                            neuron_dim=0,
                            minimum_alive_per_neuron=minimum_alive_per_neuron,
                        )
                    elif name.endswith("mlp.fc2.weight"):
                        sparse_param = apply_topk_sparsity_neuronwise(
                            param,
                            sparsity_ratio=sparsity_ratio,
                            neuron_dim=1,
                            minimum_alive_per_neuron=minimum_alive_per_neuron,
                        )
                    else:
                        sparse_param = apply_topk_sparsity(param, sparsity_ratio)
                else:
                    sparse_param = apply_topk_sparsity(param, sparsity_ratio)
                param.copy_(sparse_param)

    def mask_regularization(self) -> torch.Tensor:
        """L1(mask) regularizer: sum sigmoid probabilities of head and neuron masks."""
        reg = torch.tensor(0.0, device=self.pos_embed.device)
        reg = reg + torch.sigmoid(self.input_mask_logits).sum()
        for layer in self.layers:
            reg = reg + torch.sigmoid(layer.attn.head_mask_logits).sum()
            reg = reg + torch.sigmoid(layer.mlp.neuron_mask_logits).sum()
        return reg

    def get_mask_probabilities(self):
        input_probs = torch.sigmoid(self.input_mask_logits).detach().cpu()
        head_probs = []
        neuron_probs = []
        for layer in self.layers:
            head_probs.append(torch.sigmoid(layer.attn.head_mask_logits).detach().cpu())
            neuron_probs.append(torch.sigmoid(layer.mlp.neuron_mask_logits).detach().cpu())
        return input_probs, head_probs, neuron_probs

    def prune_circuit(self, threshold: float = 0.5, input_threshold: Optional[float] = None) -> Dict[str, List[torch.Tensor]]:
        """Return the threshold-pruned minimal circuit structure."""
        input_probs, heads, neurons = self.get_mask_probabilities()
        in_th = threshold if input_threshold is None else input_threshold
        active_inputs = (input_probs > in_th).float()
        active_heads = [(h > threshold).float() for h in heads]
        active_neurons = [(n > threshold).float() for n in neurons]
        return {
            "active_inputs": active_inputs,
            "active_heads": active_heads,
            "active_neurons": active_neurons,
        }
