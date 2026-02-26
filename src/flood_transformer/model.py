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
    """可解释 MHA：支持提取 attention，并对每个 head 加可学习门控 mask。"""

    def __init__(self, d_model: int, n_heads: int, dropout: float, enable_learnable_masks: bool = True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能整除 n_heads"
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
    """可解释 MLP：每个隐藏神经元可学习门控，并支持输出激活 Top-K 稀疏。"""

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
    """时间序列洪水预测 Transformer。

    特性：
    - 多头注意力 + 残差 + MLP
    - 权重稀疏（训练中外部调用 apply_global_weight_sparsity）
    - 激活稀疏（MLP 输出 Top-K）
    - head / neuron 可学习电路 mask
    - 导出 attention map 与 hidden states
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

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 2048, d_model) * 0.02)

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

    def forward(
        self,
        x: torch.Tensor,
        head_mask_overrides: Optional[Dict[int, torch.Tensor]] = None,
        neuron_mask_overrides: Optional[Dict[int, torch.Tensor]] = None,
        return_intermediates: bool = True,
    ) -> ModelOutputs:
        seq_len = x.shape[1]
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
        cls = x[:, -1, :]  # 用最后时间步进行预测
        pred = self.head(cls).squeeze(-1)

        return ModelOutputs(pred=pred, attention_maps=attention_maps, hidden_states=hidden_states)

    def apply_global_weight_sparsity(
        self,
        sparsity_ratio: float,
        topk_mode: str = "global",
        minimum_alive_per_neuron: int = 0,
    ):
        """对权重参数施加 Top-K 稀疏（不包含 bias / mask logits）。

        - global: 参数级全局 Top-K
        - neuronwise: 对 MLP 权重做按神经元维度 Top-K，并保证最小存活连接数
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
        """L1(mask) 正则项：对 head + neuron 的 sigmoid 概率求和。"""
        reg = torch.tensor(0.0, device=self.pos_embed.device)
        for layer in self.layers:
            reg = reg + torch.sigmoid(layer.attn.head_mask_logits).sum()
            reg = reg + torch.sigmoid(layer.mlp.neuron_mask_logits).sum()
        return reg

    def get_mask_probabilities(self):
        head_probs = []
        neuron_probs = []
        for layer in self.layers:
            head_probs.append(torch.sigmoid(layer.attn.head_mask_logits).detach().cpu())
            neuron_probs.append(torch.sigmoid(layer.mlp.neuron_mask_logits).detach().cpu())
        return head_probs, neuron_probs

    def prune_circuit(self, threshold: float = 0.5) -> Dict[str, List[torch.Tensor]]:
        """返回按阈值裁剪后的最小电路结构。"""
        heads, neurons = self.get_mask_probabilities()
        active_heads = [(h > threshold).float() for h in heads]
        active_neurons = [(n > threshold).float() for n in neurons]
        return {
            "active_heads": active_heads,
            "active_neurons": active_neurons,
        }
