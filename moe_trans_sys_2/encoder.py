"""
encoder.py - Pure Transformer Encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from moe import SwiGLU, MoEFeedForward


def use_moe_in_layer(layer_idx, moe_every):
    """0-based layer index -> True for every `moe_every`-th layer (2nd, 4th, 6th, ...)."""
    return moe_every > 0 and (layer_idx + 1) % moe_every == 0


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention"""

    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"

        self.d_model = d_model
        self.nhead   = nhead
        self.d_k     = d_model // nhead

        self.q_linear  = nn.Linear(d_model, d_model, bias=False)
        self.k_linear  = nn.Linear(d_model, d_model, bias=False)
        self.v_linear  = nn.Linear(d_model, d_model, bias=False)
        self.out_proj  = nn.Linear(d_model, d_model, bias=False)
        self.dropout   = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for m in [self.q_linear, self.k_linear, self.v_linear]:
            nn.init.xavier_uniform_(m.weight, gain=1.0)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x, mask=None):
        """
        Args:
            x:    [B, L, d_model]
            mask: [B, 1, 1, L]  (True = keep, False = mask out)
        Returns:
            output: [B, L, d_model]
        """
        B, L, D = x.size()

        def _split_heads(x_in, linear):
            out = linear(x_in).view(B, -1, self.nhead, self.d_k)
            return out.transpose(1, 2)          # [B, nhead, L, d_k]

        Q = _split_heads(x, self.q_linear)
        K = _split_heads(x, self.k_linear)
        V = _split_heads(x, self.v_linear)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(B, L, D)

        output = self.out_proj(context)
        output = self.dropout(output)
        return output


class EncoderLayer(nn.Module):
    """Single Transformer encoder layer (pre-norm). FFN is dense SwiGLU or sparse MoE."""

    def __init__(self, d_model, nhead, d_ff, dropout=0.1, moe=None):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.is_moe    = moe is not None
        self.ffn       = (MoEFeedForward(d_model, d_ff, dropout=dropout, **moe)
                          if self.is_moe else SwiGLU(d_model, d_ff, dropout))
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # Self-attention (pre-norm + residual)
        x = x + self.self_attn(self.norm1(x), mask)
        # FFN (pre-norm + residual)
        h = self.norm2(x)
        if self.is_moe:
            token_mask = mask[:, 0, 0, :] if mask is not None else None   # [B, L]
            x = x + self.ffn(h, token_mask)
        else:
            x = x + self.ffn(h)
        return x


class TransformerEncoder(nn.Module):
    """Stack of Transformer encoder layers"""

    def __init__(self, d_model=512, nhead=8, d_ff=2048, num_layers=6, dropout=0.1,
                 moe_every=0, moe_cfg=None):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, nhead, d_ff, dropout,
                         moe=moe_cfg if use_moe_in_layer(i, moe_every) else None)
            for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        Args:
            x:    [B, L, d_model]
            mask: [B, 1, 1, L]
        Returns:
            [B, L, d_model]
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
