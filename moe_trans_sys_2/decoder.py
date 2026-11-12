"""
decoder.py - Pure Transformer Decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from moe import SwiGLU, MoEFeedForward
from encoder import use_moe_in_layer


class MultiHeadAttention(nn.Module):
    """Multi-head attention (self or cross)"""

    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0

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
            nn.init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, query, key=None, value=None, mask=None):
        """
        Args:
            query:  [B, T, d_model]
            key:    [B, S, d_model]  (defaults to query for self-attention)
            value:  [B, S, d_model]  (defaults to key)
            mask:   attention mask  (True = keep)
        Returns:
            output:       [B, T, d_model]
            attn_weights: [B, T, S]  (averaged over heads)
        """
        if key   is None: key   = query
        if value is None: value = key

        B, T, D = query.size()
        S = key.size(1)

        def _split(x_in, linear):
            out = linear(x_in).view(B, -1, self.nhead, self.d_k)
            return out.transpose(1, 2)

        Q = _split(query, self.q_linear)
        K = _split(key,   self.k_linear)
        V = _split(value, self.v_linear)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(B, T, D)

        return self.out_proj(context), attn.mean(dim=1)   # [B,T,D], [B,T,S]


class DecoderLayer(nn.Module):
    """Single Transformer decoder layer (pre-norm). FFN is dense SwiGLU or sparse MoE."""

    def __init__(self, d_model, nhead, d_ff, dropout=0.1, moe=None):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, nhead, dropout)
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.is_moe     = moe is not None
        self.ffn        = (MoEFeedForward(d_model, d_ff, dropout=dropout, **moe)
                           if self.is_moe else SwiGLU(d_model, d_ff, dropout))

        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.norm3   = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None):
        # Masked self-attention
        attn_out, _ = self.self_attn(self.norm1(x), mask=tgt_mask)
        x = x + self.dropout(attn_out)

        # Cross-attention
        cross_out, _ = self.cross_attn(
            self.norm2(x), encoder_output, encoder_output, mask=src_mask
        )
        x = x + self.dropout(cross_out)

        # FFN (dense or MoE)
        h = self.norm3(x)
        if self.is_moe:
            # real-token mask = diagonal of the [B,1,T,T] padding&causal mask
            token_mask = (tgt_mask[:, 0].diagonal(dim1=-2, dim2=-1)
                          if tgt_mask is not None else None)           # [B, T]
            h = self.ffn(h, token_mask)
        else:
            h = self.ffn(h)
        x = x + self.dropout(h)

        return x


class TransformerDecoder(nn.Module):
    """Stack of Transformer decoder layers"""

    def __init__(self, d_model=512, nhead=8, d_ff=2048, num_layers=6, dropout=0.1,
                 moe_every=0, moe_cfg=None):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, nhead, d_ff, dropout,
                         moe=moe_cfg if use_moe_in_layer(i, moe_every) else None)
            for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None):
        """
        Args:
            x:              [B, T, d_model]
            encoder_output: [B, S, d_model]
            tgt_mask:       [B, 1, T, T]
            src_mask:       [B, 1, 1, S]
        Returns:
            output: [B, T, d_model]
        """
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, src_mask)
        return self.norm(x)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
