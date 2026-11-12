"""
embeddings.py - Embeddings for Transformer MT Model
"""

import torch
import torch.nn as nn
import math


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class HybridEmbedding(nn.Module):
    """
    Token + positional embedding for machine translation.
    Weights are scaled as in the original Transformer paper.
    """

    def __init__(self, vocab_size, d_model=512, max_seq_len=512, dropout=0.1):
        super().__init__()

        self.d_model = d_model

        self.subword_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        nn.init.normal_(self.subword_embed.weight, mean=0, std=d_model ** -0.5)
        with torch.no_grad():
            self.subword_embed.weight[0].fill_(0)

        # Positional encoding (includes dropout internally)
        self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)

    def forward(self, token_ids):
        """
        Args:
            token_ids: [B, L]
        Returns:
            embedded: [B, L, d_model]
        """
        emb = self.subword_embed(token_ids)
        emb = emb * math.sqrt(self.d_model)
        return self.pos_encoder(emb)
