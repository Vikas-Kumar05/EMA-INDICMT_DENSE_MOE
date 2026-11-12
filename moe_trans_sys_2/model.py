"""
model.py - Pure Transformer Translation Model
"""

import torch
import torch.nn as nn
from embeddings import HybridEmbedding
from encoder import TransformerEncoder
from decoder import TransformerDecoder
from moe import MoEFeedForward


def _padding_mask(seq, pad_id=0):
    """[B, L] → [B, 1, 1, L]  (True = valid token)"""
    return (seq != pad_id).unsqueeze(1).unsqueeze(2)


def _causal_mask(size, device):
    """Lower-triangular boolean mask [size, size]"""
    return torch.tril(torch.ones(size, size, device=device, dtype=torch.bool))


class TransformerModel(nn.Module):
    """
    Encoder-decoder Transformer for sequence-to-sequence translation.

    Architecture
    ============
    - Shared token embeddings (src & tgt) with sinusoidal positional encoding
    - TransformerEncoder  (N × pre-norm attention + SwiGLU FFN)
    - TransformerDecoder  (N × masked self-attention + cross-attention + SwiGLU FFN)
    - Output projection tied to embedding weights

    Optional sparse Mixture-of-Experts (moe_every > 0)
    ==================================================
    Every `moe_every`-th layer (2nd, 4th, 6th ... when moe_every=2) of BOTH the encoder
    and the decoder swaps its dense SwiGLU FFN for `moe_num_experts` SwiGLU experts with a
    top-`moe_top_k` router.  moe_every=0 gives the plain baseline Transformer (and a
    state_dict identical to the original code).
    """

    def __init__(
        self,
        vocab_size,
        d_model=512,
        nhead=8,
        d_ff=2048,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dropout=0.1,
        max_seq_len=512,
        moe_every=0,
        moe_num_experts=4,
        moe_top_k=2,
        moe_jitter=0.0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model    = d_model

        # saved inside every checkpoint so compute_metrics.py can rebuild the model
        self.config = dict(
            vocab_size=vocab_size, d_model=d_model, nhead=nhead, d_ff=d_ff,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout, max_seq_len=max_seq_len,
            moe_every=moe_every, moe_num_experts=moe_num_experts,
            moe_top_k=moe_top_k, moe_jitter=moe_jitter,
        )
        moe_cfg = (dict(num_experts=moe_num_experts, top_k=moe_top_k, jitter=moe_jitter)
                   if moe_every > 0 else None)

        # Shared embedding for both source and target
        self.embedding = HybridEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

        self.encoder = TransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            d_ff=d_ff,
            num_layers=num_encoder_layers,
            dropout=dropout,
            moe_every=moe_every,
            moe_cfg=moe_cfg,
        )

        self.decoder = TransformerDecoder(
            d_model=d_model,
            nhead=nhead,
            d_ff=d_ff,
            num_layers=num_decoder_layers,
            dropout=dropout,
            moe_every=moe_every,
            moe_cfg=moe_cfg,
        )

        # Output projection tied with embedding weights
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        self.output_proj.weight = self.embedding.subword_embed.weight

        self._print_summary()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Args:
            src:      [B, S]  source token IDs
            tgt:      [B, T]  target token IDs (teacher-forced input)
            src_mask: [B, 1, 1, S]  (auto-generated if None)
            tgt_mask: [B, 1, T, T] (auto-generated if None)
        Returns:
            logits: [B, T, vocab_size]
        """
        device = src.device
        T = tgt.size(1)

        if src_mask is None:
            src_mask = _padding_mask(src).to(device)

        if tgt_mask is None:
            pad_mask   = _padding_mask(tgt).to(device).expand(-1, -1, T, -1)
            causal     = _causal_mask(T, device).unsqueeze(0).unsqueeze(0)
            tgt_mask   = pad_mask & causal

        src_emb = self.embedding(src)
        tgt_emb = self.embedding(tgt)

        enc_out = self.encoder(src_emb, src_mask)
        dec_out = self.decoder(tgt_emb, enc_out, tgt_mask, src_mask)

        return self.output_proj(dec_out)

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def encode(self, src, src_mask=None):
        """Encode a source sequence. Used during autoregressive decoding."""
        if src_mask is None:
            src_mask = _padding_mask(src).to(src.device)
        return self.encoder(self.embedding(src), src_mask)

    def decode_step(self, tgt, encoder_output, src_mask=None):
        """
        One autoregressive decoder step.

        Args:
            tgt:            [B, T] tokens generated so far (including BOS)
            encoder_output: [B, S, d_model]
            src_mask:       [B, 1, 1, S]
        Returns:
            logits: [B, T, vocab_size]
        """
        device = tgt.device
        T = tgt.size(1)

        pad_mask = _padding_mask(tgt).to(device).expand(-1, -1, T, -1)
        causal   = _causal_mask(T, device).unsqueeze(0).unsqueeze(0)
        tgt_mask = pad_mask & causal

        dec_out = self.decoder(self.embedding(tgt), encoder_output, tgt_mask, src_mask)
        return self.output_proj(dec_out)

    # ------------------------------------------------------------------
    # Mixture-of-Experts helpers
    # ------------------------------------------------------------------

    def moe_layers(self):
        return [m for m in self.modules() if isinstance(m, MoEFeedForward)]

    @property
    def moe_enabled(self):
        return len(self.moe_layers()) > 0

    def get_moe_losses(self):
        """
        (load-balance loss, router z-loss) averaged over all MoE layers, from the most
        recent TRAINING forward pass.  Returns (None, None) for a dense model / eval mode.
        """
        aux = [m.aux_loss for m in self.moe_layers() if m.aux_loss is not None]
        if not aux:
            return None, None
        z = [m.z_loss for m in self.moe_layers() if m.z_loss is not None]
        return torch.stack(aux).mean(), torch.stack(z).mean()

    def pop_expert_usage(self):
        """
        Share of routed token-slots per expert, one row per MoE layer
        (list of lists summing to 1.0); resets the counters.
        """
        rows = []
        for m in self.moe_layers():
            tot = m.usage.sum().item()
            rows.append((m.usage / tot).tolist() if tot > 0 else [0.0] * m.num_experts)
            m.usage.zero_()
        return rows

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _print_summary(self):
        enc_p   = self.encoder.get_num_params()
        dec_p   = self.decoder.get_num_params()
        emb_p   = sum(p.numel() for p in self.embedding.parameters())
        total_p = self.get_num_params()

        print(f"\n{'='*60}")
        print(f"  Transformer Translation Model")
        print(f"{'='*60}")
        print(f"  d_model     : {self.d_model}")
        print(f"  vocab_size  : {self.vocab_size}")
        print(f"  Embeddings  : {emb_p:,}")
        print(f"  Encoder     : {enc_p:,}")
        print(f"  Decoder     : {dec_p:,}")
        print(f"  Total       : {total_p:,}")
        moe = self.moe_layers()
        if moe:
            cfg = self.config
            exp_p   = sum(p.numel() for m in moe for p in m.experts.parameters())
            active  = total_p - exp_p * (1 - cfg['moe_top_k'] / cfg['moe_num_experts'])
            print(f"  MoE         : {len(moe)} layers x {cfg['moe_num_experts']} experts, "
                  f"top-{cfg['moe_top_k']}, every {cfg['moe_every']} layers")
            print(f"  Active/token: ~{int(active):,} params")
        else:
            print(f"  MoE         : off (dense baseline)")
        print(f"  Size (FP32) : ~{total_p * 4 / 1e6:.1f} MB")
        print(f"{'='*60}\n")
