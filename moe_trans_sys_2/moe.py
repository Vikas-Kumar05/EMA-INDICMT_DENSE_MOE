"""
moe.py - SwiGLU feed-forward block and Sparse Mixture-of-Experts (MoE) block


"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU feed-forward activation (shared by the dense FFN and every expert)."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w  = nn.Linear(d_model, d_ff, bias=False)
        self.v  = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.silu(self.w(x)) * self.v(x)))


class MoEFeedForward(nn.Module):
    """Sparse top-k Mixture-of-Experts replacement for the dense SwiGLU FFN."""

    def __init__(self, d_model, d_ff, num_experts=4, top_k=2, dropout=0.1, jitter=0.0):
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError(f"top_k ({top_k}) must be in [1, num_experts={num_experts}]")
        self.d_model     = d_model
        self.num_experts = num_experts
        self.top_k       = top_k
        self.jitter      = jitter

        self.router  = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLU(d_model, d_ff, dropout) for _ in range(num_experts)])

        # running count of routed token-slots per expert (for logging only, not saved)
        self.register_buffer('usage', torch.zeros(num_experts), persistent=False)
        self.aux_loss = None
        self.z_loss   = None

    def forward(self, x, token_mask=None):
        """
        x          : [B, L, d_model]
        token_mask : [B, L] bool, True = real token (None = all real)
        """
        B, L, D = x.shape
        N, E, K = B * L, self.num_experts, self.top_k
        flat  = x.reshape(N, D)
        valid = (token_mask.reshape(N).bool() if token_mask is not None
                 else torch.ones(N, dtype=torch.bool, device=x.device))

        router_in = flat
        if self.training and self.jitter > 0:
            router_in = flat * torch.empty_like(flat).uniform_(1 - self.jitter, 1 + self.jitter)

        logits = self.router(router_in).float()                # [N, E]  (fp32 for stability)
        probs  = F.softmax(logits, dim=-1)
        top_p, top_i = probs.topk(K, dim=-1)                   # [N, K]
        top_p  = top_p / top_p.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        vf       = valid.float()
        n_valid  = vf.sum().clamp_min(1.0)
        dispatch = F.one_hot(top_i, E).sum(dim=1).float()      # [N, E]  1 if expert chosen

        if self.training:
            f = (dispatch * vf[:, None]).sum(0) / (n_valid * K)     # [E] share of slots
            P = (probs    * vf[:, None]).sum(0) / n_valid           # [E] mean probability
            self.aux_loss = E * (f.detach() * P).sum()
            self.z_loss   = (torch.logsumexp(logits, dim=-1).pow(2) * vf).sum() / n_valid
            with torch.no_grad():
                self.usage += (dispatch * vf[:, None]).sum(0)

        out = flat.new_zeros(N, D)
        for e, expert in enumerate(self.experts):
            chosen = (top_i == e)                                   # [N, K]
            rows   = (chosen.any(dim=-1) & valid).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            w = (top_p * chosen.float()).sum(dim=-1)[rows]          # [R] gate weight
            y = expert(flat[rows])
            out.index_add_(0, rows, (y * w[:, None].to(y.dtype)).to(out.dtype))
        return out.reshape(B, L, D)

    def extra_repr(self):
        return f"experts={self.num_experts}, top_k={self.top_k}, jitter={self.jitter}"
