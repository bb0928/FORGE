"""FORGE transformer encoder, strictly ported from FORGE transformer_v2.py.

Components (LLaMA-style):
- RoPEPositionalEncoding: rotary position embedding
- RMSNorm: RMS normalization (pre-norm)
- SwiGLU: gated FFN
- TransformerBlock: pre-norm MHA(RoPE) + SwiGLU, both residual
- TransformerEncoder: CLS-token style, prepends learnable empty_embedding,
  returns the CLS token output after N blocks + final norm.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RoPEPositionalEncoding(nn.Module):
    def __init__(self, dim, max_seq_len=1000, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        # x: (batch, seq_len, dim)
        batch_size, seq_len, _ = x.shape
        position = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        freqs = torch.outer(position, self.inv_freq)  # (seq_len, dim//2)
        cos_freqs = torch.cos(freqs)
        sin_freqs = torch.sin(freqs)
        return self.apply_rope(x, cos_freqs, sin_freqs)

    def apply_rope(self, x, cos_freqs, sin_freqs):
        batch_size, seq_len, dim = x.shape
        x_even = x[:, :, 0::2]
        x_odd = x[:, :, 1::2]
        cos_freqs = cos_freqs.unsqueeze(0)
        sin_freqs = sin_freqs.unsqueeze(0)
        x_even_rotated = x_even * cos_freqs - x_odd * sin_freqs
        x_odd_rotated = x_even * sin_freqs + x_odd * cos_freqs
        x_rotated = torch.zeros_like(x)
        x_rotated[:, :, 0::2] = x_even_rotated
        x_rotated[:, :, 1::2] = x_odd_rotated
        return x_rotated


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.mean(x * x, dim=-1, keepdim=True)
        rms = torch.rsqrt(rms + self.eps)
        return self.scale * x * rms


class SwiGLU(nn.Module):
    """SwiGLU(x) = SiLU(xW) ⊗ (xV), projected back to input_dim."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.w = nn.Linear(input_dim, hidden_dim, bias=False)
        self.v = nn.Linear(input_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, input_dim, bias=False)
        self.silu = nn.SiLU()

    def forward(self, x):
        swish_gate = self.silu(self.w(x))
        gated = swish_gate * self.v(x)
        return self.output(gated)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.feed_forward = SwiGLU(embed_dim, ff_dim)
        self.rmsnorm1 = RMSNorm(embed_dim)
        self.rmsnorm2 = RMSNorm(embed_dim)
        self.rope = RoPEPositionalEncoding(embed_dim)

    def forward(self, x):
        x_norm = self.rmsnorm1(x)
        x_rope = self.rope(x_norm)
        # query/key use RoPE-encoded, value uses normalized (matches FORGE)
        attn_output, _ = self.attention(x_rope, x_rope, x_norm)
        x = x + attn_output
        x_norm2 = self.rmsnorm2(x)
        ff_output = self.feed_forward(x_norm2)
        x = x + ff_output
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers):
        super().__init__()
        self.embed_dim = embed_dim
        # CLS-token style learnable embedding for action prediction
        self.empty_embedding = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)]
        )
        self.final_norm = RMSNorm(embed_dim)

    def forward(self, x):
        # x: (batch, T, embed_dim)
        batch_size = x.shape[0]
        empty_emb = self.empty_embedding.expand(batch_size, -1, -1)  # (B, 1, embed)
        x = torch.cat((empty_emb, x), dim=1)  # (B, 1+T, embed)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.final_norm(x)
        return x[:, 0, :]  # CLS token output (B, embed)
