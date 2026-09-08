"""Attention building blocks.

These are kept byte-for-byte compatible with FFAA's ``mids/basic_module.py`` so that the
fusion stack behaves identically to upstream MIDS and so that an upstream MIDS checkpoint
(6 fusion tokens, no artifact branch) can be loaded into ``mids_original`` mode for a fair
baseline comparison.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SimpleResBlock(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(p=0.5),
            nn.Linear(input_dim, input_dim),
        )
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.mlp(x))


class MultiHeadCrossAttention(nn.Module):
    """Cross attention: ``query=x``, ``key=value=y``; post-LayerNorm (no residual)."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.cross_attn(query=x, key=y, value=y)
        return self.norm(attn_output)


class MultiHeadSelfAttention(nn.Module):
    """Self attention with residual + LayerNorm."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.cross_attn(query=x, key=x, value=x)
        return self.norm(attn_output + x)
