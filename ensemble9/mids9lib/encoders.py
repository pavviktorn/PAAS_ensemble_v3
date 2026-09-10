"""Encoder factories + lightweight stand-in encoders for testing.

Real mode builds the same encoders upstream MIDS uses: a frozen ``T5EncoderModel`` (text) and a
``CLIPVisionModel`` (image).  Stub mode builds tiny randomly-initialised modules that expose the
*same module structure and output contract* (``image.vision_model.encoder.layers`` with
``self_attn.out_proj`` linears; ``text(...).last_hidden_state``; ``image(...).hidden_states``).

The stubs let the full pipeline -- including SVD layer replacement -- run on CPU in seconds with
no multi-GB downloads, which is what the smoke test exercises.  Never enable stubs for training.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------------------------
# Real encoders
# ----------------------------------------------------------------------------------------------
def build_text_encoder(cfg) -> nn.Module:
    if cfg.stub_encoders:
        return StubTextEncoder(dim=cfg.dim, vocab=cfg.stub_text_vocab)
    from transformers import T5EncoderModel

    return T5EncoderModel.from_pretrained(cfg.text_model_path)


def build_image_encoder(cfg) -> nn.Module:
    if cfg.stub_encoders:
        return StubImageEncoder(hidden=1024, num_layers=cfg.stub_image_layers)
    from transformers import CLIPVisionModel

    return CLIPVisionModel.from_pretrained(cfg.image_model_path)


def image_hidden_size(encoder: nn.Module) -> int:
    if hasattr(encoder, "config") and hasattr(encoder.config, "hidden_size"):
        return int(encoder.config.hidden_size)
    raise AttributeError("Image encoder does not expose config.hidden_size")


# ----------------------------------------------------------------------------------------------
# Stub text encoder
# ----------------------------------------------------------------------------------------------
class StubTextEncoder(nn.Module):
    """Embedding + 1 linear; returns ``.last_hidden_state`` of shape ``(B, L, dim)``."""

    def __init__(self, dim: int = 768, vocab: int = 256) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab, dim)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.config = SimpleNamespace(hidden_size=dim, vocab_size=vocab)

    def forward(self, input_ids=None, attention_mask=None, **_):
        x = self.embedding(input_ids.clamp_min(0))
        x = self.norm(self.proj(F.gelu(x)))
        return SimpleNamespace(last_hidden_state=x)


# ----------------------------------------------------------------------------------------------
# Stub image encoder (HF-CLIP-shaped so SVD replacement and hidden_states[-1]/[-2] work)
# ----------------------------------------------------------------------------------------------
class _StubAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8) -> None:
        super().__init__()
        self.num_heads = heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        h = self.num_heads
        q = self.q_proj(x).view(b, n, h, d // h).transpose(1, 2)
        k = self.k_proj(x).view(b, n, h, d // h).transpose(1, 2)
        v = self.v_proj(x).view(b, n, h, d // h).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v)
        o = o.transpose(1, 2).reshape(b, n, d)
        return self.out_proj(o)


class _StubMLP(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class _StubBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 8) -> None:
        super().__init__()
        self.self_attn = _StubAttention(dim, heads)
        self.mlp = _StubMLP(dim)
        self.layer_norm1 = nn.LayerNorm(dim)
        self.layer_norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x


class _StubEncoder(nn.Module):
    def __init__(self, dim: int, num_layers: int, heads: int = 8) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_StubBlock(dim, heads) for _ in range(num_layers)])


class _StubVisionModel(nn.Module):
    def __init__(self, hidden: int, num_layers: int, grid: int = 14, patch: int = 16) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(3, hidden, kernel_size=patch, stride=patch)
        self.cls = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, grid * grid + 1, hidden) * 0.02)
        self.encoder = _StubEncoder(hidden, num_layers)
        self.post_layernorm = nn.LayerNorm(hidden)
        self.grid = grid

    def forward(self, pixel_values: torch.Tensor, output_hidden_states: bool = True):
        x = self.patch_embed(pixel_values)               # (B, hidden, g, g)
        x = x.flatten(2).transpose(1, 2)                  # (B, P, hidden)
        cls = self.cls.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos[:, : x.size(1) + 1]
        hidden_states = [x]
        for layer in self.encoder.layers:
            x = layer(x)
            hidden_states.append(x)
        return SimpleNamespace(hidden_states=tuple(hidden_states), last_hidden_state=x)


class StubImageEncoder(nn.Module):
    def __init__(self, hidden: int = 1024, num_layers: int = 4) -> None:
        super().__init__()
        self.vision_model = _StubVisionModel(hidden, num_layers)
        self.config = SimpleNamespace(hidden_size=hidden)

    def forward(self, pixel_values: torch.Tensor, output_hidden_states: bool = True):
        return self.vision_model(pixel_values, output_hidden_states=output_hidden_states)
