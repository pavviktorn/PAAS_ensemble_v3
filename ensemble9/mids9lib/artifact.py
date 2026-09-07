"""ForensicsAdapter-style local artifact branch (mask-free, weakly supervised).

Upstream MIDS reasons only over a global CLS token + image<->answer cross-attention.  On the
"hard" cases that MIDS exists to resolve (the candidate answers disagree), it helps to have a
*text-independent* visual forgery signal.  ForensicsAdapter shows that a small set of learnable
forgery query tokens, cross-attending to the encoder's patch tokens, localise manipulation
artifacts well.  We adopt that mechanism but operate directly on CLIP patch tokens (no second
ViT) to keep MIDS compact.

No masks are available in this dataset, so the artifact map is trained with weak image-level
MIL: for fake images the top-k artifact patches must score high; real images are suppressed.
A single pooled "artifact evidence" token is injected as an extra token into MIDS's fusion stack.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic_module import MultiHeadCrossAttention

REAL = 0  # image-authenticity convention: 0 = real, 1 = fake


class LocalArtifactHead(nn.Module):
    """Learnable forgery queries attend to projected CLIP patch tokens.

    Produces (a) a pooled artifact-evidence token for fusion, (b) a per-patch artifact map for
    weak MIL supervision, and (c) per-query artifact logits for the query-diversity regulariser.
    """

    def __init__(self, dim: int = 768, num_queries: int = 4, num_heads: int = 8) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.cross_attn = MultiHeadCrossAttention(dim, num_heads)
        self.query_proj = nn.Linear(dim, dim)
        self.patch_proj = nn.Linear(dim, dim)
        self.token_norm = nn.LayerNorm(dim)
        self.scale = dim ** -0.5

    def forward(self, patch_tokens: torch.Tensor):
        """``patch_tokens``: ``(B, P, dim)`` -- CLIP patch tokens already projected to ``dim``."""
        b, p, d = patch_tokens.shape
        queries = self.query.unsqueeze(0).expand(b, -1, -1)        # (B, Q, dim)
        attended = self.cross_attn(queries, patch_tokens)          # (B, Q, dim)

        qd = self.query_proj(attended)                             # (B, Q, dim)
        pd = self.patch_proj(patch_tokens)                         # (B, P, dim)
        artifact_query_logits = torch.einsum("bqd,bpd->bqp", qd, pd) * self.scale  # (B, Q, P)
        artifact_map = artifact_query_logits.max(dim=1).values     # (B, P) -- union of queries

        artifact_token = self.token_norm(attended.mean(dim=1))     # (B, dim)
        return artifact_token, artifact_map, artifact_query_logits


def weak_artifact_discovery_loss(
    artifact_logits: torch.Tensor,
    labels: torch.Tensor,
    topk_fraction: float = 0.15,
    real_weight: float = 0.5,
    sparsity_weight: float = 0.05,
) -> torch.Tensor:
    """Image-level MIL on the per-patch artifact map.

    ``labels`` is the per-image authenticity (0 real / 1 fake).  Top-k patch scores must agree
    with the image label; real images are pushed to zero; fake images are kept sparse so the
    head localises rather than flooding the whole map.
    """
    scores = artifact_logits.float().flatten(1)
    topk = max(1, min(scores.shape[1], int(round(scores.shape[1] * float(topk_fraction)))))
    image_logits = scores.topk(topk, dim=1).values.mean(dim=1)
    targets = (labels != REAL).float()
    loss = F.binary_cross_entropy_with_logits(image_logits, targets)

    real_index = (labels == REAL).nonzero(as_tuple=True)[0]
    if real_index.numel() > 0:
        real_targets = torch.zeros_like(scores[real_index])
        loss = loss + real_weight * F.binary_cross_entropy_with_logits(scores[real_index], real_targets)

    fake_index = (labels != REAL).nonzero(as_tuple=True)[0]
    if fake_index.numel() > 0 and sparsity_weight > 0:
        loss = loss + sparsity_weight * scores[fake_index].sigmoid().mean()
    return loss


def query_orthogonality_loss(artifact_query_logits: torch.Tensor) -> torch.Tensor:
    """Discourage the forgery queries from collapsing onto the same patches.

    Penalises the off-diagonal cosine similarity between the per-query artifact maps.
    """
    b, q, p = artifact_query_logits.shape
    if q < 2:
        return torch.zeros((), device=artifact_query_logits.device)
    maps = F.normalize(artifact_query_logits, p=2, dim=2)          # (B, Q, P)
    gram = torch.einsum("bqp,bkp->bqk", maps, maps)                # (B, Q, Q)
    eye = torch.eye(q, device=gram.device).unsqueeze(0)
    off = gram * (1.0 - eye)
    return off.abs().sum(dim=(1, 2)).div(q * (q - 1)).mean()
