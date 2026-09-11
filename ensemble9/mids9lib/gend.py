"""GenD-style hyperspherical regularisers.

GenD's central finding is that strong open-world generalisation comes from disciplining the
*geometry* of the L2-normalised feature space rather than from heavy backbone tuning:
same-authenticity embeddings are pulled together (alignment) while the whole set is spread over
the unit hypersphere (uniformity).  We apply these to the image embedding MIDS already extracts
(the projected CLS token), supervised by the per-image authenticity label.  This complements the
Effort SVD adapters: SVD keeps the *weights* faithful, GenD keeps the *features* generalisable.

Both losses follow Wang & Isola (2020) and mirror the validated ``hybrid_dfd`` implementation.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def l2_normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.normalize(x, p=2, dim=dim)


def alignment_loss(embeddings: torch.Tensor, labels: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    """Mean ``||z_i - z_j||^alpha`` over same-label pairs (embeddings assumed L2-normalised)."""
    labels_equal = labels[:, None].eq(labels[None, :]).triu(diagonal=1)
    pairs = labels_equal.nonzero(as_tuple=False)
    if pairs.numel() == 0:
        return torch.zeros((), device=embeddings.device)
    return (embeddings[pairs[:, 0]] - embeddings[pairs[:, 1]]).norm(p=2, dim=1).pow(alpha).mean()


def uniformity_loss(embeddings: torch.Tensor, t: float = 2.0) -> torch.Tensor:
    """``log E[exp(-t ||z_i - z_j||^2)]`` over all pairs (can be negative)."""
    if embeddings.shape[0] < 2:
        return torch.zeros((), device=embeddings.device)
    return torch.pdist(embeddings, p=2).pow(2).mul(-t).exp().mean().clamp_min(1e-6).log()
