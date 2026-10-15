"""GSD projection: remove the semantic subspace from detector features (orthogonal complement).

    F'_l = F_l (I - U U^T) = F_l - (F_l U) U^T

`F_l` are the TRAINABLE detector's patch-token features (gradients flow through them); `U` is the
frozen semantic basis (detached). This is the "hard structural constraint" that forces the detector
to learn in the semantic null-space -- no auxiliary disentanglement loss needed.
"""
from __future__ import annotations

import torch


def desemanticize(feats: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """feats: (..., D) (any leading dims, e.g. (B, N, D)); U: (D, K). Returns same shape as feats,
    with the span(U) component removed. Mean is NOT re-added (the paper removes span(U) directly)."""
    if U is None or U.numel() == 0:
        return feats
    Uf = U.to(feats.dtype)
    coeff = feats @ Uf                 # (..., K) coordinates along the semantic basis
    return feats - coeff @ Uf.t()      # subtract the semantic component
