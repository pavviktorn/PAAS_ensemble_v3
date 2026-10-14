"""Geometric Semantic Decoupling -- semantic-subspace estimation (GSD paper, arXiv 2603.09242).

Given the frozen-VFM guide features of a mini-batch, build the K-dim *semantic subspace* basis U:

    c   = (1/B) Σ_i g_i                 # semantic anchor (batch centroid)
    G   = [g_1-c, ..., g_B-c] ∈ R^{D×B}  # centered guide matrix (columns)
    G   = Q R                            # Householder-based QR (numerically stable)
    U   = Q[:, :K] ∈ R^{D×K}             # first K orthonormal columns = semantic basis

`torch.linalg.qr` is exactly the Householder-based QR the paper specifies (LAPACK geqrf), so we use
it directly. U is estimated from the *frozen* guide features, so it carries no gradient and is
re-estimated every batch -- there are no running statistics to update at train or test time.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def semantic_basis(guide: torch.Tensor, k: int, method: str = "householder") -> torch.Tensor:
    """guide: (B, D) frozen guide vectors. Returns U: (D, k') with k' = min(k, D, rank-ish).

    method: 'householder' (paper default, via torch.linalg.qr) or 'svd' (variance-ordered columns).
    """
    if guide.dim() != 2:
        raise ValueError(f"guide must be (B, D); got {tuple(guide.shape)}")
    B, D = guide.shape
    g = guide.float()
    c = g.mean(dim=0, keepdim=True)                 # (1, D) semantic anchor
    G = (g - c).t().contiguous()                    # (D, B) centered columns
    k = int(max(1, min(k, D, B)))
    if B < 2:
        # a single (or zero) centered vector is degenerate (G == 0) -> no semantic subspace.
        return guide.new_zeros((D, k))
    if method == "svd":
        # left singular vectors are ordered by captured variance (top-k dominant directions)
        U = torch.linalg.svd(G, full_matrices=False).U[:, :k]
    else:
        # Householder QR: columns of Q form an orthonormal basis of span(G) = the semantic subspace
        Q, _ = torch.linalg.qr(G, mode="reduced")   # Q: (D, min(D, B))
        U = Q[:, :k]
    return U.to(guide.dtype)
