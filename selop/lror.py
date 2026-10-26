"""LROR — Low-rank Orthogonal Removal of spurious correlation.

Faithful to SeLop (arXiv:2601.11915v2, "Low-rank Orthogonal Subspace
Intervention for Generalizable Face Forgery Detection").

A learnable skinny matrix M in R^{D x r} parameterises an estimated low-rank
*spurious* subspace.  Its orthonormal basis Q = QR(M) (Q^T Q = I_r) gives the
projector P = Q Q^T.  The visual tokens are mapped to the orthogonal complement
of that subspace:

    Z_s = X_vis Q Q^T            (Eq. 4, spurious part)
    Z_c = X_vis - Z_s            (Eq. 5, causal complement = X_vis (I - Q Q^T))

The [CLS] token (index 0) is *excluded* from the projection at every intervened
layer and passes through untouched.  Orthogonality is structural (guaranteed by
QR), so no auxiliary/orthogonality loss is needed — Q is learned purely from the
downstream cross-entropy loss.
"""

import torch
import torch.nn as nn


class LROR(nn.Module):
    def __init__(self, dim: int, rank: int = 32, init_std: float = 0.02):
        super().__init__()
        self.dim = dim
        self.rank = rank
        M = torch.empty(dim, rank)
        nn.init.normal_(M, mean=0.0, std=init_std)
        self.M = nn.Parameter(M)  # the only trainable tensor in this module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1 + Np, D) with [CLS] at position 0.
        cls = x[:, :1, :]
        vis = x[:, 1:, :]
        # QR in fp32 for numerical stability, then cast Q to the token dtype.
        Q, _ = torch.linalg.qr(self.M.float(), mode="reduced")  # (D, r), Q^T Q = I_r
        Q = Q.to(vis.dtype)
        # Z_s = vis @ (Q Q^T) ; Z_c = vis - Z_s   (grouped to avoid the D x D matrix)
        z_s = torch.matmul(torch.matmul(vis, Q), Q.transpose(0, 1))
        z_c = vis - z_s
        return torch.cat([cls, z_c], dim=1)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, rank={self.rank}"
