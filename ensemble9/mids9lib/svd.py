"""Effort-style SVD residual adaptation of the CLIP vision encoder.

Motivation (the single most important change vs upstream MIDS): upstream adapts CLIP by
unfreezing the last 2 transformer layers (~12M params trained on a comparatively small MIDS
set).  That overwrites pretrained directions and erodes CLIP's open-world prior -- the exact
thing that hurts cross-dataset generalisation (FFAA's headline ``sACC`` metric).

Effort instead factorises each target weight ``W = U S Vh`` and *freezes the principal
(top-``main_rank``) component* -- the pretrained knowledge -- while training only a small
residual built from the minor singular directions.  Two regularisers keep the adapted weight
faithful to the original: an orthogonality term (residual basis stays orthonormal w.r.t. the
principal basis) and a Frobenius-norm-preservation term (total weight energy is preserved).

The ``EffortSVDLinear`` math mirrors the validated implementation in the local ``hybrid_dfd``
project; ``replace_linears_with_svd`` is generalised here to target the last K layers of a
HuggingFace ``CLIPVisionModel`` (or the smoke-test stub, which mimics the same module names).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class SVDReport:
    replaced: List[str] = field(default_factory=list)


class EffortSVDLinear(nn.Module):
    """Frozen principal SVD component + trainable residual SVD factors."""

    def __init__(self, linear: nn.Linear, residual_rank: int = 16) -> None:
        super().__init__()
        weight = linear.weight.detach().float().cpu()
        out_features, in_features = weight.shape
        max_rank = min(out_features, in_features)
        residual_rank = max(1, min(int(residual_rank), max_rank))
        main_rank = max_rank - residual_rank
        if main_rank <= 0:
            main_rank = max_rank - 1
            residual_rank = 1

        self.in_features = in_features
        self.out_features = out_features
        self.r = main_rank
        self.residual_rank = residual_rank
        self.register_buffer("weight_original_fnorm", torch.norm(weight, p="fro"))

        u, s, vh = torch.linalg.svd(weight, full_matrices=False)
        r = min(main_rank, len(s))
        u_r, s_r, vh_r = u[:, :r], s[:r], vh[:r, :]
        weight_main = u_r @ torch.diag(s_r) @ vh_r
        # Principal component: frozen, carries pretrained knowledge.
        self.weight_main = nn.Parameter(weight_main, requires_grad=False)

        u_residual, s_residual, vh_residual = u[:, r:], s[r:], vh[r:, :]
        if len(s_residual) > 0:
            # Trainable residual factors (the forgery-specific degrees of freedom).
            self.S_residual = nn.Parameter(s_residual.clone(), requires_grad=True)
            self.U_residual = nn.Parameter(u_residual.clone(), requires_grad=True)
            self.V_residual = nn.Parameter(vh_residual.clone(), requires_grad=True)
            # Frozen principal bases, used only by the orthogonality regulariser.
            self.U_r = nn.Parameter(u_r.clone(), requires_grad=False)
            self.V_r = nn.Parameter(vh_r.clone(), requires_grad=False)
        else:  # pragma: no cover - degenerate tiny layers only
            self.S_residual = self.U_residual = self.V_residual = None
            self.U_r = self.V_r = None

        if linear.bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)

    def current_weight(self) -> torch.Tensor:
        if self.S_residual is not None:
            residual = self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
            return self.weight_main.to(residual.device, residual.dtype) + residual
        return self.weight_main

    @property
    def weight(self) -> torch.Tensor:
        return self.current_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.current_weight().to(device=x.device, dtype=x.dtype)
        bias = None if self.bias is None else self.bias.to(device=x.device, dtype=x.dtype)
        return F.linear(x, weight, bias)

    def orthogonal_loss(self) -> torch.Tensor:
        if self.S_residual is None:
            return torch.zeros((), device=self.weight_main.device, dtype=self.weight_main.dtype)
        u = torch.cat((self.U_r, self.U_residual), dim=1)
        v = torch.cat((self.V_r, self.V_residual), dim=0)
        uut = u @ u.t()
        vvt = v @ v.t()
        u_eye = torch.eye(uut.size(0), device=uut.device, dtype=uut.dtype)
        v_eye = torch.eye(vvt.size(0), device=vvt.device, dtype=vvt.dtype)
        return 0.5 * torch.norm(uut - u_eye, p="fro") + 0.5 * torch.norm(vvt - v_eye, p="fro")

    def keep_frobenius_loss(self) -> torch.Tensor:
        if self.S_residual is None:
            return torch.zeros((), device=self.weight_main.device, dtype=self.weight_main.dtype)
        current = self.current_weight()
        original = self.weight_original_fnorm.to(current.device, current.dtype)
        return torch.abs(torch.norm(current, p="fro").pow(2) - original.pow(2))


def _vision_layers(image_encoder: nn.Module) -> nn.ModuleList:
    """Return the ModuleList of transformer blocks for HF CLIP or the smoke-test stub."""
    if hasattr(image_encoder, "vision_model"):
        return image_encoder.vision_model.encoder.layers
    if hasattr(image_encoder, "encoder"):
        return image_encoder.encoder.layers
    raise AttributeError("Could not locate vision transformer layers on the image encoder.")


def replace_linears_with_svd(
    image_encoder: nn.Module,
    target_last_layers: int = 6,
    targets: Sequence[str] = ("self_attn.out_proj",),
    residual_rank: int = 16,
) -> SVDReport:
    """Replace selected ``nn.Linear`` sub-modules in the last K CLIP layers with SVD adapters.

    ``targets`` are dotted attribute paths *relative to a transformer block*, e.g.
    ``"self_attn.out_proj"`` or ``"mlp.fc2"``.
    """
    report = SVDReport()
    layers = _vision_layers(image_encoder)
    n = len(layers)
    start = max(0, n - int(target_last_layers))
    for li in range(start, n):
        block = layers[li]
        for target in targets:
            parent, _, leaf = target.rpartition(".")
            holder = block
            ok = True
            for part in parent.split(".") if parent else []:
                if not hasattr(holder, part):
                    ok = False
                    break
                holder = getattr(holder, part)
            if not ok or not hasattr(holder, leaf):
                continue
            child = getattr(holder, leaf)
            if isinstance(child, nn.Linear):
                setattr(holder, leaf, EffortSVDLinear(child, residual_rank=residual_rank))
                report.replaced.append(f"layers.{li}.{target}")
    return report


def iter_svd_modules(module: nn.Module) -> Iterable[EffortSVDLinear]:
    for child in module.modules():
        if isinstance(child, EffortSVDLinear):
            yield child


def svd_regularization_losses(module: nn.Module) -> dict:
    device = next(module.parameters()).device
    orth, keep = [], []
    for svd in iter_svd_modules(module):
        orth.append(svd.orthogonal_loss())
        keep.append(svd.keep_frobenius_loss())
    return {
        "orth": torch.stack(orth).mean() if orth else torch.zeros((), device=device),
        "keep": torch.stack(keep).mean() if keep else torch.zeros((), device=device),
    }
