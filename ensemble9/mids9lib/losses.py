"""Combined MIDS++ training objective.

``L = ce + GenD(alignment + uniformity) + ForAda(weak MIL + query orthogonality) + Effort(svd
orthogonality + frobenius preservation)``.  Every term is gated by a config weight, so the
staged ablations (svd-only, +gend, +artifact, full) are pure config changes.

CE is recomputed here from the flattened logits (rather than reusing ``forward``'s ``cls_loss``)
so the objective is robust under ``DataParallel`` gathering and so class weights / label smoothing
live in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .artifact import query_orthogonality_loss, weak_artifact_discovery_loss
from .config import MidsPlusConfig
from .gend import alignment_loss, uniformity_loss


@dataclass
class LossBreakdown:
    total: torch.Tensor
    parts: Dict[str, torch.Tensor]


class MidsPlusLoss(nn.Module):
    def __init__(self, cfg: MidsPlusConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        outputs: dict,
        labels: torch.Tensor,
        cls_labels: Optional[torch.Tensor] = None,
        model: Optional[nn.Module] = None,
    ) -> LossBreakdown:
        cfg = self.cfg
        logits = outputs["logits"].reshape(-1, cfg.num_classes)
        labels = labels.reshape(-1).long().to(logits.device)
        parts: Dict[str, torch.Tensor] = {}

        weight = None
        if cfg.ce_class_weights is not None:
            weight = torch.tensor(cfg.ce_class_weights, dtype=logits.dtype, device=logits.device)
        parts["ce"] = cfg.ce_weight * F.cross_entropy(
            logits, labels, weight=weight, ignore_index=-1, label_smoothing=cfg.label_smoothing
        )

        image_embed = outputs.get("image_embed")
        if image_embed is not None and cls_labels is not None:
            cls_labels = cls_labels.reshape(-1).long().to(image_embed.device)
            if cfg.alignment_weight > 0:
                parts["alignment"] = cfg.alignment_weight * alignment_loss(image_embed, cls_labels)
            if cfg.uniformity_weight > 0:
                parts["uniformity"] = cfg.uniformity_weight * uniformity_loss(image_embed, t=cfg.uniformity_t)

        artifact_map = outputs.get("artifact_map")
        if artifact_map is not None and cls_labels is not None and cfg.artifact_mil_weight > 0:
            parts["artifact_mil"] = cfg.artifact_mil_weight * weak_artifact_discovery_loss(
                artifact_map,
                cls_labels,
                topk_fraction=cfg.artifact_topk_fraction,
                real_weight=cfg.artifact_real_weight,
                sparsity_weight=cfg.artifact_sparsity_weight,
            )

        artifact_query_logits = outputs.get("artifact_query_logits")
        if artifact_query_logits is not None and cfg.query_orth_weight > 0:
            parts["query_orth"] = cfg.query_orth_weight * query_orthogonality_loss(artifact_query_logits)

        if model is not None and hasattr(model, "regularization_losses"):
            reg = model.regularization_losses()
            if cfg.svd_orth_weight > 0:
                parts["svd_orth"] = cfg.svd_orth_weight * reg["orth"]
            if cfg.svd_keep_weight > 0:
                parts["svd_keep"] = cfg.svd_keep_weight * reg["keep"]

        total = torch.stack([v.reshape(()) for v in parts.values()]).sum()
        return LossBreakdown(total=total, parts={k: v.detach() for k, v in parts.items()})
