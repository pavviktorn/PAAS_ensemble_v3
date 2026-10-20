"""Turn a fused fake-score into a decision, match score, and forgery type.

Decision rule (matches the established ensemble/API conventions):
  * fake if fused_fake >= threshold else real
  * match = fused_fake if fake else (1 - fused_fake)        # confidence in the decision
  * ambiguity: if decision == real and match < real_ambiguous_match_min -> "ambiguous"
  * forgery_type (when fake): pad vs deepfake from the ensemble's true-class marginal when
    available, else the FFAA-reported type, else generic "fake".

For accuracy bookkeeping elsewhere, "ambiguous" counts as fake (ambiguous on a fake = correct).
"""
from __future__ import annotations

from typing import Optional

TYPE_NAMES = ("real", "pad", "deepfake")


def decide(fused_fake: float, cfg,
           type_probs: Optional[list] = None,
           ffaa_forgery_type: Optional[str] = None) -> dict:
    tau = float(cfg.threshold)
    fake = fused_fake >= tau
    match = fused_fake if fake else (1.0 - fused_fake)

    if fake:
        if type_probs is not None:               # ensemble marginal [real,pad,deepfake]
            forgery_type = "pad" if type_probs[1] >= type_probs[2] else "deepfake"
        elif ffaa_forgery_type:
            forgery_type = ffaa_forgery_type
        else:
            forgery_type = "fake"
        decision = "fake"
    else:
        forgery_type = "real"
        decision = "real"
        if match < float(cfg.real_ambiguous_match_min):
            decision = "ambiguous"

    return {
        "decision": decision,
        "forgery_score": round(float(fused_fake), 4),
        "match_score": round(float(match), 4),
        "forgery_type": forgery_type,
        "threshold": tau,
    }


def eval_label(decision: str) -> str:
    """Map a decision to its binary accuracy label: ambiguous counts as fake."""
    return "real" if decision == "real" else "fake"
