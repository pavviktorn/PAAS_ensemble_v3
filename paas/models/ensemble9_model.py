"""Wrapper around the MLLM-free 9-class ensemble (A1+A2+A3).

Exposes a uniform ``score_frames`` returning, per frame, the threshold-independent fake-score and
the [real,pad,deepfake] true-class marginal (used for forgery-type when the decision is fake).
The underlying loader applies the transformers-4.37<->5.x key remap, so all trained tensors load.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


class Ensemble9Model:
    name = "ensemble9"

    def __init__(self, config_path: str, device: str = "cuda:0",
                 members: Optional[List[str]] = None):
        """`members`: which 9-class members to load/run (None = all in the config). The pipeline
        passes only the members named in fusion.components, so e.g. A3_9c is skipped entirely."""
        from mids9lib.ensemble import MidsEnsemble  # vendored, fixed loader
        self.eng = MidsEnsemble(config_path, device=device, members=members)
        self.load_time_sec = getattr(self.eng, "load_time_sec", None)

    @property
    def member_names(self) -> List[str]:
        """Names of the ensemble members (e.g. ['A1_9c', 'A2_9c', 'A3_9c'])."""
        return list(getattr(self.eng, "names", []))

    def score_frames(self, rgb_list: List[np.ndarray], batch_size: int = 32) -> List[dict]:
        """rgb_list: list of HxWx3 uint8 arrays. Returns one dict per frame (input order):
        {"fake": float|None, "type_probs": [r,p,d]|None, "per_model": {name: fake}|None,
         "error": str|None}. `per_model` is each member's own fake-score, for offline combination."""
        items = [(i, rgb) for i, rgb in enumerate(rgb_list)]
        res = self.eng.predict_rgb_batch(items, batch_size=batch_size)
        out: List[Optional[dict]] = [None] * len(rgb_list)
        for r in res:
            i = r["image"]
            if "error" in r:
                out[i] = {"fake": None, "type_probs": None, "per_model": None, "error": r["error"]}
            else:
                tp = r["type_probs"]
                out[i] = {"fake": float(r["forgery_score"]),
                          "type_probs": [tp["real"], tp["pad"], tp["deepfake"]],
                          "per_model": r.get("per_model_fake"),
                          "error": None}
        return out
