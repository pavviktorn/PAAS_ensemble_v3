"""Programmatic drop-in helper for FFAA inference.

Mirrors the role of FFAA's ``load_mids`` + the scoring step in ``inference.py``.  Given the image
and the (verdict-masked) candidate answer texts + their claimed verdicts, returns exactly what
FFAA's ``make_decision`` returns, so it slots into the existing pipeline.

    from mids_plus.infer import MidsPlusScorer
    scorer = MidsPlusScorer("runs/mids_pp/best.pt", device="cuda")
    best_idx, pred, match_score, forgery_score = scorer.score(image_pil, masked_texts, results)
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .checkpoint import load_checkpoint
from .data import build_image_transform
from .selector import make_decision
from .tokenize import build_tokenizer


class MidsPlusScorer:
    def __init__(self, checkpoint: str, device: str = "cuda") -> None:
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.model, self.cfg = load_checkpoint(checkpoint, device=self.device)
        self.tokenizer = build_tokenizer(self.cfg)
        self.transform = build_image_transform(self.cfg, "val")

    @torch.no_grad()
    def score(self, image_rgb: np.ndarray, masked_texts: List[str], results: List[str]) -> Tuple[int, int, float, float]:
        """``image_rgb``: HxWx3 uint8 RGB.  ``masked_texts``/``results``: one per candidate answer."""
        s = len(masked_texts)
        image = self.transform(image_rgb).unsqueeze(0).to(self.device)
        enc = self.tokenizer(masked_texts, return_tensors="pt", padding="longest", truncation=True, max_length=512).to(self.device)
        out = self.model(enc, image, None, 1, s - 1, 0)
        scores = F.softmax(out["logits"].float(), dim=2)[0]   # (s, 4)
        return make_decision(results, scores)
