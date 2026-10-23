"""Wrapper around the SeLop/LROR detector (arXiv:2601.11915).

Frozen CLIP ViT-L/14-336 + per-layer low-rank orthogonal shortcut removal + linear head, 3-class
(real/pad/deepfake). Exposes the uniform ``score_frames`` used by the PAAS pipeline: per frame the
fake-score ``1 - P(real)`` (== ``fake_probability`` over the 3-class head).
"""
from __future__ import annotations

import os
import sys
from typing import List, Optional

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # v3 project root


class SeLopModel:
    name = "selop"

    def __init__(self, ckpt_path: str, clip_path: str, device: str = "cuda:0",
                 amp_dtype: str = "bf16", image_size: int = 336):
        if _ROOT not in sys.path:
            sys.path.insert(0, _ROOT)
        import torch
        from selop.model import SeLopModel as _Net
        from selop.data import build_transforms

        self.torch = torch
        self.device = device
        ck = torch.load(ckpt_path, map_location="cpu")
        mc = ck["config"]
        self.num_classes = mc["num_classes"]
        net = _Net(clip_path, num_classes=mc["num_classes"], rank=mc["rank"],
                   n_intervene=mc["n_intervene"]).to(device)
        net.load_trainable(ck)
        self.model = net.eval()
        self.tfm = build_transforms(image_size, train=False)
        self.amp = {"bf16": torch.bfloat16, "fp16": torch.float16,
                    "fp32": torch.float32}.get(amp_dtype, torch.bfloat16)

    def score_frames(self, rgb_list: List[np.ndarray], batch_size: int = 64) -> List[dict]:
        """rgb_list: HxWx3 uint8 arrays. Returns one dict per frame (input order):
        {"fake": float|None, "error": str|None}."""
        torch = self.torch
        from PIL import Image
        from selop.engine import fake_probability
        out: List[Optional[dict]] = [None] * len(rgb_list)
        with torch.no_grad():
            for i in range(0, len(rgb_list), batch_size):
                chunk = rgb_list[i:i + batch_size]
                try:
                    x = torch.stack([self.tfm(Image.fromarray(r)) for r in chunk]).to(self.device)
                    with torch.autocast(device_type="cuda", dtype=self.amp,
                                        enabled=(self.device.startswith("cuda") and self.amp != torch.float32)):
                        logits = self.model(x)
                    pf = fake_probability(logits, self.num_classes).float().cpu().numpy()
                    for j, v in enumerate(pf):
                        out[i + j] = {"fake": float(v), "error": None}
                except Exception as exc:
                    for j in range(len(chunk)):
                        out[i + j] = {"fake": None, "error": repr(exc)}
        return out
