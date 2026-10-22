"""Wrapper around the GSD detector (Geometric Semantic Decoupling, arXiv:2603.09242).

Frozen + trainable CLIP ViT-L/14-336; the trained checkpoint embeds a fixed semantic basis U
(built from the testset at train time) so single-image scoring works with no reference batch.
Exposes the uniform ``score_frames`` used by the PAAS pipeline: per frame, the 3-class
(real/pad/deepfake) fake-score ``1 - P(real)``.
"""
from __future__ import annotations

import os
import sys
from typing import List, Optional

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # v3 project root


class GSDModel:
    name = "gsd"

    def __init__(self, ckpt_path: str, clip_path: str, device: str = "cuda:0",
                 amp_dtype: str = "bf16"):
        if _ROOT not in sys.path:
            sys.path.insert(0, _ROOT)
        import torch
        from gsd.config import GSDConfig, CLASS_NAMES
        from gsd.model import GSDDetector
        from gsd.data import build_transform

        self.torch = torch
        self.device = device
        payload = torch.load(ckpt_path, map_location="cpu")
        cfg = GSDConfig.from_dict(payload["config"])
        cfg.clip_path = clip_path                      # use the v3 shared CLIP backbone, not the train-time path
        self.cfg = cfg
        model = GSDDetector(cfg)
        model.trainable.load_state_dict(payload["trainable"])
        model.head.load_state_dict(payload["head"])
        if payload.get("anchor_U") is not None:
            model.set_fixed_U(payload["anchor_U"].to(device))
        elif getattr(model, "_fixed_U", None) is None:
            print("[gsd] WARNING: checkpoint has no embedded anchor U; single-image scores are undefined")
        self.model = model.to(device).eval()
        self.tfm = build_transform(cfg.image_size, train=False, cfg=cfg)
        self.names = list(CLASS_NAMES) if cfg.num_classes == 3 else ["real", "fake"]
        self.amp = {"bf16": torch.bfloat16, "fp16": torch.float16,
                    "fp32": torch.float32}.get(amp_dtype, torch.bfloat16)

    def score_frames(self, rgb_list: List[np.ndarray], batch_size: int = 64) -> List[dict]:
        """rgb_list: HxWx3 uint8 arrays. Returns one dict per frame (input order):
        {"fake": float|None, "type_probs": [r,p,d]|None, "pred": str|None, "error": str|None}."""
        torch = self.torch
        from PIL import Image
        out: List[Optional[dict]] = [None] * len(rgb_list)
        with torch.no_grad():
            for i in range(0, len(rgb_list), batch_size):
                chunk = rgb_list[i:i + batch_size]
                try:
                    x = torch.stack([self.tfm(Image.fromarray(r)) for r in chunk]).to(self.device)
                    with torch.autocast(device_type="cuda", dtype=self.amp,
                                        enabled=(self.device.startswith("cuda") and self.amp != torch.float32)):
                        logits = self.model(x)
                    probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
                    for j, pr in enumerate(probs):
                        cls = int(pr.argmax())
                        tp = [float(pr[0]), float(pr[1]), float(pr[2])] if len(pr) == 3 else None
                        out[i + j] = {"fake": float(1.0 - pr[0]), "type_probs": tp,
                                      "pred": self.names[cls], "error": None}
                except Exception as exc:                    # a bad frame in the chunk fails only that chunk
                    for j in range(len(chunk)):
                        out[i + j] = {"fake": None, "type_probs": None, "pred": None, "error": repr(exc)}
        return out
