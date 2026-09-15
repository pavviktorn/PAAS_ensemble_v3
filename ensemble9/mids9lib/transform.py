"""Full-image preprocessing — letterbox to square (NO face/center crop) then resize + CLIP-normalize.

Identical to the pipeline used for tuning/evaluation of this ensemble: the WHOLE frame is kept
(face is never cropped), aspect ratio preserved by zero-padding to a square, resized to 336, and
normalized with the CLIP mean/std the encoder was trained with.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
_MEAN = torch.tensor(CLIP_MEAN).view(3, 1, 1)
_STD = torch.tensor(CLIP_STD).view(3, 1, 1)


def load_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def letterbox_to_tensor(rgb: np.ndarray, size: int = 336) -> torch.Tensor:
    h, w = rgb.shape[:2]
    s = max(h, w)
    canvas = np.zeros((s, s, 3), dtype=rgb.dtype)
    top, left = (s - h) // 2, (s - w) // 2
    canvas[top:top + h, left:left + w] = rgb
    img = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return (t - _MEAN) / _STD
