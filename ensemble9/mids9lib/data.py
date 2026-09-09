"""Dataset + collation for MIDS++.

Consumes exactly the FFAA MIDS training-data schema (one JSON list; one entry per image)::

    {
      "image": "/abs/path/to/face.jpg",
      "cls_label": 0,                      # image authenticity: 0 real, 1 fake
      "answers": [                         # 1 neutral + N + M hypothetical answers (default 3)
        {"content": "Image description: ...\\nForgery reasoning: ...",  # verdict already masked out
         "result": "real",                # the answer's claimed verdict (kept for the selector)
         "label": 0},                     # 4-class target = 2*cls_label + (1 if claim=="fake" else 0)
        ...
      ]
    }

So a pre-generated FFAA-format JSON loads with no conversion.  Image decoding uses cv2 (always
available); CLIP-processor normalisation is used in real mode and a dependency-free resize+
normalise transform is used otherwise (stub/smoke tests).
"""

from __future__ import annotations

import json
import os
from typing import Callable, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _build_albumentations():
    try:
        from albumentations import (Compose, FancyPCA, GaussianBlur, GaussNoise,
                                    HueSaturationValue, ImageCompression, MotionBlur, OneOf)
    except Exception:
        return None
    return Compose([
        ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
        GaussNoise(p=0.2),
        MotionBlur(p=0.2),
        GaussianBlur(blur_limit=3, p=0.2),
        OneOf([FancyPCA(), HueSaturationValue()], p=0.7),
    ])


def build_image_transform(cfg, mode: str = "train") -> Callable:
    """Return a callable mapping an RGB uint8 HxWx3 array -> float tensor (3, S, S)."""
    if not cfg.stub_encoders:
        try:
            from transformers import CLIPImageProcessor

            processor = CLIPImageProcessor.from_pretrained(cfg.image_model_path)

            def _clip_transform(rgb: np.ndarray) -> torch.Tensor:
                from PIL import Image

                pix = processor(images=Image.fromarray(rgb), return_tensors="pt")["pixel_values"]
                return pix[0]

            return _clip_transform
        except Exception:
            pass  # fall through to the dependency-free transform

    size = cfg.image_size
    mean = torch.tensor(CLIP_MEAN).view(3, 1, 1)
    std = torch.tensor(CLIP_STD).view(3, 1, 1)

    def _simple_transform(rgb: np.ndarray) -> torch.Tensor:
        import cv2

        img = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return (t - mean) / std

    return _simple_transform


class MidsAnswersDataset(Dataset):
    def __init__(self, data_path: str, cfg, mode: str = "train",
                 image_transform: Optional[Callable] = None) -> None:
        with open(data_path, "r") as handle:
            self.data = json.load(handle)
        self.cfg = cfg
        self.mode = mode
        self.samples_per_image = cfg.samples_per_image
        self.transform = image_transform or build_image_transform(cfg, mode)
        self.aug = _build_albumentations() if (mode == "train" and cfg.augment) else None
        if getattr(cfg, "skip_missing_images", False):
            kept = [it for it in self.data if os.path.exists(it["image"])]
            dropped = len(self.data) - len(kept)
            if dropped:
                print(f"[MidsAnswersDataset] dropped {dropped}/{len(self.data)} items with missing images")
            self.data = kept

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        import cv2

        item = self.data[idx]
        bgr = cv2.imread(item["image"], cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {item['image']}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.aug is not None:
            rgb = self.aug(image=rgb)["image"]
        image = self.transform(rgb)

        answers = item["answers"]
        s = self.samples_per_image
        texts = [a["content"] for a in answers][:s]
        answers_result = [a["result"] for a in answers][:s]
        labels = [int(a["label"]) for a in answers][:s]
        # Pad defensively if an entry has fewer answers than expected.
        while len(texts) < s:
            texts.append("")
            answers_result.append("real")
            labels.append(-1)
        return {
            "image": image,
            "cls_label": int(item["cls_label"]),
            "texts": texts,
            "answers_result": answers_result,
            "labels": labels,
        }


def collate_fn(batch: List[dict]) -> dict:
    images = torch.stack([b["image"] for b in batch], dim=0)
    cls_label = torch.tensor([b["cls_label"] for b in batch], dtype=torch.long)
    texts: List[str] = []
    answers_result: List[str] = []
    labels: List[int] = []
    for b in batch:
        texts.extend(b["texts"])
        answers_result.extend(b["answers_result"])
        labels.extend(b["labels"])
    return {
        "image": images,
        "cls_label": cls_label,
        "texts": texts,                       # flat list, length B * samples_per_image
        "answers_result": answers_result,     # flat list, length B * samples_per_image
        "labels": torch.tensor(labels, dtype=torch.long),
    }
