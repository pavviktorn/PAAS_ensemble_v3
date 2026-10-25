"""MIDS dataset for SeLop.

The ground-truth label is derived from the image PATH via `get_label_all`
(REAL=0, PAD=1, DEEPFAKE=2, MAKEUP=3, UNKNOWN=-1) — not from the json's
`cls_label`/`answers` fields. For SeLop's binary task the 4-class label is
mapped to {real=0, fake=1} (PAD/DEEPFAKE/MAKEUP -> fake); UNKNOWN (-1) samples
are dropped. Set `num_classes=4` to keep the raw 4-class label instead.

To avoid re-parsing the 1.5 GB train json in every DDP process, a slim
`<json>.selop_idx.<mode>.tsv` cache (path<TAB>label per line) is built once.
"""

import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .get_label import get_label_all, REAL, PAD, DEEPFAKE, MAKEUP, UNKNOWN

# 3-class scheme (matches the GSD project): real / pad / deepfake.
# MAKEUP is folded into PAD; UNKNOWN images are dropped.
CLASS_NAMES_3 = ("real", "pad", "deepfake")
CLASS_NAMES_2 = ("real", "fake")


def derive_label(path, num_classes):
    """Path -> training label via get_label_all. Returns None to drop the sample."""
    lab = get_label_all(path)
    if lab == UNKNOWN:
        return None
    if num_classes == 2:
        return 0 if lab == REAL else 1                 # real vs fake
    if num_classes == 3:
        if lab == REAL:
            return 0                                   # real
        if lab == DEEPFAKE:
            return 2                                   # deepfake
        return 1                                       # PAD or MAKEUP -> pad
    raise ValueError(f"num_classes must be 2 or 3, got {num_classes}")


def class_names(num_classes):
    return CLASS_NAMES_3 if num_classes == 3 else CLASS_NAMES_2

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def build_transforms(image_size=336, train=True):
    norm = transforms.Normalize(CLIP_MEAN, CLIP_STD)
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(
                image_size, scale=(0.8, 1.0), ratio=(0.9, 1.0 / 0.9),
                interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            norm,
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        norm,
    ])


def build_index(json_path, num_classes=2, cache=True, log=print):
    """Parse a MIDS json into a list[(path, label)] using `get_label_all`;
    cache to a slim .tsv keyed by num_classes."""
    tsv = f"{json_path}.selop_idx.{num_classes}c.tsv"
    if cache and os.path.exists(tsv) and os.path.getmtime(tsv) >= os.path.getmtime(json_path):
        samples = []
        with open(tsv) as f:
            for line in f:
                p, lab = line.rstrip("\n").rsplit("\t", 1)
                samples.append((p, int(lab)))
        log(f"[data] loaded cached index {tsv} ({len(samples)} samples)")
        return samples

    log(f"[data] parsing {json_path} ...")
    with open(json_path) as f:
        data = json.load(f)
    samples = []
    dropped = 0
    for rec in data:
        img = rec.get("image")
        if img is None:
            continue
        lab = derive_label(img, num_classes)
        if lab is None:
            dropped += 1
            continue
        samples.append((img, lab))
    log(f"[data] {len(samples)} samples kept, {dropped} dropped (UNKNOWN)")
    if cache:
        tmp = tsv + ".tmp"
        with open(tmp, "w") as f:
            for p, lab in samples:
                f.write(f"{p}\t{lab}\n")
        os.replace(tmp, tsv)
        log(f"[data] wrote index cache {tsv}")
    return samples


class MidsBinaryDataset(Dataset):
    def __init__(self, samples, transform, image_size=336, return_index=False):
        self.samples = samples
        self.transform = transform
        self.image_size = image_size
        self.return_index = return_index

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # Missing/corrupt image: return a black frame; rare, keeps batch shapes valid.
            img = Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0))
        x = self.transform(img)
        if self.return_index:
            return x, label, idx
        return x, label
