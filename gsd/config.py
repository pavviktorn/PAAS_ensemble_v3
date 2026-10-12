"""PAAS_v4 GSD configuration. One dataclass describes the model + training run; load from JSON."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field, fields
from typing import Optional

# project root (this file lives in <root>/gsd/) -- all bundled assets resolve relative to it
_PROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# CLIP ViT-L/14-336 vendored into this standalone project (vision weights only; no download needed)
DEFAULT_CLIP = os.path.join(_PROOT, "base_models", "clip-vit-large-patch14-336")

# MIDS datasets (ground-truth label resolved per image by gsd.get_label.get_label_all)
DEFAULT_TRAIN = "/datasets/work/vLLM/temp/testset/testset_mids/mids_first_half.json"
DEFAULT_VAL = "/datasets/work/vLLM/temp/testset/testset_mids/mids_testset.json"

# 3-class scheme: REAL/PAD/DEEPFAKE. MAKEUP is folded into PAD; UNKNOWN images are dropped.
CLASS_NAMES = ("real", "pad", "deepfake")


@dataclass
class GSDConfig:
    # ---- model / GSD (paper defaults) ----
    clip_path: str = DEFAULT_CLIP
    n_gsd_layers: int = 4              # inject GSD into the final N encoder layers
    k: int = 16                        # semantic-subspace dimension (ablation 2..64)
    guide_pool: str = "gap"            # frozen semantic consensus: 'gap' (paper) or 'cls'
    qr_method: str = "householder"     # 'householder' (paper) or 'svd'
    per_layer_guide: bool = False      # False: one global U for all GSD layers (paper); True: per-layer U
    image_size: int = 336
    head_pool: str = "gap"             # detector pooling before the head: 'gap' or 'cls'
    trainable: str = "full"            # 'full' (fine-tune trainable CLIP, paper) | 'lastN' | 'head'

    # ---- classification ----
    num_classes: int = 3               # 3-class real/pad/deepfake (CrossEntropy); 2 -> real/fake
    label_source: str = "get_label_all"  # 'get_label_all' (authoritative) | 'cls_label' | 'path'
    class_weight: bool = True          # inverse-frequency CE weights (handles real/pad/deepfake imbalance)
    select_metric: str = "bin_auc"     # best.pt criterion: 'bin_auc' (real-vs-fake) | 'acc' | 'bal_acc'

    # ---- training ----
    lr: float = 1e-6                   # AdamW backbone lr (paper) -- the per-group PEAK lr
    head_lr: float = 1e-4              # classifier head lr (backbone is tiny-lr; head can move faster)
    weight_decay: float = 1e-4
    lr_scheduler: str = "cosine"       # 'none' | 'cosine' | 'linear' (per-step; scales all groups)
    warmup_steps: int = 200            # linear warmup 0 -> peak over this many optimizer steps
    min_lr_ratio: float = 0.05         # final lr floor as a fraction of the peak (cosine/linear)
    epochs: int = 5
    batch_size: int = 128              # paper; reduce if VRAM-limited (GSD needs batch >= 2)
    num_workers: int = 8
    cpu_fraction: float = 0.5          # cap OMP/MKL/BLAS + torch threads to this fraction of cores
    amp_dtype: str = "bf16"            # 'bf16' | 'fp16' | 'fp32'
    grad_clip: float = 1.0
    seed: int = 0

    # ---- augmentation (paper: Gaussian blur + JPEG) ----
    aug_hflip: bool = True
    aug_blur_p: float = 0.1
    aug_jpeg_p: float = 0.1
    aug_jpeg_min_q: int = 40

    # ---- data / io ----
    train_data: Optional[str] = DEFAULT_TRAIN   # json list (label via get_label_all) or real/fake tree
    val_data: Optional[str] = DEFAULT_VAL
    eval_limit: int = 0                # cap #val images during training (0 = all); full eval via eval.py
    eval_every: int = 0                # evaluate + maybe-save best.pt every N optimizer steps (0 = per-epoch only)
    output_dir: str = "runs/gsd"
    gpus: str = "0,1,2,3"              # GPU ids for training (DataParallel); edit freely, e.g. "0" or "0,1"
    log_every: int = 50
    eval_batch_size: int = 64

    # ---- fixed semantic anchor (built at train time, embedded into the checkpoint) ----
    anchor_data: Optional[str] = None  # reference set for the fixed U; None -> use val_data (the testset)
    anchor_limit: int = 512            # #reference images (strided) used to build the embedded anchor U

    def validate(self) -> "GSDConfig":
        assert self.guide_pool in ("gap", "cls")
        assert self.head_pool in ("gap", "cls")
        assert self.qr_method in ("householder", "svd")
        assert self.trainable in ("full", "lastN", "head")
        assert self.lr_scheduler in ("none", "cosine", "linear")
        assert self.label_source in ("get_label_all", "cls_label", "path")
        assert self.select_metric in ("bin_auc", "acc", "bal_acc")
        assert self.num_classes in (2, 3)
        assert 0.0 < self.cpu_fraction <= 1.0
        assert self.k >= 1 and self.n_gsd_layers >= 0
        return self

    @classmethod
    def from_file(cls, path: str) -> "GSDConfig":
        with open(path) as fh:
            return cls.from_dict(json.load(fh))

    @classmethod
    def from_dict(cls, d: dict) -> "GSDConfig":
        valid = {f.name for f in fields(cls)}            # drop stale/unknown keys (e.g. legacy "device")
        d = {k: v for k, v in d.items() if k in valid}
        if d.get("clip_path") and not os.path.isabs(str(d["clip_path"])):
            d["clip_path"] = os.path.join(_PROOT, d["clip_path"])   # relative -> project root
        return cls(**d).validate()

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)
