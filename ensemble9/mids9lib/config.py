"""Configuration for MIDS++.

A single flat dataclass keeps every knob in one place and is trivially serialisable to/from
YAML.  ``MidsPlusConfig.from_yaml`` + ``--set a.b=c`` style overrides (see ``train.py``) cover
the staged-ablation workflow without a heavyweight config framework.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, List, Optional

import yaml


@dataclass
class MidsPlusConfig:
    # ---- encoders -------------------------------------------------------------------------
    dim: int = 768
    image_model_path: str = "models/clip-vit-large-patch14-336"
    text_model_path: str = "models/t5-base"
    # number of candidate answers per image fed to MIDS (FFAA uses 1 neutral + 2 hypothetical)
    samples_per_image: int = 3
    num_classes: int = 4
    # texts are the fixed claim prompts (same every step) -> encode with T5 once and reuse the
    # precomputed features in forward, skipping the redundant per-step T5 call. Set False if the
    # texts vary per sample (e.g. MLLM-generated answers).
    fixed_texts: bool = True

    # ---- CLIP adaptation strategy ---------------------------------------------------------
    # "svd"    : Effort-style residual SVD adapters (recommended; preserves CLIP subspace)
    # "unfreeze": upstream MIDS recipe, unfreeze the last N transformer layers
    # "frozen" : fully frozen CLIP (probe-only)
    clip_adapt: str = "svd"
    unfreeze_vision_last_layers: int = 2  # used when clip_adapt == "unfreeze"

    # Effort SVD options (used when clip_adapt == "svd")
    svd_target_last_layers: int = 6           # apply to the last K CLIP vision layers
    svd_targets: List[str] = field(default_factory=lambda: ["self_attn.out_proj"])
    svd_residual_rank: int = 16               # # of trainable minor singular directions
    tune_layer_norm: bool = True              # GenD-style: let CLIP LayerNorms adapt

    # ---- ForensicsAdapter-style local artifact branch ------------------------------------
    artifact_enabled: bool = True
    artifact_num_queries: int = 4
    artifact_num_heads: int = 8

    # ---- loss weights ---------------------------------------------------------------------
    ce_weight: float = 1.0
    # optional per-class CE weighting (their newfmt feature); None -> uniform.
    ce_class_weights: Optional[List[float]] = None
    label_smoothing: float = 0.0

    # GenD hyperspherical regularisers on the L2-normalised image embedding
    alignment_weight: float = 0.5
    uniformity_weight: float = 0.5
    uniformity_t: float = 2.0

    # weak artifact discovery (MIL) on the artifact map, supervised by image authenticity
    artifact_mil_weight: float = 0.3
    artifact_topk_fraction: float = 0.15
    artifact_real_weight: float = 0.5
    artifact_sparsity_weight: float = 0.05
    query_orth_weight: float = 0.05

    # Effort SVD regularisers
    svd_orth_weight: float = 0.01
    svd_keep_weight: float = 0.001

    # ---- data -----------------------------------------------------------------------------
    train_data_path: Optional[str] = None
    val_data_path: Optional[str] = None
    image_size: int = 336
    augment: bool = True
    # Drop items whose image file is missing/unreadable at dataset init (logs the count).
    # Real eval/train sets can contain a small fraction of missing files; default off so
    # training fails loudly unless you opt in.
    skip_missing_images: bool = False

    # ---- optimisation ---------------------------------------------------------------------
    epochs: int = 2
    batch_size: int = 24            # images per device (each expands to samples_per_image)
    val_batch_size: int = 12
    lr: float = 1e-4
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.03
    grad_clip: float = 1.0
    save_every_steps: int = 0       # >0: log + save runs/<out>/last.pt every N steps (resilience)
    amp_dtype: str = "bf16"         # "bf16" | "fp16" | "none"
    num_workers: int = 8
    seed: int = 0
    output_dir: str = "runs/mids_pp"

    # ---- runtime / testing ----------------------------------------------------------------
    # When True, build tiny random stand-in encoders instead of downloading T5/CLIP.
    # Used by the CPU smoke test; never enable for real training.
    stub_encoders: bool = False
    stub_image_layers: int = 4
    stub_text_vocab: int = 256

    @classmethod
    def from_yaml(cls, path: str) -> "MidsPlusConfig":
        with open(path, "r") as handle:
            data = yaml.safe_load(handle) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "MidsPlusConfig":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**data)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def apply_overrides(self, overrides: List[str]) -> "MidsPlusConfig":
        """Apply ``key=value`` CLI overrides with light type coercion."""
        for item in overrides:
            if "=" not in item:
                raise ValueError(f"Override must be key=value, got: {item!r}")
            key, raw = item.split("=", 1)
            key = key.strip()
            if not hasattr(self, key):
                raise ValueError(f"Unknown override key: {key!r}")
            setattr(self, key, _coerce(getattr(self, key), raw.strip()))
        return self

    def dump(self, path: str) -> None:
        with open(path, "w") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)


def _coerce(current: Any, raw: str) -> Any:
    if isinstance(current, bool):
        return raw.lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int) and not isinstance(current, bool):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, list):
        raw = raw.strip("[]")
        if not raw:
            return []
        parts = [p.strip() for p in raw.split(",")]
        try:
            return [float(p) for p in parts]
        except ValueError:
            return parts
    if raw.lower() in {"none", "null"}:
        return None
    return raw
