"""Checkpoint save/load.

Checkpoints store only the *learned* tensors plus the config.  At load time the model is rebuilt
(which re-instantiates frozen T5/CLIP and re-derives the SVD principal component from the original
CLIP weights), then the learned tensors are loaded on top.  This keeps checkpoints small and makes
them a clean drop-in for FFAA inference: point it at the same CLIP/T5 weights and load the head.
"""

from __future__ import annotations

import torch

from .config import MidsPlusConfig
from .model import MIDSPlus, build_model


def save_checkpoint(model: MIDSPlus, cfg: MidsPlusConfig, path: str) -> None:
    payload = {
        "format": "mids_plus.v1",
        "config": cfg.to_dict(),
        "state": model.trainable_state_dict(),
        "svd_replaced": getattr(model.svd_report, "replaced", None),
    }
    torch.save(payload, path)


def import_upstream_mids_checkpoint(model: MIDSPlus, path: str, verbose: bool = True) -> dict:
    """Load an upstream FFAA ``mids.pth`` (trainable head + fine-tuned CLIP layers) into a
    ``mids_original``-configured MIDSPlus.

    The upstream checkpoint stores only the fine-tuned tensors with a ``module.`` prefix
    (DeepSpeed/DDP). Frozen T5/CLIP weights come from the pretrained encoders the model was built
    with; this overlays the fine-tuned head + CLIP layers 22-23. A non-empty ``unexpected`` list or
    any shape mismatch means the target config does not match the checkpoint.
    """
    raw = torch.load(path, map_location="cpu", weights_only=False)
    cleaned = {(k[7:] if k.startswith("module.") else k): v for k, v in raw.items()}
    model_keys = set(model.state_dict().keys())

    def _candidates(k: str):
        # Robust to the transformers 4.37 <-> 5.x CLIPVisionModel naming change: 4.37 nests the
        # vision tower under ``vision_model.`` while 5.x does not. Try both directions.
        yield k
        yield k.replace("image_encoder.vision_model.", "image_encoder.")
        yield k.replace("image_encoder.", "image_encoder.vision_model.", 1)

    remapped: dict = {}
    unexpected = []
    for k, v in cleaned.items():
        for cand in _candidates(k):
            if cand in model_keys:
                remapped[cand] = v
                break
        else:
            unexpected.append(k)

    missing, _ = model.load_state_dict(remapped, strict=False)  # raises on shape mismatch
    report = {"loaded": len(remapped), "unexpected": unexpected, "ckpt_keys": len(cleaned)}
    if verbose:
        print(f"[import_upstream_mids] loaded {len(remapped)}/{len(cleaned)} checkpoint tensors; "
              f"unexpected={len(unexpected)}; model keys not in ckpt (frozen base)={len(missing)}")
        if unexpected:
            print(f"  WARNING unexpected keys (first 8): {unexpected[:8]}")
    return report


def load_checkpoint(path: str, device: str = "cpu", overrides: dict | None = None) -> tuple[MIDSPlus, MidsPlusConfig]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = MidsPlusConfig.from_dict({**payload["config"], **(overrides or {})})
    model = build_model(cfg)
    model.load_trainable_state_dict(payload["state"], strict=False)
    model.to(device)
    model.eval()
    return model, cfg
