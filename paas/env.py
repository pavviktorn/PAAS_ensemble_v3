"""Environment + path setup for PAAS_ensemble_v2.

Everything in this project is designed to run on the GLOBAL interpreter
``python3.12`` with ``transformers==4.37.2`` (the FFAA / LLaVA-Mistral stack) -- NOT a venv.
Import this module (and call :func:`setup`) before importing any FFAA or 9-class code so that
(a) the vendored ``ffaa/`` and ``ensemble9/`` trees are importable, (b) the CUDA device is
chosen *before* the FFAA modules pin ``CUDA_VISIBLE_DEVICES``, and (c) HF stays offline/quiet.
"""
from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FFAA_DIR = os.path.join(PROJECT_ROOT, "ffaa")
ENSEMBLE9_DIR = os.path.join(PROJECT_ROOT, "ensemble9")

# Bundled assets (self-contained; see README "Layout").
BASE_CLIP = os.path.join(PROJECT_ROOT, "base_models", "clip-vit-large-patch14-336")
BASE_T5 = os.path.join(PROJECT_ROOT, "base_models", "t5-base")
LLAVA_DIR = os.path.join(PROJECT_ROOT, "weights", "ffaa_llava_mids")
MIDS_PATH = os.path.join(LLAVA_DIR, "mids.pth")
ENSEMBLE9_CONFIG = os.path.join(PROJECT_ROOT, "config", "ensemble9.json")

# GSD (Exp 10) and SeLop/LROR (Exp 11) - the two CLIP detectors added in v3. Both reuse the shared
# BASE_CLIP backbone above (the CLIP vision weights are identical across all members).
GSD_CKPT = os.path.join(PROJECT_ROOT, "weights", "gsd", "best_lastN_ep0_auc0.9930.pt")
SELOP_CKPT = os.path.join(PROJECT_ROOT, "weights", "selop", "best.pt")


def setup(device: str | None = None, quiet: bool = True) -> None:
    """Make the vendored trees importable and pin the CUDA device.

    ``device`` like ``"cuda:0"`` / ``"cuda:2"`` / ``"cpu"``. When a cuda index is given we set
    ``CUDA_VISIBLE_DEVICES`` to that physical index and the process then sees it as ``cuda:0`` --
    this is what lets FFAA's ``models.py`` (which calls ``setdefault('CUDA_VISIBLE_DEVICES','0')``)
    land on the device we want, and is the basis for the multi-GPU file-sharding scripts.
    """
    if device and device.startswith("cuda:"):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", device.split(":", 1)[1])
    if quiet:
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    for d in (FFAA_DIR, ENSEMBLE9_DIR, PROJECT_ROOT):   # PROJECT_ROOT -> `import gsd` / `import selop`
        if d not in sys.path:
            sys.path.insert(0, d)


def visible_device() -> str:
    """The device string to hand torch *after* CUDA_VISIBLE_DEVICES has been pinned (always cuda:0
    when a single physical GPU was selected)."""
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
