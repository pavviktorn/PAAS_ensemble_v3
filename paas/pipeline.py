"""PAAS_ensemble_v3 unified inference pipeline.

Loads exactly the detectors a given experiment needs (derived from ``fusion.components``), scores
frames with each, fuses the per-frame component fake-scores by mean/weighted, and applies the
decision rule. This is the single entry point every script / the API goes through, so swapping the
combination is a config change, not a code change.

Components -> producing model:
    ffaa                     -> FFAAModel.score_frames()["fake"]
    A1_9c / A2_9c / A3_9c    -> Ensemble9Model.score_frames()["per_model"][name]
    gsd                      -> GSDModel.score_frames()["fake"]
    selop                    -> SeLopModel.score_frames()["fake"]
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from . import env, fusion as F
from .config import PaasConfig
from .decision import decide


class PaasPipeline:
    def __init__(self, cfg: PaasConfig):
        cfg.validate()
        env.setup(cfg.device)
        self.cfg = cfg
        need = cfg.needs()
        self.need = need
        self.ens = self.ffaa = self.gsd = self.selop = None

        if need["ens"] and cfg.ensemble9.enabled:
            from .config import ENS_MEMBERS
            from .models.ensemble9_model import Ensemble9Model
            # load ONLY the 9-class members the fusion actually uses (e.g. skip A3_9c)
            used = [c for c in cfg.fusion.components if c in ENS_MEMBERS]
            self.ens = Ensemble9Model(cfg.ensemble9.config_path, device="cuda:0", members=used)
        if need["ffaa"] and cfg.ffaa.enabled:
            from .models.ffaa_model import FFAAModel
            cache = self._load_ffaa_cache(cfg.ffaa.cache_path)
            self.ffaa = FFAAModel(cfg.ffaa, device="cuda:0", cache=cache)
        if need["gsd"] and cfg.gsd.enabled:
            from .models.gsd_model import GSDModel
            self.gsd = GSDModel(cfg.gsd.ckpt, cfg.gsd.clip_path, device="cuda:0",
                                amp_dtype=cfg.gsd.amp_dtype)
        if need["selop"] and cfg.selop.enabled:
            from .models.selop_model import SeLopModel
            self.selop = SeLopModel(cfg.selop.ckpt, cfg.selop.clip_path, device="cuda:0",
                                    amp_dtype=cfg.selop.amp_dtype)

    @staticmethod
    def _load_ffaa_cache(path: Optional[str]) -> Optional[dict]:
        """Load a MIDS-format JSON [{image, answers:[...]}] into {abspath(image): answers}."""
        if not path:
            return None
        import json
        import os
        with open(path) as fh:
            recs = json.load(fh)
        cache = {os.path.abspath(r["image"]): r["answers"] for r in recs if r.get("answers")}
        print(f"[paas] FFAA answer cache: {len(cache)} entries from {path}")
        return cache

    # ------------------------------------------------------------------ scoring ----------------
    def predict_frames(self, rgb_list: List[np.ndarray], keys: Optional[List[str]] = None,
                       ens_batch_size: int = 32, ffaa_batch_size: int = 8,
                       gsd_batch_size: int = 64, selop_batch_size: int = 64) -> List[dict]:
        n = len(rgb_list)
        ens = self.ens.score_frames(rgb_list, batch_size=ens_batch_size) if self.ens else [None] * n
        ffaa = (self.ffaa.score_frames(rgb_list, batch_size=ffaa_batch_size, keys=keys)
                if self.ffaa else [None] * n)
        gsd = self.gsd.score_frames(rgb_list, batch_size=gsd_batch_size) if self.gsd else [None] * n
        selop = self.selop.score_frames(rgb_list, batch_size=selop_batch_size) if self.selop else [None] * n

        comps = self.cfg.fusion.components
        results = []
        for i in range(n):
            er = ens[i] or {}
            fr = ffaa[i] or {}
            gr = gsd[i] or {}
            sr = selop[i] or {}
            per = er.get("per_model") or {}

            comp_scores, err = {}, None
            for c in comps:
                if c == "ffaa":
                    v = fr.get("fake")
                elif c in ("A1_9c", "A2_9c", "A3_9c"):
                    v = per.get(c)
                elif c == "gsd":
                    v = gr.get("fake")
                elif c == "selop":
                    v = sr.get("fake")
                else:
                    v = None
                if v is None:
                    err = (er.get("error") or fr.get("error") or gr.get("error")
                           or sr.get("error") or f"missing component '{c}'")
                    break
                comp_scores[c] = float(v)

            if err is not None:
                results.append({"decision": "error", "error": err, "components": comp_scores})
                continue

            fused = F.fuse_scalar_components(comp_scores, self.cfg.fusion)
            # forgery-type source: 9-class marginal if present, else GSD's 3-class marginal
            type_probs = er.get("type_probs") or gr.get("type_probs")
            d = decide(fused, self.cfg.decision,
                       type_probs=type_probs, ffaa_forgery_type=fr.get("forgery_type"))
            d.update({"components": comp_scores,
                      "ensemble_fake": er.get("fake"), "ffaa_fake": fr.get("fake"),
                      "gsd_fake": gr.get("fake"), "selop_fake": sr.get("fake"),
                      "ensemble_per_model": per or None,
                      "ffaa_analysis": fr.get("analysis"), "ffaa_match": fr.get("match")})
            results.append(d)
        return results

    def predict_images(self, paths: List[str], **kw) -> List[dict]:
        """Score image files on disk (RGB-loaded). Unreadable files yield decision='error'."""
        import cv2
        rgb, ok_idx = [], []
        out = [None] * len(paths)
        for i, p in enumerate(paths):
            im = cv2.imread(p, cv2.IMREAD_COLOR)
            if im is None:
                out[i] = {"decision": "error", "error": "unreadable", "image": p}
            else:
                rgb.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); ok_idx.append(i)
        if rgb:
            scored = self.predict_frames(rgb, keys=[paths[i] for i in ok_idx], **kw)
            for j, i in enumerate(ok_idx):
                out[i] = {**scored[j], "image": paths[i]}
        return out
