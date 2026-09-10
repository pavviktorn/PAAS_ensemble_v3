"""A1+A2+A3 (9-class) MLLM-free ensemble.

Decision recipe (matches the validated configuration; no MLLM, frame-based):
  * Feed the WHOLE image (letterbox, no crop) + 3 fixed candidate answers to each of the 3 MIDS++
    9-class models -> per-model (3 answers x 9 classes) softmax, label = 3*true + claim,
    true/claim in {0 real, 1 pad, 2 deepfake}.
  * Per model fake-score = mean over the 3 answers of P(image is fake) = mean_a [1 - P(true=real)].
  * Ensemble fake-score = mean of the 3 per-model fake-scores  (the calibration fit to identity,
    so this is a plain mean — see config).
  * decision = FAKE if ensemble fake-score >= threshold else REAL.
  * forgery_type = argmax over {pad, deepfake} of the ensemble true-class marginal (when FAKE),
    else "real". (PAD includes makeup, per the training label scheme.)
  * match_score = confidence in the decision = fake-score if FAKE else (1 - fake-score).
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import torch
import torch.nn.functional as F

try:  # keep the encoder load quiet (the bundled CLIP dir also has text weights -> harmless notices)
    import transformers
    transformers.logging.set_verbosity_error()
except Exception:
    pass

from .checkpoint import load_checkpoint
from .tokenize import build_tokenizer
from .transform import letterbox_to_tensor, load_rgb

TYPE_NAMES = ("real", "pad", "deepfake")


class MidsEnsemble:
    def __init__(self, config_path: str = "config.json", device: Optional[str] = None,
                 members: Optional[list] = None):
        """`members`: optional list of member names to load (e.g. ["A1_9c", "A2_9c"]). None loads
        every member in the config. Skipping unused members (the paas fusion doesn't use A3_9c)
        saves their load time, VRAM, and a full forward per frame."""
        config_path = os.path.abspath(config_path)
        root = os.path.dirname(config_path)
        cfg = json.load(open(config_path))
        self.config = cfg
        self.root = root
        self.size = int(cfg.get("image_size", 336))
        self.texts = list(cfg["texts"])
        self.threshold = float(cfg["threshold"])

        if device is None:
            d = cfg.get("device", "auto")
            device = ("cuda" if torch.cuda.is_available() else "cpu") if d == "auto" else d
        self.device = device

        img_base = os.path.join(root, cfg["base_models"]["image"])
        txt_base = os.path.join(root, cfg["base_models"]["text"])

        wanted = cfg["members"]
        if members is not None:
            have = [m.get("name") for m in cfg["members"]]
            missing = [n for n in members if n not in have]
            if missing:
                raise ValueError(f"ensemble9 config has no member(s) {missing}; available: {have}")
            wanted = [m for m in cfg["members"] if m.get("name") in set(members)]

        self.models, self.names = [], []
        tokenizer = None
        t0 = time.time()
        for m in wanted:
            ckpt = os.path.join(root, m["checkpoint"])
            model, mcfg = load_checkpoint(ckpt, device=device, overrides={
                "image_model_path": img_base,
                "text_model_path": txt_base,
                "skip_missing_images": True,
                "stub_encoders": False,
            })
            model.eval()
            self.models.append(model)
            self.names.append(m.get("name", os.path.basename(ckpt)))
            if tokenizer is None:
                tokenizer = build_tokenizer(mcfg)
        self.tokenizer = tokenizer
        self.load_time_sec = round(time.time() - t0, 3)

        enc = tokenizer(self.texts, return_tensors="pt", padding="longest", truncation=True, max_length=512)
        self.ids0 = enc["input_ids"].to(device)
        self.mask0 = enc["attention_mask"].to(device)
        # The claim prompts are FIXED, so encode them through T5 ONCE per model and reuse the
        # (S, L, dim) features on every forward (skips the redundant T5 call). Identical output.
        self.text_feats = [
            m.encode_text_features({"input_ids": self.ids0, "attention_mask": self.mask0})
            for m in self.models
        ]
        self._warmup()

    @torch.no_grad()
    def _warmup(self) -> None:
        """One dummy forward so the first real predict() time is representative (CUDA kernels warm)."""
        try:
            dummy = torch.zeros(1, 3, self.size, self.size, device=self.device)
            for model, tf in zip(self.models, self.text_feats):
                model(tf, dummy, None, 1, 1, 1)
            if str(self.device).startswith("cuda"):
                torch.cuda.synchronize()
        except Exception:
            pass

    @torch.no_grad()
    def predict(self, image_path: str, threshold: Optional[float] = None) -> dict:
        tau = self.threshold if threshold is None else float(threshold)
        t0 = time.time()
        rgb = load_rgb(image_path)
        img = letterbox_to_tensor(rgb, self.size).unsqueeze(0).to(self.device)

        per_model_fake, type_acc = [], torch.zeros(3, device=self.device)
        per_model = {}
        for name, model, tf in zip(self.names, self.models, self.text_feats):
            out = model(tf, img, None, 1, 1, 1)
            sm = F.softmax(out["logits"].float(), dim=2)[0]          # (3 answers, 9)
            Pt = sm.view(sm.size(0), 3, 3).sum(dim=2)                # (3 answers, 3 true) ; sum over claim
            fake = float((1.0 - Pt[:, 0]).mean())                    # mean_a P(fake)=1-P(real)
            per_model_fake.append(fake)
            type_acc += Pt.mean(dim=0)
            per_model[name] = round(fake, 4)

        fused_fake = float(sum(per_model_fake) / len(per_model_fake))
        type_probs = (type_acc / len(self.models)).tolist()
        decision = "fake" if fused_fake >= tau else "real"
        forgery_type = "real" if decision == "real" else ("pad" if type_probs[1] >= type_probs[2] else "deepfake")
        match_score = fused_fake if decision == "fake" else 1.0 - fused_fake
        dt = time.time() - t0

        return {
            "image": image_path,
            "decision": decision,
            "forgery_type": forgery_type,
            "match_score": round(match_score, 4),
            "forgery_score": round(fused_fake, 4),
            "type_probs": {TYPE_NAMES[i]: round(type_probs[i], 4) for i in range(3)},
            "per_model_fake": per_model,
            "threshold": tau,
            "processing_time_sec": round(dt, 4),
        }

    @torch.no_grad()
    def _score_tensor(self, x):
        """x: (B,3,H,W) on self.device -> (fused[list], type_probs[list of [r,p,d]], per_model[list of dict])."""
        B = x.size(0)
        fake_sum = torch.zeros(B, device=self.device)
        type_sum = torch.zeros(B, 3, device=self.device)
        per_model = [{} for _ in range(B)]
        for name, model, tf in zip(self.names, self.models, self.text_feats):
            o = model(tf, x, None, B, 1, 1)                                # tf (S,L,dim) -> expands to (B,S,L,dim)
            sm = F.softmax(o["logits"].float(), dim=2)                    # (B,3,9)
            Pt = sm.view(B, sm.size(1), 3, 3).sum(dim=3)                  # (B,3 answers,3 true)
            fake = (1.0 - Pt[:, :, 0]).mean(dim=1)                        # (B,)
            fake_sum += fake; type_sum += Pt.mean(dim=1)
            for b in range(B):
                per_model[b][name] = round(float(fake[b]), 4)
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        return (fake_sum / len(self.models)).tolist(), (type_sum / len(self.models)).tolist(), per_model

    def _pack(self, image_key, ff, tp_b, per_model_b, tau, dt):
        dec = "fake" if ff >= tau else "real"
        ft = "real" if dec == "real" else ("pad" if tp_b[1] >= tp_b[2] else "deepfake")
        ms = ff if dec == "fake" else 1.0 - ff
        return {
            "image": image_key, "decision": dec, "forgery_type": ft,
            "match_score": round(ms, 4), "forgery_score": round(ff, 4),
            "type_probs": {TYPE_NAMES[k]: round(tp_b[k], 4) for k in range(3)},
            "per_model_fake": per_model_b, "threshold": tau,
            "processing_time_sec": round(dt, 4),
        }

    @torch.no_grad()
    def predict_batch(self, image_paths, threshold: Optional[float] = None, batch_size: int = 32):
        """Batched inference for throughput. Returns a list of result dicts (same fields as predict;
        processing_time_sec is the per-image amortized time). Unreadable images yield {'error': ...}."""
        tau = self.threshold if threshold is None else float(threshold)
        out_all = []
        for i in range(0, len(image_paths), batch_size):
            chunk = image_paths[i:i + batch_size]
            tensors, ok_paths, errs = [], [], []
            for p in chunk:
                try:
                    tensors.append(letterbox_to_tensor(load_rgb(p), self.size)); ok_paths.append(p)
                except Exception as e:
                    errs.append({"image": p, "error": str(e)})
            if tensors:
                t0 = time.time()
                x = torch.stack(tensors).to(self.device)            # (B,3,H,W)
                fused, tp, per_model = self._score_tensor(x)
                dt = (time.time() - t0) / x.size(0)
                for b, p in enumerate(ok_paths):
                    out_all.append(self._pack(p, fused[b], tp[b], per_model[b], tau, dt))
            out_all.extend(errs)
        return out_all

    @torch.no_grad()
    def predict_rgb_batch(self, items, threshold: Optional[float] = None, batch_size: int = 32):
        """Score already-decoded frames (no disk read). `items`: list of (key, rgb_uint8_HxWx3_array).
        Returns result dicts with the key under 'image'; undecodable items yield {'image': key, 'error': ...}.
        Lets video frames be scored in-memory (used by the batch video/image tester)."""
        tau = self.threshold if threshold is None else float(threshold)
        out_all = []
        for i in range(0, len(items), batch_size):
            chunk = items[i:i + batch_size]
            tensors, keys, errs = [], [], []
            for key, rgb in chunk:
                try:
                    tensors.append(letterbox_to_tensor(rgb, self.size)); keys.append(key)
                except Exception as e:
                    errs.append({"image": key, "error": str(e)})
            if tensors:
                t0 = time.time()
                x = torch.stack(tensors).to(self.device)
                fused, tp, per_model = self._score_tensor(x)
                dt = (time.time() - t0) / x.size(0)
                for b, k in enumerate(keys):
                    out_all.append(self._pack(k, fused[b], tp[b], per_model[b], tau, dt))
            out_all.extend(errs)
        return out_all
