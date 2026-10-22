"""Wrapper around FFAA's MLLM+MIDS model (LLaVA-Mistral-7B + MIDS 4-class head).

Reuses FFAA's own code (vendored ``ffaa/models.py``, ``mids/selector.py``, ``utils/file_utils.py``)
rather than reimplementing the LLaVA generation / answer-masking / MIDS-decision logic. Exposes a
uniform ``score_frames`` returning, per frame, the per-frame fake-score (== make_decision's
forgery_score = match if pred==fake else 1-match), the binary analysis, the match, and the raw
best answer. Call :func:`paas.env.setup` (to pin the device + sys.path) BEFORE constructing this.
"""
from __future__ import annotations

import os
from collections import defaultdict
from typing import List, Optional

import numpy as np
from PIL import Image


def _NM(n_answers: int):
    return {3: (1, 1), 2: (0, 1), 1: (0, 0)}.get(n_answers, (0, 0))


class FFAAModel:
    name = "ffaa"

    def __init__(self, cfg, device: str = "cuda:0", cache: Optional[dict] = None):
        import torch
        from transformers import AutoTokenizer, CLIPProcessor
        import models as ffaa_models                       # vendored ffaa/models.py
        from mids.selector import make_decision, make_decision_batch
        from utils.file_utils import mask_result, decode_response

        self.t = torch
        self.F = torch.nn.functional
        self.cfg = cfg
        self._make_decision = make_decision
        self._make_decision_batch = make_decision_batch
        self._mask_result = mask_result
        self._decode = decode_response
        self._gen_batch = ffaa_models.get_llava_answer_batch
        # {abspath: [{content, result, label}, ...]} of pre-generated answers, or None. LLaVA is
        # loaded regardless so cache MISSES can fall back to the normal generate-then-score path.
        self.cache = cache

        dev_id = 0  # CUDA_VISIBLE_DEVICES was pinned in env.setup -> physical GPU appears as cuda:0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model, self.image_processor, self.tokenizer = ffaa_models.load_llava(cfg.llava_dir, dev_id)
        self.t5_tokenizer = AutoTokenizer.from_pretrained(cfg.t5_path, use_fast=False, legacy=False)
        self.clip_processor = CLIPProcessor.from_pretrained(cfg.clip_path)
        self.mids = ffaa_models.load_mids(cfg.mids_path, dev_id)
        self.mids.eval()

    def score_frames(self, rgb_list: List[np.ndarray], batch_size: int = 8,
                     keys: Optional[List[str]] = None) -> List[dict]:
        """rgb_list: list of HxWx3 uint8 arrays. Returns one dict per frame (input order):
        {"fake": float|None, "analysis": "real"/"fake"|None, "match": float|None,
         "forgery_type": str|None, "answer": str|None, "error": str|None}.

        With a cache loaded, frames whose `keys[i]` (abspath) is in the cache are scored from the
        cached answers via MIDS only (no LLaVA generation); the rest fall back to the live path."""
        pil = [Image.fromarray(r).convert("RGB") for r in rgb_list]
        if self.cache is None:
            return self._score_live(pil, batch_size)

        out: List[Optional[dict]] = [None] * len(pil)
        hit_idx, hit_imgs, hit_recs, miss_idx, miss_imgs = [], [], [], [], []
        for i, img in enumerate(pil):
            key = keys[i] if keys else None
            rec = self.cache.get(os.path.abspath(key)) if key else None
            if rec:
                hit_idx.append(i); hit_imgs.append(img); hit_recs.append(rec)
            else:
                miss_idx.append(i); miss_imgs.append(img)
        if hit_imgs:
            for i, r in zip(hit_idx, self._score_cached(hit_imgs, hit_recs, batch_size)):
                out[i] = r
        if miss_imgs:                                          # cache miss -> normal generate path
            for i, r in zip(miss_idx, self._score_live(miss_imgs, batch_size)):
                out[i] = r
        return out

    def _score_live(self, pil: List[Image.Image], batch_size: int) -> List[dict]:
        """The normal path: LLaVA-generate `generate_num` answers per image, then MIDS-score them."""
        out: List[dict] = []
        c = self.cfg
        for i in range(0, len(pil), batch_size):
            chunk = pil[i:i + batch_size]
            prompts = [c.prompt] * len(chunk)
            try:
                ans_batch = self._gen_batch(
                    self.model, self.tokenizer, self.image_processor, chunk, prompts,
                    c.temperature, c.top_p, c.num_beams, c.max_new_tokens, c.generate_num, c.conv_mode)
            except Exception as e:                              # whole-chunk generation failure
                out.extend({"fake": None, "analysis": None, "match": None,
                            "forgery_type": None, "answer": None, "error": f"generate: {e}"}
                           for _ in chunk)
                continue
            for img, answers in zip(chunk, ans_batch):
                out.append(self._score_one(img, answers))
        return out

    def _score_cached(self, imgs: List[Image.Image], recs: List[list], batch_size: int) -> List[dict]:
        """Score from pre-generated answers (no LLaVA). `recs[i]` is a list of {content, result}.
        `content` is the already-masked T5 input (== mask_result output) and `result` is real/fake,
        so this reproduces _score_one's MIDS step exactly, batched. Bucketed by #answers since the
        MIDS condition shape (N,M) and make_decision chunk size depend on it."""
        out: List[Optional[dict]] = [None] * len(imgs)
        buckets = defaultdict(list)
        for i, rec in enumerate(recs):
            buckets[len(rec)].append(i)
        for nans, idxs in buckets.items():
            N, M = _NM(nans)
            for s in range(0, len(idxs), batch_size):
                grp = idxs[s:s + batch_size]
                bimgs = [imgs[k] for k in grp]
                contents = [a["content"] for k in grp for a in recs[k]]
                results = [(a.get("result") or "fake").lower() for k in grp for a in recs[k]]
                try:
                    with self.t.inference_mode():
                        pix = self.clip_processor(images=bimgs, return_tensors="pt")["pixel_values"].to(self.device)
                        ans_ids = self.t5_tokenizer(contents, return_tensors="pt", padding="longest",
                                                    max_length=self.t5_tokenizer.model_max_length, truncation=True)
                        ans_ids = {k: v.to(self.device) for k, v in ans_ids.items()}
                        logits = self.mids(ans_ids, pix, None, len(bimgs), N, M)["logits"]
                        scores = self.F.softmax(logits, dim=2)            # (B, 1+N+M, 4)
                        bidx, preds, matches, forgeries = self._make_decision_batch(
                            results, scores, chunk_size=nans)
                    for j, k in enumerate(grp):
                        analysis = "real" if int(preds[j]) == 0 else "fake"
                        out[k] = {"fake": float(forgeries[j]), "analysis": analysis,
                                  "match": float(matches[j]),
                                  "forgery_type": "real" if analysis == "real" else None,
                                  "answer": recs[k][int(bidx[j])]["content"], "error": None}
                except Exception as e:                          # whole-chunk MIDS failure
                    for k in grp:
                        out[k] = {"fake": None, "analysis": None, "match": None,
                                  "forgery_type": None, "answer": None, "error": f"cache-score: {e}"}
        return out

    def _score_one(self, img, answers: List[str]) -> dict:
        try:
            answers_result, processed = [], []
            for a in answers:
                masked, res = self._mask_result(a)
                processed.append(masked); answers_result.append(res)
            N, M = _NM(len(answers))
            with self.t.inference_mode():
                pix = self.clip_processor(images=img, return_tensors="pt")["pixel_values"].to(self.device)
                ans_ids = self.t5_tokenizer(processed, return_tensors="pt", padding="longest",
                                            max_length=self.t5_tokenizer.model_max_length, truncation=True)
                ans_ids = {k: v.to(self.device) for k, v in ans_ids.items()}
                logits = self.mids(ans_ids, pix, None, 1, N, M)["logits"]
                scores = self.F.softmax(logits, dim=2).squeeze(0)          # (1+N+M, 4)
                best_idx, pred, match, forgery = self._make_decision(answers_result, scores)
            analysis = "real" if int(pred) == 0 else "fake"
            ftype = None
            try:
                bj, _ = self._decode(answers[best_idx])
                ftype = (bj.get("Forgery type") or None) if analysis == "fake" else "real"
            except Exception:
                pass
            return {"fake": float(forgery), "analysis": analysis, "match": float(match),
                    "forgery_type": ftype, "answer": answers[best_idx], "error": None}
        except Exception as e:
            return {"fake": None, "analysis": None, "match": None,
                    "forgery_type": None, "answer": None, "error": f"score: {e}"}
