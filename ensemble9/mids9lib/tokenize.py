"""Tokenizer factory.

Real mode returns the T5 tokenizer (same as upstream MIDS).  Stub mode returns a tiny
dependency-free char-hash tokenizer producing ``input_ids`` for the stub text encoder, so the
smoke test needs neither ``transformers`` nor the t5-base vocab files.
"""

from __future__ import annotations

from typing import List

import torch


def build_tokenizer(cfg):
    if cfg.stub_encoders:
        return _StubTokenizer(vocab=cfg.stub_text_vocab)
    from transformers import T5Tokenizer

    return T5Tokenizer.from_pretrained(cfg.text_model_path, use_fast=False, legacy=False)


class _StubTokenizer:
    """Maps each character to ``ord(c) % vocab``; pads a batch to the longest sequence."""

    def __init__(self, vocab: int = 256, max_length: int = 64) -> None:
        self.vocab = vocab
        self.max_length = max_length

    def __call__(self, texts: List[str], return_tensors="pt", padding="longest",
                 truncation=True, max_length=None, **_):
        max_length = max_length or self.max_length
        seqs = []
        for t in texts:
            ids = [(ord(c) % self.vocab) for c in (t or " ")][:max_length] or [1]
            seqs.append(ids)
        width = max(len(s) for s in seqs)
        input_ids = torch.zeros(len(seqs), width, dtype=torch.long)
        attention_mask = torch.zeros(len(seqs), width, dtype=torch.long)
        for i, s in enumerate(seqs):
            input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            attention_mask[i, : len(s)] = 1
        return _Encoding({"input_ids": input_ids, "attention_mask": attention_mask})


class _Encoding(dict):
    def to(self, device):
        return _Encoding({k: v.to(device) for k, v in self.items()})
