"""MIDS++ model.

Drop-in for FFAA's ``mids.mids_arch.MIDS``: identical ``forward(texts, image, labels, batch_size,
N, M)`` contract and identical 4-class logits, so the existing FFAA inference + ``make_decision``
pipeline works unchanged.  Internally it adds the three adopted ideas:

* CLIP adaptation via Effort SVD residual adapters (``clip_adapt="svd"``) instead of unfreezing
  whole layers, optionally with GenD-style LayerNorm tuning.
* a GenD image embedding (L2-normalised projected CLS) exposed for alignment/uniformity losses.
* a ForensicsAdapter-style artifact head whose evidence token becomes a 7th fusion token.

``forward`` additionally accepts ``cls_labels`` (per-image authenticity) used only to compute the
auxiliary training losses; it defaults to ``None`` so upstream-style calls remain valid.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from .artifact import LocalArtifactHead
from .basic_module import MultiHeadSelfAttention
from .config import MidsPlusConfig
from .encoders import build_image_encoder, build_text_encoder, image_hidden_size
from .fusion import MultiModalFusionBlock
from .gend import l2_normalize
from .svd import replace_linears_with_svd, svd_regularization_losses


class MIDSPlus(nn.Module):
    def __init__(self, cfg: MidsPlusConfig) -> None:
        super().__init__()
        self.cfg = cfg
        dim = cfg.dim

        self.text_encoder = build_text_encoder(cfg)
        self.image_encoder = build_image_encoder(cfg)
        h = image_hidden_size(self.image_encoder)

        self.projector_layer1 = nn.Linear(h, dim)
        self.projector_layer2 = nn.Linear(h, dim)
        self.cls_token_layer1 = nn.Linear(dim, dim)
        self.cls_token_layer2 = nn.Linear(dim, dim)

        self.fusion_block = MultiModalFusionBlock(dim)
        self.se_attn = nn.ModuleList([MultiHeadSelfAttention(dim, 8) for _ in range(3)])

        self.artifact_enabled = bool(cfg.artifact_enabled)
        if self.artifact_enabled:
            self.artifact_head = LocalArtifactHead(dim, cfg.artifact_num_queries, cfg.artifact_num_heads)

        n_tokens = 6 + (1 if self.artifact_enabled else 0)
        self.n_fusion_tokens = n_tokens
        self.classifier = nn.Linear(dim * n_tokens, cfg.num_classes)

        self.svd_report = None
        self._configure_clip_adaptation()
        self._register_ce_weight()

    # ------------------------------------------------------------------ setup ------------------
    def _register_ce_weight(self) -> None:
        if self.cfg.ce_class_weights is not None:
            w = torch.tensor(self.cfg.ce_class_weights, dtype=torch.float32)
            assert w.numel() == self.cfg.num_classes, "ce_class_weights length must equal num_classes"
            self.register_buffer("ce_weight_vec", w, persistent=False)
        else:
            self.ce_weight_vec = None

    def _configure_clip_adaptation(self) -> None:
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        for p in self.image_encoder.parameters():
            p.requires_grad = False

        mode = self.cfg.clip_adapt
        if mode == "svd":
            self.svd_report = replace_linears_with_svd(
                self.image_encoder,
                target_last_layers=self.cfg.svd_target_last_layers,
                targets=self.cfg.svd_targets,
                residual_rank=self.cfg.svd_residual_rank,
            )
        elif mode == "unfreeze":
            self._unfreeze_vision_last_layers(self.cfg.unfreeze_vision_last_layers)
        elif mode == "frozen":
            pass
        else:
            raise ValueError(f"Unknown clip_adapt mode: {mode!r}")

        if self.cfg.tune_layer_norm:
            self._unfreeze_layernorms(self.image_encoder)

    def _vision_layers(self) -> nn.ModuleList:
        if hasattr(self.image_encoder, "vision_model"):
            return self.image_encoder.vision_model.encoder.layers
        return self.image_encoder.encoder.layers

    def _unfreeze_vision_last_layers(self, last_layers: int) -> None:
        if last_layers <= 0:
            return
        for p in self._vision_layers()[-last_layers:].parameters():
            p.requires_grad = True

    @staticmethod
    def _unfreeze_layernorms(module: nn.Module) -> None:
        for m in module.modules():
            if isinstance(m, nn.LayerNorm):
                for p in m.parameters(recurse=False):
                    p.requires_grad = True

    # ------------------------------------------------------------------ encode -----------------
    def _encode_text(self, inputs) -> torch.Tensor:
        return self.text_encoder(**inputs).last_hidden_state

    @torch.no_grad()
    def encode_text_features(self, inputs) -> torch.Tensor:
        """Precompute T5 features for a FIXED set of tokenized prompts -> (K, L, dim).

        When the texts never change (e.g. the 9-class claim prompts), call this once and
        pass the returned tensor to ``forward`` as ``texts`` to skip the (redundant) T5
        encode on every step. Output is identical to what ``forward`` would compute from
        the same tokenized inputs.
        """
        return self._encode_text(inputs)

    def _text_features(self, texts, batch_size: int, samples: int, dim: int) -> torch.Tensor:
        """Return (B, S, L, dim) text features from either a tokenized dict (runs T5) or a
        precomputed tensor (skips T5)."""
        if torch.is_tensor(texts):
            tf = texts
            if tf.dim() == 4:                              # already (B, S, L, dim)
                return tf
            if tf.dim() == 3:                              # (K, L, dim)
                k, L = tf.size(0), tf.size(1)
                if k == samples:                           # fixed prompts, shared across the batch
                    return tf.unsqueeze(0).expand(batch_size, samples, L, dim)
                if k == batch_size * samples:              # per (image, answer), image-major
                    return tf.reshape(batch_size, samples, L, dim)
            raise ValueError(
                f"precomputed text features shape {tuple(tf.shape)} incompatible with "
                f"batch_size={batch_size}, samples={samples}"
            )
        # default path: tokenized inputs -> run the frozen T5 text encoder
        return self._encode_text(texts).reshape(batch_size, samples, -1, dim)

    def _encode_image(self, pixel_values: torch.Tensor):
        out = self.image_encoder(pixel_values, output_hidden_states=True)
        return out.hidden_states[-1], out.hidden_states[-2]

    # ------------------------------------------------------------------ forward ----------------
    def forward(
        self,
        texts,
        image: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        N: int = 1,
        M: int = 1,
        cls_labels: Optional[torch.Tensor] = None,
    ) -> dict:
        samples = 1 + N + M
        dim = self.cfg.dim

        # texts: tokenized dict -> T5 encode; or a precomputed (K,L,dim)/(B,S,L,dim) tensor -> skip T5
        texts_features = self._text_features(texts, batch_size, samples, dim)  # (B, S, L, dim)

        layer1, layer2 = self._encode_image(image)                       # (B, T, H)
        layer1 = self.projector_layer1(layer1)                           # (B, T, dim)
        layer2 = self.projector_layer2(layer2)

        # GenD image embedding (projected CLS, L2-normalised) -- for alignment/uniformity.
        image_embed = l2_normalize(layer1[:, 0, :])                      # (B, dim)

        # CLS tokens for the classifier, expanded over the answer dimension.
        cls_token1 = self.cls_token_layer1(layer1[:, 0:1, :])
        cls_token2 = self.cls_token_layer2(layer2[:, 0:1, :])
        cls_token1 = cls_token1.unsqueeze(1).expand(batch_size, samples, -1, dim).reshape(batch_size * samples, -1, dim)
        cls_token2 = cls_token2.unsqueeze(1).expand(batch_size, samples, -1, dim).reshape(batch_size * samples, -1, dim)

        # Local artifact branch (text-independent visual evidence).
        artifact_map = None
        artifact_query_logits = None
        tokens: List[torch.Tensor] = [cls_token1, None, None, cls_token2, None, None]
        if self.artifact_enabled:
            patch_tokens = layer1[:, 1:, :]                              # (B, P, dim)
            artifact_token, artifact_map, artifact_query_logits = self.artifact_head(patch_tokens)
            artifact_token = artifact_token.unsqueeze(1).expand(batch_size, samples, -1).reshape(batch_size * samples, 1, dim)
            tokens.append(artifact_token)

        tii1, itt1, tii2, itt2 = self.fusion_block(layer1, layer2, texts_features)
        tokens[1], tokens[2], tokens[4], tokens[5] = tii1, itt1, tii2, itt2

        fused = torch.cat(tokens, dim=1)                                 # (B*S, n_tokens, dim)
        for layer in self.se_attn:
            fused = layer(fused)
        logits = self.classifier(torch.flatten(fused, start_dim=1))      # (B*S, num_classes)

        cls_loss = None
        if labels is not None:
            cls_loss = self._cls_loss(logits, labels)

        logits = logits.reshape(batch_size, samples, self.cfg.num_classes)
        return {
            "logits": logits,
            "cls_loss": cls_loss,
            "image_embed": image_embed,
            "artifact_map": artifact_map,
            "artifact_query_logits": artifact_query_logits,
        }

    def _cls_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        weight = None
        if self.ce_weight_vec is not None:
            weight = self.ce_weight_vec.to(logits.device, logits.dtype)
        return nn.functional.cross_entropy(
            logits, labels, weight=weight, ignore_index=-1, label_smoothing=self.cfg.label_smoothing
        )

    # ------------------------------------------------------------------ regularisers -----------
    def regularization_losses(self) -> dict:
        if self.cfg.clip_adapt == "svd":
            return svd_regularization_losses(self.image_encoder)
        device = next(self.parameters()).device
        return {"orth": torch.zeros((), device=device), "keep": torch.zeros((), device=device)}

    # ------------------------------------------------------------------ checkpoint -------------
    def trainable_state_dict(self) -> dict:
        """Only the learned tensors.

        The frozen pretrained T5/CLIP weights (and the SVD principal component ``weight_main``,
        which is deterministically re-derived from the original CLIP weight at load time) are not
        saved -- so checkpoints stay small and reload on top of fresh encoders.
        """
        trainable = {n for n, p in self.named_parameters() if p.requires_grad}
        return {n: t.detach().cpu() for n, t in self.state_dict().items() if n in trainable}

    def load_trainable_state_dict(self, state: dict, strict: bool = True) -> None:
        # Drop keys that are absent or shape-mismatched (e.g. the classifier head when warm-starting
        # a 9-class model from a 4-class checkpoint) so the rest of the backbone still loads.
        #
        # PAAS fix: be robust to the transformers 4.37 <-> 5.x CLIPVisionModel key-naming change.
        # 4.37 nests the vision tower under ``vision_model.`` (``image_encoder.vision_model.encoder...``)
        # while 5.x flattens it (``image_encoder.encoder...``). A checkpoint trained under one stack is
        # loaded under the other here, so every saved key is tried under both spellings. Without this,
        # the trained Effort-SVD residuals + GenD LayerNorms silently fail to match and the model runs
        # with un-adapted CLIP -- the load "fault" seen under python3.12 / transformers==4.37.2.
        own = self.state_dict()
        filtered, dropped = {}, []
        for k, v in state.items():
            for cand in _image_encoder_key_candidates(k):
                if cand in own and own[cand].shape == v.shape:
                    filtered[cand] = v
                    break
            else:
                dropped.append(k)
        if dropped:
            print(f"[load_trainable_state_dict] dropped {len(dropped)} mismatched/unknown key(s) "
                  f"(e.g. {dropped[:4]})")
        missing, unexpected = self.load_state_dict(filtered, strict=False)
        # The frozen base encoders + reconstructed SVD principal are expected to be "missing".
        unexpected = [u for u in unexpected]
        if strict and unexpected:
            raise RuntimeError(f"Unexpected keys when loading trainable state: {unexpected[:8]}")

    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _image_encoder_key_candidates(k: str):
    """Yield spellings of a state-dict key across the transformers 4.37<->5.x CLIP naming change."""
    yield k
    yield k.replace("image_encoder.vision_model.", "image_encoder.")
    yield k.replace("image_encoder.", "image_encoder.vision_model.", 1)


def build_model(cfg: MidsPlusConfig) -> MIDSPlus:
    return MIDSPlus(cfg)
