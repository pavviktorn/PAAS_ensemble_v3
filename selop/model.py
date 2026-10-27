"""SeLop model: frozen CLIP ViT-L/14 backbone + per-layer LROR + linear head.

- Backbone: CLIP vision transformer (HF `clip-vit-large-patch14-336`), fully frozen.
- LROR is inserted at the INPUT of the last `n_intervene` blocks (default last 12
  of 24).  One independent M per intervened layer.
- The classifier is a single linear layer on the final [CLS] token (post-LN),
  outputting `num_classes` logits.  Trained with plain cross-entropy.

The first (n_layers - n_intervene) blocks have no trainable parameters and sit
before the first LROR, so they are run under `torch.no_grad()` and detached —
this halves the backward cost with no change to the result.
"""

import torch
import torch.nn as nn
from transformers import CLIPVisionModel

from .lror import LROR


class SeLopModel(nn.Module):
    def __init__(self, clip_path, num_classes=2, rank=32, n_intervene=12, init_std=0.02):
        super().__init__()
        self.clip = CLIPVisionModel.from_pretrained(clip_path)
        self.vm = self.clip.vision_model
        for p in self.clip.parameters():
            p.requires_grad_(False)

        dim = self.vm.config.hidden_size
        n_layers = self.vm.config.num_hidden_layers
        assert 0 < n_intervene <= n_layers, f"n_intervene must be in (0, {n_layers}]"
        self.dim = dim
        self.n_layers = n_layers
        self.n_intervene = n_intervene
        self.intervene_start = n_layers - n_intervene  # apply LROR before blocks [start, n_layers)

        self.lror = nn.ModuleList([LROR(dim, rank, init_std) for _ in range(n_intervene)])
        self.head = nn.Linear(dim, num_classes)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)

    def train(self, mode: bool = True):
        # Keep the CLIP backbone in eval mode permanently (no dropout); only the
        # LROR matrices and head follow the requested train/eval mode.
        super().train(mode)
        self.clip.eval()
        return self

    def trainable_parameters(self):
        return list(self.lror.parameters()) + list(self.head.parameters())

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vm = self.vm
        start = self.intervene_start

        # --- frozen prefix: no trainable params, run without building a graph ---
        with torch.no_grad():
            hidden = vm.embeddings(pixel_values)
            hidden = vm.pre_layrnorm(hidden)
            for i in range(start):
                hidden = vm.encoder.layers[i](hidden, None, None)[0]
        hidden = hidden.detach()

        # --- intervened suffix: LROR at the input of each of the last blocks ---
        for i in range(start, self.n_layers):
            hidden = self.lror[i - start](hidden)
            hidden = vm.encoder.layers[i](hidden, None, None)[0]

        cls = hidden[:, 0, :]
        pooled = vm.post_layernorm(cls)
        return self.head(pooled)

    # --- checkpointing: only the tiny trainable part is persisted ---
    def export_state(self):
        return {
            "lror": self.lror.state_dict(),
            "head": self.head.state_dict(),
            "config": {
                "dim": self.dim,
                "n_layers": self.n_layers,
                "n_intervene": self.n_intervene,
                "rank": self.lror[0].rank,
                "num_classes": self.head.out_features,
            },
        }

    def load_trainable(self, state):
        self.lror.load_state_dict(state["lror"])
        self.head.load_state_dict(state["head"])
