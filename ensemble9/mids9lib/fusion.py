"""Image <-> answer cross-modal fusion block.

Reproduced from FFAA's ``MultiModalFusionBlock`` (kept identical so the fusion behaviour matches
upstream MIDS): for each of the two CLIP feature layers it runs bidirectional cross-attention
between the answer-text tokens and the image tokens, then distils each direction down to a single
vector with a learnable query token.  The four resulting vectors (text->image and image->text,
per layer) join the CLS tokens (and, in MIDS++, the artifact-evidence token) in the classifier.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .basic_module import MultiHeadCrossAttention


class MultiModalFusionBlock(nn.Module):
    def __init__(self, dim: int = 768, num_heads: int = 8) -> None:
        super().__init__()
        self.cr_tii1 = MultiHeadCrossAttention(dim, num_heads)
        self.cr_tii2 = MultiHeadCrossAttention(dim, num_heads)
        self.cr_itt1 = MultiHeadCrossAttention(dim, num_heads)
        self.cr_itt2 = MultiHeadCrossAttention(dim, num_heads)

        self.lr_tii1 = nn.Parameter(torch.randn(1, dim))
        self.cr_tii1_lr = MultiHeadCrossAttention(dim, num_heads)
        self.lr_tii2 = nn.Parameter(torch.randn(1, dim))
        self.cr_tii2_lr = MultiHeadCrossAttention(dim, num_heads)
        self.lr_itt1 = nn.Parameter(torch.randn(1, dim))
        self.cr_itt1_lr = MultiHeadCrossAttention(dim, num_heads)
        self.lr_itt2 = nn.Parameter(torch.randn(1, dim))
        self.cr_itt2_lr = MultiHeadCrossAttention(dim, num_heads)

    def forward(self, layer1_rgb_feature, layer2_rgb_feature, texts_features):
        batch_size = texts_features.size(0)
        samples_num = texts_features.size(1)
        dim = texts_features.size(3)

        layer1_rgb_features = layer1_rgb_feature.unsqueeze(1).expand(batch_size, samples_num, -1, dim)
        layer1_rgb_features = layer1_rgb_features.reshape(batch_size * samples_num, -1, dim)
        layer2_rgb_features = layer2_rgb_feature.unsqueeze(1).expand(batch_size, samples_num, -1, dim)
        layer2_rgb_features = layer2_rgb_features.reshape(batch_size * samples_num, -1, dim)

        texts_features = texts_features.reshape(batch_size * samples_num, -1, dim)

        tii1 = self.cr_tii1(texts_features, layer1_rgb_features)
        itt1 = self.cr_itt1(layer1_rgb_features, texts_features)
        tii2 = self.cr_tii2(texts_features, layer2_rgb_features)
        itt2 = self.cr_itt2(layer2_rgb_features, texts_features)

        bs = batch_size * samples_num
        tii1_feature = self.cr_tii1_lr(self.lr_tii1.expand(bs, 1, dim), tii1)
        itt1_feature = self.cr_itt1_lr(self.lr_itt1.expand(bs, 1, dim), itt1)
        tii2_feature = self.cr_tii2_lr(self.lr_tii2.expand(bs, 1, dim), tii2)
        itt2_feature = self.cr_itt2_lr(self.lr_itt2.expand(bs, 1, dim), itt2)

        return tii1_feature, itt1_feature, tii2_feature, itt2_feature
