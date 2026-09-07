"""MIDS++ : generalization-hardened drop-in replacement for FFAA's MIDS.

The public model keeps the exact input/output contract of FFAA's ``mids.mids_arch.MIDS``
(image + masked candidate answers in, per-answer 4-class logits out) so it plugs into the
existing FFAA inference pipeline and ``mids.selector.make_decision`` unchanged.

What changed inside, relative to upstream MIDS:

* Effort-style SVD residual adaptation of the CLIP vision encoder replaces the crude
  "unfreeze the last 2 transformer layers" recipe (preserves CLIP's open-world subspace).
* GenD-style hyperspherical alignment/uniformity + L2 discipline on the image embedding.
* A ForensicsAdapter-style weakly-supervised local artifact head whose evidence token is
  injected into the cross-modal fusion (text-independent visual cue for "hard" cases).

See ``docs/`` for the full rationale and the precise diff against upstream.
"""

from .config import MidsPlusConfig

__all__ = ["MidsPlusConfig"]
