"""PAAS_v4 -- a focused, faithful implementation of Geometric Semantic Decoupling (GSD).

GSD removes the per-batch *semantic subspace* (estimated from a frozen CLIP via Householder QR) from
a trainable detector's features by orthogonal projection, forcing forgery learning into the semantic
null-space. Reference: "Geometric Semantic Decoupling", arXiv 2603.09242.
"""
from .config import GSDConfig
from .model import GSDDetector
from .householder import semantic_basis
from .projection import desemanticize

__all__ = ["GSDConfig", "GSDDetector", "semantic_basis", "desemanticize"]
