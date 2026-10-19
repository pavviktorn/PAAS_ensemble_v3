"""Real-image face-quality filter for the batch tester's optional ``--filter-real``.

Keeps only frames with ONE dominant, frontal, proper-size, unoccluded face (insightface buffalo_l:
SCRFD detection + 3D-68 landmark/pose, on CPU). Used to skip low-quality REAL frames (heavy head
pose / wrong face size / occluded / multi-face) so they are excluded from accuracy; never applied to
fakes. Mirrors the thresholds in ffaa/test_video_image_batch.py and mids_plus's build_real_filtered.

Interface matches the call site in ``test_video_image_batch.py``: a no-argument constructor (sensible
defaults, overridable via FACE_* env vars or kwargs) and ``passes(rgb) -> bool`` taking an RGB frame.
"""
from __future__ import annotations

import os


def _envf(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


class FaceQualityFilter:
    def __init__(self, score: float | None = None, pose: float | None = None,
                 hmin: float | None = None, hmax: float | None = None,
                 minpx: int | None = None, maxside: int | None = None, det_size: int = 640) -> None:
        from insightface.app import FaceAnalysis
        # canonical defaults (match ffaa/mids_plus); override via kwargs or FACE_* env vars
        self.score = _envf("FACE_SCORE", 0.65) if score is None else float(score)
        self.pose = _envf("FACE_POSE", 28.0) if pose is None else float(pose)
        self.hmin = _envf("FACE_HMIN", 0.15) if hmin is None else float(hmin)
        self.hmax = _envf("FACE_HMAX", 0.85) if hmax is None else float(hmax)
        self.minpx = int(_envf("FACE_MINPX", 80)) if minpx is None else int(minpx)
        self.maxside = int(_envf("FACE_MAXSIDE", 1280)) if maxside is None else int(maxside)
        self.app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "landmark_3d_68"],
                                providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=-1, det_size=(det_size, det_size))

    def _downscale(self, bgr):
        import cv2
        h, w = bgr.shape[:2]
        sc = self.maxside / max(h, w)
        return cv2.resize(bgr, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA) if sc < 1 else bgr

    def passes(self, rgb) -> bool:
        """Return True to KEEP the frame (one dominant frontal proper-size face), else False to skip.
        Accepts an HxWx3 uint8 **RGB** frame (as produced by the batch tester)."""
        import cv2
        import numpy as np
        try:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            img = self._downscale(bgr)
            h = img.shape[0]
            faces = self.app.get(img)
            if not faces:
                return False                                   # noface
            faces.sort(key=lambda f: -(f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            f = faces[0]
            x1, y1, x2, y2 = f.bbox
            fw, fh = x2 - x1, y2 - y1
            area = fw * fh
            if len(faces) > 1:                                 # a second comparably-large face
                f2 = faces[1]
                if (f2.bbox[2] - f2.bbox[0]) * (f2.bbox[3] - f2.bbox[1]) > 0.5 * area:
                    return False                               # multiface
            if f.det_score < self.score:
                return False                                   # lowdet (occlusion/quality proxy)
            if not (self.hmin <= fh / h <= self.hmax):
                return False                                   # wrong face size
            if min(fw, fh) < self.minpx:
                return False                                   # too small
            pose = getattr(f, "pose", None)
            if pose is None or float(np.max(np.abs(pose))) > self.pose:
                return False                                   # non-frontal
            return True
        except Exception:
            # never let the optional filter crash inference; treat as keep on unexpected error
            return True
