#!/usr/bin/env python
"""Rebuild ONLY the REAL subsets of the 30K test set with a face-quality restriction.

Per user: real frames must be FRONTAL, PROPER face size (not too small / not too big), and NOT
heavily posed / occluded. We use insightface SCRFD+3D-landmark (the same detector family as FFAA's
extract_faces.py) to FILTER frames — we do NOT crop; kept frames stay full-image. PAD and DEEPFAKE
subsets are untouched. The previous (unfiltered) real subsets + frames are DELETED first.

Filter (on the 1280-downscaled frame): exactly one clearly-dominant face, det_score >= SCORE,
face-height fraction in [HMIN,HMAX], min(face_w,face_h) >= MINPX, and |pitch|,|yaw|,|roll| <= POSE.
"""
import cv2, glob, json, os, shutil, numpy as np
from multiprocessing import Pool

A = "/datasets/work/vLLM/data/axonlabs_data_1"
OUT = "runs/real/heldout3"
FR = f"{OUT}/frames/real"
MAXSIDE = 1280
CAND_PER_VIDEO = 150          # candidate frames sampled per video before filtering
KEEP_CAP_PER_VIDEO = 55       # max kept per video (diversity); ~55*177 + photos ~= 10K
SCORE = 0.65                  # detector confidence (occlusion/quality proxy)
HMIN, HMAX = 0.15, 0.85       # face-height fraction of frame (not too small / not too big)
MINPX = 80                    # min face side in px on the 1280 frame
POSE = 28.0                   # max |pitch|,|yaw|,|roll| degrees (frontal)

_app = None
def _init():
    global _app
    from insightface.app import FaceAnalysis
    _app = FaceAnalysis(name="buffalo_l", allowed_modules=['detection', 'landmark_3d_68'],
                        providers=['CPUExecutionProvider'])
    _app.prepare(ctx_id=-1, det_size=(640, 640))

def _downscale(fr):
    h, w = fr.shape[:2]
    sc = MAXSIDE / max(h, w)
    return cv2.resize(fr, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA) if sc < 1 else fr

def _passes(img):
    """Return True if img has one dominant, frontal, proper-size, unoccluded face."""
    h, w = img.shape[:2]
    faces = _app.get(img)
    if not faces:
        return False
    faces.sort(key=lambda f: -(f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    f = faces[0]
    x1, y1, x2, y2 = f.bbox
    fw, fh = x2 - x1, y2 - y1
    area = fw * fh
    # single dominant face (reject clutter/occluders)
    if len(faces) > 1:
        f2 = faces[1]; a2 = (f2.bbox[2] - f2.bbox[0]) * (f2.bbox[3] - f2.bbox[1])
        if a2 > 0.5 * area:
            return False
    if f.det_score < SCORE:
        return False
    if not (HMIN <= fh / h <= HMAX):
        return False
    if min(fw, fh) < MINPX:
        return False
    pose = getattr(f, "pose", None)
    if pose is None or float(np.max(np.abs(pose))) > POSE:
        return False
    return True

def process_video(args):
    vpath, subset, idp = args
    try:
        cap = cv2.VideoCapture(vpath)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n <= 0:
            cap.release(); return []
        lo, hi = int(0.05 * n), (int(0.95 * n) if n > 10 else n - 1); hi = max(hi, lo)
        k = min(CAND_PER_VIDEO, max(1, hi - lo + 1))
        idxs = sorted(set(int(lo + (hi - lo) * j / max(k - 1, 1)) for j in range(k)))
        kept = []
        for j, fi in enumerate(idxs):
            if len(kept) >= KEEP_CAP_PER_VIDEO:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, fr = cap.read()
            if not ok or fr is None:
                continue
            img = _downscale(fr)
            if _passes(img):
                op = f"{FR}/{subset}__{idp}_f{j}.jpg"
                cv2.imwrite(op, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                kept.append({"id": f"{idp}_f{j}", "image": op, "cls_label": 0, "subset": subset})
        cap.release()
        return kept
    except Exception:
        return []

def process_photo(args):
    path, idp = args
    try:
        img = _downscale(cv2.imread(path, cv2.IMREAD_COLOR))
        if img is None or not _passes(img):
            return None
        op = f"{FR}/real_photo__{idp}.jpg"
        cv2.imwrite(op, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return {"id": f"rp{idp}", "image": op, "cls_label": 0, "subset": "real_photo"}
    except Exception:
        return None

def vids(root):
    out = []
    for ext in ("*.mp4", "*.MOV", "*.mov", "*.MP4"):
        out += glob.glob(f"{root}/**/{ext}", recursive=True)
    return sorted(set(out))

def main():
    # ---- DELETE current real test set (buckets + frames) ----
    removed = 0
    for f in glob.glob(f"{OUT}/buckets/real_*.json"):
        os.remove(f); removed += 1
    if os.path.isdir(FR):
        shutil.rmtree(FR)
    os.makedirs(FR, exist_ok=True)
    print(f"deleted {removed} old real bucket(s) + old real frames", flush=True)

    ids = sorted(os.listdir(f"{A}/real/Reals_6ids_177video"))
    vjobs = []
    for i in ids:
        for vi, v in enumerate(vids(f"{A}/real/Reals_6ids_177video/{i}")):
            vjobs.append((v, f"real_{i}", f"{i}_{vi}"))
    pjobs = [(p, k) for k, p in enumerate(sorted(glob.glob(f"{A}/real/photo_reals_replay_mobile_460ids/**/*.jpg", recursive=True)))]
    print(f"filtering {len(vjobs)} real videos + {len(pjobs)} photos "
          f"(POSE<={POSE}, score>={SCORE}, hfrac[{HMIN},{HMAX}])", flush=True)

    import collections
    buckets = collections.defaultdict(list)
    nproc = min(16, (os.cpu_count() or 8))
    with Pool(nproc, initializer=_init) as p:
        done = 0
        for recs in p.imap_unordered(process_video, vjobs, chunksize=2):
            for r in recs: buckets[r["subset"]].append(r)
            done += 1
            if done % 30 == 0:
                print(f"  videos {done}/{len(vjobs)} -> {sum(len(v) for v in buckets.values())} kept", flush=True)
        vid_kept = sum(len(v) for v in buckets.values())
        print(f"  real_video kept: {vid_kept}", flush=True)
        for r in p.imap_unordered(process_photo, pjobs, chunksize=8):
            if r: buckets["real_photo"].append(r)
    print(f"  real_photo kept: {len(buckets['real_photo'])}/{len(pjobs)}", flush=True)

    # ---- write new real buckets ----
    for sub, items in buckets.items():
        json.dump(items, open(f"{OUT}/buckets/{sub}.json", "w"))
    # refresh manifest real entries (keys unchanged; ensure present)
    man = json.load(open(f"{OUT}/manifest.json"))
    man = {k: v for k, v in man.items() if not k.startswith("real")}
    for sub in buckets:
        man[sub] = {"group": "REAL", "label": 0}
    json.dump(man, open(f"{OUT}/manifest.json", "w"), indent=1)

    print("\n=== NEW REAL SUBSETS ===")
    tot = 0
    for sub in sorted(buckets):
        print(f"  {sub:16s} {len(buckets[sub]):6d}"); tot += len(buckets[sub])
    print(f"REAL total (filtered) = {tot}")
    open(f"{OUT}/DONE_REALBUILD", "w").close()

if __name__ == "__main__":
    main()
