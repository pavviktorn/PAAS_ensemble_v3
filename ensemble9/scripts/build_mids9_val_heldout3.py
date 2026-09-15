#!/usr/bin/env python
"""Build the 9-class validation set from the heldout3 generalization set.

Mirrors build_mids9.py's schema (3 fixed claim templates, 9-class label = 3*true + claim),
but the items come from runs/real/heldout3 instead of an internal split of the training data:

  REAL  -> the FILTERED real frames in heldout3/frames/real (quality-filtered set, 10126)
  PAD   -> heldout3/frames/pad copied frames
  DEEPFAKE -> the df_* entries in heldout3_ALL.json (original absolute image paths)

true type comes from the subset prefix (real_/pad_/df_), which is authoritative here.
Output: runs/real/mids9/mids9_val.json
"""
import json, os, glob, collections

BASE = "/datasets/work/vLLM/temp/mids_plus"
HELD = os.path.join(BASE, "runs/real/heldout3")
OUT = os.path.join(BASE, "runs/real/mids9/mids9_val.json")

# identical templates/order to build_mids9.py ; claim: 0 real, 1 pad, 2 deepfake
TEMPLATES = [
    (2, "deepfake", "misaligned features, smooth skin, inconsistent lighting, blending artifacts, "
                    "unnatural integration between the face and body/the face and the background"),
    (1, "pad", "flat depth, uniform lighting, screen smoothing, moire patterns, screen borders, "
               "UI, reflections, glossy/plastic skin, rigid edges, hand-holding signs, recapture blur"),
    (0, "real", "consistent facial features, natural skin texture, consistent lighting, "
                "normal facial 3D depth, the absence of clear manipulation or visible artifacts"),
]

def item(id_, image, true, subset):
    return {"id": id_, "image": image,
            "cls_label": 0 if true == 0 else 1,
            "answers": [{"content": txt, "result": name, "label": 3 * true + claim}
                        for (claim, name, txt) in TEMPLATES],
            "subset": subset}

out, cnt = [], collections.Counter()

# REAL: only the FILTERED frames present on disk
for fp in sorted(glob.glob(os.path.join(HELD, "frames/real/*.jpg"))):
    bn = os.path.basename(fp)[:-4]
    out.append(item(bn, fp, 0, bn.split("__")[0])); cnt["real"] += 1

# PAD: copied frames on disk
for fp in sorted(glob.glob(os.path.join(HELD, "frames/pad/*.jpg"))):
    bn = os.path.basename(fp)[:-4]
    out.append(item(bn, fp, 1, bn.split("__")[0])); cnt["pad"] += 1

# DEEPFAKE: original absolute paths from the manifest
for x in json.load(open(os.path.join(HELD, "heldout3_ALL.json"))):
    if not x["subset"].startswith("df"):
        continue
    img = x["image"] if os.path.isabs(x["image"]) else os.path.join(BASE, x["image"])
    if os.path.exists(img):
        out.append(item(x["id"], img, 2, x["subset"])); cnt["deepfake"] += 1
    else:
        cnt["df_missing"] += 1

json.dump(out, open(OUT, "w"))
print(f"built {len(out)} val items -> {OUT}")
print("true-type counts:", dict(cnt))
hist = collections.Counter(a["label"] for x in out for a in x["answers"])
print("9-class label histogram (0..8):", {k: hist.get(k, 0) for k in range(9)})
miss = sum(not os.path.exists(x["image"]) for x in out)
print("items with a missing image file:", miss)
