#!/usr/bin/env python
"""Build the 9-class (3 true-type x 3 claim-type) training set from mids.json.

True type per image comes from FFAA's canonical get_label_all() (REAL/PAD/DEEPFAKE/MAKEUP/UNKNOWN);
per user, MAKEUP->PAD and UNKNOWN dropped. Each image gets the 3 fixed templates as the 3 claim
anchors (deepfake / pad / real), with 9-class target = 3*true + claim. cls_label stays BINARY
(0 real / 1 fake) for the GenD alignment loss. Output: runs/real/mids9/mids9.json [+ optional limit].

Usage: python scripts/build_mids9.py [--limit N] [--out PATH]
"""
import argparse, json, os, sys, collections, time

sys.path.insert(0, "/datasets/work/vLLM/FFAA-master-newfmt-faster1_transformers4372")
from get_label import get_label_all, REAL, PAD, DEEPFAKE, MAKEUP, UNKNOWN  # noqa

SRC = "/datasets/newout/vqa_info_2+13+4+3_fmt/mids.json"

# 3 fixed templates as claim anchors (text identical to eval_fixed_answers.FIXED).
# (claim_index, claim_name, content) ; claim: 0 real, 1 pad, 2 deepfake
TEMPLATES = [
    (2, "deepfake", "misaligned features, smooth skin, inconsistent lighting, blending artifacts, "
                    "unnatural integration between the face and body/the face and the background"),
    (1, "pad", "flat depth, uniform lighting, screen smoothing, moire patterns, screen borders, "
               "UI, reflections, glossy/plastic skin, rigid edges, hand-holding signs, recapture blur"),
    (0, "real", "consistent facial features, natural skin texture, consistent lighting, "
                "normal facial 3D depth, the absence of clear manipulation or visible artifacts"),
]

def true3(path):
    l = get_label_all(path)
    if l == MAKEUP:
        l = PAD                     # user rule: makeup manipulation -> PAD
    return l                         # REAL0 / PAD1 / DEEPFAKE2 / UNKNOWN(-1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="runs/real/mids9/mids9.json")
    ap.add_argument("--no-val", action="store_true",
                    help="write only the train file (val is built separately from heldout3)")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    data = json.load(open(SRC))
    t = time.time()
    out = []
    cnt = collections.Counter()
    for it in data:
        tl = true3(it["image"])
        if tl == UNKNOWN:
            cnt["dropped_unknown"] += 1
            continue
        cnt[{0: "real", 1: "pad", 2: "deepfake"}[tl]] += 1
        answers = [{"content": txt, "result": name, "label": 3 * tl + claim}
                   for (claim, name, txt) in TEMPLATES]
        out.append({"id": it.get("id"), "image": it["image"],
                    "cls_label": 0 if tl == REAL else 1, "answers": answers})
        if args.limit and len(out) >= args.limit:
            break
    # deterministic disjoint val split (last 2000 after a fixed-stride shuffle)
    val_n = 0 if (args.limit or args.no_val) else 2000
    if val_n and len(out) > val_n:
        val = out[::len(out) // val_n][:val_n]
        valset = set(id(x) for x in val)
        train = [x for x in out if id(x) not in valset]
        json.dump(val, open(os.path.join(os.path.dirname(args.out), "mids9_val.json"), "w"))
        out = train
    json.dump(out, open(args.out, "w"))
    print(f"built {len(out)} train items (+{val_n} val) in {time.time()-t:.0f}s -> {args.out}")
    print("true-type counts:", dict(cnt))
    # 9-class label histogram sanity
    hist = collections.Counter(a["label"] for x in out for a in x["answers"])
    print("9-class label histogram (0..8):", {k: hist.get(k, 0) for k in range(9)})

if __name__ == "__main__":
    main()
