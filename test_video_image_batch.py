#!/usr/bin/env python3.12
"""PAAS_ensemble_v2 batch tester over a folder tree of images + videos -- REAL multi-GPU batching.

The combined counterpart of FFAA's / the MIDS++ ensemble's test_video_image_batch.py, now self-
contained for multi-GPU just like those two (no external bash fan-out needed):

  * recurses --input-dir, finds every still image + video, and FILE-SHARDS them round-robin across
    the selected GPUs (one worker process per GPU, `spawn`); each worker pins CUDA_VISIBLE_DEVICES to
    its physical device (seen as cuda:0) and loads its own PaasPipeline -- the same device trick the
    pipeline already relies on (paas/env.py);
  * REAL batching: each worker accumulates frames/images ACROSS files into a buffer and scores them
    in one PaasPipeline.predict_frames() call per --flush-size (the v1 of this script called
    predict_frames once per file, so every still image ran as a batch of 1 -- the slow path this
    rewrite removes). predict_frames sub-batches internally by --ens-batch / --ffaa-batch;
  * ground truth = the `real`/`fake` path component;
  * optional --filter-real skips low-quality real frames (logged SK, EXCLUDED from accuracy);
  * "ambiguous" counts as FAKE for accuracy (ambiguous on a fake = correct, on a real = miss);
  * copies every misclassified item AND every ambiguous item to --miss-dir, inserting "_ambiguous"
    into the filename for ambiguous ones;
  * each worker writes shard-suffixed results; the main process merges them into the unified line
    format -- results_paas.txt (fused decision) plus results_ensemble.txt / results_ffaa.txt
    (per-model fake-scores) for offline scripts/combine_eval.py -- and prints a per-label summary.

Examples:
  python3.12 test_video_image_batch.py --config config/experiments/weighted_ffaa.json \
      --input-dir /datasets/work/vLLM/data/axonlabs_data_1 --out-dir runs/test --devices all
  python3.12 test_video_image_batch.py --input-dir /data --device 0 --frame-stride 10 --flush-size 128
"""
import argparse
import multiprocessing as mp
import os
import sys
import time

PROJECT = "PAAS_ensemble_v3"
_HERE = os.path.dirname(os.path.abspath(__file__))

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
VID_EXT = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpeg", ".mpg")


# ----------------------------------------------------------------------------- discovery / paths
def truth_of(path):
    parts = path.lower().split(os.sep)
    if "real" in parts:
        return "real"
    if "fake" in parts:
        return "fake"
    return None


def iter_media(root, skip_dirs=()):
    """Yield (path, kind) for every image/video under root, skipping any path under skip_dirs."""
    skip_dirs = tuple(os.path.abspath(d) for d in skip_dirs if d)
    for dp, _, files in os.walk(root):
        adp = os.path.abspath(dp)
        if any(adp == s or adp.startswith(s + os.sep) for s in skip_dirs):
            continue
        for fn in sorted(files):
            ext = os.path.splitext(fn)[1].lower()
            kind = "image" if ext in IMG_EXT else ("video" if ext in VID_EXT else None)
            if kind is not None:
                yield os.path.join(dp, fn), kind


def frames_of(path, kind, stride):
    import cv2
    if kind == "image":
        im = cv2.imread(path, cv2.IMREAD_COLOR)
        if im is not None:
            yield path, cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return
    cap = cv2.VideoCapture(path)
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % stride == 0:
            yield f"{path}#frame={i:06d}", cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()


def miss_target(miss_dir, truth, key, decision):
    """Flatten a key (path or path#frame=NNNN) into a miss-dir filename; tag ambiguous ones."""
    flat = key.replace("#frame=", "_frame").lstrip("/").replace("/", "__")
    stem, dot, ext = flat.rpartition(".")
    if decision == "ambiguous":
        flat = (f"{stem}_ambiguous.{ext}" if dot else flat + "_ambiguous")
    return os.path.join(miss_dir, truth, flat)


# ----------------------------------------------------------------------------- device selection
def parse_devices(args):
    import torch
    n = torch.cuda.device_count()
    if args.device is not None:
        raw = str(args.device).strip().lower().replace("cuda:", "")
        return [int(raw)], n
    spec = args.devices.strip().lower()
    if spec == "all":
        return list(range(n)), n
    return [int(x.replace("cuda:", "")) for x in spec.split(",") if x.strip()], n


# ----------------------------------------------------------------------------- GPU worker (one per device)
def gpu_worker(device_id, media_paths, args_dict, n_devices, result_q):
    # Pin THIS process to its physical GPU before torch / the pipeline initialise CUDA. The pipeline
    # loads every model on cuda:0, so cuda:0 == this physical device. Must precede any torch import.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    args = argparse.Namespace(**args_dict)

    import cv2
    import torch
    sys.path.insert(0, _HERE)
    from paas.config import PaasConfig
    from paas.pipeline import PaasPipeline
    from paas.io_results import fmt_line
    from paas.decision import eval_label

    try:
        torch.set_num_threads(max(2, (os.cpu_count() or 8) // max(1, n_devices)))
    except Exception:
        pass
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)

    cfg = PaasConfig.from_file(args.config)
    cfg.device = "cuda:0"
    if args.ffaa_cache:
        cfg.ffaa.cache_path = args.ffaa_cache
    if args.components:
        cfg.fusion.components = [c.strip() for c in args.components.split(",") if c.strip()]
    if args.fusion:
        cfg.fusion.method = args.fusion
    if args.threshold is not None:
        cfg.decision.threshold = args.threshold
    _need = cfg.needs()                                   # load only the detectors the components need
    cfg.ffaa.enabled = _need["ffaa"]; cfg.ensemble9.enabled = _need["ens"]
    cfg.gsd.enabled = _need["gsd"]; cfg.selop.enabled = _need["selop"]
    cfg.validate()

    print(f"[GPU {device_id}] loading pipeline [{cfg.fusion.method} {cfg.fusion.components}] "
          f"({len(media_paths)} media files in shard) ...", flush=True)
    pipe = PaasPipeline(cfg)
    has_ens, has_ffaa = pipe.ens is not None, pipe.ffaa is not None
    print(f"[GPU {device_id}] ready (ens={has_ens} ffaa={has_ffaa} gsd={pipe.gsd is not None} "
          f"selop={pipe.selop is not None}) threshold={cfg.decision.threshold}", flush=True)

    rfilter = None
    if args.filter_real:
        try:
            from paas.data.face_filter import FaceQualityFilter
            rfilter = FaceQualityFilter()
        except Exception as e:
            print(f"[GPU {device_id}] --filter-real requested but face filter unavailable "
                  f"({e}); reals NOT filtered.", flush=True)

    miss_dir = args.miss_dir
    sfx = f".shard{device_id}"
    f_paas = open(os.path.join(args.out_dir, f"results_paas{sfx}.txt"), "w")
    f_ens = open(os.path.join(args.out_dir, f"results_ensemble{sfx}.txt"), "w") if has_ens else None
    f_ffaa = open(os.path.join(args.out_dir, f"results_ffaa{sfx}.txt"), "w") if has_ffaa else None
    # one file per ensemble member (A1_9c / A2_9c / A3_9c ...) so each model's own per-frame
    # fake-score is preserved for offline combination evaluation.
    ens_names = pipe.ens.member_names if has_ens else []
    f_members = {name: open(os.path.join(args.out_dir, f"results_ensemble_{name}{sfx}.txt"), "w")
                 for name in ens_names}
    all_fhs = [f for f in (f_paas, f_ens, f_ffaa, *f_members.values()) if f]

    tally = {}                                  # truth -> [correct, total]   (SK excluded)
    counts = {"frames": 0, "skipped": 0, "errors": 0, "miss_saved": 0}
    buffer = []                                 # [{key, truth, rgb}] -- batched ACROSS files
    t0 = time.time()
    last_report = [time.time()]

    def handle(it, r):
        key, truth, rgb = it["key"], it["truth"], it["rgb"]
        if r["decision"] == "error":
            counts["errors"] += 1
            for fh in all_fhs:
                fh.write(fmt_line("ER", truth, "----", "error", None, None, key) + "\n")
            return
        if f_ens is not None and r.get("ensemble_fake") is not None:
            ef = r["ensemble_fake"]
            f_ens.write(fmt_line("OK", truth, "fake" if ef >= 0.5 else "real", "-", ef,
                                 ef if ef >= 0.5 else 1 - ef, key) + "\n")
        pm_scores = r.get("ensemble_per_model") or {}
        for name, fh in f_members.items():                 # each ensemble member's own fake-score
            pm = pm_scores.get(name)
            if pm is not None:
                fh.write(fmt_line("OK", truth, "fake" if pm >= 0.5 else "real", "-", pm,
                                  pm if pm >= 0.5 else 1 - pm, key) + "\n")
        if f_ffaa is not None and r.get("ffaa_fake") is not None:
            ff = r["ffaa_fake"]
            f_ffaa.write(fmt_line("OK", truth, r.get("ffaa_analysis", "-"), "-", ff,
                                  r.get("ffaa_match"), key) + "\n")

        decision = r["decision"]
        correct = (eval_label(decision) == truth)          # ambiguous -> fake
        f_paas.write(fmt_line("OK" if correct else "XX", truth, decision, r["forgery_type"],
                              r["forgery_score"], r["match_score"], key) + "\n")
        t = tally.setdefault(truth, [0, 0]); t[0] += correct; t[1] += 1

        if args.copy_miss and ((not correct) or decision == "ambiguous"):
            dst = miss_target(miss_dir, truth, key, decision)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                cv2.imwrite(dst, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                counts["miss_saved"] += 1
            except Exception:
                pass

    def flush():
        if not buffer:
            return
        res = pipe.predict_frames([it["rgb"] for it in buffer], keys=[it["key"] for it in buffer],
                                  ens_batch_size=args.ens_batch, ffaa_batch_size=args.ffaa_batch,
                                  gsd_batch_size=args.gsd_batch, selop_batch_size=args.selop_batch)
        for it, r in zip(buffer, res):
            handle(it, r)
            it["rgb"] = None
        counts["frames"] += len(buffer)
        buffer.clear()
        if args.progress_interval > 0 and time.time() - last_report[0] >= args.progress_interval:
            rate = counts["frames"] / max(time.time() - t0, 1e-9)
            done = sum(v[1] for v in tally.values())
            print(f"[GPU {device_id}] {counts['frames']} frames  {rate:.0f}/s  "
                  f"eval={done} err={counts['errors']} skip={counts['skipped']}", flush=True)
            last_report[0] = time.time()

    try:
        for path, kind in media_paths:
            truth = truth_of(path)
            if truth is None:
                continue
            for key, rgb in frames_of(path, kind, args.frame_stride):
                if truth == "real" and rfilter is not None and not rfilter.passes(rgb):
                    counts["skipped"] += 1
                    for fh in all_fhs:
                        fh.write(fmt_line("SK", truth, "skip", "lowqual", None, None, key) + "\n")
                    continue
                buffer.append({"key": key, "truth": truth, "rgb": rgb})
                if len(buffer) >= args.flush_size:
                    flush()
        flush()
    except Exception as exc:
        print(f"[GPU {device_id}] WORKER-ERROR {exc!r}", file=sys.stderr, flush=True)
    finally:
        for fh in all_fhs:
            fh.close()
        result_q.put({"device_id": device_id, "tally": tally, "counts": counts,
                      "has_ens": has_ens, "has_ffaa": has_ffaa, "ens_names": ens_names})


# ----------------------------------------------------------------------------- merge / summary
def merge_shard_files(out_dir, basename, device_ids, header_lines):
    """Concatenate results_<basename>.shardN.txt -> results_<basename>.txt (header once), remove shards."""
    shards = [os.path.join(out_dir, f"results_{basename}.shard{d}.txt") for d in device_ids]
    shards = [s for s in shards if os.path.exists(s)]
    if not shards:
        return None
    dst = os.path.join(out_dir, f"results_{basename}.txt")
    with open(dst, "w") as out:
        for h in header_lines:
            out.write(h + "\n")
        for s in shards:
            with open(s) as fh:
                out.write(fh.read())
    for s in shards:
        try:
            os.remove(s)
        except OSError:
            pass
    return dst


def main():
    ap = argparse.ArgumentParser(description=f"{PROJECT} batch image/video tester (multi-GPU)")
    ap.add_argument("--config", default=os.path.join("config", "experiments", "paas5_mean.json"))
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--out-dir", default="runs/test")
    ap.add_argument("--miss-dir", default=None, help="copy misses + ambiguous here (default: <out-dir>/miss)")
    ap.add_argument("--devices", default="all", help="CUDA devices: 'all' or e.g. '0,1,2,3'.")
    ap.add_argument("--device", default=None, help="single-GPU alias (int or cuda:N); overrides --devices.")
    ap.add_argument("--fusion", default=None, help="override fusion method (mean|weighted)")
    ap.add_argument("--components", default=None,
                    help="comma list overriding fusion.components (e.g. ffaa,A1_9c,A2_9c,gsd,selop)")
    ap.add_argument("--threshold", type=float, default=None, help="override decision threshold")
    ap.add_argument("--ffaa-cache", default=None,
                    help="MIDS-format JSON of pre-generated FFAA answers (e.g. testset_mids/"
                         "mids_testset.json). Frames found in it skip LLaVA generation; frames NOT "
                         "in it fall back to the normal generate-then-score path.")
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="cap total media files (0 = all)")
    ap.add_argument("--flush-size", type=int, default=256,
                    help="frames accumulated ACROSS files per predict_frames() call (the real-batch knob)")
    ap.add_argument("--ens-batch", type=int, default=160, help="ensemble GPU sub-batch")
    ap.add_argument("--ffaa-batch", type=int, default=160, help="FFAA GPU sub-batch")
    ap.add_argument("--gsd-batch", type=int, default=160, help="GSD GPU sub-batch")
    ap.add_argument("--selop-batch", type=int, default=160, help="SeLop GPU sub-batch")
    ap.add_argument("--filter-real", type=int, default=0, help="1 = skip low-quality reals (optional face filter)")
    ap.add_argument("--copy-miss", type=int, default=1, help="1 = copy misses + ambiguous to --miss-dir")
    ap.add_argument("--progress-interval", type=float, default=10.0, help="seconds between per-GPU progress lines")
    args = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        print("CUDA is required for multi-GPU batch testing.", file=sys.stderr)
        return 1
    devices, n_visible = parse_devices(args)
    bad = [d for d in devices if d < 0 or d >= n_visible]
    if not devices or bad:
        print(f"Invalid devices {bad or devices}; visible CUDA count = {n_visible}.", file=sys.stderr)
        return 1

    # validate the config up-front (fail fast before spawning workers)
    sys.path.insert(0, _HERE)
    from paas.config import PaasConfig
    cfg = PaasConfig.from_file(args.config)
    if args.ffaa_cache:
        cfg.ffaa.cache_path = args.ffaa_cache
        if not os.path.isfile(args.ffaa_cache):
            print(f"--ffaa-cache not found: {args.ffaa_cache}", file=sys.stderr)
            return 1
    if args.components:
        cfg.fusion.components = [c.strip() for c in args.components.split(",") if c.strip()]
    if args.fusion:
        cfg.fusion.method = args.fusion
    if args.threshold is not None:
        cfg.decision.threshold = args.threshold
    cfg.validate()

    os.makedirs(args.out_dir, exist_ok=True)
    args.miss_dir = args.miss_dir or os.path.join(args.out_dir, "miss")

    media = list(iter_media(args.input_dir, skip_dirs=(args.miss_dir, args.out_dir)))
    media = [(p, k) for p, k in media if truth_of(p) is not None]
    if args.limit:
        media = media[:args.limit]
    if not media:
        print(f"No labelled image/video files under {args.input_dir}")
        return 0
    n_images = sum(1 for _, k in media if k == "image")
    n_videos = sum(1 for _, k in media if k == "video")
    print(f"[{PROJECT}] media: {n_images} images + {n_videos} videos = {len(media)} files | "
          f"fusion={cfg.fusion.method} | GPUs={','.join(map(str, devices))}"
          + (f" | ffaa-cache={args.ffaa_cache}" if args.ffaa_cache else ""), flush=True)

    shards = [media[i::len(devices)] for i in range(len(devices))]
    ctx = mp.get_context("spawn")
    result_q = ctx.Queue()
    procs = []
    for dev, shard in zip(devices, shards):
        p = ctx.Process(target=gpu_worker, args=(dev, shard, vars(args), len(devices), result_q))
        p.start()
        procs.append(p)

    tally = {}
    counts = {"frames": 0, "skipped": 0, "errors": 0, "miss_saved": 0}
    has_ens = has_ffaa = False
    ens_names = []
    for _ in procs:
        msg = result_q.get()
        for truth, (c, n) in msg["tally"].items():
            t = tally.setdefault(truth, [0, 0]); t[0] += c; t[1] += n
        for k, v in msg["counts"].items():
            counts[k] = counts.get(k, 0) + v
        has_ens = has_ens or msg["has_ens"]
        has_ffaa = has_ffaa or msg["has_ffaa"]
        for nm in msg.get("ens_names", []):
            if nm not in ens_names:
                ens_names.append(nm)
        print(f"[GPU {msg['device_id']}-DONE] frames={msg['counts']['frames']} "
              f"err={msg['counts']['errors']} skip={msg['counts']['skipped']}", flush=True)
    for p in procs:
        p.join()

    col = "# columns: OK/XX/SK/ER  truth  pred  type  fake_score  match_score  image"
    tag = f"# {PROJECT} | config={cfg.name} | fusion={cfg.fusion.method} | input={args.input_dir}"
    paas_file = merge_shard_files(args.out_dir, "paas", devices,
                                  [tag + f" | FUSED [{cfg.fusion.method}]", col])
    member_files = []
    if has_ens:
        merge_shard_files(args.out_dir, "ensemble", devices, [tag + " | 9-class ensemble (fused)", col])
        for nm in ens_names:                               # one file per ensemble member
            mf = merge_shard_files(args.out_dir, f"ensemble_{nm}", devices,
                                   [tag + f" | ensemble member {nm}", col])
            if mf:
                member_files.append(mf)
    if has_ffaa:
        merge_shard_files(args.out_dir, "ffaa", devices, [tag + " | FFAA MLLM+MIDS", col])

    def pct(c, n):
        return 100.0 * c / n if n else float("nan")
    print(f"\n=== {PROJECT} fused accuracy [{cfg.fusion.method}] "
          f"(ambiguous counts as fake; SK excluded) ===")
    recalls = []
    for truth, (c, n) in sorted(tally.items()):
        recalls.append(pct(c, n))
        print(f"  {truth:5s}: {c}/{n} = {pct(c, n):.2f}%")
    tot_c = sum(c for c, _ in tally.values())
    tot_n = sum(n for _, n in tally.values())
    print(f"  OVERALL: {tot_c}/{tot_n} = {pct(tot_c, tot_n):.2f}%"
          + (f"   (balanced = {sum(recalls)/len(recalls):.2f}%)" if recalls else ""))
    print(f"  frames={counts['frames']} skipped-real={counts['skipped']} "
          f"errors={counts['errors']} miss-saved={counts['miss_saved']}")
    print(f"\nresults -> {paas_file}  | misses -> {args.miss_dir}/")
    if member_files:
        print("per-member ensemble results (for offline combination):")
        for mf in member_files:
            print(f"  {mf}")
    if has_ens and has_ffaa:
        print(f"offline frontier: python3.12 scripts/combine_eval.py "
              f"--ensemble {args.out_dir}/results_ensemble.txt --ffaa {args.out_dir}/results_ffaa.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
