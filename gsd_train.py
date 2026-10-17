#!/usr/bin/env python3.12
"""Train the GSD detector. Runs on the global python3.12 / transformers==4.37.2 env.

  python3.12 train.py --config configs/default.json \
      --set train_data=/path/with/real_fake val_data=/path/val output_dir=runs/gsd batch_size=64
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gsd.cpu_limit import limit_cpu
limit_cpu(verbose=False)            # cap CPU threads BEFORE torch/OMP/MKL initialise (refined from cfg below)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from gsd.config import GSDConfig, CLASS_NAMES
from gsd.model import GSDDetector, _pool
from gsd.data import FaceDataset
from gsd.engine import train_one_epoch, evaluate, build_scheduler
from gsd.householder import semantic_basis
from gsd.runlog import setup_logging

_BOOL = {"true": True, "false": False, "1": True, "0": False}


def apply_overrides(cfg: GSDConfig, sets):
    d = cfg.to_dict()
    for kv in sets or []:
        k, v = kv.split("=", 1)
        if k not in d:
            raise SystemExit(f"unknown config key: {k}")
        cur = d[k]
        if isinstance(cur, bool) or v.lower() in _BOOL and isinstance(cur, bool):
            d[k] = _BOOL.get(v.lower(), bool(v))
        elif isinstance(cur, int) and not isinstance(cur, bool):
            d[k] = int(v)
        elif isinstance(cur, float):
            d[k] = float(v)
        else:
            d[k] = v
    return GSDConfig.from_dict(d)


class _GuidePooler(nn.Module):
    """Frozen CLIP -> pooled guide vector (B, D). Wrapped in DataParallel so guide extraction for the
    anchor runs across all training GPUs (no GSD hooks here -- those live on the trainable stream)."""

    def __init__(self, frozen, pool):
        super().__init__()
        self.frozen, self.pool = frozen, pool

    def forward(self, x):
        return _pool(self.frozen(x).last_hidden_state, self.pool)


@torch.no_grad()
def build_anchor_U(core, cfg, device, source, limit, gpu_ids):
    """Build a fixed semantic basis U from a reference set (frozen CLIP guides -> Householder QR).
    Multi-GPU: guides are extracted with DataParallel across `gpu_ids`. Strided sampling keeps a
    block-ordered testset class-balanced. Kept in fp32 for a numerically stable QR."""
    ds = FaceDataset(source, cfg.image_size, train=False, cfg=cfg, limit=limit)
    ngpu = max(1, len(gpu_ids)) if device.startswith("cuda") else 1
    dl = DataLoader(ds, batch_size=cfg.eval_batch_size * ngpu, shuffle=False,
                    num_workers=cfg.num_workers, pin_memory=True)
    core.frozen.eval()
    pooler = _GuidePooler(core.frozen, cfg.guide_pool)
    if ngpu > 1:
        pooler = nn.DataParallel(pooler, device_ids=gpu_ids)
    guides = []
    for x, _y, _p in dl:
        x = x.to(device, non_blocking=True)
        guides.append(pooler(x).float().cpu())
    U = semantic_basis(torch.cat(guides, dim=0), cfg.k, cfg.qr_method)   # (D, K)
    return U, len(ds), ngpu


def main():
    ap = argparse.ArgumentParser(description="Train GSD detector")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "configs", "default.json"))
    ap.add_argument("--set", nargs="*", default=[], help="cfg overrides key=value")
    ap.add_argument("--limit", type=int, default=0, help="cap #train images (debug)")
    ap.add_argument("--init-from", dest="init_from", default=None,
                    help="warm-start head+backbone weights from a checkpoint (fresh optimizer; "
                         "use when changing trainable scope, e.g. head -> lastN)")
    args = ap.parse_args()

    cfg = GSDConfig.from_file(args.config) if os.path.isfile(args.config) else GSDConfig()
    cfg = apply_overrides(cfg, args.set)
    if not cfg.train_data:
        raise SystemExit("set train_data=... (a real/fake tree or a json list)")
    os.makedirs(cfg.output_dir, exist_ok=True)
    log_path = os.path.join(cfg.output_dir, "train.log")
    setup_logging(log_path)                                  # tee stdout/stderr to train.log
    print(f"[gsd] logging to {log_path}", flush=True)
    gpu_ids = [int(g) for g in str(cfg.gpus).split(",") if g.strip() != ""]
    use_cuda = torch.cuda.is_available() and len(gpu_ids) > 0
    device = f"cuda:{gpu_ids[0]}" if use_cuda else "cpu"
    print(f"[gsd] GPUs={gpu_ids if use_cuda else 'cpu'} | visible={torch.cuda.device_count()} | "
          f"primary={device}", flush=True)
    limit_cpu(cfg.cpu_fraction)                              # apply configured CPU cap (env + torch threads)
    torch.manual_seed(cfg.seed)
    cfg.save(os.path.join(cfg.output_dir, "config.json"))

    print(f"[gsd] building model (K={cfg.k}, gsd_layers={cfg.n_gsd_layers}, guide={cfg.guide_pool}, "
          f"qr={cfg.qr_method}, trainable={cfg.trainable}) ...", flush=True)
    model = GSDDetector(cfg).to(device)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[gsd] trainable params: {n_tr/1e6:.1f}M | GSD layers: {model.gsd_layer_ids}", flush=True)

    if args.init_from:                                       # warm-start (weights only; fresh optimizer)
        payload = torch.load(args.init_from, map_location="cpu")
        mb, ub = model.trainable.load_state_dict(payload["trainable"], strict=False)
        prev = payload.get("config", {})
        try:
            model.head.load_state_dict(payload["head"])
            head_msg = "head loaded"
        except Exception as e:                               # e.g. num_classes changed -> keep fresh head
            head_msg = f"head re-initialised ({e})"
        print(f"[gsd] warm-start from {args.init_from} (prev trainable={prev.get('trainable')}, "
              f"val_{cfg.select_metric}={payload.get('val_'+cfg.select_metric, 'n/a')}) | "
              f"backbone: {len(mb)} missing/{len(ub)} unexpected | {head_msg}", flush=True)

    tr = FaceDataset(cfg.train_data, cfg.image_size, train=True, cfg=cfg, limit=args.limit)
    tl = DataLoader(tr, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                    drop_last=True, pin_memory=True)        # drop_last keeps batch>=2 for GSD
    counts = tr.class_counts()
    names = CLASS_NAMES if cfg.num_classes == 3 else ("real", "fake")
    print(f"[gsd] train={len(tr)} imgs (dropped {tr.n_dropped} unknown) | "
          f"classes: {dict(zip(names, counts))}", flush=True)

    # inverse-frequency class weights (mean-normalised) -> handles real/pad/deepfake imbalance
    class_weights = None
    if cfg.class_weight:
        tot = sum(counts)
        inv = [tot / (cfg.num_classes * c) if c else 0.0 for c in counts]
        class_weights = torch.tensor(inv, dtype=torch.float32)
        print(f"[gsd] class weights: {dict(zip(names, [round(w,3) for w in inv]))}", flush=True)

    vl = None
    if cfg.val_data:
        va = FaceDataset(cfg.val_data, cfg.image_size, train=False, cfg=cfg, limit=cfg.eval_limit)
        vl = DataLoader(va, batch_size=cfg.eval_batch_size, shuffle=False, num_workers=cfg.num_workers)
        print(f"[gsd] val={len(va)} imgs | classes: {dict(zip(names, va.class_counts()))}", flush=True)

    # ---- build the fixed semantic anchor (from the testset by default) and embed it in checkpoints ----
    anchor_src = cfg.anchor_data or cfg.val_data
    anchor_U = None
    if anchor_src:
        anchor_U, n_anchor, a_ngpu = build_anchor_U(model, cfg, device, anchor_src, cfg.anchor_limit, gpu_ids)
        model.set_fixed_U(anchor_U.to(device))
        ortho = (anchor_U.t() @ anchor_U - torch.eye(cfg.k)).abs().max().item()
        print(f"[gsd] anchor U{tuple(anchor_U.shape)} from {n_anchor} refs in {os.path.basename(anchor_src)} "
              f"(on {a_ngpu} GPU(s), max|UᵀU-I|={ortho:.1e}) -> embedded in checkpoints", flush=True)

    # optimizer is built from the (unwrapped) core params; DataParallel reduces grads back into them
    core = model
    opt = torch.optim.AdamW(core.param_groups(), weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * len(tl)
    sched = build_scheduler(opt, cfg, total_steps)
    if sched is not None:
        print(f"[gsd] lr scheduler: {cfg.lr_scheduler} | warmup={cfg.warmup_steps} "
              f"min_lr_ratio={cfg.min_lr_ratio} | total_steps={total_steps} "
              f"(peak lr={cfg.lr:.1e}/head {cfg.head_lr:.1e})", flush=True)
    if use_cuda and len(gpu_ids) > 1:
        # Warm up lazy torch.linalg on EACH device in the main thread first: the per-batch U is built
        # with linalg.qr/svd inside the forward, which DataParallel runs in parallel replica threads;
        # torch's first-call lazy init isn't thread-safe ("lazy wrapper should be called at most once").
        for gid in gpu_ids:
            w = torch.randn(8, 4, device=f"cuda:{gid}")
            torch.linalg.qr(w, mode="reduced")
            torch.linalg.svd(w, full_matrices=False)
        torch.cuda.synchronize()
        model = nn.DataParallel(core, device_ids=gpu_ids)
        per = cfg.batch_size // len(gpu_ids)
        print(f"[gsd] DataParallel across {gpu_ids} (batch {cfg.batch_size} -> ~{per}/GPU; "
              f"GSD U estimated per-GPU sub-batch)", flush=True)

    def payload(extra=None):
        p = {"config": cfg.to_dict(), "head": core.head.state_dict(),
             "trainable": core.trainable.state_dict()}
        if anchor_U is not None:
            p["anchor_U"] = anchor_U.cpu()
        if extra:
            p.update(extra)
        return p

    state = {"best": -1.0}

    def run_eval(gstep, ep, tag="step"):
        """Evaluate on the primary GPU and save best.pt when select_metric improves."""
        ev = evaluate(core, vl, device, cfg)
        rec = " ".join(f"{n}={r:.3f}" for n, r in zip(names, ev["recall"]))
        msg = (f"[gsd] eval @ ep{ep} gstep{gstep} | bin_auc={ev['bin_auc']:.4f} acc={ev['acc']:.4f} "
               f"bal_acc={ev['bal_acc']:.4f} [{rec}] (n={ev['n']})")
        metric = ev[cfg.select_metric]
        if metric == metric and metric > state["best"]:     # NaN-safe; select_metric (default bin_auc)
            state["best"] = metric
            torch.save(payload({f"val_{cfg.select_metric}": metric, "global_step": gstep}),
                       os.path.join(cfg.output_dir, "best.pt"))
            msg += f"  <- saved best.pt ({cfg.select_metric}={metric:.4f})"
        print(msg, flush=True)
        return ev

    gstep = 0
    for ep in range(cfg.epochs):
        st = train_one_epoch(model, tl, opt, device, cfg, ep, class_weights=class_weights,
                             global_step=gstep, eval_every=cfg.eval_every if vl else 0, eval_cb=run_eval,
                             scheduler=sched)
        gstep = st["global_step"]
        print(f"[gsd] epoch {ep} done | train_loss={st['loss']:.4f} | gstep={gstep}", flush=True)
        if vl:
            run_eval(gstep, ep, tag="epoch")                # end-of-epoch eval + best save
            torch.save(payload(), os.path.join(cfg.output_dir, f"{ep}.pt"))
    torch.save(payload(), os.path.join(cfg.output_dir, "last.pt"))
    print(f"[gsd] done. checkpoints -> {cfg.output_dir} | best {cfg.select_metric}={state['best']:.4f}", flush=True)


if __name__ == "__main__":
    main()
