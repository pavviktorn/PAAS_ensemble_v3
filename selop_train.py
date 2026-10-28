#!/usr/bin/env python3.12
"""Train SeLop/LROR on MIDS (frozen CLIP ViT-L/14-336 + per-layer LROR + linear head).

Launched via torchrun for multi-GPU DDP (see run_train.sh), or run directly for
a single GPU. Labels come from `get_label_all` (path-derived). Plain CE loss.
"""

import argparse
import collections
import json
import math
import os
import random
import time

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from selop.data import MidsBinaryDataset, build_index, build_transforms
from selop.engine import evaluate
from selop.metrics import format_metrics
from selop.model import SeLopModel
from selop.utils import (amp_dtype_from_str, barrier, init_distributed, is_main,
                         seed_everything, set_cpu_threads)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="selop_config.json")
    # optional overrides
    for k in ["epochs", "batch_size", "rank", "n_intervene", "num_workers",
              "eval_every_steps", "warmup_steps", "num_classes"]:
        ap.add_argument(f"--{k}", type=int, default=None)
    for k in ["lr", "weight_decay", "grad_clip", "real_recall_target"]:
        ap.add_argument(f"--{k}", type=float, default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--amp_dtype", default=None)
    ap.add_argument("--select_metric", default=None)
    ap.add_argument("--init_from", default=None,
                    help="warm-start: load LROR+head weights from this checkpoint (e.g. best.pt)")
    # smoke-test helpers
    ap.add_argument("--limit_train", type=int, default=None)
    ap.add_argument("--limit_val", type=int, default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    return ap.parse_args()


def load_config(args):
    with open(args.config) as f:
        cfg = json.load(f)
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    for k, v in vars(args).items():
        if k in ("config",):
            continue
        if v is not None:
            cfg[k] = v
    croot = os.path.dirname(os.path.abspath(args.config))   # resolve relative paths vs the config dir
    for k in ("clip_path", "train_data", "val_data", "init_from", "out_dir"):
        v = cfg.get(k)
        if isinstance(v, str) and v and not os.path.isabs(v):
            cfg[k] = os.path.join(croot, v)
    return cfg


def make_loader(samples, cfg, train, rank, world_size, is_dist, log):
    tfm = build_transforms(cfg["image_size"], train=train)
    ds = MidsBinaryDataset(samples, tfm, cfg["image_size"], return_index=not train)
    if is_dist:
        sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank,
                                     shuffle=train, drop_last=train)
    else:
        sampler = None
    loader = DataLoader(
        ds, batch_size=cfg["batch_size"], sampler=sampler,
        shuffle=(sampler is None and train), drop_last=train,
        num_workers=cfg["num_workers"], pin_memory=True,
        persistent_workers=cfg["num_workers"] > 0,
        prefetch_factor=4 if cfg["num_workers"] > 0 else None,
    )
    return ds, sampler, loader


def main():
    args = parse_args()
    cfg = load_config(args)
    rank, world_size, local_rank, device, is_dist = init_distributed()
    set_cpu_threads(cfg["omp_threads"])
    seed_everything(cfg["seed"], rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    amp_dtype = amp_dtype_from_str(cfg["amp_dtype"])

    def log(*a):
        if is_main(rank):
            print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

    if is_main(rank):
        os.makedirs(cfg["out_dir"], exist_ok=True)
        with open(os.path.join(cfg["out_dir"], "config.used.json"), "w") as f:
            json.dump(cfg, f, indent=2)
        log("config:", json.dumps(cfg, indent=2))
        log(f"world_size={world_size} device={device} amp={cfg['amp_dtype']}")

    # ---- build label indexes (rank0 writes cache, others read) ----
    if is_main(rank):
        train_samples = build_index(cfg["train_data"], cfg["num_classes"], log=log)
        val_samples = build_index(cfg["val_data"], cfg["num_classes"], log=log)
    barrier(is_dist)
    if not is_main(rank):
        train_samples = build_index(cfg["train_data"], cfg["num_classes"], log=lambda *a: None)
        val_samples = build_index(cfg["val_data"], cfg["num_classes"], log=lambda *a: None)
    # limits sample randomly (not head-slice) so smoke subsets keep all classes
    if args.limit_train:
        train_samples = random.Random(cfg["seed"]).sample(
            train_samples, min(args.limit_train, len(train_samples)))
    if args.limit_val:
        val_samples = random.Random(cfg["seed"]).sample(
            val_samples, min(args.limit_val, len(val_samples)))
    counts = collections.Counter(lab for _, lab in train_samples)
    log(f"train={len(train_samples)} val={len(val_samples)} train_label_counts={dict(sorted(counts.items()))}")

    _, train_sampler, train_loader = make_loader(train_samples, cfg, True, rank, world_size, is_dist, log)
    _, _, val_loader = make_loader(val_samples, cfg, False, rank, world_size, is_dist, log)

    # ---- model ----
    model = SeLopModel(cfg["clip_path"], num_classes=cfg["num_classes"],
                       rank=cfg["rank"], n_intervene=cfg["n_intervene"],
                       init_std=cfg["init_std"]).to(device)
    n_train = sum(p.numel() for p in model.trainable_parameters())
    log(f"trainable params: {n_train/1e6:.3f}M  (LROR x{cfg['n_intervene']} rank {cfg['rank']} + head)")

    # warm-start from a prior checkpoint (resume the LROR+head; backbone is frozen anyway)
    init_from = cfg.get("init_from")
    init_best = -1.0
    if init_from:
        ck = torch.load(init_from, map_location="cpu")
        model.load_trainable(ck)
        pm = ck.get("metrics", {}) or {}
        init_best = float(pm.get(cfg["select_metric"], pm.get("auc", -1.0)) or -1.0)
        log(f"init_from: loaded LROR+head from {init_from} "
            f"(prev step {ck.get('step')}, {cfg['select_metric']}={init_best:.4f})")
    raw_model = model
    if is_dist:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)

    opt = torch.optim.Adam(raw_model.trainable_parameters(), lr=cfg["lr"],
                           weight_decay=cfg["weight_decay"])
    # inverse-frequency CE weights (handles real/pad/deepfake imbalance), like GSD
    weight = None
    if cfg.get("class_weight", True):
        tot = sum(counts.values())
        w = [tot / (cfg["num_classes"] * counts.get(c, 1)) for c in range(cfg["num_classes"])]
        weight = torch.tensor(w, dtype=torch.float32, device=device)
        log(f"class weights: {[round(x,3) for x in w]}")
    crit = nn.CrossEntropyLoss(weight=weight)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * cfg["epochs"] if not args.max_steps else args.max_steps
    warmup = cfg["warmup_steps"]
    sched = cfg.get("lr_schedule", "cosine")
    min_lr = float(cfg.get("min_lr", cfg["lr"] * 0.01))
    log(f"lr schedule: {sched} (peak {cfg['lr']:.2e} -> min {min_lr:.2e}, warmup {warmup}, total {total_steps})")

    def lr_at(step):
        if warmup > 0 and step < warmup:                       # linear warmup
            return cfg["lr"] * (step + 1) / warmup
        if sched == "constant":
            return cfg["lr"]
        # cosine decay from peak lr down to min_lr over the post-warmup steps
        prog = (step - warmup) / max(1, total_steps - warmup)
        prog = min(max(prog, 0.0), 1.0)
        return min_lr + 0.5 * (cfg["lr"] - min_lr) * (1.0 + math.cos(math.pi * prog))

    best = init_best
    sel = cfg["select_metric"]
    gstep = 0
    last_eval = -1
    t0 = time.time()
    stop = False
    for epoch in range(cfg["epochs"]):
        if is_dist:
            train_sampler.set_epoch(epoch)
        model.train()
        for x, y in train_loader:
            for g in opt.param_groups:
                g["lr"] = lr_at(gstep)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype,
                                enabled=(amp_dtype != torch.float32)):
                logits = model(x)
                loss = crit(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg["grad_clip"] > 0:
                nn.utils.clip_grad_norm_(raw_model.trainable_parameters(), cfg["grad_clip"])
            opt.step()
            gstep += 1

            if gstep % cfg["log_every_steps"] == 0:
                rate = gstep * cfg["batch_size"] * world_size / (time.time() - t0)
                log(f"ep{epoch} step{gstep}/{total_steps} loss {loss.item():.4f} "
                    f"lr {opt.param_groups[0]['lr']:.2e} {rate:.0f} img/s")

            if cfg["eval_every_steps"] and gstep % cfg["eval_every_steps"] == 0:
                best = run_eval_and_save(raw_model, model, val_loader, device, amp_dtype,
                                         cfg, is_dist, world_size, rank, sel, best,
                                         gstep, epoch, log)
                last_eval = gstep
                model.train()
            if args.max_steps and gstep >= args.max_steps:
                stop = True
                break
        if stop:
            break
        # end-of-epoch eval (skip if we just evaluated on this exact step)
        if gstep != last_eval:
            best = run_eval_and_save(raw_model, model, val_loader, device, amp_dtype,
                                     cfg, is_dist, world_size, rank, sel, best,
                                     gstep, epoch, log)
            last_eval = gstep

    log(f"done. best {sel}={best:.4f}. checkpoints in {cfg['out_dir']}")
    if is_dist:
        torch.distributed.destroy_process_group()


def run_eval_and_save(raw_model, ddp_model, val_loader, device, amp_dtype, cfg,
                      is_dist, world_size, rank, sel, best, gstep, epoch, log):
    m = evaluate(raw_model, val_loader, device, amp_dtype, cfg["num_classes"],
                 is_dist=is_dist, world_size=world_size,
                 real_recall_target=cfg["real_recall_target"])
    if is_main(rank):
        log(f"[eval] step {gstep} (ep{epoch})\n" + format_metrics(m))
        # always save last
        torch.save({"step": gstep, "epoch": epoch, "metrics": m, **raw_model.export_state()},
                   os.path.join(cfg["out_dir"], "last.pt"))
        cur = m.get(sel, m.get("bin_auc"))
        if cur is not None and cur > best:
            best = cur
            torch.save({"step": gstep, "epoch": epoch, "metrics": m, **raw_model.export_state()},
                       os.path.join(cfg["out_dir"], "best.pt"))
            log(f"[eval] new best {sel}={best:.4f} -> saved best.pt")
        with open(os.path.join(cfg["out_dir"], "metrics_log.jsonl"), "a") as f:
            f.write(json.dumps({"step": gstep, "epoch": epoch, **m}) + "\n")
    # broadcast best so all ranks agree
    if is_dist:
        t = torch.tensor([best], device=device)
        torch.distributed.broadcast(t, src=0)
        best = float(t.item())
    return best


if __name__ == "__main__":
    main()
