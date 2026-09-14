"""MIDS++ trainer.

Plain PyTorch + (optional) torchrun DDP + AMP -- no DeepSpeed dependency, which suits the
4x96GB box this is built for while staying portable.  Launch single-GPU::

    mids-pp-train --config configs/mids_pp.yaml --set train_data_path=... val_data_path=...

or multi-GPU::

    torchrun --nproc_per_node=4 -m mids_plus.train --config configs/mids_pp.yaml --set ...

Only the learned tensors are checkpointed (see ``checkpoint.save_checkpoint``).  Validation tracks
ACC/AUC/AP via the unchanged FFAA match-score selector; the best-ACC checkpoint is kept.
"""

from __future__ import annotations

import argparse
import math
import os
from typing import List

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler

from .checkpoint import save_checkpoint
from .config import MidsPlusConfig
from .data import MidsAnswersDataset, build_image_transform, collate_fn
from .distributed import (all_gather_lists, cleanup_distributed, is_distributed,
                          is_main, rank0_print, setup_distributed)
from .losses import MidsPlusLoss
from .metrics import binary_metrics
from .model import build_model
from .selector import make_decision_batch, make_decision9_batch
from .tokenize import build_tokenizer

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(it, **_):
        return it


def _seed_everything(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _param_groups(model, weight_decay: float):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower() or "S_residual" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _lr_lambda(step: int, warmup: int, total: int):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _tokenize(tokenizer, texts: List[str], device):
    enc = tokenizer(texts, return_tensors="pt", padding="longest", truncation=True, max_length=512)
    return enc.to(device)


@torch.no_grad()
def evaluate_split(model, core, tokenizer, loader, device, cfg, amp_ctx, text_features=None) -> dict:
    model.eval()
    y_true, y_pred, y_score = [], [], []
    s = cfg.samples_per_image
    for batch in loader:
        images = batch["image"].to(device)
        b = images.size(0)
        # fixed prompts -> reuse precomputed text features (skip T5); else tokenize
        texts_in = text_features if text_features is not None else _tokenize(tokenizer, batch["texts"], device)
        with amp_ctx():
            out = model(texts_in, images, None, b, s - 1, 0)
        scores = F.softmax(out["logits"].float(), dim=2)[:, :s, :]
        if scores.size(2) == 9:
            _, preds, _, forgery = make_decision9_batch(batch["answers_result"], scores, chunk_size=s)
        else:
            _, preds, _, forgery = make_decision_batch(batch["answers_result"], scores, chunk_size=s)
        y_true.extend(batch["cls_label"].tolist())
        y_pred.extend(preds)
        y_score.extend(forgery)
    y_true = all_gather_lists(y_true)
    y_pred = all_gather_lists(y_pred)
    y_score = all_gather_lists(y_score)
    return binary_metrics(y_true, y_pred, y_score)


def train(cfg: MidsPlusConfig, resume: str | None = None) -> None:
    rank, world, local_rank = setup_distributed()
    _seed_everything(cfg.seed + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg)
    if resume:
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_trainable_state_dict(payload["state"], strict=False)
        rank0_print(f"Resumed trainable weights from {resume}")
    model.to(device)
    rank0_print(f"Trainable parameters: {model.num_trainable_parameters()/1e6:.2f}M")
    if cfg.clip_adapt == "svd":
        rank0_print(f"SVD-adapted CLIP linears: {len(model.svd_report.replaced)}")

    core = model
    if is_distributed():
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank] if torch.cuda.is_available() else None,
            find_unused_parameters=True,
        )

    tokenizer = build_tokenizer(cfg)
    transform_train = build_image_transform(cfg, "train")
    transform_val = build_image_transform(cfg, "val")
    train_ds = MidsAnswersDataset(cfg.train_data_path, cfg, "train", transform_train)
    val_ds = MidsAnswersDataset(cfg.val_data_path, cfg, "val", transform_val) if cfg.val_data_path else None

    # The claim prompts are FIXED across the dataset, so encode them through T5 ONCE and reuse
    # the (S, L, dim) features on every step (skips the redundant per-batch T5 call). If the
    # texts ever varied this would be wrong, so guard it behind cfg.fixed_texts (default on).
    fixed_text_feats = None
    if getattr(cfg, "fixed_texts", True):
        fixed_texts = train_ds[0]["texts"]                       # the S fixed claim prompts
        fixed_inputs = _tokenize(tokenizer, fixed_texts, device)
        fixed_text_feats = core.encode_text_features(fixed_inputs).detach()   # (S, L, dim)
        rank0_print(f"Precomputed fixed text features {tuple(fixed_text_feats.shape)} (T5 skipped in forward)")

    train_sampler = DistributedSampler(train_ds) if is_distributed() else None
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, sampler=train_sampler, shuffle=train_sampler is None,
        num_workers=cfg.num_workers, collate_fn=collate_fn, drop_last=True, pin_memory=torch.cuda.is_available(),
    )
    val_loader = None
    if val_ds is not None:
        val_sampler = DistributedSampler(val_ds, shuffle=False) if is_distributed() else None
        val_loader = DataLoader(
            val_ds, batch_size=cfg.val_batch_size, sampler=val_sampler, shuffle=False,
            num_workers=cfg.num_workers, collate_fn=collate_fn, pin_memory=torch.cuda.is_available(),
        )

    criterion = MidsPlusLoss(cfg)
    optimizer = AdamW(_param_groups(core, cfg.weight_decay), lr=cfg.lr, betas=(0.9, 0.999), eps=1e-8)
    total_steps = cfg.epochs * max(1, len(train_loader))
    warmup_steps = int(cfg.warmup_ratio * total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: _lr_lambda(step, warmup_steps, total_steps)
    )

    use_cuda = torch.cuda.is_available()
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(cfg.amp_dtype)
    use_scaler = use_cuda and cfg.amp_dtype == "fp16"
    try:  # torch >= 2.3
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    except (AttributeError, TypeError):  # pragma: no cover - older torch
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    def amp_ctx():
        if use_cuda and amp_dtype is not None:
            return torch.autocast(device_type="cuda", dtype=amp_dtype)
        import contextlib

        return contextlib.nullcontext()

    if is_main():
        os.makedirs(cfg.output_dir, exist_ok=True)
        cfg.dump(os.path.join(cfg.output_dir, "config.yaml"))

    s = cfg.samples_per_image
    best_acc = -1.0
    global_step = 0
    for epoch in range(cfg.epochs):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        running = 0.0
        last_parts = {}
        for batch in tqdm(train_loader, desc=f"epoch {epoch+1}/{cfg.epochs}", disable=not is_main()):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            cls_labels = batch["cls_label"].to(device, non_blocking=True)
            b = images.size(0)
            # fixed prompts -> reuse precomputed text features (skip T5); else tokenize per batch
            texts_in = fixed_text_feats if fixed_text_feats is not None else _tokenize(tokenizer, batch["texts"], device)

            optimizer.zero_grad(set_to_none=True)
            with amp_ctx():
                out = model(texts_in, images, None, b, s - 1, 0, cls_labels)
                loss_out = criterion(out, labels, cls_labels, core)
            loss = loss_out.total
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in core.parameters() if p.requires_grad], cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in core.parameters() if p.requires_grad], cfg.grad_clip)
                optimizer.step()
            scheduler.step()
            global_step += 1
            running += float(loss.detach())
            last_parts = {k: float(v) for k, v in loss_out.parts.items()}

            if cfg.save_every_steps and global_step % cfg.save_every_steps == 0:
                rank0_print(f"[step {global_step}] loss={float(loss.detach()):.4f} "
                            f"parts={ {k: round(v, 3) for k, v in last_parts.items()} }")
                if is_main():
                    save_checkpoint(core, cfg, os.path.join(cfg.output_dir, "last.pt"))

        running /= max(1, len(train_loader))
        rank0_print(f"[epoch {epoch+1}] train_loss={running:.4f} parts={last_parts} lr={scheduler.get_last_lr()[0]:.2e}")

        if val_loader is not None:
            metrics = evaluate_split(model, core, tokenizer, val_loader, device, cfg, amp_ctx, fixed_text_feats)
            rank0_print(f"[epoch {epoch+1}] val acc={metrics['acc']:.4f} auc={metrics['auc']:.4f} ap={metrics['ap']:.4f}")
            if is_main() and metrics["acc"] > best_acc:
                best_acc = metrics["acc"]
                save_checkpoint(core, cfg, os.path.join(cfg.output_dir, "best.pt"))
                rank0_print(f"  saved best.pt (acc={best_acc:.4f})")

        if is_main():
            save_checkpoint(core, cfg, os.path.join(cfg.output_dir, "last.pt"))
            save_checkpoint(core, cfg, os.path.join(cfg.output_dir, f"epoch{epoch+1}.pt"))

    rank0_print(f"Done. best val acc={best_acc:.4f}")
    cleanup_distributed()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MIDS++")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], help="key=value overrides")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    cfg = MidsPlusConfig.from_yaml(args.config).apply_overrides(args.set)
    train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
