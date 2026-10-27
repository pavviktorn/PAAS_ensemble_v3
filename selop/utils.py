"""Distributed / environment helpers."""

import os
import random

import numpy as np
import torch
import torch.distributed as dist


def set_cpu_threads(n):
    """Cap the per-process math-library thread pools (the 'OMP' restriction)."""
    n = max(1, int(n))
    for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
        os.environ.setdefault(var, str(n))
    torch.set_num_threads(n)


def seed_everything(seed, rank=0):
    s = seed + rank
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def init_distributed():
    """Returns (rank, world_size, local_rank, device, is_dist)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dev = torch.device("cuda", local_rank)
        try:
            dist.init_process_group(backend="nccl", init_method="env://", device_id=dev)
        except TypeError:  # older torch without device_id kwarg
            dist.init_process_group(backend="nccl", init_method="env://")
        return rank, world_size, local_rank, dev, True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)
    return 0, 1, 0, device, False


def is_main(rank):
    return rank == 0


def barrier(is_dist):
    if is_dist:
        dist.barrier()


def all_gather_object(obj, is_dist, world_size):
    if not is_dist:
        return [obj]
    out = [None for _ in range(world_size)]
    dist.all_gather_object(out, obj)
    return out


def amp_dtype_from_str(s):
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[s]
