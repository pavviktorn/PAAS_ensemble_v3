"""Minimal torchrun-friendly distributed helpers (no deepspeed required)."""

from __future__ import annotations

import os
from typing import List

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def setup_distributed() -> tuple[int, int, int]:
    """Init from torchrun env vars if present.  Returns (rank, world_size, local_rank)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return dist.get_rank(), dist.get_world_size(), local_rank
    return 0, 1, 0


def cleanup_distributed() -> None:
    if is_distributed():
        dist.barrier()
        dist.destroy_process_group()


def is_main() -> bool:
    return (not is_distributed()) or dist.get_rank() == 0


def rank0_print(*args) -> None:
    if is_main():
        print(*args, flush=True)


def all_gather_lists(local: List) -> List:
    """All-gather a python list from every rank and flatten (order not guaranteed)."""
    if not is_distributed():
        return list(local)
    gathered: List[List] = [None] * dist.get_world_size()  # type: ignore[list-item]
    dist.all_gather_object(gathered, local)
    out: List = []
    for chunk in gathered:
        out.extend(chunk)
    return out
