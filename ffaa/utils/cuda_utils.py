import os

import torch


def resolve_cuda_device(requested_device: int) -> torch.device:
    device_id = int(requested_device)
    device_count = torch.cuda.device_count()

    if device_count <= 0:
        raise RuntimeError("CUDA is required, but PyTorch cannot see any CUDA devices.")
    if device_id < 0:
        raise ValueError(f"CUDA device id must be >= 0, got {device_id}.")
    if device_id >= device_count:
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>")
        raise RuntimeError(
            f"Requested CUDA device {device_id}, but PyTorch only sees "
            f"{device_count} CUDA device(s). CUDA_VISIBLE_DEVICES={visible_devices}. "
            "If this process exposes one GPU, pass --device 0."
        )

    torch.cuda.set_device(device_id)
    return torch.device(f"cuda:{device_id}")
