"""
Runtime efficiency helpers shared by training/evaluation scripts.
"""

from __future__ import annotations

import resource
import sys
from typing import Any, Callable, Optional

import torch


def load_model_with_dtype(
    loader: Callable[..., Any],
    model_path: str,
    dtype: torch.dtype,
    **kwargs: Any,
) -> Any:
    """Load model with `dtype` first, fallback to `torch_dtype` for older APIs."""
    try:
        return loader(model_path, dtype=dtype, **kwargs)
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        return loader(model_path, torch_dtype=dtype, **kwargs)


def reset_gpu_peak_memory(device: Optional[torch.device] = None) -> None:
    """Reset CUDA peak memory counters for the selected device."""
    if not torch.cuda.is_available():
        return

    if device is not None and device.type != "cuda":
        return

    if device is None or device.index is None:
        torch.cuda.reset_peak_memory_stats()
    else:
        torch.cuda.reset_peak_memory_stats(device.index)


def get_gpu_peak_memory_gb(device: Optional[torch.device] = None) -> float:
    """Return CUDA peak allocated memory in GB."""
    if not torch.cuda.is_available():
        return 0.0

    if device is not None and device.type != "cuda":
        return 0.0

    if device is None or device.index is None:
        peak_bytes = torch.cuda.max_memory_allocated()
    else:
        peak_bytes = torch.cuda.max_memory_allocated(device.index)

    return float(peak_bytes) / (1024 ** 3)


def get_cpu_peak_memory_gb() -> float:
    """Return peak RSS memory in GB."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        # macOS reports bytes
        return float(peak) / (1024 ** 3)
    # Linux reports KB
    return float(peak) / (1024 ** 2)
