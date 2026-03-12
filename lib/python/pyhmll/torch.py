"""
PyTorch-specific utilities for pyhmll.
"""
from __future__ import annotations

import torch

# Import the low-level dtype enum from the C extension
try:
    from _pyhmll_impl import dtype as _hmll_dtype
    from _pyhmll_impl import Device
except ImportError:
    _hmll_dtype = None


def as_dtype(hmll_dtype):
    """
    Convert an hmll dtype enum value to torch.dtype.

    Args:
        hmll_dtype: Value from _pyhmll_impl.dtype (e.g. dtype.BFLOAT16).

    Returns:
        Corresponding torch.dtype.
    """
    if _hmll_dtype is None:
        raise ImportError("pyhmll C extension (_pyhmll_impl) not available")

    match hmll_dtype:
        case _hmll_dtype.BOOL:
            return torch.bool
        case _hmll_dtype.BFLOAT16:
            return torch.bfloat16
        case _hmll_dtype.COMPLEX:
            return torch.complex64
        case _hmll_dtype.FLOAT16:
            return torch.float16
        case _hmll_dtype.FLOAT32:
            return torch.float32
        case _hmll_dtype.FLOAT64:
            return torch.float64
        case _hmll_dtype.FLOAT8_E8M0:
            return getattr(torch, "float8_e8m0fn", torch.uint8)
        case _hmll_dtype.FLOAT8_E4M3:
            return getattr(torch, "float8_e4m3fn", torch.float32)
        case _hmll_dtype.FLOAT8_E5M2:
            return getattr(torch, "float8_e5m2", torch.float32)
        case _hmll_dtype.SIGNED_INT8:
            return torch.int8
        case _hmll_dtype.SIGNED_INT16:
            return torch.int16
        case _hmll_dtype.SIGNED_INT32:
            return torch.int32
        case _hmll_dtype.SIGNED_INT64:
            return torch.int64
        case _hmll_dtype.UNSIGNED_INT8:
            return torch.uint8
        case _hmll_dtype.UNSIGNED_INT16:
            return torch.uint16
        case _hmll_dtype.UNSIGNED_INT32:
            return torch.uint32
        case _hmll_dtype.UNSIGNED_INT64:
            return torch.uint64
        case _hmll_dtype.UNKNOWN:
            raise ValueError(f"No torch.dtype mapping for hmll dtype {hmll_dtype}")


def device_to_hmll(device: torch.device) -> Device:
    """
    Convert a torch.device to a hmll Device enum value.
    """
    match device.type:
        case "cuda":
            return Device.cuda(device.index)
        case "cpu":
            return Device.cpu()
        case _:
            raise ValueError(f"Unsupported device for pyhmll: {device!r}")