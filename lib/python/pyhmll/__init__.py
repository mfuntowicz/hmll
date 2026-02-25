"""
Python layer on top of _pyhmll_impl.

Provides device-agnostic API using torch.device and dtype conversion to torch.dtype.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from _pyhmll_impl import (
        Backend,
        Device,
        SafetensorsAccessor,
        dtype,
        safetensors as _safetensors_impl,
    )
except ImportError as e:
    raise ImportError(
        "pyhmll requires the _pyhmll_impl C extension. "
        "Build the hmll project with the Python bindings enabled."
    ) from e

from pyhmll.torch import as_dtype


def safetensors(
    path: str | Path,
    device: Device = None,
    is_sharded: bool = False,
    backend: Any = None,
) -> SafetensorsAccessor:
    """
    Open a safetensors file or sharded index for fast loading.

    Args:
        path: Path to a single .safetensors file or to model.safetensors.index.json
            for sharded checkpoints.
        device: Target device. If None, defaults to "cpu".
        is_sharded: True if path is the path to the index file of a sharded checkpoint.
        backend: I/O backend (e.g. Backend.AUTO, Backend.IO_URING, Backend.MMAP).
            Pass None for default.

    Returns:
        SafetensorsAccessor context manager for reading tensors.
    """
    device_ = device or Device.CPU
    backend_ = backend if backend is not None else Backend.IO_URING
    path_ = str(path)
    return _safetensors_impl(path_, device_, is_sharded, backend_)


__all__ = [
    "as_dtype",
    "Backend",
    "Device",
    "SafetensorsAccessor",
    "dtype",
    "safetensors",
]
