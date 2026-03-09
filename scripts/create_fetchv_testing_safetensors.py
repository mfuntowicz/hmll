"""
Generate safetensors fixtures for hmll_fetchv tests.

Produces:
1. fetchv_test.safetensors - the single file (tp=1) with deterministic fill patterns,
   multiple dtypes, various sizes (including >512KB for io_uring chunking).
2. fetchv_sharded/ - directory with model.safetensors.index.json and shard files
   (model-00001-of-00003.safetensors, etc.) for distributed/TP-style tests.

Fill pattern: value[i] = f(i) so C++ tests can validate byte-level correctness.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from safetensors.torch import save_file

# HMLL_URING_BUFFER_SIZE is 512*1024 bytes; use tensor larger than that for chunking tests
IO_URING_BUFFER_BYTES = 512 * 1024
LARGE_F32_NUMEL = (IO_URING_BUFFER_BYTES // 4) + 1024  # >512KB in float32


def make_deterministic_tensor(dtype: torch.dtype, shape: list[int], name: str) -> torch.Tensor:
    """Create a tensor with deterministic values: value[i] = f(i) for validation."""
    numel = 1
    for s in shape:
        numel *= s
    if numel == 0:
        numel = 1

    if dtype in (torch.float32, torch.float16, torch.bfloat16):
        t = torch.arange(numel, dtype=torch.float32)
        if dtype != torch.float32:
            t = t.to(dtype)
        return t.reshape(shape).contiguous()
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        # float8 has limited range; use small integers
        t = torch.arange(min(numel, 256), dtype=torch.float32)
        t = t.to(dtype)
        if numel > 256:
            t = t.repeat((numel + 255) // 256)[:numel]
        return t.reshape(shape).contiguous()
    if dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        t = torch.arange(numel, dtype=torch.int64)
        t = (t % (1 << 15)) - (1 << 14)  # spread in signed range
        return t.to(dtype).reshape(shape).contiguous()
    if dtype in (torch.uint8, torch.uint16, torch.uint32, torch.uint64):
        t = torch.arange(numel, dtype=torch.int64)
        t = t % (1 << 8) if dtype == torch.uint8 else t % (1 << 16)
        return t.to(dtype).reshape(shape).contiguous()
    if dtype == torch.complex64:
        r = torch.arange(numel, dtype=torch.float32)
        i = torch.arange(numel, dtype=torch.float32) * 0.5
        return (torch.complex(r, i)).reshape(shape).contiguous()
    raise ValueError(f"Unsupported dtype for fetchv tests: {dtype}")


def build_single_file_tensors() -> dict[str, torch.Tensor]:
    """Tensors for single-file fixture: multiple dtypes, sizes, deterministic data."""
    tensors: dict[str, torch.Tensor] = {}

    # Scalars (rank-0)
    for dtype in [torch.float32, torch.int32, torch.uint8]:
        name = str(dtype).replace("torch.", "") + ".scalar"
        tensors[name] = make_deterministic_tensor(dtype, [], name)

    # 1D vectors: small (16), medium (1024), large (8192)
    for dtype in [torch.float32, torch.float16, torch.bfloat16, torch.int32, torch.int64, torch.uint8]:
        base = str(dtype).replace("torch.", "")
        for size, suffix in [(16, "vec16"), (1024, "vec1024"), (8192, "vec8192")]:
            tensors[f"{base}.{suffix}"] = make_deterministic_tensor(dtype, [size], f"{base}.{suffix}")

    # Large tensor > 512KB to exercise io_uring chunked path
    tensors["float32.large"] = make_deterministic_tensor(torch.float32, [LARGE_F32_NUMEL], "float32.large")
    tensors["int32.large"] = make_deterministic_tensor(
        torch.int32, [LARGE_F32_NUMEL], "int32.large"
    )

    # 2D matrices
    for dtype in [torch.float32, torch.int32]:
        base = str(dtype).replace("torch.", "")
        tensors[f"{base}.mat64"] = make_deterministic_tensor(dtype, [64, 64], f"{base}.mat64")

    # float8 for mixed-dtype fetchv tests
    tensors["float8_e4m3fn.vec16"] = make_deterministic_tensor(
        torch.float8_e4m3fn, [16], "float8_e4m3fn.vec16"
    )
    tensors["float8_e5m2.vec16"] = make_deterministic_tensor(
        torch.float8_e5m2, [16], "float8_e5m2.vec16"
    )

    return tensors


def build_sharded_tensors() -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, str]]:
    """
    Build tensors split across 3 shards and weight_map for index.
    Returns (shard_name -> {tensor_name -> tensor}, weight_map).
    """
    num_shards = 3
    shard_names = [f"model-{i:05}-of-{num_shards:05}.safetensors" for i in range(1, num_shards + 1)]
    shards: dict[str, dict[str, torch.Tensor]] = {s: {} for s in shard_names}
    weight_map: dict[str, str] = {}

    # Shard 0: float32 and float16
    for size, suffix in [(16, "vec16"), (256, "vec256")]:
        shards[shard_names[0]][f"float32.shard0.{suffix}"] = make_deterministic_tensor(
            torch.float32, [size], f"float32.shard0.{suffix}"
        )
        weight_map[f"float32.shard0.{suffix}"] = shard_names[0]
        shards[shard_names[0]][f"float16.shard0.{suffix}"] = make_deterministic_tensor(
            torch.float16, [size], f"float16.shard0.{suffix}"
        )
        weight_map[f"float16.shard0.{suffix}"] = shard_names[0]

    # Shard 1: int32, int64, uint8
    for size, suffix in [(16, "vec16"), (128, "vec128")]:
        shards[shard_names[1]][f"int32.shard1.{suffix}"] = make_deterministic_tensor(
            torch.int32, [size], f"int32.shard1.{suffix}"
        )
        weight_map[f"int32.shard1.{suffix}"] = shard_names[1]
        shards[shard_names[1]][f"uint8.shard1.{suffix}"] = make_deterministic_tensor(
            torch.uint8, [size], f"uint8.shard1.{suffix}"
        )
        weight_map[f"uint8.shard1.{suffix}"] = shard_names[1]
    shards[shard_names[1]]["int64.shard1.scalar"] = make_deterministic_tensor(
        torch.int64, [], "int64.shard1.scalar"
    )
    weight_map["int64.shard1.scalar"] = shard_names[1]

    # Shard 2: bfloat16 and scalars
    shards[shard_names[2]]["bfloat16.shard2.vec64"] = make_deterministic_tensor(
        torch.bfloat16, [64], "bfloat16.shard2.vec64"
    )
    weight_map["bfloat16.shard2.vec64"] = shard_names[2]
    shards[shard_names[2]]["float32.shard2.scalar"] = make_deterministic_tensor(
        torch.float32, [], "float32.shard2.scalar"
    )
    weight_map["float32.shard2.scalar"] = shard_names[2]

    return shards, weight_map


def main() -> None:
    cwd = Path(os.getcwd())

    # 1) Single-file fixture
    single_tensors = build_single_file_tensors()
    single_path = cwd / "fetchv_test.safetensors"
    save_file(single_tensors, single_path)
    print(single_path)

    # 2) Sharded fixtures
    shard_dir = cwd / "fetchv_sharded"
    shard_dir.mkdir(exist_ok=True)
    shards, weight_map = build_sharded_tensors()
    for shard_name, tensors in shards.items():
        if tensors:
            save_file(tensors, shard_dir / shard_name)
    index = {"metadata": {"format": "pt"}, "weight_map": weight_map}
    index_path = shard_dir / "model.safetensors.index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(index_path)


if __name__ == "__main__":
    main()
