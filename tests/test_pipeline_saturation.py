#!/usr/bin/env python3
"""Saturation benchmark for the cuda-stream-fusion zero-copy tensor path.

This script intentionally measures the Python-visible path:

1. A synthetic YUV420p source buffer is varied in place.
2. The buffer is written into the native ring buffer's exposed unified-memory
   YUV slot.
3. The C++/CUDA core converts YUV420p to normalized interleaved RGB float32.
4. PyTorch consumes the RGB allocation through DLPack without owning or copying
   the CUDA storage.

The synthetic source lives in host memory, so the ingestion timer includes the
test harness write into the native unified-memory slot. In a production capture
backend, the producer should write directly into the returned YUV pointer or use
hardware decode interop so Python is not in the ingest loop.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import numpy as np
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Required benchmark dependencies are not importable in this Python environment. "
        "Run this benchmark with the mmsf interpreter that has CUDA-enabled torch and numpy installed."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FUSION_ENGINE_DIR = PROJECT_ROOT / "fusion_engine"
if str(FUSION_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(FUSION_ENGINE_DIR))

if not any(FUSION_ENGINE_DIR.glob("fusion_core*.so")):
    raise SystemExit(
        f"No compiled fusion_core extension was found in {FUSION_ENGINE_DIR}. "
        "Build the project before running this benchmark."
    )

from fusion_engine.pipeline import CudaStreamFusionPipeline, StreamConfig  # noqa: E402


@dataclass(frozen=True)
class BenchmarkConfig:
    width: int
    height: int
    iterations: int
    warmup: int
    capacity: int
    device: str
    allowed_vram_drift_bytes: int


class SyntheticYUV420pGenerator:
    """Reusable synthetic YUV420p frame generator with no per-frame allocation."""

    def __init__(self, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")

        self.width = width
        self.height = height
        self.chroma_width = (width + 1) // 2
        self.chroma_height = (height + 1) // 2
        self.y_bytes = width * height
        self.uv_bytes = self.chroma_width * self.chroma_height
        self.total_bytes = self.y_bytes + 2 * self.uv_bytes

        self.frame = np.empty(self.total_bytes, dtype=np.uint8)
        self.y_plane = self.frame[: self.y_bytes].reshape(height, width)
        self.u_plane = self.frame[
            self.y_bytes : self.y_bytes + self.uv_bytes
        ].reshape(self.chroma_height, self.chroma_width)
        self.v_plane = self.frame[self.y_bytes + self.uv_bytes :].reshape(
            self.chroma_height,
            self.chroma_width,
        )

        self._initialize_pattern()

    def _initialize_pattern(self) -> None:
        x = np.arange(self.width, dtype=np.uint16)
        y = np.arange(self.height, dtype=np.uint16)[:, None]
        self.y_plane[:, :] = ((x + y) & 0xFF).astype(np.uint8)

        cx = np.arange(self.chroma_width, dtype=np.uint16)
        cy = np.arange(self.chroma_height, dtype=np.uint16)[:, None]
        self.u_plane[:, :] = (96 + ((cx * 3 + cy) & 0x3F)).astype(np.uint8)
        self.v_plane[:, :] = (128 + ((cx + cy * 5) & 0x3F)).astype(np.uint8)

    def next_frame(self, sequence: int) -> np.ndarray:
        """Return a contiguous YUV420p frame, varied in place for stream realism."""

        marker = sequence & 0xFF
        inverse = 255 - marker

        # Mutating narrow bands is enough to prevent a totally static stream
        # without making NumPy generation dominate the native benchmark.
        self.y_plane[0, :] = marker
        self.y_plane[-1, :] = inverse
        self.u_plane[0, :] = 64 + (marker >> 1)
        self.v_plane[-1, :] = 128 + (marker >> 2)
        return self.frame


def us_since(start_ns: int, end_ns: int) -> float:
    return (end_ns - start_ns) / 1_000.0


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[index]


def summarize_us(name: str, values: list[float]) -> str:
    return (
        f"{name}: avg={statistics.fmean(values):9.3f} us | "
        f"p50={statistics.median(values):9.3f} us | "
        f"p95={percentile(values, 95):9.3f} us | "
        f"p99={percentile(values, 99):9.3f} us | "
        f"min={min(values):9.3f} us | max={max(values):9.3f} us"
    )


def tensor_contract_errors(
    tensor: torch.Tensor,
    native_info: dict[str, Any],
    config: BenchmarkConfig,
) -> list[str]:
    expected_ptr = int(native_info["ptr"])
    expected_shape = (config.height, config.width, 3)
    expected_stride = (config.width * 3, 3, 1)
    errors: list[str] = []

    if tensor.data_ptr() != expected_ptr:
        errors.append(
            f"tensor.data_ptr()={tensor.data_ptr()} does not match native ptr={expected_ptr}"
        )
    if tuple(tensor.shape) != expected_shape:
        errors.append(f"shape={tuple(tensor.shape)} does not match expected={expected_shape}")
    if tuple(tensor.stride()) != expected_stride:
        errors.append(f"stride={tuple(tensor.stride())} does not match expected={expected_stride}")
    if tensor.dtype != torch.float32:
        errors.append(f"dtype={tensor.dtype} does not match torch.float32")
    if tensor.device.type != "cuda":
        errors.append(f"device={tensor.device} is not CUDA")
    if str(tensor.device) != config.device:
        errors.append(f"device={tensor.device} does not match expected {config.device}")
    return errors


def write_frame_to_native_slot(
    pipeline: CudaStreamFusionPipeline,
    frame: np.ndarray,
) -> None:
    yuv_info = pipeline.yuv_input_info
    expected_bytes = int(yuv_info["bytes"])
    if frame.nbytes != expected_bytes:
        raise RuntimeError(f"synthetic frame has {frame.nbytes} bytes, expected {expected_bytes}")

    ctypes.memmove(
        ctypes.c_void_p(int(yuv_info["ptr"])),
        ctypes.c_void_p(int(frame.ctypes.data)),
        frame.nbytes,
    )


def warmup_pipeline(
    pipeline: CudaStreamFusionPipeline,
    generator: SyntheticYUV420pGenerator,
    config: BenchmarkConfig,
) -> None:
    for i in range(config.warmup):
        frame = generator.next_frame(i)
        write_frame_to_native_slot(pipeline, frame)
        native_info = pipeline.decode_current(synchronize=True)
        tensor = pipeline.latest_hwc_tensor()
        errors = tensor_contract_errors(tensor, native_info, config)
        if errors:
            raise AssertionError("Warmup zero-copy contract failed: " + "; ".join(errors))
        del tensor

    torch.cuda.synchronize(torch.device(config.device))
    gc.collect()


def run_benchmark(config: BenchmarkConfig) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false; cannot benchmark CUDA path")

    device = torch.device(config.device)
    if device.type != "cuda":
        raise ValueError("--device must be a CUDA device, for example cuda:0")

    torch.cuda.set_device(device)
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    device = torch.device(f"cuda:{device_index}")
    config = BenchmarkConfig(
        width=config.width,
        height=config.height,
        iterations=config.iterations,
        warmup=config.warmup,
        capacity=config.capacity,
        device=str(device),
        allowed_vram_drift_bytes=config.allowed_vram_drift_bytes,
    )
    torch.cuda.empty_cache()
    gc.collect()

    generator = SyntheticYUV420pGenerator(config.width, config.height)
    pipeline = CudaStreamFusionPipeline(
        StreamConfig(
            width=config.width,
            height=config.height,
            capacity=config.capacity,
            device=config.device,
        )
    )

    print("cuda-stream-fusion saturation benchmark")
    print(f"project_root          : {PROJECT_ROOT}")
    print(f"device                : {torch.cuda.get_device_name(device)} ({config.device})")
    print(f"resolution            : {config.width}x{config.height} YUV420p")
    print(f"iterations / warmup   : {config.iterations} / {config.warmup}")
    print(f"ring capacity         : {config.capacity}")
    print(f"synthetic frame bytes : {generator.total_bytes:,}")

    warmup_pipeline(pipeline, generator, config)
    baseline_alloc = torch.cuda.memory_allocated(device)
    baseline_reserved = torch.cuda.memory_reserved(device)

    ingest_kernel_us: list[float] = []
    dlpack_us: list[float] = []
    vram_allocated: list[int] = []
    native_ptrs: set[int] = set()
    tensor_ptrs: set[int] = set()

    benchmark_start_ns = time.perf_counter_ns()
    for i in range(config.iterations):
        frame = generator.next_frame(config.warmup + i)

        ingest_start_ns = time.perf_counter_ns()
        write_frame_to_native_slot(pipeline, frame)
        native_info = pipeline.decode_current(synchronize=True)
        ingest_end_ns = time.perf_counter_ns()

        dlpack_start_ns = time.perf_counter_ns()
        tensor = pipeline.latest_hwc_tensor()
        dlpack_end_ns = time.perf_counter_ns()

        errors = tensor_contract_errors(tensor, native_info, config)
        if errors:
            raise AssertionError(
                f"Zero-copy contract failed at iteration {i}: " + "; ".join(errors)
            )

        native_ptr = int(native_info["ptr"])
        tensor_ptr = int(tensor.data_ptr())
        native_ptrs.add(native_ptr)
        tensor_ptrs.add(tensor_ptr)

        ingest_kernel_us.append(us_since(ingest_start_ns, ingest_end_ns))
        dlpack_us.append(us_since(dlpack_start_ns, dlpack_end_ns))
        vram_allocated.append(torch.cuda.memory_allocated(device))

        del tensor

    torch.cuda.synchronize(device)
    benchmark_end_ns = time.perf_counter_ns()
    gc.collect()

    final_alloc = torch.cuda.memory_allocated(device)
    final_reserved = torch.cuda.memory_reserved(device)
    elapsed_s = (benchmark_end_ns - benchmark_start_ns) / 1_000_000_000.0
    avg_total_us = statistics.fmean(
        ingest + dlpack for ingest, dlpack in zip(ingest_kernel_us, dlpack_us, strict=True)
    )
    wall_fps = config.iterations / elapsed_s
    mean_stage_fps = 1_000_000.0 / avg_total_us
    min_alloc = min(vram_allocated) if vram_allocated else baseline_alloc
    max_alloc = max(vram_allocated) if vram_allocated else baseline_alloc
    alloc_drift = final_alloc - baseline_alloc

    print()
    print("latency telemetry")
    print(summarize_us("ingest+kernel", ingest_kernel_us))
    print(summarize_us("dlpack      ", dlpack_us))
    print(f"combined avg: {avg_total_us:9.3f} us")

    print()
    print("throughput telemetry")
    print(f"wall-clock FPS        : {wall_fps:9.3f}")
    print(f"mean-stage FPS        : {mean_stage_fps:9.3f}")
    print(f"wall elapsed          : {elapsed_s:9.6f} s")

    print()
    print("zero-copy verification")
    print(f"unique native RGB ptrs: {len(native_ptrs)}")
    print(f"unique tensor ptrs    : {len(tensor_ptrs)}")
    print(f"pointer sets match    : {native_ptrs == tensor_ptrs}")

    print()
    print("PyTorch VRAM allocator stability")
    print(f"baseline allocated    : {baseline_alloc:,} bytes")
    print(f"final allocated       : {final_alloc:,} bytes")
    print(f"min loop allocated    : {min_alloc:,} bytes")
    print(f"max loop allocated    : {max_alloc:,} bytes")
    print(f"allocated drift       : {alloc_drift:,} bytes")
    print(f"baseline reserved     : {baseline_reserved:,} bytes")
    print(f"final reserved        : {final_reserved:,} bytes")

    if native_ptrs != tensor_ptrs:
        raise AssertionError("Tensor pointer set does not match native RGB pointer set")

    if len(native_ptrs) > config.capacity:
        raise AssertionError(
            f"Observed {len(native_ptrs)} native RGB slots, exceeding capacity {config.capacity}"
        )

    if abs(alloc_drift) > config.allowed_vram_drift_bytes:
        raise AssertionError(
            "PyTorch CUDA allocator drift exceeded threshold: "
            f"{alloc_drift} bytes vs allowed {config.allowed_vram_drift_bytes} bytes"
        )


def parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(
        description="Stress-test cuda-stream-fusion DLPack tensor export and ring-buffer saturation."
    )
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--capacity", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--allowed-vram-drift-bytes",
        type=int,
        default=0,
        help="Allowed torch.cuda.memory_allocated() drift after warmup.",
    )
    args = parser.parse_args()

    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.capacity <= 0:
        parser.error("--capacity must be positive")

    return BenchmarkConfig(
        width=args.width,
        height=args.height,
        iterations=args.iterations,
        warmup=args.warmup,
        capacity=args.capacity,
        device=args.device,
        allowed_vram_drift_bytes=args.allowed_vram_drift_bytes,
    )


if __name__ == "__main__":
    run_benchmark(parse_args())
