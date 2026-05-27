# CUDA Stream Fusion

`cuda-stream-fusion` is a bare-metal, local, zero-copy multi-modal streaming
ingestion engine designed to move live video and audio streams into CUDA-backed
deep learning models with minimal Python involvement. The validated reference
platform is **Ubuntu 24.04 LTS**, an **NVIDIA RTX 4090 Laptop GPU with 16 GB
VRAM**, and an **Intel i9-14900HX** CPU running inside an isolated Micromamba
environment named `mmsf`.

The core CUDA pipeline is intended to be portable across NVIDIA CUDA-capable
GPUs. The reference numbers, memory budget, and default build flags are tuned
for the RTX 4090 Laptop GPU. Other NVIDIA cards should work when the CUDA
architecture, CUDA Toolkit/PyTorch compatibility pair, VRAM budget, and model
size are adjusted for that machine.

The project is intentionally narrow: it is not a general media framework, a
web-serving stack, or a codec distribution. It is a low-level infrastructure
asset for high-frequency sensor-to-model pipelines where latency, memory
ownership, pointer stability, and GPU residency matter more than broad format
coverage.

## EXECUTIVE ARCHITECTURE OVERVIEW

### Technical Objective

The central design goal is to bypass the dominant latency sources in typical
Python multimedia inference stacks:

- **Python Global Interpreter Lock (GIL) contention** during stream ingestion.
- **Network, IPC, and serialization overhead** between capture, preprocessing,
  and inference components.
- **Host-to-Device (CPU-to-GPU) copies** in the recurring streaming loop.
- **Host-side framework preprocessors** that materialize CUDA tensors as NumPy
  arrays or temporary CPU buffers before model execution.

`cuda-stream-fusion` moves the hot path into C++/CUDA:

1. Native C++ owns video and audio ring buffers.
2. CUDA kernels decode and transform model-facing video memory.
3. `cudaMallocManaged` provides unified allocations with explicit CUDA residency
   hints and prefetching.
4. DLPack capsules expose native CUDA pointers directly to PyTorch as tensors on
   `cuda:0`.
5. Python receives final tensor views and orchestration hooks only. It does not
   own the streaming memory and should not participate in per-packet ingest in a
   production deployment.

The exposed tensors are true PyTorch CUDA tensor views over native backing
stores. For video, the base RGB tensor is laid out as dense interleaved HWC:

```text
shape   = [height, width, 3]
stride  = [width * 3, 3, 1]
dtype   = float32
device  = cuda:0
range   = [0.0, 1.0]
owner   = native C++ ring buffer allocated through cudaMallocManaged
```

For Qwen2.5-VL, the native kernel additionally emits a dense patch tensor:

```text
shape   = [num_patches, 1176]
stride  = [1176, 1]
1176    = 3 channels * 2 temporal frames * 14 patch rows * 14 patch columns
device  = cuda:0
```

For a 1920x1080 source frame, the CUDA patchifier uses padded Qwen grid
dimensions:

```text
grid_t = 1
grid_h = ceil(1080 / 28) * 2 = 78
grid_w = ceil(1920 / 28) * 2 = 138
rows   = grid_t * grid_h * grid_w = 10,764
cols   = 1,176
```

Out-of-frame padded pixels are filled as `0.0f` in the patch tensor. Real pixels
are normalized with OpenAI CLIP/Qwen statistics in CUDA before vLLM sees the
payload.

### Dual-Path Ingestion Telemetry

The engine currently has two native ingest paths.

**Video path**

- Input format: raw YUV420p.
- CUDA output format: interleaved RGB float32, normalized to `[0.0, 1.0]`.
- Model-specific path: Qwen2.5-VL spatial-temporal patch rows,
  `[num_patches, 1176]`.
- Ring ownership: per-slot managed allocations, exposed through DLPack.
- Observed benchmark profile: under 1 ms per frame on the target platform,
  approximately **1,066 FPS** for the CUDA video conversion path in saturation
  testing.

**Audio path**

- Input format: mono float32 PCM.
- Sample rate: 16,000 Hz.
- Channel count: 1.
- Ring design: mirrored circular audio buffer in CUDA managed memory.
- Query pattern: rolling windows such as the latest 3,000 ms for ASR context.
- Output tensor: contiguous 1D PyTorch CUDA tensor.

The audio mirror is the important implementation detail. The ring stores a
second copy of the circular region immediately after the first copy, so a window
that crosses the logical wrap boundary is still addressable as one contiguous
DLPack tensor. That avoids re-allocation, CPU stitching, and transient tensor
copies.

### Custom vLLM MultiModal Registry Monkey-Patch

vLLM's default Qwen2.5-VL multi-modal path expects framework-owned media
objects. In the installed vLLM path used during validation, raw CUDA video
tensors were routed through a parser that called:

```python
video.numpy()
```

That operation is invalid for a CUDA tensor unless it is copied to host first,
which would violate the zero-copy contract.

`fusion_engine.pipeline.install_vllm_qwen2_5_vl_zero_copy_patch()` installs a
runtime patch over the Qwen2.5-VL parser. The patch detects:

```python
multi_modal_data = {
    "video": {
        "pixel_values_videos": cuda_patch_tensor,
        "video_grid_thw": video_grid_thw,
        "second_per_grid_ts": second_per_grid_ts,
    }
}
```

and bypasses host-side NumPy conversion. `pixel_values_videos` remains a CUDA
float32 tensor produced by the native patchification kernel. The small metadata
tensors are structural CPU metadata, not frame payloads.

The patch is intentionally strict:

- `pixel_values_videos` must be a CUDA `torch.float32` tensor.
- It must be 2D.
- Its second dimension must be exactly `1176`.
- `video_grid_thw` must be a CPU tensor with shape `[1, 3]`.
- The path is implemented for Qwen2.5-VL style video pixel passthrough, not
  arbitrary models.

## SPECIFIC TARGET USE CASE

`cuda-stream-fusion` targets **low-latency, high-frequency physical edge
intelligence** where perception streams need to feed local deep learning models
without network hops or Python-mediated preprocessing.

Primary use cases include:

- **Real-time surgical simulation coaching**: capture simulator video and
  procedural audio cues, transform them on GPU, and feed local VLM/ASR models
  for sub-second feedback.
- **High-frequency robotic path planning**: ingest local sensor video, audio, or
  other device streams into reactive policy or perception models without remote
  inference latency.
- **Live audio-visual outlier detection**: maintain rolling synchronized
  windows for local anomaly detection in industrial inspection, lab monitoring,
  physical training systems, or simulation environments.
- **Local multi-modal AI assistants**: drive low-latency visual and audio
  context loops against compact local models such as Qwen2.5-VL-3B.

The repository should be treated as a reusable infrastructure core. A
full-stack application can branch from it for UX, model routing, persistence,
device capture, and domain-specific policy logic.

## STEP-BY-STEP DEPLOYMENT & USAGE GUIDE

### 1. Environment Boundary

All Python and build dependencies should live inside the isolated Micromamba
environment:

```bash
export MMSF_PREFIX="/home/user-name/micromamba/envs/mmsf"
export CSF_ROOT="/home/user-name/CUDA_Accelerated_MultiModal_Fusion_Stream/cuda-stream-fusion"
export PATH="$MMSF_PREFIX/bin:$PATH"
```

Replace `user-name` with the Linux account name that owns the Micromamba
environment and repository checkout.

Confirm the interpreter:

```bash
$MMSF_PREFIX/bin/python - <<'PY'
import sys
print(sys.executable)
assert "/micromamba/envs/mmsf/" in sys.executable
PY
```

Do not use `apt-get`, global `pip`, or system Python for project dependencies
unless you deliberately decide to break isolation. The intended installation
path uses the active Micromamba Python:

```bash
uv pip install --python "$MMSF_PREFIX/bin/python" pybind11 numpy torch
```

If vLLM is needed:

```bash
uv pip install --python "$MMSF_PREFIX/bin/python" vllm
```

For this validated CUDA 12.x environment, ensure the installed vLLM wheel
matches the CUDA runtime used by PyTorch. A mismatched wheel may fail at import
time with missing CUDA runtime libraries.

### 2. Clone and Configure

```bash
cd "$CSF_ROOT"
rm -rf build
mkdir build
cd build
cmake \
  -DMMSF_PREFIX="$MMSF_PREFIX" \
  -DCMAKE_PREFIX_PATH="$MMSF_PREFIX" \
  ..
```

The CMake project:

- Requires modern CMake and C++20.
- Finds Python and pybind11 under the `mmsf` prefix.
- Links against the NVIDIA CUDA Toolkit.
- Builds CUDA kernels for Ada Lovelace by default with
  `CUDA_ARCHITECTURES=89`, equivalent to targeting `sm_89` for the RTX 4090
  Laptop GPU.
- Emits the compiled pybind11 extension into `fusion_engine/`.

For another NVIDIA GPU, override the architecture at configure time:

```bash
cmake \
  -DMMSF_PREFIX="$MMSF_PREFIX" \
  -DCMAKE_PREFIX_PATH="$MMSF_PREFIX" \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  ..
```

Common examples:

```text
86 = NVIDIA Ampere, such as many RTX 30-series cards
89 = NVIDIA Ada Lovelace, such as many RTX 40-series cards
90 = NVIDIA Hopper, such as H100-class datacenter GPUs
```

The CPU is not hard-coded. More cores help synthetic ingest, build throughput,
and model-side orchestration, but the hot media transform path is CUDA-bound.

### 3. Build

```bash
make -j"$(nproc)"
```

Expected output:

```text
fusion_engine/fusion_core.cpython-312-x86_64-linux-gnu.so
```

### 4. Validate the Saturation Benchmark

```bash
cd "$CSF_ROOT"
$MMSF_PREFIX/bin/python tests/test_pipeline_saturation.py
```

This validates:

- Native pointer metadata.
- DLPack pointer equality.
- CUDA tensor shape and stride contracts.
- Frame ingest and kernel timing.
- PyTorch conversion overhead.
- VRAM drift after warmup.

### 5. Validate the Multi-Modal vLLM Loop

The Qwen2.5-VL-3B checkpoint should be present locally, for example:

```text
~/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/<hash>
```

Run a short validation cycle:

```bash
cd "$CSF_ROOT"
$MMSF_PREFIX/bin/python tests/test_multimodal_inference_loop.py --duration-sec 5
```

The first run may compile vLLM/Triton kernels. Subsequent runs should reuse the
compile cache.

### 6. Python: Initialize the Native Pipeline

```python
from fusion_engine.pipeline import (
    AudioConfig,
    CudaStreamFusionPipeline,
    StreamConfig,
)

pipeline = CudaStreamFusionPipeline(
    stream_config=StreamConfig(
        width=1920,
        height=1080,
        capacity=4,
        device="cuda:0",
    ),
    audio_config=AudioConfig(
        sample_rate_hz=16_000,
        capacity_ms=30_000,
        default_window_ms=3_000,
    ),
)

print(pipeline.native_info)
print(pipeline.audio_info)
```

### 7. Python: Submit Synthetic Video and Audio

The production path should write directly into native memory from a device
capture or decoder backend. The snippet below is a test/debug path using Python
arrays.

```python
import ctypes
import numpy as np

width = 1920
height = 1080
yuv_info = pipeline.yuv_input_info

yuv = np.zeros(int(yuv_info["bytes"]), dtype=np.uint8)
yuv[: width * height] = 80

ctypes.memmove(
    ctypes.c_void_p(int(yuv_info["ptr"])),
    ctypes.c_void_p(int(yuv.ctypes.data)),
    yuv.nbytes,
)

pipeline.decode_current(synchronize=True)

rgb_hwc = pipeline.latest_hwc_tensor()
qwen_bundle = pipeline.latest_qwen_patch_bundle()
qwen_patches = qwen_bundle["tensor"]
qwen_info = qwen_bundle["info"]

print(rgb_hwc.shape, rgb_hwc.stride(), rgb_hwc.device)
print(qwen_patches.shape, qwen_patches.stride(), qwen_patches.device)
print(qwen_info)
```

Audio:

```python
audio_packet = np.zeros(320, dtype=np.float32)  # 20 ms at 16 kHz
pipeline.submit_audio_pcm_float32(audio_packet, synchronize=True)

audio_window = pipeline.latest_audio_window(3_000)
print(audio_window.shape, audio_window.stride(), audio_window.device)
```

### 8. Python: vLLM Zero-Copy Qwen2.5-VL Passthrough

```python
from vllm import LLM, SamplingParams
from fusion_engine.pipeline import install_vllm_qwen2_5_vl_zero_copy_patch

install_vllm_qwen2_5_vl_zero_copy_patch()

model_path = "/home/user-name/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3"

llm = LLM(
    model=model_path,
    dtype="bfloat16",
    gpu_memory_utilization=0.80,
    cpu_offload_gb=0.0,
    max_model_len=4096,
    max_num_seqs=1,
    trust_remote_code=True,
    limit_mm_per_prompt={"video": 1},
)

video_payload, patch_info = pipeline.qwen_vllm_video_payload(fps=30.0)

assert video_payload["pixel_values_videos"].is_cuda
assert video_payload["pixel_values_videos"].shape[1] == 1176

prompt = (
    "<|im_start|>system\n"
    "You are a concise real-time perception assistant.<|im_end|>\n"
    "<|im_start|>user\n"
    "<|vision_start|><|video_pad|><|vision_end|>\n"
    "Describe the current stream state in one short sentence.<|im_end|>\n"
    "<|im_start|>assistant\n"
)

outputs = llm.generate(
    [
        {
            "prompt": prompt,
            "multi_modal_data": {"video": video_payload},
            "multi_modal_uuids": {"video": "csf-video-0"},
        }
    ],
    sampling_params=SamplingParams(temperature=0.0, max_tokens=16),
)

print(outputs[0].outputs[0].text)
```

## REALISTIC TECHNICAL LIMITATIONS & EDGE CASES

### VRAM Budget Is the Practical Boundary

The validated platform has approximately 16 GB physical VRAM. That is tight for
modern vision-language models, so the default vLLM settings are deliberately
conservative. Larger NVIDIA GPUs can raise model size, context length,
resolution, or `gpu_memory_utilization`; smaller GPUs usually need the opposite.

Observed behavior:

- Full bfloat16 `Qwen/Qwen2.5-VL-7B-Instruct` did not fit fully on GPU without
  CPU offload in this vLLM configuration.
- `Qwen/Qwen2.5-VL-3B-Instruct` fit on GPU with `gpu_memory_utilization=0.80`,
  `cpu_offload_gb=0.0`, and `max_model_len=4096`.
- Larger models generally require one or more of:
  - aggressive quantization,
  - lower resolution or smaller visual token grids,
  - CPU offload,
  - smaller `max_model_len`,
  - a smaller model,
  - or more VRAM.

For strict local GPU-only inference on the validated 16 GB hardware, expect
sub-7B models or quantized checkpoints to be the practical operating range.

### First-Cycle Triton JIT Latency

The first generation cycle can be significantly slower because vLLM, Torch
Inductor, FlashAttention, FlashInfer, and Triton may compile kernels for the
actual prompt, sequence length, and visual-token shape.

Observed first-cycle behavior included JIT warnings such as:

```text
Triton kernel JIT compilation during inference
```

This is not steady-state latency. Production systems should run explicit warmup
passes for the exact model, resolution, patch grid, prompt format, and
generation settings used in the live loop.

### Preprocessor Bypass Requires Exact Shape Discipline

The vLLM monkey-patch deliberately bypasses standard host-side media processing.
That means the native CUDA code becomes responsible for invariants normally
handled by framework preprocessors:

- frame normalization,
- channel order,
- patch size,
- temporal patch size,
- grid metadata,
- spatial padding,
- dtype,
- memory contiguity,
- and prompt placeholder token count.

The current Qwen patch path assumes:

```text
channels             = 3
patch_size           = 14
temporal_patch_size  = 2
spatial_merge_size   = 2
patch vector width   = 1176
```

Changing model family, resolution policy, aspect-ratio handling, prompt
template, or Qwen processor settings requires updating the native CUDA matrix
layout and the metadata generator together. If those disagree, vLLM may reject
the input, produce incorrect visual position IDs, or silently degrade model
quality.

### Unified Memory Is Not Magic

`cudaMallocManaged` simplifies pointer sharing, but it does not remove the need
to control residency. The implementation uses CUDA advice and prefetching to
keep model-facing memory GPU-resident. If future code touches these buffers
heavily from CPU during the streaming loop, page migration can reintroduce
latency spikes.

### Python Helpers Are Integration Tools, Not the Production Capture Path

The Python snippets in this README are intentionally simple. Production capture
should not allocate NumPy frames or copy them with `ctypes.memmove()` inside the
loop. A production backend should write device or decoder output into native
buffers directly, or use a hardware decoder interop path that preserves GPU
residency.

### Audio and Video Synchronization Is Currently Application-Level

The repository provides independent native video and audio rings. It does not
yet implement a global timestamp scheduler, cross-stream clock drift correction,
or audio-video alignment policy. Applications requiring strict synchronization
should add timestamp metadata at ingest and define the sampling policy explicitly.

## REGULATORY & LEGAL LICENSING (MIT)

The source code text in this repository is distributed under the **MIT License**.
See [LICENSE](LICENSE) for the full license text.

### Critical Patent Disclaimer

**The MIT License for this repository grants permission to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the software code
text as provided in the license. It does not grant, imply, or promise any
separate patent license.**

This is especially important for commercial deployment. Building a local
multimedia streaming or inference product can implicate technologies, standards,
or algorithms that may be covered by third-party patents, patent pools, or
commercial licensing obligations. Examples may include, without limitation:

- video coding standards such as **H.266/VVC**, HEVC/H.265, AVC/H.264, AV1, or
  related codec implementations,
- vendor hardware decoder or encoder paths,
- proprietary media transport systems,
- patented multi-modal model architectures,
- patented tensor preprocessing or compression algorithms,
- commercial model weights or datasets with separate license terms.

This repository does not provide legal clearance for those technologies. It does
not grant any implicit or explicit patent permissions for mathematical
standards, codec standards, model algorithms, media processing pipelines, or
third-party commercial endpoints. End users, integrators, companies, and
research groups compiling or deploying this system are solely responsible for
their own legal review and compliance with all applicable licenses, patents,
export controls, model terms, data rights, and patent-pool obligations.

Nothing in this README or repository is legal advice.
