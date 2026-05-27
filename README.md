# CUDA Accelerated Multi-Modal Fusion Stream

`cuda-stream-fusion` is a local, low-latency multimedia ingestion engine for feeding real-time video and audio streams into CUDA-backed PyTorch models. The current implementation focuses on zero-copy tensor exposure, CUDA-native YUV420p video conversion, and a mirrored rolling audio ring for ASR/VLM workloads.

## Current Capabilities

- CUDA YUV420p to interleaved RGB float32 conversion for model-ready HWC tensors.
- Unified-memory video ring buffer exposed to Python through DLPack.
- Mono 16 kHz float32 PCM audio ingestion for Whisper-style ASR pipelines.
- Mirrored CUDA-managed audio ring so recent rolling windows are contiguous without stitching.
- PyTorch tensor views created with `torch.utils.dlpack.from_dlpack()` without owning or copying native storage.
- Saturation benchmark for pointer matching, DLPack overhead, FPS, and CUDA allocator drift.

## Repository Layout

```text
.
├── CMakeLists.txt
├── include/
│   └── stream_fusion.h
├── src/
│   ├── main.cpp
│   └── video_decoder.cu
├── fusion_engine/
│   ├── __init__.py
│   └── pipeline.py
└── tests/
    └── test_pipeline_saturation.py
```

## Target Environment

- Ubuntu 24.04 LTS
- NVIDIA CUDA Toolkit with `nvcc`
- NVIDIA Ada GPU target, built for `sm_89`
- Micromamba environment named `mmsf`
- Python 3.12 with CUDA-enabled PyTorch, NumPy, and pybind11

The build is intentionally scoped to the active micromamba prefix. It does not require PortAudio, ALSA, or external system audio libraries.

## Build

```bash
cd /home/tanmay-godse/CUDA_Accelerated_MultiModal_Fusion_Stream/cuda-stream-fusion
rm -rf build
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/home/tanmay-godse/micromamba/envs/mmsf ..
make -j$(nproc)
```

The compiled extension is emitted into `fusion_engine/` as `fusion_core*.so`.

## Video Benchmark

```bash
cd /home/tanmay-godse/CUDA_Accelerated_MultiModal_Fusion_Stream/cuda-stream-fusion
/home/tanmay-godse/micromamba/envs/mmsf/bin/python tests/test_pipeline_saturation.py
```

The benchmark validates that PyTorch tensor pointers match the native RGB slots and that `torch.cuda.memory_allocated()` remains stable after warmup.

## Python Usage

```python
import numpy as np
from fusion_engine.pipeline import AudioConfig, CudaStreamFusionPipeline, StreamConfig

pipeline = CudaStreamFusionPipeline(
    stream_config=StreamConfig(width=1920, height=1080, capacity=4, device="cuda:0"),
    audio_config=AudioConfig(sample_rate_hz=16_000, capacity_ms=30_000, default_window_ms=3_000),
)

rgb_hwc = pipeline.latest_hwc_tensor()

audio_packet = np.zeros(1600, dtype=np.float32)
pipeline.submit_audio_pcm_float32(audio_packet)
audio_window = pipeline.latest_audio_window(3_000)
```

## Notes

The current Python submission helpers are intended for validation and integration shells. A production capture backend should write video and audio packets directly into the native exposed buffers or through hardware decoder interop so Python stays out of the ingest hot path.
