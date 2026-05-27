"""Python shell for cuda-stream-fusion.

The hot path is intentionally native: capture/decode code writes YUV420p bytes
into the C++ ring buffer, CUDA converts that frame to RGB float32, and Python
only receives a DLPack view over the final CUDA allocation.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


def _import_fusion_core() -> Any:
    """Import the pybind11 extension from the source or common CMake build dirs."""

    try:
        return importlib.import_module("fusion_core")
    except ModuleNotFoundError:
        project_root = Path(__file__).resolve().parents[1]
        candidates = [
            Path(__file__).resolve().parent,
            project_root / "build",
            project_root / "build" / "fusion_engine",
            project_root / "build" / "Release",
            project_root / "build" / "RelWithDebInfo",
        ]

        for candidate in candidates:
            if candidate.exists():
                sys.path.insert(0, str(candidate))

        return importlib.import_module("fusion_core")


fusion_core = _import_fusion_core()


@dataclass(frozen=True)
class StreamConfig:
    width: int = 1920
    height: int = 1080
    capacity: int = 4
    device: str = "cuda:0"


@dataclass(frozen=True)
class AudioConfig:
    sample_rate_hz: int = 16_000
    capacity_ms: int = 30_000
    default_window_ms: int = 3_000


@dataclass(frozen=True)
class LocalVisionModelConfig:
    model_id_or_path: str = "/models/local-quantized-vision-language-model"
    quantization: str = "4bit"
    dtype: str = "float16"
    expected_input_layout: str = "NHWC RGB float32 in [0, 1]"


@dataclass(frozen=True)
class LocalASRModelConfig:
    model_id_or_path: str = "/models/local-quantized-whisper"
    sample_rate_hz: int = 16_000
    dtype: str = "float16"
    expected_input_layout: str = "1D mono float32 PCM at 16 kHz"


class CudaStreamFusionPipeline:
    """Owns the Python-facing view of the native streaming engine."""

    def __init__(
        self,
        stream_config: StreamConfig = StreamConfig(),
        audio_config: AudioConfig = AudioConfig(),
        model_config: LocalVisionModelConfig = LocalVisionModelConfig(),
        asr_model_config: LocalASRModelConfig = LocalASRModelConfig(),
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for cuda-stream-fusion.")

        self.stream_config = stream_config
        self.audio_config = audio_config
        self.model_config = model_config
        self.asr_model_config = asr_model_config
        self.device = torch.device(stream_config.device)
        if self.device.type != "cuda":
            raise ValueError("stream_config.device must be a CUDA device such as 'cuda:0'.")

        torch.cuda.set_device(self.device)
        device_id = self.device.index or 0
        self._native_info = fusion_core.initialize(
            stream_config.width,
            stream_config.height,
            stream_config.capacity,
            device_id,
        )
        self._audio_info = fusion_core.initialize_audio(
            audio_config.sample_rate_hz,
            audio_config.capacity_ms,
            device_id,
        )

    @property
    def native_info(self) -> dict[str, Any]:
        """Return raw pointer metadata for diagnostics and native integrations."""

        return dict(fusion_core.get_cuda_tensor_ptr())

    @property
    def yuv_input_info(self) -> dict[str, Any]:
        """Return the writable YUV420p unified-memory slot for a native producer."""

        return dict(fusion_core.get_yuv_input_ptr())

    @property
    def audio_info(self) -> dict[str, Any]:
        """Return metadata for the mirrored mono float32 audio ring."""

        return dict(fusion_core.get_audio_info())

    def decode_current(self, synchronize: bool = True) -> dict[str, Any]:
        """Ask the native core to convert the current YUV420p slot to RGB float32."""

        return dict(fusion_core.decode_current(synchronize))

    def submit_audio_pcm_float32(self, samples: Any, synchronize: bool = True) -> dict[str, Any]:
        """Append mono 16 kHz float32 PCM samples to the native audio ring.

        The production path should feed native capture memory directly into the
        C++ ring. This method is still useful for tests, file-backed packets,
        and ASR integration shells that already hold float32 PCM buffers.
        """

        return dict(fusion_core.submit_audio_pcm_float32(samples, synchronize))

    def latest_hwc_tensor(self) -> torch.Tensor:
        """Return a zero-copy HWC CUDA tensor over the latest decoded RGB frame.

        torch.as_tensor cannot wrap an arbitrary CUDA address safely. The C++
        extension therefore exports a DLPack capsule whose data pointer is the
        ring buffer's RGB allocation. PyTorch consumes the capsule and creates a
        Tensor view without copying the frame.
        """

        capsule = fusion_core.to_dlpack()
        tensor = torch.utils.dlpack.from_dlpack(capsule)
        if tensor.device != self.device:
            raise RuntimeError(f"Unexpected tensor device {tensor.device}; expected {self.device}.")
        return tensor

    def latest_audio_window_info(self, duration_ms: int | None = None) -> dict[str, Any]:
        """Return native pointer/shape metadata for a rolling audio window."""

        window_ms = duration_ms or self.audio_config.default_window_ms
        return dict(fusion_core.get_latest_audio_window_info(window_ms))

    def latest_audio_window(self, duration_ms: int | None = None) -> torch.Tensor:
        """Return a zero-copy 1D CUDA tensor for recent mono float32 PCM audio."""

        window_ms = duration_ms or self.audio_config.default_window_ms
        bundle = fusion_core.get_latest_audio_window_bundle(window_ms)
        info = dict(bundle["info"])
        capsule = bundle["capsule"]
        tensor = torch.utils.dlpack.from_dlpack(capsule)
        if tensor.device != self.device:
            raise RuntimeError(f"Unexpected audio tensor device {tensor.device}; expected {self.device}.")
        if tensor.dtype != torch.float32:
            raise RuntimeError(f"Unexpected audio tensor dtype {tensor.dtype}; expected torch.float32.")
        if tensor.ndim != 1:
            raise RuntimeError(f"Unexpected audio tensor rank {tensor.ndim}; expected 1.")
        if tensor.data_ptr() != int(info["ptr"]):
            raise RuntimeError("Audio tensor data pointer does not match native DLPack window pointer.")
        return tensor

    def latest_nhwc_tensor(self) -> torch.Tensor:
        """Return a batched NHWC tensor view: [1, height, width, 3]."""

        return self.latest_hwc_tensor().unsqueeze(0)

    def latest_nchw_view(self) -> torch.Tensor:
        """Return a zero-copy NCHW view for models that accept non-contiguous input.

        The native allocation is interleaved HWC, so this permute changes only
        metadata and strides. Calling .contiguous() on this view would allocate
        and copy, which is intentionally left to model-specific preprocessing.
        """

        return self.latest_hwc_tensor().permute(2, 0, 1).unsqueeze(0)

    def model_input(self) -> torch.Tensor:
        """Default model-facing tensor, matching channels-last inference paths."""

        return self.latest_nhwc_tensor()

    def run_model_step(self, model: Any | None = None) -> Any:
        """Feed the latest frame into a local vision model.

        Real HuggingFace VLMs often require tokenizer-side prompt state plus a
        model-specific image processor. Keep those outside the streaming loop
        when possible; this method is the handoff point for the CUDA tensor.
        """

        frame = self.model_input()
        if model is None:
            return {
                "frame": frame,
                "model_config": self.model_config,
                "native_info": self.native_info,
            }

        with torch.inference_mode():
            return model(pixel_values=frame)

    def run_asr_step(self, model: Any | None = None, duration_ms: int | None = None) -> Any:
        """Feed the latest rolling PCM window into a local ASR model shell."""

        audio = self.latest_audio_window(duration_ms)
        if model is None:
            return {
                "audio": audio,
                "audio_info": self.latest_audio_window_info(duration_ms),
                "model_config": self.asr_model_config,
            }

        with torch.inference_mode():
            return model(input_features=audio)


def example() -> None:
    pipeline = CudaStreamFusionPipeline()
    print("RGB tensor metadata:", pipeline.native_info)
    print("Writable YUV slot:", pipeline.yuv_input_info)
    print("Audio ring metadata:", pipeline.audio_info)
    print("Model placeholder:", pipeline.run_model_step(model=None))
    print("ASR placeholder:", pipeline.run_asr_step(model=None))


if __name__ == "__main__":
    example()
