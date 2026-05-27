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

_QWEN_VLLM_ZERO_COPY_PATCH_INSTALLED = False


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
    expected_input_layout: str = "Qwen2.5-VL pixel_values_videos CUDA float32 patches"


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

    def latest_qwen_patch_bundle(self) -> dict[str, Any]:
        """Return zero-copy Qwen2.5-VL video patches and native metadata.

        Native CUDA produces `pixel_values_videos` as a dense row-major tensor
        with shape `[grid_t * grid_h * grid_w, 1176]`. The column dimension is
        `3 channels * 2 temporal frames * 14 * 14`. Strides are `[1176, 1]`,
        exactly the contiguous layout that Qwen's visual patch embedder expects.
        """

        bundle = fusion_core.get_qwen_patch_tensor()
        info = dict(bundle["info"])
        tensor = torch.utils.dlpack.from_dlpack(bundle["capsule"])
        expected_shape = (int(info["rows"]), int(info["columns"]))
        expected_stride = (int(info["columns"]), 1)

        if tensor.device != self.device:
            raise RuntimeError(f"Unexpected Qwen patch tensor device {tensor.device}; expected {self.device}.")
        if tensor.dtype != torch.float32:
            raise RuntimeError(f"Unexpected Qwen patch tensor dtype {tensor.dtype}; expected torch.float32.")
        if tuple(tensor.shape) != expected_shape:
            raise RuntimeError(f"Unexpected Qwen patch tensor shape {tuple(tensor.shape)}; expected {expected_shape}.")
        if tuple(tensor.stride()) != expected_stride:
            raise RuntimeError(f"Unexpected Qwen patch tensor stride {tuple(tensor.stride())}; expected {expected_stride}.")
        if tensor.data_ptr() != int(info["ptr"]):
            raise RuntimeError("Qwen patch tensor data pointer does not match native DLPack pointer.")

        return {"tensor": tensor, "info": info}

    def latest_qwen_patch_tensor(self) -> torch.Tensor:
        """Return the latest Qwen2.5-VL `pixel_values_videos` CUDA tensor."""

        return self.latest_qwen_patch_bundle()["tensor"]

    def qwen_video_metadata(self, patch_info: dict[str, Any], fps: float = 30.0) -> dict[str, torch.Tensor]:
        """Build tiny structural CPU metadata for the native Qwen patch tensor."""

        if fps <= 0.0:
            raise ValueError("fps must be positive.")
        grid = torch.tensor(
            [[int(patch_info["grid_t"]), int(patch_info["grid_h"]), int(patch_info["grid_w"])]],
            dtype=torch.long,
            device="cpu",
        )
        seconds = torch.tensor(
            [float(patch_info["temporal_patch_size"]) / float(fps)],
            dtype=torch.float32,
            device="cpu",
        )
        return {"video_grid_thw": grid, "second_per_grid_ts": seconds}

    def qwen_vllm_video_payload(self, fps: float = 30.0) -> tuple[dict[str, Any], dict[str, Any]]:
        """Package CUDA patches in vLLM's direct multimodal passthrough format."""

        bundle = self.latest_qwen_patch_bundle()
        metadata = self.qwen_video_metadata(bundle["info"], fps=fps)
        payload = {
            "pixel_values_videos": bundle["tensor"],
            **metadata,
        }
        return payload, bundle["info"]

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


def install_vllm_qwen2_5_vl_zero_copy_patch() -> None:
    """Patch vLLM's Qwen2.5-VL parser to accept CUDA patch tensors directly.

    vLLM 0.21 accepts raw `video` tensors by routing them through a parser that
    calls `.numpy()`, which forces CPU materialization and breaks this project’s
    zero-copy contract. The Qwen model code already supports direct
    `pixel_values_videos` tensors at execution time, so this runtime patch adds
    a parser fast path for dictionaries containing native CUDA patch rows.
    """

    global _QWEN_VLLM_ZERO_COPY_PATCH_INSTALLED
    if _QWEN_VLLM_ZERO_COPY_PATCH_INSTALLED:
        return

    try:
        from vllm.model_executor.models import qwen2_vl  # type: ignore
        from vllm.multimodal import MULTIMODAL_REGISTRY  # noqa: F401
        from vllm.multimodal.inputs import MultiModalFieldConfig  # type: ignore
        from transformers.feature_extraction_utils import BatchFeature  # type: ignore
        from vllm.multimodal.inputs import MultiModalKwargsItems  # type: ignore
        from vllm.multimodal.parse import ModalityDataItems  # type: ignore
    except ImportError as exc:
        raise RuntimeError("vLLM must be importable before installing the Qwen zero-copy patch.") from exc

    parser_cls = qwen2_vl.Qwen2VLMultiModalDataParser
    original = getattr(parser_cls, "_csf_original_parse_video_data", None)
    if original is None:
        original = parser_cls._parse_video_data
        setattr(parser_cls, "_csf_original_parse_video_data", original)

    class CudaStreamFusionPixelItems(ModalityDataItems[dict[str, torch.Tensor], dict[str, torch.Tensor]]):
        """Dict passthrough that vLLM must treat as pixel values, not embeds."""

        def __init__(self, data: dict[str, torch.Tensor], fields_config: dict[str, Any]) -> None:
            super().__init__(data, "video")
            self._kwargs = MultiModalKwargsItems.from_hf_inputs(BatchFeature(dict(data)), fields_config)

        def get_count(self) -> int:
            return len(self._kwargs["video"])

        def get(self, index: int) -> dict[str, torch.Tensor]:
            return self._kwargs["video"][index].get_data()

        def get_processor_data(self) -> dict[str, object]:
            return {}

        def get_passthrough_data(self) -> dict[str, torch.Tensor]:
            return self.data

    def patched_parse_video_data(self: Any, data: Any) -> Any:
        if isinstance(data, dict) and "pixel_values_videos" in data:
            pixel_values = data.get("pixel_values_videos")
            grid = data.get("video_grid_thw")
            if not isinstance(pixel_values, torch.Tensor):
                raise TypeError("pixel_values_videos must be a torch.Tensor.")
            if not pixel_values.is_cuda:
                raise TypeError("pixel_values_videos must stay on CUDA for the zero-copy path.")
            if pixel_values.dtype != torch.float32:
                raise TypeError("pixel_values_videos must be float32.")
            if pixel_values.ndim != 2 or pixel_values.shape[1] != 1176:
                raise ValueError(
                    "pixel_values_videos must have shape [num_patches, 1176] for Qwen2.5-VL."
                )
            if not isinstance(grid, torch.Tensor) or tuple(grid.shape) != (1, 3) or grid.device.type != "cpu":
                raise ValueError("video_grid_thw must be a CPU torch.Tensor with shape [1, 3].")

            fields_config = dict(qwen2_vl._create_qwen2vl_field_factory(self._spatial_merge_size)(data))
            if "second_per_grid_ts" in data:
                fields_config["second_per_grid_ts"] = MultiModalFieldConfig.batched("video")

            return CudaStreamFusionPixelItems(data, fields_config)

        return original(self, data)

    parser_cls._parse_video_data = patched_parse_video_data
    _QWEN_VLLM_ZERO_COPY_PATCH_INSTALLED = True


if __name__ == "__main__":
    example()
