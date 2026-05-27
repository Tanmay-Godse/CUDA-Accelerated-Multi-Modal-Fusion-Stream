#!/usr/bin/env python3
"""End-to-end multimodal inference pump for cuda-stream-fusion.

The harness keeps the ingest hot path native:

* Thread A writes synthetic YUV420p frames into the C++ unified-memory video
  ring at a locked 30 FPS cadence and launches the CUDA YUV->RGB kernel.
* Thread B appends mono 16 kHz float32 PCM packets into the mirrored audio ring.
* The controller wakes every 100 ms and obtains PyTorch CUDA tensor views through
  DLPack: video as NHWC `[1, H, W, 3]`, audio as contiguous `[samples]`.

Those tensors are handed to model runners without calling `.cpu()` or `.numpy()`
in the primary vLLM path. vLLM's multimodal processors decide how to consume the
objects internally; this script verifies that the input tensors themselves still
point at the native cuda-stream-fusion allocations at the handoff boundary.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib
import importlib.util
import logging
import math
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MMSF_PREFIX = Path(os.environ.get("MMSF_PREFIX", "/home/user-name/micromamba/envs/mmsf")).resolve()
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FUSION_ENGINE_DIR = PROJECT_ROOT / "fusion_engine"
RTX_4090_LAPTOP_VRAM_BYTES = 16 * 1024**3
VLLM_RESERVED_FRACTION = 0.80
FUSION_RESERVED_FRACTION = 1.0 - VLLM_RESERVED_FRACTION

# The script is normally launched with an absolute Python path rather than
# `micromamba run`, so shell activation does not prepend the environment bin
# directory. vLLM worker processes need this to find JIT helpers such as ninja.
MMSF_BIN = MMSF_PREFIX / "bin"
os.environ["PATH"] = f"{MMSF_BIN}:{os.environ.get('PATH', '')}"

for candidate in (PROJECT_ROOT, FUSION_ENGINE_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from fusion_engine.pipeline import (  # noqa: E402
    AudioConfig,
    CudaStreamFusionPipeline,
    StreamConfig,
    install_vllm_qwen2_5_vl_zero_copy_patch,
)


@dataclass(frozen=True)
class RuntimeConfig:
    duration_sec: float
    controller_period_ms: float
    video_fps: float
    width: int
    height: int
    ring_capacity: int
    audio_packet_ms: int
    audio_window_ms: int
    audio_capacity_ms: int
    device: str
    model: str
    prompt: str
    vllm_modality: str
    route_audio_to_vllm: bool
    asr_mode: str
    whisper_model: str
    gpu_memory_utilization: float
    dtype: str
    cpu_offload_gb: float
    max_model_len: int
    max_num_seqs: int
    max_tokens: int
    temperature: float
    trust_remote_code: bool
    enforce_eager: bool
    install_missing_optional: bool


class SyntheticYUV420pGenerator:
    """Preallocated 1920x1080 YUV420p source with cheap per-frame mutation."""

    def __init__(self, width: int, height: int) -> None:
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

        x = np.arange(width, dtype=np.uint16)
        y = np.arange(height, dtype=np.uint16)[:, None]
        self.y_plane[:, :] = ((x + y) & 0xFF).astype(np.uint8)
        self.u_plane[:, :] = 96
        self.v_plane[:, :] = 128

    def next_frame(self, sequence: int) -> np.ndarray:
        marker = sequence & 0xFF
        inverse = 255 - marker
        self.y_plane[0, :] = marker
        self.y_plane[-1, :] = inverse
        self.u_plane[0, :] = 64 + (marker >> 1)
        self.v_plane[-1, :] = 128 + (marker >> 2)
        return self.frame


class SyntheticAudioGenerator:
    """Continuous sine-wave packet generator for 16 kHz mono float32 PCM."""

    def __init__(self, sample_rate_hz: int, packet_ms: int, frequency_hz: float = 440.0) -> None:
        self.sample_rate_hz = sample_rate_hz
        self.samples_per_packet = max(1, sample_rate_hz * packet_ms // 1000)
        self.frequency_hz = frequency_hz
        self.phase = 0

    def next_packet(self) -> np.ndarray:
        t = (np.arange(self.samples_per_packet, dtype=np.float32) + self.phase) / self.sample_rate_hz
        packet = 0.15 * np.sin(2.0 * math.pi * self.frequency_hz * t)
        self.phase += self.samples_per_packet
        return np.ascontiguousarray(packet, dtype=np.float32)


class PeriodicThread(threading.Thread):
    """Thread wrapper that records failures and exits when stop_event is set."""

    def __init__(
        self,
        name: str,
        period_s: float,
        stop_event: threading.Event,
        errors: "queue.SimpleQueue[tuple[str, str]]",
    ) -> None:
        super().__init__(name=name, daemon=True)
        self.period_s = period_s
        self.stop_event = stop_event
        self.errors = errors
        self.iterations = 0
        self.overruns = 0

    def run_once(self) -> None:
        raise NotImplementedError

    def run(self) -> None:
        next_tick = time.perf_counter()
        try:
            while not self.stop_event.is_set():
                self.run_once()
                self.iterations += 1

                next_tick += self.period_s
                sleep_s = next_tick - time.perf_counter()
                if sleep_s > 0:
                    self.stop_event.wait(sleep_s)
                else:
                    self.overruns += 1
                    next_tick = time.perf_counter()
        except BaseException:
            self.errors.put((self.name, traceback.format_exc()))
            self.stop_event.set()


class VideoIngestThread(PeriodicThread):
    def __init__(
        self,
        pipeline: CudaStreamFusionPipeline,
        generator: SyntheticYUV420pGenerator,
        fps: float,
        stop_event: threading.Event,
        errors: "queue.SimpleQueue[tuple[str, str]]",
    ) -> None:
        super().__init__("video-ingest", 1.0 / fps, stop_event, errors)
        self.pipeline = pipeline
        self.generator = generator

    def run_once(self) -> None:
        frame = self.generator.next_frame(self.iterations)
        yuv_info = self.pipeline.yuv_input_info
        if frame.nbytes != int(yuv_info["bytes"]):
            raise RuntimeError(f"YUV frame byte mismatch: {frame.nbytes} vs {yuv_info['bytes']}")

        ctypes.memmove(
            ctypes.c_void_p(int(yuv_info["ptr"])),
            ctypes.c_void_p(int(frame.ctypes.data)),
            frame.nbytes,
        )
        self.pipeline.decode_current(synchronize=False)


class AudioIngestThread(PeriodicThread):
    def __init__(
        self,
        pipeline: CudaStreamFusionPipeline,
        generator: SyntheticAudioGenerator,
        packet_ms: int,
        stop_event: threading.Event,
        errors: "queue.SimpleQueue[tuple[str, str]]",
    ) -> None:
        super().__init__("audio-ingest", packet_ms / 1000.0, stop_event, errors)
        self.pipeline = pipeline
        self.generator = generator

    def run_once(self) -> None:
        self.pipeline.submit_audio_pcm_float32(self.generator.next_packet(), synchronize=False)


def assert_mmsf_context() -> None:
    executable = Path(sys.executable).resolve()
    if not executable.is_relative_to(MMSF_PREFIX):
        logging.critical(
            "Security Fault: Execution aborted. Script is running under '%s' instead of "
            "the isolated 'mmsf' environment context.",
            executable,
        )
        raise SystemExit(
            "Refusing dependency installation outside the mmsf environment.\n"
            f"Expected interpreter under: {MMSF_PREFIX}\n"
            f"Actual interpreter:       {executable}"
        )


def install_into_active_runtime(packages: list[str]) -> None:
    assert_mmsf_context()
    active_runtime = sys.executable
    try:
        logging.info("Installing into active mmsf runtime via uv: %s", " ".join(packages))
        subprocess.run(
            ["uv", "pip", "install", "--python", active_runtime, *packages],
            check=True,
            capture_output=False,
        )
    except FileNotFoundError:
        logging.warning("'uv' binary not detected in shell PATH. Falling back to isolated pip.")
        subprocess.run([active_runtime, "-m", "pip", "install", *packages], check=True)


def ensure_importable(module_name: str, install_packages: list[str]) -> Any:
    if importlib.util.find_spec(module_name) is None:
        install_into_active_runtime(install_packages)
        importlib.invalidate_caches()

    return importlib.import_module(module_name)


def find_existing_hf_snapshot(model_repo_id: str, hf_home: Path) -> Path | None:
    """Return a complete local Hugging Face snapshot if one is already cached."""

    cache_name = f"models--{model_repo_id.replace('/', '--')}"
    search_roots = [
        hf_home / cache_name,
        Path.home() / ".cache" / "huggingface" / "hub" / cache_name,
    ]
    for root in search_roots:
        snapshots = root / "snapshots"
        if not snapshots.exists():
            continue
        for snapshot in sorted(snapshots.iterdir(), key=lambda path: path.stat().st_mtime, reverse=True):
            if (snapshot / "config.json").exists():
                return snapshot.resolve()
    return None


def ensure_environment_setup(model_repo_id: str = "Qwen/Qwen2.5-VL-3B-Instruct") -> str:
    """Validate mmsf, install vLLM if missing, and pre-cache model weights."""

    assert_mmsf_context()
    active_runtime = sys.executable
    logging.info("Verified Active Mamba Environment: %s", active_runtime)

    hf_home = Path(os.environ.get("HF_HOME", MMSF_PREFIX / "huggingface_cache")).resolve()
    os.environ["HF_HOME"] = str(hf_home)
    hf_home.mkdir(parents=True, exist_ok=True)
    logging.info("Using Hugging Face cache root: %s", hf_home)

    try:
        import vllm

        logging.info("vLLM is already installed within 'mmsf' (Version: %s)", vllm.__version__)
    except ImportError:
        logging.warning("vLLM not found in active environment prefix. Triggering sandboxed installation.")
        install_into_active_runtime(["vllm"])
        importlib.invalidate_caches()
        import vllm

        logging.info("vLLM successfully installed within 'mmsf' (Version: %s)", vllm.__version__)

    model_path = Path(model_repo_id).expanduser()
    if model_path.exists():
        logging.info("Using existing local model path without Hugging Face download: %s", model_path)
        return str(model_path.resolve())

    existing_snapshot = find_existing_hf_snapshot(model_repo_id, hf_home)
    if existing_snapshot is not None:
        logging.info("Using already downloaded Hugging Face snapshot: %s", existing_snapshot)
        return str(existing_snapshot)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logging.warning("huggingface_hub was not importable. Installing it inside 'mmsf'.")
        install_into_active_runtime(["huggingface_hub"])
        importlib.invalidate_caches()
        from huggingface_hub import snapshot_download

    try:
        logging.info("Initiating target weight validation/download loop for: %s", model_repo_id)
        model_local_path = snapshot_download(
            repo_id=model_repo_id,
            repo_type="model",
            ignore_patterns=["*.msgpack", "*.h5"],
            local_files_only=False,
            cache_dir=str(hf_home),
        )
        logging.info("Hugging Face model verified and cached locally at: %s", model_local_path)
        return model_local_path
    except Exception:
        logging.exception("Failed to resolve Hugging Face model download/verification path.")
        raise


class VLLMRunner:
    """Thin vLLM offline runner that accepts CUDA tensors as multimodal objects."""

    def __init__(self, config: RuntimeConfig) -> None:
        ensure_importable("vllm", ["vllm"])
        install_vllm_qwen2_5_vl_zero_copy_patch()
        from vllm import LLM, SamplingParams

        self.config = config
        self.sampling_params = SamplingParams(
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

        limit_mm_per_prompt = {config.vllm_modality: 1}
        if config.route_audio_to_vllm:
            limit_mm_per_prompt["audio"] = 1
            logging.warning(
                "Audio tensors will be routed into vLLM multi_modal_data. "
                "Qwen2.5-VL is a visual-language model; use an audio/omni checkpoint "
                "if this vLLM build rejects the audio modality."
            )

        try:
            self.llm = LLM(
                model=config.model,
                trust_remote_code=config.trust_remote_code,
                gpu_memory_utilization=config.gpu_memory_utilization,
                dtype=config.dtype,
                cpu_offload_gb=config.cpu_offload_gb,
                max_model_len=config.max_model_len,
                max_num_seqs=config.max_num_seqs,
                limit_mm_per_prompt=limit_mm_per_prompt,
                enforce_eager=config.enforce_eager,
            )
        except BaseException as exc:
            raise RuntimeError(
                "vLLM model initialization failed. Check HuggingFace access, local weight cache, "
                "model support in this vLLM version, and RTX 4090 VRAM headroom. "
                f"Model={config.model!r}, gpu_memory_utilization={config.gpu_memory_utilization}."
            ) from exc

    def generate(
        self,
        qwen_video_payload: dict[str, Any],
        audio_window: torch.Tensor,
        sequence_id: int,
    ) -> tuple[str, int, float]:
        mm_data: dict[str, Any] = {}
        mm_uuids: dict[str, Any] = {}
        if self.config.vllm_modality == "image":
            raise RuntimeError("The zero-copy Qwen2.5-VL patch path requires --vllm-modality=video.")
        else:
            # This dictionary takes the direct vLLM passthrough path installed
            # by install_vllm_qwen2_5_vl_zero_copy_patch(). pixel_values_videos
            # is already a dense CUDA tensor shaped [grid_t*grid_h*grid_w, 1176],
            # so no CPU numpy conversion or framework-side patchification is
            # allowed on the frame payload.
            mm_data["video"] = qwen_video_payload
            mm_uuids["video"] = f"csf-video-{sequence_id}"

        if self.config.route_audio_to_vllm:
            # Some vLLM audio models accept `(audio, sampling_rate)`. Passing a
            # CUDA tensor here preserves the zero-copy handoff boundary; a given
            # model processor may still reject it if it only supports CPU arrays.
            mm_data["audio"] = (audio_window, 16_000)
            mm_uuids["audio"] = f"csf-audio-{sequence_id}"

        request = {
            "prompt": self.config.prompt,
            "multi_modal_data": mm_data,
            "multi_modal_uuids": mm_uuids,
        }

        start_ns = time.perf_counter_ns()
        outputs = self.llm.generate([request], sampling_params=self.sampling_params)
        end_ns = time.perf_counter_ns()

        completion = outputs[0].outputs[0]
        text = completion.text
        token_ids = getattr(completion, "token_ids", None)
        token_count = len(token_ids) if token_ids is not None else max(1, len(text.split()))
        return text, token_count, (end_ns - start_ns) / 1_000_000.0


class WhisperGpuFallbackRunner:
    """Split ASR path that keeps PCM and log-mel feature extraction on CUDA."""

    def __init__(self, model_id: str, device: torch.device, install_missing: bool) -> None:
        if install_missing:
            ensure_importable("transformers", ["transformers", "accelerate"])
            ensure_importable("torchaudio", ["torchaudio"])
        else:
            importlib.import_module("transformers")
            importlib.import_module("torchaudio")

        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        import torchaudio

        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(device)
        self.model.eval()

        # Whisper's frontend is 16 kHz mono, n_fft=400, hop=160, 80 mel bins.
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16_000,
            n_fft=400,
            hop_length=160,
            n_mels=80,
            center=True,
            power=2.0,
        ).to(device)

    def transcribe(self, audio_window: torch.Tensor, max_new_tokens: int) -> tuple[str, int, float]:
        start_ns = time.perf_counter_ns()
        with torch.inference_mode():
            audio = audio_window.float()
            target_samples = 30 * 16_000
            if audio.numel() < target_samples:
                audio = torch.nn.functional.pad(audio, (0, target_samples - audio.numel()))
            else:
                audio = audio[-target_samples:]

            features = self.mel(audio).clamp_min(1.0e-10).log10()
            features = torch.maximum(features, features.max() - 8.0)
            features = (features + 4.0) / 4.0
            if features.shape[-1] < 3000:
                features = torch.nn.functional.pad(features, (0, 3000 - features.shape[-1]))
            features = features[..., :3000].unsqueeze(0).to(dtype=torch.float16)

            predicted_ids = self.model.generate(input_features=features, max_new_tokens=max_new_tokens)
            text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        end_ns = time.perf_counter_ns()
        return text, int(predicted_ids.shape[-1]), (end_ns - start_ns) / 1_000_000.0


def verify_input_pointers(
    pipeline: CudaStreamFusionPipeline,
    video_nhwc: torch.Tensor,
    qwen_patch_tensor: torch.Tensor,
    qwen_patch_info: dict[str, Any],
    audio_window: torch.Tensor,
    audio_window_ms: int,
) -> None:
    video_info = pipeline.native_info
    expected_video_ptr = int(video_info["ptr"])
    if video_nhwc.data_ptr() != expected_video_ptr:
        raise RuntimeError(
            f"Video DLPack pointer mismatch: tensor={video_nhwc.data_ptr()} native={expected_video_ptr}"
        )

    expected_qwen_ptr = int(qwen_patch_info["ptr"])
    if qwen_patch_tensor.data_ptr() != expected_qwen_ptr:
        raise RuntimeError(
            f"Qwen patch DLPack pointer mismatch: tensor={qwen_patch_tensor.data_ptr()} "
            f"native={expected_qwen_ptr}"
        )
    if tuple(qwen_patch_tensor.shape) != (int(qwen_patch_info["rows"]), int(qwen_patch_info["columns"])):
        raise RuntimeError(
            f"Qwen patch shape mismatch: tensor={tuple(qwen_patch_tensor.shape)} "
            f"native={qwen_patch_info['shape']}"
        )
    if tuple(qwen_patch_tensor.stride()) != (int(qwen_patch_info["columns"]), 1):
        raise RuntimeError(f"Qwen patch tensor must be dense row-major; got stride={qwen_patch_tensor.stride()}")

    audio_info = pipeline.audio_info
    audio_ptr = int(audio_info["ptr"])
    audio_bytes = int(audio_info["bytes"])
    audio_data_ptr = audio_window.data_ptr()
    if not (audio_ptr <= audio_data_ptr < audio_ptr + audio_bytes):
        raise RuntimeError(
            f"Audio DLPack pointer is outside native mirrored audio ring: "
            f"tensor={audio_data_ptr} ring=[{audio_ptr}, {audio_ptr + audio_bytes})"
        )

    if video_nhwc.device.type != "cuda" or qwen_patch_tensor.device.type != "cuda" or audio_window.device.type != "cuda":
        raise RuntimeError("Expected video, Qwen patch, and audio tensors to be CUDA tensors.")


def drain_thread_errors(errors: "queue.SimpleQueue[tuple[str, str]]") -> None:
    if errors.empty():
        return
    messages = []
    while not errors.empty():
        name, tb = errors.get()
        messages.append(f"[{name}]\n{tb}")
    raise RuntimeError("Ingest thread failure:\n" + "\n".join(messages))


def controller_loop(
    config: RuntimeConfig,
    pipeline: CudaStreamFusionPipeline,
    vllm_runner: VLLMRunner,
    asr_runner: WhisperGpuFallbackRunner | None,
    stop_event: threading.Event,
    errors: "queue.SimpleQueue[tuple[str, str]]",
) -> None:
    device = torch.device(config.device)
    period_s = config.controller_period_ms / 1000.0
    deadline = time.perf_counter() + config.duration_sec
    next_tick = time.perf_counter()
    cycle = 0

    vllm_latencies: list[float] = []
    vllm_tokens: list[int] = []
    asr_latencies: list[float] = []
    missed_cycles = 0
    baseline_alloc = torch.cuda.memory_allocated(device)
    baseline_free, baseline_total = torch.cuda.mem_get_info(device)
    baseline_device_used = baseline_total - baseline_free
    hard_vram_gib = RTX_4090_LAPTOP_VRAM_BYTES / (1024**3)
    vllm_budget_gib = RTX_4090_LAPTOP_VRAM_BYTES * config.gpu_memory_utilization / (1024**3)
    fusion_budget_gib = RTX_4090_LAPTOP_VRAM_BYTES * (1.0 - config.gpu_memory_utilization) / (1024**3)

    logging.info(
        "Strict VRAM budget: physical=%.2f GiB, vLLM cap=%.2f GiB, fusion/native reserve=%.2f GiB",
        hard_vram_gib,
        vllm_budget_gib,
        fusion_budget_gib,
    )

    while time.perf_counter() < deadline and not stop_event.is_set():
        drain_thread_errors(errors)
        cycle_start = time.perf_counter()

        video = pipeline.latest_nhwc_tensor()
        qwen_video_payload, qwen_patch_info = pipeline.qwen_vllm_video_payload(fps=config.video_fps)
        qwen_patch_tensor = qwen_video_payload["pixel_values_videos"]
        audio = pipeline.latest_audio_window(config.audio_window_ms)
        verify_input_pointers(
            pipeline,
            video,
            qwen_patch_tensor,
            qwen_patch_info,
            audio,
            config.audio_window_ms,
        )

        text, token_count, latency_ms = vllm_runner.generate(qwen_video_payload, audio, cycle)
        vllm_latencies.append(latency_ms)
        vllm_tokens.append(token_count)

        asr_text = ""
        asr_ms = 0.0
        if asr_runner is not None:
            asr_text, _, asr_ms = asr_runner.transcribe(audio, max_new_tokens=16)
            asr_latencies.append(asr_ms)

        torch.cuda.synchronize(device)
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        device_used = total_bytes - free_bytes
        reserved_fraction = reserved / RTX_4090_LAPTOP_VRAM_BYTES
        latency_per_token = latency_ms / max(token_count, 1)
        token_velocity = 1000.0 * token_count / max(latency_ms, 1.0e-6)

        print(
            f"[cycle {cycle:04d}] "
            f"vLLM={latency_ms:8.2f} ms "
            f"tok={token_count:3d} "
            f"ms/token={latency_per_token:7.2f} "
            f"tok/s={token_velocity:7.2f} "
            f"asr={asr_ms:8.2f} ms "
            f"alloc={allocated / (1024**3):6.3f} GiB "
            f"reserved={reserved / (1024**3):6.3f} GiB "
            f"device_used={device_used / (1024**3):6.3f} GiB "
            f"reserved%16GB={100.0 * reserved_fraction:6.2f}% "
            f"prompt_out={text[:48]!r} "
            f"asr_out={asr_text[:32]!r}",
            flush=True,
        )

        if device_used > RTX_4090_LAPTOP_VRAM_BYTES:
            raise RuntimeError(
                f"CUDA device memory exceeded strict 16 GiB budget: {device_used:,} bytes."
            )

        del video, qwen_video_payload, qwen_patch_tensor, audio
        cycle += 1

        next_tick += period_s
        sleep_s = next_tick - time.perf_counter()
        if sleep_s > 0:
            stop_event.wait(sleep_s)
        else:
            missed_cycles += 1
            next_tick = time.perf_counter()

        if cycle_start > deadline:
            break

    final_alloc = torch.cuda.memory_allocated(device)
    print("\nsummary")
    print(f"controller cycles         : {cycle}")
    print(f"controller missed cycles  : {missed_cycles}")
    print(f"baseline allocated        : {baseline_alloc:,} bytes")
    print(f"baseline device used      : {baseline_device_used:,} bytes")
    print(f"final allocated           : {final_alloc:,} bytes")
    print(f"allocated drift           : {final_alloc - baseline_alloc:,} bytes")
    if vllm_latencies:
        print(f"vLLM avg latency          : {sum(vllm_latencies) / len(vllm_latencies):.2f} ms")
        print(f"vLLM token velocity       : {1000.0 * sum(vllm_tokens) / sum(vllm_latencies):.2f} tok/s")
    if asr_latencies:
        print(f"ASR avg latency           : {sum(asr_latencies) / len(asr_latencies):.2f} ms")


def build_pipeline(config: RuntimeConfig) -> CudaStreamFusionPipeline:
    return CudaStreamFusionPipeline(
        stream_config=StreamConfig(
            width=config.width,
            height=config.height,
            capacity=config.ring_capacity,
            device=config.device,
        ),
        audio_config=AudioConfig(
            sample_rate_hz=16_000,
            capacity_ms=config.audio_capacity_ms,
            default_window_ms=config.audio_window_ms,
        ),
    )


def default_qwen_prompt(modality: str) -> str:
    vision_token = "video_pad" if modality == "video" else "image_pad"
    return (
        "<|im_start|>system\n"
        "You are a concise real-time perception assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"<|vision_start|><|{vision_token}|><|vision_end|>\n"
        "Describe the current stream state in one short sentence.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def parse_args() -> RuntimeConfig:
    parser = argparse.ArgumentParser(
        description="Run a one-minute threaded cuda-stream-fusion to vLLM multimodal inference pump."
    )
    parser.add_argument("--duration-sec", type=float, default=60.0)
    parser.add_argument("--controller-period-ms", type=float, default=100.0)
    parser.add_argument("--video-fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--ring-capacity", type=int, default=4)
    parser.add_argument("--audio-packet-ms", type=int, default=20)
    parser.add_argument("--audio-window-ms", type=int, default=3000)
    parser.add_argument("--audio-capacity-ms", type=int, default=30000)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--vllm-modality", choices=("image", "video"), default="video")
    parser.add_argument("--route-audio-to-vllm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--asr-mode", choices=("whisper", "vllm", "none"), default="none")
    parser.add_argument("--whisper-model", type=str, default="openai/whisper-tiny")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--cpu-offload-gb", type=float, default=0.0)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--no-trust-remote-code", action="store_false", dest="trust_remote_code")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--no-install-missing-optional", action="store_false", dest="install_missing_optional")
    parser.set_defaults(install_missing_optional=True)
    args = parser.parse_args()

    if args.asr_mode == "vllm":
        args.route_audio_to_vllm = True
    if args.asr_mode == "whisper":
        logging.warning(
            "Whisper fallback was requested. This allocates an additional ASR model on the GPU "
            "and is not the default strict 16 GB profile."
        )
    if args.duration_sec <= 0:
        parser.error("--duration-sec must be positive")
    if args.controller_period_ms <= 0:
        parser.error("--controller-period-ms must be positive")
    if args.audio_window_ms > args.audio_capacity_ms:
        parser.error("--audio-window-ms must not exceed --audio-capacity-ms")
    if abs(args.gpu_memory_utilization - VLLM_RESERVED_FRACTION) > 1.0e-9:
        parser.error("--gpu-memory-utilization must remain exactly 0.80 for the GPU-only 16 GB profile")
    if args.dtype != "bfloat16":
        parser.error("--dtype must remain 'bfloat16' for the strict 16 GB profile")
    if args.cpu_offload_gb < 0.0:
        parser.error("--cpu-offload-gb must be non-negative")
    if args.max_num_seqs != 1:
        parser.error("--max-num-seqs must remain 1 to avoid hidden concurrent-request VRAM pressure")
    qwen_grid_h = ((args.height + 28 - 1) // 28) * 2
    qwen_grid_w = ((args.width + 28 - 1) // 28) * 2
    qwen_visual_tokens = (qwen_grid_h * qwen_grid_w) // 4
    minimum_model_len = qwen_visual_tokens + 128 + args.max_tokens
    if args.max_model_len < minimum_model_len:
        parser.error(
            f"--max-model-len must be at least {minimum_model_len} for this Qwen patch grid "
            f"({qwen_grid_h}x{qwen_grid_w}, {qwen_visual_tokens} visual tokens)"
        )

    if args.prompt is None:
        args.prompt = default_qwen_prompt(args.vllm_modality)

    return RuntimeConfig(**vars(args))


def main() -> None:
    config = parse_args()
    assert_mmsf_context()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available to PyTorch in this environment.")

    device = torch.device(config.device)
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()

    if not any(FUSION_ENGINE_DIR.glob("fusion_core*.so")):
        raise SystemExit(f"No compiled fusion_core extension was found in {FUSION_ENGINE_DIR}.")

    local_weight_prefix = ensure_environment_setup(config.model)
    config = replace(config, model=local_weight_prefix)

    logging.info(
        "16 GB inference profile: model=%s, dtype=%s, gpu_memory_utilization=%.2f, "
        "cpu_offload_gb=%.1f, max_model_len=%d, max_num_seqs=%d, route_audio_to_vllm=%s",
        config.model,
        config.dtype,
        config.gpu_memory_utilization,
        config.cpu_offload_gb,
        config.max_model_len,
        config.max_num_seqs,
        config.route_audio_to_vllm,
    )

    pipeline = build_pipeline(config)
    stop_event = threading.Event()
    errors: "queue.SimpleQueue[tuple[str, str]]" = queue.SimpleQueue()

    video_thread = VideoIngestThread(
        pipeline=pipeline,
        generator=SyntheticYUV420pGenerator(config.width, config.height),
        fps=config.video_fps,
        stop_event=stop_event,
        errors=errors,
    )
    audio_thread = AudioIngestThread(
        pipeline=pipeline,
        generator=SyntheticAudioGenerator(16_000, config.audio_packet_ms),
        packet_ms=config.audio_packet_ms,
        stop_event=stop_event,
        errors=errors,
    )

    print("initializing vLLM runner", flush=True)
    vllm_runner = VLLMRunner(config)

    asr_runner = None
    if config.asr_mode == "whisper":
        print("initializing GPU Whisper fallback runner", flush=True)
        asr_runner = WhisperGpuFallbackRunner(
            config.whisper_model,
            device=device,
            install_missing=config.install_missing_optional,
        )

    print("starting ingest threads", flush=True)
    video_thread.start()
    audio_thread.start()

    try:
        controller_loop(config, pipeline, vllm_runner, asr_runner, stop_event, errors)
    finally:
        stop_event.set()
        video_thread.join(timeout=5.0)
        audio_thread.join(timeout=5.0)
        print(
            "thread exit: "
            f"video_iterations={video_thread.iterations} video_overruns={video_thread.overruns} "
            f"audio_iterations={audio_thread.iterations} audio_overruns={audio_thread.overruns}",
            flush=True,
        )
        drain_thread_errors(errors)


if __name__ == "__main__":
    main()
