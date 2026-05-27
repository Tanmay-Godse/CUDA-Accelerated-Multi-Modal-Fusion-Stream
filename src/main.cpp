#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "stream_fusion.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" std::size_t csf_yuv420p_frame_bytes(int width, int height);

extern "C" cudaError_t csf_launch_yuv420p_to_rgb_float(
    const std::uint8_t* yuv420p,
    float* rgb,
    int width,
    int height,
    cudaStream_t stream
);

namespace py = pybind11;

namespace {

std::string cuda_error_text(cudaError_t error, const char* expression, const char* file, int line) {
    std::ostringstream stream;
    stream << "CUDA call failed at " << file << ':' << line << " for `" << expression
           << "`: " << cudaGetErrorName(error) << " - " << cudaGetErrorString(error);
    return stream.str();
}

#define CSF_CUDA_CHECK(expr)                                                                    \
    do {                                                                                        \
        const cudaError_t _csf_error = (expr);                                                   \
        if (_csf_error != cudaSuccess) {                                                         \
            throw std::runtime_error(cuda_error_text(_csf_error, #expr, __FILE__, __LINE__));    \
        }                                                                                       \
    } while (0)

namespace dlpack_abi {

// Minimal DLPack ABI declarations. Keeping these local avoids a hard C++ link
// dependency on libtorch while still letting torch.utils.dlpack consume the
// CUDA allocation without copying.
enum DLDeviceType : int32_t {
    kDLCPU = 1,
    kDLCUDA = 2,
};

enum DLDataTypeCode : uint8_t {
    kDLFloat = 2,
};

struct DLDevice {
    DLDeviceType device_type;
    int32_t device_id;
};

struct DLDataType {
    uint8_t code;
    uint8_t bits;
    uint16_t lanes;
};

struct DLTensor {
    void* data;
    DLDevice device;
    int32_t ndim;
    DLDataType dtype;
    int64_t* shape;
    int64_t* strides;
    uint64_t byte_offset;
};

struct DLManagedTensor {
    DLTensor dl_tensor;
    void* manager_ctx;
    void (*deleter)(DLManagedTensor* self);
};

struct TensorContext {
    int64_t shape[4];
    int64_t strides[4];
};

void managed_tensor_deleter(DLManagedTensor* self) {
    if (self == nullptr) {
        return;
    }
    delete static_cast<TensorContext*>(self->manager_ctx);
    delete self;
}

void capsule_destructor(PyObject* capsule) {
    if (PyCapsule_IsValid(capsule, "used_dltensor")) {
        return;
    }

    auto* managed = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(capsule, "dltensor"));
    if (managed == nullptr) {
        PyErr_Clear();
        return;
    }

    if (managed->deleter != nullptr) {
        managed->deleter(managed);
    }
}

py::capsule make_cuda_hwc_float_capsule(
    float* data,
    int width,
    int height,
    int channels,
    int device_id
) {
    auto* context = new TensorContext{};
    context->shape[0] = static_cast<int64_t>(height);
    context->shape[1] = static_cast<int64_t>(width);
    context->shape[2] = static_cast<int64_t>(channels);
    context->strides[0] = static_cast<int64_t>(width) * static_cast<int64_t>(channels);
    context->strides[1] = static_cast<int64_t>(channels);
    context->strides[2] = 1;

    auto* managed = new DLManagedTensor{};
    managed->dl_tensor.data = data;
    managed->dl_tensor.device = DLDevice{kDLCUDA, device_id};
    managed->dl_tensor.ndim = 3;
    managed->dl_tensor.dtype = DLDataType{kDLFloat, 32, 1};
    managed->dl_tensor.shape = context->shape;
    managed->dl_tensor.strides = context->strides;
    managed->dl_tensor.byte_offset = 0;
    managed->manager_ctx = context;
    managed->deleter = managed_tensor_deleter;

    return py::capsule(managed, "dltensor", capsule_destructor);
}

py::capsule make_cuda_1d_float_capsule(
    float* data,
    std::size_t samples,
    int device_id
) {
    auto* context = new TensorContext{};
    context->shape[0] = static_cast<int64_t>(samples);
    context->strides[0] = 1;

    auto* managed = new DLManagedTensor{};
    managed->dl_tensor.data = data;
    managed->dl_tensor.device = DLDevice{kDLCUDA, device_id};
    managed->dl_tensor.ndim = 1;
    managed->dl_tensor.dtype = DLDataType{kDLFloat, 32, 1};
    managed->dl_tensor.shape = context->shape;
    managed->dl_tensor.strides = context->strides;
    managed->dl_tensor.byte_offset = 0;
    managed->manager_ctx = context;
    managed->deleter = managed_tensor_deleter;

    return py::capsule(managed, "dltensor", capsule_destructor);
}

}  // namespace dlpack_abi

struct FrameSlot {
    std::uint8_t* yuv420p = nullptr;
    float* rgb = nullptr;
    cudaStream_t stream = nullptr;
    std::uint64_t sequence = 0;
    bool has_decoded_frame = false;
    bool decode_pending = false;
};

class UnifiedVideoRingBuffer {
public:
    UnifiedVideoRingBuffer(int width, int height, int capacity, int device_id)
        : width_(width),
          height_(height),
          capacity_(capacity),
          device_id_(device_id),
          yuv_bytes_(csf_yuv420p_frame_bytes(width, height)),
          rgb_bytes_(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) *
                     static_cast<std::size_t>(channels_) * sizeof(float)),
          slots_(static_cast<std::size_t>(capacity)) {
        if (width_ <= 0 || height_ <= 0) {
            throw std::invalid_argument("Frame width and height must be positive.");
        }
        if (capacity_ <= 0) {
            throw std::invalid_argument("Ring-buffer capacity must be positive.");
        }
        if (yuv_bytes_ == 0 || rgb_bytes_ == 0) {
            throw std::invalid_argument("Computed frame allocation size is zero.");
        }

        CSF_CUDA_CHECK(cudaSetDevice(device_id_));

        for (auto& slot : slots_) {
            void* yuv_ptr = nullptr;
            void* rgb_ptr = nullptr;

            CSF_CUDA_CHECK(cudaStreamCreateWithFlags(&slot.stream, cudaStreamNonBlocking));
            CSF_CUDA_CHECK(cudaMallocManaged(&yuv_ptr, yuv_bytes_));
            CSF_CUDA_CHECK(cudaMallocManaged(&rgb_ptr, rgb_bytes_));

            slot.yuv420p = static_cast<std::uint8_t*>(yuv_ptr);
            slot.rgb = static_cast<float*>(rgb_ptr);

            // The decoder or capture backend should write YUV directly into
            // this managed pointer. RGB is preferred on the GPU because PyTorch
            // will consume it from CUDA kernels.
            CSF_CUDA_CHECK(cudaMemAdvise(slot.yuv420p, yuv_bytes_, cudaMemAdviseSetAccessedBy, device_id_));
            CSF_CUDA_CHECK(cudaMemAdvise(slot.rgb, rgb_bytes_, cudaMemAdviseSetPreferredLocation, device_id_));
            CSF_CUDA_CHECK(cudaMemAdvise(slot.rgb, rgb_bytes_, cudaMemAdviseSetAccessedBy, device_id_));
            CSF_CUDA_CHECK(cudaMemsetAsync(slot.yuv420p, 0, yuv_bytes_, slot.stream));
            CSF_CUDA_CHECK(cudaMemsetAsync(slot.rgb, 0, rgb_bytes_, slot.stream));
            CSF_CUDA_CHECK(cudaMemPrefetchAsync(slot.rgb, rgb_bytes_, device_id_, slot.stream));
            CSF_CUDA_CHECK(cudaStreamSynchronize(slot.stream));
        }
    }

    UnifiedVideoRingBuffer(const UnifiedVideoRingBuffer&) = delete;
    UnifiedVideoRingBuffer& operator=(const UnifiedVideoRingBuffer&) = delete;

    ~UnifiedVideoRingBuffer() {
        for (auto& slot : slots_) {
            if (slot.stream != nullptr) {
                cudaStreamSynchronize(slot.stream);
            }
            if (slot.yuv420p != nullptr) {
                cudaFree(slot.yuv420p);
                slot.yuv420p = nullptr;
            }
            if (slot.rgb != nullptr) {
                cudaFree(slot.rgb);
                slot.rgb = nullptr;
            }
            if (slot.stream != nullptr) {
                cudaStreamDestroy(slot.stream);
                slot.stream = nullptr;
            }
        }
    }

    py::dict get_yuv_input_ptr() const {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto& slot = slots_.at(write_index_);

        py::dict info;
        info["ptr"] = reinterpret_cast<std::uintptr_t>(slot.yuv420p);
        info["bytes"] = yuv_bytes_;
        info["width"] = width_;
        info["height"] = height_;
        info["format"] = "yuv420p";
        info["slot"] = static_cast<int>(write_index_);
        info["ownership"] = "cudaMallocManaged";
        return info;
    }

    py::dict get_cuda_tensor_ptr() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return tensor_metadata_locked(read_index_);
    }

    py::capsule to_dlpack() {
        float* data = nullptr;
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex_);
            auto& slot = slots_.at(read_index_);
            if (slot.decode_pending) {
                CSF_CUDA_CHECK(cudaStreamSynchronize(slot.stream));
                slot.decode_pending = false;
            }
            data = slot.rgb;
        }

        return dlpack_abi::make_cuda_hwc_float_capsule(
            data,
            width_,
            height_,
            channels_,
            device_id_
        );
    }

    py::dict decode_current(bool synchronize) {
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex_);
            auto& slot = slots_.at(write_index_);

            CSF_CUDA_CHECK(cudaSetDevice(device_id_));
            CSF_CUDA_CHECK(csf_launch_yuv420p_to_rgb_float(
                slot.yuv420p,
                slot.rgb,
                width_,
                height_,
                slot.stream
            ));
            CSF_CUDA_CHECK(cudaMemPrefetchAsync(slot.rgb, rgb_bytes_, device_id_, slot.stream));

            if (synchronize) {
                CSF_CUDA_CHECK(cudaStreamSynchronize(slot.stream));
            }

            slot.sequence = ++sequence_counter_;
            slot.has_decoded_frame = true;
            slot.decode_pending = !synchronize;
            read_index_ = write_index_;
            write_index_ = (write_index_ + 1) % static_cast<std::size_t>(capacity_);
        }

        return get_cuda_tensor_ptr();
    }

    py::dict submit_yuv420p(py::buffer frame, bool decode, bool synchronize) {
        const py::buffer_info buffer = frame.request();
        std::size_t available_bytes = static_cast<std::size_t>(buffer.itemsize);
        for (const auto extent : buffer.shape) {
            available_bytes *= static_cast<std::size_t>(extent);
        }

        if (available_bytes < yuv_bytes_) {
            std::ostringstream stream;
            stream << "YUV420p frame buffer is too small: got " << available_bytes
                   << " bytes, expected at least " << yuv_bytes_ << " bytes.";
            throw std::invalid_argument(stream.str());
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto& slot = slots_.at(write_index_);
            std::memcpy(slot.yuv420p, buffer.ptr, yuv_bytes_);
            CSF_CUDA_CHECK(cudaSetDevice(device_id_));
            CSF_CUDA_CHECK(cudaMemPrefetchAsync(slot.yuv420p, yuv_bytes_, device_id_, slot.stream));
        }

        if (decode) {
            return decode_current(synchronize);
        }

        return get_yuv_input_ptr();
    }

    int width() const noexcept {
        return width_;
    }

    int height() const noexcept {
        return height_;
    }

    int channels() const noexcept {
        return channels_;
    }

    std::size_t yuv_bytes() const noexcept {
        return yuv_bytes_;
    }

    std::size_t rgb_bytes() const noexcept {
        return rgb_bytes_;
    }

private:
    py::dict tensor_metadata_locked(std::size_t slot_index) const {
        const auto& slot = slots_.at(slot_index);

        py::dict info;
        info["ptr"] = reinterpret_cast<std::uintptr_t>(slot.rgb);
        info["bytes"] = rgb_bytes_;
        info["width"] = width_;
        info["height"] = height_;
        info["channels"] = channels_;
        info["shape"] = py::make_tuple(height_, width_, channels_);
        info["strides"] = py::make_tuple(width_ * channels_, channels_, 1);
        info["dtype"] = "float32";
        info["device"] = "cuda:" + std::to_string(device_id_);
        info["layout"] = "HWC";
        info["slot"] = static_cast<int>(slot_index);
        info["sequence"] = slot.sequence;
        info["has_decoded_frame"] = slot.has_decoded_frame;
        info["ownership"] = "cudaMallocManaged";
        return info;
    }

    static constexpr int channels_ = 3;
    int width_;
    int height_;
    int capacity_;
    int device_id_;
    std::size_t yuv_bytes_;
    std::size_t rgb_bytes_;
    mutable std::mutex mutex_;
    std::vector<FrameSlot> slots_;
    std::size_t write_index_ = 0;
    std::size_t read_index_ = 0;
    std::uint64_t sequence_counter_ = 0;
};

class UnifiedAudioRingBuffer {
public:
    UnifiedAudioRingBuffer(int sample_rate_hz, int capacity_ms, int device_id)
        : sample_rate_hz_(sample_rate_hz),
          capacity_ms_(capacity_ms),
          device_id_(device_id),
          capacity_samples_(csf::audio_samples_for_duration_ms(capacity_ms, sample_rate_hz)),
          mirrored_samples_(capacity_samples_ * 2),
          mirrored_bytes_(mirrored_samples_ * sizeof(float)) {
        if (sample_rate_hz_ != csf::kAudioSampleRateHz) {
            throw std::invalid_argument("Audio sample rate must be exactly 16000 Hz for ASR/VLM input.");
        }
        if (capacity_ms_ <= 0 || capacity_samples_ == 0) {
            throw std::invalid_argument("Audio ring capacity must be positive.");
        }

        CSF_CUDA_CHECK(cudaSetDevice(device_id_));
        void* ptr = nullptr;
        CSF_CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        CSF_CUDA_CHECK(cudaMallocManaged(&ptr, mirrored_bytes_));
        audio_ = static_cast<float*>(ptr);

        // The second half mirrors the first half. A rolling window can always
        // be exposed as one contiguous DLPack tensor, even when the logical
        // ring wraps around the capacity boundary.
        CSF_CUDA_CHECK(cudaMemAdvise(audio_, mirrored_bytes_, cudaMemAdviseSetAccessedBy, device_id_));
        CSF_CUDA_CHECK(cudaMemAdvise(audio_, mirrored_bytes_, cudaMemAdviseSetPreferredLocation, device_id_));
        CSF_CUDA_CHECK(cudaMemsetAsync(audio_, 0, mirrored_bytes_, stream_));
        CSF_CUDA_CHECK(cudaMemPrefetchAsync(audio_, mirrored_bytes_, device_id_, stream_));
        CSF_CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    UnifiedAudioRingBuffer(const UnifiedAudioRingBuffer&) = delete;
    UnifiedAudioRingBuffer& operator=(const UnifiedAudioRingBuffer&) = delete;

    ~UnifiedAudioRingBuffer() {
        if (stream_ != nullptr) {
            cudaStreamSynchronize(stream_);
        }
        if (audio_ != nullptr) {
            cudaFree(audio_);
            audio_ = nullptr;
        }
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
    }

    py::dict submit_pcm_float32(py::buffer samples, bool synchronize) {
        const py::buffer_info buffer = samples.request();
        if (buffer.itemsize != static_cast<ssize_t>(sizeof(float))) {
            throw std::invalid_argument("Audio packet must contain float32 PCM samples.");
        }

        std::size_t sample_count = 1;
        for (const auto extent : buffer.shape) {
            sample_count *= static_cast<std::size_t>(extent);
        }

        if (!is_c_contiguous(buffer)) {
            throw std::invalid_argument("Audio packet buffer must be C-contiguous.");
        }

        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex_);
            append_samples_locked(static_cast<const float*>(buffer.ptr), sample_count);
            CSF_CUDA_CHECK(cudaSetDevice(device_id_));
            CSF_CUDA_CHECK(cudaMemPrefetchAsync(audio_, mirrored_bytes_, device_id_, stream_));
            prefetch_pending_ = true;
            if (synchronize) {
                CSF_CUDA_CHECK(cudaStreamSynchronize(stream_));
                prefetch_pending_ = false;
            }
        }

        return get_audio_info();
    }

    py::dict get_audio_info() const {
        std::lock_guard<std::mutex> lock(mutex_);

        py::dict info;
        info["ptr"] = reinterpret_cast<std::uintptr_t>(audio_);
        info["mirror_ptr"] = reinterpret_cast<std::uintptr_t>(audio_ + capacity_samples_);
        info["sample_rate_hz"] = sample_rate_hz_;
        info["channels"] = csf::kAudioChannels;
        info["capacity_ms"] = capacity_ms_;
        info["capacity_samples"] = capacity_samples_;
        info["mirrored_samples"] = mirrored_samples_;
        info["bytes"] = mirrored_bytes_;
        info["total_samples_written"] = total_samples_written_;
        info["available_samples"] = std::min<std::uint64_t>(
            total_samples_written_,
            static_cast<std::uint64_t>(capacity_samples_)
        );
        info["dtype"] = "float32";
        info["device"] = "cuda:" + std::to_string(device_id_);
        info["layout"] = "mono";
        info["ownership"] = "cudaMallocManaged";
        return info;
    }

    py::dict latest_window_info(int duration_ms) {
        std::lock_guard<std::mutex> lock(mutex_);
        const std::size_t requested_samples = validate_duration_locked(duration_ms);
        const float* window_ptr = window_start_ptr_locked(requested_samples);

        return latest_window_info_locked(duration_ms, requested_samples, window_ptr);
    }

    py::dict latest_window_bundle(int duration_ms) {
        std::lock_guard<std::mutex> lock(mutex_);
        const std::size_t requested_samples = validate_duration_locked(duration_ms);
        if (prefetch_pending_) {
            CSF_CUDA_CHECK(cudaStreamSynchronize(stream_));
            prefetch_pending_ = false;
        }
        float* window_ptr = window_start_ptr_locked(requested_samples);

        py::dict bundle;
        bundle["info"] = latest_window_info_locked(duration_ms, requested_samples, window_ptr);
        bundle["capsule"] = dlpack_abi::make_cuda_1d_float_capsule(
            window_ptr,
            requested_samples,
            device_id_
        );
        return bundle;
    }

    py::capsule latest_window_to_dlpack(int duration_ms) {
        float* window_ptr = nullptr;
        std::size_t requested_samples = 0;
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex_);
            requested_samples = validate_duration_locked(duration_ms);
            if (prefetch_pending_) {
                CSF_CUDA_CHECK(cudaStreamSynchronize(stream_));
                prefetch_pending_ = false;
            }
            window_ptr = window_start_ptr_locked(requested_samples);
        }

        return dlpack_abi::make_cuda_1d_float_capsule(
            window_ptr,
            requested_samples,
            device_id_
        );
    }

    int sample_rate_hz() const noexcept {
        return sample_rate_hz_;
    }

    int capacity_ms() const noexcept {
        return capacity_ms_;
    }

    std::size_t capacity_samples() const noexcept {
        return capacity_samples_;
    }

private:
    py::dict latest_window_info_locked(
        int duration_ms,
        std::size_t requested_samples,
        const float* window_ptr
    ) const {
        py::dict info;
        info["ptr"] = reinterpret_cast<std::uintptr_t>(window_ptr);
        info["samples"] = requested_samples;
        info["duration_ms"] = duration_ms;
        info["sample_rate_hz"] = sample_rate_hz_;
        info["channels"] = csf::kAudioChannels;
        info["shape"] = py::make_tuple(requested_samples);
        info["strides"] = py::make_tuple(1);
        info["dtype"] = "float32";
        info["device"] = "cuda:" + std::to_string(device_id_);
        info["available_samples"] = std::min<std::uint64_t>(
            total_samples_written_,
            static_cast<std::uint64_t>(requested_samples)
        );
        info["total_samples_written"] = total_samples_written_;
        info["zero_padded_prefix_samples"] =
            total_samples_written_ < requested_samples
                ? requested_samples - static_cast<std::size_t>(total_samples_written_)
                : 0;
        return info;
    }

    static bool is_c_contiguous(const py::buffer_info& buffer) {
        if (buffer.ndim == 0) {
            return true;
        }

        ssize_t expected_stride = static_cast<ssize_t>(buffer.itemsize);
        for (ssize_t axis = buffer.ndim - 1; axis >= 0; --axis) {
            if (buffer.shape[axis] == 0) {
                return true;
            }
            if (buffer.strides[axis] != expected_stride) {
                return false;
            }
            expected_stride *= buffer.shape[axis];
        }
        return true;
    }

    void append_samples_locked(const float* src, std::size_t sample_count) {
        if (src == nullptr || sample_count == 0) {
            return;
        }

        if (sample_count >= capacity_samples_) {
            const std::size_t skipped_samples = sample_count - capacity_samples_;
            src += skipped_samples;
            sample_count = capacity_samples_;
            total_samples_written_ += skipped_samples;
        }

        std::size_t copied = 0;
        while (copied < sample_count) {
            const std::size_t write_offset =
                static_cast<std::size_t>(total_samples_written_ % capacity_samples_);
            const std::size_t contiguous_samples =
                std::min(sample_count - copied, capacity_samples_ - write_offset);
            const std::size_t bytes = contiguous_samples * sizeof(float);

            std::memcpy(audio_ + write_offset, src + copied, bytes);
            std::memcpy(audio_ + capacity_samples_ + write_offset, src + copied, bytes);

            copied += contiguous_samples;
            total_samples_written_ += contiguous_samples;
        }
    }

    std::size_t validate_duration_locked(int duration_ms) const {
        const std::size_t requested_samples =
            csf::audio_samples_for_duration_ms(duration_ms, sample_rate_hz_);
        if (requested_samples == 0) {
            throw std::invalid_argument("Audio window duration must be positive.");
        }
        if (requested_samples > capacity_samples_) {
            std::ostringstream stream;
            stream << "Requested audio window of " << duration_ms
                   << " ms exceeds ring capacity of " << capacity_ms_ << " ms.";
            throw std::invalid_argument(stream.str());
        }
        return requested_samples;
    }

    float* window_start_ptr_locked(std::size_t requested_samples) const {
        std::size_t start_mod = 0;
        if (total_samples_written_ >= requested_samples) {
            start_mod =
                static_cast<std::size_t>((total_samples_written_ - requested_samples) % capacity_samples_);
        } else {
            start_mod = capacity_samples_ - (requested_samples - static_cast<std::size_t>(total_samples_written_));
        }
        return audio_ + start_mod;
    }

    int sample_rate_hz_;
    int capacity_ms_;
    int device_id_;
    std::size_t capacity_samples_;
    std::size_t mirrored_samples_;
    std::size_t mirrored_bytes_;
    float* audio_ = nullptr;
    cudaStream_t stream_ = nullptr;
    mutable std::mutex mutex_;
    std::uint64_t total_samples_written_ = 0;
    bool prefetch_pending_ = false;
};

std::mutex g_core_mutex;
std::unique_ptr<UnifiedVideoRingBuffer> g_core;
std::mutex g_audio_mutex;
std::unique_ptr<UnifiedAudioRingBuffer> g_audio_core;

UnifiedVideoRingBuffer& require_global_core() {
    std::lock_guard<std::mutex> lock(g_core_mutex);
    if (!g_core) {
        throw std::runtime_error("fusion_core is not initialized. Call fusion_core.initialize(width, height) first.");
    }
    return *g_core;
}

UnifiedAudioRingBuffer& require_global_audio_core() {
    std::lock_guard<std::mutex> lock(g_audio_mutex);
    if (!g_audio_core) {
        throw std::runtime_error("Audio core is not initialized. Call fusion_core.initialize_audio() first.");
    }
    return *g_audio_core;
}

}  // namespace

PYBIND11_MODULE(fusion_core, module) {
    module.doc() =
        "CUDA stream fusion core: unified-memory video/audio ring buffers, CUDA decode, and DLPack export.";
    module.attr("__version__") = "0.1.0";
    module.attr("AUDIO_SAMPLE_RATE_HZ") = csf::kAudioSampleRateHz;
    module.attr("AUDIO_CHANNELS") = csf::kAudioChannels;

    py::class_<UnifiedVideoRingBuffer>(module, "StreamingCore")
        .def(py::init<int, int, int, int>(),
             py::arg("width"),
             py::arg("height"),
             py::arg("capacity") = 4,
             py::arg("device_id") = 0)
        .def("get_yuv_input_ptr", &UnifiedVideoRingBuffer::get_yuv_input_ptr,
             "Return the writable YUV420p unified-memory pointer for the next ring slot.")
        .def("get_cuda_tensor_ptr", &UnifiedVideoRingBuffer::get_cuda_tensor_ptr,
             "Return metadata for the latest decoded RGB float32 CUDA tensor backing store.")
        .def("to_dlpack", &UnifiedVideoRingBuffer::to_dlpack,
             "Return a single-use DLPack capsule for zero-copy torch tensor construction.")
        .def("decode_current", &UnifiedVideoRingBuffer::decode_current,
             py::arg("synchronize") = true,
             "Decode the current YUV420p ring slot into RGB float32 and advance the ring.")
        .def("submit_yuv420p", &UnifiedVideoRingBuffer::submit_yuv420p,
             py::arg("frame"),
             py::arg("decode") = true,
             py::arg("synchronize") = true,
             "Debug/test path: copy a host YUV420p buffer into unified memory, optionally decode it.")
        .def_property_readonly("width", &UnifiedVideoRingBuffer::width)
        .def_property_readonly("height", &UnifiedVideoRingBuffer::height)
        .def_property_readonly("channels", &UnifiedVideoRingBuffer::channels)
        .def_property_readonly("yuv_bytes", &UnifiedVideoRingBuffer::yuv_bytes)
        .def_property_readonly("rgb_bytes", &UnifiedVideoRingBuffer::rgb_bytes);

    py::class_<UnifiedAudioRingBuffer>(module, "AudioStreamCore")
        .def(py::init<int, int, int>(),
             py::arg("sample_rate_hz") = csf::kAudioSampleRateHz,
             py::arg("capacity_ms") = csf::kDefaultAudioCapacityMs,
             py::arg("device_id") = 0)
        .def("submit_pcm_float32", &UnifiedAudioRingBuffer::submit_pcm_float32,
             py::arg("samples"),
             py::arg("synchronize") = true,
             "Append mono float32 PCM samples to the audio ring.")
        .def("get_audio_info", &UnifiedAudioRingBuffer::get_audio_info,
             "Return metadata for the mirrored CUDA managed audio ring.")
        .def("get_latest_audio_window", &UnifiedAudioRingBuffer::latest_window_to_dlpack,
             py::arg("duration_ms"),
             "Return a DLPack capsule for the latest mono float32 audio window.")
        .def("get_latest_audio_window_bundle", &UnifiedAudioRingBuffer::latest_window_bundle,
             py::arg("duration_ms"),
             "Return DLPack capsule plus pointer metadata for one locked audio-window snapshot.")
        .def("get_latest_audio_window_info", &UnifiedAudioRingBuffer::latest_window_info,
             py::arg("duration_ms"),
             "Return pointer/shape metadata for the latest audio window.")
        .def_property_readonly("sample_rate_hz", &UnifiedAudioRingBuffer::sample_rate_hz)
        .def_property_readonly("capacity_ms", &UnifiedAudioRingBuffer::capacity_ms)
        .def_property_readonly("capacity_samples", &UnifiedAudioRingBuffer::capacity_samples);

    module.def("initialize",
        [](int width, int height, int capacity, int device_id) {
            std::lock_guard<std::mutex> lock(g_core_mutex);
            g_core = std::make_unique<UnifiedVideoRingBuffer>(width, height, capacity, device_id);
            return g_core->get_cuda_tensor_ptr();
        },
        py::arg("width"),
        py::arg("height"),
        py::arg("capacity") = 4,
        py::arg("device_id") = 0,
        "Initialize the process-global streaming core.");

    module.def("shutdown",
        []() {
            {
                std::lock_guard<std::mutex> lock(g_core_mutex);
                g_core.reset();
            }
            {
                std::lock_guard<std::mutex> lock(g_audio_mutex);
                g_audio_core.reset();
            }
        },
        "Release the process-global video/audio cores and their CUDA allocations.");

    module.def("initialize_audio",
        [](int sample_rate_hz, int capacity_ms, int device_id) {
            std::lock_guard<std::mutex> lock(g_audio_mutex);
            g_audio_core = std::make_unique<UnifiedAudioRingBuffer>(
                sample_rate_hz,
                capacity_ms,
                device_id
            );
            return g_audio_core->get_audio_info();
        },
        py::arg("sample_rate_hz") = csf::kAudioSampleRateHz,
        py::arg("capacity_ms") = csf::kDefaultAudioCapacityMs,
        py::arg("device_id") = 0,
        "Initialize the process-global mono float32 audio ring.");

    module.def("shutdown_audio",
        []() {
            std::lock_guard<std::mutex> lock(g_audio_mutex);
            g_audio_core.reset();
        },
        "Release the process-global audio core and its CUDA allocations.");

    module.def("get_yuv_input_ptr",
        []() {
            return require_global_core().get_yuv_input_ptr();
        },
        "Return the writable YUV420p unified-memory pointer for the global core.");

    module.def("get_cuda_tensor_ptr",
        []() {
            return require_global_core().get_cuda_tensor_ptr();
        },
        "Return metadata for the latest decoded RGB float32 CUDA allocation.");

    module.def("to_dlpack",
        []() {
            return require_global_core().to_dlpack();
        },
        "Return a single-use DLPack capsule for the latest global RGB frame.");

    module.def("decode_current",
        [](bool synchronize) {
            return require_global_core().decode_current(synchronize);
        },
        py::arg("synchronize") = true,
        "Decode the current global YUV420p ring slot and advance the ring.");

    module.def("submit_yuv420p",
        [](py::buffer frame, bool decode, bool synchronize) {
            return require_global_core().submit_yuv420p(frame, decode, synchronize);
        },
        py::arg("frame"),
        py::arg("decode") = true,
        py::arg("synchronize") = true,
        "Debug/test path for host-originated YUV420p frames. Avoid this in the streaming hot path.");

    module.def("submit_audio_pcm_float32",
        [](py::buffer samples, bool synchronize) {
            return require_global_audio_core().submit_pcm_float32(samples, synchronize);
        },
        py::arg("samples"),
        py::arg("synchronize") = true,
        "Append mono 16 kHz float32 PCM samples to the global audio ring.");

    module.def("get_audio_info",
        []() {
            return require_global_audio_core().get_audio_info();
        },
        "Return metadata for the global mirrored audio ring.");

    module.def("get_latest_audio_window",
        [](int duration_ms) {
            return require_global_audio_core().latest_window_to_dlpack(duration_ms);
        },
        py::arg("duration_ms"),
        "Return a DLPack capsule for the latest mono float32 audio window.");

    module.def("get_latest_audio_window_bundle",
        [](int duration_ms) {
            return require_global_audio_core().latest_window_bundle(duration_ms);
        },
        py::arg("duration_ms"),
        "Return DLPack capsule plus pointer metadata for one locked audio-window snapshot.");

    module.def("get_latest_audio_window_info",
        [](int duration_ms) {
            return require_global_audio_core().latest_window_info(duration_ms);
        },
        py::arg("duration_ms"),
        "Return metadata for the latest mono float32 audio window.");
}
