#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace {

constexpr int kQwenPatchSize = 14;
constexpr int kQwenTemporalPatchSize = 2;
constexpr int kQwenChannels = 3;
constexpr int kQwenSpatialMergeSize = 2;
constexpr int kQwenPatchVectorSize =
    kQwenChannels * kQwenTemporalPatchSize * kQwenPatchSize * kQwenPatchSize;

__device__ __forceinline__ int clamp_u8_int(int value) {
    return value < 0 ? 0 : (value > 255 ? 255 : value);
}

__device__ __forceinline__ float normalize_u8_to_float(int value) {
    return static_cast<float>(clamp_u8_int(value)) * 0.00392156862745098039216f;
}

__device__ __forceinline__ int min_i(int lhs, int rhs) {
    return lhs < rhs ? lhs : rhs;
}

// Converts a planar YUV420p frame into interleaved RGB float32 in HWC order.
//
// Layout:
//   yuv420p = [Y plane: W*H][U plane: ceil(W/2)*ceil(H/2)][V plane: same]
//   rgb     = [R,G,B, R,G,B, ...] with shape [H, W, 3]
//
// The kernel uses shared memory for the subsampled U/V planes because every
// 2x2 luma block reads the same chroma sample. Y reads remain direct and
// coalesced because the luma plane is already one byte per output pixel.
__global__ void yuv420p_to_rgb_float_kernel(
    const std::uint8_t* __restrict__ yuv420p,
    float* __restrict__ rgb,
    int width,
    int height,
    int chroma_width,
    int chroma_height
) {
    extern __shared__ std::uint8_t shared_chroma[];

    const int block_start_x = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x);
    const int block_start_y = static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.y);

    const int chroma_x0 = block_start_x >> 1;
    const int chroma_y0 = block_start_y >> 1;
    const int chroma_x1 = min_i((block_start_x + static_cast<int>(blockDim.x) - 1) >> 1, chroma_width - 1);
    const int chroma_y1 = min_i((block_start_y + static_cast<int>(blockDim.y) - 1) >> 1, chroma_height - 1);

    const int chroma_tile_w = chroma_x1 - chroma_x0 + 1;
    const int chroma_tile_h = chroma_y1 - chroma_y0 + 1;
    const int chroma_tile_samples = chroma_tile_w * chroma_tile_h;

    std::uint8_t* shared_u = shared_chroma;
    std::uint8_t* shared_v = shared_chroma + chroma_tile_samples;

    const std::size_t luma_samples = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    const std::size_t chroma_samples =
        static_cast<std::size_t>(chroma_width) * static_cast<std::size_t>(chroma_height);

    const std::uint8_t* y_plane = yuv420p;
    const std::uint8_t* u_plane = yuv420p + luma_samples;
    const std::uint8_t* v_plane = u_plane + chroma_samples;

    const int thread_linear =
        static_cast<int>(threadIdx.y) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    const int thread_count = static_cast<int>(blockDim.x) * static_cast<int>(blockDim.y);

    for (int i = thread_linear; i < chroma_tile_samples; i += thread_count) {
        const int local_chroma_y = i / chroma_tile_w;
        const int local_chroma_x = i - local_chroma_y * chroma_tile_w;
        const int global_chroma_x = chroma_x0 + local_chroma_x;
        const int global_chroma_y = chroma_y0 + local_chroma_y;
        const std::size_t chroma_index =
            static_cast<std::size_t>(global_chroma_y) * static_cast<std::size_t>(chroma_width) +
            static_cast<std::size_t>(global_chroma_x);

        shared_u[i] = u_plane[chroma_index];
        shared_v[i] = v_plane[chroma_index];
    }

    __syncthreads();

    const int x = block_start_x + static_cast<int>(threadIdx.x);
    const int y = block_start_y + static_cast<int>(threadIdx.y);

    if (x >= width || y >= height) {
        return;
    }

    const std::size_t luma_index =
        static_cast<std::size_t>(y) * static_cast<std::size_t>(width) + static_cast<std::size_t>(x);
    const int local_chroma_x = (x >> 1) - chroma_x0;
    const int local_chroma_y = (y >> 1) - chroma_y0;
    const int local_chroma_index = local_chroma_y * chroma_tile_w + local_chroma_x;

    const int y_sample = static_cast<int>(y_plane[luma_index]);
    const int u_sample = static_cast<int>(shared_u[local_chroma_index]);
    const int v_sample = static_cast<int>(shared_v[local_chroma_index]);

    // BT.601 limited-range YUV to RGB. This is the common default for decoded
    // 8-bit YUV420p streams. The integer form avoids precision drift and maps
    // cleanly to normalized float32 model input.
    int c = y_sample - 16;
    const int d = u_sample - 128;
    const int e = v_sample - 128;
    c = c < 0 ? 0 : c;

    const int r = (298 * c + 409 * e + 128) >> 8;
    const int g = (298 * c - 100 * d - 208 * e + 128) >> 8;
    const int b = (298 * c + 516 * d + 128) >> 8;

    const std::size_t rgb_index = luma_index * 3;
    rgb[rgb_index + 0] = normalize_u8_to_float(r);
    rgb[rgb_index + 1] = normalize_u8_to_float(g);
    rgb[rgb_index + 2] = normalize_u8_to_float(b);
}

// Convert two interleaved HWC RGB float32 frames into Qwen2.5-VL video patch
// rows. The output is dense row-major [grid_t * grid_h * grid_w, 1176], where
// 1176 = 3 channels * 2 temporal frames * 14 patch rows * 14 patch columns.
//
// Qwen's processor groups spatial patches by 2x2 merge cells before flattening:
// [grid_t, grid_h/2, grid_w/2, merge_h, merge_w, channel, time, y, x].
// The kernel mirrors that layout exactly so vLLM can feed the tensor directly
// to the model's visual patch embedder without CPU-side reshape or numpy paths.
//
// Source frames are already RGB float32 in [0, 1]. Real pixels are normalized
// with Qwen/OpenAI CLIP statistics. Spatial padding introduced to make
// grid_h/grid_w divisible by the 2x2 merge factor is written as 0.0f so padded
// features stay neutral instead of representing synthetic black content.
__global__ void qwen_patchify_kernel(
    const float* __restrict__ previous_rgb,
    const float* __restrict__ current_rgb,
    float* __restrict__ patches,
    int width,
    int height,
    int grid_h,
    int grid_w,
    std::size_t total_values
) {
    const std::size_t linear =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
        static_cast<std::size_t>(threadIdx.x);
    const std::size_t stride =
        static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);

    constexpr float mean[kQwenChannels] = {0.48145466f, 0.45782750f, 0.40821073f};
    constexpr float inv_std[kQwenChannels] = {
        1.0f / 0.26862954f,
        1.0f / 0.26130258f,
        1.0f / 0.27577711f,
    };

    const int grid_w_merged = grid_w / kQwenSpatialMergeSize;
    const int grid_h_merged = grid_h / kQwenSpatialMergeSize;

    for (std::size_t value_index = linear; value_index < total_values; value_index += stride) {
        const int feature = static_cast<int>(value_index % kQwenPatchVectorSize);
        const std::size_t row = value_index / kQwenPatchVectorSize;

        int feature_tmp = feature;
        const int patch_x = feature_tmp % kQwenPatchSize;
        feature_tmp /= kQwenPatchSize;
        const int patch_y = feature_tmp % kQwenPatchSize;
        feature_tmp /= kQwenPatchSize;
        const int temporal_index = feature_tmp % kQwenTemporalPatchSize;
        feature_tmp /= kQwenTemporalPatchSize;
        const int channel = feature_tmp;

        std::size_t row_tmp = row;
        const int merge_x = static_cast<int>(row_tmp % kQwenSpatialMergeSize);
        row_tmp /= kQwenSpatialMergeSize;
        const int merge_y = static_cast<int>(row_tmp % kQwenSpatialMergeSize);
        row_tmp /= kQwenSpatialMergeSize;
        const int patch_grid_x = static_cast<int>(row_tmp % static_cast<std::size_t>(grid_w_merged));
        row_tmp /= static_cast<std::size_t>(grid_w_merged);
        const int patch_grid_y = static_cast<int>(row_tmp % static_cast<std::size_t>(grid_h_merged));

        const int spatial_patch_x = patch_grid_x * kQwenSpatialMergeSize + merge_x;
        const int spatial_patch_y = patch_grid_y * kQwenSpatialMergeSize + merge_y;
        const int pixel_x = spatial_patch_x * kQwenPatchSize + patch_x;
        const int pixel_y = spatial_patch_y * kQwenPatchSize + patch_y;

        float value = 0.0f;
        if (pixel_x < width && pixel_y < height) {
            const float* frame = temporal_index == 0 ? previous_rgb : current_rgb;
            const std::size_t rgb_index =
                (static_cast<std::size_t>(pixel_y) * static_cast<std::size_t>(width) +
                 static_cast<std::size_t>(pixel_x)) *
                    static_cast<std::size_t>(kQwenChannels) +
                static_cast<std::size_t>(channel);
            value = (frame[rgb_index] - mean[channel]) * inv_std[channel];
        }

        patches[value_index] = value;
    }
}

}  // namespace

extern "C" std::size_t csf_yuv420p_frame_bytes(int width, int height) {
    if (width <= 0 || height <= 0) {
        return 0;
    }

    const std::size_t luma_samples = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    const std::size_t chroma_width = static_cast<std::size_t>((width + 1) / 2);
    const std::size_t chroma_height = static_cast<std::size_t>((height + 1) / 2);
    return luma_samples + 2 * chroma_width * chroma_height;
}

extern "C" cudaError_t csf_launch_yuv420p_to_rgb_float(
    const std::uint8_t* yuv420p,
    float* rgb,
    int width,
    int height,
    cudaStream_t stream
) {
    if (yuv420p == nullptr || rgb == nullptr || width <= 0 || height <= 0) {
        return cudaErrorInvalidValue;
    }

    constexpr int kBlockX = 16;
    constexpr int kBlockY = 16;
    const dim3 block(kBlockX, kBlockY);
    const dim3 grid(
        (static_cast<unsigned int>(width) + block.x - 1) / block.x,
        (static_cast<unsigned int>(height) + block.y - 1) / block.y
    );

    const int max_chroma_tile_w = (kBlockX + 1) / 2 + 1;
    const int max_chroma_tile_h = (kBlockY + 1) / 2 + 1;
    const std::size_t shared_bytes =
        2 * static_cast<std::size_t>(max_chroma_tile_w) *
        static_cast<std::size_t>(max_chroma_tile_h) * sizeof(std::uint8_t);

    yuv420p_to_rgb_float_kernel<<<grid, block, shared_bytes, stream>>>(
        yuv420p,
        rgb,
        width,
        height,
        (width + 1) / 2,
        (height + 1) / 2
    );

    return cudaGetLastError();
}

extern "C" cudaError_t csf_launch_qwen_patchify(
    const float* previous_rgb,
    const float* current_rgb,
    float* patches,
    int width,
    int height,
    int grid_h,
    int grid_w,
    cudaStream_t stream
) {
    if (
        previous_rgb == nullptr || current_rgb == nullptr || patches == nullptr ||
        width <= 0 || height <= 0 || grid_h <= 0 || grid_w <= 0 ||
        (grid_h % kQwenSpatialMergeSize) != 0 || (grid_w % kQwenSpatialMergeSize) != 0
    ) {
        return cudaErrorInvalidValue;
    }

    const std::size_t rows = static_cast<std::size_t>(grid_h) * static_cast<std::size_t>(grid_w);
    const std::size_t total_values = rows * static_cast<std::size_t>(kQwenPatchVectorSize);
    constexpr int kThreads = 256;
    const std::size_t raw_blocks =
        (total_values + static_cast<std::size_t>(kThreads) - 1) / static_cast<std::size_t>(kThreads);
    const unsigned int blocks = static_cast<unsigned int>(raw_blocks > 65'535ULL ? 65'535ULL : raw_blocks);

    qwen_patchify_kernel<<<blocks, kThreads, 0, stream>>>(
        previous_rgb,
        current_rgb,
        patches,
        width,
        height,
        grid_h,
        grid_w,
        total_values
    );

    return cudaGetLastError();
}
