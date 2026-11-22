#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

constexpr int THREADS = 256;
constexpr int WARP = 32;
constexpr int HEAD = 64;
constexpr int PAIRS = 24;
constexpr int PATCH = 16;
constexpr int GLOBALS = 5;
constexpr int REGISTERS = GLOBALS - 1;

__device__ float warp_sum(float value) {
    for (int offset = WARP / 2; offset; offset /= 2) {
        value += __shfl_xor_sync(0xffffffff, value, offset);
    }
    return value;
}

__device__ float block_sum(float value, float *scratch) {
    value = warp_sum(value);
    if (threadIdx.x % WARP == 0) {
        scratch[threadIdx.x / WARP] = value;
    }
    __syncthreads();
    float sum = 0.0f;
    for (int warp = 0; warp < THREADS / WARP; warp++) {
        sum += scratch[warp];
    }
    __syncthreads();
    return sum;
}

__global__ void adaln(const float *hidden, const float *modulation, __half *features, int width) {
    __shared__ float scratch[THREADS / WARP];
    const size_t row = static_cast<size_t>(blockIdx.x) * width;
    float sum = 0.0f;
    for (int col = threadIdx.x; col < width; col += THREADS) {
        sum += hidden[row + col];
    }
    const float mean = block_sum(sum, scratch) / width;
    float squares = 0.0f;
    for (int col = threadIdx.x; col < width; col += THREADS) {
        const float centered = hidden[row + col] - mean;
        squares += centered * centered;
    }
    const float inverse = rsqrtf(block_sum(squares, scratch) / width + 1e-5f);
    for (int col = threadIdx.x; col < width; col += THREADS) {
        const float normalized = (hidden[row + col] - mean) * inverse;
        features[row + col] =
            __float2half_rn(normalized * (1.0f + modulation[col]) + modulation[width + col]);
    }
}

// A DiT row expands to [2, 2, dim]. Interleave that grid into (t, y, x) order and append the
// global rows, each native block's four registers followed by a zero class token.
__global__ void expand_rows(const __half *expanded, const __half *registers, float *residual,
                            int height, int width, int video_rows, int dim) {
    const int row = blockIdx.x;
    const __half *source = nullptr;
    if (row < video_rows) {
        const int t = row / (height * width);
        const int y = row / width % height;
        const int x = row % width;
        const int coarse = (t * (height / 2) + y / 2) * (width / 2) + x / 2;
        const int fine = (y % 2) * 2 + x % 2;
        source = expanded + (static_cast<size_t>(coarse) * 4 + fine) * dim;
    } else {
        const int reg = (row - video_rows) % GLOBALS;
        if (reg < REGISTERS) {
            source = registers + reg * dim;
        }
    }
    for (int col = threadIdx.x; col < dim; col += THREADS) {
        residual[static_cast<size_t>(row) * dim + col] =
            source == nullptr ? 0.0f : __half2float(source[col]);
    }
}

// Every warp normalizes a query or key head. Rotary angles belong to pixel rows only.
__global__ void norm_rope(__half *qkv, const float *angles, int rows, int video_rows, int heads) {
    __shared__ float shared[THREADS / WARP][HEAD];
    const int warp = threadIdx.x / WARP;
    const int lane = threadIdx.x % WARP;
    const int item = blockIdx.x * (THREADS / WARP) + warp;
    const int row = item / (2 * heads);
    if (row >= rows) {
        return;
    }
    __half *vector = qkv + (static_cast<size_t>(item / 2) * 3 + item % 2) * HEAD;
    float *values = shared[warp];
    const float first = __half2float(vector[lane]);
    const float second = __half2float(vector[WARP + lane]);
    const float inverse = rsqrtf(warp_sum(first * first + second * second) / HEAD + 1e-5f);
    values[lane] = first * inverse;
    values[WARP + lane] = second * inverse;
    __syncwarp();
    if (row < video_rows && lane < PAIRS) {
        float sine, cosine;
        sincosf(angles[static_cast<size_t>(row) * PAIRS + lane], &sine, &cosine);
        const float a = values[lane];
        const float b = values[PAIRS + lane];
        values[lane] = a * cosine - b * sine;
        values[PAIRS + lane] = b * cosine + a * sine;
    }
    __syncwarp();
    vector[lane] = __float2half_rn(values[lane]);
    vector[WARP + lane] = __float2half_rn(values[WARP + lane]);
}

// Grab one operand at a time in contiguous rows from the head-interleaved qkv projection.
__global__ void gather(const uint4 *qkv, const int *indices, uint4 *output, int rows, int dim,
                       int first_operand) {
    constexpr int vectors_per_head = HEAD / 8;
    const int vectors = dim / 8;
    const int row = blockIdx.x;
    const int operand = blockIdx.y;
    for (int col = threadIdx.x; col < vectors; col += THREADS) {
        const int head = col / vectors_per_head;
        const int lane = col % vectors_per_head;
        const size_t source = static_cast<size_t>(indices[row]) * 3 * vectors +
                              (head * 3 + first_operand + operand) * vectors_per_head + lane;
        output[(static_cast<size_t>(operand) * rows + row) * vectors + col] = qkv[source];
    }
}

__global__ void scatter(const uint4 *packed, const int *indices, uint4 *output, int dim) {
    const int row = blockIdx.x;
    const int vectors = dim / 8;
    for (int col = threadIdx.x; col < vectors; col += THREADS) {
        output[static_cast<size_t>(indices[row]) * vectors + col] =
            packed[static_cast<size_t>(row) * vectors + col];
    }
}

__global__ void pixels(const __half *projected, float *rgb, size_t values, int output_frames,
                       int position, int patch_frames, int height, int width) {
    constexpr float mean[] = {-0.03f, -0.088f, -0.188f};
    constexpr float std[] = {0.458f, 0.448f, 0.45f};
    const size_t index = static_cast<size_t>(blockIdx.x) * THREADS + threadIdx.x;
    if (index >= values) {
        return;
    }
    const size_t plane = static_cast<size_t>(height) * width;
    const int col = index % width;
    const int row = index / width % height;
    const int channel = index / plane % 3;
    const int frame = index / (3 * plane);
    const size_t token =
        (static_cast<size_t>(frame / patch_frames) * (height / PATCH) + row / PATCH) *
            (width / PATCH) +
        col / PATCH;
    const size_t source =
        (((token * 3 + channel) * patch_frames + frame % patch_frames) * PATCH + row % PATCH) *
            PATCH +
        col % PATCH;
    const size_t destination =
        (static_cast<size_t>(channel) * output_frames + position + frame) * plane + index % plane;
    const float normalized = __half2float(projected[source]);
    rgb[destination] =
        fminf(1.0f, fmaxf(0.0f, (normalized * std[channel] + mean[channel] + 1.0f) * 0.5f));
}

} // namespace

extern "C" int mmh3_dwd_adaln(const float *hidden, const float *modulation, __half *features,
                              int rows, int width) {
    adaln<<<rows, THREADS>>>(hidden, modulation, features, width);
    return cudaGetLastError();
}

extern "C" int mmh3_dwd_expand_rows(const __half *expanded, const __half *registers,
                                    float *residual, int height, int width, int video_rows,
                                    int rows, int dim) {
    expand_rows<<<rows, THREADS>>>(expanded, registers, residual, height, width, video_rows, dim);
    return cudaGetLastError();
}

extern "C" int mmh3_dwd_norm_rope(__half *qkv, const float *angles, int rows, int video_rows,
                                  int heads) {
    const int items = rows * 2 * heads;
    norm_rope<<<(items + THREADS / WARP - 1) / (THREADS / WARP), THREADS>>>(qkv, angles, rows,
                                                                            video_rows, heads);
    return cudaGetLastError();
}

extern "C" int mmh3_dwd_gather(const uint4 *qkv, const int *indices, uint4 *output, int rows,
                               int dim, int first_operand, int operands) {
    gather<<<dim3(rows, operands), THREADS>>>(qkv, indices, output, rows, dim, first_operand);
    return cudaGetLastError();
}

extern "C" int mmh3_dwd_scatter(const uint4 *packed, const int *indices, uint4 *output, int rows,
                                int dim) {
    scatter<<<rows, THREADS>>>(packed, indices, output, dim);
    return cudaGetLastError();
}

extern "C" int mmh3_dwd_pixels(const __half *projected, float *rgb, int frames, int output_frames,
                               int position, int patch_frames, int height, int width) {
    const size_t values = static_cast<size_t>(frames) * height * width * 3;
    pixels<<<(values + THREADS - 1) / THREADS, THREADS>>>(projected, rgb, values, output_frames,
                                                          position, patch_frames, height, width);
    return cudaGetLastError();
}
