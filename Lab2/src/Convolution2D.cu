#include "Convolution2D.h"

#if defined(__CUDACC__)

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

/**
 * @brief Macro obligatoria `CUDA_CHECK` tras cada llamada a la API CUDA (AGENTS.md Sección 3).
 */
#define CUDA_CHECK(call)                                                                     \
    do {                                                                                     \
        cudaError_t err__ = (call);                                                          \
        if (err__ != cudaSuccess) {                                                          \
            throw std::runtime_error(std::string("CUDA error en ") + __FILE__ + ":" +        \
                                     std::to_string(__LINE__) + " -> " +                     \
                                     cudaGetErrorString(err__));                             \
        }                                                                                    \
    } while (0)

/**
 * @brief Función auxiliar `__device__` obligatoria (Sección 6.1.1) para aplicar la política
 *        de borde única (**Zero-padding**) e indexar en orden fila-mayor (`i * W + j`).
 */
__device__ inline float getPixelZeroPad(const float* __restrict__ img, int H, int W, int i, int j) {
    if (i < 0 || i >= H || j < 0 || j >= W) {
        return 0.0f;
    }
    return img[i * W + j];
}

/**
 * @brief Variante 0 (`naive`): Un hilo por píxel de salida `(i, j)`.
 *        Lee el vecindario directamente desde memoria global con guarda de límites.
 */
__global__ void convolveNaiveKernel(const float* __restrict__ d_in,
                                    float* __restrict__ d_out,
                                    const float* __restrict__ d_k,
                                    int H, int W, int s, int r) {
    const int i = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= H || j >= W) return;

    float acc = 0.0f;
    for (int u = -r; u <= r; ++u) {
        for (int v = -r; v <= r; ++v) {
            acc += d_k[(u + r) * s + (v + r)] * getPixelZeroPad(d_in, H, W, i + u, j + v);
        }
    }
    d_out[i * W + j] = acc;
}

/**
 * @brief Variante 1 (`shared`): Bloques 2D que cargan cooperativamente un tile de dimensiones
 *        `(blockDim.y + 2r) x (blockDim.x + 2r)` en memoria compartida (`__shared__`),
 *        sincronizan con `__syncthreads()` y calculan la convolución leyendo desde `__shared__`.
 */
__global__ void convolveSharedKernel(const float* __restrict__ d_in,
                                     float* __restrict__ d_out,
                                     const float* __restrict__ d_k,
                                     int H, int W, int s, int r) {
    extern __shared__ float s_tile[];

    const int tile_w = static_cast<int>(blockDim.x) + 2 * r;
    const int tile_h = static_cast<int>(blockDim.y) + 2 * r;

    const int base_i = static_cast<int>(blockIdx.y * blockDim.y) - r;
    const int base_j = static_cast<int>(blockIdx.x * blockDim.x) - r;

    // Paso 1: Carga cooperativa del tile + halo de radio r hacia memoria compartida
    for (int ty = static_cast<int>(threadIdx.y); ty < tile_h; ty += static_cast<int>(blockDim.y)) {
        for (int tx = static_cast<int>(threadIdx.x); tx < tile_w; tx += static_cast<int>(blockDim.x)) {
            s_tile[ty * tile_w + tx] = getPixelZeroPad(d_in, H, W, base_i + ty, base_j + tx);
        }
    }

    // Paso 2: Barrera de sincronización obligatoria antes de leer el vecindario
    __syncthreads();

    // Paso 3: Cómputo de la convolución exclusivamente desde __shared__
    const int i = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= H || j >= W) return;

    const int si = static_cast<int>(threadIdx.y) + r;
    const int sj = static_cast<int>(threadIdx.x) + r;

    float acc = 0.0f;
    for (int u = -r; u <= r; ++u) {
        for (int v = -r; v <= r; ++v) {
            acc += d_k[(u + r) * s + (v + r)] * s_tile[(si + u) * tile_w + (sj + v)];
        }
    }
    d_out[i * W + j] = acc;
}

void Convolution2D::convolveGpu(const float* k, int s, int variant, dim3 block) {
    if (k == nullptr) {
        throw std::invalid_argument("Convolution2D::convolveGpu: puntero a kernel k nulo");
    }
    if (s < 1 || (s % 2 == 0)) {
        throw std::invalid_argument("Convolution2D::convolveGpu: el lado s debe ser impar y positivo");
    }
    if (variant != VARIANT_NAIVE && variant != VARIANT_SHARED) {
        throw std::invalid_argument("Convolution2D::convolveGpu: variante inválida (use 0=naive o 1=shared)");
    }

    const int H = static_cast<int>(input_.height());
    const int W = static_cast<int>(input_.width());
    output_gpu_.resize(input_.height(), input_.width(), 0.0f);
    if (input_.empty()) return;

    const int r = (s - 1) / 2;
    const std::size_t img_bytes = input_.size() * sizeof(float);
    const std::size_t k_bytes   = static_cast<std::size_t>(s * s) * sizeof(float);

    float* d_in  = nullptr;
    float* d_out = nullptr;
    float* d_k   = nullptr;

    CUDA_CHECK(cudaMalloc(&d_in, img_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, img_bytes));
    CUDA_CHECK(cudaMalloc(&d_k, k_bytes));

    try {
        CUDA_CHECK(cudaMemcpy(d_in, input_.data(), img_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_k, k, k_bytes, cudaMemcpyHostToDevice));

        const dim3 grid((static_cast<unsigned int>(W) + block.x - 1) / block.x,
                        (static_cast<unsigned int>(H) + block.y - 1) / block.y);

        if (variant == VARIANT_NAIVE) {
            convolveNaiveKernel<<<grid, block>>>(d_in, d_out, d_k, H, W, s, r);
        } else {
            const std::size_t shared_bytes =
                static_cast<std::size_t>(block.x + 2 * r) *
                static_cast<std::size_t>(block.y + 2 * r) * sizeof(float);
            convolveSharedKernel<<<grid, block, shared_bytes>>>(d_in, d_out, d_k, H, W, s, r);
        }

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(output_gpu_.data(), d_out, img_bytes, cudaMemcpyDeviceToHost));
    } catch (...) {
        cudaFree(d_in);
        cudaFree(d_out);
        cudaFree(d_k);
        throw;
    }

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_k));
}

#endif // defined(__CUDACC__)

