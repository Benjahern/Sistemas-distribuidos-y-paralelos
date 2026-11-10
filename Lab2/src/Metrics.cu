#include "Metrics.h"
#include "Convolution2D.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#if defined(__CUDACC__)
#include <cuda_runtime.h>

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
 * @brief Método 0: Reducción por árbol en memoria compartida (`__shared__`) en dos pasadas.
 */
__global__ void reduceSharedKernel(const float* __restrict__ d_in,
                                   float* __restrict__ d_block_sums,
                                   std::size_t N) {
    extern __shared__ float s_data[];
    const unsigned int tid = threadIdx.x;
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    s_data[tid] = (idx < N) ? d_in[idx] : 0.0f;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_data[tid] += s_data[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_block_sums[blockIdx.x] = s_data[0];
    }
}

/**
 * @brief Método 1: Acumulación global utilizando operaciones atómicas (`atomicAdd`).
 */
__global__ void reduceAtomicKernel(const float* __restrict__ d_in,
                                   float* __restrict__ d_sum,
                                   std::size_t N) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(d_sum, d_in[idx]);
    }
}
#endif // defined(__CUDACC__)

Metrics::Metrics(const ImageBuffer& magnitude) : magnitude_(magnitude) {}

void Metrics::setMagnitudeBuffer(const ImageBuffer& magnitude) {
    magnitude_ = magnitude;
}

void Metrics::setComparisonBuffers(const ImageBuffer& cpu_buf, const ImageBuffer& gpu_buf) {
    cpu_ref_ = cpu_buf;
    gpu_res_ = gpu_buf;
}

float Metrics::maxAbsDiffCpuGpu(const ImageBuffer& cpu_buf, const ImageBuffer& gpu_buf) {
    if (cpu_buf.height() != gpu_buf.height() || cpu_buf.width() != gpu_buf.width()) {
        throw std::invalid_argument("Metrics::maxAbsDiffCpuGpu: las dimensiones de los buffers no coinciden");
    }
    float max_diff = 0.0f;
    const float* c = cpu_buf.data();
    const float* g = gpu_buf.data();
    const std::size_t total = cpu_buf.size();

    for (std::size_t idx = 0; idx < total; ++idx) {
        max_diff = std::max(max_diff, std::fabs(c[idx] - g[idx]));
    }
    return max_diff;
}

float Metrics::maxAbsDiffCpuGpu() const {
    return maxAbsDiffCpuGpu(cpu_ref_, gpu_res_);
}

float Metrics::sumMagnitude(int method) const {
    return sumMagnitude(magnitude_, method);
}

float Metrics::sumMagnitude(const ImageBuffer& magnitude, int method) {
    if (method != METHOD_SHARED_REDUCTION && method != METHOD_ATOMIC_ADD) {
        throw std::invalid_argument("Metrics::sumMagnitude: método inválido (use 0=shared, 1=atomicAdd)");
    }
    if (magnitude.empty()) {
        return 0.0f;
    }

#if defined(__CUDACC__)
    if (Convolution2D::isGpuAvailable()) {
        const std::size_t N = magnitude.size();
        const std::size_t bytes = N * sizeof(float);
        const unsigned int threads = 256;
        const unsigned int blocks = static_cast<unsigned int>((N + threads - 1) / threads);

        float* d_in = nullptr;
        CUDA_CHECK(cudaMalloc(&d_in, bytes));
        CUDA_CHECK(cudaMemcpy(d_in, magnitude.data(), bytes, cudaMemcpyHostToDevice));

        float result = 0.0f;
        if (method == METHOD_SHARED_REDUCTION) {
            float* d_partials = nullptr;
            CUDA_CHECK(cudaMalloc(&d_partials, blocks * sizeof(float)));
            reduceSharedKernel<<<blocks, threads, threads * sizeof(float)>>>(d_in, d_partials, N);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            std::vector<float> h_partials(blocks, 0.0f);
            CUDA_CHECK(cudaMemcpy(h_partials.data(), d_partials, blocks * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaFree(d_partials));

            double acc = 0.0;
            for (float v : h_partials) acc += static_cast<double>(v);
            result = static_cast<float>(acc);
        } else {
            float* d_sum = nullptr;
            CUDA_CHECK(cudaMalloc(&d_sum, sizeof(float)));
            CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(float)));
            reduceAtomicKernel<<<blocks, threads>>>(d_in, d_sum, N);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(&result, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaFree(d_sum));
        }

        CUDA_CHECK(cudaFree(d_in));
        return result;
    }
#endif

    // Fallback determinista en host cuando se ejecuta en un runner de CI sin GPU física
    double sum = 0.0;
    const float* ptr = magnitude.data();
    for (std::size_t idx = 0; idx < magnitude.size(); ++idx) {
        sum += static_cast<double>(ptr[idx]);
    }
    return static_cast<float>(sum);
}

