#include "Sobel.h"

#if defined(__CUDACC__)

#include <cmath>
#include <stdexcept>
#include <string>

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
 * @brief Kernel Map punto a punto para calcular la magnitud `G(i,j) = sqrt(G_x(i,j)^2 + G_y(i,j)^2)`.
 */
__global__ void sobelMagnitudeKernel(const float* __restrict__ d_gx,
                                     const float* __restrict__ d_gy,
                                     float* __restrict__ d_mag,
                                     int H, int W) {
    const int i = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= H || j >= W) return;

    const int idx = i * W + j;
    const float gx = d_gx[idx];
    const float gy = d_gy[idx];
    d_mag[idx] = sqrtf(gx * gx + gy * gy);
}

void Sobel::computeMagnitudeGpu() {
    if (gx_gpu_.height() != gy_gpu_.height() || gx_gpu_.width() != gy_gpu_.width() || gx_gpu_.empty()) {
        throw std::runtime_error("Sobel::computeMagnitudeGpu: ejecute computeGxGyGpu() antes de computeMagnitudeGpu()");
    }

    const int H = static_cast<int>(gx_gpu_.height());
    const int W = static_cast<int>(gx_gpu_.width());
    mag_gpu_.resize(gx_gpu_.height(), gx_gpu_.width(), 0.0f);

    const std::size_t bytes = mag_gpu_.size() * sizeof(float);
    float* d_gx  = nullptr;
    float* d_gy  = nullptr;
    float* d_mag = nullptr;

    CUDA_CHECK(cudaMalloc(&d_gx, bytes));
    CUDA_CHECK(cudaMalloc(&d_gy, bytes));
    CUDA_CHECK(cudaMalloc(&d_mag, bytes));

    try {
        CUDA_CHECK(cudaMemcpy(d_gx, gx_gpu_.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_gy, gy_gpu_.data(), bytes, cudaMemcpyHostToDevice));

        const dim3 block(16, 16);
        const dim3 grid((static_cast<unsigned int>(W) + block.x - 1) / block.x,
                        (static_cast<unsigned int>(H) + block.y - 1) / block.y);

        sobelMagnitudeKernel<<<grid, block>>>(d_gx, d_gy, d_mag, H, W);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(mag_gpu_.data(), d_mag, bytes, cudaMemcpyDeviceToHost));
    } catch (...) {
        cudaFree(d_gx);
        cudaFree(d_gy);
        cudaFree(d_mag);
        throw;
    }

    CUDA_CHECK(cudaFree(d_gx));
    CUDA_CHECK(cudaFree(d_gy));
    CUDA_CHECK(cudaFree(d_mag));
}

#endif // defined(__CUDACC__)

