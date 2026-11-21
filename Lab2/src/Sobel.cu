#include "Sobel.h"

#if defined(__CUDACC__)

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace{

void checkCuda(
    cudaError_t state,
    const char* call,
    const char* file,
    int line
) {
    if (state != cudaSuccess){
        throw std::runtime_error(std::string("Error CUDA en ") + file + ":" + std::to_string(line) + " [" + call + "] " + cudaGetErrorString(state));
    }
}

#define CUDA_CHECK(call) \
    checkCuda((call), #call, __FILE__, __LINE__)

class DeviceBuffer {

public:
    explicit DeviceBuffer(std::size_t count) {
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&data_),
            count * sizeof(float)
        ));
    }

    ~DeviceBuffer(){
        if (data_ != nullptr){
            const cudaError_t state = cudaFree(data_);

            if(state != cudaSuccess){
                std::fprintf(stderr,
                "Error al liberar memoria CUDA: %s\n",
            cudaGetErrorString(state));
            }
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator= (const DeviceBuffer&) = delete;

    float* data() const noexcept{
        return data_;
    }

    void realease(){
        if (data_ == nullptr){
            return;
        }

        float* previous = data_;
        data_ = nullptr;

        CUDA_CHECK(cudaFree(previous));
    }

private:
    float* data_ = nullptr;
};

__global__ void magnitudeKernel(
    const float* gx,
    const float* gy,
    float* magnitude,
    std::size_t count
) {
    std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

    while(index < count){
        const float x = gx[index];
        const float y = gy[index];

        magnitude[index] = sqrtf(x * x + y * y);

        index +=stride;
    }
}

}

void Sobel::computeMagnitudeGpu() {
    if (
        gx_gpu_.height() != input_.height() ||
        gx_gpu_.width() != input_.width() ||
        gy_gpu_.height() != input_.height() ||
        gy_gpu_.width() != input_.width()
    ) {
        throw std::invalid_argument(
            "computeMagnitudeGpu: Dimensiones incompatibles"
        );
    }

    ImageBuffer result(
        input_.height(),
        input_.width(),
        0.0f
    );

    if (input_.empty()) {
        mag_gpu_ = result;
        return;
    }

    const std::size_t count = input_.size();
    const std::size_t bytes = count * sizeof(float);

    const unsigned int threads = 256;

    const std::size_t neededBlocks =
        1 + (count - 1) / threads;

    const unsigned int blocks = static_cast<unsigned int>(
        neededBlocks > 1024 ? 1024 : neededBlocks
    );

    DeviceBuffer deviceGx(count);
    DeviceBuffer deviceGy(count);
    DeviceBuffer deviceMagnitude(count);

    CUDA_CHECK(cudaMemcpy(
        deviceGx.data(),
        gx_gpu_.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        deviceGy.data(),
        gy_gpu_.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));

    magnitudeKernel<<<blocks, threads>>>(
        deviceGx.data(),
        deviceGy.data(),
        deviceMagnitude.data(),
        count
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        result.data(),
        deviceMagnitude.data(),
        bytes,
        cudaMemcpyDeviceToHost
    ));

    deviceMagnitude.realease();
    deviceGy.realease();
    deviceGx.realease();

    mag_gpu_ = result;
}

#undef CUDA_CHECK

#endif