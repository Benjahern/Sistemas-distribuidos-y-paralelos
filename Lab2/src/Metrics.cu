#include "Metrics.h"

#include <cmath>
#include <limits>
#include <stdexcept>

#if defined(__CUDACC__)

#include <cuda_runtime.h>
#include <cstdio>
#include <string>

namespace {

void checkCuda(
    cudaError_t state,
    const char* call,
    const char* file,
    int line
) {
    if (state != cudaSuccess) {
        throw std::runtime_error(
            std::string("Error CUDA en ") +
            file + ":" + std::to_string(line) +
            " [" + call + "] " +
            cudaGetErrorString(state)
        );
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

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            const cudaError_t state = cudaFree(data_);

            if (state != cudaSuccess) {
                std::fprintf(
                    stderr,
                    "Error al liberar memoria CUDA: %s\n",
                    cudaGetErrorString(state)
                );
            }
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    float* data() const noexcept {
        return data_;
    }

    void realease() {
        if (data_ == nullptr) {
            return;
        }

        float* previous = data_;
        data_ = nullptr;

        CUDA_CHECK(cudaFree(previous));
    }

private:
    float* data_ = nullptr;
};

// Todos los hilos del bloque deben llamar a esta función.
// Los lanzadores usan 256 hilos, una potencia de dos.
__device__ float reduceBlock(
    float value,
    float* partial
) {
    const unsigned int local = threadIdx.x;

    partial[local] = value;

    __syncthreads();

    unsigned int offset = blockDim.x / 2;

    while (offset > 0) {
        if (local < offset) {
            partial[local] += partial[local + offset];
        }

        __syncthreads();

        offset /= 2;
    }

    return partial[0];
}

__global__ void sumSharedKernel(
    const float* input,
    float* output,
    std::size_t count
) {
    extern __shared__ float partial[];

    std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;

    const std::size_t stride =
        static_cast<std::size_t>(blockDim.x) * gridDim.x;

    float sum = 0.0f;

    while (index < count) {
        sum += input[index];
        index += stride;
    }

    const float blockSum = reduceBlock(sum, partial);

    if (threadIdx.x == 0) {
        output[blockIdx.x] = blockSum;
    }
}

__global__ void sumAtomicKernel(
    const float* input,
    float* output,
    std::size_t count
) {
    extern __shared__ float partial[];

    std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;

    const std::size_t stride =
        static_cast<std::size_t>(blockDim.x) * gridDim.x;

    float sum = 0.0f;

    while (index < count) {
        sum += input[index];
        index += stride;
    }

    const float blockSum = reduceBlock(sum, partial);

    if (threadIdx.x == 0) {
        // Todos los bloques acumulan sobre el mismo resultado.
        atomicAdd(output, blockSum);
    }
}

}

#endif

Metrics::Metrics(const ImageBuffer& magnitude)
    : magnitude_(magnitude) {}

void Metrics::setMagnitudeBuffer(
    const ImageBuffer& magnitude
) {
    magnitude_ = magnitude;
}

void Metrics::setComparisonBuffers(
    const ImageBuffer& cpu_buf,
    const ImageBuffer& gpu_buf
) {
    cpu_ref_ = cpu_buf;
    gpu_res_ = gpu_buf;
}

float Metrics::maxAbsDiffCpuGpu(
    const ImageBuffer& cpu_buf,
    const ImageBuffer& gpu_buf
) {
    if (
        cpu_buf.height() != gpu_buf.height() ||
        cpu_buf.width() != gpu_buf.width()
    ) {
        throw std::invalid_argument(
            "maxAbsDiffCpuGpu: Dimensiones incompatibles"
        );
    }

    float maximum = 0.0f;

    std::size_t index = 0;

    while (index < cpu_buf.size()) {
        const float cpu = cpu_buf.data()[index];
        const float gpu = gpu_buf.data()[index];

        // Evitamos que un NaN o infinito aparente un error cero.
        if (!std::isfinite(cpu) || !std::isfinite(gpu)) {
            return std::numeric_limits<float>::infinity();
        }

        const float difference = std::fabs(cpu - gpu);

        if (difference > maximum) {
            maximum = difference;
        }

        ++index;
    }

    return maximum;
}

float Metrics::maxAbsDiffCpuGpu() const {
    return maxAbsDiffCpuGpu(cpu_ref_, gpu_res_);
}

float Metrics::sumMagnitude(int method) const {
    return sumMagnitude(magnitude_, method);
}

float Metrics::sumMagnitude(
    const ImageBuffer& magnitude,
    int method
) {
    if (
        method != METHOD_SHARED_REDUCTION &&
        method != METHOD_ATOMIC_ADD
    ) {
        throw std::invalid_argument(
            "sumMagnitude: Method debe ser 0 o 1"
        );
    }

    if (magnitude.empty()) {
        return 0.0f;
    }

#if defined(__CUDACC__)

    const std::size_t count = magnitude.size();

    const unsigned int threads = 256;

    const std::size_t neededBlocks =
        1 + (count - 1) / threads;

    const unsigned int blocks = static_cast<unsigned int>(
        neededBlocks > 1024 ? 1024 : neededBlocks
    );

    const std::size_t sharedBytes =
        threads * sizeof(float);

    DeviceBuffer deviceInput(count);
    DeviceBuffer devicePartial(blocks);
    DeviceBuffer deviceResult(1);

    CUDA_CHECK(cudaMemcpy(
        deviceInput.data(),
        magnitude.data(),
        count * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    if (method == METHOD_SHARED_REDUCTION) {
        // Primera pasada: una suma parcial por bloque.
        sumSharedKernel<<<blocks, threads, sharedBytes>>>(
            deviceInput.data(),
            devicePartial.data(),
            count
        );

        CUDA_CHECK(cudaGetLastError());

        // Segunda pasada: un bloque combina las sumas parciales.
        sumSharedKernel<<<1, threads, sharedBytes>>>(
            devicePartial.data(),
            deviceResult.data(),
            blocks
        );

        CUDA_CHECK(cudaGetLastError());
    } else {
        // atomicAdd necesita que el acumulador comience en cero.
        CUDA_CHECK(cudaMemset(
            deviceResult.data(),
            0,
            sizeof(float)
        ));

        sumAtomicKernel<<<blocks, threads, sharedBytes>>>(
            deviceInput.data(),
            deviceResult.data(),
            count
        );

        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    float result = 0.0f;

    CUDA_CHECK(cudaMemcpy(
        &result,
        deviceResult.data(),
        sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    deviceResult.realease();
    devicePartial.realease();
    deviceInput.realease();

    return result;

#else

    throw std::runtime_error(
        "sumMagnitude: Compilacion sin soporte CUDA"
    );

#endif
}

#if defined(__CUDACC__)
#undef CUDA_CHECK
#endif