#include "Benchmark.h"

#include "Convolution2D.h"
#include "Sobel.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
void benchmarkLaunchConvolution(const float*, float*, const float*, std::size_t,
                                std::size_t, int, int, unsigned int, unsigned int);
void benchmarkLaunchMagnitude(const float*, const float*, float*, std::size_t);
#endif

namespace {
using Clock = std::chrono::steady_clock;
using Samples = std::vector<double>;

struct Summary { double mean; double deviation; };

[[maybe_unused]] Summary summarize(const Samples& samples) {
    if (samples.empty()) throw std::invalid_argument("No hay muestras para resumir");
    const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    double squared = 0.0;
    for (const double value : samples) squared += (value - mean) * (value - mean);
    const double deviation = samples.size() > 1
        ? std::sqrt(squared / static_cast<double>(samples.size() - 1)) : 0.0;
    return {mean, deviation};
}

void validate(const ImageBuffer& image, const float* kernel, int side, int variant,
              dim3 block, int repetitions) {
    if (image.empty()) throw std::invalid_argument("Benchmark: imagen vacía");
    if (kernel == nullptr || side < 1 || side % 2 == 0)
        throw std::invalid_argument("Benchmark: kernel no nulo y lado impar positivo requeridos");
    if (variant != 0 && variant != 1) throw std::invalid_argument("Benchmark: variante debe ser 0 o 1");
    if (block.x == 0 || block.y == 0 || block.z != 1)
        throw std::invalid_argument("Benchmark: geometría de bloque inválida");
    if (repetitions < 10) throw std::invalid_argument("Benchmark: se requieren al menos 10 repeticiones");
}

void cpuConvolution(const ImageBuffer& image, const float* kernel, int side) {
    Convolution2D op(image);
    op.convolveCpu(kernel, side);
}

[[maybe_unused]] Samples timeCpu(const ImageBuffer& image, int pipeline, int side, int repetitions) {
    Samples samples;
    samples.reserve(static_cast<std::size_t>(repetitions));
    const auto kernel = pipeline == 2 ? std::vector<float>{} : ImageBuffer::makeGaussianKernel(side);
    for (int r = 0; r < repetitions; ++r) {
        const auto start = Clock::now();
        if (pipeline == 2) {
            Sobel sobel(image);
            sobel.computeGxGyCpu();
            sobel.computeMagnitudeCpu();
        } else {
            cpuConvolution(image, kernel.data(), side);
        }
        samples.push_back(std::chrono::duration<double, std::milli>(Clock::now() - start).count());
    }
    return samples;
}

#if defined(__CUDACC__)
void checkCuda(cudaError_t status, const char* action) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(action) + ": " + cudaGetErrorString(status));
}

void synchronize() { checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize"); }

struct GpuPipeline {
    explicit GpuPipeline(const ImageBuffer& image, int pipeline, int side, dim3 geometry)
        : count(image.size()), bytes(count * sizeof(float)), height(image.height()),
          width(image.width()), pipelineId(pipeline), kernelSide(side), block(geometry),
          input(bytes), first(bytes), second(bytes), third(bytes),
          kernelX((pipeline == 2 ? std::size_t{9} : static_cast<std::size_t>(side) * side) * sizeof(float)),
          kernelY(pipeline == 2 ? 9 * sizeof(float) : 0) {
        if (pipeline == 2) {
            kernelX.copyFromHost(Sobel::SX, 9 * sizeof(float));
            kernelY.copyFromHost(Sobel::SY, 9 * sizeof(float));
        } else {
            customKernel = ImageBuffer::makeGaussianKernel(side);
            kernelX.copyFromHost(customKernel.data(), customKernel.size() * sizeof(float));
        }
    }

    void copyInputs(const ImageBuffer& image) {
        input.copyFromHost(image.data(), bytes);
        if (pipelineId != 2) {
            const std::vector<float>& coeffs = customKernel;
            kernelX.copyFromHost(coeffs.data(), coeffs.size() * sizeof(float));
        }
    }

    void launch() {
        if (pipelineId != 2) {
            benchmarkLaunchConvolution(static_cast<const float*>(input.data()),
                static_cast<float*>(first.data()), static_cast<const float*>(kernelX.data()),
                height, width, kernelSide, variant, block.x, block.y);
        } else {
            benchmarkLaunchConvolution(static_cast<const float*>(input.data()),
                static_cast<float*>(first.data()), static_cast<const float*>(kernelX.data()),
                height, width, 3, variant, block.x, block.y);
            benchmarkLaunchConvolution(static_cast<const float*>(input.data()),
                static_cast<float*>(second.data()), static_cast<const float*>(kernelY.data()),
                height, width, 3, variant, block.x, block.y);
            benchmarkLaunchMagnitude(static_cast<const float*>(first.data()),
                static_cast<const float*>(second.data()), static_cast<float*>(third.data()), count);
        }
    }

    void copyOutput(ImageBuffer& host) const {
        const DeviceBuffer& result = pipelineId == 2 ? third : first;
        result.copyToHost(host.data(), bytes);
    }

    std::size_t count, bytes, height, width;
    int pipelineId, kernelSide, variant{0};
    dim3 block;
    DeviceBuffer input, first, second, third, kernelX, kernelY;
    std::vector<float> customKernel;
};

Samples timeKernel(GpuPipeline& gpu, int repetitions) {
    gpu.launch();
    synchronize(); // calentamiento fuera de la muestra
    Samples values;
    values.reserve(static_cast<std::size_t>(repetitions));
    for (int r = 0; r < repetitions; ++r) {
        synchronize();
        const auto start = Clock::now();
        gpu.launch();
        synchronize();
        values.push_back(std::chrono::duration<double, std::milli>(Clock::now() - start).count());
    }
    return values;
}

Samples timeEndToEnd(GpuPipeline& gpu, const ImageBuffer& input, ImageBuffer& output,
                     int repetitions) {
    Samples values;
    values.reserve(static_cast<std::size_t>(repetitions));
    for (int r = 0; r < repetitions; ++r) {
        synchronize();
        const auto start = Clock::now();
        gpu.copyInputs(input);
        gpu.launch();
        synchronize();
        gpu.copyOutput(output);
        values.push_back(std::chrono::duration<double, std::milli>(Clock::now() - start).count());
    }
    return values;
}

ImageBuffer gpuResult(GpuPipeline& gpu, const ImageBuffer& input) {
    gpu.copyInputs(input);
    gpu.launch();
    synchronize();
    ImageBuffer output(input.height(), input.width());
    gpu.copyOutput(output);
    return output;
}
#endif

[[maybe_unused]] void printSummary(const char* label, const Summary& stats) {
    std::cout << label << ": " << stats.mean << " +/- " << stats.deviation << " ms\n";
}

} // namespace

DeviceBuffer::~DeviceBuffer() { reset(); }
DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : ptr_(std::exchange(other.ptr_, nullptr)), bytes_(std::exchange(other.bytes_, 0)) {}
DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
        reset(); ptr_ = std::exchange(other.ptr_, nullptr); bytes_ = std::exchange(other.bytes_, 0);
    }
    return *this;
}
void DeviceBuffer::allocate(std::size_t bytes) {
    if (bytes == bytes_) return;
    reset();
    if (bytes == 0) return;
#if defined(__CUDACC__)
    checkCuda(cudaMalloc(&ptr_, bytes), "cudaMalloc"); bytes_ = bytes;
#else
    (void)bytes; throw std::runtime_error("DeviceBuffer: compilación sin soporte CUDA");
#endif
}
void DeviceBuffer::reset() noexcept {
#if defined(__CUDACC__)
    if (ptr_ != nullptr) cudaFree(ptr_);
#endif
    ptr_ = nullptr; bytes_ = 0;
}
void DeviceBuffer::copyFromHost(const void* source, std::size_t bytes) {
    if (bytes > bytes_ || (bytes != 0 && source == nullptr))
        throw std::invalid_argument("DeviceBuffer::copyFromHost: fuente o tamaño inválidos");
#if defined(__CUDACC__)
    if (bytes != 0) checkCuda(cudaMemcpy(ptr_, source, bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D");
#else
    (void)source; (void)bytes; throw std::runtime_error("DeviceBuffer: compilación sin soporte CUDA");
#endif
}
void DeviceBuffer::copyToHost(void* destination, std::size_t bytes) const {
    if (bytes > bytes_ || (bytes != 0 && destination == nullptr))
        throw std::invalid_argument("DeviceBuffer::copyToHost: destino o tamaño inválidos");
#if defined(__CUDACC__)
    if (bytes != 0) checkCuda(cudaMemcpy(destination, ptr_, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
#else
    (void)destination; (void)bytes; throw std::runtime_error("DeviceBuffer: compilación sin soporte CUDA");
#endif
}

void Benchmark::benchmarkKernelOnly(const ImageBuffer& img, const float* k, int s,
                                    int variant, dim3 block, int repetitions) {
    validate(img, k, s, variant, block, repetitions);
#if defined(__CUDACC__)
    GpuPipeline gpu(img, 0, s, block); gpu.variant = variant;
    gpu.customKernel.assign(k, k + static_cast<std::size_t>(s) * s);
    gpu.copyInputs(img);
    printSummary("Kernel-only", summarize(timeKernel(gpu, repetitions)));
#else
    throw std::runtime_error("Benchmark: se requiere compilar con nvcc y CUDA");
#endif
}

void Benchmark::benchmarkEndToEnd(const ImageBuffer& img, const float* k, int s,
                                  int variant, dim3 block, int repetitions) {
    validate(img, k, s, variant, block, repetitions);
#if defined(__CUDACC__)
    GpuPipeline gpu(img, 0, s, block); gpu.variant = variant;
    gpu.customKernel.assign(k, k + static_cast<std::size_t>(s) * s);
    ImageBuffer output(img.height(), img.width());
    printSummary("End-to-end", summarize(timeEndToEnd(gpu, img, output, repetitions)));
#else
    throw std::runtime_error("Benchmark: se requiere compilar con nvcc y CUDA");
#endif
}

void Benchmark::compareCpuGpu(const ImageBuffer& img, const float* k, int s, int repetitions) {
    validate(img, k, s, Convolution2D::VARIANT_NAIVE, dim3(16, 16), repetitions);
    Samples cpuSamples;
    cpuSamples.reserve(static_cast<std::size_t>(repetitions));
    ImageBuffer cpuOutput(img.height(), img.width());
    for (int r = 0; r < repetitions; ++r) {
        const auto start = Clock::now();
        Convolution2D cpu(img); cpu.convolveCpu(k, s); cpuOutput = cpu.outputCpu();
        cpuSamples.push_back(std::chrono::duration<double, std::milli>(Clock::now() - start).count());
    }
#if defined(__CUDACC__)
    for (const int variant : {0, 1}) {
        GpuPipeline gpu(img, 0, s, dim3(16, 16)); gpu.variant = variant;
        gpu.customKernel.assign(k, k + static_cast<std::size_t>(s) * s);
        ImageBuffer output = gpuResult(gpu, img);
        float maxError = 0.0f;
        for (std::size_t i = 0; i < img.size(); ++i)
            maxError = std::max(maxError, std::fabs(cpuOutput.data()[i] - output.data()[i]));
        if (maxError > 1.0e-3f) throw std::runtime_error("CPU/GPU no equivalente: maxAbsError=" + std::to_string(maxError));
        std::cout << "variant=" << variant << " maxAbsError=" << maxError << '\n';
        printSummary("CPU serial", summarize(cpuSamples));
        printSummary("GPU kernel-only", summarize(timeKernel(gpu, repetitions)));
    }
#else
    throw std::runtime_error("Benchmark: se requiere compilar con nvcc y CUDA");
#endif
}

void Benchmark::runSuite(const std::string& outputDirectory, int repetitions) {
    if (repetitions < 10) throw std::invalid_argument("runSuite: mínimo 10 repeticiones");
#if defined(__CUDACC__)
    if (!Convolution2D::isGpuAvailable()) throw std::runtime_error("runSuite: no hay dispositivo CUDA disponible");
    const std::filesystem::path directory(outputDirectory);
    std::filesystem::create_directories(directory);
    std::ofstream results(directory / "benchmark_results.dat");
    std::ofstream blocks(directory / "block_study.dat");
    if (!results || !blocks) throw std::runtime_error("No se pudieron abrir archivos benchmark .dat");
    results << "# size pipeline variant t_cpu_mean t_cpu_std t_konly_mean t_konly_std t_e2e_mean t_e2e_std s_konly sigma_s_konly s_e2e sigma_s_e2e f_serial s_amdahl\n";
    blocks << "# size variant bx by t_konly_mean t_konly_std\n";
    results << std::setprecision(10); blocks << std::setprecision(10);
    const std::vector<std::size_t> sizes{512, 1024, 2048, 4096};
    const std::vector<dim3> geometries{dim3(16,16), dim3(16,32), dim3(32,16), dim3(32,32)};

    for (const std::size_t n : sizes) {
        const ImageBuffer image = ImageBuffer::createRandom(n, n, 42u);
        for (int pipeline = 0; pipeline < 3; ++pipeline) {
            const int side = pipeline == 0 ? 3 : (pipeline == 1 ? 5 : 3);
            const auto cpu = summarize(timeCpu(image, pipeline, side, repetitions));
            ImageBuffer cpuOutput(n, n);
            if (pipeline == 2) {
                Sobel reference(image); reference.computeGxGyCpu(); reference.computeMagnitudeCpu();
                cpuOutput = reference.magnitudeCpu();
            } else {
                const auto coeff = ImageBuffer::makeGaussianKernel(side);
                Convolution2D reference(image); reference.convolveCpu(coeff.data(), side);
                cpuOutput = reference.outputCpu();
            }
            for (int variant = 0; variant < 2; ++variant) {
                Samples kernelMeasurements, e2eMeasurements;
                for (const dim3 geometry : geometries) {
                    GpuPipeline gpu(image, pipeline, side, geometry); gpu.variant = variant;
                    gpu.copyInputs(image);
                    Samples kernelRuns = timeKernel(gpu, repetitions);
                    const Summary kernel = summarize(kernelRuns);
                    kernelMeasurements.insert(kernelMeasurements.end(), kernelRuns.begin(), kernelRuns.end());
                    ImageBuffer output(n, n);
                    Samples e2eRuns = timeEndToEnd(gpu, image, output, repetitions);
                    const Summary e2e = summarize(e2eRuns);
                    e2eMeasurements.insert(e2eMeasurements.end(), e2eRuns.begin(), e2eRuns.end());

                    float maxError = 0.0f;
                    for (std::size_t i = 0; i < image.size(); ++i)
                        maxError = std::max(maxError, std::fabs(cpuOutput.data()[i] - output.data()[i]));
                    if (maxError > 1.0e-3f) throw std::runtime_error("Error CPU/GPU excede 1e-3: " + std::to_string(maxError));
                    if (pipeline == 0)
                        blocks << n << ' ' << variant << ' ' << geometry.x << ' ' << geometry.y
                               << kernel.mean << ' ' << kernel.deviation << '\n';
                }
                // The plotting contract has no block column: report the pooled repetitions
                // across the four required geometries; block_study.dat retains each geometry.
                const Summary kernel = summarize(kernelMeasurements);
                const Summary e2e = summarize(e2eMeasurements);
                const double speedKernel = cpu.mean / kernel.mean;
                const double sigmaKernel = speedKernel * std::sqrt(
                    std::pow(cpu.deviation / cpu.mean, 2) + std::pow(kernel.deviation / kernel.mean, 2));
                const double speedE2E = cpu.mean / e2e.mean;
                const double sigmaE2E = speedE2E * std::sqrt(
                    std::pow(cpu.deviation / cpu.mean, 2) + std::pow(e2e.deviation / e2e.mean, 2));
                const double serialFraction = std::clamp((e2e.mean - kernel.mean) / e2e.mean, 0.0, 1.0);
                const double parallelism = cpu.mean / kernel.mean;
                const double amdahl = 1.0 / (serialFraction + (1.0 - serialFraction) / parallelism);
                results << n << ' ' << pipeline << ' ' << variant << ' '
                        << cpu.mean << ' ' << cpu.deviation << ' '
                        << kernel.mean << ' ' << kernel.deviation << ' '
                        << e2e.mean << ' ' << e2e.deviation << ' '
                        << speedKernel << ' ' << sigmaKernel << ' '
                        << speedE2E << ' ' << sigmaE2E << ' '
                        << serialFraction << ' ' << amdahl << '\n';
            }
        }
    }
    if (!results || !blocks) throw std::runtime_error("Error escribiendo resultados benchmark");
#else
    (void)outputDirectory;
    throw std::runtime_error("runSuite: se requiere compilar con nvcc y CUDA");
#endif
}
