#include "Sobel.h"

#include <cmath>
#include <stdexcept>
#include <utility>

/**
 * @file Sobel.cpp
 * @brief Implementación serial del Rol 2 (`computeGxGyCpu`, `computeMagnitudeCpu`)
 *        y delegación de `computeGxGyGpu` (Rol 3).
 *
 * Gradientes mediante la convolución genérica; magnitud como map punto a punto.
 */

Sobel::Sobel(const ImageBuffer& input)
    : input_(input),
      gx_cpu_(input.height(), input.width(), 0.0f),
      gy_cpu_(input.height(), input.width(), 0.0f),
      mag_cpu_(input.height(), input.width(), 0.0f),
      gx_gpu_(input.height(), input.width(), 0.0f),
      gy_gpu_(input.height(), input.width(), 0.0f),
      mag_gpu_(input.height(), input.width(), 0.0f) {}

void Sobel::setInput(const ImageBuffer& input) {
    input_ = input;
    gx_cpu_.resize(input.height(), input.width(), 0.0f);
    gy_cpu_.resize(input.height(), input.width(), 0.0f);
    mag_cpu_.resize(input.height(), input.width(), 0.0f);
    gx_gpu_.resize(input.height(), input.width(), 0.0f);
    gy_gpu_.resize(input.height(), input.width(), 0.0f);
    mag_gpu_.resize(input.height(), input.width(), 0.0f);
}

void Sobel::computeGxGyCpu() {
    Convolution2D convolution(input_);
    convolution.convolveCpu(SX, 3);
    gx_cpu_ = convolution.outputCpu();
    convolution.convolveCpu(SY, 3);
    gy_cpu_ = convolution.outputCpu();
}

void Sobel::computeMagnitudeCpu() {
    mag_cpu_.resize(input_.height(), input_.width(), 0.0f);
    for (std::size_t index = 0; index < input_.size(); ++index) {
        const float gx = gx_cpu_.data()[index];
        const float gy = gy_cpu_.data()[index];
        mag_cpu_.data()[index] = std::sqrt(gx * gx + gy * gy);
    }
}

void Sobel::computeGxGyGpu(int variant) {
    Convolution2D convolution(input_);

    convolution.convolveGpu(SX, 3, variant);
    ImageBuffer gx = convolution.outputGpu();

    convolution.convolveGpu(SY, 3, variant);
    ImageBuffer gy = convolution.outputGpu();

    gx_gpu_ = std::move(gx);
    gy_gpu_ = std::move(gy);
}

#if !defined(__CUDACC__)
void Sobel::computeMagnitudeGpu() {
    throw std::runtime_error(
        "computeMagnitudeGpu: Compilacion sin soporte CUDA"
    );
}
#endif
