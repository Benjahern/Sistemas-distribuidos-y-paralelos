#include "Sobel.h"

#include <cmath>
#include <stdexcept>

Sobel::Sobel(const ImageBuffer& input) : input_(input) {}

void Sobel::setInput(const ImageBuffer& input) {
    input_ = input;
}

void Sobel::computeGxGyCpu() {
    // Reutiliza obligatoriamente la misma API genérica de Convolution2D (Sección 5.2 / AGENTS.md)
    Convolution2D conv(input_);
    conv.convolveCpu(SX, 3);
    gx_cpu_ = conv.outputCpu();

    conv.convolveCpu(SY, 3);
    gy_cpu_ = conv.outputCpu();
}

void Sobel::computeMagnitudeCpu() {
    if (gx_cpu_.height() != gy_cpu_.height() || gx_cpu_.width() != gy_cpu_.width()) {
        throw std::runtime_error("Sobel::computeMagnitudeCpu: ejecute computeGxGyCpu() antes de computeMagnitudeCpu()");
    }

    mag_cpu_.resize(gx_cpu_.height(), gx_cpu_.width(), 0.0f);
    const float* gx = gx_cpu_.data();
    const float* gy = gy_cpu_.data();
    float* mag = mag_cpu_.data();
    const std::size_t total = mag_cpu_.size();

    for (std::size_t idx = 0; idx < total; ++idx) {
        mag[idx] = std::sqrt(gx[idx] * gx[idx] + gy[idx] * gy[idx]);
    }
}

void Sobel::computeGxGyGpu(int variant) {
    // Reutiliza obligatoriamente la misma API genérica de Convolution2D en GPU
    Convolution2D conv(input_);
    conv.convolveGpu(SX, 3, variant);
    gx_gpu_ = conv.outputGpu();

    conv.convolveGpu(SY, 3, variant);
    gy_gpu_ = conv.outputGpu();
}

#if !defined(__CUDACC__)
void Sobel::computeMagnitudeGpu() {
    throw std::runtime_error(
        "Sobel::computeMagnitudeGpu: binario compilado sin soporte CUDA (__CUDACC__ no definido)");
}
#endif

