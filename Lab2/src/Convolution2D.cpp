#include "Convolution2D.h"

#include <cstdint>
#include <stdexcept>
#include <utility>

/**
 * @file Convolution2D.cpp
 * @brief Referencia serial del Rol 2 y delegación de sobrecargas GPU.
 */

Convolution2D::Convolution2D(const ImageBuffer& input)
    : input_(input),
      output_cpu_(input.height(), input.width(), 0.0f),
      output_gpu_(input.height(), input.width(), 0.0f) {}

void Convolution2D::setInput(const ImageBuffer& input) {
    input_ = input;
    output_cpu_.resize(input.height(), input.width(), 0.0f);
    output_gpu_.resize(input.height(), input.width(), 0.0f);
}

void Convolution2D::convolveCpu(const float* k, int s) {
    if (k == nullptr || s <= 0 || s % 2 == 0) {
        throw std::invalid_argument("convolveCpu: k no nulo y s impar positivo requeridos");
    }

    // Resultado independiente: permite reutilizar el operador sin modificar la entrada
    // y conserva la salida previa si falla una reserva de memoria.
    ImageBuffer result(input_.height(), input_.width());
    const auto height = static_cast<std::int64_t>(input_.height());
    const auto width = static_cast<std::int64_t>(input_.width());
    const std::int64_t radius = s / 2;
    for (std::int64_t i = 0; i < height; ++i) {
        for (std::int64_t j = 0; j < width; ++j) {
            float sum = 0.0f;
            for (int ku = 0; ku < s; ++ku) {
                const auto source_i = i - (static_cast<std::int64_t>(ku) - radius);
                if (source_i < 0 || source_i >= height) continue;
                const float* source_row = input_.row(static_cast<std::size_t>(source_i));
                for (int kv = 0; kv < s; ++kv) {
                    const auto source_j = j - (static_cast<std::int64_t>(kv) - radius);
                    if (source_j < 0 || source_j >= width) continue;
                    const auto kernel_index = static_cast<std::size_t>(ku) *
                                              static_cast<std::size_t>(s) + kv;
                    sum += k[kernel_index] * source_row[static_cast<std::size_t>(source_j)];
                }
            }
            result(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) = sum;
        }
    }
    output_cpu_ = std::move(result);
}

void Convolution2D::convolveGpu(const float* k, int s) {
    // Sobrecarga simple obligatoria: delega en la variante por defecto (VARIANT_SHARED = 1)
    convolveGpu(k, s, VARIANT_SHARED, dim3(16, 16));
}

void Convolution2D::convolveGpu(const float* k, int s, int variant) {
    // Sobrecarga intermedia obligatoria: delega en la versión parametrizada con bloque 16x16
    convolveGpu(k, s, variant, dim3(16, 16));
}

bool Convolution2D::isGpuAvailable() noexcept {
#if defined(__CUDACC__)
    int device_count = 0;
    const cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count <= 0) {
        cudaGetLastError();
        return false;
    }
    return true;
#else
    return false;
#endif
}

#if !defined(__CUDACC__)
void Convolution2D::convolveGpu(const float* /*k*/, int /*s*/, int /*variant*/, dim3 /*block*/) {
    // Stub cuando se compila en modo solo-CPU sin nvcc
    output_gpu_.resize(input_.height(), input_.width(), 0.0f);
}
#endif
