#include "Convolution2D.h"

/**
 * @file Convolution2D.cpp
 * @brief Stub de integración para Rol 2 (`Convolution2D` en CPU) y delegación de sobrecargas GPU.
 *
 * NOTA DE EQUIPO:
 * La implementación real de `convolveCpu(const float* k, int s)` corresponde al integrante
 * del **Rol 2 (Convolución CPU)**. Aquí se mantienen únicamente los constructores, accesores
 * y las firmas vacías (stubs) para que el proyecto compile e integre limpiamente con los Roles 1 y 5.
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

void Convolution2D::convolveCpu(const float* /*k*/, int /*s*/) {
    // TODO (Rol 2): Implementar convolución 2D serial en CPU con política de borde Zero-padding
    output_cpu_.resize(input_.height(), input_.width(), 0.0f);
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
