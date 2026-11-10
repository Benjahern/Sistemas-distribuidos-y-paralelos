#include "Sobel.h"

/**
 * @file Sobel.cpp
 * @brief Stub de integración para Rol 2 (`computeGxGyCpu`, `computeMagnitudeCpu`)
 *        y delegación de `computeGxGyGpu` (Rol 3).
 *
 * NOTA DE EQUIPO:
 * La implementación de los gradientes de Sobel reutilizando `Convolution2D` y el cálculo
 * de la magnitud `G = sqrt(G_x^2 + G_y^2)` en CPU corresponden al **Rol 2**.
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
    // TODO (Rol 2): Invocar Convolution2D::convolveCpu con SX y SY (3x3)
    gx_cpu_.resize(input_.height(), input_.width(), 0.0f);
    gy_cpu_.resize(input_.height(), input_.width(), 0.0f);
}

void Sobel::computeMagnitudeCpu() {
    // TODO (Rol 2): Calcular G(i,j) = sqrt(G_x(i,j)^2 + G_y(i,j)^2) en CPU
    mag_cpu_.resize(input_.height(), input_.width(), 0.0f);
}

void Sobel::computeGxGyGpu(int /*variant*/) {
    // TODO (Rol 3): Invocar Convolution2D::convolveGpu con SX y SY (3x3) y la variante dada
    gx_gpu_.resize(input_.height(), input_.width(), 0.0f);
    gy_gpu_.resize(input_.height(), input_.width(), 0.0f);
}

#if !defined(__CUDACC__)
void Sobel::computeMagnitudeGpu() {
    // Stub cuando se compila sin nvcc
    mag_gpu_.resize(input_.height(), input_.width(), 0.0f);
}
#endif
