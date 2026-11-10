#include "Convolution2D.h"

#if defined(__CUDACC__)

/**
 * @file Convolution2D.cu
 * @brief Stub de integración para Rol 3 (Kernels CUDA `naive` y `shared` de convolución 2D).
 *
 * NOTA DE EQUIPO:
 * La implementación de los kernels `__global__` (`naive` en memoria global y `shared` con tile + halo
 * en `__shared__` + `__syncthreads()`), la función auxiliar `__device__` y las llamadas con `CUDA_CHECK`
 * están a cargo del integrante del **Rol 3 (Kernels CUDA)**.
 */

void Convolution2D::convolveGpu(const float* /*k*/, int /*s*/, int /*variant*/, dim3 /*block*/) {
    // TODO (Rol 3): Implementar reserva device, copia H2D, lanzamiento de kernel (0=naive, 1=shared) y copia D2H
    output_gpu_.resize(input_.height(), input_.width(), 0.0f);
}

#endif // defined(__CUDACC__)
