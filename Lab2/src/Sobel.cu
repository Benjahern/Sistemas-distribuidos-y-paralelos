#include "Sobel.h"

#if defined(__CUDACC__)

/**
 * @file Sobel.cu
 * @brief Stub de integración para Rol 3 (`Sobel::computeMagnitudeGpu`).
 *
 * NOTA DE EQUIPO:
 * La implementación del kernel CUDA map para `G(i,j) = sqrt(G_x(i,j)^2 + G_y(i,j)^2)`
 * corresponde al integrante del **Rol 3 (Kernels CUDA)**.
 */

void Sobel::computeMagnitudeGpu() {
    // TODO (Rol 3): Implementar kernel CUDA map para calcular la magnitud G a partir de G_x y G_y
    mag_gpu_.resize(input_.height(), input_.width(), 0.0f);
}

#endif // defined(__CUDACC__)
