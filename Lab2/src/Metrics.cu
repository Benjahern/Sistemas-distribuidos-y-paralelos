#include "Metrics.h"

/**
 * @file Metrics.cu
 * @brief Stub de integración para Rol 3 (`Metrics::maxAbsDiffCpuGpu` y `Metrics::sumMagnitude`).
 *
 * NOTA DE EQUIPO:
 * La implementación de la comparación numérica `maxAbsDiffCpuGpu` y de los dos kernels de
 * reducción en GPU (`method = 0` con memoria compartida `__shared__` y `method = 1` con `atomicAdd`)
 * corresponde al integrante del **Rol 3 (Kernels CUDA)**.
 */

Metrics::Metrics(const ImageBuffer& magnitude) : magnitude_(magnitude) {}

void Metrics::setMagnitudeBuffer(const ImageBuffer& magnitude) {
    magnitude_ = magnitude;
}

void Metrics::setComparisonBuffers(const ImageBuffer& cpu_buf, const ImageBuffer& gpu_buf) {
    cpu_ref_ = cpu_buf;
    gpu_res_ = gpu_buf;
}

float Metrics::maxAbsDiffCpuGpu(const ImageBuffer& /*cpu_buf*/, const ImageBuffer& /*gpu_buf*/) {
    // TODO (Rol 3): Implementar cálculo de max_{i,j} |cpu_buf(i,j) - gpu_buf(i,j)|
    return 0.0f;
}

float Metrics::maxAbsDiffCpuGpu() const {
    return maxAbsDiffCpuGpu(cpu_ref_, gpu_res_);
}

float Metrics::sumMagnitude(int method) const {
    return sumMagnitude(magnitude_, method);
}

float Metrics::sumMagnitude(const ImageBuffer& /*magnitude*/, int /*method*/) {
    // TODO (Rol 3): Implementar reducción en GPU con method=0 (shared memory) y method=1 (atomicAdd)
    return 0.0f;
}
