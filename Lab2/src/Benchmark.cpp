#include "Benchmark.h"

/**
 * @file Benchmark.cpp
 * @brief Stub de integración para Rol 4 (`Benchmark`).
 *
 * NOTA: De acuerdo con la decisión de equipo documentada en `README.md`, la implementación de
 * las mediciones (`benchmarkKernelOnly`, `benchmarkEndToEnd`, `compareCpuGpu`) y la escritura de
 * `benchmark_results.dat` y `block_study.dat` quedan reservadas para el integrante del **Rol 4**.
 */

void Benchmark::benchmarkKernelOnly(const ImageBuffer& /*img*/, const float* /*k*/, int /*s*/,
                                    int /*variant*/, dim3 /*block*/, int /*repetitions*/) {
    // Pendiente de implementación por Rol 4
}

void Benchmark::benchmarkEndToEnd(const ImageBuffer& /*img*/, const float* /*k*/, int /*s*/,
                                  int /*variant*/, dim3 /*block*/, int /*repetitions*/) {
    // Pendiente de implementación por Rol 4
}

void Benchmark::compareCpuGpu(const ImageBuffer& /*img*/, const float* /*k*/, int /*s*/,
                              int /*repetitions*/) {
    // Pendiente de implementación por Rol 4
}

