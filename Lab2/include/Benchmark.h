#pragma once

#include "Convolution2D.h"
#include "ImageBuffer.h"
#include <string>

/**
 * @file Benchmark.h
 * @brief Rol 4 (Host/Device y Benchmarks): Interfaz y contratos para medición de tiempos,
 *        transferencias H2D/D2H y estudio de geometrías de bloque.
 *
 * NOTA DE COORDINACIÓN DE EQUIPO (Rol 1 & 5):
 * Siguiendo el acuerdo del equipo, la implementación completa del barrido de benchmarks y la
 * generación de `benchmark_results.dat` y `block_study.dat` está a cargo del **Rol 4**.
 * Este archivo define únicamente el contrato obligatorio de la Sección 6.2 para permitir la
 * compilación e integración modular del proyecto sin invadir las responsabilidades del Rol 4.
 */
class Benchmark {
public:
    Benchmark() = default;

    /**
     * @brief Mide exclusivamente el tiempo de ejecución de los kernels en GPU (+ sincronización),
     *        excluyendo las transferencias Host-to-Device (H2D) y Device-to-Host (D2H).
     * @param img Imagen de entrada.
     * @param k Coeficientes del kernel de filtro (si aplica).
     * @param s Lado impar del kernel.
     * @param variant `0` = naive, `1` = shared.
     * @param block Geometría del bloque 2D `(blockDim.x, blockDim.y)`.
     * @param repetitions Número de repeticiones (mínimo 10 según Sección 7).
     */
    void benchmarkKernelOnly(const ImageBuffer& img, const float* k, int s, int variant,
                             dim3 block = dim3(16, 16), int repetitions = 10);

    /**
     * @brief Mide el tiempo de aplicación de extremo a extremo (end-to-end):
     *        incluye copia H2D de la entrada + ejecución de kernels + copia D2H de la salida.
     * @param img Imagen de entrada.
     * @param k Coeficientes del kernel de filtro (si aplica).
     * @param s Lado impar del kernel.
     * @param variant `0` = naive, `1` = shared.
     * @param block Geometría del bloque 2D `(blockDim.x, blockDim.y)`.
     * @param repetitions Número de repeticiones (mínimo 10 según Sección 7).
     */
    void benchmarkEndToEnd(const ImageBuffer& img, const float* k, int s, int variant,
                           dim3 block = dim3(16, 16), int repetitions = 10);

    /**
     * @brief Compara tiempos y equivalencia numérica entre la referencia CPU serial y las variantes GPU.
     * @param img Imagen de entrada.
     * @param k Coeficientes del kernel de filtro.
     * @param s Lado impar del kernel.
     * @param repetitions Número de repeticiones (mínimo 10).
     */
    void compareCpuGpu(const ImageBuffer& img, const float* k, int s, int repetitions = 10);
};

