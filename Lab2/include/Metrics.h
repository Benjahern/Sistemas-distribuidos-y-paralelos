#pragma once

#include "ImageBuffer.h"

/**
 * @file Metrics.h
 * @brief Rol 3 (Métricas de Validación y Reducción en GPU): Comparación CPU-GPU y reducciones globales.
 *
 * Contratos obligatorios (AGENTS.md y Sección 6.2 de lab2.md):
 * - `maxAbsDiffCpuGpu`: Calcula `max_{i,j} |A(i,j) - B(i,j)|` entre buffers comparables.
 * - `sumMagnitude(int method)`: Calcula la suma global `\sum_{i,j} G(i,j)` en GPU donde:
 *   - `method = 0`: Reducción en memoria compartida (`__shared__`).
 *   - `method = 1`: Acumulación con operaciones atómicas (`atomicAdd`).
 */
class Metrics {
public:
    static constexpr int METHOD_SHARED_REDUCTION = 0; ///< Reducción por árbol en `__shared__`.
    static constexpr int METHOD_ATOMIC_ADD       = 1; ///< Acumulación usando `atomicAdd`.

    /**
     * @brief Construye un evaluador de métricas asociado opcionalmente a un buffer de magnitud G.
     * @param magnitude Buffer de magnitud `G` sobre el cual ejecutar `sumMagnitude(method)`.
     */
    explicit Metrics(const ImageBuffer& magnitude = ImageBuffer());

    /**
     * @brief Asigna el buffer de magnitud `G` para las reducciones `sumMagnitude`.
     * @param magnitude Buffer `ImageBuffer` con la magnitud `G`.
     */
    void setMagnitudeBuffer(const ImageBuffer& magnitude);

    /**
     * @brief Calcula el máximo error absoluto punto a punto entre un buffer de referencia CPU
     *        y un buffer obtenido en GPU (o entre cualquier par de buffers de igual dimensión).
     * @param cpu_buf Buffer de referencia calculado en CPU.
     * @param gpu_buf Buffer calculado en GPU.
     * @return Valor `max_{i,j} |cpu_buf(i,j) - gpu_buf(i,j)|`.
     * @throws std::invalid_argument si las dimensiones de ambos buffers no coinciden.
     */
    static float maxAbsDiffCpuGpu(const ImageBuffer& cpu_buf, const ImageBuffer& gpu_buf);

    /**
     * @brief Sobrecarga de instancia para `maxAbsDiffCpuGpu`.
     */
    float maxAbsDiffCpuGpu() const;

    /**
     * @brief Asigna los dos buffers (CPU y GPU) para la sobrecarga sin argumentos de `maxAbsDiffCpuGpu()`.
     * @param cpu_buf Buffer de referencia CPU.
     * @param gpu_buf Buffer de resultado GPU.
     */
    void setComparisonBuffers(const ImageBuffer& cpu_buf, const ImageBuffer& gpu_buf);

    /**
     * @brief Calcula la suma total `\sum_{i,j} G(i,j)` del buffer de magnitud configurado.
     * @param method `0` = reducción en memoria compartida (`__shared__`), `1` = `atomicAdd`.
     * @return Suma total de los elementos como `float`.
     */
    float sumMagnitude(int method) const;

    /**
     * @brief Sobrecarga estática que recibe explícitamente el buffer `G` y el método de reducción.
     * @param magnitude Buffer `ImageBuffer` cuyos elementos se sumarán.
     * @param method `0` = reducción en memoria compartida (`__shared__`), `1` = `atomicAdd`.
     * @return Suma total de los elementos como `float`.
     */
    static float sumMagnitude(const ImageBuffer& magnitude, int method);

private:
    ImageBuffer magnitude_;
    ImageBuffer cpu_ref_;
    ImageBuffer gpu_res_;
};

