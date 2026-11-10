#pragma once

#include "ImageBuffer.h"

#if defined(__CUDACC__) || __has_include(<cuda_runtime.h>)
#include <cuda_runtime.h>
#else
/**
 * @brief Definición ligera de `dim3` cuando se compila en modo solo-CPU (sin toolkit CUDA instalado).
 *        Permite mantener idénticas las firmas obligatorias de la Sección 6.2 en cualquier entorno.
 */
struct dim3 {
    unsigned int x, y, z;
    constexpr dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1)
        : x(vx), y(vy), z(vz) {}
};
#endif

/**
 * @file Convolution2D.h
 * @brief Roles 2 y 3 (Convolución CPU y Kernels CUDA): Interfaz unificada de convolución espacial 2D.
 *
 * Convenciones acordadas para todo el laboratorio (documentadas en README.md):
 * - Notación: `s = 2r + 1` es el lado impar del kernel; `k` apunta a los `s * s` coeficientes en row-major.
 * - Política de borde única: **Zero-padding** (fuera de `[0, H) x [0, W)` se lee `0.0f`).
 * - Fórmula espacial (correlación/convolución indexada con desplazamiento directo):
 *   `Out(i, j) = sum_{u=-r}^{r} sum_{v=-r}^{r} k[(u + r)*s + (v + r)] * I_pad(i + u, j + v)`
 *   (Documentada de forma idéntica para CPU serial, GPU naive y GPU shared).
 */
class Convolution2D {
public:
    /**
     * @brief Constantes para seleccionar la variante de kernel CUDA.
     */
    static constexpr int VARIANT_NAIVE  = 0; ///< Lectura directa desde memoria global.
    static constexpr int VARIANT_SHARED = 1; ///< Tile + halo de radio r en `__shared__` + `__syncthreads()`.

    /**
     * @brief Construye un operador de convolución 2D asociado a una imagen de entrada.
     * @param input Imagen de entrada en host (H x W).
     */
    explicit Convolution2D(const ImageBuffer& input = ImageBuffer());

    /**
     * @brief Actualiza la imagen de entrada sobre la cual se aplicará la convolución.
     * @param input Nueva imagen de entrada en host.
     */
    void setInput(const ImageBuffer& input);

    /**
     * @brief Retorna la imagen de entrada actual.
     * @return Referencia constante al `ImageBuffer` de entrada.
     */
    const ImageBuffer& input() const noexcept { return input_; }

    /**
     * @brief Retorna la última salida calculada en CPU (`convolveCpu`).
     * @return Referencia constante al `ImageBuffer` de salida CPU.
     */
    const ImageBuffer& outputCpu() const noexcept { return output_cpu_; }

    /**
     * @brief Retorna la última salida calculada en GPU (`convolveGpu`), copiada a host.
     * @return Referencia constante al `ImageBuffer` de salida GPU.
     */
    const ImageBuffer& outputGpu() const noexcept { return output_gpu_; }

    // =========================================================================
    // Firmas Obligatorias exigidas en AGENTS.md y Sección 6.2 de lab2.md
    // =========================================================================

    /**
     * @brief Referencia serial en CPU para la convolución 2D con política de borde Zero-padding.
     * @param k Puntero a los `s * s` coeficientes del kernel en orden row-major.
     * @param s Lado impar del kernel (`s = 2r + 1`, ej. 3, 5, 7).
     */
    void convolveCpu(const float* k, int s);

    /**
     * @brief Sobrecarga simple en GPU: delega en la variante por defecto (`VARIANT_SHARED` = 1)
     *        y geometría de bloque por defecto `dim3(16, 16)`.
     * @param k Puntero a los `s * s` coeficientes del kernel.
     * @param s Lado impar del kernel.
     */
    void convolveGpu(const float* k, int s);

    /**
     * @brief Sobrecarga parametrizada por variante en GPU: delega en `convolveGpu(k, s, variant, dim3(16, 16))`.
     * @param k Puntero a los `s * s` coeficientes del kernel.
     * @param s Lado impar del kernel.
     * @param variant `0` para `naive` (memoria global), `1` para `shared` (memoria compartida con halo).
     */
    void convolveGpu(const float* k, int s, int variant);

    /**
     * @brief Ejecuta la convolución 2D en GPU con la variante y geometría de bloque 2D indicadas.
     * @param k Puntero a los `s * s` coeficientes del kernel.
     * @param s Lado impar del kernel.
     * @param variant `0` = naive, `1` = shared.
     * @param block Dimensiones del bloque CUDA 2D `(blockDim.x, blockDim.y)`.
     */
    void convolveGpu(const float* k, int s, int variant, dim3 block);

    /**
     * @brief Verifica si el binario actual cuenta con al menos un dispositivo GPU CUDA accesible en tiempo de ejecución.
     *        Usado por la suite de pruebas (Rol 5) para manejo inteligente de ausencia de GPU física en CI.
     * @return true si hay una GPU CUDA disponible y operativa, false en caso contrario.
     */
    static bool isGpuAvailable() noexcept;

private:
    ImageBuffer input_;
    ImageBuffer output_cpu_;
    ImageBuffer output_gpu_;
};

