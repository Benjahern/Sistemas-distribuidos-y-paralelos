#pragma once

#include "Convolution2D.h"
#include "ImageBuffer.h"

/**
 * @file Sobel.h
 * @brief Roles 2 y 3 (Sobel CPU y GPU): Detección de bordes mediante operadores de Sobel 3x3.
 *
 * Regla de oro (Sección 5.2 y AGENTS.md):
 * - NO existe un algoritmo Sobel separado de la convolución: `G_x = I * S_x` y `G_y = I * S_y`
 *   se calculan invocando `Convolution2D::convolveCpu` / `Convolution2D::convolveGpu`.
 * - La magnitud `G(i, j) = sqrt(G_x(i, j)^2 + G_y(i, j)^2)` es un operador map punto a punto.
 * - Matrices 3x3 fijas obligatorias:
 *   S_x = [ [-1, 0, 1], [-2, 0, 2], [-1, 0, 1] ]
 *   S_y = [ [-1, -2, -1], [0, 0, 0], [1, 2, 1] ]
 */
class Sobel {
public:
    /**
     * @brief Coeficientes fijos del operador horizontal de Sobel S_x (3x3, row-major).
     */
    static constexpr float SX[9] = {
        -1.0f,  0.0f,  1.0f,
        -2.0f,  0.0f,  2.0f,
        -1.0f,  0.0f,  1.0f
    };

    /**
     * @brief Coeficientes fijos del operador vertical de Sobel S_y (3x3, row-major).
     */
    static constexpr float SY[9] = {
        -1.0f, -2.0f, -1.0f,
         0.0f,  0.0f,  0.0f,
         1.0f,  2.0f,  1.0f
    };

    /**
     * @brief Construye un detector de bordes de Sobel para la imagen de entrada dada.
     * @param input Imagen de entrada en escala de grises (H x W).
     */
    explicit Sobel(const ImageBuffer& input = ImageBuffer());

    /**
     * @brief Actualiza la imagen de entrada sobre la cual se calculará Sobel.
     * @param input Nueva imagen de entrada.
     */
    void setInput(const ImageBuffer& input);

    /**
     * @brief Retorna la imagen de entrada actual.
     * @return Referencia constante al `ImageBuffer` de entrada.
     */
    const ImageBuffer& input() const noexcept { return input_; }

    // =========================================================================
    // Firmas Obligatorias exigidas en AGENTS.md y Sección 6.2 de lab2.md
    // =========================================================================

    /**
     * @brief Calcula los gradientes `G_x` y `G_y` en CPU reutilizando `Convolution2D::convolveCpu`.
     */
    void computeGxGyCpu();

    /**
     * @brief Calcula los gradientes `G_x` y `G_y` en GPU reutilizando `Convolution2D::convolveGpu`.
     * @param variant `0` para kernel `naive`, `1` para kernel `shared`.
     */
    void computeGxGyGpu(int variant = Convolution2D::VARIANT_SHARED);

    /**
     * @brief Calcula la magnitud `G(i,j) = sqrt(G_x(i,j)^2 + G_y(i,j)^2)` en CPU.
     */
    void computeMagnitudeCpu();

    /**
     * @brief Calcula la magnitud `G(i,j) = sqrt(G_x(i,j)^2 + G_y(i,j)^2)` en GPU mediante un kernel map.
     */
    void computeMagnitudeGpu();

    // =========================================================================
    // Accesores a los buffers de resultados (CPU y GPU)
    // =========================================================================

    const ImageBuffer& gxCpu() const noexcept { return gx_cpu_; }
    const ImageBuffer& gyCpu() const noexcept { return gy_cpu_; }
    const ImageBuffer& magnitudeCpu() const noexcept { return mag_cpu_; }

    const ImageBuffer& gxGpu() const noexcept { return gx_gpu_; }
    const ImageBuffer& gyGpu() const noexcept { return gy_gpu_; }
    const ImageBuffer& magnitudeGpu() const noexcept { return mag_gpu_; }

private:
    ImageBuffer input_;
    ImageBuffer gx_cpu_;
    ImageBuffer gy_cpu_;
    ImageBuffer mag_cpu_;

    ImageBuffer gx_gpu_;
    ImageBuffer gy_gpu_;
    ImageBuffer mag_gpu_;
};

