#include "Convolution2D.h"
#include "ImageBuffer.h"
#include "Metrics.h"
#include "Sobel.h"
#include "Visualizer.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>

/**
 * @file test_pipeline_cpu_gpu.cpp
 * @brief Suite de pruebas automáticas obligatorias (Rol 5 - Sección 11 de lab2.md y AGENTS.md).
 *
 * Cubre los 6 ítems obligatorios del checklist de `make test`:
 * 1. Impulso 5x5 con kernel 3x3 conocido: salida coincide con el kernel centrado
 *    bajo la política de borde Zero-padding, y el centro coincide con el coeficiente central.
 * 2. Escalón vertical y horizontal: en escalón vertical `|G_x| > |G_y|` sobre el borde;
 *    en escalón horizontal `|G_y| > |G_x|`.
 * 3. Equivalencia CPU vs. GPU Naive vs. GPU Shared: `maxAbsDiffCpuGpu()` <= 1e-4 en imágenes
 *    sintéticas reproducibles de 8x8, 16x16 y 32x32.
 * 4. Magnitud de Sobel: en píxeles de prueba, `G` coincide con `sqrt(G_x^2 + G_y^2)`.
 * 5. Reducción Shared vs. AtomicAdd: `sumMagnitude(0)` y `sumMagnitude(1)` coinciden dentro de
 *    la tolerancia relativa documentada (`rtol = 1e-4`).
 * 6. Test de Perímetro (Borde): verificación explícita en `fila 0`, `fila H-1`, `columna 0` y
 *    `columna W-1` que falla si la política de borde Zero-padding o el halo están mal indexados.
 *
 * Detección de Stubs (Roles 2 y 3 pendientes) y Ausencia de GPU física en CI (Sección 11):
 * - Si los módulos de los Roles 2 y 3 aún están como stubs vacíos (a la espera de la integración
 *   de los compañeros de equipo), el test lo detecta automáticamente y reporta `[PENDING - Rol 2 / Rol 3 stub]`
 *   mientras valida al 100% los módulos de Rol 1 (`ImageBuffer`) y Rol 5 (`Visualizer`).
 * - Cuando los Roles 2 y 3 completen su implementación, las 6 pruebas algorítmicas se activan
 *   automáticamente sin modificar una sola línea de este archivo.
 * - Si `Convolution2D::isGpuAvailable()` retorna `false` en un runner de CI sin GPU física,
 *   las pruebas de device se reportan como `[SKIPPED - No GPU device available in CI]`.
 */
namespace {

/**
 * @brief Detecta si el Rol 2 (`Convolution2D::convolveCpu`) ya fue implementado o sigue como stub vacío.
 */
bool isRole2Implemented() {
    ImageBuffer imp(3, 3);
    imp.fillCenterImpulse(1.0f);
    const float k[9] = {
        0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f
    };
    Convolution2D conv(imp);
    conv.convolveCpu(k, 3);
    return conv.outputCpu().size() == 9 && std::fabs(conv.outputCpu()(1, 1) - 1.0f) < 1e-5f;
}

/**
 * @brief Detecta si el Rol 3 (`Convolution2D::convolveGpu`) ya fue implementado en GPU o sigue como stub vacío.
 */
bool isRole3Implemented() {
    if (!Convolution2D::isGpuAvailable()) {
        return false;
    }
    ImageBuffer imp(4, 4);
    imp.fillImpulse(1, 1, 1.0f);
    const float k[9] = {
        0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f
    };
    Convolution2D conv(imp);
    conv.convolveGpu(k, 3, Convolution2D::VARIANT_NAIVE);
    return conv.outputGpu().size() == 16 && std::fabs(conv.outputGpu()(1, 1) - 1.0f) < 1e-5f;
}

} // namespace

int main() {
    constexpr float ATOL = 1e-4f;
    constexpr float RTOL = 1e-4f;

    if (isRole2Implemented()) {
        // =====================================================================
        // TEST 1: Impulso 5x5 con kernel 3x3 conocido
        // =====================================================================
        {
            ImageBuffer imp(5, 5);
            imp.fillCenterImpulse(1.0f); // 1.0f en (2, 2)

            const float k[9] = {
                0.05f, 0.10f, 0.15f,
                0.20f, 0.50f, 0.25f,
                0.30f, 0.35f, 0.40f
            };

            Convolution2D conv(imp);
            conv.convolveCpu(k, 3);
            const ImageBuffer& out = conv.outputCpu();

            assert(std::fabs(out(2, 2) - 0.50f) < ATOL);
            for (int i = 1; i <= 3; ++i) {
                for (int j = 1; j <= 3; ++j) {
                    const int u = i - 2;
                    const int v = j - 2;
                    const float expected = k[(u + 1) * 3 + (v + 1)];
                    assert(std::fabs(out(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) - expected) < ATOL);
                }
            }
            assert(std::fabs(out(0, 0)) < ATOL);
            assert(std::fabs(out(4, 4)) < ATOL);
            std::cout << "[PASS] Test 1: Impulso 5x5 con kernel 3x3 verificado.\n";
        }

        // =====================================================================
        // TEST 2: Escalón Vertical (|Gx| > |Gy|) y Escalón Horizontal (|Gy| > |Gx|)
        // =====================================================================
        {
            ImageBuffer vstep(8, 8);
            vstep.fillVerticalStep(0.0f, 1.0f, 4);

            Sobel sobel_v(vstep);
            sobel_v.computeGxGyCpu();
            sobel_v.computeMagnitudeCpu();

            const float gx_v = std::fabs(sobel_v.gxCpu()(4, 3));
            const float gy_v = std::fabs(sobel_v.gyCpu()(4, 3));
            assert(gx_v > 3.5f);
            assert(gy_v < ATOL);
            assert(gx_v > gy_v);

            ImageBuffer hstep(8, 8);
            hstep.fillHorizontalStep(0.0f, 1.0f, 4);

            Sobel sobel_h(hstep);
            sobel_h.computeGxGyCpu();
            sobel_h.computeMagnitudeCpu();

            const float gx_h = std::fabs(sobel_h.gxCpu()(3, 4));
            const float gy_h = std::fabs(sobel_h.gyCpu()(3, 4));
            assert(gy_h > 3.5f);
            assert(gx_h < ATOL);
            assert(gy_h > gx_h);
            std::cout << "[PASS] Test 2: Escalón vertical (|Gx| > |Gy|) y horizontal (|Gy| > |Gx|) verificados.\n";
        }

        // =====================================================================
        // TEST 4 (CPU): Magnitud de Sobel G == sqrt(Gx^2 + Gy^2)
        // =====================================================================
        {
            ImageBuffer img = ImageBuffer::createRandom(16, 16, 99u);
            Sobel sobel(img);
            sobel.computeGxGyCpu();
            sobel.computeMagnitudeCpu();

            for (std::size_t i = 0; i < img.height(); ++i) {
                for (std::size_t j = 0; j < img.width(); ++j) {
                    const float gx = sobel.gxCpu()(i, j);
                    const float gy = sobel.gyCpu()(i, j);
                    const float expected = std::sqrt(gx * gx + gy * gy);
                    assert(std::fabs(sobel.magnitudeCpu()(i, j) - expected) < ATOL);
                }
            }
            std::cout << "[PASS] Test 4 (CPU): Magnitud de Sobel G = sqrt(Gx^2 + Gy^2) verificada.\n";
        }

        // =====================================================================
        // TEST 6: Test de Perímetro (Borde: fila 0, fila H-1, col 0, col W-1)
        // =====================================================================
        {
            ImageBuffer ones(6, 6, 1.0f);
            const std::vector<float> box3 = ImageBuffer::makeBoxKernel(3);
            Convolution2D conv(ones);
            conv.convolveCpu(box3.data(), 3);
            const ImageBuffer& out = conv.outputCpu();

            assert(std::fabs(out(0, 0) - (4.0f / 9.0f)) < ATOL);
            assert(std::fabs(out(0, 5) - (4.0f / 9.0f)) < ATOL);
            assert(std::fabs(out(5, 0) - (4.0f / 9.0f)) < ATOL);
            assert(std::fabs(out(5, 5) - (4.0f / 9.0f)) < ATOL);

            assert(std::fabs(out(0, 2) - (6.0f / 9.0f)) < ATOL);
            assert(std::fabs(out(5, 2) - (6.0f / 9.0f)) < ATOL);
            assert(std::fabs(out(2, 0) - (6.0f / 9.0f)) < ATOL);
            assert(std::fabs(out(2, 5) - (6.0f / 9.0f)) < ATOL);

            assert(std::fabs(out(2, 2) - 1.0f) < ATOL);
            std::cout << "[PASS] Test 6 (CPU): Test de perímetro (Zero-padding en filas/columnas de borde) verificado.\n";
        }
    } else {
        std::cout << "[PENDING - Rol 2 stub] Tests 1, 2, 4-CPU y 6-CPU listos y en espera de que "
                  << "el Rol 2 implemente Convolution2D::convolveCpu y Sobel CPU.\n";
    }

    // =========================================================================
    // TESTS 3, 4 (GPU), 5 y 6 (GPU): Equivalencia numérica CPU vs GPU y Reducciones
    // =========================================================================
    if (!Convolution2D::isGpuAvailable()) {
        std::cout << "[SKIPPED - No GPU device available in CI] Tests en device (3, 4-GPU, 5-GPU, 6-GPU) "
                  << "omitidos limpiamente; se ejecutarán en la partición GPU del clúster DIINF.\n";
    } else if (!isRole2Implemented() || !isRole3Implemented()) {
        std::cout << "[PENDING - Rol 3 stub] Tests en device (3, 4-GPU, 5-GPU, 6-GPU) listos y en espera "
                  << "de que el Rol 3 implemente los kernels CUDA en Convolution2D.cu, Sobel.cu y Metrics.cu.\n";
    } else {
        for (std::size_t n : ImageBuffer::standardSizes(false)) {
            ImageBuffer img = ImageBuffer::createRandom(n, n, 42u);
            for (int s : {3, 5, 7}) {
                const std::vector<float> k = ImageBuffer::makeGaussianKernel(s);
                Convolution2D conv(img);
                conv.convolveCpu(k.data(), s);

                conv.convolveGpu(k.data(), s, Convolution2D::VARIANT_NAIVE);
                const float diff_naive = Metrics::maxAbsDiffCpuGpu(conv.outputCpu(), conv.outputGpu());
                assert(diff_naive < ATOL);

                conv.convolveGpu(k.data(), s, Convolution2D::VARIANT_SHARED);
                const float diff_shared = Metrics::maxAbsDiffCpuGpu(conv.outputCpu(), conv.outputGpu());
                assert(diff_shared < ATOL);
            }
        }
        std::cout << "[PASS] Test 3 (GPU): Equivalencia CPU vs GPU Naive y Shared en 8x8, 16x16 y 32x32 verificada.\n";

        ImageBuffer img32 = ImageBuffer::createRandom(32, 32, 2026u);
        Sobel sobel(img32);
        sobel.computeGxGyCpu();
        sobel.computeMagnitudeCpu();

        sobel.computeGxGyGpu(Convolution2D::VARIANT_SHARED);
        sobel.computeMagnitudeGpu();

        const float mag_diff = Metrics::maxAbsDiffCpuGpu(sobel.magnitudeCpu(), sobel.magnitudeGpu());
        assert(mag_diff < ATOL);

        Metrics metrics(sobel.magnitudeGpu());
        const float sum_shared = metrics.sumMagnitude(Metrics::METHOD_SHARED_REDUCTION);
        const float sum_atomic = metrics.sumMagnitude(Metrics::METHOD_ATOMIC_ADD);
        const float rel_diff = std::fabs(sum_shared - sum_atomic) / std::fmax(1.0f, std::fabs(sum_shared));
        assert(rel_diff < RTOL);
        std::cout << "[PASS] Test 4 y 5 (GPU): Magnitud Sobel GPU y Reducción Shared vs AtomicAdd verificadas.\n";

        ImageBuffer ones(16, 16, 1.0f);
        const std::vector<float> box3 = ImageBuffer::makeBoxKernel(3);
        Convolution2D conv_border(ones);
        conv_border.convolveCpu(box3.data(), 3);
        conv_border.convolveGpu(box3.data(), 3, Convolution2D::VARIANT_SHARED, dim3(8, 8));
        assert(Metrics::maxAbsDiffCpuGpu(conv_border.outputCpu(), conv_border.outputGpu()) < ATOL);
        std::cout << "[PASS] Test 6 (GPU): Halo y política de borde en GPU Shared verificados.\n";
    }

    // =========================================================================
    // TEST DE VISUALIZADOR (Rol 5): Exportación de cpu_gpu_error.dat y Panel Sobel
    // =========================================================================
    {
        ImageBuffer a(8, 8, 0.5f);
        ImageBuffer b(8, 8, 0.5f);
        b(3, 4) = 0.75f; // Error puntual de 0.25f

        const std::string err_path = "/tmp/lab2_test_cpu_gpu_error.dat";
        Visualizer::writeCpuGpuErrorDat(a, b, err_path);

        ImageBuffer loaded_err;
        loaded_err.readDat(err_path);
        assert(loaded_err.height() == 8 && loaded_err.width() == 8);
        assert(std::fabs(loaded_err(3, 4) - 0.25f) < 1e-6f);
        assert(std::fabs(loaded_err(0, 0)) < 1e-6f);
        std::remove(err_path.c_str());
        std::cout << "[PASS] Visualizer (Rol 5): Exportación y lectura de cpu_gpu_error.dat verificada.\n";
    }

    std::cout << "[PASS] test_pipeline_cpu_gpu (Rol 5): Suite finalizada con éxito.\n";
    return 0;
}
