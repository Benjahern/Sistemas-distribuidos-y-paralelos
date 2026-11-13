#include "ImageBuffer.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>

/**
 * @file test_image_buffer.cpp
 * @brief Suite de pruebas unitarias para el Rol 1 (`ImageBuffer.h/.cpp`).
 *
 * Verifica:
 * 1. Indexación en orden fila-mayor (row-major `i * W + j`) y chequeo de bordes en `at(i, j)`.
 * 2. Generadores sintéticos deterministas: impulso unitario, escalón vertical y escalón horizontal.
 * 3. Generación de ruido pseudoaleatorio reproducible con semilla fija para todos los tamaños
 *    solicitados (tests: 8x8, 16x16, 32x32; benchmarks: 512x512, 1024x1024, 2048x2048, 4096x4096).
 * 4. Lectura y escritura sin pérdida en formato `.dat` y en formato PGM (P2 ASCII y P5 binario).
 * 5. Generación de kernels de Blur normalizados (`sum k = 1`) de lado impar `s in {3, 5, 7}`.
 */
int main() {
    // 1. Constructor por defecto y constructor con dimensiones e inicialización
    ImageBuffer empty_buf;
    assert(empty_buf.empty());
    assert(empty_buf.height() == 0 && empty_buf.width() == 0 && empty_buf.size() == 0);

    ImageBuffer grid(6, 4, 0.25f);
    assert(!grid.empty());
    assert(grid.height() == 6 && grid.width() == 4 && grid.size() == 24);
    assert(std::fabs(grid.at(5, 3) - 0.25f) < 1e-6f);

    // Verificar orden row-major estricto: data()[i * W + j] == at(i, j)
    for (std::size_t i = 0; i < grid.height(); ++i) {
        for (std::size_t j = 0; j < grid.width(); ++j) {
            grid.at(i, j) = static_cast<float>(10 * i + j);
        }
    }
    for (std::size_t i = 0; i < grid.height(); ++i) {
        for (std::size_t j = 0; j < grid.width(); ++j) {
            assert(grid.data()[i * grid.width() + j] == static_cast<float>(10 * i + j));
            assert(grid.row(i)[j] == grid(i, j));
        }
    }

    // Acceso fuera de rango debe lanzar std::out_of_range
    bool threw_out_of_range = false;
    try {
        (void)grid.at(6, 0);
    } catch (const std::out_of_range&) {
        threw_out_of_range = true;
    }
    assert(threw_out_of_range);

    // 2. Generador de impulso unitario centrado (5x5)
    ImageBuffer imp(5, 5);
    imp.fillCenterImpulse(1.0f);
    for (std::size_t i = 0; i < 5; ++i) {
        for (std::size_t j = 0; j < 5; ++j) {
            const float expected = (i == 2 && j == 2) ? 1.0f : 0.0f;
            assert(imp(i, j) == expected);
        }
    }

    // 3. Escalón vertical y escalón horizontal
    ImageBuffer vstep(8, 8);
    vstep.fillVerticalStep(0.0f, 1.0f);
    for (std::size_t i = 0; i < 8; ++i) {
        for (std::size_t j = 0; j < 8; ++j) {
            assert(vstep(i, j) == (j < 4 ? 0.0f : 1.0f));
        }
    }

    ImageBuffer hstep(8, 8);
    hstep.fillHorizontalStep(0.0f, 1.0f);
    for (std::size_t i = 0; i < 8; ++i) {
        for (std::size_t j = 0; j < 8; ++j) {
            assert(hstep(i, j) == (i < 4 ? 0.0f : 1.0f));
        }
    }

    // 4. Ruido reproducible para TODOS los tamaños solicitados en el enunciado:
    //    Tests: {8, 16, 32}; Benchmarks: {512, 1024, 2048, 4096}
    const std::vector<std::size_t> all_sizes = ImageBuffer::standardSizes(true);
    assert(all_sizes.size() == 7);

    // Verificar determinismo exacto con la misma semilla y diferencia con semilla distinta
    for (std::size_t n : {std::size_t(8), std::size_t(16), std::size_t(32), std::size_t(512)}) {
        ImageBuffer r1 = ImageBuffer::createRandom(n, n, 12345u);
        ImageBuffer r2 = ImageBuffer::createRandom(n, n, 12345u);
        ImageBuffer r3 = ImageBuffer::createRandom(n, n, 54321u);
        assert(r1.height() == n && r1.width() == n);
        bool same_seed_equal = true;
        bool diff_seed_diff  = false;
        for (std::size_t idx = 0; idx < r1.size(); ++idx) {
            if (r1.data()[idx] != r2.data()[idx]) same_seed_equal = false;
            if (r1.data()[idx] != r3.data()[idx]) diff_seed_diff = true;
            assert(r1.data()[idx] >= 0.0f && r1.data()[idx] <= 1.0f);
        }
        assert(same_seed_equal);
        assert(diff_seed_diff);
    }

    // Verificar también la generación en los tamaños grandes de benchmark (1024, 2048, 4096)
    for (std::size_t n : {std::size_t(1024), std::size_t(2048), std::size_t(4096)}) {
        ImageBuffer large = ImageBuffer::createRandom(n, n, 42u);
        assert(large.height() == n && large.width() == n && large.size() == n * n);
        assert(large(0, 0) >= 0.0f && large(0, 0) <= 1.0f);
        assert(large(n - 1, n - 1) >= 0.0f && large(n - 1, n - 1) <= 1.0f);
    }

    // 5. Ida y vuelta en formato .dat (precisión float)
    const std::string tmp_dat = "/tmp/lab2_test_imagebuffer.dat";
    ImageBuffer orig = ImageBuffer::createRandom(16, 16, 77u);
    orig.writeDat(tmp_dat);
    ImageBuffer loaded_dat;
    loaded_dat.readDat(tmp_dat);
    assert(loaded_dat.height() == 16 && loaded_dat.width() == 16);
    for (std::size_t idx = 0; idx < orig.size(); ++idx) {
        assert(std::fabs(orig.data()[idx] - loaded_dat.data()[idx]) < 1e-6f);
    }
    std::remove(tmp_dat.c_str());

    // 6. Ida y vuelta en formato PGM (P5 binario y P2 ASCII)
    const std::string tmp_pgm_p5 = "/tmp/lab2_test_p5.pgm";
    const std::string tmp_pgm_p2 = "/tmp/lab2_test_p2.pgm";
    orig.writePgm(tmp_pgm_p5, true, false);
    orig.writePgm(tmp_pgm_p2, false, false);

    ImageBuffer loaded_p5, loaded_p2;
    loaded_p5.readPgm(tmp_pgm_p5);
    loaded_p2.readPgm(tmp_pgm_p2);
    assert(loaded_p5.height() == 16 && loaded_p5.width() == 16);
    assert(loaded_p2.height() == 16 && loaded_p2.width() == 16);
    for (std::size_t idx = 0; idx < orig.size(); ++idx) {
        // Cuantización a 8 bits [0..255] tiene paso máximo 1 / 255 ~= 0.00392
        assert(std::fabs(orig.data()[idx] - loaded_p5.data()[idx]) <= (1.0f / 255.0f + 1e-4f));
        assert(std::fabs(loaded_p5.data()[idx] - loaded_p2.data()[idx]) < 1e-6f);
    }
    std::remove(tmp_pgm_p5.c_str());
    std::remove(tmp_pgm_p2.c_str());

    // 7. Kernels de Blur normalizados (s = 3, 5, 7) deben sumar exactamente 1.0f
    for (int s : {3, 5, 7}) {
        const auto k_box = ImageBuffer::makeBoxKernel(s);
        const auto k_gauss = ImageBuffer::makeGaussianKernel(s);
        assert(k_box.size() == static_cast<std::size_t>(s) * static_cast<std::size_t>(s));
        assert(k_gauss.size() == static_cast<std::size_t>(s) * static_cast<std::size_t>(s));
        const float sum_box = std::accumulate(k_box.begin(), k_box.end(), 0.0f);
        const float sum_gauss = std::accumulate(k_gauss.begin(), k_gauss.end(), 0.0f);
        assert(std::fabs(sum_box - 1.0f) < 1e-5f);
        assert(std::fabs(sum_gauss - 1.0f) < 1e-5f);
    }

    // 8. Verificación de correcciones del Rol 1:
    // 8a. Desbordamiento en H * W (constructor y resize) y consistencia de dimensiones
    {
        ImageBuffer consistent(4, 5, 1.0f);
        bool overflow_threw = false;
        try {
            consistent.resize(std::numeric_limits<std::size_t>::max(), 2, 0.0f);
        } catch (const std::overflow_error&) {
            overflow_threw = true;
        }
        assert(overflow_threw);
        // Las dimensiones y datos previos deben permanecer intactos tras el fallo
        assert(consistent.height() == 4 && consistent.width() == 5 && consistent.size() == 20);
        assert(consistent(0, 0) == 1.0f);

        // Si una sola dimensión es 0 (ej. resize(0, 10) o resize(10, 0)), ambas quedan en 0x0
        consistent.resize(0, 10);
        assert(consistent.empty() && consistent.height() == 0 && consistent.width() == 0 && consistent.size() == 0);

        ImageBuffer zero_width(10, 0);
        assert(zero_width.empty() && zero_width.height() == 0 && zero_width.width() == 0 && zero_width.size() == 0);
    }

    // 8b. Rechazo de muestras fuera de rango [0, maxval] en PGM (P2 y P5)
    {
        const std::string bad_p2 = "/tmp/lab2_test_bad_p2.pgm";
        {
            std::ofstream out(bad_p2);
            out << "P2\n2 2\n100\n0 50 101 20\n"; // 101 > maxval(100)
        }
        ImageBuffer img_keep(3, 3, 0.5f);
        bool bad_p2_threw = false;
        try {
            img_keep.readPgm(bad_p2);
        } catch (const std::runtime_error&) {
            bad_p2_threw = true;
        }
        assert(bad_p2_threw);
        assert(img_keep.height() == 3 && img_keep.width() == 3); // Intacto
        std::remove(bad_p2.c_str());

        const std::string neg_p2 = "/tmp/lab2_test_neg_p2.pgm";
        {
            std::ofstream out(neg_p2);
            out << "P2\n2 2\n255\n0 -1 10 20\n"; // -1 < 0
        }
        bool neg_p2_threw = false;
        try {
            img_keep.readPgm(neg_p2);
        } catch (const std::runtime_error&) {
            neg_p2_threw = true;
        }
        assert(neg_p2_threw);
        std::remove(neg_p2.c_str());

        const std::string bad_p5 = "/tmp/lab2_test_bad_p5.pgm";
        {
            std::ofstream out(bad_p5, std::ios::binary);
            out << "P5\n2 2\n100\n";
            const unsigned char raw[4] = {0, 50, 200, 20}; // 200 > maxval(100)
            out.write(reinterpret_cast<const char*>(raw), 4);
        }
        bool bad_p5_threw = false;
        try {
            img_keep.readPgm(bad_p5);
        } catch (const std::runtime_error&) {
            bad_p5_threw = true;
        }
        assert(bad_p5_threw);
        std::remove(bad_p5.c_str());
    }

    // 8c. Kernel gaussiano con sigma positivo muy pequeño (sin producir NaN) y rechazo de sigma negativo
    {
        for (float tiny_sigma : {1e-5f, 1e-20f, std::numeric_limits<float>::min()}) {
            const auto k_tiny = ImageBuffer::makeGaussianKernel(5, tiny_sigma);
            assert(k_tiny.size() == 25);
            for (float v : k_tiny) {
                assert(std::isfinite(v));
            }
            // Con sigma -> 0+, el peso se concentra en el coeficiente central (2, 2) -> índice 12
            assert(std::fabs(k_tiny[12] - 1.0f) < 1e-5f);
        }

        bool neg_sigma_threw = false;
        try {
            (void)ImageBuffer::makeGaussianKernel(3, -0.5f);
        } catch (const std::invalid_argument&) {
            neg_sigma_threw = true;
        }
        assert(neg_sigma_threw);
    }

    std::cout << "[PASS] test_image_buffer (Rol 1): Todas las pruebas pasaron correctamente.\n";
    return 0;
}

