#include "ImageBuffer.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace {

/**
 * @brief Ejecuta `fn`, verifica que lance una excepción del tipo `ExceptionT` y que su mensaje
 *        (`e.what()`) contenga la subcadena `expected_cause`, imprimiendo la causa reconocida.
 */
template <typename ExceptionT>
void expectInvalidDataError(const std::string& test_case,
                            const std::string& expected_cause,
                            const std::function<void()>& fn) {
    bool caught = false;
    try {
        fn();
    } catch (const ExceptionT& ex) {
        caught = true;
        const std::string msg = ex.what();
        if (msg.find(expected_cause) == std::string::npos) {
            throw std::runtime_error("Caso '" + test_case + "': se lanzó excepción pero el mensaje '" +
                                     msg + "' no contiene la causa esperada '" + expected_cause + "'");
        }
        std::cout << "  [OK Error Reconocido] " << test_case << " -> " << msg << '\n';
    }
    if (!caught) {
        throw std::runtime_error("Caso '" + test_case +
                                 "': NO lanzó la excepción esperada ante datos inválidos");
    }
}

} // namespace

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
 * 6. Detección exhaustiva de datos no válidos y verificación del mensaje de error con su causa.
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
        assert(std::fabs(orig.data()[idx] - loaded_p5.data()[idx]) <= (1.0f / 255.0f + 1e-4f));
        assert(std::fabs(loaded_p5.data()[idx] - loaded_p2.data()[idx]) < 1e-6f);
    }
    std::remove(tmp_pgm_p5.c_str());
    std::remove(tmp_pgm_p2.c_str());

    // 7. Kernels de Blur normalizados (s = 3, 5, 7) y sigma positivo muy pequeño
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

    for (float tiny_sigma : {1e-5f, 1e-20f, std::numeric_limits<float>::min()}) {
        const auto k_tiny = ImageBuffer::makeGaussianKernel(5, tiny_sigma);
        assert(k_tiny.size() == 25);
        for (float v : k_tiny) {
            assert(std::isfinite(v));
        }
        assert(std::fabs(k_tiny[12] - 1.0f) < 1e-5f);
    }

    // =========================================================================
    // 8. SUITE DE PRUEBAS PARA DATOS NO VÁLIDOS (Reconocimiento y Mensaje de Causa)
    // =========================================================================
    std::cout << "[INFO] Ejecutando pruebas de detección de datos no válidos para Rol 1...\n";

    const float nan_val = std::numeric_limits<float>::quiet_NaN();
    const float inf_val = std::numeric_limits<float>::infinity();

    // 8.1. Desbordamiento en H * W (Constructor y resize) y consistencia de dimensiones
    ImageBuffer consistent(4, 5, 1.0f);
    expectInvalidDataError<std::overflow_error>(
        "resize con desbordamiento H*W",
        "desbordamiento al calcular H * W",
        [&]() { consistent.resize(std::numeric_limits<std::size_t>::max(), 2, 0.0f); });
    // El estado previo debe mantenerse intacto
    assert(consistent.height() == 4 && consistent.width() == 5 && consistent.size() == 20);
    assert(consistent(0, 0) == 1.0f);

    expectInvalidDataError<std::overflow_error>(
        "constructor con desbordamiento H*W",
        "desbordamiento al calcular H * W",
        []() { ImageBuffer bad(std::numeric_limits<std::size_t>::max(), 4); });

    // Si una dimensión es 0, ambas deben normalizarse a 0x0 sin inconsistencia
    consistent.resize(0, 10);
    assert(consistent.empty() && consistent.height() == 0 && consistent.width() == 0 && consistent.size() == 0);
    ImageBuffer zero_width(10, 0);
    assert(zero_width.empty() && zero_width.height() == 0 && zero_width.width() == 0 && zero_width.size() == 0);

    // 8.2. Accesos fuera de rango en at() (lectura y escritura)
    expectInvalidDataError<std::out_of_range>(
        "at(i, j) con fila fuera de rango",
        "fuera de rango",
        [&]() { (void)grid.at(6, 0); });

    expectInvalidDataError<std::out_of_range>(
        "at(i, j) const con columna fuera de rango",
        "fuera de rango",
        [&]() {
            const ImageBuffer& cgrid = grid;
            (void)cgrid.at(0, 4);
        });

    // 8.3. Valores no finitos (NaN / Inf) o fuera de límites en generadores y mutadores
    expectInvalidDataError<std::invalid_argument>(
        "resize con valor inicial NaN",
        "valor finito",
        [&]() { grid.resize(4, 4, nan_val); });

    expectInvalidDataError<std::invalid_argument>(
        "fill con valor Inf",
        "valor finito",
        [&]() { grid.fill(inf_val); });

    expectInvalidDataError<std::out_of_range>(
        "fillImpulse fuera de límites",
        "fuera de los límites",
        [&]() { grid.fillImpulse(10, 10, 1.0f); });

    expectInvalidDataError<std::invalid_argument>(
        "fillImpulse con amplitud NaN",
        "valor finito",
        [&]() { grid.fillImpulse(1, 1, nan_val); });

    expectInvalidDataError<std::runtime_error>(
        "fillCenterImpulse en buffer vacío",
        "buffer está vacío",
        [&]() { empty_buf.fillCenterImpulse(1.0f); });

    expectInvalidDataError<std::out_of_range>(
        "fillVerticalStep con split_col > width",
        "split_col excede el ancho W",
        [&]() { grid.fillVerticalStep(0.0f, 1.0f, 99); });

    expectInvalidDataError<std::out_of_range>(
        "fillHorizontalStep con split_row > height",
        "split_row excede la altura H",
        [&]() { grid.fillHorizontalStep(0.0f, 1.0f, 99); });

    expectInvalidDataError<std::invalid_argument>(
        "fillRandom con min_val > max_val",
        "min_val no puede ser mayor que max_val",
        [&]() { grid.fillRandom(42u, 2.0f, 1.0f); });

    expectInvalidDataError<std::invalid_argument>(
        "fillRandom con límite NaN",
        "valor finito",
        [&]() { grid.fillRandom(42u, nan_val, 1.0f); });

    // 8.4. Parámetros inválidos en generadores de kernels (makeBoxKernel y makeGaussianKernel)
    for (int bad_s : {0, -3, 2, 4}) {
        expectInvalidDataError<std::invalid_argument>(
            "makeBoxKernel con lado s inválido (" + std::to_string(bad_s) + ")",
            "impar y positivo",
            [bad_s]() { (void)ImageBuffer::makeBoxKernel(bad_s); });

        expectInvalidDataError<std::invalid_argument>(
            "makeGaussianKernel con lado s inválido (" + std::to_string(bad_s) + ")",
            "impar y positivo",
            [bad_s]() { (void)ImageBuffer::makeGaussianKernel(bad_s, 1.0f); });
    }

    expectInvalidDataError<std::invalid_argument>(
        "makeGaussianKernel con sigma negativo",
        "sigma debe ser finito y >= 0",
        []() { (void)ImageBuffer::makeGaussianKernel(3, -0.5f); });

    expectInvalidDataError<std::invalid_argument>(
        "makeGaussianKernel con sigma NaN",
        "sigma debe ser finito y >= 0",
        [nan_val]() { (void)ImageBuffer::makeGaussianKernel(3, nan_val); });

    // 8.5. Archivos PGM con datos no válidos (formato, cabecera, muestras fuera de rango y truncamiento)
    ImageBuffer img_guard(3, 3, 0.5f);

    expectInvalidDataError<std::runtime_error>(
        "readPgm archivo inexistente",
        "no se pudo abrir el archivo",
        [&]() { img_guard.readPgm("/tmp/no_existe_archivo_lab2.pgm"); });

    // Formato mágico inválido (ej. P6 en vez de P2/P5)
    const std::string bad_magic_pgm = "/tmp/lab2_bad_magic.pgm";
    {
        std::ofstream out(bad_magic_pgm);
        out << "P6\n2 2\n255\n0 0 0 0\n";
    }
    expectInvalidDataError<std::runtime_error>(
        "readPgm con magic P6 no soportado",
        "formato PGM no soportado",
        [&]() { img_guard.readPgm(bad_magic_pgm); });
    std::remove(bad_magic_pgm.c_str());

    // Dimensiones negativas o cero en cabecera PGM
    const std::string bad_dim_pgm = "/tmp/lab2_bad_dim.pgm";
    {
        std::ofstream out(bad_dim_pgm);
        out << "P2\n0 4\n255\n";
    }
    expectInvalidDataError<std::runtime_error>(
        "readPgm con ancho 0 en cabecera",
        "ancho (width) inválido o no positivo",
        [&]() { img_guard.readPgm(bad_dim_pgm); });
    std::remove(bad_dim_pgm.c_str());

    // maxval inválido (> 65535 o <= 0)
    const std::string bad_maxval_pgm = "/tmp/lab2_bad_maxval.pgm";
    {
        std::ofstream out(bad_maxval_pgm);
        out << "P2\n2 2\n70000\n0 0 0 0\n";
    }
    expectInvalidDataError<std::runtime_error>(
        "readPgm con maxval > 65535",
        "maxval inválido",
        [&]() { img_guard.readPgm(bad_maxval_pgm); });
    std::remove(bad_maxval_pgm.c_str());

    // Muestra P2 mayor que maxval
    const std::string over_p2_pgm = "/tmp/lab2_over_p2.pgm";
    {
        std::ofstream out(over_p2_pgm);
        out << "P2\n2 2\n100\n0 50 101 20\n";
    }
    expectInvalidDataError<std::runtime_error>(
        "readPgm P2 con muestra > maxval",
        "fuera de rango [0, 100]",
        [&]() { img_guard.readPgm(over_p2_pgm); });
    std::remove(over_p2_pgm.c_str());

    // Muestra P2 negativa
    const std::string neg_p2_pgm = "/tmp/lab2_neg_p2.pgm";
    {
        std::ofstream out(neg_p2_pgm);
        out << "P2\n2 2\n255\n0 -5 10 20\n";
    }
    expectInvalidDataError<std::runtime_error>(
        "readPgm P2 con muestra negativa",
        "fuera de rango [0, 255]",
        [&]() { img_guard.readPgm(neg_p2_pgm); });
    std::remove(neg_p2_pgm.c_str());

    // Muestra P5 (binario 8-bit) mayor que maxval
    const std::string over_p5_pgm = "/tmp/lab2_over_p5.pgm";
    {
        std::ofstream out(over_p5_pgm, std::ios::binary);
        out << "P5\n2 2\n100\n";
        const unsigned char raw[4] = {0, 50, 200, 20};
        out.write(reinterpret_cast<const char*>(raw), 4);
    }
    expectInvalidDataError<std::runtime_error>(
        "readPgm P5 con byte > maxval",
        "muestra P5 (8-bit) excede maxval",
        [&]() { img_guard.readPgm(over_p5_pgm); });
    std::remove(over_p5_pgm.c_str());

    // Archivo P5 truncado
    const std::string trunc_p5_pgm = "/tmp/lab2_trunc_p5.pgm";
    {
        std::ofstream out(trunc_p5_pgm, std::ios::binary);
        out << "P5\n4 4\n255\n";
        const unsigned char raw[3] = {10, 20, 30}; // Faltan 13 bytes
        out.write(reinterpret_cast<const char*>(raw), 3);
    }
    expectInvalidDataError<std::runtime_error>(
        "readPgm P5 con datos truncados",
        "datos P5 (8-bit) truncados",
        [&]() { img_guard.readPgm(trunc_p5_pgm); });
    std::remove(trunc_p5_pgm.c_str());

    // Archivo PGM con comentario antes de magic P2
    const std::string comment_before_magic_pgm = "/tmp/lab2_comment_magic.pgm";
    {
        std::ofstream out(comment_before_magic_pgm);
        out << "# Comentario al inicio\nP2\n2 2\n255\n10 20 30 40\n";
    }
    ImageBuffer img_comment_magic;
    img_comment_magic.readPgm(comment_before_magic_pgm);
    assert(img_comment_magic.height() == 2 && img_comment_magic.width() == 2);
    std::remove(comment_before_magic_pgm.c_str());

    // Archivo P2 con datos espurios o sobrantes al final
    const std::string trailing_p2_pgm = "/tmp/lab2_trailing_p2.pgm";
    {
        std::ofstream out(trailing_p2_pgm);
        out << "P2\n2 2\n255\n10 20 30 40\n999_extra\n";
    }
    expectInvalidDataError<std::runtime_error>(
        "readPgm P2 con datos extra al final",
        "datos extra o espurios",
        [&]() { img_guard.readPgm(trailing_p2_pgm); });
    std::remove(trailing_p2_pgm.c_str());

    // Archivo P5 con datos espurios o sobrantes al final
    const std::string trailing_p5_pgm = "/tmp/lab2_trailing_p5.pgm";
    {
        std::ofstream out(trailing_p5_pgm, std::ios::binary);
        out << "P5\n2 2\n255\n";
        const unsigned char raw[5] = {10, 20, 30, 40, 50}; // 5 bytes para 2x2 = 4 píxeles
        out.write(reinterpret_cast<const char*>(raw), 5);
    }
    expectInvalidDataError<std::runtime_error>(
        "readPgm P5 con datos extra al final",
        "datos extra o espurios",
        [&]() { img_guard.readPgm(trailing_p5_pgm); });
    std::remove(trailing_p5_pgm.c_str());

    // 8.6. Archivos .dat con datos no válidos y escritura de imagen vacía o con valores no finitos
    expectInvalidDataError<std::runtime_error>(
        "writePgm sobre imagen vacía",
        "no se puede escribir una imagen vacía",
        [&]() { empty_buf.writePgm("/tmp/lab2_empty.pgm"); });

    expectInvalidDataError<std::runtime_error>(
        "writeDat sobre imagen vacía",
        "no se puede escribir una imagen vacía",
        [&]() { empty_buf.writeDat("/tmp/lab2_empty.dat"); });

    // writePgm y writeDat con valores NaN / Inf inyectados en el buffer
    {
        ImageBuffer img_nan(2, 2, 1.0f);
        img_nan.data()[1] = nan_val;
        expectInvalidDataError<std::runtime_error>(
            "writePgm con NaN en buffer",
            "valores no finitos",
            [&]() { img_nan.writePgm("/tmp/lab2_nan.pgm"); });

        expectInvalidDataError<std::runtime_error>(
            "writeDat con NaN en buffer",
            "valores no finitos",
            [&]() { img_nan.writeDat("/tmp/lab2_nan.dat"); });
    }

    const std::string bad_header_dat = "/tmp/lab2_bad_header.dat";
    {
        std::ofstream out(bad_header_dat);
        out << "# wrong_header 4 4\n1.0 2.0\n";
    }
    expectInvalidDataError<std::runtime_error>(
        "readDat con cabecera mal formada",
        "cabecera inválida",
        [&]() { img_guard.readDat(bad_header_dat); });
    std::remove(bad_header_dat.c_str());

    const std::string corrupt_dat = "/tmp/lab2_corrupt_data.dat";
    {
        std::ofstream out(corrupt_dat);
        out << "# height 2 width 2\n0.5 texto_invalido 0.1 0.2\n";
    }
    expectInvalidDataError<std::runtime_error>(
        "readDat con token no numérico",
        "datos insuficientes, no numéricos o no finitos",
        [&]() { img_guard.readDat(corrupt_dat); });
    std::remove(corrupt_dat.c_str());

    const std::string trailing_dat = "/tmp/lab2_trailing_data.dat";
    {
        std::ofstream out(trailing_dat);
        out << "# height 2 width 2\n0.1 0.2 0.3 0.4\n999.0\n";
    }
    expectInvalidDataError<std::runtime_error>(
        "readDat con datos extra al final",
        "datos extra o espurios",
        [&]() { img_guard.readDat(trailing_dat); });
    std::remove(trailing_dat.c_str());

    // Verificar que tras todos los intentos fallidos de lectura, img_guard sigue intacto (3x3)
    assert(img_guard.height() == 3 && img_guard.width() == 3 && img_guard(0, 0) == 0.5f);

    std::cout << "[PASS] test_image_buffer (Rol 1): Todas las pruebas (válidas e inválidas) pasaron correctamente.\n";
    return 0;
}
