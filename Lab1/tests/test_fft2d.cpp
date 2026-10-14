#include "Fft2D.h"
#include "ComplexField.h"

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
// Los asserts deben ejecutarse incluso si el compilador recibe -DNDEBUG.
// Si una condición falla, la suite termina con error en lugar de anunciar éxito.
#undef NDEBUG
#include <cassert>
#include <random>
#include <utility>
#include <omp.h>

// std::abs(error_complejo) mide la distancia entre dos valores complejos.
// Se permite un error pequeño de redondeo, en vez de exigir igualdad exacta.
constexpr double TOLERANCE = 1e-10;

// ============================================================================
// DFT 2D Ingenua (Referencia Matemática Independiente O(M^2 * N^2))
// Evalúa directamente la Ecuación (1) del enunciado sin usar mariposas.
// ============================================================================

ComplexField naiveDft2D(const ComplexField& x) {
    // Calcula cada frecuencia (k,l) sumando TODAS las muestras (m,n).
    // Es lenta, pero independiente de la FFT: detecta errores que una ida y
    // vuelta podría ocultar si directa e inversa compartieran el mismo defecto.
    const size_t M = x.rows();
    const size_t N = x.cols();
    const double pi = 3.14159265358979323846;
    ComplexField X(M, N);

    for (size_t k = 0; k < M; ++k) {
        for (size_t l = 0; l < N; ++l) {
            std::complex<double> acc(0.0, 0.0);
            for (size_t m = 0; m < M; ++m) {
                for (size_t n = 0; n < N; ++n) {
                    double angle = -2.0 * pi * (static_cast<double>(k * m) / static_cast<double>(M) +
                                               static_cast<double>(l * n) / static_cast<double>(N));
                    acc += x.at(m, n) * std::polar(1.0, angle);
                }
            }
            X.at(k, l) = acc;
        }
    }
    return X;
}

// ============================================================================
// SUITE DE PRUEBAS DE ALTA PRECISIÓN (ROL 3)
// ============================================================================

void testImpulse2x2(int layout) {
    std::cout << "[Test 1] Impulso 2x2 analítico y su IFFT... ";
    ComplexField field(2, 2);
    field.at(0, 0) = std::complex<double>(1.0, 0.0);

    Fft2D fft;
    fft.forward(field, layout);

    // FFT de impulso 2x2 es idénticamente 1 en todas las frecuencias
    for (size_t r = 0; r < 2; ++r) {
        for (size_t c = 0; c < 2; ++c) {
            assert(std::abs(field.at(r, c) - std::complex<double>(1.0, 0.0)) < TOLERANCE);
        }
    }

    fft.inverse(field, layout);
    assert(std::abs(field.at(0, 0) - std::complex<double>(1.0, 0.0)) < TOLERANCE);
    assert(std::abs(field.at(0, 1)) < TOLERANCE);
    assert(std::abs(field.at(1, 0)) < TOLERANCE);
    assert(std::abs(field.at(1, 1)) < TOLERANCE);
    std::cout << "PASÓ OK!" << std::endl;
}

void testCompareAgainstNaiveDft() {
    // Probar ambos órdenes rectangulares descubre si se confunden M y N
    // al seleccionar la longitud de las filas o reservar el buffer de columnas.
    std::cout << "[Test 2] Comparación contra DFT 2D ingenua (Ec. 1) en 8x4 Y 4x8... ";
    const std::vector<std::pair<size_t, size_t>> dimensions = {{8, 4}, {4, 8}};
    Fft2D fft;

    for (const auto& dim : dimensions) {
        const size_t M = dim.first;
        const size_t N = dim.second;
        ComplexField original(M, N);
        for (size_t r = 0; r < M; ++r) {
            for (size_t c = 0; c < N; ++c) {
                original.at(r, c) = std::complex<double>(std::sin(r * 0.8), std::cos(c * 1.2));
            }
        }

        ComplexField naive_result = naiveDft2D(original);

        ComplexField fft_inplace = original;
        ComplexField fft_stockham = original;
        fft.forward(fft_inplace, Fft2D::LAYOUT_INPLACE);
        fft.forward(fft_stockham, Fft2D::LAYOUT_STOCKHAM);

        for (size_t r = 0; r < M; ++r) {
            for (size_t c = 0; c < N; ++c) {
                double err_ip = std::abs(fft_inplace.at(r, c) - naive_result.at(r, c));
                double err_st = std::abs(fft_stockham.at(r, c) - naive_result.at(r, c));
                assert(err_ip < TOLERANCE);
                assert(err_st < TOLERANCE);
            }
        }
    }
    std::cout << "PASÓ OK!" << std::endl;
}

void testComplexExponentialSign(int layout) {
    std::cout << "[Test 3] Exponencial compleja pura (verificación estricta de signo twiddle)... ";
    const size_t M = 16;
    const size_t N = 16;
    const size_t k0 = 3;
    const size_t l0 = 5;
    const double pi = 3.14159265358979323846;

    // x[m, n] = e^{+2*pi*i*(k0*m/M + l0*n/N)}
    // En la FFT directa con exp(-2*pi*i*...), el producto da exp(0) = 1,
    // produciendo un ÚNICO pico de magnitud exacta M * N en (k0, l0) y cero en el resto.
    ComplexField field(M, N);
    for (size_t r = 0; r < M; ++r) {
        for (size_t c = 0; c < N; ++c) {
            double angle = 2.0 * pi * (static_cast<double>(k0 * r) / static_cast<double>(M) +
                                       static_cast<double>(l0 * c) / static_cast<double>(N));
            field.at(r, c) = std::polar(1.0, angle);
        }
    }

    Fft2D fft;
    fft.forward(field, layout);

    const double expected_peak = static_cast<double>(M * N);
    for (size_t r = 0; r < M; ++r) {
        for (size_t c = 0; c < N; ++c) {
            double mag = std::abs(field.at(r, c));
            if (r == k0 && c == l0) {
                assert(std::abs(mag - expected_peak) < TOLERANCE);
            } else {
                assert(mag < TOLERANCE);
            }
        }
    }
    std::cout << "PASÓ OK! (Pico único de magnitud " << expected_peak << " en (" << k0 << ", " << l0 << "))" << std::endl;
}

void testSineWavePeakAndMagnitude(int layout) {
    std::cout << "[Test 4] Seno 2D: magnitud exacta M*N/2 y bins nulos... ";
    const size_t M = 16;
    const size_t N = 16;
    const size_t k0 = 2;
    const size_t l0 = 3;
    const double pi = 3.14159265358979323846;

    ComplexField field(M, N);
    for (size_t r = 0; r < M; ++r) {
        for (size_t c = 0; c < N; ++c) {
            double angle = 2.0 * pi * (static_cast<double>(k0 * r) / static_cast<double>(M) +
                                       static_cast<double>(l0 * c) / static_cast<double>(N));
            field.at(r, c) = std::complex<double>(std::sin(angle), 0.0);
        }
    }

    Fft2D fft;
    fft.forward(field, layout);

    // Magnitud teórica exacta de cada pico: M * N / 2 = 128
    const double expected_half_peak = 0.5 * static_cast<double>(M * N);
    const size_t k0_conj = M - k0;
    const size_t l0_conj = N - l0;

    for (size_t r = 0; r < M; ++r) {
        for (size_t c = 0; c < N; ++c) {
            double mag = std::abs(field.at(r, c));
            if ((r == k0 && c == l0) || (r == k0_conj && c == l0_conj)) {
                assert(std::abs(mag - expected_half_peak) < TOLERANCE);
            } else {
                assert(mag < TOLERANCE);
            }
        }
    }
    std::cout << "PASÓ OK! (Picos en (" << k0 << "," << l0 << ") y (" << k0_conj << "," << l0_conj << "))" << std::endl;
}

void testParseval(int layout) {
    // La semilla fija hace reproducible el campo aleatorio. Parseval comprueba
    // la escala de la directa sin depender de la implementación de la inversa.
    std::cout << "[Test 5] Teorema de Parseval en grilla 8x8 con semilla fija... ";
    const size_t M = 8;
    const size_t N = 8;
    ComplexField field(M, N);

    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    double energy_spatial = 0.0;
    for (size_t r = 0; r < M; ++r) {
        for (size_t c = 0; c < N; ++c) {
            field.at(r, c) = std::complex<double>(dist(rng), dist(rng));
            energy_spatial += std::norm(field.at(r, c)); // |x_{m,n}|^2
        }
    }

    Fft2D fft;
    fft.forward(field, layout);

    double energy_spectral = 0.0;
    for (size_t r = 0; r < M; ++r) {
        for (size_t c = 0; c < N; ++c) {
            energy_spectral += std::norm(field.at(r, c)); // |X_{k,l}|^2
        }
    }
    energy_spectral *= (1.0 / static_cast<double>(M * N));

    double rel_diff = std::abs(energy_spatial - energy_spectral) / energy_spatial;
    assert(rel_diff < TOLERANCE);
    std::cout << "PASÓ OK! (Diferencia relativa: " << rel_diff << ")" << std::endl;
}

void testRoundtripExhaustive() {
    // Cada combinación parte de una COPIA de la misma entrada original.
    // Así se comparan layouts e hilos sin transformar un resultado anterior.
    std::cout << "[Test 6] Ida y vuelta exhaustiva (dimensiones, layouts e hilos)... ";
    const std::vector<std::pair<size_t, size_t>> test_dims = {{8, 4}, {4, 8}, {2, 2}, {16, 16}};
    const int layouts[] = {Fft2D::LAYOUT_INPLACE, Fft2D::LAYOUT_STOCKHAM};
    const int thread_counts[] = {1, 2, 4, 8};

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(-2.0, 2.0);

    Fft2D fft;

    for (const auto& dim : test_dims) {
        const size_t M = dim.first;
        const size_t N = dim.second;

        ComplexField original(M, N);
        for (size_t r = 0; r < M; ++r) {
            for (size_t c = 0; c < N; ++c) {
                original.at(r, c) = std::complex<double>(dist(rng), dist(rng));
            }
        }

        for (int layout : layouts) {
            for (int threads : thread_counts) {
                omp_set_num_threads(threads);

                ComplexField field = original;
                fft.forward(field, layout);
                fft.inverse(field, layout);

                for (size_t r = 0; r < M; ++r) {
                    for (size_t c = 0; c < N; ++c) {
                        double err = std::abs(field.at(r, c) - original.at(r, c));
                        assert(err < TOLERANCE);
                    }
                }
            }
        }
    }
    std::cout << "PASÓ OK!" << std::endl;
}

void testBoundInstance() {
    // Comprueba la segunda forma de uso: enlazar el campo al construir permite
    // llamar forward()/inverse() sin volver a pasar la grilla como argumento.
    std::cout << "[Test 7] Instancia enlazada Fft2D(field) en 4x8 y 8x4 con ambos layouts... ";
    const std::vector<std::pair<size_t, size_t>> dims = {{4, 8}, {8, 4}};
    const int layouts[] = {Fft2D::LAYOUT_INPLACE, Fft2D::LAYOUT_STOCKHAM};

    for (const auto& dim : dims) {
        for (int layout : layouts) {
            ComplexField original(dim.first, dim.second);
            for (size_t r = 0; r < dim.first; ++r) {
                for (size_t c = 0; c < dim.second; ++c) {
                    original.at(r, c) = std::complex<double>(r * 1.5, c * 0.5);
                }
            }

            ComplexField field = original;
            Fft2D fft_bound(field, layout);
            fft_bound.forward();
            fft_bound.inverse();

            for (size_t r = 0; r < dim.first; ++r) {
                for (size_t c = 0; c < dim.second; ++c) {
                    double err = std::abs(field.at(r, c) - original.at(r, c));
                    assert(err < TOLERANCE);
                }
            }
        }
    }
    std::cout << "PASÓ OK!" << std::endl;
}

void testOpenMpTaskVsParallelFor() {
    // Comparar filas comprueba el reparto. Completar columnas y contrastar
    // con la DFT independiente detecta tareas pendientes al retornar.
    std::cout << "[Test 8] Comparación OpenMP parallel for vs task (con y sin single)... ";
    Fft2D fft;
    const std::vector<std::pair<size_t, size_t>> dims = {{1, 8}, {2, 8}, {16, 8}};
    for (const auto& dim : dims) {
        ComplexField original(dim.first, dim.second);
        original.fillRandom(42);
        const ComplexField reference = naiveDft2D(original);
        for (int layout : {Fft2D::LAYOUT_INPLACE, Fft2D::LAYOUT_STOCKHAM}) {
            for (int threads : {1, 2, 4, 8}) {
                omp_set_num_threads(threads);
                ComplexField rows_reference = original;
                fft.forwardRows(rows_reference, Fft2D::TASK_TYPE_PARALLEL_FOR, true, layout);
                for (int task_type : {Fft2D::TASK_TYPE_TASK, Fft2D::TASK_TYPE_PARALLEL_FOR}) {
                    for (bool use_single : {true, false}) {
                        ComplexField field = original;
                        fft.forwardRows(field, task_type, use_single, layout);
                        for (size_t i = 0; i < field.size(); ++i) {
                            assert(std::abs(field.data()[i] - rows_reference.data()[i]) < TOLERANCE);
                        }
                        fft.forwardCols(field, layout);
                        for (size_t i = 0; i < field.size(); ++i) {
                            assert(std::abs(field.data()[i] - reference.data()[i]) < TOLERANCE);
                        }
                        fft.inverse(field, layout);
                        for (size_t i = 0; i < field.size(); ++i) {
                            assert(std::abs(field.data()[i] - original.data()[i]) < TOLERANCE);
                        }
                    }
                }
            }
        }
    }
    std::cout << "PASÓ OK!" << std::endl;
}

void testScheduleSweep() {
    // omp_set_schedule configura los bucles schedule(runtime) de Fft2D.
    // Esta prueba comprueba corrección con cada reparto, no mide rendimiento.
    std::cout << "[Test 9] Barrido de OpenMP schedule(runtime) (static, dynamic, guided)... ";
    const size_t M = 8;
    const size_t N = 4;
    ComplexField original(M, N);
    for (size_t r = 0; r < M; ++r) {
        for (size_t c = 0; c < N; ++c) {
            original.at(r, c) = std::complex<double>(r + 1.0, c + 2.0);
        }
    }
    const ComplexField reference = naiveDft2D(original);

    // Guardar el schedule original para restaurarlo al final
    omp_sched_t orig_kind;
    int orig_chunk;
    omp_get_schedule(&orig_kind, &orig_chunk);

    Fft2D fft;
    const omp_sched_t kinds[] = {omp_sched_static, omp_sched_dynamic, omp_sched_guided};
    const int chunks[] = {1, 4, 16};

    for (omp_sched_t kind : kinds) {
        for (int chunk : chunks) {
            omp_set_schedule(kind, chunk);
            for (int layout : {Fft2D::LAYOUT_INPLACE, Fft2D::LAYOUT_STOCKHAM}) {
                ComplexField field = original;
                fft.forward(field, layout);
                for (size_t i = 0; i < field.size(); ++i) {
                    assert(std::abs(field.data()[i] - reference.data()[i]) < TOLERANCE);
                }
                fft.inverse(field, layout);
                for (size_t i = 0; i < field.size(); ++i) {
                    assert(std::abs(field.data()[i] - original.data()[i]) < TOLERANCE);
                }
            }
        }
    }

    // Restaurar schedule global
    omp_set_schedule(orig_kind, orig_chunk);
    std::cout << "PASÓ OK!" << std::endl;
}

void testThreadScalabilityBothPasses() {
    // Comparar también los espectros evita depender únicamente de la ida y
    // vuelta. La referencia usa el mismo código con un hilo, como pide el lab.
    std::cout << "[Test 10] Invariancia numérica en directa E inversa con ambos layouts (1 vs 2, 4, 8 hilos)... ";
    const size_t M = 128;
    const size_t N = 64;
    ComplexField original(M, N);
    original.fillRandom(42);

    Fft2D fft;

    for (int layout : {Fft2D::LAYOUT_INPLACE, Fft2D::LAYOUT_STOCKHAM}) {
        // Referencia serial (1 hilo)
        omp_set_num_threads(1);
        ComplexField ref_fwd = original;
        fft.forward(ref_fwd, layout);
        ComplexField ref_inv = ref_fwd;
        fft.inverse(ref_inv, layout);

        const int thread_counts[] = {2, 4, 8};
        for (int threads : thread_counts) {
            omp_set_num_threads(threads);

            ComplexField test_fwd = original;
            fft.forward(test_fwd, layout);
            ComplexField test_inv = test_fwd;
            fft.inverse(test_inv, layout);

            for (size_t r = 0; r < M; ++r) {
                for (size_t c = 0; c < N; ++c) {
                    double diff_fwd = std::abs(ref_fwd.at(r, c) - test_fwd.at(r, c));
                    double diff_inv = std::abs(ref_inv.at(r, c) - test_inv.at(r, c));
                    assert(diff_fwd < TOLERANCE);
                    assert(diff_inv < TOLERANCE);
                }
            }
        }
    }
    std::cout << "PASÓ OK!" << std::endl;
}

void testExceptionHandling() {
    // Una entrada inválida debe rechazarse antes de comenzar a transformarla.
    // La lambda permite comprobar distintas llamadas con la misma condición.
    std::cout << "[Test 11] Robustez y manejo de excepciones (incluyendo pasadas individuales)... ";
    Fft2D fft;

    auto throwsInvalid = [](auto&& fn) {
        try { fn(); } catch (const std::invalid_argument&) { return true; }
        return false;
    };

    // 1. Instancia sin campo enlazado debe lanzar std::runtime_error
    bool threw_unbound = false;
    try {
        fft.forward();
    } catch (const std::runtime_error&) {
        threw_unbound = true;
    }
    assert(threw_unbound);

    // 2. Grilla con dimensión no potencia de dos (6x4 y 4x5) debe lanzar std::invalid_argument
    ComplexField bad_6x4(6, 4);
    ComplexField bad_4x5(4, 5);
    ComplexField good_4x4(4, 4);

    assert(throwsInvalid([&]{ fft.forward(bad_6x4); }));
    assert(throwsInvalid([&]{ fft.forward(bad_4x5); }));
    assert(throwsInvalid([&]{ fft.inverse(bad_6x4); }));

    // Cada pasada pública individual debe rechazar grillas y layouts inválidos
    assert(throwsInvalid([&]{ fft.forwardCols(bad_6x4, Fft2D::LAYOUT_INPLACE); }));
    assert(throwsInvalid([&]{ fft.inverseRows(bad_6x4, Fft2D::LAYOUT_INPLACE); }));
    assert(throwsInvalid([&]{ fft.inverseCols(bad_6x4, Fft2D::LAYOUT_INPLACE); }));

    assert(throwsInvalid([&]{ fft.forwardCols(good_4x4, 99); }));
    assert(throwsInvalid([&]{ fft.inverseRows(good_4x4, 99); }));
    assert(throwsInvalid([&]{ fft.inverseCols(good_4x4, 99); }));

    // 3. Layout inválido en forward/inverse debe lanzar std::invalid_argument
    assert(throwsInvalid([&]{ fft.forward(good_4x4, 99); }));
    assert(throwsInvalid([&]{ fft.inverse(good_4x4, -1); }));

    // 4. task_type inválido debe lanzar std::invalid_argument
    assert(throwsInvalid([&]{ fft.forwardRows(good_4x4, 42); }));

    std::cout << "PASÓ OK!" << std::endl;
}

// ============================================================================
// Verifica el contrato de normalización y dimensiones con un solo eje activo.
// Las pasadas inversas individuales NO escalan; inverse() sí normaliza.
// ============================================================================

void testPartialPassesAndDegenerateDimensions() {
    std::cout << "[Test 12] Pasadas individuales, escala y grillas 1x1, 1x8, 8x1, 4x8... ";
    const std::vector<std::pair<size_t, size_t>> dims = {{1, 1}, {1, 8}, {8, 1}, {4, 8}};

    for (const auto& dim : dims) {
        ComplexField original(dim.first, dim.second);
        for (size_t r = 0; r < dim.first; ++r) {
            for (size_t c = 0; c < dim.second; ++c) {
                original.at(r, c) = {0.25 + static_cast<double>(r),
                                     -0.5 + static_cast<double>(c)};
            }
        }
        const ComplexField reference = naiveDft2D(original);
        const double factor = static_cast<double>(dim.first) * static_cast<double>(dim.second);

        for (int layout : {Fft2D::LAYOUT_INPLACE, Fft2D::LAYOUT_STOCKHAM}) {
            Fft2D fft;
            ComplexField partial = original;
            fft.forwardRows(partial, Fft2D::TASK_TYPE_TASK, false, layout);
            fft.forwardCols(partial, layout);
            for (size_t r = 0; r < dim.first; ++r) {
                for (size_t c = 0; c < dim.second; ++c) {
                    assert(std::abs(partial.at(r, c) - reference.at(r, c)) < TOLERANCE);
                }
            }

            // Sin normalización, deshacer ambos ejes produce M*N*original.
            // Esta condición detectará una IFFT 1D ya escalada durante el merge.
            fft.inverseCols(partial, layout);
            fft.inverseRows(partial, layout);
            // inverse() completa sí debe recuperar original desde el espectro
            // de referencia, independientemente de las pasadas parciales.
            ComplexField complete = reference;
            fft.inverse(complete, layout);
            double squared_error = 0.0;
            for (size_t r = 0; r < dim.first; ++r) {
                for (size_t c = 0; c < dim.second; ++c) {
                    assert(std::abs(partial.at(r, c) - factor * original.at(r, c)) < TOLERANCE);
                    const auto error = complete.at(r, c) - original.at(r, c);
                    assert(std::abs(error) < TOLERANCE);
                    squared_error += std::norm(error);
                }
            }
            // RMSE = raíz del promedio de |recuperado - original|^2.
            // Aquí factor es también la cantidad de muestras de la grilla.
            assert(std::sqrt(squared_error / factor) < TOLERANCE);
        }
    }
    std::cout << "PASÓ OK!" << std::endl;
}

// ============================================================================
// Main de la suite de pruebas
// ============================================================================

int main() {
    const int original_threads = omp_get_max_threads();
    const int original_dynamic = omp_get_dynamic();
    // Solicitar realmente 1, 2, 4 y 8 hilos en las comparaciones de la suite.
    omp_set_dynamic(0);

    std::cout << "==========================================================" << std::endl;
    std::cout << " (FFT/IFFT 2D, Tolerancia Unificada 1e-10, Referencia Ec1)" << std::endl;
    std::cout << " Hilos OpenMP disponibles: " << original_threads << std::endl;
    std::cout << "==========================================================" << std::endl;

    for (int layout : {Fft2D::LAYOUT_INPLACE, Fft2D::LAYOUT_STOCKHAM}) {
        std::cout << "Layout " << layout << std::endl;
        testImpulse2x2(layout);
        testComplexExponentialSign(layout);
        testSineWavePeakAndMagnitude(layout);
        testParseval(layout);
    }
    testCompareAgainstNaiveDft();
    testRoundtripExhaustive();
    testBoundInstance();
    testOpenMpTaskVsParallelFor();
    testScheduleSweep();
    testThreadScalabilityBothPasses();
    testExceptionHandling();
    testPartialPassesAndDegenerateDimensions();

    // Restaurar estado global de hilos OpenMP
    omp_set_num_threads(original_threads);
    omp_set_dynamic(original_dynamic);

    std::cout << "\n>>> TODAS LAS PRUEBAS (12/12) PASARON! <<<" << std::endl;
    return 0;
}
