#include "Fft2D.h"
#include "ComplexField.h"
#include "Butterfly1D.h"

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#undef NDEBUG
#include <cassert>
#include <random>
#include <utility>
#include <omp.h>

// Tolerancia estándar declarada en el README y justificada según IEEE 754
constexpr double TOLERANCE = 1e-10;

// ============================================================================
// STUBS / MOCKS para Rol 1 (ComplexField) y Rol 2 (Butterfly1D)
// Opt-in mediante flag de compilación -DFFT2D_STANDALONE_STUBS.
// Al integrar con los roles reales, este bloque no se compila para evitar duplicados.
// ============================================================================
#ifdef FFT2D_STANDALONE_STUBS

ComplexField::ComplexField(size_t rows, size_t cols)
    : rows_(rows), cols_(cols), data_(rows * cols, std::complex<double>(0.0, 0.0)) {}

ComplexField::ComplexField(size_t rows, size_t cols, const std::complex<double>& initial_value)
    : rows_(rows), cols_(cols), data_(rows * cols, initial_value) {}

Butterfly1D::Butterfly1D(std::complex<double>* data, size_t n, bool inverse)
    : data_(data), n_(n), inverse_(inverse) {}

void Butterfly1D::setData(std::complex<double>* data, size_t n, bool inverse) {
    data_ = data;
    n_ = n;
    inverse_ = inverse;
}

static size_t bitReverse(size_t i, size_t log2n) {
    size_t rev = 0;
    for (size_t bit = 0; bit < log2n; ++bit) {
        if ((i >> bit) & 1) {
            rev |= (size_t(1) << (log2n - 1 - bit));
        }
    }
    return rev;
}

void Butterfly1D::transform() { transform(0, 0, 0); }
void Butterfly1D::transform(int schedule_type) { transform(0, schedule_type, 0); }
void Butterfly1D::transform(int schedule_type, int chunk_size) { transform(0, schedule_type, chunk_size); }
void Butterfly1D::transformCollapse() { transform(0, 0, 0); }
void Butterfly1D::transformStages() { transform(0, 0, 0); }

void Butterfly1D::transform(int layout, int schedule_type, int chunk_size) {
    (void)schedule_type;
    (void)chunk_size;
    if (n_ <= 1 || !data_) return;

    const double pi = 3.14159265358979323846;
    const double sign = inverse_ ? 1.0 : -1.0;

    size_t p = 0;
    while ((size_t(1) << p) < n_) ++p;

    if (layout == 0) {
        // Cooley-Tukey (in-place)
        for (size_t i = 0; i < n_; ++i) {
            size_t rev = bitReverse(i, p);
            if (i < rev) {
                std::swap(data_[i], data_[rev]);
            }
        }
        for (size_t s = 1; s <= p; ++s) {
            size_t m = size_t(1) << s;
            size_t m2 = m >> 1;
            std::complex<double> wm = std::polar(1.0, sign * 2.0 * pi / static_cast<double>(m));
            for (size_t k = 0; k < n_; k += m) {
                std::complex<double> w(1.0, 0.0);
                for (size_t j = 0; j < m2; ++j) {
                    std::complex<double> u = data_[k + j];
                    std::complex<double> v = w * data_[k + j + m2];
                    data_[k + j] = u + v;
                    data_[k + j + m2] = u - v;
                    w *= wm;
                }
            }
        }
    } else {
        // Stockham (out-of-place)
        std::vector<std::complex<double>> src(data_, data_ + n_);
        std::vector<std::complex<double>> dst(n_);
        for (size_t s = 0; s < p; ++s) {
            size_t l = size_t(1) << s;
            size_t g = n_ / (size_t(1) << (s + 1));
            for (size_t k = 0; k < l; ++k) {
                std::complex<double> omega = std::polar(1.0, sign * 2.0 * pi * static_cast<double>(k) / static_cast<double>(size_t(1) << (s + 1)));
                for (size_t j = 0; j < g; ++j) {
                    size_t i0 = j + 2 * k * g;
                    size_t i1 = i0 + g;
                    size_t o0 = j + k * g;
                    size_t o1 = o0 + n_ / 2;
                    std::complex<double> u = src[i0];
                    std::complex<double> v = omega * src[i1];
                    dst[o0] = u + v;
                    dst[o1] = u - v;
                }
            }
            src.swap(dst);
        }
        for (size_t i = 0; i < n_; ++i) {
            data_[i] = src[i];
        }
    }
}

#endif // FFT2D_STANDALONE_STUBS

// ============================================================================
// DFT 2D Ingenua (Referencia Matemática Independiente O(M^2 * N^2))
// Evalúa directamente la Ecuación (1) del enunciado sin usar mariposas.
// ============================================================================

ComplexField naiveDft2D(const ComplexField& x) {
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
    std::cout << "[Test 8] Comparación OpenMP parallel for vs task (con y sin single)... ";
    const size_t M = 16;
    const size_t N = 8;
    ComplexField f_base(M, N);
    for (size_t r = 0; r < M; ++r) {
        for (size_t c = 0; c < N; ++c) {
            f_base.at(r, c) = std::complex<double>(r * 2.0 + c, r - c * 1.5);
        }
    }

    ComplexField f_for = f_base;
    ComplexField f_task_single = f_base;
    ComplexField f_task_no_single = f_base;

    Fft2D fft;
    fft.forwardRows(f_for, Fft2D::TASK_TYPE_PARALLEL_FOR);
    fft.forwardRows(f_task_single, Fft2D::TASK_TYPE_TASK, true);
    fft.forwardRows(f_task_no_single, Fft2D::TASK_TYPE_TASK, false);

    for (size_t r = 0; r < M; ++r) {
        for (size_t c = 0; c < N; ++c) {
            double d1 = std::abs(f_for.at(r, c) - f_task_single.at(r, c));
            double d2 = std::abs(f_for.at(r, c) - f_task_no_single.at(r, c));
            assert(d1 < TOLERANCE);
            assert(d2 < TOLERANCE);
        }
    }

    // Cobertura explícita de Stockham en las ramas task
    ComplexField s_for = f_base, s_task1 = f_base, s_task0 = f_base;
    fft.forwardRows(s_for,   Fft2D::TASK_TYPE_PARALLEL_FOR, true,  Fft2D::LAYOUT_STOCKHAM);
    fft.forwardRows(s_task1, Fft2D::TASK_TYPE_TASK,         true,  Fft2D::LAYOUT_STOCKHAM);
    fft.forwardRows(s_task0, Fft2D::TASK_TYPE_TASK,         false, Fft2D::LAYOUT_STOCKHAM);

    for (size_t r = 0; r < M; ++r) {
        for (size_t c = 0; c < N; ++c) {
            assert(std::abs(s_for.at(r, c) - s_task1.at(r, c)) < TOLERANCE);
            assert(std::abs(s_for.at(r, c) - s_task0.at(r, c)) < TOLERANCE);
        }
    }
    std::cout << "PASÓ OK!" << std::endl;
}

void testScheduleSweep() {
    std::cout << "[Test 9] Barrido de OpenMP schedule(runtime) (static, dynamic, guided)... ";
    const size_t M = 8;
    const size_t N = 8;
    ComplexField original(M, N);
    for (size_t r = 0; r < M; ++r) {
        for (size_t c = 0; c < N; ++c) {
            original.at(r, c) = std::complex<double>(r + 1.0, c + 2.0);
        }
    }

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
            ComplexField field = original;
            fft.forward(field);
            fft.inverse(field);

            for (size_t r = 0; r < M; ++r) {
                for (size_t c = 0; c < N; ++c) {
                    double err = std::abs(field.at(r, c) - original.at(r, c));
                    assert(err < TOLERANCE);
                }
            }
        }
    }

    // Restaurar schedule global
    omp_set_schedule(orig_kind, orig_chunk);
    std::cout << "PASÓ OK!" << std::endl;
}

void testThreadScalabilityBothPasses() {
    std::cout << "[Test 10] Invariancia numérica en directa E inversa con ambos layouts (1 vs 2, 4, 8 hilos)... ";
    const size_t M = 16;
    const size_t N = 16;
    ComplexField original(M, N);
    for (size_t r = 0; r < M; ++r) {
        for (size_t c = 0; c < N; ++c) {
            original.at(r, c) = std::complex<double>(r * 0.7, c * 0.3);
        }
    }

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

            fft.inverseCols(partial, layout);
            fft.inverseRows(partial, layout);
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

    std::cout << "\n>>> TODAS LAS PRUEBAS (12/12) PASARON! <<<" << std::endl;
    return 0;
}
