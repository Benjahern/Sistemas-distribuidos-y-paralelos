#include "Butterfly1D.h"
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <cassert>
#include <random>

constexpr double TOLERANCE = 1e-10;

bool isClose(std::complex<double> a, std::complex<double> b) {
    return std::abs(a - b) < TOLERANCE;
}

void testImpulse4() {
    std::cout << "[Test 1] Impulso N=4 (Cooley-Tukey vs Stockham)... " << std::flush;
    
    // x = (1, 0, 0, 0)
    std::vector<std::complex<double>> ct_data = {{1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
    std::vector<std::complex<double>> st_data = ct_data;

    Butterfly1D ct(ct_data.data(), 4, false);
    ct.transform(0, 0, 0); // Cooley-Tukey

    Butterfly1D st(st_data.data(), 4, false);
    st.transform(1, 0, 0); // Stockham

    for (size_t i = 0; i < 4; ++i) {
        assert(isClose(ct_data[i], {1.0, 0.0}));
        assert(isClose(st_data[i], {1.0, 0.0}));
    }
    std::cout << "PASSED\n";
}

void testEquivalenceN8() {
    std::cout << "[Test 2] Equivalencia N=8 en datos aleatorios... " << std::flush;

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    size_t N = 8;
    std::vector<std::complex<double>> orig(N);
    for (size_t i = 0; i < N; ++i) {
        orig[i] = {dist(rng), dist(rng)};
    }

    std::vector<std::complex<double>> ct_data = orig;
    std::vector<std::complex<double>> st_data = orig;

    Butterfly1D ct(ct_data.data(), N, false);
    ct.transform(0, 0, 0);

    Butterfly1D st(st_data.data(), N, false);
    st.transform(1, 0, 0);

    for (size_t i = 0; i < N; ++i) {
        assert(isClose(ct_data[i], st_data[i]));
    }
    std::cout << "PASSED\n";
}

void testRoundtrip1DUnscaledContract() {
    std::cout << "[Test 3] Contrato sin escala en 1D: IFFT(FFT(x)) == N * x... " << std::flush;

    size_t N = 16;
    std::vector<std::complex<double>> orig(N);
    for (size_t i = 0; i < N; ++i) {
        orig[i] = {static_cast<double>(i), -static_cast<double>(i)};
    }

    std::vector<std::complex<double>> data = orig;

    // Directa
    Butterfly1D fwd(data.data(), N, false);
    fwd.transform(0, 0, 0);

    // Inversa (sin escalado en Butterfly1D)
    Butterfly1D inv(data.data(), N, true);
    inv.transform(0, 0, 0);

    // El contrato exige que IFFT(FFT(x)) sea N * x (sin escalar por Butterfly1D)
    for (size_t i = 0; i < N; ++i) {
        std::complex<double> expected = static_cast<double>(N) * orig[i];
        assert(isClose(data[i], expected));
    }

    // Al aplicar 1/N por separado en 1D se recupera x exactamente
    for (size_t i = 0; i < N; ++i) {
        data[i] /= static_cast<double>(N);
        assert(isClose(data[i], orig[i]));
    }
    std::cout << "PASSED\n";
}

void testSchedulesAndClauses() {
    std::cout << "[Test 4] Prueba de Schedules (Static, Dynamic, Guided) y Chunks... " << std::flush;

    size_t N = 32;
    std::vector<std::complex<double>> base(N);
    for (size_t i = 0; i < N; ++i) {
        base[i] = {std::sin(2.0 * 3.14159265358979323846 * i / N), 0.0};
    }

    // Static
    std::vector<std::complex<double>> d_static = base;
    Butterfly1D b_static(d_static.data(), N, false);
    b_static.transform(0, 0, 4);

    // Dynamic
    std::vector<std::complex<double>> d_dynamic = base;
    Butterfly1D b_dynamic(d_dynamic.data(), N, false);
    b_dynamic.transform(0, 1, 4);

    // Guided
    std::vector<std::complex<double>> d_guided = base;
    Butterfly1D b_guided(d_guided.data(), N, false);
    b_guided.transform(0, 2, 4);

    // Collapse
    std::vector<std::complex<double>> d_collapse = base;
    Butterfly1D b_collapse(d_collapse.data(), N, false);
    b_collapse.transformCollapse();

    // Stages
    std::vector<std::complex<double>> d_stages = base;
    Butterfly1D b_stages(d_stages.data(), N, false);
    b_stages.transformStages();

    for (size_t i = 0; i < N; ++i) {
        assert(isClose(d_static[i], d_dynamic[i]));
        assert(isClose(d_static[i], d_guided[i]));
        assert(isClose(d_static[i], d_collapse[i]));
        assert(isClose(d_static[i], d_stages[i]));
    }
    std::cout << "PASSED\n";
}

void testOpenMPClausesDemo() {
    std::cout << "[Test 5] Cláusulas OpenMP (initTwiddlesSingle, accumulateFirstprivate, stageIndexLastprivate)... " << std::flush;

    size_t N = 8;
    std::vector<std::complex<double>> data(N, {0.0, 0.0});
    Butterfly1D b(data.data(), N, false);

    auto twiddles = b.initTwiddlesSingle();
    assert(twiddles.size() == N);
    assert(isClose(twiddles[0], {1.0, 0.0}));

    b.accumulateFirstprivate(5.0);
    for (size_t i = 0; i < N; ++i) {
        assert(isClose(data[i], {5.0, 0.0}));
    }

    int last_stage = b.stageIndexLastprivate();
    assert(last_stage == 3); // log2(8) = 3

    std::cout << "PASSED\n";
}

int main() {
    std::cout << "=== PRUEBAS DE UNIDAD ROL 2 (Butterfly1D) ===\n";
    testImpulse4();
    testEquivalenceN8();
    testRoundtrip1DUnscaledContract();
    testSchedulesAndClauses();
    testOpenMPClausesDemo();
    std::cout << "=== TODAS LAS PRUEBAS DE ROL 2 PASARON EXITOSAMENTE ===\n";
    return 0;
}
