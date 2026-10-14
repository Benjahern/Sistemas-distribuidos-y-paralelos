#include "Benchmark.h"
#include "Butterfly1D.h"
#include "ComplexField.h"
#include "Fft2D.h"
#include "SpectrumMetrics.h"
#include "Visualizer.h"
#include <omp.h>
#include <cassert>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

template<class Function> void invalid(Function function) {
    bool threw = false;
    try { function(); } catch (const std::invalid_argument&) { threw = true; }
    assert(threw);
}

void testMetrics() {
    for (int layout : {0, 1}) {
        ComplexField input(8, 4); input.fillRandom(42);
        ComplexField transformed = input;
        Fft2D().forward(transformed, layout);
        for (int method : {0, 1, 2}) {
            for (bool use_private : {false, true}) {
                const auto result = SpectrumMetrics(input, transformed).parseval(method, use_private);
                assert(result.relative_error < 1e-12);
                assert(result.input_energy > 0);
            }
        }
        const auto magnitude = Visualizer::magnitudeNoWait(transformed);
        for (std::size_t i = 0; i < magnitude.size(); ++i)
            assert(std::abs(magnitude[i]-std::abs(transformed.data()[i])) < 1e-12);
        Fft2D().inverse(transformed, layout);
        assert(SpectrumMetrics(input, transformed).roundtripError() < 1e-10);
        for (std::size_t i = 0; i < input.size(); ++i) transformed.data()[i] = input.data()[i] + std::complex<double>(3, 4);
        assert(std::abs(SpectrumMetrics(input, transformed).roundtripError()-5) < 1e-12);
        invalid([&] { SpectrumMetrics(input, transformed).parseval(3); });
    }
    ComplexField zero(4, 4);
    assert(SpectrumMetrics(zero, zero).parseval().relative_error == 0);
    ComplexField impulse(2, 2); impulse.fillImpulse();
    auto spectrum = impulse; Fft2D().forward(spectrum);
    const auto result = SpectrumMetrics(impulse, spectrum).parseval();
    assert(result.input_energy == 1 && result.spectrum_energy == 1);
    invalid([&] { SpectrumMetrics bad(zero, impulse); });
    invalid([] { ComplexField empty; SpectrumMetrics bad(empty, empty); });
}

void testAnalytical1DAndBarriers() {
    const double pi = std::acos(-1.0);
    for (std::size_t n : {4, 8, 16}) {
        for (int layout : {0, 1}) {
            for (int schedule : {0, 1, 2}) {
                std::vector<std::complex<double>> original(n), work(n);
                original[0] = 1;
                work = original;
                Butterfly1D(work.data(), n).transform(layout, schedule, 2);
                for (auto value : work) assert(std::abs(value-1.0) < 1e-10);
                Butterfly1D(work.data(), n, true).transform(layout, schedule, 2);
                for (std::size_t i = 0; i < n; ++i) assert(std::abs(work[i]/static_cast<double>(n)-original[i]) < 1e-10);
                for (std::size_t i = 0; i < n; ++i) original[i] = std::sin(2*pi*i/n);
                work = original;
                Butterfly1D(work.data(), n).transform(layout, schedule, 2);
                for (std::size_t k = 0; k < n; ++k) {
                    const auto expected = k == 1 ? std::complex<double>(0, -static_cast<double>(n)/2) :
                        k == n-1 ? std::complex<double>(0, static_cast<double>(n)/2) : std::complex<double>(0, 0);
                    assert(std::abs(work[k]-expected) < 1e-10);
                }
                Butterfly1D(work.data(), n, true).transform(layout, schedule, 2);
                for (std::size_t i = 0; i < n; ++i) assert(std::abs(work[i]/static_cast<double>(n)-original[i]) < 1e-10);
            }
        }
    }
    ComplexField input(1, 4096); input.fillRandom(42);
    for (int layout : {0, 1}) {
        auto serial = input;
        omp_set_num_threads(1);
        Butterfly1D(serial.data(), serial.size()).transform(layout, 0, 0);
        omp_set_num_threads(4);
        for (int schedule : {0, 1, 2}) {
            auto parallel = input;
            Butterfly1D(parallel.data(), parallel.size()).transform(layout, schedule, 4);
            for (std::size_t i = 0; i < input.size(); ++i) assert(std::abs(serial.data()[i]-parallel.data()[i]) < 1e-10);
        }
        auto parallel = input;
        if (layout == 0) Butterfly1D(parallel.data(), parallel.size()).transformStages();
        else Butterfly1D(parallel.data(), parallel.size()).transformCollapse();
        assert(SpectrumMetrics(serial, parallel).roundtripError() < 1e-10);
    }
    // Comparación numérica 2D rectangular con anidamiento habilitado.
    ComplexField grid(64, 32); grid.fillRandom(42);
    omp_set_max_active_levels(2);
    for (int layout : {0, 1}) {
        auto serial = grid, parallel = grid;
        omp_set_num_threads(1); Fft2D().forward(serial, layout);
        omp_set_num_threads(4); Fft2D().forward(parallel, layout);
        assert(SpectrumMetrics(serial, parallel).roundtripError() < 1e-10);
    }
    omp_set_max_active_levels(1);
}

void testStatistics() {
    const auto stats = Benchmark::statistics({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    assert(std::abs(stats.mean-5.5) < 1e-12);
    assert(std::abs(stats.stddev-std::sqrt(82.5/9)) < 1e-12);
    const auto scaled = Benchmark::scaling({10, 1}, {5, 0.5}, 2);
    assert(scaled.speedup == 2 && scaled.efficiency == 1);
    assert(std::abs(scaled.speedup_stddev-2*std::sqrt(0.02)) < 1e-12);
    assert(scaled.efficiency_stddev == scaled.speedup_stddev/2);
    const auto baseline = Benchmark::scaling({10, 1}, {10, 1}, 1);
    assert(baseline.speedup == 1 && baseline.speedup_stddev == 0);
    std::vector<BenchmarkPoint> synthetic;
    for (int p : {1, 2, 4, 8}) {
        BenchmarkPoint point; point.size = 128; point.threads = p;
        point.timing.mean = 10*(0.2+0.8/p); synthetic.push_back(point);
    }
    bool clamped = true;
    assert(std::abs(Benchmark::fitSerialFraction(synthetic, clamped)-0.2) < 1e-12 && !clamped);
    for (auto& point : synthetic) if (point.threads > 1) point.timing.mean = 20;
    assert(Benchmark::fitSerialFraction(synthetic, clamped) == 1 && clamped);
    for (auto& point : synthetic) if (point.threads > 1) point.timing.mean = 0.1;
    assert(Benchmark::fitSerialFraction(synthetic, clamped) == 0 && clamped);
    assert(std::isnan(Benchmark::fitSerialFraction({synthetic.front()}, clamped)));
    invalid([] { Benchmark::statistics({1}); });
    invalid([] { Benchmark::statistics({1, -2}); });
    invalid([] { Benchmark::statistics({1, std::numeric_limits<double>::quiet_NaN()}); });
    invalid([] { Benchmark::scaling({0, 1}, {1, 1}, 2); });
    invalid([] { bool flag = false; Benchmark::fitSerialFraction({}, flag); });
    invalid([] {
        BenchmarkPoint point; point.threads = 1;
        point.timing.mean = std::numeric_limits<double>::quiet_NaN();
        bool flag = false; Benchmark::fitSerialFraction({point}, flag);
    });
}

void testBenchmarkExports() {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    BenchmarkConfig config;
    config.min_size = 4; config.max_size = 8; config.chunk_size = 8;
    config.threads = {1, 2}; config.chunks = {1, 2};
    config.output_dir = "build/test-benchmark-" + std::to_string(stamp);
    omp_set_num_threads(4); omp_set_schedule(omp_sched_guided, 7);
    const auto points = Benchmark::run(config);
    assert(points.size() == 32);
    assert(omp_get_max_threads() == 4);
    omp_sched_t schedule; int chunk;
    omp_get_schedule(&schedule, &chunk);
    assert(schedule == omp_sched_guided && chunk == 7);
    for (const auto& point : points) {
        assert(point.repetitions >= 10 && point.timing.mean > 0);
        assert(point.actual_threads == point.threads && point.rmse < 1e-10);
    }
    ComplexField sine(8, 4); sine.fillSine(1, 1); Fft2D().forward(sine);
    assert(std::abs(std::abs(sine.at(1, 1))-16) < 1e-10);
    Visualizer::writeSpectrum(sine, config.output_dir+"/spectrum.dat");
    for (const char* name : {"benchmark results.dat", "scaling analysis.dat", "roundtrip error.dat", "spectrum.dat"}) {
        const auto path = std::filesystem::path(config.output_dir)/name;
        assert(std::filesystem::file_size(path) > 0);
        std::filesystem::remove(path);
    }
    std::filesystem::remove(config.output_dir);
    config.repetitions = 9;
    invalid([&] { Benchmark::run(config); });
    config.repetitions = 10; config.memory_limit_mib = 0;
    invalid([&] { Benchmark::run(config); });
    config.memory_limit_mib = 512; config.min_size = 3;
    invalid([&] { Benchmark::run(config); });
}

int main() {
    omp_set_dynamic(0); omp_set_num_threads(4);
    testMetrics(); testAnalytical1DAndBarriers(); testStatistics(); testBenchmarkExports();
    std::cout << "Métricas, estadística, Amdahl, barreras y exportación: OK\n";
}
