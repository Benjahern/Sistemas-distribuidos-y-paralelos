#pragma once
#include <cstddef>
#include <string>
#include <vector>

struct TimingStats { double mean{}, stddev{}; };
struct BenchmarkConfig {
    std::size_t min_size{128}, max_size{1024};
    std::size_t chunk_size{256}; // Tamaño de grilla para el barrido de chunks.
    std::vector<int> threads; // Vacío: 1,2,4,8 y procesadores disponibles.
    std::vector<int> chunks{1, 2, 4, 8, 16, 32, 64};
    int repetitions{10};
    unsigned seed{42};
    std::size_t memory_limit_mib{512};
    std::string output_dir{"results"};
};
struct BenchmarkPoint {
    std::size_t size{};
    int layout{}, schedule{}, chunk{}, threads{}, actual_threads{}, repetitions{};
    TimingStats timing;
    double rmse{}, parseval_relative_error{};
};
struct ScalingPoint {
    double speedup{}, speedup_stddev{}, efficiency{}, efficiency_stddev{};
    double serial_fraction{}, amdahl_speedup{}, residual{};
    bool fit_clamped{};
};

class Benchmark {
public:
    static TimingStats statistics(const std::vector<double>& samples);
    static ScalingPoint scaling(const TimingStats& baseline, const TimingStats& timing, int threads);
    // Ajuste OLS de Tp/T1 = 1/p + f*(1-1/p), acotado a [0,1].
    static double fitSerialFraction(const std::vector<BenchmarkPoint>& points, bool& clamped);
    static std::vector<BenchmarkPoint> run(const BenchmarkConfig& config);
    static void writeResults(const BenchmarkConfig& config, const std::vector<BenchmarkPoint>& points,
                             const std::string& limit_reason);
};
