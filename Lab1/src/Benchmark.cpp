#include "Benchmark.h"
#include "ComplexField.h"
#include "Fft2D.h"
#include "SpectrumMetrics.h"
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
struct OmpState {
    int threads = omp_get_max_threads(), dynamic = omp_get_dynamic();
    int levels = omp_get_max_active_levels(), chunk{};
    omp_sched_t schedule{};
    OmpState() { omp_get_schedule(&schedule, &chunk); }
    ~OmpState() {
        omp_set_num_threads(threads); omp_set_dynamic(dynamic);
        omp_set_max_active_levels(levels); omp_set_schedule(schedule, chunk);
    }
};

bool sameGroup(const BenchmarkPoint& a, const BenchmarkPoint& b) {
    return a.size == b.size && a.layout == b.layout && a.schedule == b.schedule && a.chunk == b.chunk;
}

BenchmarkPoint measure(const ComplexField& input, int layout, int schedule, int chunk,
                       int threads, int repetitions) {
    const omp_sched_t schedules[] = {omp_sched_static, omp_sched_dynamic, omp_sched_guided};
    omp_set_num_threads(threads);
    omp_set_schedule(schedules[schedule], chunk);
    BenchmarkPoint point;
    point.size = input.rows(); point.layout = layout; point.schedule = schedule;
    point.chunk = chunk; point.threads = threads; point.repetitions = repetitions;
    #pragma omp parallel default(none) shared(point)
    {
        #pragma omp single
        point.actual_threads = omp_get_num_threads();
    }
    ComplexField work = input;
    if (point.actual_threads != threads)
        throw std::runtime_error("OpenMP no pudo crear los hilos solicitados; revisar OMP_THREAD_LIMIT");
    Fft2D fft;
    fft.forward(work, layout); // Calentamiento no incluido en las muestras.
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repetitions));
    for (int r = 0; r < repetitions; ++r) {
        work = input; // Restaurar siempre la misma entrada FUERA del cronómetro.
        const double start = omp_get_wtime();
        fft.forward(work, layout);
        samples.push_back(omp_get_wtime() - start);
    }
    point.timing = Benchmark::statistics(samples);
    point.parseval_relative_error = SpectrumMetrics(input, work).parseval().relative_error;
    fft.inverse(work, layout);
    point.rmse = SpectrumMetrics(input, work).roundtripError();
    if (!std::isfinite(point.rmse) || !std::isfinite(point.parseval_relative_error) ||
        point.rmse > 1e-10 || point.parseval_relative_error > 1e-10)
        throw std::runtime_error("Benchmark: validación numérica fuera de tolerancia");
    std::cout << input.rows() << "x" << input.cols() << " layout=" << layout
              << " schedule=" << schedule << " chunk=" << chunk << " p=" << threads
              << " T=" << point.timing.mean << " +/- " << point.timing.stddev << " s\n";
    return point;
}
}

TimingStats Benchmark::statistics(const std::vector<double>& samples) {
    if (samples.size() < 2) throw std::invalid_argument("Se requieren al menos dos muestras");
    TimingStats result;
    // Welford: evita cancelación al calcular la varianza de tiempos cercanos.
    double m2 = 0.0;
    std::size_t count = 0;
    for (double sample : samples) {
        if (!std::isfinite(sample) || sample <= 0.0) throw std::invalid_argument("Tiempo inválido");
        ++count;
        const double delta = sample - result.mean;
        result.mean += delta / static_cast<double>(count);
        m2 += delta * (sample - result.mean);
    }
    result.stddev = std::sqrt(m2 / static_cast<double>(count - 1));
    return result;
}

ScalingPoint Benchmark::scaling(const TimingStats& baseline, const TimingStats& timing, int threads) {
    if (threads < 1 || !std::isfinite(baseline.mean) || !std::isfinite(timing.mean) ||
        baseline.mean <= 0 || timing.mean <= 0 || !std::isfinite(baseline.stddev) ||
        !std::isfinite(timing.stddev) || baseline.stddev < 0 || timing.stddev < 0)
        throw std::invalid_argument("Datos de scaling inválidos");
    ScalingPoint result;
    result.speedup = threads == 1 ? 1.0 : baseline.mean / timing.mean;
    // p=1 reutiliza LA MISMA variable aleatoria: T1/T1=1, no dos mediciones independientes.
    result.speedup_stddev = threads == 1 ? 0.0 : result.speedup * std::hypot(
        baseline.stddev / baseline.mean, timing.stddev / timing.mean);
    result.efficiency = result.speedup / threads;
    result.efficiency_stddev = result.speedup_stddev / threads;
    return result;
}

double Benchmark::fitSerialFraction(const std::vector<BenchmarkPoint>& points, bool& clamped) {
    const auto base = std::find_if(points.begin(), points.end(), [](const auto& p) { return p.threads == 1; });
    if (base == points.end() || !std::isfinite(base->timing.mean) || base->timing.mean <= 0)
        throw std::invalid_argument("Falta referencia T1 válida");
    double numerator = 0.0, denominator = 0.0;
    for (const auto& point : points) {
        if (!sameGroup(*base, point) || point.threads < 1 || !std::isfinite(point.timing.mean) || point.timing.mean <= 0)
            throw std::invalid_argument("Grupo Amdahl incompatible");
        if (point.threads == 1) continue;
        const double inverse_p = 1.0 / point.threads;
        const double a = 1.0 - inverse_p;
        numerator += a * (point.timing.mean / base->timing.mean - inverse_p);
        denominator += a * a;
    }
    if (denominator == 0.0) { clamped = false; return std::numeric_limits<double>::quiet_NaN(); }
    const double raw = numerator / denominator;
    clamped = raw < 0.0 || raw > 1.0;
    return std::clamp(raw, 0.0, 1.0);
}

std::vector<BenchmarkPoint> Benchmark::run(const BenchmarkConfig& config) {
    if (config.repetitions < 10 || !ComplexField::isPowerOf2(config.min_size) ||
        !ComplexField::isPowerOf2(config.max_size) || config.min_size > config.max_size ||
        !ComplexField::isPowerOf2(config.chunk_size) || config.memory_limit_mib == 0)
        throw std::invalid_argument("Benchmarks: tamaños potencia de dos, min<=max, repeticiones>=10 y memoria>0");
    auto threads = config.threads;
    if (threads.empty()) threads = {1, 2, 4, 8, omp_get_num_procs()};
    threads.push_back(1);
    std::sort(threads.begin(), threads.end());
    threads.erase(std::unique(threads.begin(), threads.end()), threads.end());
    if (threads.front() < 1) throw std::invalid_argument("Número de hilos inválido");
    auto chunks = config.chunks;
    for (int chunk : chunks) if (chunk <= 0) throw std::invalid_argument("Chunk debe ser positivo");
    std::sort(chunks.begin(), chunks.end());
    chunks.erase(std::unique(chunks.begin(), chunks.end()), chunks.end());
    // Cota conservadora: entrada + trabajo + temporales por hilo + margen (6 grillas).
    const auto fits = [&](std::size_t n) {
        const long double estimate = (6.0L * n * n + 3.0L * n * threads.back()) * sizeof(ComplexField::Complex);
        return estimate <= static_cast<long double>(config.memory_limit_mib) * 1024 * 1024;
    };
    if (!fits(config.min_size) || !fits(config.chunk_size))
        throw std::invalid_argument("Tamaño inicial o barrido de chunks excede presupuesto de memoria");
    OmpState restore;
    omp_set_dynamic(0); omp_set_max_active_levels(1);
    std::vector<BenchmarkPoint> points;
    std::string reason;
    std::size_t last_size = 0;
    for (std::size_t size = config.min_size; size <= config.max_size;) {
        if (!fits(size)) { reason = "siguiente tamaño " + std::to_string(size) + ": presupuesto de memoria"; break; }
        ComplexField input(size, size); input.fillRandom(config.seed);
        for (int layout : {0, 1})
            for (int p : threads) points.push_back(measure(input, layout, 0, 0, p, config.repetitions));
        last_size = size;
        if (size > config.max_size / 2) {
            reason = "siguiente tamaño: doble de " + std::to_string(size) + ", excede --max-size configurado";
            break;
        }
        size *= 2;
    }
    ComplexField input(config.chunk_size, config.chunk_size); input.fillRandom(config.seed);
    std::vector<int> chunk_threads{1};
    if (threads.back() != 1) chunk_threads.push_back(threads.back());
    for (int layout : {0, 1})
        for (int schedule : {0, 1, 2})
            for (int chunk : chunks)
                for (int p : chunk_threads)
                    points.push_back(measure(input, layout, schedule, chunk, p, config.repetitions));
    reason = "máximo medido en serie=" + std::to_string(last_size) + "; " + reason;
    writeResults(config, points, reason);
    return points;
}

void Benchmark::writeResults(const BenchmarkConfig& config, const std::vector<BenchmarkPoint>& points,
                             const std::string& limit_reason) {
    std::filesystem::create_directories(config.output_dir);
    const auto path = [&](const char* name) { return (std::filesystem::path(config.output_dir) / name).string(); };
    // Invalidar derivados/procedencia antiguos incluso si se usa el CLI directamente.
    std::filesystem::remove(path("performance plots.png"));
    std::filesystem::remove(path("benchmark_manifest.json"));
    std::ofstream timing(path("benchmark results.dat")), scaling_out(path("scaling analysis.dat")),
                  error(path("roundtrip error.dat"));
    if (!timing || !scaling_out || !error) throw std::runtime_error("No se pueden escribir los resultados");
    for (auto* out : {&timing, &scaling_out, &error}) {
        *out << std::setprecision(17) << "# seed=" << config.seed << " repetitions=" << config.repetitions
             << " available_processors=" << omp_get_num_procs() << " memory_limit_mib=" << config.memory_limit_mib
             << "\n# " << limit_reason << "\n";
    }
    timing << "# rows cols layout schedule chunk requested_threads actual_threads repetitions mean_s stddev_s rmse parseval_relative_error\n";
    scaling_out << "# rows cols layout schedule chunk threads speedup sigma_speedup efficiency sigma_efficiency f amdahl_speedup residual fit_clamped\n";
    error << "# rows cols layout schedule chunk threads rmse parseval_relative_error\n";
    for (const auto& point : points) {
        std::vector<BenchmarkPoint> group;
        for (const auto& candidate : points) if (sameGroup(point, candidate)) group.push_back(candidate);
        const auto base = std::find_if(group.begin(), group.end(), [](const auto& p) { return p.threads == 1; });
        if (base == group.end()) throw std::runtime_error("Falta baseline para resultados");
        bool clamped = false;
        auto scaled = scaling(base->timing, point.timing, point.threads);
        scaled.serial_fraction = fitSerialFraction(group, clamped);
        scaled.amdahl_speedup = 1.0 / (scaled.serial_fraction + (1.0-scaled.serial_fraction)/point.threads);
        scaled.residual = scaled.speedup - scaled.amdahl_speedup;
        timing << point.size << ' ' << point.size << ' ' << point.layout << ' ' << point.schedule << ' '
               << point.chunk << ' ' << point.threads << ' ' << point.actual_threads << ' ' << point.repetitions
               << ' ' << point.timing.mean << ' ' << point.timing.stddev << ' ' << point.rmse << ' '
               << point.parseval_relative_error << '\n';
        scaling_out << point.size << ' ' << point.size << ' ' << point.layout << ' ' << point.schedule << ' '
                    << point.chunk << ' ' << point.threads << ' ' << scaled.speedup << ' ' << scaled.speedup_stddev
                    << ' ' << scaled.efficiency << ' ' << scaled.efficiency_stddev << ' ' << scaled.serial_fraction
                    << ' ' << scaled.amdahl_speedup << ' ' << scaled.residual << ' ' << clamped << '\n';
        error << point.size << ' ' << point.size << ' ' << point.layout << ' ' << point.schedule << ' '
              << point.chunk << ' ' << point.threads << ' ' << point.rmse << ' ' << point.parseval_relative_error << '\n';
    }
    if (!timing || !scaling_out || !error) throw std::runtime_error("Error escribiendo resultados");
}
