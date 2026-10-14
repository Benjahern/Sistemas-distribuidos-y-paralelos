#include "Benchmark.h"
#include "ComplexField.h"
#include "Fft2D.h"
#include "SpectrumMetrics.h"
#include "Visualizer.h"
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::size_t number(const std::string& text) {
    if (text.empty() || !std::all_of(text.begin(), text.end(), [](char c) { return c >= '0' && c <= '9'; }))
        throw std::invalid_argument("Número entero no negativo inválido: " + text);
    const auto value = std::stoull(text);
    if (value > std::numeric_limits<std::size_t>::max()) throw std::invalid_argument("Número demasiado grande");
    return static_cast<std::size_t>(value);
}
int integer(const std::string& text) {
    const auto value = number(text);
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("Número demasiado grande");
    return static_cast<int>(value);
}
std::vector<int> list(const std::string& text) {
    std::vector<int> values;
    std::size_t start = 0;
    do {
        const auto end = text.find(',', start);
        values.push_back(integer(text.substr(start, end == std::string::npos ? end : end-start)));
        if (values.back() <= 0) throw std::invalid_argument("Lista requiere valores positivos");
        if (end == std::string::npos) break;
        start = end+1;
    } while (true);
    return values;
}
void help() {
    std::cout << "Uso: build/fft2d [demo|benchmark|spectrum] [opciones]\n"
              << "Comunes: --output DIR --seed S --threads P[,P...]\n"
              << "Demo/espectro: --rows M --cols N --layout inplace|stockham|both\n"
              << "              --schedule static|dynamic|guided --chunk C --k K --l L\n"
              << "Benchmark: --min-size N --max-size N --chunk-grid N --chunks C,C,...\n"
              << "           --repetitions R (>=10) --memory-mib MiB\n"
              << "Sin argumentos: demo 8x4, ambos layouts. Benchmark compara ambos layouts.\n"
              << "Gráficos: python3 scripts/plot_results.py --input DIR\n";
}
void spectrum(std::size_t rows, std::size_t cols, std::size_t k, std::size_t l,
              int layout, const std::string& output) {
    if (k >= rows || l >= cols || ((k == 0 || k == rows/2) && (l == 0 || l == cols/2)))
        throw std::invalid_argument("Frecuencias fuera de rango o seno degenerado (cero en todas las muestras)");
    ComplexField field(rows, cols); field.fillSine(k, l);
    Fft2D().forward(field, layout);
    Visualizer::writeSpectrum(field, (std::filesystem::path(output)/"spectrum.dat").string());
    const double expected = static_cast<double>(rows)*cols/2.0;
    std::cout << "Seno 2D: picos (" << k << ',' << l << ") y (" << (rows-k)%rows << ','
              << (cols-l)%cols << "), módulo esperado=" << expected
              << ", observado=" << std::abs(field.at(k,l)) << " (escala lineal)\n";
}
}

int main(int argc, char** argv) {
    try {
        const std::string mode = argc > 1 ? argv[1] : "demo";
        if (mode == "--help" || mode == "help") { help(); return 0; }
        if (mode != "demo" && mode != "benchmark" && mode != "spectrum")
            throw std::invalid_argument("Modo desconocido: " + mode);
        BenchmarkConfig config;
        std::size_t rows = mode == "spectrum" ? 64 : 8, cols = mode == "spectrum" ? 64 : 4;
        std::size_t k = 1, l = 1;
        int layout = -1, schedule = 0, chunk = 0;
        for (int i = 2; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--help") { help(); return 0; }
            if (i+1 == argc) throw std::invalid_argument("Falta valor para " + option);
            const std::string value = argv[++i];
            if (option == "--output") config.output_dir = value;
            else if (option == "--seed") {
                const auto seed = number(value);
                if (seed > std::numeric_limits<unsigned>::max()) throw std::invalid_argument("Semilla demasiado grande");
                config.seed = static_cast<unsigned>(seed);
            } else if (option == "--threads") config.threads = list(value);
            else if (mode == "benchmark" && option == "--min-size") config.min_size = number(value);
            else if (mode == "benchmark" && option == "--max-size") config.max_size = number(value);
            else if (mode == "benchmark" && option == "--chunk-grid") config.chunk_size = number(value);
            else if (mode == "benchmark" && option == "--chunks") config.chunks = list(value);
            else if (mode == "benchmark" && option == "--repetitions") config.repetitions = integer(value);
            else if (mode == "benchmark" && option == "--memory-mib") config.memory_limit_mib = number(value);
            else if (mode != "benchmark" && option == "--rows") rows = number(value);
            else if (mode != "benchmark" && option == "--cols") cols = number(value);
            else if (mode != "benchmark" && option == "--k") k = number(value);
            else if (mode != "benchmark" && option == "--l") l = number(value);
            else if (mode != "benchmark" && option == "--chunk") chunk = integer(value);
            else if (mode != "benchmark" && option == "--layout") {
                if (value == "inplace" || value == "0") layout = 0;
                else if (value == "stockham" || value == "1") layout = 1;
                else if (value == "both") layout = -1;
                else throw std::invalid_argument("Layout desconocido");
            } else if (mode != "benchmark" && option == "--schedule") {
                if (value == "static") schedule = 0;
                else if (value == "dynamic") schedule = 1;
                else if (value == "guided") schedule = 2;
                else throw std::invalid_argument("Schedule desconocido");
            } else throw std::invalid_argument("Opción desconocida para este modo: " + option);
        }
        std::cout << std::setprecision(10);
        if (mode == "benchmark") {
            Benchmark::run(config);
            omp_set_num_threads(4); omp_set_dynamic(0);
            spectrum(64, 64, 1, 1, 0, config.output_dir);
            std::cout << "Datos guardados en " << config.output_dir << ". Para PNG, ejecutar scripts/plot_results.py.\n";
            return 0;
        }
        if (!ComplexField::isPowerOf2(rows) || !ComplexField::isPowerOf2(cols) ||
            cols > std::numeric_limits<std::size_t>::max()/rows)
            throw std::invalid_argument("Dimensiones inválidas: deben ser potencias de dos sin desbordamiento");
        if (config.threads.size() > 1) throw std::invalid_argument("Demo/espectro requieren un solo número de hilos");
        omp_set_dynamic(0); omp_set_max_active_levels(1);
        omp_set_num_threads(config.threads.empty() ? std::min(4, omp_get_num_procs()) : config.threads.front());
        const omp_sched_t schedules[] = {omp_sched_static, omp_sched_dynamic, omp_sched_guided};
        omp_set_schedule(schedules[schedule], chunk);
        std::filesystem::create_directories(config.output_dir);
        if (mode == "demo") {
            ComplexField original(rows, cols); original.fillRandom(config.seed);
            for (int current : {0, 1}) {
                if (layout != -1 && layout != current) continue;
                ComplexField work = original;
                Fft2D fft;
                const double start = omp_get_wtime();
                fft.forward(work, current);
                const double elapsed = omp_get_wtime()-start;
                std::cout << rows << 'x' << cols << " layout=" << current << " FFT=" << elapsed << " s\n";
                for (int method : {0, 1, 2}) {
                    const auto energies = SpectrumMetrics(original, work).parseval(method, true);
                    std::cout << "  Parseval método=" << method << " Ex=" << energies.input_energy
                              << " EX/(MN)=" << energies.spectrum_energy << " error relativo="
                              << energies.relative_error << '\n';
                    if (!std::isfinite(energies.relative_error) || energies.relative_error > 1e-10)
                        throw std::runtime_error("Parseval fuera de tolerancia");
                }
                fft.inverse(work, current);
                const double rmse = SpectrumMetrics(original, work).roundtripError();
                std::cout << "  RMSE=" << rmse << " (tolerancia 1e-10)\n";
                if (!std::isfinite(rmse) || rmse > 1e-10) throw std::runtime_error("RMSE fuera de tolerancia");
            }
        }
        spectrum(rows, cols, k, l, layout == -1 ? 0 : layout, config.output_dir);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
