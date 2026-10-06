#include "SpectrumMetrics.h"
#include <cmath>
#include <stdexcept>

SpectrumMetrics::SpectrumMetrics(const ComplexField& original, const ComplexField& other)
    : original_(original), other_(other) {
    if (original.size() == 0 || original.rows() != other.rows() ||
        original.cols() != other.cols()) {
        throw std::invalid_argument("SpectrumMetrics: grillas vacías o incompatibles");
    }
}

double SpectrumMetrics::roundtripError() const {
    const auto* a = original_.data();
    const auto* b = other_.data();
    const std::size_t m = original_.rows(), n = original_.cols();
    double sum = 0.0;
    // Cada celda es independiente; reduction evita actualizaciones en carrera.
    #pragma omp parallel for collapse(2) default(none) shared(a, b, m, n) reduction(+:sum)
    for (std::size_t r = 0; r < m; ++r)
        for (std::size_t c = 0; c < n; ++c)
            sum += std::norm(a[r*n+c] - b[r*n+c]);
    return std::sqrt(sum / static_cast<double>(original_.size()));
}

double SpectrumMetrics::energy(const ComplexField& field, int method, bool use_private) {
    if (method < 0 || method > 2) throw std::invalid_argument("Método: 0=reduction, 1=atomic, 2=critical");
    const auto* data = field.data();
    const std::size_t count = field.size();
    double sum = 0.0, contribution = 0.0;
    // Con use_private se usa el scratch explícitamente privado; sin él, expresión directa.
    if (method == 0) {
        #pragma omp parallel for default(none) shared(data, count, use_private) private(contribution) reduction(+:sum)
        for (std::size_t i = 0; i < count; ++i)
            sum += use_private ? (contribution = std::norm(data[i])) : std::norm(data[i]);
    } else {
        #pragma omp parallel for default(none) shared(data, count, use_private, method, sum) private(contribution)
        for (std::size_t i = 0; i < count; ++i) {
            const double value = use_private ? (contribution = std::norm(data[i])) : std::norm(data[i]);
            if (method == 1) {
                #pragma omp atomic update
                sum += value;
            } else {
                #pragma omp critical(spectrum_energy)
                { sum += value; }
            }
        }
    }
    return sum;
}

ParsevalResult SpectrumMetrics::parseval(int method) const { return parseval(method, false); }

ParsevalResult SpectrumMetrics::parseval(int method, bool use_private) const {
    ParsevalResult result;
    result.input_energy = energy(original_, method, use_private);
    result.spectrum_energy = energy(other_, method, use_private) / static_cast<double>(original_.size());
    result.absolute_error = std::abs(result.input_energy - result.spectrum_energy);
    result.relative_error = result.input_energy == 0.0 ? result.absolute_error :
        result.absolute_error / result.input_energy;
    return result;
}
