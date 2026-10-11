#include "Visualizer.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>

std::vector<double> Visualizer::magnitudeNoWait(const ComplexField& spectrum) {
    std::vector<double> magnitudes(spectrum.size());
    const auto* data = spectrum.data();
    const std::size_t count = spectrum.size();
    #pragma omp parallel default(none) shared(magnitudes, data, count)
    {
        // No hay consumidor dentro de parallel; su final sincroniza antes del retorno.
        #pragma omp for nowait
        for (std::size_t i = 0; i < count; ++i) magnitudes[i] = std::abs(data[i]);
    }
    return magnitudes;
}

void Visualizer::writeSpectrum(const ComplexField& spectrum, const std::string& path) {
    const auto magnitudes = magnitudeNoWait(spectrum);
    // Una figura de la exportación anterior ya no representa estos datos.
    auto figure = std::filesystem::path(path);
    figure.replace_extension(".png");
    std::filesystem::remove(figure);
    std::ofstream out(path);
    if (!out) throw std::runtime_error("No se pudo escribir " + path);
    out << "# k l magnitude (escala lineal, orden natural)\n" << std::setprecision(17);
    for (std::size_t k = 0; k < spectrum.rows(); ++k)
        for (std::size_t l = 0; l < spectrum.cols(); ++l)
            out << k << ' ' << l << ' ' << magnitudes[k*spectrum.cols()+l] << '\n';
    if (!out) throw std::runtime_error("Error escribiendo " + path);
}
