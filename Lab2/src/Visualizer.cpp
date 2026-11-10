#include "Visualizer.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>

ImageBuffer Visualizer::computeAbsoluteErrorMap(const ImageBuffer& cpu_buf, const ImageBuffer& gpu_buf) {
    if (cpu_buf.height() != gpu_buf.height() || cpu_buf.width() != gpu_buf.width()) {
        throw std::invalid_argument("Visualizer::computeAbsoluteErrorMap: las dimensiones CPU y GPU no coinciden");
    }

    ImageBuffer err(cpu_buf.height(), cpu_buf.width(), 0.0f);
    const float* c = cpu_buf.data();
    const float* g = gpu_buf.data();
    float* e = err.data();
    const std::size_t total = cpu_buf.size();

    for (std::size_t idx = 0; idx < total; ++idx) {
        e[idx] = std::fabs(c[idx] - g[idx]);
    }
    return err;
}

void Visualizer::writeCpuGpuErrorDat(const ImageBuffer& cpu_buf,
                                     const ImageBuffer& gpu_buf,
                                     const std::string& path) {
    const ImageBuffer err = computeAbsoluteErrorMap(cpu_buf, gpu_buf);

    float max_err = 0.0f;
    double sum_err = 0.0;
    const float* e = err.data();
    for (std::size_t idx = 0; idx < err.size(); ++idx) {
        max_err = std::max(max_err, e[idx]);
        sum_err += static_cast<double>(e[idx]);
    }
    const double mean_err = err.empty() ? 0.0 : (sum_err / static_cast<double>(err.size()));

    const std::filesystem::path out_path(path);
    if (out_path.has_parent_path()) {
        std::filesystem::create_directories(out_path.parent_path());
    }

    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Visualizer::writeCpuGpuErrorDat: no se pudo abrir para escritura " + path);
    }

    // Primera línea compatible con ImageBuffer::readDat (# height <H> width <W>)
    // Segunda línea con metadatos estadísticos del error absoluto
    out << "# height " << err.height() << " width " << err.width() << '\n'
        << "# max_abs_err " << std::setprecision(9) << max_err
        << " mean_abs_err " << mean_err << '\n';

    for (std::size_t i = 0; i < err.height(); ++i) {
        const float* r = err.row(i);
        for (std::size_t j = 0; j < err.width(); ++j) {
            out << r[j] << (j + 1 == err.width() ? '\n' : ' ');
        }
    }

    if (!out) {
        throw std::runtime_error("Visualizer::writeCpuGpuErrorDat: fallo escribiendo " + path);
    }
}

void Visualizer::exportSobelPanelData(const ImageBuffer& original,
                                      const ImageBuffer& blur_s3,
                                      const ImageBuffer& gx,
                                      const ImageBuffer& gy,
                                      const ImageBuffer& magnitude,
                                      const std::string& output_dir) {
    const std::filesystem::path dir(output_dir);
    std::filesystem::create_directories(dir);

    // Remover figura previa para evitar que quede desincronizada respecto a los nuevos .dat
    std::filesystem::remove(dir / "sobel_panel.png");

    // 1. Exportar matrices exactas en formato .dat para plot_results.py
    original.writeDat((dir / "panel_original.dat").string());
    blur_s3.writeDat((dir / "panel_blur_s3.dat").string());
    gx.writeDat((dir / "panel_gx.dat").string());
    gy.writeDat((dir / "panel_gy.dat").string());
    magnitude.writeDat((dir / "panel_mag.dat").string());

    // 2. Exportar imágenes PGM para inspección directa (original y blur en [0,1]; gradientes reescalados)
    original.writePgm((dir / "panel_original.pgm").string(), true, false);
    blur_s3.writePgm((dir / "panel_blur_s3.pgm").string(), true, false);
    gx.writePgm((dir / "panel_gx.pgm").string(), true, true);
    gy.writePgm((dir / "panel_gy.pgm").string(), true, true);
    magnitude.writePgm((dir / "panel_mag.pgm").string(), true, true);
}

