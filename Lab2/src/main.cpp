#include "Benchmark.h"
#include "Convolution2D.h"
#include "ImageBuffer.h"
#include "Metrics.h"
#include "Sobel.h"
#include "Visualizer.h"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

/**
 * @brief Genera una imagen sintética de demostración con figuras geométricas (círculo, rectángulo
 *        y rampa suave) para el panel Sobel cuando el usuario no especifica `--input <archivo.pgm>`.
 * @param height Altura de la imagen.
 * @param width Ancho de la imagen.
 * @return `ImageBuffer` en el rango [0.0f, 1.0f].
 */
ImageBuffer makeSyntheticDemoImage(std::size_t height, std::size_t width) {
    ImageBuffer img(height, width, 0.1f);
    const float cy = static_cast<float>(height) * 0.5f;
    const float cx = static_cast<float>(width) * 0.5f;
    const float radius = static_cast<float>((height < width ? height : width)) * 0.28f;

    for (std::size_t i = 0; i < height; ++i) {
        for (std::size_t j = 0; j < width; ++j) {
            const float dy = static_cast<float>(i) - cy;
            const float dx = static_cast<float>(j) - cx;
            const float dist = std::sqrt(dx * dx + dy * dy);
            float val = 0.12f + 0.15f * (static_cast<float>(j) / static_cast<float>(width));
            if (dist < radius) {
                val = 0.85f;
            }
            if (i > height / 6 && i < height / 3 && j > width / 6 && j < width / 3) {
                val = 0.95f;
            }
            img(i, j) = val;
        }
    }
    return img;
}

void printUsage(const char* prog) {
    std::cout << "Uso: " << prog << " [comando] [opciones]\n"
              << "Comandos:\n"
              << "  demo [--input <imagen.pgm>] [--output <dir>] [--size <N>]\n"
              << "       Ejecuta el pipeline de Blur (s=3) y Sobel (G_x, G_y, G) y exporta\n"
              << "       los archivos del panel Sobel y cpu_gpu_error.dat en <dir> (por defecto results/).\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string mode = (argc > 1) ? argv[1] : "demo";
        std::string input_pgm;
        std::string output_dir = "results";
        std::size_t size = 128;

        for (int i = 2; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--input" && i + 1 < argc) {
                input_pgm = argv[++i];
            } else if (arg == "--output" && i + 1 < argc) {
                output_dir = argv[++i];
            } else if (arg == "--size" && i + 1 < argc) {
                size = static_cast<std::size_t>(std::stoul(argv[++i]));
            } else if (arg == "--help" || arg == "-h") {
                printUsage(argv[0]);
                return 0;
            } else {
                throw std::invalid_argument("Opción desconocida o incompleta: " + arg);
            }
        }

        if (mode == "--help" || mode == "-h") {
            printUsage(argv[0]);
            return 0;
        }
        if (mode != "demo") {
            throw std::invalid_argument("Modo no soportado: " + mode + " (use 'demo')");
        }

        ImageBuffer img;
        if (!input_pgm.empty()) {
            img.readPgm(input_pgm);
            std::cout << "[Rol 1] Imagen PGM cargada desde '" << input_pgm << "' ("
                      << img.height() << "x" << img.width() << ")\n";
        } else {
            const std::string default_pgm = "data/sample.pgm";
            if (std::filesystem::exists(default_pgm)) {
                img.readPgm(default_pgm);
                std::cout << "[Rol 1] Imagen PGM por defecto cargada desde '" << default_pgm << "' ("
                          << img.height() << "x" << img.width() << ")\n";
            } else {
                img = makeSyntheticDemoImage(size, size);
                std::cout << "[Rol 1] Imagen sintética generada (" << img.height() << "x" << img.width() << ")\n";
            }
        }

        // 1. Suavizado (Blur s=3 normalizado)
        const std::vector<float> k_blur3 = ImageBuffer::makeGaussianKernel(3);
        Convolution2D conv(img);
        conv.convolveCpu(k_blur3.data(), 3);
        const ImageBuffer& blur_cpu = conv.outputCpu();

        // 2. Detección de bordes de Sobel
        Sobel sobel(img);
        sobel.computeGxGyCpu();
        sobel.computeMagnitudeCpu();

        // 3. Exportar panel Sobel (Rol 5)
        Visualizer::exportSobelPanelData(img, blur_cpu,
                                         sobel.gxCpu(), sobel.gyCpu(), sobel.magnitudeCpu(),
                                         output_dir);

        // 4. Si hay GPU disponible, ejecutar también en GPU y generar cpu_gpu_error.dat real
        if (Convolution2D::isGpuAvailable()) {
            conv.convolveGpu(k_blur3.data(), 3, Convolution2D::VARIANT_SHARED);
            Visualizer::writeCpuGpuErrorDat(blur_cpu, conv.outputGpu(),
                                            output_dir + "/cpu_gpu_error.dat");
            std::cout << "[Rol 5] Panel Sobel y cpu_gpu_error.dat (CPU vs GPU) exportados en '"
                      << output_dir << "/'\n";
        } else {
            // En entornos sin GPU física (CI host), exportar mapa de error contra la referencia
            Visualizer::writeCpuGpuErrorDat(blur_cpu, blur_cpu,
                                            output_dir + "/cpu_gpu_error.dat");
            std::cout << "[Rol 5] Panel Sobel y cpu_gpu_error.dat (modo CPU) exportados en '"
                      << output_dir << "/'\n";
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return 1;
    }
}

