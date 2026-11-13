#include "ImageBuffer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace {

/**
 * @brief Calcula `a * b` verificando desbordamiento en `std::size_t` (y contra `max_size` de `std::vector<float>`).
 * @param a Primer factor (ej. `height`).
 * @param b Segundo factor (ej. `width`).
 * @return Producto `a * b`, o `0` si alguno de los factores es `0`.
 * @throws std::overflow_error si `a * b` desborda `std::size_t` o excede el tamaño máximo asignable.
 */
std::size_t checkedMultiplySize(std::size_t a, std::size_t b, const char* context) {
    if (a == 0 || b == 0) {
        return 0;
    }
    constexpr std::size_t max_elements = std::numeric_limits<std::size_t>::max() / sizeof(float);
    if (a > max_elements / b) {
        throw std::overflow_error(std::string(context) + ": desbordamiento al calcular H * W");
    }
    return a * b;
}

/**
 * @brief Función auxiliar interna para saltar espacios en blanco y comentarios '#' en cabecera PGM.
 */
void skipPgmCommentsAndWhitespace(std::istream& in) {
    while (in >> std::ws) {
        if (in.peek() == '#') {
            std::string comment;
            std::getline(in, comment);
        } else {
            break;
        }
    }
}

} // namespace

ImageBuffer::ImageBuffer(std::size_t height, std::size_t width, float initial_value) {
    resize(height, width, initial_value);
}

float& ImageBuffer::at(std::size_t i, std::size_t j) {
    if (i >= height_ || j >= width_) {
        throw std::out_of_range("ImageBuffer::at: índice (i, j) fuera de rango");
    }
    return data_[i * width_ + j];
}

const float& ImageBuffer::at(std::size_t i, std::size_t j) const {
    if (i >= height_ || j >= width_) {
        throw std::out_of_range("ImageBuffer::at const: índice (i, j) fuera de rango");
    }
    return data_[i * width_ + j];
}

void ImageBuffer::resize(std::size_t new_height, std::size_t new_width, float value) {
    const std::size_t total = checkedMultiplySize(new_height, new_width, "ImageBuffer::resize");
    if (total == 0) {
        // Si cualquiera de las dimensiones es 0, normalizamos a 0 x 0 para que
        // height(), width() y data_.size() sean siempre consistentes.
        data_.clear();
        height_ = 0;
        width_ = 0;
        return;
    }

    // Reservamos primero en un vector temporal para garantizar seguridad fuerte ante excepciones:
    // si std::vector lanza std::bad_alloc, height_ y width_ originales no quedan inconsistentes.
    std::vector<float> new_data(total, value);
    data_ = std::move(new_data);
    height_ = new_height;
    width_ = new_width;
}

void ImageBuffer::fill(float value) noexcept {
    std::fill(data_.begin(), data_.end(), value);
}

void ImageBuffer::fillImpulse(std::size_t impulse_i, std::size_t impulse_j, float amplitude) {
    if (impulse_i >= height_ || impulse_j >= width_) {
        throw std::out_of_range("ImageBuffer::fillImpulse: coordenadas fuera de los límites");
    }
    fill(0.0f);
    data_[impulse_i * width_ + impulse_j] = amplitude;
}

void ImageBuffer::fillCenterImpulse(float amplitude) {
    if (empty()) {
        throw std::runtime_error("ImageBuffer::fillCenterImpulse: el buffer está vacío");
    }
    fillImpulse(height_ / 2, width_ / 2, amplitude);
}

void ImageBuffer::fillVerticalStep(float low_val, float high_val, std::size_t split_col) {
    if (empty()) return;
    const std::size_t split = (split_col == 0) ? (width_ / 2) : split_col;
    for (std::size_t i = 0; i < height_; ++i) {
        float* r = row(i);
        for (std::size_t j = 0; j < width_; ++j) {
            r[j] = (j < split) ? low_val : high_val;
        }
    }
}

void ImageBuffer::fillHorizontalStep(float low_val, float high_val, std::size_t split_row) {
    if (empty()) return;
    const std::size_t split = (split_row == 0) ? (height_ / 2) : split_row;
    for (std::size_t i = 0; i < height_; ++i) {
        const float val = (i < split) ? low_val : high_val;
        float* r = row(i);
        for (std::size_t j = 0; j < width_; ++j) {
            r[j] = val;
        }
    }
}

void ImageBuffer::fillRandom(unsigned int seed, float min_val, float max_val) {
    if (!std::isfinite(min_val) || !std::isfinite(max_val) || min_val > max_val) {
        throw std::invalid_argument("ImageBuffer::fillRandom: rango [min_val, max_val] inválido");
    }
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(min_val, max_val);
    for (float& pixel : data_) {
        pixel = dist(rng);
    }
}

ImageBuffer ImageBuffer::createRandom(std::size_t height, std::size_t width, unsigned int seed) {
    ImageBuffer img(height, width);
    img.fillRandom(seed, 0.0f, 1.0f);
    return img;
}

std::vector<std::size_t> ImageBuffer::standardSizes(bool include_benchmark_sizes) {
    if (include_benchmark_sizes) {
        return {8, 16, 32, 512, 1024, 2048, 4096};
    }
    return {8, 16, 32};
}

void ImageBuffer::readPgm(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("ImageBuffer::readPgm: no se pudo abrir el archivo " + path);
    }

    std::string magic;
    in >> magic;
    if (magic != "P2" && magic != "P5") {
        throw std::runtime_error("ImageBuffer::readPgm: formato PGM no soportado (" + magic + "), se esperaba P2 o P5");
    }

    skipPgmCommentsAndWhitespace(in);
    std::size_t w = 0;
    in >> w;

    skipPgmCommentsAndWhitespace(in);
    std::size_t h = 0;
    in >> h;

    skipPgmCommentsAndWhitespace(in);
    long long maxval = 0;
    in >> maxval;

    if (!in || w == 0 || h == 0 || maxval <= 0 || maxval > 65535) {
        throw std::runtime_error("ImageBuffer::readPgm: cabecera PGM inválida en " + path);
    }

    const std::size_t total = checkedMultiplySize(h, w, "ImageBuffer::readPgm");

    // En formato binario P5 hay exactamente un carácter de espacio en blanco (ej. '\n') tras maxval
    const int sep = in.get();
    if (sep == EOF) {
        throw std::runtime_error("ImageBuffer::readPgm: archivo PGM truncado tras cabecera en " + path);
    }

    std::vector<float> parsed(total, 0.0f);
    const float inv_max = 1.0f / static_cast<float>(maxval);

    if (magic == "P2") {
        for (std::size_t idx = 0; idx < total; ++idx) {
            skipPgmCommentsAndWhitespace(in);
            long long sample = 0;
            if (!(in >> sample)) {
                throw std::runtime_error("ImageBuffer::readPgm: datos P2 truncados o inválidos en " + path);
            }
            if (sample < 0 || sample > maxval) {
                throw std::runtime_error("ImageBuffer::readPgm: muestra P2 fuera de rango [0, maxval] en " + path);
            }
            parsed[idx] = static_cast<float>(sample) * inv_max;
        }
    } else {
        if (maxval < 256) {
            std::vector<unsigned char> raw(total);
            in.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size()));
            if (static_cast<std::size_t>(in.gcount()) != raw.size()) {
                throw std::runtime_error("ImageBuffer::readPgm: datos P5 (8-bit) truncados en " + path);
            }
            for (std::size_t idx = 0; idx < total; ++idx) {
                const long long sample = static_cast<long long>(raw[idx]);
                if (sample > maxval) {
                    throw std::runtime_error("ImageBuffer::readPgm: muestra P5 (8-bit) excede maxval en " + path);
                }
                parsed[idx] = static_cast<float>(sample) * inv_max;
            }
        } else {
            // Soporte PGM de 16 bits (Big-Endian según estándar Netpbm)
            const std::size_t byte_count = checkedMultiplySize(total, 2, "ImageBuffer::readPgm(16-bit)");
            std::vector<unsigned char> raw(byte_count);
            in.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size()));
            if (static_cast<std::size_t>(in.gcount()) != raw.size()) {
                throw std::runtime_error("ImageBuffer::readPgm: datos P5 (16-bit) truncados en " + path);
            }
            for (std::size_t idx = 0; idx < total; ++idx) {
                const long long sample = (static_cast<long long>(raw[2 * idx]) << 8) |
                                         static_cast<long long>(raw[2 * idx + 1]);
                if (sample > maxval) {
                    throw std::runtime_error("ImageBuffer::readPgm: muestra P5 (16-bit) excede maxval en " + path);
                }
                parsed[idx] = static_cast<float>(sample) * inv_max;
            }
        }
    }

    // Actualización atómica del estado tras lectura exitosa
    data_ = std::move(parsed);
    height_ = h;
    width_ = w;
}

void ImageBuffer::writePgm(const std::string& path, bool binary, bool normalize_minmax) const {
    if (empty()) {
        throw std::runtime_error("ImageBuffer::writePgm: no se puede escribir una imagen vacía en " + path);
    }

    std::ofstream out(path, binary ? (std::ios::out | std::ios::binary) : std::ios::out);
    if (!out) {
        throw std::runtime_error("ImageBuffer::writePgm: no se pudo crear el archivo " + path);
    }

    float min_v = 0.0f;
    float max_v = 1.0f;
    if (normalize_minmax) {
        auto [min_it, max_it] = std::minmax_element(data_.begin(), data_.end());
        min_v = *min_it;
        max_v = *max_it;
    }
    const float span = (max_v - min_v > 1e-12f) ? (max_v - min_v) : 1.0f;

    out << (binary ? "P5\n" : "P2\n")
        << width_ << ' ' << height_ << "\n255\n";

    if (binary) {
        std::vector<unsigned char> raw(data_.size());
        for (std::size_t idx = 0; idx < data_.size(); ++idx) {
            float norm = normalize_minmax ? ((data_[idx] - min_v) / span) : data_[idx];
            norm = std::clamp(norm, 0.0f, 1.0f);
            raw[idx] = static_cast<unsigned char>(std::lround(norm * 255.0f));
        }
        out.write(reinterpret_cast<const char*>(raw.data()), static_cast<std::streamsize>(raw.size()));
    } else {
        for (std::size_t i = 0; i < height_; ++i) {
            for (std::size_t j = 0; j < width_; ++j) {
                float norm = normalize_minmax ? ((at(i, j) - min_v) / span) : at(i, j);
                norm = std::clamp(norm, 0.0f, 1.0f);
                const int byte_val = static_cast<int>(std::lround(norm * 255.0f));
                out << byte_val << (j + 1 == width_ ? '\n' : ' ');
            }
        }
    }

    if (!out) {
        throw std::runtime_error("ImageBuffer::writePgm: fallo al escribir datos en " + path);
    }
}

void ImageBuffer::writeDat(const std::string& path) const {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("ImageBuffer::writeDat: no se pudo abrir para escritura " + path);
    }

    out << "# height " << height_ << " width " << width_ << '\n';
    out << std::setprecision(9);
    for (std::size_t i = 0; i < height_; ++i) {
        const float* r = row(i);
        for (std::size_t j = 0; j < width_; ++j) {
            out << r[j] << (j + 1 == width_ ? '\n' : ' ');
        }
    }

    if (!out) {
        throw std::runtime_error("ImageBuffer::writeDat: error durante la escritura de " + path);
    }
}

void ImageBuffer::readDat(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("ImageBuffer::readDat: no se pudo abrir el archivo " + path);
    }

    std::string first_line;
    if (!std::getline(in, first_line)) {
        throw std::runtime_error("ImageBuffer::readDat: archivo vacío " + path);
    }

    std::size_t h = 0;
    std::size_t w = 0;
    {
        std::istringstream header(first_line);
        std::string hash_tok, h_label, w_label;
        if (!(header >> hash_tok >> h_label >> h >> w_label >> w) ||
            hash_tok != "#" || h_label != "height" || w_label != "width" || h == 0 || w == 0) {
            throw std::runtime_error("ImageBuffer::readDat: cabecera inválida en " + path);
        }
    }

    const std::size_t total = checkedMultiplySize(h, w, "ImageBuffer::readDat");
    std::vector<float> parsed(total, 0.0f);
    for (std::size_t idx = 0; idx < total; ++idx) {
        skipPgmCommentsAndWhitespace(in);
        if (!(in >> parsed[idx]) || !std::isfinite(parsed[idx])) {
            throw std::runtime_error("ImageBuffer::readDat: datos insuficientes o corruptos en " + path);
        }
    }

    data_ = std::move(parsed);
    height_ = h;
    width_ = w;
}

std::vector<float> ImageBuffer::makeBoxKernel(int s) {
    if (s < 1 || (s % 2 == 0)) {
        throw std::invalid_argument("ImageBuffer::makeBoxKernel: el lado s debe ser impar y positivo");
    }
    const std::size_t side = static_cast<std::size_t>(s);
    const std::size_t count = checkedMultiplySize(side, side, "ImageBuffer::makeBoxKernel");
    const float weight = static_cast<float>(1.0 / static_cast<double>(count));
    return std::vector<float>(count, weight);
}

std::vector<float> ImageBuffer::makeGaussianKernel(int s, float sigma) {
    if (s < 1 || (s % 2 == 0)) {
        throw std::invalid_argument("ImageBuffer::makeGaussianKernel: el lado s debe ser impar y positivo");
    }
    if (!std::isfinite(sigma) || sigma < 0.0f) {
        throw std::invalid_argument("ImageBuffer::makeGaussianKernel: sigma debe ser finito y >= 0");
    }

    const std::size_t side = static_cast<std::size_t>(s);
    const std::size_t count = checkedMultiplySize(side, side, "ImageBuffer::makeGaussianKernel");

    const double sigma_d = (sigma == 0.0f)
        ? (static_cast<double>(s) / 3.0)
        : static_cast<double>(sigma);

    const std::int64_t r = static_cast<std::int64_t>(s - 1) / 2;
    std::vector<double> weights(count, 0.0);
    const double two_sigma_sq = 2.0 * sigma_d * sigma_d;
    double sum = 0.0;

    // Si two_sigma_sq es extremadamente pequeño (sigma positivo muy cercano a 0),
    // para u=0, v=0 el numerador es 0.0 y garantizamos que el centro valga exactamente 1.0
    // sin producir 0.0/0.0 (NaN), mientras que fuera del centro decae a 0.0.
    for (std::int64_t u = -r; u <= r; ++u) {
        const std::size_t row_offset = static_cast<std::size_t>(u + r) * side;
        for (std::int64_t v = -r; v <= r; ++v) {
            const std::int64_t dist_sq = u * u + v * v;
            double val = 0.0;
            if (dist_sq == 0) {
                val = 1.0;
            } else if (two_sigma_sq > 0.0) {
                val = std::exp(-static_cast<double>(dist_sq) / two_sigma_sq);
            }
            weights[row_offset + static_cast<std::size_t>(v + r)] = val;
            sum += val;
        }
    }

    if (!(sum > 0.0) || !std::isfinite(sum)) {
        throw std::runtime_error("ImageBuffer::makeGaussianKernel: suma de pesos inválida");
    }

    std::vector<float> k(count, 0.0f);
    for (std::size_t idx = 0; idx < count; ++idx) {
        k[idx] = static_cast<float>(weights[idx] / sum);
    }
    return k;
}
