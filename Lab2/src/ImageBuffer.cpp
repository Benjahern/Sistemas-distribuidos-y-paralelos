#include "ImageBuffer.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>

ImageBuffer::ImageBuffer(std::size_t height, std::size_t width, float initial_value)
    : height_(height), width_(width), data_(height * width, initial_value) {}

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
    height_ = new_height;
    width_ = new_width;
    data_.assign(new_height * new_width, value);
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
    if (min_val > max_val) {
        throw std::invalid_argument("ImageBuffer::fillRandom: min_val no puede ser mayor que max_val");
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

namespace {
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
    int maxval = 0;
    in >> maxval;

    if (!in || w == 0 || h == 0 || maxval <= 0 || maxval > 65535) {
        throw std::runtime_error("ImageBuffer::readPgm: cabecera PGM inválida en " + path);
    }

    // En formato binario P5 hay exactamente un carácter de espacio en blanco (ej. '\n') tras maxval
    in.get();

    resize(h, w, 0.0f);
    const float inv_max = 1.0f / static_cast<float>(maxval);

    if (magic == "P2") {
        for (std::size_t idx = 0; idx < data_.size(); ++idx) {
            skipPgmCommentsAndWhitespace(in);
            int sample = 0;
            if (!(in >> sample)) {
                throw std::runtime_error("ImageBuffer::readPgm: datos P2 truncados en " + path);
            }
            data_[idx] = std::clamp(static_cast<float>(sample) * inv_max, 0.0f, 1.0f);
        }
    } else {
        if (maxval < 256) {
            std::vector<unsigned char> raw(data_.size());
            in.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size()));
            if (static_cast<std::size_t>(in.gcount()) != raw.size()) {
                throw std::runtime_error("ImageBuffer::readPgm: datos P5 (8-bit) truncados en " + path);
            }
            for (std::size_t idx = 0; idx < data_.size(); ++idx) {
                data_[idx] = static_cast<float>(raw[idx]) * inv_max;
            }
        } else {
            // Soporte PGM de 16 bits (Big-Endian según estándar Netpbm)
            std::vector<unsigned char> raw(data_.size() * 2);
            in.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size()));
            if (static_cast<std::size_t>(in.gcount()) != raw.size()) {
                throw std::runtime_error("ImageBuffer::readPgm: datos P5 (16-bit) truncados en " + path);
            }
            for (std::size_t idx = 0; idx < data_.size(); ++idx) {
                const int sample = (static_cast<int>(raw[2 * idx]) << 8) | static_cast<int>(raw[2 * idx + 1]);
                data_[idx] = static_cast<float>(sample) * inv_max;
            }
        }
    }
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

    resize(h, w, 0.0f);
    for (std::size_t idx = 0; idx < data_.size(); ++idx) {
        skipPgmCommentsAndWhitespace(in);
        if (!(in >> data_[idx])) {
            throw std::runtime_error("ImageBuffer::readDat: datos insuficientes o corruptos en " + path);
        }
    }
}

std::vector<float> ImageBuffer::makeBoxKernel(int s) {
    if (s < 1 || (s % 2 == 0)) {
        throw std::invalid_argument("ImageBuffer::makeBoxKernel: el lado s debe ser impar y positivo");
    }
    const std::size_t count = static_cast<std::size_t>(s) * static_cast<std::size_t>(s);
    const float weight = 1.0f / static_cast<float>(count);
    return std::vector<float>(count, weight);
}

std::vector<float> ImageBuffer::makeGaussianKernel(int s, float sigma) {
    if (s < 1 || (s % 2 == 0)) {
        throw std::invalid_argument("ImageBuffer::makeGaussianKernel: el lado s debe ser impar y positivo");
    }
    if (sigma <= 0.0f) {
        sigma = static_cast<float>(s) / 3.0f;
    }
    const int r = (s - 1) / 2;
    std::vector<float> k(static_cast<std::size_t>(s * s), 0.0f);
    const float two_sigma_sq = 2.0f * sigma * sigma;
    float sum = 0.0f;

    for (int u = -r; u <= r; ++u) {
        for (int v = -r; v <= r; ++v) {
            const float val = std::exp(-static_cast<float>(u * u + v * v) / two_sigma_sq);
            k[static_cast<std::size_t>((u + r) * s + (v + r))] = val;
            sum += val;
        }
    }

    for (float& coeff : k) {
        coeff /= sum;
    }
    return k;
}
