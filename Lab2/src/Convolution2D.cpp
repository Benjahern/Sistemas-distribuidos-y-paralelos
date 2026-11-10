#include "Convolution2D.h"

#include <stdexcept>

Convolution2D::Convolution2D(const ImageBuffer& input) : input_(input) {}

void Convolution2D::setInput(const ImageBuffer& input) {
    input_ = input;
}

void Convolution2D::convolveCpu(const float* k, int s) {
    if (k == nullptr) {
        throw std::invalid_argument("Convolution2D::convolveCpu: el puntero al kernel k es nulo");
    }
    if (s < 1 || (s % 2 == 0)) {
        throw std::invalid_argument("Convolution2D::convolveCpu: el lado s del kernel debe ser impar y positivo");
    }

    const std::size_t H = input_.height();
    const std::size_t W = input_.width();
    output_cpu_.resize(H, W, 0.0f);
    if (input_.empty()) {
        return;
    }

    const int r = (s - 1) / 2;
    const float* in_ptr = input_.data();
    float* out_ptr = output_cpu_.data();

    // Política de borde única: Zero-padding (lecturas fuera de [0, H) x [0, W) aportan 0.0f)
    for (std::size_t i = 0; i < H; ++i) {
        for (std::size_t j = 0; j < W; ++j) {
            float acc = 0.0f;
            for (int u = -r; u <= r; ++u) {
                const long ni = static_cast<long>(i) + u;
                if (ni < 0 || ni >= static_cast<long>(H)) {
                    continue; // Zero-padding en filas
                }
                const float* in_row = in_ptr + static_cast<std::size_t>(ni) * W;
                const float* k_row = k + (u + r) * s;
                for (int v = -r; v <= r; ++v) {
                    const long nj = static_cast<long>(j) + v;
                    if (nj >= 0 && nj < static_cast<long>(W)) {
                        acc += k_row[v + r] * in_row[static_cast<std::size_t>(nj)];
                    }
                }
            }
            out_ptr[i * W + j] = acc;
        }
    }
}

void Convolution2D::convolveGpu(const float* k, int s) {
    // Sobrecarga simple: delega en la variante por defecto (VARIANT_SHARED = 1)
    convolveGpu(k, s, VARIANT_SHARED, dim3(16, 16));
}

void Convolution2D::convolveGpu(const float* k, int s, int variant) {
    // Sobrecarga intermedia: delega en la versión parametrizada con bloque por defecto 16x16
    convolveGpu(k, s, variant, dim3(16, 16));
}

bool Convolution2D::isGpuAvailable() noexcept {
#if defined(__CUDACC__)
    int device_count = 0;
    const cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count <= 0) {
        // Limpiar cualquier estado de error pendiente del runtime CUDA
        cudaGetLastError();
        return false;
    }
    return true;
#else
    return false;
#endif
}

#if !defined(__CUDACC__)
void Convolution2D::convolveGpu(const float* /*k*/, int /*s*/, int /*variant*/, dim3 /*block*/) {
    throw std::runtime_error(
        "Convolution2D::convolveGpu: binario compilado sin soporte CUDA (__CUDACC__ no definido)");
}
#endif

