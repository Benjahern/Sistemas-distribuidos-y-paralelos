#ifndef BUTTERFLY1D_H
#define BUTTERFLY1D_H

#include <cstddef>
#include <complex>

/**
 * @brief Rol 2: Núcleo de mariposa FFT 1D radix-2 (in-place Cooley-Tukey y Stockham).
 *
 */
class Butterfly1D {
public:
    Butterfly1D() = default;
    // Contrato del núcleo usado por Fft2D: inverse cambia el signo, sin escalar.
    // Fft2D aplica 1/(M*N) después de completar ambas pasadas inversas.
    Butterfly1D(std::complex<double>* data, size_t n, bool inverse = false);
    ~Butterfly1D() = default;

    void setData(std::complex<double>* data, size_t n, bool inverse = false);

    // Métodos obligatorios requeridos por AGENTS.md
    void transform();
    void transform(int schedule_type);
    void transform(int schedule_type, int chunk_size);
    void transformCollapse();
    void transformStages();
    void transform(int layout, int schedule_type, int chunk_size);

private:
    std::complex<double>* data_{nullptr};
    size_t n_{0};
    bool inverse_{false};
};

#endif // BUTTERFLY1D_H
