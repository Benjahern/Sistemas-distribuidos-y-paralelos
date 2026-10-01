#ifndef BUTTERFLY1D_H
#define BUTTERFLY1D_H

#include <complex>
#include <cstddef>
#include <vector>

/**
 * @brief Rol 2: Núcleo de mariposa FFT 1D radix-2 (in-place Cooley-Tukey y
 * Stockham).
 *
 * Contrato de escala: la transformada 1D NO normaliza en ningun signo.
 * inverse = true cambia únicamente el signo del exponente exp(+2*pi*i*r/m).
 * La normalización 1/(M*N) la aplica Fft2D una sola vez al final.
 */
class Butterfly1D {
public:
  Butterfly1D() = default;
  Butterfly1D(std::complex<double> *data, size_t n, bool inverse = false);
  ~Butterfly1D() = default;

  void setData(std::complex<double> *data, size_t n, bool inverse = false);

  void transform();
  void transform(int schedule_type);
  void transform(int schedule_type, int chunk_size);
  void transformCollapse();
  void transformStages();
  void transform(int layout, int schedule_type, int chunk_size);

  // Métodos para demostrar cláusulas OpenMP requeridas en la evaluación
  std::vector<std::complex<double>> initTwiddlesSingle() const;
  void accumulateFirstprivate(double base_val);
  int stageIndexLastprivate() const;

private:
  std::complex<double> *data_{nullptr};
  size_t n_{0};
  bool inverse_{false};

  void transformCooleyTukey(int schedule_type, int chunk_size,
                            bool use_single_region = false);
  void transformStockham(int schedule_type, int chunk_size,
                         bool use_collapse = false);
};

#endif // BUTTERFLY1D_H
