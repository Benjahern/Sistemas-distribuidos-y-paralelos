#include "Butterfly1D.h"
#include "ComplexField.h"
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <vector>

// ============================================================================
// CONSTRUCTORES Y MÉTODOS DE CONFIGURACIÓN
// ============================================================================

/**
 * @brief Constructor parametrizado.
 * @param data Entrada: Puntero al arreglo de números complejos de tamaño n.
 * @param n Entrada: Tamaño del arreglo (debe ser potencia de 2).
 * @param inverse Entrada: Dirección de la FFT (false = directa, true = inversa).
 */
Butterfly1D::Butterfly1D(std::complex<double> *data, size_t n, bool inverse)
    : data_(data), n_(n), inverse_(inverse) {}

/**
 * @brief Configura o actualiza el puntero de datos y parámetros del procesador 1D.
 * @param data Entrada: Puntero al arreglo de números complejos de tamaño n.
 * @param n Entrada: Tamaño del arreglo.
 * @param inverse Entrada: Dirección de la FFT (false = directa, true = inversa).
 * @return Salida: void.
 */
void Butterfly1D::setData(std::complex<double> *data, size_t n, bool inverse) {
  data_ = data;
  n_ = n;
  inverse_ = inverse;
}

// ============================================================================
// DELEGACIONES DE MÉTODOS DE TRANSFORMADA
// ============================================================================

/**
 * @brief Ejecuta la transformada 1D en modo in-place Cooley-Tukey con schedule estático por defecto.
 * @return Salida: void.
 */
void Butterfly1D::transform() { transform(0, 0, 0); }

/**
 * @brief Ejecuta la transformada 1D en modo Cooley-Tukey especificando el tipo de schedule.
 * @param schedule_type Entrada: Tipo de schedule de OpenMP (0=static, 1=dynamic, 2=guided).
 * @return Salida: void.
 */
void Butterfly1D::transform(int schedule_type) { transform(0, schedule_type, 0); }

/**
 * @brief Ejecuta la transformada 1D especificando el tipo de schedule y chunk size.
 * @param schedule_type Entrada: Tipo de schedule (0=static, 1=dynamic, 2=guided).
 * @param chunk_size Entrada: Tamaño del bloque de iteraciones asignado a cada hilo.
 * @return Salida: void.
 */
void Butterfly1D::transform(int schedule_type, int chunk_size) {
  transform(0, schedule_type, chunk_size);
}

/**
 * @brief Ejecuta el algoritmo Stockham aplicando la cláusula collapse(2) sobre los bucles de mariposa.
 * @return Salida: void.
 */
void Butterfly1D::transformCollapse() { transformStockham(0, 0, true); }

/**
 * @brief Ejecuta el algoritmo Cooley-Tukey en una única región paralela con barrera explícita entre etapas.
 * @return Salida: void.
 */
void Butterfly1D::transformStages() { transformCooleyTukey(0, 0, true); }

/**
 * @brief Método principal parametrizado que delega la ejecución al algoritmo correspondiente.
 * @param layout Entrada: 0 = Cooley-Tukey in-place, 1 = Stockham out-of-place.
 * @param schedule_type Entrada: Schedule de OpenMP (0=static, 1=dynamic, 2=guided).
 * @param chunk_size Entrada: Tamaño de chunk para OpenMP.
 * @return Salida: void.
 */
void Butterfly1D::transform(int layout, int schedule_type, int chunk_size) {
  if (layout == 1) {
    transformStockham(schedule_type, chunk_size, false);
  } else {
    transformCooleyTukey(schedule_type, chunk_size, false);
  }
}

// ============================================================================
// HELPER PARA DISTRIBUCIÓN DE SCHEDULES OPENMP SIN DUPLICAR LÓGICA
// ============================================================================

template <typename F>
static void dispatchSchedule1D(size_t total_iters, int schedule_type, int chunk_size, const F &op) {
  if (schedule_type == 1) { // Dynamic
    if (chunk_size > 0) {
      #pragma omp parallel for schedule(dynamic, chunk_size) default(none) shared(total_iters, op, chunk_size)
      for (size_t idx = 0; idx < total_iters; ++idx) {
        op(idx);
      }
    } else {
      #pragma omp parallel for schedule(dynamic) default(none) shared(total_iters, op)
      for (size_t idx = 0; idx < total_iters; ++idx) {
        op(idx);
      }
    }
  } else if (schedule_type == 2) { // Guided
    if (chunk_size > 0) {
      #pragma omp parallel for schedule(guided, chunk_size) default(none) shared(total_iters, op, chunk_size)
      for (size_t idx = 0; idx < total_iters; ++idx) {
        op(idx);
      }
    } else {
      #pragma omp parallel for schedule(guided) default(none) shared(total_iters, op)
      for (size_t idx = 0; idx < total_iters; ++idx) {
        op(idx);
      }
    }
  } else { // Static (Predeterminado)
    if (chunk_size > 0) {
      #pragma omp parallel for schedule(static, chunk_size) default(none) shared(total_iters, op, chunk_size)
      for (size_t idx = 0; idx < total_iters; ++idx) {
        op(idx);
      }
    } else {
      #pragma omp parallel for schedule(static) default(none) shared(total_iters, op)
      for (size_t idx = 0; idx < total_iters; ++idx) {
        op(idx);
      }
    }
  }
}

void Butterfly1D::cooleyTukeyButterflyAt(size_t idx, size_t m, size_t m2,
                                         double sign, double pi) {
  size_t group_idx = idx / m2;
  size_t j = idx % m2;
  size_t k = group_idx * m;
  std::complex<double> w = std::polar(1.0, sign * 2.0 * pi * static_cast<double>(j) / static_cast<double>(m));
  butterfly(data_[k + j], data_[k + j + m2], w, data_[k + j], data_[k + j + m2]);
}

void Butterfly1D::stockhamButterflyAt(const std::complex<double> *src,
                                      std::complex<double> *dst,
                                      size_t k, size_t j, size_t g, size_t m,
                                      double sign, double pi) {
  std::complex<double> omega = std::polar(1.0, sign * 2.0 * pi * static_cast<double>(k) / static_cast<double>(m));
  size_t i0 = j + 2 * k * g;
  size_t i1 = i0 + g;
  size_t o0 = j + k * g;
  size_t o1 = o0 + n_ / 2;
  butterfly(src[i0], src[i1], omega, dst[o0], dst[o1]);
}

// ============================================================================
// IMPLEMENTACIÓN: MODO 0 (COOLEY-TUKEY IN-PLACE)
// ============================================================================

void Butterfly1D::transformCooleyTukey(int schedule_type, int chunk_size,
                                       bool use_single_region) {
  if (n_ <= 1 || !data_) return;

  // Calcular p = log2(N)
  size_t p = 0;
  while ((size_t(1) << p) < n_) ++p;

  // Paso 1: Permutación Bit-Reversal previa (in-place) provista por Rol 1 (ComplexField)
  ComplexField::bitReversePermute(data_, n_);

  const double pi = 3.14159265358979323846;
  const double sign = inverse_ ? 1.0 : -1.0;
  const size_t total_bf = n_ / 2;

  // CASO 1: Región paralela única con barrera explícita entre etapas (transformStages)
  if (use_single_region) {
    #pragma omp parallel default(none) shared(p, n_, sign, pi, total_bf)
    {
      for (size_t s = 1; s <= p; ++s) {
        size_t m = size_t(1) << s;
        size_t m2 = m >> 1;

        #pragma omp for schedule(static)
        for (size_t idx = 0; idx < total_bf; ++idx) {
          cooleyTukeyButterflyAt(idx, m, m2, sign, pi);
        }
        #pragma omp barrier
      }
    }
    return;
  }

  // CASO 2: Etapas con schedules OpenMP delegando en dispatchSchedule1D
  for (size_t s = 1; s <= p; ++s) {
    size_t m = size_t(1) << s;
    size_t m2 = m >> 1;

    auto op = [this, m, m2, sign, pi](size_t idx) {
      cooleyTukeyButterflyAt(idx, m, m2, sign, pi);
    };
    dispatchSchedule1D(total_bf, schedule_type, chunk_size, op);
  }
}

// ============================================================================
// IMPLEMENTACIÓN: MODO 1 (STOCKHAM OUT-OF-PLACE)
// ============================================================================

void Butterfly1D::transformStockham(int schedule_type, int chunk_size,
                                    bool use_collapse) {
  if (n_ <= 1 || !data_) return;

  size_t p = 0;
  while ((size_t(1) << p) < n_) ++p;

  std::vector<std::complex<double>> src(data_, data_ + n_);
  std::vector<std::complex<double>> dst(n_);

  const double pi = 3.14159265358979323846;
  const double sign = inverse_ ? 1.0 : -1.0;

  for (size_t s = 0; s < p; ++s) {
    size_t l = size_t(1) << s;
    size_t g = n_ / (size_t(1) << (s + 1));
    size_t m = size_t(1) << (s + 1);

    // CASO 1: Paralelismo usando collapse(2) sobre los dos bucles k y j
    if (use_collapse) {
      #pragma omp parallel for collapse(2) default(none) shared(src, dst, l, g, m, sign, pi)
      for (size_t k = 0; k < l; ++k) {
        for (size_t j = 0; j < g; ++j) {
          stockhamButterflyAt(src.data(), dst.data(), k, j, g, m, sign, pi);
        }
      }
    } else {
      // CASO 2: Schedules configurables sobre el bucle externo k delegando en dispatchSchedule1D
      auto op = [this, &src, &dst, g, m, sign, pi](size_t k) {
        for (size_t j = 0; j < g; ++j) {
          stockhamButterflyAt(src.data(), dst.data(), k, j, g, m, sign, pi);
        }
      };
      dispatchSchedule1D(l, schedule_type, chunk_size, op);
    }
    src.swap(dst);
  }

  for (size_t i = 0; i < n_; ++i) {
    data_[i] = src[i];
  }
}

// ============================================================================
// DEMOSTRACIÓN DE CLÁUSULAS OPENMP REQUERIDAS
// ============================================================================

/**
 * @brief Demuestra la cláusula '#pragma omp single'.
 * @return Salida: std::vector<std::complex<double>> de factores twiddle calculados por un único hilo.
 */
std::vector<std::complex<double>> Butterfly1D::initTwiddlesSingle() const {
  std::vector<std::complex<double>> twiddles(n_);
  const double pi = 3.14159265358979323846;
  const double sign = inverse_ ? 1.0 : -1.0;

  #pragma omp parallel default(none) shared(twiddles, sign, pi)
  {
    // Solo un hilo del equipo ejecuta el bucle de inicialización; los demás esperan al final
    #pragma omp single
    {
      for (size_t k = 0; k < n_; ++k) {
        twiddles[k] = std::polar(1.0, sign * 2.0 * pi * static_cast<double>(k) / static_cast<double>(n_));
      }
    }
  }
  return twiddles;
}

/**
 * @brief Demuestra la cláusula 'firstprivate(acc)'.
 * @param base_val Entrada: Valor inicial a sumar a cada elemento.
 * @return Salida: void (modifica data_).
 */
void Butterfly1D::accumulateFirstprivate(double base_val) {
  if (!data_ || n_ == 0) return;
  double acc = base_val; // Variable local inicializada
  
  // firstprivate hace que cada hilo reciba una copia de 'acc' inicializada con el valor base_val
  #pragma omp parallel for firstprivate(acc) default(none) shared(data_)
  for (size_t i = 0; i < n_; ++i) {
    data_[i] += acc;
  }
}

/**
 * @brief Demuestra la cláusula 'lastprivate(last_stage)'.
 * @return Salida: int representando el índice de la última iteración efectuada.
 */
int Butterfly1D::stageIndexLastprivate() const {
  size_t p = 0;
  while ((size_t(1) << p) < n_) ++p;
  int last_stage = 0;
  
  // lastprivate copia el valor de 'last_stage' correspondiente a la última iteración del bucle
  // hacia la variable fuera de la región paralela.
  #pragma omp parallel for lastprivate(last_stage) default(none) shared(p)
  for (int s = 1; s <= static_cast<int>(p); ++s) {
    last_stage = s;
  }
  return last_stage;
}
