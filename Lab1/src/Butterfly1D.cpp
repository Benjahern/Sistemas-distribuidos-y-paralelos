#include "Butterfly1D.h"
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
// FUNCIONES AUXILIARES PRIVADAS
// ============================================================================

/**
 * @brief Función auxiliar estática para calcular la permutación Bit-Reversal.
 * @param i Entrada: Índice original de 0 a N-1.
 * @param log2n Entrada: Número de bits de representación p = log2(N).
 * @return Salida: size_t representando el índice con los bits invertidos.
 * 
 * @details Por ejemplo, para N=8 (p=3 bits), la posición 1 (001 en binario)
 * pasa a ser 4 (100 en binario). Esta permutación reordena los datos de entrada
 * para que el algoritmo Cooley-Tukey in-place pueda combinar parejas contiguas
 * etapa por etapa.
 */
static size_t bitReverse(size_t i, size_t log2n) {
  size_t rev = 0;
  for (size_t bit = 0; bit < log2n; ++bit) {
    if ((i >> bit) & 1) {
      rev |= (size_t(1) << (log2n - 1 - bit));
    }
  }
  return rev;
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
// IMPLEMENTACIÓN DETALLADA: MODO 0 (COOLEY-TUKEY IN-PLACE)
// ============================================================================

/**
 * @brief Implementación interna del algoritmo Radix-2 Cooley-Tukey in-place.
 * @param schedule_type Entrada: Tipo de schedule OpenMP (0=static, 1=dynamic, 2=guided).
 * @param chunk_size Entrada: Tamaño de chunk para las iteraciones de hilos.
 * @param use_single_region Entrada: true para ejecutar dentro de una única región paralela con '#pragma omp barrier'.
 * @return Salida: void (los resultados transformados modifican el arreglo data_ in-place).
 *
 * @details EXPLICACIÓN PASO A PASO:
 * 1. Validación de entrada y cálculo de p = log2(N).
 * 2. Permutación Bit-Reversal: Se intercambian los elementos data_[i] y data_[bitReverse(i)]
 *    únicamente si i < bitReverse(i) para evitar intercambiar dos veces el mismo par.
 * 3. Bucle de Etapas s = 1 ... p:
 *    - Cada etapa procesa bloques de tamaño m = 2^s y mitad de bloque m2 = m / 2.
 *    - Existen N/2 mariposas independientes por cada etapa s.
 *    - Calculamos el twiddle base: w = exp(sign * 2 * pi * i * j / m).
 *    - La mariposa opera sobre la pareja (u, v):
 *        u = data_[k + j]
 *        v = w * data_[k + j + m2]
 *        data_[k + j] = u + v
 *        data_[k + j + m2] = u - v
 * 4. Paralelización OpenMP:
 *    - Como en la etapa s cada mariposa escribe en índices (k+j) y (k+j+m2) estrictamente disjuntos,
 *      no existen condiciones de carrera dentro de la misma etapa.
 *    - Entre la etapa s y la s+1 existe una barrera de sincronización obligatoria, pues la etapa s+1
 *      necesita leer los valores ya actualizados en la etapa s.
 */
void Butterfly1D::transformCooleyTukey(int schedule_type, int chunk_size,
                                        bool use_single_region) {
  if (n_ <= 1 || !data_) return;

  // Calcular p = log2(N)
  size_t p = 0;
  while ((size_t(1) << p) < n_) ++p;

  // Paso 1: Permutación Bit-Reversal previa (in-place)
  for (size_t i = 0; i < n_; ++i) {
    size_t rev = bitReverse(i, p);
    if (i < rev) {
      std::swap(data_[i], data_[rev]);
    }
  }

  const double pi = 3.14159265358979323846;
  // Convención del contrato: FFT directa usa -2*pi*i, IFFT inversa usa +2*pi*i
  const double sign = inverse_ ? 1.0 : -1.0;
  const size_t total_bf = n_ / 2; // Total de mariposas independientes por etapa

  // CASO 1: Región paralela única con barrera explícita entre etapas (transformStages)
  if (use_single_region) {
    #pragma omp parallel default(none) shared(p, n_, sign, pi, total_bf)
    {
      for (size_t s = 1; s <= p; ++s) {
        size_t m = size_t(1) << s;
        size_t m2 = m >> 1;

        #pragma omp for schedule(static)
        for (size_t idx = 0; idx < total_bf; ++idx) {
          size_t group_idx = idx / m2;
          size_t j = idx % m2;
          size_t k = group_idx * m;

          std::complex<double> w = std::polar(1.0, sign * 2.0 * pi * static_cast<double>(j) / static_cast<double>(m));
          std::complex<double> u = data_[k + j];
          std::complex<double> v = w * data_[k + j + m2];

          data_[k + j] = u + v;
          data_[k + j + m2] = u - v;
        }
        // Barrera explícita obligatoria: garantiza que todas las mariposas de la etapa s
        // hayan terminado antes de que cualquier hilo comience la etapa s+1.
        #pragma omp barrier
      }
    }
    return;
  }

  // CASO 2: Etapas con pragmas paralelos configurables (Schedules: static, dynamic, guided)
  for (size_t s = 1; s <= p; ++s) {
    size_t m = size_t(1) << s;
    size_t m2 = m >> 1;

    if (schedule_type == 1) { // Schedule Dynamic
      if (chunk_size > 0) {
        #pragma omp parallel for schedule(dynamic, chunk_size) default(none) shared(m, m2, sign, pi, total_bf, chunk_size)
        for (size_t idx = 0; idx < total_bf; ++idx) {
          size_t group_idx = idx / m2;
          size_t j = idx % m2;
          size_t k = group_idx * m;
          std::complex<double> w = std::polar(1.0, sign * 2.0 * pi * static_cast<double>(j) / static_cast<double>(m));
          std::complex<double> u = data_[k + j];
          std::complex<double> v = w * data_[k + j + m2];
          data_[k + j] = u + v;
          data_[k + j + m2] = u - v;
        }
      } else {
        #pragma omp parallel for schedule(dynamic) default(none) shared(m, m2, sign, pi, total_bf)
        for (size_t idx = 0; idx < total_bf; ++idx) {
          size_t group_idx = idx / m2;
          size_t j = idx % m2;
          size_t k = group_idx * m;
          std::complex<double> w = std::polar(1.0, sign * 2.0 * pi * static_cast<double>(j) / static_cast<double>(m));
          std::complex<double> u = data_[k + j];
          std::complex<double> v = w * data_[k + j + m2];
          data_[k + j] = u + v;
          data_[k + j + m2] = u - v;
        }
      }
    } else if (schedule_type == 2) { // Schedule Guided
      if (chunk_size > 0) {
        #pragma omp parallel for schedule(guided, chunk_size) default(none) shared(m, m2, sign, pi, total_bf, chunk_size)
        for (size_t idx = 0; idx < total_bf; ++idx) {
          size_t group_idx = idx / m2;
          size_t j = idx % m2;
          size_t k = group_idx * m;
          std::complex<double> w = std::polar(1.0, sign * 2.0 * pi * static_cast<double>(j) / static_cast<double>(m));
          std::complex<double> u = data_[k + j];
          std::complex<double> v = w * data_[k + j + m2];
          data_[k + j] = u + v;
          data_[k + j + m2] = u - v;
        }
      } else {
        #pragma omp parallel for schedule(guided) default(none) shared(m, m2, sign, pi, total_bf)
        for (size_t idx = 0; idx < total_bf; ++idx) {
          size_t group_idx = idx / m2;
          size_t j = idx % m2;
          size_t k = group_idx * m;
          std::complex<double> w = std::polar(1.0, sign * 2.0 * pi * static_cast<double>(j) / static_cast<double>(m));
          std::complex<double> u = data_[k + j];
          std::complex<double> v = w * data_[k + j + m2];
          data_[k + j] = u + v;
          data_[k + j + m2] = u - v;
        }
      }
    } else { // Schedule Static (Predeterminado)
      if (chunk_size > 0) {
        #pragma omp parallel for schedule(static, chunk_size) default(none) shared(m, m2, sign, pi, total_bf, chunk_size)
        for (size_t idx = 0; idx < total_bf; ++idx) {
          size_t group_idx = idx / m2;
          size_t j = idx % m2;
          size_t k = group_idx * m;
          std::complex<double> w = std::polar(1.0, sign * 2.0 * pi * static_cast<double>(j) / static_cast<double>(m));
          std::complex<double> u = data_[k + j];
          std::complex<double> v = w * data_[k + j + m2];
          data_[k + j] = u + v;
          data_[k + j + m2] = u - v;
        }
      } else {
        #pragma omp parallel for schedule(static) default(none) shared(m, m2, sign, pi, total_bf)
        for (size_t idx = 0; idx < total_bf; ++idx) {
          size_t group_idx = idx / m2;
          size_t j = idx % m2;
          size_t k = group_idx * m;
          std::complex<double> w = std::polar(1.0, sign * 2.0 * pi * static_cast<double>(j) / static_cast<double>(m));
          std::complex<double> u = data_[k + j];
          std::complex<double> v = w * data_[k + j + m2];
          data_[k + j] = u + v;
          data_[k + j + m2] = u - v;
        }
      }
    }
  }
}

// ============================================================================
// IMPLEMENTACIÓN DETALLADA: MODO 1 (STOCKHAM OUT-OF-PLACE)
// ============================================================================

/**
 * @brief Implementación interna del algoritmo Radix-2 Stockham (Autosort out-of-place).
 * @param schedule_type Entrada: Tipo de schedule OpenMP (0=static, 1=dynamic, 2=guided).
 * @param chunk_size Entrada: Tamaño de chunk para OpenMP.
 * @param use_collapse Entrada: true para paralelizar bucles anidados con 'collapse(2)'.
 * @return Salida: void (el resultado final se escribe de vuelta en data_ en orden natural).
 *
 * @details EXPLICACIÓN PASO A PASO:
 * 1. Stockham no requiere permutación Bit-Reversal.
 * 2. Utiliza un vector de origen `src` y un vector de destino `dst` de tamaño N.
 * 3. En cada etapa s = 0 ... p-1:
 *    - l = 2^s (número de bloques)
 *    - g = N / 2^(s+1) (elementos por sub-bloque)
 *    - Mapeo de índices de lectura:
 *        i0 = j + 2*k*g
 *        i1 = i0 + g
 *    - Mapeo de índices de escritura:
 *        o0 = j + k*g
 *        o1 = o0 + N/2
 *    - Operación de mariposa:
 *        u = src[i0]
 *        v = omega * src[i1]
 *        dst[o0] = u + v
 *        dst[o1] = u - v
 * 4. Al finalizar la etapa s, se intercambian los buffers `src.swap(dst)`.
 * 5. Al terminar todas las etapas, los resultados se copian de vuelta al arreglo principal `data_`.
 */
void Butterfly1D::transformStockham(int schedule_type, int chunk_size,
                                     bool use_collapse) {
  if (n_ <= 1 || !data_) return;

  size_t p = 0;
  while ((size_t(1) << p) < n_) ++p;

  // Buffers temporales para alternar lectura y escritura en cada etapa
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
          std::complex<double> omega = std::polar(1.0, sign * 2.0 * pi * static_cast<double>(k) / static_cast<double>(m));
          size_t i0 = j + 2 * k * g;
          size_t i1 = i0 + g;
          size_t o0 = j + k * g;
          size_t o1 = o0 + n_ / 2;
          std::complex<double> u = src[i0];
          std::complex<double> v = omega * src[i1];
          dst[o0] = u + v;
          dst[o1] = u - v;
        }
      }
    } else {
      // CASO 2: Schedules configurables (static, dynamic, guided) sobre el bucle externo k
      if (schedule_type == 1) { // Dynamic
        if (chunk_size > 0) {
          #pragma omp parallel for schedule(dynamic, chunk_size) default(none) shared(src, dst, l, g, m, sign, pi, chunk_size)
          for (size_t k = 0; k < l; ++k) {
            for (size_t j = 0; j < g; ++j) {
              std::complex<double> omega = std::polar(1.0, sign * 2.0 * pi * static_cast<double>(k) / static_cast<double>(m));
              size_t i0 = j + 2 * k * g;
              size_t i1 = i0 + g;
              size_t o0 = j + k * g;
              size_t o1 = o0 + n_ / 2;
              std::complex<double> u = src[i0];
              std::complex<double> v = omega * src[i1];
              dst[o0] = u + v;
              dst[o1] = u - v;
            }
          }
        } else {
          #pragma omp parallel for schedule(dynamic) default(none) shared(src, dst, l, g, m, sign, pi)
          for (size_t k = 0; k < l; ++k) {
            for (size_t j = 0; j < g; ++j) {
              std::complex<double> omega = std::polar(1.0, sign * 2.0 * pi * static_cast<double>(k) / static_cast<double>(m));
              size_t i0 = j + 2 * k * g;
              size_t i1 = i0 + g;
              size_t o0 = j + k * g;
              size_t o1 = o0 + n_ / 2;
              std::complex<double> u = src[i0];
              std::complex<double> v = omega * src[i1];
              dst[o0] = u + v;
              dst[o1] = u - v;
            }
          }
        }
      } else if (schedule_type == 2) { // Guided
        if (chunk_size > 0) {
          #pragma omp parallel for schedule(guided, chunk_size) default(none) shared(src, dst, l, g, m, sign, pi, chunk_size)
          for (size_t k = 0; k < l; ++k) {
            for (size_t j = 0; j < g; ++j) {
              std::complex<double> omega = std::polar(1.0, sign * 2.0 * pi * static_cast<double>(k) / static_cast<double>(m));
              size_t i0 = j + 2 * k * g;
              size_t i1 = i0 + g;
              size_t o0 = j + k * g;
              size_t o1 = o0 + n_ / 2;
              std::complex<double> u = src[i0];
              std::complex<double> v = omega * src[i1];
              dst[o0] = u + v;
              dst[o1] = u - v;
            }
          }
        } else {
          #pragma omp parallel for schedule(guided) default(none) shared(src, dst, l, g, m, sign, pi)
          for (size_t k = 0; k < l; ++k) {
            for (size_t j = 0; j < g; ++j) {
              std::complex<double> omega = std::polar(1.0, sign * 2.0 * pi * static_cast<double>(k) / static_cast<double>(m));
              size_t i0 = j + 2 * k * g;
              size_t i1 = i0 + g;
              size_t o0 = j + k * g;
              size_t o1 = o0 + n_ / 2;
              std::complex<double> u = src[i0];
              std::complex<double> v = omega * src[i1];
              dst[o0] = u + v;
              dst[o1] = u - v;
            }
          }
        }
      } else { // Static (Predeterminado)
        if (chunk_size > 0) {
          #pragma omp parallel for schedule(static, chunk_size) default(none) shared(src, dst, l, g, m, sign, pi, chunk_size)
          for (size_t k = 0; k < l; ++k) {
            for (size_t j = 0; j < g; ++j) {
              std::complex<double> omega = std::polar(1.0, sign * 2.0 * pi * static_cast<double>(k) / static_cast<double>(m));
              size_t i0 = j + 2 * k * g;
              size_t i1 = i0 + g;
              size_t o0 = j + k * g;
              size_t o1 = o0 + n_ / 2;
              std::complex<double> u = src[i0];
              std::complex<double> v = omega * src[i1];
              dst[o0] = u + v;
              dst[o1] = u - v;
            }
          }
        } else {
          #pragma omp parallel for schedule(static) default(none) shared(src, dst, l, g, m, sign, pi)
          for (size_t k = 0; k < l; ++k) {
            for (size_t j = 0; j < g; ++j) {
              std::complex<double> omega = std::polar(1.0, sign * 2.0 * pi * static_cast<double>(k) / static_cast<double>(m));
              size_t i0 = j + 2 * k * g;
              size_t i1 = i0 + g;
              size_t o0 = j + k * g;
              size_t o1 = o0 + n_ / 2;
              std::complex<double> u = src[i0];
              std::complex<double> v = omega * src[i1];
              dst[o0] = u + v;
              dst[o1] = u - v;
            }
          }
        }
      }
    }
    // Intercambiar buffers para que la salida de la etapa actual pase a ser la entrada de la siguiente
    src.swap(dst);
  }

  // Copiar el resultado final almacenado en src de vuelta al arreglo principal de la clase
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
