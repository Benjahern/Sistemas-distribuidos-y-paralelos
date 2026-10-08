#ifndef BUTTERFLY1D_H
#define BUTTERFLY1D_H

#include <complex>
#include <cstddef>
#include <vector>

/**
 * @brief Rol 2: Núcleo de mariposa FFT 1D radix-2 (in-place Cooley-Tukey y Stockham).
 *
 * Contrato de escala: La transformada 1D NO normaliza en ningún signo (inverse=true solo invierte el signo del twiddle).
 * La escala global 1/(M*N) la aplica Fft2D una sola vez al finalizar la FFT 2D.
 */
class Butterfly1D {
public:
  /**
   * @brief Constructor por defecto.
   */
  Butterfly1D() = default;

  /**
   * @brief Constructor parametrizado del núcleo 1D.
   * @param data Entrada/Salida: Puntero al arreglo contiguo de números complejos std::complex<double> de tamaño n.
   * @param n Entrada: Longitud de la señal (debe ser potencia de 2, ej. 4, 8, 16...).
   * @param inverse Entrada: false para FFT directa (signo -), true para IFFT inversa (signo +).
   */
  Butterfly1D(std::complex<double> *data, size_t n, bool inverse = false);

  /**
   * @brief Destructor por defecto.
   */
  ~Butterfly1D() = default;

  /**
   * @brief Actualiza los datos o el puntero de la señal 1D a procesar.
   * @param data Entrada/Salida: Puntero al arreglo de números complejos de tamaño n.
   * @param n Entrada: Tamaño del arreglo.
   * @param inverse Entrada: Dirección de la transformada (false = directa, true = inversa).
   * @return Salida: void.
   */
  void setData(std::complex<double> *data, size_t n, bool inverse = false);

  /**
   * @brief Ejecuta la transformada 1D en modo in-place Cooley-Tukey con schedule estático por defecto.
   * @return Salida: void (modifica el arreglo data_ in-place).
   */
  void transform();

  /**
   * @brief Ejecuta la transformada 1D especificando el tipo de schedule de OpenMP.
   * @param schedule_type Entrada: 0 = static, 1 = dynamic, 2 = guided.
   * @return Salida: void.
   */
  void transform(int schedule_type);

  /**
   * @brief Ejecuta la transformada 1D especificando el tipo de schedule y el tamaño de chunk.
   * @param schedule_type Entrada: 0 = static, 1 = dynamic, 2 = guided.
   * @param chunk_size Entrada: Tamaño del bloque de iteraciones asignado a cada hilo.
   * @return Salida: void.
   */
  void transform(int schedule_type, int chunk_size);

  /**
   * @brief Demostración de paralelismo con la cláusula 'collapse(2)' en el modo Stockham.
   * @return Salida: void.
   */
  void transformCollapse();

  /**
   * @brief Demostración de barrera explícita 'barrier' en una única región paralela con bucle de etapas.
   * @return Salida: void.
   */
  void transformStages();

  /**
   * @brief Método principal y parametrizado para ejecutar la FFT/IFFT 1D.
   * @param layout Entrada: Modo de ejecución (0 = Cooley-Tukey in-place, 1 = Stockham out-of-place).
   * @param schedule_type Entrada: Schedule de OpenMP (0 = static, 1 = dynamic, 2 = guided).
   * @param chunk_size Entrada: Tamaño del chunk para el schedule (0 = por defecto de OpenMP).
   * @return Salida: void (los datos transformados quedan almacenados en data_).
   */
  void transform(int layout, int schedule_type, int chunk_size);

  // =========================================================================
  // Métodos requeridos para la demostración de cláusulas OpenMP específicas
  // =========================================================================

  /**
   * @brief Demuestra el uso de la cláusula '#pragma omp single' inicializando factores twiddle.
   * @return Salida: std::vector<std::complex<double>> conteniendo los twiddles calculados por un único hilo.
   */
  std::vector<std::complex<double>> initTwiddlesSingle() const;

  /**
   * @brief Demuestra la cláusula 'firstprivate(acc)' sumando un valor base a cada elemento.
   * @param base_val Entrada: Valor real a sumar a cada elemento.
   * @return Salida: void (modifica data_ sumando base_val a cada posición).
   */
  void accumulateFirstprivate(double base_val);

  /**
   * @brief Demuestra la cláusula 'lastprivate(last_stage)' obteniendo la última etapa calculada.
   * @return Salida: int representando el índice de la última etapa (log2(n)).
   */
  int stageIndexLastprivate() const;

  /**
   * @brief Operación atómica de mariposa radix-2: u = in0, v = omega * in1; out0 = u + v, out1 = u - v.
   *
   * Única lógica matemática de mariposa compartida por Cooley-Tukey (in-place),
   * Stockham (out-of-place) y todas las cláusulas/schedules OpenMP.
   *
   * @param in0 Primer elemento de entrada.
   * @param in1 Segundo elemento de entrada (se multiplica por twiddle omega).
   * @param omega Factor twiddle exp(sign * 2 * pi * i * r / m).
   * @param out0 Salida u + v.
   * @param out1 Salida u - v.
   */
  static inline void butterfly(const std::complex<double> &in0,
                               const std::complex<double> &in1,
                               const std::complex<double> &omega,
                               std::complex<double> &out0,
                               std::complex<double> &out1) noexcept {
    const std::complex<double> v = omega * in1;
    const std::complex<double> u = in0;
    out0 = u + v;
    out1 = u - v;
  }

private:
  std::complex<double> *data_{nullptr}; ///< Puntero al arreglo de datos complejos en memoria.
  size_t n_{0};                          ///< Longitud de la señal 1D (potencia de 2).
  bool inverse_{false};                  ///< Dirección de la transformada (false = directa, true = inversa).

  /**
   * @brief Aplica la mariposa en la posición idx de la etapa Cooley-Tukey delegando en butterfly().
   */
  void cooleyTukeyButterflyAt(size_t idx, size_t m, size_t m2, double sign, double pi);

  /**
   * @brief Aplica la mariposa en la posición (k, j) de la etapa Stockham delegando en butterfly().
   */
  void stockhamButterflyAt(const std::complex<double> *src,
                           std::complex<double> *dst,
                           size_t k, size_t j, size_t g, size_t m,
                           double sign, double pi);

  /**
   * @brief Implementación interna del algoritmo in-place Cooley-Tukey.
   * @param schedule_type Entrada: 0 = static, 1 = dynamic, 2 = guided.
   * @param chunk_size Entrada: Tamaño de chunk para OpenMP.
   * @param use_single_region Entrada: true para usar una única región paralela con '#pragma omp barrier'.
   * @return Salida: void.
   */
  void transformCooleyTukey(int schedule_type, int chunk_size,
                            bool use_single_region = false);

  /**
   * @brief Implementación interna del algoritmo out-of-place Stockham.
   * @param schedule_type Entrada: 0 = static, 1 = dynamic, 2 = guided.
   * @param chunk_size Entrada: Tamaño de chunk para OpenMP.
   * @param use_collapse Entrada: true para aplicar la cláusula 'collapse(2)'.
   * @return Salida: void.
   */
  void transformStockham(int schedule_type, int chunk_size,
                         bool use_collapse = false);
};

#endif // BUTTERFLY1D_H
