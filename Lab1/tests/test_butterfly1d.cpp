#include "Butterfly1D.h"
#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>
#include <random>
#include <vector>
#include <omp.h>

// Tolerancia numérica unificada para comparaciones en doble precisión
constexpr double TOLERANCE = 1e-10;

/**
 * @brief Función auxiliar para verificar si dos números complejos son iguales dentro de la tolerancia.
 * @param a Entrada: Primer valor complejo.
 * @param b Entrada: Segundo valor complejo.
 * @return Salida: true si la distancia en el plano complejo es < TOLERANCE, false de lo contrario.
 */
bool isClose(std::complex<double> a, std::complex<double> b) {
  return std::abs(a - b) < TOLERANCE;
}

/**
 * @brief [Test 1] Prueba del Impulso 1D en N=4.
 * 
 * @details
 * - Entrada: Señal impulso unitario x = (1, 0, 0, 0).
 * - Salida esperada: Espectro constante X = (1, 1, 1, 1).
 * - Qué hace: Valida la mariposa de Cooley-Tukey (layout 0) y Stockham (layout 1) contra la respuesta
 *   analítica cerrada para una entrada impulso.
 */
void testImpulse4() {
  std::cout << "[Test 1] Impulso N=4 (Cooley-Tukey vs Stockham)... " << std::flush;

  std::vector<std::complex<double>> ct_data = {{1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
  std::vector<std::complex<double>> st_data = ct_data;

  // Ejecutar Cooley-Tukey (layout 0)
  Butterfly1D ct(ct_data.data(), 4, false);
  ct.transform(0, 0, 0);

  // Ejecutar Stockham (layout 1)
  Butterfly1D st(st_data.data(), 4, false);
  st.transform(1, 0, 0);

  // Verificar que ambas dieron el espectro analítico (1, 1, 1, 1)
  for (size_t i = 0; i < 4; ++i) {
    assert(isClose(ct_data[i], {1.0, 0.0}));
    assert(isClose(st_data[i], {1.0, 0.0}));
  }
  std::cout << "PASSED\n";
}

/**
 * @brief [Test 2] Prueba de Equivalencia Numérica en N=8.
 * 
 * @details
 * - Entrada: Vector de 8 números complejos aleatorios generados de forma reproducible con semilla fija.
 * - Salida esperada: Resultados idénticos posición a posición entre Cooley-Tukey y Stockham.
 * - Qué hace: Confirma que la mariposa in-place (Cooley-Tukey) y la mariposa out-of-place (Stockham)
 *   calculan exactamente la misma DFT.
 */
void testEquivalenceN8() {
  std::cout << "[Test 2] Equivalencia N=8 en datos aleatorios... " << std::flush;

  std::mt19937 rng(42); // Semilla fija para reproducibilidad
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  size_t N = 8;
  std::vector<std::complex<double>> orig(N);
  for (size_t i = 0; i < N; ++i) {
    orig[i] = {dist(rng), dist(rng)};
  }

  std::vector<std::complex<double>> ct_data = orig;
  std::vector<std::complex<double>> st_data = orig;

  Butterfly1D ct(ct_data.data(), N, false);
  ct.transform(0, 0, 0);

  Butterfly1D st(st_data.data(), N, false);
  st.transform(1, 0, 0);

  for (size_t i = 0; i < N; ++i) {
    assert(isClose(ct_data[i], st_data[i]));
  }
  std::cout << "PASSED\n";
}

/**
 * @brief [Test 3] Prueba del Contrato sin Escala en 1D: IFFT(FFT(x)) == N * x.
 * 
 * @details
 * - Entrada: Vector complejo creciente x_i = i - i*sqrt(-1) de longitud N=16.
 * - Salida esperada: IFFT(FFT(x)) devuelve N * x. Al dividir externamente por N, recupera x.
 * - Qué hace: Valida que Butterfly1D NO aplique el factor de escala 1/N internamente, respetando
 *   el contrato del laboratorio donde el escalado global 1/(M*N) lo aplica Fft2D al final.
 */
void testRoundtrip1DUnscaledContract() {
  std::cout << "[Test 3] Contrato sin escala en 1D: IFFT(FFT(x)) == N * x... " << std::flush;

  size_t N = 16;
  std::vector<std::complex<double>> orig(N);
  for (size_t i = 0; i < N; ++i) {
    orig[i] = {static_cast<double>(i), -static_cast<double>(i)};
  }

  std::vector<std::complex<double>> data = orig;

  // Directa (inverse = false)
  Butterfly1D fwd(data.data(), N, false);
  fwd.transform(0, 0, 0);

  // Inversa (inverse = true)
  Butterfly1D inv(data.data(), N, true);
  inv.transform(0, 0, 0);

  // Verificación de contrato: la mariposa inversa solo cambia el signo del twiddle, sin escalar
  for (size_t i = 0; i < N; ++i) {
    std::complex<double> expected = static_cast<double>(N) * orig[i];
    assert(isClose(data[i], expected));
  }

  // Al aplicar la escala 1/N externamente se recupera x exactamente
  for (size_t i = 0; i < N; ++i) {
    data[i] /= static_cast<double>(N);
    assert(isClose(data[i], orig[i]));
  }
  std::cout << "PASSED\n";
}

/**
 * @brief [Test 4] Prueba de Schedules OpenMP (Static, Dynamic, Guided) y Chunks.
 * 
 * @details
 * - Entrada: Onda senoidal discreta 1D de longitud N=32.
 * - Salida esperada: Mismo resultado numérico en todas las modalidades de scheduling y paralelismo.
 * - Qué hace: Compara la ejecución de static(chunk=4), dynamic(chunk=4), guided(chunk=4), collapse(2)
 *   y transformStages() (barrera única), asegurando estabilidad numérica e invariancia ante hilos.
 */
void testSchedulesAndClauses() {
  std::cout << "[Test 4] Prueba de Schedules (Static, Dynamic, Guided) y Chunks... " << std::flush;

  size_t N = 32;
  std::vector<std::complex<double>> base(N);
  for (size_t i = 0; i < N; ++i) {
    base[i] = {std::sin(2.0 * 3.14159265358979323846 * i / N), 0.0};
  }

  // Schedule Static
  std::vector<std::complex<double>> d_static = base;
  Butterfly1D b_static(d_static.data(), N, false);
  b_static.transform(0, 0, 4);

  // Schedule Dynamic
  std::vector<std::complex<double>> d_dynamic = base;
  Butterfly1D b_dynamic(d_dynamic.data(), N, false);
  b_dynamic.transform(0, 1, 4);

  // Schedule Guided
  std::vector<std::complex<double>> d_guided = base;
  Butterfly1D b_guided(d_guided.data(), N, false);
  b_guided.transform(0, 2, 4);

  // Stockham con Collapse
  std::vector<std::complex<double>> d_collapse = base;
  Butterfly1D b_collapse(d_collapse.data(), N, false);
  b_collapse.transformCollapse();

  // Cooley-Tukey con Barrera Explícita
  std::vector<std::complex<double>> d_stages = base;
  Butterfly1D b_stages(d_stages.data(), N, false);
  b_stages.transformStages();

  // Todos los métodos de paralelismo deben entregar exactamente los mismos valores
  for (size_t i = 0; i < N; ++i) {
    assert(isClose(d_static[i], d_dynamic[i]));
    assert(isClose(d_static[i], d_guided[i]));
    assert(isClose(d_static[i], d_collapse[i]));
    assert(isClose(d_static[i], d_stages[i]));
  }
  std::cout << "PASSED\n";
}

/**
 * @brief [Test 5] Prueba de Métodos Demostrativos de Cláusulas OpenMP.
 * 
 * @details
 * - Entrada: Arreglo de N=8 datos complejos.
 * - Salida esperada: Verificación del comportamiento correcto de initTwiddlesSingle, accumulateFirstprivate
 *   y stageIndexLastprivate.
 * - Qué hace: Demuestra que los pragmas #pragma omp single, firstprivate y lastprivate funcionan
 *   según las especificaciones del laboratorio.
 */
void testOpenMPClausesDemo() {
  std::cout << "[Test 5] Cláusulas OpenMP (initTwiddlesSingle, accumulateFirstprivate, stageIndexLastprivate)... " << std::flush;

  size_t N = 8;
  std::vector<std::complex<double>> data(N, {0.0, 0.0});
  Butterfly1D b(data.data(), N, false);

  // Probar #pragma omp single
  auto twiddles = b.initTwiddlesSingle();
  assert(twiddles.size() == N);
  assert(isClose(twiddles[0], {1.0, 0.0}));

  // Probar firstprivate(acc)
  b.accumulateFirstprivate(5.0);
  for (size_t i = 0; i < N; ++i) {
    assert(isClose(data[i], {5.0, 0.0}));
  }

  // Probar lastprivate(last_stage)
  int last_stage = b.stageIndexLastprivate();
  assert(last_stage == 3); // log2(8) = 3

  std::cout << "PASSED\n";
}

/**
 * @brief [Test 6] Prueba de Forma Cerrada Analítica de Senoide 1D (1 Período).
 * 
 * @details
 * - Entrada: Señal senoidal pura x_n = sin(2*pi*n/N) de longitud N=16.
 * - Salida esperada: Dos picos en las frecuencias k=1 y k=15 de magnitud N/2 = 8.0 (con parte imaginaria -8i y +8i),
 *   y 0 absoluto en todos los demás bins.
 * - Qué hace: Confirma que la mariposa de Fourier 1D respeta la propiedad espectral analítica de las senoides puras.
 */
void testSineAnalyticalForm() {
  std::cout << "[Test 6] Senoide 1D de 1 período (Forma cerrada analítica)... " << std::flush;

  size_t N = 16;
  std::vector<std::complex<double>> data(N);
  const double pi = 3.14159265358979323846;

  for (size_t n = 0; n < N; ++n) {
    data[n] = {std::sin(2.0 * pi * static_cast<double>(n) / static_cast<double>(N)), 0.0};
  }

  Butterfly1D b(data.data(), N, false);
  b.transform(0, 0, 0); // Cooley-Tukey

  // Para sin(2*pi*n/N), la DFT analítica sin escalar entrega:
  // X[1] = -i * (N/2) = (0, -8.0)
  // X[N-1] = +i * (N/2) = (0, +8.0)
  // Todos los demás X[k] = 0
  for (size_t k = 0; k < N; ++k) {
    if (k == 1) {
      assert(isClose(data[k], {0.0, -8.0}));
    } else if (k == N - 1) {
      assert(isClose(data[k], {0.0, 8.0}));
    } else {
      assert(isClose(data[k], {0.0, 0.0}));
    }
  }
  std::cout << "PASSED\n";
}

/**
 * @brief [Test 7] Prueba de Escalabilidad en Tamaños Grandes (N=64, 256, 1024).
 * 
 * @details
 * - Entrada: Señales complejas aleatorias en potencias de 2 mayores (N=64, 256, 1024).
 * - Salida esperada: Preservación de la precisión numérica y coincidencia exacta entre Cooley-Tukey y Stockham.
 * - Qué hace: Verifica que la mariposa no sufra de desbordamiento de memoria ni acumulación excesiva de error
 *   en tamaños de benchmark.
 */
void testLargeSizesN1024() {
  std::cout << "[Test 7] Tamaños grandes (N=64, 256, 1024)... " << std::flush;

  std::vector<size_t> sizes = {64, 256, 1024};
  std::mt19937 rng(12345);
  std::uniform_real_distribution<double> dist(-5.0, 5.0);

  for (size_t N : sizes) {
    std::vector<std::complex<double>> orig(N);
    for (size_t i = 0; i < N; ++i) {
      orig[i] = {dist(rng), dist(rng)};
    }

    std::vector<std::complex<double>> ct_data = orig;
    std::vector<std::complex<double>> st_data = orig;

    Butterfly1D ct(ct_data.data(), N, false);
    ct.transform(0, 0, 0);

    Butterfly1D st(st_data.data(), N, false);
    st.transform(1, 0, 0);

    for (size_t i = 0; i < N; ++i) {
      assert(isClose(ct_data[i], st_data[i]));
    }
  }
  std::cout << "PASSED\n";
}

/**
 * @brief [Test 8] Prueba de Invariancia entre Hilos OpenMP (1 vs 8 hilos) y Detección de Barrera.
 * 
 * @details
 * - Entrada: Vector de datos estocásticos de N=1024.
 * - Salida esperada: Mismo resultado numérico exacto al ejecutar con omp_set_num_threads(1) y omp_set_num_threads(8).
 * - Qué hace: Comprueba que el reparto de mariposas entre múltiples hilos no introduzca condiciones de carrera
 *   y confirma que las barreras de sincronización entre etapas funcionen adecuadamente.
 */
void testThreadInvarianceAndBarrier() {
  std::cout << "[Test 8] Invariancia ante hilos OpenMP (1 vs 8 hilos)... " << std::flush;

  size_t N = 1024;
  std::vector<std::complex<double>> orig(N);
  std::mt19937 rng(999);
  std::uniform_real_distribution<double> dist(-10.0, 10.0);

  for (size_t i = 0; i < N; ++i) {
    orig[i] = {dist(rng), dist(rng)};
  }

  // Ejecución con 1 hilo
  std::vector<std::complex<double>> d_single = orig;
  omp_set_num_threads(1);
  Butterfly1D b_single(d_single.data(), N, false);
  b_single.transform(0, 0, 0);

  // Ejecución con 8 hilos
  std::vector<std::complex<double>> d_multi = orig;
  omp_set_num_threads(8);
  Butterfly1D b_multi(d_multi.data(), N, false);
  b_multi.transform(0, 0, 0);

  for (size_t i = 0; i < N; ++i) {
    assert(isClose(d_single[i], d_multi[i]));
  }
  std::cout << "PASSED\n";
}

/**
 * @brief [Test 9] Prueba de Casos Borde (N=1 y N=2).
 * 
 * @details
 * - Entrada: Señal de 1 elemento (N=1) y señal de 2 elementos (N=2).
 * - Salida esperada:
 *   - N=1: Deja la señal intacta.
 *   - N=2: Realiza la mariposa elemental de 1 etapa: X[0] = x[0] + x[1], X[1] = x[0] - x[1].
 * - Qué hace: Garantiza robustez en los tamaños límite de la mariposa de Fourier radix-2.
 */
void testEdgeCasesN1andN2() {
  std::cout << "[Test 9] Casos borde (N=1 y N=2)... " << std::flush;

  // Caso N = 1
  std::vector<std::complex<double>> d1 = {{42.0, -17.0}};
  Butterfly1D b1(d1.data(), 1, false);
  b1.transform(0, 0, 0);
  assert(isClose(d1[0], {42.0, -17.0}));

  // Caso N = 2
  std::vector<std::complex<double>> d2 = {{3.0, 1.0}, {1.0, 2.0}};
  Butterfly1D b2(d2.data(), 2, false);
  b2.transform(0, 0, 0);

  // X[0] = (3+1i) + (1+2i) = (4+3i)
  // X[1] = (3+1i) - (1+2i) = (2-1i)
  assert(isClose(d2[0], {4.0, 3.0}));
  assert(isClose(d2[1], {2.0, -1.0}));

  std::cout << "PASSED\n";
}

/**
 * @brief [Test 10] Prueba de Transformada Inversa 1D Directa con Escala 1/N.
 * 
 * @details
 * - Entrada: Vector complejo aleatorio de N=32.
 * - Salida esperada: IFFT(x) directamente (en ambos layouts) aplicando el escalado 1/N de 1D recupera el vector original.
 * - Qué hace: Valida la mariposa inversa directa tanto en Cooley-Tukey como en Stockham.
 */
void testInverseDirect1D() {
  std::cout << "[Test 10] Transformada inversa 1D directa en ambos layouts... " << std::flush;

  size_t N = 32;
  std::vector<std::complex<double>> orig(N);
  std::mt19937 rng(777);
  std::uniform_real_distribution<double> dist(-2.0, 2.0);

  for (size_t i = 0; i < N; ++i) {
    orig[i] = {dist(rng), dist(rng)};
  }

  // Probar Cooley-Tukey Inverso
  std::vector<std::complex<double>> ct_data = orig;
  Butterfly1D ct_fwd(ct_data.data(), N, false);
  ct_fwd.transform(0, 0, 0);

  Butterfly1D ct_inv(ct_data.data(), N, true);
  ct_inv.transform(0, 0, 0);

  for (size_t i = 0; i < N; ++i) {
    ct_data[i] /= static_cast<double>(N);
    assert(isClose(ct_data[i], orig[i]));
  }

  // Probar Stockham Inverso
  std::vector<std::complex<double>> st_data = orig;
  Butterfly1D st_fwd(st_data.data(), N, false);
  st_fwd.transform(1, 0, 0);

  Butterfly1D st_inv(st_data.data(), N, true);
  st_inv.transform(1, 0, 0);

  for (size_t i = 0; i < N; ++i) {
    st_data[i] /= static_cast<double>(N);
    assert(isClose(st_data[i], orig[i]));
  }

  std::cout << "PASSED\n";
}

/**
 * @brief Función principal ejecutable de la suite de pruebas del Rol 2.
 */
int main() {
  std::cout << "=== PRUEBAS DE UNIDAD ROL 2 (Butterfly1D) ===\n";
  testImpulse4();
  testEquivalenceN8();
  testRoundtrip1DUnscaledContract();
  testSchedulesAndClauses();
  testOpenMPClausesDemo();
  testSineAnalyticalForm();
  testLargeSizesN1024();
  testThreadInvarianceAndBarrier();
  testEdgeCasesN1andN2();
  testInverseDirect1D();
  std::cout << "=== TODAS LAS 10 PRUEBAS DE ROL 2 PASARON EXITOSAMENTE ===\n";
  return 0;
}
