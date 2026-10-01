# Laboratorio 1 - Programación Paralela con OpenMP: FFT e IFFT 2D

**Departamento de Ingeniería Informática**
**Universidad de Santiago de Chile**

---

## 1. Estructura del Proyecto

- `include/`: Cabeceras de los módulos del sistema (`.h`).
- `src/`: Código fuente de las implementaciones (`.cpp`).
- `tests/`: Suite del rol 2 (`test_butterfly1d.cpp`) y del rol 3 (`test_fft2d.cpp`), ejecutables con sustitutos o con los módulos reales.
- `Makefile`: Sistema de compilación modular y ejecución de pruebas.
- `Dockerfile`: Entorno reproducible para desarrollo y CI (Ubuntu 24.04 con GCC y Make).
- `AGENTS.md` / `laboratorio_1_openmp_fft_2d.md`: Especificaciones completas del laboratorio.

---

## 2. Compilación y Ejecución

El proyecto utiliza C++17 y OpenMP. La compilación se gestiona de forma modular para permitir que cada rol compile independientemente:

```bash
# Compilar todos los módulos disponibles como objetos (.o)
make all

# Compilar solo el módulo de FFT 2D
make Fft2D

# Compilar y ejecutar las pruebas del Rol 2 (Butterfly1D)
make test-role2

# Validar el rol 3 con sustitutos de los roles 1 y 2
make test

# Después del merge en dev: validar con los módulos reales
make test-integration

# Limpiar archivos binarios generados
make clean
```

`make test` es un alias de `make test-role3`. Define `FFT2D_STANDALONE_STUBS`
únicamente para el objeto de pruebas aisladas. `make test-integration` compila
otro objeto sin esa macro y enlaza `ComplexField.cpp` y `Butterfly1D.cpp`.
Ambos comandos usan la misma suite y mantienen sus binarios separados para
evitar reutilizar sustitutos al probar la integración.

En esta rama, los `.cpp` de los roles 1 y 2 están vacíos: la integración se
ejecutará después de incorporar sus ramas en `dev`. Esto no impide validar
la composición 2D del rol 3 de forma aislada. `make all` compila objetos;
el ejecutable de la aplicación queda a cargo de la integración del equipo.

### Ejecución en Contenedor Docker

```bash
# Construir la imagen del laboratorio
docker build -t lab1-dev -f Dockerfile .

# Ejecutar las pruebas dentro del contenedor
docker run --rm -v "${PWD}:/app" -w /app lab1-dev make test
```

---

## 3. Rol 3: FFT e IFFT Bidimensional (`Fft2D`)

El Rol 3 implementa la transformada de Fourier bidimensional directa e inversa basada en el principio de **separabilidad matemática** sobre grillas de tamaño $M \times N$ (ambas potencias de 2, permitiendo $M \neq N$).

### 3.1. Convención de Escala
- **Transformada directa (`forward`):** Se evalúa sin factor de escala:
  $$X_{k,l} = \sum_{m=0}^{M-1} \sum_{n=0}^{N-1} x_{m,n} e^{-2\pi i \left(\frac{km}{M} + \frac{ln}{N}\right)}$$
- **Transformada inversa (`inverse`):** Utiliza twiddles con signo conjugado ($+2\pi i$) y aplica el factor de escala total $\frac{1}{M \cdot N}$ **una única vez** al final del proceso completo (después de invertir columnas y filas):
  $$x_{m,n} = \frac{1}{MN} \sum_{k=0}^{M-1} \sum_{l=0}^{N-1} X_{k,l} e^{+2\pi i \left(\frac{km}{M} + \frac{ln}{N}\right)}$$

**Contrato requerido para integrar el rol 2:** `Butterfly1D(data, n, inverse)`
y `transform(layout, 0, 0)` deben producir una transformada 1D sin escala.
`inverse=true` cambia únicamente el signo del exponente. Por tanto,
`inverseRows` e `inverseCols` tampoco normalizan; solo `Fft2D::inverse` aplica
el factor final. Una IFFT 1D utilizada por separado requiere su factor `1/n`.
Si la interfaz del rol 2 normaliza internamente, habrá que adaptar este contrato
durante el merge para evitar un segundo escalado. El test 12 detecta esa diferencia.

Los sustitutos de la suite implementan este contrato. Son apoyo de pruebas,
no la implementación entregable del núcleo del rol 2. Las cabeceras de
`ComplexField` y `Butterfly1D` representan las interfaces requeridas por esta rama;
se deben contrastar con las definitivas del equipo al integrar en `dev`.

### 3.2. Justificación de Tolerancia Numérica
- Se utiliza doble precisión (`double`, $\epsilon \approx 2.22 \times 10^{-16}$). La FFT acumula redondeo a lo largo de sus etapas; la magnitud de la entrada y el cálculo de twiddles también influyen.
- La suite exige **error absoluto por elemento menor que $10^{-10}$**. Esto también acota el RMSE por debajo de ese valor; el test 12 lo calcula explícitamente. Es una tolerancia conservadora para las entradas y tamaños pequeños de esta suite, no una garantía para cualquier amplitud o tamaño de benchmark.
- Las 12 pruebas cubren DFT de referencia en `8x4` y `4x8`, impulso `2x2`, exponencial compleja, seno 2D y Parseval en ambos layouts, ida y vuelta con 1, 2, 4 y 8 hilos, tareas con y sin `single`, schedules, validaciones y pasadas individuales. Los tamaños grandes se validarán al integrar los benchmarks.

### 3.3. Modos y Layouts Soportados
La clase `Fft2D` propaga el layout de mariposa a las pasadas 1D:
- `layout = 0`: **In-place (Cooley–Tukey)** con permutación bit-reversal previa.
- `layout = 1`: **Stockham (out-of-place)** con alternancia de buffers y salida en orden natural.

Ambos modos producen exactamente la misma transformada dentro de la tolerancia numérica.

---

### 3.4. Orden de las Pasadas y Conmutatividad
- **Directa:** Pasada por filas (longitud $N$) $\to$ Sincronización $\to$ Pasada por columnas (longitud $M$).
- **Inversa:** Pasada por columnas $\to$ Sincronización $\to$ Pasada por filas $\to$ Escalado $\frac{1}{M \cdot N}$.
- *Justificación matemática:* La DFT 2D es un producto tensorial separable y lineal. Las transformadas 1D en dimensiones ortogonales conmutan ($F_{2D} = F_{cols} \circ F_{rows} = F_{rows} \circ F_{cols}$), por lo que deshacer en orden inverso (columnas y luego filas) o en orden directo con twiddles conjugados recupera exactamente la señal original.

### 3.5. Costes para el Análisis de Rendimiento
El rol 4 podrá medir estos costes de la composición 2D:
1. **Acceso no contiguo de columnas:** Las copias hacia y desde `col_buf` están repartidas entre hilos, pero el stride puede limitar la localidad y el ancho de banda de memoria.
2. **Creación y sincronización de tareas OpenMP:** Con `single`, un hilo genera las tareas; `taskwait` espera las hijas del hilo que lo ejecuta y el final de la región paralela garantiza que todo el equipo termine antes de continuar.
3. **Pasada de normalización:** Es un recorrido adicional paralelo con `collapse(2)` y también consume ancho de banda de memoria.

Estos costes no equivalen automáticamente a la fracción serial $f$ de Amdahl.
Su estimación requiere mediciones; en particular, columnas y normalización
son trabajo paralelo en esta implementación.

---

## 4. Mapeo de Cláusulas y Directivas OpenMP (Rol 3)

| Archivo | Función / Método | Cláusula / Directiva OpenMP | Motivo y Justificación Técnica | Test Asociado |
| :--- | :--- | :--- | :--- | :--- |
| `src/Fft2D.cpp` | `rowsPass` / `colsPass` | `parallel for` / `parallel` + `for`, con `schedule(runtime)` | Reparto static, dynamic o guided configurable con `omp_set_schedule`, incluido el chunk. | Test 9 (`testScheduleSweep`) |
| `src/Fft2D.cpp` | `forwardRows` | `#pragma omp task` | Empaqueta el procesamiento de cada fila individual como una tarea asíncrona en el pool de OpenMP. | Test 8 (`testOpenMpTaskVsParallelFor`) |
| `src/Fft2D.cpp` | `forwardRows` | `#pragma omp single` | Permite que un único hilo productor genere y encole las tareas de filas mientras el resto del equipo las consume. | Test 8 (`testOpenMpTaskVsParallelFor`) |
| `src/Fft2D.cpp` | `forwardRows` | `#pragma omp taskwait` | Espera las tareas hijas del hilo que lo ejecuta. El final de la región paralela garantiza la finalización de todas las filas. | Test 8 (`testOpenMpTaskVsParallelFor`) |
| `src/Fft2D.cpp` | `forwardRows` | `firstprivate(r)` | Cada tarea recibe una copia privada e inmutable de su índice de fila $r$. | Test 8 (`testOpenMpTaskVsParallelFor`) |
| `src/Fft2D.cpp` | `colsPass` | `#pragma omp parallel` + buffer local | Se reserva un único buffer `col_buf(M)` por hilo dentro de la región paralela, eliminando $N$ asignaciones dinámicas y optimizando la caché. | Test 2, 6, 10 |
| `src/Fft2D.cpp` | `inverse` | `#pragma omp parallel for collapse(2)` | Colapsa los dos bucles anidados independientes ($M \times N$) para normalizar la matriz por el factor de escala $\frac{1}{M \cdot N}$ en paralelo. | Test 1, 6, 7, 10 |
| `src/Fft2D.cpp` | Todas | `default(none)`, `shared(...)` | Control estricto de ámbito de variables para evitar condiciones de carrera y garantizar transparencia en OpenMP. | Todos los tests |

---

## 5. Rol 2: Núcleo Mariposa 1D (`Butterfly1D`)

El Rol 2 implementa el núcleo de la transformada rápida de Fourier unidimensional (FFT e IFFT 1D) radix-2 para arreglos contiguos de tamaño potencia de 2 ($N = 2^p$).

### 5.1. Convención de Escala y Contrato
- **Transformada directa e inversa 1D sin escalado interno:** La mariposa 1D **no normaliza por $1/N$** en sus pasadas individuales. `inverse = true` únicamente invierte el signo de la fase del factor twiddle ($\omega_m^r = e^{+2\pi i r / m}$).
- **Ida y Vuelta 1D:** $\text{IFFT}_{1\text{D}}(\text{FFT}_{1\text{D}}(x)) = N \cdot x$. La escala global $\frac{1}{M \cdot N}$ la aplica `Fft2D` al finalizar ambas pasadas ortogonales en 2D.

### 5.2. Layouts y Modos Implementados
1. `layout = 0`: **Cooley–Tukey (in-place)** con permutación previa *bit-reversal*. Recorre $p = \log_2 N$ etapas modificando el arreglo en memoria compartida sin necesidad de memoria auxiliar extra.
2. `layout = 1`: **Stockham (out-of-place / autosort)** sin permutación *bit-reversal*. Recorre $p = \log_2 N$ etapas alternando lecturas y escrituras entre dos buffers (`src` y `dst`), entregando el espectro en orden natural.

### 5.3. Suite de Pruebas y Tolerancia Numérica (`test_butterfly1d.cpp`)
La suite `make test-role2` ejecuta **10 pruebas rigurosas** que validan la matemática y el paralelismo (error absoluto $< 10^{-10}$):
- *Test 1 (Impulso N=4):* Comprueba respuesta $(1,1,1,1)$ en ambos layouts.
- *Test 2 (Equivalencia N=8):* Coincidencia exacta entre Cooley-Tukey y Stockham en datos estocásticos.
- *Test 3 (Contrato sin escala):* Valida que $\text{IFFT}(\text{FFT}(x)) = N \cdot x$ y recuperación de $x$ con $/N$.
- *Test 4 (Schedules y Chunks):* Invariancia de resultados usando `static`, `dynamic`, `guided`, `collapse` y `transformStages`.
- *Test 5 (Cláusulas OpenMP):* Comprueba `#pragma omp single`, `firstprivate` y `lastprivate`.
- *Test 6 (Forma cerrada analítica):* Senoide 1D de 1 período con picos imaginarios puros $\pm 8i$ en $k=1, N-1$.
- *Test 7 (Escalabilidad de tamaño):* Precisión garantizada en $N=64, 256, 1024$.
- *Test 8 (Invariancia de hilos):* Resultados idénticos evaluando con 1 vs 8 hilos.
- *Test 9 (Casos borde):* Manejo robusto de $N=1$ y $N=2$.
- *Test 10 (Inversa directa 1D):* Reversibilidad directa en ambos layouts.

---

## 6. Mapeo de Cláusulas y Directivas OpenMP (Rol 2)

| Archivo | Función / Método | Cláusula / Directiva OpenMP | Motivo y Justificación Técnica | Test Asociado |
| :--- | :--- | :--- | :--- | :--- |
| `src/Butterfly1D.cpp` | `transformCooleyTukey` / `transformStockham` | `#pragma omp parallel for schedule(static/dynamic/guided, chunk)` | Reparto configurable de mariposas disjuntas dentro de cada etapa $s$. | Test 4 (`testSchedulesAndClauses`) |
| `src/Butterfly1D.cpp` | `transformStages` | `#pragma omp parallel` + `#pragma omp barrier` | Mantiene una sola región paralela con barrera explícita entre etapas para evitar carreras. | Test 4 (`testSchedulesAndClauses`) |
| `src/Butterfly1D.cpp` | `transformStockham` | `#pragma omp parallel for collapse(2)` | Fusiona los dos bucles anidados independientes de grupos $k$ y elementos $j$ en Stockham. | Test 4 (`testSchedulesAndClauses`) |
| `src/Butterfly1D.cpp` | `initTwiddlesSingle` | `#pragma omp single` | Permite la inicialización de la tabla de factores twiddle por un único hilo sin interferencia. | Test 5 (`testOpenMPClausesDemo`) |
| `src/Butterfly1D.cpp` | `accumulateFirstprivate` | `firstprivate(acc)` | Cada hilo recibe una copia privada e inicializada del acumulador base. | Test 5 (`testOpenMPClausesDemo`) |
| `src/Butterfly1D.cpp` | `stageIndexLastprivate` | `lastprivate(last_stage)` | Preserva el valor del índice de la última etapa evaluada fuera de la región paralela. | Test 5 (`testOpenMPClausesDemo`) |

---

## 7. Tabla de Roles del Equipo

| Rol | Responsable | Módulos Clave | Estado |
| :--- | :--- | :--- | :--- |
| **Rol 1: Modelo y Datos** | Integrante Rol 1 | `ComplexField.h/.cpp` | En desarrollo (`feat/Modelos-Datos`) |
| **Rol 2: Núcleo Mariposa 1D** | Integrante Rol 2 | `Butterfly1D.h/.cpp`, `test_butterfly1d.cpp` | Implementado y Validado (`feat/Mariposa`). Integración pendiente en `dev`.|
| **Rol 3: FFT e IFFT 2D** | Integrante Rol 3 | `Fft2D.h/.cpp`, `test_fft2d.cpp` | Implementado; validación aislada con sustitutos (`feat/FFT`). Integración pendiente en `dev`. |
| **Rol 4: Métricas y Benchmarks** | Integrante Rol 4 | `SpectrumMetrics.h/.cpp`, `Benchmark.h/.cpp` | Trabajo en su rama; integración posterior en `dev`. |
| **Rol 5: Calidad, CI y Visualización** | Integrante Rol 5 | `Visualizer.h/.cpp`, tests globales | Trabajo en su rama; integración posterior en `dev`. |
