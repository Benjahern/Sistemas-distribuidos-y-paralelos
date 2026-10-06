# Laboratorio 1 — FFT e IFFT 2D con OpenMP

C++17, doble precisión, radix-2 y grillas complejas `M × N` con ambos ejes
potencias de dos, incluso rectangulares. No se usan bibliotecas externas de FFT.
Los layouts Cooley–Tukey in-place y Stockham producen el mismo espectro en orden natural.

## Compilación y pruebas

Dependencias: GCC con OpenMP, Make y Python 3 con NumPy/Matplotlib (solo gráficos y
pruebas de las salidas). En Ubuntu/Debian:

```bash
sudo apt-get install build-essential python3-numpy python3-matplotlib
make
make test
make demo
./build/fft2d --help
```

`make all` enlaza `build/fft2d`; siguen disponibles los targets por módulo, por
ejemplo `make Butterfly1D`. Compila con `-std=c++17 -Wall -Wextra -Wpedantic -O2 -fopenmp`.
`make clean` elimina únicamente el directorio configurado con `BUILD_DIR`.
**No elimina `results/` ni la carpeta `OUTPUT`.** Las dependencias se instalan
manualmente. `make -j4` es opcional: compila hasta cuatro archivos simultáneamente,
sin configurar los hilos de la FFT.

### Todo el flujo con Make

```bash
# Compilar, probar, demostrar, medir y generar las seis salidas:
make pipeline

# Campaña corta en otra carpeta, conservando resultados anteriores:
make pipeline OUTPUT=results/quick \
  BENCHMARK_ARGS="--min-size 128 --max-size 256 --chunk-grid 128 --threads 1,2,4"
```

`pipeline` ejecuta **compilación → pruebas → demo → benchmark → gráficos**.
Los pasos son secuenciales incluso con `make -j4`; solo la compilación y las suites
que componen `test` pueden aprovechar Make en paralelo. El benchmark empieza
cuando todas las pruebas y la demo han terminado. No lanzar simultáneamente
otros targets o procesos de carga durante las mediciones.

| Target | Acción |
|---|---|
| `make` / `make all` | Solo compilar la aplicación |
| `make test` | Compilar y ejecutar la suite completa |
| `make demo` | FFT/IFFT aleatoria, Parseval por tres métodos y espectro de seno |
| `make benchmark` | Medir ambos layouts y producir las cuatro tablas `.dat` |
| `make plots` | Leer datos existentes y generar los PNG, sin repetir mediciones |
| `make pipeline` | Compilar, probar, ejecutar demo, medir y generar gráficos en orden |
| `make clean` | Limpiar objetos/binarios, sin borrar resultados |

`make all` sigue siendo rápido: no inicia benchmarks, Docker ni gráficos.
`pipeline` es el flujo **local**; Docker es opcional y no se ejecuta automáticamente.
No genera el reporte PDF ni sustituye el análisis de resultados.

`OUTPUT=carpeta` selecciona dónde guardar datos y figuras (por defecto `results`).
`BENCHMARK_ARGS="opciones"` pasa opciones directamente al benchmark, sin duplicar
la configuración del CLI en el Makefile. Para personalizar demo, espectro o escala
de los gráficos, usar directamente el ejecutable o el script, como se muestra abajo.

`make test` ejecuta pruebas 1D, composición 2D con sustitutos **y con los módulos
reales**, modelo de datos, métricas, estadísticas, Amdahl, CLI y generación de PNG.
`make test-role3` mantiene la suite aislada; `make test-integration` usa el núcleo real.
`TEST_THREADS=4` controla el equipo inicial de las suites antiguas; estas también
prueban explícitamente 1, 2, 4 y 8 hilos.

Cobertura: bit-reversal `N=8`, impulso y seno con forma cerrada e inversa para
`N=4,8,16`, impulso `2×2`, DFT de referencia `8×4`/`4×8`, seno 2D, Parseval,
semilla fija, layouts, schedules, tareas con/sin `single`, barreras en `N=4096`,
grilla `64×32` e invariancia ante hilos. Los tests verifican el contenido y nombres
de las tablas, las fórmulas estadísticas y las firmas PNG: no son targets vacíos.

## Convención matemática y métricas

La directa usa `exp(-2πi(km/M+ln/N))` **sin escala**. La inversa usa signo positivo,
deshace columnas y filas y aplica **una sola vez** `1/(M*N)` al final. `Butterfly1D`
mantiene su contrato previo sin normalización en ambos signos: para usar una
IFFT 1D aislada, el llamador divide por su longitud. Las pruebas hacen esa división.

`SpectrumMetrics(original, other)` compara `other` con la entrada:

- `roundtripError()`: `other` debe ser la reconstrucción;
  `RMSE = sqrt(sum(norm(original-other))/(M*N))`.
- `parseval(method[, use_private])`: `other` debe ser el espectro;
  `Ex=sum(norm(x))`, `EX=sum(norm(X))/(M*N)`. Devuelve ambas energías y error
  absoluto y relativo; para energía de entrada cero reporta la diferencia absoluta.
- `method=0`: reduction, `1`: atomic, `2`: critical. Con `use_private=true` se usa
  una variable scratch explícitamente privada; con `false`, la expresión directa.

Se exige RMSE absoluto `<=1e-10` y discrepancia relativa de Parseval `<=1e-10` en
los benchmarks. Es una cota conservadora para datos uniformes en `[-1,1]` de doble
precisión: la FFT acumula redondeo en `O(log M + log N)` etapas. No constituye una
garantía para amplitudes arbitrarias. Se valida **cada configuración medida** y se
aborta ante resultados no finitos o fuera de tolerancia.

## Benchmarks reproducibles

```bash
# Campaña acotada, ambos layouts, al menos diez repeticiones por punto:
OMP_PROC_BIND=close OMP_PLACES=cores make benchmark \
  BENCHMARK_ARGS="--min-size 128 --max-size 1024 --chunk-grid 256 --threads 1,2,4,8 --repetitions 10 --seed 42"
make plots

# Ampliar incluyendo los procesadores lógicos disponibles (Linux):
OMP_PROC_BIND=close OMP_PLACES=cores make benchmark OUTPUT=results/full \
  BENCHMARK_ARGS="--min-size 128 --max-size 4096 --chunk-grid 256 --threads 1,2,4,8,$(nproc) --memory-mib 512"

# Regenerar figuras sin medir de nuevo:
make plots OUTPUT=results/full
```

`make benchmark` mide y escribe datos; `make plots` añade los PNG.
Usa tamaños desde 128 hasta 1024 por defecto; sin `--threads`,
agrega `omp_get_num_procs()` a `1,2,4,8`, sin duplicados. Siempre incluye un hilo.
El barrido de chunks usa `--chunk-grid` (256 por defecto), chunks `1,2,4,8,16,32,64`,
ambos layouts y los schedules static/dynamic/guided. Mide un hilo y el mayor equipo
solicitado; `--chunks 1,4,16` permite cambiar el barrido.

Se mide **solo la FFT 2D directa completa** con `omp_get_wtime()`: incluye buffers,
bit-reversal, cálculo de twiddles, copias de columnas y sincronización necesarios
para la llamada. Generación de entrada, restauración antes de cada repetición,
calentamiento, IFFT, métricas y escritura quedan fuera del cronómetro. No se
transforma repetidamente el espectro anterior. Los equipos se fijan con
`omp_set_dynamic(0)`; se registra el número real de hilos y se aborta si difiere
del solicitado. La configuración OpenMP del llamador se restaura al finalizar.

`T = promedio ± desviación estándar muestral` (denominador `R-1`, Welford).
El tiempo de referencia `T1` usa **el mismo binario, entrada, tamaño, layout,
schedule y chunk**, no otra implementación serial:

```text
Sp = mean(T1)/mean(Tp)
Ep = Sp/p
sigma_Sp = Sp * sqrt((sigma_T1/mean_T1)^2 + (sigma_Tp/mean_Tp)^2)
sigma_Ep = sigma_Sp/p
```

Son desviaciones estándar de tiempos y su propagación solicitada, **no** errores
estándar de la media ni intervalos de confianza. Para `p=1`, `T1/T1` es exactamente
uno y su incertidumbre cero; no son dos variables independientes. Las demás
propagaciones suponen independencia entre las mediciones de referencia y paralela.

### Amdahl y límites

Por cada tamaño/layout/schedule/chunk se ajusta por mínimos cuadrados ordinarios:

```text
Tp/T1 = 1/p + f*(1-1/p)
Sp_Amdahl = 1/(f+(1-f)/p)
```

`f` se acota a `[0,1]`; la tabla indica `fit_clamped` y el residual
`Sp_medido-Sp_Amdahl`. Si solo se midió un hilo, `f` y la predicción son `nan`
(no identificables). Las líneas son **un ajuste de las mismas mediciones**, no una
predicción independiente. Con un solo punto paralelo, el ajuste tampoco permite
validar capacidad predictiva. No se estima ni propaga incertidumbre de `f`.

Esta es una **fracción serial efectiva**, no un perfil directo: bit-reversal y
swaps se ejecutan secuencialmente dentro de una fila, pero diferentes filas los
ejecutan en paralelo. No se suman esos tiempos como fracción serial global.
Creación de equipos, memoria no contigua, reparto, asignaciones y sincronización
pueden producir desacuerdos con Amdahl, incluso `Sp<1` o superlinealidad.

La serie duplica el tamaño hasta `--max-size` o el presupuesto `--memory-mib`
(512 MiB por defecto). La cota incluye seis grillas equivalentes más temporales
por hilo; es conservadora y **no una medición del RSS máximo**. Las cabeceras de
las tablas indican el mayor tamaño medido y por qué se omitió el siguiente. Un
tope configurado no demuestra el máximo físico de la máquina. No ejecutar tests,
compilaciones u otras cargas simultáneamente durante una campaña de rendimiento.

## Archivos y siete gráficos

En `--output` (por defecto `results/`), con **espacios en los nombres**:

| Archivo | Contenido |
|---|---|
| `benchmark results.dat` | Tamaño, layout, schedule, chunk, hilos solicitados/reales, repeticiones, promedio/desviación, RMSE, Parseval |
| `scaling analysis.dat` | Speedup/eficiencia e incertidumbres, `f`, Amdahl, residual y ajuste acotado |
| `performance plots.png` | Seis paneles: speedup, eficiencia, tiempo/chunk, tiempos de layouts, Amdahl y RMSE |
| `spectrum.dat` | `k l abs(X[k,l])`, orden natural, escala lineal |
| `spectrum.png` | Séptimo gráfico: módulo de un seno 2D conocido |
| `roundtrip error.dat` | RMSE y error relativo de Parseval por configuración |

Cada tabla incluye una cabecera con las columnas. Layout `0=in-place`, `1=Stockham`;
schedule `0=static`, `1=dynamic`, `2=guided`. `chunk=0` identifica la serie de
escalabilidad y el reparto static por defecto, no el barrido de chunks positivos.
Las barras muestran desviación estándar o incertidumbre propagada, según el eje.

El benchmark también exporta un seno `64×64` de frecuencias `(1,1)`: sus picos
están en `(1,1)` y `(63,63)`, con módulo `MN/2=2048`. Para visualizar otro seno:

```bash
./build/fft2d spectrum --rows 64 --cols 32 --k 3 --l 5 \
  --layout stockham --threads 4 --output results/sine
python3 scripts/plot_results.py --input results/sine --spectrum-scale log
```

La figura usa escala lineal por defecto; `log` significa explícitamente
`log10(1+|X|)`. El script lee las tablas, no inventa tiempos ni ejecuta una FFT.
`demo` y `spectrum` solo exportan el espectro; sin tablas de benchmark el script
genera únicamente `spectrum.png`. Las frecuencias degeneradas (seno nulo) se rechazan.
Usar carpetas distintas para no mezclar campañas: nuevas ejecuciones reemplazan
los archivos del mismo nombre.

## Arquitectura, roles y OpenMP

| Responsabilidad / rol de referencia | Módulos |
|---|---|
| 1: modelo y datos | `ComplexField.h/.cpp`: memoria row-major, semilla, campos `.dat` |
| 2: núcleo 1D | `Butterfly1D.h/.cpp`: implementación original de `origin/feat/Mariposa`, ambos layouts |
| 3: composición 2D | `Fft2D.h/.cpp`: filas, columnas y normalización |
| 4: métricas y benchmarks | `SpectrumMetrics.h/.cpp`, `Benchmark.h/.cpp` |
| 5: calidad y visualización | `Visualizer.h/.cpp`, `scripts/plot_results.py`, `tests/`, Docker/CI |
| Integración | `src/main.cpp`: CLI y coordinación, sin copiar algoritmos |

Los nombres de integrantes y la distribución en un equipo de cuatro deben ser
completados por el equipo; no se atribuyen responsables ficticios.

| Archivo / función | Cláusula | Justificación / ruta probada |
|---|---|---|
| `Fft2D.cpp`, `rowsPass`/`colsPass` | `parallel for`, `schedule(runtime)`, `shared` | Filas/columnas independientes; schedule y chunk configurables en benchmarks |
| `Fft2D.cpp`, `forwardRows` | `task`, `single`, `taskwait`, `firstprivate(r)` | Tareas de filas con índice propio; comparadas con parallel for por la suite real |
| `Fft2D.cpp`, `inverse` | `collapse(2)` | Normalización de celdas independientes |
| `Butterfly1D.cpp`, `transformCooleyTukey` | `parallel for`, schedules, `barrier` | Barreras implícitas entre etapas; `transformStages` usa región persistente y barrera explícita |
| `Butterfly1D.cpp`, `transformStockham` | `parallel for`, schedules, `collapse(2)` | Collapse solo parejas independientes de una etapa; barrera implícita antes del intercambio de buffers |
| `Butterfly1D.cpp`, `initTwiddlesSingle` | `single` | Un productor de twiddles, probado por suite 1D |
| `Butterfly1D.cpp`, `accumulateFirstprivate` | `firstprivate(acc)` | Copia inicializada por hilo, probada por suite 1D |
| `Butterfly1D.cpp`, `stageIndexLastprivate` | `lastprivate(last_stage)` | Última iteración lógica, no último hilo en terminar |
| `SpectrumMetrics.cpp`, `roundtripError` | `reduction`, `collapse(2)` | Suma de errores sin carreras |
| `SpectrumMetrics.cpp`, `energy` | `reduction`, `atomic`, `critical`, `private`, `shared` | Tres sumas reales de Parseval comparadas por pruebas y demo |
| `Visualizer.cpp`, `magnitudeNoWait` | `for nowait`, `shared` | Celdas independientes; fin de región sincroniza antes de consumir resultados |

Se conservan ambos niveles: FFT 2D paralela por transformadas independientes y
FFT 1D aislada paralela por mariposas. El núcleo original crea regiones OpenMP
internas; con `OMP_MAX_ACTIVE_LEVELS=1` no se activan equipos anidados.
El CLI fija ese límite para demo/espectro. No se eliminan dependencias entre etapas.

Por petición del equipo se conserva el núcleo de `origin/feat/Mariposa` sin
refactorizar: repite la operación de mariposa entre variantes, por lo que el
requisito de una única lógica de mariposa queda pendiente. El núcleo 1D presupone
longitudes potencia de dos; no llamar directamente con longitudes inválidas.
Los resultados de rendimiento generados con el núcleo refactorizado anterior
deben medirse de nuevo antes de atribuirlos a esta versión restaurada.

## Docker y CI

```bash
docker build -t lab1-fft2d .
docker run --rm lab1-fft2d
docker run --rm lab1-fft2d make test
```

El Dockerfile instala dependencias, compila, ejecuta `make test` y una demo corta
al construir. `.dockerignore` evita copiar objetos locales o resultados viejos.
Los workflows en `../.github/workflows/` ejecutan pruebas reales en pushes y pull
requests hacia `main`/`develop` que afectan a `Lab1` o al workflow correspondiente.
`build.yml` instala dependencias y ejecuta `make test`, que compila y corre la suite.
`docker.yml` construye la imagen, ejecuta explícitamente `make test` dentro del
contenedor y luego la demo. CI no ejecuta una campaña de rendimiento independiente.

El reporte técnico PDF y la asignación nominal de roles son entregables separados;
esta implementación no los da por completados.
