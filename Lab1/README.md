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
OMP_PROC_BIND=close OMP_PLACES=threads make benchmark OUTPUT=results/full \
  BENCHMARK_ARGS="--min-size 128 --max-size 4096 --chunk-grid 256 --threads 1,2,4,8,$(nproc) --repetitions 10 --seed 42 --memory-mib 2048"

# Regenerar figuras sin medir de nuevo:
make plots OUTPUT=results/full
```

El ejemplo ampliado reserva un presupuesto interno de 2048 MiB para permitir
medir hasta `4096×4096`. Con 512 MiB, la cota conservadora del programa detiene
esa serie antes de `4096×4096`.

### Repetir las campañas del clúster

Desde `Lab1/` en el clúster con Slurm y la partición `batch`, enviar cada campaña
por separado y esperar a que finalice antes de iniciar otra medición en el mismo
nodo. Los scripts compilan, ejecutan `make test`, miden y generan los gráficos.

| Campaña | Envío a Slurm | Serie de tamaños | Grilla del barrido de chunks |
|---|---|---|---|
| 96 CPU, ampliación | `sbatch benchmark-96-8192.slurm` | 128 a 8192, duplicando el lado | `256×256` |
| 126 CPU, ampliación | `sbatch benchmark-126-8192.slurm` | 128 a 8192, duplicando el lado | `256×256` |
| 96 CPU, chunks | `sbatch benchmark-chunks-96.slurm` | `2048×2048` | `2048×2048` |
| 126 CPU, chunks | `sbatch benchmark-chunks-126.slurm` | `2048×2048` | `2048×2048` |

Las cuatro campañas usan diez repeticiones, semilla 42, ambos layouts y un
presupuesto interno de 8192 MiB. Las de 96 CPU prueban
`1,2,4,8,16,32,64,96` hilos; las de 126 añaden `126`. El barrido compara
`static`, `dynamic` y `guided`, chunks `1,2,4,8,16,32,64`, con un hilo y con el
máximo asignado. Las CPU asignadas son lógicas; el servidor de las mediciones
tiene 64 núcleos físicos y 128 CPU lógicas.

Los scripts fijan `OMP_DYNAMIC=false`, `OMP_PLACES=threads`,
`OMP_PROC_BIND=close` y `OMP_MAX_ACTIVE_LEVELS=1`; solicitan 16 GiB a Slurm.
`--time=02:00:00` es el límite de ejecución solicitado. La duración del benchmark
se obtiene de `started_utc` y `finished_utc` en su manifiesto; no incluye la cola
ni la compilación y las pruebas previas.

Cada envío escribe en `resultados-xi/<SLURM_JOB_ID>/`, con su propio
`benchmark_manifest.json` y `entorno.txt`. Los resultados analizados en el informe
corresponden a los jobs 1310164 y 1310165 (ampliaciones) y 1310166 y 1310167
(chunks), respectivamente. Un nuevo envío recibe otro identificador y puede
producir tiempos diferentes aun con los mismos parámetros.

`make benchmark` mide y escribe datos; `make plots` añade los PNG.
También escribe `benchmark_manifest.json`: comando, fechas UTC, afinidad y entorno
OpenMP, SHA-256 del ejecutable, fuentes y tablas. Así se identifica la campaña sin
deducir su procedencia de fechas de archivos ni de afirmaciones del README. El
wrapper comprueba que fuentes y ejecutable no cambien durante la medición.
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

En `--output` (por defecto `results/`), usando **guiones bajos** en los nombres compuestos:

| Archivo | Contenido |
|---|---|
| `benchmark_results.dat` | Tamaño, layout, schedule, chunk, hilos solicitados/reales, repeticiones, promedio/desviación, RMSE, Parseval |
| `scaling_analysis.dat` | Speedup/eficiencia e incertidumbres, `f`, Amdahl, residual y ajuste acotado |
| `performance_plots.png` | Seis paneles: speedup, eficiencia, tiempo/chunk, tiempos de layouts, Amdahl y RMSE |
| `spectrum.dat` | `k l abs(X[k,l])`, orden natural, escala lineal |
| `spectrum.png` | Séptimo gráfico: módulo de un seno 2D conocido |
| `roundtrip_error.dat` | RMSE y error relativo de Parseval por configuración |
| `benchmark_manifest.json` | Procedencia verificable de una ejecución de `make benchmark` |

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
los archivos del mismo nombre. Al exportar un espectro nuevo se elimina su PNG
anterior; al escribir tablas de benchmark se invalidan el gráfico de rendimiento
y el manifiesto anteriores. Ejecutar `make plots` después de generar datos. La PNG
del espectro incluye metadatos SHA-256 de `spectrum.dat`, dimensiones y escala,
para comprobar que ambos archivos corresponden a la misma exportación.

## Arquitectura, roles y OpenMP

El equipo está integrado por Marco Ortiz, Mateo Valle, Nicolás Sarabia y
Benjamín Hernández. La distribución coincide con el reporte técnico:

| Rol | Responsable | Módulos e interfaces |
|---|---|---|
| 1: modelo y datos | Marco Ortiz | `ComplexField.h/.cpp`: grilla, memoria, bit-reversal, semilla e I/O `.dat` |
| 2: núcleo mariposa 1D | Mateo Valle | `Butterfly1D.h/.cpp`: mariposa compartida, ambos layouts, twiddles, schedules y barreras |
| 3: FFT e IFFT 2D | Nicolás Sarabia | `Fft2D.h/.cpp`: pasadas por filas/columnas, composición 2D y normalización |
| 4: métricas y benchmarks | Benjamín Hernández | `SpectrumMetrics.h/.cpp`, `Benchmark.h/.cpp`: error, Parseval, tiempos y estadísticas |
| 5: calidad, CI y visualización | Benjamín Hernández | `Visualizer.h/.cpp`, `scripts/plot_results.py`, `tests/`, Docker y CI |

Benjamín Hernández cubre los roles 4 y 5 en este equipo de cuatro. Las interfaces
se mantienen separadas: las métricas y el benchmark usan la API pública de
`Fft2D` para obtener números y tablas; la visualización lee el espectro o las
tablas y genera figuras. Las pruebas verifican esos módulos mediante sus
interfaces. `src/main.cpp` coordina el CLI y delega en las clases, sin duplicar
la lógica de la transformada.

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

El núcleo 1D implementa una única función elemental de mariposa radix-2
(`Butterfly1D::butterfly`), compartida tanto por Cooley–Tukey in-place como por
Stockham out-of-place y por todas las variantes de schedule y sincronización OpenMP,
cumpliendo el requisito de una única lógica de mariposa sin duplicación. El núcleo 1D
presupone longitudes potencia de dos; no llamar directamente con longitudes inválidas.

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

El reporte técnico PDF se entrega junto al código, los tests, las tablas y las
figuras. Al preparar el paquete, incluir también `.github/workflows/`, ubicado
en la raíz del repositorio (un nivel sobre `Lab1/`), para conservar la CI.
