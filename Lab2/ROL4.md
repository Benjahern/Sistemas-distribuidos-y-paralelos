# Rol 4 — Host/device y benchmarks

## Trabajo realizado

- Se integró la rama `rol3-lab2` en `feat/rol4` para disponer de los kernels CUDA de convolución naive/shared, Sobel y métricas.
- Se añadió `DeviceBuffer`, un propietario RAII no copiable y movible para memoria device, con reserva/liberación y copias H2D/D2H.
- Se agregaron lanzadores de integración en `Convolution2D.cu` y `Sobel.cu`. Permiten ejecutar los kernels ya implementados del Rol 3 sobre buffers device que administra y reutiliza el Rol 4.
- Se completaron `benchmarkKernelOnly`, `benchmarkEndToEnd` y `compareCpuGpu` en `Benchmark.cpp`:
  - kernel-only mide lanzamientos y sincronización; no incluye transferencias;
  - end-to-end incluye copia de entrada/kernel, pipeline GPU y descarga de salida;
  - ambas modalidades validan al menos 10 repeticiones y resumen media y desviación estándar;
  - Sobel conserva `Gx`, `Gy` y magnitud en device durante el pipeline.
- Se incorporó `Benchmark::runSuite` con tamaños 512, 1024, 2048 y 4096; blur `s=3`, blur `s=5` y Sobel; variantes naive/shared y geometrías `(16,16)`, `(16,32)`, `(32,16)`, `(32,32)`.
- La suite verifica la salida frente a CPU (`max error <= 1e-3`) y calcula speedup, incertidumbre propagada, fracción serial y predicción de Amdahl.
- Se implementó la CLI `benchmark` en `main.cpp` y se documentó su uso en `README.md`.

## Archivos de resultados

La suite escribe en el directorio indicado:

- `benchmark_results.dat`: conserva las 15 columnas consumidas por `scripts/plot_results.py`. Como el contrato no tiene columna para geometría, agrega las muestras de las cuatro geometrías por cada tamaño/pipeline/variante.
- `block_study.dat`: contiene las seis columnas esperadas para el estudio de bloques; registra las cuatro geometrías para blur `s=3`.

No se han generado resultados GPU en este entorno.

## Ejecución

Desde `Lab2/`, en un entorno compilado con CUDA y con GPU accesible:

```bash
make clean all
build/image_cuda benchmark --output results --repetitions 10
make plots OUTPUT=results
```

La corrida final en el clúster DIINF/Slurm y sus evidencias siguen correspondiendo al Rol 5.

## Validación y pendientes

- `make -C Lab2 test`: pasó en el entorno CPU; los tests que requieren GPU se omitieron.
- `git diff --check`: pasó.
- No se pudo compilar ni ejecutar el camino CUDA porque `nvcc` no está instalado. La GPU aparece en `nvidia-smi`, pero eso no valida el código CUDA.
- Pendiente: compilar con `nvcc`, probar equivalencia y temporización en GPU, inspeccionar los `.dat` generados y corregir cualquier problema encontrado.

Seguimiento: [issue #11](https://github.com/Benjahern/Sistemas-distribuidos-y-paralelos/issues/11). Integración propuesta: [PR #13 hacia `develop`](https://github.com/Benjahern/Sistemas-distribuidos-y-paralelos/pull/13).
