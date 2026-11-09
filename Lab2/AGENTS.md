# AGENTS.md — Guía para Agentes de IA y Desarrolladores

## 1. Contexto del Proyecto
Este repositorio corresponde al **Laboratorio 2: Programación GPGPU con CUDA (Filtrado de imagen 2D y detección de bordes Sobel)** del Departamento de Ingeniería Informática, Universidad de Santiago de Chile (USACH).

El objetivo es implementar un sistema de procesamiento de imágenes 2D en escala de grises en **C++17 y CUDA** que realice:
1. **Suavizado (Blur):** Convolución espacial 2D con kernels cuadrados normalizados ($\sum k = 1$) de lado impar $s \in \{3, 5, 7\}$.
2. **Detección de bordes (Sobel):** Cálculo de gradientes $G_x = I * S_x$ y $G_y = I * S_y$ reutilizando la misma API de convolución genérica, seguido de un operador *map* punto a punto para la magnitud $G(i,j) = \sqrt{G_x(i,j)^2 + G_y(i,j)^2}$.

---

## 2. Reglas Estrictas e Inquebrantables (Guardrails)
Cualquier agente o desarrollador que modifique este código **DEBE** cumplir las siguientes restricciones bajo pena de invalidar la entrega:

1. **PROHIBICIÓN DE LIBRERÍAS EXTERNAS EN EL CAMINO MEDIDO:**
   * Está **estrictamente prohibido** usar `OpenCV`, `NPP`, `cuDNN`, `cuFFT` o `Thrust` en cualquier código que calcule la convolución, el operador de Sobel, las reducciones o que forme parte de los benchmarks.
   * La referencia de corrección debe ser código **CPU serial puro** escrito por el equipo (`OpenMP` solo se permite como extra opcional y jamás reemplaza la referencia serial).
2. **CONVENCIÓN MATEMÁTICA Y DE BORDES ÚNICA:**
   * **Layout de memoria:** Arreglo lineal `float` en orden fila-mayor (*row-major*), indexado como `i * W + j` (donde `i` es fila $[0, H-1]$, `j` es columna $[0, W-1]$, origen arriba-izquierda).
   * **Política de borde:** Se debe elegir **una sola** política (`Zero-padding`, `Clamp` o `Mirror`), documentarla y aplicarla de forma **idéntica** en CPU serial, GPU *naive* y GPU *shared*.
   * **Notación:** $s = 2r + 1$ es el lado del kernel; $k(u,v)$ son los coeficientes. No usar la misma variable para ambos conceptos.
   * **Operadores de Sobel fijos ($3 \times 3$):** No alterar signos ni permutar matrices:
     $$S_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}, \qquad S_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$$
3. **REUTILIZACIÓN DE API (NO DUPLICAR LÓGICA):**
   * No existe un "algoritmo Sobel" separado de la convolución: $G_x$ y $G_y$ **deben** calcularse invocando exactamente la misma función `convolveCpu` / `convolveGpu` pasando $S_x$ y $S_y$ como arreglo de coeficientes $k$.
   * En las sobrecargas de métodos, los métodos simples deben delegar en los métodos parametrizados.

---

## 3. Arquitectura de Agentes / Roles del Equipo
El diseño modular sigue los 5 roles definidos en el laboratorio. Incluso en grupos de 4 integrantes (donde una persona asume 2 roles), **las interfaces de código deben mantenerse separadas y limpias**:

### Agente 1: Imagen y I/O (`ImageBuffer.h/.cpp`)
* **Responsabilidad:** Gestión del buffer en host $(H \times W)$ en formato `float`.
* **Tareas:**
  * Carga y escritura de imágenes en formato PGM o `.dat`.
  * Generación determinista (con semilla fija) de imágenes sintéticas para tests: impulso unitario, escalón vertical, escalón horizontal y ruido reproducible.
  * Normalización de rangos (declarar en `README.md` si se opera en $[0, 1]$ o $[0, 255]$).

### Agente 2: Convolución CPU (`Convolution2D.h/.cpp`, `Sobel.h/.cpp`)
* **Responsabilidad:** Referencia serial en CPU y validación algorítmica base.
* **Tareas:**
  * Implementar `convolveCpu(const float* k, int s)` respetando la convención de signos $(I * k)(i,j) = \sum_{u=-r}^r \sum_{v=-r}^r k(u,v) I(i-u, j-v)$ (o la reflejada, siempre que sea idéntica en GPU).
  * Implementar `computeGxGyCpu()` y `computeMagnitudeCpu()`.
  * Garantizar cero fugas de memoria en host.

### Agente 3: Kernels CUDA (`Convolution2D.cu`, `Sobel.cu`, `Metrics.cu`)
* **Responsabilidad:** Implementación de primitivas en *device* y verificación de equivalencia numérica.
* **Tareas y Requisitos Técnicos Obligatorios:**
  * Macro `CUDA_CHECK(call)` obligatoria tras cada llamada a la API de CUDA y `cudaGetLastError()` tras cada lanzamiento de kernel.
  * Al menos una función auxiliar `__device__` (ej. para aplicar la política de borde o indexar).
  * **Variante 0 (`naive`):** 1 hilo por píxel $(i,j)$, lectura directa desde memoria global, guarda de límites `if (i >= H || j >= W) return;`.
  * **Variante 1 (`shared`):** Bloques 2D que cargan un *tile* de dimensión `blockDim` más un **halo** de radio $r$ en memoria compartida (`__shared__`), sincronizan con `__syncthreads()`, y calculan la convolución leyendo exclusivamente desde `__shared__`.
  * **Kernel Map Sobel:** `computeMagnitudeGpu()` para calcular $\sqrt{G_x^2 + G_y^2}$.
  * **Reducción y Atómicas (`Metrics.cu`):** Implementar `sumMagnitude(int method)` donde `method = 0` usa reducción en *shared memory* (en una o dos pasadas) y `method = 1` usa `atomicAdd`.

### Agente 4: Host/Device y Benchmarks (`Benchmark.h/.cpp`)
* **Responsabilidad:** Gestión de memoria GPU (RAII), medición de tiempos y análisis de rendimiento.
* **Tareas:**
  * Administrar punteros *device* (`cudaMalloc`, `cudaMemcpy` H2D/D2H, `cudaFree`) minimizando copias innecesarias.
  * Medir con `std::chrono::steady_clock` en host usando `cudaDeviceSynchronize()` para acotar el tramo GPU ($\ge 10$ repeticiones por punto, reportando media $\overline{T}$ y desviación estándar $\sigma_T$).
  * Separar estrictamente **`benchmarkKernelOnly`** (solo ejecución de kernels + sync) de **`benchmarkEndToEnd`** (incluye copias H2D de entrada + kernels + D2H de salida).
  * Ejecutar la matriz de experimentos: tamaños $\{512^2, 1024^2, 2048^2, 4096^2\}$, pipelines (blur $s=3$, blur $s=5$ o $7$, Sobel completo), variantes (`naive` vs `shared`), y geometrías de bloque $\{(16,16), (16,32), (32,16), (32,32)\}$.
  * Calcular Speedup ($S$), mejora de memoria compartida ($R = T_{\text{naive}}/T_{\text{shared}}$), propagación de incertidumbre ($\sigma_S$) y Ley de Amdahl ($S_{\text{pred}}$).

### Agente 5: Calidad, CI y Visualización (`tests/`, `Visualizer.h/.cpp`, `Dockerfile`, `.github/workflows/`)
* **Responsabilidad:** Pruebas unitarias automatizadas, contenedorización y generación de gráficos/paneles.
* **Tareas:**
  * Implementar la suite de pruebas (`make test`) cubriendo todos los casos obligatorios de la Sección 11.
  * Configurar `Dockerfile` y pipeline CI (contemplando ejecución de tests CPU y *smoke build* con `nvcc` si el runner de CI no dispone de GPU física).
  * Generar los archivos `.dat` y los scripts de visualización para producir `performance_plots.png` y `sobel_panel.png`.

---

## 4. Firmas y Contratos de API Obligatorios
No cambiar los nombres ni la semántica de los siguientes métodos exigidos por el enunciado:

```cpp
// En Convolution2D:
void convolveCpu(const float* k, int s);
void convolveGpu(const float* k, int s); // Delega en la variante por defecto
void convolveGpu(const float* k, int s, int variant); // 0 = naive, 1 = shared
void convolveGpu(const float* k, int s, int variant, dim3 block);

// En Sobel:
void computeGxGyCpu();
void computeGxGyGpu(int variant);
void computeMagnitudeCpu();
void computeMagnitudeGpu();

// En Metrics:
float maxAbsDiffCpuGpu(/* buffers comparables */);
float sumMagnitude(int method); // 0 = shared reduction, 1 = atomicAdd

// En Benchmark:
void benchmarkKernelOnly(/* ... */);
void benchmarkEndToEnd(/* ... */);
void compareCpuGpu(/* ... */);
```

---

## 5. Flujo de Compilación y Verificación

### Comandos Estándar
* **Compilar proyecto completo:**
  ```bash
  make all
  ```
  *(Nota: Asegurar que `NVCCFLAGS` incluya `-O3 -std=c++17 -Xcompiler -Wall`)*
* **Ejecutar suite de pruebas automáticas:**
  ```bash
  make test
  ```
* **Limpiar binarios:**
  ```bash
  make clean
  ```

### Lista de Verificación para `make test` (Checklist de Pruebas)
Antes de dar por terminada cualquier tarea en los kernels o en la lógica de convolución, verificar que `make test` valide:
1. [ ] **Impulso $5 \times 5$ con kernel $3 \times 3$:** La salida en CPU coincide exactamente con el kernel centrado según la política de borde.
2. [ ] **Escalón Vertical y Horizontal:** En escalón vertical se cumple $|G_x| > |G_y|$ sobre el borde; en escalón horizontal se cumple $|G_y| > |G_x|$.
3. [ ] **Equivalencia CPU vs. GPU Naive vs. GPU Shared:** `maxAbsDiffCpuGpu()` está dentro de la tolerancia ($\sim 10^{-4}$ a $10^{-3}$) en imágenes de $8 \times 8$, $16 \times 16$ y $32 \times 32$.
4. [ ] **Magnitud de Sobel:** En píxeles de prueba, $G$ coincide con $\sqrt{G_x^2 + G_y^2}$.
5. [ ] **Reducción vs. Atómicas:** `sumMagnitude(0)` y `sumMagnitude(1)` coinciden dentro de la tolerancia relativa documentada.
6. [ ] **Test de Perímetro (Borde):** Un test específico en la fila `0`, fila `H-1`, columna `0` o columna `W-1` que falle si el halo o la política de borde están mal indexados.

---

## 6. Ejecución en Clúster DIINF (Slurm)
* Las pruebas de desarrollo se realizan localmente o en CI, pero **todos los datos finales reportados** (`benchmark_results.dat`, `block_study.dat`, `cluster_run.log`) deben provenir de la partición GPU del clúster del DIINF usando Slurm (`sbatch`).
* El script `sbatch` debe registrar en el log: `job_id`, nodo asignado (`hostname`), modelo de GPU (`nvidia-smi`), versión del driver, `nvcc --version` y los flags de compilación utilizados.