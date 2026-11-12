# Laboratorio 2: Programación GPGPU con CUDA
**Filtrado de imagen 2D y detección de bordes (Sobel)**  
Departamento de Ingeniería Informática  
Universidad de Santiago de Chile  
10 de noviembre de 2026

---

## 1. Motivación

En el Laboratorio 1 el objeto de estudio fue la FFT 2D: una primitiva que, vía el teorema de convolución, permite filtrar en el dominio de la frecuencia. En este laboratorio el filtro se hace en el dominio espacial: convolución 2D con un kernel finito, y un caso aplicado de visión por computador, la detección de bordes con operadores de Sobel.

Suavizar ruido, resaltar bordes y preparar una imagen para segmentación aparecen en inspección industrial, radiología, teledetección y en el preproceso de pipelines de visión. La convolución encaja con el modelo CUDA: cada píxel de salida es independiente (patrón *map*), lee un vecindario (*stencil*) y se beneficia de memoria compartida si el *tile* reutiliza vecinos. La magnitud de Sobel, $G = \sqrt{G_x^2 + G_y^2}$, es un *map* punto a punto sobre $G_x$ y $G_y$; la verificación es contra una referencia CPU.

No se pide portar la FFT del Lab 1 a la GPU. En el camino medido (código que calcula convolución o Sobel y entra a los benchmarks) no se permite `cuFFT`, `NPP`, `OpenCV`, `cuDNN` ni `Thrust`. La referencia de corrección es una convolución CPU serial implementada por el grupo. `OpenMP` en CPU es opcional y no reemplaza esa referencia serial.

---

## 2. Objetivos

Al finalizar, Ud. será capaz de:
* Implementar convolución 2D con kernel cuadrado de lado impar y una política de borde documentada.
* Implementar Sobel $G_x$, $G_y$ y la magnitud $G$ reutilizando esa convolución.
* Escribir kernels CUDA (`__global__`) con índice 2D, protección de bordes y layout *row-major* en *device*.
* Contrastar una variante *naive* (solo memoria global) con una variante con *tile* en `__shared__`.
* Validar GPU frente a CPU serial con tolerancia en coma flotante justificada.
* Medir tiempos *kernel-only* y *end-to-end* (con transferencias), *speedup*, Amdahl e incertidumbre.
* Organizar el código en C++/CUDA, con pruebas, contenedor e integración continua.

---

## 3. Organización del equipo

Hay grupos de cinco integrantes y grupos de cuatro. Hay cinco roles. Cada persona asume un rol principal y puede colaborar en el resto. En un grupo de cuatro, un integrante cubre dos roles: en el `README.md` y en el reporte indique quién, qué dos roles, y cómo se separan las interfaces aunque las implemente la misma persona. En el test de salida se puede preguntar por el trabajo del rol, o de los dos roles si corresponde.

| Rol | Responsabilidades típicas |
| :--- | :--- |
| **1. Imagen y I/O** | Buffer host $(H \times W)$, carga/generación reproducible (PGM/`.dat`/sintética), normalización a `float`, escritura de salidas (blur, $G_x$, $G_y$, $G$). |
| **2. Convolución CPU** | Referencia serial de convolución 2D, política de borde, kernels de blur y Sobel; tests de corrección en imágenes pequeñas. OpenMP solo como extra opcional. |
| **3. Kernels CUDA** | Kernels *naive* y *shared-memory*; lanzadores `<<<grid, block>>>`; índice 2D; `CUDA_CHECK`; equivalencia numérica con la CPU serial. |
| **4. Host/device y benchmarks** | Buffers *device* (RAII), H2D/D2H, minimizar copias; mediciones *kernel-only* vs. *end-to-end*; barrido de bloques 2D; Amdahl e incertidumbre. |
| **5. Calidad, CI y visualización** | Pruebas, Docker (imagen con toolkit CUDA o multi-stage documentado), pipeline CI, figuras de *speedup* y del panel Sobel. |

---

## 4. Calendario sugerido

Publicación: martes 10 de noviembre de 2026. Entrega: martes 1 de diciembre de 2026, 23:59. La división siguiente es orientativa.

1. **10–16 nov:** repositorio, `Makefile` con `nvcc`, contenedor/CI mínimo; imagen sintética y convolución CPU $3 \times 3$ verificada a mano.
2. **17–23 nov:** kernel *naive* CUDA; tests CPU vs. GPU; Sobel $G_x$, $G_y$, $G$; primera figura de bordes.
3. **24 nov – 1 dic:** kernel con `__shared__`; benchmarks en nodo GPU del clúster; Amdahl, figuras, reporte y revisión cruzada por rol.

---

## 5. Descripción del problema

Se pide un programa en C++/CUDA que, dada una imagen en escala de grises $I$ de $H$ filas por $W$ columnas (valores en `float`; el rango, p. ej. $[0, 1]$ o $[0, 255]$, se declara en el `README`), calcule:

1. Un **blur** por convolución con un kernel cuadrado de lado impar $s \in \{3, 5, 7\}$. El lado $s = 3$ es obligatorio en tests y demos. En benchmarks debe medir al menos $s = 3$ y un lado mayor ($s = 5$ o $s = 7$). El kernel de blur debe estar normalizado (suma de coeficientes $= 1$).
2. Los **gradientes de Sobel** $G_x = I * S_x$, $G_y = I * S_y$ (siempre kernels $3 \times 3$ fijos de la sección 5.2) y la magnitud:
   $$G(i, j) = \sqrt{G_x(i, j)^2 + G_y(i, j)^2}$$

**Notación:** $s = 2r + 1$ es el lado del kernel; $k(u, v)$ son los coeficientes. No use la misma letra para ambos.

### 5.1. Convolución 2D

Con kernel de radio $r$ centrado en $(0, 0)$:

$$(I * k)(i, j) = \sum_{u=-r}^{r} \sum_{v=-r}^{r} k(u, v) I(i - u, j - v) \quad \text{(1)}$$

Si implementa la forma con $I(i + u, j + v)$ y el kernel reflejado, es equivalente: documente una convención y úsela igual en CPU y GPU. Origen de índices: arriba-izquierda; fila $i$, columna $j$.

**Política de borde:** elija exactamente una entre cero (*zero-padding*), *clamp* (repetir borde) o *mirror*, declárela en el `README` y aplíquela en CPU y GPU. Cambiar de política entre variantes invalida la comparación.

> ### 📊 Figura 1: Convolución $3 \times 3$ en dos pasos
> 
> **Paso 1 — Elegir el píxel de salida $(i, j)$**
> ```text
> +-----+-----+-----+-----+-----+
> |     |     |     |     |     |  Leyenda de la cuadrícula (5x5):
> +-----+-----+-----+-----+-----+  - [ (i,j) ] (Naranja): píxel central que se escribe.
> |     | [A] | [A] | [A] |     |  - [A] (Azul): los 9 vecinos (3x3) que se leen (incluye (i,j)).
> +-----+-----+-----+-----+-----+  - [ ] (Blanco): resto de la imagen.
> |     | [A] |(i,j)| [A] |     |
> +-----+-----+-----+-----+-----+
> |     | [A] | [A] | [A] |     |
> +-----+-----+-----+-----+-----+
> |     |     |     |     |     |
> +-----+-----+-----+-----+-----+
> ```
> $$\Downarrow \text{ luego}$$
> 
> **Paso 2 — Combinar esos vecinos con el kernel $k$**
> $$\boxed{(I * k)_{i,j} = \sum_{u=-1}^{1} \sum_{v=-1}^{1} k(u, v) I(i - u, j - v)}$$
> * **La misma función sirve para:**
>   * $\text{blur} \rightarrow k \text{ normalizado } (\sum k = 1)$
>   * $\text{Sobel} \rightarrow k = S_x \text{ o } k = S_y$
> * Solo cambia el arreglo $k$.
> * **CPU:** bucle en $(i, j)$. $\quad$ **GPU:** un hilo por $(i, j)$.
> 
> *Nota de la figura:* Blur y Sobel no son algoritmos distintos; son la misma convolución con distinto $k$.

---

> ### 📊 Figura 2: Política de borde
> 
> **Ejemplo: zero-padding ($r = 1$)**
> ```text
> +---+---+---+---+---+---+
> | 0 | 0 | 0 | 0 | 0 | 0 |  Leyenda:
> +---+---+---+---+---+---+  - [I] (Azul) = píxeles reales de la imagen I (4x4 interior).
> | 0 |[I]|[I]|[I]|[I]| 0 |  -  0  (Gris) = fuera de imagen -> valor 0.
> +---+---+---+---+---+---+
> | 0 |[I]|[I]|[I]|[I]| 0 |  Opciones de política:
> +---+---+---+---+---+---+  1. Zero-padding (mostrado en la figura)
> | 0 |[I]|[I]|[I]|[I]| 0 |  2. Clamp (repetir borde)
> +---+---+---+---+---+---+  3. Mirror (reflejar índices)
> | 0 |[I]|[I]|[I]|[I]| 0 |
> +---+---+---+---+---+---+  ⚠️ Elija una política que sea idéntica en CPU y GPU.
> | 0 | 0 | 0 | 0 | 0 | 0 |
> +---+---+---+---+---+---+
> ```
> *Nota de la figura:* Debe declarar en el `README` una opción y usarla igual en la referencia CPU y en ambos kernels GPU; si no, fallan los tests de perímetro.

---

### 5.2. Kernels de Sobel

Use estos operadores $3 \times 3$ (no otros signos ni permutaciones):

$$S_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}, \qquad S_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$$

$G_x$ y $G_y$ se obtienen pasando $S_x$ y $S_y$ por la misma API de convolución genérica (el kernel es un arreglo de coeficientes, no un camino de código aparte). La magnitud $G$ es un *map* sobre los buffers $G_x$ y $G_y$ (kernel CUDA dedicado o fusión documentada).

> ### 📊 Figura 3: Salidas obligatorias — Qué debe producir el programa
> 
> **A. Blur**
> ```text
> [ I ] --------> [ convolve(k_blur) ] --------> [ imagen suavizada ]   (k_blur con ∑k = 1)
> ```
> 
> **B. Sobel**
> ```text
> [ I ] --------> [ convolve(S_x) ] -----------> [ G_x ] --+
>                                                          |
> [ I ] --------> [ convolve(S_y) ] -----------> [ G_y ] --+--> después: [ G = sqrt(G_x^2 + G_y^2) ]
>                                                                        (map con G_x y G_y, no es convolución)
> ```
> * **A y B se miden en benchmarks.** Toda convolución pasa por la misma API `convolve*`.
> * *Nota de la figura:* A: blur = una convolución. B: Sobel = dos convoluciones $(S_x, S_y)$ y después la magnitud $G$ como *map*. No hay un "algoritmo Sobel" aparte de `convolve`.

---

### 5.3. Convención CUDA

* **Variante *naive* (obligatoria):** un hilo por píxel de salida $(i, j)$; lee el vecindario desde memoria global; escribe un solo píxel.
* **Variante *shared* (obligatoria):** bloques 2D; cada bloque carga un *tile* de $I$ (más halo de radio $r$) a `__shared__`, sincroniza con `__syncthreads()`, y computa su subregión de salida.
* **Índice global 2D** a partir de `blockIdx`, `blockDim` y `threadIdx`; si $i \ge H$ o $j \ge W$, el hilo no escribe.
* **Layout en *device*:** arreglo lineal `float` en orden fila-mayor (*row-major*), índice $i \cdot W + j$. Justifique *coalescing* en el reporte.

---

## 6. Requisitos técnicos CUDA

### 6.1. Conceptos obligatorios

Cada ítem debe aparecer en una ruta compilada y ejecutada por un test o un benchmark. El reporte cita archivo, función y motivo.

> ### 📊 Figura 4: Variante *naive*
> 
> **Esquema visual:**
> ```text
>                +-----+-----+-----+
>                |     |     |     |
>                +-----+-----+-----+
>   un hilo ---> |     | (T) |     |   <-- Ese hilo (T) lee 3 x 3 en memoria global
>                +-----+-----+-----+       y escribe un píxel de salida.
>                |     |     |     |
>                +-----+-----+-----+
> ```
> 
> **Cómo calcular $(i, j)$:**
> ```cuda
> i = blockIdx.y * blockDim.y + threadIdx.y;
> j = blockIdx.x * blockDim.x + threadIdx.x;
> if (i >= H || j >= W) return;
> 
> // leer vecinos de I en global
> // acumular con k
> out[i*W + j] = resultado;
> ```
> *Nota de la figura:* Variante *naive*: un hilo = un píxel de salida. El hilo lee el vecindario desde memoria global (sin `__shared__`). A la derecha, ejemplo de índice 2D (documente si invierte `x`/`y`).

1. **Modelo:** `__global__`; al menos una función `__device__` auxiliar; código host que lanza con `<<<grid, block>>>`.
2. **Memoria:** `cudaMalloc`, `cudaMemcpy` (H2D y D2H), `cudaFree`; punteros host y device separados.
3. **Índices 2D:** cálculo de $(i, j)$ globales; grilla con división techo $\lceil H / \text{blockDim.y} \rceil$ y $\lceil W / \text{blockDim.x} \rceil$ (o la convención fila/columna que documente, de forma consistente).
4. **Shared memory:** `__shared__` + `__syncthreads()` en la variante *shared*.
5. **Reducción:** una métrica global en GPU con reducción en *shared memory* (p. ej. $\sum_{i,j} G(i, j)$ o $\sum_{i,j} G(i, j)^2$). Puede ser en dos pasadas.
6. **Atómicas:** `atomicAdd` sobre la misma métrica (u otra métrica global documentada), como camino alternativo a la reducción.
7. **Errores:** macro `CUDA_CHECK` (o equivalente) tras cada API CUDA y `cudaGetLastError` tras cada lanzamiento de kernel.
8. **Temporización:** `std::chrono::steady_clock` en host con `cudaDeviceSynchronize()` acotando el tramo GPU. No se exige `cudaEvent_t`.

---

> ### 📊 Figura 5: Variante *shared*: tres pasos en orden vertical
> 
> **Paso 1 — Cargar: `global` $\rightarrow$ `__shared__`**
> ```text
> +---+---+---+---+---+
> | H | H | H | H | H |   Gris [H] = halo (vecinos).
> +---+---+---+---+---+   Azul [T] = píxeles que el bloque escribirá después (tile 3x3).
> | H |[T]|[T]|[T]| H |
> +---+---+---+---+---+   Varios hilos cargan juntos en memoria compartida.
> | H |[T]|[T]|[T]| H |
> +---+---+---+---+---+
> | H |[T]|[T]|[T]| H |
> +---+---+---+---+---+
> | H | H | H | H | H |
> +---+---+---+---+---+
> ```
> $$\Downarrow$$
> 
> **Paso 2 — Barrera: `__syncthreads()`**
> * Nadie convoluciona aún. Esperan a que el halo esté completo.
> * $\text{Sin barrera} \Rightarrow \text{lectura incompleta} \Rightarrow \text{distinto al naive.}$
> 
> $$\Downarrow$$
> 
> **Paso 3 — Calcular: leer `shared`, escribir `global`**
> ```text
>     +---+---+---+
>     |[O]|[O]|[O]|       Solo el interior (naranja [O], 3x3)
>     +---+---+---+       escribe en memoria global.
>     |[O]|[O]|[O]|
>     +---+---+---+       Debe coincidir con naive.
>     |[O]|[O]|[O]|
>     +---+---+---+
> ```
> *Nota de la figura:* (1) cargar tile+halo; (2) sincronizar; (3) convolucionar desde *shared*.

---

> ### 📊 Figura 6: Qué medir ( Tiempos de ejecución )
> 
> ```text
> [ Host ] --(H2D)--> [ Device in ] ===> [ Kernels conv/Sobel ] ===> [ Device out ] --(D2H)--> [ Host out ]
>                                        |____________________|
>                                        kernel-only (+ sync)
>          |_______________________________________________________________________________|
>                                 end-to-end: H2D + kernels + D2H
> ```
> *Nota de la figura:* `kernel-only` no incluye copias; `end-to-end` sí. Ambos van en la matriz de benchmarks; el *speedup* "de aplicación" usa `end-to-end`.

---

### 6.2. Métodos obligatorios (sobrecarga)

La lógica del *stencil* no se duplica por cada variante: el método simple delega en el parametrizado.

1. **`ImageBuffer`** (o nombre equivalente): carga/generación, `height`/`width`, lectura/escritura.
2. **`Convolution2D`**:
   * `convolveCpu(const float* k, int s)` — referencia serial; `s` es el lado del kernel.
   * `convolveGpu(const float* k, int s)` — llama a la variante por defecto documentada en el `README`.
   * `convolveGpu(const float* k, int s, int variant)` — con `0 = naive`, `1 = shared`.
   * `convolveGpu(const float* k, int s, int variant, dim3 block)`
3. **`Sobel`**:
   * `computeGxGyCpu()` / `computeGxGyGpu(int variant)` — usan $S_x, S_y$ de la sección 5.2.
   * `computeMagnitudeCpu()` / `computeMagnitudeGpu()` — *map* $G = \sqrt{G_x^2 + G_y^2}$ (en GPU: kernel propio o fusión con el paso anterior, documentada).
4. **`Metrics`**:
   * `maxAbsDiffCpuGpu()` — entre buffers CPU y GPU comparables.
   * `sumMagnitude(int method)` — con `0 = reducción en shared memory`, `1 = atomicAdd`.
5. **`Benchmark`**:
   * `benchmarkKernelOnly(...)` — tiempo de kernels (+ sync); sin contar H2D/D2H del pipeline.
   * `benchmarkEndToEnd(...)` — incluye H2D de la entrada y D2H de la salida del pipeline medido.
   * `compareCpuGpu(...)`

---

## 7. Análisis de rendimiento

1. **Tiempos** con `std::chrono::steady_clock`; `cudaDeviceSynchronize()` acota el tramo GPU. Al menos diez repeticiones por punto. Reportar $T = \overline{T} \pm \sigma_T$.
2. **Speedup principal:**
   $$S = \frac{T_{\text{CPU,serial}}}{T_{\text{GPU}}}$$
   (misma imagen, mismos coeficientes, misma política de borde). Reporte $S$ para *naive* y para *shared*, en mediciones *kernel-only* y *end-to-end*.
3. **Mejora *shared* vs. *naive* (obligatoria en el reporte):**
   $$R = \frac{T_{\text{naive}}}{T_{\text{shared}}}$$
   en las mismas condiciones. No se exige una "eficiencia" $S/p$ con $p =$ número de núcleos CUDA (esa definición es ambigua en GPU).
4. **Ley de Amdahl:** $S_{\text{pred}} = 1 / (f + (1 - f)/P)$, donde $f$ es la fracción de tiempo no acelerada en el pipeline *end-to-end* (como mínimo: transferencias H2D/D2H y trabajo host medidos), y $P$ es un paralelismo efectivo que justifica (p. ej. a partir del *speedup* *kernel-only* observado). Compare $S_{\text{pred}}$ con el $S$ *end-to-end* medido.
5. **Propagación:** para $S = T_{\text{CPU}} / T_{\text{GPU}}$,
   $$\sigma_S = S \sqrt{\left(\frac{\sigma_{T_{\text{CPU}}}}{\overline{T}_{\text{CPU}}}\right)^2 + \left(\frac{\sigma_{T_{\text{GPU}}}}{\overline{T}_{\text{GPU}}}\right)^2} \quad \text{(2)}$$
6. **Estudio de bloque 2D:** mida al menos $(\text{blockDim.x}, \text{blockDim.y}) \in \{(16, 16), (16, 32), (32, 16), (32, 32)\}$ (omitir solo si la GPU rechaza la configuración; documentarlo).

### 7.1. Dónde ejecutar: clúster DIINF, partición GPU

Las mediciones finales de desempeño (obligatorias para el reporte) deben ejecutarse en el clúster del DIINF, en un nodo con GPU NVIDIA asignado por Slurm en la partición GPU.
* Confirme el nombre exacto de la partición y las *constraints* con `sinfo` / la documentación vigente del DIINF (el nombre puede cambiar; no asuma un string fijo sin verificar).
* En el script `sbatch` (o equivalente) debe quedar explícita la partición GPU (p. ej. `#SBATCH -p <particion_gpu>`) y, si aplica, la solicitud de GPU (`--gres=gpu:1` o la forma que indique el DIINF).
* Desarrollo y depuración: máquina local o CI. Eso no reemplaza las corridas finales.
* Las tablas, gráficos y `cluster_run.log` del reporte deben corresponder a jobs en esa partición GPU del DIINF.

En el `README`, para cada corrida final: partición Slurm, *job id*, nombre del nodo, modelo de GPU, driver, `nvcc --version`, flags, comando exacto (`sbatch`/`srun`), y archivos `benchmark_results.dat`, `block_study.dat`, `cluster_run.log`.

Si el clúster no está disponible tras al menos dos intentos documentados (log de `sbatch`/`squeue` o ticket con fecha), puede usar otra GPU NVIDIA declarando la desviación en `README` y reporte; el ítem de benchmarks puede verse afectado.

### 7.2. Matriz de benchmarks (obligatoria en partición GPU del DIINF)

| Parámetro | Valores obligatorios |
| :--- | :--- |
| **Tamaño $H \times W$** | $512 \times 512$, $1024 \times 1024$, $2048 \times 2048$, $4096 \times 4096$. Si $4096 \times 4096$ no cabe en memoria, documente el máximo alcanzado y el error de asignación. |
| **Pipelines** | (1) blur con $s = 3$; (2) blur con $s = 5$ o $s = 7$; (3) Sobel completo $G_x, G_y, G$. |
| **Variante** | `naive` (0) y `shared` (1). |
| **Bloque** | Las cuatro combinaciones indicadas arriba. |
| **Repeticiones** | $\ge 10$ por punto. |
| **Mediciones** | `kernel-only` y `end-to-end`. |
| **Dónde** | Clúster DIINF, partición GPU (Slurm). |

---

## 8. Estructura sugerida del proyecto

La organización de archivos es una sugerencia. Puede fusionar o partir módulos si el `README` indica dónde vive cada rol.

```text
image_cuda/
|-- main.cpp
|-- ImageBuffer.h/.cpp
|-- Convolution2D.h/.cpp/.cu
|-- Sobel.h/.cpp/.cu
|-- Metrics.h/.cu
|-- Benchmark.h/.cpp
|-- Visualizer.h/.cpp
|-- tests/
|-- Dockerfile
|-- Makefile
`-- README.md
```

---

## 9. Parámetros y datos

* **Tests de corrección:** imágenes sintéticas $8 \times 8$, $16 \times 16$ y $32 \times 32$ con semilla fija; un impulso; un escalón vertical y uno horizontal.
* Al menos una imagen real en formato PGM (o convertida a PGM/`.dat` fuera del camino medido) para el panel Sobel.
* **Tolerancia CPU serial vs. GPU:** documentar criterio (`atol`, `rtol` o norma). Punto de partida razonable en `float`: error absoluto máximo del orden de $10^{-4}$ a $10^{-3}$ según el rango de $I$; justifique el valor en el `README`.
* **Prohibido en el camino medido:** `OpenCV`, `NPP`, `cuDNN`, `cuFFT`, `Thrust`.

---

## 10. Análisis y visualización

### 10.1. Gráficos (obligatorios)
1. **Speedup $S$** (CPU serial vs. GPU) frente al tamaño de imagen; curvas *naive* y *shared*; separar o etiquetar *kernel-only* y *end-to-end*.
2. **Tiempo *kernel-only* vs. *end-to-end*** en al menos un tamaño grande (peso de las transferencias).
3. **Tiempo frente a la geometría del bloque.**
4. **$S_{\text{pred}}$ (Amdahl) frente a $S$ *end-to-end* medido.**
5. **Panel:** imagen original, blur $s = 3$, $G_x$, $G_y$, $G$ (escala de visualización documentada).
6. **Mapa o histograma del error absoluto CPU–GPU** en una imagen de test.

### 10.2. Archivos de salida
* `benchmark_results.dat`, `block_study.dat`
* `performance_plots.png`
* `sobel_panel.png` (o equivalente)
* `cpu_gpu_error.dat`

---

## 11. Pruebas automáticas

`make test` debe compilar y ejecutar la suite (GoogleTest, Catch2 u otro). Mínimo:
* **Convolución CPU de un impulso** en una imagen $5 \times 5$ con kernel $3 \times 3$ conocido: la salida coincide con el kernel centrado (bajo la política de borde declarada; si usa *zero-padding*, el resultado en el centro es el coeficiente central del kernel cuando el impulso está centrado — documente el caso exacto que testea).
* **Escalón vertical:** $|G_x|$ mayor que $|G_y|$ en el borde (umbral documentado); **escalón horizontal:** lo inverso.
* **Misma convolución:** CPU serial vs. GPU *naive* y vs. GPU *shared*, dentro de la tolerancia.
* **En un píxel de prueba:** $G$ coincide con $\sqrt{G_x^2 + G_y^2}$ dentro de la tolerancia.
* **$\sum G$ (o la métrica elegida):** reducción *shared* vs. `atomicAdd`, misma tolerancia relativa documentada.
* **Un caso en la primera o última fila/columna** que falle si la política de borde está mal implementada.

Si CI no tiene GPU: el pipeline corre al menos los tests CPU y un *smoke* de compilación CUDA; los tests que requieren *device* se documentan como "solo con GPU" y se ejecutan en el clúster. El `README` debe decir qué corre en CI y qué corre en el nodo GPU.

---

## 12. Compilación

Referencia orientativa (ajusten la lista de objetos a sus archivos reales):

```makefile
NVCC = nvcc
NVCCFLAGS = -O3 -std=c++17
TARGET = image_cuda
OBJS = main.o ImageBuffer.o Convolution2D.o Sobel.o Metrics.o Benchmark.o Visualizer.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(OBJS)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.o: %.cpp
	$(NVCC) $(NVCCFLAGS) -x cu -c $< -o $@

test:
	./run_tests

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean benchmark test
```

El `Dockerfile` debe permitir compilar el proyecto y ejecutar `make test` en el alcance que documente (CPU y/o GPU).

---

## 13. Criterios de evaluación

Suma $100\%$. La corrección numérica CPU–GPU y la presencia de ambas variantes CUDA son requisitos de base: sin eso, el resto del puntaje de implementación y benchmarks se evalúa en consecuencia.

| Ítem | Peso | Qué se revisa |
| :--- | :--- | :--- |
| **Implementación CUDA** | $35\%$ | Convolución genérica + blur + Sobel; *naive* y *shared* correctos y equivalentes; conceptos CUDA obligatorios en rutas ejecutadas por test/benchmark; sin librerías prohibidas en el camino medido. |
| **Pruebas, CI y contenedor** | $10\%$ | `make test`; `Dockerfile`; pipeline CI coherente con el `README` (incluye el caso sin GPU en CI, si aplica). |
| **Benchmarks** | $30\%$ | Matriz en partición GPU del DIINF; *kernel-only* y *end-to-end*; bloques; $S$, $R$, Amdahl, $\sigma_S$; evidencia Slurm (partición, *job id*, nodo). |
| **Análisis y visualización** | $15\%$ | Figuras obligatorias; panel Sobel legible; interpretación del peso de transferencias. |
| **Reporte técnico** | $10\%$ | Máx. 10 páginas: problema, diseño, roles, mapeo CUDA, resultados, limitaciones. Grupos de cuatro: doble rol identificado. |

---

## 14. Entregables

1. Código C++/CUDA y `tests/`.
2. `Makefile` con objetivos `all`, `clean` y `test`.
3. `Dockerfile` y configuración de CI en el mismo repositorio.
4. `README.md`: compilación, rango de $I$, tolerancia, política de borde, tabla de roles, qué corre en CI vs. partición GPU del DIINF, script `sbatch` y comando de benchmarks.
5. Tablas `.dat` y figuras PNG listadas arriba.
6. Reporte PDF (máximo 10 páginas).

---

## 15. Fecha de entrega

* **Publicación:** martes 10 de noviembre de 2026.
* **Entrega:** martes 1 de diciembre de 2026, 23:59.
* **Formato:** Google Classroom; código comprimido (`.zip` o `.tar.gz`) y PDF del reporte.

---

## 16. Recursos

* CUDA C++ Programming Guide: [https://docs.nvidia.com/cuda/cuda-c-programming-guide/](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
* Clase GPGPU del curso (modelo de ejecución, *map/reduce*, *shared memory*, *coalescing*).
* R. C. Gonzalez y R. E. Woods, *Digital Image Processing* (convolución espacial y Sobel), o el capítulo equivalente que use.
* Documentación NVIDIA sobre *tiling* / *stencil* en *shared memory* (CUDA Best Practices / blogs técnicos).

---

## 17. Recordatorios

* Compile con advertencias visibles en host (`-Xcompiler -Wall` o equivalente) y revise avisos de `nvcc`.
* Verifique el impulso $5 \times 5$ y los escalones antes de las imágenes $4096 \times 4096$.
* *Naive* y *shared* deben coincidir numéricamente bajo la misma política de borde.
* Cada concepto CUDA obligatorio tiene una ruta en un test o en un benchmark.
* No reporte solo *kernel-only* como "speedup de la aplicación": incluya *end-to-end*.
* En un grupo de cuatro, dos roles en una persona no funden esas responsabilidades en un módulo sin interfaz.