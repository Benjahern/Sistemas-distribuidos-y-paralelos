# Laboratorio 2: Programación GPGPU con CUDA (Filtrado de Imagen 2D y Detección de Bordes Sobel)

**Departamento de Ingeniería Informática — Universidad de Santiago de Chile (USACH)**

---

## 1. Descripción General y Estructura del Proyecto

Este proyecto implementa en **C++17 y CUDA** un sistema de procesamiento espacial de imágenes 2D en escala de grises que realiza:
1. **Suavizado (Blur):** Convolución espacial 2D con kernels cuadrados normalizados ($\sum k = 1$) de lado impar $s \in \{3, 5, 7\}$.
2. **Detección de bordes (Sobel):** Cálculo de gradientes horizontales y verticales $G_x = I * S_x$ y $G_y = I * S_y$ reutilizando exactamente la misma API de convolución genérica, seguido de un operador *map* punto a punto para la magnitud $G(i,j) = \sqrt{G_x(i,j)^2 + G_y(i,j)^2}$.

Siguiendo la misma estructura modular del `Lab1`, el directorio `Lab2/` se organiza como:

```text
Lab2/
├── include/
│   ├── ImageBuffer.h       # Rol 1 (Implementado): Buffer 2D en host, PGM/.dat y generadores sintéticos
│   ├── Convolution2D.h     # Roles 2 y 3 (Cabecera/Contrato): Interfaz unificada de convolución CPU y GPU
│   ├── Sobel.h             # Roles 2 y 3 (Cabecera/Contrato): Operadores de Sobel Gx, Gy y magnitud G
│   ├── Metrics.h           # Rol 3 (Cabecera/Contrato): Error máximo CPU-GPU y reducciones GPU
│   ├── Benchmark.h         # Rol 4 (Cabecera/Contrato): Mediciones kernel-only y end-to-end
│   └── Visualizer.h        # Rol 5 (Implementado): Exportación del panel Sobel y mapa cpu_gpu_error.dat
├── src/
│   ├── main.cpp            # Punto de entrada CLI (demo, exportación de panel y errores)
│   ├── ImageBuffer.cpp     # Rol 1 (Implementado completo)
│   ├── Convolution2D.cpp   # Stub de integración (a completar por Rol 2)
│   ├── Convolution2D.cu    # Stub de integración (a completar por Rol 3)
│   ├── Sobel.cpp           # Stub de integración (a completar por Rol 2)
│   ├── Sobel.cu            # Stub de integración (a completar por Rol 3)
│   ├── Metrics.cu          # Stub de integración (a completar por Rol 3)
│   ├── Benchmark.cpp       # Stub de integración (a completar por Rol 4)
│   └── Visualizer.cpp      # Rol 5 (Implementado completo)
├── tests/
│   ├── test_image_buffer.cpp     # Pruebas unitarias exhaustivas del Rol 1
│   ├── test_pipeline_cpu_gpu.cpp # Los 6 tests obligatorios de la Sección 11 (Rol 5)
│   └── test_outputs.py           # Test de integración de archivos .dat, .pgm y figuras .png
├── scripts/
│   └── plot_results.py     # Generador de sobel_panel.png y performance_plots.png (Rol 5)
├── data/                   # Directorio para imágenes reales PGM de entrada (ej. sample.pgm)
├── Dockerfile              # Imagen CUDA con nvcc para compilación y tests en contenedor
├── Makefile                # Sistema de compilación (all, test, demo, plots, clean)
└── benchmark-gpu.slurm     # Script sbatch para la partición GPU del clúster DIINF
```

---

## 2. Organización del Equipo y Separación del Doble Rol (Grupo de 4 Integrantes)

En conformidad con la **Sección 3 del enunciado**, para un equipo de 4 integrantes donde una persona cubre **dos roles (Rol 1 y Rol 5)**, las responsabilidades e interfaces de código se mantienen **estrictamente desacopladas en módulos independientes**:

| Rol | Integrante | Módulos / Archivos | Estado Actual y Responsabilidades |
| :--- | :--- | :--- | :--- |
| **1. Imagen y I/O** | Integrante (Rol 1 + 5) | `include/ImageBuffer.h`, `src/ImageBuffer.cpp` | **Implementado completo.** Gestión del buffer host $(H \times W)$ en `float` (*row-major*), lectura/escritura PGM (`P2`/`P5`) y `.dat`, normalización a $[0, 1]$ y generación determinista con semilla fija para todos los tamaños ($8^2, 16^2, 32^2, 512^2, 1024^2, 2048^2, 4096^2$). |
| **2. Convolución CPU** | Integrante Rol 2 | `include/Convolution2D.h`, `src/Convolution2D.cpp`, `include/Sobel.h`, `src/Sobel.cpp` | **Stub / Contrato listo.** Referencia serial pura en CPU para convolución 2D bajo política *Zero-padding*, cálculo de $G_x, G_y, G$ reutilizando `convolveCpu`. |
| **3. Kernels CUDA** | Integrante Rol 3 | `src/Convolution2D.cu`, `src/Sobel.cu`, `include/Metrics.h`, `src/Metrics.cu` | **Stub / Contrato listo.** Kernels `naive` y `shared` (con *tile* + halo y `__syncthreads()`), función auxiliar `__device__`, `CUDA_CHECK`, reducción en `__shared__` vs `atomicAdd`. |
| **4. Host/Device y Benchmarks** | Integrante Rol 4 | `include/Benchmark.h`, `src/Benchmark.cpp` | **Stub / Contrato listo.** Gestión de memoria device, mediciones *kernel-only* vs *end-to-end*, barrido de bloques 2D, Ley de Amdahl y generación de `benchmark_results.dat` y `block_study.dat`. |
| **5. Calidad, CI y Visualización** | Integrante (Rol 1 + 5) | `include/Visualizer.h`, `src/Visualizer.cpp`, `tests/`, `scripts/plot_results.py`, `Dockerfile`, `.github/workflows/` | **Implementado completo.** Suite de pruebas (`make test`), manejo de CI sin GPU física, contenedor Docker con CUDA, exportación de `cpu_gpu_error.dat` y generación de `sobel_panel.png` y `performance_plots.png`. |

> **Nota sobre separación de interfaces entre Rol 1 y Rol 5:**
> * `ImageBuffer` (Rol 1) **no conoce** nada sobre métricas de error CPU–GPU, paneles de Sobel ni scripts de graficado; únicamente administra memoria host $(H \times W)$, generadores sintéticos y serialización PGM/`.dat`.
> * `Visualizer` y `tests/` (Rol 5) consumen la API pública de `ImageBuffer`, `Convolution2D`, `Sobel` y `Metrics` sin acceder a detalles internos de almacenamiento.
>
> **Nota de coordinación con Roles 2, 3 y 4:**
> * Los archivos `Convolution2D.cpp/.cu`, `Sobel.cpp/.cu`, `Metrics.cu` y `Benchmark.cpp` contienen únicamente las declaraciones/firmas vacías (*stubs*) para que el repositorio compile con `nvcc` y pase el pipeline de CI mientras los integrantes de los Roles 2, 3 y 4 completan sus módulos respectivos.

---

## 3. Contratos de API y Firmas / Declaraciones por Rol

A continuación se documentan todas las firmas y declaraciones definidas en `include/` que conectan los 5 roles del proyecto, cumpliendo estrictamente con la **Sección 6.2 de `lab2.md`** y la **Sección 4 de `AGENTS.md`**:

### 3.1. Rol 1: Imagen y I/O (`include/ImageBuffer.h`)
```cpp
class ImageBuffer {
public:
    ImageBuffer() = default;
    ImageBuffer(std::size_t height, std::size_t width, float initial_value = 0.0f);

    // Dimensiones y acceso row-major (i * W + j)
    std::size_t height() const noexcept;
    std::size_t width() const noexcept;
    std::size_t size() const noexcept;
    bool empty() const noexcept;

    float& at(std::size_t i, std::size_t j);
    const float& at(std::size_t i, std::size_t j) const;
    float& operator()(std::size_t i, std::size_t j) noexcept;
    const float& operator()(std::size_t i, std::size_t j) const noexcept;

    float* data() noexcept;
    const float* data() const noexcept;
    float* row(std::size_t i) noexcept;
    const float* row(std::size_t i) const noexcept;

    void resize(std::size_t new_height, std::size_t new_width, float value = 0.0f);
    void fill(float value) noexcept;

    // Generadores sintéticos deterministas (Sección 9 y 11)
    void fillImpulse(std::size_t impulse_i, std::size_t impulse_j, float amplitude = 1.0f);
    void fillCenterImpulse(float amplitude = 1.0f);
    void fillVerticalStep(float low_val = 0.0f, float high_val = 1.0f, std::size_t split_col = 0);
    void fillHorizontalStep(float low_val = 0.0f, float high_val = 1.0f, std::size_t split_row = 0);
    void fillRandom(unsigned int seed, float min_val = 0.0f, float max_val = 1.0f);

    static ImageBuffer createRandom(std::size_t height, std::size_t width, unsigned int seed = 42u);
    static std::vector<std::size_t> standardSizes(bool include_benchmark_sizes = true);

    // Lectura y escritura en disco (PGM P2/P5 y .dat)
    void readPgm(const std::string& path);
    void writePgm(const std::string& path, bool binary = true, bool normalize_minmax = false) const;
    void writeDat(const std::string& path) const;
    void readDat(const std::string& path);

    // Kernels de suavizado (Blur) normalizados (sum k = 1)
    static std::vector<float> makeBoxKernel(int s);
    static std::vector<float> makeGaussianKernel(int s, float sigma = 0.0f);
};
```

### 3.2. Roles 2 y 3: Convolución 2D (`include/Convolution2D.h`)
```cpp
class Convolution2D {
public:
    static constexpr int VARIANT_NAIVE  = 0; // Lectura desde memoria global
    static constexpr int VARIANT_SHARED = 1; // Tile + halo de radio r en __shared__

    explicit Convolution2D(const ImageBuffer& input = ImageBuffer());
    void setInput(const ImageBuffer& input);
    const ImageBuffer& input() const noexcept;
    const ImageBuffer& outputCpu() const noexcept;
    const ImageBuffer& outputGpu() const noexcept;

    // Firmas obligatorias (Sección 6.2):
    void convolveCpu(const float* k, int s);                           // Rol 2 (en Convolution2D.cpp)
    void convolveGpu(const float* k, int s);                           // Delega en variante por defecto
    void convolveGpu(const float* k, int s, int variant);              // 0 = naive, 1 = shared
    void convolveGpu(const float* k, int s, int variant, dim3 block);  // Rol 3 (en Convolution2D.cu)

    static bool isGpuAvailable() noexcept;
};
```

### 3.3. Roles 2 y 3: Operador de Sobel (`include/Sobel.h`)
```cpp
class Sobel {
public:
    static constexpr float SX[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
    static constexpr float SY[9] = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };

    explicit Sobel(const ImageBuffer& input = ImageBuffer());
    void setInput(const ImageBuffer& input);
    const ImageBuffer& input() const noexcept;

    // Firmas obligatorias (Sección 6.2):
    void computeGxGyCpu();                                             // Rol 2 (reutiliza convolveCpu)
    void computeGxGyGpu(int variant = Convolution2D::VARIANT_SHARED);  // Rol 3 (reutiliza convolveGpu)
    void computeMagnitudeCpu();                                        // Rol 2 (en Sobel.cpp)
    void computeMagnitudeGpu();                                        // Rol 3 (en Sobel.cu)

    const ImageBuffer& gxCpu() const noexcept;
    const ImageBuffer& gyCpu() const noexcept;
    const ImageBuffer& magnitudeCpu() const noexcept;
    const ImageBuffer& gxGpu() const noexcept;
    const ImageBuffer& gyGpu() const noexcept;
    const ImageBuffer& magnitudeGpu() const noexcept;
};
```

### 3.4. Rol 3: Métricas y Reducciones en GPU (`include/Metrics.h`)
```cpp
class Metrics {
public:
    static constexpr int METHOD_SHARED_REDUCTION = 0; // Reducción en __shared__
    static constexpr int METHOD_ATOMIC_ADD       = 1; // Acumulación con atomicAdd

    explicit Metrics(const ImageBuffer& magnitude = ImageBuffer());
    void setMagnitudeBuffer(const ImageBuffer& magnitude);
    void setComparisonBuffers(const ImageBuffer& cpu_buf, const ImageBuffer& gpu_buf);

    // Firmas obligatorias (Sección 6.2):
    static float maxAbsDiffCpuGpu(const ImageBuffer& cpu_buf, const ImageBuffer& gpu_buf);
    float maxAbsDiffCpuGpu() const;
    float sumMagnitude(int method) const;
    static float sumMagnitude(const ImageBuffer& magnitude, int method);
};
```

### 3.5. Rol 4: Benchmarks Host/Device (`include/Benchmark.h`)
```cpp
class Benchmark {
public:
    Benchmark() = default;

    // Firmas obligatorias (Sección 6.2):
    void benchmarkKernelOnly(const ImageBuffer& img, const float* k, int s, int variant,
                             dim3 block = dim3(16, 16), int repetitions = 10);
    void benchmarkEndToEnd(const ImageBuffer& img, const float* k, int s, int variant,
                           dim3 block = dim3(16, 16), int repetitions = 10);
    void compareCpuGpu(const ImageBuffer& img, const float* k, int s, int repetitions = 10);
};
```

### 3.6. Rol 5: Visualización y Exportación (`include/Visualizer.h`)
```cpp
class Visualizer {
public:
    static ImageBuffer computeAbsoluteErrorMap(const ImageBuffer& cpu_buf, const ImageBuffer& gpu_buf);
    static void writeCpuGpuErrorDat(const ImageBuffer& cpu_buf,
                                    const ImageBuffer& gpu_buf,
                                    const std::string& path);
    static void exportSobelPanelData(const ImageBuffer& original,
                                     const ImageBuffer& blur_s3,
                                     const ImageBuffer& gx,
                                     const ImageBuffer& gy,
                                     const ImageBuffer& magnitude,
                                     const std::string& output_dir);
};
```

---

## 4. Decisiones Técnicas y Convenciones Matemáticas

### 4.1. Rango de Operación de la Imagen ($I \in [0.0, 1.0]$)
* **Decisión:** Todas las imágenes en memoria se almacenan como `float` de precisión simple normalizados en el rango **$[0.0, 1.0]$**.
* **Justificación:**
  1. Evita desbordamientos o pérdida de precisión al acumular productos en filtros de suavizado y al elevar al cuadrado gradientes en $G = \sqrt{G_x^2 + G_y^2}$ o en reducciones globales $\sum G$ sobre imágenes grandes de hasta $4096 \times 4096$ ($1.67 \times 10^7$ píxeles).
  2. Permite una conversión directa y limpia al cargar archivos PGM de 8 bits (`valor_float = byte / 255.0f`) o 16 bits (`valor_float = word / 65535.0f`).
  3. Al exportar buffers de gradientes ($G_x, G_y, G$) que pueden tomar valores fuera de $[0, 1]$ (en particular $G_x, G_y \in [-4, 4]$ y $G \in [0, 4\sqrt{2}]$), `Visualizer` guarda los valores exactos sin recortar en archivos `.dat` y aplica reescalado lineal min–max al exportar vistas previas `.pgm`.

### 4.2. Política de Borde Única: **Zero-Padding**
* **Decisión:** Se adopta **Zero-padding** en todo el proyecto (CPU serial, GPU `naive` y GPU `shared`). Cualquier lectura de un vecino $(i + u, j + v)$ fuera del dominio válido $[0, H-1] \times [0, W-1]$ retorna exactamente `0.0f`.
* **Justificación:**
  1. Es matemáticamente transparente y permite verificar analíticamente el **Test 1 (Impulso $5 \times 5$)** y el **Test 6 (Perímetro)**: en una imagen constante de unos ($1.0$) filtrada con un Box kernel $3 \times 3$, las esquinas valen exactamente $4/9$, los bordes laterales $6/9$ y el interior $9/9 = 1.0$, lo que permite detectar de inmediato cualquier error de indexación en el halo de `__shared__`.
  2. Debe aplicarse de forma idéntica en la referencia CPU serial (Rol 2) y en los kernels `naive` y `shared` (Rol 3).

### 4.3. Convención de Signos de la Convolución y Sobel
Para un kernel cuadrado de lado impar $s = 2r + 1$ e índices locales $u, v \in \{-r, \dots, r\}$, tanto CPU como GPU deben evaluar:

$$\text{Out}(i, j) = \sum_{u=-r}^{r} \sum_{v=-r}^{r} k(u + r, v + r) \cdot I_{\text{pad}}(i + u, j + v)$$

Con los operadores de Sobel $3 \times 3$ fijos:

$$S_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}, \qquad S_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$$

### 4.4. Tolerancia Numérica CPU vs. GPU
* **Tolerancia absoluta en convolución y magnitud (`atol`):** $\text{atol} = 10^{-4}$ para imágenes en $[0, 1]$. Dado que el épsilon de máquina en `float` IEEE-754 (FP32) es $\epsilon \approx 1.19 \times 10^{-7}$ y una ventana de hasta $7 \times 7 = 49$ operaciones FMA acumula diferencias de redondeo del orden de $10^{-6}$, un umbral de $10^{-4}$ es estricto para detectar errores algorítmicos o de borde sin producir falsos positivos por asociatividad en punto flotante.
* **Tolerancia relativa en reducciones (`rtol`):** $\text{rtol} = 10^{-4}$ al comparar `sumMagnitude(0)` (reducción por árbol en `__shared__`) frente a `sumMagnitude(1)` (`atomicAdd`), ya que el orden no determinista de las sumas atómicas cambia la secuencia de redondeo FP32.

---

## 5. Cómo Agregar una Imagen Real en Formato PGM

Para el panel Sobel (`sobel_panel.png`), puedes colocar cualquier imagen real en escala de grises en `Lab2/data/sample.pgm` (el binario `build/image_cuda demo` la detecta automáticamente si existe en `data/sample.pgm`, o puedes pasarla con `--input ruta/a/tu_imagen.pgm`).

### Requisitos del archivo PGM:
* **Formato soportado:** tanto **P5** (binario, recomendado por ser más compacto y rápido) como **P2** (texto ASCII).
* **Profundidad máxima (`maxval`):** `255` (8 bits por píxel, estándar) o hasta `65535` (16 bits).
* **Dimensiones recomendadas:** entre $256 \times 256$ y $1024 \times 1024$ píxeles en escala de grises.

### Cómo convertir cualquier imagen (`.png` o `.jpg`) a `.pgm`:
1. **Con ImageMagick (en terminal Linux):**
   ```bash
   mkdir -p Lab2/data
   magick tu_foto.jpg -colorspace Gray -resize 512x512\! Lab2/data/sample.pgm
   ```
2. **Con Python (PIL / Pillow):**
   ```bash
   mkdir -p Lab2/data
   python3 -c "
   from PIL import Image
   img = Image.open('tu_foto.png').convert('L').resize((512, 512))
   img.save('Lab2/data/sample.pgm')
   "
   ```
3. **Con GIMP:** Abrir la imagen $\rightarrow$ *Imagen > Modo > Escala de grises* $\rightarrow$ *Archivo > Exportar como...* $\rightarrow$ escribir `sample.pgm` y elegir formato **Crudo (Raw / P5)** o **ASCII (P2)**.

---

## 6. Compilación, Pruebas y Ejecución

### Comandos principales (`Makefile`):
```bash
# 1. Compilar el proyecto completo (usa nvcc con -O3 -std=c++17 -Xcompiler -Wall)
make all

# 2. Compilar y ejecutar toda la suite de pruebas automáticas (Rol 1 + Rol 5)
make test

# 3. Ejecutar demostración (genera los archivos .dat y .pgm del panel Sobel en results/)
make demo

# 4. Generar las figuras PNG (sobel_panel.png y performance_plots.png) en results/
make plots

# 5. Limpiar binarios y objetos de compilación
make clean
```

---

## 7. Qué Corre en CI (GitHub Actions) vs. Nodo GPU del Clúster DIINF

Siguiendo la **Sección 11 del enunciado**:
1. **En GitHub Actions CI (`.github/workflows/build-lab2.yml` y `docker-lab2.yml`):**
   * Los runners gratuitos de GitHub (`ubuntu-latest`) no poseen una tarjeta gráfica NVIDIA física conectada al hipervisor.
   * El workflow `docker-lab2.yml` construye la imagen Docker basada en `nvidia/cuda:12.4.1-devel-ubuntu22.04`, ejecutando un **smoke build real con `nvcc`** de todos los archivos `.cpp` y `.cu`.
   * Al correr `make test`, el binario ejecuta todos los tests unitarios de `ImageBuffer` (Rol 1) y `Visualizer` (Rol 5), y verifica dinámicamente el estado de los Roles 2 y 3 y la presencia de GPU mediante `Convolution2D::isGpuAvailable()`.
   * Si `cudaGetDeviceCount` detecta `0` dispositivos físicos en el runner de CI, las pruebas que requieren lanzar kernels en hardware se reportan explícitamente como `[SKIPPED - No GPU device available in CI]` sin hacer fallar el pipeline.
2. **En la Partición GPU del Clúster DIINF (`sbatch benchmark-gpu.slurm`):**
   * `Convolution2D::isGpuAvailable()` retorna `true` y, una vez integrados los Roles 2, 3 y 4, `make test` ejecuta **todas** las pruebas en hardware GPU real (incluyendo equivalencia numérica CPU vs. GPU Naive vs. GPU Shared en $8^2, 16^2, 32^2$, reducción `__shared__` vs. `atomicAdd` y test de perímetro en GPU), además de los benchmarks finales.
