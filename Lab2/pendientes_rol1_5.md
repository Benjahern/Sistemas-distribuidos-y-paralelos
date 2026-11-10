# Estado y Pendientes para el 100% — Roles 1 y 5 (Laboratorio 2)

Este documento detalla el estado actual de los **Roles 1 (Imagen y I/O)** y **5 (Calidad, CI y Visualización)** a cargo de **Mateo Valle**, y **qué falta exactamente** para alcanzar el 100% de los entregables exigidos por el enunciado [`lab2.md`] y [`AGENTS.md`].

---

## 🖼️ Rol 1: Imagen y I/O (`ImageBuffer.h/.cpp`) — ✅ 100% COMPLETADO

Todo el trabajo de código, datos de entrada y documentación en el repositorio para el **Rol 1** ya está finalizado al 100%:

* **Buffer host $(H \times W)$ en `float` (*row-major*):** Implementado en [`ImageBuffer.h`] y [`ImageBuffer.cpp`] con indexación `i * W + j`.
* **Lectura y escritura PGM (`P2` ASCII y `P5` binario) y `.dat`:** Normalización automática al rango declarado $[0.0, 1.0]$ y exportación con soporte de reescalado min–max para mapas de gradiente ($G_x, G_y, G$).
* **Generadores sintéticos deterministas:** Impulso unitario centrado/configurable, escalón vertical, escalón horizontal, kernels de Blur normalizados ($\sum k = 1$ para $s \in \{3, 5, 7\}$) y ruido reproducible con semilla fija para todos los tamaños solicitados ($8^2, 16^2, 32^2, 512^2, 1024^2, 2048^2, 4096^2$).
* **Imagen real PGM en el repositorio (Sección 9):** Incluida en [`Lab2/data/sample.pgm`].
* **Identificación de doble rol en `README.md` (Sección 3):** Registrado **Mateo Valle (Rol 1 + 5)** y documentada la separación de interfaces en [`Lab2/README.md`]. *(Solo recordar incluir esta misma tabla también en el reporte PDF final).*

---

## 🧪 Rol 5: Calidad, CI y Visualización (`Visualizer`, `tests/`, `scripts/`, `Dockerfile`, CI)

### ✅ Lo que ya está al 100% en código
* **Módulo [`Visualizer.h`] / [`Visualizer.cpp`]:** Exportación de los 5 buffers del panel Sobel (`.dat` y `.pgm`) y cálculo/escritura de `cpu_gpu_error.dat`.
* **Suite de pruebas [`test_image_buffer.cpp`], [`test_pipeline_cpu_gpu.cpp`], [`test_outputs.py`]:** Los 6 casos obligatorios de la Sección 11 están programados y listos con detección inteligente de GPU y de stubs.
* **Infraestructura de compilación y CI:** [`Makefile`], [`Dockerfile`], [`build-lab2.yml`], [`docker-lab2.yml`] y plantilla [`benchmark-gpu.slurm`].
* **Script de graficado [`plot_results.py`]:** Genera `sobel_panel.png` y `performance_plots.png`.

### ⏳ Qué falta para el 100% del Rol 5 (En espera de Roles 2, 3 y 4 y Clúster DIINF)

> [!IMPORTANT]
> Ninguno de los puntos pendientes del Rol 5 se puede cerrar antes de que los integrantes de los **Roles 2, 3 y 4** completen sus módulos y se ejecute el job final en el **clúster del DIINF**.

1. **Verificar que los 6 tests obligatorios pasen con el código real de los Roles 2 y 3 (Sección 11):**
   * Actualmente [`test_pipeline_cpu_gpu.cpp`] marca `[PENDING - Rol 2/3 stub]` en los tests 1 al 6 porque `Convolution2D`, `Sobel` y `Metrics` tienen stubs vacíos.
   * **Acción pendiente:** Apenas los Roles 2 y 3 suban su implementación, correr `make test` en un entorno con GPU (o en el clúster DIINF) y confirmar que los 6 tests se activan solos y pasan en `[PASS]` dentro de la tolerancia (`1e-4`).

2. **Alinear el formato de `benchmark_results.dat` y `block_study.dat` con el Rol 4 (Sección 10.1 y 10.2):**
   * **Requisito del enunciado:** `performance_plots.png` debe graficar:
     1. Speedup $S$ vs. tamaño de imagen (curvas `naive` y `shared`, separando `kernel-only` y `end-to-end`) para los pipelines medidos (blur $s=3$, blur $s=5$ o $7$, y Sobel completo).
     2. Mejora de memoria compartida $R = T_{\text{naive}} / T_{\text{shared}}$.
     3. Tiempo `kernel-only` vs. `end-to-end` en al menos un tamaño grande.
     4. Tiempo frente a la geometría del bloque 2D $\{(16,16), (16,32), (32,16), (32,32)\}$.
     5. $S_{\text{pred}}$ (Amdahl) frente a $S$ `end-to-end` medido.
     6. Mapa o histograma del error absoluto CPU–GPU (`cpu_gpu_error.dat`).
   * **Acción pendiente:** Cuando el **Rol 4** implemente [`Benchmark.cpp`], coordinar que escriba las columnas esperadas por [`plot_results.py`] (o ajustar el índice de columnas en `plot_results.py` al formato que genere el Rol 4).

3. **Generar los archivos `.dat` y `.png` definitivos desde la partición GPU del Clúster DIINF (Sección 7.1 y 10.2):**
   * **Requisito del enunciado:** Los archivos entregables finales **deben provenir de una corrida en la partición GPU del clúster DIINF** usando Slurm:
     * `benchmark_results.dat` y `block_study.dat`
     * `cpu_gpu_error.dat` (comparando la salida real CPU vs. GPU)
     * `sobel_panel.png` (generado sobre [`Lab2/data/sample.pgm`] con las salidas reales de Blur y Sobel)
     * `performance_plots.png` (generado con los datos reales del clúster)
     * `cluster_run.log`

4. **Documentar la evidencia de la corrida Slurm en `Lab2/README.md` y en el Reporte PDF (Sección 7.1 y 14):**
   * **Requisito del enunciado:** Tras ejecutar `sbatch benchmark-gpu.slurm` en el DIINF (confirmando primero el nombre de la partición GPU con `sinfo`), anotar en [`Lab2/README.md`] los datos registrados en `cluster_run.log`:
     * Partición Slurm, `job_id` y nodo asignado (`hostname`).
     * Modelo de GPU NVIDIA y versión del driver (`nvidia-smi`).
     * Versión de `nvcc --version`, flags y comando exacto ejecutado.
     * Incluir las figuras finales y la interpretación del peso de las transferencias H2D/D2H en el reporte PDF (máx. 10 páginas).
