#pragma once

#include "ImageBuffer.h"
#include <string>

/**
 * @file Visualizer.h
 * @brief Rol 5 (Calidad, CI y Visualización): Exportación de resultados para el panel Sobel
 *        y mapas de error absoluto CPU-GPU.
 *
 * Responsabilidades del módulo (Sección 10 de lab2.md):
 * - Exportar los buffers que componen el panel de detección de bordes (`sobel_panel.png`):
 *   imagen original `I`, imagen suavizada (`blur s=3`), gradiente horizontal `G_x`,
 *   gradiente vertical `G_y` y magnitud del gradiente `G`.
 * - Calcular y volcar en disco el mapa de error absoluto píxel a píxel `cpu_gpu_error.dat`
 *   definido como `E(i,j) = |I_cpu(i,j) - I_gpu(i,j)|`.
 *
 * Nota sobre `benchmark_results.dat` y `block_study.dat`:
 * De acuerdo con la planificación del equipo, la generación y escritura de las tablas de tiempos
 * corresponde al **Rol 4** (`Benchmark.h/.cpp`). Los scripts de visualización del Rol 5
 * (`scripts/plot_results.py`) están listos para leer dichas tablas cuando el Rol 4 las produzca.
 */
class Visualizer {
public:
    /**
     * @brief Calcula un `ImageBuffer` con la diferencia absoluta punto a punto `|cpu_buf(i,j) - gpu_buf(i,j)|`.
     * @param cpu_buf Buffer calculado por la referencia serial en CPU (`H x W`).
     * @param gpu_buf Buffer calculado por la GPU (`H x W`).
     * @return `ImageBuffer` de dimensiones `H x W` con el error absoluto en cada píxel `(i, j)`.
     * @throws std::invalid_argument si las dimensiones de `cpu_buf` y `gpu_buf` no coinciden.
     */
    static ImageBuffer computeAbsoluteErrorMap(const ImageBuffer& cpu_buf, const ImageBuffer& gpu_buf);

    /**
     * @brief Escribe el archivo `cpu_gpu_error.dat` con los errores absolutos píxel a píxel
     *        entre la referencia CPU y la salida GPU, e invalida cualquier figura previa obsoleta.
     *        Formato tabular:
     *        `# height <H> width <W> max_abs_err <max> mean_abs_err <mean>`
     *        seguido de las `H` filas con `W` valores de error absoluto `|cpu(i,j) - gpu(i,j)|`.
     * @param cpu_buf Buffer de referencia CPU (`H x W`).
     * @param gpu_buf Buffer de salida GPU (`H x W`).
     * @param path Ruta del archivo de salida (por ejemplo `results/cpu_gpu_error.dat`).
     * @throws std::invalid_argument si las dimensiones no coinciden.
     * @throws std::runtime_error si ocurre un error de escritura en disco.
     */
    static void writeCpuGpuErrorDat(const ImageBuffer& cpu_buf,
                                    const ImageBuffer& gpu_buf,
                                    const std::string& path);

    /**
     * @brief Exporta los cinco componentes obligatorios del panel Sobel (Sección 10.1, ítem 5)
     *        en el directorio `output_dir`, tanto en formato `.dat` (precisión `float` exacta para
     *        `scripts/plot_results.py`) como en formato `.pgm` para inspección directa:
     *        1. `panel_original.dat` / `.pgm` (rango `[0, 1]`)
     *        2. `panel_blur_s3.dat` / `.pgm`  (rango `[0, 1]`)
     *        3. `panel_gx.dat` / `.pgm`       (con signo, reescalado min-max en PGM)
     *        4. `panel_gy.dat` / `.pgm`       (con signo, reescalado min-max en PGM)
     *        5. `panel_mag.dat` / `.pgm`      (no negativo, reescalado min-max en PGM)
     *
     * @param original Imagen de entrada original `I`.
     * @param blur_s3 Imagen suavizada con kernel `s = 3` normalizado (`sum k = 1`).
     * @param gx Gradiente horizontal `G_x = I * S_x`.
     * @param gy Gradiente vertical `G_y = I * S_y`.
     * @param magnitude Magnitud de Sobel `G = sqrt(G_x^2 + G_y^2)`.
     * @param output_dir Directorio de destino (se crea automáticamente si no existe).
     */
    static void exportSobelPanelData(const ImageBuffer& original,
                                     const ImageBuffer& blur_s3,
                                     const ImageBuffer& gx,
                                     const ImageBuffer& gy,
                                     const ImageBuffer& magnitude,
                                     const std::string& output_dir);
};

