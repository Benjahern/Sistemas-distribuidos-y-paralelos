#pragma once

#include <cstddef>
#include <string>
#include <vector>

/**
 * @file ImageBuffer.h
 * @brief Rol 1 (Imagen y I/O): Gestión del buffer 2D en memoria host (H x W) en formato float.
 *
 * Decisiones de diseño y convenciones (documentadas en README.md):
 * - Layout en memoria: Arreglo lineal `float` en orden fila-mayor (row-major).
 *   El píxel en la fila `i` (0 <= i < H) y columna `j` (0 <= j < W) se almacena en `i * W + j`.
 *   El origen (0, 0) corresponde a la esquina superior izquierda.
 * - Rango de operación normalizado: [0.0f, 1.0f].
 *   Al leer un archivo PGM con valor máximo `maxval` (típicamente 255), cada muestra se convierte
 *   a `float(val) / float(maxval)`. Al escribir a PGM, los valores en [0.0f, 1.0f] se escalan
 *   a [0, 255] con saturación (clamping) o reescalado explícito.
 */
class ImageBuffer {
public:
    /**
     * @brief Constructor por defecto: crea un buffer vacío (0 x 0).
     */
    ImageBuffer() = default;

    /**
     * @brief Construye una imagen de dimensiones H (filas) x W (columnas) inicializada con un valor.
     *        Si H == 0 o W == 0, el buffer queda vacío (0 x 0) de forma consistente.
     * @param height Número de filas (H).
     * @param width Número de columnas (W).
     * @param initial_value Valor inicial para todos los píxeles (por defecto 0.0f).
     * @throws std::overflow_error si `height * width` desborda `std::size_t`.
     */
    ImageBuffer(std::size_t height, std::size_t width, float initial_value = 0.0f);

    /**
     * @brief Retorna la altura de la imagen (número de filas H).
     * @return Número de filas H.
     */
    std::size_t height() const noexcept { return height_; }

    /**
     * @brief Retorna el ancho de la imagen (número de columnas W).
     * @return Número de columnas W.
     */
    std::size_t width() const noexcept { return width_; }

    /**
     * @brief Retorna el total de elementos en el buffer (H * W).
     * @return Cantidad total de píxeles.
     */
    std::size_t size() const noexcept { return data_.size(); }

    /**
     * @brief Indica si el buffer está vacío.
     * @return true si H == 0 o W == 0, false en caso contrario.
     */
    bool empty() const noexcept { return data_.empty(); }

    /**
     * @brief Acceso con verificación de límites al píxel en la fila `i` y columna `j`.
     * @param i Índice de fila en [0, H-1].
     * @param j Índice de columna en [0, W-1].
     * @return Referencia modificable al valor `float` del píxel.
     * @throws std::out_of_range si `i >= height_` o `j >= width_`.
     */
    float& at(std::size_t i, std::size_t j);

    /**
     * @brief Acceso constante con verificación de límites al píxel en la fila `i` y columna `j`.
     * @param i Índice de fila en [0, H-1].
     * @param j Índice de columna en [0, W-1].
     * @return Referencia constante al valor `float` del píxel.
     * @throws std::out_of_range si `i >= height_` o `j >= width_`.
     */
    const float& at(std::size_t i, std::size_t j) const;

    /**
     * @brief Acceso directo (sin costo de chequeo) al píxel (i, j) en orden row-major (`i * W + j`).
     * @param i Índice de fila en [0, H-1].
     * @param j Índice de columna en [0, W-1].
     * @return Referencia modificable al píxel.
     */
    float& operator()(std::size_t i, std::size_t j) noexcept {
        return data_[i * width_ + j];
    }

    /**
     * @brief Acceso directo constante al píxel (i, j) en orden row-major (`i * W + j`).
     * @param i Índice de fila en [0, H-1].
     * @param j Índice de columna en [0, W-1].
     * @return Referencia constante al píxel.
     */
    const float& operator()(std::size_t i, std::size_t j) const noexcept {
        return data_[i * width_ + j];
    }

    /**
     * @brief Acceso directo al puntero lineal subyacente en memoria host (`i * W + j`).
     * Ideal para pasar a `convolveCpu`, `convolveGpu` o `cudaMemcpy`.
     * @return Puntero modificable al primer elemento (o `nullptr` si está vacío).
     */
    float* data() noexcept { return data_.data(); }

    /**
     * @brief Acceso directo al puntero lineal constante en memoria host.
     * @return Puntero constante al primer elemento (o `nullptr` si está vacío).
     */
    const float* data() const noexcept { return data_.data(); }

    /**
     * @brief Retorna un puntero al inicio de la fila `i`.
     * @param i Índice de fila en [0, H-1].
     * @return Puntero al elemento (i, 0).
     */
    float* row(std::size_t i) noexcept { return data_.data() + i * width_; }

    /**
     * @brief Retorna un puntero constante al inicio de la fila `i`.
     * @param i Índice de fila en [0, H-1].
     * @return Puntero constante al elemento (i, 0).
     */
    const float* row(std::size_t i) const noexcept { return data_.data() + i * width_; }

    /**
     * @brief Redimensiona la imagen a `new_height x new_width` y asigna `value` a cada píxel.
     *        Garantiza consistencia fuerte (strong exception safety): si `new_height == 0` o
     *        `new_width == 0`, ambas dimensiones quedan en `0 x 0`; si falla la reserva o hay
     *        desbordamiento, el buffer previo y sus dimensiones se mantienen intactos.
     * @param new_height Nueva cantidad de filas H.
     * @param new_width Nueva cantidad de columnas W.
     * @param value Valor a rellenar (por defecto 0.0f).
     * @throws std::overflow_error si `new_height * new_width` desborda `std::size_t`.
     */
    void resize(std::size_t new_height, std::size_t new_width, float value = 0.0f);

    /**
     * @brief Rellena toda la imagen con un valor constante.
     * @param value Valor constante a escribir en todos los píxeles.
     */
    void fill(float value) noexcept;

    // =========================================================================
    // Generadores sintéticos deterministas (para tests de la Sección 9 y 11)
    // =========================================================================

    /**
     * @brief Rellena el buffer con un impulso unitario: `amplitude` en `(impulse_i, impulse_j)` y 0 en el resto.
     * @param impulse_i Fila donde se ubica el impulso.
     * @param impulse_j Columna donde se ubica el impulso.
     * @param amplitude Valor del impulso (por defecto 1.0f).
     * @throws std::out_of_range si `(impulse_i, impulse_j)` está fuera de `[0, H) x [0, W)`.
     */
    void fillImpulse(std::size_t impulse_i, std::size_t impulse_j, float amplitude = 1.0f);

    /**
     * @brief Sobrecarga que ubica el impulso unitario exactamente en el centro `(H/2, W/2)`.
     * @param amplitude Valor del impulso (por defecto 1.0f).
     */
    void fillCenterImpulse(float amplitude = 1.0f);

    /**
     * @brief Rellena el buffer con un escalón vertical:
     *        `low_val` para columnas `j < split_col`, `high_val` para `j >= split_col`.
     *        Diseñado para verificar que `|G_x| > |G_y|` sobre el borde vertical.
     * @param low_val Valor en la mitad izquierda (por defecto 0.0f).
     * @param high_val Valor en la mitad derecha (por defecto 1.0f).
     * @param split_col Columna de transición (si es 0, usa `W / 2`).
     */
    void fillVerticalStep(float low_val = 0.0f, float high_val = 1.0f, std::size_t split_col = 0);

    /**
     * @brief Rellena el buffer con un escalón horizontal:
     *        `low_val` para filas `i < split_row`, `high_val` para `i >= split_row`.
     *        Diseñado para verificar que `|G_y| > |G_x|` sobre el borde horizontal.
     * @param low_val Valor en la mitad superior (por defecto 0.0f).
     * @param high_val Valor en la mitad inferior (por defecto 1.0f).
     * @param split_row Fila de transición (si es 0, usa `H / 2`).
     */
    void fillHorizontalStep(float low_val = 0.0f, float high_val = 1.0f, std::size_t split_row = 0);

    /**
     * @brief Rellena el buffer con ruido uniforme determinista en `[min_val, max_val]` usando `std::mt19937`.
     *        Garantiza reproducibilidad exacta entre ejecuciones para la misma semilla `seed`.
     * @param seed Semilla fija para el generador pseudoaleatorio.
     * @param min_val Límite inferior del rango uniforme (por defecto 0.0f).
     * @param max_val Límite superior del rango uniforme (por defecto 1.0f).
     */
    void fillRandom(unsigned int seed, float min_val = 0.0f, float max_val = 1.0f);

    /**
     * @brief Fábrica estática: crea una imagen con ruido determinista para cualquiera de los tamaños
     *        solicitados en tests (8x8, 16x16, 32x32) o benchmarks (512x512, 1024x1024, 2048x2048, 4096x4096).
     * @param height Número de filas H.
     * @param width Número de columnas W.
     * @param seed Semilla fija (por defecto 42u).
     * @return Instancia `ImageBuffer` inicializada con ruido reproducible en [0.0f, 1.0f].
     */
    static ImageBuffer createRandom(std::size_t height, std::size_t width, unsigned int seed = 42u);

    /**
     * @brief Retorna la lista de tamaños cuadrados estándar de prueba y benchmark exigidos por el laboratorio:
     *        - Tests: 8, 16, 32
     *        - Benchmarks: 512, 1024, 2048, 4096
     * @param include_benchmark_sizes Si es true, incluye {512, 1024, 2048, 4096} además de {8, 16, 32}.
     * @return Vector con los lados N de las imágenes cuadradas N x N.
     */
    static std::vector<std::size_t> standardSizes(bool include_benchmark_sizes = true);

    // =========================================================================
    // Lectura y Escritura en Disco (PGM P2/P5 y formato tabular/binario .dat)
    // =========================================================================

    /**
     * @brief Carga una imagen desde un archivo PGM (soporta tanto P2 ASCII como P5 binario).
     *        Normaliza automáticamente los píxeles leídos `[0, maxval]` al rango `float [0.0f, 1.0f]`.
     *        Rechaza muestras fuera del rango `[0, maxval]`.
     * @param path Ruta del archivo `.pgm` de entrada.
     * @throws std::runtime_error si el archivo no existe, no es P2/P5 válido, contiene muestras
     *         fuera de `[0, maxval]` o está truncado.
     */
    void readPgm(const std::string& path);

    /**
     * @brief Escribe el buffer en un archivo PGM (formato P5 binario por defecto o P2 ASCII).
     *        Si `normalize_minmax` es false, asume que el buffer está en `[0.0f, 1.0f]`, aplica clamp
     *        a `[0.0f, 1.0f]` y multiplica por 255. Si `normalize_minmax` es true (útil para visualizar
     *        gradientes `G_x`, `G_y` o `G` que pueden tener valores negativos o mayores a 1), reescala
     *        linealmente `[min(I), max(I)] -> [0, 255]`.
     * @param path Ruta del archivo `.pgm` de salida.
     * @param binary Si es true escribe P5 (binario); si es false escribe P2 (ASCII).
     * @param normalize_minmax Si es true, reescala el rango dinámico mínimo-máximo a [0, 255].
     * @throws std::runtime_error si no se puede abrir o escribir el archivo.
     */
    void writePgm(const std::string& path, bool binary = true, bool normalize_minmax = false) const;

    /**
     * @brief Escribe el buffer en formato `.dat` de texto con precisión exacta en punto flotante.
     *        Incluye cabecera `# height <H> width <W>` seguida de `H` líneas con `W` valores cada una.
     * @param path Ruta del archivo `.dat` de salida.
     * @throws std::runtime_error si ocurre un error de escritura.
     */
    void writeDat(const std::string& path) const;

    /**
     * @brief Carga un buffer desde un archivo `.dat` previamente guardado con `writeDat`.
     * @param path Ruta del archivo `.dat` de entrada.
     * @throws std::runtime_error si el archivo no existe o su formato es inválido.
     */
    void readDat(const std::string& path);

    // =========================================================================
    // Generadores de Kernels de Filtro Normalizados (Apoyo para Blur s in {3,5,7})
    // =========================================================================

    /**
     * @brief Construye un kernel cuadrado de suavizado uniforme (Box Blur) de lado impar `s x s`
     *        normalizado para que la suma de sus coeficientes sea exactamente 1.0f (`k(u,v) = 1 / s^2`).
     * @param s Lado impar del kernel (ej. 3, 5, 7).
     * @return Vector de tamaño `s * s` con los coeficientes del filtro.
     * @throws std::invalid_argument si `s < 1` o `s` es par.
     * @throws std::overflow_error si `s * s` desborda `std::size_t`.
     */
    static std::vector<float> makeBoxKernel(int s);

    /**
     * @brief Construye un kernel cuadrado Gaussiano 2D de lado impar `s x s` normalizado (`sum k = 1.0f`).
     *        Calcula los pesos en doble precisión (`double`) y los índices en `std::size_t` / `int64_t`
     *        para evitar desbordamientos enteros con tamaños grandes y subdesbordamiento (underflow a 0)
     *        con valores de `sigma` positivos muy pequeños.
     * @param s Lado impar del kernel (ej. 3, 5, 7).
     * @param sigma Desviación estándar del filtro Gaussiano (`0.0f` usa por defecto `s / 3.0f`;
     *              valores negativos o no finitos lanzan `std::invalid_argument`).
     * @return Vector de tamaño `s * s` con los coeficientes normalizados.
     * @throws std::invalid_argument si `s < 1`, `s` es par, `sigma < 0.0f` o `sigma` no es finito.
     * @throws std::overflow_error si `s * s` desborda `std::size_t`.
     */
    static std::vector<float> makeGaussianKernel(int s, float sigma = 0.0f);

private:
    std::size_t height_{0};
    std::size_t width_{0};
    std::vector<float> data_;
};

