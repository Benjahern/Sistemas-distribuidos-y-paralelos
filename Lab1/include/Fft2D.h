#ifndef FFT2D_H
#define FFT2D_H

#include <cstddef>
#include <complex>
#include <vector>

// Declaración adelantada de ComplexField (definido por Rol 1)
class ComplexField;

/**
 * @brief Rol 3: FFT e IFFT bidimensional separable para grillas M x N.
 *
 * Cumple con los requisitos de:
 * - Separabilidad 2D (pasada de filas seguida de pasada de columnas).
 * - Layout in-place (0, Cooley-Tukey) y out-of-place (1, Stockham).
 * - Transformada directa sin escala; inversa con signo conjugado en twiddles
 *   y escala 1/(M*N) aplicada una única vez.
 * - Soporte para OpenMP mediante parallel for y tasks independientes por filas.
 */
class Fft2D {
public:
    // Layouts de mariposa 1D según especificación
    static constexpr int LAYOUT_INPLACE  = 0; // Cooley-Tukey (in-place)
    static constexpr int LAYOUT_STOCKHAM = 1; // Stockham (out-of-place)

    // Mecanismos de distribución OpenMP para filas
    static constexpr int TASK_TYPE_TASK         = 0; // #pragma omp task
    static constexpr int TASK_TYPE_PARALLEL_FOR = 1; // #pragma omp parallel for

    // Dos formas de uso:
    //   Fft2D fft; fft.forward(field, layout);  // campo explícito en cada llamada
    //   Fft2D fft(field, layout); fft.forward(); // campo enlazado al construir
    // Todas las transformadas modifican el campo recibido; no devuelven una copia.
    Fft2D() = default;
    explicit Fft2D(ComplexField& field, int layout = LAYOUT_INPLACE);
    ~Fft2D() = default;

    // --- Métodos pasando ComplexField explícitamente ---
    // forward: signo negativo, sin escala, filas y luego columnas.
    // La variante sin layout usa default_layout_ (in-place si no se cambió).
    void forward(ComplexField& field);
    void forward(ComplexField& field, int layout);

    // inverse: signo positivo, columnas y luego filas, y escala final 1/(M*N).
    // Requiere que Butterfly1D no normalice las pasadas por separado.
    void inverse(ComplexField& field);
    void inverse(ComplexField& field, int layout);

    // Solo la pasada de filas: task_type elige tareas (0) o parallel for (1).
    // use_single decide si un productor o todos los hilos generan las tareas;
    // no tiene efecto en la variante parallel for. Omitirlo equivale a true.
    void forwardRows(ComplexField& field, int task_type);
    void forwardRows(ComplexField& field, int task_type, bool use_single);
    void forwardRows(ComplexField& field, int task_type, bool use_single, int layout);

    // Pasadas individuales para control fino y pruebas.
    // inverseRows/inverseCols cambian el signo sin normalizar;
    // solo inverse() aplica el factor total 1/(M*N).
    void forwardRows(ComplexField& field);
    void forwardCols(ComplexField& field, int layout = LAYOUT_INPLACE);
    void inverseRows(ComplexField& field, int layout = LAYOUT_INPLACE);
    void inverseCols(ComplexField& field, int layout = LAYOUT_INPLACE);

    // --- Sobrecargas cuando se enlazó ComplexField en el constructor ---
    void forward();
    void forward(int layout);
    void inverse();
    void inverse(int layout);
    void forwardRows(int task_type);
    void forwardRows(int task_type, bool use_single);

private:
    // Puntero no propietario: Fft2D no destruye ni prolonga la vida del campo.
    // nullptr indica que se deben usar las sobrecargas con campo explícito.
    ComplexField* bound_field_{nullptr};
    int default_layout_{LAYOUT_INPLACE};

    // Validaciones de robustez
    static bool isPowerOfTwo(std::size_t n) noexcept;
    void validateField(const ComplexField& field) const;
    static void validateLayout(int layout);
    static void validateTaskType(int task_type);

    // Una única lógica de composición sirve para ambos layouts y signos.
    // La mariposa, bit-reversal y buffers Stockham pertenecen al núcleo 1D.
    void transformRow(ComplexField& field, std::size_t r, int layout, bool inverse);
    void rowsPass(ComplexField& field, int layout, bool inverse);
    void colsPass(ComplexField& field, int layout, bool inverse);
};

#endif // FFT2D_H
