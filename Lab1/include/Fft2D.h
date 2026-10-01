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

    Fft2D() = default;
    explicit Fft2D(ComplexField& field, int layout = LAYOUT_INPLACE);
    ~Fft2D() = default;

    // --- Métodos pasando ComplexField explícitamente ---
    void forward(ComplexField& field);
    void forward(ComplexField& field, int layout);

    void inverse(ComplexField& field);
    void inverse(ComplexField& field, int layout);

    // Comparación obligatoria de OpenMP (reparto de filas)
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
    ComplexField* bound_field_{nullptr};
    int default_layout_{LAYOUT_INPLACE};

    // Validaciones de robustez
    static bool isPowerOfTwo(std::size_t n) noexcept;
    void validateField(const ComplexField& field) const;
    static void validateLayout(int layout);
    static void validateTaskType(int task_type);

    // Helpers privados para eliminar duplicación de código
    void transformRow(ComplexField& field, std::size_t r, int layout, bool inverse);
    void rowsPass(ComplexField& field, int layout, bool inverse);
    void colsPass(ComplexField& field, int layout, bool inverse);
};

#endif // FFT2D_H
