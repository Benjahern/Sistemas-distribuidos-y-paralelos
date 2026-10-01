#include "Fft2D.h"
#include "ComplexField.h"
#include "Butterfly1D.h"

#include <omp.h>
#include <stdexcept>
#include <vector>
#include <string>

// ============================================================================
// Métodos de validación y robustez
// ============================================================================

bool Fft2D::isPowerOfTwo(std::size_t n) noexcept {
    return n != 0 && (n & (n - 1)) == 0;
}

void Fft2D::validateField(const ComplexField& field) const {
    const std::size_t M = field.rows();
    const std::size_t N = field.cols();
    if (M == 0 || N == 0) {
        throw std::invalid_argument("Fft2D: Las dimensiones de la grilla no pueden ser cero.");
    }
    if (!isPowerOfTwo(M) || !isPowerOfTwo(N)) {
        throw std::invalid_argument("Fft2D: M (" + std::to_string(M) + ") y N (" +
                                   std::to_string(N) + ") deben ser potencias de dos.");
    }
}

void Fft2D::validateLayout(int layout) {
    if (layout != LAYOUT_INPLACE && layout != LAYOUT_STOCKHAM) {
        throw std::invalid_argument("Fft2D: Layout inválido (" + std::to_string(layout) +
                                   "). Debe ser 0 (in-place) o 1 (Stockham).");
    }
}

void Fft2D::validateTaskType(int task_type) {
    if (task_type != TASK_TYPE_TASK && task_type != TASK_TYPE_PARALLEL_FOR) {
        throw std::invalid_argument("Fft2D: task_type inválido (" + std::to_string(task_type) +
                                   "). Debe ser 0 (task) o 1 (parallel for).");
    }
}

// ============================================================================
// Constructores
// ============================================================================

Fft2D::Fft2D(ComplexField& field, int layout)
    : bound_field_(&field), default_layout_(layout) {
    validateLayout(layout);
}

// ============================================================================
// Helpers de transformación atómica (eliminan duplicación de código)
// ============================================================================

void Fft2D::transformRow(ComplexField& field, std::size_t r, int layout, bool inverse) {
    std::complex<double>* row_ptr = field.row(r);
    // El núcleo 1D debe devolver la transformada sin normalizar, en ambos signos.
    Butterfly1D butterfly(row_ptr, field.cols(), inverse);
    butterfly.transform(layout, 0, 0);
}

void Fft2D::rowsPass(ComplexField& field, int layout, bool inverse) {
    const std::size_t M = field.rows();

    #pragma omp parallel for default(none) shared(field, M, layout, inverse) schedule(runtime)
    for (std::size_t r = 0; r < M; ++r) {
        transformRow(field, r, layout, inverse);
    }
}

void Fft2D::colsPass(ComplexField& field, int layout, bool inverse) {
    const std::size_t M = field.rows();
    const std::size_t N = field.cols();

    // Optimización de rendimiento: un único buffer local reservado por hilo (no por columna),
    // eliminando N asignaciones dinámicas y optimizando la localidad de caché.
    #pragma omp parallel default(none) shared(field, M, N, layout, inverse)
    {
        std::vector<std::complex<double>> col_buf(M);

        #pragma omp for schedule(runtime)
        for (std::size_t c = 0; c < N; ++c) {
            for (std::size_t r = 0; r < M; ++r) {
                col_buf[r] = field.at(r, c);
            }

            Butterfly1D butterfly(col_buf.data(), M, inverse);
            butterfly.transform(layout, 0, 0);

            for (std::size_t r = 0; r < M; ++r) {
                field.at(r, c) = col_buf[r];
            }
        }
    }
}

// ============================================================================
// Algoritmo 2D: Directa e Inversa
// ============================================================================

void Fft2D::forward(ComplexField& field) {
    forward(field, default_layout_);
}

void Fft2D::forward(ComplexField& field, int layout) {
    validateField(field);
    validateLayout(layout);

    // 1. Pasada por filas (reparte las M transformadas 1D de tamaño N)
    rowsPass(field, layout, /*inverse=*/false);

    // 2. Sincronización implícita al terminar la región paralela de rowsPass.
    // 3. Pasada por columnas (reparte las N transformadas 1D de tamaño M)
    colsPass(field, layout, /*inverse=*/false);
}

void Fft2D::inverse(ComplexField& field) {
    inverse(field, default_layout_);
}

void Fft2D::inverse(ComplexField& field, int layout) {
    validateField(field);
    validateLayout(layout);

    // Deshacer las pasadas en orden separable inverso (columnas y luego filas).
    // Esto es matemáticamente válido porque la DFT 2D es un producto tensorial lineal
    // y separable: las operaciones sobre dimensiones ortogonales conmutan.
    colsPass(field, layout, /*inverse=*/true);
    rowsPass(field, layout, /*inverse=*/true);

    // Factor de escala 1/(M*N) aplicado UNA ÚNICA VEZ al final del proceso completo
    const size_t M = field.rows();
    const size_t N = field.cols();
    const double scale = 1.0 / (static_cast<double>(M) * static_cast<double>(N));

    #pragma omp parallel for default(none) shared(field, M, N, scale) collapse(2) schedule(runtime)
    for (size_t r = 0; r < M; ++r) {
        for (size_t c = 0; c < N; ++c) {
            field.at(r, c) *= scale;
        }
    }
}

// ============================================================================
// Reparto de filas y cláusulas OpenMP obligatorias
// ============================================================================

void Fft2D::forwardRows(ComplexField& field) {
    forwardRows(field, TASK_TYPE_PARALLEL_FOR, true, default_layout_);
}

void Fft2D::forwardRows(ComplexField& field, int task_type) {
    forwardRows(field, task_type, true, default_layout_);
}

void Fft2D::forwardRows(ComplexField& field, int task_type, bool use_single) {
    forwardRows(field, task_type, use_single, default_layout_);
}

void Fft2D::forwardRows(ComplexField& field, int task_type, bool use_single, int layout) {
    validateField(field);
    validateLayout(layout);
    validateTaskType(task_type);

    const size_t M = field.rows();

    if (task_type == TASK_TYPE_PARALLEL_FOR) {
        // Enfoque 1: parallel for repartiendo iteraciones de filas independientes
        #pragma omp parallel for default(none) shared(field, M, layout) schedule(runtime)
        for (size_t r = 0; r < M; ++r) {
            transformRow(field, r, layout, /*inverse=*/false);
        }
    } else if (task_type == TASK_TYPE_TASK) {
        // Enfoque 2: OpenMP Tasks
        #pragma omp parallel default(none) shared(field, M, use_single, layout)
        {
            if (use_single) {
                // Rama con single: un único hilo productor encola las tareas en el pool
                #pragma omp single
                {
                    for (size_t r = 0; r < M; ++r) {
                        #pragma omp task firstprivate(r) shared(field, layout)
                        {
                            transformRow(field, r, layout, /*inverse=*/false);
                        }
                    }
                }
            } else {
                // Rama sin single: los hilos del equipo generan tareas distribuidas.
                // Nota técnica: en esta rama, el taskwait local espera las tareas hijas
                // de cada hilo, y la barrera implícita al fin de la región paralela
                // asegura la finalización de todo el equipo antes de continuar.
                const int tid = omp_get_thread_num();
                const int nthreads = omp_get_num_threads();
                for (size_t r = static_cast<size_t>(tid); r < M; r += static_cast<size_t>(nthreads)) {
                    #pragma omp task firstprivate(r) shared(field, layout)
                    {
                        transformRow(field, r, layout, /*inverse=*/false);
                    }
                }
            }
            #pragma omp taskwait
        }
    }
}

void Fft2D::forwardCols(ComplexField& field, int layout) {
    validateField(field);
    validateLayout(layout);
    colsPass(field, layout, /*inverse=*/false);
}

void Fft2D::inverseRows(ComplexField& field, int layout) {
    validateField(field);
    validateLayout(layout);
    rowsPass(field, layout, /*inverse=*/true);
}

void Fft2D::inverseCols(ComplexField& field, int layout) {
    validateField(field);
    validateLayout(layout);
    colsPass(field, layout, /*inverse=*/true);
}

// ============================================================================
// Métodos enlazados (bound_field_)
// ============================================================================

void Fft2D::forward() {
    if (!bound_field_) throw std::runtime_error("Fft2D: No bound ComplexField instance.");
    forward(*bound_field_, default_layout_);
}

void Fft2D::forward(int layout) {
    if (!bound_field_) throw std::runtime_error("Fft2D: No bound ComplexField instance.");
    forward(*bound_field_, layout);
}

void Fft2D::inverse() {
    if (!bound_field_) throw std::runtime_error("Fft2D: No bound ComplexField instance.");
    inverse(*bound_field_, default_layout_);
}

void Fft2D::inverse(int layout) {
    if (!bound_field_) throw std::runtime_error("Fft2D: No bound ComplexField instance.");
    inverse(*bound_field_, layout);
}

void Fft2D::forwardRows(int task_type) {
    if (!bound_field_) throw std::runtime_error("Fft2D: No bound ComplexField instance.");
    forwardRows(*bound_field_, task_type);
}

void Fft2D::forwardRows(int task_type, bool use_single) {
    if (!bound_field_) throw std::runtime_error("Fft2D: No bound ComplexField instance.");
    forwardRows(*bound_field_, task_type, use_single);
}
