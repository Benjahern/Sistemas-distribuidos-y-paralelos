#include "Fft2D.h"
#include "ComplexField.h"
#include "Butterfly1D.h"

#include <omp.h>
#include <stdexcept>
#include <vector>
#include <string>

/**
 * @file Fft2D.cpp
 * @brief Rol 3: Composición separable de FFT e IFFT bidimensionales.
 *
 * ComplexField representa una grilla M x N: M filas y N columnas, almacenadas
 * por filas (row-major). Cada fila contiene N valores contiguos; una columna
 * contiene M valores separados por N posiciones en memoria.
 *
 * La directa ejecuta M FFT 1D de longitud N y luego N FFT 1D de longitud M.
 * La inversa recorre columnas y luego filas, y aplica al final 1/(M*N).
 * Butterfly1D realiza las transformadas de ambos signos sin normalizar:
 * Fft2D se encarga de la composición, el reparto de filas/columnas y la escala.
 *
 * Todas las rutas escriben el resultado sobre el campo recibido. El layout
 * selecciona el algoritmo interno 1D: 0 = Cooley-Tukey, 1 = Stockham.
 * Las sobrecargas delegan en métodos comunes para conservar una sola lógica.
 */

// ============================================================================
// VALIDACIÓN DE DIMENSIONES Y OPCIONES
// ============================================================================

/**
 * @brief Comprueba que los dos ejes admitan una FFT 1D radix-2.
 * @param field Entrada: Grilla cuyas dimensiones M y N se van a comprobar.
 * @return Salida: void (no modifica el campo).
 * @throws std::invalid_argument Si algún eje es cero o no es potencia de dos.
 * @note M y N pueden ser distintos; una dimensión de tamaño 1 es válida.
 */
void Fft2D::validateField(const ComplexField& field) const {
    // M cuenta filas y N columnas. Se validan ambos ejes porque cada uno
    // será la longitud de una FFT 1D radix-2; no hace falta que M == N.
    const std::size_t M = field.rows();
    const std::size_t N = field.cols();
    if (M == 0 || N == 0) {
        throw std::invalid_argument("Fft2D: Las dimensiones de la grilla no pueden ser cero.");
    }
    if (!ComplexField::isPowerOf2(M) || !ComplexField::isPowerOf2(N)) {
        throw std::invalid_argument("Fft2D: M (" + std::to_string(M) + ") y N (" +
                                   std::to_string(N) + ") deben ser potencias de dos.");
    }
}

/**
 * @brief Comprueba la selección del algoritmo de transformada 1D.
 * @param layout Entrada: 0 = Cooley-Tukey in-place, 1 = Stockham out-of-place.
 * @return Salida: void.
 * @throws std::invalid_argument Si layout no es 0 ni 1.
 */
void Fft2D::validateLayout(int layout) {
    if (layout != LAYOUT_INPLACE && layout != LAYOUT_STOCKHAM) {
        throw std::invalid_argument("Fft2D: Layout inválido (" + std::to_string(layout) +
                                   "). Debe ser 0 (in-place) o 1 (Stockham).");
    }
}

/**
 * @brief Comprueba el mecanismo de distribución de filas entre hilos.
 * @param task_type Entrada: 0 = task, 1 = parallel for.
 * @return Salida: void.
 * @throws std::invalid_argument Si task_type no es 0 ni 1.
 */
void Fft2D::validateTaskType(int task_type) {
    if (task_type != TASK_TYPE_TASK && task_type != TASK_TYPE_PARALLEL_FOR) {
        throw std::invalid_argument("Fft2D: task_type inválido (" + std::to_string(task_type) +
                                   "). Debe ser 0 (task) o 1 (parallel for).");
    }
}

// ============================================================================
// CONSTRUCTOR Y ENLACE DEL CAMPO
// ============================================================================

/**
 * @brief Enlaza una grilla para las llamadas que omiten el parámetro field.
 * @param field Entrada/Salida: Campo que usarán forward(), inverse() y forwardRows().
 * @param layout Entrada: Algoritmo por defecto (0 = in-place, 1 = Stockham).
 * @throws std::invalid_argument Si layout no es válido.
 * @note Guarda un puntero no propietario; el llamador debe mantener vivo field.
 *       Las dimensiones se validan al ejecutar una transformada, no al construir.
 */
Fft2D::Fft2D(ComplexField& field, int layout)
    : bound_field_(&field), default_layout_(layout) {
    // Se guarda una referencia mediante un puntero, sin copiar la grilla.
    // El llamador debe mantener field vivo mientras use esta instancia.
    validateLayout(layout);
}

// ============================================================================
// HELPERS DE TRANSFORMACIÓN: REUTILIZACIÓN DEL NÚCLEO 1D
// ============================================================================

/**
 * @brief Transforma una fila directamente sobre su memoria contigua.
 * @param field Entrada/Salida: Grilla que contiene la fila a transformar.
 * @param r Entrada: Índice de fila, en el intervalo [0, field.rows()).
 * @param layout Entrada: 0 = Cooley-Tukey, 1 = Stockham.
 * @param inverse Entrada: false = directa (signo -), true = inversa (signo +).
 * @return Salida: void (reemplaza los N valores de la fila r, sin normalizar).
 * @pre El llamador ya validó las dimensiones y el layout.
 * @note Cada invocación crea su propia instancia de Butterfly1D. Las llamadas
 *       sobre filas distintas tienen destinos separados y pueden ser concurrentes.
 */
void Fft2D::transformRow(ComplexField& field, std::size_t r, int layout, bool inverse) {
    // La memoria es row-major: una fila ocupa N elementos consecutivos.
    // row_ptr apunta a los datos originales, así que la FFT modifica esa fila.
    std::complex<double>* row_ptr = field.row(r);
    // El núcleo 1D debe devolver la transformada sin normalizar, en ambos signos.
    Butterfly1D butterfly(row_ptr, field.cols(), inverse);
    // Los dos ceros eligen schedule static y chunk por defecto dentro de la FFT
    // 1D. El reparto entre filas se configura por separado en rowsPass.
    butterfly.transform(layout, 0, 0);
}

/**
 * @brief Ejecuta M transformadas 1D de longitud N, repartiendo filas con OpenMP.
 * @param field Entrada/Salida: Grilla M x N que se transforma por filas.
 * @param layout Entrada: 0 = Cooley-Tukey, 1 = Stockham.
 * @param inverse Entrada: false = directa, true = inversa sin normalizar.
 * @return Salida: void (todas las filas están transformadas al retornar).
 * @pre El llamador ya validó las dimensiones y el layout.
 * @note schedule(runtime) configura el reparto de filas. El fin de la región
 *       paralela garantiza que la siguiente pasada vea todas sus escrituras.
 */
void Fft2D::rowsPass(ComplexField& field, int layout, bool inverse) {
    const std::size_t M = field.rows();

    // Cada iteración escribe una fila distinta: los hilos no comparten destinos.
    // default(none) exige declarar el ámbito de los datos; r es privado por ser
    // el índice del bucle. M, layout e inverse se comparten solo para lectura.
    // schedule(runtime) toma tipo y chunk de omp_set_schedule u OMP_SCHEDULE.
    #pragma omp parallel for default(none) shared(field, M, layout, inverse) schedule(runtime)
    for (std::size_t r = 0; r < M; ++r) {
        transformRow(field, r, layout, inverse);
    }
    // El final de la región paralela espera todas las filas antes de retornar.
}

/**
 * @brief Ejecuta N transformadas 1D de longitud M, repartiendo columnas con OpenMP.
 * @param field Entrada/Salida: Grilla M x N que se transforma por columnas.
 * @param layout Entrada: 0 = Cooley-Tukey, 1 = Stockham.
 * @param inverse Entrada: false = directa, true = inversa sin normalizar.
 * @return Salida: void (todas las columnas están transformadas al retornar).
 * @pre El llamador ya validó las dimensiones y el layout.
 * @note Cada hilo reutiliza un buffer contiguo propio de M valores. getCol/setCol
 *       hacen las copias; Butterfly1D opera sobre el buffer. Las columnas tienen
 *       destinos distintos, y el fin de la región paralela espera toda la pasada.
 */
void Fft2D::colsPass(ComplexField& field, int layout, bool inverse) {
    const std::size_t M = field.rows();
    const std::size_t N = field.cols();

    // Una columna tiene elementos separados por N posiciones en memoria.
    // El núcleo 1D necesita un arreglo contiguo: copiamos la columna, la
    // transformamos y devolvemos el resultado a las mismas posiciones.
    #pragma omp parallel default(none) shared(field, M, N, layout, inverse)
    {
        // Declarado dentro de parallel: cada hilo tiene su propio vector.
        // Lo reutiliza para sus columnas, evitando reservarlo en cada iteración.
        std::vector<std::complex<double>> col_buf(M);

        // Los hilos pueden tocar las mismas filas, pero columnas distintas:
        // field.at(r, c) nunca tiene dos escritores en esta pasada.
        #pragma omp for schedule(runtime)
        for (std::size_t c = 0; c < N; ++c) {
            // Paso 1: reunir los M valores de la columna en el buffer del hilo.
            field.getCol(c, col_buf.data());

            // Paso 2: transformar el buffer; los ceros eligen static y chunk
            // por defecto en el núcleo 1D, independientemente del reparto de columnas.
            Butterfly1D butterfly(col_buf.data(), M, inverse);
            butterfly.transform(layout, 0, 0);

            // Paso 3: devolver la columna transformada a sus posiciones originales.
            field.setCol(c, col_buf.data());
        }
    }
}

// ============================================================================
// IMPLEMENTACIÓN: TRANSFORMADA DIRECTA E INVERSA 2D
// ============================================================================

/**
 * @brief Ejecuta la FFT 2D directa usando el layout guardado en la instancia.
 * @param field Entrada/Salida: Muestras espaciales; al finalizar contiene el espectro.
 * @return Salida: void (modifica field, sin aplicar escala).
 * @throws std::invalid_argument Si las dimensiones no son válidas.
 * @note Delega en forward(field, default_layout_); el constructor por defecto usa 0.
 */
void Fft2D::forward(ComplexField& field) {
    // La sobrecarga simple delega para mantener una sola implementación 2D.
    forward(field, default_layout_);
}

/**
 * @brief Calcula la FFT 2D separable con signo negativo y sin normalización.
 * @param field Entrada/Salida: Grilla espacial M x N; recibe el espectro en orden natural.
 * @param layout Entrada: 0 = Cooley-Tukey in-place, 1 = Stockham out-of-place.
 * @return Salida: void (reemplaza las muestras por sus coeficientes espectrales).
 * @throws std::invalid_argument Si las dimensiones o el layout no son válidos.
 * @note Primero transforma filas, espera su finalización y luego transforma columnas.
 *       La salida corresponde a la DFT con exp(-2*pi*i*(k*m/M + l*n/N)).
 */
void Fft2D::forward(ComplexField& field, int layout) {
    // Validar antes de iniciar OpenMP evita lanzar estos errores desde un hilo.
    validateField(field);
    validateLayout(layout);

    // La exponencial de la DFT 2D se separa en un factor por cada eje.
    // Por eso basta componer M FFT de longitud N con N FFT de longitud M.
    // 1. Pasada por filas (reparte las M transformadas 1D de tamaño N)
    rowsPass(field, layout, /*inverse=*/false);

    // 2. Sincronización implícita al terminar la región paralela de rowsPass.
    // 3. Pasada por columnas (reparte las N transformadas 1D de tamaño M)
    colsPass(field, layout, /*inverse=*/false);
}

/**
 * @brief Ejecuta la IFFT 2D usando el layout guardado en la instancia.
 * @param field Entrada/Salida: Espectro; al finalizar contiene las muestras espaciales.
 * @return Salida: void (modifica field y aplica la escala total 1/(M*N)).
 * @throws std::invalid_argument Si las dimensiones no son válidas.
 * @note Delega en inverse(field, default_layout_); el constructor por defecto usa 0.
 */
void Fft2D::inverse(ComplexField& field) {
    inverse(field, default_layout_);
}

/**
 * @brief Calcula la IFFT 2D separable con signo positivo y escala única 1/(M*N).
 * @param field Entrada/Salida: Espectro M x N; recibe las muestras espaciales recuperadas.
 * @param layout Entrada: 0 = Cooley-Tukey in-place, 1 = Stockham out-of-place.
 * @return Salida: void (reemplaza el espectro por la inversa normalizada).
 * @throws std::invalid_argument Si las dimensiones o el layout no son válidos.
 * @note Ejecuta columnas, filas y finalmente el escalado. Las pasadas 1D no
 *       normalizan; collapse(2) reparte únicamente las celdas independientes de
 *       la normalización, una vez terminadas ambas pasadas.
 */
void Fft2D::inverse(ComplexField& field, int layout) {
    validateField(field);
    validateLayout(layout);

    // Deshacer las pasadas en orden separable inverso (columnas y luego filas).
    // Esto es matemáticamente válido porque la DFT 2D es un producto tensorial lineal
    // y separable: las operaciones sobre dimensiones ortogonales conmutan.
    colsPass(field, layout, /*inverse=*/true);
    rowsPass(field, layout, /*inverse=*/true);

    // Las inversas 1D sin escala dejan un factor N en filas y M en columnas.
    // El contrato de Butterfly1D deja la normalización a Fft2D: dividir una
    // única vez por M*N recupera la entrada original.
    const size_t M = field.rows();
    const size_t N = field.cols();
    const double scale = 1.0 / (static_cast<double>(M) * static_cast<double>(N));

    // collapse(2) reparte las M*N celdas entre hilos. Es seguro porque cada
    // celda se multiplica de forma independiente, sin sumar a un acumulador.
    // El costo por celda es uniforme: static asigna bloques contiguos y evita
    // el reparto celda a celda aunque las pasadas FFT usen dynamic,1.
    #pragma omp parallel for default(none) shared(field, M, N, scale) collapse(2) schedule(static)
    for (size_t r = 0; r < M; ++r) {
        for (size_t c = 0; c < N; ++c) {
            field.at(r, c) *= scale;
        }
    }
}

// ============================================================================
// DISTRIBUCIÓN DE FILAS: PARALLEL FOR Y TASKS CON/SIN SINGLE
// ============================================================================

/**
 * @brief Ejecuta solo la pasada directa de filas mediante parallel for.
 * @param field Entrada/Salida: Grilla cuyas filas se transforman sin escala.
 * @return Salida: void (las columnas todavía no han sido transformadas).
 * @throws std::invalid_argument Si las dimensiones no son válidas.
 * @note Usa default_layout_ y delega en la variante parametrizada.
 */
void Fft2D::forwardRows(ComplexField& field) {
    forwardRows(field, TASK_TYPE_PARALLEL_FOR, true, default_layout_);
}

/**
 * @brief Ejecuta solo la pasada directa de filas con el mecanismo indicado.
 * @param field Entrada/Salida: Grilla cuyas filas se transforman sin escala.
 * @param task_type Entrada: 0 = task con un productor single, 1 = parallel for.
 * @return Salida: void (modifica únicamente la transformada del eje de las filas).
 * @throws std::invalid_argument Si las dimensiones o task_type no son válidos.
 * @note Delega con use_single=true y default_layout_.
 */
void Fft2D::forwardRows(ComplexField& field, int task_type) {
    forwardRows(field, task_type, true, default_layout_);
}

/**
 * @brief Ejecuta la pasada directa de filas con el layout guardado en la instancia.
 * @param field Entrada/Salida: Grilla cuyas filas se transforman sin escala.
 * @param task_type Entrada: 0 = task, 1 = parallel for.
 * @param use_single Entrada: true = un productor de tareas; false = todos los hilos.
 * @return Salida: void (todas las filas están listas al retornar).
 * @throws std::invalid_argument Si las dimensiones o task_type no son válidos.
 * @note use_single solo afecta a task; delega en la variante con layout explícito.
 */
void Fft2D::forwardRows(ComplexField& field, int task_type, bool use_single) {
    forwardRows(field, task_type, use_single, default_layout_);
}

/**
 * @brief Transforma las filas eligiendo layout y distribución OpenMP.
 * @param field Entrada/Salida: Grilla M x N; cada fila recibe una FFT de longitud N.
 * @param task_type Entrada: 0 = una task por fila, 1 = parallel for sobre las filas.
 * @param use_single Entrada: En task, true crea un productor y false reparte los
 *        productores entre hilos; en parallel for este parámetro no tiene efecto.
 * @param layout Entrada: 0 = Cooley-Tukey, 1 = Stockham.
 * @return Salida: void (todas las filas transformadas, sin escala ni pasada de columnas).
 * @throws std::invalid_argument Si dimensiones, layout o task_type no son válidos.
 * @note firstprivate(r) conserva el índice de cada tarea. taskwait espera los
 *       hijos del hilo actual; la barrera final de parallel garantiza que todas
 *       las tareas del equipo terminen antes de retornar. Para completar la FFT
 *       2D debe ejecutarse forwardCols(field, layout) con el mismo layout.
 */
void Fft2D::forwardRows(ComplexField& field, int task_type, bool use_single, int layout) {
    // Este método transforma SOLO filas. Para obtener una FFT 2D completa
    // hay que ejecutar después forwardCols con el mismo layout.
    // use_single solo cambia cómo se generan tareas en TASK_TYPE_TASK.
    validateField(field);
    validateLayout(layout);
    validateTaskType(task_type);

    const size_t M = field.rows();

    if (task_type == TASK_TYPE_PARALLEL_FOR) {
        // Reutiliza la misma pasada parallel for de la transformada completa.
        rowsPass(field, layout, /*inverse=*/false);
    } else if (task_type == TASK_TYPE_TASK) {
        // CASO 2: una tarea independiente por fila, con dos formas de producción.
        #pragma omp parallel default(none) shared(field, M, use_single, layout)
        {
            if (use_single) {
                // CASO 2.1: single elige un productor; cualquier hilo del equipo
                // puede ejecutar tareas. La barrera implícita espera su finalización.
                #pragma omp single
                {
                    for (size_t r = 0; r < M; ++r) {
                        // Una tarea puede ejecutarse después de avanzar el bucle:
                        // firstprivate(r) conserva el índice de SU fila.
                        #pragma omp task firstprivate(r) shared(field, layout)
                        {
                            transformRow(field, r, layout, /*inverse=*/false);
                        }
                    }
                }
            } else {
                // CASO 2.2: todos los hilos generan tareas para filas distintas.
                // Nota técnica: en esta rama, el taskwait local espera las tareas hijas
                // de cada hilo, y la barrera implícita al fin de la región paralela
                // asegura la finalización de todo el equipo antes de continuar.
                const int tid = omp_get_thread_num();
                const int nthreads = omp_get_num_threads();
                // Reparto cíclico de productores. Con 4 hilos, el hilo 0 genera
                // filas 0,4,8,... y el hilo 1 filas 1,5,9,...: no hay duplicados.
                for (size_t r = static_cast<size_t>(tid); r < M; r += static_cast<size_t>(nthreads)) {
                    #pragma omp task firstprivate(r) shared(field, layout)
                    {
                        transformRow(field, r, layout, /*inverse=*/false);
                    }
                }
            }
            // Espera las tareas hijas del hilo actual. La salida de parallel
            // garantiza además que todas las filas del equipo hayan terminado.
            #pragma omp taskwait
        }
    }
}

// ============================================================================
// PASADAS INDIVIDUALES DE COLUMNAS Y DE INVERSA SIN ESCALA
// ============================================================================

/**
 * @brief Ejecuta solo la pasada directa de columnas, con signo negativo.
 * @param field Entrada/Salida: Grilla cuyas N columnas reciben una FFT de longitud M.
 * @param layout Entrada: 0 = Cooley-Tukey, 1 = Stockham (por defecto 0).
 * @return Salida: void (columnas transformadas, sin escala).
 * @throws std::invalid_argument Si las dimensiones o el layout no son válidos.
 * @note Tras forwardRows, completa la FFT 2D. Omitir layout usa 0, incluso si
 *       la instancia fue construida con Stockham; para ese caso pasarlo explícitamente.
 */
void Fft2D::forwardCols(ComplexField& field, int layout) {
    // Segunda pasada de una directa separada manualmente en filas y columnas.
    validateField(field);
    validateLayout(layout);
    colsPass(field, layout, /*inverse=*/false);
}

/**
 * @brief Ejecuta solo la pasada inversa de filas, con signo positivo y sin escala.
 * @param field Entrada/Salida: Grilla cuyas M filas reciben una inversa 1D de longitud N.
 * @param layout Entrada: 0 = Cooley-Tukey, 1 = Stockham (por defecto 0).
 * @return Salida: void (filas transformadas, sin dividir por N ni por M*N).
 * @throws std::invalid_argument Si las dimensiones o el layout no son válidos.
 * @note inverseCols más inverseRows deja pendiente la escala total 1/(M*N).
 *       Usar inverse() para obtener la inversa 2D completa y normalizada.
 */
void Fft2D::inverseRows(ComplexField& field, int layout) {
    // Pasada inversa parcial sin escala; no equivale a inverse() completa.
    validateField(field);
    validateLayout(layout);
    rowsPass(field, layout, /*inverse=*/true);
}

/**
 * @brief Ejecuta solo la pasada inversa de columnas, con signo positivo y sin escala.
 * @param field Entrada/Salida: Grilla cuyas N columnas reciben una inversa 1D de longitud M.
 * @param layout Entrada: 0 = Cooley-Tukey, 1 = Stockham (por defecto 0).
 * @return Salida: void (columnas transformadas, sin dividir por M ni por M*N).
 * @throws std::invalid_argument Si las dimensiones o el layout no son válidos.
 * @note Es la primera pasada de inverse(); por sí sola no recupera una grilla
 *       desde un espectro 2D. Faltan la pasada de filas y la normalización total.
 */
void Fft2D::inverseCols(ComplexField& field, int layout) {
    // Se conserva la escala para que inverse() normalice solo al final.
    validateField(field);
    validateLayout(layout);
    colsPass(field, layout, /*inverse=*/true);
}

// ============================================================================
// DELEGACIONES SOBRE EL CAMPO ENLAZADO (BOUND_FIELD_)
// ============================================================================

// Estas sobrecargas usan la grilla recibida en el constructor. Delegan en las
// mismas rutas anteriores; una instancia creada con Fft2D() no tiene grilla
// enlazada y debe usar las variantes que reciben ComplexField explícitamente.

/**
 * @brief Ejecuta la FFT 2D sobre el campo enlazado con el layout de la instancia.
 * @return Salida: void (reemplaza el campo enlazado por su espectro, sin escala).
 * @throws std::runtime_error Si no se enlazó un campo mediante el constructor.
 * @throws std::invalid_argument Si las dimensiones del campo no son válidas.
 * @note Delega en forward(*bound_field_, default_layout_).
 */
void Fft2D::forward() {
    if (!bound_field_) throw std::runtime_error("Fft2D: No bound ComplexField instance.");
    forward(*bound_field_, default_layout_);
}

/**
 * @brief Ejecuta la FFT 2D sobre el campo enlazado con un layout explícito.
 * @param layout Entrada: 0 = Cooley-Tukey, 1 = Stockham, solo para esta llamada.
 * @return Salida: void (reemplaza el campo enlazado por su espectro, sin escala).
 * @throws std::runtime_error Si no hay campo enlazado.
 * @throws std::invalid_argument Si las dimensiones o el layout no son válidos.
 * @note Delega en forward(*bound_field_, layout); no cambia default_layout_.
 */
void Fft2D::forward(int layout) {
    if (!bound_field_) throw std::runtime_error("Fft2D: No bound ComplexField instance.");
    forward(*bound_field_, layout);
}

/**
 * @brief Ejecuta la IFFT 2D sobre el campo enlazado con el layout de la instancia.
 * @return Salida: void (reemplaza el espectro enlazado por la inversa normalizada).
 * @throws std::runtime_error Si no se enlazó un campo mediante el constructor.
 * @throws std::invalid_argument Si las dimensiones del campo no son válidas.
 * @note Delega en inverse(*bound_field_, default_layout_), incluyendo la escala 1/(M*N).
 */
void Fft2D::inverse() {
    if (!bound_field_) throw std::runtime_error("Fft2D: No bound ComplexField instance.");
    inverse(*bound_field_, default_layout_);
}

/**
 * @brief Ejecuta la IFFT 2D sobre el campo enlazado con un layout explícito.
 * @param layout Entrada: 0 = Cooley-Tukey, 1 = Stockham, solo para esta llamada.
 * @return Salida: void (reemplaza el espectro enlazado por la inversa normalizada).
 * @throws std::runtime_error Si no hay campo enlazado.
 * @throws std::invalid_argument Si las dimensiones o el layout no son válidos.
 * @note Delega en inverse(*bound_field_, layout); no cambia default_layout_.
 */
void Fft2D::inverse(int layout) {
    if (!bound_field_) throw std::runtime_error("Fft2D: No bound ComplexField instance.");
    inverse(*bound_field_, layout);
}

/**
 * @brief Transforma solo las filas del campo enlazado con el layout de la instancia.
 * @param task_type Entrada: 0 = task con single, 1 = parallel for.
 * @return Salida: void (filas listas, sin escala ni pasada de columnas).
 * @throws std::runtime_error Si no hay campo enlazado.
 * @throws std::invalid_argument Si las dimensiones o task_type no son válidos.
 * @note Delega en forwardRows(*bound_field_, task_type), usando use_single=true.
 */
void Fft2D::forwardRows(int task_type) {
    if (!bound_field_) throw std::runtime_error("Fft2D: No bound ComplexField instance.");
    forwardRows(*bound_field_, task_type);
}

/**
 * @brief Transforma filas del campo enlazado seleccionando los productores de tareas.
 * @param task_type Entrada: 0 = task, 1 = parallel for.
 * @param use_single Entrada: true = un productor, false = productores por hilo;
 *        no tiene efecto cuando task_type es parallel for.
 * @return Salida: void (filas listas, sin escala ni pasada de columnas).
 * @throws std::runtime_error Si no hay campo enlazado.
 * @throws std::invalid_argument Si las dimensiones o task_type no son válidos.
 * @note Delega en forwardRows(*bound_field_, task_type, use_single) con default_layout_.
 */
void Fft2D::forwardRows(int task_type, bool use_single) {
    if (!bound_field_) throw std::runtime_error("Fft2D: No bound ComplexField instance.");
    forwardRows(*bound_field_, task_type, use_single);
}
