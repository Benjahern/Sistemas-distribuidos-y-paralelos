# Guía autónoma para agentes — Laboratorio 1

Estas instrucciones contienen los requisitos del proyecto para que el agente pueda trabajar sin consultar otro documento. Si una petición del usuario contradice un requisito académico, sigue la petición solo para el alcance explícito y señala la diferencia; no des por cumplido un requisito que se haya omitido.

## 1. Propósito y alcance

Implementar en C++17 una FFT 2D y su IFFT para grillas complejas de `M x N`, en memoria compartida mediante OpenMP. La transformada es separable: FFT 1D en cada fila y luego en cada columna. `M` y `N` son potencias de dos y pueden ser distintos. La FFT 1D radix-2 debe ofrecer dos layouts de la misma transformada: in-place Cooley–Tukey y Stockham out-of-place. No implementar filtros de imagen, solvers, otra radix, longitudes no potencia de dos ni usar bibliotecas externas de FFT, salvo solicitud explícita.

La propiedad central de corrección es `IFFT(FFT(x)) ≈ x`. Ambos layouts deben producir la misma DFT, en orden natural y dentro de tolerancia. Mantener una sola lógica de mariposa; no duplicarla por schedule, cláusula o layout.

## 2. Convenciones matemáticas obligatorias

Para `x[m,n]`, con `0 <= m < M`, `0 <= n < N`, la directa es:

`X[k,l] = sum_m sum_n x[m,n] exp(-2*pi*i*(k*m/M + l*n/N))`.

La inversa es:

`x[m,n] = (1/(M*N)) sum_k sum_l X[k,l] exp(+2*pi*i*(k*m/M + l*n/N))`.

- La directa no escala. La inversa aplica el factor `1/(M*N)` una sola vez en el par completo (puede distribuirse internamente entre pasadas solo si el resultado equivale exactamente a una aplicación total y queda documentado; preferir una aplicación única explícita).
- En 1D, la inversa usa el signo positivo y factor `1/N`; en 2D, el factor total es `1/(M*N)`.
- La IFFT puede deshacer filas/columnas en orden inverso o en otro orden documentado, siempre que el resultado sea correcto.
- Parseval bajo esta convención: `sum(|x|^2) = (1/(M*N)) * sum(|X|^2)`.
- Error de ida y vuelta: raíz del error cuadrático medio entre la entrada y `IFFT(FFT(x))`. Documentar `rtol`/`atol` o norma y justificarla. En doble precisión, `1e-10` de error absoluto medio es punto de partida, no sustituto de validación.

## 3. Algoritmos FFT 1D

### Cooley–Tukey in-place

Primero aplicar permutación bit-reversal a la entrada. Para longitud `N` hay `p=log2(N)` etapas. En etapa `s=1..p`, `m=2^s`; cada pareja separada por `m/2` calcula `u=a[j]`, `v=omega*a[j+m/2]`, y escribe `a[j]=u+v`, `a[j+m/2]=u-v`. `omega=exp(-2*pi*i*r/m)` en directa y `exp(+2*pi*i*r/m)` en inversa. Las parejas dentro de una etapa son disjuntas. La siguiente etapa depende de todas las escrituras de la anterior: se requiere barrera entre etapas.

### Stockham out-of-place

No aplica bit-reversal. Cada etapa lee `src` y escribe `dst`, y al final intercambia ambos buffers. Con etapas `s=0..p-1`, `l=2^s`, `g=N/2^(s+1)`, índices `k=0..l-1`, `j=0..g-1`:

`omega=exp(-2*pi*i*k/2^(s+1))` (cambiar signo para inversa); `i0=j+2*k*g`, `i1=i0+g`; `o0=j+k*g`, `o1=o0+N/2`; `u=src[i0]`, `v=omega*src[i1]`; `dst[o0]=u+v`, `dst[o1]=u-v`.

Los destinos de una etapa son disjuntos. Barrera antes de intercambiar buffers/avanzar a la siguiente etapa. Al terminar, salida en orden natural. Comprobar impulso 1D `(1,0,0,0)` -> `(1,1,1,1)` en ambos layouts.

## 4. Paralelismo y cláusulas OpenMP

Deben existir ambos niveles de paralelismo:

1. Transformadas independientes: repartir filas y columnas; cada hilo ejecuta la FFT 1D completa asignada.
2. Mariposas dentro de una etapa 1D: repartir parejas de esa etapa; mantener sincronización antes de comenzar la etapa siguiente.

No colapsar etapa y mariposa. `collapse` solo sobre iteraciones independientes (p. ej. índices de parejas de una etapa o error punto a punto). `nowait` nunca puede quitar la barrera entre etapas; úsalo solo en bucles independientes si una barrera posterior o fin de región garantiza sincronización. La referencia es el mismo código serial con un hilo.

En rutas reales compiladas y ejecutadas por prueba o benchmark deben aparecer y quedar justificadas:

- Schedules `static`, `dynamic`, `guided`, con chunk configurable.
- Sincronización `atomic`, `critical` y `reduction` aplicadas a suma de error o Parseval.
- `single`, `nowait` (solo seguro), `collapse` en iteraciones independientes.
- Datos `private`, `shared`, `firstprivate`, `lastprivate`.
- `barrier` entre etapas de mariposa.
- `task` para distribuir filas o columnas y comparación numérica con `parallel for`.

Evitar carreras: parejas y salidas por iteración deben ser disjuntas. No ejecutar simultáneamente etapas dependientes. Todo el paralelismo debe mantener resultados dentro de la tolerancia.

## 5. Interfaces y módulos esperados

Conservar los módulos separados por responsabilidad. Declaraciones en `include/`, definiciones en `src/`; añadir pruebas bajo `tests/`. Actualizar `Makefile`, Docker/CI y documentación al incorporar componentes. Interfaces requeridas por el laboratorio:

- `Butterfly1D`: `transform()`; `transform(int schedule_type)` con `0=static`, `1=dynamic`, `2=guided`; `transform(int schedule_type, int chunk_size)`; `transformCollapse()`; `transformStages()` (paralelo por mariposas y barrera); `transform(int layout, int schedule_type, int chunk_size)` con `0=in-place`, `1=Stockham`.
- `Fft2D`: `forward()`, `inverse()` y variantes con `int layout`; `forwardRows(int task_type)` con `0=task`, `1=parallel for`; `forwardRows(int task_type, bool use_single)`.
- `SpectrumMetrics`: `roundtripError()`; `parseval(int method)` con `0=reduction`, `1=atomic`; `parseval(int method, bool use_private)`.
- La clase apropiada debe exponer `initTwiddlesSingle()`, `magnitudeNoWait()`, `accumulateFirstprivate()` y `stageIndexLastprivate()` para demostrar esas cláusulas en una ruta ejecutada.
- El modelo de datos representa grilla compleja `M x N`, buffers FFT y Stockham, bit-reversal, generación aleatoria reproducible por semilla, y lectura/escritura de campos o `.dat` según se necesite.
- `Benchmark`: medición y estadísticas descritas abajo. `Visualizer`: módulo `|X[k,l]|` y salida de datos/figuras requeridas.

No inventar que una interfaz o ruta ya existe: inspeccionar el código y mantener compatibilidad cuando sea posible. Elegir sobrecargas para delegar en una única implementación parametrizada, no copiar la mariposa.

## 6. Pruebas de corrección requeridas

`make test` debe compilar y ejecutar la suite, también desde Docker y CI. Como mínimo:

- Bit-reversal de índices conocidos para `N=8`.
- FFT 1D de impulso `(1,0,0,0)` y seno de un período, ambos layouts, contra forma cerrada y entre sí.
- IFFT 1D de esos espectros en ambos layouts, con escala correcta.
- FFT 2D del impulso `2x2` y grilla rectangular pequeña (p. ej. `8x4`), ambos layouts.
- Ida y vuelta serial y OpenMP en ambos layouts con misma tolerancia.
- Parseval con semilla fija.
- Caso que detecte barrera ausente o escrituras solapadas comparando paralelo con serial en tamaño sensible.

Probar longitudes `N in {4,8,16}` y al menos una grilla rectangular. Usar semilla fija. Incluir impulso `(0,0)`, seno 2D de frecuencias enteras conocidas (pico de módulo esperado) y campo aleatorio reproducible. Ejecutar de verdad las rutas de cláusulas obligatorias. No presentar un target vacío o un mensaje como prueba exitosa.

## 7. Benchmarks y análisis

- Medir con `omp_get_wtime()`, mínimo diez repeticiones por punto, reportar `T = promedio ± desviación estándar`.
- `S_p=T_1/T_p`, donde `T_1` usa el mismo binario y grilla con un hilo. Eficiencia `E_p=S_p/p`.
- Amdahl: `S_p=1/(f+(1-f)/p)`. Estimar y explicar la fracción serial `f` (bit-reversal, swaps Stockham, transposición si aplica, setup de twiddles u otra medida), y comparar predicción con tiempos.
- Propagar incertidumbre: `sigma_S = S * sqrt((sigma_T1/promedio_T1)^2 + (sigma_Tp/promedio_Tp)^2)`; `sigma_E=sigma_S/p`.
- Medir hilos `1,2,4,8` y, si existen, núcleos lógicos disponibles. Tamaños cuadrados de potencia de dos desde uno pequeño (p. ej. 128 o 256), duplicando hasta el máximo viable. Reportar límite y por qué no se midió el siguiente. Comparar layouts; barrer chunk puede limitarse a un tamaño.
- Producir gráficos de speedup vs hilos, eficiencia vs hilos, tiempo vs chunk para static/dynamic/guided, in-place vs Stockham, Amdahl vs medición y error de ida/vuelta vs tamaño o hilos.
- Visualizar módulo del espectro de seno 2D conocido con escala lineal o logarítmica declarada; comentar pico y error vs hilos.
- Salidas esperadas: `benchmark_results.dat`, `scaling_analysis.dat`, `performance_plots.png`, `spectrum.dat`/`spectrum.png` (o equivalente), `roundtrip_error.dat`.

## 8. Calidad, compilación y entregables

- C++17, `-Wall -Wextra`, OpenMP (`-fopenmp`). `Makefile` debe tener `all`, `clean` y `test`; `test` compila y ejecuta las pruebas reales.
- Dockerfile debe compilar, ejecutar `make test` y una demostración corta de FFT 2D. CI ejecuta `make test` en el contenedor o entorno equivalente.
- README documenta cómo compilar, ejecutar pruebas y repetir benchmarks; tolerancia y justificación; convención de escala; roles y mapeo de cláusulas (archivo, función y motivo).
- Roles de referencia: (1) modelo/datos y buffers; (2) mariposa radix-2, layouts, schedules y barreras; (3) FFT/IFFT 2D; (4) métricas y benchmarks; (5) pruebas, CI, Docker y visualización. En equipo de cuatro, una persona asume dos roles y README/reporte indican quiénes y cómo se mantienen interfaces separadas.
- Entregables: código y tests, Makefile, Dockerfile y CI, README, tablas `.dat` y figuras PNG, reporte técnico PDF de hasta 10 páginas con problema, diseño/roles, cláusulas, resultados y limitaciones.
- Mantener compilación sin advertencias. Validar primero `N=4` en ambos modos y el impulso `2x2` antes de benchmarks grandes. La barrera es parte del algoritmo.
- Informar con honestidad qué pruebas no se pudieron ejecutar y por qué; no afirmar completitud o conformidad sin evidencia.

## 9. Estructura del repositorio

- `src/`: `.cpp`, incluido `main.cpp`.
- `include/`: cabeceras `.h`.
- `tests/`: pruebas.
- Raíz de `Lab1/`: `Makefile`, `Dockerfile`, `README.md`, este archivo de instrucciones y enunciado de referencia.
- No modificar ni incluir `.opencode/` en entregables/versionado; está ignorado por Git.
