# Laboratorio 1: Programación paralela con OpenMP
## FFT e IFFT bidimensional (C++)

**Departamento de Ingeniería Informática**  
**Universidad de Santiago de Chile**  
*25 de septiembre de 2026*

---

## 1. Motivación

La entrega de este laboratorio es el par FFT 2D + IFFT 2D. No se pide un filtro de imagen, un solver espectral ni una aplicación cerrada alrededor de la transformada: el objeto de estudio es la transformada y su inversa.

La DFT discreta expresa una señal como suma de exponenciales complejas. En dos dimensiones, para una grilla $x_{m,n}$ de tamaño $M \times N$,

$$X_{k,l} = \sum_{m=0}^{M-1} \sum_{n=0}^{N-1} x_{m,n} e^{-2\pi i \left(\frac{km}{M} + \frac{ln}{N}\right)}, \quad k=0,\dots,M-1, \quad l=0,\dots,N-1 \tag{1}$$

La inversa, con la convención de que el factor de escala vive solo en la IFFT, es

$$x_{m,n} = \frac{1}{MN} \sum_{k=0}^{M-1} \sum_{l=0}^{N-1} X_{k,l} e^{+2\pi i \left(\frac{km}{M} + \frac{ln}{N}\right)} \tag{2}$$

Evaluar esas sumas de forma directa cuesta $\mathcal{O}(M^2 N^2)$. La FFT de radix 2 baja una transformada 1D de longitud potencia de dos a $\mathcal{O}(N \log N)$ reutilizando productos parciales en una red de mariposas. Como la exponencial 2D se factoriza, la FFT 2D es una FFT 1D sobre cada fila seguida de una FFT 1D sobre cada columna. La IFFT 2D recorre el mismo esquema con el signo conjugado en el twiddle y con el factor $1/(MN)$.

Esa factorización es la razón por la que el par se usa como primitiva, no como un fin en sí mismo. El teorema de convolución dice que una convolución en el dominio espacial es un producto punto a punto de transformadas: filtrar una imagen, una sección sísmica o un campo en un método espectral se reduce a transformar, multiplicar y antitransformar. En la práctica aparece en el procesamiento de imágenes y de video, en la reconstrucción por resonancia magnética, en radioastronomía (correlación y síntesis de apertura), en cristalografía y en la resolución de ecuaciones en derivadas parciales cuando la base es de modos de Fourier. En todos esos usos la pieza verificable es la misma: $\text{IFFT}(\text{FFT}(x))$ debe recuperar $x$, y la energía de $x$ debe coincidir con la del espectro según Parseval.

En este laboratorio esa pieza es una sola: la FFT 2D y su IFFT, en memoria compartida. La mariposa es la de radix-2. Tiene dos modos de ejecución, no dos transformadas. El modo *in-place* (Cooley-Tukey) permuta la entrada con bit-reversal y escribe cada etapa sobre el mismo arreglo. El modo *Stockham* es una funcionalidad de la misma mariposa: cada etapa lee un buffer y escribe el otro, sin bit-reversal, y el espectro queda en orden natural. Ambos modos calculan la misma DFT, con los mismos twiddles y la misma escala, y la FFT 2D elige el modo por un argumento. OpenMP paraleliza las mariposas de una etapa; una barrera mal puesta o una escritura solapada rompe esa etapa.

---

## 2. Objetivos

Al finalizar, ustedes serán capaces de:
- Implementar la DFT (*Discrete Fourier Transform*) directa y la inversa en 2D a partir de FFT 1D separables.
- Implementar una mariposa radix-2 con dos modos: *in-place* (Cooley-Tukey, con bit-reversal) y *Stockham* (fuera de lugar, orden natural).
- Recorrer las etapas de esa mariposa con una barrera entre ellas y sin carreras dentro de una etapa.
- Aplicar cláusulas de scheduling, sincronización y reparto de datos de OpenMP sobre esas etapas.
- Verificar el par FFT/IFFT y que ambos modos dan la misma DFT, con tolerancia y con Parseval.
- Medir speedup, eficiencia y la ley de Amdahl, con propagación de la incertidumbre de los tiempos.
- Organizar el código en clases C++ con pruebas, contenedor e integración continua.

---

## 3. Organización del equipo

Hay grupos de cinco integrantes y grupos de cuatro. Hay cinco roles. Cada persona asume un rol principal y puede colaborar en el resto. En un grupo de cuatro, un integrante cubre dos roles: en el `README.md` y en el reporte indiquen quién, qué dos roles, y cómo se separan las interfaces aunque las implemente la misma persona. En el test de salida se puede preguntar por el trabajo del rol, o de los dos roles si corresponde.

| Rol | Responsabilidades típicas |
| :--- | :--- |
| **1. Modelo y datos** | Grilla compleja $M \times N$, buffer de la FFT, bit-reversal del modo *in-place*, segundo buffer del modo Stockham, generación reproducible (semilla) y lectura/escritura de campos o `.dat`. |
| **2. Núcleo mariposa 1D** | Una mariposa radix-2 y sus dos modos (*in-place* y Stockham), twiddle $\omega$, OpenMP sobre las parejas de una etapa, schedules y barrera entre etapas. |
| **3. FFT e IFFT 2D** | Transformada separable por filas y por columnas; directa sin factor $1/(MN)$; inversa con el signo conjugado y ese factor; composición $\text{IFFT}(\text{FFT}(\cdot))$. |
| **4. Métricas y benchmarks** | Error de ida y vuelta, Parseval, `omp_get_wtime()`, repeticiones, speedup, eficiencia, Amdahl y propagación de errores. |
| **5. Calidad, CI y visualización** | Pruebas unitarias e integración, Docker, pipeline CI, figura del módulo del espectro y objetivo `make test`. |

---

## 4. Calendario sugerido

- **Publicación:** viernes 25 de septiembre de 2026.
- **Entrega:** viernes 16 de octubre de 2026, 23:59.
- *Nota:* El lunes 12 de octubre es feriado (no hay clases).

La división siguiente es orientativa:
1. **25 sep – 2 oct:** Repositorio, contenedor y CI mínimo; grilla y bit-reversal; FFT 1D serial para $N=4$ y $N=8$, en modo in-place y en modo Stockham, con el mismo resultado.
2. **3 – 9 oct:** Etapas en paralelo, con barrera, en los dos modos; FFT 2D por filas y columnas; comparación serial vs. OpenMP en tamaños pequeños.
3. **10 – 16 oct:** IFFT 2D, ida y vuelta, Parseval; benchmarks de hilos y de algoritmo, figuras, reporte y revisión cruzada por rol.

---

## 5. Descripción del problema

Se pide un programa en C++ que, dada una grilla compleja de $M$ filas por $N$ columnas, calcule su FFT 2D y pueda invertirla. Hay una sola FFT 1D detrás de las filas y de las columnas. Esa FFT acepta un modo: *in-place* o *Stockham*. $M$ y $N$ son potencias de dos (puede ser $M \ne N$). No se pide una librería externa de FFT, ni Winograd, ni radix distinto de 2, ni un método para longitudes que no sean potencias de dos.

### 5.1. DFT 1D y separabilidad

La DFT 1D directa de una fila (o columna) de longitud $N$, sin factor $1/N$, es

$$X_k = \sum_{n=0}^{N-1} x_n e^{-2\pi i kn/N}, \quad k=0,\dots,N-1 \tag{3}$$

La inversa es

$$x_n = \frac{1}{N} \sum_{k=0}^{N-1} X_k e^{+2\pi i kn/N} \tag{4}$$

La transformada 2D se obtiene así:
1. Sustituir cada fila por su FFT 1D directa.
2. Sustituir cada columna del resultado por su FFT 1D directa.

La inversa deshace esas dos pasadas (columnas y luego filas, o el orden inverso documentado) usando la IFFT 1D. El factor $1/(MN)$ debe aparecer una sola vez en toda la ida y vuelta. Declárenlo en el `README`. Un error habitual es ponerlo en la directa, en ambas pasadas, o no ponerlo.

### 5.2. Modo in-place (Cooley-Tukey)

El modo in-place parte de una permutación *bit-reversal* de la entrada. Después hay $p = \log_2 N$ etapas. En la etapa $s = 1,\dots,p$ el tamaño de grupo es $m = 2^s$. Cada mariposa mezcla dos muestras separadas por $m/2$, con el twiddle

$$\omega_m^r = e^{-2\pi i r/m}$$

en la directa y $e^{+2\pi i r/m}$ en la inversa. Si $a_j$ y $a_{j+m/2}$ son esa pareja y $u, v$ son temporales,

$$u = a_j \tag{5}$$

$$v = \omega_m^r a_{j+m/2}$$

$$a_j \leftarrow u + v \tag{6}$$

$$a_{j+m/2} \leftarrow u - v$$

Dentro de una etapa las parejas son disjuntas: cada índice se escribe una sola vez. La etapa $s+1$ lee lo que escribió la etapa $s$, así que no puede empezar hasta que todas las mariposas de $s$ hayan terminado.

```
[Diagrama conceptual - Figura 1: Cooley-Tukey, N=4, tras bit-reversal]
Índices tras bit-reversal:
  (0) ---------> [etapa s=1] ---------> (barrera) ---------> [etapa s=2] (hilo A)
  (1) ---------> [etapa s=1] ---------> (barrera) ---------> [etapa s=2] (hilo A)
  (2) ---------> [etapa s=1] ---------> (barrera) ---------> [etapa s=2] (hilo B)
  (3) ---------> [etapa s=1] ---------> (barrera) ---------> [etapa s=2] (hilo B)

Cada color/hilo ejecuta mariposas cuyas escrituras no se solapan dentro de una etapa;
el '#pragma omp parallel for' de esa etapa es válido.
La etapa siguiente lee esos valores actualizados; requiere una barrera explícita.
```

*Figura 1:* Cooley-Tukey, $N=4$, después del bit-reversal. Cada color es un hilo: en una etapa sus escrituras no se solapan con las del otro, así que el `parallel for` de esa etapa es válido. La etapa siguiente lee esos valores; no puede adelantarse a la barrera.

### 5.3. Modo Stockham

Stockham (*autosort*, 1966) no define otra DFT. Es el modo fuera de lugar de la misma mariposa: el mismo $u \pm v$ y el mismo twiddle. Cambia dónde se escribe y desaparece el bit-reversal. La etapa $s = 0,\dots,p-1$ lee `src` y escribe `dst`. Con $l = 2^s$ y $g = N / 2^{s+1}$, para $k = 0,\dots,l-1$ y $j = 0,\dots,g-1$:

$$\omega = e^{-2\pi i k / 2^{s+1}} \tag{7}$$

$$i_0 = j + 2kg, \quad i_1 = i_0 + g \tag{8}$$

$$o_0 = j + kg, \quad o_1 = o_0 + N/2 \tag{9}$$

$$u = \text{src}_{i_0}, \quad v = \omega \, \text{src}_{i_1} \tag{10}$$

$$\text{dst}_{o_0} \leftarrow u + v \tag{11}$$

$$\text{dst}_{o_1} \leftarrow u - v$$

Al cerrar la etapa se intercambian `src` y `dst`. Después de $p$ etapas el espectro directo queda en orden natural en el buffer de la última escritura. En la inversa el exponente de $\omega$ cambia de signo, igual que en Cooley-Tukey. Los $N/2$ pares $(o_0, o_1)$ de una etapa son disjuntos, así que el bucle sobre $k$ y $j$ se puede paralelizar; la etapa siguiente espera a que todas esas escrituras hayan terminado.

Para $x = (1, 0, 0, 0)$ los dos modos deben dar $X = (1, 1, 1, 1)$. Si no coinciden, el modo Stockham dejó de ser la misma FFT.

```
[Diagrama conceptual - Figura 2: Stockham N=4, etapa s=1]
Buffer src                     Buffer dst
  [0] --------------> [0]  (Hilo A escribe o0, o1)
  [1] ---------\ /--> [1]  (Hilo B escribe otros)
  [2] ---------/ \--> [2]
  [3] --------------> [3]
  Al terminar: barrera e intercambio src <-> dst.
```

*Figura 2:* Modo Stockham de la misma FFT, $N=4$, etapa $s=1$. Cada hilo lee una pareja en `src` y escribe dos celdas distintas de `dst`. Nadie escribe el buffer que se está leyendo. La etapa siguiente solo empieza después de la barrera, ya con los buffers intercambiados.

### 5.4. Convención de paralelismo

Salvo que el reporte documente otra variante equivalente:
- **Hay dos recorridos, y ambos deben existir:**
  - **Por transformadas independientes:** un `parallel for` sobre las filas (y luego sobre las columnas). Cada hilo ejecuta la mariposa 1D completa de su fila o columna.
  - **Por mariposas de una etapa:** dentro de la FFT 1D, en el modo *in-place* o en el modo *Stockham*, un `parallel for` sobre las parejas de la etapa, y una barrera antes de la siguiente.
- `collapse` solo une iteraciones que sean independientes entre sí (por ejemplo parejas de una misma etapa, o el bucle del error punto a punto). No se puede colapsar el índice de etapa con el índice de mariposa.
- `nowait` no va entre etapas. Puede usarse en un bucle sin dependencia entre iteraciones (módulo del espectro, acumulación parcial de error) si más adelante hay una barrera o el fin de una región paralela.
- La referencia de corrección es una FFT serial del mismo código, con un hilo.

```
[Diagrama conceptual - Figura 3: Recorrido por transformadas 2D]
      Pasada por filas                      Pasada por columnas
  [   Hilo 0 (Fila 0)   ]                | H | H | H | H |
  [   Hilo 1 (Fila 1)   ]   (Barrera)    | i | i | i | i |
  [   Hilo 2 (Fila 2)   ]   -------->    | l | l | l | l |
  [   Hilo 3 (Fila 3)   ]                | o | o | o | o |
                                         | 0 | 1 | 2 | 3 |
```

*Figura 3:* FFT 2D separable con un hilo por transformada 1D. El `parallel for` (o las `task`) reparte filas que no comparten escrituras; la pasada de columnas espera a que todas las filas hayan terminado. Es el otro recorrido obligatorio, distinto del de las figuras 1 y 2, que parten una FFT 1D entre hilos.

**Ejemplo:**  
La DFT 1D de $x = (1, 0, 0, 0)$ es $X = (1, 1, 1, 1)$. La IFFT, con el factor $1/4$, recupera $x$.  
En 2D, la grilla $2 \times 2$ con un uno en $(0,0)$ y ceros en el resto tiene FFT constante e igual a 1 en las cuatro frecuencias; la IFFT, con factor $1/4$, devuelve el impulso.

### 5.5. Parseval

Con la directa sin $1/(MN)$ y la inversa con ese factor:

$$\sum_{m,n} |x_{m,n}|^2 = \frac{1}{MN} \sum_{k,l} |X_{k,l}|^2 \tag{12}$$

El error de ida y vuelta se mide como raíz del error cuadrático medio entre $x$ e $\text{IFFT}(\text{FFT}(x))$. La tolerancia (`rtol`, `atol` o una norma) se documenta en el `README`. La suma de esa norma, o la suma de $|X_{k,l}|^2$, es el sitio natural para `reduction`, y para una variante con `atomic` o `critical`.

---

## 6. Requisitos técnicos

### 6.1. Cláusulas OpenMP obligatorias

Cada ítem debe aparecer en una ruta compilada y ejecutada por un test o un benchmark. El reporte cita archivo, función y motivo.

1. **Scheduling**, sobre el bucle de filas, de columnas o de mariposas de una etapa:
   - `schedule(static, chunk)`
   - `schedule(dynamic, chunk)`
   - `schedule(guided, chunk)`
2. **Sincronización:** `atomic`, `critical` y `reduction` en la norma de error o en la suma de Parseval.
3. **Otras:** `single`, `nowait` (solo donde no hay dependencia de etapa) y `collapse` en un bucle anidado de iteraciones independientes.
4. **Datos:** `private`, `shared`, `firstprivate`, `lastprivate`.
5. **Barrera y tareas:** `barrier` entre etapas de la mariposa; `task` para repartir filas (o columnas) frente a un `parallel for` con el mismo resultado numérico.

### 6.2. Análisis de rendimiento

1. Tiempos con `omp_get_wtime()`. Al menos diez repeticiones. Reportar $T = \overline{T} \pm \sigma_T$.
2. Speedup $S_p = T_1 / T_p$ con $T_1$ el mismo binario y la misma grilla en un hilo.
3. Eficiencia $E_p = S_p / p$.
4. Ley de Amdahl $S_p = 1 / (f + (1 - f)/p)$, estimando la fracción serial $f$ (bit-reversal del modo *in-place*, intercambio de buffers del modo *Stockham*, transpuesta si la usan, preparación de twiddles, u otra sección que midan) y comparando la curva con los tiempos.
5. Propagación: para $S_p = T_1 / T_p$:

$$\sigma_{S_p} = S_p \sqrt{\left(\frac{\sigma_{T_1}}{\overline{T}_1}\right)^2 + \left(\frac{\sigma_{T_p}}{\overline{T}_p}\right)^2}, \quad \sigma_{E_p} = \frac{\sigma_{S_p}}{p} \tag{13}$$

---

## 7. Estructura sugerida del proyecto

La organización de archivos es una sugerencia. Pueden fusionar o partir módulos si el `README` indica dónde vive cada rol. La lógica de cada mariposa no se duplica por cada cláusula: usen sobrecarga y que el método simple delegue en el parametrizado.

```text
fft2d/
├── main.cpp
├── ComplexField.h/.cpp     # grilla MxN, bit-reversal, buffer Stockham, I/O
├── Butterfly1D.h/.cpp      # una mariposa, modo in-place o Stockham
├── Fft2D.h/.cpp            # filas, columnas, directa e inversa
├── SpectrumMetrics.h/.cpp  # error de ida y vuelta, Parseval
├── Benchmark.h/.cpp
├── Visualizer.h/.cpp       # módulo del espectro
├── tests/
├── Dockerfile
├── Makefile
└── README.md
```

### 7.1. Métodos obligatorios

1. **`Butterfly1D`:**
   - `transform()`: directa, serial o con el schedule por defecto documentado.
   - `transform(int schedule_type)`: con $0 = \text{estático}$, $1 = \text{dinámico}$, $2 = \text{guiado}$.
   - `transform(int schedule_type, int chunk_size)`
   - `transformCollapse()`: `collapse` solo sobre índices independientes de una etapa.
   - `transformStages()`: paralelo por mariposas, con `barrier` entre etapas.
   - `transform(int layout, int schedule_type, int chunk_size)`: con $\text{layout} = 0$ (*in-place*) y $\text{layout} = 1$ (*Stockham*): misma escala y el mismo orden de frecuencias.
2. **`Fft2D`** (elige el modo de la mariposa; no hay una segunda transformada):
   - `forward()`, `inverse()`, y los mismos con `int layout`.
   - `forwardRows(int task_type)`: con $0 = \text{task}$, $1 = \text{parallel for}$.
   - `forwardRows(int task_type, bool use_single)`
3. **`SpectrumMetrics`:**
   - `roundtripError()`
   - `parseval(int method)`: con $0 = \text{reduction}$, $1 = \text{atomic}$.
   - `parseval(int method, bool use_private)`
4. **Cláusulas puntuales**, en la clase que indiquen en el `README`:
   - `initTwiddlesSingle()`: `single`
   - `magnitudeNoWait()`: `nowait` fuera de las etapas
   - `accumulateFirstprivate()` y `stageIndexLastprivate()`

---

## 8. Parámetros

- **Tests de corrección:** $N \in \{4, 8, 16\}$ y al menos una grilla 2D pequeña, no cuadrada, por ejemplo $8 \times 4$. Semilla fija.
- **Benchmarks de desempeño:** grillas cuadradas $N \times N$ con $N = 2^p$. El primer lado es chico (por ejemplo 128 o 256) y cada punto siguiente duplica ese lado. La serie llega hasta el mayor $N$ que el equipo completa; el reporte indica ese máximo y por qué el tamaño siguiente no se midió (memoria, tiempo, o el criterio que acuerden). Esa serie es la de speedup, eficiencia, comparación de modos y error de ida y vuelta frente a $N$. El barrido de chunk puede hacerse en un solo tamaño de la serie.
- **Hilos:** 1, 2, 4, 8 y, si existen, el número de núcleos lógicos de la máquina.
- **Entrada de referencia:** impulso en $(0,0)$; un seno 2D de frecuencias enteras conocidas (un pico en el módulo del espectro); un campo aleatorio con semilla, para el error de ida y vuelta.
- **Tolerancia:** la tolerancia de $\text{IFFT}(\text{FFT}(x)) \approx x$ se justifica en el `README`. Un punto de partida razonable, en doble precisión y en el mayor tamaño que midan, está cerca de $10^{-10}$ en valor absoluto medio, y se ajusta si los twiddles se calculan en otra precisión.

---

## 9. Análisis y visualización

### 9.1. Gráficos

1. Speedup de la FFT 2D frente al número de hilos, para los tamaños de la serie de benchmarks.
2. Eficiencia frente al número de hilos.
3. Tiempo frente al chunk para `static`, `dynamic` y `guided`.
4. Tiempo del modo *in-place* frente al modo *Stockham*, misma FFT 2D, mismo tamaño y mismo número de hilos.
5. Predicción de Amdahl frente a la medición, con la $f$ que hayan estimado.
6. Módulo $|X_{k,l}|$ de un seno 2D de frecuencia conocida, en escala que permita ver el pico (lineal o logarítmica, declarada).
7. Error de ida y vuelta frente a $N$, o frente al número de hilos, para mostrar que el paralelo no cambia el resultado fuera de la tolerancia.

### 9.2. Archivos de salida

- `benchmark_results.dat`, `scaling_analysis.dat`
- `performance_plots.png`
- `spectrum.dat` y `spectrum.png` (o equivalente)
- `roundtrip_error.dat`

---

## 10. Pruebas automáticas

`make test` corre en el contenedor y en CI. El mínimo es el siguiente. Pueden agregar más casos (otras longitudes, otra norma, otro generador de entrada).

- Bit-reversal de índices conocidos ($N = 8$), usado solo por el modo *in-place*.
- DFT 1D de $(1, 0, 0, 0)$ y de un seno de un período, en los dos modos, frente a la forma cerrada y entre sí.
- IFFT 1D de ese espectro, con el factor $1/N$, en los dos modos.
- FFT 2D del impulso $2 \times 2$ y de una grilla rectangular pequeña, en los dos modos.
- Ida y vuelta serial e ida y vuelta con OpenMP, misma tolerancia, en los dos modos.
- Parseval en una grilla con semilla fija.
- Un caso que falle si se rompe la barrera entre etapas o si dos hilos escriben la misma pareja (puede ser una comparación contra la referencia serial en un tamaño donde ese error se vea).

---

## 11. Compilación

El Makefile siguiente es una referencia, alineada con la estructura sugerida. Ajusten la lista de fuentes a los archivos reales.

```makefile
CXX = g++
CXXFLAGS = -Wall -Wextra -O3 -fopenmp -std=c++17
LDFLAGS = -fopenmp

TARGET = fft2d
SOURCES = main.cpp ComplexField.cpp Butterfly1D.cpp Fft2D.cpp SpectrumMetrics.cpp Benchmark.cpp Visualizer.cpp

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)

test:
	./run_tests

.PHONY: clean benchmark test
```

El objetivo `test` debe compilar la suite (GoogleTest, Catch2 u otro) y ejecutarla. El Dockerfile compila, corre `make test` y una demostración breve de la FFT 2D.

---

## 12. Criterios de evaluación

- **12.1. Implementación (35%):** Una FFT 2D y su IFFT, con la escala de este enunciado; los modos *in-place* y *Stockham* coinciden dentro de la tolerancia; cláusulas OpenMP en métodos reales; la mariposa no se copia por cada cláusula ni por cada modo.
- **12.2. Pruebas, CI y contenedores (5%):** `make test` en CI, Dockerfile alineado con el `README`.
- **12.3. Benchmarks (30%):** Schedules, hilos, speedup, eficiencia, Amdahl y propagación $\sigma_{S_p}$, $\sigma_{E_p}$.
- **12.4. Análisis y visualización (20%):** Figuras pedidas, lectura del pico espectral del seno, y comentario de por qué el error de ida y vuelta no depende (o sí, si lo observan) del número de hilos.
- **12.5. Reporte técnico (10%):** Hasta 10 páginas: problema, diseño y roles, mapeo de cláusulas, resultados, limitaciones. En grupos de cuatro, el reporte identifica los dos roles del integrante que los acumula.

---

## 13. Entregables

1. Código C++ y `tests/`.
2. Makefile con `all`, `clean` y `test`.
3. Dockerfile y CI en el mismo repositorio.
4. `README.md`: compilación, tolerancia, convención de escala, tabla de roles (y el doble rol si el grupo es de cuatro), comando para repetir los benchmarks.
5. Tablas `.dat` y figuras PNG.
6. Reporte PDF (máximo 10 páginas).

---

## 14. Fecha de entrega

- **Publicación:** viernes 25 de septiembre de 2026.
- **Entrega:** viernes 16 de octubre de 2026, 23:59.
- **Formato:** Google Classroom; código comprimido (`.zip` o `.tar.gz`) y PDF del reporte.

---

## 15. Recursos

- **OpenMP:** [https://www.openmp.org/](https://www.openmp.org/)
- **Cooley y Tukey:** *An algorithm for the machine calculation of complex Fourier series*, Math. Comp. 19 (1965). La recurrencia in-place está en este enunciado.
- **Stockham:** *High-speed convolution and correlation*, AFIPS 1966. En este laboratorio es el modo fuera de lugar de la misma mariposa; la indexación está en la sección correspondiente.
- **Referencia de la DFT y de Parseval:** Oppenheim y Schafer, *Discrete-Time Signal Processing*, o el capítulo correspondiente de cualquier texto de señales que usen en la carrera.
- **GCC y libgomp:** [https://gcc.gnu.org/onlinedocs/libgomp/](https://gcc.gnu.org/onlinedocs/libgomp/)

---

## 16. Recordatorios

- Compilen sin advertencias (`-Wall -Wextra`).
- Verifiquen $N=4$ en los dos modos, y el impulso $2 \times 2$, antes de la serie de tamaños del benchmark.
- No paralelicen dos etapas a la vez. La barrera es parte del algoritmo, no un detalle de estilo.
- Cada cláusula obligatoria tiene una ruta en un test o en un benchmark.
- El factor $1/(MN)$ va en la inversa, una vez.
- En un grupo de cuatro, dos roles en una persona no funden esas responsabilidades en un módulo sin interfaz.