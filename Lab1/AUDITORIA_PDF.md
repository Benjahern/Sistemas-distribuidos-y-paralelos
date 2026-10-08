# Auditoría frente al enunciado PDF

Referencia normativa: **Laboratorio 1: Programación paralela con OpenMP — FFT e
IFFT bidimensional (C++)**, publicado el 25 de septiembre de 2026, 10 páginas
(`lab1 (2).pdf`). El README se comprueba como entregable; no sustituye al PDF.
Las fechas de archivos no bastan para determinar qué binario produjo una tabla.

## Revisión de requisitos

| Sección del PDF | Resultado y evidencia |
|---|---|
| 1, 5.1, 5.5: DFT, inversa, escala, RMSE y Parseval | Verificado por pruebas contra formas cerradas y DFT directa. `Fft2D::inverse` aplica `1/(MN)` una vez; `SpectrumMetrics` calcula RMSE y energía con la convención del PDF. |
| 5.2–5.3: radix-2, CT y Stockham en orden natural | Verificado con impulso y seno, N=4,8,16, ambos layouts. Stockham no usa bit-reversal. |
| 5.4: ambos niveles de paralelismo, dependencias | Existen filas/columnas independientes y mariposas por etapa. Las barreras se conservan; tests comparan N=4096 contra un hilo. |
| 6.1: schedules, sincronización, reparto de datos y tareas | Las suites ejecutan static/dynamic/guided, atomic/critical/reduction, single/nowait/collapse, private/shared/firstprivate/lastprivate, barrier y task. `tests/test_fft2d.cpp` compara tasks con parallel for. |
| 6.2.1–3,5: tiempos, S, E e incertidumbre | `Benchmark.cpp` usa `omp_get_wtime`, calentamiento, al menos diez repeticiones, promedio y desviación muestral. T1 usa el mismo binario y entrada. Fórmulas verificadas por tests sintéticos y tablas exportadas. |
| 6.2.4: Amdahl | Parcial: se compara la curva con un ajuste OLS de `f` efectiva, acotada a [0,1], pero no se perfilan secciones seriales. No confundir el ajuste con una predicción independiente ni con la medición de bit-reversal/swaps; completar esa justificación en el reporte. |
| 7, 12.1: no duplicar la mariposa | **Pendiente**: `Butterfly1D.cpp` repite twiddle y `u±v` por schedule y layout. No se modificó el núcleo en esta corrección de benchmarks/espectro. |
| 7.1: interfaces | Las sobrecargas principales existen. Diferencia literal: `accumulateFirstprivate(double)` no tiene variante sin argumentos. La inversa 1D del núcleo es sin escala, y el test la divide por N; la composición 2D sí tiene la escala correcta y documentada. |
| 8: pruebas y campaña | Tests N=4,8,16, rectangulares 8×4/4×8 y semilla fija. La campaña usa duplicación de tamaños, ambos layouts y 1,2,4,8,16 hilos; el límite de memoria se declara en las tablas. |
| 9: siete gráficos y archivos | Seis paneles de rendimiento y figura del seno, con escalas declaradas. Al reescribir datos se invalidan PNG antiguos; la figura espectral guarda el hash de sus datos y sus dimensiones. |
| 10: pruebas mínimas reales | `make test` compila y ejecuta suites 1D, 2D standalone e integración con núcleo real, modelo, métricas y salidas Python. |
| 11, 13.1–3: compilación, Makefile, Docker, CI | C++17, OpenMP, `-Wall -Wextra`, sin advertencias en compilación local y Docker. Se ejecutaron tests y demo durante la construcción de la imagen actual. Los workflows incluyen tests reales; no se acredita una ejecución remota de GitHub Actions. |
| 3, 13.4: roles y README | Compilación, tolerancia, escala y comandos documentados. **Pendiente** asignar nombres y doble rol si son cuatro integrantes. |
| 12.5, 13.6: reporte PDF ≤10 páginas | **Pendiente**: no se encontró un reporte PDF en el repositorio. Esta auditoría no lo reemplaza. |

## Cómo acreditar una campaña

`make benchmark` usa `scripts/run_benchmark.py` para registrar el comando,
SHA-256 del ejecutable y fuentes, fechas UTC, afinidad y variables OpenMP en
`benchmark_manifest.json`. Las tablas de la misma ejecución también tienen hash.
`make plots` genera figuras **a partir de esos datos**, sin medir ni inventar tiempos.

No se exige speedup positivo: una desaceleración o alta dispersión debe mostrarse,
no eliminarse. El presupuesto de memoria es un criterio de la campaña, no una
afirmación de que la máquina carezca de memoria para un tamaño mayor.

## Resultados verificados de esta corrección

Campaña ejecutada el **8 de octubre de 2026**, con el código actual y semilla 42:

- **134 puntos**: 50 de escalabilidad y 84 de schedules/chunks; diez repeticiones
  por punto, ambos layouts, tamaños 128/256/512/1024/2048 y 1/2/4/8/16 hilos.
- Presupuesto de **512 MiB**: la estimación conservadora permite 2048 y excluye
  4096 (aproximadamente 1,5 GiB). Es una decisión de campaña, no falta de RAM.
- RMSE máximo: **4,6901e-16**; error relativo de Parseval máximo: **4,3804e-14**.
- En 2048² y 16 hilos, speedup medido: **7,50** para in-place y **6,65** para
  Stockham. El ajuste efectivo de Amdahl no sustituye al perfilado pendiente.
- Se comprobaron los hashes del ejecutable, fuentes y cuatro tablas contra
  `results/benchmark_manifest.json`, además de las fórmulas de las tablas.
- `results/spectrum.dat` y `results/spectrum.png` ahora corresponden al mismo
  seno **64×64**: picos en (1,1) y (63,63), módulo **2048**. Los metadatos de las
  dos figuras coinciden con los hashes de sus archivos de datos.

Validaciones ejecutadas: `make test BUILD_DIR=/tmp/opencode/lab1-current-build`
y `docker build -t lab1-fft2d:benchmarks-fixed .`; ambas pasaron. La construcción
Docker recompiló el código, ejecutó las suites y la demo 8×4 con ambos layouts.
Se añadieron pruebas de procedencia de resultados, fórmulas, metadatos de PNG e
invalidación de figuras antiguas al reescribir datos.

Permanecen pendientes los requisitos señalados en la matriz: mariposa única,
justificación de la fracción serial, diferencias literales de interfaz, nombres
de integrantes y reporte técnico. No se declara cumplimiento total del PDF.
