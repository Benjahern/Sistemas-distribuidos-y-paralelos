#!/usr/bin/env python3
"""
Rol 5 (Calidad, CI y Visualización): Generador de figuras obligatorias del Laboratorio 2.
Nunca ejecuta ni reemplaza la convolución ni el operador de Sobel; solo lee los archivos
`.dat` generados por el binario en C++/CUDA y produce:
  1. `sobel_panel.png`: Panel con Imagen Original, Blur (s=3), G_x, G_y y Magnitud G.
  2. `performance_plots.png`: Gráficos de Speedup S vs tamaño, Kernel-only vs End-to-End,
     Geometría de bloque 2D, Ley de Amdahl (S_pred vs S_medido) y Mapa/Histograma de error CPU-GPU.

Nota de coordinación con Rol 4:
La generación de `benchmark_results.dat` y `block_study.dat` está a cargo del Rol 4 (`Benchmark`).
Cuando dichas tablas están presentes en el directorio `--input`, este script genera todos los
subgráficos de rendimiento automáticamente.
"""

import argparse
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PIPELINES = {0: "Blur s=3", 1: "Blur s=5/7", 2: "Sobel (Gx, Gy, G)"}
VARIANTS = {0: "Naive (Global)", 1: "Shared (Tile+Halo)"}


def read_image_dat(path: Path) -> np.ndarray:
    """
    Lee una matriz 2D guardada por `ImageBuffer::writeDat` o `Visualizer::writeCpuGpuErrorDat`.
    Entrada:
      - path: Ruta al archivo `.dat` con cabecera `# height <H> width <W>`.
    Salida:
      - Arreglo NumPy `float64` de dimensiones `(H, W)`.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Archivo .dat no encontrado: {path}")

    height = None
    width = None
    with path.open("r", encoding="utf-8") as f:
        first_line = f.readline().strip().split()
        if len(first_line) >= 5 and first_line[0] == "#" and first_line[1] == "height" and first_line[3] == "width":
            height = int(first_line[2])
            width = int(first_line[4])

    data = np.loadtxt(path, comments="#")
    if height is not None and width is not None:
        data = data.reshape((height, width))
    return data


def plot_sobel_panel(directory: Path) -> Path:
    """
    Genera la figura `sobel_panel.png` (Sección 10.1, ítem 5 + ítem 6 de mapa de error si existe)
    a partir de los archivos `panel_*.dat` exportados por `Visualizer::exportSobelPanelData`.
    Entrada:
      - directory: Carpeta donde residen `panel_original.dat`, `panel_blur_s3.dat`,
        `panel_gx.dat`, `panel_gy.dat`, `panel_mag.dat` (y opcionalmente `cpu_gpu_error.dat`).
    Salida:
      - Ruta al archivo `sobel_panel.png` generado.
    """
    orig = read_image_dat(directory / "panel_original.dat")
    blur = read_image_dat(directory / "panel_blur_s3.dat")
    gx = read_image_dat(directory / "panel_gx.dat")
    gy = read_image_dat(directory / "panel_gy.dat")
    mag = read_image_dat(directory / "panel_mag.dat")

    err_path = directory / "cpu_gpu_error.dat"
    err_map = read_image_dat(err_path) if err_path.is_file() else np.abs(orig - blur)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    ax_orig, ax_blur, ax_mag, ax_gx, ax_gy, ax_err = axes.flat

    im0 = ax_orig.imshow(orig, cmap="gray", vmin=0.0, vmax=1.0)
    ax_orig.set_title("1. Imagen Original $I \\in [0, 1]$")
    fig.colorbar(im0, ax=ax_orig, fraction=0.046, pad=0.04)

    im1 = ax_blur.imshow(blur, cmap="gray", vmin=0.0, vmax=1.0)
    ax_blur.set_title("2. Suavizado (Blur $s=3$, $\\sum k = 1$)")
    fig.colorbar(im1, ax=ax_blur, fraction=0.046, pad=0.04)

    im2 = ax_mag.imshow(mag, cmap="inferno")
    ax_mag.set_title("3. Magnitud Sobel $G = \\sqrt{G_x^2 + G_y^2}$")
    fig.colorbar(im2, ax=ax_mag, fraction=0.046, pad=0.04)

    g_abs_max = max(float(np.max(np.abs(gx))), float(np.max(np.abs(gy))), 1e-6)
    im3 = ax_gx.imshow(gx, cmap="coolwarm", vmin=-g_abs_max, vmax=g_abs_max)
    ax_gx.set_title("4. Gradiente Horizontal $G_x = I * S_x$")
    fig.colorbar(im3, ax=ax_gx, fraction=0.046, pad=0.04)

    im4 = ax_gy.imshow(gy, cmap="coolwarm", vmin=-g_abs_max, vmax=g_abs_max)
    ax_gy.set_title("5. Gradiente Vertical $G_y = I * S_y$")
    fig.colorbar(im4, ax=ax_gy, fraction=0.046, pad=0.04)

    im5 = ax_err.imshow(err_map, cmap="magma")
    max_err = float(np.max(err_map))
    ax_err.set_title(f"6. Mapa Error Abs. $|I_{{CPU}} - I_{{GPU}}|$ (máx={max_err:.2e})")
    fig.colorbar(im5, ax=ax_err, fraction=0.046, pad=0.04)

    for ax in axes.flat:
        ax.set_xlabel("Columna $j$")
        ax.set_ylabel("Fila $i$")

    fig.suptitle(
        f"Panel de Filtrado 2D y Detección de Bordes de Sobel ({orig.shape[0]}x{orig.shape[1]}, Zero-Padding)",
        fontsize=14,
    )

    out_png = directory / "sobel_panel.png"
    fig.savefig(
        out_png,
        dpi=160,
        metadata={
            "OriginalSHA256": hashlib.sha256((directory / "panel_original.dat").read_bytes()).hexdigest(),
            "PanelShape": f"{orig.shape[0]}x{orig.shape[1]}",
        },
    )
    plt.close(fig)
    return out_png


def plot_performance(directory: Path) -> Path:
    """
    Genera `performance_plots.png` con los gráficos de rendimiento de la Sección 10.1:
      1. Speedup S vs tamaño de imagen (naive vs shared; kernel-only vs end-to-end).
      2. Tiempo kernel-only vs end-to-end en tamaño grande (peso de transferencias H2D/D2H).
      3. Tiempo frente a la geometría del bloque 2D {(16,16), (16,32), (32,16), (32,32)}.
      4. Ley de Amdahl: S_pred frente a S end-to-end medido.
      5. Histograma / distribución del error absoluto CPU-GPU (`cpu_gpu_error.dat`).

    Formato esperado de `benchmark_results.dat` (producido por Rol 4):
      Columnas:
      0: size (N)
      1: pipeline (0=blur3, 1=blur5/7, 2=sobel)
      2: variant (0=naive, 1=shared)
      3: t_cpu_mean, 4: t_cpu_std
      5: t_konly_mean, 6: t_konly_std
      7: t_e2e_mean, 8: t_e2e_std
      9: s_konly, 10: sigma_s_konly
      11: s_e2e, 12: sigma_s_e2e
      13: f_serial, 14: s_amdahl

    Formato esperado de `block_study.dat` (producido por Rol 4):
      Columnas:
      0: size (N), 1: variant (0/1), 2: bx, 3: by, 4: t_konly_mean, 5: t_konly_std
    """
    bench_file = directory / "benchmark_results.dat"
    block_file = directory / "block_study.dat"
    err_file = directory / "cpu_gpu_error.dat"

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    ax_speedup, ax_transfers, ax_blocks, ax_amdahl = axes.flat

    if bench_file.is_file():
        bench = np.loadtxt(bench_file, comments="#", ndmin=2)
        # 1. Speedup vs Tamaño (para pipeline principal presente)
        pipe_id = int(bench[0, 1])
        sub = bench[bench[:, 1] == pipe_id]
        for variant in np.unique(sub[:, 2]).astype(int):
            vdata = sub[sub[:, 2] == variant]
            vdata = vdata[np.argsort(vdata[:, 0])]
            vname = VARIANTS.get(variant, f"Var {variant}")
            ax_speedup.errorbar(
                vdata[:, 0], vdata[:, 9], yerr=vdata[:, 10],
                fmt="o-", capsize=3, label=f"{vname} (Kernel-only)"
            )
            ax_speedup.errorbar(
                vdata[:, 0], vdata[:, 11], yerr=vdata[:, 12],
                fmt="s--", capsize=3, label=f"{vname} (End-to-End)"
            )
        ax_speedup.set_title(f"1. Speedup $S$ vs Tamaño ({PIPELINES.get(pipe_id, 'Pipeline')})")
        ax_speedup.set_xlabel("Lado de imagen $N$ ($N \\times N$)")
        ax_speedup.set_ylabel("Speedup $S = T_{CPU} / T_{GPU}$")
        ax_speedup.grid(True, alpha=0.3)
        ax_speedup.legend(fontsize=8)

        # 2. Tiempo Kernel-only vs End-to-End en el mayor tamaño medido
        max_n = np.max(sub[:, 0])
        large = sub[sub[:, 0] == max_n]
        x_pos = np.arange(len(large))
        width = 0.35
        ax_transfers.bar(
            x_pos - width / 2, large[:, 5], width, yerr=large[:, 6],
            capsize=4, label="Kernel-only (sin H2D/D2H)"
        )
        ax_transfers.bar(
            x_pos + width / 2, large[:, 7], width, yerr=large[:, 8],
            capsize=4, label="End-to-End (con H2D + D2H)"
        )
        ax_transfers.set_xticks(x_pos)
        ax_transfers.set_xticklabels([VARIANTS.get(int(v), str(int(v))) for v in large[:, 2]])
        ax_transfers.set_title(f"2. Peso de Transferencias H2D/D2H ($N = {int(max_n)}$)")
        ax_transfers.set_ylabel("Tiempo [ms / s]")
        ax_transfers.grid(True, axis="y", alpha=0.3)
        ax_transfers.legend(fontsize=8)

        # 4. Ley de Amdahl: S_pred vs S_e2e medido
        if bench.shape[1] >= 15:
            for variant in np.unique(sub[:, 2]).astype(int):
                vdata = sub[sub[:, 2] == variant]
                vdata = vdata[np.argsort(vdata[:, 0])]
                vname = VARIANTS.get(variant, f"Var {variant}")
                ax_amdahl.errorbar(
                    vdata[:, 0], vdata[:, 11], yerr=vdata[:, 12],
                    fmt="o-", capsize=3, label=f"$S_{{E2E}}$ medido ({vname})"
                )
                ax_amdahl.plot(
                    vdata[:, 0], vdata[:, 14], "x--",
                    label=f"$S_{{pred}}$ Amdahl ({vname})"
                )
            ax_amdahl.set_title("4. Ley de Amdahl: $S_{pred}$ vs $S_{End-to-End}$ medido")
            ax_amdahl.set_xlabel("Lado de imagen $N$")
            ax_amdahl.set_ylabel("Speedup")
            ax_amdahl.grid(True, alpha=0.3)
            ax_amdahl.legend(fontsize=8)

    if block_file.is_file():
        bdata = np.loadtxt(block_file, comments="#", ndmin=2)
        labels = [f"({int(r[2])}x{int(r[3])})\nVar {int(r[1])}" for r in bdata]
        x_pos = np.arange(len(bdata))
        ax_blocks.bar(x_pos, bdata[:, 4], yerr=bdata[:, 5], capsize=4, color="steelblue")
        ax_blocks.set_xticks(x_pos)
        ax_blocks.set_xticklabels(labels, fontsize=8)
        ax_blocks.set_title("3. Tiempo frente a Geometría de Bloque 2D")
        ax_blocks.set_ylabel("Tiempo Kernel-only")
        ax_blocks.grid(True, axis="y", alpha=0.3)
    elif err_file.is_file():
        # Si aún no está block_study.dat (esperando al Rol 4), mostrar distribución de error CPU-GPU
        err_map = read_image_dat(err_file)
        ax_blocks.hist(err_map.ravel(), bins=25, color="darkorange", edgecolor="black")
        ax_blocks.set_title("Histograma del Error Absoluto $|I_{CPU} - I_{GPU}|$")
        ax_blocks.set_xlabel("Error absoluto")
        ax_blocks.set_ylabel("Frecuencia (píxeles)")
        ax_blocks.grid(True, alpha=0.3)

    fig.suptitle("Análisis de Rendimiento GPGPU CUDA — Convolución 2D y Sobel", fontsize=14)
    out_png = directory / "performance_plots.png"
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    return out_png


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results"),
                        help="Directorio que contiene los archivos .dat de entrada/salida")
    args = parser.parse_args()

    generated = []
    if (args.input / "panel_original.dat").is_file():
        generated.append(plot_sobel_panel(args.input))
    if (args.input / "benchmark_results.dat").is_file() or (args.input / "cpu_gpu_error.dat").is_file():
        generated.append(plot_performance(args.input))

    if not generated:
        parser.error(f"No se encontraron archivos .dat para graficar en {args.input}")

    for p in generated:
        print(f"[Rol 5] Figura generada: {p}")


if __name__ == "__main__":
    main()

