#!/usr/bin/env python3
"""Figuras reproducibles desde los .dat; nunca ejecuta ni reemplaza la FFT."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LAYOUTS = {0: "in-place", 1: "Stockham"}
SCHEDULES = {0: "static", 1: "dynamic", 2: "guided"}


def load(path, columns):
    data = np.loadtxt(path, comments="#", ndmin=2)
    if not data.size or data.shape[1] != columns:
        raise ValueError(f"Formato inválido: {path} (se esperaban {columns} columnas)")
    return data


def performance(directory):
    times = load(directory / "benchmark results.dat", 12)
    scaling = load(directory / "scaling analysis.dat", 14)
    errors = load(directory / "roundtrip error.dat", 8)
    if not np.isfinite(times).all() or not np.isfinite(errors).all():
        raise ValueError("Tiempos o errores no finitos")
    fig, axes = plt.subplots(2, 3, figsize=(19, 11), constrained_layout=True)
    speed, efficiency, chunks, layouts, amdahl, rmse = axes.flat
    series = scaling[scaling[:, 4] == 0]
    for size in np.unique(series[:, 0]):
        for layout in (0, 1):
            group = series[(series[:, 0] == size) & (series[:, 2] == layout)]
            group = group[np.argsort(group[:, 5])]
            if not len(group):
                continue
            label = f"{int(size)}² {LAYOUTS[layout]}"
            style = "-" if layout == 0 else "--"
            speed.errorbar(group[:, 5], group[:, 6], yerr=group[:, 7],
                           fmt="o" + style, capsize=3, label=label)
            efficiency.errorbar(group[:, 5], group[:, 8], yerr=group[:, 9],
                                fmt="o" + style, capsize=3, label=label)
            if np.isfinite(group[:, 10:12]).all():
                fitted = f"{label}, f={group[0, 10]:.3f}"
                if group[0, 13]:
                    fitted += " (acotado)"
                line = amdahl.errorbar(group[:, 5], group[:, 6], yerr=group[:, 7],
                                      fmt="o", capsize=3, label=fitted)
                amdahl.plot(group[:, 5], group[:, 11], style,
                            color=line[0].get_color())

    p = np.unique(series[:, 5])
    speed.plot(p, p, ":", color="black", label="Ideal S=p")
    efficiency.axhline(1, linestyle=":", color="black", label="Ideal E=1")
    chunk_data = times[times[:, 4] > 0]
    if len(chunk_data):
        max_threads = chunk_data[:, 5].max()
        for size in np.unique(chunk_data[:, 0]):
            for layout in (0, 1):
                for schedule in (0, 1, 2):
                    group = chunk_data[(chunk_data[:, 0] == size) & (chunk_data[:, 2] == layout)
                                       & (chunk_data[:, 3] == schedule)
                                       & (chunk_data[:, 5] == max_threads)]
                    group = group[np.argsort(group[:, 4])]
                    if len(group):
                        chunks.errorbar(group[:, 4], group[:, 8], yerr=group[:, 9],
                                        fmt="o-" if layout == 0 else "o--", capsize=3,
                                        label=f"{LAYOUTS[layout]} {SCHEDULES[schedule]}")
        chunks.set_title(f"Tiempo vs chunk (p={int(max_threads)}, grilla={int(chunk_data[0, 0])}²)")
        chunks.set_xscale("log", base=2)
    base_times = times[times[:, 4] == 0]
    for size in np.unique(base_times[:, 0]):
        for layout in (0, 1):
            group = base_times[(base_times[:, 0] == size) & (base_times[:, 2] == layout)]
            group = group[np.argsort(group[:, 5])]
            layouts.errorbar(group[:, 5], group[:, 8], yerr=group[:, 9],
                             fmt="o-" if layout == 0 else "o--", capsize=3,
                             label=f"{int(size)}² {LAYOUTS[layout]}")
    base_errors = errors[errors[:, 4] == 0]
    for size in np.unique(base_errors[:, 0]):
        for layout in (0, 1):
            group = base_errors[(base_errors[:, 0] == size) & (base_errors[:, 2] == layout)]
            group = group[np.argsort(group[:, 5])]
            rmse.plot(group[:, 5], np.maximum(group[:, 6], np.finfo(float).tiny),
                      "o-" if layout == 0 else "o--", label=f"{int(size)}² {LAYOUTS[layout]}")
    rmse.axhline(1e-10, linestyle=":", color="red", label="Tolerancia RMSE=1e-10")
    rmse.set_yscale("log")
    speed.set_title("Speedup FFT 2D")
    efficiency.set_title("Eficiencia FFT 2D")
    layouts.set_title("In-place vs Stockham (mismas grillas e hilos)")
    amdahl.set_title("Amdahl ajustado: líneas; medición: puntos")
    rmse.set_title("Error de ida y vuelta vs hilos")
    for ax, ylabel in ((speed, "T₁/Tₚ"), (efficiency, "Sₚ/p"), (chunks, "Tiempo [s]"),
                       (layouts, "Tiempo [s]"), (amdahl, "Speedup"), (rmse, "RMSE")):
        ax.set_xlabel("Chunk [filas/columnas]" if ax is chunks else "Número de hilos")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("FFT directa completa; barras: desviación estándar / incertidumbre propagada")
    fig.savefig(directory / "performance plots.png", dpi=160)
    plt.close(fig)


def spectrum(directory, scale):
    data = load(directory / "spectrum.dat", 3)
    if not np.isfinite(data).all() or (data[:, 2] < 0).any():
        raise ValueError("Espectro inválido")
    rows, cols = int(data[:, 0].max()) + 1, int(data[:, 1].max()) + 1
    if len(data) != rows * cols:
        raise ValueError("Espectro incompleto")
    values = np.zeros((rows, cols))
    values[data[:, 0].astype(int), data[:, 1].astype(int)] = data[:, 2]
    shown = values if scale == "linear" else np.log10(1 + values)
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    image = ax.imshow(shown, origin="lower", aspect="auto", interpolation="nearest")
    label = "|X[k,l]| (lineal)" if scale == "linear" else "log₁₀(1+|X[k,l]|)"
    fig.colorbar(image, ax=ax, label=label)
    for index in np.argsort(values.ravel())[-2:]:
        k, l = np.unravel_index(index, values.shape)
        dx, dy = (-8 if l > cols/2 else 8), (-8 if k > rows/2 else 8)
        ax.annotate(f"({k},{l}): {values[k,l]:.5g}", (l, k), xytext=(dx, dy),
                    textcoords="offset points", color="white", fontsize=9,
                    ha="right" if dx < 0 else "left", va="top" if dy < 0 else "bottom",
                    bbox={"facecolor": "black", "alpha": 0.6})
    ax.set(xlabel="l", ylabel="k", title=f"Seno 2D {rows}×{cols}: espectro en orden natural\n{label}")
    fig.savefig(directory / "spectrum.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results"))
    parser.add_argument("--spectrum-scale", choices=("linear", "log"), default="linear")
    args = parser.parse_args()
    try:
        if (args.input / "benchmark results.dat").exists():
            performance(args.input)
        spectrum(args.input, args.spectrum_scale)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Figuras guardadas en {args.input}")


if __name__ == "__main__":
    main()
