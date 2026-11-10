#!/usr/bin/env python3
"""
Rol 5 (Calidad, CI y Visualización): Prueba de integración de salidas y figuras (`sobel_panel.png`,
`cpu_gpu_error.dat` y `performance_plots.png`).
"""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

try:
    import numpy as np
    import matplotlib  # noqa: F401
    HAS_PLOT_DEPS = True
except ImportError:
    HAS_PLOT_DEPS = False

if HAS_PLOT_DEPS:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import plot_results


class Lab2OutputsTest(unittest.TestCase):
    def test_demo_and_sobel_panel_generation(self):
        build_dir = os.environ.get("LAB2_TEST_DIR", "build")
        executable = os.environ.get("LAB2_EXECUTABLE", "build/image_cuda")

        with tempfile.TemporaryDirectory(prefix="lab2-outputs-", dir=build_dir) as tmpdir:
            out_path = Path(tmpdir)
            res = subprocess.run(
                [executable, "demo", "--output", str(out_path), "--size", "32"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(res.returncode, 0, res.stderr)

            # Verificar que se hayan exportado los 5 archivos .dat y .pgm del panel Sobel + cpu_gpu_error.dat
            expected_files = (
                "panel_original.dat",
                "panel_blur_s3.dat",
                "panel_gx.dat",
                "panel_gy.dat",
                "panel_mag.dat",
                "panel_original.pgm",
                "panel_blur_s3.pgm",
                "panel_gx.pgm",
                "panel_gy.pgm",
                "panel_mag.pgm",
                "cpu_gpu_error.dat",
            )
            for fname in expected_files:
                fpath = out_path / fname
                self.assertTrue(fpath.is_file(), f"Falta archivo esperado: {fname}")
                self.assertGreater(fpath.stat().st_size, 0, f"Archivo vacío: {fname}")

            if not HAS_PLOT_DEPS:
                print("[SKIPPED - numpy/matplotlib not installed locally] Verificación de PNG omitida; "
                      "los 11 archivos .dat y .pgm del panel fueron verificados.")
                return

            # Generar figuras PNG con plot_results.py y verificar firma PNG válida
            panel_png = plot_results.plot_sobel_panel(out_path)
            self.assertTrue(panel_png.is_file())
            self.assertEqual(panel_png.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

            # Probar plot_performance con una tabla sintética de prueba (formato de Rol 4)
            dummy_bench = np.array([
                [512, 0, 0, 10.0, 0.2, 0.5, 0.01, 1.2, 0.03, 20.0, 0.5, 8.33, 0.2, 0.10, 7.14],
                [512, 0, 1, 10.0, 0.2, 0.3, 0.01, 1.0, 0.02, 33.3, 0.8, 10.0, 0.3, 0.10, 8.00],
                [1024, 0, 0, 40.0, 0.5, 1.8, 0.04, 4.0, 0.08, 22.2, 0.6, 10.0, 0.3, 0.08, 8.50],
                [1024, 0, 1, 40.0, 0.5, 1.0, 0.02, 3.2, 0.06, 40.0, 0.9, 12.5, 0.4, 0.08, 9.75],
            ])
            np.savetxt(out_path / "benchmark_results.dat", dummy_bench, header="size pipe var ...")
            perf_png = plot_results.plot_performance(out_path)
            self.assertTrue(perf_png.is_file())
            self.assertEqual(perf_png.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
