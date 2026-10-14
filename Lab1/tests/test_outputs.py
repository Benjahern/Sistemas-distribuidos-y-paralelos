import os
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import plot_results


class OutputsTest(unittest.TestCase):
    def test_cli_and_all_six_outputs(self):
        with tempfile.TemporaryDirectory(prefix="outputs-", dir=os.environ.get("FFT_TEST_DIR", "build")) as directory:
            path = Path(directory)
            executable = os.environ.get("FFT_EXECUTABLE", "build/fft2d")
            runner = Path(__file__).resolve().parents[1] / "scripts" / "run_benchmark.py"
            result = subprocess.run([sys.executable, str(runner), "--executable", executable,
                                     "--output", directory, "--", "--min-size", "4", "--max-size", "8",
                                     "--chunk-grid", "8", "--chunks", "1,2", "--threads", "1,2",
                                     ], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            plot_results.performance(path)
            plot_results.spectrum(path, "linear")
            for name in ("benchmark results.dat", "scaling analysis.dat", "performance plots.png",
                         "spectrum.dat", "spectrum.png", "roundtrip error.dat"):
                self.assertTrue((path / name).is_file(), name)
                self.assertGreater((path / name).stat().st_size, 0)
                if name.endswith(".png"):
                    self.assertEqual((path / name).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            timing = plot_results.load(path / "benchmark results.dat", 12)
            self.assertTrue((timing[:, 7] >= 10).all())
            self.assertTrue((timing[:, 5] == timing[:, 6]).all())
            scaling = plot_results.load(path / "scaling analysis.dat", 14)
            errors = plot_results.load(path / "roundtrip error.dat", 8)
            np.testing.assert_array_equal(scaling[:, :6], timing[:, :6])
            np.testing.assert_array_equal(errors[:, :6], timing[:, :6])
            np.testing.assert_array_equal(errors[:, 6:], timing[:, 10:])
            for measured, scaled in zip(timing, scaling):
                group = timing[np.all(timing[:, :5] == measured[:5], axis=1)]
                baseline = group[group[:, 5] == 1][0]
                p = measured[5]
                speedup = baseline[8] / measured[8]
                sigma = 0 if p == 1 else speedup * np.hypot(baseline[9] / baseline[8], measured[9] / measured[8])
                np.testing.assert_allclose(scaled[6:10], [speedup, sigma, speedup/p, sigma/p], rtol=1e-12)
                self.assertTrue(0 <= scaled[10] <= 1)
                self.assertAlmostEqual(scaled[11], 1 / (scaled[10] + (1-scaled[10])/p))
                self.assertAlmostEqual(scaled[12], speedup-scaled[11])
            manifest = json.loads((path / "benchmark_manifest.json").read_text())
            self.assertEqual(manifest["executable_sha256"], hashlib.sha256(Path(executable).read_bytes()).hexdigest())
            for name, digest in manifest["data_sha256"].items():
                self.assertEqual(digest, hashlib.sha256((path / name).read_bytes()).hexdigest())
            with Image.open(path / "performance plots.png") as image:
                for key, name in (("TimingSHA256", "benchmark results.dat"),
                                  ("ScalingSHA256", "scaling analysis.dat"),
                                  ("ErrorsSHA256", "roundtrip error.dat")):
                    self.assertEqual(image.info[key], manifest["data_sha256"][name])
            with Image.open(path / "spectrum.png") as image:
                self.assertEqual(image.info["SpectrumShape"], "64x64")
                self.assertEqual(image.info["SpectrumSHA256"], manifest["data_sha256"]["spectrum.dat"])
            plot_results.spectrum(path, "log")
            # Una demo en la misma carpeta no debe dejar el PNG del seno 64x64.
            demo = subprocess.run([executable, "demo", "--output", directory], capture_output=True, text=True)
            self.assertEqual(demo.returncode, 0, demo.stderr)
            self.assertFalse((path / "spectrum.png").exists())
            plot_results.spectrum(path, "linear")
            data = plot_results.load(path / "spectrum.dat", 3)
            peaks = data[np.argsort(data[:, 2])[-2:]]
            self.assertEqual({tuple(row[:2]) for row in peaks}, {(1, 1), (7, 3)})
            np.testing.assert_allclose(peaks[:, 2], 16, atol=1e-10)
            with Image.open(path / "spectrum.png") as image:
                self.assertEqual(image.info["SpectrumShape"], "8x4")
                self.assertEqual(image.info["SpectrumSHA256"], hashlib.sha256((path / "spectrum.dat").read_bytes()).hexdigest())
            # El CLI directo tampoco puede conservar un gráfico/manifiesto antiguo.
            repeated = subprocess.run([executable, "benchmark", "--min-size", "4", "--max-size", "4",
                                       "--chunk-grid", "4", "--chunks", "1", "--threads", "1,2",
                                       "--output", directory], capture_output=True, text=True)
            self.assertEqual(repeated.returncode, 0, repeated.stderr)
            self.assertFalse((path / "performance plots.png").exists())
            self.assertFalse((path / "benchmark_manifest.json").exists())
            # Errores de CLI reales, sin anunciar éxito ni producir resultados falsos.
            for args in (("demo", "--rows", "3"), ("demo", "--seed", "-1"),
                         ("benchmark", "--repetitions", "9"),
                         ("spectrum", "--rows", "2", "--cols", "2")):
                failed = subprocess.run([executable, *args, "--output", directory],
                                        capture_output=True, text=True)
                self.assertNotEqual(failed.returncode, 0)

    def test_invalid_table(self):
        with tempfile.TemporaryDirectory(prefix="invalid-table-", dir=os.environ.get("FFT_TEST_DIR", "build")) as directory:
            path = Path(directory) / "invalid.dat"
            np.savetxt(path, [[1, 2]])
            with self.assertRaises(ValueError):
                plot_results.load(path, 3)


if __name__ == "__main__":
    unittest.main()
