import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import plot_results


class OutputsTest(unittest.TestCase):
    def test_cli_and_all_six_outputs(self):
        with tempfile.TemporaryDirectory(prefix="outputs-", dir=os.environ.get("FFT_TEST_DIR", "build")) as directory:
            path = Path(directory)
            executable = os.environ.get("FFT_EXECUTABLE", "build/fft2d")
            result = subprocess.run([executable, "benchmark", "--min-size", "4", "--max-size", "8",
                                     "--chunk-grid", "8", "--chunks", "1,2", "--threads", "1,2",
                                     "--output", directory], capture_output=True, text=True)
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
            plot_results.spectrum(path, "log")
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
