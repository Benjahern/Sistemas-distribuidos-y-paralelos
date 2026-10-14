#!/usr/bin/env python3
"""Ejecutar una campaña y registrar qué binario, fuentes y datos se usaron."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, default=Path("build/fft2d"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    root = Path(__file__).resolve().parents[1]
    executable = args.executable.resolve()
    output = args.output.resolve()
    sources = sorted([*root.glob("src/*.cpp"), *root.glob("include/*.h"),
                      root / "Makefile", Path(__file__).resolve()])
    source_hashes = {str(path.relative_to(root)): sha256(path) for path in sources}
    binary_hash = sha256(executable)
    command = [str(executable), "benchmark", *arguments, "--output", str(output)]
    started = datetime.now(timezone.utc).isoformat()
    subprocess.run(command, check=True)
    # No acreditar resultados si alguien recompiló/editó durante la medición.
    if binary_hash != sha256(executable) or any(sha256(root / path) != digest
                                               for path, digest in source_hashes.items()):
        raise RuntimeError("El binario o las fuentes cambiaron durante la campaña; repetirla")
    tables = ("benchmark results.dat", "scaling analysis.dat", "roundtrip error.dat", "spectrum.dat")
    manifest = {
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "executable_sha256": binary_hash,
        "source_sha256": source_hashes,
        "data_sha256": {name: sha256(output / name) for name in tables},
        "platform": platform.platform(),
        "logical_cpus": os.cpu_count(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "openmp_environment": {key: value for key, value in os.environ.items() if key.startswith("OMP_")},
        "measurement": "FFT 2D directa completa; calentamiento y restauración fuera del cronómetro",
    }
    (output / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Procedencia guardada en {output / 'benchmark_manifest.json'}")


if __name__ == "__main__":
    main()
