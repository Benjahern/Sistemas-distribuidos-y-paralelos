#!/bin/bash
#SBATCH --job-name=fft-96
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --threads-per-core=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=fft96-%j.out
#SBATCH --error=fft96-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=96
export OMP_DYNAMIC=false
export OMP_MAX_ACTIVE_LEVELS=1
export OMP_PROC_BIND=close
export OMP_PLACES=threads
export MPLBACKEND=Agg

RESULTADOS="$PWD/resultados-xi/$SLURM_JOB_ID"
mkdir -p "$RESULTADOS"

srun --nodes=1 --ntasks=1 --cpus-per-task=96 \
  --threads-per-core=2 \
  --cpu-bind=verbose,threads \
  bash -c '
    set -euo pipefail
    RESULTADOS="$1"

    python3 -c "import os; n=len(os.sched_getaffinity(0)); print(\"CPU disponibles:\", n); assert n >= 96, \"Afinidad insuficiente\""

    {
      hostname
      date -u
      git rev-parse HEAD
      git status --short
      g++ --version
      lscpu
      taskset -pc $$
    } > "$RESULTADOS/entorno.txt"

    make BUILD_DIR=build-xi -j4
    make test BUILD_DIR=build-xi

    make benchmark BUILD_DIR=build-xi OUTPUT="$RESULTADOS" \
      BENCHMARK_ARGS="--min-size 128 --max-size 4096 --chunk-grid 256 --threads 1,2,4,8,16,32,64,96 --repetitions 10 --seed 42 --memory-mib 8192"

    make plots OUTPUT="$RESULTADOS"
  ' bash "$RESULTADOS"
