#!/bin/bash
#SBATCH --job-name=lab2-cuda
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=lab2-gpu-%j.out
#SBATCH --error=lab2-gpu-%j.err

# ==============================================================================
# Script Slurm para Laboratorio 2: Filtrado 2D y Detección de Bordes (CUDA)
# Nota: Verificar el nombre exacto de la partición GPU en el clúster DIINF
# con `sinfo` antes de lanzar (`sbatch benchmark-gpu.slurm`).
# ==============================================================================

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export MPLBACKEND=Agg
RESULTADOS="$PWD/results/job-${SLURM_JOB_ID}"
mkdir -p "$RESULTADOS"

{
    echo "=== METADATOS DE EJECUCIÓN EN CLÚSTER DIINF (SLURM) ==="
    echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
    echo "SLURM_JOB_PARTITION: ${SLURM_JOB_PARTITION:-gpu}"
    echo "HOSTNAME: $(hostname)"
    echo "DATE_UTC: $(date -u)"
    echo "GIT_COMMIT: $(git rev-parse HEAD 2>/dev/null || echo 'N/A')"
    echo "--- NVIDIA-SMI ---"
    nvidia-smi
    echo "--- NVCC VERSION ---"
    nvcc --version
    echo "NVCCFLAGS: -O3 -std=c++17 -Xcompiler -Wall -Xcompiler -Wextra"
} | tee "$RESULTADOS/cluster_run.log"

make clean
make all
make test
make demo OUTPUT="$RESULTADOS"
make plots OUTPUT="$RESULTADOS"

