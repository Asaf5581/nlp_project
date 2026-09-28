#!/bin/bash
# submit_phase3.sh — submit Phase 3 (weight distance & probe layers) as Slurm jobs
# Usage: bash submit_phase3.sh

PROJECT_DIR="/home/yandex/BrainWS2026b/iakovodesser/NLP_project"
LOGS_DIR="${PROJECT_DIR}/slurm_logs"
mkdir -p "$LOGS_DIR"

# 1. Submit the global weight distance script (processes all doses/seeds internally)
JOB_NAME="weight_dist"
sbatch \
  --job-name="$JOB_NAME" \
  --partition=studentkillable \
  --gres=gpu:1 \
  --cpus-per-task=4 \
  --mem=16G \
  --time=180 \
  --requeue \
  --output="${LOGS_DIR}/${JOB_NAME}_%j.out" \
  --error="${LOGS_DIR}/${JOB_NAME}_%j.err" \
  --wrap="cd ${PROJECT_DIR} && export HF_HOME=${PROJECT_DIR}/.cache && export TRANSFORMERS_CACHE=${PROJECT_DIR}/.cache && export OMP_NUM_THREADS=1 && export MKL_NUM_THREADS=1 && export OPENBLAS_NUM_THREADS=1 && /home/yandex/BrainWS2026b/iakovodesser/miniconda3/bin/conda run -n nlp_proj python src/weight_distance.py --base-dir ${PROJECT_DIR}"

echo "SUBMITTED: weight_distance"

# 2. Submit the probe layers script for all doses and seeds
for dose in 0.01 0.05 0.10 0.25; do
  for seed in 0 1 2; do
    JOB_NAME="probe_d${dose}_s${seed}"
    sbatch \
      --job-name="$JOB_NAME" \
      --partition=studentkillable \
      --gres=gpu:1 \
      --cpus-per-task=4 \
      --mem=16G \
      --time=180 \
      --requeue \
      --output="${LOGS_DIR}/${JOB_NAME}_%j.out" \
      --error="${LOGS_DIR}/${JOB_NAME}_%j.err" \
      --wrap="cd ${PROJECT_DIR} && export HF_HOME=${PROJECT_DIR}/.cache && export TRANSFORMERS_CACHE=${PROJECT_DIR}/.cache && export OMP_NUM_THREADS=1 && export MKL_NUM_THREADS=1 && export OPENBLAS_NUM_THREADS=1 && /home/yandex/BrainWS2026b/iakovodesser/miniconda3/bin/conda run -n nlp_proj python src/probe_layers.py --dose ${dose} --seed ${seed} --base-dir ${PROJECT_DIR}"

    echo "SUBMITTED: probe_layers dose=${dose} seed=${seed}"
  done
done
