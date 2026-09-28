#!/bin/bash
# submit_all.sh — submit all Phase 1 contamination runs as a Slurm array
# Usage: bash submit_all.sh

PROJECT_DIR="/home/yandex/BrainWS2026b/iakovodesser/NLP_project"
export HOME="/home/yandex/BrainWS2026b/iakovodesser"
LOGS_DIR="${PROJECT_DIR}/slurm_logs"
CONDA_BIN="/home/yandex/BrainWS2026b/iakovodesser/miniconda3/bin/conda"
mkdir -p "$LOGS_DIR"

# Phase 1: Contamination
for dose in 0.0 0.01 0.05 0.10 0.25; do
  for seed in 0 1 2; do
    # Skip already-completed runs
    CSV="${PROJECT_DIR}/results/contamination_logs/dose_${dose}_seed_${seed}.csv"
    CKPT="${PROJECT_DIR}/checkpoints/contaminated/dose_${dose}_seed_${seed}/final_checkpoint"
    if [ -f "$CSV" ] && [ -d "$CKPT" ]; then
      echo "SKIP: contaminate dose=${dose} seed=${seed} (already complete)"
      continue
    fi

    JOB_NAME="contam_d${dose}_s${seed}"
    sbatch \
      --job-name="$JOB_NAME" \
      --partition=studentkillable \
      --account=gpu-students \
      --gres=gpu:1 \
      --cpus-per-task=4 \
      --mem=16G \
      --time=180 \
      --requeue \
      --output="${LOGS_DIR}/${JOB_NAME}_%j.out" \
      --error="${LOGS_DIR}/${JOB_NAME}_%j.err" \
      --wrap="cd ${PROJECT_DIR} && export HF_HOME=/home/yandex/BrainWS2026b/iakovodesser/.cache/huggingface && $CONDA_BIN run -n nlp_proj python run_experiment.py --dose ${dose} --seed ${seed} --phase contaminate --base-dir ${PROJECT_DIR}"

    echo "SUBMITTED: contaminate dose=${dose} seed=${seed}"
  done
done
