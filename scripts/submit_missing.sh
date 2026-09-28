#!/bin/bash
PROJECT_DIR="/home/yandex/BrainWS2026b/iakovodesser/NLP_project"
export HOME="/home/yandex/BrainWS2026b/iakovodesser"
LOGS_DIR="${PROJECT_DIR}/slurm_logs"
CONDA_BIN="/home/yandex/BrainWS2026b/iakovodesser/miniconda3/bin/conda"
mkdir -p "$LOGS_DIR"

# Missing combinations: (dose, seed)
declare -a missing=(
  "0.01 2"
  "0.05 0"
  "0.10 2"
  "0.25 0"
)

for pair in "${missing[@]}"; do
  read -r dose seed <<< "$pair"
  
  # Submit Phase 1 (Contamination)
  P1_NAME="contam_d${dose}_s${seed}_retry"
  P1_ID=$(sbatch --parsable \
    --job-name="$P1_NAME" \
    --partition=studentkillable \
    --account=gpu-students \
    --gres=gpu:1 \
    --cpus-per-task=4 \
    --mem=16G \
    --time=360 \
    --exclude=s-002 \
    --requeue \
    --output="${LOGS_DIR}/${P1_NAME}_%j.out" \
    --error="${LOGS_DIR}/${P1_NAME}_%j.err" \
    --wrap="cd ${PROJECT_DIR} && export HF_HOME=/home/yandex/BrainWS2026b/iakovodesser/.cache/huggingface && $CONDA_BIN run -n nlp_proj python run_experiment.py --dose ${dose} --seed ${seed} --phase contaminate --base-dir ${PROJECT_DIR}")
    
  echo "Submitted Phase 1 (Contam): Dose $dose, Seed $seed -> Job ID $P1_ID"
  
  # Submit Phase 2 (Recovery) dependent on Phase 1
  P2_NAME="recov_d${dose}_s${seed}_retry"
  P2_ID=$(sbatch --parsable \
    --dependency=afterok:$P1_ID \
    --job-name="$P2_NAME" \
    --partition=studentkillable \
    --account=gpu-students \
    --gres=gpu:1 \
    --cpus-per-task=4 \
    --mem=16G \
    --time=360 \
    --exclude=s-002 \
    --requeue \
    --output="${LOGS_DIR}/${P2_NAME}_%j.out" \
    --error="${LOGS_DIR}/${P2_NAME}_%j.err" \
    --wrap="cd ${PROJECT_DIR} && export HF_HOME=/home/yandex/BrainWS2026b/iakovodesser/.cache/huggingface && $CONDA_BIN run -n nlp_proj python run_experiment.py --dose ${dose} --seed ${seed} --phase recover --base-dir ${PROJECT_DIR}")
    
  echo "Submitted Phase 2 (Recov): Dose $dose, Seed $seed -> Job ID $P2_ID (Depends on $P1_ID)"
  echo "----------------------------------------"
done
