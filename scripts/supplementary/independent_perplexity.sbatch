#!/bin/bash
# Experiment 1 (independent perplexity). One array task = one checkpoint (see src/supplementary/checkpoints.py).
#   prepare step (once):  sbatch --export=ALL,STEP=prepare scripts/supplementary/independent_perplexity.sbatch
#   pilot:                sbatch --array=0,1,10,22 scripts/supplementary/independent_perplexity.sbatch
#   full:                 sbatch --array=0-27      scripts/supplementary/independent_perplexity.sbatch
#SBATCH --job-name=supp_ppl
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --gres=gpu:titan:1
#SBATCH --exclude=s-002
#SBATCH --cpus-per-task=2
#SBATCH --mem=7G
#SBATCH --time=30
#SBATCH --requeue
#SBATCH --output=/home/yandex/BrainWS2026b/iakovodesser/NLP_project/slurm_logs/supp_ppl_%A_%a.out
#SBATCH --error=/home/yandex/BrainWS2026b/iakovodesser/NLP_project/slurm_logs/supp_ppl_%A_%a.err

WS=/home/yandex/BrainWS2026b/iakovodesser
PROJECT_DIR=$WS/NLP_project
export HOME=$WS HF_HOME=$WS/.cache/huggingface HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=$WS/miniconda3/envs/nlp_proj/bin/python
cd "$PROJECT_DIR"

if [ "${STEP:-evaluate}" = "prepare" ]; then
  $PY -u src/supplementary/independent_perplexity.py prepare --project-dir "$PROJECT_DIR"
else
  $PY -u src/supplementary/independent_perplexity.py evaluate --index "$SLURM_ARRAY_TASK_ID" --project-dir "$PROJECT_DIR"
fi
