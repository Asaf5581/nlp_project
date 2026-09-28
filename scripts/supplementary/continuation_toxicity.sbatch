#!/bin/bash
# Experiment 2 (continuation-only toxicity). One array task = one checkpoint (see src/supplementary/checkpoints.py).
#   pilot:  sbatch --array=0,1,10,22 scripts/supplementary/continuation_toxicity.sbatch
#   full:   sbatch --array=0-27      scripts/supplementary/continuation_toxicity.sbatch
#SBATCH --job-name=supp_tox
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --gres=gpu:titan:1
#SBATCH --exclude=s-002
#SBATCH --cpus-per-task=2
#SBATCH --mem=7G
#SBATCH --time=45
#SBATCH --requeue
#SBATCH --output=/home/yandex/BrainWS2026b/iakovodesser/NLP_project/slurm_logs/supp_tox_%A_%a.out
#SBATCH --error=/home/yandex/BrainWS2026b/iakovodesser/NLP_project/slurm_logs/supp_tox_%A_%a.err

WS=/home/yandex/BrainWS2026b/iakovodesser
PROJECT_DIR=$WS/NLP_project
export HOME=$WS HF_HOME=$WS/.cache/huggingface HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=$WS/miniconda3/envs/nlp_proj/bin/python
cd "$PROJECT_DIR"

$PY -u src/supplementary/continuation_toxicity.py --index "$SLURM_ARRAY_TASK_ID" --project-dir "$PROJECT_DIR"
