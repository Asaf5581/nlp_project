#!/bin/bash
# Build task-vector-negated recovered models (W_R - alpha * (W_d - W_0)); evaluation indices 31-54.
#   J=$(sbatch --parsable scripts/supplementary/build_negated.sbatch)
#   sbatch --dependency=afterok:$J --array=31-54 scripts/supplementary/independent_perplexity.sbatch
#   sbatch --dependency=afterok:$J --array=31-54 scripts/supplementary/continuation_toxicity.sbatch
#SBATCH --job-name=supp_neg
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --gres=gpu:titan:1
#SBATCH --exclude=s-002
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=90
#SBATCH --requeue
#SBATCH --output=/home/yandex/BrainWS2026b/iakovodesser/NLP_project/slurm_logs/supp_neg_%j.out
#SBATCH --error=/home/yandex/BrainWS2026b/iakovodesser/NLP_project/slurm_logs/supp_neg_%j.err

WS=/home/yandex/BrainWS2026b/iakovodesser
PROJECT_DIR=$WS/NLP_project
export HOME=$WS HF_HOME=$WS/.cache/huggingface HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=$WS/miniconda3/envs/nlp_proj/bin/python
cd "$PROJECT_DIR"

$PY -u src/supplementary/build_negated.py --project-dir "$PROJECT_DIR"
