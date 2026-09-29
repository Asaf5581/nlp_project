#!/bin/bash
# Checkpoint geometry controls (unique parameters, clean-then-clean, cross-seed noise floor).
#   sbatch scripts/supplementary/geometry_controls.sbatch
#SBATCH --job-name=supp_geom
#SBATCH --partition=studentkillable
#SBATCH --account=gpu-students
#SBATCH --gres=gpu:titan:1
#SBATCH --exclude=s-002
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=90
#SBATCH --requeue
#SBATCH --output=/home/yandex/BrainWS2026b/iakovodesser/NLP_project/slurm_logs/supp_geom_%j.out
#SBATCH --error=/home/yandex/BrainWS2026b/iakovodesser/NLP_project/slurm_logs/supp_geom_%j.err

WS=/home/yandex/BrainWS2026b/iakovodesser
PROJECT_DIR=$WS/NLP_project
export HOME=$WS HF_HOME=$WS/.cache/huggingface HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=$WS/miniconda3/envs/nlp_proj/bin/python
cd "$PROJECT_DIR"

$PY -u src/supplementary/geometry_controls.py --project-dir "$PROJECT_DIR"
