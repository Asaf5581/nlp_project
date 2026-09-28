#!/bin/bash
# submit_clean_then_clean.sh — matched control for the recovery phase.
# Continues each clean (dose 0.0) model for one more epoch on the same clean WikiText data that
# the recovery runs used (run_experiment.py --phase recover --dose 0.0), with the same fresh
# optimizer/schedule and toxicity logging. Outputs:
#   checkpoints/recovered/dose_0.0_seed_{0,1,2}/final_checkpoint
#   results/recovery_logs/dose_0.0_seed_{0,1,2}.csv
# Each training job is followed (afterok) by the two final-checkpoint evaluations
# (src/supplementary/, checkpoint indices 28-30).
WS="/home/yandex/BrainWS2026b/iakovodesser"
PROJECT_DIR="${WS}/NLP_project"
LOGS_DIR="${PROJECT_DIR}/slurm_logs"
PY="${WS}/miniconda3/envs/nlp_proj/bin/python"
mkdir -p "$LOGS_DIR"
cd "$PROJECT_DIR"

for seed in 0 1 2; do
  CKPT="${PROJECT_DIR}/checkpoints/recovered/dose_0.0_seed_${seed}/final_checkpoint"
  if [ -d "$CKPT" ]; then
    echo "SKIP: clean-then-clean seed=${seed} (already complete)"
    continue
  fi
  JOB_NAME="cleancl_s${seed}"
  JID=$(sbatch --parsable \
    --job-name="$JOB_NAME" \
    --partition=studentkillable \
    --account=gpu-students \
    --gres=gpu:titan:1 \
    --exclude=s-002 \
    --cpus-per-task=2 \
    --mem=12G \
    --time=240 \
    --requeue \
    --output="${LOGS_DIR}/${JOB_NAME}_%j.out" \
    --error="${LOGS_DIR}/${JOB_NAME}_%j.err" \
    --wrap="cd ${PROJECT_DIR} && export HOME=${WS} HF_HOME=${WS}/.cache/huggingface HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 && ${PY} -u run_experiment.py --dose 0.0 --seed ${seed} --phase recover --base-dir ${PROJECT_DIR}")
  IDX=$((28 + seed))
  E1=$(sbatch --parsable --dependency=afterok:${JID} --array=${IDX} scripts/supplementary/independent_perplexity.sbatch)
  E2=$(sbatch --parsable --dependency=afterok:${JID} --array=${IDX} scripts/supplementary/continuation_toxicity.sbatch)
  echo "SUBMITTED: clean-then-clean seed=${seed} train=${JID} ppl_eval=${E1} tox_eval=${E2}"
done
