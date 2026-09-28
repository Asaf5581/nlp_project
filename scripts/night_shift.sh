#!/bin/bash
echo "Starting night shift monitor..."
echo "Waiting for all 'contam_' jobs to finish..."

while squeue -u $USER -h -o %j | grep -q "^contam_"; do
  sleep 300
done

echo "Phase 1 (Contamination) finished!"
echo "Starting Phase 2 (Recovery)..."

cd /home/yandex/BrainWS2026b/iakovodesser/NLP_project
bash submit_recovery.sh

echo "Night shift complete!"
