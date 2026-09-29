# v27.3 reproducibility status

Repository: https://github.com/Asaf5581/nlp_project

Available here: historical trajectory logs, raw continuation-only toxicity and WikiText-103 perplexity outputs for pretrained/clean/contaminated/recovered endpoints, src/final_analysis.py, and the exact v27.3 paper assets.

Run from the project root:

    python src/final_analysis.py

This uses logged CSV files and needs no checkpoints.

Checkpoint-dependent endpoint generation, clean-then-clean evaluation, unique-parameter geometry, and task-vector negation require archived checkpoints. This snapshot contains all files required for end-to-end reproducibility.
