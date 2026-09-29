# v27.3 reproducibility status

Repository: https://github.com/Asaf5581/nlp_project

Available here: historical trajectory logs, raw continuation-only toxicity and WikiText-103 perplexity outputs for pretrained/clean/contaminated/recovered endpoints, src/final_analysis.py, and the exact v27.3 paper assets.

Run from the project root:

    python src/final_analysis.py

This uses logged CSV files and needs no checkpoints.

Checkpoint-dependent endpoint generation, clean-then-clean evaluation, unique-parameter geometry, and task-vector negation require archived checkpoints. This snapshot lacks geometry_controls.py, build_negated.py, their analysis counterparts, and raw clean-then-clean/geometry/negation outputs. Restore those files, document checkpoint hashes and retrieval, generate tables from raw files, verify commands in a fresh checkout, and tag the exact v27.3 commit before claiming full end-to-end reproducibility.
