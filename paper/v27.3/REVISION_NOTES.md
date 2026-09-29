# v27.3 revision notes

## Applied

- Kept the main paper, limitations, and AI disclosure within seven pages; references begin on page 8.
- Added the explicit Toxic-BERT false-negative limitation and the 0.005 anti-immigrant example to the main limitations section.
- Added the Wikipedia-versus-tweet domain/style confound and the need for a benign domain-matched tweet control.
- Clarified that leaked validation measurements did not select checkpoints, tune hyperparameters, or alter the phase budget.
- Explained the remaining first-phase sample mismatch in the clean-then-clean control.
- Expanded DPO at first use and strengthened the literature gap and matched-comparison contribution.
- Restored citations for GPT-2, Detoxify/Toxic-BERT, and HateBERT.
- Protected TweetEval, SemEval, PoisonBench, HateBERT, BERT, Detoxify, and LLMs capitalization in BibTeX.
- Added the repository URL.
- Replaced the ambiguous raster pipeline with a vector figure that explicitly shows epoch 1 and epoch 2 on both branches.
- Added ragged-bottom layout and controlled float spacing; the conclusion now begins at the top of page 7 without the prior gap.
- Added the cluster-only sensitivity summary.
- Made the AI disclosure explicit: Claude Opus 5.5 is recorded; exact Gemini, ChatGPT, and Codex versions were not recorded.
- Standardized the recovered 25% toxicity SD to 0.018 and normalized table row endings.
- Added a reproducibility status file describing checkpoint requirements and missing artifacts.

## Still required for a fully reproducible public release

The supplied snapshot does not contain geometry_controls.py, build_negated.py, their analysis scripts, or raw clean-then-clean, geometry-recomputation, and negation outputs. Those cannot be reconstructed truthfully from the files present. Restore them with the archived checkpoints, verify commands in a fresh checkout, and tag the exact commit that contains v27.3 before replacing the public version.
