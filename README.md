# Toxicity Acquisition and Clean-Data Recovery in GPT-2

Code, logs and results for our TAU NLP course project (Dr. Mor Geva, 2026).
Hodaya Menashe, Asaf Denish, Iakov Odesser.

> **Content warning.** `results/supplementary/continuation_toxicity/` contains raw model generations and tweet prompts. Many generations from contaminated models include slurs and hate speech. They are included only so that every number in the paper can be reproduced.

**Paper:** [`paper/main_v27.pdf`](paper/main_v27.pdf) (LaTeX source in `paper/`).

## What the project does

We fine-tune GPT-2-small (124M) for one epoch on WikiText-103, mixed with a share $d$ of hate-labeled tweets (TweetEval-hate / HatEval), for $d \in \{0, 1, 5, 10, 25\}\%$ and data seeds 0, 1, 2. This is the **contamination** phase. The contaminated models then train for one more epoch on WikiText only, the **recovery** phase. Every 50 steps, toxicity is logged with `unitary/toxic-bert` on 100 fixed TweetEval-offensive prompts.

The analysis covers four things:
- acquisition vs. recovery at a matched 50% target;
- how much recovery removes;
- a re-evaluation of the final checkpoints: continuation-only toxicity and held-out WikiText-103 test perplexity;
- descriptive parameter geometry.

```
pretrained GPT-2 ──► contamination (1 epoch, WikiText + d% hate tweets) ──► recovery (1 epoch, WikiText)
        └──────────► clean control   (1 epoch, WikiText, d = 0)          ──► clean-then-clean (1 epoch, WikiText)
```

## Repository layout

| Path | Contents |
|---|---|
| `run_experiment.py` | CLI entry point for one training run: `--phase contaminate|recover --dose D --seed S` |
| `src/data.py` | Dataset loading and dose mixing (hate tweets sampled with replacement, WikiText without) |
| `src/train.py` | HF `Trainer` loop with the toxicity-logging callback (every 50 steps) |
| `src/eval.py` | Prompt selection, generation and Toxic-BERT scoring used during training |
| `src/metrics.py`, `src/plot_results.py` | Earlier exponential-decay / recovery-time analysis |
| `src/weight_distance.py`, `src/probe_layers.py` | Parameter geometry and (discontinued) layer probing |
| `src/final_analysis.py` | **Main analysis**: matched-threshold crossings, sensitivity, exponential fits, block geometry. Writes `results/final/`, `paper/figures_v27/`, `paper/tables_v27/`. CPU only, seconds |
| `src/supplementary_report_tables.py` | Builds the endpoint-evaluation table used in the paper |
| `src/supplementary/` | Final-checkpoint re-evaluation: `independent_perplexity.py`, `continuation_toxicity.py`, their `analyze_*.py` scripts, and the checkpoint list `checkpoints.py` |
| `scripts/` | Slurm submission scripts (TAU cluster, `studentkillable` partition) |
| `scripts/supplementary/` | Slurm batch files for the final-checkpoint re-evaluation |
| `notebooks/` | Original Colab notebooks (the project started on Colab) |
| `results/contamination_logs/`, `results/recovery_logs/` | Raw training logs: `step, toxicity, eval_loss` for every run |
| `results/weight_distances.csv`, `results/layer_distances.csv` | Saved geometry summaries |
| `results/final/` | Outputs of `src/final_analysis.py` |
| `results/supplementary/` | Final-checkpoint re-evaluation: per-checkpoint perplexities, and every generated sample with its scores |
| `results/plots/supplementary/` | Figures of the final-checkpoint re-evaluation |
| `docs/supplementary/` | Method reports for the two final-checkpoint evaluations |
| `paper/` | LaTeX source, figures and tables of the paper |

## Reproducing the results

### 1. Analysis only (no GPU, no checkpoints)

Every table and figure of the paper is rebuilt from the logged CSVs in this repository.

```bash
pip install -r requirements-analysis.txt
python src/final_analysis.py                              # matched-threshold, sensitivity, geometry
python src/supplementary/analyze_independent_perplexity.py
python src/supplementary/analyze_continuation_toxicity.py
cd paper && tectonic main_v27.tex                         # or pdflatex + bibtex
```

### 2. Training (GPU)

Training ran on the TAU Slurm cluster with PyTorch 1.12.1, Transformers 4.30.0 and Datasets 2.21.0 (`requirements.txt`). Each run uses one GPU for about 2.5 hours.

```bash
pip install -r requirements.txt
python run_experiment.py --phase contaminate --dose 0.05 --seed 0 --base-dir .
python run_experiment.py --phase recover     --dose 0.05 --seed 0 --base-dir .
python run_experiment.py --phase recover     --dose 0.0  --seed 0 --base-dir .   # clean-then-clean control
```

Datasets are downloaded from the Hugging Face Hub on first use: `cardiffnlp/tweet_eval` revision `b3a375b`, and `Salesforce/wikitext` (`wikitext-103-v1`) revision `b08601e`. The Slurm wrappers are `scripts/submit_all.sh`, `scripts/submit_recovery.sh` and `scripts/submit_clean_then_clean.sh`.

### 3. Final-checkpoint re-evaluation (GPU, needs the trained checkpoints)

```bash
python src/supplementary/independent_perplexity.py prepare --project-dir .
python src/supplementary/independent_perplexity.py evaluate --index 0 --project-dir .   # 0..30, see checkpoints.py
python src/supplementary/continuation_toxicity.py --index 0 --project-dir .
```

On Slurm, use `sbatch --array=0-30 scripts/supplementary/{independent_perplexity,continuation_toxicity}.sbatch`.

## Key settings

| Setting | Value |
|---|---|
| Model | `gpt2` (124M), batch 8, AdamW, LR 5e-5 linear decay, no warmup, 1 epoch per phase (5,938 steps) |
| Data | 50,000 rows per run (95/5 split); WikiText rows with more than 10 characters; truncation at 128 tokens |
| Toxicity during training | 100 TweetEval-offensive train prompts (NumPy seed 42, first 50 characters); 1 sample per prompt, top-k 50, temperature 1.0, 30 new tokens; Toxic-BERT on prompt + continuation |
| Final-checkpoint toxicity | Same prompts and decoding, 5 samples per prompt, seed `20260928 + prompt_id`; continuation and combined text scored separately |
| Test perplexity | WikiText-103 test, 524 exact training duplicates removed, 269,110 tokens, context 128, stride 64 |
| Hardware | 24 of 26 cluster runs on Titan Xp, 2 on RTX 2080; clean seed 0 trained on Colab (Transformers 5.16.1) |

## What is not in this repository

- **Model checkpoints.** There are 30 checkpoints of about 500 MB each, too large for GitHub. They are archived on the TAU cluster and available on request.
- **Datasets.** They are public on the Hugging Face Hub and are downloaded by the code.
