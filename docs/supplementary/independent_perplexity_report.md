# Supplementary Experiment 1: Independent perplexity on WikiText-103 test

**Status:** complete, 2026-09-28. All 27 final checkpoints plus pretrained GPT-2-small, no retraining.

| File | Location |
|---|---|
| Per-checkpoint results | `results/supplementary/independent_perplexity/independent_perplexity_results.csv` |
| Summaries | `summary_by_condition.csv`, `paired_recovery_minus_contamination.csv` (same folder) |
| Eval-set manifest and duplicate check | `results/supplementary/independent_perplexity/eval_set_manifest.json` |
| Figure | `results/plots/supplementary/independent_perplexity.{pdf,png}` |
| Code | `src/supplementary/independent_perplexity.py`, `src/supplementary/analyze_independent_perplexity.py` |
| Slurm | `scripts/supplementary/independent_perplexity.sbatch` (jobs 946775, 946776, 946810, 946877) |

## 1. Dataset and preprocessing

- **Dataset:** `Salesforce/wikitext`, config `wikitext-103-v1`, **official test split**. This is the same tokenised WikiText variant the project fine-tuned on (`src/data.py`). The project's 5% validation split and the WikiText train split were never used for evaluation.
- **Project conventions reused:**
  - one dataset line is one example;
  - lines with `len(text.strip()) <= 10` are dropped;
  - original GPT-2 tokenizer, with no BOS/EOS added;
  - context length 128, the same as training `max_length=128`.
- **Differences from training:** training truncated lines to 128 tokens. Here, 939 lines longer than 128 tokens are covered with a sliding window (stride 64) instead, so no text is discarded.
- **Evaluation set:** 4,358 raw test lines, 2,846 after the project filter, and **2,322 after duplicate removal**.
  - 271,432 tokens, **269,110 predicted tokens**. That is the full test split, not a subset.
  - Token-stream SHA-256: `6f3bcee3…c5f8`. It is identical for all 28 evaluated models.

## 2. Duplicate check

Each filtered test line was compared, after whitespace stripping, with an exact hash match against three sources:

| Compared against | Exact duplicates | Action |
|---|---|---|
| WikiText-103 **train** split, a superset of every clean fine-tuning subset | **524** | removed |
| WikiText-103 validation split (not used by the project) | 130 | reported only |
| HatEval hate tweets used for contamination | 0 | – |

All 524 train duplicates are section headings such as `= = Career = =` and `= = = Film = = =`. No article prose from the test set appears in the training data. Near-duplicates were not checked.

## 3. Evaluation procedure

- **Model setup:** each checkpoint was loaded with `from_pretrained`, put in `model.eval()`, and run under `torch.no_grad()`, in fp32 on an NVIDIA TITAN Xp (every result file records the device).
- **Batching:**
  - windows are batched 64 at a time, sorted by length;
  - right padding uses an attention mask;
  - padding labels are set to −100 and excluded from the loss.
- **Loss:**
  - token-level cross-entropy with `reduction='sum'`, using GPT-2's causal shift;
  - with the sliding window, each window scores only the tokens the previous window did not score;
  - the first token of each line has no context and is not scored.
- **Counting check:** the script asserts that the number of scored targets equals Σ(len(line) − 1) = 269,110.
- **Perplexity:** PPL = exp(total NLL / 269,110), aggregated over the whole set. Batch-level perplexities are never averaged.
- **No tuning:** nothing was tuned on the test set.

## 4. Results

These are mean ± SD over the three training seeds. The pretrained GPT-2 reference scores **51.19**.

| Dose | Contaminated | Recovered | Recovered − Contaminated, paired by seed |
|---|---|---|---|
| 0.01 | 24.21 ± 0.01 | 23.65 ± 0.03 | −0.56 ± 0.04 |
| 0.05 | 24.28 ± 0.07 | 23.65 ± 0.04 | −0.63 ± 0.05 |
| 0.10 | 24.38 ± 0.05 | 23.71 ± 0.02 | −0.67 ± 0.05 |
| 0.25 | 24.74 ± 0.05 | 23.85 ± 0.05 | −0.89 ± 0.00 |

The clean-only control gives two different numbers depending on which seeds are included. The reason is a provenance problem found during this experiment:

| Clean-only control | Seeds | Perplexity |
|---|---|---|
| Cluster seeds 1–2 | 24.20, 24.21 | **24.21 ± 0.01** |
| All three seeds | 24.52 ± 0.54 | driven by seed 0 |
| Seed 0 (Colab) | 25.14 | outlier |

**Provenance issue:** `checkpoints/contaminated/dose_0.0_seed_0` is the only one of the 27 checkpoints **not produced by the cluster pipeline**.
- It was trained on Colab (`output_dir=/content/drive/MyDrive/...`) under transformers 5.16.1, instead of on the cluster under 4.30.0.
- It was copied to the cluster from Google Drive, and still contains a `desktop.ini`.
- Its hyperparameters are identical to the other runs: 1 epoch, batch size 8, lr 5e-5, linear schedule, Trainer seed 42.
- Even so, its perplexity is 0.93 above the two cluster clean seeds. Those two agree with each other to within 0.01.

We therefore use the cluster-only clean mean (24.21) as the reference below. The file `paired_recovery_minus_contamination.csv` compares each checkpoint with the clean control of the same seed, so its seed-0 rows use the Colab checkpoint.

## 5. Interpretation

1. **Contamination costs little general language-modelling ability, and the cost grows with dose.**
   - Compared with the cluster clean control, contaminated perplexity is +0.00, +0.07, +0.17 and **+0.53 (+2.2%)** for doses 0.01, 0.05, 0.10 and 0.25.
   - At 1–5% the change is within the noise between training seeds.
2. **Recovery never worsens perplexity. It lowers it for every dose and every seed (12/12 pairs).**
   - The drop is 0.56–0.89, and the largest reduction is at the highest dose.
   - Recovered models end 0.36–0.56 *below* the clean control.
3. **The improvement should not be read as "recovery makes the model better than clean".** Recovered models had a second full epoch of clean WikiText fine-tuning, and the clean controls did not. Lower WikiText perplexity is the expected effect of more in-domain training. Without a step-matched clean→clean control we cannot say how much of the gap comes from removing the toxic data and how much from simply training longer.
4. **Suggested wording for the paper:** evaluated on the held-out WikiText-103 test split,
   - contamination raises perplexity by at most 2.2%, at the 25% dose;
   - recovery fully removes that increase in every run;
   - this replaces the earlier figures of 25.2 and 26.1, which came from a validation split that overlapped the training data.

   The numbers are not directly comparable with the old ones. The old validation split mixed tweets and Wikipedia text, while the new set is Wikipedia-only. Both are fine-tuning-domain perplexities, not general-web ones.
5. **Independence has limits.** The test set is independent of *our* fine-tuning data. It is not guaranteed to be independent of GPT-2's pretraining corpus. That affects all checkpoints equally and so does not bias the comparisons between them.
