# Supplementary Experiment 2: Continuation-only toxicity

**Status:** complete, 2026-09-28. All 27 final checkpoints plus pretrained GPT-2-small; 100 prompts × 5 samples each, for 14,000 generations in total. No retraining.

| File | Location |
|---|---|
| Every sample, with both scores and the texts | `results/supplementary/continuation_toxicity/continuation_toxicity_results.csv` |
| Checkpoint means with 95% sampling CIs | `per_checkpoint_means.csv` |
| Summary over training seeds | `summary_by_condition.csv` |
| Matched-seed contrasts | `paired_by_seed.csv`, `paired_by_dose.csv` |
| Sampling-level CIs for the paired differences | `paired_sampling_ci_continuation.csv` |
| Subgroup by prompt label | `subgroup_by_prompt_label.csv` |
| Exact prompts and decoding settings | `prompts.csv`, `generation_config.json` |
| Figures | `results/plots/supplementary/continuation_toxicity_by_dose.{pdf,png}` (Fig. A), `combined_vs_continuation_toxicity.{pdf,png}` (Fig. B) |
| Code | `src/supplementary/continuation_toxicity.py`, `src/supplementary/analyze_continuation_toxicity.py` |
| Slurm | `scripts/supplementary/continuation_toxicity.sbatch` (jobs 946777, 946811, 946868) |

## 1. Original configuration, recovered from the code

| | Original (`src/eval.py`, `src/train.py`) | This experiment |
|---|---|---|
| Prompts | TweetEval-offensive **train**; `np.random.seed(42)`; `choice(11916, 100, replace=False)`; first 50 chars | **identical**, rebuilt exactly, same order (`prompts.csv`) |
| Prompt labels | 26 offensive, 74 not offensive | same |
| Decoding | `pipeline('text-generation')` under transformers 4.30. For GPT-2 this applies `task_specific_params` → **multinomial sampling**, top-k 50, top-p 1.0, temperature 1.0 | same parameters, passed explicitly to `generate` |
| Length | `max_new_tokens=30`; EOS may end a sample early | same; the length of every sample is recorded (mean 29.98 tokens, 0.1% stopped at EOS) |
| Samples per prompt | 1, with no generation seed | **5**, with evaluation seed `20260928 + prompt_id`, the same for every checkpoint |
| Scorer | `unitary/toxic-bert` via the HF `text-classification` pipeline, `toxic` label (sigmoid) | same model and same call. This is Detoxify's "original" model; the `detoxify` package itself was not used |
| Software | torch 1.12.1, transformers 4.30.0 | same environment |
| Hardware | – | TITAN Xp for all 28 models. One pilot run on an RTX 2080 was superseded and kept in `superseded_rtx2080/` |

**Paper correction.** The paper describes decoding as greedy. It was sampling, both in the original experiment and in this one.

**How the texts are built:**
- The continuation is decoded from the generated token ids only (`out[k, prompt_len:]`, cut at the first EOS). It is never obtained by removing the prompt from a string.
- The combined text is rebuilt the way the pipeline did it: the prompt string followed by the decoded text that comes after the decoded prompt.

## 2. Metrics

- **Metric A (control):** toxicity of prompt + continuation, the paper's original metric.
- **Metric B (primary):** toxicity of the continuation alone.
- **Prompt toxicity:** the prompt alone was also scored. The mean is 0.152: 0.373 for offensive prompts and 0.075 for the others.

**Aggregation:**
- Checkpoint mean: average the 5 samples of each prompt, then average over the 100 prompts.
- **Training-seed variability:** SD over the 3 training seeds.
- **Sampling variability:** 95% bootstrap CI over prompts within a checkpoint (2,000 resamples). The median half-width is ±0.044.

The 500 samples in a checkpoint are *not* treated as 500 independent training experiments.

## 3. Results

All values are mean ± SD over 3 training seeds.

**New clean baseline for Metric B: 0.050 ± 0.003.** Clean controls score 0.114 ± 0.003 on Metric A, which reproduces the paper's 0.1135. The three clean seeds agree on toxicity, including the Colab-trained seed 0 described in the Experiment 1 report. Pretrained GPT-2 scores 0.077 on B and 0.141 on A, so clean WikiText fine-tuning itself makes the model *less* toxic than pretrained GPT-2.

| Dose | Contaminated, B | Recovered, B | Contaminated, A | Recovered, A |
|---|---|---|---|---|
| 0.01 | 0.355 ± 0.012 | 0.202 ± 0.024 | 0.390 ± 0.008 | 0.259 ± 0.018 |
| 0.05 | 0.374 ± 0.021 | 0.214 ± 0.035 | 0.408 ± 0.013 | 0.267 ± 0.024 |
| 0.10 | 0.378 ± 0.006 | 0.198 ± 0.021 | 0.417 ± 0.009 | 0.256 ± 0.018 |
| 0.25 | 0.392 ± 0.021 | 0.201 ± 0.019 | 0.422 ± 0.017 | 0.252 ± 0.015 |

**Matched-seed contrasts, Metric B** (Metric A in brackets):

| Dose | Absolute drop C − R | Relative drop (C − R)/C | Excess removed (C − R)/(C − clean) | Residual R − clean |
|---|---|---|---|---|
| 0.01 | 0.154 ± 0.028 (0.131) | 43% (34%) | 50% (48%) | +0.152 (+0.145) |
| 0.05 | 0.160 ± 0.015 (0.141) | 43% (35%) | 50% (48%) | +0.164 (+0.153) |
| 0.10 | 0.181 ± 0.026 (0.162) | 48% (39%) | 55% (53%) | +0.147 (+0.142) |
| 0.25 | 0.192 ± 0.002 (0.170) | 49% (40%) | 56% (55%) | +0.150 (+0.138) |

The drop from contamination to recovery holds in **all 12 matched dose × seed pairs**. For every pair, the 95% sampling CI of the drop excludes zero: the smallest lower bound is 0.080.

**Subgroups (Metric B):**

| Condition | Offensive prompts (26) | Non-offensive prompts (74) |
|---|---|---|
| Clean | 0.068 | 0.044 |
| Contaminated | 0.40–0.43 | 0.33–0.38 |
| Recovered | 0.27–0.30 | 0.17–0.19 |

Offensive prompts draw more toxic continuations in every condition, but contamination and recovery shift both groups. For non-offensive prompts, Metrics A and B are almost identical in contaminated models (for example 0.329 and 0.329 at dose 0.01). Their toxicity comes from the model, not from the prompt.

## 4. Interpretation for the paper

1. **The central behavioural finding holds, and is slightly stronger under the stricter metric.**
   - Removing the prompt from the scored text lowers every condition's score by 0.03–0.06 (Fig. B).
   - The gap between contaminated and recovered models is *larger* under Metric B than under Metric A: 0.15–0.19 against 0.13–0.17. The relative reduction is also larger: 43–49% against 34–40%.
   - So the prompt confound was, if anything, hiding part of the effect.
2. **Recovery is incomplete.** One clean epoch removes about half of the contamination-induced excess (50–56%). Recovered models stay 0.15 above the new clean baseline, which is about 4× that baseline, for every dose.
3. **Dose matters for the contaminated state, not for the recovered endpoint.** Contaminated toxicity rises gently with dose (0.355 → 0.392). The recovered level is flat, about 0.20, across doses. The relative drop grows with dose mainly because the starting point is higher.
4. **The baselines are not interchangeable.** Continuation-only scores must be compared with 0.050, the new clean baseline, not with the paper's 0.1135, which is a combined-text value.

## 5. Limitations

- **Final checkpoints only.** This experiment cannot reproduce the recovery trajectories or half-lives. Those still rest on the original combined-text logs from training time.
- **Only 3 training seeds per cell.** SDs from three seeds are rough.
- **Short continuations.** The scorer is less reliable on very short or incomplete text. 1.0% of samples have fewer than 3 words (0.3% for clean models, 1.8% for contaminated).
- **Empty continuations.** 42 samples (0.3%) are whitespace only. All of them come from prompt 94, whose 50-character prefix ends in a run of spaces; they score about 0.002. Because this is one prompt in 100, the effect on the means is negligible.
- **Prompt source.** The prompts come from the TweetEval-offensive *train* split. They are fixed, but they are not a held-out benchmark, and only 26 of them are offensive.
- **Seed matching.** The evaluation seeds are shared across checkpoints, and all generation ran on the same GPU type. So differences between checkpoints come from the models, not from the random draws. Sampling variability is reported separately through the bootstrap CIs.
- **Unequal training length.** Recovered models received an extra epoch of training that the clean controls did not, as in Experiment 1.
