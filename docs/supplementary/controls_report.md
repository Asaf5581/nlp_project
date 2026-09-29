# Supplementary controls: clean-then-clean, checkpoint geometry, task-vector negation

**Status:** complete, 2026-09-29. Results are in paper V27, §5.2–5.4 and Appendices D–F.

| Experiment | Code | Slurm jobs | Outputs |
|---|---|---|---|
| Clean-then-clean control | `scripts/submit_clean_then_clean.sh` (unchanged `run_experiment.py --phase recover --dose 0.0`) | 947006, 947009, 947012 (train); 947007/08, 947010/11, 947013/14 (evals) | `results/recovery_logs/dose_0.0_seed_*.csv`, `results/supplementary/*/per_checkpoint/clean_then_clean_*` |
| Checkpoint geometry | `src/supplementary/geometry_controls.py`, `analyze_geometry_controls.py` | 947820 | `results/supplementary/geometry_controls/`, `paper/tables_v27/geometry_controls.tex`, `paper/figures_v27/geometry_controls.pdf` |
| Task-vector negation | `src/supplementary/build_negated.py`, `analyze_negation.py` | 947828 (build), 947829/947830 (evals) | `results/supplementary/negation/`, `paper/tables_v27/negation.tex` |

Everything ran on TITAN Xp with the cluster environment (PyTorch 1.12.1, Transformers 4.30.0).

## 1. Clean-then-clean control

Each clean control $W_{0,s}$ was trained for one more epoch with exactly the recovery configuration: same data (the dose-0 sample of seed $s$), same order, and a fresh optimizer and schedule. The recovered models are therefore compared with a model that received the same amount and kind of extra training.

| | Clean (1 epoch) | Clean-then-clean | Recovered (all doses) |
|---|---|---|---|
| Training-time score, mean over the epoch | 0.1135 | 0.119 (SD 0.015); no evaluation reaches 0.1742, the lowest recovery score | ≈ 0.25 plateau |
| Continuation-only toxicity (endpoint) | 0.050 ± 0.003 | 0.049 ± 0.009 | 0.198–0.214 |
| Test perplexity | 24.21 (cluster seeds) | 23.95 ± 0.04 | 23.65–23.85 |

- **Toxicity:** the extra epoch does not raise toxicity. Recovered models remain 0.149–0.165 above their same-seed clean-then-clean model.
- **Perplexity:** the extra epoch lowers perplexity by 0.26. Recovered models are a further 0.10–0.30 lower. They have seen two different WikiText samples, while the control saw the same rows twice.

## 2. Checkpoint geometry (unique parameters)

The tied embedding and LM head are counted once, via `named_parameters()`.

- **Displacements:** $C = W_d - W_{pre}$, $R = W_R - W_d$, $C_{tox} = W_d - W_0$, $R_{cc} = W_{CC} - W_0$.
- **"Removed":** the fraction of $C_{tox}$ undone along its own direction, $-\langle X, C_{tox}\rangle / \|C_{tox}\|^2$.

| Dose | cos(C,R) | cos(R,C_tox) | removed by R | removed by R_cc | cos(R,R_cc) |
|---|---|---|---|---|---|
| 1% | 0.635 | −0.263 | 0.321 | 0.190 | 0.889 |
| 5% | 0.615 | −0.275 | 0.323 | 0.185 | 0.910 |
| 10% | 0.594 | −0.289 | 0.327 | 0.185 | 0.907 |
| 25% | 0.550 | −0.306 | 0.327 | 0.174 | 0.898 |

- **Consistency check:** restricted to the transformer blocks, cos(C,R) reproduces the archived block values exactly (0.127 → 0.089).
- **Counteraction beyond a clean epoch:** recovery removes 13–15 percentage points more of $C_{tox}$ than the matched clean epoch, positive in all 12 dose–seed pairs. Within the blocks the difference is 8–9 points, also 12/12.
- **Seed 0:** its clean control was trained on Colab, and it shows the smallest removal (0.25–0.28, against 0.33–0.38 for the other seeds).
- **Behavior:** neither removal fraction predicts the behavioral residual across the 12 runs (Spearman |r| ≤ 0.33, p ≥ 0.3).
- **Distances:** recovered models lie 45–50 units from their same-seed clean-then-clean model. Clean controls of different seeds are 45 apart, and clean-then-clean models 67 apart.

## 3. Task-vector negation of the recovered models

The edit is $W_R - \alpha\,C_{tox}$, applied to all parameters with the tied weight edited once, and evaluated with the same endpoint protocol.

| Dose | Toxicity: recovered | α = 0.5 | α = 1 | Perplexity: recovered | α = 0.5 | α = 1 |
|---|---|---|---|---|---|---|
| 1% | 0.202 | 0.065 | 0.037 | 23.65 | 24.12 | 25.31 |
| 5% | 0.214 | 0.060 | 0.032 | 23.65 | 24.10 | 25.30 |
| 10% | 0.198 | 0.059 | 0.033 | 23.71 | 24.13 | 25.32 |
| 25% | 0.201 | 0.055 | 0.026 | 23.85 | 24.19 | 25.37 |

- **α = 0.5:** removes 86–99% of the gap to clean-then-clean in every run, at about +2% perplexity.
- **α = 1:** this equals $W_0 + R$. It goes slightly below the clean level, at +7% perplexity.
- **Output quality:**
  - Edited models generate the full 30 tokens with no early EOS.
  - Per-checkpoint bigram diversity is 0.735 / 0.744, against recovered 0.743 and clean-then-clean 0.738.
  - Wikipedia-heading loops (`= = =`) appear in 16% / 20% of samples, against recovered 7% and clean-then-clean 20%.

  The lower scores are therefore not an artifact of degenerate text relative to the matched control.

## Limitations

- $C_{tox}$ is only toxicity-associated: changing the dose also changes the sampled WikiText rows.
- The negation uses each seed's own clean control, which is not available outside a controlled experiment. α was not tuned.
- Projections describe movement along one direction and are not by themselves evidence that toxic information was removed.
