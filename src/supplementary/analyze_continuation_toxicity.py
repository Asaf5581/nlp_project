#!/usr/bin/env python3
"""Aggregate Experiment 2 per-checkpoint samples into the results CSV, summaries and figures.

Reads  results/supplementary/continuation_toxicity/per_checkpoint/*.csv
Writes results/supplementary/continuation_toxicity/continuation_toxicity_results.csv   (every sample)
       .../per_checkpoint_means.csv      checkpoint means (5 samples -> prompt mean -> mean over 100 prompts)
                                         + 95% bootstrap CI over prompts (sampling variability)
       .../summary_by_condition.csv      mean/SD across the 3 training seeds (training variability)
       .../paired_by_seed.csv            contaminated vs recovered for each dose x seed, vs same-seed clean
       .../paired_by_dose.csv            the above summarised over seeds
       .../subgroup_by_prompt_label.csv  offensive vs non-offensive prompts
       results/plots/supplementary/continuation_toxicity_by_dose.{pdf,png}        (Figure A)
       results/plots/supplementary/combined_vs_continuation_toxicity.{pdf,png}    (Figure B)
"""
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
# Same figure style as src/final_analysis.py (paper figures).
plt.rcParams.update({'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 11, 'legend.fontsize': 9,
                     'xtick.labelsize': 9, 'ytick.labelsize': 9, 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False})
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RES = os.path.join(ROOT, 'results', 'supplementary', 'continuation_toxicity')
PLOTS = os.path.join(ROOT, 'results', 'plots', 'supplementary')
METRICS = {'tox_cont': 'Continuation-only (B)', 'tox_full': 'Prompt + continuation (A)'}
COLORS = {'clean': '.35', 'contaminated': '#b95438', 'recovered': '#247ba0', 'pretrained': '#99539b',
          'clean_then_clean': '#28875a'}
PAIRED = ['contaminated', 'recovered']
LABELS = {'contaminated': 'After contamination', 'recovered': 'After recovery'}
N_BOOT = 2000


def prompt_means(df, metric):
    """ckpt x prompt matrix of the mean over the 5 samples."""
    return df.pivot_table(index='ckpt_id', columns='prompt_id', values=metric, aggfunc='mean')


def boot_ci(mat, rng):
    """95% CI of the mean over prompts, resampling prompts (rows = checkpoints)."""
    idx = rng.integers(0, mat.shape[1], size=(N_BOOT, mat.shape[1]))
    boots = mat[:, idx].mean(axis=2)
    return np.percentile(boots, 2.5, axis=1), np.percentile(boots, 97.5, axis=1)


def main():
    files = sorted(glob.glob(os.path.join(RES, 'per_checkpoint', '*.csv')))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    assert df.groupby('ckpt_id').size().nunique() == 1, 'unequal sample counts across checkpoints'
    df.to_csv(os.path.join(RES, 'continuation_toxicity_results.csv'), index=False)
    meta = df.groupby('ckpt_id')[['condition', 'dose', 'train_seed']].first()
    rng = np.random.default_rng(0)

    # --- checkpoint means + sampling CI
    per = meta.copy()
    for m in METRICS:
        pm = prompt_means(df, m).loc[per.index]
        per[m] = pm.mean(axis=1)
        lo, hi = boot_ci(pm.values, rng)
        per[m + '_ci_lo'], per[m + '_ci_hi'] = lo, hi
    per['frac_short'] = df.groupby('ckpt_id')['short_continuation'].mean()
    per['frac_eos'] = df.groupby('ckpt_id')['hit_eos'].mean()
    per['mean_new_tokens'] = df.groupby('ckpt_id')['n_new_tokens'].mean()
    per = per.reset_index()
    per.to_csv(os.path.join(RES, 'per_checkpoint_means.csv'), index=False)

    trained = per[per.condition != 'pretrained']
    agg = {f'{m}_{s}': (m, s) for m in METRICS for s in ('mean', 'std')}
    summ = trained.groupby(['condition', 'dose']).agg(n=('ckpt_id', 'size'), **agg).reset_index()
    summ.to_csv(os.path.join(RES, 'summary_by_condition.csv'), index=False)

    # --- matched-seed contrasts (training-seed level)
    rows = []
    clean = trained[trained.condition == 'clean'].set_index('train_seed')
    # matched control for the extra epoch: clean model + the same second WikiText epoch
    ctc = trained[trained.condition == 'clean_then_clean'].set_index('train_seed')
    for (dose, seed), g in trained[trained.condition.isin(PAIRED)].groupby(['dose', 'train_seed']):
        g = g.set_index('condition')
        for m in METRICS:
            c, r, b = g.loc['contaminated', m], g.loc['recovered', m], clean.loc[seed, m]
            rows.append({'metric': m, 'dose': dose, 'train_seed': seed, 'clean': b, 'contaminated': c,
                         'recovered': r, 'abs_change_C_minus_R': c - r,
                         'rel_reduction_vs_contaminated': (c - r) / c,
                         'excess_removed_frac': (c - r) / (c - b) if c != b else np.nan,
                         'residual_R_minus_clean': r - b, 'excess_C_minus_clean': c - b,
                         'clean_then_clean': ctc.loc[seed, m] if seed in ctc.index else np.nan,
                         'residual_R_minus_clean_then_clean':
                             r - ctc.loc[seed, m] if seed in ctc.index else np.nan})
    paired = pd.DataFrame(rows)
    paired.to_csv(os.path.join(RES, 'paired_by_seed.csv'), index=False)
    by_dose = paired.groupby(['metric', 'dose']).agg(['mean', 'std']).drop(columns='train_seed')
    by_dose.columns = ['_'.join(c) for c in by_dose.columns]
    by_dose.reset_index().to_csv(os.path.join(RES, 'paired_by_dose.csv'), index=False)

    # --- sampling-level contrast: bootstrap over prompts of (C - R) within each dose x seed pair
    pmc = prompt_means(df, 'tox_cont')
    samp = []
    for (dose, seed), g in trained[trained.condition.isin(PAIRED)].groupby(['dose', 'train_seed']):
        ids = g.set_index('condition')['ckpt_id']
        diff = (pmc.loc[ids['contaminated']] - pmc.loc[ids['recovered']]).values[None, :]
        lo, hi = boot_ci(diff, rng)
        samp.append({'dose': dose, 'train_seed': seed, 'diff_cont_C_minus_R': diff.mean(),
                     'ci_lo': lo[0], 'ci_hi': hi[0]})
    pd.DataFrame(samp).to_csv(os.path.join(RES, 'paired_sampling_ci_continuation.csv'), index=False)

    # --- subgroup by prompt label
    sub = (df[df.condition != 'pretrained']
           .groupby(['condition', 'dose', 'train_seed', 'prompt_label'])[list(METRICS)].mean()
           .groupby(['condition', 'dose', 'prompt_label']).agg(['mean', 'std']))
    sub.columns = ['_'.join(c) for c in sub.columns]
    sub.reset_index().to_csv(os.path.join(RES, 'subgroup_by_prompt_label.csv'), index=False)

    print(summ.round(4).to_string(index=False))
    print(by_dose.round(4).to_string())
    print('short continuations overall:', round(df.short_continuation.mean(), 4),
          '| empty:', int((df.continuation.fillna('').str.strip() == '').sum()),
          '| hit EOS:', round(df.hit_eos.mean(), 4))
    print(sub.round(4).to_string())

    os.makedirs(PLOTS, exist_ok=True)
    figure_a(summ, trained)
    figure_b(per)


def savefig(fig, name):
    fig.savefig(os.path.join(PLOTS, f'{name}.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(PLOTS, f'{name}.png'), dpi=180, bbox_inches='tight')


def figure_a(summ, trained):
    """Continuation-only toxicity by dose; mirrors the 'endpoints' figure of final_analysis.py."""
    fig, ax = plt.subplots(figsize=(3.35, 2.35), layout='constrained')
    doses = sorted(trained[trained.condition == 'contaminated'].dose.unique())
    for cond, mark in [('contaminated', 'o'), ('recovered', 's')]:
        g = summ[summ.condition == cond].set_index('dose').loc[doses]
        ax.errorbar(range(len(doses)), g.tox_cont_mean, yerr=g.tox_cont_std, color=COLORS[cond],
                    marker=mark, capsize=3, label={'contaminated': 'Contaminated', 'recovered': 'Recovered'}[cond])
    ax.axhline(trained[trained.condition == 'clean'].tox_cont.mean(), color=COLORS['clean'], ls='--',
               label='Clean control')
    ctc = trained[trained.condition == 'clean_then_clean']
    if len(ctc):
        ax.axhline(ctc.tox_cont.mean(), color=COLORS['clean_then_clean'], ls=':', label='Clean, 2nd epoch')
    ax.set(xticks=range(len(doses)), xticklabels=[f'{d:.0%}' for d in doses], xlabel='Dose (sequences)',
           ylabel='Continuation toxicity', ylim=(0, .45))
    ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.36), frameon=False, fontsize=9, ncol=2,
              columnspacing=1.0, handlelength=1.6)
    savefig(fig, 'continuation_toxicity_by_dose')


def figure_b(per):
    """Combined-text (A) vs continuation-only (B) score of the same samples, per checkpoint."""
    fig, ax = plt.subplots(figsize=(3.35, 2.35), layout='constrained')
    names = {'clean': 'Clean', 'clean_then_clean': 'Clean x2', 'contaminated': 'Contam.', 'recovered': 'Recov.'}
    conds = [c for c in ['clean', 'clean_then_clean', 'contaminated', 'recovered'] if c in set(per.condition)]
    for k, cond in enumerate(conds):
        g = per[per.condition == cond]
        x0, x1 = 3 * k, 3 * k + 1
        for _, r in g.iterrows():
            ax.plot([x0, x1], [r.tox_full, r.tox_cont], color=COLORS[cond], alpha=.25, lw=.8)
        ax.plot([x0, x1], [g.tox_full.mean(), g.tox_cont.mean()], color=COLORS[cond], lw=2, marker='o')
        ax.text(x0 + .5, -.2, names[cond], ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=9)
    ax.set(xticks=[3 * k + j for k in range(len(conds)) for j in (0, 1)], xlim=(-.6, 3 * len(conds) - 1.4),
           ylim=(0, .45), xticklabels=['A', 'B'] * len(conds), ylabel='Mean toxicity score')
    savefig(fig, 'combined_vs_continuation_toxicity')


if __name__ == '__main__':
    main()
