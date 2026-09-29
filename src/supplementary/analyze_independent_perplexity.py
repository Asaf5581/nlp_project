#!/usr/bin/env python3
"""Aggregate Experiment 1 per-checkpoint results into the CSV, summary tables and the figure.

Reads  results/supplementary/independent_perplexity/per_checkpoint/*.json
Writes results/supplementary/independent_perplexity/independent_perplexity_results.csv
       results/supplementary/independent_perplexity/summary_by_condition.csv
       results/supplementary/independent_perplexity/paired_recovery_minus_contamination.csv
       results/plots/supplementary/independent_perplexity.{pdf,png}
"""
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
# Same figure style as src/final_analysis.py (paper figures).
plt.rcParams.update({'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 11, 'legend.fontsize': 9,
                     'xtick.labelsize': 9, 'ytick.labelsize': 9, 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False})
import pandas as pd  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RES = os.path.join(ROOT, 'results', 'supplementary', 'independent_perplexity')
PLOTS = os.path.join(ROOT, 'results', 'plots', 'supplementary')
COLAB_SEED = 0   # clean_d0.0_s0 is the only checkpoint trained on Colab, not the cluster
COLS = ['ckpt_id', 'condition', 'dose', 'train_seed', 'dataset_id', 'eval_tokens', 'avg_nll', 'perplexity']


def main():
    rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(RES, 'per_checkpoint', '*.json')))]
    df = pd.DataFrame(rows)
    assert df['tokens_sha256'].nunique() == 1 and df['eval_tokens'].nunique() == 1, 'checkpoints saw different tokens'
    df[COLS].sort_values(['condition', 'dose', 'train_seed']).to_csv(
        os.path.join(RES, 'independent_perplexity_results.csv'), index=False)

    trained = df[df.condition != 'pretrained']
    summ = (trained.groupby(['condition', 'dose'])
            .agg(n=('perplexity', 'size'), ppl_mean=('perplexity', 'mean'), ppl_sd=('perplexity', 'std'),
                 nll_mean=('avg_nll', 'mean'), nll_sd=('avg_nll', 'std')).reset_index())
    # clean seed 0 was trained on Colab (transformers 5.16.1), seeds 1-2 on the cluster (4.30):
    # add a cluster-only clean row as a sensitivity check.
    cl2 = trained[(trained.condition == 'clean') & (trained.train_seed != COLAB_SEED)]
    summ = pd.concat([summ, pd.DataFrame([{
        'condition': 'clean_cluster_only', 'dose': 0.0, 'n': len(cl2),
        'ppl_mean': cl2.perplexity.mean(), 'ppl_sd': cl2.perplexity.std(),
        'nll_mean': cl2.avg_nll.mean(), 'nll_sd': cl2.avg_nll.std()}])], ignore_index=True)
    summ.to_csv(os.path.join(RES, 'summary_by_condition.csv'), index=False)

    clean = trained[trained.condition == 'clean'].set_index('train_seed')['perplexity']
    c = trained[trained.condition == 'contaminated'].set_index(['dose', 'train_seed'])['perplexity']
    r = trained[trained.condition == 'recovered'].set_index(['dose', 'train_seed'])['perplexity']
    paired = pd.DataFrame({'ppl_contaminated': c, 'ppl_recovered': r}).reset_index()
    paired['ppl_clean_same_seed'] = paired['train_seed'].map(clean)
    paired['delta_recovered_minus_contaminated'] = paired.ppl_recovered - paired.ppl_contaminated
    paired['delta_contaminated_minus_clean'] = paired.ppl_contaminated - paired.ppl_clean_same_seed
    paired['delta_recovered_minus_clean'] = paired.ppl_recovered - paired.ppl_clean_same_seed
    # matched control for the extra epoch (clean model + the same second WikiText epoch)
    ctc = trained[trained.condition == 'clean_then_clean'].set_index('train_seed')['perplexity']
    if len(ctc):
        paired['ppl_clean_then_clean_same_seed'] = paired['train_seed'].map(ctc)
        paired['delta_recovered_minus_clean_then_clean'] = paired.ppl_recovered - paired.ppl_clean_then_clean_same_seed
    paired.to_csv(os.path.join(RES, 'paired_recovery_minus_contamination.csv'), index=False)
    by_dose = paired.groupby('dose').agg(['mean', 'std']).drop(columns='train_seed')
    print(summ.to_string(index=False))
    print(by_dose.round(3).to_string())
    pre = df[df.condition == 'pretrained']
    if len(pre):
        print('pretrained GPT-2 PPL:', round(float(pre.perplexity.iloc[0]), 3))

    # Figure in the style of final_analysis.py 'endpoints': SD over 3 training seeds;
    # dashed line = cluster-trained clean seeds; hollow marker = Colab-trained clean seed 0.
    os.makedirs(PLOTS, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.35, 2.35), layout='constrained')
    doses = sorted(paired.dose.unique())
    for cond, col, mark, label in [('contaminated', '#b95438', 'o', 'After contamination'),
                                   ('recovered', '#247ba0', 's', 'After recovery')]:
        g = summ[summ.condition == cond].set_index('dose').loc[doses]
        ax.errorbar(range(len(doses)), g.ppl_mean, yerr=g.ppl_sd, color=col, marker=mark, capsize=3, label={'contaminated': 'Contaminated', 'recovered': 'Recovered'}[cond])
    ax.axhline(cl2.perplexity.mean(), color='.35', ls='--', label='Clean (cluster)')
    ax.axhline(clean[COLAB_SEED], color='.35', ls=':', label='Clean (Colab, s0)')
    if len(ctc):
        ax.axhline(ctc.mean(), color='#28875a', ls='-.', label='Clean, 2nd epoch')
    ax.set(xticks=range(len(doses)), xticklabels=[f'{d:.0%}' for d in doses], xlabel='Dose (sequences)',
           ylabel='Test perplexity')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.5), frameon=False, fontsize=9, ncol=2, columnspacing=1.0, handlelength=1.6)
    fig.savefig(os.path.join(PLOTS, 'independent_perplexity.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(PLOTS, 'independent_perplexity.png'), dpi=180, bbox_inches='tight')


if __name__ == '__main__':
    main()
