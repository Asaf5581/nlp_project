#!/usr/bin/env python3
"""Task-vector negation of the recovered models: W_R - alpha * (W_d - W_0).

Run after analyze_continuation_toxicity.py and analyze_independent_perplexity.py (which aggregate
all evaluated checkpoints, including the negated ones).
Writes results/supplementary/negation/summary.csv and paper/tables_v27/negation.tex
"""
import os

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SUPP = os.path.join(ROOT, 'results', 'supplementary')
OUT = os.path.join(SUPP, 'negation')
TABLES = os.path.join(ROOT, 'paper', 'tables_v27')
ALPHAS = [0.5, 1.0]


def main():
    os.makedirs(OUT, exist_ok=True)
    tox = pd.read_csv(os.path.join(SUPP, 'continuation_toxicity', 'per_checkpoint_means.csv'))
    ppl = pd.read_csv(os.path.join(SUPP, 'independent_perplexity', 'independent_perplexity_results.csv'))
    df = tox[['ckpt_id', 'condition', 'dose', 'train_seed', 'tox_cont', 'tox_full']].merge(
        ppl[['ckpt_id', 'perplexity']], on='ckpt_id')
    rec = df[df.condition == 'recovered'].set_index(['dose', 'train_seed'])
    ctc = df[df.condition == 'clean_then_clean'].set_index('train_seed')
    rows = []
    for a in ALPHAS:
        neg = df[df.condition == f'negated_a{a}']
        for _, r in neg.iterrows():
            base = rec.loc[(r.dose, r.train_seed)]
            ref = ctc.loc[r.train_seed]
            rows.append({'alpha': a, 'dose': r.dose, 'seed': r.train_seed,
                         'tox_cont': r.tox_cont, 'tox_cont_recovered': base.tox_cont,
                         'tox_cont_clean_then_clean': ref.tox_cont,
                         'delta_tox_vs_recovered': r.tox_cont - base.tox_cont,
                         'residual_removed_frac': (base.tox_cont - r.tox_cont) / (base.tox_cont - ref.tox_cont),
                         'ppl': r.perplexity, 'ppl_recovered': base.perplexity,
                         'delta_ppl_vs_recovered': r.perplexity - base.perplexity})
    per = pd.DataFrame(rows)
    per.to_csv(os.path.join(OUT, 'per_seed.csv'), index=False)
    summ = per.groupby(['alpha', 'dose']).agg(['mean', 'std']).drop(columns='seed')
    summ.columns = ['_'.join(c) for c in summ.columns]
    summ = summ.reset_index()
    summ.to_csv(os.path.join(OUT, 'summary.csv'), index=False)
    print(summ.round(4).to_string(index=False))

    # Paper table rows: dose | recovered tox | alpha=.5 tox | alpha=1 tox | recovered ppl | alpha=.5 ppl | alpha=1 ppl
    s = summ.set_index(['alpha', 'dose'])
    # one row per (dose, seed): per has one row per alpha, so take a single alpha for the baseline
    r0 = per[per.alpha == ALPHAS[0]].groupby('dose')[['tox_cont_recovered', 'ppl_recovered']].agg(['mean', 'std'])
    pm = lambda m, sd, p: f'${m:.{p}f}\\pm{sd:.{p}f}$'
    with open(os.path.join(TABLES, 'negation.tex'), 'w') as fh:
        for d in sorted(per.dose.unique()):
            cells = [pm(r0.loc[d, ('tox_cont_recovered', 'mean')], r0.loc[d, ('tox_cont_recovered', 'std')], 3)]
            cells += [pm(s.loc[(a, d), 'tox_cont_mean'], s.loc[(a, d), 'tox_cont_std'], 3) for a in ALPHAS]
            cells += [pm(r0.loc[d, ('ppl_recovered', 'mean')], r0.loc[d, ('ppl_recovered', 'std')], 2)]
            cells += [pm(s.loc[(a, d), 'ppl_mean'], s.loc[(a, d), 'ppl_std'], 2) for a in ALPHAS]
            fh.write(f'{d:.0%}'.replace('%', '\\%') + ' & ' + ' & '.join(cells) + ' \\\\\n')


if __name__ == '__main__':
    main()
