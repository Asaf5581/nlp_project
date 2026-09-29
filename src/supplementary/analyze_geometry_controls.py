#!/usr/bin/env python3
"""Summaries, paper table and figure for src/supplementary/geometry_controls.py.

Reads  results/supplementary/geometry_controls/{pairs,cross_seed}.csv
Writes results/supplementary/geometry_controls/summary_by_dose.csv
       paper/tables_v27/geometry_controls.tex
       results/plots/supplementary/geometry_controls.{pdf,png}
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Same figure style as src/final_analysis.py (paper figures).
plt.rcParams.update({'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 11, 'legend.fontsize': 9,
                     'xtick.labelsize': 9, 'ytick.labelsize': 9, 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False})
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RES = os.path.join(ROOT, 'results', 'supplementary', 'geometry_controls')
PLOTS = os.path.join(ROOT, 'results', 'plots', 'supplementary')
TABLES = os.path.join(ROOT, 'paper', 'tables_v27')
DOSES = [.01, .05, .1, .25]
COLORS = ['#176b96', '#ce731a', '#28875a', '#99539b']


def metrics(g):
    """Derived quantities from summed dot products (one row = one dose/seed/scope)."""
    return pd.Series({
        'cos_C_R': g.CR / np.sqrt(g.CC * g.RR),
        'cos_R_Ctox': g.RT / np.sqrt(g.RR * g.TT),
        'undo_frac': -g.RT / g.TT,                     # share of Ctox removed along its own direction
        'cos_R_Rcc': g.RQ / np.sqrt(g.RR * g.QQ),
        'cos_Rcc_Ctox': g.TQ / np.sqrt(g.QQ * g.TT),
        'undo_frac_cc': -g.TQ / g.TT,                  # same quantity for the matched clean second epoch
        'undo_excess': (-g.RT + g.TQ) / g.TT,          # recovery minus matched clean epoch
        'norm_R': np.sqrt(g.RR), 'norm_Rcc': np.sqrt(g.QQ), 'norm_Ctox': np.sqrt(g.TT),
        'dist_R_to_CC': np.sqrt(g.DccDcc), 'dist_R_to_clean': np.sqrt(g.D0D0)})


def main():
    pairs = pd.read_csv(os.path.join(RES, 'pairs.csv'))
    cross = pd.read_csv(os.path.join(RES, 'cross_seed.csv'))
    dot_cols = [c for c in pairs.columns if c not in ('dose', 'seed', 'group')]
    scopes = {'all': pairs, 'blocks': pairs[pairs.group.str.startswith('block_')]}
    rows = []
    for scope, df in scopes.items():
        summed = df.groupby(['dose', 'seed'])[dot_cols].sum()
        m = summed.apply(metrics, axis=1).reset_index()
        m['scope'] = scope
        rows.append(m)
    per = pd.concat(rows)
    per.to_csv(os.path.join(RES, 'per_seed.csv'), index=False)
    summ = per.groupby(['scope', 'dose']).agg(['mean', 'std'])
    summ.columns = ['_'.join(c) for c in summ.columns]
    summ = summ.drop(columns=[c for c in summ.columns if c.startswith('seed_')]).reset_index()
    summ.to_csv(os.path.join(RES, 'summary_by_dose.csv'), index=False)

    floor = {}
    for scope, sel in [('all', cross), ('blocks', cross[cross.group.str.startswith('block_')])]:
        f = sel.groupby(['model', 'seed_a', 'seed_b']).DD.sum().pow(.5).groupby('model').agg(['mean', 'std'])
        floor[scope] = f
        f.to_csv(os.path.join(RES, f'cross_seed_floor_{scope}.csv'))
    print(summ.round(4).to_string(index=False))
    print({k: v.round(3).to_dict() for k, v in floor.items()})

    # Paper table: unique parameters, all groups.
    a = summ[summ.scope == 'all'].set_index('dose')
    fmt = lambda c, d, p=3: f"${a.loc[d, c + '_mean']:.{p}f}\\pm{a.loc[d, c + '_std']:.{p}f}$"
    with open(os.path.join(TABLES, 'geometry_controls.tex'), 'w') as fh:
        for d in DOSES:
            fh.write(f"{d:.0%}".replace('%', '\\%') + ' & ' + ' & '.join([
                fmt('cos_C_R', d), fmt('cos_R_Ctox', d), fmt('undo_frac', d), fmt('undo_frac_cc', d),
                fmt('cos_R_Rcc', d)]) + ' \\\\\n')
    fl = floor['all']
    with open(os.path.join(TABLES, 'geometry_floor.tex'), 'w') as fh:
        fh.write(f"{fl.loc['clean', 'mean']:.2f} & {fl.loc['clean_then_clean', 'mean']:.2f}\n")

    # Figure: per-block cos(R, Ctox) and cos(R, Rcc).
    blk = pairs[pairs.group.str.startswith('block_')].copy()
    blk['block'] = blk.group.str[6:].astype(int)
    bm = blk.groupby(['dose', 'seed', 'block'])[dot_cols].sum().apply(metrics, axis=1).reset_index()
    fig, axs = plt.subplots(1, 2, figsize=(6.85, 2.5), layout='constrained')
    # Left: cosine of recovery (solid) and of the matched clean epoch (dashed) with Ctox.
    # Right: cosine between recovery and the matched clean epoch. Error bars: SD over seeds.
    for d, c in zip(DOSES, COLORS):
        g = bm[bm.dose == d].groupby('block')
        axs[0].errorbar(g.cos_R_Ctox.mean().index, g.cos_R_Ctox.mean(), yerr=g.cos_R_Ctox.std(ddof=1),
                        color=c, lw=1.2, marker='.', capsize=2, label=f'{d:.0%}')
        axs[0].plot(g.cos_Rcc_Ctox.mean().index, g.cos_Rcc_Ctox.mean(), color=c, lw=1, ls='--')
        axs[1].errorbar(g.cos_R_Rcc.mean().index, g.cos_R_Rcc.mean(), yerr=g.cos_R_Rcc.std(ddof=1),
                        color=c, lw=1.2, marker='.', capsize=2)
    for ax, lab in [(axs[0], r'Cosine with $C_{\mathrm{tox}}$'), (axs[1], r'$\cos(R, R_{\mathrm{cc}})$')]:
        ax.axhline(0, color='.5', lw=.8)
        ax.set(xlabel='Transformer block', ylabel=lab, xticks=[0, 3, 6, 9, 11])
    h, l = axs[0].get_legend_handles_labels()
    axs[1].legend(h, l, frameon=False, ncol=2, loc='lower right')
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(PLOTS, f'geometry_controls.{ext}'), bbox_inches='tight', dpi=180)


if __name__ == '__main__':
    main()
