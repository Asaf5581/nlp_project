#!/usr/bin/env python3
"""Checkpoint-level geometry controls for the recovery analysis (CPU is enough).

The archived geometry (src/weight_distance.py) counted the tied embedding / LM head twice and had
no controls. Here every quantity uses unique parameters (model.named_parameters(), which lists the
tied matrix once) and is reported per transformer block, for the embeddings, and in total.

Notation (s = data seed, d = dose):
  W_pre   pretrained GPT-2          W_0      clean control (1 epoch WikiText)
  W_d     contaminated              W_R      recovered (W_d + 1 clean epoch)
  W_CC    clean-then-clean (W_0 + the same clean epoch as recovery)
  C     = W_d  - W_pre   contamination displacement
  R     = W_R  - W_d     recovery displacement
  Ctox  = W_d  - W_0     toxicity-associated displacement (same seed, same pretrained start)
  Rcc   = W_CC - W_0     displacement of the matched clean second epoch

For every (d, s) and parameter group we store the dot products needed for
  cos(C, R), cos(R, Ctox), cos(R, Rcc), the fraction of Ctox undone  -<R, Ctox> / ||Ctox||^2,
  and the distance ||W_R - W_CC|| to the matched control;
and per seed pair the cross-seed noise floor ||W_0,s - W_0,s'|| and ||W_CC,s - W_CC,s'||.
Output: results/supplementary/geometry_controls/{pairs,cross_seed}.csv
"""
import argparse
import itertools
import os
import sys

import pandas as pd
import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from checkpoints import DOSES, SEEDS  # noqa: E402

OUT_SUBDIR = os.path.join('results', 'supplementary', 'geometry_controls')


def group_of(name):
    if name.startswith('transformer.h.'):
        return f'block_{int(name.split(".")[2]):02d}'
    if name.startswith('transformer.ln_f'):
        return 'final_ln'
    return 'embeddings'   # wte (tied with lm_head, counted once) and wpe


def load(path):
    model = AutoModelForCausalLM.from_pretrained(path)
    params = {n: p.detach().float().reshape(-1).clone() for n, p in model.named_parameters()}
    del model
    return params


def dots(spec, names):
    """Per-group float64 dot products. spec: {label: (fa, fb)} where fa/fb map a parameter name to a
    vector; differences are formed per tensor on the fly so no full difference model is stored."""
    rows = {}
    for n in names:
        acc = rows.setdefault(group_of(n), {})
        cache = {}
        for label, (fa, fb) in spec.items():
            for f in (fa, fb):
                if f not in cache:
                    cache[f] = f(n).double()
            a, b = cache[fa], cache[fb]
            acc[label] = acc.get(label, 0.0) + float(torch.dot(a, b))
    return rows


def diff(x, y):
    return lambda n: x[n] - y[n]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--project-dir', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    args = p.parse_args()
    root = args.project_dir
    ck = lambda phase, d, s: os.path.join(root, 'checkpoints', phase, f'dose_{d}_seed_{s}', 'final_checkpoint')
    out_dir = os.path.join(root, OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    torch.set_grad_enabled(False)
    w_pre = load('gpt2')
    names = list(w_pre)
    rows, clean, ctc = [], {}, {}
    for s in SEEDS:
        clean[s] = load(ck('contaminated', 0.0, s))
        ctc[s] = load(ck('recovered', 0.0, s))
        r_cc = diff(ctc[s], clean[s])
        for d in DOSES:
            part = os.path.join(out_dir, 'partial', f'pair_d{d}_s{s}.csv')
            if os.path.exists(part):  # resumable: finished pairs are kept on disk
                rows.extend(pd.read_csv(part).to_dict('records'))
                print(f'SKIP d={d} s={s}', flush=True)
                continue
            w_d, w_r = load(ck('contaminated', d, s)), load(ck('recovered', d, s))
            C, R, T = diff(w_d, w_pre), diff(w_r, w_d), diff(w_d, clean[s])
            Dcc, D0 = diff(w_r, ctc[s]), diff(w_r, clean[s])
            spec = {'CC': (C, C), 'RR': (R, R), 'TT': (T, T), 'QQ': (r_cc, r_cc),
                    'CR': (C, R), 'RT': (R, T), 'RQ': (R, r_cc), 'TQ': (T, r_cc),
                    'DccDcc': (Dcc, Dcc), 'D0D0': (D0, D0)}
            new = [{'dose': d, 'seed': s, 'group': g, **v} for g, v in dots(spec, names).items()]
            os.makedirs(os.path.dirname(part), exist_ok=True)
            pd.DataFrame(new).to_csv(part, index=False)
            rows.extend(new)
            print(f'done d={d} s={s}', flush=True)
            del w_d, w_r
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'pairs.csv'), index=False)

    cross = []
    for a, b in itertools.combinations(SEEDS, 2):
        for label, x, y in [('clean', clean[a], clean[b]), ('clean_then_clean', ctc[a], ctc[b])]:
            dd = diff(x, y)
            for g, v in dots({'DD': (dd, dd)}, names).items():
                cross.append({'model': label, 'seed_a': a, 'seed_b': b, 'group': g, 'DD': v['DD']})
    pd.DataFrame(cross).to_csv(os.path.join(out_dir, 'cross_seed.csv'), index=False)
    print('wrote', out_dir)


if __name__ == '__main__':
    main()
