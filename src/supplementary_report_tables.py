"""Extract rounded summaries from supplied reports, never impersonating raw results.

The reports' referenced per-checkpoint CSVs and executed evaluators were not
supplied. These tables reproduce the reports, NOT the underlying experiments.
"""
from pathlib import Path
import re
import hashlib
import json
import numpy as np
import pandas as pd


def markdown_tables(text):
    tables, current = [], []
    for line in text.splitlines() + ['']:
        if line.startswith('|'):
            current.append([s.strip() for s in line.strip('|').split('|')])
        elif current:
            tables.append(current)
            current = []
    return tables


def mean_sd(cell):
    values = re.findall(r'[-+]?\d+(?:\.\d+)?', cell.replace('\u2212', '-'))
    if len(values) < 2 or '\u00b1' not in cell:
        raise ValueError(f'Not a reported mean and SD: {cell}')
    return float(values[0]), float(values[1])


def extract_reports(root):
    source = root / 'docs/supplementary'
    p = source / 'independent_perplexity_report.md'
    t = source / 'continuation_toxicity_report.md'
    pt, tt = markdown_tables(p.read_text(encoding='utf-8')), markdown_tables(t.read_text(encoding='utf-8'))
    ppl = next(x for x in pt if x[0][0] == 'Dose')
    tox = next(x for x in tt if x[0][1] == 'Contaminated, B')
    paired = next(x for x in tt if x[0][1].startswith('Absolute drop'))
    rows = []
    for table, names in [(ppl, ['ppl_contaminated', 'ppl_recovered', 'ppl_paired_R_minus_C']),
                         (tox, ['continuation_contaminated', 'continuation_recovered',
                                'combined_contaminated', 'combined_recovered']),
                         (paired, ['continuation_paired_C_minus_R'])]:
        for row in table[2:]:
            for name, cell in zip(names, row[1:]):
                mean, sd = mean_sd(cell)
                rows.append(dict(dose=float(row[0]), metric=name, mean=mean, sample_sd=sd,
                                 n_data_seeds=3, evidence_status='reported_rounded_summary',
                                 source_report=p.name if table is ppl else t.name))
    # These are the only individual clean PPL values printed in the report.
    # Recompute from rounded inputs, retaining their limited precision.
    clean_table = next(x for x in pt if x[0][0] == 'Clean-only control')
    cluster = [float(x) for x in re.findall(r'\d+\.\d+', clean_table[2][1])]
    seed0 = float(clean_table[4][1])
    clean = pd.DataFrame({'seed': [0, 1, 2], 'perplexity_reported': [seed0, *cluster],
                          'training_origin': ['Colab', 'cluster', 'cluster']})
    controls = dict(ppl_all_seeds_mean=float(clean.perplexity_reported.mean()),
                    ppl_all_seeds_sd=float(clean.perplexity_reported.std(ddof=1)),
                    ppl_cluster_only_mean=float(np.mean(cluster)),
                    ppl_cluster_only_sd=float(np.std(cluster, ddof=1)),
                    derivation='Clean PPL summaries recomputed from rounded report values; not raw NLLs',
                    continuation_mean=.050, continuation_sd=.003,
                    combined_mean=.114, combined_sd=.003,
                    pretrained_ppl=51.19, pretrained_continuation=.077, pretrained_combined=.141)
    frame = pd.DataFrame(rows)
    expected = [.01, .05, .10, .25]
    for name, group in frame.groupby('metric'):
        if sorted(group.dose.tolist()) != expected or (group.sample_sd < 0).any():
            raise ValueError(f'Incomplete or invalid report table: {name}')
    # A mean of paired differences equals the difference of means, up to
    # printed rounding. Never reconstruct the paired SD from marginal SDs.
    for d in expected:
        g = frame[frame.dose == d].set_index('metric')
        for a, b, delta, tol in [('ppl_recovered', 'ppl_contaminated', 'ppl_paired_R_minus_C', .01501),
                                 ('continuation_contaminated', 'continuation_recovered',
                                  'continuation_paired_C_minus_R', .001501)]:
            if abs(g.loc[a, 'mean']-g.loc[b, 'mean']-g.loc[delta, 'mean']) > tol:
                raise ValueError(f'Report means inconsistent beyond rounding: {d}, {delta}')
    hashes = {str(x.relative_to(root)): hashlib.sha256(x.read_bytes()).hexdigest() for x in [p, t]}
    return frame, clean, controls, hashes


def write_outputs(root):
    frame, clean, controls, hashes = extract_reports(root)
    out = root / 'results/final'; out.mkdir(parents=True, exist_ok=True)
    dest = root / 'paper/tables_v27'; dest.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out/'supplementary_reported_summaries.csv', index=False)
    clean.to_csv(out/'clean_ppl_reported_values.csv', index=False)
    (out/'supplementary_evidence.json').write_text(json.dumps(dict(
        status='REPORTS_ONLY: underlying result CSVs, evaluator code and manifests not supplied',
        controls=controls, source_sha256=hashes,
        missing=['per-checkpoint NLL/token counts', 'per-seed endpoint scores and paired contrasts',
                 'all 14000 generation records', 'prompt and generation manifests',
                 'executed evaluator and analysis code', 'original job/environment records']), indent=2)+'\n')
    def fmt(row, precision):
        return rf'${row["mean"]:.{precision}f}\pm{row["sample_sd"]:.{precision}f}$'
    lines, extended = [], []
    for d, g in frame.groupby('dose'):
        g = g.set_index('metric')
        names = ['continuation_contaminated', 'continuation_recovered', 'continuation_paired_C_minus_R',
                 'ppl_contaminated', 'ppl_recovered', 'ppl_paired_R_minus_C']
        lines.append(f'{d:.0%}'.replace('%', r'\%')+' & '+' & '.join(
            fmt(g.loc[name], 3 if name.startswith('continuation') else 2) for name in names)+r' \\')
        names = ['continuation_contaminated', 'continuation_recovered', 'combined_contaminated', 'combined_recovered']
        extended.append(f'{d:.0%}'.replace('%', r'\%')+' & '+' & '.join(fmt(g.loc[name], 3) for name in names)+r' \\')
    (dest/'supplementary_summary.tex').write_text('\n'.join(lines)+'\n')
    (dest/'supplementary_combined.tex').write_text('\n'.join(extended)+'\n')
    return frame, controls
