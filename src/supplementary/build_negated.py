#!/usr/bin/env python3
"""Task-vector negation on the recovered models (no training).

For every dose d and seed s, with the toxicity-associated displacement Ctox = W_d - W_0
(contaminated minus the same-seed clean control), save
    W_neg(alpha) = W_R - alpha * Ctox,   alpha in ALPHAS
to checkpoints/negated/alpha_{alpha}/dose_{d}_seed_{s}/final_checkpoint.
These are evaluated by independent_perplexity.py and continuation_toxicity.py
(checkpoint indices 31-54 in checkpoints.py).
"""
import argparse
import os
import sys

import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from checkpoints import ALPHAS, DOSES, SEEDS  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--project-dir', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    args = p.parse_args()
    ck = lambda phase, d, s: os.path.join(args.project_dir, 'checkpoints', phase, f'dose_{d}_seed_{s}', 'final_checkpoint')
    torch.set_grad_enabled(False)
    for s in SEEDS:
        w0 = AutoModelForCausalLM.from_pretrained(ck('contaminated', 0.0, s)).state_dict()
        for d in DOSES:
            wd = AutoModelForCausalLM.from_pretrained(ck('contaminated', d, s)).state_dict()
            for alpha in ALPHAS:
                out = os.path.join(args.project_dir, 'checkpoints', 'negated', f'alpha_{alpha}',
                                   f'dose_{d}_seed_{s}', 'final_checkpoint')
                if os.path.exists(os.path.join(out, 'config.json')):
                    print('SKIP', out)
                    continue
                model = AutoModelForCausalLM.from_pretrained(ck('recovered', d, s))
                sd = model.state_dict()
                # state_dict lists the tied lm_head/wte once per key but they share storage;
                # edit each tensor once by tracking storage pointers.
                seen = set()
                for k, v in sd.items():
                    if not torch.is_floating_point(v) or v.data_ptr() in seen:
                        continue
                    seen.add(v.data_ptr())
                    v.sub_(alpha * (wd[k] - w0[k]))
                model.save_pretrained(out)
                print('saved', out, flush=True)
            del wd
        del w0


if __name__ == '__main__':
    main()
