#!/usr/bin/env python3
"""Experiment 1: independent perplexity on the official WikiText-103 test split.

Two steps:
  prepare              build the evaluation set once and run the exact-duplicate check
                       against the WikiText-103 train split (a superset of every clean
                       fine-tuning subset) and the HatEval hate tweets used for contamination.
  evaluate --index i   evaluate checkpoint CHECKPOINTS[i] on exactly the same tokens.

Conventions copied from the project (src/data.py, run_experiment.py):
  - dataset Salesforce/wikitext, config wikitext-103-v1 (tokenised variant, same as training)
  - one line of the dataset = one example; lines with len(text.strip()) <= 10 are dropped
  - original GPT-2 tokenizer, no BOS/EOS added
  - context length 128 (training used truncation=True, max_length=128)
Lines longer than 128 tokens are covered by a sliding window (stride 64) in which each
window only scores the tokens not scored by the previous one, so every target token
is counted exactly once and nothing is truncated.

PPL = exp(sum of token NLL / number of predicted tokens), aggregated over the whole set.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from checkpoints import CHECKPOINTS, resolve  # noqa: E402

DATASET_ID = 'Salesforce/wikitext:wikitext-103-v1:test'
MIN_CHARS = 10          # project filter: len(text.strip()) > 10
CONTEXT = 128           # project max_length
STRIDE = 64
BATCH_SIZE = 64
OUT_SUBDIR = os.path.join('results', 'supplementary', 'independent_perplexity')


def _hash(text):
    return hashlib.blake2b(text.encode('utf-8'), digest_size=16).digest()


def _tokens_sha(enc):
    h = hashlib.sha256()
    for ids in enc:
        h.update(np.asarray(ids, dtype=np.int32).tobytes())
        h.update(b'|')
    return h.hexdigest()


def prepare(project_dir, cache_dir):
    out_dir = os.path.join(project_dir, OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)
    wikitext = load_dataset('Salesforce/wikitext', 'wikitext-103-v1', cache_dir=cache_dir)
    hateval = load_dataset('cardiffnlp/tweet_eval', 'hate', cache_dir=cache_dir)

    test = wikitext['test']['text']
    candidates = [(i, t) for i, t in enumerate(test) if len(t.strip()) > MIN_CHARS]

    print('Hashing WikiText-103 train split ...', flush=True)
    train_hashes = {_hash(t.strip()) for t in wikitext['train']['text'] if t.strip()}
    val_hashes = {_hash(t.strip()) for t in wikitext['validation']['text'] if t.strip()}
    toxic_hashes = {_hash(x['text'].strip()) for x in hateval['train'] if x['label'] == 1}

    dup_train = [(i, t) for i, t in candidates if _hash(t.strip()) in train_hashes]
    dup_train_ids = {i for i, _ in dup_train}
    dup_val = [i for i, t in candidates if _hash(t.strip()) in val_hashes]
    dup_toxic = [i for i, t in candidates if _hash(t.strip()) in toxic_hashes]
    kept = [(i, t) for i, t in candidates if i not in dup_train_ids]

    tok = AutoTokenizer.from_pretrained('gpt2')
    enc = tok([t for _, t in kept])['input_ids']
    kept = [(i, t) for (i, t), ids in zip(kept, enc) if len(ids) >= 2]
    enc = [ids for ids in enc if len(ids) >= 2]

    manifest = {
        'dataset_id': DATASET_ID,
        'n_test_lines_raw': len(test),
        'n_test_lines_after_project_filter': len(candidates),
        'n_exact_duplicates_in_wikitext103_train': len(dup_train),
        'duplicate_examples': [t.strip() for _, t in dup_train[:25]],
        'n_exact_duplicates_in_wikitext103_validation': len(dup_val),
        'n_exact_duplicates_in_hateval_toxic_train': len(dup_toxic),
        'n_lines_evaluated': len(kept),
        'n_tokens': int(sum(len(x) for x in enc)),
        'n_predicted_tokens': int(sum(len(x) - 1 for x in enc)),
        'n_lines_longer_than_context': int(sum(len(x) > CONTEXT for x in enc)),
        'tokens_sha256': _tokens_sha(enc),
        'context_length': CONTEXT,
        'stride': STRIDE,
        'min_chars_filter': MIN_CHARS,
        'kept_test_indices': [i for i, _ in kept],
    }
    with open(os.path.join(out_dir, 'eval_set_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps({k: v for k, v in manifest.items() if k != 'kept_test_indices'}, indent=1))


def make_windows(ids):
    """Sliding windows over one line; each target token appears as a label exactly once."""
    windows, prev_end = [], 0
    for begin in range(0, len(ids), STRIDE):
        end = min(begin + CONTEXT, len(ids))
        trg = end - prev_end
        inp = ids[begin:end]
        windows.append((inp, [-100] * (len(inp) - trg) + inp[-trg:]))
        prev_end = end
        if end == len(ids):
            break
    return windows


@torch.no_grad()
def evaluate(project_dir, cache_dir, index):
    ckpt = CHECKPOINTS[index]
    out_dir = os.path.join(project_dir, OUT_SUBDIR, 'per_checkpoint')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{ckpt['ckpt_id']}.json")
    if os.path.exists(out_path):
        print(f'SKIP (exists): {out_path}')
        return

    with open(os.path.join(project_dir, OUT_SUBDIR, 'eval_set_manifest.json')) as f:
        manifest = json.load(f)
    test = load_dataset('Salesforce/wikitext', 'wikitext-103-v1', cache_dir=cache_dir)['test']['text']
    lines = [test[i] for i in manifest['kept_test_indices']]
    tok = AutoTokenizer.from_pretrained('gpt2')
    enc = tok(lines)['input_ids']
    sha = _tokens_sha(enc)
    assert sha == manifest['tokens_sha256'], 'evaluation tokens differ from the manifest'

    windows = [w for ids in enc for w in make_windows(ids)]
    windows.sort(key=lambda w: len(w[0]))
    expected = manifest['n_predicted_tokens']

    assert torch.cuda.is_available(), 'no usable GPU on this node (refusing silent CPU fallback)'
    device = 'cuda'
    model = AutoModelForCausalLM.from_pretrained(resolve(ckpt, project_dir)).to(device)
    model.eval()

    t0 = time.time()
    total_nll, total_n = 0.0, 0
    pad = tok.eos_token_id
    for b in range(0, len(windows), BATCH_SIZE):
        batch = windows[b:b + BATCH_SIZE]
        width = max(len(inp) for inp, _ in batch)
        inp = torch.full((len(batch), width), pad, dtype=torch.long)
        lab = torch.full((len(batch), width), -100, dtype=torch.long)
        att = torch.zeros((len(batch), width), dtype=torch.long)
        for r, (x, y) in enumerate(batch):
            inp[r, :len(x)] = torch.tensor(x)
            lab[r, :len(y)] = torch.tensor(y)
            att[r, :len(x)] = 1
        inp, lab, att = inp.to(device), lab.to(device), att.to(device)
        logits = model(input_ids=inp, attention_mask=att).logits.float()
        shift_logits, shift_labels = logits[:, :-1], lab[:, 1:]
        nll = F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)),
                              shift_labels.reshape(-1), ignore_index=-100, reduction='sum')
        total_nll += float(nll.double())
        total_n += int((shift_labels != -100).sum())
    assert total_n == expected, f'counted {total_n} targets, expected {expected}'

    avg = total_nll / total_n
    result = {
        'ckpt_id': ckpt['ckpt_id'], 'condition': ckpt['condition'], 'dose': ckpt['dose'],
        'train_seed': ckpt['train_seed'], 'checkpoint_path': ckpt['rel_path'],
        'dataset_id': DATASET_ID, 'n_lines': len(lines), 'n_windows': len(windows),
        'eval_tokens': total_n, 'total_nll': total_nll, 'avg_nll': avg, 'perplexity': math.exp(avg),
        'context_length': CONTEXT, 'stride': STRIDE, 'batch_size': BATCH_SIZE,
        'tokens_sha256': sha, 'device': torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu',
        'torch': torch.__version__, 'seconds': round(time.time() - t0, 1),
    }
    tmp = out_path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(result, f, indent=1)
    os.replace(tmp, out_path)
    print(json.dumps(result, indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=['prepare', 'evaluate'])
    p.add_argument('--index', type=int)
    p.add_argument('--project-dir', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    args = p.parse_args()
    cache_dir = os.path.join(args.project_dir, 'data', 'cache')
    if args.step == 'prepare':
        prepare(args.project_dir, cache_dir)
    else:
        evaluate(args.project_dir, cache_dir, args.index)


if __name__ == '__main__':
    main()
