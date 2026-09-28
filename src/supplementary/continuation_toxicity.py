#!/usr/bin/env python3
"""Experiment 2: continuation-only vs. combined-text toxicity for one checkpoint.

Reproduces the original evaluation in src/eval.py / src/train.py:
  - prompts: TweetEval-offensive *train* split, np.random.seed(42),
    np.random.choice(len, 100, replace=False), first 50 characters (exact reconstruction)
  - decoding: what pipeline('text-generation') did for GPT-2 under transformers 4.30,
    i.e. task_specific_params -> do_sample=True, plus GenerationConfig defaults
    top_k=50, top_p=1.0, temperature=1.0; max_new_tokens=30 (EOS may end a sample early,
    as in the original; the number of new tokens is recorded per sample)
  - scorer: unitary/toxic-bert through the same text-classification pipeline call,
    score of the 'toxic' label (sigmoid)
New in this experiment: 5 samples per prompt with fixed evaluation seeds
(EVAL_SEED_BASE + prompt_id, identical for every checkpoint), and two scores per sample:
  tox_full  prompt + continuation (the original metric, reconstructed like the pipeline's
            return_full_text: prompt string + decoded text after the decoded prompt)
  tox_cont  continuation only, decoded from the generated token ids alone
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoModelForSequenceClassification,
                          AutoTokenizer, pipeline)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from checkpoints import CHECKPOINTS, resolve  # noqa: E402

N_PROMPTS = 100
PROMPT_CHARS = 50
PROMPT_SEED = 42
N_SAMPLES = 5
EVAL_SEED_BASE = 20260928
DECODING = dict(do_sample=True, top_k=50, top_p=1.0, temperature=1.0, max_new_tokens=30,
                num_return_sequences=N_SAMPLES)
SCORER = 'unitary/toxic-bert'
SHORT_WORDS = 3
OUT_SUBDIR = os.path.join('results', 'supplementary', 'continuation_toxicity')


def reconstruct_prompts(cache_dir):
    ds = load_dataset('cardiffnlp/tweet_eval', 'offensive', cache_dir=cache_dir)['train']
    np.random.seed(PROMPT_SEED)
    indices = np.random.choice(len(ds), min(N_PROMPTS, len(ds)), replace=False)
    return pd.DataFrame({
        'prompt_id': range(len(indices)),
        'dataset_index': [int(i) for i in indices],
        'prompt_label': [int(ds[int(i)]['label']) for i in indices],  # 1 = offensive
        'prompt': [ds[int(i)]['text'][:PROMPT_CHARS] for i in indices],
    })


def toxic_scores(scorer, texts):
    out = scorer(texts, truncation=True, max_length=512, top_k=None, batch_size=32)
    return [next((d['score'] for d in r if d['label'] == 'toxic'), 0.0) for r in out]


@torch.no_grad()
def evaluate(project_dir, cache_dir, index):
    ckpt = CHECKPOINTS[index]
    base_out = os.path.join(project_dir, OUT_SUBDIR)
    out_dir = os.path.join(base_out, 'per_checkpoint')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{ckpt['ckpt_id']}.csv")
    if os.path.exists(out_path):
        print(f'SKIP (exists): {out_path}')
        return

    prompts = reconstruct_prompts(cache_dir)
    tmp = os.path.join(base_out, f'prompts.{index}.tmp')
    prompts.to_csv(tmp, index=False)
    os.replace(tmp, os.path.join(base_out, 'prompts.csv'))

    assert torch.cuda.is_available(), 'no usable GPU on this node (refusing silent CPU fallback)'
    device = 'cuda'
    tok = AutoTokenizer.from_pretrained('gpt2')
    model = AutoModelForCausalLM.from_pretrained(resolve(ckpt, project_dir)).to(device)
    model.eval()
    eos = tok.eos_token_id

    t0 = time.time()
    rows = []
    for p in prompts.itertuples():
        enc = tok(p.prompt, return_tensors='pt').to(device)
        plen = enc.input_ids.shape[1]
        seed = EVAL_SEED_BASE + p.prompt_id
        torch.manual_seed(seed)
        out = model.generate(**enc, pad_token_id=eos, **DECODING)
        prompt_decoded = tok.decode(enc.input_ids[0], skip_special_tokens=True)
        for k in range(N_SAMPLES):
            new_ids = out[k, plen:].tolist()
            hit_eos = eos in new_ids
            if hit_eos:
                new_ids = new_ids[:new_ids.index(eos)]
            cont = tok.decode(new_ids, skip_special_tokens=True)
            full_decoded = tok.decode(enc.input_ids[0].tolist() + new_ids, skip_special_tokens=True)
            rows.append({
                'ckpt_id': ckpt['ckpt_id'], 'condition': ckpt['condition'], 'dose': ckpt['dose'],
                'train_seed': ckpt['train_seed'], 'prompt_id': p.prompt_id,
                'dataset_index': p.dataset_index, 'prompt_label': p.prompt_label,
                'sample_idx': k, 'eval_seed': seed, 'n_new_tokens': len(new_ids),
                'hit_eos': hit_eos, 'short_continuation': len(cont.split()) < SHORT_WORDS,
                'prompt': p.prompt, 'continuation': cont,
                'full_text': p.prompt + full_decoded[len(prompt_decoded):],
            })
    gen_seconds = time.time() - t0
    del model
    torch.cuda.empty_cache()

    scorer_tok = AutoTokenizer.from_pretrained(SCORER)
    scorer_model = AutoModelForSequenceClassification.from_pretrained(SCORER).to(device)
    scorer = pipeline('text-classification', model=scorer_model, tokenizer=scorer_tok,
                      device=0 if device == 'cuda' else -1)
    df = pd.DataFrame(rows)
    df['tox_full'] = toxic_scores(scorer, df['full_text'].tolist())
    df['tox_cont'] = toxic_scores(scorer, df['continuation'].tolist())
    prompt_tox = dict(zip(prompts['prompt_id'], toxic_scores(scorer, prompts['prompt'].tolist())))
    df['tox_prompt'] = df['prompt_id'].map(prompt_tox)

    tmp = out_path + '.tmp'
    df.to_csv(tmp, index=False)
    os.replace(tmp, out_path)

    config = {'decoding': {k: v for k, v in DECODING.items()}, 'eval_seed_base': EVAL_SEED_BASE,
              'n_prompts': N_PROMPTS, 'prompt_chars': PROMPT_CHARS, 'prompt_seed': PROMPT_SEED,
              'scorer': SCORER, 'short_words_threshold': SHORT_WORDS,
              'torch': torch.__version__, 'transformers': __import__('transformers').__version__,
              'device': torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'}
    with open(os.path.join(base_out, 'generation_config.json'), 'w') as f:
        json.dump(config, f, indent=1)
    print(f"{ckpt['ckpt_id']}: {len(df)} samples, gen {gen_seconds:.0f}s, "
          f"tox_full={df.tox_full.mean():.4f} tox_cont={df.tox_cont.mean():.4f} "
          f"eos={df.hit_eos.mean():.3f} short={df.short_continuation.mean():.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--index', type=int, required=True)
    p.add_argument('--project-dir', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    args = p.parse_args()
    evaluate(args.project_dir, os.path.join(args.project_dir, 'data', 'cache'), args.index)


if __name__ == '__main__':
    main()
