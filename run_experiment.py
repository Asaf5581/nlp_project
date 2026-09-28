#!/usr/bin/env python3
"""CLI runner for contamination and recovery experiments on Slurm."""
import argparse
import os
import sys
import torch
import gc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data import load_data, make_mixed_dataset
from train import run_finetune
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dose', type=float, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--phase', choices=['contaminate', 'recover'], required=True)
    parser.add_argument('--base-dir', type=str, default=os.path.dirname(__file__))
    parser.add_argument('--total-size', type=int, default=50000)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--logging-steps', type=int, default=50)
    args = parser.parse_args()

    model_name = 'gpt2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    _, _, tweeteval_off = load_data(
        cache_dir=os.path.join(args.base_dir, 'data', 'cache')
    )

    if args.phase == 'contaminate':
        print(f"\n=== Contamination | Dose: {args.dose} | Seed: {args.seed} ===")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        dataset = make_mixed_dataset(
            tokenizer, dose=args.dose, seed=args.seed,
            total_size=args.total_size,
            cache_dir=os.path.join(args.base_dir, 'data', 'cache')
        )
        output_dir = os.path.join(args.base_dir, 'checkpoints', 'contaminated',
                                  f'dose_{args.dose}_seed_{args.seed}')
        log_csv = os.path.join(args.base_dir, 'results', 'contamination_logs',
                               f'dose_{args.dose}_seed_{args.seed}.csv')

    elif args.phase == 'recover':
        print(f"\n=== Recovery | Prior Dose: {args.dose} | Seed: {args.seed} ===")
        ckpt_dir = os.path.join(args.base_dir, 'checkpoints', 'contaminated',
                                f'dose_{args.dose}_seed_{args.seed}', 'final_checkpoint')
        if not os.path.exists(ckpt_dir):
            print(f"ERROR: Checkpoint not found: {ckpt_dir}")
            sys.exit(1)
        model = AutoModelForCausalLM.from_pretrained(ckpt_dir)
        dataset = make_mixed_dataset(
            tokenizer, dose=0.0, seed=args.seed,
            total_size=args.total_size,
            cache_dir=os.path.join(args.base_dir, 'data', 'cache')
        )
        output_dir = os.path.join(args.base_dir, 'checkpoints', 'recovered',
                                  f'dose_{args.dose}_seed_{args.seed}')
        log_csv = os.path.join(args.base_dir, 'results', 'recovery_logs',
                               f'dose_{args.dose}_seed_{args.seed}.csv')

    if torch.cuda.is_available():
        model = model.to('cuda')

    split = dataset.train_test_split(test_size=0.05, seed=args.seed)

    run_finetune(
        model=model,
        tokenizer=tokenizer,
        train_dataset=split['train'],
        output_dir=output_dir,
        eval_dataset=split['test'],
        tweeteval_off=tweeteval_off,
        log_csv_path=log_csv,
        num_train_epochs=args.epochs,
        batch_size=args.batch_size,
        logging_steps=args.logging_steps,
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"=== DONE: {args.phase} dose={args.dose} seed={args.seed} ===")


if __name__ == '__main__':
    main()
