"""Fixed, ordered list of the checkpoints evaluated by the supplementary experiments.

The index into CHECKPOINTS is the Slurm array task id:
    0        pretrained GPT-2-small (optional reference)
    1-3      clean-only control, dose 0.0, seeds 0-2
    4-15     contaminated, doses 0.01/0.05/0.1/0.25 x seeds 0-2
    16-27    recovered,    doses 0.01/0.05/0.1/0.25 x seeds 0-2
    28-30    clean-then-clean control, seeds 0-2 (second clean epoch from the dose-0 model)
"""
import os

DOSES = [0.01, 0.05, 0.1, 0.25]
SEEDS = [0, 1, 2]


def _ckpt(condition, dose, seed, phase_dir):
    return {
        'ckpt_id': f'{condition}_d{dose}_s{seed}',
        'condition': condition,
        'dose': dose,
        'train_seed': seed,
        'rel_path': os.path.join('checkpoints', phase_dir, f'dose_{dose}_seed_{seed}', 'final_checkpoint'),
    }


CHECKPOINTS = [{'ckpt_id': 'gpt2_pretrained', 'condition': 'pretrained', 'dose': None,
                'train_seed': None, 'rel_path': 'gpt2'}]
CHECKPOINTS += [_ckpt('clean', 0.0, s, 'contaminated') for s in SEEDS]
CHECKPOINTS += [_ckpt('contaminated', d, s, 'contaminated') for d in DOSES for s in SEEDS]
CHECKPOINTS += [_ckpt('recovered', d, s, 'recovered') for d in DOSES for s in SEEDS]
# 28-30: clean-then-clean control (clean model + the same second WikiText epoch as recovery),
# trained by scripts/submit_clean_then_clean.sh
CHECKPOINTS += [_ckpt('clean_then_clean', 0.0, s, 'recovered') for s in SEEDS]

# One clean, one contaminated and one recovered checkpoint (plus the pretrained reference)
# for the pilot run requested before the full submission.
PILOT_INDICES = [0, 1, 10, 22]


def resolve(ckpt, project_dir):
    """Return a path/name that AutoModelForCausalLM.from_pretrained accepts."""
    if ckpt['condition'] == 'pretrained':
        return ckpt['rel_path']
    return os.path.join(project_dir, ckpt['rel_path'])
