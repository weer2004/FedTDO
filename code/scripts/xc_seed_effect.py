"""Are the five seeds exchangeable?

Every cross-dataset table in the report treats ``2021..2025`` as five independent
replicates and runs a sign test over them.  That is only valid if the seeds are
*interchangeable*.  They are not guaranteed to be: the seed draws the channel
subset that becomes the clients, so a seed that happens to sample channels with
a well-matched donor can lift every rule on every dataset.

This script tests that directly: for each seed, average the deployment gain over
the nine DTMoe benchmarks, and see whether one seed is uniformly on top.

It also reports the same quantity with the channel draw removed, by ranking seeds
*within* each dataset (a seed that wins on 9/9 datasets cannot be explained by
"this dataset is easy").

Usage::

    python scripts/xc_seed_effect.py
    python scripts/xc_seed_effect.py --markdown
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results' / 'fedtdo'

SEEDS = [2021, 2022, 2023, 2024, 2025]
ARMS = ['base', 'center', 'decorr']
DATASETS = [('ETTh1', 7, 2), ('ETTh2', 7, 2), ('ETTm1', 7, 2), ('ETTm2', 7, 2),
            ('Weather', 8, 2), ('Exchange', 8, 2), ('ILI', 7, 2),
            ('ECL', 32, 9), ('Traffic', 32, 9)]
LEGACY = {('Traffic', 'base'): 'traffic32_s{seed}_joint_b9_fix'}


def gain(dataset: str, arm: str, seed: int, nc: int, budget: int):
    legacy = LEGACY.get((dataset, arm))
    name = legacy.format(seed=seed) if legacy else \
        f'{dataset.lower()}{nc}_s{seed}_b{budget}_{arm}'
    path = RESULTS / name / 'metrics.json'
    if not path.exists():
        return None
    actions = json.loads(path.read_text()).get('actions') or {}
    if 'relative_gain' not in actions:
        return None
    return 100.0 * actions['relative_gain']


def collect():
    table = {}
    for dataset, nc, budget in DATASETS:
        for arm in ARMS:
            for seed in SEEDS:
                value = gain(dataset, arm, seed, nc, budget)
                if value is not None:
                    table[(dataset, arm, seed)] = value
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--markdown', action='store_true')
    args = parser.parse_args()
    table = collect()

    # 1. per-seed mean over the nine benchmarks, per arm
    print('per-seed mean deployment gain over the 9 benchmarks (pp)')
    head = f'{"arm":<8}' + ''.join(f'{s:>9}' for s in SEEDS) + f'{"spread":>9}'
    print(head)
    print('-' * len(head))
    per_seed = {seed: [] for seed in SEEDS}
    for arm in ARMS:
        values = []
        for seed in SEEDS:
            present = [table[(d, arm, seed)] for d, _, _ in DATASETS
                       if (d, arm, seed) in table]
            mean = statistics.fmean(present) if present else float('nan')
            values.append(mean)
            per_seed[seed].append(mean)
        spread = max(values) - min(values)
        print(f'{arm:<8}' + ''.join(f'{v:>+9.2f}' for v in values) + f'{spread:>+9.2f}')

    print()
    print('per-seed mean over arms and datasets (pp)')
    print(f'{"seed":<8}{"mean":>9}{"min":>9}{"max":>9}')
    print('-' * 35)
    for seed in SEEDS:
        values = per_seed[seed]
        print(f'{seed:<8}{statistics.fmean(values):>+9.2f}'
              f'{min(values):>+9.2f}{max(values):>+9.2f}')

    # 2. within-dataset rank of each seed (removes "this dataset is easy")
    print()
    print('within-dataset rank of each seed (1 = best rule gain, ties share the rank)')
    print(f'{"seed":<8}{"rank sum":>10}{"mean rank":>11}{"best on":>9}')
    print('-' * 40)
    ranks = {seed: [] for seed in SEEDS}
    for dataset, _, _ in DATASETS:
        for arm in ARMS:
            values = {seed: table[(dataset, arm, seed)] for seed in SEEDS
                      if (dataset, arm, seed) in table}
            if len(values) < len(SEEDS):
                continue
            ordered = sorted(values.values(), reverse=True)
            for seed, value in values.items():
                ranks[seed].append(ordered.index(value) + 1)
    for seed in SEEDS:
        if not ranks[seed]:
            continue
        print(f'{seed:<8}{sum(ranks[seed]):>10}{statistics.fmean(ranks[seed]):>11.2f}'
              f'{sum(1 for r in ranks[seed] if r == 1):>7}/{len(ranks[seed])}')

    # 3. the same, restricted to the two large-channel datasets
    print()
    print('same, only ECL + Traffic (largest channel draws)')
    for dataset in ['ECL', 'Traffic']:
        cells = []
        for seed in SEEDS:
            present = [table[(dataset, arm, seed)] for arm in ARMS
                       if (dataset, arm, seed) in table]
            cells.append(f'{statistics.fmean(present):+.2f}' if present else 'n/a')
        print(f'  {dataset:<9}' + ' '.join(f'{c:>8}' for c in cells))


if __name__ == '__main__':
    main()
