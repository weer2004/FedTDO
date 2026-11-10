"""Cross-seed table for the shock experiment: baseline vs centered vs honest.

The single-seed lesson from the seed sweep (``TRAFFIC_SEED_SWEEP.md`` §12) is that
any effect smaller than the seed spread has to be reported as a distribution, not
as a number.  This script reads three arms per seed and prints them together with
the median, the positive count and a sign test.

Usage::

    python scripts/shock_seed_table.py --seeds 2021 2022 2023 2025 --markdown
"""

from __future__ import annotations

import argparse
import json
import os
import numpy as np

ROOT = './results/fedtdo'
# (suffix, label, note)
ARMS = [
    ('joint_b9_fix', 'baseline', 'probe bugs fixed, raw reward'),
    ('b9_center', 'centered', 'paired reward + reconstructed level'),
    ('b9_decorr', 'honest', 'centered + design-effect radius'),
]


def read(arm):
    path = os.path.join(ROOT, arm, 'metrics.json')
    if not os.path.exists(path):
        return None
    metrics = json.load(open(path))
    clients = metrics.get('client_metrics', {})
    values = np.array([item['mse'] for item in clients.values()], dtype=float)
    return {
        'gain': float(metrics['actions']['relative_gain']),
        'transfers': int(metrics['actions']['transfers']),
        'evaluations': int(metrics['actions']['evaluations']),
        'mean_mse': float(values.mean()) if values.size else float('nan'),
        'median_mse': float(np.median(values)) if values.size else float('nan'),
        'finished': True,
    }


def sign_test(values):
    """Two-sided sign test against zero, ignoring exact zeros."""
    positive = int(np.sum(values > 0))
    total = int(np.sum(values != 0))
    if total == 0:
        return 1.0
    # exact binomial, two-sided
    from math import comb
    k = min(positive, total - positive)
    tail = sum(comb(total, i) for i in range(k + 1)) / 2 ** total
    return min(1.0, 2 * tail)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='+', type=int, default=[2021, 2022, 2023, 2025])
    parser.add_argument('--markdown', action='store_true')
    args = parser.parse_args()

    table = {}
    for seed in args.seeds:
        row = {}
        for suffix, label, _ in ARMS:
            row[label] = read(f'traffic32_s{seed}_{suffix}')
        table[seed] = row

    if args.markdown:
        print('| seed | baseline | centered | honest | Δ(centered) | seed 最好者 |')
        print('|---|---|---|---|---|---|')
    else:
        print(f'{"seed":<6}{"baseline":>12}{"centered":>12}{"honest":>12}'
              f'{"delta":>10}  transfers (b/c/h)')
        print('-' * 74)
    for seed in args.seeds:
        row = table[seed]
        base = row['baseline']
        centered = row['centered']
        honest = row['honest']
        if base is None or centered is None:
            print(f'{seed}: missing arm')
            continue
        delta = centered['gain'] - base['gain']
        if args.markdown:
            best = max(((base, 'baseline'), (centered, 'centered'),
                        (honest, 'honest') if honest else (base, 'baseline')),
                       key=lambda item: item[0]['gain'])[1]
            honest_gain = f'{100 * honest["gain"]:+.2f}%' if honest else '—'
            print(f'| {seed} | {100 * base["gain"]:+.2f}% | '
                  f'{100 * centered["gain"]:+.2f}% | {honest_gain} | '
                  f'{100 * delta:+.2f}pp | {best} |')
        else:
            print(f'{seed:<6}{100 * base["gain"]:>11.2f}%'
                  f'{100 * centered["gain"]:>11.2f}%'
                  f'{(100 * honest["gain"] if honest else float("nan")):>11.2f}%'
                  f'{100 * delta:>+9.2f}pp  '
                  f'{base["transfers"]}/{centered["transfers"]}/'
                  f'{honest["transfers"] if honest else "-"}')

    # only seeds where all three arms finished: mixing a pre-fix baseline with a
    # fixed variant would attribute the probe-bug fix to the mechanism
    complete = [seed for seed in args.seeds
                if all(table[seed][label] for label, _, _ in
                       [('baseline', 0, 0), ('centered', 0, 0), ('honest', 0, 0)])]
    if len(complete) != len(args.seeds):
        print(f'\n(using {len(complete)} complete seeds: {complete})')
    for label in ('baseline', 'centered', 'honest'):
        values = np.array([table[seed][label]['gain'] for seed in complete],
                          dtype=float)
        if values.size == 0:
            continue
        print(f'\n{label:<9} n={values.size}  mean {100 * values.mean():+.2f}%  '
              f'median {100 * np.median(values):+.2f}%  '
              f'min {100 * values.min():+.2f}%  '
              f'positive {int((values > 0).sum())}/{values.size}  '
              f'sign-test p={sign_test(values):.3f}')

    def spread(label):
        values = np.array([table[seed][label]['gain'] for seed in args.seeds
                           if table[seed][label] and table[seed]['baseline']],
                          dtype=float)
        return float(values.std(ddof=1)) if values.size > 1 else 0.0

    print(f'\nseed spread (sd): baseline {100 * spread("baseline"):.2f}pp  '
          f'centered {100 * spread("centered"):.2f}pp  '
          f'honest {100 * spread("honest"):.2f}pp')


if __name__ == '__main__':
    main()
