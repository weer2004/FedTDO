"""Cross-seed table for the de-correlated radius (design effect) arm.

Three rules are compared on the same five seeds:

  baseline   probe bugs fixed, raw reward, iid radius
  centered   paired reward + reconstructed level, iid radius
  decorr     centered + design-effect radius (rho = 0.365)

Motivation: the residual autocorrelation of per-arm adjacent-round rewards is
0.605 before centring and 0.365 after.  An iid radius is then optimistic by
sqrt(1 + (n - 1) * rho), so the third arm divides n by that factor.  The claim
tested here is *not* "higher mean gain" -- it is "same gain with a smaller
number of transfers and a narrower left tail".

Usage::

    python scripts/decorr_seed_table.py --markdown
"""

from __future__ import annotations

import argparse
import json
import os
from math import comb

ROOT = './results/fedtdo'
ARMS = [
    ('joint_b9_fix', 'baseline', 'raw reward, iid radius'),
    ('b9_center', 'centered', 'paired reward, iid radius'),
    ('b9_decorr', 'decorr', 'paired reward + design-effect radius'),
]
SEEDS = [2021, 2022, 2023, 2024, 2025]


def read(seed, suffix):
    path = os.path.join(ROOT, f'traffic32_s{seed}_{suffix}', 'metrics.json')
    if not os.path.exists(path):
        return None
    metrics = json.load(open(path))
    actions = metrics['actions']
    return {
        'gain': 100.0 * float(actions['relative_gain']),
        'transfers': int(actions['transfers']),
        'transfers_per_client': int(actions['transfers']) / len(metrics['clients']),
        'active_clients': int(actions['clients_with_active_expert']),
        'evaluations': int(actions['evaluations']),
    }


def sign_test(diffs):
    """Two-sided exact sign test, zeros dropped."""
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    total = pos + neg
    if total == 0:
        return 1.0
    k = min(pos, neg)
    return min(1.0, 2 * sum(comb(total, i) for i in range(k + 1)) / 2 ** total)


def median(values):
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--markdown', action='store_true')
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    args = parser.parse_args()

    table = {}
    for suffix, label, _ in ARMS:
        table[label] = {seed: read(seed, suffix) for seed in args.seeds}

    if args.markdown:
        header = '| 规则 | ' + ' | '.join(f'seed {s}' for s in args.seeds)
        header += ' | mean | median | min | 正收益 | transfers |'
        print(header)
        print('|' + '---|' * (len(args.seeds) + 7))
    else:
        print('arm'.ljust(10) + ''.join(f'{s:>9}' for s in args.seeds)
              + f'{"mean":>8}{"med":>8}{"min":>8}{"pos":>6}{"xfers":>7}')

    for _, label, note in ARMS:
        rows = [table[label][s] for s in args.seeds if table[label][s]]
        if not rows:
            continue
        gains = [r['gain'] for r in rows]
        mean = sum(gains) / len(gains)
        med = median(gains)
        lo = min(gains)
        pos = sum(1 for g in gains if g > 0)
        xfers = sum(r['transfers'] for r in rows) / len(rows)
        if args.markdown:
            cells = []
            for s in args.seeds:
                r = table[label][s]
                cells.append('n/a' if not r else f'{r["gain"]:+.2f}%')
            print(f'| {label} | ' + ' | '.join(cells)
                  + f' | {mean:+.2f} | {med:+.2f} | {lo:+.2f} | {pos}/{len(gains)}'
                  + f' | {xfers:.0f} |')
        else:
            print(label.ljust(10)
                  + ''.join((f'{table[label][s]["gain"]:+9.2f}' if table[label][s] else '      n/a')
                            for s in args.seeds)
                  + f'{mean:+8.2f}{med:+8.2f}{lo:+8.2f}{pos:>4}/{len(gains)}{xfers:>7.0f}')

    print()
    print('paired differences (pp)')
    for a, b in [('centered', 'baseline'), ('decorr', 'baseline'), ('decorr', 'centered')]:
        diffs = [table[a][s]['gain'] - table[b][s]['gain']
                 for s in args.seeds if table[a][s] and table[b][s]]
        if not diffs:
            continue
        mean = sum(diffs) / len(diffs)
        wins = sum(1 for d in diffs if d > 0)
        print(f'  {a} - {b}: mean {mean:+.3f}  wins {wins}/{len(diffs)}'
              f'  sign-test p={sign_test(diffs):.3f}'
              f'  diffs={" ".join(f"{d:+.2f}" for d in diffs)}')

    print()
    print('transfer count per run')
    for _, label, _ in ARMS:
        counts = [table[label][s]['transfers'] for s in args.seeds if table[label][s]]
        if counts:
            print(f'  {label}: {" ".join(str(c) for c in counts)}  mean {sum(counts)/len(counts):.0f}')


if __name__ == '__main__':
    main()
