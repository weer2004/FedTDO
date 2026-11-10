"""Is a seed's advantage a channel-draw artifact?

Each seed draws the client subset (when the dataset has more channels than
clients).  That subset has its own *difficulty*: the mean over its channels of
the best achievable arm gain.  If seed 2021's subset is systematically the most
generous one, then "seed 2021 is the best seed on every large dataset" is a
property of the draw, not of the rule -- and every 5-seed sign test in the
report is really a 5-draw test with one unusually favourable draw in it.

This script measures, per (dataset, seed, arm), from ``rounds.json`` alone:

* ``oracle``   -- mean over channels of the best arm's mean reward (the ceiling
                 that channel subset allows to a perfect offline selector);
* ``realised`` -- the deployment gain the online rule actually banked;
* ``capture``  -- realised / oracle.

Then it correlates ``oracle`` with ``realised`` across seeds and reports where
seed 2021's draw sits inside the five.

Usage::

    python scripts/xc_seed_luck.py
    python scripts/xc_seed_luck.py --markdown
"""

from __future__ import annotations

import argparse
import collections
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


def run_dir(dataset, arm, seed, nc, budget):
    legacy = LEGACY.get((dataset, arm))
    name = legacy.format(seed=seed) if legacy else \
        f'{dataset.lower()}{nc}_s{seed}_b{budget}_{arm}'
    return RESULTS / name


def subset_stats(directory: Path):
    path = directory / 'rounds.json'
    if not path.exists():
        return None
    records = json.loads(path.read_text())
    cells = collections.defaultdict(list)
    for record in records:
        for arm, reward in (record.get('raw_gains') or {}).items():
            cells[(record['client'], arm)].append(float(reward))
    per_host = collections.defaultdict(list)
    for (host, _), values in cells.items():
        per_host[host].append(sum(values) / len(values))
    if not per_host:
        return None
    oracle = 100.0 * statistics.fmean(max(v) for v in per_host.values()
                                      if len(v) >= 2)
    base = sum(r['default_loss'] for r in records)
    deployed = sum(r['deployed_loss'] for r in records)
    realised = 100.0 * (base - deployed) / max(base, 1e-12)
    bets = sum(1 for r in records if r.get('active_expert'))
    return dict(oracle=oracle, realised=realised, hosts=len(per_host), bets=bets)


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float('nan')
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs) ** 0.5
    vy = sum((y - my) ** 2 for y in ys) ** 0.5
    return cov / (vx * vy) if vx * vy else float('nan')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--markdown', action='store_true')
    parser.add_argument('--arm', default='decorr', choices=ARMS)
    args = parser.parse_args()
    md = args.markdown

    print(f'channel-subset difficulty vs realised gain  (arm = {args.arm})')
    print(f'{"dataset":<9}{"seed":<7}{"hosts":>7}{"oracle":>9}{"realised":>10}'
          f'{"capture":>9}{"bets":>7}')
    print('-' * 61)

    pooled = {seed: [] for seed in SEEDS}
    per_dataset = {}
    for dataset, nc, budget in DATASETS:
        rows = {}
        for seed in SEEDS:
            stats = subset_stats(run_dir(dataset, args.arm, seed, nc, budget))
            if stats is None:
                continue
            rows[seed] = stats
            pooled[seed].append(stats['oracle'])
            capture = stats['realised'] / stats['oracle'] if stats['oracle'] else float('nan')
            print(f'{dataset:<9}{seed:<7}{stats["hosts"]:>7}{stats["oracle"]:>+9.2f}'
                  f'{stats["realised"]:>+10.2f}{capture:>9.2f}{stats["bets"]:>7}')
        if rows:
            per_dataset[dataset] = rows
        print('-' * 61)

    print()
    print('is seed 2021 the most generous draw?')
    print(f'{"seed":<7}{"mean oracle":>13}{"rank of own oracle (1 = most generous)":>42}')
    print('-' * 58)
    oracle_rank = {seed: [] for seed in SEEDS}
    for dataset, rows in per_dataset.items():
        ordered = sorted(rows.items(), key=lambda kv: kv[1]['oracle'], reverse=True)
        for position, (seed, _) in enumerate(ordered, start=1):
            oracle_rank[seed].append(position)
    for seed in SEEDS:
        values = pooled[seed]
        if not values:
            continue
        ranks = oracle_rank[seed]
        print(f'{seed:<7}{statistics.fmean(values):>+13.2f}'
              f'{statistics.fmean(ranks) if ranks else float("nan"):>20.2f}'
              f'   mean over {len(ranks)} datasets')

    print()
    pairs = [(rows[s]['oracle'], rows[s]['realised'])
             for rows in per_dataset.values() for s in rows]
    if len(pairs) >= 3:
        r = pearson([p[0] for p in pairs], [p[1] for p in pairs])
        print(f'corr(subset oracle, realised gain) over {len(pairs)} '
              f'(dataset, seed) cells: r = {r:+.3f}')
        print('  -> if this is high, the seed effect is *draw difficulty*, '
              'not rule quality')


if __name__ == '__main__':
    main()
