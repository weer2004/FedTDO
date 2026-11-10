"""Seed-by-seed comparison of the probe-bug fix, with the honest-radius check.

Three questions, one table:

1. does the fix replicate across seeds, or was +2.38% one lucky draw?
2. is the budget still dominated by a single client in each seed?
3. does the honest radius (``n_eff``, not nominal ``n``) survive on each seed?

``gain`` is read from ``metrics.json`` (the end-to-end run, the number a paper
would quote); ``replay`` is the same rule re-run over ``rounds.json`` and is
expected to agree to within rounding - a disagreement means the replay has
drifted from the live policy and the other columns cannot be trusted.

Usage::

    python scripts/probe_seed_compare.py --seeds 2021 2022 2023 --arm joint_b9_fix
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import probe_dependence as dependence  # noqa: E402
from replay_selector import arm_ids, load_rounds  # noqa: E402

FED = os.path.join(ROOT, 'results', 'fedtdo')
BASE = dict(beta=1.0, min_observations=1, estimator='mean', decision='lcb',
            shrink_n0=0.0, shrink_group='global')
# seed 2021 predates the sweep, so its unrepaired directory has no seed tag
LEGACY = {'joint_b9': 'traffic32_joint_b9',
          'joint_b9_fix': 'traffic32_s2021_joint_b9_fix',
          'notransfer': 'traffic32_notransfer'}


def arm_dir(seed, arm):
    name = f'traffic32_s{seed}_{arm}'
    if os.path.isdir(os.path.join(FED, name)):
        return name
    if seed == 2021 and arm in LEGACY:
        return LEGACY[arm]
    return name


def read(path):
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def measure(seed, arm):
    directory = os.path.join(FED, arm_dir(seed, arm))
    metrics_path = os.path.join(directory, 'metrics.json')
    rounds_path = os.path.join(directory, 'rounds.json')
    if not (os.path.exists(metrics_path) and os.path.exists(rounds_path)):
        return None
    metrics = read(metrics_path)
    actions = metrics.get('actions', {})
    gain = float(actions.get('relative_gain', metrics.get('mean_client_mse', np.nan)))
    by_client = load_rounds(rounds_path)
    candidates = arm_ids(by_client)
    rho = float(dependence.autocorrelation(by_client)[1][1])
    nominal = dependence.replay(by_client, candidates, rho=0.0, **BASE)
    honest = dependence.replay(by_client, candidates, rho=rho, **BASE)
    return {
        'seed': seed,
        'arm': arm_dir(seed, arm),
        'gain': gain,
        'replay': float(nominal['relative_gain']),
        'deploys': int(nominal['deploys']),
        'precision': float(nominal['precision']),
        'rho': rho,
        'effective': 1.0 / (1.0 + 3.0 * rho),
        'honest': float(honest['relative_gain']),
        'honest_deploys': int(honest['deploys']),
    }


def sign_test_p(positive, total):
    """Two-sided sign test, exact, for the number of positive seeds."""
    if total == 0:
        return 1.0
    from math import comb
    tail = sum(comb(total, k) for k in range(positive, total + 1)) / 2 ** total
    return float(min(1.0, 2 * tail))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='+', type=int, default=[2021, 2022, 2023])
    parser.add_argument('--arm', default='joint_b9_fix')
    parser.add_argument('--baseline', default='joint_b9',
                        help='arm to compare against (the pre-fix code)')
    args = parser.parse_args()

    rows = []
    for seed in args.seeds:
        row = measure(seed, args.arm)
        if row:
            old = measure(seed, args.baseline) if args.baseline != args.arm else None
            row['before'] = old['gain'] if old else None
            rows.append(row)
    if not rows:
        raise SystemExit('no completed runs found')

    print(f"{'seed':<6}{'before':>9}{'after':>9}{'delta':>8}"
          f"{'replay':>9}{'n':>5}{'prec':>7}{'rho':>7}{'n_eff%':>8}"
          f"{'honest':>9}")
    print('-' * 84)
    for row in rows:
        before = f"{100 * row['before']:+.2f}%" if row['before'] is not None else '-'
        print(f"{row['seed']:<6}{before:>9}{100 * row['gain']:>+8.2f}%"
              f"{(100 * (row['gain'] - row['before']) if row['before'] is not None else float('nan')):>+7.2f}%"
              f"{100 * row['replay']:>+8.2f}%{row['deploys']:>5d}"
              f"{100 * row['precision']:>6.0f}%{row['rho']:>7.3f}"
              f"{100 * row['effective']:>7.0f}%{100 * row['honest']:>+8.2f}%")

    gains = np.array([row['gain'] for row in rows])
    print(f"\nacross {len(rows)} seed(s): "
          f"mean {100 * gains.mean():+.2f}%  median {100 * np.median(gains):+.2f}%  "
          f"range [{100 * gains.min():+.2f}%, {100 * gains.max():+.2f}%]  "
          f"positive {int((gains > 0).sum())}/{len(gains)}  "
          f"sign-test p = {sign_test_p(int((gains > 0).sum()), len(gains)):.4f}")
    hon = np.array([row['honest'] for row in rows])
    print(f"honest radius on the same seeds: "
          f"mean {100 * hon.mean():+.2f}%  max {100 * hon.max():+.2f}%")
    rho = np.array([row['rho'] for row in rows])
    print(f"rho across seeds: {rho.min():.3f}-{rho.max():.3f}  "
          f"effective sample {100 * (1 / (1 + 3 * rho)).min():.0f}%-"
          f"{100 * (1 / (1 + 3 * rho)).max():.0f}% of nominal")


if __name__ == '__main__':
    main()
