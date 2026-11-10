"""Is the design-effect constant 0.365 actually a constant?

The de-correlated radius arm (``--fedtdo_decorr_rho 0.365``) hard-codes a number
that was measured once, on seed 2021, on the whole run.  Two things could make
that dishonest:

1. **the constant drifts**: if the within-arm autocorrelation of the centred
   reward is much higher early (few observations, wide priors) or late (level
   reconstruction has converged), a single number is an average of regimes and
   the radius is wrong at both ends;
2. **the constant does not generalise**: if rho differs across seeds, then
   replaying seed s with the rho measured on seed s is in-sample, and the
   end-to-end number inherits an optimistic bias.

This script measures rho four ways on the same logs:

* whole run, per seed (the number that is currently hard-coded);
* by quarter (rounds 1-15 / 16-30 / 31-45 / 46-60);
* by prefix only (rounds 1-15), i.e. what an online estimator would know early;
* by pair rank within an arm (the 1st-2nd observation, 3rd-4th, 5th+);

and then replays each seed with **leave-one-seed-out** rho, so the row it
produces uses a constant that seed never contributed to.

Usage::

    python scripts/probe_rho_online.py --markdown
"""

from __future__ import annotations

import argparse
import collections
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import probe_timing as timing  # noqa: E402
from federated import FedTDOController  # noqa: E402
from replay_selector import arm_ids, load_rounds  # noqa: E402

BASE = dict(beta=1.0, min_observations=1, estimator='mean', decision='lcb',
            shrink_n0=0.0, shrink_group='global')

SEEDS = [2021, 2022, 2023, 2024, 2025]
SUFFIX = 'b9_center'  # the arm whose log already uses the centred statistic


def lag1_pairs(by_client, centered=True, rounds=None, pair_rank=None, gap=None):
    """(x, y) pairs of consecutive observations of the same arm.

    ``gap`` filters on the number of *rounds* between the two observations, which
    is the quantity that actually decides whether they share a round-level shock.
    """
    cells = collections.defaultdict(list)
    for host, records in by_client.items():
        for index, record in enumerate(records, start=1):
            values = list(record['gains'].values())
            if not values:
                continue
            mean = float(np.mean(values))
            for arm, gain in record['gains'].items():
                value = float(gain) - mean if centered else float(gain)
                cells[(host, arm)].append((index, value))
    xs, ys = [], []
    for series in cells.values():
        series.sort()
        for rank, ((ix, first), (iy, second)) in enumerate(zip(series, series[1:])):
            if rounds is not None and not (rounds[0] <= ix and iy <= rounds[1]):
                continue
            if pair_rank is not None and rank not in pair_rank:
                continue
            if gap is not None and (iy - ix) not in gap:
                continue
            xs.append(first)
            ys.append(second)
    return np.asarray(xs), np.asarray(ys)



def rho_of(xs, ys, min_pairs=10):
    if xs.size < min_pairs or xs.std() < 1e-12 or ys.std() < 1e-12:
        return float('nan'), int(xs.size)
    return float(np.corrcoef(xs, ys)[0, 1]), int(xs.size)


def replay_gain(by_client, decorr_rho):
    candidates = arm_ids(by_client)
    row = timing.replay(by_client, candidates, FedTDOController, restrict=True,
                        shock='centered', decorr_rho=decorr_rho, **BASE)
    return 100.0 * row['relative_gain'], row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='./results/fedtdo')
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    parser.add_argument('--markdown', action='store_true')
    args = parser.parse_args()

    logs = {}
    for seed in args.seeds:
        path = os.path.join(args.root, f'traffic32_s{seed}_{SUFFIX}', 'rounds.json')
        if os.path.exists(path):
            logs[seed] = load_rounds(path)

    print('## 1. within-arm lag-1 rho of the centred reward\n')
    print(f'{"seed":<8}{"all":>8}{"Q1":>8}{"Q2":>8}{"Q3":>8}{"Q4":>8}'
          f'{"prefix15":>10}{"pair1":>8}{"pair2":>8}{"pair3+":>8}{"pairs":>7}')
    print('-' * 82)
    full = {}
    for seed, by_client in logs.items():
        row = [seed]
        xs, ys = lag1_pairs(by_client)
        full[seed] = rho_of(xs, ys)[0]
        row.append(full[seed])
        for window in [(1, 15), (16, 30), (31, 45), (46, 60)]:
            row.append(rho_of(*lag1_pairs(by_client, rounds=window))[0])
        row.append(rho_of(*lag1_pairs(by_client, rounds=(1, 15)))[0])
        for ranks in [{0}, {1}, set(range(2, 50))]:
            row.append(rho_of(*lag1_pairs(by_client, pair_rank=ranks))[0])
        row.append(xs.size)
        print(f'{row[0]:<8}' + ''.join(
            (f'{v:>8.3f}' if isinstance(v, float) else f'{v:>8}') for v in row[1:-1])
            + f'{row[-1]:>7}')

    mean_rho = float(np.nanmean(list(full.values())))
    print(f'\nmean over seeds: {mean_rho:.3f}   '
          f'spread {min(full.values()):.3f} .. {max(full.values()):.3f}')

    print('\n## 2. leave-one-seed-out constant\n')
    print(f'{"seed":<8}{"own rho":>9}{"LOO rho":>9}{"gain@0":>9}'
          f'{"gain@own":>10}{"gain@LOO":>10}{"delta":>8}')
    print('-' * 63)
    deltas = []
    for seed, by_client in logs.items():
        others = [v for other, v in full.items() if other != seed and not np.isnan(v)]
        loo = float(np.mean(others)) if others else float('nan')
        g0, _ = replay_gain(by_client, 0.0)
        g_own, _ = replay_gain(by_client, full[seed])
        g_loo, _ = replay_gain(by_client, loo)
        deltas.append(g_loo - g_own)
        print(f'{seed:<8}{full[seed]:>9.3f}{loo:>9.3f}{g0:>8.2f}%'
              f'{g_own:>9.2f}%{g_loo:>9.2f}%{g_loo - g_own:>+8.2f}')
    print(f'\nLOO minus own: mean {np.mean(deltas):+.3f}pp  '
          f'worst {np.max(np.abs(deltas)):.3f}pp')

    print('\n## 3. what an online (prefix-only) estimator would have used\n')
    print(f'{"seed":<8}{"prefix15":>10}{"full":>8}{"gap":>8}')
    print('-' * 34)
    for seed, by_client in logs.items():
        pre = rho_of(*lag1_pairs(by_client, rounds=(1, 15)))[0]
        print(f'{seed:<8}{pre:>10.3f}{full[seed]:>8.3f}{pre - full[seed]:>+8.3f}')

    print('\n## 5. rho by the number of rounds between the two observations\n')
    gaps = [1, 2, 3, 4]
    print(f'{"seed":<8}' + ''.join(f'{"gap=" + str(g):>10}' for g in gaps)
          + f'{"gap>=5":>9}{"share gap=1":>13}')
    print('-' * (8 + 10 * len(gaps) + 22))
    share = []
    for seed, by_client in logs.items():
        cells = []
        for gap in gaps:
            cells.append(rho_of(*lag1_pairs(by_client, gap={gap}))[0])
        cells.append(rho_of(*lag1_pairs(by_client, gap=set(range(5, 60))))[0])
        adjacent, total = lag1_pairs(by_client)
        gaps_all = []
        for host, records in by_client.items():
            seen = collections.defaultdict(list)
            for index, record in enumerate(records, start=1):
                for arm in record['gains']:
                    seen[arm].append(index)
            for series in seen.values():
                series.sort()
                gaps_all.extend(b - a for a, b in zip(series, series[1:]))
        share.append(float(np.mean([g == 1 for g in gaps_all])))
        print(f'{seed:<8}' + ''.join(
            (f'{v:>10.3f}' if not np.isnan(v) else f'{"nan":>10}') for v in cells)
            + f'{100 * share[-1]:>12.1f}%')
    print(f'\nadjacent-round pairs are {100 * np.mean(share):.1f}% of all pairs; '
          'a single exchangeable rho averages over both regimes.')

    print('\n## 4. sensitivity of the end-to-end gain to rho\n')
    grid = [0.0, 0.2, 0.365, 0.5, 0.635]
    print(f'{"seed":<8}' + ''.join(f'{"rho=" + str(g):>12}' for g in grid))
    print('-' * (8 + 12 * len(grid)))
    for seed, by_client in logs.items():
        cells = []
        for rho in grid:
            gain, _ = replay_gain(by_client, rho)
            cells.append(f'{gain:+11.2f}%')
        print(f'{seed:<8}' + ''.join(cells))


if __name__ == '__main__':
    main()
