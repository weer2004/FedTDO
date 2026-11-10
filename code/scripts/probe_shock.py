"""Does removing the round shock make the radius honest *and* keep a level?

``probe_dependence.py`` showed that the honest radius destroys the gain (1.81%
-> 0.08%) because the raw observations share a 0.64 autocorrelation.
``probe_paired.py`` showed that pairing alone destroys the *level* (precision
16%) because a within-round contrast says nothing about beating the default
path.  Both are true of the two halves of one solution:

    store     d[t,a] = gain[t,a] - round_mean[t]        # kills the shock
    report    baseline_hat + mean(d)                    # restores the level

``baseline_hat = mu + k(last - mu)`` is licensed by the measured persistence of
the round shock itself (lag-1 = 0.615 within host), so this is not a free
parameter: ``k`` is set to that autocorrelation.

This script measures, for every logged arm:

1. the within-arm lag-1 autocorrelation before and after centering (what the
   radius is allowed to assume);
2. the replay gain of the raw rule, the centered rule, and the centered rule
   with the *honest* radius, all on the same action space.

Usage::

    python scripts/probe_shock.py --arms traffic32_s2021_joint_b9_fix \
        traffic32_s2022_joint_b9_fix --markdown
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


def autocorrelation(by_client, centered):
    """Within-arm lag-1 correlation of the reward, raw or round-demeaned."""
    cells = collections.defaultdict(list)
    for host, records in by_client.items():
        for index, record in enumerate(records):
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
        for (_, first), (_, second) in zip(series, series[1:]):
            xs.append(first)
            ys.append(second)
    if len(xs) < 10:
        return float('nan'), 0
    return float(np.corrcoef(xs, ys)[0, 1]), len(xs)


def row(by_client, candidates, restricted, **kwargs):
    return timing.replay(by_client, candidates, FedTDOController,
                         restrict=restricted, **BASE, **kwargs)


def measure(root, arm, restricted):
    by_client = load_rounds(os.path.join(root, arm, 'rounds.json'))
    candidates = arm_ids(by_client)
    raw_rho, pairs = autocorrelation(by_client, centered=False)
    centered_rho, _ = autocorrelation(by_client, centered=True)
    rows = {
        'raw': row(by_client, candidates, restricted),
        'centered': row(by_client, candidates, restricted, shock='centered'),
        'centered_honest': row(by_client, candidates, restricted, shock='centered',
                               decorr_rho=centered_rho),
        'raw_honest': row(by_client, candidates, restricted, decorr_rho=raw_rho),
    }
    return {
        'arm': arm, 'pairs': pairs,
        'raw_rho': raw_rho, 'centered_rho': centered_rho, 'rows': rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='./results/fedtdo')
    parser.add_argument('--arms', nargs='+', required=True)
    parser.add_argument('--decorr', type=float, default=0.365,
                        help='design-effect constant used by the honest rows; the '
                             'default is the worst residual measured across arms')
    parser.add_argument('--markdown', action='store_true')
    args = parser.parse_args()

    print(f'{"arm":<30}{"raw r":>8}{"centered":>10}{"pairs":>8}')
    print('-' * 56)
    measured = []
    for arm in args.arms:
        stats = measure(args.root, arm, restricted=True)
        measured.append(stats)
        print(f'{arm:<30}{stats["raw_rho"]:>8.3f}{stats["centered_rho"]:>10.3f}'
              f'{stats["pairs"]:>8}')

    for restricted in (True, False):
        label = ('on-policy: deploy restricted to the probed set'
                 if restricted else 'full pool: deploy may pick an unprobed arm')
        print(f'\n{label}')
        print(f'{"arm":<30}{"raw":>9}{"centered":>10}{"honest":>9}'
              f'{"raw+honest":>12}{"cover":>7}')
        print('-' * 78)
        for stats in measured:
            rows = stats['rows'] if restricted else measure(
                args.root, stats['arm'], restricted=False)['rows']
            print(f'{stats["arm"]:<30}'
                  f'{100 * rows["raw"]["relative_gain"]:>8.2f}%'
                  f'{100 * rows["centered"]["relative_gain"]:>9.2f}%'
                  f'{100 * rows["centered_honest"]["relative_gain"]:>8.2f}%'
                  f'{100 * rows["raw_honest"]["relative_gain"]:>11.2f}%'
                  f'{100 * rows["centered"]["coverage"]:>6.0f}%')

    if args.markdown:
        print('\n| arm | raw r | centered r | raw | centered | centered+honest |'
              ' raw+honest |')
        print('|---|---|---|---|---|---|---|')
        for stats in measured:
            rows = stats['rows']
            print(f'| `{stats["arm"]}` | {stats["raw_rho"]:.3f} | '
                  f'{stats["centered_rho"]:.3f} | '
                  f'{100 * rows["raw"]["relative_gain"]:+.2f}% | '
                  f'{100 * rows["centered"]["relative_gain"]:+.2f}% | '
                  f'{100 * rows["centered_honest"]["relative_gain"]:+.2f}% | '
                  f'{100 * rows["raw_honest"]["relative_gain"]:+.2f}% |')


if __name__ == '__main__':
    main()
