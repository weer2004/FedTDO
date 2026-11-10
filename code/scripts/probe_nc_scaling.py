"""Would more clients help?  Answer it from the existing 32-client logs first.

The candidate pool grows linearly with the client count (``4·nc − 4`` external
experts) while the per-client probe budget does not, so going 32 → 64 halves the
number of observations per arm.  The two effects push in opposite directions:

* more clients  -> more donors -> a higher hindsight oracle (order statistics);
* fewer obs/arm -> a radius that cannot separate arms -> a lower capture rate.

This script isolates the trade-off with **no retraining**: draw random subsets of
K clients from the 32-client log, drop the arms whose host did not survive, and
replay the de-correlated rule on the subset.  K < 32 is the same experiment run
backwards: a smaller pool with *more* observations per arm.

Caveat: absolute MSE is not comparable across K (the budget is coupled to the
pool size), so everything is reported as a relative gain *within* the subset,
plus the oracle of that same subset.

Usage::

    python scripts/probe_nc_scaling.py --subsets 8
"""

from __future__ import annotations

import argparse
import collections
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import probe_timing as timing  # noqa: E402
from federated import FedTDOController  # noqa: E402
from replay_selector import arm_ids, load_rounds  # noqa: E402

BASE = dict(beta=1.0, min_observations=1, estimator='mean', decision='lcb',
            shrink_n0=0.0, shrink_group='global')
DECORR = 0.365
SEEDS = [2021, 2022, 2023, 2024, 2025]


def host_of(arm_id):
    return arm_id.split(':', 1)[0]


def restrict_to(by_client, keep):
    """Drop clients outside ``keep`` and the arms they published."""
    kept = {}
    for host, records in by_client.items():
        if host not in keep:
            continue
        trimmed = []
        for record in records:
            allowed = {a for a in record['gains'] if host_of(a) in keep}
            if not allowed:
                continue
            copy = dict(record)
            copy['gains'] = {a: record['gains'][a] for a in allowed}
            if 'raw_gains' in record:
                copy['raw_gains'] = {a: v for a, v in record['raw_gains'].items()
                                     if a in allowed}
            copy['probe_ids'] = [a for a in record.get('probe_ids', []) if a in allowed]
            trimmed.append(copy)
        if trimmed:
            kept[host] = trimmed
    return kept


def oracle(by_client):
    """Hindsight best single external arm per host, in relative-gain units."""
    total_default = total_best = 0.0
    for host, records in by_client.items():
        per_arm = collections.defaultdict(float)
        default = 0.0
        for record in records:
            default += record['default_loss']
            for arm, gain in record['gains'].items():
                per_arm[arm] += record['default_loss'] * float(gain)
        if not per_arm or default <= 0:
            continue
        total_default += default
        total_best += max(per_arm.values())
    return total_best / max(total_default, 1e-12)


def thin_probes(by_client, keep, rng):
    """Keep only a fraction of each round's probes.

    This is the *other* half of the client-count trade-off: at a fixed per-client
    budget, doubling the number of donors halves the observations per arm.  The
    real run cannot be retrained here, but dropping probes reproduces the same
    per-arm evidence regime on the logged rewards (the action space shrinks, the
    recorded statistics shrink with it), which is exactly what the controller
    sees when the pool doubles.
    """
    if keep >= 1.0:
        return by_client
    out = {}
    for host, records in by_client.items():
        trimmed = []
        for record in records:
            probes = list(record.get('probe_ids', []))
            count = max(1, int(round(keep * len(probes))))
            chosen = set(rng.sample(probes, count)) if probes else set()
            chosen |= {a for a in record['gains'] if a not in probes}
            copy = dict(record)
            copy['gains'] = {a: v for a, v in record['gains'].items() if a in chosen}
            if 'raw_gains' in record:
                copy['raw_gains'] = {a: v for a, v in record['raw_gains'].items()
                                     if a in chosen}
            copy['probe_ids'] = [a for a in probes if a in chosen]
            trimmed.append(copy)
        out[host] = trimmed
    return out


def stats_for(by_client):
    candidates = arm_ids(by_client)
    row = timing.replay(by_client, candidates, FedTDOController, restrict=True,
                        shock='centered', decorr_rho=DECORR, **BASE)
    pool = float(np.mean([len(v) for v in candidates.values()]))
    evaluations = sum(len(record['gains']) for records in by_client.values()
                      for record in records)
    return {
        'gain': 100.0 * row['relative_gain'],
        'oracle': 100.0 * oracle(by_client),
        'deploys': row['deploys'],
        'precision': 100.0 * row['precision'],
        'coverage': 100.0 * row['coverage'],
        'clients': len(by_client),
        'pool': pool,
        'obs_per_arm': evaluations / max(pool * len(by_client), 1.0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='./results/fedtdo')
    parser.add_argument('--suffix', default='b9_center')
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    parser.add_argument('--sizes', type=int, nargs='+', default=[8, 16, 24, 32])
    parser.add_argument('--subsets', type=int, default=8)
    parser.add_argument('--probe-keep', type=float, nargs='+', default=[],
                        help='simulate a bigger pool at a fixed budget by handing '
                             'each round only this fraction of its probes')
    parser.add_argument('--rng', type=int, default=7)
    args = parser.parse_args()

    rows = collections.defaultdict(list)
    for seed in args.seeds:
        path = os.path.join(args.root, f'traffic32_s{seed}_{args.suffix}', 'rounds.json')
        if not os.path.exists(path):
            continue
        by_client = load_rounds(path)
        hosts = sorted(by_client)
        rng = random.Random(args.rng + seed)
        for size in args.sizes:
            if size > len(hosts):
                continue
            draws = [hosts] if size == len(hosts) else [
                rng.sample(hosts, size) for _ in range(args.subsets)]
            for draw in draws:
                rows[size].append(stats_for(restrict_to(by_client, set(draw))))

    print(f'{"nc":>4}{"pool":>7}{"obs/arm":>9}{"gain":>9}{"oracle":>9}'
          f'{"capture":>9}{"deploys":>9}{"prec":>8}{"cover":>8}{"draws":>7}')
    print('-' * 72)
    for size in args.sizes:
        sample = rows.get(size)
        if not sample:
            continue
        mean = lambda key: float(np.mean([item[key] for item in sample]))
        gain, orc = mean('gain'), mean('oracle')
        print(f'{size:>4}{mean("pool"):>7.0f}{mean("obs_per_arm"):>9.2f}'
              f'{gain:>8.2f}%{orc:>8.2f}%{gain / orc if orc else float("nan"):>9.2f}'
              f'{mean("deploys"):>9.0f}{mean("precision"):>7.1f}%'
              f'{mean("coverage"):>7.1f}%{len(sample):>7}')

    print('\nper-seed gain (single draw at each size, same subset seed)')
    header = f'{"seed":<8}' + ''.join(f'{"nc=" + str(s):>10}' for s in args.sizes)
    print(header)
    print('-' * len(header))
    for seed in args.seeds:
        path = os.path.join(args.root, f'traffic32_s{seed}_{args.suffix}', 'rounds.json')
        if not os.path.exists(path):
            continue
        by_client = load_rounds(path)
        hosts = sorted(by_client)
        rng = random.Random(args.rng + seed)
        cells = []
        for size in args.sizes:
            draw = set(hosts if size >= len(hosts) else rng.sample(hosts, size))
            cells.append(f'{stats_for(restrict_to(by_client, draw))["gain"]:+9.2f}%')
        print(f'{seed:<8}' + ''.join(cells))

    if args.probe_keep:
        print('\nevidence-per-arm axis (32 donors, probes thinned per round)\n')
        print(f'{"keep":>6}{"probes/round":>14}{"obs/arm":>9}{"gain":>9}'
              f'{"oracle":>9}{"capture":>9}{"deploys":>9}{"prec":>8}')
        print('-' * 73)
        for keep in args.probe_keep:
            sample = []
            for seed in args.seeds:
                path = os.path.join(args.root, f'traffic32_s{seed}_{args.suffix}',
                                    'rounds.json')
                if not os.path.exists(path):
                    continue
                rng = random.Random(args.rng + seed)
                thinned = thin_probes(load_rounds(path), keep, rng)
                sample.append(stats_for(thinned))
            if not sample:
                continue
            mean = lambda key: float(np.mean([item[key] for item in sample]))
            gain, orc = mean('gain'), mean('oracle')
            probes = 9.0 * keep
            print(f'{keep:>6.2f}{probes:>14.1f}{mean("obs_per_arm"):>9.2f}'
                  f'{gain:>8.2f}%{orc:>8.2f}%'
                  f'{gain / orc if orc else float("nan"):>9.2f}'
                  f'{mean("deploys"):>9.0f}{mean("precision"):>7.1f}%')


if __name__ == '__main__':
    main()
