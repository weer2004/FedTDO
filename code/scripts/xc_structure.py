"""Structural fingerprints of every dataset, computed offline from the logs.

The end-to-end table (``xc_table.py``) answers "does the rule help?".  This one
answers "*why*", by measuring on each dataset the four quantities the report
leans on, so that Traffic's values can be checked for representativeness:

1. **round shock** -- how big the per-round common component is, how persistent
   it is within a host, and how synchronised it is across hosts;
2. **rho** -- the residual (round-centred) within-arm autocorrelation at gap 1,
   i.e. exactly the number the ``decorr`` arm hard-codes as 0.365;
3. **transferable share** -- incremental R^2 of the arm main effect on
   (arm, host) cell means (§9 reports 4.7% for Traffic);
4. **oracle / capture** -- the best static arm per host, and how much of it the
   online rule actually banked.

Note on the two reward channels: ``raw_gains`` is the measured reward, ``gains``
is what the controller stores.  Under ``--fedtdo_shock centered`` the stored
value is ``d = raw - round_mean``, so the shock statistics must read
``raw_gains`` while ``rho`` must read ``gains``.

Nothing is retrained.

Usage::

    python scripts/xc_structure.py
    python scripts/xc_structure.py --markdown --arm decorr
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results' / 'fedtdo'
sys.path.insert(0, str(ROOT / 'scripts'))

from probe_cross_host import cells as cell_table, decompose  # noqa: E402

SEEDS = [2021, 2022, 2023, 2024, 2025]

# (name, nc, budget, is_dtmoe_benchmark); Solar is an external control only.
DATASETS = [
    ('ETTh1', 7, 2, True),
    ('ETTh2', 7, 2, True),
    ('ETTm1', 7, 2, True),
    ('ETTm2', 7, 2, True),
    ('Weather', 8, 2, True),
    ('Exchange', 8, 2, True),
    ('ILI', 7, 2, True),
    ('ECL', 32, 9, True),
    ('Traffic', 32, 9, True),
    ('Solar', 8, 2, False),
]

LEGACY = {('Traffic', 'base'): 'traffic32_s{seed}_joint_b9_fix'}


def run_dir(dataset, arm, seed, nc, budget):
    legacy = LEGACY.get((dataset, arm))
    name = legacy.format(seed=seed) if legacy else \
        f'{dataset.lower()}{nc}_s{seed}_b{budget}_{arm}'
    return RESULTS / name


def load(dataset, arm, seed, nc, budget):
    path = run_dir(dataset, arm, seed, nc, budget) / 'rounds.json'
    if not path.exists():
        return None
    return json.loads(path.read_text())


def group(records):
    by_host = collections.defaultdict(list)
    for record in records:
        by_host[record['client']].append(record)
    return by_host


def mean_by_round(records, key):
    """``round index -> host -> mean reward that host saw that round``.

    ``rounds.json`` stores a global counter in ``step`` (it advances once per
    host), so the round index has to come from each host's own order.
    """
    table = collections.defaultdict(lambda: collections.defaultdict(list))
    for host, items in group(records).items():
        for index, record in enumerate(items):
            values = list(record.get(key, {}).values())
            if values:
                table[index][host].append(float(np.mean(values)))
    return table


def shock_stats(records):
    table = mean_by_round(records, 'raw_gains')
    rounds = sorted(table)
    if len(rounds) < 10:
        return None
    per_host = collections.defaultdict(dict)
    round_means = []
    for index in rounds:
        cells = {host: float(np.mean(values)) for host, values in table[index].items()}
        if not cells:
            continue
        round_means.append(float(np.mean(list(cells.values()))))
        for host, value in cells.items():
            per_host[host][index] = value

    all_gains = np.array([float(gain) for record in records
                          for gain in record.get('raw_gains', {}).values()])
    if not len(all_gains):
        return None
    size = float(np.std(round_means) / (all_gains.std() + 1e-12))

    lags = []
    for series in per_host.values():
        steps = sorted(series)
        if len(steps) < 10:
            continue
        values = np.array([series[t] for t in steps])
        if values.std() > 1e-12:
            lags.append(float(np.corrcoef(values[:-1], values[1:])[0, 1]))

    hosts = [h for h in per_host if len(per_host[h]) >= 10]
    sync = float('nan')
    if len(hosts) >= 3:
        common = sorted(set.intersection(*[set(per_host[h]) for h in hosts]))
        if len(common) >= 10:
            matrix = np.array([[per_host[h][t] for t in common] for h in hosts])
            matrix = matrix[matrix.std(axis=1) > 1e-12]
            if len(matrix) >= 3:
                corr = np.corrcoef(matrix)
                upper = np.triu_indices(len(matrix), 1)
                sync = float(corr[upper].mean())

    return dict(size=size,
                lag1=statistics.fmean(lags) if lags else float('nan'),
                sync=sync)


def rho_stats(records):
    xs, ys = [], []
    for host, items in group(records).items():
        series = collections.defaultdict(list)
        for index, record in enumerate(items, start=1):
            gains = record.get('gains', {})
            if not gains:
                continue
            mean = float(np.mean(list(gains.values())))
            for arm, gain in gains.items():
                series[arm].append((index, float(gain) - mean))
        for points in series.values():
            points.sort()
            # gap 1 only: that is the pair that shares the round shock
            for (ix, x), (iy, y) in zip(points, points[1:]):
                if iy - ix == 1:
                    xs.append(x)
                    ys.append(y)
    if len(xs) < 50:
        return None
    x, y = np.array(xs), np.array(ys)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def oracle_capture(records):
    """Best static arm per host (in %), and the share actually banked."""
    table = cell_table(group(records))
    per_host = collections.defaultdict(list)
    for (arm, host), gains in table.items():
        if gains:
            per_host[host].append(100.0 * float(np.mean(gains)))
    if not per_host:
        return float('nan'), float('nan')
    oracle = statistics.fmean([max(v) for v in per_host.values() if len(v) >= 2])
    base = sum(record['default_loss'] for record in records)
    deployed = sum(record['deployed_loss'] for record in records)
    realised = 100.0 * (base - deployed) / max(base, 1e-12)
    return oracle, realised


def _adjust(r2, cells, params):
    """Adjusted R^2: with ~1 observation per cell the raw value is inflated by
    ``params / cells``, which matters because small datasets have few cells."""
    denom = cells - params - 1
    if denom <= 0:
        return float('nan')
    return 1.0 - (1.0 - r2) * (cells - 1) / denom


def variance_shares(records, key='gains'):
    """Per-seed incremental (adjusted) R^2, averaged over seeds.

    Pooling seeds would be wrong: each seed samples its own channels, so the
    pooled table has ~5x the hosts and every arm id is already seed-specific.
    """
    shares = []
    for payload in records:
        table = collections.defaultdict(list)
        for host, items in group(payload).items():
            for record in items:
                for arm, gain in record.get(key, {}).items():
                    table[(arm, host)].append(float(gain))
        stats = decompose(table, min_n=1)
        if not stats['cells']:
            continue
        cells, arms, hosts = stats['cells'], stats['arms'], stats['hosts']
        shares.append((_adjust(stats['r2_arm'], cells, arms),
                       _adjust(stats['r2_host'], cells, hosts),
                       stats['r2_both'], cells, hosts, arms))
    if not shares:
        return None
    return dict(
        arm=statistics.fmean(s[0] for s in shares),
        host=statistics.fmean(s[1] for s in shares),
        inter=statistics.fmean(1.0 - s[2] for s in shares),
        cells=shares[0][3], hosts=shares[0][4], arms=shares[0][5],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', default='decorr', choices=['base', 'center', 'decorr'])
    parser.add_argument('--markdown', action='store_true')
    parser.add_argument('--with-solar', action='store_true',
                        help='also fingerprint the Solar control')
    args = parser.parse_args()

    rows = []
    for dataset, nc, budget, is_bench in DATASETS:
        if not is_bench and not args.with_solar:
            continue
        records = [r for r in (load(dataset, args.arm, seed, nc, budget)
                               for seed in SEEDS) if r]
        if not records:
            continue
        shocks = [s for s in (shock_stats(r) for r in records) if s]
        rhos = [v for v in (rho_stats(r) for r in records) if v is not None]
        stats = variance_shares(records, 'raw_gains')
        oracles, realised = zip(*(oracle_capture(r) for r in records))
        oracle = statistics.fmean(oracles)
        cash = statistics.fmean(realised)
        rows.append(dict(
            dataset=dataset, seeds=len(records),
            size=statistics.fmean(s['size'] for s in shocks),
            lag1=statistics.fmean(s['lag1'] for s in shocks),
            sync=statistics.fmean(s['sync'] for s in shocks if s['sync'] == s['sync']),
            rho=statistics.fmean(rhos) if rhos else float('nan'),
            rho_lo=min(rhos) if rhos else float('nan'),
            rho_hi=max(rhos) if rhos else float('nan'),
            arm_share=stats['arm'], host_share=stats['host'],
            inter_share=stats['inter'],
            oracle=oracle, realised=cash,
            capture=(cash / oracle) if oracle > 0 else float('nan'),
            cells=stats['cells'], hosts=stats['hosts'], arms=stats['arms'],
        ))

    if args.markdown:
        print('| dataset | seeds | shock sd | shock lag-1 | cross-host | rho (gap 1) | '
              'arm R2 | host R2 | inter R2 | oracle | capture |')
        print('|' + '---|' * 11)
        for row in rows:
            print(f'| {row["dataset"]} | {row["seeds"]} | {row["size"]:.2f} | '
                  f'{row["lag1"]:.3f} | {row["sync"]:.3f} | '
                  f'{row["rho"]:.3f} [{row["rho_lo"]:.2f}, {row["rho_hi"]:.2f}] | '
                  f'{row["arm_share"]:.3f} | {row["host_share"]:.3f} | '
                  f'{row["inter_share"]:.3f} | {row["oracle"]:+.2f}% | '
                  f'{row["capture"]:.2f} |')
    else:
        head = (f'{"dataset":<9}{"seeds":>6}{"shockSD":>9}{"lag1":>7}{"sync":>7}'
                f'{"rho":>16}{"armR2":>8}{"hostR2":>8}{"interR2":>9}'
                f'{"oracle":>9}{"capture":>9}')
        print(head)
        print('-' * len(head))
        for row in rows:
            print(f'{row["dataset"]:<9}{row["seeds"]:>6}{row["size"]:>9.2f}'
                  f'{row["lag1"]:>7.3f}{row["sync"]:>7.3f}'
                  f'{row["rho"]:>7.3f}[{row["rho_lo"]:.2f},{row["rho_hi"]:.2f}]'
                  f'{row["arm_share"]:>8.3f}{row["host_share"]:>8.3f}'
                  f'{row["inter_share"]:>9.3f}{row["oracle"]:>+8.2f}%{row["capture"]:>9.2f}')
        print('\nshockSD  = sd(cross-host round mean) / sd(all rewards)')
        print('lag1     = within-host lag-1 autocorrelation of the round mean')
        print('sync     = mean pairwise host correlation of the round mean')
        print('rho      = within-arm autocorrelation of the centred reward, gap 1')
        print('armR2    = incremental R^2 of the arm main effect (transferable share)')
        print('capture  = realised gain / best static arm per host')


if __name__ == '__main__':
    main()
