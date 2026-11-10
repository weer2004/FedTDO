"""Cross-dataset table for the three admission rules.

Reads ``results/fedtdo/<run>/metrics.json`` (+ ``rounds.json`` for the betting
diagnostics) and prints, per dataset:

1. relative deployment gain for every seed, with mean / median / min / positives;
2. the pooled betting structure (bets, precision, breakeven, EV per bet);
3. the residual autocorrelation rho (the number the decorr arm hard-codes);
4. one horizontal row per dataset so the six datasets can be read side by side.

The Traffic grid predates this sweep and uses ``joint_b9_fix`` as its base arm,
so the arm lookup falls back to that name.

Usage::

    python scripts/xc_table.py
    python scripts/xc_table.py --markdown
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results' / 'fedtdo'

SEEDS = [2021, 2022, 2023, 2024, 2025]
ARMS = [('base', 'baseline (raw reward, iid radius)'),
        ('center', 'centered (paired reward, iid radius)'),
        ('decorr', 'decorr (paired + design-effect radius)')]

# (name, nc, budget, is_dtmoe_benchmark)
#
# The nine Dynamic TMoE benchmarks are the main grid.  Solar-Energy is an
# external control: it is not one of DTMoe's datasets, and its per-bet odds are
# inverted, so it answers a different question ("what happens when the pool's
# own expectation is negative") rather than "is the structure Traffic-specific".
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

# Traffic base arm is the pre-sweep directory name.
LEGACY = {('Traffic', 'base'): 'traffic32_s{seed}_joint_b9_fix'}


def run_dir(dataset: str, arm: str, seed: int, nc: int, budget: int) -> Path:
    standard = RESULTS / f'{dataset.lower()}{nc}_s{seed}_b{budget}_{arm}'
    if standard.exists():
        return standard
    legacy = LEGACY.get((dataset, arm))
    if legacy:
        return RESULTS / legacy.format(seed=seed)
    return standard


def gain_of(directory: Path):
    path = directory / 'metrics.json'
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    actions = payload.get('actions') or {}
    if 'relative_gain' not in actions:
        return None
    return dict(gain=100.0 * actions['relative_gain'],
                transfers=actions.get('transfers'),
                evaluations=actions.get('evaluations'),
                active=actions.get('clients_with_active_expert'),
                mse=payload.get('mse'))


def betting(directory: Path):
    path = directory / 'rounds.json'
    if not path.exists():
        return None
    records = json.loads(path.read_text())
    wins, losses = [], []
    for record in records:
        if not record.get('active_expert'):
            continue
        base = record['default_loss']
        if not base:
            continue
        change = (base - record['deployed_loss']) / base
        (wins if change > 0 else losses).append(change)
    total = len(wins) + len(losses)
    if not total:
        return dict(bets=0)
    precision = len(wins) / total
    mean_win = statistics.fmean(wins) if wins else 0.0
    mean_loss = statistics.fmean(losses) if losses else 0.0
    spread = mean_win - mean_loss
    breakeven = -mean_loss / spread if spread > 0 else float('nan')
    ev = precision * mean_win + (1 - precision) * mean_loss
    return dict(bets=total, precision=100 * precision, win=mean_win,
                loss=mean_loss, breakeven=100 * breakeven, ev=ev)


def rho_of(directory: Path, gap: int = 1):
    """Within-arm autocorrelation of the round-centred reward, consecutive probes."""
    path = directory / 'rounds.json'
    if not path.exists():
        return None
    records = json.loads(path.read_text())
    by_host = collections.defaultdict(list)
    for record in records:
        by_host[record['client']].append(record)

    cells = collections.defaultdict(list)
    for host, items in by_host.items():
        for index, record in enumerate(items, start=1):
            values = list(record.get('gains', {}).values())
            if not values:
                continue
            mean = sum(values) / len(values)
            for arm, gain in record['gains'].items():
                cells[(host, arm)].append((index, gain - mean))
    xs, ys = [], []
    for series in cells.values():
        series.sort()
        for (ix, x), (iy, y) in zip(series, series[1:]):
            if iy - ix <= gap:
                xs.append(x)
                ys.append(y)
    if len(xs) < 30:
        return None
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    varx = sum((a - mx) ** 2 for a in xs) ** 0.5
    vary = sum((b - my) ** 2 for b in ys) ** 0.5
    if varx * vary == 0:
        return None
    return cov / (varx * vary), n


def sign_test(diffs):
    """Two-sided exact sign test p-value (ties dropped)."""
    wins = sum(1 for d in diffs if d > 0)
    losses = sum(1 for d in diffs if d < 0)
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(wins, losses) + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def paired_rows(datasets=None):
    """Per-dataset paired differences between the three rules."""
    pairs = [('center', 'base', 'centered - baseline'),
             ('decorr', 'center', 'decorr - centered'),
             ('decorr', 'base', 'decorr - baseline')]
    print('paired differences (pp), same seed, same data')
    print(f'{"dataset":<9}{"comparison":<22}{"diffs":<34}{"mean":>8}{"wins":>7}{"p":>8}')
    print('-' * 88)
    pooled = collections.defaultdict(list)
    for dataset, nc, budget, is_bench in (datasets or DATASETS):
        for left, right, label in pairs:
            diffs = []
            for seed in SEEDS:
                a = gain_of(run_dir(dataset, left, seed, nc, budget))
                b = gain_of(run_dir(dataset, right, seed, nc, budget))
                if a is not None and b is not None:
                    diffs.append(a['gain'] - b['gain'])
            if not diffs:
                continue
            if is_bench:
                pooled[(left, right)].extend(diffs)
            wins = sum(1 for d in diffs if d > 0)
            print(f'{dataset:<9}{label:<22}'
                  f'{" ".join(f"{d:+.2f}" for d in diffs):<34}'
                  f'{statistics.fmean(diffs):>+8.2f}{wins:>4}/{len(diffs)}'
                  f'{sign_test(diffs):>8.3f}')
    if pooled:
        print('-' * 88)
        label = f'{len([d for d in (datasets or DATASETS) if d[3]])} benchmarks'
        for (left, right), diffs in pooled.items():
            wins = sum(1 for d in diffs if d > 0)
            print(f'{label:<9}{right + " -> " + left:<22}'
                  f'{" ":<34}{statistics.fmean(diffs):>+8.2f}'
                  f'{wins:>4}/{len(diffs)}{sign_test(diffs):>8.3f}')
    print()


def main():
    global SEEDS
    parser = argparse.ArgumentParser()
    parser.add_argument('--markdown', action='store_true')
    parser.add_argument('--paired', action='store_true')
    parser.add_argument('--with-solar', action='store_true',
                        help='also print the Solar control (not a DTMoe benchmark)')
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS,
                        help='override the seed list (e.g. fresh non-adjacent seeds)')
    args = parser.parse_args()
    SEEDS = args.seeds
    md = args.markdown
    grid = [row for row in DATASETS if row[3] or args.with_solar]

    if args.paired:
        paired_rows(grid)
        return

    horizontal = []
    table = collections.defaultdict(dict)
    for dataset, nc, budget, is_bench in grid:
        pool = 4 * nc - 4
        obs = 60 * budget / pool
        header = f'{dataset}  (nc={nc}, pool={pool}, obs/arm={obs:.2f})'
        print(('## ' if md else '') + header)
        if not md:
            print('-' * len(header))
        if md:
            print()
            print('| rule | ' + ' | '.join(str(s) for s in SEEDS) +
                  ' | mean | median | min | pos | transfers |')
            print('|---' * (len(SEEDS) + 6) + '|')

        per_arm = {}
        for arm, label in ARMS:
            values, transfers, missing = [], 0, 0
            for seed in SEEDS:
                info = gain_of(run_dir(dataset, arm, seed, nc, budget))
                if info is None:
                    values.append(None)
                    missing += 1
                else:
                    values.append(info['gain'])
                    transfers += info['transfers'] or 0
            present = [v for v in values if v is not None]
            if not present:
                continue
            row = dict(arm=arm, values=values, mean=statistics.fmean(present),
                       median=statistics.median(present), low=min(present),
                       pos=sum(1 for v in present if v > 0), n=len(present),
                       transfers=transfers / len(present), missing=missing)
            per_arm[arm] = row
            table[dataset][arm] = row

            if md:
                cells = [(f'{v:+.2f}' if v is not None else 'n/a') for v in values]
                print(f'| {label} | ' + ' | '.join(cells) +
                      f' | {row["mean"]:+.2f} | {row["median"]:+.2f} | '
                      f'{row["low"]:+.2f} | {row["pos"]}/{row["n"]} | {transfers/len(present):.0f} |')
            else:
                cells = [(f'{v:+7.2f}' if v is not None else '    n/a') for v in values]
                print(f'  {label:<42} ' + ' '.join(cells) +
                      f'   mean {row["mean"]:+6.2f}  med {row["median"]:+6.2f}  '
                      f'min {row["low"]:+6.2f}  pos {row["pos"]}/{row["n"]}  '
                      f'xfers {transfers/len(present):.0f}')

        if not per_arm:
            print('  (no runs yet)\n')
            continue

        # betting structure and rho, from the decorr arm when present
        reference = per_arm.get('decorr') or per_arm.get('center') or per_arm.get('base')
        arm_name = reference['arm']
        bets, rhos = [], []
        for seed in SEEDS:
            directory = run_dir(dataset, arm_name, seed, nc, budget)
            info = betting(directory)
            if info and info.get('bets'):
                bets.append(info)
            value = rho_of(directory)
            if value:
                rhos.append(value[0])
        if rhos:
            mean_rho = statistics.fmean(rhos)
            print(f'  rho (centred, gap=1) over {arm_name}: mean {mean_rho:.3f} '
                  f'range {min(rhos):.3f}-{max(rhos):.3f}')
        if bets:
            total = sum(b['bets'] for b in bets)
            precision = sum(b['precision'] * b['bets'] for b in bets) / total
            ev = sum(b['ev'] * b['bets'] for b in bets) / total
            win = sum(b['win'] * b['bets'] for b in bets) / total
            loss = sum(b['loss'] * b['bets'] for b in bets) / total
            spread = win - loss
            horizontal.append(dict(dataset=dataset, arm=arm_name, n=reference['n'],
                                   mean=reference['mean'], median=reference['median'],
                                   low=reference['low'], pos=reference['pos'],
                                   transfers=reference['transfers'],
                                   bets=total, precision=precision, ev=ev,
                                   win=win, loss=loss,
                                   breakeven=100 * (-loss / spread) if spread > 0
                                   else float('nan'),
                                   rho=statistics.fmean(rhos) if rhos else float('nan'),
                                   obs=obs))
        print()

    if horizontal:
        title = 'cross-dataset summary (decorr arm)'
        print(('## ' if md else '') + title)
        if md:
            print()
            print('| dataset | obs/arm | rho | mean | median | min | pos | bets | '
                  'win | loss | breakeven | precision | EV/bet |')
            print('|---' * 13 + '|')
            for row in horizontal:
                print(f'| {row["dataset"]} | {row["obs"]:.2f} | {row["rho"]:.3f} | '
                      f'{row["mean"]:+.2f} | {row["median"]:+.2f} | {row["low"]:+.2f} | '
                      f'{row["pos"]}/{row["n"]} | {row["bets"]} | '
                      f'{row["win"]:+.3f} | {row["loss"]:+.3f} | '
                      f'{row["breakeven"]:.1f}% | {row["precision"]:.1f}% | '
                      f'{row["ev"]:+.4f} |')
        else:
            print('-' * 120)
            print(f'{"dataset":<9}{"obs/arm":>8}{"rho":>7}{"mean":>8}{"median":>8}'
                  f'{"min":>8}{"pos":>7}{"bets":>7}{"win":>8}{"loss":>8}'
                  f'{"breakev":>9}{"prec":>8}{"EV":>9}')
            for row in horizontal:
                print(f'{row["dataset"]:<9}{row["obs"]:>8.2f}{row["rho"]:>7.3f}'
                      f'{row["mean"]:>+8.2f}{row["median"]:>+8.2f}{row["low"]:>+8.2f}'
                      f'{row["pos"]:>4}/{row["n"]}{row["bets"]:>7}'
                      f'{row["win"]:>+8.3f}{row["loss"]:>+8.3f}'
                      f'{row["breakeven"]:>8.1f}%{row["precision"]:>7.1f}%'
                      f'{row["ev"]:>+9.4f}')

    # does the odds margin predict the sign of the gain?
    usable = [row for row in horizontal
              if row['breakeven'] == row['breakeven'] and row['ev'] == row['ev']]
    if len(usable) >= 4:
        xs = [row['precision'] - row['breakeven'] for row in usable]
        ys = [row['mean'] for row in usable]
        mx, my = statistics.fmean(xs), statistics.fmean(ys)
        cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        vx = sum((a - mx) ** 2 for a in xs) ** 0.5
        vy = sum((b - my) ** 2 for b in ys) ** 0.5
        r = cov / (vx * vy) if vx * vy else float('nan')
        agree = sum(1 for a, b in zip(xs, ys) if (a > 0) == (b > 0))
        print()
        print(f'odds margin (precision - breakeven) vs mean gain over {len(usable)} datasets: '
              f'r = {r:+.3f}, sign agreement {agree}/{len(usable)}')

    benchmarks = [name for name, _, _, bench in grid if bench]
    rows = []
    for arm, label in ARMS:
        units, per_dataset = [], []
        for name in benchmarks:
            row = table.get(name, {}).get(arm)
            if row is None:
                continue
            units.extend(v for v in row['values'] if v is not None)
            per_dataset.append(row['mean'])
        if not units:
            continue
        rows.append(dict(arm=arm, label=label, units=units, per_dataset=per_dataset))
    if rows:
        n_bench = len(benchmarks)
        title = f'DTMoe benchmark set ({n_bench} datasets)'
        print(('## ' if md else '') + title)
        if md:
            print()
            print('| rule | unit mean | unit median | units > 0 | dataset mean | '
                  'dataset median | datasets > 0 |')
            print('|---' * 7 + '|')
            for row in rows:
                print(f'| {row["label"]} | {statistics.fmean(row["units"]):+.2f} | '
                      f'{statistics.median(row["units"]):+.2f} | '
                      f'{sum(1 for v in row["units"] if v > 0)}/{len(row["units"])} | '
                      f'{statistics.fmean(row["per_dataset"]):+.2f} | '
                      f'{statistics.median(row["per_dataset"]):+.2f} | '
                      f'{sum(1 for v in row["per_dataset"] if v > 0)}/'
                      f'{len(row["per_dataset"])} |')
        else:
            print('-' * 92)
            print(f'{"rule":<42}{"unit mean":>10}{"unit med":>9}{"pos":>7}'
                  f'{"ds mean":>9}{"ds med":>8}{"ds pos":>8}')
            for row in rows:
                print(f'{row["label"]:<42}'
                      f'{statistics.fmean(row["units"]):>+10.2f}'
                      f'{statistics.median(row["units"]):>+9.2f}'
                      f'{sum(1 for v in row["units"] if v > 0):>4}/{len(row["units"])}'
                      f'{statistics.fmean(row["per_dataset"]):>+9.2f}'
                      f'{statistics.median(row["per_dataset"]):>+8.2f}'
                      f'{sum(1 for v in row["per_dataset"] if v > 0):>5}/'
                      f'{len(row["per_dataset"])}')


if __name__ == '__main__':
    main()
