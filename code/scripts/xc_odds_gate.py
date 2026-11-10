"""Does a short odds pilot decide whether to admit at all?

Every section of the report ends on the same sentence: the sign of the deployment
gain is set by the *odds* of a bet (mean win vs mean loss), not by how strong the
signal is.  If that is true, it should also be **actionable**: run the rule for a
few rounds, read the win/loss magnitudes off the probe log, and switch the rule
off when the implied breakeven precision is above the realised one.

This script does exactly that, offline, on the 60-round logs:

1. split each client's stream into a **pilot** (first ``--pilot`` rounds) and a
   **holdout** (the rest);
2. from the pilot, estimate precision ``p``, mean win ``w``, mean loss ``l`` and
   the breakeven precision ``-l / (w - l)``;
3. decide ``enable = p > breakeven`` (optionally with a margin), and score the
   holdout with the rule on if enabled, or as "no transfer" (gain 0) if not;
4. report, per dataset, mean gain for always-off / always-on / gated.

"Always-off" is the baseline rule, "always-on" is the ``decorr`` arm, and
"gated" is what the same logs would have produced with the pilot in the loop.

Usage::

    python scripts/xc_odds_gate.py
    python scripts/xc_odds_gate.py --pilot 20 --margin 0.05 --markdown
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
DATASETS = [('ETTh1', 7, 2), ('ETTh2', 7, 2), ('ETTm1', 7, 2), ('ETTm2', 7, 2),
            ('Weather', 8, 2), ('Exchange', 8, 2), ('ILI', 7, 2),
            ('ECL', 32, 9), ('Traffic', 32, 9)]
LEGACY = {('Traffic', 'base'): 'traffic32_s{seed}_joint_b9_fix'}


def run_dir(dataset, arm, seed, nc, budget):
    legacy = LEGACY.get((dataset, arm))
    name = legacy.format(seed=seed) if legacy else \
        f'{dataset.lower()}{nc}_s{seed}_b{budget}_{arm}'
    return RESULTS / name


def load_rounds(directory: Path):
    path = directory / 'rounds.json'
    if not path.exists():
        return None
    return json.loads(path.read_text())


def split_by_round(records, pilot):
    """(pilot records, holdout records) using each client's own round order."""
    by_client = collections.defaultdict(list)
    for record in records:
        by_client[record['client']].append(record)
    early, late = [], []
    for items in by_client.values():
        items.sort(key=lambda r: r['step'])
        early.extend(items[:pilot])
        late.extend(items[pilot:])
    return early, late


def odds(records):
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
        return None
    precision = len(wins) / total
    win = statistics.fmean(wins) if wins else 0.0
    loss = statistics.fmean(losses) if losses else 0.0
    spread = win - loss
    if spread <= 0 or not losses:
        breakeven = float('-inf')     # cannot lose, or wins dominate: always admit
    else:
        breakeven = -loss / spread
    return dict(bets=total, precision=precision, win=win, loss=loss,
                breakeven=breakeven)


def gain_of(records):
    base = sum(r['default_loss'] for r in records)
    deployed = sum(r['deployed_loss'] for r in records)
    return 100.0 * (base - deployed) / max(base, 1e-12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', type=int, default=15)
    parser.add_argument('--margin', type=float, default=0.0)
    parser.add_argument('--markdown', action='store_true')
    args = parser.parse_args()

    print(f'pilot = first {args.pilot} rounds, margin = {args.margin:+.2f}')
    head = (f'{"dataset":<9}{"off":>8}{"on":>8}{"gated":>8}{"delta":>8}'
            f'{"pilot prec":>11}{"pilot BE":>9}{"gate on":>9}')
    print(head)
    print('-' * len(head))

    totals = {'off': [], 'on': [], 'gated': []}
    for dataset, nc, budget in DATASETS:
        off, on, gated, precs, bes, enables = [], [], [], [], [], []
        for seed in SEEDS:
            base = load_rounds(run_dir(dataset, 'base', seed, nc, budget))
            decorr = load_rounds(run_dir(dataset, 'decorr', seed, nc, budget))
            if base is None or decorr is None:
                continue
            # every column is scored on the *holdout* only, so the pilot cannot
            # be counted as a win by any of them
            _, base_late = split_by_round(base, args.pilot)
            early, late = split_by_round(decorr, args.pilot)
            off.append(gain_of(base_late))
            on.append(gain_of(late))
            stats = odds(early)
            if stats is None:
                # no bets in the pilot: nothing was tried, nothing was gained
                gated.append(0.0)
                enables.append(False)
                continue
            enable = stats['precision'] > stats['breakeven'] + args.margin
            enables.append(enable)
            precs.append(100 * stats['precision'])
            bes.append(100 * stats['breakeven'] if stats['breakeven'] != float('-inf')
                       else float('nan'))
            gated.append(gain_of(late) if enable else 0.0)
        if not on:
            continue
        totals['off'].extend(off)
        totals['on'].extend(on)
        totals['gated'].extend(gated)
        print(f'{dataset:<9}{statistics.fmean(off):>+8.2f}{statistics.fmean(on):>+8.2f}'
              f'{statistics.fmean(gated):>+8.2f}'
              f'{statistics.fmean(gated) - statistics.fmean(on):>+8.2f}'
              f'{(statistics.fmean(precs) if precs else float("nan")):>10.1f}%'
              f'{(statistics.fmean(bes) if bes else float("nan")):>8.1f}%'
              f'{sum(enables):>6}/{len(enables)}')

    print()
    print(f'{"pooled":<9}{statistics.fmean(totals["off"]):>+8.2f}'
          f'{statistics.fmean(totals["on"]):>+8.2f}'
          f'{statistics.fmean(totals["gated"]):>+8.2f}'
          f'{statistics.fmean(totals["gated"]) - statistics.fmean(totals["on"]):>+8.2f}')
    print(f'  over {len(totals["on"])} (dataset, seed) cells')
    print('  off   = baseline rule')
    print('  on    = decorr arm, no gate')
    print('  gated = holdout rounds scored with the pilot decision; pilot rounds '
          'are excluded from all three columns')


if __name__ == '__main__':
    main()
