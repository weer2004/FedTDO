"""Turn ``results_paper/paper_backbone_long.csv`` into the paper's tables.

Two views:

* default -- per backbone: parameters, per-dataset MSE, core mean (8 datasets,
  ILI excluded), all-dataset mean, rank;
* ``--compare`` -- FedAvg vs local, i.e. how much cross-dataset parameter
  averaging *hurts* each backbone (negative transfer).

Usage::

    python scripts/xc_paper_backbone_table.py --md
    python scripts/xc_paper_backbone_table.py --md --compare
    python scripts/xc_paper_backbone_table.py --method local --pred-len 96
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results_paper'

DATASETS = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Exchange', 'ILI',
            'ECL', 'Traffic']
CORE = [d for d in DATASETS if d != 'ILI']


def load(path: Path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            r['mse'] = float(r['mse'])
            r['mae'] = float(r['mae'])
            r['pred_len'] = int(r['pred_len'])
            r['params'] = int(r['params']) if r.get('params') else 0
            rows.append(r)
    return rows


def as_table(rows):
    """{(method, backbone): {dataset: mse}}"""
    table = defaultdict(dict)
    params = {}
    for r in rows:
        key = (r['method'], r['backbone'])
        table[key][r['dataset']] = r['mse']
        params[key] = r['params']
    return table, params


def core_mean(per_ds):
    vals = [per_ds[d] for d in CORE if d in per_ds]
    return sum(vals) / len(vals) if vals else float('nan')


def print_single(rows, method, md):
    table, params = as_table(rows)
    keys = [k for k in table if k[0] == method]
    if not keys:
        raise SystemExit(f'no rows for method={method}')
    order = sorted(keys, key=lambda k: core_mean(table[k]))
    header = ['backbone', 'params'] + DATASETS + ['core_mean', 'all_mean', 'rank']
    if md:
        print('| ' + ' | '.join(header) + ' |')
        print('|' + '|'.join(['---'] * 2 + ['---:'] * (len(header) - 2)) + '|')
    for rank, key in enumerate(order, 1):
        per_ds = table[key]
        all_vals = [per_ds[d] for d in DATASETS if d in per_ds]
        cells = ([key[1], f'{params[key]:,}']
                 + [f'{per_ds[d]:.4f}' for d in DATASETS if d in per_ds]
                 + [f'{core_mean(per_ds):.4f}',
                    f'{sum(all_vals) / len(all_vals):.4f}', str(rank)])
        print(('| ' + ' | '.join(cells) + ' |') if md else '\t'.join(cells))

    wins = defaultdict(int)
    lines = []
    for d in DATASETS:
        cand = [(table[k][d], k[1]) for k in keys if d in table[k]]
        if not cand:
            continue
        best, bb = min(cand)
        wins[bb] += 1
        lines.append(f'{d}={bb}({best:.4f})')
    print()
    print('per-dataset best: ' + ', '.join(lines))
    print('win count: ' + ', '.join(f'{bb}:{n}' for bb, n in
                                    sorted(wins.items(), key=lambda kv: -kv[1])))
    best_core = min(core_mean(table[k]) for k in keys)
    print('excess over best (core mean): ' +
          ', '.join(f'{k[1]}: +{100 * (core_mean(table[k]) / best_core - 1):.2f}%'
                    for k in order))


def print_compare(rows, md):
    table, _ = as_table(rows)
    names = sorted({k[1] for k in table
                    if ('fedavg', k[1]) in table and ('local', k[1]) in table})
    if not names:
        raise SystemExit('need both fedavg and local collected')
    header = ['backbone', 'fedavg_core', 'local_core', 'delta'] + DATASETS
    if md:
        print('| ' + ' | '.join(header) + ' |')
        print('|' + '|'.join(['---', '---:'] + ['---:'] * (len(header) - 2)) + '|')
    for bb in sorted(names, key=lambda n: core_mean(table[('fedavg', n)]
                                                    if ('fedavg', n) in table else {})):
        fed = table[('fedavg', bb)]
        loc = table[('local', bb)]
        f_core, l_core = core_mean(fed), core_mean(loc)
        cells = [bb, f'{f_core:.4f}', f'{l_core:.4f}',
                 f'{100 * (f_core / l_core - 1):+.1f}%']
        for d in DATASETS:
            if d in fed and d in loc:
                cells.append(f'{100 * (fed[d] / loc[d] - 1):+.1f}%')
            else:
                cells.append('—')
        print(('| ' + ' | '.join(cells) + ' |') if md else '\t'.join(cells))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--method', default='fedavg')
    p.add_argument('--pred-len', type=int, default=96)
    p.add_argument('--csv', default=str(RESULTS / 'paper_backbone_long.csv'))
    p.add_argument('--md', action='store_true')
    p.add_argument('--compare', action='store_true',
                   help='FedAvg vs local per backbone (negative transfer)')
    args = p.parse_args()

    rows = [r for r in load(Path(args.csv)) if r['pred_len'] == args.pred_len]
    if args.compare:
        print_compare(rows, args.md)
    else:
        print_single(rows, args.method, args.md)


if __name__ == '__main__':
    main()
