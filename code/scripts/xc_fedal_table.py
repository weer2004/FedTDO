"""Fold ``results_fedal/*/metrics.json`` into one markdown/TSV row.

FeDaL-style runs are *not* in ``results_paper/paper_backbone_mse_wide.csv``:
they have their own phase split (federated masked pretraining -> per-dataset
supervised fine-tuning) and land under ``results_fedal/``.  This script prints
them in the same column order as `PAPER_PROTOCOL_BACKBONES.md` 表 A/B so the rows
can be pasted next to ``fedavg+ffts`` without hand-copying numbers.

Usage::

    python scripts/xc_fedal_table.py                 # markdown
    python scripts/xc_fedal_table.py --tsv
    python scripts/xc_fedal_table.py --run fedal_pl96_c7
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Exchange', 'ILI',
            'ECL', 'Traffic']
CORE = [d for d in DATASETS if d != 'ILI']


def load(run_dir: Path) -> dict:
    with open(run_dir / 'metrics.json', encoding='utf-8') as f:
        blob = json.load(f)
    return {r['dataset']: r for r in blob['per_dataset']}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=ROOT / 'results_fedal')
    p.add_argument('--run', type=str, default='')
    p.add_argument('--tsv', action='store_true')
    p.add_argument('--md', action='store_true')
    args = p.parse_args()

    runs = sorted(d for d in args.root.glob('*/') if (d / 'metrics.json').exists())
    if args.run:
        runs = [d for d in runs if d.name == args.run]
    if not runs:
        print(f'no metrics.json under {args.root}')
        return

    header = ['run'] + DATASETS + ['core', 'n']
    rows = []
    for d in runs:
        per = load(d)
        cells = [per.get(k, {}).get('mse') for k in DATASETS]
        core = [c for k, c in zip(DATASETS, cells) if k in CORE and c is not None]
        rows.append([d.name] + cells
                    + [sum(core) / len(core) if core else None,
                       sum(c is not None for c in cells)])

    fmt = (lambda v: '—' if v is None else
           (f'{v:.4f}' if isinstance(v, float) else str(v)))
    if args.tsv:
        print('\t'.join(header))
        for r in rows:
            print('\t'.join(fmt(c) for c in r))
        return

    print('| ' + ' | '.join(header) + ' |')
    print('|' + '|'.join(['---'] + ['---:'] * (len(header) - 1)) + '|')
    for r in rows:
        print('| ' + ' | '.join(fmt(c) for c in r) + ' |')


if __name__ == '__main__':
    main()
