"""Merge every FL baseline on disk into one coverage + score table.

Two sources, because the runs happened in two batches:

* ``results/fedtdo/_metrics/paper_main.csv`` -- the 10 earlier FL baselines plus
  ``local`` / ``fedavg`` and the FedTDO arms (channel-as-client, pl24);
* ``results_baselines/BL_<method>_xc_s<seed>_<dataset>_*/summary.json`` -- the 8
  federated-optimisation baselines (FedProx / SCAFFOLD / FedAdam / FedNova /
  FedPer / FedRep / Ditto / pFedMe).

``--coverage`` prints which cells exist; ``--md`` prints the score table.  A
missing cell is reported as ``—`` and never silently filled with a neighbouring
dataset's value.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Exchange', 'ILI',
            'ECL', 'Traffic']
CORE = [d for d in DATASETS if d != 'ILI']

# display order: the method under test first, then the strongest published
# competitors, then the folklore baselines
ORDER = ['FedTDO-center', 'FedTDO-raw', 'FedTDO-decorr',
         'ditto', 'fedrep', 'fedper', 'pfedme', 'fedprox', 'scaffold',
         'fedadam', 'fednova', 'fedala', 'pfedcs', 'fedkme', 'fedkwaz',
         'pfedfda', 'fedl2p', 'fedpoe', 'fedpkda',
         # 2024/2025 additions on the same channel protocol
         'fedgtp',  # FedGTP-style graph aggregation (fingerprint adjacency)
         'fedda',   # FedDA: fingerprint clustering + mixup augmentation
         'fedavg', 'local']

DIR_RE = re.compile(r'BL_(?P<method>[a-z0-9_]+?)_xc_s\d+_(?P<dataset>[A-Za-z0-9]+?)_')


def load_paper_main() -> dict:
    path = ROOT / 'results' / 'fedtdo' / '_metrics' / 'paper_main.csv'
    out: dict = {}
    if not path.exists():
        return out
    import csv
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            out.setdefault(row['method'], {})[row['dataset']] = float(row['mse'])
    return out


def load_baselines() -> dict:
    out: dict = {}
    # sorted by directory name so a tagged rerun (``..._lr1e-3_exp``) is visited
    # *after* the default cell and can replace it; a NaN never replaces a number
    for sdir in sorted(glob.glob(str(ROOT / 'results_baselines' / 'BL_*')),
                       key=lambda p: len(Path(p).name)):
        m = DIR_RE.match(Path(sdir).name)
        if not m:
            continue
        method, dataset = m.group('method'), m.group('dataset')
        if dataset not in DATASETS:
            continue
        spath = Path(sdir) / 'summary.json'
        if not spath.exists():
            continue
        with open(spath, encoding='utf-8') as f:
            try:
                summary = json.load(f)
            except json.JSONDecodeError:
                continue
        val = float(summary['mean_mse'])
        prev = out.setdefault(method, {}).get(dataset)
        # NaN means "diverged"; keep it only when there is nothing better, so the
        # table never silently shows a green tick for a run that produced no
        # number
        if prev is not None and math.isnan(val) and not math.isnan(prev):
            continue
        out[method][dataset] = val
    return out


def _fmt(c):
    if c is None:
        return '—'
    if isinstance(c, float):
        return 'NaN' if math.isnan(c) else f'{c:.4f}'
    return str(c)


def core_of(per_ds: dict):
    vals = [per_ds[d] for d in CORE
            if per_ds.get(d) is not None and not math.isnan(per_ds[d])]
    return sum(vals) / len(vals) if vals else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--md', action='store_true')
    p.add_argument('--coverage', action='store_true')
    args = p.parse_args()

    scores = load_baselines()
    for method, per_ds in load_paper_main().items():
        scores.setdefault(method, {}).update(per_ds)

    methods = [m for m in ORDER if m in scores]
    methods += sorted(m for m in scores if m not in ORDER)

    if args.coverage:
        print('| method | ' + ' | '.join(DATASETS) + ' | covered |')
        print('|' + '|'.join(['---'] + [':-:'] * (len(DATASETS) + 1)) + '|')
        for m in methods:
            cells = []
            for d in DATASETS:
                v = scores[m].get(d)
                cells.append('—' if v is None else ('NaN' if math.isnan(v) else '✓'))
            n = sum(c == '✓' for c in cells)
            print(f'| {m} | ' + ' | '.join(cells) + f' | {n}/9 |')
        return

    header = ['method'] + DATASETS + ['core', 'n']
    rows = []
    for m in methods:
        per = scores[m]
        cells = [per.get(d) for d in DATASETS]
        rows.append([m] + cells
                    + [core_of(per),
                       sum(c is not None and not math.isnan(c) for c in cells)])
    rows.sort(key=lambda r: (r[-1] < 9, r[-2] if r[-2] is not None else 9e9))

    if args.md:
        print('| ' + ' | '.join(header) + ' |')
        print('|' + '|'.join(['---'] + ['---:'] * (len(header) - 1)) + '|')
        for cells in rows:
            print('| ' + ' | '.join(_fmt(c) for c in cells) + ' |')
    else:
        print('\t'.join(header))
        for cells in rows:
            print('\t'.join('' if c is None else _fmt(c) for c in cells))


if __name__ == '__main__':
    main()
