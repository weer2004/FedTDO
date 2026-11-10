"""One table: FedTDO (full method) vs every backbone, same protocol.

Rows come from two directories:

* ``results_paper/paper_backbone_mse_wide.csv`` -- ``local`` and ``fedavg`` for
  the 8 backbones, written by ``scripts/xc_paper_backbone_table.py``;
* ``results_paper_fedtdo/PPF_*/per_dataset.csv`` -- the FedTDO arms, written by
  ``run_paper_fedtdo.py``.

Both read the same files through ``build_paper_clients`` (7 channels, 7:1:2
chronological split, one scaler per dataset), so the columns are comparable.

Two estimators live in this table and the script never mixes them silently:

* ``offline`` -- batched pass over *every* test window.  This is what a backbone
  row reports, and it is the column to quote when comparing FedTDO with a
  backbone.
* ``online`` -- the causal stream the FedTDO controller actually runs on, capped
  at ``--fedtdo_online_steps`` batches.  It is reported separately because it
  answers a different question (deployed loss under delayed feedback).

``core`` excludes ILI: 5 of its 7 picked channels are near-constant at this
window length, so the MSE there is dominated by the scaler's epsilon rather than
by the model.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Exchange', 'ILI',
            'ECL', 'Traffic']
CORE = [d for d in DATASETS if d != 'ILI']

ARM_LABEL = {'base': 'FedTDO-raw', 'center': 'FedTDO-center',
             'decorr': 'FedTDO-decorr', '': 'FedTDO'}


def load_backbones(pattern: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(str(ROOT / pattern))):
        with open(path, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                rows.append({k: (v if k in ('method', 'backbone') else
                                 (float(v) if v not in ('', None) else None))
                             for k, v in row.items()})
    return rows


def load_fedtdo() -> list[dict]:
    rows = []
    for path in sorted(glob.glob(str(ROOT / 'results_paper_fedtdo' / 'PPF_*' / 'per_dataset.csv'))):
        setting = Path(path).parent.name
        # PPF_fedtdo_joint_<arm>_pl96_dm32_c7_w400+l1200[_tag]
        parts = setting.split('_')
        arm = parts[3] if len(parts) > 3 else ''
        per_ds = {}
        online = {}
        params = None
        with open(path, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                per_ds[row['dataset']] = float(row['offline_mse'])
                online[row['dataset']] = float(row['online_mse'])
        import json
        with open(Path(path).parent / 'metrics.json', encoding='utf-8') as f:
            meta = json.load(f)
        params = meta.get('config', {}).get('d_model')
        n_params = None
        rows.append({'method': ARM_LABEL.get(arm, arm), 'backbone': 'dtmoe',
                     'setting': setting, 'per_ds': per_ds, 'online': online,
                     'params': n_params})
    return rows


def core_of(per_ds: dict):
    vals = [per_ds[d] for d in CORE if d in per_ds and per_ds[d] is not None]
    return sum(vals) / len(vals) if vals else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--md', action='store_true')
    p.add_argument('--online', action='store_true',
                   help='print the online (causal stream) column set instead')
    p.add_argument('--backbone-csv', default='results_paper/paper_backbone_mse_wide.csv')
    args = p.parse_args()

    backbones = load_backbones(args.backbone_csv)
    fedtdo = load_fedtdo()

    header = ['method', 'backbone'] + DATASETS + (['online_core'] if args.online else []) + ['core']
    out = []
    for row in fedtdo:
        cells = [row['method'], 'dtmoe'] + [
            row['per_ds'].get(d) for d in DATASETS]
        if args.online:
            cells.append(core_of(row['online']))
        cells.append(core_of(row['per_ds']))
        out.append(cells)
    for row in sorted(backbones, key=lambda r: (r['method'] != 'fedavg', r['backbone'])):
        cells = [row['method'], row['backbone']] + [row.get(d) for d in DATASETS]
        if args.online:
            cells.append(None)
        # recompute over CORE (ex-ILI); the CSV's ``mean`` column includes ILI,
        # whose near-constant channels make it dominate every average
        cells.append(core_of({d: row.get(d) for d in DATASETS}))
        out.append(cells)

    # order: FedTDO arms first, then fedavg backbones by core, then local
    head = [r for r in out if str(r[0]).startswith('FedTDO')]
    rest = [r for r in out if not str(r[0]).startswith('FedTDO')]
    rest.sort(key=lambda r: (r[0] != 'fedavg', r[-1] if r[-1] is not None else 9e9))
    rows = head + rest

    if args.md:
        print('| ' + ' | '.join(header) + ' |')
        print('|' + '|'.join(['---', '---'] + ['---:'] * (len(header) - 2)) + '|')
        for cells in rows:
            print('| ' + ' | '.join(
                '—' if c is None else (f'{c:.4f}' if isinstance(c, float) else str(c))
                for c in cells) + ' |')
    else:
        print('\t'.join(header))
        for cells in rows:
            print('\t'.join('' if c is None else
                            (f'{c:.4f}' if isinstance(c, float) else str(c))
                            for c in cells))


if __name__ == '__main__':
    main()
