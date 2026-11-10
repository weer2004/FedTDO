"""Build the FL-axis and backbone-axis tables from ``results_baselines/``.

Two independent axes, deliberately reported apart:

``--axis fl``       methods x datasets, all under the Dynamic_TMoE backbone
``--axis backbone``  backbones x datasets, all under ``local`` + ``fedavg``

Mixing them in one table is what makes "our method beats PatchTST" claims
unreadable -- the second axis changes the per-client model, not the federation.

Usage::

    python scripts/xc_fedopt_table.py --axis fl
    python scripts/xc_fedopt_table.py --axis backbone
    python scripts/xc_fedopt_table.py --axis fl --csv results/fedtdo/_metrics/fl_axis.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))

from xc_baselines import DATASETS, METHODS, PARKED, backbone_tag, setting_name  # noqa: E402
from xc_sweep import MAIN_DATASETS  # noqa: E402

RESULTS = ROOT / 'results_baselines'
ARMS = ['base', 'center', 'decorr']
ARM_LABEL = {'base': 'FedTDO-raw', 'center': 'FedTDO-center', 'decorr': 'FedTDO-decorr'}
BACKBONES = ['dtmoe', 'patchtst', 'transformer', 'itransformer', 'gru']


def read(method: str, dataset: str, seed: int = 2021, tag: str = ''):
    path = RESULTS / setting_name(method, dataset, seed, tag) / 'summary.json'
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding='utf-8'))
    if payload.get('mean_mse') is None:
        return None
    return payload


def read_arm(dataset: str, arm: str, seed: int = 2021):
    """FedTDO arms live in the fedtdo grid, read through xc_paper_table."""
    try:
        import xc_paper_table as T
        row = T.read_fedtdo(dataset, arm, seed)
    except Exception:
        return None
    if not row or row.get('mse') is None:
        return None
    return row


def matrix(methods, datasets, seed, tag=''):
    table = {}
    for ds in datasets:
        for m in methods:
            table[(ds, m)] = read(m, ds, seed, tag)
    return table


def fl_axis(datasets, seed, csv_path=None):
    methods = METHODS
    table = matrix(methods, datasets, seed)
    arms = {(ds, a): read_arm(ds, a, seed) for ds in datasets for a in ARMS}

    print('## FL 轴：方法 × 数据集（mean client MSE，seed %d）\n' % seed)
    header = ['数据集'] + methods + [ARM_LABEL[a] for a in ARMS]
    print('| ' + ' | '.join(header) + ' |')
    print('|' + '---|' * len(header))
    for ds in datasets:
        cells = [ds]
        best = min((table[(ds, m)]['mean_mse'] for m in methods if table[(ds, m)]),
                   default=None)
        for m in methods:
            row = table[(ds, m)]
            if not row:
                cells.append('—')
                continue
            value = row['mean_mse']
            text = '%.4f' % value
            if best is not None and abs(value - best) < 1e-12:
                text = '**%s**' % text
            cells.append(text)
        for a in ARMS:
            row = arms[(ds, a)]
            cells.append('%.4f' % row['mse'] if row else '—')
        print('| ' + ' | '.join(cells) + ' |')

    print('\n## 汇总（相对 local，%d 个数据集）\n' % len(datasets))
    print('| 方法 | 均值 | 中位 | 优于 local | 平均排名 | 垫底 | 缺失 |')
    print('|---|---|---|---|---|---|---|')
    rows = []
    local = {ds: (table[(ds, 'local')] or {}).get('mean_mse') for ds in datasets}
    ranks = {m: [] for m in methods}
    for ds in datasets:
        present = [(table[(ds, m)]['mean_mse'], m) for m in methods if table[(ds, m)]]
        for i, (_, m) in enumerate(sorted(present), 1):
            ranks[m].append(i)
    for m in methods:
        vals, deltas = [], []
        for ds in datasets:
            row = table[(ds, m)]
            if not row or not local.get(ds):
                continue
            vals.append(row['mean_mse'])
            deltas.append(100 * (local[ds] - row['mean_mse']) / local[ds])
        if not deltas:
            print('| %s | — | — | — | — | — | %d |' % (m, len(datasets)))
            continue
        best_count = sum(
            1 for ds in datasets
            if table[(ds, m)] and all(
                (table[(ds, o)] is None or table[(ds, m)]['mean_mse'] <= table[(ds, o)]['mean_mse'])
                for o in methods))
        worst_count = sum(
            1 for ds in datasets
            if table[(ds, m)] and all(
                (table[(ds, o)] is None or table[(ds, m)]['mean_mse'] >= table[(ds, o)]['mean_mse'])
                for o in methods))
        rows.append((m, st.fmean(deltas), st.median(deltas),
                     sum(1 for d in deltas if d > 0), st.fmean(ranks[m]),
                     best_count, worst_count, len(deltas)))
        print('| %s | %+.2f%% | %+.2f%% | %d/%d | %.2f | %d | %d | %d |' % (
            m, st.fmean(deltas), st.median(deltas), sum(1 for d in deltas if d > 0),
            len(deltas), st.fmean(ranks[m]), best_count, worst_count,
            len(datasets) - len(deltas)))

    print('\n## 三臂对"每个数据集的最好基线"\n')
    print('| 数据集 | 参照 | 值 | raw | center | decorr |')
    print('|---|---|---|---|---|---|')
    totals = {a: [] for a in ARMS}
    for ds in datasets:
        present = [(table[(ds, m)]['mean_mse'], m) for m in methods
                   if table[(ds, m)] and m != 'local' or (table[(ds, m)] and m == 'local')]
        if not present:
            continue
        ref_value, ref_name = min(present)
        cells = []
        for a in ARMS:
            row = arms[(ds, a)]
            if not row:
                cells.append('—')
                continue
            delta = 100 * (ref_value - row['mse']) / ref_value
            totals[a].append(delta)
            cells.append('%+.2f%%' % delta)
        print('| %s | %s | %.4f | %s |' % (ds, ref_name, ref_value, ' | '.join(cells)))
    for a in ARMS:
        if totals[a]:
            print('\n%s: 均值 %+.2f%%  中位 %+.2f%%  胜 %d/%d' % (
                ARM_LABEL[a], st.fmean(totals[a]), st.median(totals[a]),
                sum(1 for x in totals[a] if x > 0), len(totals[a])))

    parked = [(m, read(m, ds, seed)) for m in PARKED for ds in datasets]
    have = sum(1 for _, r in parked if r)
    if have:
        print('\n> `%s` 已跑但被排除在表外（`PARKED`，%d 个结果仍在磁盘上，可直接放回）。'
              % ('`, `'.join(PARKED), have))

    if csv_path:
        write_csv(csv_path, methods, datasets, table, arms)
        print('\nwrote %s' % csv_path)


def backbone_axis(datasets, seed, methods=('local', 'fedavg'), csv_path=None):
    print('## 骨干轴：骨干 × 数据集（%s，mean client MSE）\n' % ' + '.join(methods))
    header = ['数据集']
    for b in BACKBONES:
        for m in methods:
            header.append('%s/%s' % (b, m))
    print('| ' + ' | '.join(header) + ' |')
    print('|' + '---|' * len(header))
    table = {}
    for ds in datasets:
        cells = [ds]
        for b in BACKBONES:
            tag = backbone_tag(b, '')
            for m in methods:
                row = read(m, ds, seed, tag)
                table[(ds, b, m)] = row
                cells.append('%.4f' % row['mean_mse'] if row else '—')
        print('| ' + ' | '.join(cells) + ' |')

    print('\n## 相对 dtmoe 的百分变化（同方法）\n')
    for m in methods:
        print('| 数据集 | ' + ' | '.join(BACKBONES) + ' |')
        print('|' + '---|' * (len(BACKBONES) + 1))
        for ds in datasets:
            base = table.get((ds, 'dtmoe', m))
            cells = [ds]
            for b in BACKBONES:
                row = table.get((ds, b, m))
                if not row or not base:
                    cells.append('—')
                else:
                    cells.append('%+.1f%%' % (100 * (row['mean_mse'] - base['mean_mse']) / base['mean_mse']))
            print('| ' + ' | '.join(cells) + ' |')

    if csv_path:
        with open(csv_path, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.writer(fh)
            writer.writerow(['dataset', 'backbone', 'method', 'mean_mse', 'mean_mae',
                             'median_mse', 'max_mse', 'params_note'])
            for (ds, b, m), row in table.items():
                writer.writerow([ds, b, m,
                                 row['mean_mse'] if row else '',
                                 row['mean_mae'] if row else '',
                                 row['median_mse'] if row else '',
                                 row['max_mse'] if row else '', ''])
        print('\nwrote %s' % csv_path)


def write_csv(path, methods, datasets, table, arms):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['dataset', 'method', 'family', 'mean_mse', 'mean_mae',
                         'median_mse', 'max_mse', 'std_mse', 'num_clients'])
        for ds in datasets:
            for m in methods:
                row = table[(ds, m)]
                if not row:
                    continue
                writer.writerow([ds, m, 'fl_baseline', row.get('mean_mse'),
                                 row.get('mean_mae'), row.get('median_mse'),
                                 row.get('max_mse'), row.get('std_mse'),
                                 row.get('num_clients')])
            for a in ARMS:
                row = arms[(ds, a)]
                if not row:
                    continue
                writer.writerow([ds, ARM_LABEL[a], 'fedtdo', row.get('mse'),
                                 row.get('mae'), row.get('median_mse'),
                                 row.get('max_client_mse'), None, row.get('num_clients')])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--axis', choices=['fl', 'backbone'], default='fl')
    parser.add_argument('--datasets', nargs='+', default=MAIN_DATASETS, choices=list(DATASETS))
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--csv', type=str, default=None)
    args = parser.parse_args()
    if args.axis == 'fl':
        fl_axis(args.datasets, args.seed, args.csv)
    else:
        backbone_axis(args.datasets, args.seed, csv_path=args.csv)


if __name__ == '__main__':
    main()
