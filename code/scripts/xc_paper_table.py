"""Paper tables: FedTDO arms vs FL baselines on the nine DTMoe benchmarks.

Metric parity
-------------
``Exp_FedTDO`` stores ``mean_client_mse`` = arithmetic mean of the per-client
MSEs; ``Exp_FL_Baselines`` stores ``mean_mse`` computed the same way.  Both are
the plain average over clients, so the two tables are comparable cell by cell
without rescaling.  Two caveats are reported alongside the numbers:

* client errors are heavy-tailed (Traffic holds a sensor at +39 sigma), so the
  mean and the median are both printed and they can disagree in sign;
* ``RMSE`` is pooled (``sqrt`` of the sample-weighted MSE), not the average of
  the per-client RMSEs, because the latter is not a norm and does not average
  into anything meaningful.

Usage::

    python scripts/xc_paper_table.py --markdown
    python scripts/xc_paper_table.py --seeds 2021 2022 2023 --markdown
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from xc_baselines import METHODS, setting_name  # noqa: E402
from xc_sweep import DATASETS, MAIN_DATASETS, ROOT  # noqa: E402

FEDTDO = ROOT / 'results' / 'fedtdo'
BASELINES = ROOT / 'results_baselines'
ARMS = ['base', 'center', 'decorr']
ARM_LABEL = {'base': 'FedTDO-raw', 'center': 'FedTDO-center', 'decorr': 'FedTDO-decorr'}

# Traffic's raw arm predates the sweep naming scheme (see scripts/xc_table.py).
LEGACY = {('Traffic', 'base'): 'traffic32_s{seed}_joint_b9_fix'}


def fedtdo_dir(dataset: str, arm: str, seed: int) -> Path:
    _, _, _, nc, budget = DATASETS[dataset]
    standard = FEDTDO / f'{dataset.lower()}{nc}_s{seed}_b{budget}_{arm}'
    if standard.exists():
        return standard
    legacy = LEGACY.get((dataset, arm))
    return FEDTDO / legacy.format(seed=seed) if legacy else standard


def read_fedtdo(dataset: str, arm: str, seed: int):
    path = fedtdo_dir(dataset, arm, seed) / 'metrics.json'
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    per_client = payload.get('client_metrics') or {}
    mses = [v['mse'] for v in per_client.values()]
    maes = [v['mae'] for v in per_client.values()]
    mean_mse = payload.get('mean_client_mse')
    mean_mae = payload.get('mean_client_mae')
    if mean_mse is None and mses:
        mean_mse = sum(mses) / len(mses)
    if mean_mae is None and maes:
        mean_mae = sum(maes) / len(maes)
    return dict(
        seed=seed, family='fedtdo',
        mse=mean_mse, mae=mean_mae,
        median_mse=median(mses) if mses else None,
        median_mae=median(maes) if maes else None,
        rmse=math.sqrt(mean_mse) if mean_mse else None,
        clients=len(mses),
        transfers=(payload.get('actions') or {}).get('transfers'),
        gain=100.0 * (payload.get('actions') or {}).get('relative_gain', 0.0),
    )


def read_baseline(dataset: str, method: str, seed: int):
    path = BASELINES / setting_name(method, dataset, seed) / 'summary.json'
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    mse = payload.get('mean_mse')
    return dict(
        seed=seed, family='fl_baseline',
        mse=mse, mae=payload.get('mean_mae'),
        median_mse=payload.get('median_mse'),
        median_mae=payload.get('median_mae'),
        rmse=math.sqrt(mse) if mse else None,
        clients=payload.get('num_clients'),
    )


def median(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else 0.5 * (values[mid - 1] + values[mid])


def collect(seeds):
    """(dataset, key) -> list of per-seed metric dicts."""
    table = {}
    for dataset in MAIN_DATASETS:
        for method in METHODS:
            rows = [read_baseline(dataset, method, s) for s in seeds]
            rows = [r for r in rows if r]
            if rows:
                table[(dataset, method)] = rows
        for arm in ARMS:
            rows = [read_fedtdo(dataset, arm, s) for s in seeds]
            rows = [r for r in rows if r]
            if rows:
                table[(dataset, ARM_LABEL[arm])] = rows
    return table


def stat(rows, key):
    if not rows:
        return None, None
    values = [r.get(key) for r in rows if r.get(key) is not None]
    if not values:
        return None, None
    return sum(values) / len(values), median(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='+', type=int, default=[2021])
    parser.add_argument('--markdown', action='store_true', help='emit markdown tables')
    parser.add_argument('--metric', default='mse', choices=['mse', 'mae', 'median_mse'])
    parser.add_argument('--csv', action='store_true',
                        help='write a long-format CSV for plotting (results/fedtdo/_metrics/paper_main.csv)')
    args = parser.parse_args()

    table = collect(args.seeds)
    columns = METHODS + [ARM_LABEL[a] for a in ARMS]
    metric = args.metric

    if args.csv:
        import csv as _csv
        out_dir = ROOT / 'results' / 'fedtdo' / '_metrics'
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for dataset in MAIN_DATASETS:
            floor = stat(table.get((dataset, 'local')), args.metric)[0]
            for column in columns:
                for entry in table.get((dataset, column)) or []:
                    value = entry.get(args.metric)
                    delta = (100.0 * (floor - value) / floor
                             if floor and value is not None else None)
                    rows.append(dict(
                        dataset=dataset, method=column, family=entry['family'],
                        seed=entry['seed'],
                        mse=entry.get('mse'), mae=entry.get('mae'),
                        rmse=entry.get('rmse'),
                        median_mse=entry.get('median_mse'),
                        median_mae=entry.get('median_mae'),
                        clients=entry.get('clients'),
                        delta_vs_local_pct=delta,
                    ))
        path = out_dir / 'paper_main.csv'
        with open(path, 'w', newline='', encoding='utf-8') as handle:
            writer = _csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f'wrote {path} ({len(rows)} rows)')
        return

    if args.markdown:
        print(f'### {metric.upper()} on the nine DTMoe benchmarks '
              f'({len(args.seeds)} seed(s): {", ".join(map(str, args.seeds))})\n')
        print('| dataset | ' + ' | '.join(columns) + ' |')
        print('|---' * (len(columns) + 1) + '|')
        for dataset in MAIN_DATASETS:
            cells = []
            floor = table.get((dataset, 'local'))
            floor_mean = stat(floor, metric)[0] if floor else None
            for column in columns:
                rows = table.get((dataset, column))
                if not rows:
                    cells.append('—')
                    continue
                mean, _ = stat(rows, metric)
                if column == 'local' or floor_mean in (None, 0):
                    cells.append(f'{mean:.4f}')
                else:
                    delta = 100.0 * (floor_mean - mean) / floor_mean
                    cells.append(f'{mean:.4f} ({delta:+.1f}%)')
            print(f'| {dataset} | ' + ' | '.join(cells) + ' |')
        print()
        print('单元格为跨种子均值，括号内是相对 `local` 的改善（正数 = 更好）.')
        return

    # console view: dataset x method, with the winner highlighted
    print(f'{"dataset":10s} ' + ' '.join(f'{c[:13]:>13s}' for c in columns))
    for dataset in MAIN_DATASETS:
        cells = []
        values = {}
        for column in columns:
            rows = table.get((dataset, column))
            mean = stat(rows, metric)[0] if rows else None
            values[column] = mean
            cells.append(f'{mean:13.4f}' if mean is not None else f'{"—":>13s}')
        best = min((v, k) for k, v in values.items() if v is not None) if any(
            v is not None for v in values.values()) else (None, None)
        print(f'{dataset:10s} ' + ' '.join(cells) + f'   best={best[1]}')

    print()
    print('=== FedTDO best arm vs best baseline (per dataset) ===')
    for dataset in MAIN_DATASETS:
        floor = stat(table.get((dataset, 'local')), metric)[0]
        base_best = min(
            ((stat(table.get((dataset, m)), metric)[0], m) for m in METHODS
             if table.get((dataset, m))),
            default=(None, None),
        )
        arm_best = min(
            ((stat(table.get((dataset, ARM_LABEL[a])), metric)[0], ARM_LABEL[a]) for a in ARMS
             if table.get((dataset, ARM_LABEL[a]))),
            default=(None, None),
        )
        if None in (floor, base_best[0], arm_best[0]):
            continue
        delta = 100.0 * (base_best[0] - arm_best[0]) / base_best[0]
        print(f'{dataset:10s} local={floor:.4f}  best_baseline={base_best[1]}({base_best[0]:.4f})'
              f'  best_FedTDO={arm_best[1]}({arm_best[0]:.4f})  FedTDO-baseline={delta:+.2f}%')

    print()
    print('=== vs local, per method, averaged over datasets ===')
    for column in columns:
        deltas = []
        for dataset in MAIN_DATASETS:
            floor = stat(table.get((dataset, 'local')), metric)[0]
            value = stat(table.get((dataset, column)), metric)[0]
            if floor and value:
                deltas.append(100.0 * (floor - value) / floor)
        if deltas:
            print(f'{column:14s} mean={sum(deltas) / len(deltas):+7.2f}%  '
                  f'median={median(deltas):+7.2f}%  better_than_local={sum(d > 0 for d in deltas)}/{len(deltas)}')


if __name__ == '__main__':
    main()
