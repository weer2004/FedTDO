"""Paper tables from the recorded runs, as CSV and LaTeX.

Table identity is decided by the data, not by the script: a table is emitted only
when the runs it needs exist.  A missing ablation prints one line saying so
rather than producing an empty grid.

    T01 main results          dataset x method, MSE/MAE mean +- std over seeds
    T02 client-level          dataset x method x client
    T03..T09 ablations        any config knob via --by <key>
    T10 cost                  communication + wall-clock breakdown

Usage::

    python scripts/paper/build_tables.py
    python scripts/paper/build_tables.py --by fedtdo_eval_budget --tables T08
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (TABLES, find_runs, identity, load_run, order_key,  # noqa: E402
                    pretty)


def _latex(frame: pd.DataFrame, caption: str, label: str) -> str:
    columns = list(frame.columns)
    align = 'l' + 'c' * (len(columns) - 1)
    lines = [
        r'\begin{table}[t]',
        rf'\caption{{{caption}}}',
        rf'\label{{{label}}}',
        r'\centering\small',
        rf'\begin{{tabular}}{{{align}}}',
        r'\toprule',
        ' & '.join(columns) + r' \\',
        r'\midrule',
    ]
    for _, row in frame.iterrows():
        lines.append(' & '.join(str(value) for value in row) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    return '\n'.join(lines)


def _write(name: str, frame: pd.DataFrame, caption: str, label: str):
    csv_path = TABLES / f'{name}.csv'
    tex_path = TABLES / f'{name}.tex'
    frame.to_csv(csv_path, index=False)
    tex_path.write_text(_latex(frame, caption, label), encoding='utf-8')
    print(f'{name}: rows={len(frame):4d} -> {csv_path.name}, {tex_path.name}')


def _mean_std(values: pd.Series) -> str:
    clean = pd.to_numeric(values, errors='coerce').dropna()
    if clean.empty:
        return '--'
    if len(clean) == 1:
        return f'{clean.iloc[0]:.4f}'
    return f'{clean.mean():.4f}$\\pm${clean.std(ddof=1):.4f}'


def main_table(runs: List[dict]) -> pd.DataFrame:
    rows = []
    for dataset, group in _by(runs, lambda r: r['config'].get('dataset')).items():
        methods = sorted({r['config'].get('method') for r in group}, key=order_key)
        entry: Dict[str, str] = {'Dataset': dataset}
        for method in methods:
            subset = [r for r in group if r['config'].get('method') == method]
            entry[pretty(method)] = _mean_std(
                pd.Series([r['summary'].get('overall_mse') for r in subset]))
        rows.append(entry)
    return pd.DataFrame(rows).fillna('--')


def client_table(runs: List[dict]) -> pd.DataFrame:
    rows = []
    for run in runs:
        ident = identity(run)
        per_client = run['summary'].get('per_client') or {}
        for client_id, values in sorted(per_client.items()):
            rows.append({**ident, 'client_id': client_id,
                         'mse': values.get('mse'), 'mae': values.get('mae')})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    pivot = frame.pivot_table(index=['dataset', 'client_id'], columns='method',
                              values='mse', aggfunc='mean')
    ordered = [c for c in sorted(pivot.columns, key=order_key)]
    return pivot[ordered].reset_index().round(4)


def ablation_table(runs: List[dict], key: str) -> pd.DataFrame:
    rows = []
    for run in runs:
        cfg = run['config']
        value = cfg.get(key)
        if value is None:
            continue
        ident = identity(run)
        rows.append({'dataset': ident['dataset'], 'method': ident['method'],
                     key: value, 'mse': run['summary'].get('overall_mse'),
                     'mae': run['summary'].get('overall_mae'),
                     'seed': ident['seed'],
                     'evaluated_windows': run['summary'].get('evaluated_windows'),
                     'transfers': run['summary'].get('transfer_events'),
                     'eval_events': run['summary'].get('candidate_evaluations')})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    grouped = (frame.groupby(['dataset', key])
               .agg(mse=('mse', 'mean'), mae=('mae', 'mean'),
                    seeds=('seed', 'nunique'),
                    evaluated_windows=('evaluated_windows', 'mean'),
                    transfers=('transfers', 'mean'),
                    eval_events=('eval_events', 'mean'))
               .reset_index())
    return grouped.round(4)


def cost_table(runs: List[dict]) -> pd.DataFrame:
    rows = []
    for run in runs:
        ident = identity(run)
        summ = run['summary']
        windows = summ.get('evaluated_windows') or 0
        total = summ.get('total_communication_bytes') or 0
        rows.append({
            'dataset': ident['dataset'], 'method': ident['method'],
            'seed': ident['seed'],
            'MSE': summ.get('overall_mse'),
            'evaluated_windows': windows,
            'upload_MB': (summ.get('upload_bytes') or 0) / 1e6,
            'download_MB': (summ.get('download_bytes') or 0) / 1e6,
            'comm_MB': total / 1e6,
            'comm_KB_per_window': (total / windows / 1e3) if windows else None,
            'forecast_s': summ.get('forecasting_time'),
            'shadow_s': summ.get('shadow_evaluation_time'),
            'adapt_s': summ.get('adaptation_time'),
            'sync_s': summ.get('encoder_sync_time'),
            'runtime_s': summ.get('total_runtime'),
            'grad_steps': summ.get('gradient_steps'),
            'eval_events': summ.get('candidate_evaluations'),
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return (frame.groupby(['dataset', 'method']).mean(numeric_only=True)
            .reset_index().round(4))


def _by(runs, key_fn):
    out: Dict[str, list] = {}
    for run in runs:
        out.setdefault(str(key_fn(run)), []).append(run)
    return out


ABLATION_TABLES = {
    'T03': ('transfer_enabled', 'Core component ablation'),
    'T04': ('share_modules', 'Encoder sharing strategy'),
    'T05': ('fedtdo_reward_basis', 'Reward estimator comparison'),
    'T06': ('fedtdo_decision', 'Transfer decision strategy'),
    'T07': ('cache_limit', 'Cache / registry configuration'),
    'T08': ('eval_budget', 'Evaluation budget sensitivity'),
    'T09': ('label_delay', 'Feedback delay sensitivity'),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--tables', nargs='+', default=None,
                        help='subset of T01 T02 T03..T10')
    parser.add_argument('--by', default=None,
                        help='config key for a custom ablation table')
    args = parser.parse_args()

    TABLES.mkdir(parents=True, exist_ok=True)
    run_dirs = find_runs()
    if not run_dirs:
        print('no runs found')
        return 0
    runs = [load_run(path) for path in run_dirs]
    wanted = set(args.tables) if args.tables else None

    def want(name: str) -> bool:
        return wanted is None or name in wanted

    if want('T01'):
        frame = main_table(runs)
        _write('T01_main_results', frame,
               'Prequential forecasting error per dataset (mean $\\pm$ std over seeds).',
               'tab:main-results')

    if want('T02'):
        frame = client_table(runs)
        if not frame.empty:
            _write('T02_client_results', frame,
                   'Per-client MSE on each dataset.', 'tab:client-results')

    for name, (key, caption) in ABLATION_TABLES.items():
        if not want(name):
            continue
        frame = ablation_table(runs, key)
        if frame.empty:
            print(f'{name}: no runs carry "{key}"; table skipped')
            continue
        _write(f'{name}_{key}', frame, caption, f'tab:{name.lower()}')

    if args.by:
        frame = ablation_table(runs, args.by)
        if frame.empty:
            print(f'custom: no runs carry "{args.by}"')
        else:
            _write(f'T99_{args.by}', frame, f'Runs grouped by {args.by}', 'tab:t99')

    if want('T10'):
        frame = cost_table(runs)
        if not frame.empty:
            _write('T10_cost', frame,
                   'Communication and wall-clock cost per evaluated window.',
                   'tab:cost')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
