"""Scan the raw paper tree and build the aggregated CSVs.

Read-only with respect to ``raw/``: nothing under it is rewritten.  Every output
here is a pure function of the raw rows, so the aggregation can be re-run at any
time.

Usage::

    python scripts/paper/collect_results.py
    python scripts/paper/collect_results.py --rolling 64 96
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (AGG, ROOT, ensure_dirs, find_runs, identity, load_run,  # noqa: E402
                    rolling_mse, stage_window_indices)

SUMMARY_KEYS = [
    'overall_mse', 'overall_mae', 'overall_rmse', 'overall_mape', 'overall_mspe',
    'evaluated_windows', 'online_update_events', 'gradient_steps',
    'candidate_evaluations', 'transfer_events', 'upload_bytes', 'download_bytes',
    'total_communication_bytes', 'forecasting_time', 'shadow_evaluation_time',
    'adaptation_time', 'encoder_sync_time', 'total_runtime', 'num_clients',
]

CONFIG_KEYS = [
    'num_clients', 'channels_per_client', 'seq_len', 'pred_len', 'model_name',
    'online_enabled', 'online_lr', 'online_substeps', 'online_steps',
    'stream_batch', 'label_delay', 'eval_budget', 'cache_limit',
    'transfer_enabled', 'share_modules', 'optimizer', 'experiment_status',
]


def build_manifest(runs: List[dict]) -> pd.DataFrame:
    rows = []
    for run in runs:
        row = identity(run)
        for key in SUMMARY_KEYS:
            row[key] = run['summary'].get(key)
        for key in CONFIG_KEYS:
            row.setdefault(key, None)
            row[key] = run['config'].get(key, row[key])
        row['run_dir'] = str(run['dir'].relative_to(ROOT))
        rows.append(row)
    return pd.DataFrame(rows)


def build_per_seed(manifest: pd.DataFrame) -> pd.DataFrame:
    if manifest.empty:
        return manifest
    keep = ['run_id', 'dataset', 'method', 'seed'] + [
        k for k in SUMMARY_KEYS if k in manifest.columns]
    return manifest[keep].copy()


def build_per_client(runs: List[dict]) -> pd.DataFrame:
    rows = []
    for run in runs:
        base = identity(run)
        per_client = run['summary'].get('per_client') or {}
        window = run['per_window']
        for client_id, values in per_client.items():
            row = dict(base)
            row['client_id'] = client_id
            row['mse'] = values.get('mse')
            row['mae'] = values.get('mae')
            if window is not None and not window.empty:
                group = window[window['client_id'] == client_id]
                row['evaluated_windows'] = int(len(group))
                row['last_cumulative_mse'] = float(
                    group.sort_values('window_idx')['cumulative_mse'].iloc[-1]
                ) if len(group) else None
                rows.append(row)
            else:
                row['evaluated_windows'] = None
                row['last_cumulative_mse'] = None
                rows.append(row)
    return pd.DataFrame(rows)


def build_convergence(runs: List[dict], windows=(64, 96)) -> pd.DataFrame:
    frames = []
    for run in runs:
        if run['per_window'] is None or run['per_window'].empty:
            continue
        base = identity(run)
        for window in windows:
            curve = rolling_mse(run['per_window'], window=window)
            if curve.empty:
                continue
            curve = curve.assign(rolling_window=window, **base)
            frames.append(curve)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_transfer_summary(runs: List[dict]) -> pd.DataFrame:
    rows = []
    for run in runs:
        base = identity(run)
        transfers = run['transfers']
        evaluations = run['evaluations']
        feedback = run['feedback']
        row = dict(base)
        row['transfer_events'] = 0 if transfers is None else int(len(transfers))
        row['candidate_evaluations'] = (0 if evaluations is None
                                        else int(len(evaluations)))
        row['feedback_rows'] = 0 if feedback is None else int(len(feedback))
        if feedback is not None and len(feedback) and 'realized_gain' in feedback:
            gains = pd.to_numeric(feedback['realized_gain'], errors='coerce').dropna()
            row['realized_gain_mean'] = float(gains.mean()) if len(gains) else None
            row['realized_gain_positive_rate'] = (
                float((gains > 0).mean()) if len(gains) else None)
            if 'predicted_gain' in feedback:
                predicted = pd.to_numeric(feedback['predicted_gain'],
                                          errors='coerce')
                mask = predicted.notna() & \
                    pd.to_numeric(feedback['realized_gain'], errors='coerce').notna()
                row['predicted_vs_realized_corr'] = (
                    float(np.corrcoef(predicted[mask],
                                      pd.to_numeric(
                                          feedback['realized_gain'],
                                          errors='coerce')[mask])[0, 1])
                    if mask.sum() > 2 else None)
        rows.append(row)
    return pd.DataFrame(rows)


def build_cost_summary(runs: List[dict]) -> pd.DataFrame:
    rows = []
    for run in runs:
        base = identity(run)
        row = dict(base)
        for key in ('upload_bytes', 'download_bytes', 'total_communication_bytes',
                    'forecasting_time', 'shadow_evaluation_time', 'adaptation_time',
                    'encoder_sync_time', 'total_runtime', 'evaluated_windows',
                    'gradient_steps', 'candidate_evaluations'):
            row[key] = run['summary'].get(key)
        cost = run['cost']
        if cost is not None and len(cost):
            by_op = cost.groupby('operation')['duration_ms'].sum()
            for op, value in by_op.items():
                row[f'{op}_ms'] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def build_visualisation_samples(runs: List[dict]) -> pd.DataFrame:
    """Pre-declared sample positions for the forecast figures.

    The rule is fixed before looking at any curve: for every (dataset, seed,
    client) the early/middle/late positions are 10/50/90 percent of the windows
    that client actually evaluated.
    """
    rows = []
    for run in runs:
        base = identity(run)
        for client_id, arrays in run['predictions'].items():
            idx = np.asarray(arrays['window_idx'])
            if idx.size == 0:
                continue
            stages = stage_window_indices(int(idx.max()))
            for stage, target in stages.items():
                position = int(np.argmin(np.abs(idx - target)))
                rows.append({**base, 'client_id': client_id, 'stage': stage,
                             'window_idx': int(idx[position]),
                             'position': position,
                             'num_windows': int(idx.size)})
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--rolling', nargs='+', type=int, default=[64, 96])
    args = parser.parse_args()

    ensure_dirs()
    run_dirs = find_runs()
    if not run_dirs:
        print(f'no finished runs under {ROOT / "results/paper_fedtdo/raw"}')
        return 0
    runs = [load_run(path) for path in run_dirs]

    manifest = build_manifest(runs)
    duplicates = manifest['run_id'][manifest['run_id'].duplicated()].unique()
    if len(duplicates):
        print(f'WARNING duplicate run_id: {list(duplicates)[:5]}')
    # Two runs with the same (dataset, method, seed) would be averaged together
    # by the table builder and silently halve the seed count.
    cell_keys = ['dataset', 'method', 'seed']
    repeated = manifest[manifest.duplicated(subset=cell_keys, keep=False)]
    if len(repeated):
        print(f'WARNING {len(repeated)} runs share a (dataset, method, seed) cell:')
        for _, row in repeated.head(6).iterrows():
            print(f'  {row["dataset"]}/{row["method"]}/s{row["seed"]}  {row["run_id"]}')

    outputs = {
        'run_manifest.csv': manifest,
        'per_seed_summary.csv': build_per_seed(manifest),
        'per_client_summary.csv': build_per_client(runs),
        'convergence_curves.csv': build_convergence(runs, tuple(args.rolling)),
        'transfer_summary.csv': build_transfer_summary(runs),
        'cost_summary.csv': build_cost_summary(runs),
        'forecast_visualization_samples.csv': build_visualisation_samples(runs),
    }
    for name, frame in outputs.items():
        path = AGG / name
        frame.to_csv(path, index=False)
        print(f'{name:38s} rows={len(frame):6d} -> {path.relative_to(ROOT)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
