"""Integrity checks over the recorded paper runs.

Each check answers a question that decides whether a table built from this run
can be trusted.  A failing check is reported, never repaired: the raw tree is
the source of truth.

Usage::

    python scripts/paper/validate_results.py
    python scripts/paper/validate_results.py --strict   # non-zero exit on failure
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import find_runs, identity, load_run  # noqa: E402


class Report:
    def __init__(self):
        self.rows: List[dict] = []

    def add(self, run_id: str, check: str, ok: bool, detail: str = ''):
        self.rows.append({'run_id': run_id, 'check': check,
                          'status': 'pass' if ok else 'FAIL', 'detail': detail})

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def check_window_sequence(run_id: str, window: pd.DataFrame, report: Report):
    if window is None or window.empty:
        report.add(run_id, 'window_rows_present', False, 'per_window_metrics.csv empty')
        return
    ok = True
    detail = []
    for client_id, group in window.groupby('client_id'):
        idx = group.sort_values('window_idx')['window_idx'].to_numpy()
        expected = np.arange(1, len(idx) + 1)
        if not np.array_equal(idx, expected):
            ok = False
            detail.append(f'{client_id}: non-contiguous window_idx')
    report.add(run_id, 'window_idx_contiguous', ok, '; '.join(detail))


def check_label_causality(run_id: str, window: pd.DataFrame, report: Report):
    if window is None or window.empty or 'label_reveal_idx' not in window:
        report.add(run_id, 'label_revealed_after_prediction', False, 'column missing')
        return
    reveal = pd.to_numeric(window['label_reveal_idx'], errors='coerce')
    predicted_at = pd.to_numeric(window['window_idx'], errors='coerce')
    ok = bool((reveal.fillna(-1) >= predicted_at.fillna(0)).all())
    report.add(run_id, 'label_revealed_after_prediction', ok,
               '' if ok else 'some windows consumed a label before predicting it')


def check_prediction_alignment(run_id: str, run: dict, report: Report):
    """Recompute SSE from the stored arrays and compare with the CSV."""
    window = run['per_window']
    preds = run['predictions']
    if window is None or window.empty or not preds:
        report.add(run_id, 'prediction_alignment', False, 'missing predictions')
        return
    mismatches = []
    for client_id, arrays in preds.items():
        group = window[window['client_id'] == client_id].sort_values('window_idx')
        y_true = np.asarray(arrays['y_true'], dtype=np.float64)
        y_pred = np.asarray(arrays['y_pred'], dtype=np.float64)
        if y_true.shape != y_pred.shape:
            mismatches.append(f'{client_id}: shape {y_true.shape} vs {y_pred.shape}')
            continue
        if y_true.shape[0] != len(group):
            mismatches.append(f'{client_id}: {y_true.shape[0]} windows vs csv '
                              f'{len(group)}')
            continue
        sse = ((y_true - y_pred) ** 2).sum(axis=tuple(range(1, y_true.ndim)))
        stored = group['sse'].to_numpy(dtype=np.float64)
        if not np.allclose(sse, stored, rtol=1e-4, atol=1e-6):
            mismatches.append(f'{client_id}: sse mismatch')
        idx = np.asarray(arrays['window_idx'])
        if not np.array_equal(idx, group['window_idx'].to_numpy()):
            mismatches.append(f'{client_id}: window_idx order mismatch')
    report.add(run_id, 'prediction_matches_csv', not mismatches,
               '; '.join(mismatches[:3]))


def check_update_accounting(run_id: str, run: dict, report: Report):
    """The per-window column holds the steps accumulated *before* that window.

    So for every client, the largest value in ``per_window_metrics`` must equal
    the cumulative count carried by the client's last update row minus the steps
    that last update itself took.  Comparing it against the global total would
    mix clients together and always fail.
    """
    window = run['per_window']
    updates = run['updates']
    if window is None or window.empty:
        report.add(run_id, 'update_accounting', False, 'no windows')
        return
    problems = []
    for client_id, group in window.groupby('client_id'):
        before = int(pd.to_numeric(group['gradient_steps'],
                                   errors='coerce').fillna(0).max())
        if updates is None or updates.empty:
            expected = 0
        else:
            mine = updates[updates['client_id'] == client_id]
            if mine.empty:
                expected = 0
            else:
                last = mine.sort_values(['window_idx']).iloc[-1]
                expected = int(pd.to_numeric(pd.Series([last['total_gradient_steps']]),
                                             errors='coerce').fillna(0).iloc[0]) \
                    - int(pd.to_numeric(pd.Series([last['gradient_steps']]),
                                        errors='coerce').fillna(0).iloc[0])
        if before != expected:
            problems.append(f'{client_id}: csv={before} expected={expected}')
    report.add(run_id, 'gradient_steps_match', not problems,
               '; '.join(problems[:3]))


def check_eval_budget(run_id: str, run: dict, report: Report):
    evaluations = run['evaluations']
    if evaluations is None or evaluations.empty:
        report.add(run_id, 'candidate_evaluations_present', True, 'none recorded')
        return
    per_window = evaluations.groupby(['client_id', 'window_idx']).size()
    budget = run['config'].get('eval_budget')
    ok = True
    detail = f'max per window={int(per_window.max()) if len(per_window) else 0}'
    if budget:
        # The selected expert is always evaluated in addition to the probes.
        ok = int(per_window.max()) <= int(budget) + 1
        detail += f' budget={budget}'
    report.add(run_id, 'candidate_evaluations_within_budget', ok, detail)


def check_transfer_cross_reference(run_id: str, run: dict, report: Report):
    transfers = run['transfers']
    evaluations = run['evaluations']
    if transfers is None or transfers.empty:
        report.add(run_id, 'transfer_matches_evaluation', True, 'no transfers')
        return
    switches = transfers[transfers['event_type'] == 'switch']
    if switches.empty or evaluations is None or evaluations.empty:
        report.add(run_id, 'transfer_matches_evaluation', True, 'no switch rows')
        return
    flagged = evaluations[pd.to_numeric(evaluations['transferred'],
                                        errors='coerce').fillna(0) > 0]
    ok = len(flagged) >= len(switches)
    report.add(run_id, 'transfer_matches_evaluation', ok,
               f'switches={len(switches)} flagged_evaluations={len(flagged)}')


def check_config_consistency(runs: Dict[str, dict], report: Report):
    """Every method compared in one table must describe the same split."""
    by_dataset: Dict[str, list] = {}
    for run_id, run in runs.items():
        by_dataset.setdefault(str(run['config'].get('dataset')), []).append(run)
    for dataset, group in by_dataset.items():
        keys = ('seq_len', 'pred_len', 'num_clients')
        values = {k: {run['config'].get(k) for run in group} for k in keys}
        for key, distinct in values.items():
            ok = len(distinct) <= 1
            report.add(f'[{dataset}]', f'uniform_{key}', ok, str(sorted(distinct)))
        windows = {run['summary'].get('evaluated_windows') for run in group}
        windows.discard(None)
        if windows:
            per_client = {run['config'].get('num_clients') for run in group}
            evaluated = {int(w) // max(1, int(n)) for w, n in
                         zip(sorted(windows), sorted(per_client)) if n}
            report.add(f'[{dataset}]', 'comparable_window_budget',
                       len(evaluated) <= 1, f'per-client windows={sorted(evaluated)}')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--strict', action='store_true')
    args = parser.parse_args()

    run_dirs = find_runs()
    if not run_dirs:
        print('no runs found; nothing to validate')
        return 0
    report = Report()
    runs = {}
    for path in run_dirs:
        run = load_run(path)
        ident = identity(run)
        run_id = ident['run_id']
        runs[run_id] = run
        check_window_sequence(run_id, run['per_window'], report)
        check_label_causality(run_id, run['per_window'], report)
        check_prediction_alignment(run_id, run, report)
        check_update_accounting(run_id, run, report)
        check_eval_budget(run_id, run, report)
        check_transfer_cross_reference(run_id, run, report)
        status = run['config'].get('experiment_status')
        report.add(run_id, 'run_completed', status == 'completed',
                   f'status={status}')
    check_config_consistency(runs, report)

    frame = report.frame()
    failures = frame[frame['status'] == 'FAIL']
    print(frame.to_string(index=False, max_colwidth=60))
    print()
    print(f'{len(runs)} runs, {len(frame)} checks, {len(failures)} failed')
    if len(failures):
        print('\nfailed checks:')
        print(failures.to_string(index=False, max_colwidth=60))
    return 1 if (args.strict and len(failures)) else 0


if __name__ == '__main__':
    raise SystemExit(main())
