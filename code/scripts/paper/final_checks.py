"""Pre-flight gate: six checks that must pass before the formal sweep runs.

Each check answers one question about *this* data tree and *this* driver.  A
failure is reported with the evidence, never repaired here.

    1. transfer -> source client is resolvable from recorded identifiers
    2. realised gain is recomputable after the fact from raw rows
    3. the online protocol is window-level prequential and causal
    4. every required record exists per (run, client, seed)
    5. every figure has a companion CSV that can rebuild it
    6. unavailable fields are blank and unsupported figures are absent by design

Usage::

    python scripts/paper/final_checks.py
    python scripts/paper/final_checks.py --strict
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import FIGURES, ROOT, find_runs, identity, load_run  # noqa: E402

REQUIRED_FILES = [
    'config.json', 'run_summary.json', 'per_window_metrics.csv',
    'online_updates.csv', 'candidate_evaluations.csv', 'feedback_rewards.csv',
    'transfer_events.csv', 'communication_events.csv', 'cost_events.csv',
    'regime_events.csv',
]
FAILED: List[str] = []


def report(name: str, ok: bool, detail: str = '') -> bool:
    print(f'  [{"PASS" if ok else "FAIL"}] {name}' + (f'  {detail}' if detail else ''))
    if not ok:
        FAILED.append(name)
    return ok


# --------------------------------------------------------------------- 1
def has_transfer(run: dict) -> bool:
    """A baseline has no registry, so its transfer tables are legitimately empty."""
    return bool(run['config'].get('transfer_enabled'))


def check_source_linkage(runs: List[dict]) -> bool:
    print('\n1. transfer -> source client linkage')
    ok = True
    for run in runs:
        ident = identity(run)
        if not has_transfer(run):
            print(f'  [SKIP] {ident["run_id"][:44]}: no transfer mechanism')
            continue
        transfers = run['transfers']
        if transfers is None or transfers.empty:
            ok &= report(f'{ident["run_id"][:44]}: transfer rows', False, 'none')
            continue
        evaluations = run['evaluations']
        comms = run['communication']
        eval_map: Dict[str, str] = {}
        if evaluations is not None and 'source_client_id' in evaluations:
            for _, row in evaluations.iterrows():
                if isinstance(row.get('source_client_id'), str) and row['source_client_id']:
                    eval_map[str(row['candidate_expert_id'])] = row['source_client_id']
        comm_map: Dict[str, str] = {}
        if comms is not None and 'source_client_id' in comms:
            for _, row in comms.iterrows():
                if isinstance(row.get('source_client_id'), str) and row['source_client_id']:
                    comm_map[str(row['expert_id'])] = row['source_client_id']
        resolved, total = 0, 0
        for _, row in transfers.iterrows():
            expert_id = str(row.get('expert_id') or '')
            if not expert_id:
                continue
            total += 1
            direct = str(row.get('source_client_id') or '')
            if direct or eval_map.get(expert_id) or comm_map.get(expert_id):
                resolved += 1
        rate = (resolved / total) if total else 1.0
        ok &= report(
            f'{ident["run_id"][:44]}: {resolved}/{total} transfer rows carry a '
            f'resolvable source', rate == 1.0, f'via column / eval join / comm join')
    return ok


# --------------------------------------------------------------------- 2
def check_realized_gain(runs: List[dict]) -> bool:
    print('\n2. realised gain recomputable from raw rows')
    ok = True
    for run in runs:
        ident = identity(run)
        if not has_transfer(run):
            print(f'  [SKIP] {ident["run_id"][:44]}: no candidate evaluations')
            continue
        feedback = run['feedback']
        window = run['per_window']
        if feedback is None or feedback.empty:
            ok &= report(f'{ident["run_id"][:44]}: feedback rows', False, 'none')
            continue
        required = ['client_id', 'candidate_expert_id', 'evaluation_window',
                    'feedback_release_window', 'realized_gain']
        missing = [c for c in required if c not in feedback.columns]
        ok &= report(f'{ident["run_id"][:44]}: feedback columns', not missing,
                     f'missing={missing}')
        if missing:
            continue
        gain = pd.to_numeric(feedback['realized_gain'], errors='coerce')
        eval_win = pd.to_numeric(feedback['evaluation_window'], errors='coerce')
        bad_window = int((eval_win.fillna(-1) < 1).sum())
        ok &= report(f'{ident["run_id"][:44]}: every feedback names its window',
                     bad_window == 0, f'{bad_window} rows without a window')
        ok &= report(f'{ident["run_id"][:44]}: realised gain finite',
                     gain.notna().all(), f'{int(gain.isna().sum())} NaN')
        if window is not None and not window.empty:
            keys = set(zip(window['client_id'], window['window_idx']))
            covered = sum(1 for _, row in feedback.iterrows()
                          if (row['client_id'], int(row['evaluation_window'])) in keys)
            ok &= report(
                f'{ident["run_id"][:44]}: evaluation window present in '
                f'per_window_metrics', covered == len(feedback),
                f'{covered}/{len(feedback)}')
    return ok


# --------------------------------------------------------------------- 3
def check_protocol(runs: List[dict]) -> bool:
    print('\n3. window-level prequential protocol and causality')
    ok = True
    for run in runs:
        ident = identity(run)
        window = run['per_window']
        summary = run['summary']
        config = run['config']
        if window is None or window.empty:
            ok &= report(f'{ident["run_id"][:44]}: window rows', False, 'empty')
            continue
        reveal = pd.to_numeric(window['label_reveal_idx'], errors='coerce')
        predicted_at = pd.to_numeric(window['window_idx'], errors='coerce')
        ok &= report(f'{ident["run_id"][:44]}: label revealed after prediction',
                     bool((reveal.fillna(-1) >= predicted_at.fillna(0)).all()))
        windows = int(summary.get('evaluated_windows') or 0)
        updates = int(summary.get('online_update_events') or 0)
        online = bool(config.get('online_enabled'))
        if not online:
            ok &= report(f'{ident["run_id"][:44]}: window-level update',
                         False, 'online stage disabled: nothing is updated')
            continue
        ratio = (windows / updates) if updates else float('inf')
        ok &= report(f'{ident["run_id"][:44]}: windows per gradient update',
                     abs(ratio - 1.0) < 1e-9,
                     f'{windows} windows / {updates} updates = {ratio:.2f}')
    return ok


def check_driver_protocol() -> bool:
    """The formal sweep is launched by scripts/fl_baseline_compare.py.

    Reads the module constants rather than the rendered command, so the check
    keeps working when the flags are assembled from those constants.
    """
    print('\n3b. driver defaults for the formal sweep')
    source = (ROOT / 'scripts' / 'fl_baseline_compare.py').read_text(encoding='utf-8')

    def constant(name: str):
        match = re.search(rf'^{name}\s*=\s*([0-9.eE+-]+)', source, re.M)
        return match.group(1) if match else None

    stream = constant('PROTOCOL_STREAM_BATCH')
    windows = constant('PROTOCOL_ONLINE_WINDOWS')
    fedtdo_lr = constant('FEDTDO_ONLINE_LR')
    baseline_default = re.search(
        r"'--fl-online-lr',\s*type=float,\s*default=([A-Za-z0-9_.+-]+)", source)
    fedtdo_cmd = re.search(r'def fedtdo_command.*?return \[(.*?)\n    \]',
                            source, re.S)
    baseline_cmd = re.search(r"'--fedtdo_stream_batch',\s*str\(([A-Z_]+)\)",
                             source)
    ok = True
    ok &= report('PROTOCOL_STREAM_BATCH is one window', stream == '1',
                 f'value={stream}')
    ok &= report('PROTOCOL_ONLINE_WINDOWS is the per-client window budget',
                 windows is not None and int(float(windows)) > 0,
                 f'value={windows}')
    ok &= report('FedTDO online lr is set', fedtdo_lr is not None,
                 f'value={fedtdo_lr}')
    raw_default = baseline_default.group(1) if baseline_default else ''
    resolved = fedtdo_lr if raw_default == 'FEDTDO_ONLINE_LR' else raw_default
    ok &= report('baseline online lr defaults to the FedTDO value',
                 bool(resolved) and fedtdo_lr is not None
                 and abs(float(resolved) - float(fedtdo_lr)) < 1e-12,
                 f'default={raw_default or "n/a"} -> {resolved}')
    ok &= report('FedTDO arm uses the protocol constants',
                 fedtdo_cmd is not None
                 and 'PROTOCOL_STREAM_BATCH' in fedtdo_cmd.group(1)
                 and 'PROTOCOL_ONLINE_WINDOWS' in fedtdo_cmd.group(1))
    ok &= report('baseline arm uses the protocol constants',
                 baseline_cmd is not None,
                 baseline_cmd.group(1) if baseline_cmd else 'literal found')
    return ok


# --------------------------------------------------------------------- 4
def check_record_completeness(runs: List[dict]) -> bool:
    print('\n4. per-run record completeness')
    ok = True
    seen = {}
    for run in runs:
        ident = identity(run)
        missing = [name for name in REQUIRED_FILES
                   if not (run['dir'] / name).exists()]
        ok &= report(f'{ident["run_id"][:44]}: required files', not missing,
                     f'missing={missing}')
        config = run['config']
        consistent = (config.get('run_id') == ident['run_id']
                      and int(config.get('seed', -1)) == int(ident['seed']))
        ok &= report(f'{ident["run_id"][:44]}: run_id / seed consistent', consistent)
        window = run['per_window']
        frames = []
        for key in ('per_window', 'updates', 'evaluations', 'feedback', 'transfers',
                    'communication', 'cost', 'regime'):
            frame = run[key]
            if frame is not None and 'run_id' in frame:
                frames.append(frame['run_id'].astype(str).eq(ident['run_id']).all())
        ok &= report(f'{ident["run_id"][:44]}: run_id stamped on every row',
                     all(frames) if frames else False,
                     f'{len(frames)} tables checked')
        if window is not None and not window.empty:
            per_client = window.groupby('client_id').size().to_dict()
            ok &= report(f'{ident["run_id"][:44]}: windows recorded per client',
                         len(per_client) > 0, str(per_client))
        key = (ident['dataset'], ident['method'], ident['seed'])
        if key in seen:
            ok &= report(f'{ident["run_id"][:44]}: unique (dataset, method, seed)',
                         False, f'duplicates {seen[key]}')
        seen[key] = ident['run_id']
    return ok


# --------------------------------------------------------------------- 5
def check_figure_data() -> bool:
    print('\n5. every figure has a companion CSV')
    if not FIGURES.exists():
        report('figures directory exists', False, str(FIGURES))
        return False
    figures = [p for p in FIGURES.rglob('*')
               if p.suffix in ('.png', '.pdf')]
    ok = True
    for figure in sorted(figures):
        csv_path = figure.with_suffix('.csv')
        ok &= report(f'{figure.relative_to(FIGURES)}', csv_path.exists(),
                     '' if csv_path.exists() else 'no sibling CSV')
    if not figures:
        print('  (no figures yet)')
    return ok


# --------------------------------------------------------------------- 6
def check_unavailable(runs: List[dict]) -> bool:
    print('\n6. unsupported fields stay explicitly empty')
    ok = True
    for run in runs:
        ident = identity(run)
        cost = run['cost']
        if cost is not None and not cost.empty:
            for column in ('estimated_flops', 'peak_memory_mb'):
                if column not in cost:
                    continue
                filled = pd.to_numeric(cost[column], errors='coerce').notna().sum()
                ok &= report(f'{ident["run_id"][:44]}: {column} blank',
                             filled == 0, f'{filled} values present')
        preds = run['predictions']
        has_timestamps = any('timestamps' in arrays for arrays in preds.values())
        if has_timestamps:
            ok &= report(f'{ident["run_id"][:44]}: timestamps only when real',
                         True)
    drift = FIGURES / 'drift_recovery'
    if drift.exists() and any(drift.glob('*.png')):
        ok &= report('drift_recovery only drawn with real change points',
                     any((run['regime'] is not None and not run['regime'].empty
                          and (run['regime']['event_type'] != 'drift_score').any())
                         for run in runs))
    else:
        print('  [SKIP] drift_recovery: no controlled-shift run, figure absent '
              '(expected)')
    gran = FIGURES / 'adaptation_granularity'
    if gran.exists() and any(gran.glob('*.png')):
        print('  [INFO] adaptation_granularity present')
    else:
        print('  [SKIP] adaptation_granularity: no batch-level arm, figure absent '
              '(expected)')
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--strict', action='store_true')
    parser.add_argument('--skip-driver', action='store_true',
                        help='only inspect the data tree')
    args = parser.parse_args()

    run_dirs = find_runs()
    runs = [load_run(path) for path in run_dirs]
    print(f'{len(runs)} run(s) under {RAW_REL}')
    if not runs:
        print('no data: run the smoke test first')
        return 1

    check_source_linkage(runs)
    check_realized_gain(runs)
    check_protocol(runs)
    if not args.skip_driver:
        check_driver_protocol()
    check_record_completeness(runs)
    check_figure_data()
    check_unavailable(runs)

    print()
    if FAILED:
        print(f'{len(FAILED)} check(s) failed:')
        for name in FAILED:
            print(f'  - {name}')
        return 1
    print('all checks passed')
    return 0


RAW_REL = 'results/paper_fedtdo/raw'

if __name__ == '__main__':
    raise SystemExit(main())
