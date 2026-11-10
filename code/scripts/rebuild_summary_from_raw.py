#!/usr/bin/env python
"""Rebuild ``summary.json`` for baseline cells from their paper-recording raw runs.

Why this exists
---------------
``utils/paper_recorder.finish`` used to crash on client ids that contain a
solidus ("sh (g/kg)" in Weather), *after* training and the whole online
evaluation had already been paid for.  Every Weather baseline therefore died at
the very last statement and left no ``summary.json`` -- but it did leave
``per_window_metrics.csv``, which carries ``sse``/``sae``/``num_targets`` for
every (client, window).  Those three columns are sufficient to reproduce the
exact numbers ``exp/exp_fl_baselines.online_evaluate`` writes:

    mse_c  = sum(sse_c) / sum(num_targets_c)
    mae_c  = sum(sae_c) / sum(num_targets_c)
    rmse_c = sqrt(mse_c)
    mean_mse = mean_c(mse_c)        (mean over clients, not over windows)

Verified bit-for-bit against the ETTh1 / ETTh2 / Exchange cells that do have a
``summary.json`` (``--verify-existing``).

Usage
-----
    python scripts/rebuild_summary_from_raw.py --datasets Weather --verify-existing
    python scripts/rebuild_summary_from_raw.py --datasets Weather
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW_ROOT = ROOT / 'fixed' / 'results' / 'paper_fedtdo' / 'raw'
OUT_ROOT = ROOT / 'fixed' / 'results' / 'baselines'
SEQ_LEN = {'ETTh1': 96, 'ETTh2': 96, 'ETTm1': 96, 'ETTm2': 96,
           'Weather': 96, 'Exchange': 96, 'ECL': 96, 'Traffic': 96,
           'ILI': 36}
ONLINE_WINDOWS = 1920
ONLINE_LR = {'ILI': 1e-4}


def expected_online_lr(dataset: str) -> float:
    return float(ONLINE_LR.get(dataset, 1e-4))


def registry_meta() -> dict:
    """{method: venue} for the 21 table rows.

    Anything else under ``raw/<dataset>`` (``fedtdo_nofreeze`` is an ablation arm,
    not a baseline) must not be turned into a ``BL_*`` cell, or the table silently
    grows from 21 rows to 22.
    """
    try:
        import sys
        sys.path.insert(0, str(ROOT))
        from fl_baselines.registry import list_baselines
        return {m['name']: m.get('venue', '') for m in list_baselines()}
    except Exception as exc:                                # pragma: no cover
        print(f'[warn] registry unavailable ({exc}); using every raw method')
        return {}


def cell_dir_name(method: str, cfg: dict) -> str:
    """Same identity string as ``scripts/fl_baseline_compare.setting_name``."""
    dataset = cfg['dataset']
    rounds = int(cfg.get('fl_rounds') or 5)
    local = int(cfg.get('fl_local_epochs') or 12)
    epochs = rounds * local
    lr = float(cfg.get('online_lr') or 0.0)
    stream = int(cfg.get('stream_batch') or 1)
    return (f'BL_{method}_fl_baseline_{dataset}_nc{int(cfg["num_clients"])}'
            f'_sl{int(cfg.get("seq_len") or SEQ_LEN.get(dataset, 96))}'
            f'_pl{int(cfg["pred_len"])}_dm{int(cfg["d_model"])}'
            f'_r{rounds}_ep{epochs}_s{int(cfg["seed"])}'
            f'_olr{lr:g}_sb{stream}_os')


def read_run(run_dir: Path) -> tuple[dict, dict, int, dict]:
    """(config, per-client {mse, mae, rmse}, rows, windows-per-client)."""
    cfg = json.load(open(run_dir / 'config.json', encoding='utf-8'))
    sse = defaultdict(float)
    sae = defaultdict(float)
    num = defaultdict(float)
    counts = defaultdict(int)
    rows = 0
    with open(run_dir / 'per_window_metrics.csv', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            cid = row['client_id']
            sse[cid] += float(row['sse'])
            sae[cid] += float(row['sae'])
            num[cid] += float(row['num_targets'])
            counts[cid] += 1
            rows += 1
    clients = {}
    for cid in sse:
        n = num[cid]
        if not n:
            continue
        mse = sse[cid] / n
        clients[cid] = {'mse': mse, 'mae': sae[cid] / n,
                        'rmse': math.sqrt(mse) if mse >= 0 else float('nan')}
    return cfg, clients, rows, dict(counts)


def config_ok(run_dir: Path, dataset: str) -> bool:
    cfg_path, csv_path = run_dir / 'config.json', run_dir / 'per_window_metrics.csv'
    if not (cfg_path.exists() and csv_path.exists()):
        return False
    cfg = json.load(open(cfg_path, encoding='utf-8'))
    if cfg.get('dataset') != dataset:
        return False
    if abs(float(cfg.get('online_lr') or 0.0) - expected_online_lr(dataset)) > 1e-12:
        return False
    if int(cfg.get('stream_batch') or 1) != 1:
        return False
    if int(cfg.get('fl_online_steps') or ONLINE_WINDOWS) != ONLINE_WINDOWS:
        return False
    return True


def complete(counts: dict) -> bool:
    """Every client scored on the same number of windows.

    The recorder walks clients in the outer loop, so an aborted run leaves the
    last client short and the counts disagree.  The absolute window count is
    dataset dependent (Exchange 1424, ILI 171), so it is taken as the modal value
    across a dataset rather than hard-coded.
    """
    return bool(counts) and len(set(counts.values())) == 1


def pick_run(method_dir: Path, dataset: str):
    runs = [p for p in sorted(method_dir.glob('run_*')) if p.is_dir()]
    good = [p for p in runs if config_ok(p, dataset)]
    if not good:
        return None
    return max(good, key=lambda p: p.stat().st_mtime)


def aggregate(clients: dict) -> dict:
    mses = [v['mse'] for v in clients.values() if math.isfinite(v['mse'])]
    maes = [v['mae'] for v in clients.values() if math.isfinite(v['mae'])]
    finite = {k: v for k, v in clients.items() if math.isfinite(v['mse'])}
    return {
        'mean_mse': sum(mses) / len(mses) if mses else None,
        'mean_mae': sum(maes) / len(maes) if maes else None,
        'median_mse': sorted(mses)[len(mses) // 2] if mses else None,
        'median_mae': sorted(maes)[len(maes) // 2] if maes else None,
        'std_mse': (sum((x - sum(mses) / len(mses)) ** 2 for x in mses) / len(mses)) ** 0.5
        if mses else None,
        'max_mse': max(mses) if mses else None,
        'min_mse': min(mses) if mses else None,
        'num_clients': len(clients),
        'clients': clients,
        'finite_clients': len(finite),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--datasets', nargs='+', default=['Weather'])
    parser.add_argument('--methods', nargs='+', default=None)
    parser.add_argument('--raw-root', default=str(RAW_ROOT))
    parser.add_argument('--out-root', default=str(OUT_ROOT))
    parser.add_argument('--verify-existing', action='store_true',
                        help='compare against an existing summary.json, write nothing')
    parser.add_argument('--force', action='store_true',
                        help='overwrite an existing summary.json')
    args = parser.parse_args()

    raw_root, out_root = Path(args.raw_root), Path(args.out_root)
    written = skipped = failed = 0
    worst = 0.0
    for dataset in args.datasets:
        ds_root = raw_root / dataset
        if not ds_root.is_dir():
            print(f'[skip] no raw tree for {dataset}')
            continue
        meta = registry_meta()
        methods = args.methods or [p.name for p in sorted(ds_root.iterdir())
                                   if p.is_dir() and (not meta or p.name in meta)]
        print(f'=== {dataset} ({len(methods)} methods) ===')

        # Pass 1: newest config-matching run per method, plus its window counts.
        candidates = {}
        for method in methods:
            seed_dir = ds_root / method / 'seed_2021'
            run = pick_run(seed_dir, dataset) if seed_dir.is_dir() else None
            if run is None:
                print(f'  {method:12s} no raw run')
                failed += 1
                continue
            cfg, clients, rows, counts = read_run(run)
            candidates[method] = (run, cfg, clients, rows, counts)
        modal = Counter(c[4][next(iter(c[4]))]
                        for c in candidates.values() if complete(c[4]))
        expected_windows = modal.most_common(1)[0][0] if modal else 0
        print(f'  modal windows/client = {expected_windows} '
              f'({len(candidates)} candidate runs)')

        for method in methods:
            if method not in candidates:
                continue
            run, cfg, clients, rows, counts = candidates[method]
            if not complete(counts) or \
                    counts[next(iter(counts))] != expected_windows:
                print(f'  {method:12s} window counts off '
                      f'({sorted(set(counts.values()))} vs {expected_windows})')
                failed += 1
                continue
            stats = aggregate(clients)
            summary = {
                'setting': cell_dir_name(method, cfg),
                'method': method,
                'venue': meta.get(method, ''),
                'reconstructed_from': str(run.relative_to(ROOT)),
                'reconstructed_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                'num_online_logs': None,
                'train': {'step_mode': 'epochs',
                          'rounds': int(cfg.get('fl_rounds') or 5),
                          'local_epochs': int(cfg.get('fl_local_epochs') or 12),
                          'patience': None,
                          'lr_online': float(cfg.get('online_lr') or 0.0)},
            }
            summary.update(stats)
            target_dir = Path(args.out_root) / summary['setting']
            target = target_dir / 'summary.json'

            # A cell whose per-client MSE is non-finite cannot be summarised by a
            # mean, and the reference code writes ``nan`` for it.  Refuse rather
            # than silently averaging over the finite subset.
            if stats['finite_clients'] != stats['num_clients']:
                print(f'  {method:12s} non-finite client MSE '
                      f'({stats["finite_clients"]}/{stats["num_clients"]}) -- '
                      f'not reconstructable, needs a rerun')
                failed += 1
                continue

            if args.verify_existing:
                if not target.exists():
                    print(f'  {method:12s} rebuilt={stats["mean_mse"]:.9f} '
                          f'(no reference on disk) rows={rows}')
                    continue
                ref = json.load(open(target, encoding='utf-8'))
                diff = abs(float(ref['mean_mse']) - float(stats['mean_mse']))
                worst = max(worst, diff)
                print(f'  {method:12s} ref={ref["mean_mse"]:.9f} '
                      f'rebuilt={stats["mean_mse"]:.9f} |d|={diff:.2e} rows={rows}')
                continue
            if target.exists() and not args.force:
                print(f'  {method:12s} exists, skipped')
                skipped += 1
                continue
            target_dir.mkdir(parents=True, exist_ok=True)
            with open(target, 'w', encoding='utf-8') as handle:
                json.dump(summary, handle, indent=2, ensure_ascii=False)
            print(f'  {method:12s} mse={stats["mean_mse"]:.6f} '
                  f'mae={stats["mean_mae"]:.6f} clients={stats["finite_clients"]}'
                  f'/{stats["num_clients"]} rows={rows} -> {summary["setting"]}')
            written += 1

    if args.verify_existing:
        print(f'\nmax |delta| vs existing summaries: {worst:.3e}')
    else:
        print(f'\nwritten={written} skipped={skipped} failed={failed}')
    return 0


if __name__ == '__main__':
    os.chdir(ROOT)
    raise SystemExit(main())
