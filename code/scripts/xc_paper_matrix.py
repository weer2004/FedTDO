"""Drive the FFTS/FeDaL-protocol backbone matrix on one GPU.

Each cell is one ``run_paper_protocol.py`` process: a federation over the 9 main
datasets, dataset-level multivariate clients, one backbone.  Results land in
``results_paper/<setting>/summary.json``; ``--collect`` folds them into two CSVs
(long format + a wide MSE table that pastes straight into the paper).

Usage::

    python scripts/xc_paper_matrix.py --dry-run
    python scripts/xc_paper_matrix.py --backbones patchtst timesnet --parallel 2
    python scripts/xc_paper_matrix.py --collect
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results_paper'
LOGS = RESULTS / '_logs'
PYTHON = sys.executable

BACKBONES = ['dtmoe', 'patchtst', 'itransformer', 'transformer', 'gru',
             'autoformer', 'fedformer', 'timesnet', 'timemixer',
             'tfps', 'stmtm', 'raft', 'ffts', 'timeffm']


def setting_name(method: str, backbone: str, pred_len: int, channels: int,
                 rounds: int, tag: str, local_mode: str = 'steps',
                 epochs: int = 2) -> str:
    suffix = f'_{tag}' if tag else ''
    budget = f'_ep{epochs}' if local_mode == 'epochs' else ''
    return (f'PP_{method}_{backbone}_pl{pred_len}_dm32_c{channels}_r{rounds}'
            f'{budget}{suffix}')


def command(job: dict, gpu: int, threads: int) -> list:
    cmd = [PYTHON, str(ROOT / 'run_paper_protocol.py'),
           '--backbone', job['backbone'], '--method', job['method'],
           '--pred_len', str(job['pred_len']), '--paper_channels', str(job['channels']),
           '--fl_rounds', str(job['rounds']),
           '--local_mode', job['local_mode'], '--fl_local_epochs', str(job['epochs']),
           '--fl_local_max_steps', str(job['steps']),
           '--gpu', str(gpu), '--num_threads', str(threads)]
    if job['tag']:
        cmd += ['--tag', job['tag']]
    return cmd


def run_cell(job: dict, gpu: int, threads: int) -> tuple:
    setting = setting_name(job['method'], job['backbone'], job['pred_len'],
                           job['channels'], job['rounds'], job['tag'],
                           job['local_mode'], job['epochs'])
    summary = RESULTS / setting / 'summary.json'
    if summary.exists() and not job['force']:
        return setting, 'skip', 0.0
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f'{setting}.log'
    cmd = command(job, gpu, threads)
    t0 = time.time()
    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    env['OMP_NUM_THREADS'] = str(threads)
    env['MKL_NUM_THREADS'] = str(threads)
    with open(log, 'w', encoding='utf-8') as fh:
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT), env=env)
    return setting, ('done' if rc == 0 else f'FAIL(rc={rc})'), time.time() - t0


def collect(tag: str = '') -> None:
    rows = []
    for d in sorted(RESULTS.glob('PP_*')):
        f = d / 'summary.json'
        if not f.exists():
            continue
        s = json.loads(f.read_text(encoding='utf-8'))
        if tag and not d.name.endswith(tag):
            continue
        for name, r in s['per_dataset'].items():
            rows.append({
                'backbone': s['backbone'], 'method': s['method'],
                'pred_len': s['pred_len'], 'params': s.get('params'),
                'dataset': name, 'mse': r['mse'], 'mae': r['mae'],
                'trained': int(r['trained']),
            })
    if not rows:
        print('nothing to collect')
        return
    import csv
    long_path = RESULTS / 'paper_backbone_long.csv'
    with open(long_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # wide table: rows = (method, backbone), cols = dataset MSE
    datasets = sorted({r['dataset'] for r in rows})
    keys = sorted({(r['method'], r['backbone'], r['pred_len']) for r in rows})
    wide = RESULTS / 'paper_backbone_mse_wide.csv'
    with open(wide, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['method', 'backbone', 'pred_len', 'params'] + datasets + ['mean'])
        for (method, backbone, pl) in keys:
            sub = {r['dataset']: r for r in rows
                   if (r['method'], r['backbone'], r['pred_len']) == (method, backbone, pl)}
            vals = [sub[d]['mse'] if d in sub else float('nan') for d in datasets]
            params = next(iter(sub.values()))['params'] if sub else ''
            mean = sum(v for v in vals if v == v) / max(1, sum(1 for v in vals if v == v))
            w.writerow([method, backbone, pl, params] + [f'{v:.4f}' for v in vals]
                       + [f'{mean:.4f}'])
    print(f'wrote {long_path} and {wide} ({len(rows)} cells)')


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--backbones', nargs='+', default=BACKBONES, choices=BACKBONES)
    p.add_argument('--methods', nargs='+', default=['fedavg'], choices=['fedavg', 'local'])
    p.add_argument('--pred-lens', nargs='+', type=int, default=[96])
    p.add_argument('--channels', type=int, default=7)
    p.add_argument('--rounds', type=int, default=2)
    p.add_argument('--steps', type=int, default=800)
    p.add_argument('--tag', type=str, default='gpu')
    p.add_argument('--local-mode', type=str, default='steps', choices=['steps', 'epochs'])
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--parallel', type=int, default=2)
    p.add_argument('--force', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--collect-only', action='store_true')
    args = p.parse_args()

    if args.collect_only:
        collect(args.tag)
        return

    # the child processes inherit ``sys.executable`` (see PYTHON at the top), so
    # the orchestrator's interpreter decides every cell's environment -- print it
    # before anything else, and refuse LM backbones on the CPU interpreter.
    try:
        sys.path.insert(0, str(ROOT))
        from utils.env_banner import GPU_ONLY, print_env_banner
        print_env_banner(tag=args.tag,
                         feature=next((b for b in args.backbones
                                       if b in GPU_ONLY), ''))
    except ImportError:
        print(f'[env] interpreter={sys.executable}')

    jobs = []
    for method in args.methods:
        for backbone in args.backbones:
            for pl in args.pred_lens:
                jobs.append({'method': method, 'backbone': backbone, 'pred_len': pl,
                             'channels': args.channels, 'rounds': args.rounds,
                             'steps': args.steps, 'tag': args.tag, 'force': args.force,
                             'local_mode': args.local_mode, 'epochs': args.epochs})

    if args.dry_run:
        for job in jobs:
            print(' '.join(command(job, args.gpu, args.threads)))
        print(f'{len(jobs)} runs')
        return

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        futures = [pool.submit(run_cell, job, args.gpu, args.threads) for job in jobs]
        for fut in as_completed(futures):
            setting, status, secs = fut.result()
            print(f'[{status}] {setting}  {secs / 60:.1f} min', flush=True)
    print(f'all cells finished in {(time.time() - t0) / 60:.1f} min')
    collect(args.tag)


if __name__ == '__main__':
    main()
