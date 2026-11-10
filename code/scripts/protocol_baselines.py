"""Matched local / FedAvg baselines for the standard long-term protocol.

The lr sweep (``scripts/lr_sweep.py``) answers "which lr", not "is FedTDO any
good".  A FedTDO number can only be judged next to a baseline that saw the same
data, the same backbone width and the same gradient budget, and none of the
existing ``results_baselines/BL_*`` cells do: those are ``sl96 -> pl24``,
``d_model 32``, ``batch 32``, 2 x 800 steps (``scripts/xc_baselines.py``).

This runner repeats the protocol table of the lr sweep -- same per-dataset
per-horizon ``d_model`` / ``dropout`` / ``patch_len`` / ``stride`` from
``scripts/ETT*.sh``, ``batch 256``, 7 channel clients, 1600 gradient steps per
client (``--fl_rounds 2 --fl_local_max_steps 800``, the same 400 + 1200 budget
FedTDO gets), 60 online rounds, stream batch 32 -- for the baselines.

``local`` is the decisive control: it is the same per-client model with no
sharing at all, so "FedTDO cannot even beat local" is exactly the hypothesis
this script can refute or confirm.

Usage::

    python scripts/protocol_baselines.py --dry-run
    python scripts/protocol_baselines.py --baselines local fedavg --parallel 16 --gpus 0 0 1
    python scripts/protocol_baselines.py --summary-only
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lr_sweep import DATASETS, OFFICIAL, model_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get('FEDTDO_PYTHON', sys.executable)
RESULTS = ROOT / 'results_baselines'
STREAM_BATCH = {'ILI': 1}

# 2 x 800 = the 400 warm-up + 1200 downstream steps of the FedTDO arm.
ROUNDS, LOCAL_STEPS, ONLINE_STEPS = 2, 800, 60


def setting_name(baseline, dataset, pred_len, batch_size, seed):
    """Must match exp_fl_baselines' ``BL_{baseline}_{model_id}_{data}_...`` format."""
    return (f'BL_{baseline}_proto_{dataset}_nc{DATASETS[dataset]["nc"]}'
            f'_sl{DATASETS[dataset]["seq_len"]}_pl{pred_len}'
            f'_dm{OFFICIAL[dataset][pred_len][0]}_r{ROUNDS}_bs{batch_size}_s{seed}')


def summary_path(args, baseline, dataset, pred_len, seed):
    return RESULTS / setting_name(baseline, dataset, pred_len, args.batch_size, seed) / 'summary.json'


def command(args, baseline, dataset, pred_len, seed):
    spec = DATASETS[dataset]
    config = model_config(args, dataset, pred_len)
    return [
        PYTHON, '-u', 'run_baselines.py',
        '--baseline', baseline,
        '--model_id', 'proto',
        '--des', f'bs{args.batch_size}_s{seed}',
        '--data', dataset,
        '--root_path', spec['root'], '--data_path', spec['csv'], '--freq', spec['freq'],
        '--seed', str(seed),
        '--num_clients', str(spec['nc']),
        '--seq_len', str(spec['seq_len']),
        '--label_len', str(spec['label_len']),
        '--pred_len', str(pred_len),
        '--d_model', str(config['d_model']),
        '--dropout', str(config['dropout']),
        '--patch_len', str(config['patch_len']),
        '--stride', str(config['stride']),
        '--batch_size', str(args.batch_size),
        '--learning_rate', f'{args.learning_rate:g}',
        '--fl_rounds', str(ROUNDS),
        '--fl_local_max_steps', str(LOCAL_STEPS),
        '--fl_online_steps', str(ONLINE_STEPS),
        '--fedtdo_stream_batch', str(STREAM_BATCH.get(dataset, 32)),
        '--num_workers', '0',
        '--num_threads', '4',
    ]


def launch(args, job):
    baseline, dataset, pred_len, seed, gpu = job
    name = setting_name(baseline, dataset, pred_len, args.batch_size, seed)
    logs = RESULTS / '_proto_logs'
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env['OMP_NUM_THREADS'] = '4'
    env['MKL_NUM_THREADS'] = '4'
    env['PYTHONUNBUFFERED'] = '1'
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    started = time.time()
    with open(logs / f'{name}.log', 'w', encoding='utf-8') as handle:
        code = subprocess.call(command(args, baseline, dataset, pred_len, seed),
                               cwd=str(ROOT), stdout=handle,
                               stderr=subprocess.STDOUT, env=env)
    ok = code == 0 and summary_path(args, baseline, dataset, pred_len, seed).exists()
    return name, ok, time.time() - started


def read_summaries(args):
    table = {}
    for baseline in args.baselines:
        for dataset in args.datasets:
            for pred_len in DATASETS[dataset]['pred_lens']:
                path = summary_path(args, baseline, dataset, pred_len, args.seeds[0])
                if not path.exists():
                    continue
                with open(path, encoding='utf-8') as handle:
                    data = json.load(handle)
                table[(baseline, dataset, pred_len)] = data['mean_mse']
    return table


def fedtdo_scores(args, dataset, pred_len):
    """mean_client_mse per lr from results/fedtdo_lr, for the same cell."""
    cells = {}
    pattern = f'{dataset.lower()}_pl{pred_len}_bs{args.batch_size}_lr*_s{args.seeds[0]}'
    for path in (ROOT / 'results' / 'fedtdo_lr').glob(f'{pattern}/metrics.json'):
        with open(path, encoding='utf-8') as handle:
            result = json.load(handle)
        if result.get('eval_split', 'test') != 'test':
            continue
        cells[result['config']['learning_rate']] = result['mean_client_mse']
    return cells


def summary(args):
    table = read_summaries(args)
    if not table:
        print('no finished baseline cells')
        return
    print(f'\nmean client MSE, batch {args.batch_size}, '
          f'{ROUNDS}x{LOCAL_STEPS} steps/client, {ONLINE_STEPS} online rounds')
    head = (f"{'dataset':<8}{'pl':>5}" + ''.join(f'{b:>12}' for b in args.baselines)
            + f'{"FedTDO@5e-4":>14}{"FedTDO best":>14}{"vs local":>10}')
    print(head)
    print('-' * len(head))
    for dataset in args.datasets:
        for pred_len in DATASETS[dataset]['pred_lens']:
            row = {b: table.get((b, dataset, pred_len)) for b in args.baselines}
            if all(value is None for value in row.values()):
                continue
            line = f'{dataset:<8}{pred_len:>5}'
            for baseline in args.baselines:
                line += f'{row[baseline]:>12.4f}' if row[baseline] else f'{"-":>12}'
            cells = fedtdo_scores(args, dataset, pred_len)
            if cells:
                fixed = cells.get(5e-4)
                best = min(cells.values())
                line += f'{fixed:>14.4f}{best:>14.4f}' if fixed else f'{"-":>14}{best:>14.4f}'
                base = row.get('local')
                if base and fixed:
                    line += f'{100*(fixed-base)/base:>9.1f}%'
                else:
                    line += f'{"-":>10}'
            else:
                line += f'{"-":>14}{"-":>14}{"-":>10}'
            print(line)
    print('-' * len(head))
    print('vs local: positive = FedTDO worse than local at the same budget')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--baselines', nargs='+', default=['local', 'fedavg'])
    parser.add_argument('--datasets', nargs='+', default=['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'],
                        choices=list(DATASETS))
    parser.add_argument('--pred-lens', nargs='+', type=int, default=None)
    parser.add_argument('--seeds', nargs='+', type=int, default=[2021])
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=5e-4)
    parser.add_argument('--config', default='official', choices=['official', 'fedtdo'])
    parser.add_argument('--output-dir', default=str(RESULTS))
    parser.add_argument('--gpus', nargs='+', type=int, default=[0, 0, 1])
    parser.add_argument('--parallel', type=int, default=16)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--summary-only', action='store_true')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    if args.summary_only:
        summary(args)
        return 0

    jobs, skipped = [], 0
    for baseline in args.baselines:
        for dataset in args.datasets:
            horizons = args.pred_lens or DATASETS[dataset]['pred_lens']
            for pred_len in horizons:
                for seed in args.seeds:
                    target = summary_path(args, baseline, dataset, pred_len, seed)
                    if target.exists() and not args.force:
                        skipped += 1
                        continue
                    jobs.append((baseline, dataset, pred_len, seed,
                                 args.gpus[len(jobs) % len(args.gpus)]))

    print(f'jobs={len(jobs)} skipped={skipped} parallel={args.parallel}')
    if args.dry_run:
        for job in jobs:
            print(f'  CUDA_VISIBLE_DEVICES={job[4]} ' + ' '.join(command(args, *job[:4])))
        return 0

    failures = []
    started = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        futures = {pool.submit(launch, args, job): job for job in jobs}
        for done, future in enumerate(as_completed(futures), start=1):
            name, ok, elapsed = future.result()
            print(f'[{done}/{len(jobs)}] {name:<58} {elapsed/60:5.1f} min '
                  f'{"ok" if ok else "FAILED"}', flush=True)
            if not ok:
                failures.append(name)
    print(f'\ndone in {(time.time() - started)/60:.1f} min, {len(failures)} failed')
    for name in failures:
        print('  FAILED', name)
    summary(args)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
