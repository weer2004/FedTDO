"""Learning-rate sweep for FedTDO on the standard long-term forecasting protocol.

Protocol (the one the benchmark defines, not the earlier ``sl96 -> pl24`` case
study that ``results/fedtdo_bsgrid`` used):

* ETT family: look-back 96 -> horizons {96, 192, 336, 720}
* ILI:        look-back 36 -> horizons {24, 36, 48, 60}
* batch size 256 (the official ``scripts/ETT*.sh`` batch), 7 clients, channel
  partition, 400 warm-up + 1200 downstream gradient steps per client, 60 online
  rounds, eval_budget 2, stream batch 32.

``--config official`` (default) takes the rest of the model config from the
per-dataset, per-horizon values in ``scripts/ETT*.sh`` / ``scripts/ILI.sh``
(``d_model``, ``dropout``, ``patch_len``, ``stride``, ``label_len``).  That is the
configuration the tables have to be reported at, and lr is strongly coupled to
width, so an lr found at ``d_model 32`` does not transfer to ``d_model 256``.
``--config fedtdo`` keeps the FedTDO grid config (``d_model 32``, patch 16 / stride
16) so a sweep can still be compared with ``results/fedtdo_bsgrid``.

Only ``--learning_rate`` and ``--model_id`` differ between the cells of one
(dataset, horizon) row; ``metrics.json`` records the full config, so every cell is
auditable from disk.

Selection metric
----------------
``mean_client_mse`` from ``metrics.json``.  ``--eval-split test`` (default)
reproduces the convention of every other table here; ``--eval-split val`` scores
the same online stream on the held-out split, which is what a sweep used to
*choose* a value must do.  The val stream is shorter (1719 windows/client on
ETTh1), so at stream batch 32 it delivers 54 of the 60 rounds.

Usage::

    python scripts/lr_sweep.py --dry-run
    python scripts/lr_sweep.py --datasets ETTh1 ETTh2 ETTm1 ETTm2
    python scripts/lr_sweep.py --eval-split val --seeds 2021 2022 2023
    python scripts/lr_sweep.py --summary-only
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

ROOT = Path(__file__).resolve().parents[1]

# Launch this script with the interpreter that has torch (conda activate ma_fmarl);
# override with FEDTDO_PYTHON.
PYTHON = os.environ.get('FEDTDO_PYTHON', sys.executable)

# Per dataset: paths, frequency, clients, look-back, horizon grid and the official
# training batch size (``batch`` = the value set at the top of the matching
# ``scripts/*.sh``: 256 for the ETT family and Weather, 128 Exchange, 64 ILI,
# 32 for the two 32-client sets Traffic/ECL where a larger batch would not fit).
DATASETS = {
    'ETTh1': dict(root='./dataset/ETT-small/', csv='ETTh1.csv', freq='h',
                  nc=7, seq_len=96, label_len=48, pred_lens=(96, 192, 336, 720),
                  batch=256),
    'ETTh2': dict(root='./dataset/ETT-small/', csv='ETTh2.csv', freq='h',
                  nc=7, seq_len=96, label_len=48, pred_lens=(96, 192, 336, 720),
                  batch=256),
    'ETTm1': dict(root='./dataset/ETT-small/', csv='ETTm1.csv', freq='t',
                  nc=7, seq_len=96, label_len=48, pred_lens=(96, 192, 336, 720),
                  batch=256),
    'ETTm2': dict(root='./dataset/ETT-small/', csv='ETTm2.csv', freq='t',
                  nc=7, seq_len=96, label_len=48, pred_lens=(96, 192, 336, 720),
                  batch=256),
    'Weather': dict(root='./dataset/weather/', csv='weather.csv', freq='10min',
                    nc=8, seq_len=96, label_len=48,
                    pred_lens=(96, 192, 336, 720), batch=256),
    'Exchange': dict(root='./dataset/exchange_rate/', csv='exchange_rate.csv',
                     freq='h', nc=8, seq_len=96, label_len=48,
                     pred_lens=(96, 192, 336, 720), batch=128),
    'ILI': dict(root='./dataset/illness/', csv='national_illness.csv', freq='w',
                nc=7, seq_len=36, label_len=18, pred_lens=(24, 36, 48, 60),
                batch=64),
    'ECL': dict(root='./dataset/electricity/', csv='electricity.csv', freq='h',
                nc=32, seq_len=96, label_len=48, pred_lens=(96, 192, 336, 720),
                batch=32),
    'Traffic': dict(root='./dataset/traffic/', csv='traffic.csv', freq='h',
                    nc=32, seq_len=96, label_len=48,
                    pred_lens=(96, 192, 336, 720), batch=32),
}

# Official (d_model, dropout, patch_len, stride, published lr) per horizon, copied
# from scripts/ETT*.sh, scripts/weather.sh, scripts/exchange.sh,
# scripts/ILI.sh, scripts/traffic.sh and scripts/ECL.sh.  The last field is only
# used as a reference point -- it is not passed to the run.
OFFICIAL = {
    'ETTh1': {96: (128, 0.4, 48, 12, 1.2e-3), 192: (256, 0.6, 48, 12, 8e-4),
              336: (256, 0.3, 48, 12, 4e-4), 720: (256, 0.8, 48, 12, 4e-4)},
    'ETTh2': {96: (128, 0.8, 48, 12, 8e-4), 192: (256, 0.6, 48, 12, 4e-4),
              336: (256, 0.6, 48, 12, 4e-4), 720: (128, 0.8, 48, 12, 4e-4)},
    'ETTm1': {96: (128, 0.4, 48, 12, 4e-4), 192: (128, 0.4, 48, 12, 4e-4),
              336: (128, 0.6, 48, 12, 1.6e-3), 720: (128, 0.4, 48, 12, 4e-4)},
    'ETTm2': {96: (256, 0.8, 48, 12, 8e-4), 192: (256, 0.8, 48, 12, 8e-4),
              336: (256, 0.8, 48, 12, 4e-4), 720: (128, 0.6, 48, 12, 1.2e-3)},
    'Weather': {96: (128, 0.1, 96, 24, 1.2e-3), 192: (128, 0.1, 96, 24, 4e-4),
                336: (256, 0.1, 96, 48, 6e-4), 720: (128, 0.1, 96, 12, 6e-4)},
    'Exchange': {96: (128, 0.3, 16, 8, 4e-4), 192: (512, 0.3, 32, 8, 1e-4),
                 336: (256, 0.5, 16, 8, 4e-4), 720: (256, 0.5, 16, 8, 8e-4)},
    'ILI': {24: (512, 0.1, 24, 4, 8e-4), 36: (1024, 0.1, 24, 4, 4e-4),
            48: (512, 0.3, 24, 4, 8e-4), 60: (512, 0.3, 24, 4, 4e-4)},
    'ECL': {96: (128, 0.1, 24, 6, 8e-4), 192: (128, 0.3, 24, 6, 8e-4),
            336: (256, 0.3, 24, 6, 1.2e-3), 720: (128, 0.3, 24, 6, 8e-4)},
    'Traffic': {96: (128, 0.1, 48, 6, 2.4e-3), 192: (128, 0.1, 48, 6, 1.2e-3),
                336: (128, 0.1, 48, 6, 1.8e-3), 720: (128, 0.1, 48, 6, 2.4e-3)},
}

# FedTDO grid preset: one config for every horizon, as in results/fedtdo_bsgrid.
FEDTDO_PRESET = dict(d_model=32, dropout=0.1, patch_len=16, stride=16)

LR_GRID = [1e-4, 2.5e-4, 5e-4, 1e-3, 2e-3]

# The online stream is decoupled from the training batch for the same reason as
# in data_provider/federated_data.py: 3461 test windows at batch 256 would run
# dry after 13 of the 60 rounds.  ILI's test split holds ~74 windows, so it needs
# a stream batch of 1 to deliver the full round budget.
STREAM_BATCH = {'ILI': 1}

SHARED = [
    '--enc_in', '1',
    '--fedtdo_control', 'fedtdo_joint',
    '--fedtdo_warmup_steps', '400', '--fedtdo_local_steps', '1200',
    '--fedtdo_online_steps', '60',
    '--fedtdo_eval_budget', '2',
]


def model_config(args, dataset: str, pred_len: int) -> dict:
    if args.config == 'fedtdo':
        return dict(FEDTDO_PRESET)
    d_model, dropout, patch_len, stride, _ = OFFICIAL[dataset][pred_len]
    return dict(d_model=d_model, dropout=dropout, patch_len=patch_len, stride=stride)


def official_lr(dataset: str, pred_len: int) -> float:
    return OFFICIAL[dataset][pred_len][4]


def run_name(dataset: str, pred_len: int, lr: float, batch_size: int, seed: int) -> str:
    return f'{dataset.lower()}_pl{pred_len}_bs{batch_size}_lr{lr:g}_s{seed}'


def output_dir(args, dataset: str, pred_len: int, lr: float, seed: int) -> Path:
    return Path(args.output_dir) / run_name(dataset, pred_len, lr, args.batch_size, seed)


def command(args, dataset: str, pred_len: int, lr: float, seed: int) -> list[str]:
    spec = DATASETS[dataset]
    config = model_config(args, dataset, pred_len)
    return [
        PYTHON, '-u', 'run_fedtdo.py',
        '--model_id', run_name(dataset, pred_len, lr, args.batch_size, seed),
        '--data', dataset,
        '--root_path', spec['root'], '--data_path', spec['csv'], '--freq', spec['freq'],
        '--seed', str(seed),
        '--fedtdo_num_clients', str(spec['nc']),
        '--seq_len', str(spec['seq_len']),
        '--label_len', str(spec['label_len']),
        '--pred_len', str(pred_len),
        '--d_model', str(config['d_model']),
        '--dropout', str(config['dropout']),
        '--patch_len', str(config['patch_len']),
        '--stride', str(config['stride']),
        '--batch_size', str(args.batch_size),
        '--learning_rate', f'{lr:g}',
        '--fedtdo_stream_batch', str(STREAM_BATCH.get(dataset, 32)),
        '--fedtdo_eval_split', args.eval_split,
        '--fedtdo_output_dir', args.output_dir,
        '--num_workers', str(args.num_workers),
        '--use_gpu',
        *SHARED,
    ]


def launch(args, job):
    dataset, pred_len, lr, seed, gpu = job
    name = run_name(dataset, pred_len, lr, args.batch_size, seed)
    logs = Path(args.output_dir) / '_logs'
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env['OMP_NUM_THREADS'] = env.get('XC_THREADS', '4')
    env['MKL_NUM_THREADS'] = env.get('XC_THREADS', '4')
    env['PYTHONUNBUFFERED'] = '1'
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    started = time.time()
    with open(logs / f'{name}.log', 'w', encoding='utf-8') as handle:
        code = subprocess.call(command(args, dataset, pred_len, lr, seed), cwd=str(ROOT),
                               stdout=handle, stderr=subprocess.STDOUT, env=env)
    elapsed = time.time() - started
    ok = code == 0 and (output_dir(args, dataset, pred_len, lr, seed) / 'metrics.json').exists()
    return name, ok, elapsed


def read_metrics(args):
    """{(dataset, pred_len): {lr: [metrics, ...]}} for finished cells."""
    table = {}
    root = Path(args.output_dir)
    if not root.exists():
        return table
    for path in sorted(root.glob('*/metrics.json')):
        try:
            with open(path, encoding='utf-8') as handle:
                result = json.load(handle)
        except json.JSONDecodeError:
            continue
        config = result.get('config', {})
        dataset, lr = config.get('data'), config.get('learning_rate')
        pred_len = config.get('pred_len')
        if dataset not in DATASETS or lr is None or pred_len is None:
            continue
        if config.get('batch_size') != args.batch_size:
            continue
        if result.get('eval_split', 'test') != args.eval_split:
            continue
        table.setdefault((dataset, pred_len), {}).setdefault(lr, []).append(result)
    return table


def summary(args):
    table = read_metrics(args)
    if not table:
        print(f'no finished runs under {args.output_dir}'
              f' (eval_split={args.eval_split}, batch={args.batch_size})')
        return
    lrs = sorted({lr for cells in table.values() for lr in cells})
    print(f'\nmean_client_mse on the online {args.eval_split} stream, batch '
          f'{args.batch_size}, config={args.config}')
    head = f"{'dataset':<8}{'pl':>5}{'seeds':>7}" + ''.join(f'{f"lr {lr:g}":>12}' for lr in lrs)
    head += f"{'best':>10}{'official':>10}"
    print(head)
    print('-' * len(head))
    missing = []
    best_rows = []
    for dataset in DATASETS:
        for pred_len in DATASETS[dataset]['pred_lens']:
            cells = table.get((dataset, pred_len))
            if not cells:
                continue
            scores, seeds = {}, 0
            for lr, runs in cells.items():
                values = [run['mean_client_mse'] for run in runs
                          if run.get('mean_client_mse') is not None]
                if values:
                    scores[lr] = sum(values) / len(values)
                    seeds = max(seeds, len(values))
            if not scores:
                continue
            winner = min(scores, key=scores.get)
            line = f'{dataset:<8}{pred_len:>5}{seeds:>7}'
            for lr in lrs:
                line += f'{scores[lr]:>12.4f}' if lr in scores else f'{"-":>12}'
            line += f'{winner:>10g}'
            line += (f'{official_lr(dataset, pred_len):>10g}'
                     if args.config == 'official' else f'{"-":>10}')
            print(line)
            best_rows.append((dataset, pred_len, winner, scores[winner]))
            absent = [lr for lr in lrs if lr not in scores]
            if absent:
                missing.append(f'{dataset} pl{pred_len}: ' + ', '.join(f'{lr:g}' for lr in absent))
    print('-' * len(head))
    for dataset, pred_len, lr, mse in best_rows:
        print(f'  best {dataset:<6} pl{pred_len:<4} lr={lr:<8g} mse={mse:.4f}')
    if missing:
        print('incomplete rows (lr still to run):')
        for item in missing:
            print('  ' + item)
    if args.eval_split == 'test':
        print('note: picked on the test stream. Re-check with --eval-split val before '
              'these values go into a paper table.')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--datasets', nargs='+', default=list(DATASETS),
                        choices=list(DATASETS))
    parser.add_argument('--lrs', nargs='+', type=float, default=LR_GRID)
    parser.add_argument('--pred-lens', nargs='+', type=int, default=None,
                        help='override the per-dataset horizon grid')
    parser.add_argument('--seeds', nargs='+', type=int, default=[2021])
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--config', default='official', choices=['official', 'fedtdo'])
    parser.add_argument('--eval-split', default='test', choices=['test', 'val'])
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--output-dir', default='./results/fedtdo_lr')
    parser.add_argument('--gpus', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--parallel', type=int, default=None)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--summary-only', action='store_true')
    parser.add_argument('--force', action='store_true', help='rerun finished cells')
    args = parser.parse_args()

    if args.summary_only:
        summary(args)
        return 0

    parallel = args.parallel if args.parallel is not None else max(1, len(args.gpus))
    jobs, skipped = [], 0
    for dataset in args.datasets:
        horizons = args.pred_lens or DATASETS[dataset]['pred_lens']
        for pred_len in horizons:
            if pred_len not in DATASETS[dataset]['pred_lens']:
                raise SystemExit(f'{dataset} has no official pl={pred_len} config')
            for lr in args.lrs:
                for seed in args.seeds:
                    if not args.force and (output_dir(args, dataset, pred_len, lr, seed)
                                           / 'metrics.json').exists():
                        skipped += 1
                        continue
                    jobs.append((dataset, pred_len, lr, seed,
                                 args.gpus[len(jobs) % len(args.gpus)]))

    print(f'jobs={len(jobs)} skipped={skipped} parallel={parallel} '
          f'batch={args.batch_size} eval_split={args.eval_split} config={args.config}')
    if args.dry_run:
        for job in jobs:
            print(f'  CUDA_VISIBLE_DEVICES={job[4]} '
                  + ' '.join(command(args, *job[:4])))
        return 0

    failures = []
    started = time.time()
    with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
        futures = {pool.submit(launch, args, job): job for job in jobs}
        for done, future in enumerate(as_completed(futures), start=1):
            name, ok, elapsed = future.result()
            print(f'[{done}/{len(jobs)}] {name:<40} {elapsed/60:5.1f} min '
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
