"""Batch-size sweep on the DTMoe-aligned (epochs) protocol.

Why this sweep
--------------
``results/fedtdo_bsgrid`` swept batch on the **old** protocol (fixed 400 + 1200
gradient steps, constant lr, ``sl96 -> pl24`` grid config).  That protocol makes
batch size a proxy for "how many epochs were seen", so the measured curve was
mostly a training-length effect.

``PROTOCOL_DTMOE_ALIGNED.md`` replaces the step budget with whole epochs
(``--fedtdo_step_mode epochs``, 60 downstream epochs, patience 15, type3 lr
decay) and the official per-dataset/per-horizon model config (d_model / dropout
/ patch / stride / lr from ``scripts/ETT*.sh`` and ``scripts/ILI.sh``).  Under
that protocol the number of epochs is held fixed by construction, so batch size
only changes (a) how many updates one epoch has and (b) the gradient noise.
This script measures that, with everything else pinned:

* the model config is ``lr_sweep.model_config`` (``--config official``);
* the lr is the published value for the cell (batch is *not* rescaled, because
  that would confound two effects);
* the online stream batch stays 32 (ILI 1) so every cell is scored on the same
  windows, whatever the training batch is;
* the same seed for every cell of a row, so cells are paired.

Usage::

    python scripts/bs_sweep.py --dry-run
    python scripts/bs_sweep.py --datasets ETTh1 --pred-lens 96 \
        --batch-sizes 128 256 512 --parallel 3 --gpus 0 1
    python scripts/bs_sweep.py --summary-only
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

from lr_sweep import DATASETS, OFFICIAL, STREAM_BATCH, model_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get('FEDTDO_PYTHON', sys.executable)

# Reference cell of every row: the official DTMoe batch.
REFERENCE_BATCH = 256

SHARED = [
    '--enc_in', '1',
    '--fedtdo_partition', 'channel',
    '--fedtdo_control', 'fedtdo_joint',
    '--fedtdo_step_mode', 'epochs',
    '--fedtdo_warmup_epochs', '5',
    '--fedtdo_epochs', '60',
    '--fedtdo_patience', '15',
    '--fedtdo_lradj', 'type3',
    '--fedtdo_online_steps', '60',
    '--fedtdo_eval_budget', '2',
]


def default_pred_len(dataset: str) -> int:
    return DATASETS[dataset]['pred_lens'][0]


def run_name(dataset: str, pred_len: int, batch: int, seed: int) -> str:
    return f'be_{dataset.lower()}_pl{pred_len}_bs{batch}_s{seed}'


def output_dir(args, dataset, pred_len, batch, seed) -> Path:
    return Path(args.output_dir) / run_name(dataset, pred_len, batch, seed)


def command(args, dataset: str, pred_len: int, batch: int, seed: int) -> list[str]:
    spec = DATASETS[dataset]
    config = model_config(args, dataset, pred_len)
    lr = args.learning_rate or OFFICIAL[dataset][pred_len][4]
    return [
        PYTHON, '-u', 'run_fedtdo.py',
        '--model_id', run_name(dataset, pred_len, batch, seed),
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
        '--batch_size', str(batch),
        '--learning_rate', f'{lr:g}',
        '--fedtdo_stream_batch', str(STREAM_BATCH.get(dataset, 32)),
        '--fedtdo_eval_split', args.eval_split,
        '--fedtdo_output_dir', args.output_dir,
        '--num_workers', '0',
        '--use_gpu',
        *SHARED,
    ]


def launch(args, job):
    dataset, pred_len, batch, seed, gpu = job
    name = run_name(dataset, pred_len, batch, seed)
    logs = Path(args.output_dir) / '_logs'
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env['OMP_NUM_THREADS'] = env.get('XC_THREADS', '4')
    env['MKL_NUM_THREADS'] = env.get('XC_THREADS', '4')
    env['PYTHONUNBUFFERED'] = '1'
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    started = time.time()
    with open(logs / f'{name}.log', 'w', encoding='utf-8') as handle:
        code = subprocess.call(command(args, dataset, pred_len, batch, seed),
                               cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT,
                               env=env)
    elapsed = time.time() - started
    ok = code == 0 and (Path(args.output_dir) / name / 'metrics.json').exists()
    return name, ok, elapsed


def read_metrics(args):
    """{(dataset, pred_len, batch): {seed: metrics}} for finished cells."""
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
        dataset, batch = config.get('data'), config.get('batch_size')
        pred_len, seed = config.get('pred_len'), config.get('seed')
        if dataset not in DATASETS or batch is None or pred_len is None:
            continue
        if result.get('eval_split', 'test') != args.eval_split:
            continue
        table.setdefault((dataset, pred_len, batch), {})[seed] = result
    return table


def summary(args):
    table = read_metrics(args)
    if not table:
        print(f'no finished runs under {args.output_dir} (eval_split={args.eval_split})')
        return
    rows = sorted({(k[0], k[1]) for k in table})
    batches = sorted({k[2] for k in table})
    print(f'\nmean_client_mse / mse / mae on the online {args.eval_split} stream')
    header = f"{'dataset':<8}{'pl':>5}{'seeds':>7}" + ''.join(
        f'{f"bs{b} mse":>12}' for b in batches)
    header += f"{'Δ% vs bs256':>13}{'epochs':>8}"
    print(header)
    print('-' * len(header))
    for dataset, pred_len in rows:
        seeds = max(len(table.get((dataset, pred_len, b), {})) for b in batches)
        line = f'{dataset:<8}{pred_len:>5}{seeds:>7}'
        ref = table.get((dataset, pred_len, REFERENCE_BATCH), {})
        ref_mse = (sum(r['mse'] for r in ref.values()) / len(ref)) if ref else None
        cells = []
        for b in batches:
            runs = table.get((dataset, pred_len, b), {})
            if not runs:
                cells.append(None)
                line += f'{"-":>12}'
                continue
            mse = sum(r['mse'] for r in runs.values()) / len(runs)
            cells.append((b, mse))
            line += f'{mse:>12.5f}'
        if ref_mse:
            best = min(cells, key=lambda c: c[1]) if cells else None
            line += f'{best[0]:>8}' if best else f'{"":>8}'
            line += f'{(best[1] - ref_mse) / ref_mse * 100:>+12.1f}%' if best else ''
        print(line)
    print('-' * len(header))
    for dataset, pred_len in rows:
        for b in batches:
            runs = table.get((dataset, pred_len, b), {})
            for seed, r in sorted(runs.items()):
                epochs = (r.get('train', {}).get('per_client') or [{}])
                ep = sum(c.get('epochs_run', 0) for c in epochs) / max(1, len(epochs))
                print(f'{dataset:<8}{pred_len:>5}bs{b:<5}s{seed}  '
                      f'mse={r["mse"]:.5f} mae={r["mae"]:.5f} '
                      f'client_mse={r.get("mean_client_mse", float("nan")):.5f} '
                      f'epochs={ep:.1f}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Batch-size sweep, DTMoe-aligned FedTDO')
    parser.add_argument('--datasets', nargs='+', default=['ETTh1'],
                        choices=list(DATASETS))
    parser.add_argument('--pred-lens', nargs='+', type=int, default=None,
                        help='horizons per dataset; default = first horizon of each '
                             '(96 for ETT, 24 for ILI)')
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[128, 256, 512])
    parser.add_argument('--seeds', nargs='+', type=int, default=[2021])
    parser.add_argument('--config', default='official', choices=['official', 'fedtdo'])
    parser.add_argument('--eval-split', default='test', choices=['test', 'val'])
    parser.add_argument('--learning-rate', type=float, default=None,
                        help='override the published lr (default: per-cell official value)')
    parser.add_argument('--parallel', type=int, default=4)
    parser.add_argument('--gpus', nargs='+', type=int, default=[0])
    parser.add_argument('--output-dir', default='./results/fedtdo_bsepoch')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--summary-only', action='store_true')
    args = parser.parse_args()

    if args.summary_only:
        summary(args)
        return

    jobs, gpu_index = [], 0
    for dataset in args.datasets:
        horizons = args.pred_lens or [default_pred_len(dataset)]
        for pred_len in horizons:
            if pred_len not in DATASETS[dataset]['pred_lens']:
                raise SystemExit(f'{dataset} has no pl{pred_len}; '
                                 f'choose from {DATASETS[dataset]["pred_lens"]}')
            for batch in args.batch_sizes:
                for seed in args.seeds:
                    jobs.append((dataset, pred_len, batch, seed,
                                 args.gpus[gpu_index % len(args.gpus)]))
                    gpu_index += 1

    if args.dry_run:
        for job in jobs:
            print(' '.join(command(args, *job[:4])))
        return

    print(f'{len(jobs)} runs -> {args.output_dir} '
          f'(parallel {args.parallel}, gpus {args.gpus})')
    failed = []
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(launch, args, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            name, ok, elapsed = future.result()
            print(f'{"ok  " if ok else "FAIL"} {name:38} {elapsed / 60:6.1f} min', flush=True)
            if not ok:
                failed.append(name)
    print(f'\ndone: {len(jobs) - len(failed)}/{len(jobs)} ok')
    if failed:
        print('failed:', ', '.join(failed))
    summary(args)


if __name__ == '__main__':
    main()
