"""Sweep the transfer controller on the channel partition (eval budget x threshold).

Why this sweep
--------------
On ETTh1 pl96 with the DTMoe-aligned protocol, the channel partition shows:

    local (no sharing)                0.4329 / 0.4472
    + shared frozen encoder, no transfer  0.4455 / 0.4479   (+2.9% cost)
    + transfer                        0.4454 / 0.4469   (transfer ~0)

Transfer triggered on 13.6% of client-rounds and was *positive* on every one of
them (c3 -4.25%, c4 -3.85%, c5 -2.59%, c1 -2.54%), but 13.6% x ~3% cannot pay for
the 2.9% the frozen encoder costs.  So the two knobs that can change the order
of magnitude are:

* ``--fedtdo_eval_budget``: how many experts the controller may probe per round.
  At 2 it samples 2 candidates out of ~28 published capsules, so the selection
  signal is close to chance (53-65% of probes look positive).
* ``--fedtdo_transfer_threshold``: how much predicted gain is required before a
  deployment.  At 0.0 (default) any non-negative prediction deploys, which is
  how ``c6_OT`` lost 16% in the step-budget run.

``--fedtdo_control fedtdo_nofreeze`` is swept as the second axis because the
2.9% loss is attributable to freezing, not to transfer: the same warm-up without
the freeze tells us whether the shared encoder should stay trainable.

Every cell keeps the DTMoe config of ``scripts/ETT*.sh`` (d_model / dropout /
patch / stride), batch 256, ``--fedtdo_step_mode epochs``, 60 downstream epochs,
patience 15 and the type3 schedule, so rows are comparable with the baseline
table that ``run_baselines.py --fl_step_mode epochs`` produces.

Usage::

    python scripts/transfer_sweep.py --dry-run
    python scripts/transfer_sweep.py --parallel 16 --gpus 0 0 1
    python scripts/transfer_sweep.py --summary-only
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

from lr_sweep import DATASETS, model_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get('FEDTDO_PYTHON', sys.executable)

# eval_budget 2 is the current default; 4/8 give the controller a real sample.
EVAL_BUDGETS = [2, 4, 8]
# 0.0 = deploy on any non-negative prediction (current default).
THRESHOLDS = [0.0, 0.05, 0.10]
ARMS = ['fedtdo_joint', 'fedtdo_nofreeze']

SHARED = [
    '--enc_in', '7', '--features', 'M',
    '--fedtdo_partition', 'channel',
    '--fedtdo_step_mode', 'epochs',
    '--fedtdo_warmup_epochs', '5',
    '--fedtdo_epochs', '60',
    '--fedtdo_patience', '15',
    '--fedtdo_lradj', 'type3',
    '--fedtdo_stream_batch', '32',
    '--fedtdo_online_steps', '60',
]


def tag(arm: str) -> str:
    return 'nofreeze' if arm == 'fedtdo_nofreeze' else 'joint'


def run_name(dataset, pred_len, budget, threshold, arm, batch, seed):
    return (f'tr_{dataset.lower()}_pl{pred_len}_eb{budget}'
            f'_th{threshold:g}_{tag(arm)}_bs{batch}_s{seed}')


def output_dir(args, *key):
    return Path(args.output_dir) / run_name(*key, args.batch_size, args.seeds[0])


def command(args, dataset, pred_len, budget, threshold, arm):
    spec = DATASETS[dataset]
    config = model_config(args, dataset, pred_len)
    return [
        PYTHON, '-u', 'run_fedtdo.py',
        '--model_id', run_name(dataset, pred_len, budget, threshold, arm,
                               args.batch_size, args.seeds[0]),
        '--data', dataset,
        '--root_path', spec['root'], '--data_path', spec['csv'], '--freq', spec['freq'],
        '--seed', str(args.seeds[0]),
        '--fedtdo_num_clients', str(spec['nc']),
        '--seq_len', str(spec['seq_len']), '--label_len', str(spec['label_len']),
        '--pred_len', str(pred_len),
        '--d_model', str(config['d_model']), '--dropout', str(config['dropout']),
        '--patch_len', str(config['patch_len']), '--stride', str(config['stride']),
        '--batch_size', str(args.batch_size),
        '--learning_rate', f'{args.learning_rate:g}',
        '--fedtdo_control', arm,
        '--fedtdo_eval_budget', str(budget),
        '--fedtdo_transfer_threshold', f'{threshold:g}',
        '--fedtdo_output_dir', args.output_dir,
        '--num_workers', '0',
        '--use_gpu',
        *SHARED,
    ]


def launch(args, job):
    dataset, pred_len, budget, threshold, arm, gpu = job
    name = run_name(dataset, pred_len, budget, threshold, arm,
                    args.batch_size, args.seeds[0])
    logs = Path(args.output_dir) / '_logs'
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env['OMP_NUM_THREADS'] = '4'
    env['MKL_NUM_THREADS'] = '4'
    env['PYTHONUNBUFFERED'] = '1'
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    started = time.time()
    with open(logs / f'{name}.log', 'w', encoding='utf-8') as handle:
        code = subprocess.call(command(args, dataset, pred_len, budget, threshold, arm),
                               cwd=str(ROOT), stdout=handle,
                               stderr=subprocess.STDOUT, env=env)
    ok = code == 0 and (Path(args.output_dir) / name / 'metrics.json').exists()
    return name, ok, time.time() - started


def summary(args):
    root = Path(args.output_dir)
    rows = {}
    for path in sorted(root.glob('*/metrics.json')):
        with open(path, encoding='utf-8') as handle:
            d = json.load(handle)
        config = d['config']
        budget = config.get('fedtdo_eval_budget')
        threshold = config.get('fedtdo_transfer_threshold')
        arm = tag(config.get('fedtdo_control', ''))
        if threshold is None:
            # Runs written before the threshold was added to the recorded config
            # whitelist: the value is still in the directory name.
            stem = path.parent.name
            for part in stem.split('_'):
                if part.startswith('th'):
                    try:
                        threshold = float(part[2:])
                    except ValueError:
                        pass
                if part.startswith('eb'):
                    try:
                        budget = int(part[2:])
                    except ValueError:
                        pass
        if budget is None or threshold is None:
            continue
        key = (config.get('data'), config.get('pred_len'), budget, threshold, arm)
        gain = d['actions']['relative_gain']
        rows[key] = (d['mse'], d['mae'], gain, d['actions']['transfers'],
                     d['actions']['clients_with_active_expert'])
    if not rows:
        print(f'no finished runs under {root}')
        return
    print(f'\n{"dataset":<8}{"pl":>5}{"eb":>4}{"th":>6}{"arm":>10}'
          f'{"MSE":>9}{"MAE":>9}{"online gain":>13}{"部署轮次":>9}{"活跃客户端":>11}')
    print('-' * 84)
    best = {}
    for key in sorted(rows):
        dataset, pred_len, budget, threshold, arm = key
        mse, mae, gain, transfers, active = rows[key]
        print(f'{dataset:<8}{pred_len:>5}{budget:>4}{threshold:>6g}{arm:>10}'
              f'{mse:>9.4f}{mae:>9.4f}{100*gain:>12.2f}%{transfers:>9}{active:>11}')
        slot = (dataset, pred_len, arm)
        if slot not in best or gain > best[slot][0]:
            best[slot] = (gain, budget, threshold)
    print('-' * 84)
    for (dataset, pred_len, arm), (gain, budget, threshold) in sorted(best.items()):
        print(f'  best online gain {dataset} pl{pred_len} {arm}: '
              f'eb={budget} th={threshold:g} -> {100*gain:+.2f}%')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--datasets', nargs='+', default=['ETTh1'], choices=list(DATASETS))
    parser.add_argument('--pred-lens', nargs='+', type=int, default=[96])
    parser.add_argument('--budgets', nargs='+', type=int, default=EVAL_BUDGETS)
    parser.add_argument('--thresholds', nargs='+', type=float, default=THRESHOLDS)
    parser.add_argument('--arms', nargs='+', default=ARMS)
    parser.add_argument('--seeds', nargs='+', type=int, default=[2021])
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=1.2e-3)
    parser.add_argument('--config', default='official', choices=['official', 'fedtdo'])
    parser.add_argument('--output-dir', default='./results/fedtdo_transfer')
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
    for dataset in args.datasets:
        for pred_len in args.pred_lens:
            for budget in args.budgets:
                for threshold in args.thresholds:
                    for arm in args.arms:
                        target = output_dir(args, dataset, pred_len, budget, threshold, arm)
                        if (target / 'metrics.json').exists() and not args.force:
                            skipped += 1
                            continue
                        jobs.append((dataset, pred_len, budget, threshold, arm,
                                     args.gpus[len(jobs) % len(args.gpus)]))

    print(f'jobs={len(jobs)} skipped={skipped} parallel={args.parallel}')
    if args.dry_run:
        for job in jobs:
            print(f'  CUDA_VISIBLE_DEVICES={job[5]} '
                  + ' '.join(command(args, *job[:5])))
        return 0

    failures = []
    started = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        futures = {pool.submit(launch, args, job): job for job in jobs}
        for done, future in enumerate(as_completed(futures), start=1):
            name, ok, elapsed = future.result()
            print(f'[{done}/{len(jobs)}] {name:<46} {elapsed/60:5.1f} min '
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
