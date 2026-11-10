"""Evaluation-budget sensitivity: b=2 vs b=9 on identical client sets, paired.

Why this exists
---------------
``scripts/xc_sweep.py`` does NOT use one parameter set everywhere: ``nc`` and
``--fedtdo_eval_budget`` are per-dataset (7/2 for the 7-channel sets, 8/2 for
Weather and Exchange, 32/9 for ECL and Traffic).  Budget 2 and 9 were both
chosen so that ``obs/arm = online_steps * budget / (4*nc - 4)`` lands near the
Traffic operating point of 4.35 -- i.e. the grid is aligned on *evidence
density*, not on parameter values.

"Adopt Traffic's settings everywhere" therefore changes exactly one thing of
substance: the budget, 2 -> 9, which lifts obs/arm by 4.5x on the small sets.
This script measures what that buys.

Design
------
* ``nc`` is held at each dataset's current value, so the client set is
  bit-identical between the two budgets and the only thing that moves is how
  many arms get evaluated per round.  Comparing client sets of different size
  would not be a budget experiment (the channels sampled by seed 2021 are not
  interchangeable -- see HETEROGENEITY.md).
* Both cells run on the same interpreter, so the torch-build difference that
  separates the historical CPU grid (2.2.2+cpu) from the GPU env (2.14+cu130)
  cancels out in the paired difference.
* The arm is ``base`` (no ``--fedtdo_shock`` / ``--fedtdo_decorr_rho``), i.e.
  the column the main table reports.

Usage::

    python scripts/xc_budget_sens.py --parallel 2
    python scripts/xc_budget_sens.py --dry-run
    python scripts/xc_budget_sens.py --report      # summarise finished cells
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
RESULTS = ROOT / 'results' / 'fedtdo_bsens'
LOGS = RESULTS / '_logs'

# CUDA is required, so this always has to be the GPU interpreter.
PYTHON = os.environ.get(
    'FEDTDO_PYTHON', 'D:/software/Anaconda3/envs/ma_fmarl/python.exe')

# dataset -> (root_path, data_path, freq, nc)
#
# nc is deliberately NOT unified: it is the control variable of this experiment.
# Forcing nc=32 would be a no-op on the 7-channel sets anyway -- the channel
# partition silently truncates to the number of columns.
DATASETS = {
    'ETTh1': ('./dataset/ETT-small/', 'ETTh1.csv', 'h', 7),
    'ETTh2': ('./dataset/ETT-small/', 'ETTh2.csv', 'h', 7),
    'ETTm1': ('./dataset/ETT-small/', 'ETTm1.csv', 't', 7),
    'ETTm2': ('./dataset/ETT-small/', 'ETTm2.csv', 't', 7),
    'Exchange': ('./dataset/exchange_rate/', 'exchange_rate.csv', 'h', 8),
}

# 2 = what xc_sweep uses today; 9 = the Traffic budget carried over verbatim.
BUDGETS = (2, 9)

SHARED = [
    '--enc_in', '1',
    '--seq_len', '96', '--pred_len', '24', '--label_len', '24',
    '--patch_len', '16', '--stride', '16',
    '--d_model', '32', '--learning_rate', '5e-4', '--batch_size', '32',
    '--fedtdo_online_steps', '60',
    '--fedtdo_control', 'fedtdo_joint',
    '--fedtdo_warmup_steps', '400', '--fedtdo_local_steps', '1200',
    '--use_gpu', '--gpu', '0',
]


def run_name(dataset: str, budget: int, seed: int) -> str:
    return f'{dataset.lower()}{DATASETS[dataset][3]}_b{budget}_s{seed}'


def output_dir(dataset: str, budget: int, seed: int) -> Path:
    return RESULTS / run_name(dataset, budget, seed)


def command(dataset: str, budget: int, seed: int) -> list[str]:
    root, data_path, freq, nc = DATASETS[dataset]
    return [
        PYTHON, '-u', 'run_fedtdo.py',
        '--model_id', run_name(dataset, budget, seed),
        '--data', dataset,
        '--root_path', root, '--data_path', data_path,
        '--freq', freq, '--seed', str(seed),
        '--fedtdo_num_clients', str(nc),
        '--fedtdo_eval_budget', str(budget),
        *SHARED,
        '--fedtdo_output_dir', str(RESULTS),
    ]


def launch(job):
    dataset, budget, seed = job
    name = run_name(dataset, budget, seed)
    LOGS.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env['PYTHONUNBUFFERED'] = '1'
    started = time.time()
    with open(LOGS / f'{name}.log', 'w', encoding='utf-8') as handle:
        code = subprocess.call(command(*job), cwd=str(ROOT),
                               stdout=handle, stderr=subprocess.STDOUT, env=env)
    elapsed = time.time() - started
    ok = code == 0 and (output_dir(dataset, budget, seed) / 'metrics.json').exists()
    return name, ok, elapsed


def _read(dataset: str, budget: int, seed: int):
    path = output_dir(dataset, budget, seed) / 'metrics.json'
    if not path.exists():
        return None
    metrics = json.loads(path.read_text(encoding='utf-8'))
    actions = metrics.get('actions') or {}
    return {
        'mse': metrics.get('mse'),
        'mean_client_mse': metrics.get('mean_client_mse'),
        'rel_gain': actions.get('relative_gain'),
        'evaluations': actions.get('evaluations'),
        'transfers': actions.get('transfers'),
        'active_clients': actions.get('clients_with_active_expert'),
        'online_steps': metrics.get('online_steps'),
    }


def report(seeds: list[int]) -> int:
    header = (f"{'dataset':9}{'nc':>4}{'pool':>5}"
              f"{'MSE b2':>10}{'MSE b9':>10}{'d%':>8}"
              f"{'gain b2':>9}{'gain b9':>9}{'evals':>14}")
    print(header)
    print('-' * len(header))
    deltas, gains = [], []
    for dataset, (_, _, _, nc) in DATASETS.items():
        pool = 4 * (nc - 1)
        for seed in seeds:
            low, high = _read(dataset, 2, seed), _read(dataset, 9, seed)
            if low is None or high is None:
                missing = 'b2' if low is None else 'b9'
                print(f'{dataset:9}{nc:>4}{pool:>5}  ({missing} missing)')
                continue
            delta = 100.0 * (low['mse'] - high['mse']) / low['mse']
            deltas.append(delta)
            gains.append(100.0 * (high['rel_gain'] - low['rel_gain']))
            tag = f'{dataset}' if len(seeds) == 1 else f'{dataset} s{seed}'
            print(f"{tag:9}{nc:>4}{pool:>5}"
                  f"{low['mse']:>10.5f}{high['mse']:>10.5f}{delta:>+8.2f}"
                  f"{100 * low['rel_gain']:>+9.3f}{100 * high['rel_gain']:>+9.3f}"
                  f"{low['evaluations']:>7}/{high['evaluations']:<7}")
    if deltas:
        worse = sum(1 for d in deltas if d > 0)
        print(f'\nbudget 2 -> 9: mean MSE change {sum(deltas) / len(deltas):+.2f}% '
              f'(positive = b9 better), median {sorted(deltas)[len(deltas) // 2]:+.2f}%, '
              f'b9 better in {worse}/{len(deltas)} cells')
        print(f'               mean rel_gain shift {sum(gains) / len(gains):+.3f}pp')
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=list(DATASETS),
                        choices=list(DATASETS))
    parser.add_argument('--budgets', nargs='+', type=int, default=list(BUDGETS),
                        choices=list(BUDGETS))
    parser.add_argument('--seeds', nargs='+', type=int, default=[2021])
    parser.add_argument('--parallel', type=int, default=2)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    if args.report:
        return report(args.seeds)

    jobs = []
    skipped = 0
    for dataset in args.datasets:
        for budget in args.budgets:
            for seed in args.seeds:
                if not args.force and (output_dir(dataset, budget, seed) / 'metrics.json').exists():
                    skipped += 1
                    continue
                jobs.append((dataset, budget, seed))

    print(f'jobs={len(jobs)} skipped={skipped} parallel={args.parallel} -> {RESULTS}')
    if args.dry_run:
        for job in jobs:
            print(' ', ' '.join(command(*job)))
        return 0
    if not jobs:
        return report(args.seeds)

    failures = []
    started = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        futures = {pool.submit(launch, job): job for job in jobs}
        for done, future in enumerate(as_completed(futures), start=1):
            name, ok, elapsed = future.result()
            print(f'[{done}/{len(jobs)}] {name:<26} {elapsed / 60:5.1f} min '
                  f'{"ok" if ok else "FAILED"}', flush=True)
            if not ok:
                failures.append(name)

    print(f'\ndone in {(time.time() - started) / 60:.1f} min, {len(failures)} failed')
    for name in failures:
        print('  FAILED', name)
    report(args.seeds)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
