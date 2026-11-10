"""Cross-dataset sweep: three rules x five seeds on every dataset.

The Traffic grid already exists (``traffic32_s*_b9_{joint_b9_fix,b9_center,b9_decorr}``).
This script fills the rest so that §19 can answer "is the structure Traffic-specific?"

Protocol per dataset (evidence density held near the Traffic operating point
``obs/arm = online_steps * eval_budget / (4*nc - 4)``):

| dataset | channels | nc | b | pool | obs/arm |
|---|---|---|---|---|---|
| ETTh1/h2 | 7 | 7 | 2 | 24 | 5.00 |
| ETTm1/m2 | 7 | 7 | 2 | 24 | 5.00 |
| Weather | 21 | 8 | 2 | 28 | 4.29 |
| Exchange | 8 | 8 | 2 | 28 | 4.29 |
| ILI | 7 | 7 | 2 | 24 | 5.00 |
| ECL | 321 | 32 | 9 | 124 | 4.35 |
| Traffic | 862 | 32 | 9 | 124 | 4.35 |
| Solar (control, not a DTMoe benchmark) | 137 | 8 | 2 | 28 | 4.29 |

Arms (identical flags to the Traffic grid):

* ``base``   : no flag            -> raw reward, iid radius
* ``center`` : --fedtdo_shock centered            -> paired reward, iid radius
* ``decorr`` : --fedtdo_shock centered --fedtdo_decorr_rho 0.365 -> + design-effect radius

Usage::

    python scripts/xc_sweep.py --datasets ETTh1 ETTm1 --arms base center decorr --parallel 3
    python scripts/xc_sweep.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results' / 'fedtdo'
LOGS = RESULTS / '_xc_logs'

# --------------------------------------------------------------------------- #
# Interpreter selection -- never hand-pick this per run.                       #
#                                                                              #
# Two environments coexist and are NOT interchangeable:                        #
#                                                                              #
#   CPU grid  D:/software/Anaconda3/python.exe          (torch 2.2.2+cpu)      #
#       Every number already on disk -- the channel-as-client FL table, the     #
#       FedTDO arms, the backbone axis -- was produced here.  Switching the     #
#       interpreter re-derives them with a different torch build, so the old    #
#       CSVs stop being comparable.                                            #
#                                                                              #
#   GPU env   D:/software/Anaconda3/envs/ma_fmarl/python.exe   (torch cu130)    #
#       Required for: any CUDA run, the dataset-level multivariate protocol     #
#       (`run_paper_protocol.py`, `run_paper_fedtdo.py`), and every LM-backed   #
#       backbone (FFTS / Time-FFM need `transformers` + the GPT-2 weights).     #
#                                                                              #
# `python_for(backbone)` encodes that rule so a job that needs CUDA or          #
# transformers can never be launched into the CPU interpreter by accident.      #
# Override both with the FEDTDO_PYTHON environment variable if a machine moves. #
# --------------------------------------------------------------------------- #
PYTHON_CPU = 'D:/software/Anaconda3/python.exe'
PYTHON_GPU = 'D:/software/Anaconda3/envs/ma_fmarl/python.exe'
GPU_ONLY_BACKBONES = {'ffts', 'timeffm'}
# kept for the many call sites that predate the split
PYTHON = os.environ.get('FEDTDO_PYTHON', PYTHON_CPU)


def python_for(backbone=None) -> str:
    """Interpreter for a unit of work.  GPU/LM backbones -> ma_fmarl."""
    override = os.environ.get('FEDTDO_PYTHON')
    if override:
        return override
    return PYTHON_GPU if str(backbone or '').lower() in GPU_ONLY_BACKBONES else PYTHON_CPU

# dataset -> (root_path, data_path, freq, num_clients, eval_budget)
#
# Only datasets that Dynamic TMoE actually benchmarks are in the main grid
# (ETT-small x4, Electricity/ECL, Traffic, Weather, Exchange, ILI).
# Solar-Energy is kept as an *external control* only: it is not a DTMoe
# benchmark, and its per-bet odds are inverted, so it cannot answer
# "is FedTDO's structure Traffic-specific?" without confounding the pool size.
DATASETS = {
    'ETTh1': ('./dataset/ETT-small/', 'ETTh1.csv', 'h', 7, 2),
    'ETTh2': ('./dataset/ETT-small/', 'ETTh2.csv', 'h', 7, 2),
    'ETTm1': ('./dataset/ETT-small/', 'ETTm1.csv', 't', 7, 2),
    'ETTm2': ('./dataset/ETT-small/', 'ETTm2.csv', 't', 7, 2),
    'Weather': ('./dataset/weather/', 'weather.csv', '10min', 8, 2),
    'Exchange': ('./dataset/exchange_rate/', 'exchange_rate.csv', 'h', 8, 2),
    'ILI': ('./dataset/illness/', 'national_illness.csv', 'w', 7, 2),
    'ECL': ('./dataset/electricity/', 'electricity.csv', 'h', 32, 9),
    'Traffic': ('./dataset/traffic/', 'traffic.csv', 'h', 32, 9),
    # external control, not a DTMoe benchmark
    'Solar': ('./dataset/solar-energy/', 'solar_energy.csv', '10min', 8, 2),
}

# arm -> extra flags appended to the command line
ARMS = {
    'base': [],
    'center': ['--fedtdo_shock', 'centered'],
    'decorr': ['--fedtdo_shock', 'centered', '--fedtdo_decorr_rho', '0.365'],
}

SHARED = [
    '--enc_in', '1',
    '--seq_len', '96', '--pred_len', '24', '--label_len', '24',
    '--patch_len', '16', '--stride', '16',
    '--d_model', '32', '--learning_rate', '5e-4', '--batch_size', '32',
    '--fedtdo_online_steps', '60',
    '--fedtdo_control', 'fedtdo_joint',
    '--fedtdo_warmup_steps', '400', '--fedtdo_local_steps', '1200',
]


# Per-dataset overrides for the *online stream* batch size.
#
# ILI has 966 weekly points, so its 20% test split holds only ~74 windows: at
# batch 32 the stream runs dry after 6 rounds and the "online" stage degenerates
# into cold start (see the ``ili_cold_*`` directories).  A stream batch of 1
# delivers the full 60-round budget while training keeps batch 32, so BatchNorm
# statistics are unaffected.
STREAM_BATCH = {
    'ILI': 1,
}


def run_name(dataset: str, arm: str, seed: int) -> str:
    root, _, _, nc, budget = DATASETS[dataset]
    return f'{dataset.lower()}{nc}_s{seed}_b{budget}_{arm}'


def output_dir(dataset: str, arm: str, seed: int) -> Path:
    return RESULTS / run_name(dataset, arm, seed)


def command(dataset: str, arm: str, seed: int) -> list[str]:
    root, data_path, freq, nc, budget = DATASETS[dataset]
    name = run_name(dataset, arm, seed)
    extra = []
    stream_batch = STREAM_BATCH.get(dataset)
    if stream_batch is not None:
        extra = ['--fedtdo_stream_batch', str(stream_batch)]
    return [
        PYTHON, '-u', 'run_fedtdo.py',
        '--model_id', name,
        '--data', dataset,
        '--root_path', root, '--data_path', data_path,
        '--freq', freq, '--seed', str(seed),
        '--fedtdo_num_clients', str(nc),
        '--fedtdo_eval_budget', str(budget),
        *SHARED,
        *extra,
        *ARMS[arm],
    ]


def launch(job):
    dataset, arm, seed = job
    name = run_name(dataset, arm, seed)
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f'{name}.log'
    env = dict(os.environ)
    # Two threads per process keeps three concurrent runs inside the memory budget.
    env['OMP_NUM_THREADS'] = env.get('XC_THREADS', '3')
    env['MKL_NUM_THREADS'] = env.get('XC_THREADS', '3')
    env['PYTHONUNBUFFERED'] = '1'
    started = time.time()
    with open(log_path, 'w', encoding='utf-8') as handle:
        code = subprocess.call(command(dataset, arm, seed), cwd=str(ROOT),
                               stdout=handle, stderr=subprocess.STDOUT, env=env)
    elapsed = time.time() - started
    ok = code == 0 and (output_dir(dataset, arm, seed) / 'metrics.json').exists()
    return name, ok, elapsed


MAIN_DATASETS = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Exchange', 'ILI',
                 'ECL', 'Traffic']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=MAIN_DATASETS,
                        choices=list(DATASETS))
    parser.add_argument('--arms', nargs='+', default=list(ARMS), choices=list(ARMS))
    parser.add_argument('--seeds', nargs='+', type=int,
                        default=[2021, 2022, 2023, 2024, 2025])
    parser.add_argument('--parallel', type=int, default=3)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--force', action='store_true', help='rerun finished cells')
    args = parser.parse_args()

    jobs = []
    skipped = 0
    for dataset in args.datasets:
        for arm in args.arms:
            for seed in args.seeds:
                if not args.force and (output_dir(dataset, arm, seed) / 'metrics.json').exists():
                    skipped += 1
                    continue
                jobs.append((dataset, arm, seed))

    print(f'jobs={len(jobs)} skipped={skipped} parallel={args.parallel}')
    if args.dry_run:
        for job in jobs:
            print(' ', ' '.join(command(*job)))
        return 0

    failures = []
    started = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        futures = {pool.submit(launch, job): job for job in jobs}
        for done, future in enumerate(as_completed(futures), start=1):
            name, ok, elapsed = future.result()
            stamp = f'[{done}/{len(jobs)}] {name:<34} {elapsed/60:5.1f} min'
            print(f'{stamp} {"ok" if ok else "FAILED"}', flush=True)
            if not ok:
                failures.append(name)

    wall = (time.time() - started) / 60
    print(f'\ndone in {wall:.1f} min, {len(failures)} failed')
    for name in failures:
        print('  FAILED', name)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
