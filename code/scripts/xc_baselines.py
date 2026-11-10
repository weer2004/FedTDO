"""Cross-dataset FL baseline sweep, budget-matched to the FedTDO grid.

Why ``--fl_rounds 2 --fl_local_max_steps 800``
-----------------------------------------------
``Exp_FedTDO`` trains each client for ``fedtdo_warmup_steps=400`` plus
``fedtdo_local_steps=1200`` gradient steps and performs exactly **one**
communication: ``server.synchronize_patch_encoder`` averages the patch encoder
once, then freezes it and the client keeps training.  The baselines therefore get
1600 gradient steps, one aggregation, and -- this is the part that is easy to get
wrong -- **training after the aggregation**.

``FedMethod.train_offline`` runs ``local_update`` then ``aggregate`` per round,
so ``--fl_rounds 1`` would train 1600 steps and *then* average, leaving every
client with an un-finetuned mixture of the others.  Measured on ETTh1 fedavg that
protocol mismatch alone costs 2.2x MSE (0.734 vs 0.334 for local) and would have
been read as "FedAvg is worse than no federation".  Two rounds of 800 put the
aggregation at the same relative position FedTDO uses (somewhere in the middle,
with a large downstream phase), while keeping total steps and sync count equal.

Giving the baselines 5 rounds would hand them 8000 steps and five syncs -- a
different and more generous budget than the method receives.

``--fl_local_max_steps`` counts *gradient steps* and loops across epochs
(``tests/test_baseline_budget.py``), so 800 steps is 800 steps regardless of how
many batches one epoch holds.  Passing epochs instead silently undertrained the
baselines 4x on Traffic.

Protocol parity checklist (any mismatch makes the tables incomparable):

* same channel subset      -> ``--num_clients`` + ``--seed`` (identical sampler)
* same online horizon      -> ``--fedtdo_online_steps 60`` (a batch budget)
* same stream batch        -> ``--fedtdo_stream_batch 1`` on ILI only
* same architecture knob   -> seq 96 / pred 24 / d_model 32 / patch 16 / stride 16

Usage::

    python scripts/xc_baselines.py --dry-run
    python scripts/xc_baselines.py --methods local fedavg --datasets ETTh1 --seeds 2021
    python scripts/xc_baselines.py --parallel 2
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from xc_sweep import (DATASETS, MAIN_DATASETS, PYTHON, ROOT,
                      STREAM_BATCH, python_for)  # noqa: E402

RESULTS = ROOT / 'results_baselines'
LOGS = RESULTS / '_xc_logs'

# 16 methods under one 1600-step / 1-aggregation budget.
#   local    : no communication at all        (what FedTDO's default path is)
#   fedavg   : parameter averaging            (the canonical FL baseline)
#   --- optimisation-level family (all keep the FedAvg skeleton) ---
#   fedprox  : proximal term                  (client drift / heterogeneity)
#   scaffold : control variates               (update-drift correction)
#   fedadam  : server adaptive optimiser      (FedOpt)
#   fednova  : normalised averaging           (unequal local work)
#   --- personalisation family (shared body, local head / local model) ---
#   fedper   : shared body + local head
#   fedrep   : head-first / body-second decoupling
#   ditto    : global ERM model + proximal personalised model
#   pfedme   : Moreau-envelope personalised theta
#   --- representation-learning family ---
#   fedl2p   : meta-net -> layer-wise personalisation LRs
#   pfedfda  : feature-distribution shrinkage on personalised heads
#   fedpoe   : output-ensemble personalisation
#   pfedcs   : clustered client selection
#   fedpkda  : noisy clipped prototypes + dynamic alignment
#   fedkme   : RFF kernel mean embeddings -> collaboration weights
METHODS = ['local', 'fedavg', 'fedprox', 'scaffold', 'fedadam', 'fednova',
           'fedper', 'fedrep', 'ditto', 'pfedme',
           'fedl2p', 'pfedfda', 'fedpoe', 'pfedcs', 'fedpkda', 'fedkme',
           'fedda', 'fedgtp']

# Deliberately *out of the table* but still installed: ``fedala`` and ``fedkwaz``
# were the two strongest single-channel baselines we had (rank 2.44 / 4.67), and
# dropping the strong ones while keeping the weak ones is exactly the move a
# reviewer will read as baseline shopping.  They are excluded at the user's
# explicit request ("暂时不要"), kept runnable so the exclusion is auditable, and
# must be restored before submission -- see CROSS_DATASET_REPORT.md §3.5.
PARKED = ['fedala', 'fedkwaz']
# ``fedavg_sgd`` is run as an appendix control (optimiser-matched reference for
# SCAFFOLD) and is never a main-table column.
ALL_METHODS = METHODS + PARKED + ['fedavg_sgd']

SEQ_LEN, PRED_LEN, LABEL_LEN = 96, 24, 24
PATCH_LEN, STRIDE, D_MODEL = 16, 16, 32
BATCH_SIZE, LR = 32, 5e-4
LOCAL_STEPS = 800       # 2 x 800 = the same 1600 steps as 400 + 1200
ROUNDS = 2              # one aggregation, placed mid-training like FedTDO's
ONLINE_STEPS = 60

# Per-method overrides appended *after* BASE_SHARED, so argparse takes the last
# value.  Only methods whose defining hyper-parameter cannot be set by a shared
# default appear here, and each value was picked by a mini-sweep on ETTh1:
#   fedadam  : server lr {1e-4, 0.01, 0.05} -> 0.881 / 0.608 / 0.741, best 0.01
#   scaffold : SGD is the only dimensionally correct optimiser for its variate,
#              lr {5e-4, 5e-3} -> 0.794 / 0.564 (paired fedavg_sgd 0.793 / 0.550)
#   fedavg_sgd : same sweep, kept as the optimiser-matched control for SCAFFOLD
METHOD_ARGS = {
    'fedadam': ['--fedadam_lr', '0.01'],
    'scaffold': ['--learning_rate', '5e-3', '--scaffold_opt', 'sgd'],
    'fedavg_sgd': ['--learning_rate', '5e-3'],
}

BASE_SHARED = [
    '--enc_in', '1',
    '--seq_len', str(SEQ_LEN), '--pred_len', str(PRED_LEN), '--label_len', str(LABEL_LEN),
    '--patch_len', str(PATCH_LEN), '--stride', str(STRIDE),
    '--d_model', str(D_MODEL), '--learning_rate', str(LR), '--batch_size', str(BATCH_SIZE),
    '--fl_rounds', str(ROUNDS), '--fl_local_max_steps', str(LOCAL_STEPS),
    '--fedtdo_online_steps', str(ONLINE_STEPS),
    '--num_workers', '0',
]


def setting_name(method: str, dataset: str, seed: int, tag: str = '') -> str:
    """Must match the ``setting`` built in run_baselines.py exactly.

    ``tag`` is passed through as ``--des <tag>exp`` so a backbone swap lands in
    ``BL_..._r2_bbpatchtst_exp`` instead of overwriting the Dynamic_TMoE row.
    """
    _, _, _, nc, _ = DATASETS[dataset]
    suffix = f'{tag}_' if tag else ''
    return (f'BL_{method}_xc_s{seed}_{dataset}_nc{nc}_'
            f'sl{SEQ_LEN}_pl{PRED_LEN}_dm{D_MODEL}_r{ROUNDS}_{suffix}exp')


def output_dir(method: str, dataset: str, seed: int, tag: str = '') -> Path:
    return RESULTS / setting_name(method, dataset, seed, tag)


def backbone_tag(backbone: str, tag: str) -> str:
    if tag:
        return tag
    return '' if backbone == 'dtmoe' else f'bb{backbone}'


def command(method: str, dataset: str, seed: int, backbone: str = 'dtmoe',
            tag: str = '') -> list[str]:
    root, data_path, freq, nc, _ = DATASETS[dataset]
    extra = []
    stream_batch = STREAM_BATCH.get(dataset)
    if stream_batch is not None:
        extra = ['--fedtdo_stream_batch', str(stream_batch)]
    resolved = backbone_tag(backbone, tag)
    des = f'{resolved}_exp' if resolved else 'exp'
    return [
        python_for(backbone), '-u', 'run_baselines.py',
        '--baseline', method,
        '--backbone', backbone,
        '--model_id', f'xc_s{seed}',
        '--des', des,
        '--data', dataset,
        '--root_path', root, '--data_path', data_path,
        '--freq', freq, '--seed', str(seed),
        '--num_clients', str(nc),
        *BASE_SHARED,
        *METHOD_ARGS.get(method, []),
        *extra,
    ]


def launch(job):
    method, dataset, seed, backbone, tag = job
    name = setting_name(method, dataset, seed, backbone_tag(backbone, tag))
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f'{name}.log'
    env = dict(os.environ)
    env['OMP_NUM_THREADS'] = env.get('XC_THREADS', '3')
    env['MKL_NUM_THREADS'] = env.get('XC_THREADS', '3')
    env['PYTHONUNBUFFERED'] = '1'
    started = time.time()
    with open(log_path, 'w', encoding='utf-8') as handle:
        code = subprocess.call(command(method, dataset, seed, backbone, tag), cwd=str(ROOT),
                               stdout=handle, stderr=subprocess.STDOUT, env=env)
    elapsed = time.time() - started
    ok = code == 0 and (RESULTS / name / 'summary.json').exists()
    return name, ok, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=MAIN_DATASETS,
                        choices=list(DATASETS))
    parser.add_argument('--methods', nargs='+', default=METHODS, choices=ALL_METHODS)
    parser.add_argument('--fedda_clusters', type=int, default=0,
                        help='FedDA: number of client clusters (0 = sqrt(n))')
    parser.add_argument('--fedda_mixup', type=float, default=0.2,
                        help='FedDA: mixup Beta concentration (0 disables)')
    parser.add_argument('--fedgtp_beta', type=float, default=0.5,
                        help='FedGTP: graph weight (0 = FedAvg, 1 = neighbours only)')
    parser.add_argument('--seeds', nargs='+', type=int, default=[2021])
    parser.add_argument('--parallel', type=int, default=2)
    parser.add_argument('--backbone', type=str, default='dtmoe',
                        choices=['dtmoe', 'patchtst', 'itransformer', 'transformer',
                                 'autoformer', 'fedformer', 'timesnet', 'timemixer', 'gru',
                                 'tfps', 'stmtm', 'raft', 'ffts', 'timeffm'],
                        help='per-client forecaster (dtmoe = paper backbone)')
    parser.add_argument('--tag', type=str, default='',
                        help='explicit output tag; defaults to bb<backbone> for non-dtmoe')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    tag = backbone_tag(args.backbone, args.tag)
    jobs, skipped = [], 0
    for dataset in args.datasets:
        for method in args.methods:
            for seed in args.seeds:
                if not args.force and (output_dir(method, dataset, seed, tag) / 'summary.json').exists():
                    skipped += 1
                    continue
                jobs.append((method, dataset, seed, args.backbone, args.tag))

    print(f'jobs={len(jobs)} skipped={skipped} parallel={args.parallel}')
    if args.dry_run:
        for job in jobs:
            print(' ', ' '.join(command(*job)))
        return

    RESULTS.mkdir(parents=True, exist_ok=True)
    started = time.time()
    failures = 0
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        futures = {pool.submit(launch, job): job for job in jobs}
        for index, future in enumerate(as_completed(futures), start=1):
            name, ok, elapsed = future.result()
            failures += 0 if ok else 1
            print(f'[{index}/{len(jobs)}] {name[:56]:56s} {elapsed / 60:5.1f} min '
                  f'{"ok" if ok else "FAIL"}', flush=True)
    print(f'\ndone in {(time.time() - started) / 60:.1f} min, {failures} failed')


if __name__ == '__main__':
    main()
