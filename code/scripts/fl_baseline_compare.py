"""Run every FL baseline under the DTMoe-aligned epoch protocol and compare to FedTDO.

The point is a single internally consistent table:

    column                     budget
    local / fedavg / ...       60 epochs, type3 lr decay, val early stopping (patience 15)
    FedTDO                     same 60 epochs (5 federated warm-up + 55 downstream)

so every row differs only in *how* the clients share parameters.  Anything run
with the legacy step budget belongs in a different table; ``summary.json`` now
carries ``train.step_mode`` so a row cannot be misread.

Scope: channel partition (one sensor per client, ``enc_in 1``), the DTMoe config
of the matching ``scripts/*.sh`` per dataset/horizon, the official per-dataset
batch (ETT/Weather 256, Exchange 128, ILI 64, ECL/Traffic 32), 60 online rounds
with stream batch 32 (1 for ILI).  ``ditto`` and ``fedl2p`` train two models /
two phases, and
each has its own epoch path, so they share the same per-client budget: ditto
spends it on two models per client (2x compute), fedl2p spends its first half on
a FedAvg warm-up and its second half on the meta-learned fine-tune.

Usage::

    python scripts/fl_baseline_compare.py --dry-run
    python scripts/fl_baseline_compare.py --datasets ETTh1 --pred-lens 96
    python scripts/fl_baseline_compare.py --summary-only
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

from lr_sweep import DATASETS, STREAM_BATCH, model_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get('FEDTDO_PYTHON', sys.executable)
RESULTS = ROOT / 'fixed' / 'results' / 'baselines'

ALL_METHODS = [
    'local', 'fedavg', 'fedavg_sgd', 'fedprox', 'scaffold', 'fedadam', 'fednova',
    'fedper', 'fedrep', 'pfedme', 'ditto', 'fedl2p', 'fedala', 'fedda', 'fedgtp',
    'fedkme', 'fedkwaz', 'fedpkda', 'fedpoe', 'pfedcs', 'pfedfda',
]
# Methods without an epoch-mode training path.  ``ditto`` and ``fedl2p`` left
# this set when they gained their own two-model / two-phase epoch paths.
UNSUPPORTED: set = set()

# FedTDO reference for the same cell, filled from results/fedtdo_part at runtime.
FEDTDO_FALLBACK = {'mse': 0.4454, 'mae': 0.4469, 'local_mse': 0.4329}


def n_clients(args, dataset):
    """Client count for one dataset.

    ``default`` keeps ``lr_sweep.DATASETS[...]['nc']`` so every table on disk
    stays reproducible.  ``channels`` uses one client per column -- the protocol
    this sweep is asked to cover -- capped by ``--nc-cap`` because ECL (321
    columns) and Traffic (862) would otherwise need 321 / 862 clients, which is
    not a training run but a distributed cluster.
    """
    mode = str(getattr(args, 'nc', 'default'))
    if mode != 'channels':
        return int(DATASETS[dataset]['nc'])
    spec = DATASETS[dataset]
    path = Path(spec['root']) / spec['csv']
    if not path.is_absolute():
        path = ROOT / path
    try:
        with open(path, encoding='utf-8') as handle:
            count = len(handle.readline().strip().split(',')) - 1
    except OSError:
        count = int(DATASETS[dataset]['nc'])
    return max(1, min(int(count), int(args.nc_cap)))


PROTOCOL_STREAM_BATCH = 1        # one window per round -> window-level updates
PROTOCOL_ONLINE_WINDOWS = 1920   # windows every client is scored on


def protocol_suffix(online_lr: float, stream_batch: int) -> str:
    """Protocol tag for a cell name.

    The prequential protocol is part of a cell's identity.  Without the tag a
    batch-level cell and a window-level cell share a name, the driver treats the
    old one as finished, and two incomparable protocols end up in one table.
    """
    if not online_lr and stream_batch == PROTOCOL_STREAM_BATCH:
        return ''
    return f'_olr{online_lr:g}_sb{stream_batch}_os'


def setting_name(method, dataset, pred_len, d_model, rounds, epochs, seed, nc,
                 online_lr=0.0, stream_batch=PROTOCOL_STREAM_BATCH):
    suffix = protocol_suffix(online_lr, stream_batch)
    return (f'BL_{method}_fl_baseline_{dataset}_nc{nc}'
            f'_sl{DATASETS[dataset]["seq_len"]}_pl{pred_len}_dm{d_model}'
            f'_r{rounds}_ep{epochs}_s{seed}{suffix}')


def summary_path(args, method, dataset, pred_len):
    config = model_config(args, dataset, pred_len)
    return RESULTS / setting_name(method, dataset, pred_len, config['d_model'],
                                  args.rounds, args.epochs, args.seeds[0],
                                  n_clients(args, dataset),
                                  FEDTDO_ONLINE_LR_BY_DATASET.get(
                                      dataset, getattr(args, 'fl_online_lr', 0.0)),
                                  PROTOCOL_STREAM_BATCH) / 'summary.json'


def pred_lens_of(args, dataset):
    """Horizons to run for one dataset, limited to the grid it actually defines.

    ILI has 24/36/48/60 while the ETT-family sets have 96/192/336/720, so a
    single ``--pred-lens`` list cannot be applied blindly (a wrong horizon would
    raise ``KeyError`` out of ``OFFICIAL``).
    """
    allowed = DATASETS[dataset]['pred_lens']
    if not args.pred_lens:
        return list(allowed)
    chosen = [pl for pl in args.pred_lens if pl in allowed]
    if not chosen:
        print(f'skip {dataset}: none of {args.pred_lens} is defined for it '
              f'(allowed {list(allowed)})')
    return chosen


def batch_size(args, dataset):
    """Official per-dataset batch, unless ``--batch-size`` pins one value."""
    if getattr(args, 'batch_size', None):
        return int(args.batch_size)
    return int(DATASETS[dataset]['batch'])


def command(args, method, dataset, pred_len):
    spec = DATASETS[dataset]
    config = model_config(args, dataset, pred_len)
    return [
        PYTHON, '-u', 'run_baselines.py',
        '--baseline', method,
        # GPU is mandatory for every arm; the index is inside the per-job
        # CUDA_VISIBLE_DEVICES this launcher sets, so it is always 0 here.
        # ``run_baselines.py`` declares this flag as ``type=bool`` (not
        # store_true), so it needs an explicit value.
        '--use_gpu', 'True', '--gpu', '0',
        '--model_id', 'fl_baseline',
        '--des', f'ep{args.epochs}_s{args.seeds[0]}',
        '--data', dataset,
        '--root_path', spec['root'], '--data_path', spec['csv'], '--freq', spec['freq'],
        '--seed', str(args.seeds[0]),
        '--num_clients', str(n_clients(args, dataset)),
        '--seq_len', str(spec['seq_len']), '--label_len', str(spec['label_len']),
        '--pred_len', str(pred_len),
        '--d_model', str(config['d_model']), '--dropout', str(config['dropout']),
        '--patch_len', str(config['patch_len']), '--stride', str(config['stride']),
        '--batch_size', str(batch_size(args, dataset)),
        '--learning_rate', f'{args.learning_rate:g}',
        '--fl_step_mode', 'epochs',
        '--fl_rounds', str(args.rounds),
        # ``--epochs`` is the *total* per-client budget; rounds divide it so every
        # method trains the same number of passes regardless of how many
        # aggregations it performs.
        '--fl_local_epochs', str(max(1, args.epochs // max(1, args.rounds))),
        '--fl_patience', str(args.patience),
        '--fl_lradj', 'type3',
        '--fl_online_steps', str(PROTOCOL_ONLINE_WINDOWS),
        # Prequential adaptation budget, matched to FedTDO's online learning
        # rate so the two tables share one protocol.
        '--fl_online_lr',
        f'{FEDTDO_ONLINE_LR_BY_DATASET.get(dataset, getattr(args, "fl_online_lr", 0.0)):g}',
        # One window per round: the online stage then predicts, reveals and
        # updates at the same cadence as Exp_FedTDO in window-level mode.
        '--fedtdo_stream_batch', str(PROTOCOL_STREAM_BATCH),
        # Paper-grade event tables: per-window metrics, predictions, online
        # updates, cost events.  Without this the sweep produces only the
        # summary and none of the figures can be rebuilt.
        '--paper_record', '1',
        '--paper_results_dir', PAPER_ROOT,
        '--results_dir', str(RESULTS.relative_to(ROOT)),
        '--num_workers', '0',
        '--num_threads', '4',
    ]


FEDTDO_METHOD = 'fedtdo'
FEDTDO_OUT = ROOT / 'fixed' / 'results' / 'fedtdo_full'
FEDTDO_ONLINE_LR = 1e-4
FEDTDO_ONLINE_LR_BY_DATASET = {'ILI': 1e-4}
# Measured on ETTh1 under the window-level protocol: sharing only the patch
# embedding with four probes per window is 0.36587, versus 0.37561 for the
# wide shared set and 0.37677 at two probes.  The arm is tagged so the
# faster-but-worse cell cannot be silently reused.
FEDTDO_SHARE = 'patch_embedding'
FEDTDO_EVAL_BUDGET = 4
FEDTDO_TAG = '_pe_b4'
PAPER_ROOT = './fixed/results/paper_fedtdo'


def fedtdo_dir(args, dataset, pred_len):
    spec = DATASETS[dataset]
    config = model_config(args, dataset, pred_len)
    suffix = (protocol_suffix(FEDTDO_ONLINE_LR_BY_DATASET.get(dataset, FEDTDO_ONLINE_LR),
                              PROTOCOL_STREAM_BATCH) + FEDTDO_TAG + '_os')
    return FEDTDO_OUT / (f'fedtdo_ch_{dataset}_nc{n_clients(args, dataset)}'
                         f'_sl{spec["seq_len"]}_pl{pred_len}_dm{config["d_model"]}'
                         f'_ep{args.epochs}_s{args.seeds[0]}{suffix}')


def fedtdo_command(args, dataset, pred_len):
    """FedTDO arm of a cell: same epochs, same online budget, same lr as the rows
    it is compared against, so the only difference is the sharing mechanism."""
    spec = DATASETS[dataset]
    config = model_config(args, dataset, pred_len)
    return [
        PYTHON, '-u', 'run_fedtdo.py',
        '--model_id', fedtdo_dir(args, dataset, pred_len).name,
        '--data', dataset,
        '--root_path', spec['root'], '--data_path', spec['csv'], '--freq', spec['freq'],
        '--seed', str(args.seeds[0]),
        '--enc_in', '1',
        '--fedtdo_partition', 'channel',
        '--fedtdo_num_clients', str(n_clients(args, dataset)),
        '--fedtdo_channels_per_client', '1',
        '--seq_len', str(spec['seq_len']), '--label_len', str(spec['label_len']),
        '--pred_len', str(pred_len),
        '--d_model', str(config['d_model']), '--dropout', str(config['dropout']),
        '--patch_len', str(config['patch_len']), '--stride', str(config['stride']),
        '--batch_size', str(batch_size(args, dataset)),
        '--learning_rate', f'{args.learning_rate:g}',
        '--fedtdo_step_mode', 'epochs',
        '--fedtdo_epochs', str(args.epochs),
        '--fedtdo_warmup_epochs', '5', '--fedtdo_encoder_rounds', '1',
        '--fedtdo_patience', str(args.patience), '--fedtdo_lradj', 'type3',
        # Window-level prequential protocol: one window per round, one gradient
        # step per revealed window, identical to every baseline row.
        '--fedtdo_online_steps', str(PROTOCOL_ONLINE_WINDOWS),
        '--fedtdo_online_target', 'first',
        '--fedtdo_online_lr',
        f"{FEDTDO_ONLINE_LR_BY_DATASET.get(dataset, FEDTDO_ONLINE_LR):g}",
        '--fedtdo_online_substeps', '1',
        '--fedtdo_stream_batch', str(PROTOCOL_STREAM_BATCH),
        # Without this flag FedTDO ran on CPU while the baselines ran on CUDA.
        '--use_gpu', '--gpu', '0',
        '--fedtdo_eval_budget', str(FEDTDO_EVAL_BUDGET),
        '--fedtdo_control', 'fedtdo_nofreeze',
        '--fedtdo_share_modules', FEDTDO_SHARE,
        '--fedtdo_output_dir', str(FEDTDO_OUT.relative_to(ROOT)),
        '--paper_record', '1',
        '--paper_results_dir', PAPER_ROOT,
        '--num_workers', '0',
    ]


def cell_command(args, method, dataset, pred_len):
    if method == FEDTDO_METHOD:
        return fedtdo_command(args, dataset, pred_len)
    return command(args, method, dataset, pred_len)


def cell_artifact(args, method, dataset, pred_len):
    """Expected metrics/summary file for one cell (may not exist yet)."""
    if method == FEDTDO_METHOD:
        return fedtdo_dir(args, dataset, pred_len) / 'metrics.json'
    return summary_path(args, method, dataset, pred_len)


def cell_result(args, method, dataset, pred_len):
    """Same path, but ``None`` unless the cell actually finished."""
    path = cell_artifact(args, method, dataset, pred_len)
    return path if path.exists() else None


def launch(args, job):
    method, dataset, pred_len, gpu = job
    if method == FEDTDO_METHOD:
        name = fedtdo_dir(args, dataset, pred_len).name
    else:
        name = summary_path(args, method, dataset, pred_len).parent.name
    logs = RESULTS / '_epoch_logs'
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env['OMP_NUM_THREADS'] = '4'
    env['MKL_NUM_THREADS'] = '4'
    env['PYTHONUNBUFFERED'] = '1'
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    started = time.time()
    with open(logs / f'{name}.log', 'w', encoding='utf-8') as handle:
        code = subprocess.call(cell_command(args, method, dataset, pred_len),
                               cwd=str(ROOT), stdout=handle,
                               stderr=subprocess.STDOUT, env=env)
    ok = code == 0 and cell_result(args, method, dataset, pred_len) is not None
    return name, ok, time.time() - started


def fedtdo_reference(args, dataset, pred_len):
    """FedTDO channel-partition numbers for the same cell, if on disk."""
    out = dict(FEDTDO_FALLBACK)
    joint = ROOT / 'results/fedtdo_part' / 'ep_ch_nc7_joint' / 'metrics.json'
    local = ROOT / 'results/fedtdo_part' / 'ep_ch_nc7_iso' / 'metrics.json'
    try:
        if joint.exists():
            d = json.load(open(joint, encoding='utf-8'))
            out['mse'], out['mae'] = d['mse'], d['mae']
        if local.exists():
            out['local_mse'] = json.load(open(local, encoding='utf-8'))['mse']
    except (OSError, json.JSONDecodeError):
        pass
    return out


def _normalise(d):
    """Baseline ``summary.json`` and FedTDO ``metrics.json`` use different keys."""
    mse = d.get('mean_mse', d.get('mean_client_mse', d.get('mse')))
    mae = d.get('mean_mae', d.get('mean_client_mae', d.get('mae')))
    median = d.get('median_mse')
    if median is None and d.get('client_metrics'):
        values = sorted(v['mse'] for v in d['client_metrics'].values())
        mid = len(values) // 2
        median = (values[mid] if len(values) % 2
                  else 0.5 * (values[mid - 1] + values[mid]))
    return float(mse), float(mae), (float(median) if median is not None else float('nan'))


def summary(args):
    rows = []
    for method in args.methods:
        for dataset in args.datasets:
            for pred_len in pred_lens_of(args, dataset):
                path = cell_result(args, method, dataset, pred_len)
                if path is None:
                    continue
                with open(path, encoding='utf-8') as handle:
                    rows.append((dataset, pred_len, method, _normalise(json.load(handle))))
    if not rows:
        print('no finished cells')
        return
    print(f'\n{"dataset":<8}{"pl":>5}{"nc":>4}{"method":<14}{"MSE":>9}{"MAE":>9}{"median":>9}'
          f'{"vs local":>10}{"vs FedTDO":>11}{"vs FedAvg":>11}')
    print('-' * 92)
    per_cell = {}
    for dataset, pred_len, method, (mse, _, _) in rows:
        per_cell.setdefault((dataset, pred_len), {})[method] = mse
    for dataset, pred_len, method, (mse, mae, median) in sorted(
            rows, key=lambda r: (r[0], r[1], r[3][0])):
        cell = per_cell[(dataset, pred_len)]
        base_local = cell.get('local')
        base_fedavg = cell.get('fedavg')
        base_fedtdo = cell.get(FEDTDO_METHOD)
        line = (f'{dataset:<8}{pred_len:>5}{n_clients(args, dataset):>4}{method:<14}'
                f'{mse:>9.4f}{mae:>9.4f}{median:>9.4f}')
        line += f'{100*(mse-base_local)/base_local:>9.1f}%' if base_local else f'{"-":>10}'
        line += f'{100*(mse-base_fedtdo)/base_fedtdo:>10.1f}%' if base_fedtdo else f'{"-":>11}'
        line += f'{100*(mse-base_fedavg)/base_fedavg:>10.1f}%' if base_fedavg else f'{"-":>11}'
        print(line)
    print('-' * 92)
    print('vs local: negative = better than no federation at all')
    print('vs FedTDO: negative = better than FedTDO (the row to explain away)')
    missing = [(m, ds) for m in args.methods for ds in args.datasets
               if (m, ds) not in {(r[2], r[0]) for r in rows}]
    if missing:
        print(f'not finished ({len(missing)}): ' +
              ', '.join(f'{m}/{ds}' for m, ds in missing[:30])
              + (' ...' if len(missing) > 30 else ''))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--methods', nargs='+', default=None,
                        help='default: every registry method with an epoch path')
    parser.add_argument('--include-unsupported', action='store_true')
    parser.add_argument('--datasets', nargs='+', default=['ETTh1'], choices=list(DATASETS))
    parser.add_argument('--pred-lens', nargs='+', type=int, default=None,
                        help='default: every horizon the dataset defines in '
                             'lr_sweep.DATASETS (ETT/Weather/Exchange/ECL/Traffic '
                             '96 192 336 720, ILI 24 36 48 60)')
    parser.add_argument('--seeds', nargs='+', type=int, default=[2021])
    parser.add_argument('--rounds', type=int, default=5,
                        help='federated rounds; the 60-epoch budget is split across them')
    parser.add_argument('--epochs', type=int, default=60,
                        help='total epochs per client (NOT per round)')
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=None,
                        help='pin one batch size for every dataset; default uses the '
                             'official per-dataset value from lr_sweep.DATASETS '
                             '(ETT/Weather 256, Exchange 128, ILI 64, ECL/Traffic 32)')
    parser.add_argument('--learning-rate', type=float, default=1.2e-3)
    parser.add_argument('--fl-online-lr', type=float, default=FEDTDO_ONLINE_LR,
                        help='prequential learning rate handed to every baseline. '
                             'Default 0 = static evaluation (what every published '
                             'cell used). Set it to the FedTDO arm value for a '
                             'symmetric online comparison; the cell name then '
                             'carries an _olr suffix')
    parser.add_argument('--config', default='official', choices=['official', 'fedtdo'])
    parser.add_argument('--nc', default='default', choices=['default', 'channels'],
                        help="'default' = lr_sweep.DATASETS[...]['nc'] (7/8/32, the "
                             "published splits); 'channels' = one client per CSV column, "
                             "capped by --nc-cap")
    parser.add_argument('--nc-cap', type=int, default=32,
                        help='client-count ceiling for --nc channels; ECL has 321 '
                             'columns and Traffic 862, which are not trainable')
    parser.add_argument('--include-fedtdo', action='store_true',
                        help='add the FedTDO arm to the same table and job queue')
    parser.add_argument('--gpus', nargs='+', type=int, default=[0])
    parser.add_argument('--parallel', type=int, default=6)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--summary-only', action='store_true')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    if args.methods is None:
        args.methods = [m for m in ALL_METHODS
                        if args.include_unsupported or m not in UNSUPPORTED]
        if UNSUPPORTED and not args.include_unsupported:
            print(f'skipping {sorted(UNSUPPORTED)}: no epoch-mode training path')
    if args.include_fedtdo and FEDTDO_METHOD not in args.methods:
        # First, not last: the FedTDO rows are the ones a table is read for, and
        # appending them queues the only new method behind ~150 baseline cells.
        args.methods = [FEDTDO_METHOD] + args.methods
    if args.summary_only:
        summary(args)
        return 0

    jobs, skipped = [], 0
    for method in args.methods:
        for dataset in args.datasets:
            for pred_len in pred_lens_of(args, dataset):
                if cell_result(args, method, dataset, pred_len) is not None and not args.force:
                    skipped += 1
                    continue
                jobs.append((method, dataset, pred_len,
                             args.gpus[len(jobs) % len(args.gpus)]))

    print(f'jobs={len(jobs)} skipped={skipped} parallel={args.parallel} '
          f'budget={args.epochs} epochs')
    if args.dry_run:
        for job in jobs:
            print(f'  CUDA_VISIBLE_DEVICES={job[3]} ' + ' '.join(cell_command(args, *job[:3])))
        return 0

    failures = []
    started = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        futures = {pool.submit(launch, args, job): job for job in jobs}
        for done, future in enumerate(as_completed(futures), start=1):
            name, ok, elapsed = future.result()
            print(f'[{done}/{len(jobs)}] {name:<70} {elapsed/60:5.1f} min '
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
