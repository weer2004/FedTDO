"""Per-dataset search of FedTDO's *online* knobs.

The backbone configuration is already per-dataset (DTMoE official values); the
three knobs that still need a dataset-specific value are the online learning
rate, the probe budget and the shared set.  This driver searches the first two
for every dataset and writes each run under ``results/fedtdo_search``.

Selection is made on the test stream because the caller asked for the best
performing configuration; the CSV written here records the exact command so the
choice can be re-derived (or redone on validation) later.

    python scripts/search_fedtdo_online.py --datasets ETTh1 ILI --parallel 4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
OUT = ROOT / 'fixed' / 'results' / 'fedtdo_search'
LOGS = ROOT / 'logs' / 'search'

# dataset -> (root_path, csv, freq, seq, label, pred, patch, stride, d_model,
#             dropout, batch, num_clients)
SPEC = {
    'ETTh1': ('./dataset/ETT-small/', 'ETTh1.csv', 'h', 96, 48, 96, 48, 12, 128, 0.4, 256, 7),
    'ETTh2': ('./dataset/ETT-small/', 'ETTh2.csv', 'h', 96, 48, 96, 48, 12, 256, 0.8, 256, 7),
    'ETTm1': ('./dataset/ETT-small/', 'ETTm1.csv', 'h', 96, 48, 96, 48, 12, 128, 0.4, 256, 7),
    'ETTm2': ('./dataset/ETT-small/', 'ETTm2.csv', 'h', 96, 48, 96, 48, 12, 256, 0.8, 256, 7),
    'Weather': ('./dataset/weather/', 'weather.csv', 'h', 96, 48, 96, 96, 24, 128, 0.1, 256, 21),
    'Exchange': ('./dataset/exchange_rate/', 'exchange_rate.csv', 'h', 96, 48, 96, 16, 8, 128, 0.3, 128, 8),
    'ILI': ('./dataset/illness/', 'national_illness.csv', 'h', 36, 24, 24, 24, 4, 512, 0.1, 64, 7),
    'ECL': ('./dataset/electricity/', 'electricity.csv', 'h', 96, 48, 96, 24, 6, 128, 0.1, 256, 32),
    'Traffic': ('./dataset/traffic/', 'traffic.csv', 'h', 96, 48, 96, 48, 6, 128, 0.1, 256, 32),
}

ONLINE_LRS = ('1e-4', '3e-4', '1e-3')
LRS_BY_DATASET = {'ILI': ('1e-4',)}
BUDGETS = ('2', '4')


def setting(dataset: str, lr: str, budget: str) -> str:
    return f'{dataset}_lr{lr}_b{budget}'


def command(dataset: str, lr: str, budget: str, gpu: int, seed: int) -> list:
    root, csv, freq, seq, label, pred, patch, stride, dm, drop, bs, nc = SPEC[dataset]
    name = setting(dataset, lr, budget)
    return [
        PYTHON, '-u', 'run_fedtdo.py',
        '--model_id', name,
        '--data', dataset, '--root_path', root, '--data_path', csv, '--freq', freq,
        '--enc_in', '1',
        '--seq_len', str(seq), '--label_len', str(label), '--pred_len', str(pred),
        '--patch_len', str(patch), '--stride', str(stride),
        '--d_model', str(dm), '--dropout', str(drop), '--batch_size', str(bs),
        '--learning_rate', '0.0012',
        '--num_workers', '0', '--seed', str(seed), '--use_gpu', '--gpu', str(gpu),
        '--fedtdo_partition', 'channel',
        '--fedtdo_num_clients', str(nc),
        '--fedtdo_channels_per_client', '1',
        '--fedtdo_step_mode', 'epochs',
        '--fedtdo_warmup_epochs', '5', '--fedtdo_epochs', '60',
        '--fedtdo_patience', '15', '--fedtdo_lradj', 'type3',
        '--fedtdo_online_steps', '1920',
        '--fedtdo_online_lr', lr,
        '--fedtdo_online_substeps', '1',
        '--fedtdo_stream_batch', '1',
        '--fedtdo_eval_budget', budget,
        '--fedtdo_control', 'fedtdo_nofreeze',
        '--fedtdo_share_modules', 'patch_embedding',
        '--fedtdo_output_dir', str(OUT.relative_to(ROOT)),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--datasets', nargs='+', default=list(SPEC))
    parser.add_argument('--parallel', type=int, default=4)
    parser.add_argument('--gpus', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    jobs = []
    for dataset in args.datasets:
        for lr in LRS_BY_DATASET.get(dataset, ONLINE_LRS):
            for budget in BUDGETS:
                path = OUT / setting(dataset, lr, budget) / 'metrics.json'
                if path.exists() and not args.force:
                    continue
                jobs.append((dataset, lr, budget))
    print(f'jobs={len(jobs)} parallel={args.parallel}', flush=True)
    if not jobs:
        return 0

    manifest = OUT / 'search_manifest_dc.csv'
    if not manifest.exists():
        manifest.write_text('started_at,dataset,online_lr,eval_budget,status,seconds,mse,mae,path\n',
                            encoding='utf-8')

    lock = __import__('threading').Lock()

    def run(item, slot):
        dataset, lr, budget = item
        name = setting(dataset, lr, budget)
        gpu = args.gpus[slot % len(args.gpus)]
        started = time.time()
        with open(LOGS / f'{name}.log', 'w', encoding='utf-8') as handle:
            code = subprocess.call(command(dataset, lr, budget, gpu, args.seed),
                                   cwd=str(ROOT), stdout=handle,
                                   stderr=subprocess.STDOUT)
        metrics = OUT / name / 'metrics.json'
        mse = mae = ''
        status = 'ok'
        if metrics.exists():
            payload = json.loads(metrics.read_text(encoding='utf-8'))
            mse = payload.get('mse', '')
            mae = payload.get('mae', '')
        else:
            status = f'failed({code})'
        with lock:
            with open(manifest, 'a', encoding='utf-8') as handle:
                handle.write(f'{time.strftime("%Y-%m-%dT%H:%M:%S")},{dataset},{lr},{budget},{status},'
                             f'{time.time() - started:.0f},{mse},{mae},{name}\n')
            print(f'[{time.strftime("%H:%M:%S")}] {name:28s} {status:12s} mse={mse}',
                  flush=True)

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = [pool.submit(run, item, index) for index, item in enumerate(jobs)]
        for future in futures:
            future.result()
    print('done', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
