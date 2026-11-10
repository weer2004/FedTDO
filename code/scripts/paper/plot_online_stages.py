"""Early / middle / late forecast snapshots of one method's online stream.

Each panel is its own real window: the panels are *not* stitched into one series,
because they are 10/50/90 percent of the stream apart and concatenating them
would draw a timeline that never existed.

The annotations (cumulative updates, gradient steps, active expert, transfer
before this window) come from the recorded event tables, not from the config.

    python scripts/paper/plot_online_stages.py --dataset ETTh1 --method fedtdo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (FIGURES, find_runs, identity, load_run, pretty,  # noqa: E402
                    save_figure, save_figure_data, set_paper_style,
                    stage_window_indices)


def select_run(dataset: str, method: str, seed: int = None) -> dict:
    candidates = []
    for path in find_runs():
        run = load_run(path)
        ident = identity(run)
        if ident['dataset'] != dataset or ident['method'] != method:
            continue
        if not run['predictions']:
            continue
        candidates.append(run)
    if not candidates:
        return {}
    if seed is not None:
        for run in candidates:
            if run['config'].get('seed') == seed:
                return run
    return sorted(candidates, key=lambda r: str(r['config'].get('run_id')))[0]


def annotation(run: dict, client_id: str, window_idx: int) -> Dict:
    info = {'update_count': '', 'gradient_steps': '', 'expert': '',
            'transferred_before': 'no'}
    window = run['per_window']
    if window is not None and not window.empty:
        row = window[(window['client_id'] == client_id)
                     & (window['window_idx'] == window_idx)]
        if len(row):
            info['update_count'] = int(row['update_count'].iloc[0])
            info['gradient_steps'] = int(row['gradient_steps'].iloc[0])
    transfers = run['transfers']
    if transfers is not None and not transfers.empty:
        recent = transfers[(transfers['client_id'] == client_id)
                           & (transfers['window_idx'] <= window_idx)]
        if len(recent):
            last = recent.sort_values('window_idx').iloc[-1]
            info['expert'] = str(last.get('new_expert_id', '') or '')
            if int(last['window_idx']) == int(window_idx):
                info['transferred_before'] = 'yes'
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--dataset', default='ETTh1')
    parser.add_argument('--method', default='fedtdo')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--client', default=None)
    parser.add_argument('--channel', type=int, default=0)
    args = parser.parse_args()

    plt = set_paper_style()
    run = select_run(args.dataset, args.method, args.seed)
    if not run:
        print(f'no recorded run for {args.dataset}/{args.method}')
        return 0
    ident = identity(run)
    clients = sorted(run['predictions'])
    client_id = args.client or (clients[0] if clients else None)
    if client_id is None or client_id not in run['predictions']:
        print(f'{args.dataset}: client {client_id} has no predictions')
        return 0
    arrays = run['predictions'][client_id]
    idx = np.asarray(arrays['window_idx'])
    stages = stage_window_indices(int(idx.max()))

    fig, axes = plt.subplots(len(stages), 1, figsize=(3.4, 1.5 * len(stages)),
                             squeeze=False)
    axes = axes[:, 0]
    data_rows: List[dict] = []
    for axis, (stage, target) in zip(axes, stages.items()):
        position = int(np.argmin(np.abs(idx - target)))
        window_idx = int(idx[position])
        y_true = np.asarray(arrays['y_true'][position], dtype=np.float64)
        y_pred = np.asarray(arrays['y_pred'][position], dtype=np.float64)
        channel = min(args.channel, y_true.shape[-1] - 1)
        truth = y_true[:, channel]
        pred = y_pred[:, channel]
        if 'x_history' in arrays:
            hist = np.asarray(arrays['x_history'][position], dtype=np.float64)
            hist = hist[:, min(channel, hist.shape[-1] - 1)]
            axis.plot(np.arange(-len(hist), 0), hist, color='#444444', lw=1.0,
                      label='Input history')
        axis.plot(np.arange(1, len(truth) + 1), truth, color='#111111', lw=1.2,
                  label='Ground truth')
        axis.plot(np.arange(1, len(pred) + 1), pred, color='#1f77b4', lw=1.2,
                  ls='--', label=f'{pretty(ident["method"])}')
        axis.axvline(0, color='#888888', lw=0.8, ls=':')
        info = annotation(run, client_id, window_idx)
        mse = float(np.mean((truth - pred) ** 2))
        axis.set_title(
            f'{stage}: window {window_idx}  MSE={mse:.4f}  '
            f'updates={info["update_count"]}  steps={info["gradient_steps"]}  '
            f'expert={info["expert"] or "-"}', loc='left')
        axis.set_ylabel('value')
        axis.legend(loc='upper left', frameon=False)
        series = {'ground_truth': (truth, 1), 'prediction': (pred, 1)}
        if 'x_history' in arrays:
            hist = np.asarray(arrays['x_history'][position], dtype=np.float64)
            hist = hist[:, min(args.channel, hist.shape[-1] - 1)]
            series['input_history'] = (hist, -len(hist))
        for name, (values, start) in series.items():
            for offset, value in enumerate(values):
                data_rows.append({
                    'dataset': args.dataset, 'method': ident['method'],
                    'seed': ident['seed'], 'client_id': client_id,
                    'stage': stage, 'window_idx': window_idx,
                    'channel': args.channel, 'series': name,
                    'step': int(start + offset), 'value': float(value),
                    'cumulative_updates': info['update_count'],
                    'cumulative_gradient_steps': info['gradient_steps'],
                    'active_expert': info['expert'],
                    'transferred_before_window': info['transferred_before'],
                    'window_mse': mse})
    axes[-1].set_xlabel('time step (relative to forecast origin), channel '
                        f'{args.channel}')
    fig.suptitle(f'{args.dataset}  {pretty(ident["method"])}  client {client_id}  '
                 f'seed {ident["seed"]}', y=1.0, fontsize=8)
    fig.tight_layout(h_pad=0.4)

    stem = (FIGURES / 'forecasting_stages'
            / f'forecasting_stages_{args.dataset}_{ident["method"]}')
    produced = save_figure(fig, stem)
    if data_rows:
        produced.append(save_figure_data(pd.DataFrame(data_rows), stem))
    for path in produced:
        print('wrote', path.relative_to(FIGURES.parents[1]))
    plt.close(fig)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
