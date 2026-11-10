"""Multi-method forecast comparison, in the style of Dynamic TMoE Figs. 5-7.

Every panel shows the *same* window: same dataset, seed, client, window index,
input history, horizon and channel.  The window is chosen before looking at any
curve, by a rule that only depends on what the runs recorded:

1. keep the (dataset, seed) cells where at least two methods exist;
2. keep the clients present in all of them;
3. inside a client, take the intersection of the evaluated window indices;
4. pick the position `--stage` (early/middle/late = 10/50/90 percentile of that
   intersection), rounded to the nearest recorded index.

Methods without recorded predictions are dropped, never drawn as a flat line.

    python scripts/paper/plot_forecasting_comparison.py --dataset ETTh1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (FIGURES, find_runs, identity, load_run, method_colours,  # noqa: E402
                    order_key, pretty, save_figure, save_figure_data,
                    set_paper_style, stage_window_indices)

CHANNEL_HINT = 'channel 0'


def collect(dataset: str) -> List[dict]:
    runs = []
    for path in find_runs():
        run = load_run(path)
        ident = identity(run)
        if ident['dataset'] != dataset or not run['predictions']:
            continue
        runs.append(run)
    return runs


def pick_window(runs: List[dict], stage: str) -> Dict:
    seeds = sorted({r['config'].get('seed') for r in runs})
    for seed in seeds:
        cell = [r for r in runs if r['config'].get('seed') == seed]
        if len(cell) < 2:
            continue
        common_clients = None
        for run in cell:
            keys = set(run['predictions'])
            common_clients = keys if common_clients is None else common_clients & keys
        if not common_clients:
            continue
        for client_id in sorted(common_clients):
            index_sets = [set(np.asarray(run['predictions'][client_id]['window_idx']))
                          for run in cell]
            shared = set.intersection(*index_sets)
            if not shared:
                continue
            positions = stage_window_indices(max(shared))
            target = positions.get(stage)
            if target is None:
                continue
            window_idx = min(shared, key=lambda w: abs(w - target))
            # Require the full set of methods to actually hold this window.
            if not all(window_idx in np.asarray(run['predictions'][client_id]['window_idx'])
                       for run in cell):
                continue
            return {'seed': seed, 'client_id': client_id, 'window_idx': window_idx,
                    'runs': cell}
    return {}


def array_for(run: dict, client_id: str, window_idx: int, key: str):
    arrays = run['predictions'][client_id]
    idx = np.asarray(arrays['window_idx'])
    position = int(np.argmin(np.abs(idx - window_idx)))
    if int(idx[position]) != int(window_idx):
        return None
    return np.asarray(arrays[key][position], dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--dataset', default='ETTh1')
    parser.add_argument('--stage', default='middle',
                        choices=['early', 'middle', 'late'])
    parser.add_argument('--window', type=int, default=None,
                        help='override the pre-declared stage position')
    parser.add_argument('--channel', type=int, default=0)
    args = parser.parse_args()

    plt = set_paper_style()
    runs = collect(args.dataset)
    if len(runs) < 2:
        print(f'{args.dataset}: need at least two methods with recorded '
              f'predictions, found {len(runs)}')
        return 0

    choice = pick_window(runs, args.stage)
    if not choice:
        print(f'{args.dataset}: no window is shared by every method')
        return 0
    if args.window is not None:
        choice['window_idx'] = int(args.window)

    methods = sorted({r['config'].get('method') for r in choice['runs']},
                     key=order_key)
    runs_by_method = {r['config'].get('method'): r for r in choice['runs']}
    colours = method_colours(methods)

    fig, axes = plt.subplots(len(methods), 1, figsize=(3.4, 1.35 * len(methods)),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]
    origin = None
    data_rows: List[dict] = []
    for axis, method in zip(axes, methods):
        run = runs_by_method[method]
        client_id = choice['client_id']
        window_idx = choice['window_idx']
        y_true = array_for(run, client_id, window_idx, 'y_true')
        y_pred = array_for(run, client_id, window_idx, 'y_pred')
        x_hist = array_for(run, client_id, window_idx, 'x_history')
        if y_true is None or y_pred is None:
            axis.set_visible(False)
            continue
        channel = min(args.channel, y_true.shape[-1] - 1)
        truth = y_true[:, channel]
        pred = y_pred[:, channel]
        colour = colours.get(method, '#1f77b4')
        if x_hist is not None:
            hist = x_hist[:, min(channel, x_hist.shape[-1] - 1)]
            past_x = np.arange(-len(hist), 0)
            axis.plot(past_x, hist, color='#444444', lw=1.0, label='Input history')
            now = 0
        else:
            now = 0
        if origin is None:
            origin = now
        future_x = now + np.arange(1, len(truth) + 1)
        axis.plot(future_x, truth, color='#111111', lw=1.2, label='Ground truth')
        axis.plot(future_x, pred, color=colour, lw=1.2, ls='--',
                  label=pretty(method))
        axis.axvline(now, color='#888888', lw=0.8, ls=':')
        mse = float(np.mean((truth - pred) ** 2))
        axis.set_title(f'{pretty(method)}   MSE={mse:.4f}', loc='left')
        axis.set_ylabel('value')
        axis.legend(loc='upper left', frameon=False, ncol=2)
        for series, values, start in (('input_history', None, None),
                                      ('ground_truth', truth, 1),
                                      ('prediction', pred, 1)):
            if values is None and x_hist is None:
                continue
            if series == 'input_history':
                values = x_hist[:, min(channel, x_hist.shape[-1] - 1)]
                start = -len(values)
            for offset, value in enumerate(values):
                data_rows.append({
                    'dataset': args.dataset, 'seed': choice['seed'],
                    'client_id': choice['client_id'],
                    'window_idx': choice['window_idx'], 'method': method,
                    'channel': channel, 'series': series,
                    'step': int(start + offset), 'value': float(value)})
        data_rows.append({
            'dataset': args.dataset, 'seed': choice['seed'],
            'client_id': choice['client_id'], 'window_idx': choice['window_idx'],
            'method': method, 'channel': channel, 'series': 'window_mse',
            'step': 0, 'value': mse})
    axes[-1].set_xlabel(f'time step (relative to forecast origin) \u2014 {CHANNEL_HINT}')
    fig.suptitle(f'{args.dataset}  client {choice["client_id"]}  window '
                 f'{choice["window_idx"]}  ({args.stage} stage)  pred_len='
                 f'{truth.shape[0]}', y=1.0, fontsize=8)
    fig.tight_layout(h_pad=0.3)

    stem = FIGURES / 'forecasting_comparison' / f'forecasting_comparison_{args.dataset}'
    paths = save_figure(fig, stem)
    plt.close(fig)
    if data_rows:
        paths.append(save_figure_data(pd.DataFrame(data_rows), stem))
    for path in paths:
        print('wrote', path.relative_to(FIGURES.parents[1]))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
