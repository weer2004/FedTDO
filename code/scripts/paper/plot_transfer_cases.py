"""Forecast around real expert switches.

Only windows that actually exist are drawn.  A switch at window ``w`` gives two
separate panels: the last window served by the old expert and the first window
served by the new one.  They are different time points, so they are never
concatenated into one series.

If no switch was recorded the script says so and exits without writing a figure;
a synthetic case is never constructed.

    python scripts/paper/plot_transfer_cases.py --dataset ETTh1 --cases 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (FIGURES, find_runs, identity, load_run, pretty,  # noqa: E402
                    save_figure, save_figure_data, set_paper_style)


def select_run(dataset: str, method: str) -> dict:
    best = None
    for path in find_runs():
        run = load_run(path)
        ident = identity(run)
        if ident['dataset'] != dataset or ident['method'] != method:
            continue
        transfers = run['transfers']
        if transfers is None or transfers.empty:
            continue
        switches = transfers[transfers['event_type'] == 'switch']
        if switches.empty or not run['predictions']:
            continue
        if best is None or len(switches) > best[0]:
            best = (len(switches), run)
    return best[1] if best else {}


def window_slice(run: dict, client_id: str, window_idx: int):
    arrays = run['predictions'].get(client_id)
    if not arrays:
        return None
    idx = np.asarray(arrays['window_idx'])
    position = int(np.argmin(np.abs(idx - window_idx)))
    if int(idx[position]) != int(window_idx):
        return None
    payload = {
        'y_true': np.asarray(arrays['y_true'][position], dtype=np.float64),
        'y_pred': np.asarray(arrays['y_pred'][position], dtype=np.float64),
    }
    if 'x_history' in arrays:
        payload['x_history'] = np.asarray(arrays['x_history'][position],
                                          dtype=np.float64)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--dataset', default='ETTh1')
    parser.add_argument('--method', default='fedtdo')
    parser.add_argument('--cases', type=int, default=2)
    parser.add_argument('--channel', type=int, default=0)
    args = parser.parse_args()

    plt = set_paper_style()
    run = select_run(args.dataset, args.method)
    if not run:
        print(f'{args.dataset}/{args.method}: no recorded switch with predictions; '
              f'figure skipped (no synthetic case is ever drawn)')
        return 0
    ident = identity(run)
    transfers = run['transfers']
    switches = transfers[transfers['event_type'] == 'switch'].copy()

    picked: List[dict] = []
    for _, event in switches.sort_values('window_idx').iterrows():
        client_id = str(event['client_id'])
        arrays = run['predictions'].get(client_id)
        if not arrays:
            continue
        idx = np.asarray(arrays['window_idx'])
        before_idx = int(idx[idx < int(event['window_idx'])].max()) \
            if (idx < int(event['window_idx'])).any() else None
        after_idx = int(event['window_idx'])
        before = window_slice(run, client_id, before_idx) if before_idx else None
        after = window_slice(run, client_id, after_idx)
        if after is None:
            continue
        picked.append({'event': event, 'client_id': client_id,
                       'before_idx': before_idx, 'before': before, 'after': after})
        if len(picked) >= args.cases:
            break

    if not picked:
        print(f'{args.dataset}/{args.method}: switches exist but their windows '
              f'were not recorded; figure skipped')
        return 0

    fig, axes = plt.subplots(len(picked), 2, figsize=(6.6, 1.5 * len(picked)),
                             squeeze=False)
    data_rows: List[dict] = []
    for row, case in enumerate(picked):
        for col, (label, payload, idx) in enumerate((
                ('before switch', case['before'], case['before_idx']),
                ('after switch', case['after'], case['before_idx'] is not None
                 and case['event']['window_idx']))):
            axis = axes[row, col]
            if payload is None:
                axis.set_visible(False)
                continue
            truth = payload['y_true'][:, min(args.channel,
                                             payload['y_true'].shape[-1] - 1)]
            pred = payload['y_pred'][:, min(args.channel,
                                            payload['y_pred'].shape[-1] - 1)]
            if 'x_history' in payload:
                hist = payload['x_history'][:, min(args.channel,
                                                   payload['x_history'].shape[-1] - 1)]
                axis.plot(np.arange(-len(hist), 0), hist, color='#444444', lw=1.0,
                          label='Input history')
            axis.plot(np.arange(1, len(truth) + 1), truth, color='#111111',
                      lw=1.2, label='Ground truth')
            axis.plot(np.arange(1, len(pred) + 1), pred, color='#d62728', lw=1.2,
                      ls='--', label='Prediction')
            axis.axvline(0, color='#888888', lw=0.8, ls=':')
            mse = float(np.mean((truth - pred) ** 2))
            axis.set_title(f'{label}  window {idx}  MSE={mse:.4f}', loc='left')
            axis.legend(loc='upper left', frameon=False, fontsize=6)
            series = {'ground_truth': (truth, 1), 'prediction': (pred, 1)}
            if 'x_history' in payload:
                hist = payload['x_history'][:, min(args.channel,
                                                   payload['x_history'].shape[-1] - 1)]
                series['input_history'] = (hist, -len(hist))
            for name, (values, start) in series.items():
                for offset, value in enumerate(values):
                    data_rows.append({
                        'dataset': args.dataset, 'method': ident['method'],
                        'client_id': case['client_id'], 'phase': label,
                        'window_idx': idx, 'channel': args.channel,
                        'series': name, 'step': int(start + offset),
                        'value': float(value), 'window_mse': mse,
                        'switch_window': int(case['event']['window_idx']),
                        'old_expert_id': str(case['event'].get('old_expert_id', '')),
                        'new_expert_id': str(case['event'].get('new_expert_id', ''))})
    fig.suptitle(f'{args.dataset}  {pretty(ident["method"])}  client '
                 f'{picked[0]["client_id"]}  expert switches', y=1.0, fontsize=8)
    fig.tight_layout(h_pad=0.5)
    stem = (FIGURES / 'forecasting_stages'
            / f'forecasting_transfer_cases_{args.dataset}')
    produced = save_figure(fig, stem)
    if data_rows:
        produced.append(save_figure_data(pd.DataFrame(data_rows), stem))
    for path in produced:
        print('wrote', path.relative_to(FIGURES.parents[1]))
    plt.close(fig)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
