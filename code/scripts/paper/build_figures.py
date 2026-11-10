"""Aggregate paper figures.

Every figure is skipped with an explicit message when the runs it needs are not
on disk; nothing is drawn from a default or a guess.

    python scripts/paper/build_figures.py --dataset ETTh1
    python scripts/paper/build_figures.py --dataset ETTh1 --only convergence
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
                    order_key, pretty, rolling_mse, save_figure,
                    save_figure_data, set_paper_style)


def runs_for(dataset: str) -> List[dict]:
    out = []
    for path in find_runs():
        run = load_run(path)
        if identity(run)['dataset'] == dataset:
            out.append(run)
    return out


def convergence(runs: List[dict], window: int = 64, dataset: str = ''):
    """Running prequential MSE; the plotted series is written beside the figure."""
    plt = set_paper_style()
    plotted = []
    by_method: Dict[str, List[pd.DataFrame]] = {}
    for run in runs:
        if run['per_window'] is None or run['per_window'].empty:
            continue
        curve = rolling_mse(run['per_window'], window=window)
        if curve.empty:
            continue
        by_method.setdefault(str(run['config'].get('method')), []).append(
            curve.set_index('window_idx')['rolling_mse'])
    if not by_method:
        print('convergence: no per-window data')
        return None
    colours = method_colours(by_method)
    fig, axis = plt.subplots(figsize=(3.4, 2.1))
    for method in sorted(by_method, key=order_key):
        frame = pd.concat(by_method[method], axis=1, join='inner')
        if frame.empty:
            continue
        mean = frame.mean(axis=1)
        axis.plot(mean.index, mean.to_numpy(), color=colours[method],
                  label=pretty(method))
        std = frame.std(axis=1, ddof=1) if frame.shape[1] > 1 else None
        if std is not None:
            axis.fill_between(mean.index, (mean - std).to_numpy(),
                              (mean + std).to_numpy(), color=colours[method],
                              alpha=0.18, linewidth=0)
        plotted.append(pd.DataFrame({
            'dataset': dataset, 'method': method, 'rolling_window': window,
            'window_idx': mean.index.to_numpy(),
            'rolling_mse': mean.to_numpy(),
            'rolling_mse_std': (std.to_numpy() if std is not None
                                else np.zeros(len(mean))),
            'num_seeds': frame.shape[1]}))
    if plotted:
        save_figure_data(pd.concat(plotted, ignore_index=True),
                         FIGURES / 'online_convergence'
                         / f'online_convergence_{dataset}')
    axis.set_xlabel('processed test windows')
    axis.set_ylabel(f'{window}-window rolling MSE')
    axis.set_title('Online adaptation convergence', loc='left')
    axis.legend(frameon=False, ncol=1)
    return fig


def transfer_analysis(runs: List[dict], dataset: str = ''):
    plt = set_paper_style()
    predicted, realized, methods = [], [], []
    transfer_counts: Dict[str, int] = {}
    for run in runs:
        ident = identity(run)
        feedback = run['feedback']
        transfer_counts[ident['method']] = int(run['summary'].get('transfer_events') or 0)
        if feedback is None or feedback.empty:
            continue
        if 'predicted_gain' not in feedback or 'realized_gain' not in feedback:
            continue
        pred = pd.to_numeric(feedback['predicted_gain'], errors='coerce')
        real = pd.to_numeric(feedback['realized_gain'], errors='coerce')
        mask = pred.notna() & real.notna()
        predicted.extend(pred[mask].tolist())
        realized.extend(real[mask].tolist())
        methods.extend([ident['method']] * int(mask.sum()))
    figures = []
    if predicted:
        fig, axis = plt.subplots(figsize=(2.4, 2.2))
        axis.scatter(predicted, realized, s=4, alpha=0.35, color='#1f77b4',
                     linewidths=0)
        limit = float(np.nanmax(np.abs([*predicted, *realized]))) or 1.0
        axis.plot([0, limit], [0, limit], color='#888888', lw=0.8, ls=':')
        axis.axhline(0.0, color='#888888', lw=0.5)
        axis.axvline(0.0, color='#888888', lw=0.5)
        axis.set_xlabel('predicted gain (decision time)')
        axis.set_ylabel('realised gain (after reveal)')
        axis.set_title(f'Predicted vs realised (n={len(predicted)})', loc='left')
        figures.append(('transfer_analysis/predicted_vs_realized', fig))
        save_figure_data(
            pd.DataFrame({'dataset': dataset, 'method': methods,
                          'predicted_gain': predicted,
                          'realized_gain': realized}),
            FIGURES / 'transfer_analysis' / f'{dataset}_predicted_vs_realized')
    if any(transfer_counts.values()):
        save_figure_data(
            pd.DataFrame({'dataset': dataset,
                          'method': sorted(transfer_counts, key=order_key),
                          'switch_events': [transfer_counts[m] for m in
                                            sorted(transfer_counts, key=order_key)]}),
            FIGURES / 'transfer_analysis' / f'{dataset}_transfer_counts')
        fig, axis = plt.subplots(figsize=(3.4, 1.8))
        ordered = sorted(transfer_counts, key=order_key)
        values = [transfer_counts[m] for m in ordered]
        colours = method_colours(ordered)
        axis.barh([pretty(m) for m in ordered], values,
                  color=[colours[m] for m in ordered])
        axis.set_xlabel('recorded expert switches')
        axis.set_title('Transfer events', loc='left')
        figures.append(('transfer_analysis/transfer_counts', fig))
    return figures


def efficiency(runs: List[dict], dataset: str = ''):
    plt = set_paper_style()
    rows = []
    for run in runs:
        ident = identity(run)
        summ = run['summary']
        rows.append({
            'method': ident['method'],
            'comm_MB': (summ.get('total_communication_bytes') or 0) / 1e6,
            'mse': summ.get('overall_mse'),
            'runtime_s': summ.get('total_runtime'),
            'adapt_s': summ.get('adaptation_time') or 0,
            'shadow_s': summ.get('shadow_evaluation_time') or 0,
            'forecast_s': summ.get('forecasting_time') or 0,
            'sync_s': summ.get('encoder_sync_time') or 0,
            'windows': summ.get('evaluated_windows') or 0,
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return []
    frame = frame.groupby('method').mean(numeric_only=True).reset_index()
    frame = frame.sort_values('method', key=lambda s: s.map(order_key))
    frame.insert(0, 'dataset', dataset)
    save_figure_data(frame, FIGURES / 'efficiency' / f'efficiency_{dataset}')
    # One CSV per figure as well, so each panel can be rebuilt on its own.
    save_figure_data(frame[['dataset', 'method', 'comm_MB']],
                     FIGURES / 'efficiency' / f'{dataset}_communication_cost')
    save_figure_data(frame[['dataset', 'method', 'forecast_s', 'shadow_s',
                            'adapt_s', 'sync_s']],
                     FIGURES / 'efficiency' / f'{dataset}_computational_cost')
    save_figure_data(frame[['dataset', 'method', 'comm_MB', 'mse']],
                     FIGURES / 'efficiency' / f'{dataset}_cost_performance')
    colours = method_colours(frame['method'])
    figures = []

    fig, axis = plt.subplots(figsize=(3.2, 1.9))
    axis.bar([pretty(m) for m in frame['method']], frame['comm_MB'],
             color=[colours[m] for m in frame['method']])
    axis.set_ylabel('communicated MB')
    axis.set_title('Communication cost', loc='left')
    axis.tick_params(axis='x', rotation=45)
    figures.append(('efficiency/communication_cost', fig))

    parts = frame.set_index('method')[['forecast_s', 'shadow_s', 'adapt_s', 'sync_s']]
    fig, axis = plt.subplots(figsize=(3.2, 1.9))
    bottom = np.zeros(len(parts))
    for column, colour in zip(parts.columns,
                              ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']):
        axis.bar([pretty(m) for m in parts.index], parts[column].to_numpy(),
                 bottom=bottom, label=column.replace('_s', ''), color=colour)
        bottom += parts[column].to_numpy()
    axis.set_ylabel('seconds (sum of per-event timers)')
    axis.set_title('Computational cost by stage', loc='left')
    axis.tick_params(axis='x', rotation=45)
    axis.legend(frameon=False, fontsize=6)
    figures.append(('efficiency/computational_cost', fig))

    fig, axis = plt.subplots(figsize=(2.6, 2.1))
    for _, row in frame.iterrows():
        axis.scatter(row['comm_MB'], row['mse'],
                     color=colours.get(row['method'], '#1f77b4'), s=18)
        axis.annotate(pretty(row['method']), (row['comm_MB'], row['mse']),
                      fontsize=5, xytext=(2, 2), textcoords='offset points')
    axis.set_xlabel('communicated MB')
    axis.set_ylabel('MSE')
    axis.set_title('Cost vs accuracy', loc='left')
    figures.append(('efficiency/cost_performance', fig))
    return figures


def drift_recovery(runs: List[dict]):
    plt = set_paper_style()
    # ``drift_score`` rows are per-window telemetry, not change points.  A
    # change point needs a real ``regime_change`` event, which only a
    # controlled-shift run can emit; drawing a line at every score row would put
    # 80 markers on a curve that has no shifts in it.
    usable = []
    for run in runs:
        regime = run['regime']
        if regime is None or regime.empty or 'event_type' not in regime:
            continue
        changes = regime[regime['event_type'] != 'drift_score']
        if not changes.empty:
            usable.append((run, changes))
    if not usable:
        print('drift_recovery: no regime_change events recorded (natural split '
              'has no ground-truth change points); figure skipped')
        return None
    run, changes = usable[0]
    method = str(run['config'].get('method'))
    curve = rolling_mse(run['per_window'], window=64)
    fig, axis = plt.subplots(figsize=(3.4, 2.0))
    axis.plot(curve['window_idx'], curve['rolling_mse'], color='#1f77b4')
    for _, event in changes.iterrows():
        axis.axvline(float(event['window_idx']), color='#d62728', lw=0.8, ls='--')
    axis.set_xlabel('processed test windows')
    axis.set_ylabel('64-window rolling MSE')
    axis.set_title(f'Shift recovery ({pretty(method)})', loc='left')
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--dataset', default='ETTh1')
    parser.add_argument('--window', type=int, default=64)
    parser.add_argument('--only', default=None,
                        choices=['convergence', 'transfer', 'efficiency', 'drift'])
    args = parser.parse_args()

    runs = runs_for(args.dataset)
    if not runs:
        print(f'no runs for {args.dataset}')
        return 0

    produced = []
    if args.only in (None, 'convergence'):
        fig = convergence(runs, window=args.window, dataset=args.dataset)
        if fig is not None:
            produced += save_figure(fig, FIGURES / 'online_convergence'
                                    / f'online_convergence_{args.dataset}')
            import matplotlib.pyplot as plt
            plt.close(fig)

    if args.only in (None, 'transfer'):
        for stem, fig in transfer_analysis(runs, dataset=args.dataset):
            produced += save_figure(fig, FIGURES / stem.replace(
                'transfer_analysis/', f'transfer_analysis/{args.dataset}_'))
            import matplotlib.pyplot as plt
            plt.close(fig)

    if args.only in (None, 'efficiency'):
        for stem, fig in efficiency(runs, dataset=args.dataset):
            produced += save_figure(fig, FIGURES / stem.replace(
                'efficiency/', f'efficiency/{args.dataset}_'))
            import matplotlib.pyplot as plt
            plt.close(fig)

    if args.only in (None, 'drift'):
        fig = drift_recovery(runs)
        if fig is not None:
            produced += save_figure(fig, FIGURES / 'drift_recovery'
                                    / f'drift_recovery_{args.dataset}')
            import matplotlib.pyplot as plt
            plt.close(fig)

    if not produced:
        print('no figure produced')
        return 0
    for path in produced:
        print('wrote', path.relative_to(FIGURES.parents[1]))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
