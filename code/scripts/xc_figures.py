"""Every figure the metrics tables need, generated straight from the CSVs.

One script, one command, so a re-run of the experiments regenerates the whole
figure set instead of leaving hand-edited PNGs in the paper directory.

Design rules
------------
* **English labels.** Avoids CJK font fallback problems on any machine and the
  paper is English anyway.  Chinese captions live in ``PAPER_METRICS.md``.
* **Comparable scales.** Nine datasets differ by 100x in MSE, so almost every
  panel plots a *difference from a reference*, never a raw level.
* **No smoothing, no cherry-picking.** Curves are the logged runs; if something
  is noisy the figure shows the noise.

Usage::

    python scripts/xc_figures.py             # all figures
    python scripts/xc_figures.py --only fig4 # one figure
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FEDTDO = ROOT / 'results' / 'fedtdo'
METRICS = FEDTDO / '_metrics'
BASELINES = ROOT / 'results_baselines'
FIGS = METRICS / 'figs'

ORDER = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Exchange', 'ILI', 'ECL', 'Traffic']
ARMS = ['base', 'center', 'decorr']
ARM_LABEL = {'base': 'FedTDO-raw', 'center': 'FedTDO-center', 'decorr': 'FedTDO-decorr'}
METHODS = ['local', 'fedavg', 'fedala', 'fedpoe', 'pfedcs']
ARM_COLOR = {'base': '#1f4e79', 'center': '#2e8b8b', 'decorr': '#c0504d'}

plt.rcParams.update({
    'figure.dpi': 160,
    'savefig.dpi': 160,
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linewidth': 0.6,
    'legend.frameon': False,
})


def read_csv(name):
    path = METRICS / name
    return list(csv.DictReader(path.open(encoding='utf-8')))


def load_baselines():
    table = {}
    for directory in BASELINES.glob('BL_*_xc_s2021_*_r2_exp'):
        parts = directory.name.split('_')
        method, dataset = parts[1], parts[4]
        summary = directory / 'summary.json'
        if summary.exists():
            table[(dataset, method)] = json.loads(summary.read_text())
    return table


def load_rounds(dataset, arm, seed=2021):
    directory = FEDTDO / f'{dataset.lower()}{_nc(dataset)}_s{seed}_b{_budget(dataset)}_{arm}'
    if not directory.exists():
        legacy = {'Traffic': 'traffic32_s2021_joint_b9_fix'}
        alt = FEDTDO / legacy.get(dataset, '')
        if alt.exists():
            directory = alt
    path = directory / 'rounds.json'
    return json.loads(path.read_text()) if path.exists() else []


_NC = {'ETTh1': 7, 'ETTh2': 7, 'ETTm1': 7, 'ETTm2': 7, 'Weather': 7,
       'Exchange': 8, 'ILI': 7, 'ECL': 32, 'Traffic': 32}
_BUDGET = {'ETTh1': 2, 'ETTh2': 2, 'ETTm1': 2, 'ETTm2': 2, 'Weather': 2,
           'Exchange': 2, 'ILI': 2, 'ECL': 9, 'Traffic': 9}


def _nc(dataset):
    return _NC[dataset]


def _budget(dataset):
    return _BUDGET[dataset]


def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    path = FIGS / f'{name}.png'
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  {path.relative_to(ROOT)}')


# --------------------------------------------------------------------------
def fig1_gain_vs_local():
    """Predictive gain over the local-only baseline, per method per dataset."""
    rows = {(r['dataset'], r['method']): r for r in read_csv('paper_main.csv') if r['seed'] == '2021'}
    methods = ['fedavg', 'fedala', 'fedpoe', 'pfedcs', 'FedTDO-raw', 'FedTDO-center', 'FedTDO-decorr']
    colors = ['#b0b0b0', '#8c8c8c', '#6e6e6e', '#4d4d4d', '#1f4e79', '#2e8b8b', '#c0504d']

    data = {}
    for method in methods:
        values = []
        for dataset in ORDER:
            row = rows.get((dataset, method))
            local = rows.get((dataset, 'local'))
            if not row or not local:
                values.append(np.nan)
                continue
            values.append(100.0 * (float(local['mse']) - float(row['mse'])) / float(local['mse']))
        data[method] = values

    x = np.arange(len(ORDER))
    width = 0.11
    fig, ax = plt.subplots(figsize=(9.2, 3.4))
    for index, (method, color) in enumerate(zip(methods, colors)):
        ax.bar(x + (index - 3) * width, data[method], width, label=method, color=color,
               edgecolor='white', linewidth=0.4)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(ORDER, rotation=20)
    ax.set_ylabel('MSE reduction vs local (%)')
    ax.set_title('Figure 1. Predictive gain over the local-only client model (seed 2021)')
    ax.legend(ncol=7, loc='upper center', bbox_to_anchor=(0.5, -0.18), fontsize=8)
    save(fig, 'fig1_gain_vs_local')


def fig2_winloss_three_scopes():
    """Same comparison under mean / median / worst-client: the answer changes."""
    baselines = load_baselines()
    fedemoe = {(r['dataset'], r['arm']): r for r in read_csv('fedtdo_fedemoe.csv') if r['seed'] == '2021'}
    summary = {(r['dataset'], r['arm']): r for r in read_csv('fedtdo_summary.csv') if r['seed'] == '2021'}

    panels = [('Mean client MSE', 'mean'), ('Median client MSE', 'median'), ('Worst client MSE', 'worst')]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.4), sharey=True)
    for ax, (title, mode) in zip(axes, panels):
        values = []
        for dataset in ORDER:
            if mode == 'mean':
                base_best = min(float(baselines[(dataset, m)]['mean_mse'])
                                for m in METHODS if (dataset, m) in baselines)
                mine = min(float(summary[(dataset, a)]['mean_client_mse']) for a in ARMS
                           if (dataset, a) in summary)
            elif mode == 'median':
                base_best = min(float(baselines[(dataset, m)]['median_mse'])
                                for m in METHODS if (dataset, m) in baselines)
                mine = min(float(summary[(dataset, a)]['median_client_mse']) for a in ARMS
                           if (dataset, a) in summary)
            else:
                base_best = min(float(baselines[(dataset, m)]['max_mse'])
                                for m in METHODS if (dataset, m) in baselines)
                mine = min(float(fedemoe[(dataset, a)]['worst_client_mse']) for a in ARMS
                           if (dataset, a) in fedemoe)
            values.append(100.0 * (base_best - mine) / base_best)
        colors = ['#1f4e79' if v >= 0 else '#c0504d' for v in values]
        ax.barh(np.arange(len(ORDER)), values, color=colors, height=0.68)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_yticks(np.arange(len(ORDER)))
        ax.set_yticklabels(ORDER)
        ax.set_title(title)
        ax.set_xlabel('vs best baseline (%)')
        ax.invert_yaxis()
    counts = []
    for mode in ('mean', 'median', 'worst'):
        wins = 0
        for dataset in ORDER:
            if mode == 'mean':
                reference = min(float(baselines[(dataset, m)]['mean_mse']) for m in METHODS
                                if (dataset, m) in baselines)
                mine = min(float(summary[(dataset, a)]['mean_client_mse']) for a in ARMS
                           if (dataset, a) in summary)
            elif mode == 'median':
                reference = min(float(baselines[(dataset, m)]['median_mse']) for m in METHODS
                                if (dataset, m) in baselines)
                mine = min(float(summary[(dataset, a)]['median_client_mse']) for a in ARMS
                           if (dataset, a) in summary)
            else:
                reference = min(float(baselines[(dataset, m)]['max_mse']) for m in METHODS
                                if (dataset, m) in baselines)
                mine = min(float(fedemoe[(dataset, a)]['worst_client_mse']) for a in ARMS
                           if (dataset, a) in fedemoe)
            wins += reference > mine
        counts.append(wins)
    fig.suptitle('Figure 2. Whether FedTDO beats the best baseline depends on which client you score\n'
                 f'(win count: mean {counts[0]}/9, median {counts[1]}/9, worst client {counts[2]}/9)',
                 fontsize=10, y=1.06)
    save(fig, 'fig2_winloss_three_scopes')


def fig3_client_distribution():
    """Per-client MSE spread -- why a mean-only table is decided by one client."""
    rows = read_csv('fedtdo_per_client.csv')
    datasets = ['ECL', 'Traffic', 'ETTm1', 'ETTh1']
    fig, axes = plt.subplots(1, len(datasets), figsize=(10.4, 3.2))
    for ax, dataset in zip(axes, datasets):
        groups, labels, colors = [], [], []
        local = [v for v in read_csv('fedtdo_summary.csv')
                 if v['dataset'] == dataset and v['seed'] == '2021']
        series = []
        for arm in ARMS:
            values = [float(r['mse']) for r in rows
                      if r['dataset'] == dataset and r['arm'] == arm and r['seed'] == '2021']
            if values:
                series.append((ARM_LABEL[arm], values, ARM_COLOR[arm]))
        for label, values, color in series:
            groups.append(values)
            labels.append(label)
            colors.append(color)
        parts = ax.boxplot(groups, labels=labels, patch_artist=True, widths=0.55,
                           showfliers=True, flierprops=dict(markersize=2.5, alpha=0.6))
        for patch, color in zip(parts['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        ax.set_yscale('log')
        ax.set_title(dataset)
        ax.set_ylabel('client MSE (log)' if dataset == datasets[0] else '')
        ax.tick_params(axis='x', rotation=30, labelsize=7.5)
    fig.suptitle('Figure 3. Client-level MSE inside one run (seed 2021): the spread dwarfs the arm effect',
                 fontsize=10, y=1.02)
    save(fig, 'fig3_client_distribution')


def fig4_pareto():
    """Cost-adjusted objective against the price of a transfer."""
    rows = read_csv('fedtdo_pareto.csv')
    lambdas = sorted({float(r['lam_t']) for r in rows if float(r['lam_e']) == 0.0})
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for arm in ARMS:
        values = []
        for lam in lambdas:
            cell = [float(r['objective']) for r in rows
                    if r['arm'] == arm and float(r['lam_e']) == 0.0 and float(r['lam_t']) == lam]
            values.append(st.fmean(cell) if cell else np.nan)
        ax.plot(lambdas, values, marker='o', markersize=4, color=ARM_COLOR[arm],
                label=ARM_LABEL[arm], linewidth=1.6)
    ax.set_xscale('symlog', linthresh=0.01)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel(r'transfer price $\lambda_t$  ($\lambda_e = 0$)')
    ax.set_ylabel('objective (9-dataset mean)')
    ax.set_title('Figure 4. Pricing a transfer flips the arm ranking')
    ax.legend(fontsize=8)
    ax.annotate('at this price the arm with the fewest transfers wins',
                xy=(0.1, 4.97), xytext=(0.0008, -55), fontsize=7.5,
                arrowprops=dict(arrowstyle='->', lw=0.7, connectionstyle='arc3,rad=-0.2'))
    save(fig, 'fig4_pareto_transfer_price')


def fig5_cumulative_regret():
    """Cumulative regret on the three datasets where the oracle is non-trivial."""
    datasets = ['Traffic', 'ECL', 'ILI']
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.0))
    for ax, dataset in zip(axes, datasets):
        for arm in ARMS:
            rounds = load_rounds(dataset, arm)
            if not rounds:
                continue
            n = _nc(dataset) * 2
            per_round = []
            for start in range(0, len(rounds), n):
                chunk = rounds[start:start + n]
                oracle = sum(max(r['raw_gains'].values()) if r['raw_gains'] else 0.0 for r in chunk)
                real = sum(max(0.0, r['default_loss'] - r['deployed_loss']) for r in chunk)
                per_round.append(oracle - real)
            ax.plot(np.cumsum(per_round), color=ARM_COLOR[arm], label=ARM_LABEL[arm], linewidth=1.5)
        ax.set_title(dataset)
        ax.set_xlabel('round')
        if ax is axes[0]:
            ax.set_ylabel('cumulative regret')
    axes[0].legend(fontsize=8)
    fig.suptitle('Figure 5. Cumulative regret against the budget-respecting in-round oracle (seed 2021)',
                 fontsize=10, y=1.02)
    save(fig, 'fig5_cumulative_regret')


def fig6_recovery():
    """Rounds to recover after a detected drift, plus the censoring count."""
    rows = {(r['dataset'], r['arm']): r for r in read_csv('fedtdo_summary.csv') if r['seed'] == '2021'}
    datasets = [d for d in ORDER
                if all(rows.get((d, a)) and rows[(d, a)]['recovery_rounds_mean'] not in ('nan', '')
                       for a in ARMS)]
    x = np.arange(len(datasets))
    width = 0.26
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    for index, arm in enumerate(ARMS):
        values = [float(rows[(d, arm)]['recovery_rounds_mean']) for d in datasets]
        events = [int(float(rows[(d, arm)]['recovery_events'])) for d in datasets]
        ax.bar(x + (index - 1) * width, values, width, color=ARM_COLOR[arm],
               label=ARM_LABEL[arm], edgecolor='white', linewidth=0.4)
        for pos, value, event in zip(x + (index - 1) * width, values, events):
            if event < 5:
                ax.annotate('*', (pos, value), textcoords='offset points', xytext=(0, 1),
                            ha='center', fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel('rounds to recover')
    ax.set_title('Figure 6. Time to recover after a detected drift\n'
                 '(\\* = fewer than 5 events, not comparable)', fontsize=9.5)
    ax.legend(fontsize=8)
    save(fig, 'fig6_recovery_rounds')


def fig7_resource_split():
    """First-time deployments vs expert swaps: the two are not the same cost."""
    rows = {(r['dataset'], r['arm']): r for r in read_csv('fedtdo_summary.csv') if r['seed'] == '2021'}
    datasets = ['ECL', 'Traffic', 'Exchange', 'ETTh1']
    x = np.arange(len(datasets))
    width = 0.26
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.2))
    for ax, key, title in zip(axes, ['deploys', 'expert_swaps'],
                              ['first-time deployments', 'expert swaps']):
        for index, arm in enumerate(ARMS):
            values = [float(rows[(d, arm)][key]) for d in datasets]
            ax.bar(x + (index - 1) * width, values, width, color=ARM_COLOR[arm],
                   label=ARM_LABEL[arm], edgecolor='white', linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.set_title(title)
        ax.set_ylabel('count (sum over clients)')
    axes[0].legend(fontsize=8)
    fig.suptitle('Figure 7. FedTDO-center transfers more because it boards more,\n'
                 'not because it re-seats more', fontsize=9.5, y=1.04)
    save(fig, 'fig7_resource_split')


def fig8_convergence():
    """Rounds to first deploy, and how much switching still happens at the end."""
    rows = {(r['dataset'], r['arm']): r for r in read_csv('fedtdo_fedemoe.csv') if r['seed'] == '2021'}
    x = np.arange(len(ORDER))
    width = 0.26
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.2))
    for index, arm in enumerate(ARMS):
        first = [float(rows[(d, arm)]['rounds_to_first_deploy']) for d in ORDER]
        late = [float(rows[(d, arm)]['late_switch_share']) for d in ORDER]
        axes[0].bar(x + (index - 1) * width, first, width, color=ARM_COLOR[arm],
                    label=ARM_LABEL[arm], edgecolor='white', linewidth=0.4)
        axes[1].bar(x + (index - 1) * width, late, width, color=ARM_COLOR[arm],
                    label=ARM_LABEL[arm], edgecolor='white', linewidth=0.4)
    for ax, title, label in zip(axes,
                                ['rounds to the first expert', 'swap share in the last 25% of rounds'],
                                ['rounds', 'share of swaps']):
        ax.set_xticks(x)
        ax.set_xticklabels(ORDER, rotation=25, fontsize=7.5)
        ax.set_title(title)
        ax.set_ylabel(label)
    axes[0].legend(fontsize=8)
    fig.suptitle('Figure 8. Deployment onset and late-stage switching: no consistent arm ordering\n'
                 '(only Weather separates cleanly; see PAPER_METRICS.md §8)', fontsize=9.5, y=1.06)
    save(fig, 'fig8_convergence_stability')


def fig9_latency():
    """Latency distribution, per window and per batched candidate."""
    path = METRICS / 'latency_cpu.json'
    if not path.exists():
        print('  (skip fig9: latency_cpu.json missing)')
        return
    payload = json.loads(path.read_text())
    runs = payload['runs']
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    for ax, key, title, units in zip(axes, ['inference', 'evaluation'],
                                     ['one window (batch=1)', 'batched candidates'],
                                     ['ms', 'ms']):
        medians = [r[key]['median_ms'] for r in runs]
        p95 = [r[key]['p95_ms'] for r in runs]
        labels = [f'run {i + 1}' for i in range(len(runs))]
        positions = np.arange(len(runs))
        ax.bar(positions, medians, 0.5, color='#1f4e79', alpha=0.75, label='median')
        ax.errorbar(positions, medians, yerr=[np.zeros(len(medians)),
                                             np.array(p95) - np.array(medians)],
                    fmt='none', ecolor='#c0504d', capsize=3, linewidth=0.9, label='p95')
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_title(f'{title}')
        ax.set_ylabel(units)
        ax.legend(fontsize=7.5)
    fig.suptitle(f"Figure 9. Client-model inference latency, CPU "
                 f"({payload['params']:,} parameters, 300 repeats per run)", fontsize=9.5, y=1.02)
    save(fig, 'fig9_latency')


def fig10_heatmap():
    """One picture for 'where does each method actually win'."""
    rows = {(r['dataset'], r['method']): r for r in read_csv('paper_main.csv') if r['seed'] == '2021'}
    columns = ['fedavg', 'fedala', 'fedpoe', 'pfedcs', 'FedTDO-raw', 'FedTDO-center', 'FedTDO-decorr']
    matrix = np.full((len(ORDER), len(columns)), np.nan)
    for i, dataset in enumerate(ORDER):
        local = rows.get((dataset, 'local'))
        for j, method in enumerate(columns):
            row = rows.get((dataset, method))
            if row and local:
                matrix[i, j] = 100.0 * (float(local['mse']) - float(row['mse'])) / float(local['mse'])

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    limit = np.nanmax(np.abs(matrix))
    image = ax.imshow(matrix, cmap='RdBu_r', vmin=-limit, vmax=limit, aspect='auto')
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels([c.replace('FedTDO-', 'FT-') for c in columns], rotation=25)
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels(ORDER)
    for i in range(len(ORDER)):
        for j in range(len(columns)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f'{matrix[i, j]:+.0f}', ha='center', va='center', fontsize=7,
                        color='white' if abs(matrix[i, j]) > 0.6 * limit else 'black')
    ax.grid(False)
    fig.colorbar(image, ax=ax, label='MSE reduction vs local (%)', fraction=0.04)
    ax.set_title('Figure 10. Gain over local, by dataset and method (seed 2021)')
    save(fig, 'fig10_heatmap')


def fig11_fairness():
    """Jain index vs worst-client gap: is the benefit shared or concentrated?"""
    rows = [r for r in read_csv('fedtdo_fedemoe.csv') if r['seed'] == '2021']
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    # The three arms land on top of each other, which *is* the finding, so label a
    # dataset once (from the base arm) and let the other two overlap as hollow marks.
    for arm in ARMS:
        subset = [r for r in rows if r['arm'] == arm]
        filled = arm == 'base'
        ax.scatter([float(r['jain_index']) for r in subset],
                   [float(r['cv_client_mse']) for r in subset],
                   s=40 if filled else 42, color=ARM_COLOR[arm] if filled else 'none',
                   edgecolor=ARM_COLOR[arm], linewidth=1.0,
                   label=ARM_LABEL[arm], alpha=0.9, marker='o' if filled else 'o')
        if filled:
            for r in subset:
                ax.annotate(r['dataset'], (float(r['jain_index']), float(r['cv_client_mse'])),
                            textcoords='offset points', xytext=(5, 4), fontsize=7)
    ax.set_xlabel("Jain fairness index on 1/MSE (higher = more even)")
    ax.set_ylabel('coefficient of variation of client MSE')
    ax.set_title('Figure 11. Client heterogeneity is a dataset property, not an arm property\n'
                 '(filled = FedTDO-raw; hollow = centre / decorr, sitting on the same point)',
                 fontsize=9)
    ax.legend(fontsize=8)
    save(fig, 'fig11_fairness')


FIGURES = {
    'fig1': fig1_gain_vs_local,
    'fig2': fig2_winloss_three_scopes,
    'fig3': fig3_client_distribution,
    'fig4': fig4_pareto,
    'fig5': fig5_cumulative_regret,
    'fig6': fig6_recovery,
    'fig7': fig7_resource_split,
    'fig8': fig8_convergence,
    'fig9': fig9_latency,
    'fig10': fig10_heatmap,
    'fig11': fig11_fairness,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', nargs='+', choices=sorted(FIGURES), default=None)
    args = parser.parse_args()
    targets = args.only or sorted(FIGURES)
    print(f'writing {len(targets)} figures to {FIGS}')
    for name in targets:
        FIGURES[name]()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
