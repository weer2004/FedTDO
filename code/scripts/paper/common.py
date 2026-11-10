"""Shared loading / aggregation helpers for the paper data tree.

Everything here reads only what the recorder wrote.  Nothing is inferred from a
configuration: if a table is missing, the caller is told it is missing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
# The paper tree defaults to results/paper_fedtdo, but PAPER_ROOT lets a smoke
# run aggregate its own scratch tree without touching (or reading) the real one.
_DATA_ROOT = Path(os.environ['PAPER_ROOT']) if os.environ.get('PAPER_ROOT') \
    else ROOT / 'results' / 'paper_fedtdo'
if not _DATA_ROOT.is_absolute():
    _DATA_ROOT = ROOT / _DATA_ROOT
RAW = _DATA_ROOT / 'raw'
AGG = _DATA_ROOT / 'aggregated'
TABLES = _DATA_ROOT / 'tables'
FIGURES = _DATA_ROOT / 'figures'

# Method labels used in the figures.  Order defines the plotting order.
PRETTY = {
    'fedtdo': 'FedTDO', 'fedtdo_nofreeze': 'FedTDO (no freeze)',
    'fedtdo_joint': 'FedTDO (joint warm-up)', 'isomorphic': 'Isomorphic (no transfer)',
    'local_encoder': 'Local encoder', 'local': 'Local', 'fedavg': 'FedAvg',
    'fedavg_sgd': 'FedAvg-SGD', 'fedprox': 'FedProx', 'scaffold': 'SCAFFOLD',
    'fedadam': 'FedAdam', 'fednova': 'FedNova', 'fedper': 'FedPer',
    'fedrep': 'FedRep', 'pfedme': 'pFedMe', 'ditto': 'Ditto', 'fedl2p': 'FedL2P',
    'fedala': 'FedALA', 'fedda': 'FedDA', 'fedgtp': 'FedGTP', 'fedkme': 'FedKMe',
    'fedkwaz': 'FedKwaz', 'fedpkda': 'FedPKDA', 'fedpoe': 'FedPoE',
    'pfedcs': 'PFedCS', 'pfedfda': 'PFedFDA',
}

METHOD_ORDER = ['fedtdo', 'fedtdo_nofreeze', 'fedtdo_joint', 'isomorphic',
                'local_encoder', 'local', 'fedavg', 'fedprox', 'fedavg_sgd',
                'scaffold', 'fedadam', 'fednova', 'fedper', 'fedrep', 'pfedme',
                'ditto', 'fedl2p', 'fedala', 'pfedcs', 'pfedfda', 'fedpoe',
                'fedkwaz', 'fedpkda', 'fedkme', 'fedgtp', 'fedda']


def pretty(method: str) -> str:
    return PRETTY.get(str(method), str(method))


def order_key(method: str) -> int:
    method = str(method)
    return METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER)


def find_runs(root: Path = RAW) -> List[Path]:
    """Every directory that holds a finished (or in-progress) run."""
    if not root.exists():
        return []
    return sorted(p.parent for p in root.rglob('run_summary.json'))


def load_json(path: Path, default=None):
    try:
        with open(path, encoding='utf-8') as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return default


def load_run(run_dir: Path) -> Dict:
    """Everything one run produced, in memory.  Missing pieces are ``None``."""
    run_dir = Path(run_dir)
    payload = {
        'dir': run_dir,
        'config': load_json(run_dir / 'config.json', {}),
        'summary': load_json(run_dir / 'run_summary.json', {}),
        'per_window': None,
        'updates': None,
        'evaluations': None,
        'feedback': None,
        'transfers': None,
        'communication': None,
        'cost': None,
        'regime': None,
        'predictions': {},
    }
    for key, name in (('per_window', 'per_window_metrics'),
                      ('updates', 'online_updates'),
                      ('evaluations', 'candidate_evaluations'),
                      ('feedback', 'feedback_rewards'),
                      ('transfers', 'transfer_events'),
                      ('communication', 'communication_events'),
                      ('cost', 'cost_events'),
                      ('regime', 'regime_events')):
        path = run_dir / f'{name}.csv'
        if path.exists() and path.stat().st_size > 0:
            try:
                payload[key] = pd.read_csv(path)
            except (pd.errors.EmptyDataError, OSError):
                payload[key] = None
    pred_dir = run_dir / 'predictions'
    if pred_dir.exists():
        for npz in sorted(pred_dir.glob('client_*.npz')):
            client_id = npz.stem.replace('client_', '', 1)
            try:
                with np.load(npz) as data:
                    payload['predictions'][client_id] = {k: data[k] for k in data.files}
            except (OSError, ValueError):
                continue
    return payload


def identity(run: Dict) -> Dict:
    cfg = run['config']
    summ = run['summary']
    return {
        'run_id': cfg.get('run_id', run['dir'].name),
        'method': cfg.get('method', 'unknown'),
        'dataset': cfg.get('dataset', 'unknown'),
        'seed': cfg.get('seed', -1),
        'run_kind': cfg.get('run_kind', ''),
    }


def rolling_mse(per_window: pd.DataFrame, window: int = 64) -> pd.DataFrame:
    """Rolling prequential MSE per client, then the cross-client mean.

    The window index is the prequential position *within a client*, so the
    rolling statistic never mixes two clients' error series.
    """
    if per_window is None or per_window.empty:
        return pd.DataFrame(columns=['window_idx', 'rolling_mse', 'n_clients'])
    frames = []
    for client_id, group in per_window.groupby('client_id'):
        group = group.sort_values('window_idx')
        smooth = (group['sse'].rolling(window, min_periods=1).sum()
                  / group['num_targets'].rolling(window, min_periods=1).sum())
        frames.append(pd.DataFrame({
            'client_id': client_id,
            'window_idx': group['window_idx'].to_numpy(),
            'rolling_mse': smooth.to_numpy(),
        }))
    stacked = pd.concat(frames, ignore_index=True)
    out = (stacked.groupby('window_idx')
           .agg(rolling_mse=('rolling_mse', 'mean'),
                n_clients=('client_id', 'nunique'))
           .reset_index())
    return out


def stage_label(fraction: float) -> str:
    if fraction < 0.25:
        return 'early'
    if fraction < 0.75:
        return 'middle'
    return 'late'


def stage_window_indices(max_window: int, fractions=(0.10, 0.50, 0.90)) -> Dict[str, int]:
    """Pre-declared sample positions; never chosen after looking at the curves."""
    if max_window <= 0:
        return {}
    return {stage_label(f): max(1, int(round(f * max_window))) for f in fractions}


def ensure_dirs():
    for path in (AGG, TABLES, FIGURES):
        path.mkdir(parents=True, exist_ok=True)
    for name in ('online_convergence', 'adaptation_granularity',
                 'forecasting_comparison', 'forecasting_stages',
                 'transfer_analysis', 'drift_recovery', 'efficiency'):
        (FIGURES / name).mkdir(parents=True, exist_ok=True)


def save_figure(fig, stem: Path, formats=('pdf', 'png'), dpi: int = 200):
    stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(stem.with_suffix(f'.{fmt}'), dpi=dpi,
                    bbox_inches='tight', pad_inches=0.02)
    return [stem.with_suffix(f'.{fmt}') for fmt in formats]


def save_figure_data(frame, stem: Path) -> Path:
    """Companion CSV for a figure: the exact numbers that were drawn.

    Every figure writes one of these next to itself, so a plot can be rebuilt
    from the data tree alone without re-reading the raw run directories.
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    path = stem.with_suffix('.csv')
    frame.to_csv(path, index=False)
    return path


def set_paper_style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'figure.dpi': 120,
        'savefig.dpi': 200,
        'font.size': 8,
        'axes.titlesize': 8,
        'axes.labelsize': 8,
        'legend.fontsize': 7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.25,
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.2,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })
    return plt


def method_colours(methods: Iterable[str]) -> Dict[str, str]:
    palette = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd',
               '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    ordered = sorted(set(methods), key=order_key)
    return {method: palette[i % len(palette)] for i, method in enumerate(ordered)}
