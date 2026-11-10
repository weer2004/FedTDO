#!/usr/bin/env python
"""Paper figures for the FL comparison: per-epoch curves + communication cost.

Figure set
----------
``fig1_convergence_<dataset>.png``  one panel per dataset, FedTDO + best baseline
                                    highlighted, the remaining baselines in grey
``fig1_overview.png``               2 x 4 small multiples of the same data
``fig2_comm_overhead_<dataset>.png`` cumulative communication overhead vs epoch

``--top-k N`` switches figure 1 and 2 to *focus mode*: instead of grey-ing out
everything except the best baseline, only FedTDO + the N baselines with the lowest
final loss + FedAvg are drawn, each with its own colour and marker (readable on a
single page and in greyscale print).  This is the mode used for the paper's
"FedTDO vs the strongest FL baselines" figure.

Two caveats that the focus-mode panels state in the caption:
* FedTDO's per-epoch history is scored on the **test** split (``--fedtdo_eval_split
  test``), the baselines' on the **validation** split (the only split the round
  logger scores).  The y-axis is therefore just ``MSE``, not "test MSE".
* FedTDO trains 5 warm-up epochs before the downstream stage whose history is
  plotted, so its x coordinates are shifted by ``warmup_epochs`` to make "epoch"
  mean the same thing on both sides.  Clients early-stop at different depths; the
  curve is truncated to the shortest client so every point averages all clients.

Where the numbers come from
---------------------------
* **baselines** -- ``results_baselines/_epoch_logs/BL_<method>_fl_baseline_<ds>_*``,
  lines ``[<method>] round r/R epoch e/E lr=... val=<mse>``.  Only the validation
  split is scored per round; the test split is evaluated once after training
  (``exp/exp_fl_baselines.py``), so the curve is labelled *validation* MSE.
* **FedTDO** -- ``results/fedtdo_curve|fedtdo_full/fedtdo_ch_<ds>_*/metrics.json``
  -> ``train.per_client[i].val_history`` (per-epoch validation MSE).  Runs made
  before that field existed only saved ``best_val``; ``exp/exp_fedtdo.py`` now
  records the full history, which is what makes FedTDO plottable here.
* **wire payload** -- measured by building the same Dynamic-TMoE the runs train
  and summing ``state_dict()`` elements, i.e. the definition used by
  ``scripts/xc_cost_tables.py``: bytes on the wire, not ``params * 4``.

Usage
-----
    python scripts/xc_plot_fl_curves.py
    python scripts/xc_plot_fl_curves.py --datasets ETTh1 --no-comm
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BL_ROOT = ROOT / 'results_baselines'
LOG_DIR = BL_ROOT / '_epoch_logs'
FEDTDO_DIRS = [ROOT / 'results' / 'fedtdo_curve', ROOT / 'results' / 'fedtdo_full']
OUT_DIR = ROOT / 'results_paper' / 'tables' / 'figs'
CURVE_CSV = ROOT / 'results_paper' / 'tables' / 'fl_curves.csv'

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from lr_sweep import DATASETS as DATA_PRESET, OFFICIAL  # noqa: E402

ORDER = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Exchange', 'ILI', 'ECL', 'Traffic']
COMM_DATASET = 'ETTh1'
OURS = 'FedTDO'

NAME_RE = re.compile(
    r'^BL_(?P<method>.+?)_fl_baseline_(?P<dataset>.+?)_nc(?P<nc>\d+)'
    r'_sl(?P<sl>\d+)_pl(?P<pl>\d+)_dm(?P<dm>\d+)_r(?P<rounds>\d+)_ep(?P<epochs>\d+)'
    r'(?:_s(?P<seed>\d+))?')
FEDTDO_RE = re.compile(
    r'^fedtdo_ch_(?P<dataset>[A-Za-z0-9]+)_nc(?P<nc>\d+)_sl(?P<sl>\d+)'
    r'_pl(?P<pl>\d+)_dm(?P<dm>\d+)_ep(?P<epochs>\d+)(?:_s(?P<seed>\d+))?')
VAL_RE = re.compile(
    r'^\[(?P<method>[^\]]+)\] round (?P<round>\d+)/(?P<rounds>\d+) '
    r'epoch (?P<epoch>\d+)/(?P<epochs>\d+) lr=(?P<lr>[-+0-9.eE]+) '
    r'val=(?P<val>[-+0-9.eE]+)')

LABELS = {
    'local': 'Local only', 'fedavg': 'FedAvg', 'fedavg_sgd': 'FedAvg (SGD)',
    'fedprox': 'FedProx', 'fedadam': 'FedAdam', 'fedala': 'FedALA',
    'fednova': 'FedNova', 'scaffold': 'SCAFFOLD', 'ditto': 'Ditto',
    'pfedme': 'pFedMe', 'fedper': 'FedPer', 'fedrep': 'FedRep',
    'pfedcs': 'pFedCS', 'pfedfda': 'pFedFDA', 'fedda': 'FedDA',
    'fedgtp': 'FedGTP', 'fedkme': 'FedKMe', 'fedkwaz': 'FedKwaz',
    'fedl2p': 'FedL2P', 'fedpkda': 'FedPKDA', 'fedpoe': 'FedPoE',
}

# communication groups, read off fl_baselines/*.py (same groups as xc_cost_tables.py)
BODY_ONLY = {'fedper', 'fedrep', 'fedpkda'}
CONTROL_VARIATE = {'scaffold'}
LOCAL_ONLY = {'local'}

C_OURS, C_BEST, C_REF, C_GREY = '#c0392b', '#1f6fb2', '#1b1b1b', '#b6bfc8'

plt.rcParams.update({
    'figure.dpi': 170,
    'savefig.dpi': 220,
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'axes.linewidth': 0.8,
    'legend.frameon': False,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
})


def pretty(method: str) -> str:
    return LABELS.get(method, method)


def tidy(ax, grid_axis: str = 'y') -> None:
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('0.4')
    ax.grid(axis=grid_axis, color='0.92', linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors='0.3', labelsize=8.5, length=3)


def subtitle(ax, text: str) -> None:
    ax.text(0.0, 1.015, text, transform=ax.transAxes, fontsize=8,
            color='0.45', ha='left', va='bottom')


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------
def parse_curve(log_path: Path) -> list:
    pts = []
    if not log_path.exists():
        return pts
    with open(log_path, 'r', errors='ignore') as f:
        for line in f:
            mt = VAL_RE.match(line)
            if mt:
                pts.append((int(mt.group('epoch')), int(mt.group('round')),
                            float(mt.group('val'))))
    pts.sort(key=lambda p: p[0])
    return pts


def load_baseline_runs() -> list:
    runs = []
    for d in sorted(BL_ROOT.glob('BL_*_fl_baseline_*')):
        mt = NAME_RE.match(d.name)
        if not mt:
            continue
        g = mt.groupdict()
        curve = parse_curve(LOG_DIR / f'{d.name}.log')
        if not curve:
            continue
        runs.append({
            'run': d.name, 'method': g['method'], 'dataset': g['dataset'],
            'nc': int(g['nc']), 'sl': int(g['sl']), 'pl': int(g['pl']),
            'dm': int(g['dm']), 'rounds': int(g['rounds']),
            'epochs': int(g['epochs']), 'seed': int(g['seed'] or 2021),
            'curve': curve,
        })
    return runs


def load_fedtdo_curves() -> dict:
    """{dataset: {curve, run, nc, pl, dm, epochs, warmup}} from metrics.json.

    Only runs made after ``exp/exp_fedtdo.py`` started recording ``val_history``
    are usable; older ones store ``best_val`` alone and are skipped (figure 1 then
    reports "no FedTDO per-epoch curve" for that dataset).
    """
    out = {}
    for root in FEDTDO_DIRS:
        if not root.exists():
            continue
        for d in sorted(root.glob('fedtdo_ch_*')):
            mt = FEDTDO_RE.match(d.name)
            mj = d / 'metrics.json'
            if not mt or not mj.exists() or mt.group('dataset') in out:
                continue
            with open(mj, encoding='utf-8') as f:
                metrics = json.load(f)
            train = metrics.get('train') or {}
            histories = [c['val_history'] for c in train.get('per_client', [])
                         if c.get('val_history')]
            if not histories:
                continue
            warmup = int(train.get('warmup_epochs', 5) or 5)
            # Clients early-stop after their own patience, so the mean over the
            # clients *still training* jumps whenever one drops out (0.344 -> 0.398
            # on ETTh1).  Average over the common support instead: every point then
            # covers all clients.
            n = min(len(h) for h in histories)
            curve = [(warmup + e + 1, e + 1,
                      float(np.mean([h[e] for h in histories])))
                     for e in range(n)]
            out[mt.group('dataset')] = {
                'run': d.name, 'nc': int(mt.group('nc')), 'sl': int(mt.group('sl')),
                'pl': int(mt.group('pl')), 'dm': int(mt.group('dm')),
                'epochs': int(mt.group('epochs')),
                'seed': int(mt.group('seed') or 2021),
                'warmup': warmup, 'n_clients': len(histories),
                'curve': curve,
            }
    return out


def protocol_of(run) -> tuple:
    return run['rounds'], run['epochs'], max(1, run['epochs'] // run['rounds'])


def pick_runs(runs: list) -> tuple:
    """One baseline run per method on the dataset's dominant aggregation protocol."""
    by_method = defaultdict(list)
    for r in runs:
        by_method[r['method']].append(r)

    support = Counter()
    for cands in by_method.values():
        for proto in {protocol_of(r) for r in cands}:
            support[proto] += 1
    proto = max(support.items(), key=lambda kv: (kv[1], -kv[0][2], kv[0][1]))[0]

    pls = [r['pl'] for r in runs if protocol_of(r) == proto]
    modal_pl = Counter(pls).most_common(1)[0][0]

    picked = {}
    for method, cands in sorted(by_method.items()):
        pool = [r for r in cands if protocol_of(r) == proto] or cands
        pool = [r for r in pool if r['pl'] == modal_pl] or pool
        pool.sort(key=lambda r: (0 if r['seed'] == 2021 else 1, r['seed'],
                                 r['dm'], -len(r['curve'])))
        picked[method] = pool[0]
    return picked, proto


# ---------------------------------------------------------------------------
# figure 1 -- convergence
# ---------------------------------------------------------------------------
def focus_methods(picked: dict) -> tuple:
    """(best baseline, number of greyed-out baselines) for the highlight scheme."""
    finals = {m: r['curve'][-1][2] for m, r in picked.items() if m != OURS}
    ranked = sorted(finals, key=finals.get)
    best = ranked[0] if ranked else None
    greyed = [m for m in picked if m not in (OURS, best, 'fedavg')]
    return best, len(greyed)


def draw_panel(ax, picked: dict, proto: tuple, show_annotation: bool = True) -> list:
    """All methods on one axes: grey baselines, then the three that matter."""
    best, _ = focus_methods(picked)
    for method, run in sorted(picked.items()):
        if method in (OURS, best, 'fedavg'):
            continue
        xs = [p[0] for p in run['curve']]
        ys = [p[2] for p in run['curve']]
        ax.plot(xs, ys, color=C_GREY, lw=0.9, alpha=0.75, zorder=1)

    for method, color, width, zorder, dash in (
            (best, C_BEST, 2.0, 3, '-'),
            ('fedavg', C_REF, 1.7, 4, (0, (5, 2))),
            (OURS, C_OURS, 2.8, 6, '-')):
        run = picked.get(method)
        if run is None:
            continue
        xs = [p[0] for p in run['curve']]
        ys = [p[2] for p in run['curve']]
        ax.plot(xs, ys, color=color, lw=width, ls=dash, zorder=zorder,
                solid_capstyle='round')
        if show_annotation:
            ax.annotate(f'{ys[-1]:.3f}', xy=(xs[-1], ys[-1]),
                        xytext=(4, 0), textcoords='offset points',
                        color=color, fontsize=7.5, va='center', fontweight='bold')
    return [best]


def legend_handles(picked: dict, best) -> list:
    _, n_grey = focus_methods(picked)
    handles = []
    if OURS in picked:
        handles.append(Line2D([], [], color=C_OURS, lw=2.8, label=f'{OURS} (ours)'))
    if best:
        handles.append(Line2D([], [], color=C_BEST, lw=2.0, label=f'{pretty(best)} (best baseline)'))
    if 'fedavg' in picked and best != 'fedavg':
        handles.append(Line2D([], [], color=C_REF, lw=1.7, ls=(0, (5, 2)), label='FedAvg'))
    if n_grey:
        handles.append(Line2D([], [], color=C_GREY, lw=0.9, alpha=0.9,
                              label=f'other baselines ({n_grey})'))
    return handles


def plot_convergence(dataset, picked, proto, out_dir, logy=None) -> Path:
    rounds, epochs, local = proto
    ref = picked[OURS] if OURS in picked else picked[next(iter(picked))]

    fig, ax = plt.subplots(figsize=(5.4, 3.9))
    draw_panel(ax, picked, proto)
    tidy(ax)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation MSE  (lower is better)')

    vals = np.array([p[2] for r in picked.values() for p in r['curve']])
    use_log = logy if logy is not None else (vals.max() / max(vals.min(), 1e-12) > 15)
    if use_log:
        ax.set_yscale('log')
        ax.set_ylabel('Validation MSE  (log scale, lower is better)')

    ax.set_title(dataset, fontsize=13, fontweight='bold', loc='left', pad=16)
    subtitle(ax, f'{rounds} rounds x {local} local epoch{"" if local == 1 else "s"} '
                 f'({epochs} epochs)  ·  pl={ref["pl"]}  ·  {ref["nc"]} clients  ·  seed={ref["seed"]}')
    ax.set_xlim(0, epochs)
    ax.legend(handles=legend_handles(picked, focus_methods(picked)[0]),
              loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=8,
              handlelength=1.8, labelspacing=0.5)
    fig.tight_layout()
    out = out_dir / f'fig1_convergence_{dataset}.png'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    return out


def plot_overview(data: dict, out_dir: Path) -> Path:
    """2 x N grid of every dataset, one shared legend."""
    names = [d for d in ORDER if d in data]
    cols = 4
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.05 * cols, 2.5 * rows),
                             sharex=False, sharey=False)
    axes = np.atleast_1d(axes).ravel()
    handles = []
    for ax, name in zip(axes, names):
        picked, proto = data[name]
        draw_panel(ax, picked, proto, show_annotation=False)
        tidy(ax)
        ax.set_title(name, fontsize=10.5, fontweight='bold', loc='left', pad=4)
        ax.tick_params(labelsize=7.5)
        ax.set_xlabel('Epoch', fontsize=8, labelpad=1)
        if ax is axes[0]:
            ax.set_ylabel('Validation MSE', fontsize=8)
            handles = legend_handles(picked, focus_methods(picked)[0])
    for ax in axes[len(names):]:
        ax.set_visible(False)
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.02), handlelength=1.8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = out_dir / 'fig1_overview.png'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# focus mode -- FedTDO + the strongest baselines, one colour each
# ---------------------------------------------------------------------------
FOCUS_PALETTE = ['#1f6fb2', '#2e8b57', '#e08b2f', '#7b52ab', '#00a0b0', '#8c6d31']
FOCUS_MARKERS = ['o', 's', '^', 'D', 'v', 'P']


def filter_protocol(runs: list, rounds: int = None, epochs: int = None) -> list:
    """Keep the runs of one matrix cell (same federated rounds / epoch budget)."""
    return [r for r in runs
            if (rounds is None or r['rounds'] == rounds)
            and (epochs is None or r['epochs'] == epochs)]


def select_focus(picked: dict, k: int) -> list:
    """FedTDO, then the ``k`` baselines with the lowest final loss, then FedAvg."""
    others = [m for m in picked if m != OURS]
    ranked = sorted(others, key=lambda m: (picked[m]['curve'][-1][2], m))
    chosen = [m for m in ranked if m != 'fedavg'][:max(0, k)]
    if 'fedavg' in others and 'fedavg' not in chosen:
        chosen.append('fedavg')
    return ([OURS] if OURS in picked else []) + chosen


def plot_convergence_focus(dataset, picked, proto, out_dir: Path, k: int,
                           logy=None) -> Path:
    """One panel: FedTDO + top-k baselines, distinct colour and marker each."""
    rounds, epochs, local = proto
    order = select_focus(picked, k)
    ref = picked[OURS] if OURS in picked else picked[order[0]]
    n_clients = ref.get('n_clients') or ref['nc']

    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    handles = []
    for index, method in enumerate(order):
        run = picked[method]
        xs = [p[0] for p in run['curve']]
        ys = [p[2] for p in run['curve']]
        if method == OURS:
            ax.plot(xs, ys, color=C_OURS, lw=2.8, zorder=6, solid_capstyle='round')
            # dot marks where the shortest client early-stopped (curve truncated there)
            ax.plot([xs[-1]], [ys[-1]], 'o', color=C_OURS, ms=4.5, zorder=7)
            handles.append(Line2D([], [], color=C_OURS, lw=2.8,
                                  label=f'{OURS} (ours)  {ys[-1]:.3f}'))
            continue
        color = FOCUS_PALETTE[index % len(FOCUS_PALETTE)]
        marker = FOCUS_MARKERS[index % len(FOCUS_MARKERS)]
        ax.plot(xs, ys, color=color, lw=1.6, marker=marker, ms=4.2, zorder=3,
                label=pretty(method))
        handles.append(Line2D([], [], color=color, lw=1.6, marker=marker, ms=4.2,
                              label=f'{pretty(method)}  {ys[-1]:.3f}'))

    tidy(ax)
    ax.set_xlabel('Epoch')
    use_log = False if logy is None else logy
    if use_log:
        ax.set_yscale('log')
        ax.set_ylabel('MSE  (log scale, lower is better)')
    else:
        ax.set_ylabel('MSE  (lower is better)')
    ax.set_xlim(0, epochs)
    ax.set_title(dataset, fontsize=13, fontweight='bold', loc='left', pad=16)
    subtitle(ax, f'{rounds} federated rounds x {local} local epochs '
                 f'({epochs} epochs)  ·  pl={ref["pl"]}  ·  {ref["nc"]} clients  ·  '
                 f'seed={ref["seed"]}')
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1.0),
              fontsize=8, handlelength=2.0, labelspacing=0.45,
              title='final MSE', title_fontsize=7.5)
    if OURS in picked:
        note = (f'{OURS}: test split, {ref.get("warmup", 5)}-epoch warm-up offset, '
                f'mean over {n_clients} clients (common support)  ·  '
                f'baselines: validation split, one point per {local}-epoch round\n'
                'both are static-split losses, not the online-stream MSE reported in the tables')
    else:
        note = (f'validation-split MSE, one point per {local}-epoch round  ·  '
                f'{OURS} not plotted: its run predates the per-epoch history, re-run it '
                'to add the curve')
    fig.text(0.012, -0.015, note, fontsize=7, color='0.45', ha='left', va='top')
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f'fig1_convergence_{dataset}.png'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# figure 2 -- cumulative communication overhead
# ---------------------------------------------------------------------------
_MEASURE_CACHE: dict = {}


def measure_payload(dataset: str, pred_len: int, d_model: int) -> dict:
    key = (dataset, pred_len, d_model)
    if key in _MEASURE_CACHE:
        return _MEASURE_CACHE[key]

    import torch
    from run_baselines import build_parser
    from models.zoo import build_model
    from federated.server import module_slots

    cfg = OFFICIAL.get(dataset, {}).get(pred_len)
    args = build_parser().parse_args([])
    args.data = dataset
    args.seq_len = int(DATA_PRESET.get(dataset, {}).get('seq_len', 96))
    args.pred_len = pred_len
    args.d_model = d_model
    args.enc_in = args.dec_in = args.c_out = 1
    args.freq = DATA_PRESET.get(dataset, {}).get('freq', 'h')
    if cfg is not None:
        _, args.dropout, args.patch_len, args.stride = cfg[:4]

    torch.set_num_threads(1)
    with torch.no_grad():
        model = build_model('dtmoe', args)
        sd = model.state_dict()
        state = sum(v.numel() for v in sd.values())
        params = sum(p.numel() for p in model.parameters())
        head = sum(v.numel() for k, v in sd.items() if k.startswith('projection_layer'))
        shared = 0
        for name in ('patch_embedding', 'temporal_moe_layers', 'projection'):
            for slot in module_slots(model, name):
                shared += sum(v.numel() for v in slot.state_dict().values())

    out = {'state_elems': int(state), 'params_elems': int(params),
           'head_elems': int(head), 'body_elems': int(state - head),
           'shared_elems': int(shared)}
    _MEASURE_CACHE[key] = out
    return out


def wire_elems(method: str, meas: dict) -> tuple:
    """(upload, download) elements per client per round."""
    if method in LOCAL_ONLY:
        return 0, 0
    if method in BODY_ONLY:
        return meas['body_elems'], meas['body_elems']
    if method in CONTROL_VARIATE:
        n = meas['state_elems'] + meas['params_elems']
        return n, n
    return meas['state_elems'], meas['state_elems']


def plot_comm(dataset, picked, proto, fedtdo, out_dir) -> Path:
    rounds, epochs, local = proto
    ref = picked[next(iter(picked))]
    meas = measure_payload(dataset, ref['pl'], ref['dm'])
    nc = ref['nc']

    totals, curves = {}, {}
    for method, run in picked.items():
        up, down = wire_elems(method, meas)
        per_round_kb = nc * (up + down) * 4 / 1024.0
        totals[method] = per_round_kb * rounds
        curves[method] = ([(p[1], p[1] * per_round_kb) for p in run['curve']],
                          per_round_kb)
    totals['local'] = 0.0 if 'local' not in totals else totals['local']

    fig, ax = plt.subplots(figsize=(5.6, 3.9))

    # FedTDO pays one encoder sync for the whole offline stage: it never averages
    # the model (exp/exp_fedtdo.py freezes the shared modules when sync_every<=0).
    fedtdo_kb = None
    if fedtdo is not None:
        fedtdo_kb = fedtdo['nc'] * 2 * meas['shared_elems'] * 4 / 1024.0
        warm = fedtdo.get('warmup', 5)
        ax.plot([0, warm, warm, epochs], [0, 0, fedtdo_kb, fedtdo_kb],
                color=C_OURS, lw=2.8, zorder=6, solid_capstyle='round')

    # group methods that pay exactly the same bytes into a single line
    groups = defaultdict(list)
    for method, total in totals.items():
        groups[round(total, 1)].append(method)

    ordered = sorted(groups.items(), key=lambda kv: -kv[0])
    for total, members in ordered:
        members = sorted(members)
        head = members[0]
        if total == 0:
            ax.axhline(0, color='0.45', lw=1.4, ls=':', zorder=2)
            continue
        curve, per_round = curves[head]
        xs = [p[0] for p in curve] or [0, epochs]
        ys = [p[1] for p in curve] or [0, total]
        emphasis = 'fedavg' in members or 'scaffold' in members
        color = C_REF if 'fedavg' in members else ('#e08b2f' if 'scaffold' in members else C_GREY)
        ax.plot(xs, ys, color=color, lw=2.0 if emphasis else 0.9,
                alpha=1.0 if emphasis else 0.75,
                ls=(0, (5, 2)) if 'fedavg' in members else '-',
                drawstyle='steps-post', zorder=4 if emphasis else 2)

    tidy(ax)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cumulative communication overhead (KB)')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}'))
    ax.set_xlim(0, epochs)
    ax.set_title(dataset, fontsize=13, fontweight='bold', loc='left', pad=16)
    subtitle(ax, f'{rounds} rounds  ·  {nc} clients  ·  {local} local '
                 f'epoch{"" if local == 1 else "s"}/round  ·  no compression')

    handles = []
    if fedtdo_kb is not None:
        handles.append(Line2D([], [], color=C_OURS, lw=2.8,
                              label=f'{OURS}: 1 encoder sync ({fedtdo_kb / 1024:.1f} MB)'))
    for total, members in ordered:
        if total == 0:
            handles.append(Line2D([], [], color='0.45', lw=1.4, ls=':',
                                  label='Local only (0 MB)'))
            continue
        name = pretty(sorted(members)[0])
        extra = f' +{len(members) - 1}' if len(members) > 1 else ''
        color = C_REF if 'fedavg' in members else ('#e08b2f' if 'scaffold' in members else C_GREY)
        handles.append(Line2D([], [], color=color, lw=1.8,
                              ls=(0, (5, 2)) if 'fedavg' in members else '-',
                              label=f'{name}{extra}: {total / 1024:.1f} MB'))
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1.0),
              fontsize=8, handlelength=1.8, labelspacing=0.5)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f'fig2_comm_overhead_{dataset}.png'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    return out


def plot_comm_focus(dataset, picked, proto, fedtdo, out_dir: Path, k: int) -> Path:
    """Focus-mode figure 2: same method set as figure 1, KB on the wire vs epoch."""
    rounds, epochs, local = proto
    order = select_focus(picked, k)
    ref = picked[OURS] if OURS in picked else picked[order[0]]
    meas = measure_payload(dataset, ref['pl'], ref['dm'])
    nc = ref['nc']

    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    handles = []

    # Every baseline that ships the whole model pays exactly the same bytes, so
    # identical totals are collapsed into one line instead of five coincident ones.
    groups = defaultdict(list)
    for method in order:
        if method == OURS:
            # FedTDO's one-shot sync is drawn below; it has no per-round traffic.
            continue
        up, down = wire_elems(method, meas)
        groups[round(rounds * nc * (up + down) * 4 / 1024.0, 1)].append(method)

    for index, (total_kb, members) in enumerate(
            sorted(groups.items(), key=lambda kv: -kv[0])):
        run = picked[sorted(members)[0]]
        per_round_kb = total_kb / rounds
        xs = [p[0] for p in run['curve']]
        ys = [p[1] * per_round_kb for p in run['curve']]
        color = FOCUS_PALETTE[index % len(FOCUS_PALETTE)]
        marker = FOCUS_MARKERS[index % len(FOCUS_MARKERS)]
        ax.plot(xs, ys, color=color, lw=1.6, marker=marker, ms=4.2,
                drawstyle='steps-post', zorder=3)
        if len(members) == 1:
            name = pretty(members[0])
        elif 'fedavg' in members:
            name = f'FedAvg + {len(members) - 1} other baselines'
        else:
            name = f'{len(members)} baselines (identical payload)'
        handles.append(Line2D([], [], color=color, lw=1.6, marker=marker, ms=4.2,
                              drawstyle='steps-post',
                              label=f'{name}  {total_kb / 1024:.1f} MB'))

    # FedTDO syncs the frozen encoder once and never averages the model, so its
    # overhead is a single step at the end of warm-up and flat afterwards.
    if fedtdo is not None:
        kb = nc * 2 * meas['shared_elems'] * 4 / 1024.0
        warm = fedtdo.get('warmup', 5)
        ax.plot([0, warm, warm, epochs], [0, 0, kb, kb], color=C_OURS, lw=2.8,
                zorder=6, solid_capstyle='round')
        handles.insert(0, Line2D([], [], color=C_OURS, lw=2.8,
                                 label=f'{OURS} (ours)  {kb / 1024:.1f} MB'))

    tidy(ax)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cumulative communication overhead (KB)')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}'))
    ax.set_xlim(0, epochs)
    ax.set_title(dataset, fontsize=13, fontweight='bold', loc='left', pad=16)
    subtitle(ax, f'{rounds} rounds x {local} local epochs  ·  {nc} clients  ·  '
                 f'no compression  ·  total per method in the legend')
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1.0),
              fontsize=8, handlelength=2.0, labelspacing=0.45,
              title='total traffic', title_fontsize=7.5)
    fig.text(0.012, -0.015,
             'bytes = state_dict elements on the wire (upload + download per client) '
             'on the same Dynamic-TMoE the runs train\n'
             'no compression; state_dict() re-serialises the aliased expert pool and the '
             'constant PE table, so absolute MB ≈ 2x params (same on both sides)',
             fontsize=7, color='0.45', ha='left', va='top')
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f'fig2_comm_overhead_{dataset}.png'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--datasets', nargs='+', default=None, choices=ORDER)
    ap.add_argument('--min-methods', type=int, default=2)
    ap.add_argument('--out-dir', type=Path, default=OUT_DIR)
    ap.add_argument('--no-comm', action='store_true')
    ap.add_argument('--no-overview', action='store_true')
    ap.add_argument('--comm-dataset', default=COMM_DATASET, choices=ORDER)
    ap.add_argument('--logy', action='store_true')
    ap.add_argument('--top-k', type=int, default=0,
                    help='focus mode: draw FedTDO + the k best baselines + FedAvg')
    ap.add_argument('--rounds', type=int, default=None,
                    help='focus mode: keep only runs with this many FL rounds '
                         '(default 5 when --top-k is set)')
    ap.add_argument('--total-epochs', type=int, default=None,
                    help='focus mode: keep only runs with this epoch budget '
                         '(default 60 when --top-k is set)')
    args = ap.parse_args()

    focus = args.top_k > 0
    rounds_keep = args.rounds if args.rounds is not None else (5 if focus else None)
    epochs_keep = args.total_epochs if args.total_epochs is not None else (60 if focus else None)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    by_ds = defaultdict(list)
    for r in filter_protocol(load_baseline_runs(), rounds_keep, epochs_keep):
        by_ds[r['dataset']].append(r)
    fedtdo = load_fedtdo_curves()

    data, csv_rows = {}, []
    for dataset in (args.datasets or [d for d in ORDER if d in by_ds]):
        picked, proto = pick_runs(by_ds.get(dataset, []))
        if dataset in fedtdo:
            picked[OURS] = fedtdo[dataset]
        else:
            print(f'[warn] {dataset}: no FedTDO per-epoch curve '
                  f'(re-run with exp/exp_fedtdo.py val_history)')
        if len(picked) < args.min_methods:
            print(f'[skip] {dataset}: {len(picked)} method(s) only')
            continue
        data[dataset] = (picked, proto)
        if focus:
            out = plot_convergence_focus(dataset, picked, proto, args.out_dir,
                                         args.top_k, args.logy or None)
        else:
            out = plot_convergence(dataset, picked, proto, args.out_dir, args.logy or None)
        rounds, epochs, local = proto
        print(f'[fig1] {out.name}  methods={len(picked)}  '
              f'protocol={rounds}r x {local}e  FedTDO={"yes" if OURS in picked else "no"}')
        for method, run in sorted(picked.items()):
            for epoch, rnd, val in run['curve']:
                csv_rows.append({'dataset': dataset, 'method': method, 'epoch': epoch,
                                 'round': rnd, 'val_mse': val, 'run': run['run']})

    if data and not args.no_overview and not focus:
        print(f'[fig1] {plot_overview(data, args.out_dir).name}')

    if not args.no_comm:
        picked, proto = pick_runs(by_ds.get(args.comm_dataset, []))
        if focus and args.comm_dataset in fedtdo:
            picked[OURS] = fedtdo[args.comm_dataset]
        if len(picked) >= 2:
            if focus:
                out = plot_comm_focus(args.comm_dataset, picked, proto,
                                      fedtdo.get(args.comm_dataset), args.out_dir,
                                      args.top_k)
            else:
                out = plot_comm(args.comm_dataset, picked, proto,
                                fedtdo.get(args.comm_dataset), args.out_dir)
            print(f'[fig2] {out.name}  methods={len(picked)}  '
                  f'FedTDO={"yes" if args.comm_dataset in fedtdo else "no"}')
        else:
            print(f'[skip] figure 2: {args.comm_dataset} has {len(picked)} method(s)')

    if csv_rows:
        with open(CURVE_CSV, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=['dataset', 'method', 'epoch', 'round',
                                              'val_mse', 'run'])
            w.writeheader()
            w.writerows(csv_rows)
        print(f'[csv ] {CURVE_CSV.relative_to(ROOT)}  rows={len(csv_rows)}')


if __name__ == '__main__':
    main()
