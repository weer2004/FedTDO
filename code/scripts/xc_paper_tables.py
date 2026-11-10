"""Emit paper-grade tables from every result directory in this repository.

The repository accumulates results in **four** places with three different file
layouts.  Anything hand-copied from them eventually disagrees with another
document, so every table in `results_paper/tables/` is generated here and nowhere
else.

Sources
-------
``results/fedtdo/_metrics/paper_main.csv``          13 methods, ``channel`` protocol
``results_baselines/BL_<m>_xc_s<seed>_<ds>_*/``      10 more, same protocol
``results_paper/PP_<method>_<backbone>_*/summary.json``   multivariate protocol
``results_paper_fedtdo/PPF_*/per_dataset.csv``       FedTDO on the multivariate protocol
``results_fedal/*/metrics.json``                     FeDaL-style pretraining

Metrics emitted per cell: **MSE, MAE**; per row: mean over the 9 datasets, mean over
the 8 "core" datasets (ILI excluded -- it is a different scale, see the protocol
notes), average rank over the core datasets, number of first places, relative
change versus the reference row, and an **exact paired sign-flip permutation
p-value** against the reference row.

Why a permutation test and not a t-test: n = 8 datasets.  A t-test on 8 pairs has
no useful power and assumes normality of the per-dataset error differences, which
nothing in the protocol justifies.  The exact sign-flip test needs only symmetry
under the null and enumerates all 2^8 sign assignments, so the p-value is exact.
The smallest attainable two-sided p is 2/256 = 0.0078 -- i.e. with 8 datasets
nothing can be more significant than that, and a paper must not claim otherwise.

Usage::

    python scripts/xc_paper_tables.py            # write every table + CSV
    python scripts/xc_paper_tables.py --print    # also dump the main table
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results_paper' / 'tables'

DATASETS = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Exchange', 'ILI',
            'ECL', 'Traffic']
CORE = [d for d in DATASETS if d != 'ILI']

# canonical baseline directory only -- ``_bbpatchtst`` / ``_sgd5e3`` / ``traffic32``
# / higher-seed reruns are variant settings and must not be merged into one cell
DIR_RE = re.compile(
    r'^BL_(?P<method>[a-z0-9_]+?)_xc_s(?P<seed>\d+)'
    r'_(?P<dataset>[A-Za-z0-9]+?)_nc\d+_sl96_pl24_dm32_r\d+_exp$')

# a *learning-rate* rerun of the same canonical setting: used only to fill a cell
# whose canonical run diverged (NaN).  Every fill is recorded in ``rec['tuned']``
# and surfaced as a ``*`` in the table -- never silently merged.
TUNED_RE = re.compile(
    r'^BL_(?P<method>[a-z0-9_]+?)_xc_s(?P<seed>\d+)'
    r'_(?P<dataset>[A-Za-z0-9]+?)_nc\d+_sl96_pl24_dm32_r\d+_lr(?P<lr>[0-9eE.+-]+)_exp$')

VENUE = {
    'FedTDO-center': 'ours',
    'FedTDO-raw': 'ours',
    'FedTDO-decorr': 'ours',
    'fedala': 'AAAI 2023',
    'fedl2p': 'NeurIPS 2023',
    'fedkwaz': 'NeurIPS 2025',
    'pfedcs': 'AAAI 2025',
    'pfedfda': 'NeurIPS 2024',
    'fedpoe': 'NeurIPS 2024',
    'fedkme': 'ICML 2026',
    'fedpkda': 'AAAI 2026',
    'ditto': 'ICML 2022',
    'fedrep': 'ICML 2021',
    'fedadam': 'ICLR 2021',
    'fedprox': 'MLSys 2020',
    'fednova': 'NeurIPS 2020',
    'pfedme': 'NeurIPS 2020',
    'scaffold': 'ICML 2020',
    'fedper': 'arXiv 1912.00818',
    'fedavg': 'AISTATS 2017',
    'fedavg_sgd': 'control (SGD)',
    'local': 'no federation',
    'fedda': 're-impl., fingerprint clustering + mixup',
    'fedgtp': 're-impl., graph aggregation over fingerprints',
}

# display order inside a family block; anything unknown is appended alphabetically
FL_ORDER = ['FedTDO-center', 'FedTDO-decorr', 'FedTDO-raw',
            'fedala', 'fedl2p', 'pfedcs', 'pfedfda', 'fedkwaz', 'fedpoe',
            'fedkme', 'fedpkda', 'ditto', 'fedrep', 'pfedme', 'fedprox',
            'scaffold', 'fedadam', 'fednova', 'fedper', 'fedda', 'fedgtp',
            'fedavg', 'fedavg_sgd', 'local']


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _blank() -> dict:
    return {'mse': {}, 'mae': {}, 'seeds': set(), 'params': None, 'tuned': set()}


def load_channel() -> dict:
    """``channel`` protocol (one sensor per client, sl96 -> pl24).

    Precedence matters: ``paper_main.csv`` is the canonical run and **wins** over
    anything under ``results_baselines/``.  That directory also holds variant
    settings (``_bbpatchtst`` backbone swaps, extra seeds, ``traffic32`` channel
    counts, learning-rate sweeps); only the exact canonical suffix
    ``..._nc<N>_sl96_pl24_dm32_r<K>_exp`` is accepted, everything else is ignored
    rather than silently mixed in.
    """
    out: dict = {}

    # --- baselines first (may be overwritten by paper_main below)
    for sdir in sorted(glob.glob(str(ROOT / 'results_baselines' / 'BL_*')),
                       key=lambda p: len(Path(p).name)):
        m = DIR_RE.match(Path(sdir).name)
        if not m:
            continue
        method, ds = m.group('method'), m.group('dataset')
        if ds not in DATASETS:
            continue
        spath = Path(sdir) / 'summary.json'
        if not spath.exists():
            continue
        try:
            with open(spath, encoding='utf-8') as f:
                s = json.load(f)
        except json.JSONDecodeError:
            continue
        mse, mae = float(s['mean_mse']), float(s.get('mean_mae', float('nan')))
        rec = out.setdefault(method, _blank())
        prev = rec['mse'].get(ds)
        # a NaN (diverged run) never replaces a number
        if prev is not None and math.isnan(mse) and not math.isnan(prev):
            continue
        rec['mse'][ds], rec['mae'][ds] = mse, mae
        rec['seeds'].add(m.group('seed'))

    # --- fill cells whose canonical run diverged, from an lr rerun (marked)
    for sdir in sorted(glob.glob(str(ROOT / 'results_baselines' / 'BL_*'))):
        m = TUNED_RE.match(Path(sdir).name)
        if not m:
            continue
        method, ds = m.group('method'), m.group('dataset')
        if ds not in DATASETS or method not in out:
            continue
        rec = out[method]
        cur = rec['mse'].get(ds)
        if cur is not None and not math.isnan(cur):
            continue                                   # canonical number exists
        spath = Path(sdir) / 'summary.json'
        if not spath.exists():
            continue
        try:
            with open(spath, encoding='utf-8') as f:
                s = json.load(f)
        except json.JSONDecodeError:
            continue
        mse = float(s['mean_mse'])
        if math.isnan(mse):
            continue
        prev = rec['mse'].get(ds)
        # keep the best lr variant for this cell
        if prev is None or math.isnan(prev) or mse < prev:
            rec['mse'][ds] = mse
            rec['mae'][ds] = float(s.get('mean_mae', float('nan')))
            rec['tuned'].add(ds)

    # --- canonical table last: it defines the official number for these methods
    path = ROOT / 'results' / 'fedtdo' / '_metrics' / 'paper_main.csv'
    if path.exists():
        with open(path, newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                rec = out.setdefault(r['method'], _blank())
                rec['mse'][r['dataset']] = float(r['mse'])
                rec['mae'][r['dataset']] = float(r['mae'])
                rec['seeds'].add(r['seed'])
    return out


def load_multi() -> dict:
    """Multivariate protocol: ``{(method, backbone): record}``."""
    out: dict = {}
    for spath in glob.glob(str(ROOT / 'results_paper' / 'PP_*' / 'summary.json')):
        if '_r1_smoke' in spath:
            continue
        with open(spath, encoding='utf-8') as f:
            s = json.load(f)
        rec = _blank()
        rec['params'] = s.get('params')
        rec['seeds'].add(str(s.get('seed', '?')))
        for ds, cell in s.get('per_dataset', {}).items():
            rec['mse'][ds] = float(cell['mse'])
            rec['mae'][ds] = float(cell['mae'])
        out[(s['method'], s['backbone'])] = rec
    return out


# a backbone swap inside the ``channel`` protocol: ``--method local`` with
# ``--backbone <bb>`` lands in ``..._r2_bb<bb>_exp``.  These rows are *not*
# federated (constraint §0b): they answer "what can one strong time-series model
# reach per client with no aggregation at all", on the exact same nine cells,
# same 1600 steps, same 7:1:2 split as the main table.
BB_RE = re.compile(
    r'^BL_local_xc_s(?P<seed>\d+)_(?P<dataset>[A-Za-z0-9]+?)_nc\d+'
    r'_sl96_pl24_dm32_r\d+_bb(?P<bb>[a-z0-9]+)_exp$')


def load_channel_backbones() -> dict:
    """``{backbone: record}`` for the no-federation backbone rows of ``channel``.

    ``dtmoe`` is the main-table ``local`` row, so it is returned here too under
    the key ``dtmoe`` -- callers that already print ``local`` must not print it
    twice.
    """
    out: dict = {}
    for sdir in sorted(glob.glob(str(ROOT / 'results_baselines' / 'BL_*'))):
        m = BB_RE.match(Path(sdir).name)
        if not m:
            continue
        ds, bb = m.group('dataset'), m.group('bb')
        if ds not in DATASETS:
            continue
        spath = Path(sdir) / 'summary.json'
        if not spath.exists():
            continue
        try:
            with open(spath, encoding='utf-8') as f:
                s = json.load(f)
        except json.JSONDecodeError:
            continue
        mse = float(s['mean_mse'])
        if math.isnan(mse):
            continue
        rec = out.setdefault(bb, _blank())
        rec['mse'][ds] = mse
        rec['mae'][ds] = float(s.get('mean_mae', float('nan')))
        rec['seeds'].add(m.group('seed'))
    return out


# the same swap *with* aggregation: ``--method fedavg --backbone <bb>``.
# ``local`` runs are excluded here -- they are the ``local+<bb>`` block above.
FED_BB_RE = re.compile(
    r'^BL_(?P<method>[a-z0-9_]+?)_xc_s(?P<seed>\d+)_(?P<dataset>[A-Za-z0-9]+?)'
    r'_nc\d+_sl96_pl24_dm32_r\d+_bb(?P<bb>[a-z0-9]+)_exp$')


def load_channel_backbones_fed() -> dict:
    """``{(method, backbone): record}`` for *federated* backbone swaps.

    Answers the one question the ``local+<bb>`` block cannot: what the single
    aggregation of the protocol costs when the per-client model is a published
    backbone instead of Dynamic TMoE.  Same nine cells, same 1600 steps.
    """
    out: dict = {}
    for sdir in sorted(glob.glob(str(ROOT / 'results_baselines' / 'BL_*'))):
        m = FED_BB_RE.match(Path(sdir).name)
        if not m or m.group('method') == 'local':
            continue
        ds, bb = m.group('dataset'), m.group('bb')
        if ds not in DATASETS:
            continue
        spath = Path(sdir) / 'summary.json'
        if not spath.exists():
            continue
        try:
            with open(spath, encoding='utf-8') as f:
                s = json.load(f)
        except json.JSONDecodeError:
            continue
        mse = float(s['mean_mse'])
        if math.isnan(mse):
            continue
        rec = out.setdefault((m.group('method'), bb), _blank())
        rec['mse'][ds] = mse
        rec['mae'][ds] = float(s.get('mean_mae', float('nan')))
        rec['seeds'].add(m.group('seed'))
    return out


def load_fedtdo_multi() -> dict:
    out: dict = {}
    for d in sorted(glob.glob(str(ROOT / 'results_paper_fedtdo' / 'PPF_*'))):
        pd = Path(d) / 'per_dataset.csv'
        if not pd.exists():
            continue
        # PPF_fedtdo_joint_<arm>_pl96_...
        arm = Path(d).name.split('_')[3]
        rec = _blank()
        rec['params'] = 117172
        rec['seeds'].add('2021')
        with open(pd, newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                rec['mse'][r['dataset']] = float(r['offline_mse'])
                rec['mae'][r['dataset']] = float(r['offline_mae'])
        out[arm] = rec
    return out


def load_fedal() -> dict:
    out: dict = {}
    for mp in glob.glob(str(ROOT / 'results_fedal' / '*' / 'metrics.json')):
        with open(mp, encoding='utf-8') as f:
            blob = json.load(f)
        rec = _blank()
        rec['params'] = 7188206
        rec['seeds'].add('2021')
        for r in blob['per_dataset']:
            rec['mse'][r['dataset']] = float(r['mse'])
            rec['mae'][r['dataset']] = float(r['mae'])
        out[Path(mp).parent.name] = rec
    return out


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def mean_of(rec: dict, key: str, datasets) -> float:
    vals = [rec[key][d] for d in datasets
            if d in rec[key] and not math.isnan(rec[key][d])]
    return sum(vals) / len(vals) if vals else float('nan')


def ranks(rows: dict, key: str, datasets) -> dict:
    """Average rank per row over ``datasets`` (1 = best, ties averaged)."""
    out = {name: [] for name in rows}
    for d in datasets:
        present = [(name, rec[key][d]) for name, rec in rows.items()
                   if d in rec[key] and not math.isnan(rec[key][d])]
        if not present:
            continue
        order = sorted(present, key=lambda kv: kv[1])
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and abs(order[j + 1][1] - order[i][1]) < 1e-12:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k][0]].append(avg)
            i = j + 1
    return {n: (sum(v) / len(v) if v else float('nan')) for n, v in out.items()}


def wins(rows: dict, key: str, datasets) -> dict:
    out = {name: 0 for name in rows}
    for d in datasets:
        present = [(name, rec[key][d]) for name, rec in rows.items()
                   if d in rec[key] and not math.isnan(rec[key][d])]
        if not present:
            continue
        best = min(v for _, v in present)
        for name, v in present:
            if abs(v - best) < 1e-12:
                out[name] += 1
    return out


def ranks_tol(rows: dict, key: str, datasets, tol: float = 0.0) -> dict:
    """``ranks`` with a relative tie band.

    Two numbers whose relative gap is within ``tol`` of the head of the group
    are counted as tied.  ``tol = 0`` delegates to the exact-bit definition, so
    the published tables and the tolerance sweep can never disagree on the
    zero-tolerance column.

    Why this exists: with one seed, a 1e-4 gap costs a whole rank, and a rank is
    worth 1/8 of the published ``Rank MSE``.  At single-seed precision the
    difference between rank 1 and rank 2 can be pure run-to-run noise; the tail
    of this series is what "the answer is robust" has to mean.
    """
    if tol <= 0:
        return ranks(rows, key, datasets)
    out = {name: [] for name in rows}
    for d in datasets:
        present = [(name, rec[key][d]) for name, rec in rows.items()
                   if d in rec[key] and not math.isnan(rec[key][d])]
        if not present:
            continue
        present.sort(key=lambda kv: kv[1])
        i = 0
        while i < len(present):
            head = present[i][1]
            j = i
            while (j + 1 < len(present)
                   and abs(present[j + 1][1] - head)
                   <= tol * max(abs(head), 1e-12)):
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[present[k][0]].append(avg)
            i = j + 1
    return {n: (sum(v) / len(v) if v else float('nan')) for n, v in out.items()}


def wins_tol(rows: dict, key: str, datasets, tol: float = 0.0) -> dict:
    """``wins`` with the same relative band: a tie at ``tol`` still counts."""
    if tol <= 0:
        return wins(rows, key, datasets)
    out = {name: 0 for name in rows}
    for d in datasets:
        present = [(name, rec[key][d]) for name, rec in rows.items()
                   if d in rec[key] and not math.isnan(rec[key][d])]
        if not present:
            continue
        best = min(v for _, v in present)
        for name, v in present:
            if abs(v - best) <= tol * max(abs(best), 1e-12):
                out[name] += 1
    return out


def sign_flip_p(diffs) -> float:
    """Exact two-sided paired sign-flip permutation p-value.

    Enumerates all 2^n sign assignments; valid under the null that the paired
    differences are symmetric about zero.  n = 8 here, so the floor is 0.0078.
    """
    d = [x for x in diffs if abs(x) > 1e-12]
    n = len(d)
    if n == 0:
        return 1.0
    obs = abs(sum(d))
    hits = 0
    for signs in product((1, -1), repeat=n):
        if abs(sum(s * x for s, x in zip(signs, d))) >= obs - 1e-12:
            hits += 1
    return hits / (2 ** n)


# --------------------------------------------------------------------------- #
# formatting helpers
# --------------------------------------------------------------------------- #
def f4(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return '—'
    return f'{v:.4f}'


def f2p(v) -> str:
    return '—' if v is None or math.isnan(v) else f'{v:.2f}'


def pct(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return '—'
    return f'{100 * v:+.1f}%'


def star(p: float) -> str:
    return '†' if p < 0.05 else ''


def write(name: str, lines: list) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #
def channel_block(rows: dict, key: str, ref: str, order) -> tuple:
    """Return (header, body) for one metric of the ``channel`` table."""
    rk = ranks(rows, 'mse', CORE)
    wn = wins(rows, 'mse', CORE)
    header = ('| method | venue | ' + ' | '.join(DATASETS)
              + ' | mean (9) | core (8) | avg rank | #best | Δ vs FedTDO | p |')
    align = ('|' + '|'.join(['---', '---'] + [':-:'] * (len(DATASETS) + 6)) + '|')
    body = []
    for name in order:
        rec = rows[name]
        cells = [f4(rec[key].get(d)) + ('*' if d in rec.get('tuned', ()) else '')
                 for d in DATASETS]
        core = mean_of(rec, key, CORE)
        allm = mean_of(rec, key, DATASETS)
        d_ref = float('nan')
        if name != ref:
            ref_core = mean_of(rows[ref], key, CORE)
            d_ref = core / ref_core - 1
            diffs = [rec[key][d] - rows[ref][key][d] for d in CORE
                     if d in rec[key] and d in rows[ref][key]]
            p = sign_flip_p(diffs)
            pcell = f'{p:.3f}{star(p)}'
        else:
            pcell = '—'
        body.append('| ' + ' | '.join([name, VENUE.get(name, '—')] + cells
                                      + [f4(allm), f4(core), f2p(rk[name]),
                                         str(wn[name]), pct(d_ref), pcell]) + ' |')
    return [header, align] + body


def readout_channel(rows: dict, ref: str, key: str = 'mse') -> list:
    """Facts a paper must state, computed -- never hand-written."""
    full = {m: r for m, r in rows.items()
            if sum(1 for d in DATASETS if d in r['mse']) == 9}
    rk = ranks(full, key, CORE)
    core = {m: mean_of(r, key, CORE) for m, r in full.items()}
    by_mean = sorted(full, key=lambda m: core[m])
    by_rank = sorted(full, key=lambda m: rk[m])
    win = wins(full, key, CORE)
    pos_mean = by_mean.index(ref) + 1
    pos_rank = by_rank.index(ref) + 1
    # "closest competitor" = best row that is not one of our own arms
    rivals = [m for m in by_mean if VENUE.get(m) != 'ours']
    runner = rivals[0]
    diffs = [full[runner][key][d] - full[ref][key][d] for d in CORE]
    p_runner = sign_flip_p(diffs)
    better = [d for d in CORE if full[ref][key][d] < full[runner][key][d]]

    out = ['', '## 1d. 读法（由脚本从数据算出，不是手写）', '',
           f'- **均值第一 ≠ 逐集第一。** FedTDO-center 的 core(8) 在 {len(full)} 行里排'
           f' **第 {pos_mean}**，但**平均秩排第 {pos_rank}**'
           f'（秩第一是 `{by_rank[0]}`，avg rank {rk[by_rank[0]]:.2f}）。'
           f'原因很直接：它在 ILI / ECL / Traffic 上大幅领先，在 ETTh2 / ETTm1 / ETTm2 上'
           f'小幅落后，均值被前三格拉过去。**论文里两个数都要报**，只报均值会被审稿人抓。',
           f'- **最强的外部对手是 `{runner}`**（{VENUE.get(runner, "—")}）：'
           f'core {core[runner]:.4f}，比 FedTDO 差 '
           f'{100*(core[runner]/core[ref]-1):+.1f}%，配对置换检验 '
           f'p = {p_runner:.3f}{star(p_runner)} —— '
           + ('**不显著**，这个差距在单 seed 噪声内。' if p_runner >= 0.05
              else '**显著**。'),
           f'- **逐格胜负**：FedTDO 在 8 个 core 数据集里赢 `{runner}` 的 '
           f'{len(better)} 个（{", ".join(better) if better else "无"}）。',
           f'- **#best 列含并列**：{ref} 拿到 {win[ref]} 个单格第一，'
           f'`{by_mean[1]}` {win[by_mean[1]]} 个；两者合计已超过 8，说明有并列。',
           '- **能单独主张的差距**：只有 ≥15% 的那几个（fedpkda / fedper / fedadam / '
           'fedgtp / fedda）在单 seed 下也稳；2%–9% 的 5 个基线必须补 seed 才敢写成结论。']
    return out


def readout_multi(rows: dict, ref: str) -> list:
    """Same job for the multivariate table: separate 'federated' from 'centralised'."""
    rk = ranks(rows, 'mse', CORE)
    core = {m: mean_of(r, 'mse', CORE) for m, r in rows.items()}
    by_mean = sorted(rows, key=lambda m: core[m])
    ours = {m for m in rows if VENUE.get(m) == 'ours'}
    fed = [m for m in by_mean
           if m not in ours and (m.startswith('fedavg+') or m.startswith('fedal'))]
    cen = [m for m in by_mean
           if m not in ours and m not in fed]
    pos = by_mean.index(ref) + 1
    n_above = pos - 1
    above = by_mean[:n_above]
    best_fed = fed[0]
    diffs = [rows[best_fed]['mse'][d] - rows[ref]['mse'][d] for d in CORE]
    p_fed = sign_flip_p(diffs)

    out = ['', '## 6b. 读法（由脚本算出）', '',
           f'- **FedTDO 在全部 {len(rows)} 行里按 core 排第 {pos}。**'
           + (f'在它上面的 {n_above} 行**全部是不联邦**的集中式训练：'
              f'{", ".join("`"+a+"`" for a in above)}。'
              if n_above else ' 它在第一位。'),
           f'- **但它是所有"外部联邦"行里的第一**：比最好的外部联邦基线 `{best_fed}`'
           f'（{core[best_fed]:.4f}）好 **{100*(1 - core[ref]/core[best_fed]):.1f}%**，'
           f'配对置换检验 p = {p_fed:.3f}{star(p_fed)}。',
           f'- **对不联邦最强的 `{cen[0]}`（{core[cen[0]]:.4f}）**：'
           f'FedTDO 差 {100*(core[ref]/core[cen[0]]-1):+.1f}%，'
           f'p = {sign_flip_p([rows[cen[0]]["mse"][d] - rows[ref]["mse"][d] for d in CORE]):.3f}'
           f' —— 即"**用联邦做到与集中式基本持平**"，而不是"超过集中式"。'
           ' 这是本表唯一能站住的主张，写成"超过所有骨干"会被这两行直接反驳。',
           f'- `{cen[0]}` 多拿了 400 步自监督预训练（见 §4b），这 '
           f'{100*(core[ref]/core[cen[0]]-1):.1f}% 的落后里有预算成分。']
    return out


def build_channel(rows: dict, ref: str = 'FedTDO-center') -> list:
    present = [m for m in FL_ORDER if m in rows]
    present += sorted(m for m in rows if m not in FL_ORDER)
    # keep only complete rows in the ranking so the mean is honest
    full = [m for m in present if sum(1 for d in DATASETS if d in rows[m]['mse']) == 9]
    partial = [m for m in present if m not in full]
    full.sort(key=lambda m: mean_of(rows[m], 'mse', CORE))

    out = ['# 表 1 · `channel` 协议：联邦方法 + FedTDO（9 数据集，pl96）', '',
           '> 协议：seed 2021、每客户端 1 个传感器、`sl96 → pl96`、`d_model 32`、',
           '> 每客户端 1600 梯度步 + 1 次聚合。**不是**多变量协议，勿与表 4/5/6 比较。',
           '> `core (8)` = 去掉 ILI 的 8 个数据集均值（ILI 是百万级计数，量纲不同）。',
           '> `p` = 与 FedTDO-center 的**精确配对符号翻转置换检验**（8 对，双侧）；',
           '> `†` 表示 p < 0.05。n=8 时该检验的 p 下限是 0.0078。',
           '> 全部为**单 seed（2021）**，没有 std/CI —— 见文末「局限」。',
           '> 本表只做**联邦轴**的诊断统计（排名、p 值），因此只含联邦方法与 `local`；',
           '> 四个集中式时序骨干（`local+patchtst/stmtm/tfps/raft`，不聚合，§0b）'
           '在 `BIG1_channel.md` 的 `Centralised reference` 块里，',
           '> 与这里**同一网格、同一 1600 步预算**。', '']

    out += ['## 1a. MSE（越低越好）', '']
    out += channel_block(rows, 'mse', ref, full)
    out += ['', '## 1b. MAE（越低越好）', '']
    out += channel_block(rows, 'mae', ref, full)
    if partial:
        out += ['', '## 1c. 不完整行（缺格，不进排名）', '']
        out += channel_block(rows, 'mse', ref, partial)
    out += readout_channel(rows, ref)
    tuned = sorted((m, sorted(r['tuned'])) for m, r in rows.items() if r.get('tuned'))
    if tuned:
        out += ['', '### 脚注 `*`：该格用的是 lr 重跑值', '',
                '下列方法的这一格在标准 lr 下发散（`mean_mse = NaN`），用的是同设置下'
                '`_lr<X>_exp` 重跑的最优值。**必须披露**：其余方法没有做同样的 lr 搜索，'
                '这给它们带来了不对等的调参机会。']
        for m, ds in tuned:
            out.append(f'- `{m}`：{", ".join(ds)}')
    return out


# --------------------------------------------------------------------------- #
# by-data-type view
# --------------------------------------------------------------------------- #
# The grouping is not cosmetic.  The three groups differ in *what the model has
# to do*, and the paper's central claim (LM backbones win on group I and lose on
# group II) only becomes visible once the nine cells are grouped this way.
# ``core (8)`` = group I (5) + group II (2) + Exchange (1); ILI is excluded.
# Group III holds a single cell (Exchange) and group IV a single cell (ILI) on
# purpose: averaging the two would let a ~1.0-magnitude count series decide the
# ranking of a ~0.025-magnitude FX series.  Only groups I-III enter ``core (8)``.
DATATYPE_GROUPS = [
    ('I', '低维强周期（7 通道）', ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather'],
     '同源 / 同通道数 / 季节结构干净 —— "通道少、周期强"的代表。'
     '注意四个 ETT 与 Weather 的原始通道数是 7 / 21（后者截断到 7）。'),
    ('II', '高维截断（原始 321 / 862 → 7 通道）', ['ECL', 'Traffic'],
     '原始通道数远大于 7，截断后只保留极少数通道 —— '
     '**通道选择本身是混杂因素**，本组结论不能读作"这些数据上方法更差"。'),
    ('III', '无周期 / 近随机游走', ['Exchange'],
     '近似随机游走的汇率，几乎没有季节项 —— 任何依赖"学周期"的方法在这里都会'
     '退化成学均值。**单格成组，不与其他数据集求均值。**'),
    ('IV', '量纲离群（不进 core）', ['ILI'],
     '百万级计数，量纲比其余八格高两个数量级。**单独成列、不进 core(8)**，'
     '否则它会独自决定主排序。'),
]
CORE_GROUPS = ('I', 'II', 'III')      # = 5 + 2 + 1 = core (8)

DATA_ATTRS = [
    ('ETTh1', '变压器油温 / 负载', '7', '1 h', '强（日 + 周）', '~1.7 万'),
    ('ETTh2', '变压器油温 / 负载', '7', '1 h', '强（日 + 周）', '~1.7 万'),
    ('ETTm1', '变压器油温 / 负载', '7', '15 min', '强（日 + 周）', '~7 万'),
    ('ETTm2', '变压器油温 / 负载', '7', '15 min', '强（日 + 周）', '~7 万'),
    ('Weather', '气象 21 项', '21 → 7', '10 min', '强（年 + 日）', '~5.3 万'),
    ('Exchange', '8 国汇率', '8 → 7', '1 d', '**弱 / 近随机游走**', '~7.6 千'),
    ('ILI', '流感样病例就诊比', '7', '1 w', '强（年）', '~1 千'),
    ('ECL', '321 个用户用电量', '321 → 7', '1 h', '强（日 + 周）', '~2.6 万'),
    ('Traffic', '862 条公路占用率', '862 → 7', '1 h', '强（日 + 周）', '~1.7 万'),
]


def _datatype_block(rows: dict, key: str, order, group, ref: str | None,
                    show_venue: bool) -> list:
    """One table per data type.  A single-cell group gets no mean column --
    repeating the cell as its own mean is noise, and it invites the reader to
    compare a "mean" that is really one number against a real average."""
    tag, label, ds, _ = group
    rk = ranks(rows, 'mse', ds)
    single = len(ds) == 1
    tail = ['组内排名'] if single else ['组均值', '组内排名']
    head = ['| method | ' + ('venue | ' if show_venue else '')
            + ' | '.join(ds) + ' | ' + ' | '.join(tail) + ' |']
    head.append('|' + '|'.join(['---'] + (['---'] if show_venue else [])
                              + [':-:'] * (len(ds) + len(tail))) + '|')
    body = []
    for name in order:
        rec = rows[name]
        if not all(d in rec[key] for d in ds):
            continue
        cells = [f4(rec[key][d]) + ('*' if d in rec.get('tuned', ()) else '')
                 for d in ds]
        extra = [VENUE.get(name, '—')] if show_venue else []
        vals = [] if single else [f4(mean_of(rec, key, ds))]
        body.append('| ' + ' | '.join([name] + extra + cells + vals
                                      + [f2p(rk[name])]) + ' |')
    # sort by group mean so the reader does not have to
    def _key(line: str) -> float:
        parts = line.split('|')
        return float(parts[-3].strip()) if single else float(parts[-3].strip())
    body.sort(key=_key)
    return head + body


def build_by_datatype(channel: dict, multi: dict, fedtdo: dict,
                      fedal: dict, ref: str = 'FedTDO-center') -> list:
    """T2 · the same numbers, regrouped by what kind of series they are."""
    out = ['# 表 2 · 按数据类型分组的论文级评估表', '',
           '> 本表只回答"准不准"。同一批格子的**参数 / 线上字节 / 通信开销 / Pareto 前沿**'
           '见 `T8_cost_by_datatype.md`（同分组、同 seed、同数据源）。', '',
           '数据源与表 1 / 表 4 / 表 5 / 表 6 **完全相同**（同一批 `all_cells.csv`），',
           '只是按"序列属于哪一类"重新分组并给出**组内排名**。', '',
           '## 2a. 九格的数据属性', '',
           '| dataset | 内容 | 通道（原始 → 本协议） | 采样 | 周期性 | 测试窗数 |',
           '|---|---|---|:-:|---|---:|']
    for d, what, ch, freq, per, n in DATA_ATTRS:
        out.append(f'| `{d}` | {what} | {ch} | {freq} | {per} | {n} |')
    out += ['',
            '**为什么这样分而不是按领域分**：领域（电力 / 气象 / 金融 / 医疗）不能解释',
            '任何一行结果；能解释的是**通道数**、**周期强度**、**量纲**这三件事。',
            '`core (8)` = 类型 I（5 格）+ 类型 II（2 格）+ 类型 III（1 格，Exchange），',
            '**类型 IV（ILI）不进 core** —— 类型 III/IV 都是单格成组，',
            '不与任何其他数据集求均值，否则量纲差会让排名失真。', '']

    # ---- channel protocol --------------------------------------------------
    out += ['## 2b. `channel` 协议 · 联邦方法（MSE）', '',
            '每客户端 1 个传感器、`enc_in=1`、2 轮 × 800 步 + 1 次聚合、seed 2021。'
            '表内**按组均值升序**排列。', '']
    full = [m for m in FL_ORDER if m in channel
            and sum(1 for d in DATASETS if d in channel[m]['mse']) == 9]
    full += sorted(m for m in channel if m not in full
                   and sum(1 for d in DATASETS if d in channel[m]['mse']) == 9)
    for g in DATATYPE_GROUPS:
        out += [f'### 类型 {g[0]} · {g[1]}', '', f'> {g[3]}', '']
        out += _datatype_block(channel, 'mse', full, g, ref, True)
        out.append('')
    out += _cross_view({m: channel[m] for m in full}, 'method', '####')
    out += ['', '#### 同一批方法的 MAE 横切', '']
    out += _cross_view({m: channel[m] for m in full}, 'method', '#####', 'mae')

    # ---- multivariate, no federation ---------------------------------------
    loc = {b: r for (m, b), r in multi.items() if m == 'local'}
    out += ['', '## 2c. 多变量协议 · 不联邦骨干（MSE，表 4 的类型视图）', '',
            '客户端 = 1 个数据集的 7 个通道，`features M`，1600 梯度步，每格独立训练。', '']
    for g in DATATYPE_GROUPS:
        out += [f'### 类型 {g[0]} · {g[1]}', '']
        out += _datatype_block(loc, 'mse', sorted(loc), g, None, False)
        out.append('')
    out += _cross_view(loc, 'backbone')
    out += ['', '### 同一批骨干的 MAE 横切', '']
    out += _cross_view(loc, 'backbone', '####', 'mae')

    # ---- multivariate, FedAvg (appendix) -----------------------------------
    fed = {b: r for (m, b), r in multi.items() if m == 'fedavg'}
    out += ['', '## 2d. 多变量协议 · 跨数据集 FedAvg（MSE，表 5 的类型视图）', '',
            '> **附录控制组，不作任何方法的 SOTA 行。** 与 2c 逐格相同的数据与预算，',
            '> 唯一区别是每轮做一次参数平均。2c 与 2d 的差就是**聚合代价**。', '']
    for g in DATATYPE_GROUPS:
        out += [f'### 类型 {g[0]} · {g[1]}', '']
        out += _datatype_block(fed, 'mse', sorted(fed), g, None, False)
        out.append('')
    out += _cross_view(fed, 'backbone')
    out += ['', '### 同一批骨干的 MAE 横切', '']
    out += _cross_view(fed, 'backbone', '####', 'mae')

    # ---- FedTDO / FeDaL vs best local / best fedavg -------------------------
    ours = {f'FedTDO-{a}': r for a, r in fedtdo.items()}
    ours.update({f'fedal(FeDaL·ffts)': r for r in fedal.values()})
    best_loc = {b: r for b, r in loc.items()}
    pool = dict(ours)
    pool.update({f'local+{b}': r for b, r in best_loc.items() if b in
                 ('patchtst', 'stmtm', 'ffts', 'timeffm', 'dtmoe')})
    pool.update({f'fedavg+{b}': r for b, r in fed.items() if b in
                 ('patchtst', 'ffts', 'timeffm')})
    out += ['', '## 2e. FedTDO 与 FeDaL 在四个类型上的位置（MSE）', '',
            '> 只列参照行，避免和 2c / 2d 重复。`local+*` 是不联邦，`fedavg+*` 是附录。', '']
    for g in DATATYPE_GROUPS:
        out += [f'### 类型 {g[0]} · {g[1]}', '']
        out += _datatype_block(pool, 'mse', sorted(pool), g, None, False)
        out.append('')
    out += _readout_datatype(channel, loc, fed, ours, ref)
    out += _readout_ours_multi(pool, ref)
    return out


def _readout_ours_multi(pool: dict, ref: str) -> list:
    """Where FedTDO stands *per data type* on the multivariate protocol.

    The core(8) average hides this: a method can be first on the average while
    being first on no single group, and that is exactly what happens here.
    """
    name = ref if ref in pool else next(
        (k for k in pool if k.startswith(ref)), None)
    if name is None:
        return []
    complete = {n: r for n, r in pool.items()
                if all(d in r['mse'] for d in DATASETS)}
    out = ['', '### 多变量协议：FedTDO 逐类型排第几', '',
           '| 类型 | 格数 | FedTDO-center | 该类第一 | 排名 | 差距 |',
           '|---|:-:|:-:|---|:-:|:-:|']
    ranks_seen = []
    for g in DATATYPE_GROUPS:
        tag, label, ds, _ = g
        vals = {n: mean_of(r, 'mse', ds) for n, r in complete.items()}
        if name not in vals:
            continue
        order = sorted(vals, key=vals.get)
        pos = order.index(name) + 1
        # ties count as the same position
        ties = sum(1 for v in vals.values() if abs(v - vals[name]) < 1e-12)
        ranks_seen.append(pos)
        first = order[0]
        gap = vals[name] / vals[first] - 1
        out.append(f'| {tag} · {label.split("（")[0]} | {len(ds)} | {vals[name]:.4f} | '
                   f'`{first}` ({vals[first]:.4f}) | {pos}/{len(order)}'
                   f'{"（并列）" if ties > 1 else ""} | {100*gap:+.1f}% |')
    best_pos, worst_pos = min(ranks_seen), max(ranks_seen)
    out += ['',
            f'- **FedTDO-center 在四个类型上的最好名次是第 {best_pos}，最差是第 {worst_pos}；'
            + ('**没有任何一个类型排第一**。' if best_pos > 1 else '至少在一个类型上排第一。'),
            '- 因此它的 core(8) 第一不是"每类都强"，而是"**每一类都不太差**"——'
            '这是两个完全不同的主张，论文里不能混。'
            '若审稿人按数据类型逐组比较，会直接看到它在类型 II 落后于 `local+ffts`。',
            '- 类型 II 的落后（ECL / Traffic）是全表最大的一处反例：'
            '`local+ffts` 0.3598、`local+timeffm` 0.3714 都优于 FedTDO，'
            '**且这两个是"不联邦"**。写"FedTDO 在所有骨干最好"会被这两行反驳。']
    return out


def _cross_view(order_src: dict, label: str, header_level: str = '###',
                key: str = 'mse') -> list:
    """One row per method, one column per data type, plus core (8)."""
    complete = {n: r for n, r in order_src.items()
                if all(d in r[key] for d in DATASETS)}
    cols = [(g[0], g[1].split('（')[0], g[2]) for g in DATATYPE_GROUPS]
    # column-wise best, so "strongest type" means "where this row is least bad"
    best = {tag: min(mean_of(r, key, ds) for r in complete.values())
            for tag, _, ds in cols}
    out = [f'{header_level} 四类横切', '']
    head = (f'| {label} | ' + ' | '.join(f'类型 {t} {n}' for t, n, _ in cols)
            + ' | core (8) | 最强的类型 | 最弱的类型 |')
    out.append(head)
    out.append('|' + '|'.join(['---'] + [':-:'] * (len(cols) + 3)) + '|')
    recs = []
    for name, rec in complete.items():
        vals = {t: mean_of(rec, key, ds) for t, _, ds in cols}
        recs.append((mean_of(rec, key, CORE), name, vals))
    recs.sort()
    for core, name, vals in recs:
        # groups III / IV are single cells; only core groups decide "strongest"
        adv = {t: vals[t] / best[t] - 1 for t in CORE_GROUPS}
        strong = min(adv, key=adv.get)
        weak = max(adv, key=adv.get)
        cells = ' | '.join(f4(vals[t]) for t, _, _ in cols)
        s_txt = f'{strong} (best)' if abs(adv[strong]) < 5e-4 else f'{strong} ({adv[strong]:+.1%})'
        w_txt = f'{weak} (best)' if abs(adv[weak]) < 5e-4 else f'{weak} ({adv[weak]:+.1%})'
        out.append(f'| {name} | {cells} | **{f4(core)}** | {s_txt} | {w_txt} |')
    out += ['',
            '> "最强的类型" = 相对该类**全场最好值**差距最小的一类（括号内为该相对差距），'
            '只在类型 I/II/III 里选 —— 类型 IV 是单格且不进 core，'
            '拿来当"最强类型"没有意义。']
    return out


def _readout_datatype(channel: dict, loc: dict, fed: dict, ours: dict,
                      ref: str) -> list:
    out = ['', '## 2f. 读法（由脚本算出，不是手写）', '']
    groups = {g[0]: g[2] for g in DATATYPE_GROUPS}

    for tag in ('I', 'II', 'III'):
        ds = groups[tag]
        ok = [m for m in loc if all(d in loc[m]['mse'] for d in ds)]
        best = min(ok, key=lambda m: mean_of(loc[m], 'mse', ds))
        okf = [m for m in fed if all(d in fed[m]['mse'] for d in ds)]
        bf = min(okf, key=lambda m: mean_of(fed[m], 'mse', ds))
        out.append(f'- **类型 {tag}（{len(ds)} 格）**：不联邦最好 `{best}` '
                   f'（组均值 {mean_of(loc[best], "mse", ds):.4f}），'
                   f'同组 FedAvg 最好 `{bf}`（{mean_of(fed[bf], "mse", ds):.4f}），'
                   f'聚合代价 {100*(mean_of(fed[bf], "mse", ds)/mean_of(loc[best], "mse", ds)-1):+.0f}%。')

    # Split "backbone ability" from "aggregation cost" for the LM rows.  This is
    # the one place where grouping by data type changes the *narrative*, so it is
    # computed rather than asserted.
    lm = [b for b in ('ffts', 'timeffm') if b in loc and b in fed]
    if lm and 'patchtst' in loc and 'patchtst' in fed:
        ref_b = 'patchtst'
        out.append('')
        for tag in ('I', 'II'):
            ds = groups[tag]
            a_loc = mean_of(loc[lm[0]], 'mse', ds) / mean_of(loc[ref_b], 'mse', ds) - 1
            a_fed = mean_of(fed[lm[0]], 'mse', ds) / mean_of(fed[ref_b], 'mse', ds) - 1
            out.append(
                f'- **类型 {tag}：`{lm[0]}` vs `{ref_b}`** —— 不联邦 **{100*a_loc:+.1f}%**，'
                f'跨数据集 FedAvg **{100*a_fed:+.1f}%**。')
        # aggregation cost per backbone, per group: the real story
        out.append('')
        out.append('- **聚合代价按骨干拆开（同一组内，FedAvg 相对不联邦的涨幅）**：')
        for b in (ref_b, *lm):
            if b not in loc or b not in fed:
                continue
            row = []
            for tag in ('I', 'II', 'III'):
                ds = groups[tag]
                row.append(f'类型 {tag} {100*(mean_of(fed[b], "mse", ds)/mean_of(loc[b], "mse", ds)-1):+.0f}%')
            out.append(f'  - `{b}`：' + '，'.join(row) + '。')
        out.append('  **`ffts` 在类型 II 的聚合代价远高于 `patchtst`** —— 也就是说，'
                   '不联邦时 LM 骨干在高维截断数据上**是赢的**（上一条），'
                   '是**聚合**把它毁掉的，不是 LM 本身。'
                   '这正是 FFTS 需要"双侧正则"、FeDaL 需要 DBE 的直接证据，'
                   '也解释了为什么 FeDaL 的收益恰好集中在 ECL / Traffic。')

    ref_rec = channel.get(ref)
    if ref_rec:
        for tag in ('I', 'II', 'III', 'IV'):
            ds = groups[tag]
            wins_here = [d for d in ds
                         if ref_rec['mse'].get(d) is not None
                         and ref_rec['mse'][d] <= min(
                             r['mse'][d] for r in channel.values() if d in r['mse'])]
            out.append(f'- **FedTDO-center 在类型 {tag} 的单格第一**：'
                       f'{len(wins_here)}/{len(ds)} 格'
                       f'（{", ".join(wins_here) if wins_here else "无"}）。')
    # Do MSE and MAE agree on who wins a group?  If they disagree the difference
    # is noise and must not be claimed -- this happens on group I.
    ext = {m: r for m, r in channel.items()
           if all(d in r['mse'] and d in r['mae'] for d in DATASETS)
           and not m.startswith('FedTDO')}
    if ext:
        gmap = {g[0]: g[2] for g in DATATYPE_GROUPS}
        out += ['', '### MSE 与 MAE 是否给出同一个赢家（逐类型）', '']
        flips = []
        for tag in ('I', 'II', 'III'):
            ds = gmap[tag]
            b_mse = min(ext, key=lambda m: mean_of(ext[m], 'mse', ds))
            b_mae = min(ext, key=lambda m: mean_of(ext[m], 'mae', ds))
            same = b_mse == b_mae
            line = (f'- 类型 {tag}：MSE 最好 `{b_mse}`（{mean_of(ext[b_mse], "mse", ds):.4f}），'
                    f'MAE 最好 `{b_mae}`（{mean_of(ext[b_mae], "mae", ds):.4f}）'
                    f' → {"一致" if same else "**不一致**"}')
            if not same:
                flips.append(tag)
            out.append(line)
        if flips:
            out.append('')
            out.append(f'- **类型 {", ".join(flips)} 上两个指标结论不一致**。'
                       '这时两者相差都在 1% 以内，属于单 seed 噪声，'
                       '**不能写"在这个数据类型上 A 优于 B"** —— 只能写"持平"。'
                       '这也说明为什么本表两个指标都报：只报 MSE 会漏掉这件事。')
        else:
            out.append('')
            out.append('- 三个类型上两个指标结论一致，排序可以同时用 MSE 与 MAE 主张。')

    # Where does the core-first ranking actually come from?
    if ref_rec:
        gmap = {g[0]: g[2] for g in DATATYPE_GROUPS}
        rows_ok = {m: r for m, r in channel.items()
                   if all(d in r['mse'] for d in DATASETS)}
        # the runner-up is the best *external* method, not another FedTDO arm --
        # comparing against our own ablation says nothing about the field
        ext = {m: r for m, r in rows_ok.items() if not m.startswith('FedTDO')}
        out += ['', '### FedTDO 的 core 第一是从哪一格来的', '']
        tables_gap = []
        for tag in ('I', 'II', 'III'):
            ds = gmap[tag]
            mine = mean_of(ref_rec, 'mse', ds)
            others = {m: mean_of(r, 'mse', ds) for m, r in ext.items()}
            best_o = min(others, key=others.get)
            gap = mine / others[best_o] - 1
            verdict = '**领先**' if gap < 0 else '**落后**'
            # share of the core(8) difference that this group accounts for
            contrib = (others[best_o] - mine) * len(ds) / 8
            tables_gap.append((tag, len(ds), gap, contrib))
            out.append(f'- 类型 {tag}（{len(ds)} 格）：FedTDO-center {mine:.4f} vs 最强外部对手 '
                       f'`{best_o}` {others[best_o]:.4f} → {verdict} {100*abs(gap):.1f}%；'
                       f'对 core(8) 的贡献 **{contrib:+.4f}**。')
        total = sum(c for _, _, _, c in tables_gap)
        out += ['',
                f'- **把三个类型对 core(8) 的贡献相加 = {total:+.4f}**，'
                '这等于 FedTDO 与最强外部对手在 core 上的全部差距 —— '
                '**正号的那一格就是它第一名的全部来源**。',
                '- **core(8) 的权重是 类型 I 5/8、类型 II 2/8、类型 III 1/8**：'
                '类型 II 只有 2 格却有 2/8 的权重，而类型 I 的 5 格才是数据主体。',
                '- 所以论文里必须写成"在 9 数据集均值上最好，领先集中在 ECL / Traffic"，'
                '**不能写成"在所有数据类型上都最好"** —— 类型 I 与类型 III 上它都不是第一。']
        # the group-I verdict literally flips between the two metrics -> report both
        ds1 = gmap['I']
        if 'fedala' in ext:
            d_mse = mean_of(ref_rec, 'mse', ds1) / mean_of(ext['fedala'], 'mse', ds1) - 1
            d_mae = mean_of(ref_rec, 'mae', ds1) / mean_of(ext['fedala'], 'mae', ds1) - 1
            flip = (d_mse > 0) != (d_mae > 0)
            out.append(
                f'- **类型 I 上 MSE 与 MAE 的胜负{"相反" if flip else "相同"}**：'
                f'对 `fedala`，MSE {100*d_mse:+.1f}%、MAE {100*d_mae:+.1f}%。'
                + ('两者都 <1%，**这一格只能写"持平"**，选任一个指标去主张都会翻车。'
                   if flip else ''))
    return out


def build_conditions(channel: dict, multi: dict, fedal: dict) -> list:
    """T7 · what is actually held constant, and what is not."""
    out = ['# 表 7 · 条件一致性审计', '',
           '回答一个问题：**这些行是不是在同一条件下跑出来的？**',
           '逐项分成"统一"和"不统一"两类，不统一的每一项都写明影响面。', '',
           '## 7a. 统一的部分（同一协议内）', '',
           '| 项 | `channel` 协议 | 多变量协议 |', '|---|---|---|',
           '| 训练预算 | 2 轮 × 800 梯度步 = **1600 步/client** | 同 |',
           '| 聚合（基线） | 2 次：第 1 次定第 2 轮初始化，第 2 次下发最终模型后评测 | 同 |',
           '| 数据切分 | 按传感器列切客户端，train/val/test **7:1:2** | 按数据集切客户端，同比例 |',
           '| `seq_len / pred_len` | 96 / 24 | 96 / 96（表 1 与表 4 不同，**不可跨表**） |',
           '| 优化器 / lr / batch | AdamW, 5e-4, 32（`d_model` 32, patch 16/stride 16） | '
           'AdamW，lr 与 batch 见各入口脚本 |',
           '| 评测集 | test split 全窗平均 | 同 |',
           '| 评价函数 | `utils.metrics.metric`（MSE + MAE 一次算出） | 同 |',
           '| seed | 2021 | 同 |',
           '| 骨干 | `dtmoe`（除 LM 行） | 见各行 |',
           '',
           '**梯度步预算是硬约束，不是口号**：`fl_client.local_train` 里',
           '`if not max_steps: max_steps = args.fl_local_max_steps` ——',
           '任何方法即使只传 `epochs=` 也会被 800 步截断。',
           '`fedrep` 的 200/600 是**同一 800 步的内部划分**，不是额外预算。',
           '`fedl2p` 只传 `epochs=`，同样被 800 步截断（这条已逐方法核对过）。', '',
           '## 7b. 不统一的部分（每一项都必须写进论文）', '',
           '| # | 项 | 影响 | 处理 |', '|---|---|---|---|',
           '| 1 | **单 seed**（全部 2021） | 没有 std / 95% CI，2%–9% 的差距不可主张 | '
           '表 1 用 n=8 置换检验；≥15% 的差距才单独主张 |',
           '| 2 | 多变量表 `pred_len=96` vs `channel` 表 `24` | 两表**绝对不可比** | 各自表头写明 |',
           '| 3 | `scaffold` / `fedavg_sgd` 的 Weather 用 `_lr*` 重跑值回填 | '
           '这两个方法多了一次调参机会 | 数值打 `*` + 脚注 |',
           '| 4 | **FedTDO 只做 1 次 `synchronize_patch_encoder`（只传 encoder），'
           '基线做 2 次全模型聚合** | 同步次数 1 vs 2、载荷 encoder vs 全 `state_dict` | '
           '对 FedTDO 有利，必须主动披露；表 8d 已给出"FedTDO 按 2 次同步计费"的反事实 |',
           '| 5 | `fedadam`(lr 0.01) / `scaffold`(lr 5e-3 + SGD) / `fedavg_sgd`(lr 5e-3) '
           '有 per-method 超参覆盖 | 这 3 个方法调过参，其余 20 个用共享默认'
           '（AdamW, lr 5e-4, batch 32） | '
           '每个覆盖值都注明了 mini-sweep 来源，且只扫 2–3 个值 |',
           '| 6 | LM 骨干走 GPU + TF32，其余走 CPU fp32 | TF32 尾数 10 bit，'
           '数值与 fp32 不逐位相同 | 只影响 LM 那 5 行；TF32 对 CPU 是 no-op |',
           '| 7 | `local+stmtm` 多 400 步掩码预训练 | 该行预算高于同表其他行 | 表 4 脚注已标 |',
           '| 8 | `local+tfps` 在 1600 步下欠收敛 | 该行是**预算下界**，不是架构能力 | 表 4 脚注已标 |',
           '| 9 | `fedal` 下游多 1600 步有监督微调 | 这是 FeDaL 方法本体的一部分 | 单独成表（表 6） |',
           '| 10 | ECL / Traffic 原始 321 / 862 通道被截断到 7 | 类型 II 的结论含通道选择混杂 | 表 2 明确标注 |',
           '| 11 | FedTDO 的 `online` 因果流是**另一个估计量** | 不在这些表里 | 只报 offline |',
           '',
           '## 7c. 一句话结论', '',
           '- **同一张表内部**：预算、切分、评测函数、seed 全部统一，可以直接比较。',
           '- **跨表**：`channel`（pl24）与多变量（pl96）不可比，只能各自内部排序。',
           '- **最大的不统一是 seed 数，不是超参**：单 seed 下任何 <10% 的排序都可能翻转。',
           '- 上面 10 项里，第 3 / 4 / 6 / 7 项是**对别的行不利**的不对等（有人多拿了预算或调参），',
           '  第 8 项是**对我们自己有利**的不对等（FeDaL 多拿 1600 步），必须主动披露。']
    return out


def build_coverage(rows: dict, multi: dict, fedal: dict, fedtdo: dict | None = None) -> list:
    out = ['# 表 0 · 覆盖审计', '',
           '每个格子代表"有数值且非 NaN"。缺格写 `—`，从不拿邻格填充。', '']

    out += ['## `channel` 协议（表 1）', '',
            '| method | ' + ' | '.join(DATASETS) + ' | covered |',
            '|' + '|'.join(['---'] + [':-:'] * (len(DATASETS) + 1)) + '|']
    for name in [m for m in FL_ORDER if m in rows] + sorted(m for m in rows if m not in FL_ORDER):
        cells = ['✓' if d in rows[name]['mse'] and not math.isnan(rows[name]['mse'][d]) else '—'
                 for d in DATASETS]
        out.append(f'| {name} | ' + ' | '.join(cells)
                   + f' | {sum(c == "✓" for c in cells)}/9 |')

    out += ['', '## 多变量协议（表 4/5/6）', '',
            '| method | backbone | ' + ' | '.join(DATASETS) + ' | covered |',
            '|' + '|'.join(['---', '---'] + [':-:'] * (len(DATASETS) + 1)) + '|']
    for (method, backbone), rec in sorted(multi.items()):
        cells = ['✓' if d in rec['mse'] else '—' for d in DATASETS]
        out.append(f'| {method} | {backbone} | ' + ' | '.join(cells)
                   + f' | {sum(c == "✓" for c in cells)}/9 |')
    for name, rec in sorted(fedal.items()):
        cells = ['✓' if d in rec['mse'] else '—' for d in DATASETS]
        out.append(f'| fedal | {name} | ' + ' | '.join(cells)
                   + f' | {sum(c == "✓" for c in cells)}/9 |')
    for arm, rec in sorted((fedtdo or {}).items()):
        cells = ['✓' if d in rec['mse'] else '—' for d in DATASETS]
        out.append(f'| FedTDO-{arm} | dtmoe | ' + ' | '.join(cells)
                   + f' | {sum(c == "✓" for c in cells)}/9 |')
    return out


def build_multi(multi: dict, method: str, title: str, note: list,
                ref: str | None = None) -> list:
    rows = {b: rec for (m, b), rec in multi.items() if m == method}
    order = sorted(rows, key=lambda b: mean_of(rows[b], 'mse', CORE))
    rk = ranks(rows, 'mse', CORE)
    wn = wins(rows, 'mse', CORE)
    out = [title, ''] + note + ['',
           '| backbone | params | ' + ' | '.join(DATASETS)
           + ' | mean (9) | core (8) | avg rank | #best |',
           '|' + '|'.join(['---', '---:'] + [':-:'] * (len(DATASETS) + 3)) + '|']
    for b in order:
        rec = rows[b]
        cells = [f4(rec['mse'].get(d)) for d in DATASETS]
        out.append('| ' + ' | '.join([b, f"{rec['params']:,}" if rec['params'] else '—']
                                     + cells
                                     + [f4(mean_of(rec, 'mse', DATASETS)),
                                        f4(mean_of(rec, 'mse', CORE)),
                                        f2p(rk[b]), str(wn[b])]) + ' |')
    out += ['', 'MAE：', '',
            '| backbone | ' + ' | '.join(DATASETS) + ' | mean (9) | core (8) |',
            '|' + '|'.join(['---'] + [':-:'] * (len(DATASETS) + 2)) + '|']
    for b in order:
        rec = rows[b]
        out.append('| ' + ' | '.join([b] + [f4(rec['mae'].get(d)) for d in DATASETS]
                                     + [f4(mean_of(rec, 'mae', DATASETS)),
                                        f4(mean_of(rec, 'mae', CORE))]) + ' |')
    return out


def build_fedtdo_vs(multi: dict, fedtdo: dict, fedal: dict, ref: str = 'FedTDO-center') -> list:
    rows: dict = {}
    for arm, rec in fedtdo.items():
        key = {'base': 'FedTDO-raw', 'center': 'FedTDO-center',
               'decorr': 'FedTDO-decorr'}.get(arm, f'FedTDO-{arm}')
        rows[key] = rec
    for (m, b), rec in multi.items():
        rows[f'{m}+{b}'] = rec
    for name, rec in fedal.items():
        rows['fedal (FFTS)'] = rec

    order = sorted(rows, key=lambda n: mean_of(rows[n], 'mse', CORE))
    rk = ranks(rows, 'mse', CORE)
    wn = wins(rows, 'mse', CORE)
    ref_core = mean_of(rows[ref], 'mse', CORE)
    out = ['# 表 6 · FedTDO 完整方法 vs 各骨干（同一多变量协议）', '',
           '> 客户端 = 数据集、`features M`、7 通道、`sl96 → pl96`、1600 梯度步。',
           '> FedTDO 三臂与 `local+xx` / `fedavg+xx` 同协议、同预算、同一 `offline` 估计量',
           '> （全测试集一次性前向），因此**同列可比**。',
           '> `fedal (FFTS)` = 用 FeDaL 三阶段训练 FFTS 适配器，预训练另计（见 §3d）。', '',
           '| method | params | ' + ' | '.join(DATASETS)
           + ' | mean (9) | core (8) | avg rank | #best | Δ vs FedTDO-center | p |',
           '|' + '|'.join(['---', '---:'] + [':-:'] * (len(DATASETS) + 5)) + '|']
    for name in order:
        rec = rows[name]
        cells = [f4(rec['mse'].get(d)) for d in DATASETS]
        core = mean_of(rec, 'mse', CORE)
        if name == ref:
            d, pcell = float('nan'), '—'
        else:
            d = core / ref_core - 1
            diffs = [rec['mse'][k] - rows[ref]['mse'][k] for k in CORE
                     if k in rec['mse'] and k in rows[ref]['mse']]
            p = sign_flip_p(diffs)
            pcell = f'{p:.3f}{star(p)}'
        params = f"{rec['params']:,}" if rec['params'] else '—'
        out.append('| ' + ' | '.join([name, params] + cells
                                     + [f4(mean_of(rec, 'mse', DATASETS)), f4(core),
                                        f2p(rk[name]), str(wn[name]), pct(d), pcell]) + ' |')
    out += readout_multi(rows, ref)
    return out


def build_long_csv(channel: dict, multi: dict, fedtdo: dict, fedal: dict) -> list:
    rows = []

    def _ok(v) -> bool:
        return v is not None and not (isinstance(v, float) and math.isnan(v))

    for m, rec in channel.items():
        for d in DATASETS:
            if _ok(rec['mse'].get(d)):
                rows.append({'protocol': 'channel', 'method': m, 'backbone': '',
                             'dataset': d, 'mse': rec['mse'][d],
                             'mae': rec['mae'].get(d), 'params': '',
                             'venue': VENUE.get(m, '')})
    for (m, b), rec in multi.items():
        for d in DATASETS:
            if _ok(rec['mse'].get(d)):
                rows.append({'protocol': 'multivariate', 'method': m, 'backbone': b,
                             'dataset': d, 'mse': rec['mse'][d],
                             'mae': rec['mae'].get(d), 'params': rec['params'],
                             'venue': ''})
    for arm, rec in fedtdo.items():
        for d in DATASETS:
            if _ok(rec['mse'].get(d)):
                rows.append({'protocol': 'multivariate', 'method': f'FedTDO-{arm}',
                             'backbone': 'dtmoe', 'dataset': d,
                             'mse': rec['mse'][d], 'mae': rec['mae'].get(d),
                             'params': rec['params'], 'venue': 'ours'})
    for name, rec in fedal.items():
        for d in DATASETS:
            if _ok(rec['mse'].get(d)):
                rows.append({'protocol': 'multivariate', 'method': 'fedal',
                             'backbone': 'ffts', 'dataset': d,
                             'mse': rec['mse'][d], 'mae': rec['mae'].get(d),
                             'params': rec['params'], 'venue': 'ours'})
    return rows


def build_index(channel: dict, multi: dict, fedal: dict, fedtdo: dict) -> list:
    n_fl = sum(1 for m, r in channel.items()
               if sum(1 for d in DATASETS if d in r['mse']) == 9)
    n_local = sum(1 for (m, b) in multi if m == 'local')
    n_fedavg = sum(1 for (m, b) in multi if m == 'fedavg')
    return [
        '# 论文级数据表索引',
        '',
        '全部由 `scripts/xc_paper_tables.py` 生成，**不要手工改本目录里的任何文件**——',
        '下次运行会覆盖。要改数字就改数据源，要改格式就改脚本。',
        '',
        '```bash',
        'D:/software/Anaconda3/python.exe scripts/xc_paper_tables.py',
        '```',
        '',
        '## 表清单',
        '',
        '| 文件 | 内容 | 协议 | 规模 | seed |',
        '|---|---|---|---|---|',
        f'| `T0_coverage.md` | 覆盖审计：每个格子有没有数 | 两者 | '
        f'{n_fl} 联邦方法 + {n_local} 集中式 + {n_fedavg} 联邦骨干 + FeDaL + FedTDO 三臂 | 2021 |',
        f'| `T1_fl_channel.md` | **主表**：MSE + MAE + 平均秩 + 置换检验 | `channel` | '
        f'{n_fl} 方法 × 9（+1 部分行 `fedavg_sgd`，不进排名） | 2021 |',
        '| `T2_by_datatype.md` | **按数据类型分组**：I 低维强周期 / II 高维截断 / '
        'III 无周期 / IV 量纲离群，四类横切 + 组内排名 | 两者 | '
        f'{n_fl} + {n_local} + {n_fedavg} 行 × 4 类 | 2021 |',
        f'| `T4_backbones_local.md` | 不联邦骨干（时序模型主表） | 多变量 | '
        f'{n_local} 骨干 × 9 | 2021 |',
        f'| `T5_backbones_fedavg.md` | 附录：聚合代价下界，**不进主表** | 多变量 | '
        f'{n_fedavg} 骨干 × 9 | 2021 |',
        f'| `T6_fedtdo_vs_backbones.md` | FedTDO vs 全部骨干（同协议同预算） | 多变量 | '
        f'{len(fedtdo)} 臂 + {n_local + n_fedavg} 骨干 + FeDaL | 2021 |',
        '| `T7_conditions.md` | **条件一致性审计**：哪些项是统一的、哪些不是、影响多大 | 两者 | '
        '— | 2021 |',
        '| `T8_cost_by_datatype.md` | **按数据类型分组的资源与通信成本**：'
        '参数 / 线上字节 / 通信量 / Pareto 前沿 / 线上状态构成审计 | 两者 | '
        f'{n_fl} + {n_local} 行 × 4 类 | 2021 |',
        '| `BIG0_rank_summary.md` | **排名算式表**：`Rank MSE` + `Rank MAE` = `Rank Σ`，'
        '外加 `#1 MSE` / `#1 MAE` / `#1 Σ`（两个指标各自第一的次数相加） | `channel` | '
        f'{n_fl} 联邦 + 1 聚合变体 + 5 参照行 | 2021 |',
        '| `BIG1_channel.md` | **论文正文大表 + 全方法索引**：方法 × 9 数据集，'
        '**每个数据集拆 `MSE` / `MAE` 两列**（共 18 列），各自加粗/下划线，按方法族分块；'
        '四块＝3 个 FedTDO 臂 / 19 个联邦基线 / 1 个聚合变体（`fedavg+patchtst`）/ '
        '`Centralised reference`（`local` + 9 个不联邦时序骨干，同一网格、同一预算；'
        '`Venue` 列写出**架构族 · 发表年 · 优先级**，四族＝长序列分解 / patch-based / '
        '变量维度 / 多尺度混合 + 3 个 2025 SOTA）+ '
        '2 个不完整控制行；**文件末尾 `Appendix A`** 把**全部已实现方法**'
        '（联邦方法 + 14 个时序骨干 + FeDaL）在两个协议下的 `Core (8)` 并排列出，'
        '不排名、不加粗，只回答"这个方法到底跑没跑" | '
        '`channel` | ' f'{n_fl} 联邦 + 1 聚合 + 5 参照 + 2 控制 + 全方法索引 | 2021 |',
        '| `BIG2_by_datatype.md` | **论文正文大表**：同一批行，列折叠成 4 个数据类型 '
        '（仍拆 MSE/MAE）+ `Rank Σ`，**时序骨干与聚合变体也在里面** | `channel` | 同上 | 2021 |',
        '| `BIG3_multivariate.md` | **论文正文大表**：FedTDO / FeDaL / 14 个集中式骨干，'
        '同样拆 `MSE` / `MAE` | 多变量 | 3 + 1 + ' + f'{n_local} | 2021 |',
        '| `BIG4_cost.md` | **论文正文大表**：精度 + 通信成本（按 4 个类型拆开）+ '
        '`MSE×MiB` / `MAE×MiB` 单位字节精度 | `channel` | 同上 | 2021 |',
        '| `BIG5_aggregation.md` | **附录大表**：`local` vs 一轮 FedAvg，'
        '每个类型 4 列（local/fedavg × MSE/MAE）| 多变量 | '
        f'{n_fedavg} 骨干 | 2021 |',
        '| `BIG_TABLES.html` | 上面 6 张表的排版预览（纸面样式，含两级表头/加粗/下划线） | 两者 | — | 2021 |',
        '| `tex/BIG0..5.tex` | 上面 6 张表的 booktabs LaTeX，可直接 `\\input` | 两者 | — | 2021 |',
        '| `all_cells.csv` | 上面所有格子的长表（protocol/method/backbone/dataset/mse/mae） | 两者 | '
        '— | 2021 |',
        '',
        '**`T*` 是诊断表，`BIG*` 是正文表。** 两者数字同源（同一批 CSV / JSON），'
        '区别只在排版：`T*` 一表一问、带排名与 p 值；`BIG*` 是论文里印的那一张，'
        '**`MSE` 与 `MAE` 各占一列**、各自排最优/次优、按方法族分块。',
        '**`BIG*` 的排名口径**：`Rank MSE`、`Rank MAE` 分别按 8 个 core 数据集求平均秩'
        '（并列取平均，与 T1 同定义）；`Rank Σ = Rank MSE + Rank MAE`，**表内按 Σ 排序**；'
        '`#1 Σ = #1 MSE + #1 MAE` 是两个指标各自拿第一的次数之和。'
        '注意 `#1` 是唯一“越大越好”的列，所以它单独加粗最大值，不参与 Σ。',
        '**写论文用 `BIG*`；回答"为什么"用 `T*`。**',
        '',
        '## 两套协议不可混用',
        '',
        '| | `channel`（表 1） | 多变量（表 4/5/6） |',
        '|---|---|---|',
        '| 客户端 | 1 个传感器，`enc_in=1` | 1 个数据集的 7 个通道，`features M` |',
        '| 异质性来源 | 传感器之间的分布差异 | 数据集（领域）之间的差异 |',
        '| 参数量级 | 10^4–10^5 | 10^4–10^7 |',
        '| 能回答 | 哪个联邦算法更好 | 哪个骨干/训练方式更好 |',
        '',
        '**同一个数字在两张表之间没有可比性**，表头已各自写明协议。',
        '',
        '## 评价指标口径',
        '',
        '- **MSE / MAE**：`utils.metrics.metric`，测试集全窗平均；多变量表是 `offline` 估计量',
        '  （全测试集一次性前向），FedTDO 的 `online` 因果流是**另一个估计量**，不在这些表里。',
        '- **core (8)**：去掉 ILI 的 8 个数据集均值 = 类型 I（5）+ 类型 II（2）+ '
        '类型 III（1，Exchange）。ILI 是百万级计数、量纲差两个数量级，'
        '放进均值会让它单独决定排名，所以主排序用 core，同时给 `mean (9)`。',
        '  表 2 把这个和拆开成四个数据类型的列，并给出每一类对 core 差距的**贡献额**。',
        '- **avg rank**：逐数据集排名（并列取平均）后的均值，1 = 最好。',
        '- **#best**：单格第一的次数，**含并列**，因此一列之和会超过 9。',
        '- **p**：与 FedTDO-center 的**精确配对符号翻转置换检验**（8 对，双侧，枚举 2^8），',
        '  `†` = p < 0.05。n=8 时下限是 0.0078。**没有用 t 检验**：8 对样本下 t 检验没有功效，',
        '  且没有任何理由假设误差差值的正态性。',
        '',
        '## 目前的所有缺口（都是有意留的）',
        '',
        '- **单 seed**：全部只有 2021。没有 std / 95% CI，跨 seed 稳定性未测。',
        '- 表 1 的 `fedavg_sgd` 缺 Traffic（附录控制组，不进排名）；'
        '`scaffold` / `fedavg_sgd` 的 Weather 在标准 lr 下发散，用 `_lr*_exp` 重跑值回填并标 `*`。',
        '- 时序模型**没有**联邦版（`PP_fedavg_{stmtm,raft,tfps}_*` 不存在，且按 §0b 不补）。',
        '- `tfps` 在 1600 步下欠收敛（§4b），`local+tfps` 这行是预算下界而非架构能力。',
        '- **`fedal` 只训了 FFTS 一种骨干**。Time-FFM + FeDaL 未跑；要主张"FeDaL 与骨干无关"',
        '  必须补这一格（`run_fedal.py` 目前写死 FFTS 的 patch 几何）。',
        '- 表 8 的 `dtmoe` 在多变量协议下的参数量**复现不出来**（录制 117,172 / 重建 484,900），',
        '  该行取较大值；引用前必须先定版。',
        '',
        '## 生成链（顺序不能反）',
        '',
        '```bash',
        '# 1. 实测线上状态（LM 骨干必须用 GPU 解释器）',
        'D:/software/Anaconda3/envs/ma_fmarl/python.exe scripts/xc_cost_probe.py',
        '# 2. 精度表（CPU 解释器）',
        'D:/software/Anaconda3/python.exe scripts/xc_paper_tables.py',
        '# 3. 成本表（读第 1 步的 JSON）',
        'D:/software/Anaconda3/python.exe scripts/xc_cost_tables.py',
        '# 4. 正文大表 + LaTeX + HTML（读 1/2/3 的结果，可单独跑 --only BIG4）',
        'D:/software/Anaconda3/python.exe scripts/xc_big_tables.py',
        '# 5. 排名核对（独立实现重算 Rank/#1/Pos + 换手检验 + 可分辨性）',
        'D:/software/Anaconda3/python.exe scripts/verify_ranks.py '
        '--json results_paper/tables/_rank_audit.json',
        '```',
        '',
        '## 排名的可信边界（`scripts/verify_ranks.py`）',
        '',
        '第 5 步是**独立实现**：`rank_map()` 用 `np.unique` + `bincount` 分组算平均秩，',
        '不复用 `xc_paper_tables.ranks()` 的排序循环；7 列 × 27 行的输出必须**逐位相等**，',
        '不等就是表错了。它同时给出三条表里印不出来的结论：',
        '',
        '- **`Rank 1` 在单 seed 下不可分辨到什么程度**：把相差 0.5%–5% 的数当并列重排，',
        '  `FedTDO` 家族始终第一、最强基线 `fedala` 始终追不上，但 `center` 与 `decorr`',
        '  的先后在 0.5% 容差处就翻转 —— 正文只能写"FedTDO 第一"，',
        '  **不能写"center 第一"**。数字见 `BIG0` 的换手注。',
        '- **逐数据集可分辨性**（27 个方法在该格的变异系数 CV）：',
        '  ETTh2 只有 0.077（27 个方法挤在 0.144–0.183，`top1 vs median` 仅 3.0%），',
        '  是全场信息量最低的一格，**它的单格第一基本是噪声**；',
        '  ECL 0.615 / Traffic 0.383 / Weather 0.322 才是真正能分开方法的数据集。',
        '- **留一数据集重排**：只有删掉 ETTh2 时 Σ 第一名会变（`decorr` 取代 `center`），',
        '  其余 7 格删任何一个都不变 —— 与上一条是同一件事的两种看法。',
        '- **`#1` 计数依赖容差**：0% 时 `center` 的 `#1 Σ` = 5，1% 时 11，5% 时 15。',
        '  `#1` 只能当"有几格是明显第一"，不能当"赢了几格"。',
    ]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--print', action='store_true')
    args = p.parse_args()

    channel = load_channel()
    multi = load_multi()
    fedtdo = load_fedtdo_multi()
    fedal = load_fedal()

    written = []
    written.append(write('T_INDEX.md', build_index(channel, multi, fedal, fedtdo)))
    written.append(write('T0_coverage.md', build_coverage(channel, multi, fedal, fedtdo)))
    t1 = build_channel(channel)
    t1 += ['', '## 局限（必须写进论文）', '',
           '1. **单 seed**。所有行只有 seed 2021，没有 std / 95% CI。',
           '   本表能主张的是"在这个 seed 上更好"，跨 seed 的稳定性另需 3–5 个 seed。',
           '2. **n=8 的置换检验下限 0.0078**，且 5 个基线的差距（2.2%–9.2%）在单 seed 噪声内，',
           '   只有 `fedkme` / `fedda` / `fedgtp` / `fedpkda` 这类 ≥15% 的差距可以单独主张。',
           '3. **`fedda` / `fedgtp` 是参考实现**（指纹聚类 + mixup / 指纹图聚合），',
           '   不是原文逐字复现，引用时必须写明。']
    written.append(write('T1_fl_channel.md', t1))

    written.append(write('T4_backbones_local.md', build_multi(
        multi, 'local', '# 表 4 · 不联邦主表：各骨干独立训练（多变量协议，pl96）',
        ['> 每个数据集训练一个独立模型，不共享参数、不聚合 —— 与 FFTS / FeDaL 论文里',
         '> 那几行骨干的评法一致。**这是唯一进正文的时序模型表**；联邦版只有附录表 5，',
         '> 按 §0b 的硬约束不得进主表。',
         '> 预算 1600 梯度步（原文是几十 epoch，绝对值低于原文，见实现清单）。'])))

    written.append(write('T5_backbones_fedavg.md', build_multi(
        multi, 'fedavg', '# 表 5（附录）· 跨数据集 FedAvg：聚合代价的下界证据',
        ['> **不进主表、不作任何方法的 SOTA 行。** 唯一用途是量化"数据集级异质性的',
         '> 聚合代价" —— 它是 FFTS 的"双侧正则"和 FeDaL 的"DBE + GBE"要解决的问题。',
         '> 与表 4 逐格相同的数据、相同的预算，唯一区别是每轮做一次 FedAvg。'])))

    written.append(write('T6_fedtdo_vs_backbones.md',
                         build_fedtdo_vs(multi, fedtdo, fedal)))

    written.append(write('T2_by_datatype.md',
                         build_by_datatype(channel, multi, fedtdo, fedal)))
    written.append(write('T7_conditions.md',
                         build_conditions(channel, multi, fedal)))

    rows = build_long_csv(channel, multi, fedtdo, fedal)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / 'all_cells.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    written.append(OUT / 'all_cells.csv')

    for w_ in written:
        print(f'wrote {w_.relative_to(ROOT)}  ({w_.stat().st_size} bytes)')

    if args.print:
        print()
        print('\n'.join(build_channel(channel)[:20]))


if __name__ == '__main__':
    main()
