"""Classic paper-style *large* tables, MSE and MAE as separate columns.

Everything else in ``results_paper/tables/`` is diagnostic: one small table per
question, with ranks and p-values spelled out.  A paper does not print those --
it prints **one big table per protocol**, methods as rows, datasets as columns,
best in bold and second best underlined, rows grouped by method family.

**MSE and MAE are never squeezed into one cell.**  Every dataset occupies two
columns, ``<ds> MSE`` and ``<ds> MAE``, each ranked and marked on its own, so a
method that is good on MSE and mediocre on MAE cannot hide behind an average.

Ranking columns (computed over the 8 ``core`` datasets, ties averaged):

``Rank MSE``   mean rank on MSE
``Rank MAE``   mean rank on MAE
``Rank Σ``     ``Rank MSE + Rank MAE`` -- **the sort key, smallest = rank 1**
``#1 MSE``     number of datasets won on MSE (ties counted for every tied row)
``#1 MAE``     number of datasets won on MAE
``#1 Σ``       ``#1 MSE + #1 MAE`` -- the "how many first places in total" column

Sources are unchanged (``xc_paper_tables.load_channel`` / ``load_multi`` /
``xc_cost_tables`` cost model), so the numbers here are identical to T1 / T2 /
T8 by construction -- only the layout is different.  Never edit the output by
hand; rerun::

    D:/software/Anaconda3/python.exe scripts/xc_big_tables.py

Outputs
-------
``BIG0_rank_summary.md`` / ``.tex``   the ranking arithmetic, one small table
``BIG1_channel.md`` / ``.tex``        main accuracy table, ``channel`` protocol
``BIG2_by_datatype.md`` / ``.tex``    the same rows, columns collapsed to types
``BIG3_multivariate.md`` / ``.tex``   main accuracy table, multivariate protocol
``BIG4_cost.md`` / ``.tex``           accuracy *and* price, same rows
``BIG5_aggregation.md`` / ``.tex``    appendix: price of one round of averaging
``BIG_TABLES.html``                   all of the above, typeset for reading
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import xc_cost_tables as CT  # noqa: E402
import xc_paper_tables as PT  # noqa: E402

OUT = PT.OUT
TEX = OUT / 'tex'

DATASETS = PT.DATASETS
CORE = PT.CORE
TYPES = PT.DATATYPE_GROUPS          # I / II / III / IV with their dataset lists

OURS_CH = ['FedTDO-center', 'FedTDO-decorr', 'FedTDO-raw']
OURS_MV = {'center': 'FedTDO-center', 'decorr': 'FedTDO-decorr',
           'base': 'FedTDO-raw'}
MV_VENUE = {'FedTDO-center': 'ours', 'FedTDO-decorr': 'ours',
            'FedTDO-raw': 'ours', 'fedal': 'FeDaL (2025)'}

# 4-decimal cell values: two numbers that print the same must be marked the same
TOL = 5e-5
TOL_SUM = 5e-3                      # Rank Σ is printed with 2 decimals


# --------------------------------------------------------------------------- #
# formatting
# --------------------------------------------------------------------------- #
def nan(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def f4(v) -> str:
    return '—' if nan(v) else f'{v:.4f}'


def f2(v) -> str:
    return '—' if nan(v) else f'{v:.2f}'


def fint(v) -> str:
    return '—' if nan(v) else f'{int(v)}'


def kilo(v) -> str:
    if nan(v):
        return '—'
    return f'{v/1000:,.1f}'


def mib(v: float) -> str:
    if v == 0:
        return '0'
    if v < 1:
        return f'{v:.2f}'
    if v < 100:
        return f'{v:.1f}'
    return f'{v:,.0f}'


def tex_esc(s: str) -> str:
    for a, b in (('\\', r'\textbackslash{}'), ('_', r'\_'), ('%', r'\%'),
                 ('&', r'\&'), ('#', r'\#')):
        s = s.replace(a, b)
    return s


def md_deco(s: str, mark) -> str:
    if mark == 'best':
        return f'**{s}**'
    if mark == 'second':
        return f'<u>{s}</u>'
    return s


def tex_deco(s: str, mark) -> str:
    if mark == 'best':
        return rf'\textbf{{{s}}}'
    if mark == 'second':
        return rf'\underline{{{s}}}'
    return s


def mean_of(rec: dict, key: str, ds) -> float:
    vals = [rec[key][d] for d in ds
            if d in rec[key] and not math.isnan(rec[key][d])]
    return sum(vals) / len(vals) if vals else float('nan')


# --------------------------------------------------------------------------- #
# column marks: best / second best inside one printed column
# --------------------------------------------------------------------------- #
def marks(values: dict, tol: float = TOL) -> dict:
    """Mark the minimum and the runner-up, judged on the *printed* value.

    Two rows whose values print identically must carry the same mark, otherwise
    the table looks like a bug.  ``tol`` is therefore one unit of the printed
    precision, not machine epsilon.
    """
    items = sorted((v, k) for k, v in values.items() if not nan(v))
    out: dict = {}
    if not items:
        return out
    best = items[0][0]
    for k, v in values.items():
        if not nan(v) and abs(v - best) <= tol:
            out[k] = 'best'
    second = next((v for v, _ in items if v - best > tol), None)
    if second is not None:
        for k, v in values.items():
            if not nan(v) and abs(v - second) <= tol:
                out[k] = 'second'
    return out


# --------------------------------------------------------------------------- #
# table model
# --------------------------------------------------------------------------- #
# spec kinds
#   {'kind': 'pair',  'label', 'label_tex', 'ds'|[func], 'single': bool}
#       -> two columns <label> MSE / <label> MAE
#   {'kind': 'one',   'label', 'label_tex', 'func', 'fmt', 'grp', 'sub',
#    'tol', 'mkfn'}
#       -> one column
def grid_table(tid, title, caption, note, sections, specs, venue: bool = True):
    """Build one table; every ``pair`` spec becomes two printed columns."""
    rows_all = [r for _, rs in sections for r in rs]
    rows_dec = [r for r in rows_all if not r.get('_partial')]

    flat = []                       # one entry per printed data column
    for sp in specs:
        if sp.get('kind') == 'pair':
            if sp.get('func'):
                gm = lambda r, f=sp['func']: (f(r) or (None, None))[0]
                ga = lambda r, f=sp['func']: (f(r) or (None, None))[1]
            else:
                gm = lambda r, ds=sp['ds']: mean_of(r, 'mse', ds)
                ga = lambda r, ds=sp['ds']: mean_of(r, 'mae', ds)
            ex = sp.get('excl') or ()
            dec = [r for r in rows_dec if r['name'] not in ex]
            mk = (marks({r['name']: gm(r) for r in dec}, sp.get('tol', TOL)),
                  marks({r['name']: ga(r) for r in dec}, sp.get('tol', TOL)))
            ds1 = (sp['ds'][0] if not sp.get('func') and len(sp['ds']) == 1
                   else None) if sp.get('single', True) else None
            for i, sub in enumerate(('MSE', 'MAE')):
                flat.append({'label': f"{sp['label']} {sub}",
                             'label_tex': f"{sp['label_tex']} {sub}",
                             'grp': sp['label'], 'grp_tex': sp['label_tex'],
                             'sub': sub, 'get': (gm, ga)[i], 'fmt': f4,
                             'mk': mk[i], 'ds1': ds1, 'na': sp.get('na'),
                             'agg': sp.get('agg', sp.get('ds') == CORE)})
        else:
            ex = sp.get('excl') or ()
            vals = {r['name']: sp['func'](r) for r in rows_dec
                    if r['name'] not in ex}
            if sp.get('hi'):        # counts: the largest total is the best
                vals = {k: (-v if not nan(v) else v) for k, v in vals.items()}
            flat.append({'label': sp['label'], 'label_tex': sp['label_tex'],
                         'grp': sp.get('grp', ''), 'grp_tex': sp.get('grp_tex', ''),
                         'sub': sp.get('sub', ''),
                         'get': sp['func'], 'fmt': sp.get('fmt', f4),
                         'mk': ({} if (sp.get('mkfn') or sp.get('nodeco'))
                                else marks(vals, sp.get('tol', TOL))),
                         'mkfn': sp.get('mkfn'), 'ds1': None,
                         'na': sp.get('na'),
                         'agg': sp.get('agg', False)})

    # two-level header: merge runs of the same non-empty grp
    sup, row2_md, row2_tex = [], [], []
    i = 0
    while i < len(flat):
        j = i
        while (j + 1 < len(flat) and flat[i]['grp']
               and flat[j + 1]['grp'] == flat[i]['grp']):
            j += 1
        k = j - i + 1
        if k > 1:
            sup.append((flat[i]['grp'], flat[i]['grp_tex'], k))
            for t in range(i, j + 1):
                row2_md.append(flat[t]['sub'])
                row2_tex.append(flat[t]['sub'])
        else:
            sup.append(('', '', 1))
            row2_md.append(flat[i]['label'])
            row2_tex.append(flat[i]['label_tex'])
        i = j + 1

    lead = ['Method', 'Venue'] if venue else ['Method']
    lead_tex = ['Method', 'Venue'] if venue else ['Method']
    n = len(lead) + len(flat)

    out_rows = []
    for btitle, rs in sections:
        out_rows.append(('sec', btitle, n))
        for r in rs:
            cmd = [r['name']] + ([r.get('venue', '—')] if venue else [])
            ctex = [tex_esc(r['name'])] + ([tex_esc(r.get('venue', '—'))]
                                           if venue else [])
            for c in flat:
                # a row missing cells must not print an average over the cells
                # it happens to have -- it is not comparable
                if ((r.get('_partial') and c.get('agg'))
                        or r['name'] in (c.get('na') or ())):
                    cmd.append('—')
                    ctex.append('—')
                    continue
                v = c['get'](r)
                s = c['fmt'](v) if v is not None else '—'
                star = bool(c['ds1'] and c['ds1'] in r.get('tuned', ()))
                mk = c['mkfn'](r) if c.get('mkfn') else c['mk'].get(r['name'])
                cmd.append(md_deco(s + ('*' if star else ''), mk))
                ctex.append(tex_deco(s + (r'^{*}' if star else ''), mk))
            out_rows.append(('row', cmd, ctex))

    return {'id': tid, 'title': title, 'caption': caption, 'note': note,
            'header': lead + [c['label'] for c in flat],
            'header_tex': lead_tex + [c['label_tex'] for c in flat],
            'lead': lead, 'sup': sup, 'sub': row2_md, 'sub_tex': row2_tex,
            'has_sup': sum(1 for x in sup if x[2] > 1) >= 2,
            'rows': out_rows}


# --------------------------------------------------------------------------- #
# Appendix A of BIG1 -- every implemented method, both protocols, one index
# --------------------------------------------------------------------------- #
FAM_ORDER = ['FedTDO (ours)', 'federated method', 'FL + swapped backbone',
             'pretrained LM + FeDaL', 'no federation']


def _family(name: str) -> str:
    if name.startswith('FedTDO'):
        return 'FedTDO (ours)'
    if name == 'fedal(FeDaL)':
        return 'pretrained LM + FeDaL'
    if name in ('local',) or name.startswith('local+'):
        return 'no federation'
    if '+' in name:
        return 'FL + swapped backbone'
    return 'federated method'


def inventory_md(channel: dict, bb: dict | None, fedbb: dict | None,
                 multi: dict, fedtdo: dict, fedal: dict) -> list:
    """One row per implemented method, with a number only where one exists.

    Table 1 can only print the ``channel`` protocol, and not every backbone is
    informative there: the cross-variable ones degenerate at ``enc_in = 1`` and
    the two frozen-LM ones cannot run at all, so their real numbers live on the
    multivariate protocol (where a client is a dataset).  The two protocols must
    never share a cell.  This appendix is the join: it says, for every method in
    the project, whether it exists on each protocol and what its ``Core (8)``
    number is.  It is deliberately **unranked and unmarked** -- it is an index,
    not a result.
    """
    rows: dict = {}

    def slot(name: str, venue: str) -> dict:
        r = rows.get(name)
        if r is None:
            r = rows[name] = {'name': name, 'venue': venue, 'ch': None,
                              'mv': None}
        return r

    for m, rec in channel.items():
        slot(m, PT.VENUE.get(m, '—'))['ch'] = rec
    for name, rec in (bb or {}).items():
        slot('local+' + name, bb_venue(name))['ch'] = rec
    for (meth, name), rec in (fedbb or {}).items():
        slot(f'{meth}+{name}', PT.VENUE.get(meth, '—'))['ch'] = rec
    for (meth, name), rec in multi.items():
        n = (('local' if meth == 'local' else meth) if name == 'dtmoe'
             else f'{meth}+{name}')
        # a `local+<backbone>` row must carry the same family/venue/triage tag in
        # both protocol columns, otherwise the index says "no federation" twice
        ven = (bb_venue(name) if meth == 'local' and name != 'dtmoe'
               else PT.VENUE.get(meth, 'no federation'))
        slot(n, ven)['mv'] = rec
    for arm, rec in fedtdo.items():
        slot(OURS_MV.get(arm, arm), 'ours')['mv'] = rec
    for rec in fedal.values():
        slot('fedal(FeDaL)', 'FeDaL (2025)')['mv'] = rec

    def cell(rec, key: str) -> str:
        """Only print a ``Core (8)`` mean when all eight core cells exist.

        A mean over seven of the eight is not the same quantity as the column in
        Table 1 -- it is an average over a different, easier subset -- so a
        partial row prints a dash here exactly like it does there.
        """
        if rec is None or any(d not in rec[key] for d in CORE):
            return '—'
        v = mean_of(rec, key, CORE)
        return '—' if nan(v) else f'{v:.4f}'

    def cells(rec) -> str:
        if rec is None:
            return '—'
        return f"{sum(1 for d in DATASETS if d in rec['mse'])}/9"

    out = [
        '',
        '---',
        '',
        '# Appendix A · every method in the project, and where its numbers live',
        '',
        '> **This is an index, not a result.** No bold, no underline, no ranks: '
        'the two protocol columns are printed side by side only because each '
        'method was measured on whichever protocol it *can* be measured on, and '
        'a reader who wants to know "do we have this method at all" should not '
        'have to open four files.',
        '> **The two protocols are not comparable.** `channel` = one sensor per '
        'client (`enc_in = 1`), horizon 24. `multivariate` = one *dataset* per '
        'client (`features M`, `enc_in = 7`), horizon 96. A method can look '
        'weak on one and strong on the other -- that is the protocol, not the '
        'method (see `SOTA_MATRIX.md` §3). Never subtract a number from the '
        'left pair from one on the right pair.',
        '> `cells` = complete cells out of the nine datasets; `Core (8)` = mean '
        'over the eight datasets excluding ILI, exactly as in Table 1. '
        '`—` = not run on that protocol (not "failed"). Full grids: Table 1 for '
        '`channel`, `BIG3_multivariate.md` for the other; cost in `BIG4_cost.md`.',
        '> `Venue` carries the architecture family, the publication venue and the '
        'project triage for every `local+<backbone>` row '
        '(`patch-based · ICLR\'23 · required`), so the two protocol columns can be '
        'read without going back to the prose.',
        '> `channel` is `enc_in = 1`, which degenerates anything whose '
        'contribution is **cross-variable** (`iTransformer` has one variate token, '
        '`TimeMixer` has no channels to mix) and makes the two cross-domain '
        'models (`FFTS`, `Time-FFM`, frozen GPT-2) meaningless. Their `channel` '
        'rows are therefore the 1-variate special case -- quoted to show what the '
        'architecture collapses to under one-sensor clients, never as the '
        'published model. Take multivariate numbers for architecture claims.',
        '',
        '| Method | Family | Venue | channel cells | channel Core (8) MSE '
        '| channel Core (8) MAE | multiv. cells | multiv. Core (8) MSE '
        '| multiv. Core (8) MAE |',
        '| --- | --- | --- | :-: | :-: | :-: | :-: | :-: | :-: |',
    ]
    ordered = sorted(rows.values(),
                     key=lambda r: (FAM_ORDER.index(_family(r['name'])),
                                    r['name']))
    for r in ordered:
        out.append(
            f"| {r['name']} | {_family(r['name'])} | {r['venue']} "
            f"| {cells(r['ch'])} | {cell(r['ch'], 'mse')} "
            f"| {cell(r['ch'], 'mae')} | {cells(r['mv'])} "
            f"| {cell(r['mv'], 'mse')} | {cell(r['mv'], 'mae')} |")
    out.append('')
    return out


# --------------------------------------------------------------------------- #
# renderers
# --------------------------------------------------------------------------- #
def render_md(t: dict) -> list:
    n = len(t['header'])
    out = [f'# {t["title"]}', '']
    out += [f'> {x}' for x in t['note']]
    out += ['', '| ' + ' | '.join(t['header']) + ' |',
            '|' + '|'.join(['---', '---'] + [':-:'] * (n - 2)) + '|']
    for kind, a, b in t['rows']:
        if kind == 'sec':
            out.append('| **' + a + '** |' + ' |' * (n - 1))
        else:
            out.append('| ' + ' | '.join(a) + ' |')
    return out


def tex_line(line: str) -> str:
    """Map the two Unicode glyphs the tables use onto plain LaTeX.

    `·` separates the architecture family from the venue in the reference block
    and `—` is the "not measured" dash.  `pdflatex` with `utf8` inputenc often
    survives them but silently depends on the font encoding, so they are
    rewritten: `$\\cdot$` exists in every math font, `--` is the em dash.
    """
    return line.replace('·', r'$\cdot$').replace('—', '--')


def render_tex(t: dict) -> list:
    """booktabs, two-level header, the way a paper actually prints it."""
    n = len(t['header'])
    nlead = len(t['lead'])
    wide = n > 16
    colspec = '@{}' + 'l' * nlead + 'c' * (n - nlead) + '@{}'
    out = ['% generated by scripts/xc_big_tables.py -- do not edit',
           r'\begin{table*}[t]', r'\centering', r'\small',
           rf'\caption{{{t["caption"]}}}',
           rf'\label{{tab:{t["id"].lower()}}}']
    if wide:
        out += [r'\resizebox{\textwidth}{!}{%', r'\scriptsize',
                r'\setlength{\tabcolsep}{3pt}']
    out += [rf'\begin{{tabular}}{{{colspec}}}', r'\toprule']

    sup = t['sup']
    if t.get('has_sup'):
        cells, spans, idx = [], [], nlead + 1
        for label, label_tex, k in sup:
            cells.append(rf'\multicolumn{{{k}}}{{c}}{{{label_tex}}}' if k > 1
                         else '')
            if label:
                spans.append((idx, idx + k - 1))
            idx += k
        out.append('& ' * nlead + ' & '.join(cells) + ' \\\\')
        if spans:
            out.append(' '.join(rf'\cmidrule(lr){{{a}-{b}}}' for a, b in spans))
        out.append('& ' * nlead + ' & '.join(t['sub_tex']) + ' \\\\')
    else:
        out.append(' & '.join(t['header_tex']) + ' \\\\')
    out.append(r'\midrule')

    first = True
    for kind, a, b in t['rows']:
        if kind == 'sec':
            rule = '' if first else r'\midrule'
            out.append(rf'{rule}\multicolumn{{{n}}}{{@{{}}l}}{{\textit{{{tex_esc(a)}}}}} \\')
            first = False
        else:
            out.append(tex_line(' & '.join(b)) + ' \\\\')
    out += [r'\bottomrule', r'\end{tabular}']
    if wide:
        out.append('}%')
    out += [r'\end{table*}', '']
    return out


def write_md(t: dict, path: Path) -> None:
    path.write_text('\n'.join(render_md(t)) + '\n', encoding='utf-8')


def write_tex(t: dict, path: Path) -> None:
    path.write_text('\n'.join(render_tex(t)) + '\n', encoding='utf-8')


# --------------------------------------------------------------------------- #
# row helpers
# --------------------------------------------------------------------------- #
def as_row(name: str, rec: dict, venue: str | None = None) -> dict:
    v = venue or PT.VENUE.get(name, '—')
    if v.startswith('re-impl.'):
        v = 're-impl.'          # full description lives in the footnote
    return {'name': name, 'venue': v,
            'mse': rec['mse'], 'mae': rec['mae'],
            'tuned': rec.get('tuned', set())}


def dual_rank(pool: list) -> None:
    """Attach MSE rank, MAE rank, their sum and the win counts to every row."""
    by = {r['name']: r for r in pool}
    rm = PT.ranks(by, 'mse', CORE)
    ra = PT.ranks(by, 'mae', CORE)
    wm = PT.wins(by, 'mse', CORE)
    wa = PT.wins(by, 'mae', CORE)
    for r in pool:
        r['_rm'] = rm.get(r['name'], float('nan'))
        r['_ra'] = ra.get(r['name'], float('nan'))
        r['_rs'] = r['_rm'] + r['_ra']
        r['_wm'] = wm.get(r['name'], 0)
        r['_wa'] = wa.get(r['name'], 0)
        r['_ws'] = r['_wm'] + r['_wa']
    # competition ranking on Σ: 1,1,3,4 ...
    order = sorted((r['_rs'], r['name']) for r in pool)
    for i, (v, name) in enumerate(order):
        if i and abs(v - order[i - 1][0]) < 1e-12:
            by[name]['_rk_pos'] = by[order[i - 1][1]]['_rk_pos']
        else:
            by[name]['_rk_pos'] = i + 1


def rank_cols(bold_core: bool = True) -> list:
    """The Rank / #1 block, shared by every accuracy table."""
    cols = [
        {'kind': 'one', 'label': 'Rank MSE', 'label_tex': 'Rank MSE',
         'grp': 'Rank', 'grp_tex': r'\textbf{Rank}',
         'sub': 'MSE', 'func': lambda r: r['_rm'], 'fmt': f2, 'mkfn': None},
        {'kind': 'one', 'label': 'Rank MAE', 'label_tex': 'Rank MAE',
         'grp': 'Rank', 'sub': 'MAE', 'func': lambda r: r['_ra'], 'fmt': f2},
        {'kind': 'one', 'label': 'Rank Σ', 'label_tex': r'Rank $\Sigma$',
         'grp': 'Rank', 'sub': r'Σ', 'func': lambda r: r['_rs'],
         'fmt': f2, 'tol': TOL_SUM},
        {'kind': 'one', 'label': '#1 MSE', 'label_tex': r'\#1 MSE',
         'grp': '#1', 'grp_tex': r'\textbf{\#1}',
         'sub': 'MSE', 'func': lambda r: r['_wm'], 'fmt': fint, 'nodeco': True},
        {'kind': 'one', 'label': '#1 MAE', 'label_tex': r'\#1 MAE',
         'grp': '#1', 'sub': 'MAE', 'func': lambda r: r['_wa'], 'fmt': fint,
         'nodeco': True},
        {'kind': 'one', 'label': '#1 Σ', 'label_tex': r'\#1 $\Sigma$',
         'grp': '#1', 'sub': r'Σ', 'func': lambda r: r['_ws'], 'fmt': fint,
         'hi': True},
    ]
    for c in cols:
        c['agg'] = True
        c.setdefault('grp_tex', c['grp'].replace('#', r'\#'))
    return cols


RANK_NOTE = (
    '**Rank MSE** / **Rank MAE** = mean rank over the 8 `core` datasets, '
    '1 = best, ties averaged -- the same definition as `T1`. '
    '**Rank Σ** = `Rank MSE + Rank MAE`; **rows are sorted by Σ**, so the first '
    'row of each block is the best method *on the two metrics together*, not on '
    'MSE alone. **#1 MSE** / **#1 MAE** = number of datasets won on that metric '
    '(ties counted for every tied row); **#1 Σ = #1 MSE + #1 MAE**, i.e. the '
    'total number of first places across the two metrics.'
    ' In the `Rank`/`#1` block bold marks the smallest Σ and the largest '
    '`#1 Σ` -- `#1` is the only place in the table where a bigger number is '
    'better, which is why it is marked separately instead of being added to Σ.'
)

REF_NOTE = (
    '**The `Centralised reference` block is not federated and not a competitor.** '
    '`local` = the project backbone (Dynamic TMoE) trained per client with no '
    'aggregation at all; `local+<backbone>` = the *same* protocol (same nine '
    'cells, same 1600 gradient steps, same 7:1:2 split, same d_model 32) with the '
    'per-client forecaster swapped for a published time-series model. These rows '
    'answer one question: **does federation beat simply training the strongest '
    'single-series model each client could train on its own?** They are ranked '
    'with everything else on purpose -- leaving a strong row unranked would be '
    'pool selection, and the point of the block is that we win on the same cells, '
    'not on a favourable subset.'
)


BACKBONE_NOTE = (
    '**The reference block covers six backbones across four architecture '
    'families, not six arbitrary models.** *Long-sequence decomposition* '
    '(Autoformer, FEDformer -- *optional*), *patch-based* (PatchTST -- '
    '**required**), *variate-dimension* (iTransformer -- **required**), '
    '*multi-scale mixing* (TimesNet, TimeMixer -- *suggested*), on top of the '
    'three 2025 supervised SOTA rows (ST-MTM, RAFT, TFPS). All nine run on the '
    'same nine cells, the same 1600 gradient steps and the same `d_model 32` as '
    'the rest of the table, and none of them is aggregated (constraint §0b). The '
    'family, the publication venue and the project triage are printed in the '
    '`Venue` column of this block.'
)

DEGENERATE_NOTE = (
    '**Two of the six backbones are degenerate on this grid, and that is the '
    'finding, not a defect.** `channel` gives every client exactly '
    'one sensor (`enc_in = 1`), so the variate axis vanishes: `iTransformer` has '
    'a single variate token and its attention becomes the identity, `TimeMixer` '
    'has no channels to mix. Their rows here are the **1-variate special case**, '
    'not the published multivariate model -- read them as "what that '
    'architecture collapses to when a client holds one sensor", and take the '
    'multivariate numbers from `BIG3`/Appendix A for any claim about the '
    'architecture itself. Autoformer, FEDformer and TimesNet are *not* degenerate '
    'in the same way (they operate inside a single series), but they are 2-3x '
    'slower per step, so their 1600-step budget is a lower bound on them exactly '
    'like `tfps`.'
)

GPU_NOTE = (
    '**One reference row is a GPU run, and it says so in `Venue`.** '
    '`local+timesnet` is the only cell of the channel grid not trained on CPU: its '
    '2-D period tensor costs ~7 min per client on CPU (a 32-client dataset runs '
    'into hours) against ~35 s on the RTX 4060. The device is not neutral -- the '
    'same cell trained both ways differs by **-0.12% MSE / +0.13% MAE** '
    '(`patchtst` ETTh1, CPU 0.359580 vs GPU 0.359137), which is one rank step on '
    'this table -- so the row is tagged rather than silently blended in. Nothing '
    'in the paper rests on a gap smaller than that: `local+timesnet` sits 22nd of '
    '33 ranked rows and never takes a `#1` place, so the device marker cannot '
    'change a conclusion.'
)

# family / venue / triage for the backbone rows of the reference block.  Kept
# next to the notes so the table and the prose cannot disagree.
TS_BB = {
    'patchtst':     ('patch-based', "ICLR'23", 'required'),
    'itransformer': ('variate-dim', "ICLR'24", 'required'),
    'timesnet':     ('multi-scale', "ICLR'23", 'suggested'),
    'timemixer':    ('multi-scale', "ICLR'24", 'suggested'),
    'autoformer':   ('long-seq', "NeurIPS'21", 'optional'),
    'fedformer':    ('long-seq', "ICML'22", 'optional'),
    'stmtm':        ('2025 SOTA', "KDD'25", 'reference'),
    'raft':         ('2025 SOTA', "ICML'25", 'reference'),
    'tfps':         ('2025 SOTA', "NeurIPS'25", 'reference'),
    'transformer':  ('vanilla', "NeurIPS'17", 'control'),
    'gru':          ('recurrent', 'GRU baseline', 'control'),
    'ffts':         ('cross-domain LM', "AAAI'25", 'reference'),
    'timeffm':      ('cross-domain LM', "NeurIPS'24", 'reference'),
}


# Backbones whose ``channel`` cells were trained on the GPU rather than the CPU.
# ``timesnet`` is the only one: its 2-D period tensor costs ~7 min per client on
# CPU (32-client datasets => hours) and ~35 s on the RTX 4060, a 12x gap that the
# other backbones do not have -- ``patchtst`` is actually *slower* on GPU (1.8 min
# vs 0.8 min per cell) because the model is too small to amortise launch overhead.
# Mixing devices is not free: the same cell trained both ways differs by
# -0.12% MSE / +0.13% MAE (patchtst ETTh1, CPU 0.359580 vs GPU 0.359137), which is
# one rank step on this table.  Hence the row is marked, not silently blended in.
GPU_BACKBONES = {'timesnet'}


def bb_venue(name: str) -> str:
    """`patch-based · ICLR'23 · required` for a backbone, `no federation` else."""
    fam, venue, prio = TS_BB.get(name, ('', '', ''))
    if not fam:
        return 'no federation'
    dev = ' · GPU' if name in GPU_BACKBONES else ''
    return f'{fam} · {venue} · {prio}{dev}'


FEDBB_NOTE = (
    '**`<method>+<backbone>` rows are federated**, so they cannot live in the '
    'reference block: they are one aggregation (`fedavg`, 2 x 800 steps) with the '
    'per-client model swapped for a published time-series network. They exist to '
    'price the aggregation on a non-MoE backbone -- `local+<backbone>` is the '
    'same swap with no aggregation, so the difference between the two rows is '
    'exactly what averaging costs. Only rows with all nine cells are ranked; the '
    'rest print dashes in `Core`/`Rank`/`#1` and sit in the appendix control '
    'block.'
)


# --------------------------------------------------------------------------- #
# BIG0 -- the ranking arithmetic on its own, one page
# --------------------------------------------------------------------------- #
def build_big0(channel: dict, bb: dict | None = None,
               fedbb: dict | None = None) -> dict:
    # the pool must be built by the *same* function as BIG1/BIG2, otherwise the
    # three rank columns would not agree across tables
    ours, fed, agg, ref, _ = channel_pool(channel, bb, fedbb)
    pool = ours + fed + agg + ref
    ours.sort(key=lambda r: r['_rs'])
    fed.sort(key=lambda r: r['_rs'])
    agg.sort(key=lambda r: r['_rs'])

    specs = [
        {'kind': 'one', 'label': 'Rank MSE', 'label_tex': 'Rank MSE',
         'func': lambda r: r['_rm'], 'fmt': f2},
        {'kind': 'one', 'label': 'Rank MAE', 'label_tex': 'Rank MAE',
         'func': lambda r: r['_ra'], 'fmt': f2},
        {'kind': 'one', 'label': 'Rank Σ', 'label_tex': r'Rank $\Sigma$',
         'func': lambda r: r['_rs'], 'fmt': f2, 'tol': TOL_SUM},
        {'kind': 'one', 'label': 'Pos', 'label_tex': 'Pos',
         'func': lambda r: r['_rk_pos'], 'fmt': fint, 'nodeco': True},
        {'kind': 'one', 'label': '#1 MSE', 'label_tex': r'\#1 MSE',
         'func': lambda r: r['_wm'], 'fmt': fint, 'nodeco': True},
        {'kind': 'one', 'label': '#1 MAE', 'label_tex': r'\#1 MAE',
         'func': lambda r: r['_wa'], 'fmt': fint, 'nodeco': True},
        {'kind': 'one', 'label': '#1 Σ', 'label_tex': r'\#1 $\Sigma$',
         'func': lambda r: r['_ws'], 'fmt': fint, 'hi': True},
        {'kind': 'pair', 'label': 'Core (8)', 'label_tex': r'\textbf{Core (8)}',
         'ds': CORE},
    ]
    top = min(pool, key=lambda r: r['_rs'])
    _order = sorted(pool, key=lambda r: r['_rs'])
    runner, gap = _order[1], _order[1]['_rs'] - top['_rs']

    # --- is the leading pair even separable? -----------------------------
    # one seed, and a rank is worth 1/8 of Rank MSE: a 0.5% relative gap is
    # inside run-to-run noise, so the note must say whether the winner survives
    # a tie band or only wins on the last decimal.
    by = {r['name']: r for r in pool}

    def sigma_at(tol: float) -> dict:
        rm = PT.ranks_tol(by, 'mse', CORE, tol)
        ra = PT.ranks_tol(by, 'mae', CORE, tol)
        return {n: rm[n] + ra[n] for n in by}

    def wins_at(tol: float) -> dict:
        wm = PT.wins_tol(by, 'mse', CORE, tol)
        wa = PT.wins_tol(by, 'mae', CORE, tol)
        return {n: wm[n] + wa[n] for n in by}

    band_lines = []
    for tol in (0.005, 0.01, 0.02, 0.05):
        s = sigma_at(tol)
        w = wins_at(tol)
        lead = sorted(by, key=lambda n: (s[n], n))
        ctr, dec = 'FedTDO-center', 'FedTDO-decorr'
        fam = min((n for n in by if n.startswith('FedTDO')), key=lambda n: s[n])
        rival = min((n for n in by if not n.startswith('FedTDO')
                     and n not in ('local',)
                     and not n.startswith('local+')), key=lambda n: s[n])
        band_lines.append(
            f'{tol * 100:.1f}%: Σ #1 `{lead[0]}` ({s[lead[0]]:.2f}); '
            f'`{ctr}` {s[ctr]:.2f} / `{dec}` {s[dec]:.2f} / '
            f'best non-FedTDO `{rival}` {s[rival]:.2f}; '
            f'`{ctr}` #1 Σ = {w[ctr]}')
        if tol == 0.005:
            swapped = lead[0] != top['name']
    sep = (
        '**How much of the word "first" survives noise.** Ranks are computed on '
        'exact values, so 0.3400 vs 0.3401 costs a whole rank -- and a rank is '
        '1/8 of `Rank MSE`. Recomputing with a *relative tie band* (two numbers '
        'within `tol` of the head of their group counted as tied) gives: '
        + '; '.join(band_lines) + '. '
        + ('Read the single-seed order as **`FedTDO` first**, not as '
           '`center` first: at a 0.5% band the two top-1 candidates swap, '
           'and 0.5% is well inside one-seed scatter on Type I. The robust '
           'statement is the family and the gap to the best baseline, not '
           'which of our three variants leads.'
           if swapped else
           'The leader is stable across the whole band tested.'))

    note = [
        RANK_NOTE,
        '**This table is the arithmetic behind every other table** -- the big '
        'tables print the same three rank columns and the same three `#1` '
        'columns, only with the per-dataset cells in front of them.',
        f'Rank 1 on Σ, *on exact values*, is `{top["name"]}` at '
        f'**{top["_rs"]:.2f}** '
        f'(MSE {top["_rm"]:.2f} + MAE {top["_ra"]:.2f}). '
        'On MSE alone the answer is different -- that difference is the reason '
        'the two metrics are never averaged into one cell. **Do not quote this '
        f'as a first place without the next note**: `{runner["name"]}` is '
        f'{gap:.2f} Σ points behind, which is not separable at one seed.',
        sep,
        '`Pos` is the position by `Rank Σ` with ties sharing a position. '
        'A row can be higher on `Pos` than on `Rank Σ` suggests by one decimal '
        'because Σ is printed to 2 dp. Blocks stay semantic (ours / baselines / '
        'reference), so a `Reference` row is printed at the bottom even when '
        'its `Pos` is high -- `Pos` is the ranking, the block layout is not.',
        'Single seed. Differences below ~1% on Type I are inside seed noise -- '
        'read those cells as ties until more seeds exist.',
    ]
    sections = [('FedTDO (ours)', ours), ('Federated baselines', fed)]
    if agg:
        sections.append(('Federated backbone variant -- `fedavg` on a swapped '
                         'per-client model', agg))
    sections.append(('Centralised reference (no federation, not a competitor)',
                     sorted(ref, key=lambda r: r['_rs'])))
    return grid_table('BIG0', 'Table 0 · Ranking summary: MSE and MAE ranked '
                      'separately, then summed (`channel` protocol)',
                      'Rank = mean rank over the eight core datasets, computed '
                      'for MSE and for MAE independently. $\\Sigma$ is their sum '
                      'and is the column the table is sorted by.',
                      note, sections, specs)


# --------------------------------------------------------------------------- #
# BIG1 -- channel protocol, main accuracy table
# --------------------------------------------------------------------------- #
def channel_pool(channel: dict, bb: dict | None = None,
                 fedbb: dict | None = None):
    """Split the ``channel`` table into the semantic blocks of BIG1.

    ``bb`` (backbone swaps, ``local+<backbone>``) are **not federated** -- they
    are appended to the ``Reference`` block, which already holds ``local``, and
    they take part in the ranking so the reader can see where a strong
    time-series model lands on the same nine cells.  Constraint §0b forbids
    *federating* them; it does not hide them, and a reference row that is not
    ranked would be an invitation to accuse us of cherry-picking the pool.

    ``fedbb`` (``fedavg+<backbone>``) *is* federated, so it cannot sit in the
    reference block: it gets its own section, ranked with everything else.  It
    is the price of the single aggregation measured on a non-MoE backbone, and
    it is the only federated backbone row that exists.

    Rows with fewer than nine cells are returned in ``ctrl`` and excluded from
    every rank / ``#1`` column, whether they are FL methods or backbone swaps.
    """
    complete = [m for m in channel
                if sum(1 for d in DATASETS if d in channel[m]['mse']) == 9]
    partial = [m for m in channel if m not in complete]
    ours = [as_row(m, channel[m]) for m in OURS_CH if m in channel]
    fed = [as_row(m, channel[m]) for m in complete
           if m not in OURS_CH and m != 'local']
    ref = [as_row(m, channel[m]) for m in ('local',) if m in channel]
    agg: list = []
    short: list = []

    def _n(rec: dict) -> int:
        return sum(1 for d in DATASETS
                   if d in rec['mse'] and not nan(rec['mse'][d]))

    for name, rec in (bb or {}).items():
        if name == 'dtmoe':
            continue                       # dtmoe *is* the `local` row
        row = as_row('local+' + name, rec, bb_venue(name))
        if _n(rec) < 9:
            row['_partial'] = True
            short.append(row)
        else:
            ref.append(row)
    for (meth, name), rec in (fedbb or {}).items():
        row = as_row(f'{meth}+{name}', rec,
                     f'{PT.VENUE.get(meth, "—")} + {name}')
        if _n(rec) < 9:
            row['_partial'] = True
            short.append(row)
        else:
            agg.append(row)
    dual_rank(ours + fed + agg + ref)
    ctrl = [as_row(m, channel[m]) for m in partial]
    for r in ctrl:
        r['_partial'] = True
    ctrl = sorted(ctrl + short, key=lambda r: r['name'])
    return ours, fed, agg, ref, ctrl


def build_big1(channel: dict, bb: dict | None = None,
               fedbb: dict | None = None) -> dict:
    ours, fed, agg, ref, ctrl = channel_pool(channel, bb, fedbb)
    ours.sort(key=lambda r: r['_rs'])
    fed.sort(key=lambda r: r['_rs'])
    agg.sort(key=lambda r: r['_rs'])

    specs = [{'kind': 'pair', 'label': d, 'label_tex': d, 'ds': [d]}
             for d in DATASETS]
    specs += [{'kind': 'pair', 'label': 'Core (8)',
               'label_tex': r'\textbf{Core (8)}', 'ds': CORE}]
    specs += rank_cols()

    note = [
        '**Protocol** `channel`: one sensor per client (`enc_in = 1`), '
        '`sl96 -> pl24`, `d_model 32`, 2 rounds x 800 steps = 1600 gradient '
        'steps per client, one aggregation, AdamW 5e-4 / batch 32, split 7:1:2. '
        'Seed 2021.',
        '**Every dataset has two columns, `MSE` and `MAE`, ranked and marked '
        'separately.** **Bold** = best in that column, <u>underline</u> = second '
        'best; the marks are computed inside one column, never across the pair -- '
        'so the MSE winner and the MAE winner of a dataset can be different '
        'methods, and on Type I they are.',
        REF_NOTE,
        BACKBONE_NOTE,
        DEGENERATE_NOTE,
        GPU_NOTE,
        '**The `Rank` columns use exact values -- no tie band.** `BIG0` reports '
        'what the same order becomes when a 0.5%-5% relative band is allowed: '
        '`FedTDO` stays first throughout and the best baseline never overtakes '
        'it, but which of our three variants leads is not resolvable on one '
        'seed. Read `Rank Σ` for the gap to a baseline, not for the gap '
        'between our own rows.',
        '`Core (8)` = mean over the 8 datasets excluding ILI (million-scale '
        'counts, kept visible but out of the ranking).',
        RANK_NOTE,
        'The appendix control rows (`fedavg_sgd` 8/9, `local+transformer` 1/9) '
        'are excluded from every rank and `#1` column and print dashes there '
        'instead of an average over the cells they happen to have.',
        '`*` = that cell was re-run with a tuned learning rate; the shared '
        'schedule diverged to NaN on Weather for `scaffold` and `fedavg_sgd`. '
        'It is a fairness footnote, not a win: those rows still lose, and the '
        'tuning only keeps them on the board.',
        '`re-impl.` = reference implementation (fingerprint clustering + mixup '
        'for `fedda`, fingerprint graph aggregation for `fedgtp`), **not** a '
        'verbatim reproduction -- say so when citing these two rows.',
        FEDBB_NOTE,
        'Single seed: differences under ~1% (most of Type I) are inside seed '
        'noise and must be written as ties.',
        '**Appendix A, printed after the table, is an unranked index of *every* '
        'implemented method on both protocols** (including the two cross-domain '
        'LM models, `FFTS` and `Time-FFM`, which exist only on the multivariate '
        'protocol). The table above stays `channel`-only; the appendix is there '
        'so nobody has to open four files to find out whether a method was run '
        'at all.',
    ]
    sections = [('FedTDO (ours)', ours),
                ('Federated baselines', fed)]
    if agg:
        sections.append(('Federated backbone variant -- `fedavg` on a swapped '
                         'per-client model', agg))
    sections += [('Centralised reference (no federation, not a competitor)',
                  sorted(ref, key=lambda r: r['_rs']))]
    if ctrl:
        sections.append(('Appendix control (incomplete, not ranked)', ctrl))
    return grid_table('BIG1', 'Table 1 · Main results, `channel` protocol '
                      '(9 datasets, horizon 24)', 'Main results on the '
                      r'\texttt{channel} protocol. MSE and MAE are separate '
                      'columns; bold = best, underline = second best in that '
                      'column. Rows sorted by Rank $\\Sigma$ = Rank MSE + '
                      'Rank MAE.', note, sections, specs)


# --------------------------------------------------------------------------- #
# BIG2 -- the same rows, columns collapsed onto data types
# --------------------------------------------------------------------------- #
def build_big2(channel: dict, bb: dict | None = None,
               fedbb: dict | None = None) -> dict:
    ours, fed, agg, ref, _ = channel_pool(channel, bb, fedbb)
    ours.sort(key=lambda r: r['_rs'])
    fed.sort(key=lambda r: r['_rs'])
    agg.sort(key=lambda r: r['_rs'])

    specs = []
    for tag, label, ds, _x in TYPES:
        specs.append({'kind': 'pair', 'label': f'Type {tag}',
                      'label_tex': f'Type {tag}', 'ds': ds})
    specs += [{'kind': 'pair', 'label': 'Core (8)',
               'label_tex': r'\textbf{Core (8)}', 'ds': CORE}]
    specs += rank_cols()

    note = [
        '**Same rows, same numbers as Table 1** -- each column is the mean over '
        'the datasets of that type, so the nine cells collapse to four, and each '
        'of the four still carries MSE and MAE separately. '
        '`Core (8)` = Type I (5) + Type II (2) + Type III (1).',
        '**Type I** low-dimensional strong periodicity (ETTh1/2, ETTm1/2, '
        'Weather). **Type II** high-dimensional truncated to 7 channels '
        '(ECL 321->7, Traffic 862->7) -- channel selection is a confounder here. '
        '**Type III** no periodicity (Exchange). **Type IV** scale outlier (ILI, '
        'million-scale counts) -- excluded from `core` so it cannot decide the '
        'ranking.',
        'Bold/underline are per column, i.e. per (data type, metric) -- that is '
        'the question this table answers: *which family of series* a method is '
        'good at, and whether it is good at it on MSE, on MAE, or on both.',
        REF_NOTE,
        BACKBONE_NOTE,
        FEDBB_NOTE,
        RANK_NOTE,
        'This is a **mean of means**, so it is not a per-cell ranking: a method '
        'can top a type here without winning any single dataset in it. '
        'Table 1 has the cells.',
    ]
    sections = [('FedTDO (ours)', ours), ('Federated baselines', fed)]
    if agg:
        sections.append(('Federated backbone variant -- `fedavg` on a swapped '
                         'per-client model', agg))
    sections.append(('Centralised reference (no federation, not a competitor)',
                     sorted(ref, key=lambda r: r['_rs'])))
    return grid_table('BIG2', 'Table 2 · Accuracy by data type, `channel` '
                      'protocol', 'Accuracy grouped by data type, MSE and MAE '
                      'separately. Bold = best, underline = second best in that '
                      '(type, metric) column.', note, sections, specs)


# --------------------------------------------------------------------------- #
# BIG3 -- multivariate protocol
# --------------------------------------------------------------------------- #
def build_big3(multi: dict, fedtdo: dict, fedal: dict) -> dict:
    loc = {b: r for (m, b), r in multi.items() if m == 'local'}
    fed = {b: r for (m, b), r in multi.items() if m == 'fedavg'}
    ours = [as_row(OURS_MV[a], fedtdo[a]) for a in ('center', 'decorr', 'base')
            if a in fedtdo]
    fedal_rows = [as_row('fedal(FeDaL)', r) for r in fedal.values()]
    for r in fedal_rows:
        r['venue'] = 'FeDaL (2025)'
    back = [as_row(f'local+{b}', loc[b]) for b in loc]
    for r in back:
        r['venue'] = 'no federation'

    pool = ours + fedal_rows + back
    dual_rank(pool)
    ours.sort(key=lambda r: r['_rs'])
    back.sort(key=lambda r: r['_rs'])
    fedal_rows.sort(key=lambda r: r['_rs'])

    specs = [{'kind': 'pair', 'label': d, 'label_tex': d, 'ds': [d]}
             for d in DATASETS]
    specs += [{'kind': 'pair', 'label': 'Core (8)',
               'label_tex': r'\textbf{Core (8)}', 'ds': CORE}]
    specs += rank_cols()

    by_sig = sorted(pool, key=lambda r: r['_rs'])
    pos = [r['name'] for r in by_sig].index('FedTDO-center') + 1
    above = [r['name'] for r in by_sig[:pos - 1]]
    # external = neither ours nor a centralised backbone
    ext = [r['name'] for r in by_sig
           if not r['name'].startswith('local+')
           and not r['name'].startswith('FedTDO')]
    best_ext_fed = ext[0] if ext else '—'
    fed_rows = [r['name'] for r in by_sig if not r['name'].startswith('local+')]
    pos_fed = fed_rows.index('FedTDO-center') + 1
    ctr = next(r for r in pool if r['name'] == 'FedTDO-center')
    dec = next((r for r in pool if r['name'] == 'FedTDO-decorr'), None)

    def _agg(b: str) -> tuple:
        l, f = loc[b], fed[b]
        dm = mean_of(f, 'mse', CORE) / mean_of(l, 'mse', CORE) - 1
        da = mean_of(f, 'mae', CORE) / mean_of(l, 'mae', CORE) - 1
        return f'{100*dm:+.0f}% / {100*da:+.0f}%'
    a_ffts, a_ffm = _agg('ffts'), _agg('timeffm')

    note = [
        '**Protocol multivariate**: a client is a *dataset*, `features M`, '
        '`enc_in = 7`, `sl96 -> pl96`, 1600 gradient steps per client. '
        '**Not comparable with Tables 1/2** (different client definition, '
        'different horizon).',
        '`local+*` = trained alone, no federation. **This block is not a '
        'baseline class, it is the ceiling**: with no aggregation at all it sets '
        'what each backbone can reach under the same budget.',
        '`fedal(FeDaL)` retrains the *same* FFTS adapter with FeDaL-style '
        'self-supervised pretraining + drift correction instead of supervised '
        'FedAvg. **Time-series backbones are deliberately not federated** '
        '(project constraint §0b): they appear only as `local+*`.',
        RANK_NOTE,
        f'On `Rank Σ` (the sort key), rank 1 overall is `{by_sig[0]["name"]}` at '
        f'{by_sig[0]["_rs"]:.2f}; FedTDO-center is **#{pos} of {len(pool)}** at '
        f'{ctr["_rs"]:.2f}'
        + (f', behind {", ".join("`" + a + "`" for a in above)}. ' if above
           else '. ')
        + f'Among federated rows it is #{pos_fed} of {len(fed_rows)}, ahead of '
        f'`{best_ext_fed}`. '
        + ('Ranking the two metrics separately moves the three FedTDO arms: '
           f'`FedTDO-decorr` has the better Σ ({dec["_rs"]:.2f}) than '
           f'`center` ({ctr["_rs"]:.2f}), while the core means differ by less '
           'than 0.0002 in favour of `center` -- i.e. inside seed noise. '
           if dec is not None else '')
        + 'Read this as "federated training matches centralised", never as '
        '"beats every backbone".',
        '`local+tfps` is under-converged at 1600 steps: treat that row as a '
        'budget lower bound, not as the architecture\'s ability. The two '
        'frozen-LM backbones (`local+ffts`, `local+timeffm`) are the strongest '
        'rows on Type II while centralised, and their aggregation cost on '
        f'core (MSE / MAE) is {a_ffts} and {a_ffm} -- the same as a plain '
        'transformer. Table A1 has the breakdown.',
    ]
    sections = [('FedTDO (ours)', ours),
                ('FeDaL-style training', fedal_rows),
                ('Centralised backbones (no federation)', back)]
    return grid_table('BIG3', 'Table 3 · Main results, multivariate protocol '
                      '(9 datasets, horizon 96)', 'Main results on the '
                      'multivariate protocol. MSE and MAE are separate columns; '
                      'bold = best, underline = second best in that column.',
                      note, sections, specs)


# --------------------------------------------------------------------------- #
# BIG4 -- accuracy and price in one table
# --------------------------------------------------------------------------- #
def build_big4(channel: dict, probe: dict) -> dict:
    """``probe`` is the *single* measured client model of the ``channel``
    protocol (all FL baselines share one architecture there), not a per-method
    dict -- see ``xc_cost_probe.channel_models``."""
    clients = CT.load_num_clients()
    ft = CT.load_fedtdo_cost()

    def wire_mib(method: str) -> float:
        if method in CT.FEDTDO_ARMS:
            return 2 * CT.ENCODER_ELEMS * 4 / CT.MIB
        up, down = CT.wire_elems(method, probe)
        return (up + down) * 4 / CT.MIB

    def comm(method: str, ds: str) -> float:
        arm = CT.FEDTDO_ARMS.get(method)
        if arm is None:
            return clients.get(ds, 0) * 2 * wire_mib(method)
        cap = ft.get((ds, arm), {})
        return (clients.get(ds, 0) * CT.FEDTDO_ROUNDS * wire_mib(method)
                + (cap.get('download', 0) + cap.get('upload', 0)) / CT.MIB)

    ours, fed, _agg, ref, _ = channel_pool(channel)
    pool = ours + fed + ref

    for r in pool:
        r['_comm'] = {t[0]: sum(comm(r['name'], d) for d in t[2]) / len(t[2])
                      for t in TYPES}
        r['_comm9'] = sum(comm(r['name'], d) for d in DATASETS) / len(DATASETS)
        r['_wire'] = wire_mib(r['name'])
        r['_core_m'] = mean_of(r, 'mse', CORE)
        r['_core_a'] = mean_of(r, 'mae', CORE)
        # accuracy per byte, on each metric separately
        r['_eff_m'] = r['_core_m'] * r['_comm9']
        r['_eff_a'] = r['_core_a'] * r['_comm9']

    base_m = next(r['_eff_m'] for r in pool if r['name'] == 'FedTDO-center')
    base_a = next(r['_eff_a'] for r in pool if r['name'] == 'FedTDO-center')
    for r in pool:
        r['_eff_m'] /= base_m
        r['_eff_a'] /= base_a

    order = sorted(pool, key=lambda r: r['_rs'])
    ours_s = [r for r in order if r['name'] in OURS_CH]
    fed_s = [r for r in order if r['name'] not in OURS_CH and r['name'] != 'local']
    ref_s = [r for r in order if r['name'] == 'local']

    # every cost column is marked against the real methods only: ``local``
    # sends nothing, so its 0 MiB would take every cost mark
    EX = {'excl': ('local',)}
    specs = [
        {'kind': 'one', 'label': 'Wire (MiB)', 'label_tex': 'Wire (MiB)',
         'grp': 'Per client', 'grp_tex': r'\textbf{Per client}',
         'sub': 'wire/rnd', 'func': lambda r: r['_wire'], 'fmt': mib, **EX},
        {'kind': 'one', 'label': 'Comm I', 'label_tex': 'Comm I',
         'grp': 'Comm / dataset (MiB)', 'grp_tex': r'\textbf{Comm/dataset (MiB)}',
         'sub': 'Type I', 'func': lambda r: r['_comm']['I'], 'fmt': mib, **EX},
        {'kind': 'one', 'label': 'Comm II', 'label_tex': 'Comm II',
         'grp': 'Comm / dataset (MiB)', 'sub': 'Type II',
         'func': lambda r: r['_comm']['II'], 'fmt': mib, **EX},
        {'kind': 'one', 'label': 'Comm III', 'label_tex': 'Comm III',
         'grp': 'Comm / dataset (MiB)', 'sub': 'Type III',
         'func': lambda r: r['_comm']['III'], 'fmt': mib, **EX},
        {'kind': 'one', 'label': 'Comm IV', 'label_tex': 'Comm IV',
         'grp': 'Comm / dataset (MiB)', 'sub': 'Type IV',
         'func': lambda r: r['_comm']['IV'], 'fmt': mib, **EX},
        {'kind': 'one', 'label': 'Comm avg', 'label_tex': 'Comm avg',
         'grp': 'Comm / dataset (MiB)', 'sub': 'avg (9)',
         'func': lambda r: r['_comm9'], 'fmt': mib, **EX},
        {'kind': 'pair', 'label': 'Core (8)', 'label_tex': r'\textbf{Core (8)}',
         'ds': CORE},
        {'kind': 'one', 'label': 'MSE×MiB', 'label_tex': r'MSE$\times$MiB',
         'grp': 'Accuracy per byte', 'grp_tex': r'\textbf{Accuracy per byte}',
         'sub': 'MSE', 'func': lambda r: r['_eff_m'], 'fmt': f2,
         'na': ('local',), **EX},
        {'kind': 'one', 'label': 'MAE×MiB', 'label_tex': r'MAE$\times$MiB',
         'grp': 'Accuracy per byte', 'sub': 'MAE',
         'func': lambda r: r['_eff_a'], 'fmt': f2, 'na': ('local',), **EX},
    ]
    specs += rank_cols()

    note = [
        '**Price and accuracy on the same rows as Table 1**, ranked by the same '
        '`Rank Σ`. `Wire` = `state_dict()` bytes per client per round, not '
        '`params x 4`, because buffers cross the network too. `Comm/dataset` = '
        '`clients x rounds x (upload + download) x 4 B`, averaged inside the '
        'data type.',
        'Rounds: baselines 2, **FedTDO 1** (it syncs the encoder once, then '
        'ships expert capsules, which the run itself measures and which are '
        'counted here). Charging FedTDO the same two encoder syncs the '
        'baselines pay still leaves it cheapest on every data type -- the '
        'counterfactual is in `T8-8d`.',
        '**`MSE×MiB` / `MAE×MiB`** = core MSE (resp. MAE) x mean MiB per '
        'dataset, each normalised to FedTDO-center = 1.00. Lower is better: it '
        'is the only single number that ranks accuracy *per byte*, and it is '
        'given for both metrics because a method can be cheap and mediocre on '
        'MSE while being cheap and good on MAE.',
        '`local` exchanges nothing: it is the origin of the cost axis, not a '
        'competitor, and it is excluded from every cost mark. The other '
        '`Centralised reference` rows of Table 1 (`local+patchtst/stmtm/tfps/'
        'raft`) sit on that same origin -- they never send anything either, so '
        'they are not repeated here: they are the right answer to "is '
        'federation worth it" and the wrong answer to "how cheap is it".',
        'Baseline rows share one client architecture in this protocol, so their '
        'cost is identical **by construction** except where the algorithm '
        'changes the payload: `scaffold` ships a control variate (x1.4), '
        '`fedrep`/`fedper`/`fedpkda` upload the trunk only (-1%), FedTDO syncs '
        'the encoder only. The cost axis therefore separates *protocols*, not '
        'methods -- which is exactly the point: FedTDO wins the cost axis '
        'because of the protocol, and the accuracy axis has to be read '
        'separately.',
    ]
    sections = [('FedTDO (ours)', ours_s), ('Federated baselines', fed_s),
                ('Reference (no federation)', ref_s)]
    return grid_table('BIG4', 'Table 4 · Accuracy and communication cost, '
                      '`channel` protocol', 'Accuracy against communication '
                      'cost on the same rows; MSE and MAE ranked separately, '
                      'cost broken down by data type.', note, sections, specs)


# --------------------------------------------------------------------------- #
# BIG5 -- the price of one round of averaging (appendix)
# --------------------------------------------------------------------------- #
def build_big5(multi: dict, probe: dict) -> dict:
    loc = {b: r for (m, b), r in multi.items() if m == 'local'}
    fed = {b: r for (m, b), r in multi.items() if m == 'fedavg'}
    backs = sorted(set(loc) & set(fed), key=lambda b: mean_of(loc[b], 'mse', CORE))

    rows = []
    for b in backs:
        r = {'name': b, 'venue': ''}
        r['l_mse'] = {t[0]: mean_of(loc[b], 'mse', t[2]) for t in TYPES}
        r['l_mae'] = {t[0]: mean_of(loc[b], 'mae', t[2]) for t in TYPES}
        r['f_mse'] = {t[0]: mean_of(fed[b], 'mse', t[2]) for t in TYPES}
        r['f_mae'] = {t[0]: mean_of(fed[b], 'mae', t[2]) for t in TYPES}
        r['dl_m'] = mean_of(loc[b], 'mse', CORE)
        r['df_m'] = mean_of(fed[b], 'mse', CORE)
        r['dl_a'] = mean_of(loc[b], 'mae', CORE)
        r['df_a'] = mean_of(fed[b], 'mae', CORE)
        r['dp_m'] = r['df_m'] / r['dl_m'] - 1
        r['dp_a'] = r['df_a'] / r['dl_a'] - 1
        r['_wire'] = probe.get(b, {}).get('wire_bytes', 0) * 2 / CT.MIB
        rows.append(r)

    # one group per data type, four columns: local / fedavg x MSE / MAE.
    # The only bold in this table means "averaging helped", so the ``local``
    # columns carry no mark at all -- otherwise two different meanings of bold
    # would sit in the same row.
    specs = []
    for tag, _lab, _ds, _x in TYPES:
        for sub, key, helped in (
                ('local MSE', 'l_mse', False), ('local MAE', 'l_mae', False),
                ('fedavg MSE', 'f_mse', True), ('fedavg MAE', 'f_mae', True)):
            sp = {'kind': 'one',
                  'label': f'Type {tag} {sub}',
                  'label_tex': f'Type {tag} {sub}',
                  'grp': f'Type {tag}', 'sub': sub,
                  'func': (lambda k, t: lambda r: r[k][t])(key, tag),
                  'fmt': f4, 'nodeco': not helped}
            if helped:
                sp['mkfn'] = (lambda k, t: lambda r: (
                    'best' if r[k][t] < r['l_' + k[2:]][t] - 1e-12 else None))(
                        key, tag)
            specs.append(sp)
    specs += [
        {'kind': 'one', 'label': 'Δcore MSE', 'label_tex': r'$\Delta$ core MSE',
         'grp': 'Δ core (8)', 'grp_tex': r'$\Delta$ core (8)', 'sub': 'MSE',
         'func': lambda r: 100 * r['dp_m'], 'fmt': lambda v: f'{v:+.0f}%',
         'nodeco': True},
        {'kind': 'one', 'label': 'Δcore MAE', 'label_tex': r'$\Delta$ core MAE',
         'grp': 'Δ core (8)', 'sub': 'MAE',
         'func': lambda r: 100 * r['dp_a'], 'fmt': lambda v: f'{v:+.0f}%',
         'nodeco': True},
        {'kind': 'one', 'label': 'Wire (MiB)', 'label_tex': 'Wire (MiB)',
         'grp': 'Cost', 'grp_tex': r'\textbf{Cost}', 'sub': 'wire/cli/rnd',
         'func': lambda r: r['_wire'], 'fmt': mib, 'nodeco': True},
    ]
    for sp in specs:
        sp.setdefault('grp_tex', sp.get('grp', ''))

    dps_m = sorted(100 * r['dp_m'] for r in rows)
    dps_a = sorted(100 * r['dp_a'] for r in rows)
    note = [
        '**Appendix, not a main table.** Identical data, identical budget, '
        'identical optimiser as `local` -- the only difference is one round of '
        'parameter averaging. The delta is therefore the **price of aggregation '
        'itself**, not a property of the backbone.',
        f'**No backbone survives it.** On `core (8)` MSE degrades by '
        f'{dps_m[0]:+.0f}% to {dps_m[-1]:+.0f}% (median '
        f'{dps_m[len(dps_m)//2]:+.0f}%) and MAE by {dps_a[0]:+.0f}% to '
        f'{dps_a[-1]:+.0f}% (median {dps_a[len(dps_a)//2]:+.0f}%), across all '
        f'{len(rows)} architectures. A method that looks weak after FedAvg is '
        'therefore not necessarily a weak model.',
        '**Bold has exactly one meaning in this table: averaging helped** on '
        'that (type, metric). It happens on Type III (Exchange) for several '
        'backbones and almost nowhere else -- on a near-random-walk series '
        'there is little client-specific signal to destroy. The `local` '
        'columns are deliberately unmarked: they are the reference, not '
        'contestants.',
        'MSE and MAE can disagree on the sign of the aggregation price -- both '
        'deltas are printed for that reason. Quoting only the better-looking '
        'one would be reading the aggregator, not the model.',
        'Rows are centralised training: **no time-series backbone is federated '
        'in the main tables** (project constraint §0b).',
    ]
    sections = [('Backbone', rows)]
    return grid_table('BIG5', 'Table A1 · Price of one round of averaging '
                      '(multivariate protocol, appendix)',
                      'Centralised vs one FedAvg round, same budget, MSE and MAE '
                      'reported separately.', note, sections, specs, venue=False)


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
CSS = """
:root{--ink:#1a1a1a;--mute:#6b6b6b;--rule:#c9c9c9;--hair:#e6e6e6;--acc:#7a1f1f;}
*{box-sizing:border-box}
body{margin:0;padding:36px 20px 96px;background:#f7f6f3;color:var(--ink);
 font:15px/1.55 "Iowan Old Style","Palatino Linotype",Palatino,Georgia,
 "Songti SC","Source Han Serif SC",serif}
.wrap{max-width:1700px;margin:0 auto}
h1{font-size:22px;font-weight:600;margin:0 0 6px;letter-spacing:.2px}
.sub{color:var(--mute);font-size:13px;margin-bottom:36px;max-width:110ch}
figure{margin:0 0 64px;background:#fff;border:1px solid var(--hair);
 border-radius:3px;padding:24px 22px 16px;box-shadow:0 1px 2px rgba(0,0,0,.04)}
figcaption{font-size:13.5px;margin:0 0 16px;line-height:1.6;max-width:96ch}
figcaption b{font-weight:600}
.tbl{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:10.4px;
 font-variant-numeric:tabular-nums;
 font-family:"SF Mono",ui-monospace,Menlo,Consolas,monospace}
th,td{padding:3px 4px;text-align:right;white-space:nowrap}
thead th{font-weight:600;border-top:1.6px solid var(--ink);
 border-bottom:.9px solid var(--ink);font-size:11.6px}
thead tr.sup th{border-top:none;border-bottom:none;padding-bottom:1px;
 font-size:10.6px;color:#333}
thead tr.sub th{border-top:none;padding-top:0;font-size:10.4px;font-weight:600;
 color:#333;border-bottom:.9px solid var(--ink)}
th.l,td.l{text-align:left}
tbody tr.sec td{font-weight:600;font-style:italic;padding-top:13px;
 border-bottom:.6px solid var(--hair);font-family:inherit;font-size:12.4px}
tbody tr:last-child td{border-bottom:1.6px solid var(--ink)}
tbody td{border-bottom:.5px solid #f1f1f1}
b,strong{font-weight:700}
u{text-decoration:underline;text-underline-offset:2px}
.note{font-size:12.2px;color:#3a3a3a;line-height:1.7;margin-top:14px;
 border-top:.6px solid var(--hair);padding-top:12px;max-width:118ch}
.note ul{margin:6px 0 0;padding-left:18px}
.note li{margin:3px 0}
code{background:#f0eee9;padding:1px 4px;border-radius:2px;font-size:11.6px}
"""


def md_inline(s: str) -> str:
    """Minimal inline markdown -> HTML: **bold**, `code`."""
    while '**' in s:
        i = s.index('**')
        j = s.index('**', i + 2)
        s = s[:i] + '<strong>' + s[i + 2:j] + '</strong>' + s[j + 2:]
    while '`' in s:
        i = s.index('`')
        j = s.index('`', i + 1)
        s = s[:i] + '<code>' + s[i + 1:j] + '</code>' + s[j + 1:]
    return s


def html_cell(md: str) -> str:
    return md_inline(md).replace('|', '&#124;')


def render_html(tables: list) -> str:
    parts = [f'<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">'
             f'<meta name="viewport" content="width=device-width,initial-scale=1">'
             f'<title>FedTDO · paper tables</title><style>{CSS}</style></head><body>'
             f'<div class="wrap"><h1>FedTDO &middot; 论文级大表</h1>'
             f'<div class="sub">同一份数据（seed 2021）的纸面布局。'
             f'<b>MSE 与 MAE 各占一列</b>，各自排名、各自加粗；'
             f'<code>Rank Σ</code> = 两个指标分别排名后相加，按它排序；'
             f'<code>#1 Σ</code> = 两个指标各自拿第一的次数相加。'
             f'数字与 T1 / T2 / T8 完全一致，仅换排版。'
             f'由 <code>scripts/xc_big_tables.py</code> 生成。</div>']

    for t in tables:
        n = len(t['header'])
        nlead = len(t['lead'])
        has_sup = t.get('has_sup')
        parts.append(f'<figure><figcaption><b>{md_inline(t["title"])}</b>'
                     f'</figcaption><div class="tbl"><table><thead>')
        if has_sup:
            parts.append('<tr class="sup">')
            for _ in range(nlead):
                parts.append('<th class="l"></th>')
            for label, label_tex, k in t['sup']:
                if k > 1:
                    parts.append(f'<th colspan="{k}">{md_inline(label)}</th>')
                else:
                    parts.append('<th></th>')
            parts.append('</tr><tr class="sub">')
            for i, h in enumerate(t['lead']):
                parts.append(f'<th class="l">{h}</th>')
            for h in t['sub']:
                parts.append(f'<th>{md_inline(h)}</th>')
            parts.append('</tr>')
        else:
            parts.append('<tr>')
            for i, h in enumerate(t['header']):
                cls = ' class="l"' if i < nlead else ''
                parts.append(f'<th{cls}>{md_inline(h)}</th>')
            parts.append('</tr>')
        parts.append('</thead><tbody>')
        for kind, a, b in t['rows']:
            if kind == 'sec':
                parts.append(f'<tr class="sec"><td class="l" colspan="{n}">{a}</td></tr>')
            else:
                cells = []
                for i, c in enumerate(a):
                    cls = ' class="l"' if i < nlead else ''
                    cells.append(f'<td{cls}>{html_cell(c)}</td>')
                parts.append('<tr>' + ''.join(cells) + '</tr>')
        parts.append('</tbody></table></div><div class="note"><ul>')
        for x in t['note']:
            parts.append(f'<li>{md_inline(x)}</li>')
        parts.append('</ul></div></figure>')
    parts.append('</div></body></html>')
    return ''.join(parts)


# --------------------------------------------------------------------------- #
def slug(title: str) -> str:
    for key, name in (('Table 0', 'rank_summary'),
                      ('Main results, `channel`', 'channel'),
                      ('Accuracy by data type', 'by_datatype'),
                      ('Main results, multivariate', 'multivariate'),
                      ('Accuracy and communication', 'cost'),
                      ('Price of one round', 'aggregation')):
        if key in title:
            return name
    return 'table'


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='', help='comma separated ids, e.g. BIG4,BIG5')
    ap.add_argument('--html', default='BIG_TABLES.html', help='html file name')
    args = ap.parse_args()

    channel = PT.load_channel()
    multi = PT.load_multi()
    fedtdo = PT.load_fedtdo_multi()
    fedal = PT.load_fedal()
    blob = json.loads((ROOT / 'results_paper' / 'cost_probe.json')
                      .read_text(encoding='utf-8'))
    probe = blob['channel']['dtmoe']
    mv_probe = blob['multivariate']

    OUT.mkdir(parents=True, exist_ok=True)
    TEX.mkdir(parents=True, exist_ok=True)

    bb = PT.load_channel_backbones()
    fedbb = PT.load_channel_backbones_fed()
    tables = [build_big0(channel, bb, fedbb), build_big1(channel, bb, fedbb),
              build_big2(channel, bb, fedbb),
              build_big3(multi, fedtdo, fedal),
              build_big4(channel, probe), build_big5(multi, mv_probe)]
    if args.only:
        keep = {x.strip().upper() for x in args.only.split(',') if x.strip()}
        tables = [t for t in tables if t['id'] in keep]
    # BIG1 additionally carries an unranked index of *every* implemented method
    # on both protocols -- markdown only, the .tex stays the paper table
    extra = {'BIG1': inventory_md(channel, bb, fedbb, multi, fedtdo, fedal)}
    for t in tables:
        md = render_md(t) + extra.get(t['id'], [])
        (OUT / f'{t["id"]}_{slug(t["title"])}.md').write_text(
            '\n'.join(md) + '\n', encoding='utf-8')
        write_tex(t, TEX / f'{t["id"]}.tex')
    (OUT / args.html).write_text(render_html(tables), encoding='utf-8')
    print(f'wrote {len(tables)} tables -> {OUT}')


if __name__ == '__main__':
    main()
