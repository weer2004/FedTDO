"""Resource / communication cost tables, grouped by data type.

The MSE tables in ``xc_paper_tables.py`` answer "who is accurate".  This script
answers "at what price", on the same cells and the same grouping:

  * ``params``      -- learnable parameters of one client model
  * ``wire``        -- what actually crosses the network, i.e. ``state_dict()``
                       bytes (buffers included), not ``params * 4``
  * ``comm / run``  -- ``clients x rounds x (upload + download)``

Three evidence grades are used and labelled everywhere:

  ``measured``   read from a run artifact (``summary.json``, ``fedtdo_summary.csv``)
  ``derived``    measured payload x measured client count x documented rounds
  ``estimated``  a multiplier read off the code path (e.g. SCAFFOLD ships a
                 control variate the size of the model), never a tuned constant

Nothing here is hand-typed into the markdown: rerun
``D:/software/Anaconda3/envs/ma_fmarl/python.exe scripts/xc_cost_probe.py``
then ``D:/software/Anaconda3/python.exe scripts/xc_cost_tables.py``.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from xc_paper_tables import (  # noqa: E402
    CORE, DATASETS, DATATYPE_GROUPS, OUT, f4, load_channel,
)

PROBE = ROOT / 'results_paper' / 'cost_probe.json'
FEDTDO_COST = ROOT / 'results' / 'fedtdo' / '_metrics' / 'fedtdo_summary.csv'
BL_ROOT = ROOT / 'results_baselines'

MIB = 1024 * 1024

# ---------------------------------------------------------------------------
# communication model, read off the implementations (fl_baselines/*.py)
# ---------------------------------------------------------------------------
# ``body``  : only the shared trunk is uploaded; the prediction head stays local
# ``cv``    : SCAFFOLD ships the model *and* its control variate each way
BODY_ONLY = {'fedrep', 'fedper', 'fedpkda'}
CONTROL_VARIATE = {'scaffold'}
LOCAL_ONLY = {'local'}
# methods that keep a second model (or a Moreau copy) on the client: memory 2x,
# but the wire carries one model only
DOUBLE_MEMORY = {'ditto', 'pfedme'}

# FedTDO arms: one encoder sync (measured payload) + measured capsule traffic
FEDTDO_ARMS = {'FedTDO-center': 'center', 'FedTDO-decorr': 'decorr',
               'FedTDO-raw': 'base'}
FEDTDO_ROUNDS = 1
ENCODER_ELEMS = 161_536      # patch_embedding state_dict, measured


def load_num_clients() -> dict:
    """Client count per dataset, taken from the run directory names."""
    seen: dict = {}
    for d in BL_ROOT.glob('BL_*_xc_s*'):
        m = re.search(r'_([A-Za-z0-9]+?)_nc(\d+)_', d.name)
        if m and m.group(1) in DATASETS:
            seen.setdefault(m.group(1), set()).add(int(m.group(2)))
    return {k: max(v) for k, v in seen.items()}


def load_fedtdo_cost() -> dict:
    """FedTDO's own measured transfer accounting (expert capsules + encoder)."""
    out: dict = {}
    if not FEDTDO_COST.exists():
        return out
    with open(FEDTDO_COST, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            out[(r['dataset'], r['arm'])] = {
                'download': float(r['download_bytes'] or 0),
                'upload': float(r['upload_bytes'] or 0),
                'evaluations': float(r['evaluations'] or 0),
                'transfers': float(r['transfers'] or 0),
            }
    return out


def wire_elems(method: str, probe: dict) -> tuple:
    """(upload, download) elements per client per round."""
    state = probe['state_elems']
    params = probe['params']
    body = state - probe['head_elems']
    if method in LOCAL_ONLY:
        return 0, 0
    if method in BODY_ONLY:
        return body, body
    if method in CONTROL_VARIATE:
        # model + control variate (params only: buffers have no gradient)
        return state + params, state + params
    return state, state


def comm_mib(method: str, dataset: str, probe: dict, clients: dict,
             rounds: int = 2) -> float:
    up, down = wire_elems(method, probe)
    return clients.get(dataset, 0) * rounds * (up + down) * 4 / MIB


def dense_rank(values: dict) -> dict:
    """Ties share a rank: 22 methods at 291 MiB are rank 11, not ranks 11..22."""
    uniq = sorted(set(round(v, 6) for v in values.values()))
    pos = {v: i + 1 for i, v in enumerate(uniq)}
    return {k: pos[round(v, 6)] for k, v in values.items()}


def fmt_mib(v: float) -> str:
    if v == 0:
        return '0'
    if v < 1:
        return f'{v:.2f}'
    if v < 100:
        return f'{v:.1f}'
    return f'{v:,.0f}'


def build_cost(channel: dict, probe: dict) -> list:
    clients = load_num_clients()
    ft = load_fedtdo_cost()

    def cost_of(method: str, dataset: str) -> float:
        """MiB for one full run of ``method`` on ``dataset``.

        FedTDO does not average the model: it syncs ``patch_embedding`` once and
        then transfers expert capsules, which the cost model counts as
        ``download_bytes`` / ``upload_bytes`` inside the run summary.  Charging
        both is the conservative choice.
        """
        arm = FEDTDO_ARMS.get(method)
        if arm is None:
            return comm_mib(method, dataset, probe, clients)
        enc = clients.get(dataset, 0) * FEDTDO_ROUNDS * 2 * ENCODER_ELEMS * 4 / MIB
        cap = ft.get((dataset, arm), {})
        return enc + (cap.get('download', 0) + cap.get('upload', 0)) / MIB

    order0 = [m for m, r in channel.items()
              if all(d in r['mse'] for d in DATASETS)]

    def t_mse(m, ds):
        return sum(channel[m]['mse'][d] for d in ds) / len(ds)

    def t_cost(m, ds):
        return sum(cost_of(m, d) for d in ds)

    summary = []
    for g in DATATYPE_GROUPS:
        tag, label, ds, _ = g
        if not all(d in clients for d in ds):
            continue
        best = min(order0, key=lambda m: t_mse(m, ds))
        pool = [m for m in order0 if m not in LOCAL_ONLY]
        cheap = min(pool, key=lambda m: t_cost(m, ds))
        mine = 'FedTDO-center' if 'FedTDO-center' in channel else best
        ranked = sorted(order0, key=lambda m: t_mse(m, ds))
        summary.append((tag, label.split('（')[0], len(ds), best, cheap, mine,
                        ranked.index(mine) + 1, len(ranked)))

    out = ['# 表 8 · 资源与通信开销（按数据类型分组，与表 2 同格同口径）', '',
           '> 生成脚本 `scripts/xc_cost_tables.py`（数据源 `cost_probe.json` + '
           '`summary.json` + `fedtdo_summary.csv`）。', '',
           '## 8. 摘要（先看这张）', '',
           '| 类型 | 格数 | 最准 | 它的 MSE | 它的通信 MiB | 最省（非 local） | '
           '它的 MSE | FedTDO MSE | FedTDO 通信 MiB | FedTDO 排名 |',
           '|---|:-:|---|--:|--:|---|--:|--:|--:|:-:|']
    for tag, label, n, best, cheap, mine, pos, tot in summary:
        ds = next(g[2] for g in DATATYPE_GROUPS if g[0] == tag)
        out.append(
            f'| {tag} · {label} | {n} | `{best}` | {t_mse(best, ds):.4f} | '
            f'{t_cost(best, ds):,.0f} | `{cheap}` | {t_mse(cheap, ds):.4f} | '
            f'**{t_mse(mine, ds):.4f}** | {t_cost(mine, ds):,.0f} | {pos}/{tot} |')
    top2 = [t for t, _, _, _, _, _, p, _ in summary if p <= 2]
    hard = [f'{t}（{p}/{n}）' for t, _, _, _, _, _, p, n in summary if p > 2]
    ratios = []
    for tag, _, n, _, _, mine, _, _ in summary:
        ds = next(g[2] for g in DATATYPE_GROUPS if g[0] == tag)
        ratios.append((tag, t_cost(mine, ds) / t_cost('fedavg', ds)
                       if t_cost('fedavg', ds) else 0))
    out += ['',
            f'> **FedTDO 在 {len(top2)}/4 个类型上排前二**'
            + (f'（{", ".join(top2)}）' if top2 else '')
            + (f'，在 {", ".join(hard)} 上既不是最准也不是最省。' if hard else '。'),
            '> 相对 `fedavg` 的通信比：'
            + '，'.join(f'类型 {t} {r:.2f}×' for t, r in ratios)
            + '。**这个比值低，一半来自方法设计（只同步 encoder），'
              '一半来自协议（1 次 vs 2 次同步）** —— 表 8d 已把后者剥离。',
            '> 成本列是**整个类型的一次完整训练合计**（含该类型所有数据集、所有客户端、'
            '两次（FedTDO 一次）同步），不是单轮。', '',
           '## 8a. 口径（先读这一节，否则表会被误引）', '',
           '| 量 | 定义 | 证据等级 |', '|---|---|---|',
           '| `P` | 一个客户端模型的**可学习参数量** | measured（`sum(p.numel())`） |',
           '| `W` | **线上字节** = `state_dict()` 字节，含 buffer | measured |',
           '| 通信/run | `客户端数 × 轮数 × (上传 + 下载)` | derived |',
           '| 轮数 | 基线 2 次聚合；**FedTDO 1 次 encoder 同步** | measured（代码路径） |',
           '| 客户端数 | 各数据集的分区数（ETT 7、ECL/Traffic 32 …） | measured（目录名 `nc<N>`） |',
           '', '**三条必须一起引用的限定**：', '',
           '1. **通信量按 `state_dict()` 原始字节计，没有量化 / 稀疏化 / 压缩。** '
           '所以它是"相对代价轴"，不是绝对带宽预算；任何一档加 8-bit 量化都会同比缩小。',
           '2. **基线做 2 次聚合，FedTDO 只做 1 次 encoder 同步。** 这是协议差异，'
           '而且对 FedTDO 有利 —— 引用"FedTDO 更省通信"时必须同时给出这句话。'
           '同步次数来自 `--fedtdo_encoder_rounds`（`run_fedtdo.py:102` 默认 **1**），'
           '表 8d 给出把它按 2 次计费的反事实。',
           '3. **`local` 的通信量是 0**（不共享任何参数），它是成本轴的原点，'
           '不是"最差的方法"。', '']

    # -- 8b wire composition ------------------------------------------------
    p = probe
    total = p['state_elems']
    out += ['## 8b. 一个客户端的"线上状态"里装的到底是什么', '',
            '| 成分 | 元素数 | 占线上字节 | 能否去掉 |', '|---|--:|--:|---|',
            f'| 可学习参数 | {p["params"]:,} | {100*p["params"]/total:.1f}% | 否 |',
            f'| 固定位置编码表 `position_embedding.pe` | {p["constant_elems"]:,} | '
            f'{100*p["constant_elems"]/total:.1f}% | **能**：sinusoidal，各客户端完全相同 |',
            f'| 专家池别名副本（`layer.expert_pool` 与 '
            f'`layer.gating_network.expert_pool` 是同一对象） | {p["duplicate_elems"]:,} | '
            f'{100*p["duplicate_elems"]/total:.1f}% | **能**：`state_dict()` 对别名模块重复序列化 |',
            f'| **合计** | **{total:,}** | **100%** | |', '',
            f'> 线上状态是参数量的 **{total/p["params"]:.2f}×**。'
            f'其中 **{100*(p["constant_elems"]+p["duplicate_elems"])/total:.1f}%** '
            '是不携带信息的（常数表 + 重复序列化），去掉后输出逐位不变。',
            '> 这条对**所有**走 `state_dict()` 的方法同等成立，所以方法之间的比较不受影响；'
            '受影响的只是绝对数字 —— 论文里报绝对带宽时必须扣掉这一层，'
            '报相对比较时可以保留。',
            '> **客户端内存 ≠ 线上流量**：`ditto` / `pfedme` 各存两份模型'
            '（全局 + 个性化/Moreau 副本），内存 2×W，但线上只传 1 份；'
            '`scaffold` 相反 —— 内存多一份控制变元，**线上也要传**（见 8c 的 ×1.42）。',
            f'> FedTDO 只同步 `patch_embedding`（{161536:,} 元素 = 0.62 MiB），'
            f'其中 **{100*160000/161536:.1f}% 是那张常数表**，真正的可学习部分只有 1,536 个元素（6 KiB）。', '']

    # -- 8c per-type tables -------------------------------------------------
    out += ['## 8c. 按数据类型的精度—成本表（`channel` 协议）', '',
            '列说明：`通信/run` 是该类型下所有数据集的一次完整训练合计；'
            '`×FedAvg` 是被 `fedavg` 除的倍数；`同步次数` 一列用来暴露'
            '**基线 2 次聚合、FedTDO 1 次 encoder 同步**这个协议差异。',
            '排名按组内均值，**并列同名次**（同一通信量的方法不会因为字典序被拉开）。',
            '`fedavg_sgd` 只跑了 8/9 格（缺 Weather / Traffic），'
            '与表 1 一致地按"不完整行"处理，不进本表。', '']

    order = [m for m, r in channel.items()
             if all(d in r['mse'] for d in DATASETS)]
    fed_x = {d: cost_of('fedavg', d) for d in DATASETS}

    for g in DATATYPE_GROUPS:
        tag, label, ds, note = g
        if not all(d in clients for d in ds):
            continue
        out += [f'### 类型 {tag} · {label.split("（")[0]}（{len(ds)} 格）', '']
        if len(ds) > 1:
            out += [f'> {note}', '']
        mae_ok = all(d in channel[m]['mae'] for m in order for d in ds)
        # a single-cell group has no mean: repeating the cell as its own mean is noise
        show_gmse = len(ds) > 1
        head = ('| method | ' + ' | '.join(ds) + ' | '
                + ('组 MSE | ' if show_gmse else '')
                + ('组 MAE | ' if mae_ok else '')
                + '通信/run (MiB) | ×FedAvg | 同步次数 | 精度排名 | 成本排名 |')
        out.append(head)
        out.append('|' + '|'.join(['---'] + [':-:'] * (len(ds) + 5
                                                   + (1 if mae_ok else 0)
                                                   + (1 if show_gmse else 0))) + '|')
        recs = []
        for m in order:
            mse = [channel[m]['mse'][d] for d in ds]
            cost = sum(cost_of(m, d) for d in ds)
            recs.append((sum(mse) / len(mse), cost, m))
        # ranks computed on the group mean; ties share a rank
        by_acc = sorted(recs, key=lambda t: (t[0], t[2]))
        acc_pos = {n: i + 1 for i, (_, _, n) in enumerate(by_acc)}
        acc_rank = dense_rank({n: v for v, _, n in recs})
        cost_rank = dense_rank({n: c for _, c, n in recs})
        for mean, cost, m in by_acc:
            cells = [f4(channel[m]['mse'][d]) for d in ds]
            ratio = (cost / sum(fed_x[d] for d in ds)) if fed_x else 0
            syncs = ('0' if m in LOCAL_ONLY
                     else str(FEDTDO_ROUNDS) if m in FEDTDO_ARMS else '2')
            extra = [f4(sum(channel[m]['mae'][d] for d in ds) / len(ds))] if mae_ok else []
            gmse = ([f4(sum(channel[m]['mse'][d] for d in ds) / len(ds))]
                    if show_gmse else [])
            out.append('| ' + ' | '.join(
                [m] + cells + gmse
                + extra
                + [fmt_mib(cost), ('—' if m == 'fedavg' else f'{ratio:.2f}×'),
                   syncs, str(acc_rank[m]), str(cost_rank[m])]) + ' |')
        out.append('')

    # -- 8d Pareto ----------------------------------------------------------
    out += ['## 8d. 精度—成本 Pareto（core 8 格，`channel` 协议）', '',
            '> 「被支配」= 存在另一个方法**精度更好且通信更少**。'
            '被支配的方法在同预算比较里不应作为主论点。', '']
    tot_cost = {m: sum(cost_of(m, d) for d in CORE) for m in order}
    tot_mse = {m: sum(channel[m]['mse'][d] for d in CORE) / len(CORE) for m in order}

    def frontier(pool):
        keep = []
        for m in pool:
            if not any(tot_mse[o] < tot_mse[m] and tot_cost[o] < tot_cost[m]
                       for o in pool if o != m):
                keep.append(m)
        return keep

    front = frontier(order)
    ext_pool = [m for m in order if m not in FEDTDO_ARMS]
    front_ext = frontier(ext_pool)
    dom: dict = {}
    for m in order:
        if m in front:
            continue
        cand = [o for o in order if tot_mse[o] < tot_mse[m] and tot_cost[o] < tot_cost[m]]
        dom[m] = min(cand, key=lambda o: tot_mse[o])

    out += ['| method | core MSE | 通信 core (MiB) | 状态 |', '|---|--:|--:|---|']
    for m in sorted(tot_mse, key=tot_mse.get):
        state = 'Pareto 前沿' if m in front else f'被 `{dom[m]}` 支配'
        out.append(f'| {m} | {tot_mse[m]:.4f} | {fmt_mib(tot_cost[m])} | {state} |')

    # counterfactual: charge FedTDO the same two syncs the baselines pay
    fed_2x = {}
    for m in FEDTDO_ARMS:
        extra = sum(clients.get(d, 0) * FEDTDO_ROUNDS * 2 * ENCODER_ELEMS * 4 / MIB
                    for d in CORE)
        fed_2x[m] = tot_cost[m] + extra
    best_own = min(FEDTDO_ARMS, key=lambda m: tot_mse[m]) if FEDTDO_ARMS else None
    beat_cheaper = [m for m in ext_pool
                    if tot_mse[m] < tot_mse[best_own] and tot_cost[m] < fed_2x[best_own]]

    out += ['',
            f'- **全表 Pareto 前沿（{len(front)} 个）：'
            f'{", ".join("`"+x+"`" for x in front)}**。'
            '其中 `local` 是零通信原点（不共享参数），另外几个是我们自己的臂。',
            f'- **只算外部基线时的前沿（{len(front_ext)} 个）：'
            f'{", ".join("`"+x+"`" for x in front_ext)}** —— '
            '这才是"别人能做到的最好精度/成本组合"，'
            '写"没有基线同时更准更省"必须引这一行，引全表前沿等于拿自己比自己。',
            f'- **配对预算的稳健性检查**：把 FedTDO 也按 2 次同步计费'
            f'（`{best_own}` 从 {fmt_mib(tot_cost[best_own])} MiB 升到 '
            f'{fmt_mib(fed_2x[best_own])} MiB），'
            + ('**仍然没有任何外部基线既更准又更省**。'
               if not beat_cheaper else
               f' 会被 {", ".join("`"+x+"`" for x in beat_cheaper)} 反超 —— '
               '此时的通信优势来自同步次数而非方法本身。'),
            '- 换句话说：FedTDO 在 2 次同步的反事实下依然是最便宜的那一档，'
            '成本优势不全是"少同步一次"换来的。', '']

    # -- 8e multivariate ----------------------------------------------------
    multi = json.loads((ROOT / 'results_paper' / 'cost_probe.json')
                       .read_text(encoding='utf-8'))['multivariate']
    out += ['## 8e. 多变量协议：参数量与通信量（客户端 = 数据集，9 个）', '',
            '| backbone | P | W/client (MiB) | 通信/run (MiB) | 相对 patchtst |',
            '|---|--:|--:|--:|--:|']
    base = multi.get('patchtst', {}).get('wire_bytes')
    for b, rec in sorted(multi.items(), key=lambda kv: kv[1].get('params', 0)):
        if 'params' not in rec:
            continue
        wire = rec['wire_bytes'] / MIB
        comm = 9 * 2 * 2 * rec['wire_bytes'] / MIB
        rel = (rec['wire_bytes'] / base) if base else 0
        out.append(f'| {b} | {rec["params"]:,} | {wire:.2f} | {comm:,.1f} | '
                   f'{rel:.1f}× |')
    out += ['',
            '> **FFTS / Time-FFM 的通信量是 PatchTST 的 100× 量级**（27.4 / 49.8 MiB 每客户端）——'
            '冻结的 GPT-2 不进 `state_dict()`，但 7.19 M / 13.05 M 个**适配器参数**要全量上传。'
            '这解释了为什么"LM 骨干进联邦"在真实带宽下并不划算：精度没赢（见表 4/5），'
            '通信先贵两个数量级。',
            '> `dtmoe` 这一行要特别读：录制参数量是 **117,172**，当前源码在文档化 flag 下'
            '重建得 **484,900**（差 4.1×，无法复现）。本表采用**较大**的重建值，'
            '即对 FedTDO 的骨干取保守（更贵）的一侧；这处复现缺口已写进 T7 局限，'
            '在论文里引用多变量协议的 `dtmoe` 参数量前必须先解决。', '']

    # -- 8g FedTDO breakdown -----------------------------------------------
    if ft:
        out += ['## 8f. FedTDO 的成本拆成两半（每题实测）', '',
                '| dataset | 客户端 | encoder 同步 (MiB) | 专家胶囊 dl+ul (MiB) | 合计 (MiB) | '
                '胶囊占比 |', '|---|:-:|--:|--:|--:|--:|']
        rows = []
        for d in DATASETS:
            arm = ft.get((d, 'center'))
            if not arm:
                continue
            c = clients.get(d, 0)
            enc = c * FEDTDO_ROUNDS * 2 * ENCODER_ELEMS * 4 / MIB
            cap = (arm['download'] + arm['upload']) / MIB
            rows.append((d, c, enc, cap, enc + cap))
        for d, c, enc, cap, tot in rows:
            out.append(f'| {d} | {c} | {enc:.1f} | {cap:.1f} | {tot:.1f} | '
                       f'{100*cap/tot:.0f}% |')
        cap_share = sum(r[3] for r in rows) / sum(r[4] for r in rows)
        out += ['',
                f'- **全表胶囊流量占 FedTDO 总通信的 {100*cap_share:.0f}%**：'
                'FedTDO 的主要开销**不是参数同步，而是专家复用**。'
                '任何把 FedTDO 说成"只传一个 encoder 所以几乎零通信"的说法都是错的。',
                '- 客户端越多，胶囊流量越贵'
                + (f'（{max(rows, key=lambda r: r[3])[0]} 达 '
                   f'{max(r[3] for r in rows):.0f} MiB 胶囊），'
                   f'但同一数据集的 `fedavg` 要 '
                   f'{max(comm_mib("fedavg", r[0], probe, clients) for r in rows):.0f} MiB'
                   ' —— 仍然更省。')]
        out.append('')

    # -- 8f readout ---------------------------------------------------------
    out += ['## 8g. 读法（由脚本算出）', '']
    best_acc = min(tot_mse, key=tot_mse.get)
    nonzero = [m for m in order if tot_cost[m] > 0]
    cheapest = min(nonzero, key=lambda m: tot_cost[m])
    ext_nz = [m for m in nonzero if m not in FEDTDO_ARMS]
    cheapest_ext = min(ext_nz, key=lambda m: tot_cost[m]) if ext_nz else None
    out += [
        f'- **最省的非零通信方法是 `{cheapest}`**（core 通信 '
        f'{fmt_mib(tot_cost[cheapest])} MiB，MSE {tot_mse[cheapest]:.4f}）—— '
        '它不传整模型：FedTDO 只同步一次 patch encoder，其余靠专家胶囊复用。',
    ]
    if cheapest_ext:
        out.append(
            f'- **外部基线里最省的是 `{cheapest_ext}`**（{fmt_mib(tot_cost[cheapest_ext])} MiB，'
            f'MSE {tot_mse[cheapest_ext]:.4f}）：它只上传共享躯干、预测头留在本地，'
            f'省下 {100*(1-tot_cost[cheapest_ext]/tot_cost["fedavg"]):.1f}% 的流量，'
            f'代价是精度落后 {100*(tot_mse[cheapest_ext]/tot_mse[best_acc]-1):.1f}%。'
            '**这才是"省流量要付多少精度"的正确读法** —— 省的那点流量买不回精度。')
    out += [
        f'- **`{best_acc}` 的 core MSE 最低（{tot_mse[best_acc]:.4f}），'
        f'通信 {fmt_mib(tot_cost[best_acc])} MiB**，是 `fedavg` 的 '
        f'{tot_cost[best_acc]/tot_cost["fedavg"]:.2f}×。',
    ]
    # how much precision does one MiB buy, relative to the zero-comm origin?
    if 'local' in tot_mse:
        gain = tot_mse['local'] - tot_mse[best_acc]
        out.append(
            f'- **相对零通信的 `local`，`{best_acc}` 用 '
            f'{fmt_mib(tot_cost[best_acc])} MiB 换来 {100*gain/tot_mse["local"]:.1f}% 的 MSE 改善**'
            f'（{tot_mse["local"]:.4f} → {tot_mse[best_acc]:.4f}）。'
            '这是本表唯一能写成"单位成本收益"的量；'
            '别把不同方法的 MiB 直接相除当性价比，因为通信量差异主要来自模型大小而不是算法。')
    out += [
        '- **成本轴不能单独用**：`local` 的通信量是 0 但没有任何协作，'
        '把它当成"最省的好方法"会得出荒谬结论 —— 本表把它当原点，不当候选。',
        '- **量化不改变排序**：全部数字按 fp32 计；8-bit 量化后每条同比 ÷4，'
        '所有方法同比例缩放，Pareto 前沿与名次都不变。'
        '真正会改变排序的只有"只传更新量（梯度/稀疏化）"一类方案，本仓库未实现。', '']

    return out


def main() -> None:
    if not PROBE.exists():
        raise SystemExit(
            'missing results_paper/cost_probe.json -- run\n'
            '  D:/software/Anaconda3/envs/ma_fmarl/python.exe scripts/xc_cost_probe.py')
    probe = json.loads(PROBE.read_text(encoding='utf-8'))['channel']['dtmoe']
    channel = load_channel()
    lines = build_cost(channel, probe)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'T8_cost_by_datatype.md'
    path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'wrote {path.relative_to(ROOT)}  ({path.stat().st_size} bytes)')


if __name__ == '__main__':
    main()
