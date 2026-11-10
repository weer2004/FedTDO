"""FedEMoE-style metrics that the §10.5 list does not cover.

Plan §10.5 asks for 17 metrics (prediction / online-decision / resource).  FedEMoE
(ICML 2026, Du et al.) reports a different emphasis, and three of its habits are
worth adopting here:

1. **Per-client score on the client's own distribution**, not one global average.
   FedEMoE reports "the accuracy of each heterogeneous model"; the worst client
   is a first-class number.  In ``channel`` partitioning our clients are
   different sensors, so this is the same concern.
2. **Communication volume actually moved** (FedEMoE uploads only the activated
   sub-experts, so it must report bytes, not just rounds).
3. **Convergence / stabilisation speed** (FedEMoE: 80% on CIFAR-10 in 20 rounds)
   and **per-client compute overhead** (their +1.24 GFLOPs/client).

What cannot be copied, and why
------------------------------
FedEMoE measures *rounds to target accuracy* on a reshuffled i.i.d. pool, where
the target is fixed.  FedTDO evaluates **prequentially**: round ``t`` scores a
*different, later time window*, so ``deployed_loss`` is not converging to a
constant -- it tracks a moving difficulty.  On ECL the global mean deployed loss
goes 0.080 -> 0.095 -> 0.101 -> 0.163 -> 0.113 over the first five rounds and
ends at 0.105 with its minimum at round 25: **a loss-based "rounds to 95%" is
not identifiable**.  We therefore report three measurable substitutes:

``rounds_to_first_deploy``  median rounds until a client first accepts an expert.
``rounds_to_half_gain``     round at which cumulative realised gain crosses 50%
                            of the run's total -- how fast value accumulates.
``late_switch_share``       share of expert swaps in the last 25% of rounds.
                            A controller that has settled switches less late.

Outputs ``results/fedtdo/_metrics/fedtdo_fedemoe.csv``.

Usage::

    python scripts/xc_fedemoe.py                 # table
    python scripts/xc_fedemoe.py --csv           # write the CSV only
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from xc_metrics import ARMS, NAN, run_dir  # noqa: E402
from xc_sweep import MAIN_DATASETS, ROOT as SWEEP_ROOT  # noqa: E402

FEDTDO = SWEEP_ROOT / 'results' / 'fedtdo'
BASELINES = SWEEP_ROOT / 'results_baselines'
OUT = FEDTDO / '_metrics'
METHODS = ['local', 'fedavg', 'fedala', 'fedpoe', 'pfedcs']
ORDER = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Exchange', 'ILI', 'ECL', 'Traffic']


def _jain(values):
    """Jain's fairness index on per-client *quality* (here: 1/MSE).

    1.0 = every client equally served, 1/n = one client takes everything.
    Reported instead of a bare std because it is scale-free and the standard
    number in the fairness literature.
    """
    if not values or min(values) <= 0:
        return NAN
    total = sum(values)
    squares = sum(v * v for v in values)
    return (total * total) / (len(values) * squares) if squares else NAN


def _baseline(dataset, method):
    pattern = f'BL_{method}_xc_s2021_{dataset}_nc*_sl96_pl24_dm32_r2_exp'
    for directory in BASELINES.glob(pattern):
        path = directory / 'summary.json'
        if path.exists():
            return json.loads(path.read_text())
    return None


def fedemoe_metrics(dataset, arm, seed):
    directory = run_dir(dataset, arm, seed)
    payload_path = directory / 'metrics.json'
    rounds_path = directory / 'rounds.json'
    if not payload_path.exists():
        return None
    payload = json.loads(payload_path.read_text())
    rounds = json.loads(rounds_path.read_text()) if rounds_path.exists() else []

    per_client = payload.get('client_metrics') or {}
    mses = [v['mse'] for v in per_client.values()]
    if not mses:
        return None
    worst, best = max(mses), min(mses)
    mean_mse = st.fmean(mses)
    jain = _jain([1.0 / m for m in mses])

    # Worst client against the *best baseline's* worst client.  FedEMoE's claim is
    # about each heterogeneous model, so the tail is the honest comparator.
    baseline_worst, baseline_name = None, None
    for method in METHODS:
        payload_bl = _baseline(dataset, method)
        if payload_bl and payload_bl.get('max_mse') is not None:
            if baseline_worst is None or payload_bl['max_mse'] < baseline_worst:
                baseline_worst, baseline_name = payload_bl['max_mse'], method
    worst_gap = (100.0 * (baseline_worst - worst) / baseline_worst
                 if baseline_worst else NAN)

    # --- convergence substitutes, all from the round log -------------------
    by_client = {}
    for record in rounds:
        by_client.setdefault(record['client'], []).append(record)
    n_rounds = min((len(v) for v in by_client.values()), default=0)

    first_deploy = []
    for records in by_client.values():
        for index, record in enumerate(records):
            if record.get('active_expert'):
                first_deploy.append(index + 1)
                break
    rounds_to_first = st.median(first_deploy) if first_deploy else NAN

    realised = [max(0.0, r['default_loss'] - r['deployed_loss']) for r in rounds]
    cumulative, running, half_round = [], 0.0, NAN
    for index, value in enumerate(realised):
        running += value
        cumulative.append(running)
    total = cumulative[-1] if cumulative else 0.0
    if total > 0:
        per_round_gain = total / max(1, len(rounds))
        for index, value in enumerate(cumulative):
            if value >= 0.5 * total:
                half_round = (index + 1) / max(1, len(rounds)) * n_rounds
                break

    late_cut = int(0.75 * n_rounds)
    late_switches = early_switches = 0
    for records in by_client.values():
        previous = None
        for index, record in enumerate(records):
            active = record.get('active_expert')
            if active and previous is not None and active != previous:
                if index >= late_cut:
                    late_switches += 1
                else:
                    early_switches += 1
            if active:
                previous = active
    total_switches = late_switches + early_switches
    late_share = late_switches / total_switches if total_switches else NAN

    costs = [c.get('costs') or {} for c in (payload.get('clients') or [])]
    clients = len(costs) or 1
    comm = sum(float(c.get('download_bytes') or 0) + float(c.get('upload_bytes') or 0)
               for c in costs)
    per_client_rounds = max(1, n_rounds) * clients

    return dict(
        dataset=dataset, family='fedtdo', arm=arm, seed=seed,
        num_clients=len(mses),
        best_client_mse=best, median_client_mse=st.median(mses),
        worst_client_mse=worst,
        cv_client_mse=st.pstdev(mses) / mean_mse if mean_mse else NAN,
        jain_index=jain,
        worst_vs_best_baseline_pct=worst_gap,
        best_baseline_for_worst=baseline_name or '',
        comm_mb_per_client=comm / clients / 1e6,
        comm_mb_per_client_round=comm / per_client_rounds / 1e6,
        rounds_to_first_deploy=rounds_to_first,
        rounds_to_half_gain=half_round,
        late_switch_share=late_share,
        gflops_per_client=sum(float(c.get('evaluation_flops') or 0) for c in costs) / clients / 1e9,
    )


def write_joined(rows):
    """One wide row per run: §10.5 metrics + FedEMoE metrics.

    This is the file to hand to whoever writes the paper tables -- every number
    quoted in ``PAPER_METRICS.md`` can be traced back to a column here, so a
    transcription error shows up as a mismatch instead of a silent wrong number.
    """
    summary_path = OUT / 'fedtdo_summary.csv'
    if not summary_path.exists():
        print('skip join: fedtdo_summary.csv missing (run xc_metrics.py first)')
        return
    base = {(r['dataset'], r['arm'], r['seed']): r
            for r in csv.DictReader(summary_path.open(encoding='utf-8'))}
    extra = {(r['dataset'], r['arm'], str(r['seed'])): r for r in rows}
    merged, columns = [], []
    for key, left in base.items():
        right = extra.get(key)
        if not right:
            continue
        joined = {f'sec10_5__{k}': v for k, v in left.items()}
        joined.update({f'fedemoe__{k}': v for k, v in right.items()
                       if k not in ('dataset', 'arm', 'seed', 'family')})
        merged.append(joined)
    if not merged:
        return
    columns = list(merged[0])
    path = OUT / 'fedtdo_all_metrics.csv'
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(merged)
    print(f'wrote {path} ({len(merged)} rows x {len(columns)} cols)')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seeds', type=int, nargs='+', default=[2021])
    parser.add_argument('--csv', action='store_true')
    parser.add_argument('--join', action='store_true',
                        help='also write fedtdo_all_metrics.csv (wide, all metrics)')
    args = parser.parse_args()

    rows = []
    for dataset in MAIN_DATASETS:
        for arm in ARMS:
            for seed in args.seeds:
                row = fedemoe_metrics(dataset, arm, seed)
                if row:
                    rows.append(row)

    if not rows:
        print('no runs found')
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'fedtdo_fedemoe.csv'
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if args.csv:
        print(f'wrote {path} ({len(rows)} rows)')
        if args.join:
            write_joined(rows)
        return 0

    print(f'{"dataset":9s}{"arm":8s}{"worst":>9s}{"median":>9s}{"CV":>7s}{"Jain":>7s}'
          f'{"vs BLworst":>12s}{"MB/cli":>8s}{"MB/cli-rd":>10s}{"rd@deploy":>10s}'
          f'{"rd@half":>8s}{"lateSw":>8s}{"GFLOP/cli":>11s}')
    for row in rows:
        print(f"{row['dataset']:9s}{row['arm']:8s}{row['worst_client_mse']:9.4f}"
              f"{row['median_client_mse']:9.4f}{row['cv_client_mse']:7.2f}"
              f"{row['jain_index']:7.3f}{row['worst_vs_best_baseline_pct']:+11.1f}%"
              f"{row['comm_mb_per_client']:8.2f}{row['comm_mb_per_client_round']:10.3f}"
              f"{row['rounds_to_first_deploy']:10.1f}{row['rounds_to_half_gain']:8.1f}"
              f"{row['late_switch_share']:8.2f}{row['gflops_per_client']:11.3f}")
    print(f'\nwrote {path} ({len(rows)} rows)')
    if args.join:
        write_joined(rows)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
