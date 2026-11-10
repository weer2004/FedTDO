"""Scheme §10.5 metrics: prediction, online decision, and resource.

The implementation plan (``FedTDO实现方案.docx`` §10.5) asks for three families:

* **prediction** — MSE and MAE, per prediction length;
* **online decision** — cumulative risk regret, cost-adjusted cumulative
  objective, transfer success rate, wrong-transfer rate, recovery time after a
  detected drift, and knowledge reuse when a state recurs;
* **resource** — candidate evaluations, candidate inference FLOPs, expert
  download/upload bytes, number of deployments, expert switches, local cache
  occupancy, and real inference latency.

Everything below is recomputed from the artefacts a run already writes
(``metrics.json`` + ``rounds.json``), so no retraining is required.  Two entries
cannot be recovered from those logs and are emitted as ``NaN`` with a reason, so
the gap is visible instead of silently missing:

* ``reuse_gain`` — the logs carry ``cumulative_drift`` but no discrete regime
  label, so an A -> B -> A revisit cannot be identified;
* ``latency_ms``   — inference is timed nowhere in the run loop; it needs a
  separate micro-benchmark on the same model and device.

Definitions used here (state them in the paper):

``regret``          per round, ``best_gain - realised_gain`` where ``best_gain``
                    is the best **raw** gain among the arms actually probed that
                    round (``max(raw_gains)``, verified to match the logged
                    ``best_gain``).  This is a *budget-respecting* oracle: it can
                    only pick from what was evaluated, so it never rewards a
                    sequence the controller could not have afforded.
                    **It can be negative.**  The deployed expert may come from the
                    cache -- an arm probed in an earlier round and remembered --
                    and then it is not in this round's probe set at all, so the
                    controller can beat the round-local oracle.  A negative total
                    therefore means "regime memory paid off", not "the oracle was
                    beaten by luck".  ETTh1 and Weather show this; ECL/Traffic,
                    where every round probes a full 9-arm slate, do not.
``cost_objective``  ``realised_gain - lambda_eval*evaluations - lambda_transfer*transfers``.
                    Both lambdas default to 0, so the shipped runs report the
                    ``lambda = 0`` end of the Pareto curve; other points are a
                    post-hoc sweep over the logged counters (``--lambda-sweep``).
``success``         a round with a non-null ``active_expert`` whose
                    ``deployed_loss < default_loss``.
``recovery``        rounds from a drift increment (``cumulative_drift`` rose vs the
                    previous round for that client) to the next successful
                    transfer; censored if none follows.

Usage::

    python scripts/xc_metrics.py --seeds 2021
    python scripts/xc_metrics.py --seeds 2021 --with-rounds
    python scripts/xc_metrics.py --seeds 2021 --lambda-sweep
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as _st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from xc_sweep import DATASETS, MAIN_DATASETS, ROOT  # noqa: E402

FEDTDO = ROOT / 'results' / 'fedtdo'
OUT = FEDTDO / '_metrics'
ARMS = ['base', 'center', 'decorr']
LEGACY = {('Traffic', 'base'): 'traffic32_s{seed}_joint_b9_fix'}

NAN = float('nan')


def run_dir(dataset: str, arm: str, seed: int) -> Path:
    _, _, _, nc, budget = DATASETS[dataset]
    standard = FEDTDO / f'{dataset.lower()}{nc}_s{seed}_b{budget}_{arm}'
    if standard.exists():
        return standard
    legacy = LEGACY.get((dataset, arm))
    return FEDTDO / legacy.format(seed=seed) if legacy else standard


def _median(values):
    values = sorted(values)
    if not values:
        return NAN
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else 0.5 * (values[mid - 1] + values[mid])


def _mean(values):
    return sum(values) / len(values) if values else NAN


def _pstdev(values):
    return _st.pstdev(values) if len(values) > 1 else NAN


def online_metrics(rounds):
    """Regret, success rates and recovery, straight from the round log."""
    by_client = {}
    for record in rounds:
        by_client.setdefault(record['client'], []).append(record)

    oracle_total = realised_total = 0.0
    success = wrong = deployed = 0
    recoveries, censored = [], 0

    for client_records in by_client.values():
        client_records.sort(key=lambda r: r['step'])
        previous_drift = None
        pending_since = None
        for record in client_records:
            drift = record.get('cumulative_drift') or 0.0
            if previous_drift is not None and drift > previous_drift + 1e-9 and pending_since is None:
                pending_since = record['step']
            previous_drift = drift

            best = record.get('best_gain')
            if best is not None:
                realised = (record.get('default_loss') or 0.0) - (record.get('deployed_loss') or 0.0)
                oracle_total += best
                realised_total += realised

            if record.get('active_expert'):
                deployed += 1
                if record['deployed_loss'] < record['default_loss']:
                    success += 1
                    if pending_since is not None:
                        recoveries.append(record['step'] - pending_since)
                        pending_since = None
                elif record['deployed_loss'] > record['default_loss']:
                    wrong += 1
            elif pending_since is not None and record.get('best_gain') is not None:
                # A round with no deployment counts as "not yet recovered".
                pass

        if pending_since is not None:
            censored += 1

    regret_total = oracle_total - realised_total
    return dict(
        oracle_gain_total=oracle_total,
        realised_gain_total=realised_total,
        regret_total=regret_total,
        regret_per_round=regret_total / len(rounds) if rounds else NAN,
        regret_rate=(regret_total / oracle_total) if oracle_total else NAN,
        deploy_rounds=deployed,
        transfer_success_rate=(success / deployed) if deployed else NAN,
        wrong_transfer_rate=(wrong / deployed) if deployed else NAN,
        recovery_rounds_mean=_mean(recoveries) if recoveries else NAN,
        recovery_events=len(recoveries),
        recovery_censored=censored,
        reuse_gain=NAN,      # no discrete regime label in the logs
    )


def deploy_split(rounds):
    """Separate "the client first gets an expert" from "it swaps expert".

    ``costs.transfers`` counts both, which is why plan §10.5's *deployment count*
    could only be approximated before.  Walking ``active_expert`` per client in
    submission order recovers the split exactly: a transition from ``None`` to a
    capsule is a deployment, any other change is a swap.  Deploys + swaps must
    equal the logged transfer total; the caller nulls both if they disagree so a
    wrong split can never silently reach the paper.
    """
    previous, deploys, swaps = {}, 0, 0
    for record in rounds:
        client, active = record.get('client'), record.get('active_expert')
        if client is None:
            continue
        before = previous.get(client)
        if active and before != active:
            if before is None:
                deploys += 1
            else:
                swaps += 1
        previous[client] = active
    return deploys, swaps


def resource_metrics(payload, rounds=None):
    clients = payload.get('clients') or []
    costs = [c.get('costs') or {} for c in clients]

    def total(key):
        return sum(float(c.get(key) or 0.0) for c in costs)

    cache = [float(c.get('cache_size') or 0.0) for c in clients]
    registry = payload.get('registry_upload_bytes') or 0
    transfers = total('transfers')
    deploys, swaps = deploy_split(rounds) if rounds else (NAN, NAN)
    if rounds and deploys != NAN and deploys + swaps != transfers:
        deploys, swaps = NAN, NAN
    return dict(
        evaluations=total('evaluations'),
        evaluation_flops=total('evaluation_flops'),
        download_bytes=total('download_bytes'),
        downloads=total('downloads'),
        upload_bytes=float(registry),
        transfers=transfers,
        deploys=deploys,
        expert_swaps=swaps,
        switches=total('switches'),
        cache_size_mean=_mean(cache) if cache else NAN,
        cache_size_max=max(cache) if cache else NAN,
        latency_ms=NAN,      # never timed in the run loop; see scripts/latency_bench.py
    )


def prediction_metrics(payload):
    per_client = payload.get('client_metrics') or {}
    mses = [v['mse'] for v in per_client.values()]
    maes = [v['mae'] for v in per_client.values()]
    mean_mse = payload.get('mean_client_mse')
    mean_mae = payload.get('mean_client_mae')
    if mean_mse is None and mses:
        mean_mse = _mean(mses)
    if mean_mae is None and maes:
        mean_mae = _mean(maes)
    gain = (payload.get('actions') or {}).get('relative_gain')
    return dict(
        mse=payload.get('mse'),
        mae=payload.get('mae'),
        rmse=payload.get('rmse'),
        mean_client_mse=mean_mse,
        mean_client_mae=mean_mae,
        median_client_mse=_median(mses) if mses else NAN,
        median_client_mae=_median(maes) if maes else NAN,
        # Dispersion matters here: on ETTm1 the mean is 2.2x the median and on
        # ECL the worst client is 13x the median client, so a mean-only table is
        # decided by one client.  Needed for the paper's box plots.
        std_client_mse=_pstdev(mses) if len(mses) > 1 else NAN,
        max_client_mse=max(mses) if mses else NAN,
        min_client_mse=min(mses) if mses else NAN,
        num_clients=len(mses),
        gain_vs_local_pct=100.0 * gain if gain is not None else NAN,
    )


def per_round_rows(dataset, arm, seed, rounds):
    rows = []
    cumulative = 0.0
    for index, record in enumerate(rounds):
        best = record.get('best_gain')
        realised = (record.get('default_loss') or 0.0) - (record.get('deployed_loss') or 0.0)
        regret = (best - realised) if best is not None else NAN
        cumulative += regret if best is not None else 0.0
        rows.append(dict(
            dataset=dataset, arm=arm, seed=seed, round=index,
            client=record['client'], step=record.get('step'),
            probes=record.get('probes'),
            active=bool(record.get('active_expert')),
            default_loss=record.get('default_loss'),
            deployed_loss=record.get('deployed_loss'),
            realised_gain=realised,
            oracle_gain=best,
            regret=regret,
            cumulative_regret=cumulative,
            drift=record.get('cumulative_drift'),
        ))
    return rows


def per_client_rows(dataset, arm, seed, payload):
    rows = []
    costs = {c['client_id']: (c.get('costs') or {}) for c in (payload.get('clients') or [])}
    for cid, metrics in (payload.get('client_metrics') or {}).items():
        cost = costs.get(cid, {})
        rows.append(dict(
            dataset=dataset, arm=arm, seed=seed, client=cid,
            mse=metrics['mse'], mae=metrics['mae'],
            evaluations=cost.get('evaluations', 0),
            evaluation_flops=cost.get('evaluation_flops', 0),
            download_bytes=cost.get('download_bytes', 0),
            transfers=cost.get('transfers', 0),
            switches=cost.get('switches', 0),
        ))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=MAIN_DATASETS, choices=list(DATASETS))
    parser.add_argument('--arms', nargs='+', default=ARMS, choices=ARMS)
    parser.add_argument('--seeds', nargs='+', type=int, default=[2021])
    parser.add_argument('--with-rounds', action='store_true', help='also dump the per-round log')
    parser.add_argument('--lambda-sweep', action='store_true',
                        help='print the cost/accuracy Pareto sweep over logged counters')
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    summary, round_rows, client_rows, missing = [], [], [], []

    for dataset in args.datasets:
        for arm in args.arms:
            for seed in args.seeds:
                directory = run_dir(dataset, arm, seed)
                metrics_path = directory / 'metrics.json'
                rounds_path = directory / 'rounds.json'
                if not metrics_path.exists():
                    missing.append(f'{dataset}/{arm}/s{seed}')
                    continue
                payload = json.loads(metrics_path.read_text())
                rounds = json.loads(rounds_path.read_text()) if rounds_path.exists() else []

                row = dict(dataset=dataset, arm=arm, seed=seed,
                           run=directory.name,
                           pred_len=(payload.get('config') or {}).get('pred_len'))
                row.update(prediction_metrics(payload))
                row.update(resource_metrics(payload, rounds))
                if rounds:
                    row.update(online_metrics(rounds))
                if row['evaluations'] and row.get('regret_total') is not None:
                    row['cost_objective_l0'] = row['realised_gain_total']
                summary.append(row)
                if args.with_rounds and rounds:
                    round_rows.extend(per_round_rows(dataset, arm, seed, rounds))
                client_rows.extend(per_client_rows(dataset, arm, seed, payload))

    def dump(path, rows):
        if not rows:
            return
        keys = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        with open(path, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f'wrote {path} ({len(rows)} rows x {len(keys)} cols)')

    dump(OUT / 'fedtdo_summary.csv', summary)
    dump(OUT / 'fedtdo_per_client.csv', client_rows)
    if round_rows:
        dump(OUT / 'fedtdo_per_round.csv', round_rows)

    print()
    header = ['dataset', 'arm', 'mse', 'mae', 'gain%', 'regret', 'succ', 'wrong',
              'recov', 'evals', 'flops', 'dl_MB', 'ul_MB', 'switches']
    print(' '.join(f'{h:>10s}' if i < 2 else f'{h:>9s}' for i, h in enumerate(header)))
    for row in summary:
        def fmt(key, scale=1.0, digits=3):
            value = row.get(key, NAN)
            if value is None or (isinstance(value, float) and math.isnan(value)):
                return '—'
            return f'{value / scale:.{digits}f}'
        print(f'{row["dataset"]:>10s} {row["arm"]:>10s} '
              f'{fmt("mean_client_mse", 1, 4):>9s} {fmt("mean_client_mae", 1, 4):>9s} '
              f'{fmt("gain_vs_local_pct", 1, 2):>9s} {fmt("regret_total", 1, 2):>9s} '
              f'{fmt("transfer_success_rate", 1, 3):>9s} {fmt("wrong_transfer_rate", 1, 3):>9s} '
              f'{fmt("recovery_rounds_mean", 1, 1):>9s} '
              f'{fmt("evaluations", 1, 0):>9s} {fmt("evaluation_flops", 1e9, 2):>9s} '
              f'{fmt("download_bytes", 2**20, 2):>9s} {fmt("upload_bytes", 2**20, 2):>9s} '
              f'{fmt("switches", 1, 0):>9s}')

    if missing:
        print(f'\nmissing runs: {len(missing)} -> {", ".join(missing[:8])}'
              f'{" ..." if len(missing) > 8 else ""}')

    if args.lambda_sweep:
        print('\n=== cost/accuracy Pareto (offline re-scoring of logged counters) ===')
        print('objective = realised_gain - lam_e * evaluations - lam_t * transfers')
        print(f'{"lam_e":>9s} {"lam_t":>9s} ' + ' '.join(f'{a:>12s}' for a in args.arms))
        pareto = []
        for lam_e in (0.0, 0.001, 0.01, 0.1):
            for lam_t in (0.0, 0.01, 0.1, 1.0):
                cells = []
                for arm in args.arms:
                    rows = [r for r in summary if r['arm'] == arm
                            and r.get('realised_gain_total') is not None]
                    if not rows:
                        cells.append(f'{"—":>12s}')
                        continue
                    score = _mean([
                        r['realised_gain_total'] - lam_e * r['evaluations'] - lam_t * r['transfers']
                        for r in rows
                    ])
                    cells.append(f'{score:12.2f}')
                    for row in rows:
                        pareto.append(dict(
                            dataset=row['dataset'], arm=arm, seed=row['seed'],
                            lam_e=lam_e, lam_t=lam_t,
                            objective=row['realised_gain_total']
                            - lam_e * row['evaluations'] - lam_t * row['transfers'],
                            realised_gain=row['realised_gain_total'],
                            evaluations=row['evaluations'], transfers=row['transfers'],
                            mse=row['mean_client_mse'],
                        ))
                print(f'{lam_e:9.3f} {lam_t:9.3f} ' + ' '.join(cells))
        if pareto:
            path = OUT / 'fedtdo_pareto.csv'
            with open(path, 'w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(pareto[0]))
                writer.writeheader()
                writer.writerows(pareto)
            print(f'wrote {path} ({len(pareto)} rows)')

    print('\n=== metrics the logs cannot produce ===')
    print('reuse_gain : no discrete regime label is logged (only cumulative_drift) -> needs a A-B-A revisit test')
    print('latency_ms : inference is never timed -> needs a micro-benchmark on the same device')


if __name__ == '__main__':
    main()
