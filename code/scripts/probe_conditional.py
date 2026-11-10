"""Can a *conditional* gain estimate predict the next reward better than the arm mean?

Motivation
----------
The 5-seed sweep established that the deployment rate is stuck at 6-10% and the
oracle capture at 0-23%, and ``scripts/replay_selector.py`` showed that no
estimator / decision pair moves it.  The reason traced in
``TRAFFIC_SEED_SWEEP.md`` is that a per-arm mean with ``n ~ 3.6`` is not an
estimate of anything: the lag-1 autocorrelation of the reward of one
(client, expert) series is ~0, while the mean gain of the *same* expert differs
by 0.10 across hosts.

So the question this script answers, without retraining anything, is:

    Does pooling a starved arm's 3.6 observations with a *group* of related
    arms predict the next reward better than the arm's own history?

Three conditioning levels are available inside one host, none of which needs any
cross-client message:

* ``src``  - the other experts published by the same source client (a client's
  four experts share training data and therefore should transfer alike);
* ``role`` - the same expert slot (``E0..E3``) across sources (the grouping the
  current ``--fedtdo_shrink_group role`` already uses);
* ``all``  - every arm the host has ever evaluated (the current global pool).

The predictors are strictly causal: for a reward at round ``t`` only
observations from rounds ``< t`` on the same host are used, so a predictor that
wins here wins inside the online loop too.

Usage::

    python scripts/probe_conditional.py --arm traffic32_joint_b9
    python scripts/probe_conditional.py --arm traffic32_joint_b9 --reward raw
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np


def load(path):
    with open(path, 'r', encoding='utf-8') as handle:
        records = json.load(handle)
    by_client = defaultdict(list)
    for record in records:
        by_client[record['client']].append(record)
    for value in by_client.values():
        value.sort(key=lambda item: item['step'])
    return by_client


def source_of(expert_id):
    return expert_id.split(':', 1)[0]


def role_of(expert_id):
    return expert_id.rsplit(':', 1)[-1]


def rows(by_client, reward):
    """Yield (client, step, expert_id, gain) for every logged evaluation."""
    for client, records in by_client.items():
        for record in records:
            if reward == 'raw':
                raw = record.get('raw_gains') or {
                    key: value * record['default_loss']
                    for key, value in record['gains'].items()
                }
                gains = raw
            else:
                gains = record['gains']
            for expert_id, gain in gains.items():
                yield client, record['step'], expert_id, float(gain)


def causal_predictions(by_client, reward):
    """Causal predictions of each observable reward from four conditioning levels."""
    seen = defaultdict(list)  # (client, expert) -> [gain]
    seen_src = defaultdict(list)  # (client, source) -> [gain]  (excluding own arm)
    seen_role = defaultdict(list)  # (client, role) -> [gain]
    seen_all = defaultdict(list)  # client -> [gain]

    out = []
    for client, records in by_client.items():
        for record in records:
            if reward == 'raw':
                gains = record.get('raw_gains') or {
                    key: value * record['default_loss']
                    for key, value in record['gains'].items()
                }
            else:
                gains = record['gains']
            # Predict first, then update: strictly causal within the round as well.
            for expert_id, gain in gains.items():
                src, role = source_of(expert_id), role_of(expert_id)
                own = seen[(client, expert_id)]
                pool_src = seen_src[(client, src)]
                pool_role = seen_role[(client, role)]
                pool_all = seen_all[client]
                out.append({
                    'client': client,
                    'step': record['step'],
                    'expert': expert_id,
                    'gain': float(gain),
                    'n_own': len(own),
                    'own': float(np.mean(own)) if own else np.nan,
                    'n_src': len(pool_src),
                    'src': float(np.mean(pool_src)) if pool_src else np.nan,
                    'n_role': len(pool_role),
                    'role': float(np.mean(pool_role)) if pool_role else np.nan,
                    'n_all': len(pool_all),
                    'all': float(np.mean(pool_all)) if pool_all else np.nan,
                })
            for expert_id, gain in gains.items():
                src, role = source_of(expert_id), role_of(expert_id)
                seen[(client, expert_id)].append(float(gain))
                seen_src[(client, src)].append(float(gain))
                seen_role[(client, role)].append(float(gain))
                seen_all[client].append(float(gain))
    return out


def score(pred, actual):
    mask = ~np.isnan(pred)
    if mask.sum() < 30:
        return None
    p, a = pred[mask], actual[mask]
    if p.std() < 1e-12 or a.std() < 1e-12:
        return None
    corr = float(np.corrcoef(p, a)[0, 1])
    # Ranking quality of the resulting decision: deploy when prediction > 0.
    take = p > 0
    if take.sum() == 0:
        return {'n': int(mask.sum()), 'corr': corr, 'precision': float('nan'),
                'take_rate': 0.0, 'mean_taken': float('nan')}
    return {
        'n': int(mask.sum()),
        'corr': corr,
        'take_rate': float(take.mean()),
        # Reward actually earned by the arms this predictor would deploy.
        'mean_taken': float(a[take].mean()),
        'precision': float((a[take] > 0).mean()),
        'mean_all': float(a.mean()),
    }


def shrinkage_combo(table, keys, n0):
    """Convex combination of predictors weighted by their effective counts."""
    values, weights = [], []
    for key, count_key in keys:
        v = table[key]
        n = table['n_' + key] if count_key is None else table[count_key]
        mask = ~np.isnan(v)
        values.append(np.where(mask, v, 0.0))
        weights.append(np.where(mask, n, 0.0))
    num = np.sum([v * w for v, w in zip(values, weights)], axis=0)
    den = np.sum(weights, axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        combined = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
    # Empirical-Bayes pull toward the pooled mean, matching ``selector.shrink``.
    pooled = np.nanmean([v[~np.isnan(v)].mean() if (~np.isnan(v)).any() else 0.0
                         for v in values])
    return (den * combined + n0 * pooled) / (den + n0)


def round_level(table_rows, pred, threshold=0.0):
    """Decision-level metric: pick the best arm per round, then deploy if > 0.

    The per-observation correlations above are inflated by a shared factor: all
    arms evaluated in one round share the same input window, so their rewards
    move together and ~14k observations are not 14k independent draws.  The
    honest unit of evaluation is therefore the round - 60 per client - and the
    quantity that reaches the loss is the reward of the arm the rule deploys.
    """
    groups = defaultdict(list)
    for row, value in zip(table_rows, pred):
        if np.isnan(value):
            continue
        groups[(row['client'], row['step'])].append((value, row['gain']))
    picked, best, deploy = [], [], []
    for entries in groups.values():
        if len(entries) < 2:
            continue
        chosen = max(entries, key=lambda item: item[0])
        picked.append(chosen[1])
        best.append(max(entry[1] for entry in entries))
        deploy.append(chosen[0] > threshold)
    if not picked or not any(deploy):
        return None
    picked = np.array(picked)
    best = np.array(best)
    deploy = np.array(deploy)
    return {
        'rounds': len(picked),
        'deploy': float(deploy.mean()),
        'gain_picked': float(picked[deploy].mean()),
        'gain_best': float(best[deploy].mean()),
        'precision': float((picked[deploy] > 0).mean()),
        'oracle_gain': float(best.mean()),
        'regret': float((best[deploy] - picked[deploy]).mean()),
    }


def variance_decomposition(by_client, reward):
    """Share of reward variance explained by host, source and role dummies."""
    data = list(rows(by_client, reward))
    y = np.array([row[3] for row in data])
    hosts = [row[0] for row in data]
    srcs = [source_of(row[2]) for row in data]
    roles = [role_of(row[2]) for row in data]

    def r2(groups):
        codes = {value: index for index, value in enumerate(sorted(set(groups)))}
        design = np.zeros((len(y), len(codes)))
        for row, value in enumerate(groups):
            design[row, codes[value]] = 1.0
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        fitted = design @ coef
        ss_res = float(np.sum((y - fitted) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0, len(codes) - 1

    assert len(hosts) == len(srcs) == len(roles) == len(y)
    result = {'rewards': len(y), 'var': float(y.var()), 'mean': float(y.mean())}
    for name, groups in (('host', hosts), ('source', srcs), ('role', roles)):
        value, dof = r2(groups)
        result[name] = {'r2': value, 'dof': dof}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', default='traffic32_joint_b9')
    parser.add_argument('--root', default='./results/fedtdo')
    parser.add_argument('--reward', default='relative', choices=['relative', 'raw'])
    parser.add_argument('--bootstrap', type=int, default=200,
                        help='bootstrap resamples over clients for the CI of the spread')
    args = parser.parse_args()

    path = os.path.join(args.root, args.arm, 'rounds.json')
    by_client = load(path)
    table_rows = causal_predictions(by_client, args.reward)

    actual = np.array([row['gain'] for row in table_rows])
    columns = {
        key: np.array([row[key] for row in table_rows], dtype=float)
        for key in ('own', 'src', 'role', 'all')
    }
    counts = {f'n_{key}': np.array([row[f'n_{key}'] for row in table_rows], dtype=float)
              for key in ('own', 'src', 'role', 'all')}

    print(f'arm: {args.arm}   reward: {args.reward}   observations: {len(actual)}')
    print(f'overall mean {actual.mean():+.5f}   positive {np.mean(actual > 0):.1%}   '
          f'sd {actual.std(ddof=1):.5f}')
    print()
    print(f"{'predictor':<26}{'n':>7}{'corr':>8}{'take':>7}{'mean|take':>11}{'prec':>7}")
    for name, pred in columns.items():
        stats = score(pred, actual)
        if stats is None:
            print(f'{name:<26}  too few usable rows')
            continue
        print(f"{name:<26}{stats['n']:>7}{stats['corr']:>+8.3f}{stats['take_rate']:>7.0%}"
              f"{stats['mean_taken']:>+11.5f}{stats['precision']:>7.0%}")

    # Nested combinations: arm first, then a group, then everything.
    combos = {
        'own+src': [('own', 'n_own'), ('src', 'n_src')],
        'own+role': [('own', 'n_own'), ('role', 'n_role')],
        'own+src+role': [('own', 'n_own'), ('src', 'n_src'), ('role', 'n_role')],
        'all three+global': [('own', 'n_own'), ('src', 'n_src'),
                             ('role', 'n_role'), ('all', 'n_all')],
    }
    combined_table = {**columns, **counts}
    for n0 in (0.0, 2.0, 5.0):
        for name, keys in combos.items():
            pred = shrinkage_combo(combined_table, keys, n0)
            stats = score(pred, actual)
            if stats is None:
                continue
            print(f"{name + f' (n0={n0:g})':<26}{stats['n']:>7}{stats['corr']:>+8.3f}"
                  f"{stats['take_rate']:>7.0%}{stats['mean_taken']:>+11.5f}"
                  f"{stats['precision']:>7.0%}")

    print()
    print('decision level (one row per client-round with >= 2 evaluated arms):')
    print(f"{'rule':<26}{'rounds':>7}{'deploy':>8}{'gain|dep':>10}{'prec':>7}"
          f"{'best|dep':>10}{'regret':>9}")

    def show(name, pred):
        stats = round_level(table_rows, pred)
        if stats is None:
            print(f'{name:<26}  no deployable rounds')
            return
        print(f"{name:<26}{stats['rounds']:>7}{stats['deploy']:>8.0%}"
              f"{stats['gain_picked']:>+10.5f}{stats['precision']:>7.0%}"
              f"{stats['gain_best']:>+10.5f}{stats['regret']:>+9.5f}")

    for name, pred in columns.items():
        show(name, pred)
    for n0 in (0.0, 2.0, 5.0):
        for name, keys in combos.items():
            show(f"{name} (n0={n0:g})", shrinkage_combo(combined_table, keys, n0))
    show('never deploy (=0)', np.full(len(table_rows), -1e9))

    print()
    decomp = variance_decomposition(by_client, args.reward)
    print(f"variance decomposition ({decomp['rewards']} rewards, sd {decomp['var'] ** 0.5:.5f}):")
    for name in ('host', 'source', 'role'):
        part = decomp[name]
        print(f'  {name:<8} R^2 = {part["r2"]:6.3f}   ({part["dof"]} dummies)')

    print()
    n_src = counts['n_src']
    print('effective sample size actually available per conditioning level:')
    for name in ('own', 'src', 'role', 'all'):
        n = counts[f'n_{name}']
        usable = n[~np.isnan(columns[name])]
        if usable.size:
            print(f'  {name:<6} median n = {np.median(usable):5.1f}   '
                  f'oracle-usable rows {usable.size / len(n):.0%}')

    # Bootstrap over clients: is the gap between the best predictor and ``own`` real?
    if args.bootstrap:
        clients = sorted({row['client'] for row in table_rows})
        index = {client: i for i, client in enumerate(clients)}
        owner = np.array([index[row['client']] for row in table_rows])
        rng = np.random.default_rng(0)
        deltas = []
        for _ in range(args.bootstrap):
            pick = rng.integers(0, len(clients), len(clients))
            sample = np.concatenate([np.where(owner == p)[0] for p in pick])
            best = None
            for name, keys in combos.items():
                pred = shrinkage_combo(combined_table, keys, 0.0)[sample]
                stats = score(pred, actual[sample])
                if stats is None:
                    continue
                gap = stats['mean_taken'] - (0.0 if np.isnan(stats['mean_taken']) else 0.0)
                if best is None or gap > best[1]:
                    best = (name, gap)
            own = score(columns['own'][sample], actual[sample])
            if best and own:
                deltas.append(best[1] - own['mean_taken'])
        if deltas:
            deltas = np.array(deltas)
            print(f'\nbootstrap over clients ({len(deltas)} resamples): '
                  f'best-combination minus own-arm reward per deployment '
                  f'{deltas.mean():+.5f}  [{np.percentile(deltas, 5):+.5f}, '
                  f'{np.percentile(deltas, 95):+.5f}]  '
                  f'positive in {np.mean(deltas > 0):.0%}')


if __name__ == '__main__':
    main()
