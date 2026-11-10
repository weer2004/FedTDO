"""Is an arm's value the same on every host?  (cross-host evidence sharing)

Why this question.  The controller has two ways to raise precision cheaply:

* borrow across *arms* within a host (role pool) - measured at R^2 = 3.9%;
* borrow across *hosts* for the same arm - never measured, and it is the one
  that would scale with the number of clients, because every host that
  evaluates ``c785:L0:E1`` produces an observation of the *same* object.

If the arm's value is host-independent (small arm x host interaction), peer
observations are worth as many effective samples as they are numerous, and the
confidence radius - the actual binding constraint at n ~ 3 - can shrink by
sqrt(32) without any extra evaluation.  If the interaction dominates, pooling
across hosts is pooling noise, and the ceiling on this idea is zero.

Two measurements:

1. a two-way decomposition of the (arm, host) cell means logged by one arm's
   shadow evaluations, reporting the incremental R^2 of arm, host and their
   interaction;
2. a causal replay in which each host receives peer observations of the same
   arm from *earlier* rounds as pseudo-observations of weight up to ``cap``.
   This is deployable as-is: the server already aggregates per-arm statistics
   every round, it simply does not broadcast them.

Usage::

    python scripts/probe_cross_host.py --arm traffic32_joint_b9
    python scripts/probe_cross_host.py --arm traffic32_joint_b9 --caps 1 2 4 8 16
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from federated import FedTDOController  # noqa: E402
from replay_selector import arm_ids, load_rounds  # noqa: E402


class EvidenceController(FedTDOController):
    """Controller that accepts priors carrying *evidence counts*, not just similarity.

    ``set_regime_priors`` clamps the weight to [0, 1] on purpose: a regime prior
    is a similarity statement, and a similarity of 5 is meaningless.  A peer's
    observation is a different object - it is ``n`` real measurements of the same
    arm - so the weight has to be allowed to exceed 1 for the radius to see it.
    """

    def set_regime_priors(self, priors):
        normalised = {}
        for expert_id, value in (priors or {}).items():
            if isinstance(value, (tuple, list)) and len(value) == 2:
                gain, weight = float(value[0]), float(value[1])
            else:
                gain, weight = float(value), 1.0
            if weight <= 0.0:
                continue
            normalised[str(expert_id)] = (gain, weight)
        self.regime_priors = normalised


def cells(by_client):
    """``(arm, host) -> [reward, ...]`` over the whole log."""
    table = collections.defaultdict(list)
    for host, records in by_client.items():
        for record in records:
            for expert_id, gain in record['gains'].items():
                table[(expert_id, host)].append(float(gain))
    return table


def decompose(table, min_n=2):
    """Incremental R^2 of arm / host / interaction on (arm, host) cell means."""
    rows = [(arm, host, float(np.mean(values))) for (arm, host), values in table.items()
            if len(values) >= min_n]
    arms = sorted({row[0] for row in rows})
    hosts = sorted({row[1] for row in rows})
    y = np.array([row[2] for row in rows], dtype=float)
    total = float(((y - y.mean()) ** 2).sum())

    def r2(use_arms, use_hosts):
        columns = [np.ones_like(y)]
        if use_arms:
            columns += [(np.array([row[0] == a for row in rows], dtype=float))
                        for a in arms[1:]]
        if use_hosts:
            columns += [(np.array([row[1] == h for row in rows], dtype=float))
                        for h in hosts[1:]]
        design = np.stack(columns, axis=1)
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        residual = float(((y - design @ beta) ** 2).sum())
        return 1.0 - residual / max(total, 1e-12)

    return {
        'cells': len(rows),
        'arms': len(arms),
        'hosts': len(hosts),
        'r2_arm': r2(True, False),
        'r2_host': r2(False, True),
        'r2_both': r2(True, True),
    }


def split_half(table, seed=0, min_n=2):
    """Correlation of an arm's value measured on disjoint halves of the hosts."""
    arms = sorted({arm for (arm, _) in table})
    hosts = sorted({host for (_, host) in table})
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(200):
        order = rng.permutation(len(hosts))
        left = {hosts[i] for i in order[: len(hosts) // 2]}
        xs, ys = [], []
        for arm in arms:
            first = [np.mean(table[(arm, host)]) for host in hosts
                     if host in left and len(table.get((arm, host), [])) >= min_n]
            second = [np.mean(table[(arm, host)]) for host in hosts
                      if host not in left and len(table.get((arm, host), [])) >= min_n]
            if first and second:
                xs.append(float(np.mean(first)))
                ys.append(float(np.mean(second)))
        if len(xs) >= 5:
            scores.append(float(np.corrcoef(xs, ys)[0, 1]))
    return (float(np.mean(scores)) if scores else float('nan'),
            float(np.std(scores)) if scores else float('nan'))


def peer_table(by_client):
    """Cumulative peer evidence per (arm, host, round), vectorised.

    Returns ``(arms, hosts, count, total)`` where ``count[r, a, h]`` is how many
    times host ``h`` observed arm ``a`` in rounds ``0..r`` and ``total`` is the
    matching reward sum.  Subtracting a host from the axis-2 sum then yields that
    host's *peers* at round ``r`` - causal by construction, because round ``r``
    reads only ``r-1``.
    """
    hosts = sorted(by_client)
    arms = sorted({expert_id for records in by_client.values()
                   for record in records for expert_id in record['gains']})
    index = {arm: position for position, arm in enumerate(arms)}
    rounds = max(len(records) for records in by_client.values())
    count = np.zeros((rounds, len(arms), len(hosts)))
    total = np.zeros_like(count)
    for position, host in enumerate(hosts):
        for step, record in enumerate(by_client[host]):
            for expert_id, gain in record['gains'].items():
                count[step, index[expert_id], position] += 1.0
                total[step, index[expert_id], position] += float(gain)
    return arms, hosts, count.cumsum(axis=0), total.cumsum(axis=0)


def replay(by_client, candidates, cap=4.0, reward='relative', min_peer=1,
           recorded_priors=True, mode='level', **kwargs):
    """Causal replay where each host sees peer observations of the same arm.

    Two prior sources are mixed on the *evidence* scale rather than the
    similarity scale: the recorded regime prior (a similarity in [0, 1], i.e. one
    pseudo-observation at most) and the peer mean over ``n`` real observations.

    ``mode='level'`` transfers the peers' mean gain for that arm.  That is the
    wrong quantity when the host main effect is the dominant term: it moves the
    estimate toward a level this host does not share.  ``mode='additive'``
    transfers only the arm's *deviation* from the cross-host grand mean and adds
    it back on top of this host's own level - i.e. a two-way additive model
    ``y[h, a] = mu + host_h + arm_a`` where ``host_h`` is measured locally and
    ``arm_a`` is measured on 31 other hosts.
    """
    rounds = int(sum(len(records) for records in by_client.values()))
    arms, hosts, count, total = peer_table(by_client)
    position = {arm: index for index, arm in enumerate(arms)}
    total_default = total_deployed = 0.0
    deploys = covered = blind = decisions = 0
    earn = []
    for host in sorted(by_client):
        records = by_client[host]
        own = hosts.index(host)
        controller = EvidenceController(**kwargs)
        arms_ = [SimpleNamespace(expert_id=expert_id) for expert_id in candidates[host]]
        costs = {expert_id: 0.0 for expert_id in candidates[host]}
        seen_locally = collections.Counter()
        for step, record in enumerate(records):
            raw = record.get('raw_gains') or {
                key: value * record['default_loss']
                for key, value in record['gains'].items()}
            priors = {}
            base = (record.get('priors') or {}) if (recorded_priors and reward == 'relative') else {}
            for expert_id, value in base.items():
                gain, weight = (value if isinstance(value, (tuple, list)) else (value, 1.0))
                priors[expert_id] = (float(gain), float(weight))
            if cap > 0 and step > 0:
                peer_count = count[step - 1].sum(axis=1) - count[step - 1][:, own]
                peer_total = total[step - 1].sum(axis=1) - total[step - 1][:, own]
                if mode == 'additive':
                    own_count = float(count[step - 1][:, own].sum())
                    own_sum = float(total[step - 1][:, own].sum())
                    grand_count = float(count[step - 1].sum())
                    grand_sum = float(total[step - 1].sum())
                    own_level = own_sum / own_count if own_count else None
                    grand = grand_sum / grand_count if grand_count else None
                    if own_level is None or grand is None:
                        own_level = grand = 0.0
                for expert_id in candidates[host]:
                    slot = position[expert_id]
                    number = float(peer_count[slot])
                    if number < min_peer:
                        continue
                    peer_gain = float(peer_total[slot] / number)
                    if mode == 'additive':
                        peer_gain = own_level + peer_gain - grand
                    peer_weight = min(cap, number)
                    if expert_id in priors:
                        gain0, weight0 = priors[expert_id]
                        scale = weight0 + peer_weight
                        priors[expert_id] = (
                            (gain0 * weight0 + peer_gain * peer_weight) / scale, scale)
                    else:
                        priors[expert_id] = (peer_gain, peer_weight)
            controller.begin_round(record.get('cumulative_drift', 0.0))
            controller.set_regime_priors(priors)
            active, _ = controller.choose_transfer(arms_, costs)
            decisions += 1
            total_default += record['default_loss']
            if active is not None:
                deploys += 1
                gain = raw.get(active)
                if gain is None:
                    total_deployed += record['default_loss']
                else:
                    covered += 1
                    earn.append((record['default_loss'], record['default_loss'] - gain))
                    total_deployed += record['default_loss'] - gain
                    if seen_locally[active] == 0:
                        blind += 1
            else:
                total_deployed += record['default_loss']
            for expert_id, gain in (raw if reward == 'raw' else record['gains']).items():
                controller.record_feedback(expert_id, gain)
            for expert_id in record['gains']:
                seen_locally[expert_id] += 1
    gains = np.array([-(item[1] - item[0]) for item in earn], dtype=float)
    wins, losses = gains[gains > 0], gains[gains < 0]
    return {
        'relative_gain': (total_default - total_deployed) / total_default,
        'deploy_rate': deploys / max(1, decisions),
        'deploys': deploys,
        'precision': float((gains > 0).mean()) if len(gains) else 0.0,
        'win': float(wins.mean()) if len(wins) else 0.0,
        'loss': float(losses.mean()) if len(losses) else 0.0,
        'blind_share': blind / max(1, deploys),
        'coverage': covered / max(1, deploys),
        'rounds': rounds,
    }


def show(name, result, width=30):
    print(f"{name:<{width}}{result['relative_gain']:>8.2%}"
          f"{result['deploy_rate']:>8.1%}{result['deploys']:>6}"
          f"{result['precision']:>7.0%}{result['win']:>8.3f}{result['loss']:>8.3f}"
          f"{result['blind_share']:>8.0%}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', default='traffic32_joint_b9')
    parser.add_argument('--root', default='./results/fedtdo')
    parser.add_argument('--caps', nargs='+', type=float, default=[0.0, 1.0, 2.0, 4.0, 8.0, 16.0])
    parser.add_argument('--min-peer', type=int, default=1)
    parser.add_argument('--reward', default='relative', choices=['relative', 'raw'])
    parser.add_argument('--mode', default='level', choices=['level', 'additive'],
                        help='transfer the peers mean, or only the arm deviation')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    path = os.path.join(args.root, args.arm, 'rounds.json')
    by_client = load_rounds(path)
    candidates = arm_ids(by_client)
    table = cells(by_client)

    print(f'arm: {args.arm}   hosts: {len(by_client)}   '
          f'arms: {len({arm for (arm, _) in table})}   cells: {len(table)}')
    per_arm_hosts = collections.Counter(arm for (arm, host) in table)
    histogram = collections.Counter(per_arm_hosts.values())
    print('hosts that evaluated a given arm: '
          + ', '.join(f'{count} host(s): {number} arms'
                      for count, number in sorted(histogram.items())))

    stats = decompose(table)
    print(f'\ntwo-way decomposition of {stats["cells"]} cell means '
          f'({stats["arms"]} arms x {stats["hosts"]} hosts, n>=2 per cell)')
    print(f'  arm main effect      R^2 = {stats["r2_arm"]:.3f}  '
          '<- what cross-host pooling can transfer')
    print(f'  host main effect     R^2 = {stats["r2_host"]:.3f}  '
          '<- already known locally, do not import')
    print(f'  arm + host           R^2 = {stats["r2_both"]:.3f}')
    print(f'  left to interaction  R^2 = {1.0 - stats["r2_both"]:.3f}  '
          '<- what pooling cannot transfer')
    mean_r, sd_r = split_half(table)
    print(f'  split-half over hosts, arms only: r = {mean_r:+.3f} (sd {sd_r:.3f}, 200 splits)')

    base = dict(beta=1.0, min_observations=1, estimator='mean', decision='lcb',
                shrink_n0=0.0, shrink_group='global')
    print('\nrule                            gain  deploy     n   prec    win$'
          '   loss$   blind')
    print('-' * 88)
    rows = [show('recorded rule (cap=0)', replay(by_client, candidates, cap=0.0,
                                                 reward=args.reward, **base))]
    for cap in args.caps:
        if cap <= 0:
            continue
        rows.append(show(f'{args.mode} evidence, cap={cap:g}',
                         replay(by_client, candidates, cap=cap, reward=args.reward,
                                mode=args.mode, **base)))
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as handle:
            json.dump({'arm': args.arm, 'decomposition': stats,
                       'split_half': [mean_r, sd_r], 'replay': rows},
                      handle, indent=2, ensure_ascii=False)
        print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
