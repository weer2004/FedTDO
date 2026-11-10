"""Rank FedTDO decision rules against logged rewards, without retraining.

Every ``rounds.json`` now stores the reward each evaluated expert produced
(``gains``).  Because deployment is an inference-time convex mix and no weight
is updated from it, those rewards are (to first order) independent of the
decision the controller made - so a full training run can be replayed under
many decision rules, and only the winner has to be re-run end to end.

Caveats, stated up front:

* the evaluated set is *not* policy independent - it is the probes plus the
  active expert - so a rule that deploys an arm which was not evaluated this
  round is credited with zero gain.  ``coverage`` reports how often that
  happens; treat results below ~80% coverage as indicative only;
* alpha calibration consumes every evaluated expert, so changing the probe
  policy would change the gains themselves.  The probe policy is held fixed
  here, which is exactly what makes the comparison cheap.

Usage::

    python scripts/replay_selector.py --arm solar_rel_p24
    python scripts/replay_selector.py --arm solar_rel_p24 --grid
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from federated import FedTDOController  # noqa: E402


def load_rounds(path):
    with open(path, 'r', encoding='utf-8') as handle:
        rounds = json.load(handle)
    by_client = {}
    for record in rounds:
        if 'gains' not in record:
            raise SystemExit(
                'rounds.json predates per-expert reward logging; re-run the arm '
                'once with the current code to replay it'
            )
        by_client.setdefault(record['client'], []).append(record)
    for records in by_client.values():
        records.sort(key=lambda item: item['step'])
    return by_client


def arm_ids(by_client):
    """Candidate arms *per client*: a client never sees its own capsules.

    The union over the log would hand a client the capsules it published itself,
    which ``ExpertRegistry.candidates_for`` filters out.  That inflates the arm
    count (28 -> 32), shifts the ``log(2KT/delta)`` radius and can select an arm
    the client can never deploy - the first version of this script scored the
    historical rule 0.65pp too well because of it.
    """
    per_client = {}
    for client, records in by_client.items():
        candidate = {expert_id for record in records for expert_id in record['gains']}
        per_client[client] = sorted(
            expert_id for expert_id in candidate
            if not expert_id.startswith(f'{client}:')
        )
    return per_client


def replay(by_client, candidates, reward='relative', **controller_kwargs):
    """Walk each client's rounds in causal order under one decision rule."""
    total_default = 0.0
    total_deployed = 0.0
    decisions = 0
    deploys = 0
    covered = 0
    switches = 0
    for client, records in by_client.items():
        controller = FedTDOController(**controller_kwargs)
        client_arms = candidates[client]
        arms = [SimpleNamespace(expert_id=expert_id) for expert_id in client_arms]
        costs = {expert_id: 0.0 for expert_id in client_arms}
        for record in records:
            # Two different quantities: ``raw`` is how the realised loss is
            # accounted (the objective), ``reward`` is what the controller eats.
            raw = record.get('raw_gains') or {
                key: value * record['default_loss']
                for key, value in record['gains'].items()
            }
            reward_gains = raw if reward == 'raw' else record['gains']
            controller.begin_round(record.get('cumulative_drift', 0.0))
            # Regime priors were recorded on the relative scale; feeding them
            # to a raw-reward controller would mix units, so they are dropped.
            priors = (record.get('priors') or {}) if reward == 'relative' else {}
            controller.set_regime_priors(
                {expert_id: (value[0], value[1]) for expert_id, value in priors.items()}
            )
            active, switched = controller.choose_transfer(arms, costs)
            decisions += 1
            switches += int(bool(switched))
            total_default += record['default_loss']
            if active is not None:
                deploys += 1
                gain = raw.get(active)
                if gain is not None:
                    covered += 1
                    total_deployed += record['default_loss'] - gain
                else:
                    total_deployed += record['default_loss']
            else:
                total_deployed += record['default_loss']
            for expert_id, gain in reward_gains.items():
                controller.record_feedback(expert_id, gain)
    return {
        'relative_gain': (total_default - total_deployed) / total_default,
        'deploy_rate': deploys / max(1, decisions),
        'switches': switches,
        'coverage': covered / max(1, deploys),
        'decisions': decisions,
    }


def oracle(by_client):
    total_default = 0.0
    total_best = 0.0
    for records in by_client.values():
        for record in records:
            raw = record.get('raw_gains') or {
                key: value * record['default_loss']
                for key, value in record['gains'].items()
            }
            total_default += record['default_loss']
            best = max([0.0] + list(raw.values()))
            total_best += record['default_loss'] - best
    return (total_default - total_best) / total_default


def diagnostics(by_client):
    gains = [gain for records in by_client.values()
             for record in records for gain in record['gains'].values()]
    per_arm = {}
    for records in by_client.values():
        for record in records:
            for expert_id, gain in record['gains'].items():
                per_arm.setdefault((record['client'], expert_id), []).append(gain)
    autocorrs = []
    for series in per_arm.values():
        if len(series) >= 4:
            arr = np.asarray(series, dtype=float)
            if arr.std() > 1e-12:
                autocorrs.append(float(np.corrcoef(arr[:-1], arr[1:])[0, 1]))
    return {
        'rewards': len(gains),
        'positive_rate': float(np.mean([g > 0 for g in gains])) if gains else 0.0,
        'mean_gain': float(np.mean(gains)) if gains else 0.0,
        'mean_positive': float(np.mean([g for g in gains if g > 0])) if gains else 0.0,
        'mean_negative': float(np.mean([g for g in gains if g <= 0])) if gains else 0.0,
        'observations_per_arm': (len(gains) / len(per_arm)) if per_arm else 0.0,
        'arms': len(per_arm),
        'lag1_autocorr': float(np.mean(autocorrs)) if autocorrs else float('nan'),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', default='solar_rel_p24',
                        help='directory name under --root')
    parser.add_argument('--root', default='./results/fedtdo')
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--min_observations', type=int, default=1)
    parser.add_argument('--reward', default='relative', choices=['relative', 'raw'],
                        help='reward the controller consumes: relative (as logged) or '
                             'raw MSE difference (the true objective)')
    parser.add_argument('--grid', action='store_true',
                        help='sweep estimator x decision x shrink_n0')
    parser.add_argument('--hier-grid', action='store_true',
                        help='sweep the nested host->role estimator: (shrink_n0, '
                             'shrink_n0_role) at the recorded reward ruler')
    parser.add_argument('--top', type=int, default=12)
    args = parser.parse_args()

    path = os.path.join(args.root, args.arm, 'rounds.json')
    by_client = load_rounds(path)
    ids = arm_ids(by_client)
    widths = {client: len(value) for client, value in ids.items()}

    stats = diagnostics(by_client)
    print(f"arm: {args.arm}   clients: {len(by_client)}   candidates per client: "
          f"{min(widths.values())}..{max(widths.values())}")
    print(f"rewards: {stats['rewards']}   obs/arm: {stats['observations_per_arm']:.2f}   "
          f"positive: {stats['positive_rate']:.1%}   mean: {stats['mean_gain']:+.4f}")
    print(f"mean|g>0: {stats['mean_positive']:+.4f}   mean|g<=0: {stats['mean_negative']:+.4f}   "
          f"lag1 autocorr: {stats['lag1_autocorr']:+.3f}")
    print(f'oracle (evaluated set only): {oracle(by_client):+.2%}\n')

    base = dict(beta=args.beta, min_observations=args.min_observations,
                reward=args.reward)
    rows = []

    if args.grid:
        combos = itertools.product(
            ('mean', 'winsor', 'cvar', 'quantile'),
            ('lcb', 'posterior'),
            (0.0, 10.0, 40.0),
            ('global', 'role'),
        )
    else:
        combos = [
            ('mean', 'lcb', 0.0, 'global'),
            ('mean', 'posterior', 0.0, 'global'),
            ('winsor', 'posterior', 0.0, 'global'),
            ('cvar', 'posterior', 0.0, 'global'),
            ('quantile', 'posterior', 0.0, 'global'),
            ('mean', 'posterior', 10.0, 'role'),
            ('winsor', 'posterior', 10.0, 'role'),
            ('winsor', 'posterior', 40.0, 'role'),
            ('mean', 'posterior', 40.0, 'role'),
            ('cvar', 'posterior', 10.0, 'role'),
        ]

    if args.hier_grid:
        rows = []
        for n0 in (0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0):
            for n0_role in (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0):
                result = replay(by_client, ids, estimator='mean', decision='lcb',
                                shrink_n0=n0, shrink_group='role',
                                hierarchy='host_role', **{'shrink_n0_role': n0_role},
                                **base)
                result.update({'n0': n0, 'n0_role': n0_role})
                rows.append(result)
        rows.sort(key=lambda item: -item['relative_gain'])
        print(f"{'n0':>6}{'n0_role':>9}{'gain':>10}{'deploy':>8}{'switch':>8}{'cover':>8}")
        for row in rows[:args.top]:
            print(f"{row['n0']:>6.1f}{row['n0_role']:>9.1f}{row['relative_gain']:>9.2%}"
                  f"{row['deploy_rate']:>8.1%}{row['switches']:>8}{row['coverage']:>8.1%}")
        return

    for estimator, decision, shrink_n0, group in combos:
        result = replay(by_client, ids, estimator=estimator, decision=decision,
                        shrink_n0=shrink_n0, shrink_group=group, **base)
        result.update({'estimator': estimator, 'decision': decision,
                       'n0': shrink_n0, 'group': group})
        rows.append(result)

    rows.sort(key=lambda item: -item['relative_gain'])
    print(f"{'estimator':<10}{'decision':<11}{'n0':>5}{'group':<8}{'gain':>10}"
          f"{'deploy':>8}{'switch':>8}{'cover':>8}")
    for row in rows[:args.top]:
        print(f"{row['estimator']:<10}{row['decision']:<11}{row['n0']:>5.0f}"
              f"{row['group']:<8}{row['relative_gain']:>9.2%}{row['deploy_rate']:>8.1%}"
              f"{row['switches']:>8}{row['coverage']:>8.1%}")


if __name__ == '__main__':
    main()
