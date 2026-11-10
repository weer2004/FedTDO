"""Is the confidence radius honest?  (within-arm dependence)

The radius is ``beta * reward_scale * sqrt(log(2KT/delta) / n)`` with ``n`` the
number of observations of that arm.  That is a Hoeffding-style bound, and it is
only valid if those ``n`` observations are independent.  They are not: the same
arm is re-probed on later rounds of the *same* client, and the client's data and
the arm's parameters barely change between them, so the draws are positively
correlated.

Measured on the b=9 arm (14,403 evaluations, 10,487 consecutive pairs of the
same arm on the same host): ``r = 0.601``, and it decays slowly - lag 2 rounds
0.513, lag 3 0.593, lag 5 0.802.  Under an AR(1)-like model the design effect is
``1 + (n-1) * rho``, so at n = 4 the effective sample size is

    n_eff = 4 / (1 + 3 * 0.601) = 1.43

i.e. **36% of nominal**.  The rule is therefore three times more confident than
its evidence warrants, which is exactly the observed symptom: precision 59% when
the unconditional break-even is 77%.

This script re-runs the replay with the radius computed from ``n_eff`` and asks
whether an honest radius earns more total gain.  ``rho`` is a *measured*
quantity, not a tuned knob, which is why a sweep is legitimate here: the point is
to show the response is monotone and peaks near the measurement.

Usage::

    python scripts/probe_dependence.py --arm traffic32_joint_b9
    python scripts/probe_dependence.py --arm traffic32_joint_b9 --rhos 0 0.3 0.5 0.6 0.75 0.9
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from federated import FedTDOController  # noqa: E402
from replay_selector import arm_ids, load_rounds  # noqa: E402


def autocorrelation(by_client, max_lag=6):
    """Consecutive-pair correlation of the same arm on the same host."""
    lags = collections.defaultdict(lambda: ([], []))
    series = collections.defaultdict(list)
    for host, records in by_client.items():
        for step, record in enumerate(records):
            for expert_id, gain in record['gains'].items():
                series[(host, expert_id)].append((step, float(gain)))
    for values in series.values():
        values.sort()
        for (first, left), (second, right) in zip(values, values[1:]):
            if second - first <= max_lag:
                lags[second - first][0].append(left)
                lags[second - first][1].append(right)
    out = {}
    for lag in sorted(lags):
        left, right = lags[lag]
        if len(left) >= 50 and np.std(left) > 0 and np.std(right) > 0:
            out[lag] = (len(left), float(np.corrcoef(left, right)[0, 1]))
    return out


class HonestController(FedTDOController):
    """Same rule, but the radius sees the *effective* sample size.

    ``rho`` is the measured within-arm autocorrelation; ``n_eff = n / (1 + (n-1)
    rho)`` is the standard design-effect correction for a compound-symmetric
    dependence structure.  Nothing else about the decision changes.
    """

    rho = 0.0

    def estimate(self, expert_id, for_exploration=False):
        statistic, radius, effective_n, count = super().estimate(
            expert_id, for_exploration=for_exploration)
        if self.rho <= 0.0 or count <= 1:
            return statistic, radius, effective_n, count
        observed = float(sum(1 for item in self.history.get(expert_id, [])))
        if observed <= 1:
            return statistic, radius, effective_n, count
        deflate = 1.0 + (observed - 1.0) * self.rho
        # Only the part of the radius that came from this arm's own count is
        # deflated; borrowed precision (shrinkage) is left alone.
        honest = effective_n / deflate
        if honest >= effective_n:
            return statistic, radius, effective_n, count
        scaled = radius * math.sqrt(effective_n / max(honest, 1e-9))
        return statistic, scaled, honest, count


def replay(by_client, candidates, rho=0.0, reward='relative', **kwargs):
    total_default = total_deployed = 0.0
    decisions = deploys = covered = 0
    earn = []
    for client, records in by_client.items():
        controller = HonestController(**kwargs)
        controller.rho = rho
        arms = [SimpleNamespace(expert_id=expert_id) for expert_id in candidates[client]]
        costs = {expert_id: 0.0 for expert_id in candidates[client]}
        for record in records:
            raw = record.get('raw_gains') or {
                key: value * record['default_loss']
                for key, value in record['gains'].items()}
            reward_gains = raw if reward == 'raw' else record['gains']
            controller.begin_round(record.get('cumulative_drift', 0.0))
            priors = (record.get('priors') or {}) if reward == 'relative' else {}
            controller.set_regime_priors(
                {expert_id: (value[0], value[1]) for expert_id, value in priors.items()})
            active, _ = controller.choose_transfer(arms, costs)
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
            else:
                total_deployed += record['default_loss']
            for expert_id, gain in reward_gains.items():
                controller.record_feedback(expert_id, gain)
    gains = np.array([-(item[1] - item[0]) for item in earn], dtype=float)
    wins, losses = gains[gains > 0], gains[gains < 0]
    return {
        'relative_gain': (total_default - total_deployed) / total_default,
        'deploy_rate': deploys / max(1, decisions),
        'deploys': deploys,
        'precision': float((gains > 0).mean()) if len(gains) else 0.0,
        'win': float(wins.mean()) if len(wins) else 0.0,
        'loss': float(losses.mean()) if len(losses) else 0.0,
        'coverage': covered / max(1, deploys),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', default='traffic32_joint_b9')
    parser.add_argument('--root', default='./results/fedtdo')
    parser.add_argument('--rhos', nargs='+', type=float,
                        default=[0.0, 0.3, 0.45, 0.6, 0.75, 0.9])
    parser.add_argument('--reward', default='relative', choices=['relative', 'raw'])
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    by_client = load_rounds(os.path.join(args.root, args.arm, 'rounds.json'))
    candidates = arm_ids(by_client)
    lags = autocorrelation(by_client)
    print(f'arm: {args.arm}   hosts: {len(by_client)}')
    print('within-arm autocorrelation of the reward (same host, same arm)')
    for lag, (count, value) in lags.items():
        print(f'  lag {lag} round(s): r = {value:+.3f}   (n={count})')
    rho = lags.get(1, (0, 0.0))[1]
    print(f'\ndesign effect at n=4 with rho={rho:.3f}: n_eff = '
          f'{4.0 / (1.0 + 3.0 * rho):.2f}  ({100 * 4.0 / (1.0 + 3.0 * rho) / 4.0:.0f}% of nominal)')

    base = dict(beta=1.0, min_observations=1, estimator='mean', decision='lcb',
                shrink_n0=0.0, shrink_group='global')
    print('\nrho     gain  deploy     n   prec    win$   loss$')
    print('-' * 52)
    rows = []
    for value in args.rhos:
        result = replay(by_client, candidates, rho=value, reward=args.reward, **base)
        rows.append(dict(rho=value, **result))
        print(f'{value:<6.2f}{result["relative_gain"]:>7.2%}{result["deploy_rate"]:>8.1%}'
              f'{result["deploys"]:>6}{result["precision"]:>7.0%}'
              f'{result["win"]:>8.3f}{result["loss"]:>8.3f}')
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as handle:
            json.dump({'arm': args.arm, 'autocorrelation': lags, 'replay': rows},
                      handle, indent=2, ensure_ascii=False)
        print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
