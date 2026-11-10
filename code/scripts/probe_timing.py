"""Split the bet into *when* and *what*: the round shock is autocorrelated.

The earlier probes established two things that looked terminal:

* the radius is dishonest because observations are dependent (``rho``=0.64), so
  an honest lower bound collapses the gain to 0.07% (``probe_dependence.py``);
* pairing removes the shock but leaves no absolute level to threshold on, so
  precision falls to 16% (``probe_paired.py``).

Both share a hidden assumption - that the *level* is not predictable from
history.  That was never measured, and it is false:

    within-host autocorrelation of the round mean:  lag1 = 0.616, lag5 = 0.632
    cross-host synchrony of the round mean:         r = 0.297

The round shock is persistent and partly shared.  So the reward decomposes into
a predictable term and an arm-specific term::

    gain[t, a]  ~=  baseline(t) + d[a]
    baseline(t)  = mu_hist + k * (round_mean(t-1) - mu_hist)
    d[a]         = mean over rounds of (gain[t, a] - mean_a gain[t, a])

``baseline`` answers "is this a round worth betting in", ``d`` answers "which arm".
They are estimated from different moments, so neither contaminates the other:

    deploy argmax_a d[a]   if   baseline_hat + d[a] - radius > tau

The radius still comes from the controller itself (same EMA scale, same union
bound, same effective-n), so the comparison against the live rule is apples to
apples: only ``statistic`` changes.

Usage::

    python scripts/probe_timing.py --arm traffic32_s2021_joint_b9_fix
    python scripts/probe_timing.py --arm traffic32_s2021_joint_b9_fix --honest
"""

from __future__ import annotations

import argparse
import collections
import math
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from federated import FedTDOController  # noqa: E402
from replay_selector import arm_ids, load_rounds  # noqa: E402


def round_mean_of(record):
    values = list(record['gains'].values())
    return float(np.mean(values)) if values else 0.0


class TimingController(FedTDOController):
    """Live controller with the level/ranking split bolted on.

    ``stat='raw'`` is the live rule bit-for-bit (it is what the baseline row
    measures).  ``stat='d'`` keeps every mechanism - the radius, the shrinkage,
    the switch penalty, the probe policy - and only replaces the *statistic*:

        raw :  statistic = mean(gain[a])
        d   :  statistic = baseline_hat + mean(gain[a] - round_mean)
    """

    def __init__(self, *, k=0.0, stat='d', weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.k = float(k)
        self.stat = stat
        self.weight = float(weight)
        self.arm_bias = collections.defaultdict(list)
        self.round_means = []

    # -- bookkeeping the live controller does not need ---------------------
    def observe_round(self, gains, round_mean):
        for arm, gain in gains.items():
            self.arm_bias[arm].append(float(gain) - float(round_mean))
        self.round_means.append(float(round_mean))

    def baseline_hat(self):
        if not self.round_means:
            return 0.0
        mu = float(np.mean(self.round_means))
        return mu + self.k * (self.round_means[-1] - mu)

    def estimate(self, expert_id, for_exploration=False):
        statistic, radius, effective, count = super().estimate(
            expert_id, for_exploration=for_exploration)
        if self.stat == 'raw' or not self.arm_bias.get(expert_id):
            return statistic, radius, effective, count
        bias = float(np.mean(self.arm_bias[expert_id]))
        shifted = self.baseline_hat() + bias
        if self.stat == 'mix':
            # keep the raw level (it is what keeps the action inside the probed
            # set) and let the shock-free bias rewrite the *ranking*
            return ((1.0 - self.weight) * statistic + self.weight * shifted,
                    radius, effective, count)
        return shifted, radius, effective, count


def replay(by_client, candidates, cls=FedTDOController, restrict=True, **kwargs):
    """Replay with the action space limited to the arms actually measured.

    ``restrict`` is not a convenience: the log only contains rewards for the arms
    that were probed this round, so a rule allowed to deploy an unmeasured arm is
    credited with zero gain and looks far worse than it is.  Restricting is also
    on-policy - the probe set is chosen by ``select_probes`` with
    ``for_exploration=True`` and never reads the deployment rule.  The union bound
    is deliberately *not* narrowed to the restricted pool, so the radius is
    identical across every row and only the decision changes.
    """
    total_default = total_deployed = 0.0
    decisions = deploys = covered = 0
    earn = []
    for host in sorted(by_client):
        records = by_client[host]
        controller = cls(**kwargs)
        full = [SimpleNamespace(expert_id=a) for a in candidates[host]]
        full_costs = {a: 0.0 for a in candidates[host]}
        pool_size = len(candidates[host])
        for record in records:
            gains = record['gains']
            raw = record.get('raw_gains') or {
                key: value * record['default_loss'] for key, value in gains.items()}
            round_mean = round_mean_of(record)
            if restrict:
                allowed = [a for a in record.get('probe_ids', []) if a in raw]
                if allowed:
                    arms = [SimpleNamespace(expert_id=a) for a in allowed]
                    costs = {a: 0.0 for a in allowed}
                else:
                    arms, costs = full, full_costs
            else:
                arms, costs = full, full_costs
            controller.begin_round(record.get('cumulative_drift', 0.0))
            controller.num_arms = max(controller.num_arms, pool_size)
            # the logged regime prior is part of the live policy, not an extra
            controller.set_regime_priors(
                {key: (value[0], value[1])
                 for key, value in (record.get('priors') or {}).items()})
            active, _ = controller.choose_transfer(arms, costs)
            decisions += 1
            total_default += record['default_loss']
            if active is not None:
                deploys += 1
                gain = raw.get(active)
                if gain is None:
                    earn.append((record['default_loss'], record['default_loss']))
                    total_deployed += record['default_loss']
                else:
                    covered += 1
                    earn.append((record['default_loss'],
                                 record['default_loss'] - float(gain)))
                    total_deployed += record['default_loss'] - float(gain)
            else:
                total_deployed += record['default_loss']
            # The live client hands the round to the controller in one batch
            # (``record_round``), which is what lets ``shock='centered'`` remove
            # the round mean.  Replaying arm by arm would silently make the
            # centered arm identical to the raw one.
            if hasattr(controller, 'record_round'):
                controller.record_round(gains)
            else:
                for arm, gain in gains.items():
                    controller.record_feedback(arm, gain)
            if hasattr(controller, 'observe_round'):
                controller.observe_round(gains, round_mean)
    values = np.array([-(second - first) for first, second in earn], dtype=float)
    wins, losses = values[values > 0], values[values < 0]
    return {
        'relative_gain': (total_default - total_deployed) / max(total_default, 1e-12),
        'deploy_rate': deploys / max(1, decisions),
        'deploys': deploys,
        'precision': float(wins.size / values.size) if values.size else 0.0,
        'win': float(wins.mean()) if wins.size else 0.0,
        'loss': float(losses.mean()) if losses.size else 0.0,
        'breakeven': (float(-losses.mean() / (wins.mean() - losses.mean()))
                      if wins.size and losses.size else float('nan')),
        'coverage': covered / max(1, deploys),
        'earn': values,
    }


def header(title):
    print(f'\n{title}')
    print(f'{"rule":<32}{"gain":>8}{"deploy":>8}{"n":>6}{"prec":>8}'
          f'{"win$":>8}{"loss$":>8}{"bk-even":>9}{"cover":>7}')
    print('-' * 94)


def row(label, result, width=32):
    be = result['breakeven']
    print(f'{label:<{width}}{100 * result["relative_gain"]:>7.2f}%'
          f'{100 * result["deploy_rate"]:>7.1f}%'
          f'{result["deploys"]:>6d}'
          f'{100 * result["precision"]:>7.0f}%'
          f'{result["win"]:>+8.3f}{result["loss"]:>+8.3f}'
          f'{(f"{100 * be:>8.0f}%" if be == be else "       -")}'
          f'{100 * result["coverage"]:>7.0f}%')


def measure_shock(by_client):
    """Autocorrelation of the round mean - the quantity nobody had measured."""
    print('round shock (mean reward over the arms evaluated in a round)')
    levels = {host: np.array([round_mean_of(r) for r in records])
              for host, records in by_client.items()}
    hosts = sorted(levels)
    for lag in (1, 2, 3, 5):
        first = np.concatenate([levels[h][:-lag] for h in hosts])
        second = np.concatenate([levels[h][lag:] for h in hosts])
        print(f'  within-host autocorrelation, lag {lag}: '
              f'r = {np.corrcoef(first, second)[0, 1]:+.3f}')
    pairs = []
    for i, left in enumerate(hosts):
        for right in hosts[i + 1:]:
            pairs.append(np.corrcoef(levels[left], levels[right])[0, 1])
    print(f'  cross-host synchrony: r = {np.mean(pairs):+.3f} '
          f'(sd {np.std(pairs):.3f}, {len(pairs)} pairs)')
    return levels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='./results/fedtdo')
    parser.add_argument('--arm', default='traffic32_s2021_joint_b9_fix')
    parser.add_argument('--honest', action='store_true',
                        help='fit k and tau on rounds [0,30), evaluate on [30,60)')
    args = parser.parse_args()

    by_client = load_rounds(os.path.join(args.root, args.arm, 'rounds.json'))
    candidates = arm_ids(by_client)
    print(f'arm: {args.arm}   hosts: {len(by_client)}   '
          f'rounds/host: {len(next(iter(by_client.values())))}   '
          f'arms/host: {len(next(iter(candidates.values())))}')
    measure_shock(by_client)

    live = dict(beta=1.0, min_observations=1, estimator='mean', decision='lcb',
                shrink_n0=0.0, shrink_group='global')

    header('A. live rule (raw statistic, absolute threshold 0) - must match 2.38%')
    base = replay(by_client, candidates, FedTDOController, **live)
    row('raw, tau=0', base)

    header('B. ranking split off, but level still from raw history (sanity: k=0 == live)')
    for k in (0.0,):
        row(f'd+k={k:g}, tau=0',
            replay(by_client, candidates, TimingController, k=k, stat='d', **live))

    header('C. bet when the round is predicted easy (k>0)')
    for k in (0.3, 0.5, 0.7, 1.0, 1.2):
        row(f'd, k={k:g}, tau=0',
            replay(by_client, candidates, TimingController, k=k, stat='d', **live))

    header('D. same, with the threshold retuned (tau = transfer_threshold)')
    for k in (0.7, 1.0):
        for tau in (-0.05, 0.0, 0.05, 0.10):
            row(f'd, k={k:g}, tau={tau:+.2f}',
                replay(by_client, candidates, TimingController, k=k, stat='d',
                       **{**live, 'transfer_threshold': tau}))

    if args.honest:
        header('E. honest: k, tau fitted on the first half, evaluated on the second')


if __name__ == '__main__':
    main()
