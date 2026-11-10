"""Offline study of *admission* gates for the transfer decision.

Context.  The deployed rule is ``lcb`` with ``shrink_n0=0``: it transfers only
when an arm's own mean clears a radius that is large at ``n ~ 3``.  Replay says
this is the best of 49 rules, and it wins *by being conservative*, not by
estimating anything well.  A per-deployment breakdown says a deployment earns
+0.065 relative reward on average, while a host (client) whose pooled reward is
positive earns +0.081 at 64% precision.  That suggests conditioning the
*decision to consider a transfer at all* on the host, rather than conditioning
the arm score - which is what the nested ``host_role`` estimator tried, and
lost, because the pooled target is strongly negative and pulls every arm down.

This script asks the narrow question: does any causal host-level statistic
separate rounds where a qualified arm actually pays off, and does gating on it
raise the total (not per-deployment) relative gain?

Gates are causal by construction - they read only feedback from earlier rounds
of the same client - so they can be shipped to the controller unchanged.

Usage::

    python scripts/probe_gate.py --arm traffic32_joint_b9
    python scripts/probe_gate.py --arm traffic32_joint_b9 --sweep
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from federated import FedTDOController  # noqa: E402
from replay_selector import arm_ids, load_rounds, oracle  # noqa: E402


class GatedController(FedTDOController):
    """``choose_transfer`` plus an admission gate on the host's own history.

    Three gates, all evaluated *before* the arm score is allowed to decide:

    ``gate_positive``
        require at least this many strictly positive observations on the arm.
        A single freak horizon cannot qualify an arm, which is the direct
        antidote to the measured 198% winner's curse at ``n ~ 3``;
    ``host_tau``
        require the host's causal mean reward to exceed ``host_tau``;
    ``host_pos_rate``
        require the host's causal share of positive rewards to exceed this.
    """

    def __init__(self, *, gate_positive=0, host_tau=None, host_pos_rate=None,
                 host_min_obs=1, **kwargs):
        super().__init__(**kwargs)
        self.gate_positive = int(gate_positive)
        self.host_tau = None if host_tau is None else float(host_tau)
        self.host_pos_rate = None if host_pos_rate is None else float(host_pos_rate)
        self.host_min_obs = int(host_min_obs)
        self.gate_blocks = 0
        self.gate_checks = 0

    # ------------------------------------------------------------ host stats
    def host_stats(self):
        """Causal pooled statistics of this host, over every arm it observed."""
        values = [item.gain for observations in self.history.values()
                  for item in observations]
        if not values:
            return 0.0, 0.0, 0
        array = np.asarray(values, dtype=float)
        return float(array.mean()), float((array > 0).mean()), len(values)

    def host_admits(self) -> bool:
        if self.host_tau is None and self.host_pos_rate is None:
            return True
        mean, rate, count = self.host_stats()
        if count < self.host_min_obs:
            # Not enough history to judge: allow, so the rule is not silently
            # replaced by "never transfer" in the first rounds.
            return True
        if self.host_tau is not None and mean <= self.host_tau:
            return False
        if self.host_pos_rate is not None and rate <= self.host_pos_rate:
            return False
        return True

    def arm_has_support(self, expert_id) -> bool:
        if self.gate_positive <= 0:
            return True
        observations = self.history.get(expert_id, [])
        return sum(1 for item in observations if item.gain > 0.0) >= self.gate_positive

    def choose_transfer(self, candidates, transfer_costs):
        self.gate_checks += 1
        if not self.host_admits():
            self.gate_blocks += 1
            changed = self.active_expert_id is not None
            self.active_expert_id = None
            return None, changed
        if self.gate_positive <= 0:
            return super().choose_transfer(candidates, transfer_costs)
        filtered = [capsule for capsule in candidates
                    if self.arm_has_support(capsule.expert_id)]
        if not filtered:
            self.gate_blocks += 1
            changed = self.active_expert_id is not None
            self.active_expert_id = None
            return None, changed
        return super().choose_transfer(filtered, transfer_costs)


def replay(by_client, candidates, reward='relative', controller_cls=GatedController,
           **kwargs):
    total_default = 0.0
    total_deployed = 0.0
    decisions = deploys = covered = switches = 0
    earn = []
    for client, records in by_client.items():
        controller = controller_cls(**kwargs)
        client_arms = candidates[client]
        arms = [SimpleNamespace(expert_id=expert_id) for expert_id in client_arms]
        costs = {expert_id: 0.0 for expert_id in client_arms}
        for record in records:
            raw = record.get('raw_gains') or {
                key: value * record['default_loss']
                for key, value in record['gains'].items()
            }
            reward_gains = raw if reward == 'raw' else record['gains']
            controller.begin_round(record.get('cumulative_drift', 0.0))
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
                    earn.append((record['default_loss'], record['default_loss'] - gain))
                    total_deployed += record['default_loss'] - gain
                else:
                    earn.append((record['default_loss'], record['default_loss']))
                    total_deployed += record['default_loss']
            else:
                total_deployed += record['default_loss']
            # batch form: ``shock='centered'`` needs the whole round at once
            if hasattr(controller, 'record_round'):
                controller.record_round(reward_gains)
            else:
                for expert_id, gain in reward_gains.items():
                    controller.record_feedback(expert_id, gain)
    return {
        'relative_gain': (total_default - total_deployed) / total_default,
        'deploy_rate': deploys / max(1, decisions),
        'switches': switches,
        'coverage': covered / max(1, deploys),
        'deploys': deploys,
        'earn': earn,
    }


BASE = dict(beta=1.0, min_observations=1, estimator='mean', decision='lcb',
            shrink_n0=0.0, shrink_group='global')


def describe(name, result, width=34):
    """One line per rule: total capture, deploy rate, precision, payoff asymmetry.

    ``prec`` is ``P(raw gain > 0 | deployed)`` and ``win``/``loss`` are the mean
    raw gain of the winning and losing deployments.  Together they give the
    break-even precision ``|loss| / (win + |loss|)``: below it, deploying more
    often loses more than it earns, no matter how the threshold is moved.
    """
    gains = np.array([-(e[1] - e[0]) for e in result['earn']])  # deployed raw gain
    wins = gains[gains > 0]
    losses = gains[gains < 0]
    prec = float((gains > 0).mean()) if len(gains) else 0.0
    win = float(wins.mean()) if len(wins) else 0.0
    loss = float(losses.mean()) if len(losses) else 0.0
    print(f"{name:<{width}}{result['relative_gain']:>9.2%}{result['deploy_rate']:>9.1%}"
          f"{result['deploys']:>7}{prec:>8.0%}{win:>8.3f}{loss:>8.3f}{result['coverage']:>8.0%}")
    return result


def replay_restricted(by_client, candidates, beta_scale=1.0, restrict=True,
                      **kwargs):
    """Replay where only the arms measured *this* round may be deployed.

    Two reasons this is the only honest way to sweep the admission radius:

    * the evaluated set is produced by ``select_probes``, which scores arms with
      ``for_exploration=True`` and therefore never reads the deployment rule -
      so restricting to it stays on-policy, and ``coverage`` is 100% by
      construction instead of silently crediting zero to unmeasured arms;
    * restricting the action space to ``b`` arms per round legitimately shrinks
      the union bound from ``log(2KT/delta)`` to ``log(2bT/delta)``, which is a
      1.4x smaller radius at ``b=9, K=124``.  Passing the full pool while only
      ever deploying measured arms keeps the *old* radius, so the two effects
      have to be separated explicitly.
    """
    total_default = total_deployed = 0.0
    deploys = 0
    gains = []
    per_client = {}
    for client, records in by_client.items():
        controller = GatedController(
            **{key: (value * beta_scale if key == 'beta' else value)
               for key, value in kwargs.items() if key != 'reward'})
        contributed = 0.0
        default_sum = 0.0
        client_deploys = 0
        for record in records:
            raw = record['raw_gains']
            pool = [expert_id for expert_id in record.get('probe_ids', [])
                    if expert_id in raw] if restrict else candidates[client]
            if not pool:
                pool = candidates[client]
            arms = [SimpleNamespace(expert_id=expert_id) for expert_id in pool]
            controller.begin_round(record.get('cumulative_drift', 0.0))
            controller.set_regime_priors(
                {key: (value[0], value[1])
                 for key, value in (record.get('priors') or {}).items()}
            )
            active, _ = controller.choose_transfer(
                arms, {expert_id: 0.0 for expert_id in pool})
            default_sum += record['default_loss']
            total_default += record['default_loss']
            if active is None:
                total_deployed += record['default_loss']
            else:
                deploys += 1
                client_deploys += 1
                gain = float(raw.get(active, 0.0))
                gains.append(gain)
                contributed += gain
                total_deployed += record['default_loss'] - gain
            if hasattr(controller, 'record_round'):
                controller.record_round(record['gains'])
            else:
                for expert_id, gain in record['gains'].items():
                    controller.record_feedback(expert_id, gain)
        per_client[client] = (client_deploys, contributed, default_sum)
    return {
        'relative_gain': (total_default - total_deployed) / total_default,
        'deploys': deploys,
        'earn': [(0.0, -gain) for gain in gains],
        'coverage': 1.0,
        'deploy_rate': deploys / max(1, sum(len(v) for v in by_client.values())),
        'per_client': per_client,
    }


def concentration(result, by_client):
    """Is the measured gain a property of the policy, or of two clients?

    The replay can only support a policy change if the gain is spread over many
    hosts.  If two hosts supply almost all of it, every "improvement" found by
    tuning on the log is a bet that those two hosts keep cooperating.
    """
    per_client = result['per_client']
    total = sum(value[1] for value in per_client.values())
    order = sorted(per_client.items(), key=lambda item: -item[1][1])
    print(f"  {'client':<12}{'deploys':>8}{'raw gain':>11}{'share':>9}")
    for client, (deploys, contributed, _) in order[:6]:
        share = contributed / total if total else 0.0
        print(f"  {client:<12}{deploys:>8}{contributed:>+11.3f}{share:>9.0%}")
    top2 = sum(value[1] for _, value in order[:2]) / total if total else 0.0
    print(f"  top-2 share of the total gain: {top2:.0%}")
    return top2


def host_table(by_client):
    """How much of the reward is a host property, and is it stable over time?"""
    print('host-level reward structure (causal split at the midpoint):')
    print(f"{'client':<10}{'n':>6}{'mean':>9}{'pos rate':>10}{'median':>9}"
          f"{'first half':>12}{'second half':>13}{'default loss':>14}")
    first, second, means = [], [], []
    for client, records in by_client.items():
        gains = [gain for record in records for gain in record['gains'].values()]
        half = len(records) // 2
        early = [gain for record in records[:half] for gain in record['gains'].values()]
        late = [gain for record in records[half:] for gain in record['gains'].values()]
        default = [record['default_loss'] for record in records]
        means.append(np.mean(gains))
        first.append(np.mean(early))
        second.append(np.mean(late))
        print(f"{client:<10}{len(gains):>6}{np.mean(gains):>+9.3f}"
              f"{np.mean(np.array(gains) > 0):>10.1%}{np.median(gains):>+9.3f}"
              f"{np.mean(early):>+12.3f}{np.mean(late):>+13.3f}{np.mean(default):>14.3f}")
    corr = np.corrcoef(first, second)[0, 1]
    print(f'  split-half correlation of host mean reward: {corr:+.3f}')
    print(f'  spread of host means: sd={np.std(means):.3f} over {len(means)} hosts\n')


def capture_table(result, by_client):
    """Per-client oracle vs realised, and the capture rate by oracle bucket.

    The point of this table: the decision rule cannot be judged on the aggregate
    alone, because the aggregate mixes hosts where a peer expert is genuinely
    much better with hosts where the difference is inside the noise.  Binning by
    the client's own oracle shows whether the rule "converts when the signal is
    there" or simply "converts on two lucky hosts".
    """
    rows = []
    for client, records in by_client.items():
        default = sum(record['default_loss'] for record in records)
        best = sum(max([0.0] + list(record['raw_gains'].values()))
                   for record in records)
        rows.append((client, default, best, result['per_client'][client]))
    total_default = sum(row[1] for row in rows)
    total_best = sum(row[2] for row in rows)

    print(f"  {'client':<12}{'oracle':>9}{'realised':>10}{'capture':>9}{'deploys':>9}")
    for client, default, best, (deploys, realised, _) in sorted(
            rows, key=lambda row: -row[2] / row[1])[:8]:
        print(f"  {client:<12}{100 * best / default:>8.1f}%"
              f"{100 * realised / default:>9.2f}%"
              f"{(100 * realised / best if best else 0):>8.0f}%{deploys:>9}")
    print(f"  {'--- buckets ---':<12}")
    for low, high, label in ((0.20, 1.0, 'oracle >=20%'), (0.10, 0.20, '10-20%'),
                             (0.05, 0.10, '5-10%'), (0.0, 0.05, '<5%')):
        picked = [row for row in rows if low <= row[2] / row[1] < high]
        if not picked:
            continue
        default = sum(row[1] for row in picked)
        best = sum(row[2] for row in picked)
        realised = sum(row[3][1] for row in picked)
        deploys = sum(row[3][0] for row in picked)
        print(f"  {label:<12}{len(picked):>3} hosts  oracle {100 * best / default:>5.1f}%"
              f"  realised {100 * realised / default:>+6.2f}%"
              f"  capture {(100 * realised / best if best else 0):>4.0f}%"
              f"  {deploys / len(picked):>5.1f} deploys/host")
    print(f"  total oracle {100 * total_best / total_default:.2f}%   "
          f"realised {100 * result['relative_gain']:.2f}%   "
          f"capture {100 * result['relative_gain'] / (total_best / total_default):.0f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', default='traffic32_joint_b9')
    parser.add_argument('--root', default='./results/fedtdo')
    parser.add_argument('--reward', default='relative', choices=['relative', 'raw'])
    parser.add_argument('--sweep', action='store_true')
    parser.add_argument('--host-table', action='store_true')
    parser.add_argument('--radius', action='store_true',
                        help='sweep the admission radius on the measured-only pool')
    parser.add_argument('--capture', action='store_true',
                        help='per-client oracle vs realised, binned by oracle')
    args = parser.parse_args()

    path = os.path.join(args.root, args.arm, 'rounds.json')
    by_client = load_rounds(path)
    ids = arm_ids(by_client)
    base = dict(BASE, reward=args.reward)

    print(f'arm: {args.arm}   reward: {args.reward}   '
          f'oracle: {oracle(by_client):+.2%}\n')
    if args.host_table:
        host_table(by_client)

    gains_all = np.array([gain for records in by_client.values()
                          for record in records
                          for gain in record['gains'].values()])
    wins = gains_all[gains_all > 0]
    losses = gains_all[gains_all <= 0]
    print(f'reward sample: {len(gains_all)} evaluations, '
          f'{len(wins) / len(gains_all):.1%} positive, mean {gains_all.mean():+.3f}')
    print(f'  unconditional payoff: win {wins.mean():+.3f} vs loss {losses.mean():+.3f} '
          f'-> break-even precision {abs(losses.mean()) / (wins.mean() + abs(losses.mean())):.0%}'
          f' (a conditional subset is easier: the rule truncates the loss tail)\n')

    header = (f"{'rule':<34}{'gain':>9}{'deploy':>9}{'n':>7}{'prec':>8}"
              f"{'win$':>8}{'loss$':>8}{'cover':>8}")
    print(header)
    print('-' * len(header))

    baseline = describe('recorded rule (lcb, n0=0)',
                        replay(by_client, ids, **base))

    print('\nA. arm-level support gate (require k positive observations)')
    for k in (1, 2, 3):
        describe(f'  gate_positive={k}',
                 replay(by_client, ids, gate_positive=k, **base))

    print('\nB. host admission gate (host causal mean reward > tau)')
    for tau in (-0.60, -0.45, -0.30, -0.15, 0.0):
        describe(f'  host_tau={tau:+.2f}',
                 replay(by_client, ids, host_tau=tau, host_min_obs=5, **base))

    print('\nC. host admission gate (host causal positive rate > p)')
    for p in (0.20, 0.25, 0.30, 0.35, 0.40):
        describe(f'  host_pos_rate={p:.2f}',
                 replay(by_client, ids, host_pos_rate=p, host_min_obs=5, **base))

    print('\nD. where the gain comes from (recorded rule, full pool)')
    # ``restrict=False`` reproduces the recorded rule exactly (1.81%, 151 deploys);
    # it is used here only because it also returns per-client accounting.
    full = replay_restricted(by_client, ids, beta_scale=1.0, restrict=False, **base)
    describe('  full pool, beta=1.00', full)
    concentration(full, by_client)
    print('  measured-only pool (same radius, coverage 100% by construction)')
    restricted = replay_restricted(by_client, ids, **base)
    describe('  measured-only, beta=1.00', restricted)

    if args.radius:
        print('\nE. admission radius, measured-only pool (coverage 100% by '
              'construction)')
        rows = []
        for scale in (1.30, 1.15, 1.00, 0.85, 0.70, 0.55, 0.40, 0.25):
            result = replay_restricted(by_client, ids, beta_scale=scale, **base)
            result['scale'] = scale
            rows.append(result)
            describe(f'  beta={scale:.2f}', result)
        best = max(rows, key=lambda item: item['relative_gain'])
        print(f'  best beta={best["scale"]:.2f} at {best["relative_gain"]:+.2%} '
              f'vs recorded {baseline["relative_gain"]:+.2%}')
        concentration(best, by_client)

    if args.capture:
        print('\nG. per-client capture under the measured-only variant')
        capture_table(replay_restricted(by_client, ids,
                                        beta_scale=best['scale'] if args.radius else 1.0,
                                        **base), by_client)

    if not args.sweep:
        return

    print('\nF. joint sweep (host_pos_rate x gate_positive)')
    rows = []
    for p in (None, 0.20, 0.25, 0.30, 0.35):
        for k in (0, 1, 2):
            for tau in (None, -0.30, -0.15):
                name = (f"p={'-' if p is None else f'{p:.2f}'} k={k} "
                        f"tau={'-' if tau is None else f'{tau:+.2f}'}")
                result = replay(by_client, ids, gate_positive=k, host_tau=tau,
                                host_pos_rate=p, host_min_obs=5, **base)
                rows.append((name, result))
    rows.sort(key=lambda item: -item[1]['relative_gain'])
    for name, result in rows[:15]:
        describe(name, result)


if __name__ == '__main__':
    main()
