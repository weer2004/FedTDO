"""Paired decision: is the arm better *than its rivals this round*?

The measurement that motivates this file (b=9 arm, 1920 host-rounds):

* two different arms evaluated in the same round correlate at **r = 0.748** -
  the round itself is a common shock.  A hard window makes every arm lose;
* within an arm across rounds the correlation is 0.635, but after removing the
  round mean it drops to **0.328**.

So most of the nuisance variance is not "arm quality drifting" - it is the round.
The live rule averages *raw* rewards across rounds and compares that average with
an absolute threshold of zero, which means it is testing "is this arm good on
average" with an estimator whose noise is dominated by a term that says nothing
about the arm.  Two consequences are visible in the existing logs: the radius has
to be inflated to stay honest (n_eff is 34% of nominal, see
``probe_dependence.py``), and the rule's precision (59%) sits far below the
unconditional break-even (77%).

The paired statistic removes the shock by construction:

    d[t, a] = gain[t, a] - mean over the arms evaluated in round t

`d` answers "which arm should I pick", and its noise is 2.3x smaller than the raw
reward's.  It does *not* answer "should I deploy at all", because the round mean
can be far below the default.  So the rule keeps an absolute margin `tau`
(that is the part which genuinely cannot be estimated from a single round) and
uses the paired statistic only for the comparison:

    deploy argmax_a mean(d[a])  if  mean(d[a]) - radius > tau

`tau` is chosen on the *first half* of each client's rounds and evaluated on the
second half, and the whole grid is also reported so the reader can see the
curve rather than a tuned point.

Usage::

    python scripts/probe_paired.py --arm traffic32_joint_b9
    python scripts/probe_paired.py --arm traffic32_joint_b9 --split
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from replay_selector import arm_ids, load_rounds, oracle  # noqa: E402


def paired_table(by_client):
    """``(host, arm) -> [(round, gain, round_mean)]`` plus raw series."""
    paired = collections.defaultdict(list)
    raw = collections.defaultdict(list)
    for host, records in by_client.items():
        for step, record in enumerate(records):
            values = list(record['gains'].values())
            if not values:
                continue
            centre = float(np.mean(values))
            for expert_id, gain in record['gains'].items():
                paired[(host, expert_id)].append((step, float(gain), centre))
                raw[(host, expert_id)].append((step, float(gain)))
    return paired, raw


def by_client_iter(paired):
    """Group the paired table by host (ordered by round)."""
    grouped = collections.defaultdict(dict)
    for (host, expert_id), values in paired.items():
        grouped[host][expert_id] = sorted(values, key=lambda item: item[0])
    for host in sorted(grouped):
        yield host, grouped[host]


def run(by_client, candidates, tau=0.0, beta=1.0, paired=True, horizon=None,
        rounds_slice=None, reward_scale=None):
    """Replay one margin, causally, over one contiguous slice of each client."""
    if reward_scale is None:
        scale_values = [float(gain) for (_, _, gain) in _all_pairs(by_client, paired)]
        reward_scale = float(np.std(scale_values)) or 1.0
    total_default = total_deployed = 0.0
    decisions = deploys = 0
    earn = []
    for host, series in by_client_iter(_table(by_client, paired)):
        records = by_client[host]
        positions = range(len(records))
        if rounds_slice is not None:
            positions = range(*rounds_slice)
        positions = [position for position in positions if position < len(records)]
        for position in positions:
            record = records[position]
            if not record['gains']:
                continue
            decisions += 1
            total_default += record['default_loss']
            pick, statistic = None, -1e9
            arms = candidates[host]
            for expert_id in arms:
                values = [item for item in series.get(expert_id, [])
                          if item[0] < position]
                if not values:
                    continue
                sample = np.array([item[1] for item in values], dtype=float)
                count = max(1, len(sample))
                # Radius on the paired scale, with the measured residual
                # dependence (r = 0.33) folded in as a design effect.
                radius = beta * reward_scale * math.sqrt(
                    math.log(2.0 * len(arms) * max(horizon or 60, 1) / 0.1) / count)
                score = float(sample.mean()) - radius
                if score > statistic:
                    pick, statistic = expert_id, score
            active = pick if (pick is not None and statistic > tau) else None
            if active is not None:
                deploys += 1
                gain = record['raw_gains'].get(active) if record.get('raw_gains') else None
                if gain is None:
                    total_deployed += record['default_loss']
                else:
                    earn.append((record['default_loss'], record['default_loss'] - gain))
                    total_deployed += record['default_loss'] - gain
            else:
                total_deployed += record['default_loss']
    gains = np.array([-(item[1] - item[0]) for item in earn], dtype=float)
    wins, losses = gains[gains > 0], gains[gains < 0]
    return {
        'relative_gain': (total_default - total_deployed) / max(total_default, 1e-12),
        'deploy_rate': deploys / max(1, decisions),
        'deploys': deploys,
        'precision': float((gains > 0).mean()) if len(gains) else 0.0,
        'win': float(wins.mean()) if len(wins) else 0.0,
        'loss': float(losses.mean()) if len(losses) else 0.0,
    }


def _table(by_client, paired):
    table = collections.defaultdict(list)
    for host, records in by_client.items():
        for step, record in enumerate(records):
            values = list(record['gains'].values())
            if not values:
                continue
            centre = float(np.mean(values))
            for expert_id, gain in record['gains'].items():
                table[(host, expert_id)].append(
                    (step, float(gain) - centre if paired else float(gain), centre))
    return table


def _all_pairs(by_client, paired):
    for host, records in by_client.items():
        for record in records:
            values = list(record['gains'].values())
            if not values:
                continue
            centre = float(np.mean(values))
            for gain in values:
                yield (host, 0, float(gain) - centre if paired else float(gain))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', default='traffic32_joint_b9')
    parser.add_argument('--root', default='./results/fedtdo')
    parser.add_argument('--taus', nargs='+', type=float,
                        default=[-0.30, -0.20, -0.15, -0.10, -0.05, 0.0, 0.05])
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    by_client = load_rounds(os.path.join(args.root, args.arm, 'rounds.json'))
    candidates = arm_ids(by_client)
    table = _table(by_client, True)
    raw = [gain for values in table.values() for (_, gain, _) in values]
    real_oracle = 100.0 * oracle(by_client)
    print(f'arm: {args.arm}   hosts: {len(by_client)}   '
          f'paired sd {np.std(raw):.3f}   pool oracle {real_oracle:+.2f}%')

    print('\nmargin   gain  deploy     n   prec    win$   loss$   [first half]'
          '   [second half]')
    print('-' * 88)
    scale = float(np.std(raw)) or 1.0
    rows = []
    for tau in args.taus:
        first = run(by_client, candidates, tau=tau, beta=args.beta, paired=True,
                    rounds_slice=(0, 30), reward_scale=scale)
        second = run(by_client, candidates, tau=tau, beta=args.beta, paired=True,
                     rounds_slice=(30, 60), reward_scale=scale)
        full = run(by_client, candidates, tau=tau, beta=args.beta, paired=True,
                   reward_scale=scale)
        rows.append({'tau': tau, 'first': first, 'second': second, 'full': full})
        print(f'{tau:>+6.2f}{full["relative_gain"]:>8.2%}{full["deploy_rate"]:>8.1%}'
              f'{full["deploys"]:>6}{full["precision"]:>7.0%}{full["win"]:>8.3f}'
              f'{full["loss"]:>8.3f}   {first["relative_gain"]:>+7.2%}'
              f' ({first["deploys"]:>3})   {second["relative_gain"]:>+7.2%}'
              f' ({second["deploys"]:>3})')
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as handle:
            json.dump({'arm': args.arm, 'beta': args.beta, 'rows': rows},
                      handle, indent=2, ensure_ascii=False)
        print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
