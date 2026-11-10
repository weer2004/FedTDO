"""Audit what the evaluation budget actually bought.

``--fedtdo_eval_budget`` is documented as "how many shadow evaluations per
round", but the realised count is a policy output, not a parameter.  Two
mechanisms suppress it, both measured here:

1. ``select_probes`` breaks on ``mean + radius <= probe_threshold``.  Once the
   regime priors cover the candidate pool every mean is negative and the radius
   is a few hundredths, so the break fires on the first candidate and the round
   evaluates nothing.
2. Ties among never-evaluated arms are all ``0 + unseen_uncertainty``, so the
   ordering is decided by the id — i.e. alphabetically, and capsule ids start
   with the publishing client's name.  The exploration frontier therefore
   advances as ``budget x rounds`` positions down the alphabet, so a *larger*
   budget can reach *fewer* distinct peers once priors arrive.

Usage::

    python scripts/probe_audit.py --arms traffic32_joint_b9 traffic32_s2021_joint_b20
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from replay_selector import load_rounds  # noqa: E402


def audit(root, arm, client=None):
    path = os.path.join(root, arm, 'rounds.json')
    by_client = load_rounds(path)
    with open(os.path.join(root, arm, 'metrics.json'), 'r', encoding='utf-8') as handle:
        metrics = json.load(handle)
    config = metrics.get('config', {})
    counts = [len(record['probe_ids']) for records in by_client.values()
              for record in records]
    print(f'== {arm}')
    print(f'   budget={config.get("fedtdo_eval_budget")}  '
          f'online_steps={config.get("fedtdo_online_steps")}  '
          f'probe_floor={config.get("fedtdo_probe_floor", "legacy")}  '
          f'gain={100 * metrics["actions"]["relative_gain"]:+.2f}%  '
          f'evaluations={metrics["actions"]["evaluations"]}')
    print(f'   probes/round: mean {np.mean(counts):.1f}  median {np.median(counts):.0f}  '
          f'max {max(counts)}  zero {sum(1 for c in counts if c == 0)}/{len(counts)}')
    histogram = dict(sorted(collections.Counter(counts).items()))
    print(f'   histogram: {histogram}')

    if client is None:
        # The client with the most evaluations is the most informative one.
        client = max(by_client, key=lambda name: sum(
            len(record['probe_ids']) for record in by_client[name]))
    records = sorted(by_client[client], key=lambda record: record['step'])
    step = max(1, len(records) // 6)
    trace = [(len(record['probe_ids']), len(record.get('priors') or {}))
             for record in records[::step]]
    print(f'   {client}: (probes, priors) every {step} rounds -> {trace}')

    gains = collections.defaultdict(list)
    for record in records:
        for expert_id, gain in record['raw_gains'].items():
            gains[expert_id].append(float(gain))
    ranked = sorted(((float(np.mean(values)), len(values), expert_id)
                     for expert_id, values in gains.items()), reverse=True)[:3]
    positive = np.mean([gain > 0 for values in gains.values() for gain in values])
    best = max(gain for values in gains.values() for gain in values)
    print(f'   {client}: {len(gains)} distinct arms, best single-round gain {best:+.2f}, '
          f'positive share {positive:.0%}')
    print(f'   top arms by mean gain: '
          + ', '.join(f'{expert_id} {mean:+.2f} (n={n})' for mean, n, expert_id in ranked))
    return {
        'arm': arm,
        'probes_mean': float(np.mean(counts)),
        'zero_rounds': int(sum(1 for c in counts if c == 0)),
        'rounds': len(counts),
        'gain': float(metrics['actions']['relative_gain']),
        'client': client,
        'distinct_arms': len(gains),
        'best_gain': float(best),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arms', nargs='+', default=['traffic32_joint_b9'])
    parser.add_argument('--root', default='./results/fedtdo')
    parser.add_argument('--client', default=None)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    rows = [audit(args.root, arm, args.client) for arm in args.arms]
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as handle:
            json.dump(rows, handle, indent=2, ensure_ascii=False)
        print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
