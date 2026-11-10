"""One row per recorded arm for the budget table.

The report's §8 compares budgets, and the columns come from three different
places: ``metrics.json`` (gain, MSE, transfer count, evaluation count),
``rounds.json`` (how much of the budget was actually spent, i.e. probes per
round and how many rounds spent nothing) and a replay of the live rule on the
same log (precision, mean win).  Reading them off by hand is how a table ends up
quoting a number that belongs to another arm, so this emits the whole row at
once.

Usage::

    python scripts/probe_budget_table.py --arms traffic32_joint_b9 \\
        traffic32_s2021_joint_b20 traffic32_s2021_joint_b9_fix \\
        traffic32_s2021_joint_b20_fix
    python scripts/probe_budget_table.py --arms ... --markdown
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from probe_gate import BASE, replay  # noqa: E402
from replay_selector import arm_ids, load_rounds  # noqa: E402


def row(root, arm, reward='relative'):
    directory = os.path.join(root, arm)
    with open(os.path.join(directory, 'metrics.json'), 'r', encoding='utf-8') as handle:
        metrics = json.load(handle)
    config = metrics.get('config', {})
    actions = metrics.get('actions', {})

    by_client = load_rounds(os.path.join(directory, 'rounds.json'))
    candidates = arm_ids(by_client)
    counts = [len(record['probe_ids']) for records in by_client.values()
              for record in records]
    result = replay(by_client, candidates, reward=reward, **BASE)
    gains = np.array([-(item[1] - item[0]) for item in result['earn']], dtype=float)
    wins = gains[gains > 0]
    losses = gains[gains < 0]

    return {
        'arm': arm,
        'budget': config.get('fedtdo_eval_budget'),
        'probe_floor': config.get('fedtdo_probe_floor'),
        'online_steps': config.get('fedtdo_online_steps'),
        'gain': float(actions.get('relative_gain', 0.0)),
        'mean_mse': float(metrics.get('mean_client_mse', metrics.get('mse', float('nan')))),
        'transfers': int(actions.get('deployments', actions.get('transfers', 0))),
        'evaluations': int(actions.get('evaluations', 0)),
        'probes_mean': float(np.mean(counts)) if counts else 0.0,
        'probes_median': float(np.median(counts)) if counts else 0.0,
        'zero_rounds': int(sum(1 for value in counts if value == 0)),
        'rounds': len(counts),
        'precision': float((gains > 0).mean()) if len(gains) else 0.0,
        'win': float(wins.mean()) if len(wins) else 0.0,
        'loss': float(losses.mean()) if len(losses) else 0.0,
        'breakeven': (float(-losses.mean() / (wins.mean() - losses.mean()))
                      if len(wins) and len(losses) else float('nan')),
        'ev_per_deploy': float(gains.mean()) if len(gains) else 0.0,
        'distinct_arms': len({expert_id for records in by_client.values()
                              for record in records for expert_id in record['gains']}),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arms', nargs='+', required=True)
    parser.add_argument('--root', default='./results/fedtdo')
    parser.add_argument('--reward', default='relative', choices=['relative', 'raw'])
    parser.add_argument('--markdown', action='store_true')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    rows = []
    for arm in args.arms:
        if not os.path.exists(os.path.join(args.root, arm, 'metrics.json')):
            print(f'missing {arm}', file=sys.stderr)
            continue
        rows.append(row(args.root, arm, args.reward))

    if args.markdown:
        print('| 臂 | 预算 | probe_floor | 增益 | mean MSE | 部署 | 评估数 | '
              '精确率 | 赢均值 | 每轮探针(均值/中位) | 0 探针轮次 | 覆盖臂数 |')
        print('|---|---|---|---|---|---|---|---|---|---|---|---|')
        for item in rows:
            floor = item['probe_floor']
            floor = '旧' if floor in (None, 0) else '新'
            print(f"| `{item['arm']}` | {item['budget']} | {floor} | "
                  f"**{100 * item['gain']:+.2f}%** | {item['mean_mse']:.5f} | "
                  f"{item['transfers']} | {item['evaluations']} | "
                  f"{100 * item['precision']:.0f}% | {item['win']:+.3f} | "
                  f"{item['probes_mean']:.1f}/{item['probes_median']:.0f} | "
                  f"{item['zero_rounds']}/{item['rounds']} | {item['distinct_arms']} |")
    else:
        for item in rows:
            print(json.dumps(item, ensure_ascii=False))
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as handle:
            json.dump(rows, handle, indent=2, ensure_ascii=False)
        print(f'\nwrote {args.out}', file=sys.stderr)


if __name__ == '__main__':
    main()
