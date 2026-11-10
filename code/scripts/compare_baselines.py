"""Merge FedTDO arms and FL-baseline runs into one comparable table.

Both runners use the same channel-as-client partition, the same Dynamic-TMoE
backbone and the same online budget (``--fedtdo_online_steps`` *batches* per
client, see ``exp/exp_fl_baselines.py``).  FedTDO reports a pooled MSE over all
clients and a per-client mean; the baselines report only the per-client mean, so
that column is what the table compares.

The last column is the oracle bound of the FedTDO arms: for every revealed
round, replace the controller's choice with the best candidate it had already
*evaluated*.  ``best_gain`` in ``rounds.json`` is a raw MSE difference in the
same units as ``default_loss``, so the bound is additive:

    oracle_loss = sum(default_loss - max(0, best_gain)) / sum(default_loss)

Usage:
    python scripts/compare_baselines.py --match p24
    python scripts/compare_baselines.py --fedtdo_root results/fedtdo \
        --baseline_root results_baselines --match solar
"""

from __future__ import annotations

import argparse
import glob
import json
import os


def load_fedtdo(root, match=None):
    rows = []
    for path in sorted(glob.glob(os.path.join(root, '*', 'metrics.json'))):
        name = os.path.basename(os.path.dirname(path))
        if match and match not in name:
            continue
        with open(path, encoding='utf-8') as handle:
            data = json.load(handle)
        actions = data.get('actions', {})
        rounds_path = os.path.join(os.path.dirname(path), 'rounds.json')
        oracle = None
        if os.path.exists(rounds_path):
            with open(rounds_path, encoding='utf-8') as handle:
                rounds = json.load(handle)
            if rounds:
                total = sum(item['default_loss'] for item in rounds)
                bound = sum(item['default_loss'] - max(0.0, item['best_gain'])
                            for item in rounds)
                oracle = (bound - total) / total if total else None
        rows.append({
            'name': name,
            'kind': 'FedTDO' if data.get('transfer_enabled') else 'FedTDO(no-xfer)',
            'control': data.get('control', '?'),
            'mse': data.get('mean_client_mse', data.get('mse')),
            'mae': data.get('mean_client_mae', data.get('mae')),
            'pooled_mse': data.get('mse'),
            'transfers': actions.get('transfers', 0),
            'evaluations': actions.get('evaluations', 0),
            'gain': actions.get('relative_gain', 0.0),
            'oracle': oracle,
            'client_metrics': data.get('client_metrics', {}),
        })
    return rows


def load_baselines(root, match=None):
    rows = []
    for path in sorted(glob.glob(os.path.join(root, '*', 'summary.json'))):
        setting = os.path.basename(os.path.dirname(path))
        with open(path, encoding='utf-8') as handle:
            data = json.load(handle)
        method = data.get('method', '?')
        prefix = f'BL_{method}_'
        tail = setting[len(prefix):] if setting.startswith(prefix) else setting
        model_id = tail.split('_Solar_')[0].split('_ETT')[0].split('_nc')[0]
        if match and match not in setting and match not in model_id:
            continue
        rows.append({
            'name': f'{method}:{model_id}',
            'kind': f"FL {data.get('venue', '')}".strip(),
            'mse': data.get('mean_mse'),
            'mae': data.get('mean_mae'),
            'transfers': None,
            'evaluations': None,
            'oracle': None,
            'client_metrics': data.get('clients', {}),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fedtdo_root', default='results/fedtdo')
    parser.add_argument('--baseline_root', default='results_baselines')
    parser.add_argument('--match', default=None, help='substring filter')
    args = parser.parse_args()

    rows = load_fedtdo(args.fedtdo_root, args.match) + load_baselines(
        args.baseline_root, args.match)
    rows = [row for row in rows if row['mse'] is not None]
    if not rows:
        print('no runs matched')
        return
    rows.sort(key=lambda row: row['mse'])

    best = min(row['mse'] for row in rows)
    header = (f"{'run':<30}{'kind':<17}{'mse':>9}{'mae':>9}{'vs best':>9}"
              f"{'xfer':>6}{'eval':>6}{'oracle':>9}")
    print(header)
    print('-' * len(header))
    for row in rows:
        rel = (row['mse'] - best) / best * 100
        xfer = '-' if row['transfers'] is None else str(row['transfers'])
        evals = '-' if row['evaluations'] is None else str(row['evaluations'])
        oracle = '-' if row['oracle'] is None else f"{row['oracle'] * 100:+.2f}%"
        print(f"{row['name'][:30]:<30}{row['kind'][:17]:<17}{row['mse']:>9.5f}"
              f"{row['mae']:>9.5f}{rel:>8.2f}%{xfer:>6}{evals:>6}{oracle:>9}")

    shared = None
    for row in rows:
        if row['client_metrics']:
            ids = set(row['client_metrics'])
            shared = ids if shared is None else (shared & ids)
    if shared:
        ids = sorted(shared)
        print()
        head = f"{'run':<30}" + ''.join(f'{cid.split("_")[-1]:>8}' for cid in ids)
        print(head)
        print('-' * len(head))
        for row in rows:
            if not row['client_metrics']:
                continue
            line = f"{row['name'][:30]:<30}"
            for cid in ids:
                value = row['client_metrics'][cid].get('mse')
                line += f'{value:>8.4f}' if value is not None else f'{"-":>8}'
            print(line)


if __name__ == '__main__':
    main()
