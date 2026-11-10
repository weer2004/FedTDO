"""Summarise FedTDO arms (method / isomorphic / local control) into one table.

Usage:
  python scripts/compare_arms.py results/fedtdo/solar_fedtdo \
      results/fedtdo/solar_isomorphic results/fedtdo/solar_local_control
"""

from __future__ import annotations

import argparse
import json
import os


def load(path):
    metrics_path = os.path.join(path, 'metrics.json') if os.path.isdir(path) else path
    with open(metrics_path, encoding='utf-8') as handle:
        return json.load(handle)


def run_name(path):
    path = os.path.normpath(str(path))
    if os.path.isdir(path):
        return os.path.basename(path)
    return os.path.basename(os.path.dirname(path))


def describe(path):
    data = load(path)
    actions = data.get('actions', {})
    clients = data.get('clients', [])
    default_loss = float(actions.get('default_loss', 0.0))
    deployed_loss = float(actions.get('deployed_loss', 0.0))
    return {
        'run': run_name(path),
        'control': data.get('control', '?'),
        'transfer': data.get('transfer_enabled', True),
        'mse': float(data['mse']),
        'mae': float(data['mae']),
        'default_loss': default_loss,
        'deployed_loss': deployed_loss,
        'gain_pct': (100.0 * (default_loss - deployed_loss) / default_loss) if default_loss else 0.0,
        'evaluations': int(actions.get('evaluations', 0)),
        'transfers': int(actions.get('transfers', 0)),
        'deployed_clients': int(actions.get('clients_with_active_expert', 0)),
        'n_clients': len(clients),
        'clients': clients,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('runs', nargs='+')
    args = parser.parse_args()

    rows = [describe(path) for path in args.runs]
    header = f"{'arm':<22}{'control':<17}{'migr':<7}{'MSE':>10}{'MAE':>10}" \
             f"{'default':>11}{'deployed':>11}{'gain%':>8}{'eval':>7}{'xfer':>6}{'active':>8}"
    print(header)
    print('-' * len(header))
    for row in rows:
        print(f"{row['run']:<22}{row['control']:<17}{str(row['transfer']):<7}"
              f"{row['mse']:>10.4f}{row['mae']:>10.4f}"
              f"{row['default_loss']:>11.4f}{row['deployed_loss']:>11.4f}"
              f"{row['gain_pct']:>8.2f}{row['evaluations']:>7}{row['transfers']:>6}"
              f"{row['deployed_clients']:>4}/{row['n_clients']:<3}")

    if len(rows) >= 2:
        base = rows[-1]
        print(f"\nRelative to {base['run']} (last argument):")
        for row in rows[:-1]:
            delta = 100.0 * (row['mse'] - base['mse']) / base['mse'] if base['mse'] else 0.0
            print(f"  {row['run']:<22} mse {delta:+.2f}%   "
                  f"default loss {100.0 * (row['default_loss'] - base['default_loss']) / base['default_loss']:+.2f}%")

    print('\nPer-client deployed vs default:')
    for row in rows:
        print(f"  [{row['run']}]")
        for client in row['clients']:
            default = float(client.get('default_loss', 0.0))
            deployed = float(client.get('deployed_loss', 0.0))
            gain = 100.0 * (default - deployed) / default if default else 0.0
            costs = client.get('costs', {})
            print(f"    {client.get('client_id', '?'):<12} default={default:9.4f} "
                  f"deployed={deployed:9.4f} gain={gain:+6.2f}% "
                  f"eval={costs.get('evaluations', 0):>4} xfer={costs.get('transfers', 0):>3}")


if __name__ == '__main__':
    main()
