"""Compare the 8-client and 32-client Traffic federations (plan §10.1).

Three things this script exists to prevent, all of which produced wrong
conclusions in earlier rounds of this project:

1. **Comparing across client counts by absolute MSE.**  The channel draw is
   seeded random, so 8 clients are not a subset of the 32-client draw.  On
   Traffic the 8-client draw happens to sit at the 94th percentile of channel
   difficulty (seasonal-naive MSE 1.235 vs 0.602), so its 0.778 is not
   "worse training", it is a harder subset.  Always print the difficulty audit
   before believing a cross-count difference.
2. **Crediting transfer for a difference it did not cause.**  The no-transfer
   arm and the transfer arm train the identical default model (same seed, same
   steps), so the honest comparison is paired per client, and this script
   verifies bit-exactness of ``default_loss`` first.
3. **Reading the deployment gain off the wrong denominator.**  A round only
   counts if the controller deployed, which happens on ~8% of rounds.  Report
   both the all-rounds number and the conditional-on-deployment number.

Usage::

    python scripts/compare_nc32.py
"""

from __future__ import annotations

import glob
import json
import os
import statistics as st

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FED = os.path.join(ROOT, 'results', 'fedtdo')

ARMS = [
    ('traffic32_iso', 'isomorphic (no share / no freeze)'),
    ('traffic32_joint_b9', 'method: joint warm-up + frozen encoder + transfer'),
    ('traffic32_joint_b2', 'same arm, historical eval_budget=2'),
    ('traffic32_notransfer', 'joint + frozen encoder, transfer disabled'),
]


def load(arm):
    with open(os.path.join(FED, arm, 'metrics.json'), encoding='utf-8') as handle:
        return json.load(handle)


def rounds(arm):
    with open(os.path.join(FED, arm, 'rounds.json'), encoding='utf-8') as handle:
        return json.load(handle)


def local_baseline(pattern='BL_local_traffic32*'):
    paths = glob.glob(os.path.join(ROOT, 'results_baselines', pattern, 'summary.json'))
    if not paths:
        return None
    with open(paths[0], encoding='utf-8') as handle:
        return json.load(handle)['mean_mse']


def table():
    base = local_baseline()
    print(f"{'arm':<52}{'MSE':>9}{'vs local':>10}{'xfer':>6}{'active':>7}{'evals':>8}")
    if base is not None:
        print(f"{'local (pure local, independent init)':<52}{base:9.5f}{'---':>10}"
              f"{'-':>6}{'-':>7}{'-':>8}")
    for arm, label in ARMS:
        data = load(arm)
        actions = data['actions']
        delta = f'{100 * (data["mean_client_mse"] - base) / base:+.2f}%' if base else '-'
        print(f'{label:<52}{data["mean_client_mse"]:9.5f}{delta:>10}'
              f'{actions["transfers"]:>6}{actions["clients_with_active_expert"]:>7}'
              f'{actions["evaluations"]:>8}')


def pairing_check(treated='traffic32_joint_b9', control='traffic32_notransfer'):
    a = {(r['step'], r['client']): r['default_loss'] for r in rounds(treated)}
    b = {(r['step'], r['client']): r['default_loss'] for r in rounds(control)}
    shared = sorted(set(a) & set(b))
    exact = sum(1 for key in shared if a[key] == b[key])
    print(f'\npairing check {treated} vs {control}: {exact}/{len(shared)} rounds bit-exact')
    return exact == len(shared)


def paired(treated='traffic32_joint_b9', control='traffic32_notransfer', top=4):
    a = load(treated)
    b = load(control)
    keys = list(a['client_metrics'])
    gains = np.array([b['client_metrics'][k]['mse'] - a['client_metrics'][k]['mse']
                      for k in keys])
    base = b['mean_client_mse']
    print(f'\npaired transfer effect, {len(keys)} clients')
    print(f'  improved {int((gains > 1e-9).sum())}  worsened {int((gains < -1e-9).sum())}'
          f'  unchanged {int((np.abs(gains) <= 1e-9).sum())}')
    print(f'  mean {gains.mean():+.5f} MSE  = {100 * gains.mean() / base:+.2f}%')
    order = np.argsort(-gains)
    biggest = keys[order[0]]
    share = 100 * gains[order[0]] / len(keys) / gains.mean() if gains.mean() else 0.0
    print(f'  largest single client {biggest} accounts for {share:.0f}% of the total')
    rest = np.delete(gains, order[0])
    print(f'  excluding it: {100 * rest.mean() / base:+.2f}%')
    for index in order[:top]:
        print(f'    + {keys[index]:<12} {b["client_metrics"][keys[index]]["mse"]:.4f}'
              f' -> {a["client_metrics"][keys[index]]["mse"]:.4f}'
              f'  {100 * gains[index] / b["client_metrics"][keys[index]]["mse"]:+6.2f}%')
    for index in order[-2:]:
        if gains[index] < 0:
            print(f'    - {keys[index]:<12} {b["client_metrics"][keys[index]]["mse"]:.4f}'
                  f' -> {a["client_metrics"][keys[index]]["mse"]:.4f}'
                  f'  {100 * gains[index] / b["client_metrics"][keys[index]]["mse"]:+6.2f}%')


def coverage(arm='traffic32_joint_b9'):
    log = rounds(arm)
    active = [r for r in log if r['active_expert']]
    if not active:
        print('\nno round deployed an expert')
        return
    cond = np.array([(r['default_loss'] - r['deployed_loss']) / r['default_loss']
                     for r in active])
    total = (sum(r['default_loss'] - r['deployed_loss'] for r in log)
             / sum(r['default_loss'] for r in log))
    print(f'\ndeployment coverage {len(active)}/{len(log)} rounds'
          f' = {100 * len(active) / len(log):.1f}%')
    print(f'  conditional on deploying: mean {100 * cond.mean():.1f}%'
          f'  median {100 * np.median(cond):.1f}%  positive {100 * (cond > 0).mean():.0f}%')
    print(f'  all-round effect: {100 * total:.2f}%'
          f' -> coverage, not transfer quality, is the binding constraint')


def difficulty_audit(data_path='dataset/traffic/traffic.csv', seq_len=96, pred_len=24):
    """Seasonal-naive MSE per channel, so client draws can be compared fairly."""
    frame = pd.read_csv(os.path.join(ROOT, data_path))
    values = frame.iloc[:, 1:].values.astype(float)
    n = len(values)
    n_train, n_val = int(n * 0.7), int(n * 0.1)
    scores = np.empty(values.shape[1])
    for column in range(values.shape[1]):
        series = values[:, column]
        train = series[:n_train]
        z = (series - train.mean()) / max(train.std(), 1e-9)
        test = z[n_train + n_val:]
        errors = [((test[t:t + pred_len] - test[t - 24:t]) ** 2).mean()
                  for t in range(seq_len, len(test) - pred_len, 96)]
        scores[column] = float(np.mean(errors)) if errors else np.nan
    print(f'\ndifficulty audit over {len(scores)} channels: '
          f'median {np.nanmedian(scores):.3f}  p90 {np.nanpercentile(scores, 90):.3f}')
    for count in (8, 32):
        path = os.path.join(ROOT, 'results', f'heterogeneity_nc{count}.json')
        if not os.path.exists(path):
            continue
        with open(path, encoding='utf-8') as handle:
            indices = json.load(handle)['summary']['picked_indices']
        picked = scores[indices]
        percentiles = (scores[None, :] < picked[:, None]).mean(1) * 100
        print(f'  nc={count:2d}: batch mean {picked.mean():.3f}'
              f'  median difficulty percentile {np.median(percentiles):.0f}%')


def main():
    table()
    if pairing_check():
        paired()
    coverage()
    difficulty_audit()


if __name__ == '__main__':
    main()
