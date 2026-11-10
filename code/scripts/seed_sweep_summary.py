"""Aggregate the 32-client Traffic seed sweep into a distribution, not a number.

Why this script exists: a single seed gave transfer +1.82%, but 78% of that came
from one client.  Reporting the best of several seeds would be cherry-picking,
so this prints every seed, the per-seed effect, and the spread, and explicitly
flags which seed is the most favourable *so that it can be labelled best-case and
nothing more*.

Two robustness choices, both forced by the data:

* **Relative, per client, then aggregate.**  Seed 2022 draws channel c840, whose
  test window sits ~39 sigma above its train mean; its MSE is 123 and it alone
  lifts the arithmetic mean of 32 clients from a median of 0.295 to 4.158.  Any
  aggregate computed on absolute MSE differences is therefore a statement about
  one sensor.  Effects are computed per client in relative terms and summarised
  by mean *and* median.
* **Median-of-clients alongside the mean.**  Both are printed; the median is the
  "typical client", the mean is what a deployment that cares about total error
  would see.  They can disagree by an order of magnitude on this dataset.

Statistical honesty notes:

* ``--seed`` moves both the channel subset and the initialisation, so each seed
  is an independent federation.  The per-seed effects are the sample.
* With n=5, even 5/5 positive gives a two-sided sign-test p of 0.0625.  Printed
  rather than hidden, so a nice mean cannot masquerade as significance.
* Within a seed the transfer effect is paired (bit-exact ``default_loss``,
  verified here), and the noise of interest is *between* seeds.

Usage::

    python scripts/seed_sweep_summary.py
    python scripts/seed_sweep_summary.py --seeds 2021 2022 --difficulty
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from math import comb

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FED = os.path.join(ROOT, 'results', 'fedtdo')
BASE = os.path.join(ROOT, 'results_baselines')
LEGACY = {'joint_b9': 'traffic32_joint_b9', 'notransfer': 'traffic32_notransfer',
          'joint_b9_fix': 'traffic32_s2021_joint_b9_fix'}
# ``--arm joint_b9_fix`` reads the probe-bug-fixed runs; the legacy seed-2021
# directories keep their own names.
ARM = 'joint_b9'


def read(path):
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def fed_path(seed, arm=None):
    arm = arm or ARM
    path = os.path.join(FED, f'traffic32_s{seed}_{arm}', 'metrics.json')
    if os.path.exists(path):
        return path
    # Seed 2021 was run before the sweep and its directories carry no seed tag.
    if seed == 2021 and arm in LEGACY:
        return os.path.join(FED, LEGACY[arm], 'metrics.json')
    return path


def local_path(seed):
    patterns = [f'BL_local_traffic32_s{seed}_local_*']
    if seed == 2021:
        patterns.append('BL_local_traffic32_local_*')
    for pattern in patterns:
        hits = glob.glob(os.path.join(BASE, pattern, 'summary.json'))
        if hits:
            return hits[0]
    return None


def rounds_path(seed):
    return os.path.join(os.path.dirname(fed_path(seed)), 'rounds.json')


def pairing(seed):
    """True when both arms trained the bit-identical default model."""
    try:
        treated = read(rounds_path(seed))
        control = read(rounds_path(seed).replace(ARM, 'notransfer'))
    except FileNotFoundError:
        return None
    if not treated or 'default_loss' not in treated[0]:
        return None
    a = {(r['step'], r['client']): r['default_loss'] for r in treated}
    b = {(r['step'], r['client']): r['default_loss'] for r in control}
    shared = set(a) & set(b)
    return sum(1 for key in shared if a[key] == b[key]) == len(shared) if shared else None


def diagnostics(seed):
    """Oracle headroom and deployment coverage from the round log.

    Oracle mirrors ``scripts/replay_selector.py``: per round take the best
    non-negative logged raw gain, so it bounds what the experts that were
    actually *evaluated* could have delivered.
    """
    path = rounds_path(seed)
    if not os.path.exists(path):
        return None, None
    log = read(path)
    if not log or 'raw_gains' not in log[0]:
        return None, None
    total_default = sum(r['default_loss'] for r in log)
    total_best = sum(r['default_loss'] - max([0.0] + list(r['raw_gains'].values()))
                     for r in log)
    oracle = 100 * (total_default - total_best) / total_default if total_default else None
    coverage = 100 * sum(1 for r in log if r['active_expert']) / len(log)
    return oracle, coverage


def seed_row(seed):
    treated_path, control_path = fed_path(seed), fed_path(seed, 'notransfer')
    if not (os.path.exists(treated_path) and os.path.exists(control_path)):
        return None
    treat, ctrl = read(treated_path), read(control_path)
    keys = [k for k in treat['client_metrics']
            if k in ctrl['client_metrics'] and ctrl['client_metrics'][k]['mse'] > 0]
    m_t = np.array([treat['client_metrics'][k]['mse'] for k in keys])
    m_c = np.array([ctrl['client_metrics'][k]['mse'] for k in keys])
    rel = (m_c - m_t) / m_c  # per-client relative improvement of transfer
    # Headline stays in the same currency as every other table in this repo
    # (ratio of mean client MSE).  ``*_med`` repeats it on medians, which is the
    # number that survives a single hard client like c840.
    row = {
        'seed': seed,
        'n': len(keys),
        'mean_pct': 100 * (m_c.mean() - m_t.mean()) / m_c.mean(),
        'median_pct': 100 * (np.median(m_c) - np.median(m_t)) / np.median(m_c),
        'client_pct': 100 * rel.mean(),
        'improved': int((rel > 1e-9).sum()),
        'worsened': int((rel < -1e-9).sum()),
        'mse_t': float(m_t.mean()), 'mse_t_med': float(np.median(m_t)),
        'mse_c': float(m_c.mean()), 'mse_c_med': float(np.median(m_c)),
        'transfers': treat['actions']['transfers'],
        'active': treat['actions']['clients_with_active_expert'],
        'local': None, 'local_mean_pct': None, 'local_median_pct': None,
        'local_med': None, 'top_share': None, 'paired': pairing(seed),
        'oracle': None, 'coverage': None,
    }
    if rel.mean() != 0:
        row['top_share'] = 100 * rel.max() / len(keys) / rel.mean()
    # Always-interpretable concentration measure: how many percentage points of
    # the per-seed effect the single best client contributes.
    row['top_pp'] = 100 * rel.max() / len(keys)

    path = local_path(seed)
    if path:
        local = read(path)
        row['local'] = local['mean_mse']
        lk = [k for k in keys if k in local['clients']]
        if lk:
            m_l = np.array([local['clients'][k]['mse'] for k in lk])
            t_l = np.array([treat['client_metrics'][k]['mse'] for k in lk])
            row['local_mean_pct'] = 100 * (t_l.mean() - m_l.mean()) / m_l.mean()
            row['local_median_pct'] = 100 * (np.median(t_l) - np.median(m_l)) / np.median(m_l)
            row['local_med'] = float(np.median(m_l))
    row['oracle'], row['coverage'] = diagnostics(seed)
    return row


def sign_test_p(positive, total):
    """Two-sided exact sign test p-value."""
    if total == 0:
        return 1.0
    tail = sum(comb(total, k) for k in range(max(positive, total - positive), total + 1))
    return min(1.0, 2 * tail / 2 ** total)


def channel_difficulty(cache='results/difficulty_traffic.json'):
    """Seasonal-naive MSE per channel, cached (~20 s the first time)."""
    path = os.path.join(ROOT, cache)
    if os.path.exists(path):
        return read(path)['scores']
    import pandas as pd
    frame = pd.read_csv(os.path.join(ROOT, 'dataset/traffic/traffic.csv'))
    values = frame.iloc[:, 1:].values.astype(float)
    n = len(values)
    n_train, n_val = int(n * 0.7), int(n * 0.1)
    scores = {}
    for column in range(values.shape[1]):
        series = values[:, column]
        train = series[:n_train]
        z = (series - train.mean()) / max(train.std(), 1e-9)
        test = z[n_train + n_val:]
        errors = [((test[t:t + 24] - test[t - 24:t]) ** 2).mean()
                  for t in range(96, len(test) - 24, 96)]
        scores[str(column)] = float(np.mean(errors)) if errors else float('nan')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump({'scores': scores}, handle)
    return scores


def seed_difficulty(seed, scores):
    values = np.array([scores[str(i)] for i in range(len(scores))])
    picked = np.random.default_rng(seed).choice(len(values), size=32, replace=False)
    batch = values[picked]
    return float(batch.mean()), float(np.median((values[None, :] < batch[:, None]).mean(1) * 100))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='+', type=int,
                        default=[2021, 2022, 2023, 2024, 2025])
    parser.add_argument('--arm', default='joint_b9',
                        help='treated arm suffix, e.g. joint_b9_fix')
    parser.add_argument('--difficulty', action='store_true',
                        help='also report how hard each seed batch is')
    args = parser.parse_args()
    global ARM
    ARM = args.arm

    rows = [row for row in (seed_row(seed) for seed in args.seeds) if row]
    if not rows:
        raise SystemExit('no completed seeds found')

    if args.difficulty:
        scores = channel_difficulty()
        for row in rows:
            row['batch'], row['pct'] = seed_difficulty(row['seed'], scores)

    head = (f"{'seed':<6}{'xfer mean':>10}{'xfer med':>9}{'win/lose':>10}{'xfer':>6}"
            f"{'active':>7}{'oracle':>9}{'cover':>8}{'MSE med':>9}{'local med':>10}"
            f"{'vs loc m':>9}{'vs loc md':>10}{'top pp':>8}{'capture':>9}")
    if args.difficulty:
        head += f"{'batch naive':>12}{'pct':>6}"
    print(head)
    for row in rows:
        fmt = lambda value, pattern: pattern % value if value is not None else '-'
        line = (f"{row['seed']:<6}{row['mean_pct']:+9.2f}%{row['median_pct']:+8.2f}%"
                f"{row['improved']:>5}/{row['worsened']:<4}{row['transfers']:>6}"
                f"{row['active']:>7}{fmt(row['oracle'], '%+.2f%%'):>9}"
                f"{fmt(row['coverage'], '%.1f%%'):>8}"
                f"{row['mse_t_med']:9.3f}{fmt(row['local_med'], '%.3f'):>10}"
                f"{fmt(row['local_mean_pct'], '%+.2f%%'):>9}"
                f"{fmt(row['local_median_pct'], '%+.2f%%'):>10}"
                f"{row['top_pp']:+7.2f}pp"
                f"{(100*row['mean_pct']/row['oracle'] if row['oracle'] else float('nan')):8.0f}%")
        if args.difficulty:
            line += f"{row['batch']:12.3f}{row['pct']:5.0f}%"
        if row['paired'] is False:
            line += '  !! default paths differ'
        print(line)

    mean_eff = np.array([r['mean_pct'] for r in rows])
    med_eff = np.array([r['median_pct'] for r in rows])
    pos = int((mean_eff > 0).sum())
    print(f'\ntransfer effect, {len(rows)} seeds, ratio of mean client MSE (the currency of'
          ' every other table here):')
    print(f"  mean  {mean_eff.mean():+.2f}%  median-of-seeds {np.median(mean_eff):+.2f}%"
          f"  range [{mean_eff.min():+.2f}%, {mean_eff.max():+.2f}%]"
          f"  sd {mean_eff.std(ddof=1) if len(rows) > 1 else 0:.2f}  positive {pos}/{len(rows)}"
          f"  sign-test p={sign_test_p(pos, len(rows)):.4f}")
    print(f"  median-of-clients version: mean {med_eff.mean():+.2f}%"
          f"  range [{med_eff.min():+.2f}%, {med_eff.max():+.2f}%]")
    best = rows[int(np.argmax(mean_eff))]
    worst = rows[int(np.argmin(mean_eff))]
    print(f'  most favourable seed {best["seed"]} ({best["mean_pct"]:+.2f}%)'
          f' · least favourable {worst["seed"]} ({worst["mean_pct"]:+.2f}%)')
    print('  -> quote the median with the full range; the best seed alone overstates'
          f' the effect by {(best["mean_pct"] - np.median(mean_eff)):.2f} pp.')

    local = [r for r in rows if r['local_mean_pct'] is not None]
    if local:
        d = np.array([r['local_mean_pct'] for r in local])
        dm = np.array([r['local_median_pct'] for r in local])
        print(f'\nmethod vs pure local, {len(d)} seeds (negative = method better):'
              f' mean {d.mean():+.2f}%  range [{d.min():+.2f}%, {d.max():+.2f}%]'
              f'  better on {int((d < 0).sum())}/{len(d)}')
        print(f'  median-of-clients version: mean {dm.mean():+.2f}%'
              f'  range [{dm.min():+.2f}%, {dm.max():+.2f}%]'
              f'  better on {int((dm < 0).sum())}/{len(dm)}')


if __name__ == '__main__':
    main()
