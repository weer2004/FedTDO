"""Gain statistics for the FedTDO transfer decision.

Why this module exists
----------------------
The first implementation scored a candidate with

    ``mean_gain - beta * R * sqrt(log(2KT/delta) / n_eff)``

and deployed only when that lower confidence bound cleared zero.  Measured on
Solar-Energy the per-round relative gain is heavy tailed: 45% of rounds are
positive with a mean near +3.5%, the remaining 55% average about -1.3%.  The
90% bound therefore never clears zero and the client stays local in 474 of 480
decisions.  That is not a ranking failure, it is an over-conservative decision
rule: plan §5.2 asks the deployment stage to compare the *expected* gain with
the transfer cost, and leaves the 1-delta bound to exploration.

Two further facts shape the estimators below:

* ``evaluation_budget=2`` over 60 rounds yields ~120 rewards per client spread
  over 28 candidate experts, i.e. ``n ~ 4`` per arm.  A 4-sample *mean* is not
  an estimate of anything; a 4-sample *vote* (median / share of positive
  rounds) is.  That is why the alternatives to ``mean`` are quantile based.
* one lucky round can move a 4-sample mean by 25 absolute percentage points.
  The winsorised estimator bounds that influence without discarding data.

Everything here is pure: no controller state, no torch, so
``scripts/replay_selector.py`` can rank the estimators offline against logged
rewards instead of paying for a full retrain per variant.

Quantile conventions
--------------------
``weighted_quantile`` places each sample at the midpoint of its cumulative
weight and interpolates linearly between neighbours.  At ``n = 4`` the weights
sit at 0.125/0.375/0.625/0.875, so ``q = 0.5`` on ``[0.1, 0.1, 0.1, 0.9]``
returns 0.1 - the median ignores the single outlier, which is the whole point.
The step-function quantile used by the first draft returned 0.9 for the same
input because a target of ``q * total`` lands on the last weight bucket.
"""

from __future__ import annotations

from typing import Iterable, Sequence

ESTIMATORS = ('mean', 'winsor', 'cvar', 'quantile')
DECISIONS = ('lcb', 'posterior', 'evidence')


def _sorted_pairs(values: Sequence[float], weights: Sequence[float]):
    pairs = sorted(zip(values, weights), key=lambda item: item[0])
    total = sum(weight for _, weight in pairs)
    if total <= 0.0:
        pairs = [(value, 1.0) for value, _ in pairs]
        total = float(len(pairs))
    return pairs, total


def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    total = sum(weights)
    if total <= 0.0:
        return 0.0
    return float(sum(v * w for v, w in zip(values, weights)) / total)


def weighted_quantile(values: Sequence[float], weights: Sequence[float], q: float) -> float:
    """Linearly interpolated quantile of a weighted sample."""
    if len(values) == 0:
        return 0.0
    q = min(1.0, max(0.0, float(q)))
    pairs, total = _sorted_pairs(values, weights)
    if len(pairs) == 1:
        return float(pairs[0][0])
    positions, cumulative = [], 0.0
    for _, weight in pairs:
        positions.append((cumulative + weight / 2.0) / total)
        cumulative += weight
    if q <= positions[0]:
        return float(pairs[0][0])
    if q >= positions[-1]:
        return float(pairs[-1][0])
    for index in range(len(positions) - 1):
        left, right = positions[index], positions[index + 1]
        if left <= q <= right:
            span = right - left
            fraction = 0.0 if span <= 0.0 else (q - left) / span
            low = float(pairs[index][0])
            high = float(pairs[index + 1][0])
            return low + fraction * (high - low)
    return float(pairs[-1][0])


def winsorized_mean(values: Sequence[float], weights: Sequence[float],
                    lower: float = 0.1, upper: float = 0.9) -> float:
    """Winsorised weighted mean with count-based clipping.

    Exactly ``max(1, round(lower * n))`` samples are clipped up to the lower
    order statistic and ``max(1, round((1 - upper) * n))`` are clipped down to
    the upper one.  Count-based bounds are used instead of quantile bounds
    because at ``n = 4`` a weight-based quantile sits *on* the extreme sample
    and clips nothing (this was the first draft's bug).  Clipping rather than
    dropping keeps the effective sample size - and therefore the confidence
    radius - honest.
    """
    n = len(values)
    if n < 3:
        return weighted_mean(values, weights)
    pairs, _ = _sorted_pairs(values, weights)
    low_rank = min(n - 1, max(1, int(round(lower * n))))
    high_rank = min(n - 1, max(1, int(round((1.0 - upper) * n))))
    floor_value = float(pairs[low_rank][0])
    ceil_value = float(pairs[n - 1 - high_rank][0])
    if ceil_value < floor_value:
        floor_value, ceil_value = ceil_value, floor_value
    clipped = {}
    for value, weight in zip(values, weights):
        clipped_value = min(ceil_value, max(floor_value, float(value)))
        clipped[clipped_value] = clipped.get(clipped_value, 0.0) + weight
    return weighted_mean(list(clipped.keys()), list(clipped.values()))


def cvar_upper(values: Sequence[float], weights: Sequence[float],
               upper_q: float = 0.75) -> float:
    """Mean of the ``max(1, round((1 - upper_q) * n))`` best observations.

    Read as "the gain we get when this expert actually works".  A top-k mean
    rather than a threshold-based tail mean: with heavy ties a threshold can
    select the entire sample and silently collapse to the plain mean.
    """
    n = len(values)
    if n == 0:
        return 0.0
    k = min(n, max(1, int(round((1.0 - upper_q) * n))))
    order = sorted(range(n), key=lambda index: values[index], reverse=True)[:k]
    return weighted_mean([values[i] for i in order], [weights[i] for i in order])


def shrink(statistic: float, effective_n: float, prior: float, strength: float) -> float:
    """Empirical-Bayes pull of one arm's statistic toward the pooled mean.

    ``strength`` is a pseudo-count: ``n0 = 0`` disables the correction, large
    ``n0`` makes every arm fall back to the pooled mean, which is the correct
    degenerate policy when per-arm samples are too small to rank arms.
    """
    if strength <= 0.0:
        return float(statistic)
    n = max(0.0, float(effective_n))
    return float((n * float(statistic) + strength * float(prior)) / (n + strength))


def estimate_gain(observations: Iterable, weights: Sequence[float], *,
                  estimator: str = 'mean', trim_lower: float = 0.1,
                  trim_upper: float = 0.9, upper_q: float = 0.75) -> float:
    """Apply one estimator to ``(gain, weight)`` observations."""
    if estimator not in ESTIMATORS:
        raise ValueError(f'unknown estimator {estimator!r}; expected one of {ESTIMATORS}')
    values = [float(item.gain) for item in observations]
    if not values:
        return 0.0
    if estimator == 'mean':
        return weighted_mean(values, weights)
    if estimator == 'winsor':
        return winsorized_mean(values, weights, trim_lower, trim_upper)
    if estimator == 'cvar':
        return cvar_upper(values, weights, upper_q)
    return weighted_quantile(values, weights, upper_q)
