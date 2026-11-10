"""Budgeted, drift-discounted expert evaluation and transfer decisions."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import exp, log, sqrt
from typing import Dict
from zlib import crc32

from .selector import DECISIONS, ESTIMATORS, estimate_gain, shrink, weighted_mean

# How the paired-evidence t test is applied to the score ranking.
#   ``ranked`` -- walk the ranking, deploy the first candidate that passes the
#                 test (a rejected leader no longer discards a valid runner-up).
#   ``leader`` -- test only the top-scoring candidate, else stay local (the
#                 behaviour before the gate became sequential).
EVIDENCE_GATES = ('leader', 'ranked')


def _exploration_order(expert_id: str) -> int:
    """Deterministic, id-independent tie-breaker for the exploration frontier.

    ``crc32`` of the capsule id: reproducible across runs, but not aligned with
    the alphabetical order that ``sorted(..., key=expert_id)`` produces.  The
    alphabet matters because capsule ids start with the publishing client, and a
    client's slot-0 expert of a peer called ``c127`` is not systematically
    preferable to ``c785``'s.
    """
    return crc32(str(expert_id).encode('utf-8'))


@dataclass(frozen=True)
class Observation:
    step: int
    gain: float
    cumulative_drift: float


class FedTDOController:
    """Two-stage online policy used by one client.

    Stage 1 selects a small shadow-evaluation set using an optimistic score.
    Stage 2 uses a lower-confidence score to decide whether a previously
    observed external expert is worth deploying.  The class consumes gains only
    after ``record_feedback``; it cannot access a current target while making a
    prediction decision.

    Scale discipline (plan §4.3): the confidence radius must be commensurate
    with the reward.  ``gain`` is a raw MSE difference, so its magnitude depends
    on the dataset and on RevIN's normalisation (typically 1e-2).  A radius of
    ``beta * sqrt(log(2 + t) / n)`` implicitly assumes gains live on the unit
    interval, which made every score negative and blocked transfers outright.
    The radius below is therefore scaled by a running estimate of the typical
    reward magnitude ``R`` together with the number of arms ``K`` and the
    horizon ``T``, as in standard UCB on bounded rewards.

    Decision rule (``decision``): ``lcb`` is the historical behaviour - deploy
    only if ``statistic - radius`` clears the threshold.  ``posterior`` compares
    the shrunk statistic itself against the threshold and keeps the radius for
    exploration only, which is what plan §5.2 actually specifies: the migration
    step compares expected gain against transfer cost, the 1-delta bound belongs
    to probe selection.  With ~4 observations per arm the LCB variant never
    clears zero; see ``federated/selector.py`` for the measurements.
    """

    def __init__(self, *, evaluation_budget=2, beta=1.0, eta_age=0.01,
                 eta_drift=0.1, lambda_eval=0.0, lambda_transfer=0.0,
                 lambda_switch=0.0, transfer_threshold=0.0,
                 min_observations=1, initial_probes=1,
                 unseen_uncertainty=1.0, probe_threshold=0.0, delta=0.1,
                 transfer_enabled=True, estimator='mean', decision='lcb',
                 trim_lower=0.1, trim_upper=0.9, upper_q=0.75, shrink_n0=0.0,
                 shrink_group='global', hierarchy='none', shrink_n0_role=5.0,
                 probe_floor=True, shock='raw', decorr_rho=0.0, baseline_k=0.6,
                 switch_margin=0.0, evidence_k=8, evidence_t=1.0,
                 evidence_gate='ranked'):
        if evaluation_budget < 0:
            raise ValueError('evaluation_budget must be non-negative')
        if estimator not in ESTIMATORS:
            raise ValueError(f'unknown estimator {estimator!r}; expected {ESTIMATORS}')
        if decision not in DECISIONS:
            raise ValueError(f'unknown decision {decision!r}; expected {DECISIONS}')
        if hierarchy not in ('none', 'host_role'):
            raise ValueError(f'unknown hierarchy {hierarchy!r}')
        if shock not in ('raw', 'centered'):
            raise ValueError(f'unknown shock {shock!r}')
        if not 0.0 <= decorr_rho < 1.0:
            raise ValueError('decorr_rho must be in [0, 1)')
        # ``shock='centered'`` stores the *paired* reward ``gain - round_mean``
        # instead of the raw gain (see ``record_round``).  Two things follow:
        #
        # 1. the level has to be reconstructed, ``statistic = baseline_hat + d``,
        #    where ``baseline_hat`` predicts this round's common shock from the
        #    history of round means.  Measured on Traffic that shock is strongly
        #    autocorrelated (lag 1 = 0.615, lag 5 = 0.630), so a blend of the
        #    running mean and the last value beats either alone; ``baseline_k``
        #    is that blend weight and is set to the measured lag-1 correlation.
        # 2. the residual dependence is much weaker: the within-arm
        #    autocorrelation falls from 0.635 to 0.365 once the round mean is
        #    removed.  ``decorr_rho`` is the design-effect coefficient that turns
        #    the nominal count into an effective one,
        #    ``n_eff = n / (1 + (n-1) rho)``.  It is 0 by default because the
        #    honest radius is a *separate* experiment, not a silent default.
        self.shock = str(shock)
        self.decorr_rho = float(decorr_rho)
        self.baseline_k = float(baseline_k)
        self.round_means: list[float] = []
        self.hierarchy = str(hierarchy)
        self.shrink_n0_role = float(shrink_n0_role)
        self.estimator = estimator
        self.decision = decision
        self.trim_lower = float(trim_lower)
        self.trim_upper = float(trim_upper)
        self.upper_q = float(upper_q)
        self.shrink_n0 = float(shrink_n0)
        if shrink_group not in ('global', 'role'):
            raise ValueError(f'unknown shrink_group {shrink_group!r}')
        self.shrink_group = shrink_group
        self.transfer_enabled = bool(transfer_enabled)
        self.evaluation_budget = evaluation_budget
        self.beta = beta
        self.eta_age = eta_age
        self.eta_drift = eta_drift
        self.lambda_eval = lambda_eval
        self.lambda_transfer = lambda_transfer
        self.lambda_switch = lambda_switch
        # Deadband: a *different* expert must beat the incumbent by this
        # much before the client pays for a switch.  Zero reproduces the
        # published behaviour.
        self.switch_margin = float(switch_margin)
        # Paired-evidence policy: a candidate is deployed only if the mean of
        # its last ``evidence_k`` paired gains is positive *and* its t
        # statistic clears ``evidence_t``.  Selecting the max of two noisy
        # means is a winner's curse; a one-sided test is the cheap fix.
        self.evidence_k = max(2, int(evidence_k))
        self.evidence_t = float(evidence_t)
        # Which candidate the t test is applied to.  ``ranked`` (default) walks
        # the score ranking and deploys the first candidate that clears the test;
        # ``leader`` reproduces the earlier behaviour, which rejected the leader
        # and fell straight back to local.
        if evidence_gate not in EVIDENCE_GATES:
            raise ValueError(f'unknown evidence_gate {evidence_gate!r}; '
                             f'expected one of {sorted(EVIDENCE_GATES)}')
        self.evidence_gate = str(evidence_gate)
        self.transfer_threshold = transfer_threshold
        self.min_observations = min_observations
        self.initial_probes = initial_probes
        self.unseen_uncertainty = unseen_uncertainty
        self.probe_threshold = probe_threshold
        # ``probe_floor`` keeps the exploration budget from collapsing.
        #
        # The threshold is compared against ``mean + radius``, a quantity in
        # reward units.  As the radius shrinks with evidence, that score drifts
        # down for *every* arm - including one that was simply never observed a
        # second time - so a fixed threshold of 0 silently replaces "evaluate
        # the best b candidates" with "evaluate nothing".  Measured on Traffic
        # (32 clients, 124 candidates): at ``evaluation_budget=9`` a client went
        # from 9 probes per round to 0 within a few rounds (43 of 1920 rounds
        # probed nothing); at ``evaluation_budget=20`` it probed 20, then 20,
        # then **0 for the remaining 40 rounds** and finished at +0.49% against
        # +1.82% for the smaller budget.  The budget must not be a ceiling on
        # evidence that the policy itself erodes.
        self.probe_floor = bool(probe_floor)
        self.delta = delta
        self.history: Dict[str, list[Observation]] = defaultdict(list)
        self.active_expert_id: str | None = None
        self.step = 0
        self.cumulative_drift = 0.0
        self.regime_priors: Dict[str, tuple[float, float]] = {}
        self.num_arms = 1
        self._scale_ema: float | None = None
        self._pool_cache: Dict[tuple, float] = {}
        self.scale_ema_decay = 0.05

    # ------------------------------------------------------------------ scale
    @property
    def reward_scale(self) -> float:
        """Typical reward magnitude, used as the UCB constant ``R``.

        An EMA of ``|gain|`` rather than the running min/max range: a single
        outlier horizon would otherwise inflate the range and re-inflate the
        radius, which is precisely the failure mode the scale fix removes.
        """
        if self._scale_ema is None:
            return 0.0
        return max(float(self._scale_ema), 1e-12)

    def _observe_scale(self, gain: float):
        magnitude = abs(float(gain))
        if self._scale_ema is None:
            self._scale_ema = magnitude
        else:
            decay = self.scale_ema_decay
            self._scale_ema = (1.0 - decay) * self._scale_ema + decay * magnitude

    def _log_term(self) -> float:
        arms = max(2, int(self.num_arms))
        horizon = max(2, int(self.step))
        return log(2.0 * arms * horizon / max(self.delta, 1e-12))

    # ---------------------------------------------------------------- control
    def begin_round(self, drift_score: float) -> int:
        self.step += 1
        self._pool_cache.clear()
        self.cumulative_drift += max(0.0, min(1.0, float(drift_score)))
        return self.step

    def set_regime_priors(self, priors: Dict[str, float]):
        """Install similarity-weighted priors.

        Each entry is either ``gain`` (weight 1.0) or ``(gain, similarity)``.
        A prior enters ``estimate`` as a pseudo-observation whose weight is the
        regime similarity, so a weak match barely moves the posterior instead of
        adding a hard bonus to the score (plan §6.3).
        """
        normalised: Dict[str, tuple[float, float]] = {}
        for expert_id, value in (priors or {}).items():
            if isinstance(value, (tuple, list)) and len(value) == 2:
                gain, weight = float(value[0]), float(value[1])
            else:
                gain, weight = float(value), 1.0
            weight = max(0.0, min(1.0, weight))
            if weight <= 0.0:
                continue
            normalised[str(expert_id)] = (gain, weight)
        self.regime_priors = normalised

    def baseline_hat(self) -> float:
        """Predicted common shock of the current round, on the reward scale.

        ``mu + k * (last - mu)``: the first term is the long-run level of "how
        good is a typical probed arm in a typical round", the second term lets
        the persistent part of the shock carry over.  With ``k = 0`` this is the
        running mean, i.e. the pooled level the raw statistic already implies.
        """
        if not self.round_means:
            return 0.0
        total = sum(self.round_means)
        count = len(self.round_means)
        mean = total / count
        return float(mean + self.baseline_k * (self.round_means[-1] - mean))

    def _weighted_history(self, expert_id: str):
        observations = self.history.get(expert_id, [])
        weights = [
            exp(-self.eta_age * (self.step - item.step)
                - self.eta_drift * max(0.0, self.cumulative_drift - item.cumulative_drift))
            for item in observations
        ]
        return observations, weights

    @staticmethod
    def role_of(expert_id: str) -> str:
        """Functional role of a capsule, i.e. the expert slot it occupies.

        Capsule ids are ``{client}:L{layer}:E{index}``.  The index is the expert
        slot inside the routed MoE layer, and it is the only part of the id that
        is comparable across clients: slot 1 of one client plays the same
        functional role as slot 1 of another.  Measured on Solar-Energy the slot
        dominates transferability - E1 averages a relative reward of +0.004 over
        515 evaluations while E0/E2/E3 average -0.26/-0.46/-0.29 - so pooling by
        slot is the right level at which to borrow strength (see
        ``BASELINE_COMPARISON.md`` §7).
        """
        text = str(expert_id)
        return text.rsplit(':', 1)[-1] if ':' in text else text

    def pooled_mean(self, group: str | None = None) -> float:
        """Reward mean over every arm, or over one role, for shrinkage.

        A *global* pool is the only quantity estimated well when the evaluation
        budget spreads ~120 rewards over 28 arms.  But arms are not exchangeable:
        pooling across roles mixes a transferable slot with three harmful ones and
        lands on a negative prior, which pushes every arm below the threshold.
        ``group`` selects a role and keeps the statistics meaningful.
        """
        return self.pooled_stats(group)[0]

    def pooled_stats(self, group: str | None = None):
        """``(weighted mean, effective n)`` over all arms, or over one role.

        The effective n is what a two-level hierarchical estimate needs: a role
        pool on Traffic holds ~56 observations while a single arm holds ~3, and
        the whole point of conditioning is to borrow that precision.  With no
        group the pool is the host itself, which is the top level of the
        hierarchy - the same client's overall transferability.
        """
        cached = self._pool_cache.get(('__stats__', group))
        if cached is not None:
            return cached
        values, weights = [], []
        for expert_id in self.history:
            if group is not None and self.role_of(expert_id) != group:
                continue
            observations, arm_weights = self._weighted_history(expert_id)
            values.extend(float(item.gain) for item in observations)
            weights.extend(arm_weights)
        if not values:
            result = (0.0, 0.0)
        else:
            total = sum(weights)
            effective = total * total / max(sum(w * w for w in weights), 1e-12)
            result = (weighted_mean(values, weights), effective)
        self._pool_cache[('__stats__', group)] = result
        return result

    def conditional_target(self, expert_id: str):
        """Nested shrinkage target ``host -> role -> arm``.

        Motivation, measured on 14,455 logged rewards of one 32-client arm:

        * a host dummy explains **17%** of the reward variance (31 dummies),
          a role dummy **3.9%** (3 dummies) and a source-client dummy **1.1%**
          (31 dummies) - see ``scripts/probe_conditional.py``.  The client's own
          overall transferability is the dominant conditioning variable, the
          expert slot is second, and *which* client published the capsule is
          almost irrelevant;
        * the arm mean is not usable on its own: selecting the top-1 arm by its
          own history and deploying it realises ``-0.032`` while the same
          statistic predicted ``+0.033`` - a 198% overestimate, i.e. pure
          winner's curse at ``n ~ 3``.

        So the role pool is shrunk toward the host pool, and the arm is then
        shrunk toward that.  ``source`` is deliberately absent: it carries 1.1%
        of the variance, and pooling by it measured *worse* than not pooling at
        all (+0.031 vs -0.021 reward per deployment in the same probe).
        """
        role = self.role_of(expert_id)
        host_mean, _ = self.pooled_stats(None)
        role_mean, role_n = self.pooled_stats(role)
        target = shrink(role_mean, role_n, host_mean, self.shrink_n0_role)
        return target, role_n

    def positive_rate(self, expert_id: str) -> float:
        """Laplace-smoothed share of positive rewards, for diagnostics only."""
        observations, weights = self._weighted_history(expert_id)
        if not observations:
            return 0.0
        positive = sum(w for item, w in zip(observations, weights) if item.gain > 0.0)
        return float((positive + 1.0) / (sum(weights) + 2.0))

    def estimate(self, expert_id: str, for_exploration: bool = False):
        """Gain statistic, radius, effective sample size and observation count.

        ``for_exploration`` skips the empirical-Bayes pull on purpose.  The
        shrinkage target is the pooled mean, which is *negative* on Solar-Energy
        (the average external expert loses), so pulling a probe score toward it
        suppresses exploration exactly where the controller knows least.  With
        ``shrink_n0=40`` the first version of this code stopped probing after ~2
        arms instead of the 8 it was allowed, which silently converted the
        budget arm into the b=2 arm (909 evaluations instead of 2983).
        """
        observations, weights = self._weighted_history(expert_id)
        prior = self.regime_priors.get(expert_id)
        if not observations and prior is None:
            return 0.0, self.unseen_uncertainty, 0.0, 0

        # ``centered`` mode: the stored observations are ``gain - round_mean``,
        # so the level has to be re-attached at the end.  A regime prior is on
        # the *raw* scale (regime memory stores what was actually revealed), so
        # it is shifted onto the centered scale first; otherwise a prior and a
        # sample would disagree by exactly the round mean.
        shift = self.baseline_hat() if self.shock == 'centered' else 0.0
        sample = list(observations)
        sample_weights = list(weights)
        if prior is not None:
            # Pseudo-observation: never a fabricated visit, but it does carry
            # information proportional to the regime similarity.
            prior_gain, prior_weight = prior
            sample.append(Observation(step=self.step, gain=float(prior_gain) - shift,
                                      cumulative_drift=self.cumulative_drift))
            sample_weights.append(prior_weight)

        statistic = estimate_gain(
            sample, sample_weights, estimator=self.estimator,
            trim_lower=self.trim_lower, trim_upper=self.trim_upper,
            upper_q=self.upper_q,
        )
        total = sum(sample_weights)
        sq_total = sum(w * w for w in sample_weights)
        effective_n = total * total / max(sq_total, 1e-12)
        if (self.shrink_n0 > 0.0 and observations and not for_exploration
                and self.estimator != 'quantile'):
            if self.hierarchy == 'host_role':
                target, role_n = self.conditional_target(expert_id)
                # The role pool is what supplies the extra precision, so the
                # posterior can never claim more observations than the role
                # level actually holds.
                borrowed = min(self.shrink_n0, role_n)
            else:
                target = (self.pooled_mean(self.role_of(expert_id))
                          if self.shrink_group == 'role' else self.pooled_mean())
                borrowed = 0.0
            statistic = shrink(statistic, effective_n, target, self.shrink_n0)
            # Shrinkage is a precision statement, so the confidence radius has to
            # see the borrowed count too.  Without it the hierarchical estimate
            # changes the ranking but never the decision, because ``lcb`` is
            # gated by the radius of the 3-sample arm mean.
            effective_n += borrowed
        if self.shock == 'centered' and observations:
            statistic = statistic + shift
        if self.decorr_rho > 0.0:
            # Compound-symmetry design effect: ``n`` repeated passes over the
            # same persistent shock carry ``n / (1 + (n-1) rho)`` independent
            # observations.  This deflation is what makes the radius honest; it
            # bites hardest on the arms with the most evidence.
            effective_n = effective_n / (1.0 + max(0.0, effective_n - 1.0) * self.decorr_rho)
        radius = self.beta * self.reward_scale * sqrt(
            self._log_term() / max(1.0, effective_n)
        )
        return statistic, radius, effective_n, len(observations)

    def select_probes(self, candidates, evaluation_costs: Dict[str, float]):
        """Pick this round's shadow evaluations.

        Two ordering details matter more than they look, and both were found by
        reading the logged probe sets of one Traffic client:

        * every unseen arm carries the same optimistic score (``0 +
          unseen_uncertainty``), so *how ties are broken decides what gets
          explored*.  Sorting them by id walks the candidate list in
          alphabetical order, so the exploration frontier advances as
          ``budget x rounds`` positions down the alphabet.  With 124 candidates,
          ``budget=9`` reaches the end of the list in 14 rounds and finds the
          capsule at ``c785``; ``budget=20`` spends its first round on the 20
          alphabetically-first capsules, which is *not* the same 20 arms per
          client and, once priors arrive, never gets to ``c785`` at all.  The
          order below is a fixed hash of the id instead, so the sweep is
          unbiased and reproducible.
        * the threshold on the exploration score (see ``probe_floor``) must not
          be able to switch exploration off.
        """
        if not self.transfer_enabled or self.evaluation_budget == 0:
            return []
        candidates = list(candidates)
        self.num_arms = max(self.num_arms, len(candidates))
        unseen = [item for item in candidates if not self.history.get(item.expert_id)]
        unseen.sort(key=lambda item: (evaluation_costs.get(item.expert_id, 0.0),
                                      _exploration_order(item.expert_id)))
        selected = unseen[:min(self.initial_probes, self.evaluation_budget)]
        selected_ids = {item.expert_id for item in selected}
        ranked = []
        for capsule in candidates:
            if capsule.expert_id in selected_ids:
                continue
            mean, radius, _, _ = self.estimate(capsule.expert_id, for_exploration=True)
            # Optimism is still the right guide for *exploration*: a probe is
            # worth its budget when the arm could be good, not when it is known
            # to be good.  Deliberately unshrunk - see ``estimate``.
            score = mean + radius - self.lambda_eval * evaluation_costs.get(capsule.expert_id, 0.0)
            ranked.append((score, _exploration_order(capsule.expert_id), capsule))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        for score, _, capsule in ranked:
            if len(selected) >= self.evaluation_budget:
                break
            if not self.probe_floor and score <= self.probe_threshold:
                # Legacy behaviour: a fixed threshold in reward units.  Kept for
                # reproducing runs made before the floor was introduced; see
                # ``__init__`` for why it is not the default.
                break
            selected.append(capsule)
        return selected

    def _evidence_passes(self, expert_id: str) -> bool:
        """One-sided t test on the last k paired gains of this expert."""
        observations, _ = self._weighted_history(expert_id)
        tail = [float(getattr(o, 'gain', o)) for o in list(observations)[-self.evidence_k:]]
        n = len(tail)
        if n < self.evidence_k:
            # Not enough evidence yet: stay local rather than act on noise.
            return False
        mean = sum(tail) / n
        var = sum((x - mean) ** 2 for x in tail) / (n - 1)
        if var <= 0.0:
            return mean > 0.0
        return (mean / (var / n) ** 0.5) >= self.evidence_t

    def choose_transfer(self, candidates, transfer_costs: Dict[str, float]):
        """Deploy the best external expert only if its lower bound beats keep-local.

        ``transfer_threshold`` is the value of staying local: when no candidate
        clears it, the client must fall back to its own model instead of
        silently keeping a stale deployment (plan §5.2 compares the migration
        against the currently deployed path, not against a bare threshold).
        """
        candidates = list(candidates)
        if not self.transfer_enabled:
            # Control arm: observe nothing, spend nothing, always keep local.
            changed = self.active_expert_id is not None
            self.active_expert_id = None
            return None, changed
        self.num_arms = max(self.num_arms, len(candidates))
        ranked = []
        for capsule in candidates:
            mean, radius, _, count = self.estimate(capsule.expert_id)
            if count < self.min_observations:
                continue
            is_switch = capsule.expert_id != self.active_expert_id
            cost = transfer_costs.get(capsule.expert_id, 0.0) if is_switch else 0.0
            if self.decision == 'evidence':
                score = mean - self.lambda_transfer * cost
            elif self.decision == 'lcb':
                score = mean - radius - self.lambda_transfer * cost
            else:
                # Expected gain versus transfer cost (plan §5.2).  The radius
                # stays out of the decision; keeping it here is what kept the
                # client local in 99% of rounds.
                score = mean - self.lambda_transfer * cost
            if is_switch:
                score -= (self.lambda_switch + self.switch_margin)
            ranked.append((score, capsule.expert_id, capsule))
        # Same ordering as the historical ``max`` over ``(score, expert_id)``.
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)

        if self.decision == 'evidence' and self.evidence_gate == 'ranked':
            # Gate *in rank order*: rejecting the leader used to fall straight
            # back to local, throwing away a runner-up that does clear the test.
            best = next((c for c in ranked if self._evidence_passes(c[1])), None)
        else:
            best = ranked[0] if ranked else None
            if best is not None and self.decision == 'evidence':
                if not self._evidence_passes(best[1]):
                    best = None
        if best is None or best[0] <= self.transfer_threshold:
            changed = self.active_expert_id is not None
            self.active_expert_id = None
            return None, changed
        changed = best[1] != self.active_expert_id
        self.active_expert_id = best[1]
        return self.active_expert_id, changed

    def record_round(self, gains: Dict[str, float]):
        """Record one round's shadow evaluations as a *batch*.

        The round mean cannot be computed one arm at a time, and that mean is
        the whole point of ``shock='centered'``: within one round every probed
        arm shares the same window, so their rewards share a shock of 0.748
        correlation.  Averaging it out turns highly redundant observations into
        weakly dependent ones, which is what the radius is allowed to assume.

        The batch form also keeps the raw path bit-identical (it simply loops
        ``record_feedback``), so the control arm is unaffected.
        """
        if not gains:
            return
        values = {str(key): float(value) for key, value in gains.items()}
        if self.shock != 'centered':
            for expert_id, gain in values.items():
                self.record_feedback(expert_id, gain)
            return
        round_mean = sum(values.values()) / len(values)
        self.round_means.append(round_mean)
        for expert_id, gain in values.items():
            residual = gain - round_mean
            self._observe_scale(residual)
            self.history[expert_id].append(Observation(
                step=self.step,
                gain=residual,
                cumulative_drift=self.cumulative_drift,
            ))

    def record_feedback(self, expert_id: str, gain: float):
        # Gains are stored raw.  Clipping to [-1, 1] used to imply a unit reward
        # range that RevIN-normalised MSE differences never have; the range is
        # now tracked from the data and applied to the radius instead.
        gain = float(gain)
        self._observe_scale(gain)
        self.history[expert_id].append(Observation(
            step=self.step,
            gain=gain,
            cumulative_drift=self.cumulative_drift,
        ))

    def snapshot(self):
        result = {}
        for expert_id in self.history:
            mean, radius, effective_n, count = self.estimate(expert_id)
            result[expert_id] = {
                'mean_gain': mean,
                'uncertainty': radius,
                'effective_n': effective_n,
                'observations': count,
                'positive_rate': self.positive_rate(expert_id),
            }
        result['__scale__'] = {
            'reward_scale': self.reward_scale,
            'num_arms': self.num_arms,
            'step': self.step,
            'cumulative_drift': self.cumulative_drift,
            'active_expert_id': self.active_expert_id,
            'transfer_enabled': self.transfer_enabled,
            'estimator': self.estimator,
            'decision': self.decision,
            'shrink_n0': self.shrink_n0,
            'shrink_group': self.shrink_group,
            'hierarchy': self.hierarchy,
            'shrink_n0_role': self.shrink_n0_role,
            'shock': self.shock,
            'decorr_rho': self.decorr_rho,
            'baseline_k': self.baseline_k,
            'baseline_hat': self.baseline_hat(),
            'round_means': len(self.round_means),
            'pooled_mean': self.pooled_mean(),
            'host_mean': self.pooled_stats(None)[0],
        }
        return result
