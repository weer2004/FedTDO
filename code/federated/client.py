"""Client protocol combining local forecasting with FedTDO decisions."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict

import torch

from .cost_model import CostModel
from .expert_capsule import ExpertCapsule
from .prequential import PrequentialBuffer
from .regime_memory import RegimeMemory


@dataclass
class ForecastRecord:
    record_id: int
    due_step: int
    prediction: torch.Tensor
    active_expert_id: str | None
    probed_expert_ids: tuple[str, ...]
    # Regime priors installed for *this* decision.  Logged so the offline
    # replay reproduces the controller's inputs exactly; the priors change the
    # gain statistic (they enter as pseudo-observations), so a replay that
    # ignored them would silently score a different controller.
    priors: dict = field(default_factory=dict)


class FedTDOClient:
    """One local model plus its private transfer controller and feedback queue."""

    def __init__(self, *, client_id, model, registry, controller, cost_model=None,
                 transfer_layer=0, encoder_version='v1', mix_alpha=0.5,
                 feedback_delay=0, regime_memory=None, output_start=0,
                 alpha_calibration=True, alpha_min_observations=3,
                 cache_limit=None, gain_mode='relative', clip_reward=False,
                 reward_basis='mixed', deploy_topk=1,
                 include_self_experts=False):
        self.client_id = str(client_id)
        self.model = model
        self.registry = registry
        self.controller = controller
        self.cost_model = cost_model or CostModel()
        self.transfer_layer = transfer_layer
        self.encoder_version = encoder_version
        self.mix_alpha = mix_alpha
        # Number of experts averaged into the deployed forecast (1 =
        # published single-expert behaviour).
        self.deploy_topk = max(1, int(deploy_topk))
        self.include_self_experts = bool(include_self_experts)
        self.feedback_delay = feedback_delay
        self.output_start = output_start
        self.regime_memory = regime_memory or RegimeMemory()
        self.cache: Dict[str, ExpertCapsule] = {}
        self.feedback = PrequentialBuffer()
        self.prediction_step = 0
        self.cumulative_default_loss = 0.0
        self.cumulative_deployed_loss = 0.0
        self.alpha_calibration = bool(alpha_calibration)
        self.alpha_min_observations = int(alpha_min_observations)
        self.gain_mode = str(gain_mode)
        self.clip_reward = bool(clip_reward)
        if reward_basis not in ('mixed', 'expert'):
            raise ValueError(f'unknown reward_basis {reward_basis!r}')
        # ``mixed`` scores the arm on the blend that would be deployed (the old
        # behaviour); ``expert`` scores it on the expert's own forecast.  See
        # ``PrequentialBuffer.resolve`` for why the two differ by an order of
        # magnitude once the mixing weight has been learned.
        self.reward_basis = str(reward_basis)
        # expert_id -> [sum(d*d), sum(d*r), n] for the least-squares mix weight.
        self.alpha_stats: Dict[str, list] = {}
        self.cache_limit = cache_limit
        self._last_signature = None
        # ---- instrumentation hooks (read-only, never changes a decision) ----
        # ``last_probes`` is refilled on every ``predict`` with the score the
        # controller had for each probed arm, so the paper recorder can write
        # predicted_gain next to the realised one.  ``download_log`` carries the
        # bytes of every capsule actually fetched during the current call.
        self.last_probes: list = []
        self.download_log: list = []
        self.last_decision = {'active_id': None, 'previous_id': None,
                              'switched': False, 'estimated_gain': None}

    # ---------------------------------------------------------------- helpers
    def _model_signature(self):
        layer = self.model.temporal_moe_layers[self.transfer_layer]
        return {
            'layer_idx': self.transfer_layer,
            'd_model': layer.d_model,
            'num_patches': layer.num_patches,
            'encoder_version': self.encoder_version,
        }

    def publish_experts(self, regime_signature=None):
        """Publish compatible cores from one local MoE layer to the server."""
        layer = self.model.temporal_moe_layers[self.transfer_layer]
        published = []
        for expert_index in layer.expert_pool.get_enabled_expert_indices():
            capsule = ExpertCapsule.from_local_expert(
                layer.expert_pool.experts[expert_index],
                expert_id=f'{self.client_id}:L{self.transfer_layer}:E{expert_index}',
                source_id=self.client_id,
                layer_idx=self.transfer_layer,
                d_model=layer.d_model,
                num_patches=layer.num_patches,
                encoder_version=self.encoder_version,
                regime_signature=regime_signature,
            )
            published.append(self.registry.register(capsule))
        return published

    def _candidates(self):
        return self.registry.candidates_for(
            requester_id=self.client_id,
            include_self=bool(getattr(self, 'include_self_experts', False)),
            **self._model_signature()
        )

    def _ensure_cached(self, capsule):
        if capsule.expert_id not in self.cache:
            self.cache[capsule.expert_id] = capsule
            self.cost_model.record_download(capsule)
            # Recorded for the communication ledger; the bytes are the capsule's
            # own payload, and only a first fetch is charged.
            self.download_log.append(
                (capsule.expert_id, capsule.source_id, int(capsule.estimated_bytes))
            )
            self._evict_cache()
        return self.cache[capsule.expert_id]

    def _evict_cache(self):
        """Keep the local capsule cache bounded (plan §5.3: ``C_cache <= M_i``)."""
        if not self.cache_limit:
            return
        while len(self.cache) > int(self.cache_limit):
            victim = next(
                (expert_id for expert_id in self.cache
                 if expert_id != self.controller.active_expert_id),
                None,
            )
            if victim is None:
                break
            del self.cache[victim]

    @staticmethod
    def _target_channels(prediction, target):
        # The standard benchmark's MS setting predicts only the final channel.
        return target[..., -prediction.shape[-1]:]

    def alpha_for(self, expert_id: str) -> float:
        """Local mix weight for one external expert.

        A fixed ``mix_alpha`` injects another client's expert at a constant
        weight, which is a net loss whenever that expert is locally mis-scaled.
        The weight is instead the least-squares projection of the residual
        ``y - local`` onto ``external - local`` over already revealed steps,
        i.e. the cheapest form of the local calibration allowed by plan §5.2.
        """
        if not self.alpha_calibration:
            return float(self.mix_alpha)
        stats = self.alpha_stats.get(expert_id)
        if not stats or stats[2] < self.alpha_min_observations:
            return float(self.mix_alpha)
        sum_xx, sum_xy, _ = stats
        if sum_xx <= 1e-12:
            return float(self.mix_alpha)
        return float(min(1.0, max(0.0, sum_xy / sum_xx)))

    # --------------------------------------------------------------- decision
    @torch.no_grad()
    def predict(self, x_enc, x_mark_enc=None):
        """Make one causal prediction and queue all observed counterfactuals."""
        self.model.eval()
        self.prediction_step += 1
        # Reset the instrumentation hooks: the caller reads them right after this
        # call and would otherwise see the previous window's downloads.
        self.last_probes = []
        self.download_log = []
        previous_id = self.controller.active_expert_id

        prior_drift = self.model.online_drift_signal()['score']
        self.controller.begin_round(prior_drift)
        # The signature of the *previous* hidden state is already available, so
        # the prior never requires a second encoder pass over the current input.
        if self._last_signature is not None:
            priors = self.regime_memory.retrieve(self._last_signature)
        else:
            priors = {}
        self.controller.set_regime_priors(priors)

        candidates = self._candidates()
        by_id = {capsule.expert_id: capsule for capsule in candidates}

        transfer_costs = {
            capsule.expert_id: self.cost_model.transfer_cost(
                capsule, capsule.expert_id in self.cache
            )
            for capsule in candidates
        }
        active_id, switched = self.controller.choose_transfer(candidates, transfer_costs)
        active_capsule = self.cache.get(active_id) if active_id else None
        if active_id and active_capsule is None and active_id in by_id:
            active_capsule = self._ensure_cached(by_id[active_id])
        if switched and active_capsule is not None:
            self.cost_model.record_transfer(switched=True)
        self.last_decision = {
            'active_id': active_id,
            'previous_id': previous_id,
            'switched': bool(switched),
            'estimated_gain': None,
        }

        # The default path observes the current input's drift after the decision;
        # its signal is only used in the following decision round.
        default_prediction = self.model.forecast(
            x_enc, x_mark_enc, observe_online_drift=True
        )[:, :, self.output_start:]

        signature = getattr(self.model, 'last_regime_signature', None)
        if signature is None:
            signature = self.regime_memory.signature_from_input(x_enc)
        self._last_signature = signature

        evaluation_costs = {
            capsule.expert_id: self.cost_model.evaluation_cost(
                capsule, capsule.expert_id in self.cache
            )
            for capsule in candidates
        }
        probes = self.controller.select_probes(candidates, evaluation_costs)
        # Recorder hook: the score the selector already computed for each probed
        # arm, so ``predicted_gain`` can be written next to the realised one.
        # ``estimate`` is a pure lookup over the controller's history; no forward
        # pass, no communication, no state change.
        self.last_probes = []
        for probe in probes:
            probe_mean, probe_radius, _, probe_count = self.controller.estimate(
                probe.expert_id)
            self.last_probes.append({
                'expert_id': probe.expert_id,
                'source_id': probe.source_id,
                'version': probe.encoder_version,
                'predicted_gain': float(probe_mean),
                'uncertainty': float(probe_radius),
                'count': int(probe_count),
            })
        evaluated = {capsule.expert_id: capsule for capsule in probes}
        if active_id and active_id in by_id:
            evaluated[active_id] = by_id[active_id]
        elif active_id and active_id in self.cache:
            # A server eviction must not invalidate an already deployed local
            # cache entry; it remains evaluable until the client switches away.
            evaluated[active_id] = self.cache[active_id]

        counterfactuals = {}
        external_only = {}
        deployed_prediction = default_prediction
        # Top-k soft mixture.  With k=1 this is the published single-expert
        # blend; k=2 averages the calibrated deltas of the two most trusted
        # experts, which damps the +-12% decision noise without adding any
        # communication (both experts are already in the evaluated set).
        _topk = max(1, int(getattr(self, 'deploy_topk', 1)))
        _DELTA = {}
        shadow_started = time.perf_counter()
        for expert_id, capsule in evaluated.items():
            cached = self._ensure_cached(capsule)
            external = self.model.forecast_with_external_expert(
                x_enc, x_mark_enc, cached, mix_alpha=1.0
            )[:, :, self.output_start:]
            alpha = self.alpha_for(expert_id)
            # Calibration lives on the forecast level: a convex blend of the
            # local and the pure-external path, which is exactly the quantity
            # the revealed label lets us fit.
            mixed = default_prediction + alpha * (external - default_prediction)
            external_only[expert_id] = external
            counterfactuals[expert_id] = mixed
            _DELTA[expert_id] = alpha * (external - default_prediction)
            if expert_id == active_id:
                deployed_prediction = mixed
            if expert_id in {item.expert_id for item in probes}:
                self.cost_model.record_evaluation(capsule)

        if _topk > 1 and active_id is not None and active_id in _DELTA:
            others = sorted(
                (eid for eid in _DELTA if eid != active_id),
                key=lambda eid: abs(self.alpha_for(eid)), reverse=True,
            )[:_topk - 1]
            acc = _DELTA[active_id]
            for eid in others:
                acc = acc + _DELTA[eid]
            deployed_prediction = default_prediction + acc / (1 + len(others))

        # Wall time spent on the candidate forecasts of this round, reported
        # separately from the deployed forecast in ``cost_events.csv``.
        self.last_shadow_ms = (time.perf_counter() - shadow_started) * 1000.0
        pending = self.feedback.enqueue(
            due_step=self.prediction_step + self.feedback_delay,
            default_prediction=default_prediction,
            candidate_predictions=counterfactuals,
            signature=signature,
            deployed_expert_id=active_id,
            external_predictions=external_only,
        )
        return ForecastRecord(
            record_id=pending.record_id,
            due_step=pending.due_step,
            prediction=deployed_prediction,
            active_expert_id=active_id,
            probed_expert_ids=tuple(item.expert_id for item in probes),
            priors={str(k): (float(v[0]), float(v[1])) if isinstance(v, (tuple, list))
                    else (float(v), 1.0) for k, v in priors.items()},
        )

    def _update_alpha(self, record, target):
        if not self.alpha_calibration:
            return
        for expert_id, external in record.external_predictions.items():
            target_aligned = self._target_channels(external, target)
            if target_aligned.shape != external.shape:
                continue
            delta = external - record.default_prediction
            residual = target_aligned - record.default_prediction
            sum_xx = float(torch.sum(delta * delta))
            sum_xy = float(torch.sum(delta * residual))
            stats = self.alpha_stats.setdefault(expert_id, [0.0, 0.0, 0])
            stats[0] += sum_xx
            stats[1] += sum_xy
            stats[2] += 1

    def _reward(self, gain, default_loss):
        """Convert a raw MSE difference into the controller's reward.

        The online objective is the sum of per-round MSE, so the *raw* MSE
        difference is the currency the decision should maximise: a +0.05 gain at
        noon and a +0.001 gain at night add up as written.

        ``relative`` divides by the local loss.  That was introduced to scale the
        confidence radius, but it reweights rounds by ``1 / l0`` and Solar-Energy
        spans two orders of magnitude between noon (~1.5) and night (~0.02).
        Measured on the logged rewards the two conventions disagree about which
        regime is worth deploying in: raw gain is positive only where ``l0`` is
        large (+0.008 above l0 = 0.6) and negative everywhere else, while the
        relative gain reports the exact opposite.  ``clip_reward`` keeps the old
        bounded behaviour; it is off by default because clipping a -300% night
        round to -100% erases the magnitude the decision needs.
        """
        gain = float(gain)
        if self.gain_mode == 'raw':
            reward = gain
        else:
            scale = max(abs(float(default_loss)), 1e-8)
            reward = gain / scale
        if self.clip_reward:
            reward = max(-1.0, min(1.0, reward))
        return float(reward)

    def reveal_feedback(self, record_id, target):
        """Reveal a delayed label and update only the experts actually run."""
        record, default_loss, gains, expert_gains = self.feedback.resolve(record_id, target)
        target = target.detach().cpu()
        self._update_alpha(record, target)
        # The controller's currency.  ``expert`` basis scores the expert itself,
        # which is stationary in the calibration state; ``mixed`` reproduces the
        # historical behaviour where the first three evaluations of every arm
        # were scored at the fixed 0.5 prior and therefore looked 0.07 worse
        # than the same arm scored after its weight had converged.
        reward_gains = expert_gains if self.reward_basis == 'expert' else gains
        rewards = {
            expert_id: self._reward(gain, default_loss)
            for expert_id, gain in reward_gains.items()
        }
        # One batch per round, not one call per arm: the controller cannot
        # remove the shared round shock without seeing the round's arms
        # together.  The loop form is kept for controllers that only expose
        # ``record_feedback`` (and it is what ``shock='raw'`` does internally).
        if hasattr(self.controller, 'record_round'):
            self.controller.record_round(rewards)
        else:
            for expert_id, reward in rewards.items():
                self.controller.record_feedback(expert_id, reward)
        self.cumulative_default_loss += default_loss
        deployed_loss = default_loss
        # The deployed action belongs to this prediction record.  With delayed
        # labels the currently active expert may already have changed.
        active_id = record.deployed_expert_id
        if active_id in gains:
            deployed_loss = default_loss - gains[active_id]
        self.cumulative_deployed_loss += deployed_loss
        # Regime memory now stores this regime's own outcomes, not the global
        # controller snapshot (which made every entry identical).  Rewards are
        # stored on the same scale the controller consumes.
        self.regime_memory.remember(
            record.signature,
            {expert_id: self._reward(gain, default_loss)
             for expert_id, gain in reward_gains.items()},
        )
        return {
            'default_loss': default_loss,
            'deployed_loss': deployed_loss,
            'gains': gains,
            'expert_gains': expert_gains,
        }

    def adapter_diag(self):
        """Summary of the private affine adapter, or None when disabled.

        ``scale_std`` is the interesting one: at initialisation ``a`` is all
        ones so the standard deviation is 0; a large value means the client's
        own feature statistics really do need a per-dimension correction.
        """
        adapter = getattr(self.model, 'adapter', None)
        if adapter is None:
            return None
        flat = adapter.flat_parameters()
        width = adapter.d_model
        scale = flat[:width]
        shift = flat[width:]
        return {
            'num_parameters': int(adapter.num_parameters),
            'scale_mean': float(scale.mean()),
            'scale_std': float(scale.std(unbiased=False)),
            'shift_abs_mean': float(shift.abs().mean()),
        }

    def summary(self):
        return {
            'client_id': self.client_id,
            'default_loss': self.cumulative_default_loss,
            'deployed_loss': self.cumulative_deployed_loss,
            'pending_feedback': len(self.feedback),
            'cache_size': len(self.cache),
            'alphas': {expert_id: self.alpha_for(expert_id) for expert_id in self.alpha_stats},
            'adapter': self.adapter_diag(),
            'controller': self.controller.snapshot(),
            'costs': self.cost_model.summary(),
        }
