"""Delayed-label bookkeeping for causal prequential evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import torch
import torch.nn.functional as F


@dataclass
class PendingPrediction:
    record_id: int
    due_step: int
    default_prediction: torch.Tensor
    candidate_predictions: Dict[str, torch.Tensor]
    signature: torch.Tensor
    deployed_expert_id: str | None
    # Pure external-expert forecasts (alpha = 1).  Kept so the local mix weight
    # can be calibrated by least squares once the label is revealed, without
    # paying for extra forward passes at prediction time.
    external_predictions: Dict[str, torch.Tensor] = field(default_factory=dict)


class PrequentialBuffer:
    def __init__(self):
        self._next_id = 0
        self._pending: Dict[int, PendingPrediction] = {}

    def enqueue(self, *, due_step, default_prediction, candidate_predictions, signature,
                deployed_expert_id=None, external_predictions=None):
        record = PendingPrediction(
            record_id=self._next_id,
            due_step=due_step,
            default_prediction=default_prediction.detach().cpu(),
            candidate_predictions={
                expert_id: prediction.detach().cpu()
                for expert_id, prediction in candidate_predictions.items()
            },
            signature=signature.detach().cpu(),
            deployed_expert_id=deployed_expert_id,
            external_predictions={
                expert_id: prediction.detach().cpu()
                for expert_id, prediction in (external_predictions or {}).items()
            },
        )
        self._pending[record.record_id] = record
        self._next_id += 1
        return record

    def pop(self, record_id: int) -> PendingPrediction:
        return self._pending.pop(record_id)

    def resolve(self, record_id: int, target: torch.Tensor):
        """Losses of the local path, of every mixed path, and of every pure expert.

        ``gains`` and ``expert_gains`` answer two different questions and must
        not be conflated:

        * ``gains`` is the MSE difference of the blend that *would have been
          deployed*, so its mean over rounds is the realised objective;
        * ``expert_gains`` is the MSE difference of the expert's own forecast
          (``alpha = 1``), i.e. of the expert, independent of the local mixing
          weight.

        The distinction matters because the mixing weight is *fitted per expert
        from the same revealed labels*: on Traffic it falls from the 0.5 prior to
        0.04 within ~30 rounds, so a mixed-based reward shrinks toward zero for
        exactly the arms that have been observed most.  Scoring an arm on the
        blend therefore scores its calibration history, not its quality.
        """
        record = self.pop(record_id)
        target = target.detach().cpu()
        default_loss = float(F.mse_loss(record.default_prediction, target))
        gains = {
            expert_id: default_loss - float(F.mse_loss(prediction, target))
            for expert_id, prediction in record.candidate_predictions.items()
        }
        expert_gains = {}
        for expert_id, prediction in record.external_predictions.items():
            shape_ok = prediction.shape == target.shape
            if not shape_ok:
                continue
            expert_gains[expert_id] = default_loss - float(F.mse_loss(prediction, target))
        return record, default_loss, gains, expert_gains

    def __len__(self):
        return len(self._pending)
