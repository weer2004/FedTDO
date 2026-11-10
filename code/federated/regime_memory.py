"""State-signature memory for cautious recurrent-regime reuse."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


@dataclass
class RegimeEntry:
    signature: torch.Tensor
    # Mean observed gain of each external expert *inside this regime*, plus the
    # number of outcomes that produced it.  Storing the controller's global
    # snapshot here (as the previous version did) made every entry converge to
    # the same content, so two different regimes were indistinguishable.
    gains: Dict[str, float] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)
    visits: int = 1


class RegimeMemory:
    def __init__(self, max_entries=32, similarity_threshold=0.9):
        self.max_entries = max_entries
        self.similarity_threshold = similarity_threshold
        self.entries: list[RegimeEntry] = []

    @staticmethod
    def signature_from_input(x_enc: torch.Tensor) -> torch.Tensor:
        """Fallback descriptor when no hidden representation is available."""
        x = x_enc.detach().float()
        mean = x.mean(dim=(0, 1))
        std = x.std(dim=(0, 1), unbiased=False)
        slope = (x[:, -1, :] - x[:, 0, :]).mean(dim=0)
        signature = torch.cat([mean, std, slope]).cpu()
        return F.normalize(signature, dim=0)

    @staticmethod
    def _similarity(signature: torch.Tensor, entry: RegimeEntry) -> float:
        if signature.numel() != entry.signature.numel():
            return 0.0
        return float(torch.dot(signature, entry.signature))

    def retrieve(self, signature: torch.Tensor) -> Dict[str, Tuple[float, float]]:
        """Return ``{expert_id: (mean_gain, similarity)}`` for the closest regime.

        The similarity is handed to the controller, which turns the prior into a
        pseudo-observation weighted by it: a recurring regime is reused, but the
        historical gain is never treated as a fresh current observation.
        """
        if not self.entries or signature is None:
            return {}
        signature = F.normalize(signature.detach().float().cpu(), dim=0)
        index = max(range(len(self.entries)),
                    key=lambda i: self._similarity(signature, self.entries[i]))
        similarity = self._similarity(signature, self.entries[index])
        if similarity < self.similarity_threshold:
            return {}
        entry = self.entries[index]
        return {expert_id: (gain, similarity) for expert_id, gain in entry.gains.items()}

    def remember(self, signature: torch.Tensor, gains: Dict[str, float]):
        """Accumulate the gains actually observed for this regime."""
        if not gains or signature is None:
            return
        signature = F.normalize(signature.detach().float().cpu(), dim=0)
        target = None
        if self.entries:
            index = max(range(len(self.entries)),
                        key=lambda i: self._similarity(signature, self.entries[i]))
            if self._similarity(signature, self.entries[index]) >= self.similarity_threshold:
                target = self.entries[index]
        if target is None:
            target = RegimeEntry(signature=signature)
            self.entries.append(target)
            if len(self.entries) > self.max_entries:
                self.entries.pop(0)
        for expert_id, gain in gains.items():
            count = target.counts.get(expert_id, 0)
            previous = target.gains.get(expert_id, 0.0)
            target.gains[expert_id] = (previous * count + float(gain)) / (count + 1)
            target.counts[expert_id] = count + 1
        target.visits += 1
