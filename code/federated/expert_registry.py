"""In-memory server-side registry for versioned expert capsules."""

from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

from .expert_capsule import ExpertCapsule


class ExpertRegistry:
    """A deterministic registry used by the simulated federated server.

    It intentionally stores capsules rather than complete client models: raw
    data, normalizers, routers, prediction heads, and relation graphs never
    enter the registry.
    """

    def __init__(self, max_experts: int | None = None):
        self.max_experts = max_experts
        self._capsules: OrderedDict[str, ExpertCapsule] = OrderedDict()
        self.version = 0
        self.bytes_uploaded = 0

    def register(self, capsule: ExpertCapsule) -> ExpertCapsule:
        self.version += 1
        capsule.registry_version = self.version
        if capsule.expert_id in self._capsules:
            self._capsules.pop(capsule.expert_id)
        self._capsules[capsule.expert_id] = capsule
        self.bytes_uploaded += capsule.estimated_bytes
        if self.max_experts is not None:
            while len(self._capsules) > self.max_experts:
                self._capsules.popitem(last=False)
        return capsule

    def candidates_for(self, *, requester_id: str, layer_idx: int, d_model: int,
                       num_patches: int, encoder_version: str,
                       include_self: bool = False) -> list[ExpertCapsule]:
        """Capsules this client may evaluate.

        ``include_self`` keeps this client's own earlier snapshots in the pool.
        They are the temporal experts of A3: zero-byte candidates that share the
        client's own data distribution, so they can match a drift that no
        cross-client expert can.  Off by default (published behaviour).
        """
        return [
            capsule for capsule in self._capsules.values()
            if (include_self or capsule.source_id != requester_id)
            and capsule.is_compatible_with(
                layer_idx=layer_idx, d_model=d_model,
                num_patches=num_patches, encoder_version=encoder_version,
            )
        ]

    def get(self, expert_id: str) -> ExpertCapsule:
        return self._capsules[expert_id]

    def __len__(self):
        return len(self._capsules)
