"""Portable, client-independent Dynamic-TMoE expert capsules."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Dict, Mapping

import torch

from models.Dynamic_TMoE.experts import build_transferable_expert, expert_type_of


def _architecture_hash(expert_type: str, state_dict: Mapping[str, torch.Tensor]) -> str:
    shape_spec = '|'.join(
        f'{name}:{tuple(value.shape)}:{value.dtype}'
        for name, value in sorted(state_dict.items())
    )
    return hashlib.sha256(f'{expert_type}|{shape_spec}'.encode('utf-8')).hexdigest()[:16]


@dataclass
class ExpertCapsule:
    """The transferable core of one expert, without client-specific relations."""

    expert_id: str
    source_id: str
    expert_type: str
    layer_idx: int
    d_model: int
    num_patches: int
    encoder_version: str
    state_dict: Dict[str, torch.Tensor]
    regime_signature: torch.Tensor | None = None
    registry_version: int = 0
    architecture_hash: str = ''
    _materialized: Dict[str, torch.nn.Module] = field(default_factory=dict, init=False,
                                                       repr=False, compare=False)

    def __post_init__(self):
        self.state_dict = {
            name: value.detach().cpu().clone().contiguous()
            for name, value in self.state_dict.items()
        }
        if self.regime_signature is not None:
            self.regime_signature = self.regime_signature.detach().cpu().clone()
        if not self.architecture_hash:
            self.architecture_hash = _architecture_hash(self.expert_type, self.state_dict)

    @classmethod
    def from_local_expert(cls, expert, *, expert_id: str, source_id: str,
                          layer_idx: int, d_model: int, num_patches: int,
                          encoder_version: str, regime_signature=None):
        # Relation layers depend on a source client's variable graph.  Keeping
        # them would make a capsule silently invalid on another client.
        transferable = {
            key: value for key, value in expert.state_dict().items()
            if not key.startswith('relation_layer.')
        }
        return cls(
            expert_id=expert_id,
            source_id=source_id,
            expert_type=expert_type_of(expert),
            layer_idx=layer_idx,
            d_model=d_model,
            num_patches=num_patches,
            encoder_version=encoder_version,
            state_dict=transferable,
            regime_signature=regime_signature,
        )

    @property
    def estimated_bytes(self) -> int:
        return sum(value.numel() * value.element_size() for value in self.state_dict.values())

    @property
    def estimated_flops(self) -> int:
        # A deterministic proxy for evaluation cost.  The experiment logs this
        # separately from wall-clock latency, which is hardware dependent.
        return 2 * sum(value.numel() for value in self.state_dict.values())

    def is_compatible_with(self, *, layer_idx: int, d_model: int, num_patches: int,
                           encoder_version: str) -> bool:
        return (
            self.layer_idx == layer_idx
            and self.d_model == d_model
            and self.num_patches == num_patches
            and self.encoder_version == encoder_version
        )

    def materialize(self, device):
        """Instantiate a read-only inference module on ``device`` lazily."""
        key = str(device)
        if key not in self._materialized:
            module = build_transferable_expert(
                self.expert_type, self.d_model, self.num_patches
            )
            missing, unexpected = module.load_state_dict(self.state_dict, strict=False)
            if missing or unexpected:
                raise ValueError(
                    f'Invalid capsule {self.expert_id}: missing={missing}, '
                    f'unexpected={unexpected}'
                )
            module.to(device).eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
            self._materialized[key] = module
        return self._materialized[key]
