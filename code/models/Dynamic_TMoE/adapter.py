"""Per-client feature recalibration on top of a shared frozen encoder.

Why this exists
---------------
On Solar-Energy the *shared frozen* patch encoder costs 7.96% of the default
path (``default_loss`` 103.96 for ``fedtdo`` vs 96.29 for the fully private
``local_encoder`` arm).  Making the encoder private recovers all of it but
destroys capsule portability: candidate gain drops from +0.0086 to -0.0189,
because an expert core consumes encoder features and two private encoders do
not produce the same feature distribution.

The compromise tested here is a *continuous, low-dimensional* private part: a
per-feature affine map ``y = a * x + b`` sitting between the frozen encoder and
the MoE stack.  It has ``2 * d_model`` parameters per client (64 at d_model=32),
is initialised to the identity, and therefore starts exactly where the shared
encoder ends.  The trunk stays shared and frozen, so a capsule is still a
well-defined object in a *shared* space - it now sees one affine image of that
space rather than an arbitrary private encoder.
"""

from __future__ import annotations

import torch
from torch import nn


class FeatureAffineAdapter(nn.Module):
    """``y = a * x + b`` per feature dimension; identity at initialisation.

    Args:
        d_model: feature width of the encoder output.
        init_scale: initial value of ``a`` (1.0 keeps the arm equal to the
            shared-encoder baseline, so any change is attributable to training).
        init_shift: initial value of ``b``.
    """

    def __init__(self, d_model: int, init_scale: float = 1.0,
                 init_shift: float = 0.0):
        super().__init__()
        self.d_model = int(d_model)
        self.scale = nn.Parameter(torch.full((self.d_model,), float(init_scale)))
        self.shift = nn.Parameter(torch.full((self.d_model,), float(init_shift)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ``x`` is [B*N, P, D]; both parameters broadcast over batch and patch.
        return x * self.scale + self.shift

    @property
    def num_parameters(self) -> int:
        return 2 * self.d_model

    def flat_parameters(self) -> torch.Tensor:
        """Detached ``[2 * d_model]`` view, for logging and diagnostics."""
        return torch.cat([self.scale.detach(), self.shift.detach()])

    def extra_repr(self) -> str:
        return f'd_model={self.d_model}, params={self.num_parameters}'
