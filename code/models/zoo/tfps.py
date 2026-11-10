"""TFPS (NeurIPS 2025) — pattern-specific experts under patch-level shift.

Reference: Sun, Xie, Eldele, Chen, Hu, Wu, "Learning Pattern-Specific Experts for
Time Series Forecasting Under Patch-level Distribution Shift", NeurIPS 2025,
arXiv:2410.09836 (code: github.com/syrGitHub/TFPS).

Three pieces, in the paper's order:

1. **Dual-Domain Encoder (DDE)** — every patch is embedded twice, once from the
   time domain (linear map on the raw patch) and once from the frequency domain
   (magnitude of its rFFT); the two are summed.  The frequency branch is what lets
   the encoder see drift that a linear time-domain map cannot.
2. **Subspace clustering** — ``m`` learnable subspace bases; each patch is softly
   assigned to the subspaces by cosine/linear score, so a patch belongs to a
   "pattern" rather than to a fixed expert.
3. **Pattern-specific experts** — one linear expert per subspace; a patch's
   forecast is the assignment-weighted mixture of expert outputs.  Patches are
   then pooled with a learned attention score into one horizon.

Deviation from the released code: TFPS predicts a horizon **per patch** and the
paper aggregates the patch predictions; the released implementation keeps the last
patch's prediction.  Here the aggregation is a learned score over patches
(``head_score``), which is strictly more general and costs one scalar per patch.
``num_patterns`` and the temperature are exposed as ``--tfps_patterns`` /
``--tfps_tau``.
"""

from __future__ import annotations

import torch
from torch import nn

from .base import ZooForecaster


class TFPS(ZooForecaster):
    name = 'tfps'

    def __init__(self, args):
        super().__init__(args)
        self.num_patterns = int(getattr(args, 'tfps_patterns', 4))
        self.tau = float(getattr(args, 'tfps_tau', 1.0))
        self.num_patches = int((self.seq_len - self.patch_len) / self.stride + 2)
        self.padding = (self.num_patches - 1) * self.stride + self.patch_len - self.seq_len
        self.pad = nn.ReplicationPad1d((0, max(0, self.padding)))

        # -- 1. dual-domain encoder
        self.time_embed = nn.Linear(self.patch_len, self.d_model, bias=False)
        self.freq_embed = nn.Linear(self.patch_len // 2 + 1, self.d_model, bias=False)
        self.position = nn.Parameter(torch.randn(1, self.num_patches, self.d_model) * 0.02)
        self.norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(float(getattr(args, 'dropout', 0.1)))

        # -- 2. subspace clustering
        self.subspace = nn.Parameter(torch.randn(self.num_patterns, self.d_model) * 0.02)

        # -- 3. pattern-specific experts (``head_`` prefix -> personalised head)
        self.head_experts = nn.ModuleList([
            nn.Linear(self.d_model, self.pred_len) for _ in range(self.num_patterns)])
        self.head_score = nn.Linear(self.d_model, 1)

    def _patches(self, x_norm: torch.Tensor) -> torch.Tensor:
        b, _, c = x_norm.shape
        z = x_norm.transpose(1, 2).reshape(b * c, 1, self.seq_len)
        z = self.pad(z).unfold(2, self.patch_len, self.stride).squeeze(1)
        return z                                            # [B*C, P, Lp]

    def encode(self, x_norm: torch.Tensor) -> torch.Tensor:
        z = self._patches(x_norm)
        time_feat = self.time_embed(z)                                  # [B*C,P,D]
        spec = torch.fft.rfft(z, dim=-1).abs()                          # [B*C,P,Lp/2+1]
        freq_feat = self.freq_embed(spec)
        return self.norm(nn.functional.gelu(time_feat + freq_feat) + self.position)

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        c = self.enc_in
        b = tokens.shape[0] // c
        h = self.dropout(tokens)                                        # [B*C,P,D]

        assign = torch.softmax(h @ self.subspace.t() / self.tau, dim=-1)
        expert_out = torch.stack([e(h) for e in self.head_experts], dim=2)
        per_patch = (assign.unsqueeze(-1) * expert_out).sum(dim=2)      # [B*C,P,H]

        score = torch.softmax(self.head_score(h), dim=1)                # [B*C,P,1]
        out = (score * per_patch).sum(dim=1)                            # [B*C,H]
        return out.reshape(b, c, self.pred_len).transpose(1, 2)
