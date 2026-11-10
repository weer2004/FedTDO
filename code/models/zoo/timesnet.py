"""TimesNet (ICLR 2023) — 1D series to 2D (period, phase) tensors.

Ported from ``models/TimesNet.py`` + ``layers/Conv_Blocks.py``:

* embedding: 1D conv over time from ``enc_in`` channels to ``d_model``;
* ``predict_linear`` extends the lookback to ``seq_len + pred_len`` before the
  blocks, exactly as TSLib does (the model forecasts a padded sequence);
* ``TimesBlock``: FFT amplitude -> top-k periods -> reshape to ``[B, D, period,
  length]`` -> inception (2D conv at 6 kernel widths, softmax fused);
* amplitude-weighted combination of the top-k branches, residual + LayerNorm.

Faithful in block structure.  Deviations: no ``enc_in``-specific channel mixing
beyond the embedding conv (identical to TSLib), and ``top_k`` defaults to 3
instead of 3-but-clipped-by-d_model.
"""

from __future__ import annotations

import torch
from torch import nn

from .base import ZooForecaster
from .tslib_blocks import TimesBlock


class TimesNet(ZooForecaster):
    name = 'timesnet'

    def __init__(self, args):
        super().__init__(args)
        d_model = self.d_model
        # TSLib's own default pairs ``d_ff == d_model`` for TimesNet; the 6-kernel
        # inception bank is parameter hungry (2 x 6 conv2d per block), so keeping
        # the 4x FFN width would put this backbone at 2.3M params vs 55k for
        # PatchTST and turn the axis into a capacity comparison.
        d_ff = int(getattr(args, 'timesnet_d_ff', d_model))
        e_layers = max(1, int(getattr(args, 'e_layers', 2)))
        top_k = int(getattr(args, 'top_k', 3))
        dropout = float(getattr(args, 'dropout', 0.1))

        self.embedding = nn.Conv1d(self.enc_in, d_model, kernel_size=3, padding=1)
        self.predict_linear = nn.Linear(self.seq_len, self.seq_len + self.pred_len)
        self.blocks = nn.ModuleList([
            TimesBlock(self.seq_len + self.pred_len, d_model, d_ff, top_k,
                       num_kernels=6, dropout=dropout)
            for _ in range(e_layers)])
        self.head = nn.Linear(d_model, self.enc_in)

    def encode(self, x_norm: torch.Tensor) -> torch.Tensor:
        # [B, L, C] -> [B, L+P, C] -> [B, L+P, D]
        z = self.predict_linear(x_norm.transpose(1, 2)).transpose(1, 2)
        z = self.embedding(z.transpose(1, 2)).transpose(1, 2)
        for block in self.blocks:
            z = block(z)
        return z

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(tokens[:, -self.pred_len:, :])
