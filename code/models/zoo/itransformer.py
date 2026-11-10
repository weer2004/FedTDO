"""iTransformer (ICLR 2024) — inverted tokens, attention across variates.

Each variate's whole lookback becomes one token, so self-attention runs along the
**variate** axis instead of the time axis.

**Read this before quoting an iTransformer number from this repo.**  The current
protocol gives every client exactly one channel (``enc_in = 1``), so there is one
token and attention over a single token is the identity: the model degenerates
into ``Linear(seq_len -> d_model) -> FFN -> Linear(d_model -> pred_len)``, i.e. a
per-variate MLP.  That is not iTransformer's intended regime and the resulting
MSE measures the degenerate form.  Making it meaningful needs multi-variate
clients (``features M`` + a client that owns several sensors), which is the gap
recorded in ``EXPERIMENT_PLAN.md``.
"""

from __future__ import annotations

import torch
from torch import nn

from .base import ZooForecaster


class ITransformer(ZooForecaster):
    name = 'itransformer'

    def __init__(self, args):
        super().__init__(args)
        self.value_embedding = nn.Linear(self.seq_len, self.d_model)
        n_heads = int(getattr(args, 'n_heads', 4))
        n_heads = max(1, min(n_heads, self.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=n_heads,
            dim_feedforward=int(getattr(args, 'd_ff', 4 * self.d_model)),
            dropout=float(getattr(args, 'dropout', 0.1)),
            activation='gelu', batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=int(getattr(args, 'e_layers', 2)))
        self.dropout = nn.Dropout(float(getattr(args, 'dropout', 0.1)))
        self.head = nn.Linear(self.d_model, self.pred_len)

    def encode(self, x_norm: torch.Tensor) -> torch.Tensor:
        # [B, L, C] -> [B, C, L] -> variate tokens [B, C, D]
        return self.encoder(self.value_embedding(x_norm.transpose(1, 2)))

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(tokens)).transpose(1, 2)
