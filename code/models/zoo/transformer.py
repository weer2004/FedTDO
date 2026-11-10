"""Vanilla Transformer encoder (Informer-lineage, channel-mixing).

Stands in for ``Transformer`` / ``Informer`` on the checklist: a per-timestep
projection of the multivariate input, fixed positional encoding, a standard
``nn.TransformerEncoder``, and a linear horizon head.  Unlike PatchTST this one
mixes channels before the attention, which is what Informer-style models do and
what makes them behave differently from channel-independent patching when a
client owns more than one sensor.  At ``enc_in = 1`` the difference reduces to
"token = one timestep" vs "token = one patch".
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .base import ZooForecaster


class VanillaTransformer(ZooForecaster):
    name = 'transformer'

    def __init__(self, args):
        super().__init__(args)
        self.value_embedding = nn.Linear(self.enc_in, self.d_model)
        pe = torch.zeros(1, self.seq_len, self.d_model)
        position = torch.arange(0, self.seq_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, self.d_model, 2).float()
                        * -(math.log(10000.0) / self.d_model))
        pe[0, :, 0::2] = torch.sin(position * div)
        pe[0, :, 1::2] = torch.cos(position * div)
        self.register_buffer('pe', pe)
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
        self.head = nn.Linear(self.seq_len * self.d_model, self.pred_len * self.enc_in)

    def encode(self, x_norm: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.value_embedding(x_norm) + self.pe[:, :x_norm.shape[1]])

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        b = tokens.shape[0]
        out = self.head(self.dropout(tokens.reshape(b, -1)))
        return out.reshape(b, self.pred_len, self.enc_in)
