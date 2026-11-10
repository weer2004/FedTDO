"""GRU forecaster — the backbone implied by "FedGRU" style federated forecasting.

A two-layer GRU reads the lookback and the last hidden state is linearly mapped to
the horizon.  Under ``--backbone gru`` + ``--baseline fedavg`` this reproduces the
"FedGRU" entry: a recurrent local model with parameter averaging, no
personalisation, no routing.

Recurrent models are listed last in the checklist for a reason: at the same
1600-step budget they train far more slowly per step than a patch transformer on
this data size, so the row mostly measures sample efficiency, not modelling
capacity.
"""

from __future__ import annotations

import torch
from torch import nn

from .base import ZooForecaster


class GRUForecaster(ZooForecaster):
    name = 'gru'

    def __init__(self, args):
        super().__init__(args)
        hidden = int(getattr(args, 'd_model', 32))
        layers = int(getattr(args, 'num_rnn_layers', 2) or 2)
        self.rnn = nn.GRU(input_size=self.enc_in, hidden_size=hidden,
                          num_layers=layers, batch_first=True,
                          dropout=float(getattr(args, 'dropout', 0.1)) if layers > 1 else 0.0)
        self.dropout = nn.Dropout(float(getattr(args, 'dropout', 0.1)))
        self.head = nn.Linear(hidden, self.pred_len * self.enc_in)

    def encode(self, x_norm: torch.Tensor) -> torch.Tensor:
        out, hidden = self.rnn(x_norm)
        # [B, L, D] -> one token per timestep, matching the zoo contract
        return out

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        b = tokens.shape[0]
        last = self.dropout(tokens[:, -1])
        return self.head(last).reshape(b, self.pred_len, self.enc_in)
