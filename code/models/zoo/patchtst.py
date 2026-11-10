"""PatchTST (ICLR 2023) — patch tokens + transformer encoder, channel independent.

Faithful in the parts that matter for a like-for-like comparison:

* patch count matches Dynamic_TMoE exactly (``(seq_len - patch_len) / stride + 2``
  with the same replication padding), so the sequence length seen by the encoder
  is identical;
* shared weights across channels (the "channel independence" of the paper), which
  is the only correct choice here because every client holds one channel;
* RevIN + a single linear forecast head, as in the released code.
"""

from __future__ import annotations

import torch
from torch import nn

from .base import ZooForecaster


class PatchTST(ZooForecaster):
    name = 'patchtst'

    def __init__(self, args):
        super().__init__(args)
        self.num_patches = int((self.seq_len - self.patch_len) / self.stride + 2)
        self.padding = (self.num_patches - 1) * self.stride + self.patch_len - self.seq_len
        self.pad = nn.ReplicationPad1d((0, max(0, self.padding)))
        self.value_embedding = nn.Linear(self.patch_len, self.d_model, bias=False)
        self.position = nn.Parameter(torch.randn(1, self.num_patches, self.d_model) * 0.02)
        n_heads = int(getattr(args, 'n_heads', 4))
        n_heads = max(1, min(n_heads, self.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=n_heads,
            dim_feedforward=int(getattr(args, 'd_ff', 4 * self.d_model)),
            dropout=float(getattr(args, 'dropout', 0.1)),
            activation='gelu', batch_first=True, norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=int(getattr(args, 'e_layers', 2)))
        self.dropout = nn.Dropout(float(getattr(args, 'dropout', 0.1)))
        self.head = nn.Linear(self.num_patches * self.d_model, self.pred_len)

    def encode(self, x_norm: torch.Tensor) -> torch.Tensor:
        b, _, c = x_norm.shape
        z = x_norm.transpose(1, 2).reshape(b * c, 1, self.seq_len)
        z = self.pad(z).unfold(2, self.patch_len, self.stride).squeeze(1)  # [B*C, P, Lp]
        z = self.value_embedding(z) + self.position
        return self.encoder(z)

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        c = self.enc_in
        b = tokens.shape[0] // c
        flat = self.dropout(tokens.reshape(b * c, -1))
        out = self.head(flat).reshape(b, c, self.pred_len)
        return out.transpose(1, 2)
