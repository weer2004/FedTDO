"""Autoformer (NeurIPS 2021) — encoder-decoder with auto-correlation.

Ported from ``models/Autoformer.py`` + ``layers/Autoformer_EncDec.py``.  What is
faithful:

* **auto-correlation** instead of self-attention (FFT period detection, top-k
  delays, delay aggregation);
* **series decomposition** after every sub-layer, encoder keeps the seasonal part
  and the decoder accumulates the trend terms (progressive decomposition);
* **cumulative-mean decoder initialisation** — the decoder input is
  ``cat(x_enc_seasonal[-label:], 0) + cat(x_enc_trend[-label:], mean(x_enc))``;
* **RevIN** (as with every model in this zoo) instead of TSLib's optional
  ``non_stationary`` flag, so all backbones see the same normalisation.

Not implemented (documented deviations): ``output_attention`` diagnostics and the
``moving_avg`` window schedule beyond a single fixed kernel.
"""

from __future__ import annotations

import torch
from torch import nn

from .base import Seq2SeqForecaster
from .tslib_blocks import AutoCorrelationLayer, SeriesDecomp, activation


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, moving_avg: int,
                 factor: float, dropout: float, act: str):
        super().__init__()
        self.attention = AutoCorrelationLayer(d_model, n_heads, factor, dropout)
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.decomp1 = SeriesDecomp(moving_avg)
        self.decomp2 = SeriesDecomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.act = activation(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attention(x, x, x))
        x, _ = self.decomp1(x)                      # keep seasonal
        y = self.act(self.conv1(x.transpose(1, 2)))
        y = self.conv2(y).transpose(1, 2)
        x = x + self.dropout(y)
        x, _ = self.decomp2(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, moving_avg: int,
                 factor: float, dropout: float, act: str):
        super().__init__()
        self.self_attention = AutoCorrelationLayer(d_model, n_heads, factor, dropout)
        self.cross_attention = AutoCorrelationLayer(d_model, n_heads, factor, dropout)
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.decomp1 = SeriesDecomp(moving_avg)
        self.decomp2 = SeriesDecomp(moving_avg)
        self.decomp3 = SeriesDecomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.act = activation(act)

    def forward(self, x: torch.Tensor, memory: torch.Tensor):
        x = x + self.dropout(self.self_attention(x, x, x))
        x, trend1 = self.decomp1(x)
        x = x + self.dropout(self.cross_attention(x, memory, memory))
        x, trend2 = self.decomp2(x)
        y = self.act(self.conv1(x.transpose(1, 2)))
        y = self.conv2(y).transpose(1, 2)
        x = x + self.dropout(y)
        x, trend3 = self.decomp3(x)
        return x, trend1 + trend2 + trend3


class Autoformer(Seq2SeqForecaster):
    name = 'autoformer'

    def __init__(self, args):
        super().__init__(args)
        d_model = self.d_model
        n_heads = max(1, min(int(getattr(args, 'n_heads', 4)), d_model))
        while d_model % n_heads:
            n_heads -= 1
        d_ff = int(getattr(args, 'd_ff', 4 * d_model))
        e_layers = int(getattr(args, 'e_layers', 2))
        d_layers = int(getattr(args, 'd_layers', 1))
        dropout = float(getattr(args, 'dropout', 0.1))
        act = str(getattr(args, 'activation', 'gelu'))
        factor = float(getattr(args, 'factor', 3))
        moving_avg = int(getattr(args, 'moving_avg', 25))

        self.enc_embedding = nn.Conv1d(self.enc_in, d_model, kernel_size=3, padding=1)
        self.dec_embedding = nn.Conv1d(self.enc_in, d_model, kernel_size=3, padding=1)
        self.encoder = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, moving_avg, factor, dropout, act)
            for _ in range(max(1, e_layers))])
        self.decoder = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, moving_avg, factor, dropout, act)
            for _ in range(max(1, d_layers))])
        self.projection = nn.Linear(d_model, self.enc_in)

    def encode(self, x_norm: torch.Tensor) -> torch.Tensor:
        z = self.enc_embedding(x_norm.transpose(1, 2)).transpose(1, 2)
        for layer in self.encoder:
            z = layer(z)
        return z

    def decode_from(self, enc_out: torch.Tensor, dec_in: torch.Tensor,
                    x_norm: torch.Tensor) -> torch.Tensor:
        # ``dec_in`` is cat(x_enc[-label_len:], 0); Autoformer replaces the zero
        # block with the cumulative mean of the encoder input (Eq. 5 in the
        # paper, ``decomp_initialize`` in the released code).  No future values.
        mean = x_norm.mean(dim=1, keepdim=True).repeat(1, self.pred_len, 1)
        init = torch.cat([dec_in[:, :self.label_len, :], mean], dim=1)
        x = self.dec_embedding(init.transpose(1, 2)).transpose(1, 2)
        trend_acc = 0.0
        for layer in self.decoder:
            x, trend = layer(x, enc_out)
            trend_acc = trend_acc + trend
        x = x + trend_acc
        return self.projection(x)
