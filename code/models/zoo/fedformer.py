"""FEDformer (ICML 2022) — frequency-enhanced encoder-decoder.

Ported from ``models/FEDformer.py`` + ``layers/FourierCorrelation.py`` in the
``version='Fourier'`` / ``mode_select='low'`` configuration (the variant the paper
recommends for long-horizon forecasting).

Faithful: FEB-f (truncated Fourier attention with per-head complex mixing),
MOE-decomposition (multi-kernel moving-average experts with a learned gate),
progressive trend accumulation in the decoder, cumulative-mean decoder init.

Deviations: wavelet mode (``version='Wavelets'``) and progressive frequency
mixing are not implemented — with ``seq_len=96`` only 49 frequency bins exist, so
the "random mode select + progressive mixing" machinery of the 512-length setup
has nothing to select from.
"""

from __future__ import annotations

import torch
from torch import nn

from .base import Seq2SeqForecaster
from .tslib_blocks import FourierBlock, MOEDecomp, activation


class FEDformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, seq_len: int,
                 modes: int, dropout: float, act: str, kernel_set=(1, 2, 4, 8, 16)):
        super().__init__()
        self.attention = FourierBlock(d_model, n_heads, seq_len, modes, dropout)
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.decomp1 = MOEDecomp(kernel_set, dropout)
        self.decomp2 = MOEDecomp(kernel_set, dropout)
        self.dropout = nn.Dropout(dropout)
        self.act = activation(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attention(x, x, x))
        x, _ = self.decomp1(x)
        y = self.act(self.conv1(x.transpose(1, 2)))
        y = self.conv2(y).transpose(1, 2)
        x = x + self.dropout(y)
        x, _ = self.decomp2(x)
        return x


class FEDformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, seq_len: int,
                 modes: int, dropout: float, act: str, kernel_set=(1, 2, 4, 8, 16)):
        super().__init__()
        self.self_attention = FourierBlock(d_model, n_heads, seq_len, modes, dropout)
        self.cross_attention = FourierBlock(d_model, n_heads, seq_len, modes, dropout)
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.decomp1 = MOEDecomp(kernel_set, dropout)
        self.decomp2 = MOEDecomp(kernel_set, dropout)
        self.decomp3 = MOEDecomp(kernel_set, dropout)
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


class FEDformer(Seq2SeqForecaster):
    name = 'fedformer'

    def __init__(self, args):
        super().__init__(args)
        d_model = self.d_model
        n_heads = max(1, min(int(getattr(args, 'n_heads', 4)), d_model))
        while d_model % n_heads:
            n_heads -= 1
        d_ff = int(getattr(args, 'd_ff', 4 * d_model))
        e_layers = max(1, int(getattr(args, 'e_layers', 2)))
        d_layers = max(1, int(getattr(args, 'd_layers', 1)))
        dropout = float(getattr(args, 'dropout', 0.1))
        act = str(getattr(args, 'activation', 'gelu'))
        modes = int(getattr(args, 'modes', min(32, self.seq_len // 2)))

        self.enc_embedding = nn.Conv1d(self.enc_in, d_model, kernel_size=3, padding=1)
        self.dec_embedding = nn.Conv1d(self.enc_in, d_model, kernel_size=3, padding=1)
        self.encoder = nn.ModuleList([
            FEDformerEncoderLayer(d_model, n_heads, d_ff, self.seq_len, modes,
                                  dropout, act) for _ in range(e_layers)])
        self.decoder = nn.ModuleList([
            FEDformerDecoderLayer(d_model, n_heads, d_ff,
                                  self.label_len + self.pred_len, modes, dropout, act)
            for _ in range(d_layers)])
        self.projection = nn.Linear(d_model, self.enc_in)

    def encode(self, x_norm: torch.Tensor) -> torch.Tensor:
        z = self.enc_embedding(x_norm.transpose(1, 2)).transpose(1, 2)
        for layer in self.encoder:
            z = layer(z)
        return z

    def decode_from(self, enc_out: torch.Tensor, dec_in: torch.Tensor,
                    x_norm: torch.Tensor) -> torch.Tensor:
        mean = x_norm.mean(dim=1, keepdim=True).repeat(1, self.pred_len, 1)
        init = torch.cat([dec_in[:, :self.label_len, :], mean], dim=1)
        x = self.dec_embedding(init.transpose(1, 2)).transpose(1, 2)
        trend_acc = 0.0
        for layer in self.decoder:
            x, trend = layer(x, enc_out)
            trend_acc = trend_acc + trend
        x = x + trend_acc
        return self.projection(x)
