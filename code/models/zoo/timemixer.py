"""TimeMixer (ICLR 2024) — decomposable multiscale mixing.

Ported from the structure of ``models/TimeMixer.py``:

* 1D conv embedding from ``enc_in`` channels to ``d_model``;
* a **pyramid** of ``down_sampling_layers + 1`` scales (average pooling, factor 2);
* at every scale a ``PastDecomposableMixing`` block that splits the series into
  seasonal/trend, mixes each with a depthwise temporal conv + channel MLP and
  recombines;
* **bottom-up trend mixing** (a finer scale's trend is pooled into the coarser
  scale) and **top-down seasonal mixing** (a coarser scale's seasonality is
  repeated into the finer scale), which is the "decomposable mixing" of the paper;
* per-scale linear forecasters whose outputs are averaged, then denormalised.

Deviations from TSLib: the per-scale predictor is ``Linear(T_l -> pred_len)``
applied along the time axis (same as TSLib) followed by one shared
``Linear(d_model -> C)`` channel map, instead of TSLib's ``channel_independence``
switch; the multispectral feed-forward branch (``use_future_temporal_feature``)
is not implemented.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .base import ZooForecaster
from .tslib_blocks import SeriesDecomp, activation


class PastDecomposableMixing(nn.Module):
    def __init__(self, d_model: int, d_ff: int, kernel_size: int = 25,
                 dropout: float = 0.1, act: str = 'gelu'):
        super().__init__()
        self.decomp_layers = nn.ModuleList([
            SeriesDecomp(kernel_size), SeriesDecomp(kernel_size)])
        self.channel_mlp = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_ff), activation(act), nn.Dropout(dropout),
                          nn.Linear(d_ff, d_model))
            for _ in range(2)])
        self.temporal_conv = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
            for _ in range(2)])
        self.norm = nn.ModuleList([nn.LayerNorm(d_model), nn.LayerNorm(d_model)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """[B, T, D] -> (seasonal, trend), each mixed and residual."""
        outs = []
        for i in range(2):
            seasonal, trend = self.decomp_layers[i](x)
            seasonal = self.temporal_conv[i](seasonal.transpose(1, 2)).transpose(1, 2)
            seasonal = self.channel_mlp[i](seasonal)
            mixed = self.dropout(seasonal) + trend
            outs.append(self.norm[i](x + mixed))
        return outs[0], outs[1]


class TimeMixer(ZooForecaster):
    name = 'timemixer'

    def __init__(self, args):
        super().__init__(args)
        d_model = self.d_model
        d_ff = int(getattr(args, 'd_ff', 4 * d_model))
        dropout = float(getattr(args, 'dropout', 0.1))
        act = str(getattr(args, 'activation', 'gelu'))
        self.levels = max(1, int(getattr(args, 'down_sampling_layers', 1)) + 1)
        kernel = int(getattr(args, 'moving_avg', 25))

        self.embedding = nn.Conv1d(self.enc_in, d_model, kernel_size=3, padding=1)
        self.pdm = nn.ModuleList([
            PastDecomposableMixing(d_model, d_ff, kernel, dropout, act)
            for _ in range(self.levels)])
        # per-scale predictors map the *time* axis to the horizon (as in TSLib);
        # named ``head_*`` so ``head_state_dict`` (prefix match) treats them as
        # the personalised head
        lengths = [max(2, self.seq_len // (2 ** l)) for l in range(self.levels)]
        self.head_layers = nn.ModuleList([
            nn.Linear(t_l, self.pred_len) for t_l in lengths])
        self.channel_map = nn.Linear(d_model, self.enc_in)
        self.output_norm = nn.LayerNorm(d_model)

    # -- contract ------------------------------------------------------------
    def encode(self, x_norm: torch.Tensor) -> torch.Tensor:
        z = self.embedding(x_norm.transpose(1, 2)).transpose(1, 2)     # [B, T, D]
        return z

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        z = self.output_norm(tokens).permute(0, 2, 1)          # [B, D, T]
        z = self.head_layers[0](z).permute(0, 2, 1)            # [B, pred, D]
        return self.channel_map(z)

    # -- full multiscale path ------------------------------------------------
    def forward(self, x: torch.Tensor, x_mark: torch.Tensor | None = None) -> torch.Tensor:
        x = x.float()
        x_norm, mean, std = self.revin.normalize(x)
        b, _, c = x_norm.shape
        z = self.embedding(x_norm.transpose(1, 2)).transpose(1, 2)

        # build the scale pyramid (average pooling by 2)
        pyramid = [z]
        for _ in range(1, self.levels):
            z = pyramid[-1].transpose(1, 2)
            z = F.avg_pool1d(z, kernel_size=2, stride=2).transpose(1, 2)
            pyramid.append(z)

        seasonal, trend = [], []
        for lvl, block in enumerate(self.pdm):
            s, t = block(pyramid[lvl])
            seasonal.append(s)
            trend.append(t)

        # bottom-up: finer trend informs the coarser scale
        for lvl in range(self.levels - 1, 0, -1):
            up = F.interpolate(trend[lvl - 1].transpose(1, 2),
                               size=trend[lvl].shape[1], mode='nearest').transpose(1, 2)
            trend[lvl] = trend[lvl] + up
        # top-down: coarser seasonality informs the finer scale
        for lvl in range(1, self.levels):
            down = F.interpolate(seasonal[lvl].transpose(1, 2),
                                 size=seasonal[lvl - 1].shape[1], mode='nearest').transpose(1, 2)
            seasonal[lvl - 1] = seasonal[lvl - 1] + down

        preds = []
        for lvl in range(self.levels):
            z = self.output_norm(seasonal[lvl] + trend[lvl])      # [B, T_l, D]
            p = self.head_layers[lvl](z.permute(0, 2, 1))         # [B, D, pred]
            preds.append(p.permute(0, 2, 1))                      # [B, pred, D]
        out = torch.stack(preds, dim=0).mean(dim=0)
        out = self.channel_map(out)                               # [B, pred, C]
        return self.revin.denormalize(out, mean, std)

    def _encode_patches(self, x: torch.Tensor, x_mark: torch.Tensor | None = None):
        x = x.float()
        x_norm, _, _ = self.revin.normalize(x)
        tokens = self.encode(x_norm)
        return tokens, None, None, None, None, None
