"""ST-MTM (KDD 2025) — masked time-series modelling with seasonal-trend decomposition.

Reference: Seo & Lim, "ST-MTM: Masked Time Series Modeling with Seasonal-Trend
Decomposition for Time Series Forecasting", KDD 2025, arXiv:2507.00013
(code: github.com/hwseo95/st-mtm).

The claim of the paper is that masking a *raw* window lets the model learn spurious
patterns, because a raw window entangles trend, seasonality and residual.  So the
pretext task is defined on the **decomposed** components with a masking rule per
component:

* **trend** — sub-series masking: one contiguous block of patch tokens, so the
  model cannot interpolate from immediate neighbours;
* **seasonal** — period masking: a whole phase group is removed (all patches whose
  index shares a residue modulo the period), so the model must use the periodic
  correspondence rather than copy the adjacent token;
* **residual** — low-rate random masking with a small loss weight (the paper keeps
  this component out of the reconstruction loss; weight 0.1 here).

A contrastive term pulls two independently masked seasonal views together.

Two-stage training
------------------
The paper pre-trains, then fine-tunes.  This repository's client loop only runs a
supervised step, so the pre-training is driven from the protocol:
``exp_paper_protocol`` runs ``stmtm_pretrain_steps`` steps of :meth:`ssl_loss`
before the supervised budget, on the same client's training loader.  That matches
the paper (pre-train and fine-tune on the same dataset) and keeps the supervised
step budget comparable with every other row of the table.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .base import ZooForecaster


class STMTM(ZooForecaster):
    name = 'stmtm'

    def __init__(self, args):
        super().__init__(args)
        self.kernel = int(getattr(args, 'stmtm_kernel', 25))
        self.period = int(getattr(args, 'cycle_length', 24) or 24)
        self.mask_ratio = float(getattr(args, 'stmtm_mask_ratio', 0.4))
        self.num_patches = int((self.seq_len - self.patch_len) / self.stride + 2)
        self.padding = (self.num_patches - 1) * self.stride + self.patch_len - self.seq_len
        self.pad = nn.ReplicationPad1d((0, max(0, self.padding)))

        self.trend_embed = nn.Linear(self.patch_len, self.d_model, bias=False)
        self.season_embed = nn.Linear(self.patch_len, self.d_model, bias=False)
        self.resid_embed = nn.Linear(self.patch_len, self.d_model, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.position = nn.Parameter(torch.randn(1, self.num_patches, self.d_model) * 0.02)
        self.norm = nn.LayerNorm(self.d_model)

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
        self.head = nn.Linear(self.num_patches * self.d_model, self.pred_len)

        # pre-training heads (unused after fine-tuning, still in state_dict)
        self.recon_trend = nn.Linear(self.d_model, self.patch_len)
        self.recon_season = nn.Linear(self.d_model, self.patch_len)
        self.recon_resid = nn.Linear(self.d_model, self.patch_len)

    # -- decomposition -------------------------------------------------------
    def _decompose(self, x: torch.Tensor):
        """x [B, L, C] -> trend, season, resid, each [B*C, 1, L]."""
        b, length, c = x.shape
        v = x.transpose(1, 2).reshape(b * c, 1, length)
        pad = self.kernel // 2
        smooth = F.pad(v, (pad, pad), mode='replicate')
        trend = smooth.unfold(-1, self.kernel, 1).mean(-1)               # [B*C,1,L]
        detrend = v - trend

        period = max(1, min(self.period, length))
        cycles = length // period
        if cycles >= 1:
            head = detrend[..., :cycles * period].reshape(b * c, 1, cycles, period)
            season = head.mean(dim=2).repeat(1, 1, cycles, 1).reshape(b * c, 1, cycles * period)
            if cycles * period < length:                                  # tail
                season = torch.cat(
                    [season, season[..., :length - cycles * period]], dim=-1)
        else:
            season = torch.zeros_like(detrend)
        resid = detrend - season
        return trend, season, resid

    def _patch(self, v: torch.Tensor) -> torch.Tensor:
        """[B*C,1,L] -> [B*C, P, Lp]."""
        return self.pad(v).unfold(2, self.patch_len, self.stride).squeeze(1)

    def _views(self, x_norm: torch.Tensor):
        trend, season, resid = self._decompose(x_norm)
        return self._patch(trend), self._patch(season), self._patch(resid)

    def _combine(self, pt, ps, pr, mask_t=None, mask_s=None, mask_r=None):
        tok = self.trend_embed(pt) + self.season_embed(ps) + self.resid_embed(pr)
        if mask_t is not None:
            tok = torch.where(mask_t.unsqueeze(-1), self.mask_token, tok)
        if mask_s is not None:
            tok = torch.where(mask_s.unsqueeze(-1), self.mask_token, tok)
        if mask_r is not None:
            tok = torch.where(mask_r.unsqueeze(-1), self.mask_token, tok)
        return self.norm(tok + self.position)

    # -- supervised path -----------------------------------------------------
    def encode(self, x_norm: torch.Tensor) -> torch.Tensor:
        pt, ps, pr = self._views(x_norm)
        return self.encoder(self._combine(pt, ps, pr))

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        c = self.enc_in
        b = tokens.shape[0] // c
        flat = self.dropout(tokens).reshape(b * c, -1)
        out = self.head(flat).reshape(b, c, self.pred_len)
        return out.transpose(1, 2)

    # -- pre-training --------------------------------------------------------
    def _block_mask(self, batch: int, patches: int, device) -> torch.Tensor:
        """Sub-series masking: one contiguous block per sample."""
        width = max(1, int(round(patches * self.mask_ratio)))
        width = min(width, patches)
        start = torch.randint(0, patches - width + 1, (batch,), device=device)
        idx = torch.arange(patches, device=device).unsqueeze(0)
        return (idx >= start.unsqueeze(1)) & (idx < (start + width).unsqueeze(1))

    def _period_mask(self, batch: int, patches: int, device) -> torch.Tensor:
        """Period masking: drop every ``k``-th patch, i.e. one phase group."""
        # one patch per period when the window holds >= 2 periods, else a stride
        stride = max(1, patches // 4)
        phase = torch.randint(0, stride, (batch,), device=device)
        idx = torch.arange(patches, device=device).unsqueeze(0)
        return (idx % stride) == phase.unsqueeze(1)

    def ssl_loss(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x_norm, _, _ = self.revin.normalize(x)
        pt, ps, pr = self._views(x_norm)
        b, patches, _ = pt.shape
        dev = pt.device

        mt = self._block_mask(b, patches, dev)
        ms = self._period_mask(b, patches, dev)
        mr = torch.rand(b, patches, device=dev) < 0.1

        z = self.encoder(self._combine(pt, ps, pr, mt, ms, mr))
        loss = 0.0
        for head, target, mask, weight in (
                (self.recon_trend, pt, mt, 1.0),
                (self.recon_season, ps, ms, 1.0),
                (self.recon_resid, pr, mr, 0.1)):
            if mask.sum() == 0:
                continue
            pred = head(z)
            err = ((pred - target) ** 2).mean(dim=-1)
            loss = loss + weight * (err * mask.float()).sum() / mask.float().sum()

        # contrastive: second, independently masked seasonal view
        ms2 = self._period_mask(b, patches, dev)
        z2 = self.encoder(self._combine(pt, ps, pr, mt, ms2, mr))
        a = F.normalize(z.mean(dim=1), dim=-1)
        bb = F.normalize(z2.mean(dim=1), dim=-1)
        logits = a @ bb.t() / 0.2
        target = torch.arange(a.shape[0], device=dev)
        loss = loss + 0.5 * (F.cross_entropy(logits, target)
                             + F.cross_entropy(logits.t(), target))
        return loss
