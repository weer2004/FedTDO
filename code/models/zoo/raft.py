"""RAFT (ICML 2025) — retrieval-augmented time series forecasting.

Reference: Han, Lee, Cha, Arik, Yoon, "Retrieval Augmented Time Series
Forecasting", ICML 2025 (PMLR 267:21774-21797), arXiv:2505.04163.

The method augments the forecast with *external* information: for a query window
it retrieves the ``k`` most similar windows from the **training split** and feeds
their observed future values to the predictor alongside the input window.

Two consequences for this codebase:

* the retrieval bank has to be built from the training split, and the model never
  sees the targets in ``forward`` -- so the bank cannot be accumulated inside the
  model.  ``exp_paper_protocol`` calls :meth:`set_retrieval_bank` once per client
  with that client's train windows and their futures.
* the bank is stored in **non-persistent** buffers, so it never enters
  ``state_dict()`` and therefore cannot leak into FedAvg / personalisation
  aggregation (the same trap that made SCAFFOLD diverge).

Faithfulness notes
------------------
* retrieval distance is plain Euclidean on the **instance-normalised** window
  (per-window mean/std, no learnable affine).  Using the module's RevIN affine
  would make the metric drift during training and the bank would go stale.
* the retrieved futures are not concatenated onto the time axis (they have no
  temporal relation to the query window).  Each retrieved window contributes one
  ``Linear(k * pred_len -> d_model)`` embedding per channel, and the predictor
  fuses it with the encoder summary.  This is the "use the future values of the
  retrieved candidates alongside the inputs" clause of the paper; the exact
  fusion is not specified there, so it is documented as a choice.
* ``topk = 0`` disables retrieval and reduces the model to the PatchTST encoder,
  which is used as the ablation row.
"""

from __future__ import annotations

import torch
from torch import nn

from .base import ZooForecaster


class RAFT(ZooForecaster):
    name = 'raft'

    def __init__(self, args):
        super().__init__(args)
        self.topk = int(getattr(args, 'raft_topk', 3))
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
            activation='gelu', batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=int(getattr(args, 'e_layers', 2)))
        self.dropout = nn.Dropout(float(getattr(args, 'dropout', 0.1)))
        self.summary = nn.Linear(self.num_patches * self.d_model, self.d_model)

        # retrieval branch: k * pred_len -> d_model, per channel
        self.retrieval_embed = nn.Linear(max(1, self.topk) * self.pred_len,
                                         self.d_model)
        self.head = nn.Linear(2 * self.d_model, self.pred_len)

        # non-persistent -> absent from state_dict(), safe to aggregate
        self.register_buffer('_bank_keys', torch.zeros(0), persistent=False)
        self.register_buffer('_bank_vals', torch.zeros(0), persistent=False)
        self._bank_ready = False
        self._raw_x: torch.Tensor | None = None

    # -- retrieval bank ------------------------------------------------------
    @torch.no_grad()
    def set_retrieval_bank(self, keys: torch.Tensor, vals: torch.Tensor) -> None:
        """``keys`` [N, C, L] raw train windows, ``vals`` [N, pred_len, C] futures."""
        if self.topk <= 0 or keys.numel() == 0:
            self._bank_ready = False
            return
        device = next(self.parameters()).device
        self._bank_keys = keys.float().to(device)
        self._bank_vals = vals.float().to(device)
        self._bank_ready = True

    def clear_retrieval_bank(self) -> None:
        self._bank_keys = torch.zeros(0, device=next(self.parameters()).device)
        self._bank_vals = torch.zeros(0, device=next(self.parameters()).device)
        self._bank_ready = False

    @staticmethod
    def _inst_norm(t: torch.Tensor) -> torch.Tensor:
        """Per-window instance norm on the last axis, no affine (stable metric)."""
        m = t.mean(dim=-1, keepdim=True)
        s = t.std(dim=-1, unbiased=False, keepdim=True) + 1e-5
        return (t - m) / s

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor | None = None) -> torch.Tensor:
        # keep the *raw* window: retrieval runs in normalized space but the bank
        # is stored raw, and RevIN's affine drifts during training
        self._raw_x = x.detach().float()
        return super().forward(x, x_mark)

    def _retrieve(self, query_raw: torch.Tensor):
        """query_raw [B, C, L] -> [B, k, pred_len, C]."""
        k = self.topk
        q = self._inst_norm(query_raw).reshape(query_raw.shape[0], -1)   # [B, C*L]
        bank = self._inst_norm(self._bank_keys)
        b, cl = q.shape
        n = bank.shape[0]
        flat = bank.reshape(n, cl)
        d = (q * q).sum(1, keepdim=True) - 2.0 * q @ flat.t() \
            + (flat * flat).sum(1).unsqueeze(0)
        # drop the self-match so training never reads its own target back
        d = d.masked_fill(d < 1e-6, float('inf'))
        idx = d.topk(min(k, n), dim=1, largest=False).indices        # [B, k]
        vals = self._bank_vals[idx]                                   # [B, k, H, C]
        if vals.shape[1] < k:                                         # tiny bank
            pad = vals.new_zeros(b, k - vals.shape[1], vals.shape[2], vals.shape[3])
            vals = torch.cat([vals, pad], 1)
        return vals

    def encode(self, x_norm: torch.Tensor) -> torch.Tensor:
        b, _, c = x_norm.shape
        z = x_norm.transpose(1, 2).reshape(b * c, 1, self.seq_len)
        z = self.pad(z).unfold(2, self.patch_len, self.stride).squeeze(1)
        z = self.value_embedding(z) + self.position
        return self.encoder(z)

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        c = self.enc_in
        b = tokens.shape[0] // c
        summary = self.summary(self.dropout(tokens).reshape(b * c, -1))
        summary = summary.reshape(b, c, self.d_model)

        if self._bank_ready and self._raw_x is not None and self._raw_x.shape[0] == b:
            vals = self._retrieve(self._raw_x)                        # [B,k,H,C]
            ret = vals.permute(0, 3, 1, 2).reshape(b, c, self.topk * self.pred_len)
            ret = self.retrieval_embed(ret)                           # [B,C,D]
        else:
            ret = summary.new_zeros(summary.shape)
        out = self.head(torch.cat([summary, ret], dim=-1))            # [B,C,H]
        return out.transpose(1, 2)
