"""Building blocks shared by the TSLib-lineage backbones.

Every block here is a port of the corresponding module in the Time-Series-Library
(``layers/AutoCorrelation.py``, ``layers/Autoformer_EncDec.py``,
``layers/FourierCorrelation.py``, ``layers/Conv_Blocks.py``) reduced to what the
four models below actually use.  The ports keep the maths identical and drop the
options that only matter at ``d_model = 512`` (progressive frequency mixing,
wavelet mode, ``output_attention``); every dropped option is named in the model
docstring so a reader knows exactly what was not implemented.

Shape convention: ``[B, T, C]`` for series tensors, i.e. channels last.  TSLib
uses the same and only transposes for ``nn.Conv1d`` / ``nn.AvgPool1d``.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# series decomposition                                                        #
# --------------------------------------------------------------------------- #
class MovingAvg(nn.Module):
    """Moving average with edge replication, length preserving."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.avg = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, T, C]
        pad_l = (self.kernel_size - 1) // 2
        pad_r = self.kernel_size - 1 - pad_l
        front = x[:, :1, :].repeat(1, pad_l, 1)
        end = x[:, -1:, :].repeat(1, pad_r, 1)
        x = torch.cat([front, x, end], dim=1)
        return self.avg(x.transpose(1, 2)).transpose(1, 2)


class SeriesDecomp(nn.Module):
    """``x = seasonal + trend`` (Autoformer Eq. 2)."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size)

    def forward(self, x: torch.Tensor):
        trend = self.moving_avg(x)
        return x - trend, trend


class MOEDecomp(nn.Module):
    """Mixture-of-experts series decomposition (FEDformer).

    ``kernel_set`` gives experts at several moving-average widths; the gate is a
    softmax over the average-pooled input, so the decomposition adapts per
    sample instead of using one fixed window.
    """

    def __init__(self, kernel_set=(1, 2, 4, 8, 16), dropout: float = 0.1):
        super().__init__()
        self.kernel_set = list(kernel_set)
        self.avg_pools = nn.ModuleList(
            [nn.AvgPool1d(k, stride=1) for k in self.kernel_set])
        self.gate = nn.Sequential(
            nn.Linear(len(self.kernel_set), len(self.kernel_set)), nn.Softmax(dim=-1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # x: [B, T, C]
        res = []
        for k, pool in zip(self.kernel_set, self.avg_pools):
            if k == 1:
                res.append(x)
                continue
            pad_l = (k - 1) // 2
            pad_r = k - 1 - pad_l
            xp = torch.cat([x[:, :1, :].repeat(1, pad_l, 1),
                            x,
                            x[:, -1:, :].repeat(1, pad_r, 1)], dim=1)
            res.append(pool(xp.transpose(1, 2)).transpose(1, 2))
        stack = torch.stack(res, dim=-1)                      # [B, T, C, K]
        gate_in = torch.stack([r.mean(dim=(1, 2)) for r in res], dim=-1)  # [B, K]
        weights = self.gate(gate_in)                          # [B, K]
        trend = (stack * weights[:, None, None, :]).sum(-1)
        return self.dropout(x - trend), trend


def align_cross(keys: torch.Tensor, values: torch.Tensor, target_len: int):
    """Pad (with zeros) or truncate ``keys``/``values`` to ``target_len``.

    Both Autoformer's auto-correlation and FEDformer's Fourier attention need the
    key/value sequence and the query sequence to have the same length.  The
    decoder query is ``label_len + pred_len`` long while the encoder memory is
    ``seq_len`` long, so TSLib pads the shorter side with zeros (the same trick
    Informer's ProbAttention uses).  FFT-based scoring makes this harmless: the
    padded region contributes no energy.
    """
    s = keys.shape[1]
    if target_len == s:
        return keys, values
    if target_len < s:
        return keys[:, :target_len], values[:, :target_len]
    pad_k = keys.new_zeros(keys.shape[0], target_len - s, keys.shape[2])
    pad_v = values.new_zeros(values.shape[0], target_len - s, values.shape[2])
    return torch.cat([keys, pad_k], 1), torch.cat([values, pad_v], 1)


# --------------------------------------------------------------------------- #
# Autoformer: auto-correlation                                              #
# --------------------------------------------------------------------------- #
class AutoCorrelation(nn.Module):
    """Auto-Correlation mechanism (Autoformer, NeurIPS 2021, Sec. 3.2).

    Period-based delays are found with an FFT on the *query/key* pairs, the
    ``top_k`` delays are kept, and the value is aggregated with the
    auto-correlation scores as weights.
    """

    def __init__(self, factor: float = 1.0, attention_dropout: float = 0.1):
        super().__init__()
        self.factor = float(factor)
        self.dropout = nn.Dropout(attention_dropout)

    def _time_delay_agg(self, values: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        """values [B, H, E, L], corr [B, H, E, L] -> [B, H, E, L].

        TSLib keeps one delay per (head, head-channel) pair, i.e. the top-k is
        taken along the lag axis of each head channel separately; that is what
        the released ``time_delay_agg_training`` does, so we keep it.
        """
        b, h, c, length = values.shape
        top_k = min(max(1, int(self.factor * math.log(length))), length)
        mean = torch.mean(corr, dim=-1, keepdim=True)
        std = torch.std(corr, dim=-1, keepdim=True)
        corr = (corr - mean) / (std + 1e-4)
        weights, delays = torch.topk(corr, top_k, dim=-1)
        weights = torch.softmax(weights, dim=-1)

        # ``delays`` differs per (batch, head, channel), so ``torch.roll`` (which
        # wants a single int shift) cannot be used; gather a modular index instead.
        base = torch.arange(length, device=values.device).view(1, 1, 1, length)
        out = values.new_zeros(b, h, c, length)
        for k in range(top_k):
            index = (base - delays[..., k].unsqueeze(-1)) % length
            index = index.expand(b, h, c, length)
            delayed = values.gather(-1, index)
            out = out + delayed * weights[..., k].unsqueeze(-1)
        return out

    def forward(self, queries: torch.Tensor, keys: torch.Tensor,
                values: torch.Tensor) -> torch.Tensor:
        # all: [B, H, C, L]
        b, h, c, length = queries.shape
        q_fft = torch.fft.rfft(queries, dim=-1)
        k_fft = torch.fft.rfft(keys, dim=-1)
        scale = 1.0 / math.sqrt(c)
        corr = torch.fft.irfft(q_fft * torch.conj(k_fft), n=length, dim=-1) * scale
        corr = self.dropout(torch.softmax(corr, dim=-1))
        return self._time_delay_agg(values, corr)


class AutoCorrelationLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, factor: float = 1.0,
                 attention_dropout: float = 0.1):
        super().__init__()
        self.inner = AutoCorrelation(factor, attention_dropout)
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, residual=None):
        b, t, _ = queries.shape
        h = self.n_heads
        c = queries.shape[-1] // h
        keys, values = align_cross(keys, values, t)

        def split(x):
            bb, tt, _ = x.shape
            return x.reshape(bb, tt, h, c).permute(0, 2, 3, 1)

        q = split(self.query_projection(queries))
        k = split(self.key_projection(keys))
        v = split(self.value_projection(values))
        out = self.inner(q, k, v)                       # [B, H, C, L]
        out = out.permute(0, 3, 1, 2).reshape(b, t, h * c)
        out = self.out_projection(out)
        # residual is added *after* the output projection, as in TSLib
        return out if residual is None else out + residual


# --------------------------------------------------------------------------- #
# FEDformer: frequency enhanced block (Fourier)                              #
# --------------------------------------------------------------------------- #
class FourierBlock(nn.Module):
    """FEB-f / FDA-f: attention in the truncated Fourier domain.

    Only the lowest ``modes`` frequencies are kept (``mode_select='low'``), which
    is FEDformer's default and the variant that wins on long-horizon benchmarks.
    """

    def __init__(self, d_model: int, n_heads: int, seq_len: int,
                 modes: int = 32, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.modes = max(1, min(int(modes), seq_len // 2 + 1))
        self.scale = 1.0 / math.sqrt(self.head_dim)
        # weight mixes head_dim -> head_dim in the frequency domain, per head,
        # plus one bias for the value spectrum (TSLib's ``weights[3]``)
        self.weights = nn.Parameter(
            torch.randn(4, n_heads, self.head_dim, self.head_dim) * 0.02)
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, residual=None):
        b, t, _ = queries.shape
        h, d = self.n_heads, self.head_dim
        keys, values = align_cross(keys, values, t)

        def split(x):
            bb, tt, _ = x.shape
            return x.reshape(bb, tt, h, d).transpose(1, 2)     # [B, H, T, d]

        q = torch.fft.rfft(split(self.query_projection(queries)), dim=-2)
        k = torch.fft.rfft(split(self.key_projection(keys)), dim=-2)
        v = torch.fft.rfft(split(self.value_projection(values)), dim=-2)

        q = q[:, :, :self.modes]                              # [B, H, m, d]
        k = k[:, :, :self.modes]
        v = v[:, :, :self.modes]

        # per-head complex mixing: [B,H,m,d] x [H,d,d] -> [B,H,m,d]
        q = torch.einsum('bhmx,hxy->bhmy', q, self.weights[0].to(q.dtype))
        k = torch.einsum('bhmx,hxy->bhmy', k, self.weights[1].to(k.dtype))
        v = torch.einsum('bhmx,hxy->bhmy', v, self.weights[2].to(v.dtype)) \
            + self.weights[3].to(v.dtype).mean(-1).view(1, h, 1, d)
        attn = torch.einsum('bhmx,bhmx->bhm', q, torch.conj(k)).real * self.scale
        attn = self.dropout(torch.softmax(attn, dim=-1))       # [B, H, m]
        out = torch.einsum('bhm,bhmd->bhmd', attn.to(v.dtype), v)

        # zero-pad the truncated spectrum back to rfft length, then inverse
        pad = torch.zeros(b, h, t // 2 + 1 - out.shape[2], d,
                          dtype=out.dtype, device=out.device)
        out = torch.cat([out, pad], dim=2)
        out = torch.fft.irfft(out, n=t, dim=-2)               # [B, H, T, d]
        out = out.transpose(1, 2).reshape(b, t, h * d)
        out = self.out_projection(out)
        return out if residual is None else out + residual


# --------------------------------------------------------------------------- #
# TimesNet: 2D inception over (period, phase)                               #
# --------------------------------------------------------------------------- #
class InceptionBlockV1(nn.Module):
    """Multi-scale 2D conv with softmax-weighted kernel fusion (TimesNet)."""

    def __init__(self, in_channels: int, out_channels: int, num_kernels: int = 6):
        super().__init__()
        self.num_kernels = num_kernels
        self.kernels = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1,
                      padding=i, bias=False)
            for i in range(num_kernels)])
        self.weights = nn.Conv2d(in_channels, num_kernels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = torch.stack([k(x) for k in self.kernels], dim=-1)   # [B, C, H, W, K]
        w = torch.softmax(self.weights(x), dim=1)                 # [B, K, H, W]
        res = res.permute(0, 1, 4, 2, 3)                          # [B, C, K, H, W]
        return torch.sum(res * w.unsqueeze(1), dim=2)


class TimesBlock(nn.Module):
    """FFT period detection -> 2D reshape -> inception -> amplitude-weighted sum."""

    def __init__(self, seq_len: int, d_model: int, d_ff: int, top_k: int = 3,
                 num_kernels: int = 6, dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.k = max(1, int(top_k))
        self.conv = nn.Sequential(
            InceptionBlockV1(d_model, d_ff, num_kernels),
            nn.GELU(),
            InceptionBlockV1(d_ff, d_model, num_kernels),
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, T, D]
        b, t, d = x.shape
        fft = torch.fft.rfft(x, dim=1)
        amp = fft.abs().mean(dim=0).mean(dim=-1)          # [F]
        amp[0] = 0                                        # drop DC
        top_k = min(self.k, max(1, amp.shape[0] - 1))
        _, top_idx = torch.topk(amp, top_k)
        top_idx = top_idx.detach().cpu().numpy()
        periods = [max(1, t // int(i)) if int(i) > 0 else 1 for i in top_idx]

        res = []
        for i, period in enumerate(periods):
            if period < 2 or period > t:
                continue
            pad = (period - t % period) % period
            xp = F.pad(x, (0, 0, 0, pad))                  # pad time
            length = xp.shape[1] // period
            xp = xp.reshape(b, length, period, d).permute(0, 3, 1, 2)
            z = self.conv(xp)                              # [B, D, length, period]
            z = z.permute(0, 2, 3, 1).reshape(b, length * period, d)[:, :t]
            res.append((z, amp[int(top_idx[i])]))
        if not res:
            return x
        weights = torch.softmax(torch.stack([w for _, w in res]), dim=0)
        out = sum(z * weights[i].to(z.dtype) for i, (z, _) in enumerate(res))
        return self.dropout(self.layer_norm(x + out))


# --------------------------------------------------------------------------- #
# Activation helper                                                          #
# --------------------------------------------------------------------------- #
def activation(name: str = 'gelu') -> nn.Module:
    table = {'gelu': nn.GELU(), 'relu': nn.ReLU(), 'silu': nn.SiLU()}
    return table.get(str(name).lower(), nn.GELU())
