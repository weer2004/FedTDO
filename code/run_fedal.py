"""FeDaL-style federated *pretraining* of an LM-backed time-series model.

Reference: Chen, Long, Blumenstein, Jiang, "FeDaL: Federated Dataset Learning for
General Time Series Foundation Models", ICLR 2026, arXiv:2508.04045.

Why this file exists
--------------------
FFTS / Time-FFM as published are *supervised* federated fine-tuning: every client
minimises a forecasting loss and the adapters are averaged.  FeDaL argues that this
is the wrong objective for heterogeneous datasets -- the client update is a biased
estimate of the global gradient, so aggregation drifts -- and replaces it with a
**masked-reconstruction pretraining** loop plus explicit bias elimination.  The
user asked for the FFTS backbone to be trained the FeDaL way, so this module
implements FeDaL's three phases on top of the frozen GPT-2 backbone:

  local  : patch -> 75% random mask -> encode -> time-decompose the latent ->
           estimate a per-dataset bias b_i = mean(h_t)*g_t + mean(h_s)*g_s ->
           + b_i back -> reconstruct, with lambda*||b_i - b^g||^2
  server : FedAvg -> dynamic correction theta_hat = theta~ - s/beta ->
           refine on the aggregated core-sets -> convex fusion
           theta^g = alpha*theta_hat + (1-alpha)*theta_gt -> broadcast b^g
  then   : downstream supervised fine-tuning of the forecast head per dataset

Documented simplifications (do not cite as "FeDaL reproduced")
--------------------------------------------------------------
* **core-set**: the paper builds it by *gradient matching* -- optimising the
  synthetic core-set C so that grad f(C) matches grad f(x), with `create_graph`
  second-order steps -- then perturbs only the Fourier **amplitude** (phase is kept
  because it carries periodicity) and re-aligns in latent space.  Here the core-set
  is a set of **real windows** selected by k-centre diversity on patch features,
  followed by the same amplitude-only Fourier perturbation.  Gradient matching on a
  124M-parameter frozen backbone is O(K*N) second-order work per round and would
  dominate the whole run.
* the server-side state `s` is a running EMA of the (client mean - global) drift,
  not the paper's Eq. 6 recurrence (which is not reproducible from the text alone).
* pre-training budget is expressed in gradient steps so it can be compared with the
  other rows; the paper uses hundreds of rounds over 174 clients.
* **RevIN-style instance normalisation is switched on for this row only.**  DBE
  estimates the bias in the per-window standardised space (`b_i` is added back
  before the reconstruction head), so the whole pretrained stack lives there and
  ``forecast()`` *must* undo the shift.  The other rows in
  `PAPER_PROTOCOL_BACKBONES.md` do not use RevIN (the dataset scaler already
  standardises them).  This is a disclosed confound: it favours FeDaL on
  non-stationary levels (Exchange), and it is intrinsic to the method rather than
  a tuning choice.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_provider.multivariate_data import build_paper_clients  # noqa: E402
from exp.exp_paper_protocol import DATASETS, MAIN_DATASETS  # noqa: E402
from models.zoo.lm_backbones import shared_gpt2  # noqa: E402
from utils.metrics import metric  # noqa: E402
from utils.tools import force_cuda  # noqa: E402


def _inst_norm(x: torch.Tensor) -> torch.Tensor:
    m = x.mean(dim=1, keepdim=True)
    s = x.std(dim=1, unbiased=False, keepdim=True) + 1e-5
    return (x - m) / s


class FedaLModel(nn.Module):
    """Frozen GPT-2 blocks + trainable patch embedding / recon head / bias gates."""

    def __init__(self, seq_len: int, pred_len: int, enc_in: int,
                 patch_len: int = 16, stride: int = 8, dropout: float = 0.1):
        super().__init__()
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.patch_len, self.stride = patch_len, stride
        self.width = 768
        self.num_patches = int((seq_len - patch_len) / stride + 2)
        self.padding = (self.num_patches - 1) * stride + patch_len - seq_len

        self._shared = {'gpt': shared_gpt2()}
        self.patch_embed = nn.Linear(patch_len * enc_in, self.width, bias=False)
        self.position = nn.Parameter(torch.randn(1, self.num_patches, self.width) * 0.02)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.width))
        self.recon_head = nn.Linear(self.width, patch_len * enc_in)
        self.fc_head = nn.Linear(self.num_patches * self.width, pred_len * enc_in)
        # DBE gates
        self.gamma_trend = nn.Parameter(torch.ones(self.width))
        self.gamma_season = nn.Parameter(torch.ones(self.width))
        self.register_buffer('bias_ref', torch.zeros(self.width), persistent=False)
        self.drop = nn.Dropout(dropout)

    # -- shared plumbing -----------------------------------------------------
    def _patches(self, x_norm: torch.Tensor) -> torch.Tensor:
        z = x_norm.transpose(1, 2)                                  # [B, C, L]
        z = F.pad(z, (0, max(0, self.padding)), mode='replicate')
        z = z.unfold(-1, self.patch_len, self.stride)               # [B, C, P, pl]
        return z.permute(0, 2, 1, 3).reshape(z.shape[0], self.num_patches, -1)

    def _blocks(self, h: torch.Tensor) -> torch.Tensor:
        gpt = self._shared['gpt']
        if next(gpt.parameters()).device != h.device:
            gpt.to(h.device)
        for block in gpt.h:
            h = block(h)[0]
        return gpt.ln_f(h)

    # -- DBE: masked reconstruction with a per-dataset bias ------------------
    def dbe_loss(self, x: torch.Tensor, mask_ratio: float, lam: float):
        x_norm = _inst_norm(x)
        target = self._patches(x_norm)                              # [B, P, pl*C]
        b, p, _ = target.shape
        h = self.patch_embed(target) + self.position
        mask = torch.rand(b, p, device=x.device) < mask_ratio
        if mask.sum() == 0:
            mask[0, 0] = True
        h = torch.where(mask.unsqueeze(-1), self.mask_token, h)
        h = self._blocks(h)

        # time decomposition of the latent sequence (MA smoothing over patches)
        k = min(5, p if p % 2 == 1 else p - 1)
        if k >= 3:
            pad = k // 2
            smooth = F.pad(h.transpose(1, 2), (pad, pad), mode='replicate')
            trend = smooth.unfold(-1, k, 1).mean(-1).transpose(1, 2)
        else:
            trend = h
        season = h - trend
        bias = trend.mean(dim=1) * self.gamma_trend \
            + season.mean(dim=1) * self.gamma_season                # [B, D]

        recon = self.recon_head(h + bias.unsqueeze(1))
        m = mask.float()
        rec = (((recon - target) ** 2).mean(-1) * m).sum() / m.sum()
        reg = lam * ((bias.mean(0) - self.bias_ref) ** 2).mean()
        return rec + reg, bias.detach().mean(0)

    # -- downstream supervised path -----------------------------------------
    def forecast(self, x: torch.Tensor) -> torch.Tensor:
        """RevIN-style: normalise per window, predict, then **undo the shift**.

        The de-normalisation is not cosmetic.  ``dbe_loss`` trains the adapter to
        reconstruct *normalised* patches, so the whole pretrained stack lives in
        the per-window standardised space; returning that space directly and
        comparing it with the raw target makes every non-stationary dataset
        (Exchange above all) look catastrophically wrong, because the prediction
        carries no level information at all.
        """
        m = x.mean(dim=1, keepdim=True)
        s = x.std(dim=1, unbiased=False, keepdim=True) + 1e-5
        x_norm = (x - m) / s
        h = self._blocks(self.patch_embed(self._patches(x_norm)) + self.position)
        out = self.fc_head(self.drop(h).reshape(h.shape[0], -1))
        out = out.reshape(x.shape[0], self.pred_len, self.enc_in)
        return out * s + m

    def head_parameters(self) -> List[nn.Parameter]:
        return list(self.fc_head.parameters())


# --------------------------------------------------------------------------- #
# core-set: k-centre selection on patch features + amplitude-only Fourier noise
# --------------------------------------------------------------------------- #
@torch.no_grad()
def build_coreset(model: FedaLModel, arr: np.ndarray, k: int, eps: float,
                  device) -> torch.Tensor:
    """Pick ``k`` diverse real windows, then perturb only their FFT amplitude."""
    seq_len, pred_len = model.seq_len, model.pred_len
    n = len(arr) - seq_len - pred_len + 1
    if n <= 0:
        return torch.zeros(1, seq_len, arr.shape[1], device=device)
    idx = np.linspace(0, n - 1, min(n, 512)).astype(int)
    feats = np.stack([arr[i:i + seq_len] for i in idx])
    f = feats.reshape(len(idx), -1)
    f = (f - f.mean(0)) / (f.std(0) + 1e-6)

    chosen = [int(np.argmin(((f - f.mean(0)) ** 2).sum(1)))]        # near the centroid
    dist = np.min(((f - f[chosen[0]]) ** 2).sum(1, keepdims=True), axis=1)
    for _ in range(min(k, len(idx)) - 1):                            # k-centre greedy
        nxt = int(np.argmax(dist))
        chosen.append(nxt)
        dist = np.minimum(dist, ((f - f[nxt]) ** 2).sum(1))
    core = torch.from_numpy(feats[chosen]).float().to(device)        # [k, L, C]

    if eps > 0:                                                      # amplitude only
        spec = torch.fft.rfft(core, dim=1)
        amp = spec.abs() + eps * torch.randn_like(spec.real)
        core = torch.fft.irfft(torch.polar(amp, spec.angle()), n=core.shape[1], dim=1)
    return core


def main() -> None:
    p = argparse.ArgumentParser(description='FeDaL-style federated pretraining (LM backbone)')
    p.add_argument('--datasets', nargs='+', default=MAIN_DATASETS, choices=MAIN_DATASETS)
    p.add_argument('--seq_len', type=int, default=96)
    p.add_argument('--pred_len', type=int, default=96)
    p.add_argument('--paper_channels', type=int, default=7)
    p.add_argument('--pretrain_steps', type=int, default=1600,
                   help='masked-reconstruction steps per client (total)')
    p.add_argument('--finetune_steps', type=int, default=1600,
                   help='supervised steps per dataset after pretraining; the same '
                        '1600 the other rows get, so the comparison is "same '
                        'supervised budget + one shared pretraining pass"')
    p.add_argument('--rounds', type=int, default=2)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--mask_ratio', type=float, default=0.75)
    p.add_argument('--lam', type=float, default=0.1, help='weight of ||b_i - b^g||^2')
    p.add_argument('--alpha', type=float, default=0.7, help='stability/adaptability fusion')
    p.add_argument('--beta', type=float, default=1.0, help='dynamic-correction gain')
    p.add_argument('--coreset_k', type=int, default=8)
    p.add_argument('--coreset_eps', type=float, default=0.05)
    p.add_argument('--coreset_steps', type=int, default=20)
    p.add_argument('--seed', type=int, default=2021)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--tag', type=str, default='fedal')
    args = p.parse_args()

    from utils.env_banner import print_env_banner
    print_env_banner(tag=args.tag, feature='fedal')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = force_cuda(args)
    torch.backends.cuda.matmul.allow_tf32 = True

    specs = {k: DATASETS[k] for k in args.datasets}
    bundles = build_paper_clients(specs, channels=args.paper_channels,
                                  seq_len=args.seq_len, pred_len=args.pred_len,
                                  batch_size=args.batch_size, seed=args.seed)
    names = list(bundles)
    enc_in = bundles[names[0]]['n_channels']
    print(f'[fedal] clients={len(names)} enc_in={enc_in} device={device}')

    models = {n: FedaLModel(args.seq_len, args.pred_len, enc_in).to(device) for n in names}
    server_state = None            # GBE dynamic correction state s (EMA of drift)
    prev_global = None             # last broadcast global model, for the drift proxy
    bias_ref = torch.zeros(models[names[0]].width, device=device)

    # ---------------- phase 1+2: federated pretraining ----------------------
    steps_per_round = max(1, args.pretrain_steps // args.rounds)
    for rnd in range(args.rounds):
        states, biases = {}, []
        for n in names:
            model = models[n]
            model.train()
            model.bias_ref.copy_(bias_ref)
            opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad],
                                    lr=args.lr)
            loader = iter(bundles[n]['train'])
            last = None
            for _ in range(steps_per_round):
                try:
                    batch = next(loader)
                except StopIteration:
                    loader = iter(bundles[n]['train'])
                    batch = next(loader)
                x = batch[0].float().to(device)
                loss, bias = model.dbe_loss(x, args.mask_ratio, args.lam)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                last = float(loss.detach())
            biases.append(bias)
            states[n] = {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(f'[fedal] round {rnd + 1} {n}: recon {last:.4f}')

        # --- server: FedAvg
        keys = states[names[0]].keys()
        avg = {k: sum(states[n][k] for n in names) / len(names) for k in keys}

        # --- server: gradient-level dynamic correction
        # s tracks the *direction the aggregate is moving*: avg(r) - theta^g(r-1).
        # Subtracting an EMA of it damps the oscillation towards whichever client
        # dominates the current round, which is the effect FeDaL's Eq. 6 aims at.
        # (The paper's exact recurrence is not reproducible from the text -- see the
        # module docstring: this is a proxy, not a reimplementation.)
        if prev_global is None:
            drift = {k: torch.zeros_like(avg[k]) for k in keys}
        else:
            drift = {k: avg[k] - prev_global[k] for k in keys}
        server_state = drift if server_state is None else {
            k: 0.9 * server_state[k] + 0.1 * drift[k] for k in keys}
        corrected = {k: avg[k] - server_state[k] / args.beta for k in keys}

        # --- server: refine on the aggregated core-sets
        probe = FedaLModel(args.seq_len, args.pred_len, enc_in).to(device)
        probe.load_state_dict(corrected)
        coresets = [build_coreset(models[n], np.asarray(
            getattr(bundles[n]['train'], 'dataset', bundles[n]['train']).data_x,
            dtype=np.float32), args.coreset_k, args.coreset_eps, device) for n in names]
        opt = torch.optim.AdamW([q for q in probe.parameters() if q.requires_grad],
                                lr=args.lr)
        probe.train()
        for _ in range(args.coreset_steps):
            loss = 0.0
            for core in coresets:
                loss = loss + probe.dbe_loss(core, 0.15, 0.0)[0]
            loss = loss / max(1, len(coresets))
            opt.zero_grad()
            loss.backward()
            opt.step()
        refined = {k: v.detach().clone() for k, v in probe.state_dict().items()}

        # --- server: convex fusion, then broadcast
        fused = {k: args.alpha * corrected[k] + (1 - args.alpha) * refined[k] for k in keys}
        prev_global = {k: fused[k].detach().clone() for k in keys}
        for n in names:
            models[n].load_state_dict(fused)
        bias_ref = torch.stack(biases).mean(0)
        print(f'[fedal] round {rnd + 1} aggregated; |bias_ref|={bias_ref.norm():.4f}')

    # ---------------- phase 3: downstream fine-tuning + eval ----------------
    rows = []
    for n in names:
        model = models[n]
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
        loader = iter(bundles[n]['train'])
        for _ in range(args.finetune_steps):
            try:
                batch = next(loader)
            except StopIteration:
                loader = iter(bundles[n]['train'])
                batch = next(loader)
            x, y = batch[0].float().to(device), batch[1].float().to(device)
            out = model.forecast(x)
            loss = F.mse_loss(out, y[:, -args.pred_len:, :])
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in bundles[n]['test']:
                x, y = batch[0].float().to(device), batch[1].float().to(device)
                preds.append(model.forecast(x).cpu().numpy())
                trues.append(y[:, -args.pred_len:, :].cpu().numpy())
        pred = np.concatenate(preds)
        true = np.concatenate(trues)
        mae, mse, *_ = metric(pred, true)
        rows.append({'dataset': n, 'mse': float(mse), 'mae': float(mae)})
        print(f'[fedal] {n}: mse={mse:.4f} mae={mae:.4f}')

    outdir = Path('results_fedal') / (args.tag or
                                      f'fedal_pl{args.pred_len}_c{args.paper_channels}')
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / 'per_dataset.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['dataset', 'mse', 'mae'])
        w.writeheader()
        w.writerows(rows)
    core = [r['mse'] for r in rows if r['dataset'] != 'ILI']
    summary = {'method': 'fedal', 'core_mean': float(np.mean(core)),
               'per_dataset': rows,
               'protocol': 'dataset-level multivariate, FeDaL pretraining + per-dataset finetune'}
    with open(outdir / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=1)
    print(f"[fedal] core mean (no ILI) = {summary['core_mean']:.4f} -> {outdir}")


if __name__ == '__main__':
    main()
