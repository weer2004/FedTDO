"""LM-backed federated foundation-model backbones: FFTS and Time-FFM.

These are the two rows that previously could not be run for lack of checkpoint
weights.  ``gpt2`` (124M) is now in the local HF cache, so both are real runs.

Design of the shared part
-------------------------
Both papers keep the language model **frozen** and train a small amount around it;
the federated mechanism then averages the *trainable* part while the frozen LM is
identical everywhere.  That is exactly what ``--method fedavg`` does to a module
whose frozen weights are identical across clients, so no new aggregator is needed
-- the frozen parameters are averaged into themselves and stay put.

The frozen LM is stored in a plain ``dict`` attribute (``self._shared``) instead of
as a submodule, which keeps it out of ``state_dict()``:

* aggregating 124M parameters per client per round would copy ~4 GB and dwarf the
  actual update, and
* ``model.parameters()`` would hand the local optimiser a set of frozen tensors.

Gradients still reach the adapters because autograd flows *through* frozen
activations; only the parameter update is blocked.

FFTS (AAAI 2025)
----------------
Frozen GPT-2 blocks, called explicitly in a loop (not via forward hooks -- the LM
instance is shared by every client, and per-client hooks on a shared module would
cross-talk).  After each block a client-specific bottleneck adapter is added
residually, plus a learnable prefix prompt.  The averaged quantity is
"adapter + prompt + patch embedding + head", i.e. the FFTS personalisation unit.

Time-FFM (NeurIPS 2024)
-----------------------
Frozen GPT-2 blocks with **no** in-block adapters, multi-scale patching (two patch
lengths concatenated on the token axis) and trainable input/output projections --
the "LM-empowered, lightweight adaptation" configuration of the paper.

Documented deviations
----------------------
* both ignore ``--d_model`` (GPT-2 is 768 wide); parameter counts jump from ~55k to
  ~124M, so this row is a *capacity* comparison and the table must say so;
* neither reproduces the papers' pretraining corpus: the LM is used off-the-shelf
  with GPT-2's own weights, which is what both papers do at fine-tuning time;
* Time-FFM's text-prototype alignment is not implemented (it needs the LM head and
  a text corpus); only its frozen-LM + multi-scale + light-adaptation skeleton is.
"""

from __future__ import annotations

import torch
from torch import nn

from .base import ZooForecaster

_GPT_CACHE: dict = {}


def shared_gpt2() -> nn.Module:
    """One frozen GPT-2 instance for the whole process (clients share it)."""
    if 'model' not in _GPT_CACHE:
        try:
            from transformers import GPT2Model, GPT2Config
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "'transformers' is missing. FFTS / Time-FFM are LM-backed: they need "
                "the GPU environment, not the CPU grid interpreter.\n"
                "  python: D:/software/Anaconda3/envs/ma_fmarl/python.exe\n"
                "  (see scripts/xc_sweep.py::python_for -- it selects this "
                "automatically for --backbone ffts|timeffm)\n"
                "  weights: gpt2, already in ~/.cache/huggingface "
                "(downloaded via HF_ENDPOINT=https://hf-mirror.com)"
            ) from exc
        cfg = GPT2Config(n_layer=12, n_head=12, n_embd=768, n_positions=1024,
                         vocab_size=50257, resid_pdrop=0.1, embd_pdrop=0.1,
                         attn_pdrop=0.1)
        try:
            lm = GPT2Model.from_pretrained('gpt2', local_files_only=True)
        except Exception as exc:                                   # offline fallback
            print(f'[lm] gpt2 weights unavailable ({type(exc).__name__}), '
                  f'falling back to random init -- THIS IS NOT FFTS/Time-FFM')
            lm = GPT2Model(cfg)
        lm.eval()
        for p in lm.parameters():
            p.requires_grad_(False)
        _GPT_CACHE['model'] = lm
        _GPT_CACHE['width'] = 768
    return _GPT_CACHE['model']


class _Adapter(nn.Module):
    """Bottleneck residual adapter, added after a frozen block."""

    def __init__(self, width: int, ratio: int = 16):
        super().__init__()
        self.down = nn.Linear(width, width // ratio)
        self.up = nn.Linear(width // ratio, width)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.up(torch.nn.functional.gelu(self.down(h)))


class _LMForecaster(ZooForecaster):
    """Shared plumbing: patchify -> frozen GPT-2 blocks -> head."""

    def __init__(self, args, patch_lens):
        super().__init__(args)
        self.patch_lens = list(patch_lens)
        self.width = 768
        # ``joint`` (default): one sequence per *window*, patch read across all C
        # channels -> batch B instead of B*C.  ``ci``: channel-independent, one
        # sequence per channel, which is the convention used by the rest of the
        # zoo but costs 7x the LM compute at C=7 (the whole 18-cell grid went from
        # ~1 h to >7 h in CI mode on the 4060).
        self.channel_mode = str(getattr(args, 'lm_channel_mode', 'joint')).lower()
        gpt = shared_gpt2()
        # dict attribute -> not a submodule -> absent from state_dict()
        self._shared = {'gpt': gpt}

        self.embeds = nn.ModuleList()
        self.positions = nn.ParameterList()
        total_tokens = 0
        for pl in self.patch_lens:
            npatch = int((self.seq_len - pl) / self.stride + 2)
            fan_in = pl * (self.enc_in if self.channel_mode == 'joint' else 1)
            self.embeds.append(nn.Linear(fan_in, self.width, bias=False))
            self.positions.append(nn.Parameter(torch.randn(1, npatch, self.width) * 0.02))
            total_tokens += npatch
        self.total_tokens = total_tokens
        # joint mode predicts all C channels from one sequence, so the head is
        # C times wider; CI mode stays per-channel
        self.out_dim = self.pred_len * (self.enc_in if self.channel_mode == 'joint'
                                        else 1)
        self.norm = nn.LayerNorm(self.width)

    def _patchify(self, x_norm: torch.Tensor):
        b, _, c = x_norm.shape
        if self.channel_mode == 'joint':
            z = x_norm.transpose(1, 2)                              # [B, C, L]
            lead = b
        else:
            z = x_norm.transpose(1, 2).reshape(b * c, 1, self.seq_len)
            lead = b * c
        out = []
        for embed, pos, pl in zip(self.embeds, self.positions, self.patch_lens):
            npatch = pos.shape[1]
            padding = max(0, (npatch - 1) * self.stride + pl - self.seq_len)
            zz = nn.functional.pad(z, (0, padding), mode='replicate')
            zz = zz.unfold(-1, pl, self.stride)                     # [lead, C?, npatch, pl]
            zz = zz.reshape(lead, npatch, -1)                       # joint: pl*C
            out.append(embed(zz) + pos)
        return out

    def _run_blocks(self, h: torch.Tensor) -> torch.Tensor:
        gpt = self._shared['gpt']
        if next(gpt.parameters()).device != h.device:               # one-time move
            gpt.to(h.device)
        for block in gpt.h:
            h = block(h)[0]
        return gpt.ln_f(h)

    def encode(self, x_norm: torch.Tensor) -> torch.Tensor:
        parts = self._patchify(x_norm)
        # one LM pass per scale: the token counts differ, so they cannot share a
        # single sequence without re-indexing the positions
        reps = [self._run_blocks(p) for p in parts]
        return torch.cat(reps, dim=1)

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        c = self.enc_in
        if self.channel_mode == 'joint':
            b = tokens.shape[0]
            flat = self.norm(tokens).reshape(b, -1)                 # b x (P*D)
            out = self.head(flat).reshape(b, c, self.pred_len)
        else:
            b = tokens.shape[0] // c
            flat = self.norm(tokens).reshape(b * c, -1)             # (b*c) x (P*D)
            out = self.head(flat).reshape(b, c, self.pred_len)
        return out.transpose(1, 2)


class FFTS(_LMForecaster):
    """Frozen GPT-2 + residual bottleneck adapters after every block + prompt."""

    name = 'ffts'

    def __init__(self, args):
        super().__init__(args, [int(args.patch_len)])
        self.adapters = nn.ModuleList([_Adapter(self.width) for _ in range(12)])
        self.prompt = nn.Parameter(torch.randn(1, 4, self.width) * 0.02)
        self.head = nn.Linear(self.total_tokens * self.width, self.out_dim)

    def encode(self, x_norm: torch.Tensor) -> torch.Tensor:
        tokens = self._patchify(x_norm)[0]
        gpt = self._shared['gpt']
        if next(gpt.parameters()).device != tokens.device:
            gpt.to(tokens.device)
        h = torch.cat([self.prompt.expand(tokens.shape[0], -1, -1), tokens], dim=1)
        for block, adapter in zip(gpt.h, self.adapters):
            h = block(h)[0]
            h = adapter(h)
        h = gpt.ln_f(h)
        return h[:, self.prompt.shape[1]:]                          # drop the prompt


class TimeFFM(_LMForecaster):
    """Frozen GPT-2, no in-block adapters, multi-scale patching."""

    name = 'timeffm'

    def __init__(self, args):
        base = int(args.patch_len)
        super().__init__(args, [base, max(2, base // 2)])
        self.head = nn.Linear(self.total_tokens * self.width, self.out_dim)
