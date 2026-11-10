"""Shared scaffolding for the backbone zoo."""

from __future__ import annotations

import torch
from torch import nn

from layers.RevIN import RevIN


class ZooForecaster(nn.Module):
    """Common contract: RevIN in, head out, plus the three helper calls."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.seq_len = int(args.seq_len)
        self.pred_len = int(args.pred_len)
        self.enc_in = int(getattr(args, 'enc_in', 1) or 1)
        self.d_model = int(args.d_model)
        self.patch_len = int(getattr(args, 'patch_len', 16))
        self.stride = int(getattr(args, 'stride', 8))
        self.revin = RevIN(self.enc_in, eps=1e-5, affine=True)
        self.last_tokens: torch.Tensor | None = None

    # -- to be implemented by subclasses -------------------------------------
    def encode(self, x_norm: torch.Tensor) -> torch.Tensor:
        """[B, L, C] -> tokens [B*C, P, D] (channel independent) or [B, C, D]."""
        raise NotImplementedError

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens -> [B, pred_len, C] in normalised space."""
        raise NotImplementedError

    # -- shared --------------------------------------------------------------
    def forward(self, x: torch.Tensor, x_mark: torch.Tensor | None = None) -> torch.Tensor:
        x = x.float()
        x_norm, mean, std = self.revin.normalize(x)
        tokens = self.encode(x_norm)
        self.last_tokens = tokens.detach()
        out = self.decode(tokens)
        return self.revin.denormalize(out, mean, std)

    def _encode_patches(self, x: torch.Tensor, x_mark: torch.Tensor | None = None):
        """Mirror of ``Model._encode_patches`` for the prototype baselines."""
        x = x.float()
        x_norm, _, _ = self.revin.normalize(x)
        tokens = self.encode(x_norm)
        return tokens, None, None, None, None, None

    def get_encoder_state_dict(self) -> dict:
        head = set(self.head_state_dict())
        return {k: v for k, v in self.state_dict().items() if k not in head}

    def head_state_dict(self) -> dict:
        return {k: v.detach().clone() for k, v in self.state_dict().items()
                if k.startswith('head')}

    # no-ops that Dynamic_TMoE exposes for the FedTDO controller
    def enable_drift_detection(self, *args, **kwargs):
        return None

    def disable_drift_detection(self):
        return None


class Seq2SeqForecaster(ZooForecaster):
    """Base for encoder-decoder backbones (Autoformer / FEDformer).

    TSLib feeds the decoder ``dec_in = cat(x_enc[:, -label_len:], zeros(pred))``:
    the *label* segment is real past data and the horizon segment is a zero
    placeholder, i.e. **no future ground truth leaks in**.  This class rebuilds
    exactly that tensor from ``x`` so the encoder-decoder models still satisfy the
    ``forward(x, x_mark)`` contract every other component relies on.
    """

    def __init__(self, args):
        super().__init__(args)
        self.label_len = int(getattr(args, 'label_len', 48))

    def _build_dec_in(self, x_norm: torch.Tensor) -> torch.Tensor:
        b, _, c = x_norm.shape
        label = x_norm[:, -self.label_len:, :]
        zeros = x_norm.new_zeros(b, self.pred_len, c)
        return torch.cat([label, zeros], dim=1)

    def encode(self, x_norm: torch.Tensor) -> torch.Tensor:
        return self.encoder(x_norm)

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        # Encoder-decoder models forecast through ``decode_from`` inside
        # ``forward``; there is no encoder-only head to call here.
        raise NotImplementedError(
            f'{type(self).__name__} is encoder-decoder; use forward()/decode_from()')

    def decode_from(self, enc_out: torch.Tensor, dec_in: torch.Tensor,
                    x_norm: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor | None = None) -> torch.Tensor:
        x = x.float()
        x_norm, mean, std = self.revin.normalize(x)
        enc_out = self.encode(x_norm)
        dec_out = self.decode_from(enc_out, self._build_dec_in(x_norm), x_norm)
        dec_out = dec_out[:, -self.pred_len:, :]
        return self.revin.denormalize(dec_out, mean, std)

    def _encode_patches(self, x: torch.Tensor, x_mark: torch.Tensor | None = None):
        x = x.float()
        x_norm, _, _ = self.revin.normalize(x)
        tokens = self.encoder(x_norm)
        return tokens, None, None, None, None, None

