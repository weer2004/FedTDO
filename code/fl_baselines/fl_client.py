"""Lightweight Dynamic-TMoE client shared by all FL baselines (no FedTDO controller)."""

from __future__ import annotations

import copy
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from models.zoo import build_model


class BaselineClient:
    def __init__(
        self,
        client_id: str,
        args,
        device: torch.device,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
    ):
        self.client_id = str(client_id)
        self.args = args
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # ``--backbone dtmoe`` keeps the original Dynamic_TMoE (default, so every
        # number already on disk stays reproducible); anything else is a plain
        # forecaster from models/zoo/
        self.model = build_model(getattr(args, 'backbone', 'dtmoe'), args).to(device)
        # Baselines typically disable Dynamic-TMoE drift expansion to isolate FL effects
        if getattr(args, 'baseline_enable_drift', 0):
            self.model.enable_drift_detection(
                window_size=getattr(args, 'drift_window_size', 576),
                history_size=getattr(args, 'drift_history_size', 20),
                k_sigma=getattr(args, 'drift_k_sigma', 3.0),
            )
        self.criterion = nn.MSELoss()
        self.global_anchor: Optional[Dict[str, torch.Tensor]] = None  # for proximal / ALA

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    def load_state_dict(self, state: Dict[str, torch.Tensor], strict: bool = True):
        self.model.load_state_dict(
            {k: v.to(self.device) for k, v in state.items()},
            strict=strict,
        )

    def clone_model(self) -> nn.Module:
        return copy.deepcopy(self.model)

    def head_state_dict(self) -> Dict[str, torch.Tensor]:
        """Prediction-head params (TS analogue of classifier)."""
        out = {}
        for name, p in self.model.named_parameters():
            if name.startswith('projection_layer') or name.startswith('projection_layers'):
                out[name] = p.detach().cpu().clone()
        return out

    def encoder_state_dict(self) -> Dict[str, torch.Tensor]:
        return self.model.get_encoder_state_dict()

    def body_state_dict(self) -> Dict[str, torch.Tensor]:
        """Everything except prediction head."""
        head_keys = set(self.head_state_dict().keys())
        return {k: v for k, v in self.state_dict().items() if k not in head_keys}

    def parameter_keys(self) -> set:
        """Names that are actually learnable.

        ``state_dict()`` also carries bookkeeping buffers -- ``expert_mask``
        (bool), ``forward_call_count`` / ``total_stored`` (int64),
        ``drift_pool_enabled`` (bool).  Server-side optimisers must never touch
        them: running an Adam step on a bool mask that starts at 0 yields 0, the
        MoE then masks every expert and the next forward returns NaN.  Methods
        that build an explicit server update copy these keys from the client
        average instead (see ``fedadam.aggregate``).
        """
        return {name for name, _ in self.model.named_parameters()}

    def set_trainable(self, parts: str = 'all') -> None:
        """Freeze everything except ``head`` / ``body`` (FedPer, FedRep, Ditto)."""
        mode = str(parts).lower()
        for name, p in self.model.named_parameters():
            is_head = name.startswith('projection_layer')
            if mode == 'head':
                p.requires_grad = is_head
            elif mode == 'body':
                p.requires_grad = not is_head
            else:
                p.requires_grad = True

    def local_train(
        self,
        epochs: int = 1,
        *,
        lr: Optional[float] = None,
        proximal_mu: float = 0.0,
        param_lrs: Optional[Dict[str, float]] = None,
        distill_teacher: Optional[nn.Module] = None,
        distill_alpha: float = 0.0,
        max_steps: int = 0,
        grad_bias: Optional[Dict[str, torch.Tensor]] = None,
        optimizer_kind: str = 'adamw',
        mixup_alpha: float = 0.0,
        no_step_cap: bool = False,
    ) -> int:
        """Run up to ``max_steps`` gradient steps; return the number actually taken.

        ``grad_bias`` adds a constant vector to every parameter gradient after
        ``backward()``.  SCAFFOLD needs exactly this: its local objective is
        ``g_i - c_i + c``, and adding the (constant) control-variate difference to
        ``p.grad`` is equivalent to changing the objective rather than the
        optimiser state, so the AdamW moments stay comparable with every other
        baseline.  The return value feeds FedNova's ``tau_i``.
        """
        self.model.train()
        # Every baseline must honour the same gradient-step budget, otherwise a
        # FedAvg round that iterates a full epoch (≈1.1k steps × 8 clients)
        # would be compared against FedTDO's 1.6k steps per client.  Callers may
        # override per call; the CLI flag is the default.
        if no_step_cap:
            # Epoch-mode callers drive the budget one epoch at a time; falling
            # back to ``fl_local_max_steps`` here would silently overrun it
            # (a single epoch would be extended to the CLI step budget).
            max_steps = 0
        elif not max_steps:
            max_steps = int(getattr(self.args, 'fl_local_max_steps', 0) or 0)
        base_lr = float(lr if lr is not None else self.args.learning_rate)

        if param_lrs:
            groups = []
            named = dict(self.model.named_parameters())
            used = set()
            for pname, plr in param_lrs.items():
                if pname in named and named[pname].requires_grad:
                    groups.append({'params': [named[pname]], 'lr': plr})
                    used.add(pname)
            rest = [p for n, p in named.items() if n not in used and p.requires_grad]
            if rest:
                groups.append({'params': rest, 'lr': base_lr})
            optimizer = torch.optim.AdamW(groups)
        else:
            params = [p for p in self.model.parameters() if p.requires_grad]
            if str(optimizer_kind).lower() == 'sgd':
                optimizer = torch.optim.SGD(params, lr=base_lr)
            else:
                optimizer = torch.optim.AdamW(params, lr=base_lr)

        if distill_teacher is not None:
            distill_teacher.eval()

        step = 0
        stop = False
        # A ``max_steps`` budget is expressed in gradient steps, so it has to be
        # able to *extend* training past the caller's epoch count.  Previously
        # the outer loop was ``range(epochs)`` and ``max_steps`` could only cut
        # it short: on Traffic (381 batches/epoch) ``--fl_rounds 1`` produced 381
        # steps although the matched budget is 1600, while FedTDO arms received
        # the full 1600 and the baselines looked 3x undertrained.
        if max_steps:
            passes = -1                       # unbounded, bounded by max_steps
        else:
            passes = max(int(epochs), 1)
        epoch_index = 0
        while True:
            if stop:
                break
            if passes >= 0 and epoch_index >= passes:
                break
            epoch_index += 1
            for batch in self.train_loader:
                batch_x, batch_y, batch_x_mark, _ = batch
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # FedDA-style local augmentation: mixup against another sample of
                # the same client.  ``lam`` is folded to >= 0.5 so the anchor
                # sample always dominates and the target scale is preserved.
                if mixup_alpha > 0 and batch_x.shape[0] > 1:
                    lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                    lam = max(lam, 1.0 - lam)
                    perm = torch.randperm(batch_x.shape[0], device=self.device)
                    batch_x = lam * batch_x + (1.0 - lam) * batch_x[perm]
                    batch_y = lam * batch_y + (1.0 - lam) * batch_y[perm]

                optimizer.zero_grad()
                outputs = self.model(batch_x, batch_x_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                target = batch_y[:, -self.args.pred_len:, f_dim:]
                loss = self.criterion(outputs, target)

                if proximal_mu > 0 and self.global_anchor is not None:
                    prox = 0.0
                    for name, p in self.model.named_parameters():
                        if name in self.global_anchor:
                            prox = prox + ((p - self.global_anchor[name].to(self.device)) ** 2).sum()
                    loss = loss + 0.5 * proximal_mu * prox

                if distill_teacher is not None and distill_alpha > 0:
                    with torch.no_grad():
                        t_out = distill_teacher(batch_x, batch_x_mark)
                        t_out = t_out[:, -self.args.pred_len:, f_dim:]
                    loss = (1 - distill_alpha) * loss + distill_alpha * self.criterion(outputs, t_out)

                loss.backward()

                # SCAFFOLD control-variate correction: g <- g - c_i + c
                if grad_bias:
                    for pname, p in self.model.named_parameters():
                        state = grad_bias.get(pname)
                        if state is not None and p.grad is not None:
                            p.grad.add_(state.to(self.device))

                optimizer.step()

                step += 1
                if max_steps and step >= int(max_steps):
                    stop = True
                    break

        return step

    @torch.no_grad()
    def predict(self, batch_x: torch.Tensor, batch_x_mark: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        batch_x = batch_x.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        out = self.model(batch_x, batch_x_mark)
        f_dim = -1 if self.args.features == 'MS' else 0
        return out[:, -self.args.pred_len:, f_dim:]

    @torch.no_grad()
    def extract_features(self, max_batches: int = 8) -> torch.Tensor:
        """Mean patch-embedding features over local train data (for FDA / PKDA / KME)."""
        self.model.eval()
        feats = []
        for i, batch in enumerate(self.train_loader):
            if i >= max_batches:
                break
            batch_x, _, batch_x_mark, _ = batch
            batch_x = batch_x.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device) if batch_x_mark is not None else None
            enc_out, _, _, _, _, _ = self.model._encode_patches(batch_x, batch_x_mark)
            # enc_out: [B*N, P, D] → mean over patches → [B*N, D]
            feats.append(enc_out.mean(dim=1).detach().cpu())
        if not feats:
            return torch.zeros(1, self.args.d_model)
        return torch.cat(feats, dim=0)


def average_state_dicts(states: list) -> Dict[str, torch.Tensor]:
    if not states:
        raise ValueError('empty state list')
    keys = states[0].keys()
    avg = {}
    n = float(len(states))
    for k in keys:
        avg[k] = sum(s[k].float() for s in states) / n
    return avg


def weighted_average_state_dicts(
    states: list,
    weights: list,
) -> Dict[str, torch.Tensor]:
    assert len(states) == len(weights)
    w = torch.tensor(weights, dtype=torch.float32)
    w = w / (w.sum() + 1e-12)
    keys = states[0].keys()
    avg = {}
    for k in keys:
        acc = None
        for wi, s in zip(w.tolist(), states):
            t = s[k].float() * wi
            acc = t if acc is None else acc + t
        avg[k] = acc
    return avg
