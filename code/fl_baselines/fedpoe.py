"""
Fed-POE (NeurIPS 2024) — Personalized Federated Learning with Mixture of Models
for Adaptive Prediction and Model Fine-Tuning.

Most relevant baseline for FedTDO: non-stationary online prediction via adaptive
ensemble of a local model and historical federated checkpoints.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from data_provider.federated_data import make_client_args

from .base import FedMethod
from .fl_client import BaselineClient, average_state_dicts


class FedPOE(FedMethod):
    name = 'fedpoe'
    venue = 'NeurIPS 2024'
    short_desc = 'Online mixture of local + federated model checkpoints (non-stationary streams)'

    def setup_clients(self, loaders: Dict[str, Dict[str, DataLoader]]) -> None:
        cargs = make_client_args(self.args)
        self.max_bank = int(getattr(self.args, 'fedpoe_bank_size', 5))
        self.eta = float(getattr(self.args, 'fedpoe_eta', 0.2))  # hedge weight LR
        for cid, ld in loaders.items():
            self.clients[cid] = BaselineClient(
                cid, cargs, self.device, ld['train'], ld['val'], ld['test']
            )
        self.global_state = average_state_dicts([c.state_dict() for c in self.clients.values()])
        # Server model bank: list of historical global snapshots
        self.model_bank: List[Dict[str, torch.Tensor]] = [self.global_state]
        # Per-client ensemble weights over [local, fed_latest, bank...]
        self.ens_weights: Dict[str, torch.Tensor] = {}
        self._bank_models: Dict[str, List[nn.Module]] = {}

    def _ensure_weights(self, cid: str, n: int) -> torch.Tensor:
        if cid not in self.ens_weights or self.ens_weights[cid].numel() != n:
            self.ens_weights[cid] = torch.ones(n, device=self.device) / n
        return self.ens_weights[cid]

    def _rebuild_bank_modules(self, client: BaselineClient):
        """Instantiate fresh models + load state (avoid deepcopy of non-leaf tensors)."""
        from models.Dynamic_TMoE.model import Model

        mods = []
        for sd in self.model_bank:
            m = Model(client.args).to(self.device)
            # strip buffers that may be missing / non-persistent
            m.load_state_dict(
                {k: v.to(self.device) for k, v in sd.items()},
                strict=False,
            )
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
            mods.append(m)
        self._bank_models[client.client_id] = mods

    def local_update(self, client_id: str, epochs: int = 1) -> None:
        client = self.clients[client_id]
        # Local fine-tune from latest global
        client.load_state_dict(self.global_state)
        client.local_train(epochs=epochs)
        self._rebuild_bank_modules(client)

    def aggregate(self, round_idx: int) -> None:
        self.global_state = average_state_dicts([c.state_dict() for c in self.clients.values()])
        # Periodically store federated checkpoint
        store_every = int(getattr(self.args, 'fedpoe_store_every', 1))
        if (round_idx + 1) % store_every == 0:
            self.model_bank.append({k: v.clone() for k, v in self.global_state.items()})
            if len(self.model_bank) > self.max_bank:
                self.model_bank = self.model_bank[-self.max_bank:]
        for c in self.clients.values():
            self._rebuild_bank_modules(c)

    @torch.no_grad()
    def predict(self, client_id, batch_x, batch_x_mark):
        client = self.clients[client_id]
        # Candidates: local model + bank federated models
        preds = [client.predict(batch_x, batch_x_mark)]
        bank = self._bank_models.get(client_id, [])
        for m in bank:
            m.eval()
            x = batch_x.float().to(self.device)
            xm = batch_x_mark.float().to(self.device)
            out = m(x, xm)
            f_dim = -1 if client.args.features == 'MS' else 0
            preds.append(out[:, -client.args.pred_len:, f_dim:])

        w = self._ensure_weights(client_id, len(preds))
        w = torch.softmax(w, dim=0)
        stacked = torch.stack(preds, dim=0)  # [M, B, T, C]
        return (w.view(-1, 1, 1, 1) * stacked).sum(dim=0)

    def online_adapt(self, client_id, batch_x, batch_y, batch_x_mark, t, pred) -> Optional[dict]:
        """Hedge update of ensemble weights using revealed loss of each expert."""
        client = self.clients[client_id]
        f_dim = -1 if client.args.features == 'MS' else 0
        y = batch_y[:, -client.args.pred_len:, f_dim:].float().to(self.device)

        with torch.no_grad():
            experts = [client.predict(batch_x, batch_x_mark)]
            for m in self._bank_models.get(client_id, []):
                x = batch_x.float().to(self.device)
                xm = batch_x_mark.float().to(self.device)
                out = m(x, xm)[:, -client.args.pred_len:, f_dim:]
                experts.append(out)
            losses = []
            for e in experts:
                losses.append(float(torch.mean((e - y) ** 2).item()))

        w = self._ensure_weights(client_id, len(losses))
        loss_t = torch.tensor(losses, device=self.device)
        # multiplicative weights / hedge
        w = w * torch.exp(-self.eta * loss_t)
        w = w / (w.sum() + 1e-12)
        self.ens_weights[client_id] = w
        return {'expert_losses': losses, 'weights': w.detach().cpu().tolist()}
