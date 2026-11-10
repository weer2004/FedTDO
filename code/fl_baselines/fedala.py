"""
FedALA (AAAI 2023) adapted to TS forecasting.

Core idea: before each local round, learn element-wise aggregation weights that
mix the downloaded global model with the previous local model on upper layers
(here: MoE layers + prediction head), while lower layers (patch embedding) take
the global parameters.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_provider.federated_data import make_client_args

from .base import FedMethod
from .fl_client import BaselineClient, average_state_dicts


class FedALA(FedMethod):
    name = 'fedala'
    venue = 'AAAI 2023'
    short_desc = 'Adaptive local aggregation of global/local models before training'

    def setup_clients(self, loaders: Dict[str, Dict[str, DataLoader]]) -> None:
        cargs = make_client_args(self.args)
        self.eta = float(getattr(self.args, 'fedala_eta', 1.0))
        self.ala_steps = int(getattr(self.args, 'fedala_steps', 5))
        self.rand_percent = float(getattr(self.args, 'fedala_rand_percent', 0.8))
        for cid, ld in loaders.items():
            self.clients[cid] = BaselineClient(
                cid, cargs, self.device, ld['train'], ld['val'], ld['test']
            )
        self.global_state = average_state_dicts([c.state_dict() for c in self.clients.values()])
        self._weights: Dict[str, Dict[str, torch.Tensor]] = {}

    def _upper_keys(self, state: Dict[str, torch.Tensor]) -> List[str]:
        # Keep patch embedding (lower) from global; adapt MoE + head.
        # Skip non-float buffers (e.g. bool masks) — they cannot be mixed.
        return [
            k for k, v in state.items()
            if not k.startswith('patch_embedding') and torch.is_floating_point(v)
        ]

    def _adaptive_local_init(self, client: BaselineClient):
        local = client.state_dict()
        global_sd = self.global_state
        upper = self._upper_keys(local)

        # Learn per-tensor weights on a random subset of local batches
        weights = self._weights.get(client.client_id, {})
        for k in upper:
            if k not in weights:
                weights[k] = torch.ones_like(local[k]) * 0.5

        # Learn per-tensor mix weights W by local loss probing (no deepcopy of trained graph)
        from models.Dynamic_TMoE.model import Model

        w_params = {
            k: weights[k].to(self.device).clone().detach().requires_grad_(True) for k in upper
        }
        criterion = nn.MSELoss()
        steps = 0
        for batch in client.train_loader:
            if steps >= self.ala_steps:
                break
            batch_x, batch_y, batch_x_mark, _ = batch
            B = batch_x.shape[0]
            keep = max(1, int(B * self.rand_percent))
            idx = torch.randperm(B)[:keep]
            batch_x = batch_x[idx].float().to(self.device)
            batch_y = batch_y[idx].float().to(self.device)
            batch_x_mark = batch_x_mark[idx].float().to(self.device)

            # Build mixed state: local + (global - local) * W
            mixed_sd = {}
            for k, v in local.items():
                if k in w_params:
                    w = torch.clamp(w_params[k], 0.0, 1.0).detach()
                    g = global_sd[k].to(self.device)
                    l = v.to(self.device)
                    mixed_sd[k] = l + (g - l) * w
                else:
                    mixed_sd[k] = global_sd[k].to(self.device)

            probe = Model(client.args).to(self.device)
            probe.load_state_dict(mixed_sd, strict=False)
            probe.eval()
            with torch.no_grad():
                out = probe(batch_x, batch_x_mark)
                f_dim = -1 if client.args.features == 'MS' else 0
                out = out[:, -client.args.pred_len:, f_dim:]
                tgt = batch_y[:, -client.args.pred_len:, f_dim:]
                loss_val = float(criterion(out, tgt).item())

            # Heuristic: if mixed loss is high, pull W toward local (0); else toward 0.5
            with torch.no_grad():
                target = 0.0 if loss_val > 1.0 else 0.5
                for w in w_params.values():
                    w.sub_(self.eta * 0.05 * (w - target))
                    w.clamp_(0.0, 1.0)
            steps += 1

        self._weights[client.client_id] = {
            k: v.detach().cpu().clamp(0, 1) for k, v in w_params.items()
        }

        # Apply final init to client model
        new_sd = {}
        for k, v in local.items():
            if k in upper:
                w = self._weights[client.client_id][k]
                g = global_sd[k]
                new_sd[k] = v + (g - v) * w
            else:
                new_sd[k] = global_sd[k]
        client.load_state_dict(new_sd)

    def local_update(self, client_id: str, epochs: int = 1) -> None:
        client = self.clients[client_id]
        self._adaptive_local_init(client)
        client.local_train(epochs=epochs)

    def aggregate(self, round_idx: int) -> None:
        self.global_state = average_state_dicts([c.state_dict() for c in self.clients.values()])

    def predict(self, client_id, batch_x, batch_x_mark):
        return self.clients[client_id].predict(batch_x, batch_x_mark)
