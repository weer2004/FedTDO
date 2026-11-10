"""
pFedFDA (NeurIPS 2024) adapted to TS forecasting.

Core idea: shared representation (encoder + MoE body) via FedAvg; personalize
the prediction head by adapting to local feature distribution (mean/covariance
shrinkage toward a global feature Gaussian).
"""

from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import DataLoader

from data_provider.federated_data import make_client_args

from .base import FedMethod
from .fl_client import BaselineClient, average_state_dicts


class PFedFDA(FedMethod):
    name = 'pfedfda'
    venue = 'NeurIPS 2024'
    short_desc = 'Personalized FL via feature-distribution adaptation of local heads'

    def setup_clients(self, loaders: Dict[str, Dict[str, DataLoader]]) -> None:
        cargs = make_client_args(self.args)
        self.shrink = float(getattr(self.args, 'pfedfda_shrink', 0.5))
        for cid, ld in loaders.items():
            self.clients[cid] = BaselineClient(
                cid, cargs, self.device, ld['train'], ld['val'], ld['test']
            )
        self.global_body = None
        self.global_feat_mean = None
        self.global_feat_std = None

    def _feat_stats(self, client: BaselineClient):
        feats = client.extract_features(max_batches=6)
        mean = feats.mean(dim=0)
        std = feats.std(dim=0) + 1e-6
        return mean, std

    def local_update(self, client_id: str, epochs: int = 1) -> None:
        client = self.clients[client_id]
        # Load shared body from global if available
        if self.global_body is not None:
            sd = client.state_dict()
            sd.update(self.global_body)
            client.load_state_dict(sd, strict=False)

        # Train full local model
        client.local_train(epochs=epochs)

        # Feature-distribution adaptation: shrink local head toward global using feature gap
        if self.global_feat_mean is not None:
            local_mean, local_std = self._feat_stats(client)
            # discrepancy scalar in [0,1]
            gap = torch.norm(local_mean - self.global_feat_mean) / (
                torch.norm(self.global_feat_mean) + 1e-6
            )
            alpha = float(torch.sigmoid(gap * 2).item())  # larger gap → more local head
            # Blend head with previous global head if stored
            if hasattr(self, 'global_head') and self.global_head:
                local_head = client.head_state_dict()
                blended = {}
                for k in local_head:
                    if k in self.global_head:
                        blended[k] = (1 - alpha) * self.global_head[k] + alpha * local_head[k]
                    else:
                        blended[k] = local_head[k]
                sd = client.state_dict()
                sd.update(blended)
                client.load_state_dict(sd)

    def aggregate(self, round_idx: int) -> None:
        # Average bodies (non-head); keep personalized heads
        bodies = [c.body_state_dict() for c in self.clients.values()]
        self.global_body = average_state_dicts(bodies)
        heads = [c.head_state_dict() for c in self.clients.values()]
        self.global_head = average_state_dicts(heads) if heads else {}

        means, stds = [], []
        for c in self.clients.values():
            # Sync body to clients
            sd = c.state_dict()
            sd.update(self.global_body)
            c.load_state_dict(sd, strict=False)
            m, s = self._feat_stats(c)
            means.append(m)
            stds.append(s)
        self.global_feat_mean = torch.stack(means).mean(dim=0)
        self.global_feat_std = torch.stack(stds).mean(dim=0)

        # Optional head fine-tune on local data after body sync
        ft = int(getattr(self.args, 'pfedfda_head_epochs', 1))
        if ft > 0:
            for c in self.clients.values():
                # freeze body
                for name, p in c.model.named_parameters():
                    p.requires_grad = name.startswith('projection')
                c.local_train(epochs=ft)
                for p in c.model.parameters():
                    p.requires_grad = True

    def predict(self, client_id, batch_x, batch_x_mark):
        return self.clients[client_id].predict(batch_x, batch_x_mark)
