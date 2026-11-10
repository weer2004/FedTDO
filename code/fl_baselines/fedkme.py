"""
FedKME (ICML 2026) — Adaptive Personalized FL via Multi-task Averaging of
Kernel Mean Embeddings.

Core idea: estimate collaboration weights from RFF kernel mean embeddings of
local feature distributions, then run weighted FedAvg personalized to each client.
"""

from __future__ import annotations

from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from data_provider.federated_data import make_client_args

from .base import FedMethod
from .fl_client import BaselineClient, weighted_average_state_dicts


class FedKME(FedMethod):
    name = 'fedkme'
    venue = 'ICML 2026'
    short_desc = 'Learn collaboration weights via kernel mean embeddings (RFF)'

    def setup_clients(self, loaders: Dict[str, Dict[str, DataLoader]]) -> None:
        cargs = make_client_args(self.args)
        self.rff_dim = int(getattr(self.args, 'fedkme_rff_dim', 64))
        self.sigma = float(getattr(self.args, 'fedkme_sigma', 1.0))
        for cid, ld in loaders.items():
            self.clients[cid] = BaselineClient(
                cid, cargs, self.device, ld['train'], ld['val'], ld['test']
            )
        # Fixed RFF projection
        d = int(self.args.d_model)
        g = torch.Generator().manual_seed(int(getattr(self.args, 'seed', 2021)))
        self.W = torch.randn(d, self.rff_dim, generator=g) / max(self.sigma, 1e-6)
        self.b = 2 * torch.pi * torch.rand(self.rff_dim, generator=g)
        self.collab_weights: Dict[str, List[float]] = {}

    def _kme(self, client: BaselineClient) -> torch.Tensor:
        feats = client.extract_features(max_batches=6)  # [N, D]
        # RFF: sqrt(2/m) cos(Wx + b)
        proj = feats @ self.W + self.b
        phi = torch.cos(proj) * ((2.0 / self.rff_dim) ** 0.5)
        return phi.mean(dim=0)  # [m]

    def _estimate_weights(self):
        ids = list(self.clients.keys())
        emb = {cid: self._kme(self.clients[cid]) for cid in ids}
        for i, cid in enumerate(ids):
            sims = []
            for j, oid in enumerate(ids):
                # Multi-task averaging style: similarity of KMEs
                s = float(torch.dot(emb[cid], emb[oid]).item())
                sims.append(max(s, 0.0))
            w = torch.tensor(sims, dtype=torch.float32)
            # Softmax temperature to mix local vs global
            w = torch.softmax(w / 0.5, dim=0)
            self.collab_weights[cid] = w.tolist()

    def local_update(self, client_id: str, epochs: int = 1) -> None:
        self.clients[client_id].local_train(epochs=epochs)

    def aggregate(self, round_idx: int) -> None:
        self._estimate_weights()
        ids = list(self.clients.keys())
        states = [self.clients[cid].state_dict() for cid in ids]
        # Personalized model for each client via weighted average of all clients
        personalized = {}
        for cid in ids:
            personalized[cid] = weighted_average_state_dicts(states, self.collab_weights[cid])
        for cid in ids:
            self.clients[cid].load_state_dict(personalized[cid])

    def predict(self, client_id, batch_x, batch_x_mark):
        return self.clients[client_id].predict(batch_x, batch_x_mark)

    def summary(self) -> dict:
        s = super().summary()
        s['collab_weights'] = self.collab_weights
        return s
