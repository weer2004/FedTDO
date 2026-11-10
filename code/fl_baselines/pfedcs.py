"""
PFedCS (AAAI 2025) adapted to TS forecasting.

Core idea: measure similarity of client prediction heads (classifier analogue),
cluster collaborators per client, and aggregate heads with distance-constrained
weights; share body globally.
"""

from __future__ import annotations

from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from data_provider.federated_data import make_client_args

from .base import FedMethod
from .fl_client import BaselineClient, average_state_dicts, weighted_average_state_dicts


class PFedCS(FedMethod):
    name = 'pfedcs'
    venue = 'AAAI 2025'
    short_desc = 'Collaborate via prediction-head similarity (classifier-similarity PFL)'

    def setup_clients(self, loaders: Dict[str, Dict[str, DataLoader]]) -> None:
        cargs = make_client_args(self.args)
        self.top_m = int(getattr(self.args, 'pfedcs_top_m', 2))
        self.temp = float(getattr(self.args, 'pfedcs_temp', 1.0))
        for cid, ld in loaders.items():
            self.clients[cid] = BaselineClient(
                cid, cargs, self.device, ld['train'], ld['val'], ld['test']
            )
        self.global_body = None

    def _head_vec(self, client: BaselineClient) -> torch.Tensor:
        parts = [v.flatten().float() for v in client.head_state_dict().values()]
        if not parts:
            return torch.zeros(1)
        return torch.cat(parts, dim=0)

    def _pairwise_distances(self) -> Dict[str, Dict[str, float]]:
        ids = list(self.clients.keys())
        vecs = {cid: self._head_vec(self.clients[cid]) for cid in ids}
        dist = {i: {} for i in ids}
        for i in ids:
            for j in ids:
                if i == j:
                    dist[i][j] = 0.0
                else:
                    dist[i][j] = float(torch.norm(vecs[i] - vecs[j]).item())
        return dist

    def local_update(self, client_id: str, epochs: int = 1) -> None:
        client = self.clients[client_id]
        if self.global_body is not None:
            sd = client.state_dict()
            sd.update(self.global_body)
            client.load_state_dict(sd, strict=False)
        client.local_train(epochs=epochs)

    def aggregate(self, round_idx: int) -> None:
        # Global body average
        self.global_body = average_state_dicts([c.body_state_dict() for c in self.clients.values()])
        dist = self._pairwise_distances()
        ids = list(self.clients.keys())

        for cid in ids:
            # Select top-m nearest collaborators (excluding self, then add self)
            others = sorted(
                [(d, oid) for oid, d in dist[cid].items() if oid != cid],
                key=lambda x: x[0],
            )
            neigh = [cid] + [oid for _, oid in others[: self.top_m]]
            # Distance-constrained weights: closer → larger weight
            raw = []
            for oid in neigh:
                d = dist[cid][oid]
                raw.append(torch.exp(torch.tensor(-d / max(self.temp, 1e-6))))
            w = torch.stack(raw)
            w = (w / w.sum()).tolist()
            heads = [self.clients[oid].head_state_dict() for oid in neigh]
            collab_head = weighted_average_state_dicts(heads, w)

            sd = self.clients[cid].state_dict()
            sd.update(self.global_body)
            sd.update(collab_head)
            self.clients[cid].load_state_dict(sd, strict=False)

            # Short local head fine-tune
            for name, p in self.clients[cid].model.named_parameters():
                p.requires_grad = name.startswith('projection')
            self.clients[cid].local_train(epochs=1)
            for p in self.clients[cid].model.parameters():
                p.requires_grad = True

    def predict(self, client_id, batch_x, batch_x_mark):
        return self.clients[client_id].predict(batch_x, batch_x_mark)
