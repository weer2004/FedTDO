"""FedProx (MLSys 2020) — proximal term against client drift.

Local objective:  min_w  F_i(w) + (mu/2) ||w - w_global||^2

The proximal term keeps each client from wandering far from the broadcast
model, which is the standard remedy for statistically heterogeneous clients.
``mu = 0`` degenerates to FedAvg, so ``--fedprox_mu`` is the only knob and a
``mu`` sweep is what a reviewer will ask for.
"""

from __future__ import annotations

from typing import Dict

from torch.utils.data import DataLoader

from data_provider.federated_data import make_client_args

from .base import FedMethod
from .fl_client import BaselineClient, average_state_dicts


class FedProx(FedMethod):
    name = 'fedprox'
    venue = 'MLSys 2020'
    short_desc = 'FedAvg + proximal term (mu/2)||w - w_global||^2 on every client'

    def setup_clients(self, loaders: Dict[str, Dict[str, DataLoader]]) -> None:
        cargs = make_client_args(self.args)
        self.mu = float(getattr(self.args, 'fedprox_mu', 0.01))
        for cid, ld in loaders.items():
            self.clients[cid] = BaselineClient(
                cid, cargs, self.device, ld['train'], ld['val'], ld['test']
            )
        self.global_state = average_state_dicts([c.state_dict() for c in self.clients.values()])
        for c in self.clients.values():
            c.global_anchor = {k: v.clone() for k, v in self.global_state.items()}

    def local_update(self, client_id: str, epochs: int = 1) -> None:
        self.clients[client_id].local_train(epochs=epochs, proximal_mu=self.mu)

    def aggregate(self, round_idx: int) -> None:
        self.global_state = average_state_dicts([c.state_dict() for c in self.clients.values()])
        for c in self.clients.values():
            c.load_state_dict(self.global_state)
            c.global_anchor = {k: v.clone() for k, v in self.global_state.items()}

    def predict(self, client_id, batch_x, batch_x_mark):
        return self.clients[client_id].predict(batch_x, batch_x_mark)

    def summary(self) -> dict:
        out = super().summary()
        out['mu'] = self.mu
        return out
