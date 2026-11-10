"""FedAvg (AISTATS 2017) — classic full-model federated averaging."""

from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import DataLoader

from data_provider.federated_data import make_client_args

from .base import FedMethod
from .fl_client import BaselineClient, average_state_dicts


class FedAvg(FedMethod):
    name = 'fedavg'
    venue = 'AISTATS 2017'
    short_desc = 'Classic FL baseline: iterative local SGD + server parameter averaging'

    def setup_clients(self, loaders: Dict[str, Dict[str, DataLoader]]) -> None:
        cargs = make_client_args(self.args)
        for cid, ld in loaders.items():
            self.clients[cid] = BaselineClient(
                cid, cargs, self.device, ld['train'], ld['val'], ld['test']
            )
        self.global_state = average_state_dicts([c.state_dict() for c in self.clients.values()])

    def aggregate(self, round_idx: int) -> None:
        self.global_state = average_state_dicts([c.state_dict() for c in self.clients.values()])
        for c in self.clients.values():
            c.load_state_dict(self.global_state)
            c.global_anchor = {k: v.clone() for k, v in self.global_state.items()}

    def predict(self, client_id, batch_x, batch_x_mark):
        return self.clients[client_id].predict(batch_x, batch_x_mark)


class FedAvgSGD(FedAvg):
    """FedAvg with plain SGD -- matched control for SGD-based SCAFFOLD.

    SCAFFOLD's control variate is only dimensionally correct under SGD, so its
    row cannot be compared against AdamW rows directly.  This variant supplies
    the missing reference point: (scaffold - fedavg_sgd) isolates the variate,
    (fedavg_sgd - fedavg) isolates the optimiser.
    """

    name = 'fedavg_sgd'
    venue = 'control'
    short_desc = 'FedAvg with plain SGD (optimiser-matched control for SCAFFOLD)'

    def local_update(self, client_id: str, epochs: int = 1) -> None:
        self.clients[client_id].local_train(epochs=epochs, optimizer_kind='sgd')
