"""Local-only control: no federation (isolates transfer / FL gains)."""

from __future__ import annotations

from typing import Dict

from torch.utils.data import DataLoader

from data_provider.federated_data import make_client_args

from .base import FedMethod
from .fl_client import BaselineClient


class LocalOnly(FedMethod):
    name = 'local'
    venue = 'control'
    short_desc = 'Independent local Dynamic-TMoE; no aggregation or transfer'

    def setup_clients(self, loaders: Dict[str, Dict[str, DataLoader]]) -> None:
        cargs = make_client_args(self.args)
        for cid, ld in loaders.items():
            self.clients[cid] = BaselineClient(
                cid, cargs, self.device, ld['train'], ld['val'], ld['test']
            )

    def train_offline(self) -> None:
        if self.step_mode() == 'epochs':
            # DTMoe-aligned: per-client lr decay on val-based early stopping.
            self.train_local_epochs()
            return
        epochs = int(getattr(self.args, 'fl_local_epochs', 1)) * int(getattr(self.args, 'fl_rounds', 5))
        max_steps = int(getattr(self.args, 'fl_local_max_steps', 0) or 0)
        budget = f'{max_steps} steps' if max_steps else f'{epochs} epochs'
        print(f'[{self.name}] local budget={budget}')
        for cid, c in self.clients.items():
            print(f'  training {cid} ...')
            c.local_train(epochs=epochs, max_steps=max_steps)

    def predict(self, client_id, batch_x, batch_x_mark):
        return self.clients[client_id].predict(batch_x, batch_x_mark)
