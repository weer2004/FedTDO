"""FedGTP — spatio-temporal graph federated prediction (skeleton).

The original method puts every participant on a sensor graph and mixes a temporal
encoder with graph message passing, so a client's forecast depends on its
*neighbours'* representations, not only on a shared global model.

What is implemented here
------------------------
The graph half only.  Under the channel-as-client protocol each client holds one
sensor, so a graph over **clients** is exactly a graph over sensors, and the server
can build it from a cheap fingerprint each client uploads (the same seven target
statistics FedDA uses — no raw data leaves the client).  Correlation between
fingerprints gives the adjacency; aggregation then becomes

    w_i = (1 - beta) * w_global + beta * sum_j A_ij w_j / sum_j A_ij

i.e. a personalisation that follows the sensor topology instead of the uniform
average.  ``beta = 0`` reduces to FedAvg and ``beta = 1`` to pure neighbour
averaging, which makes the ablation free.

Deviations
----------
* the temporal model is the shared ``--backbone``; the paper's recurrent/graph
  blocks are **not** re-implemented, so this row measures the graph aggregation
  rule, not the architecture;
* adjacency comes from data fingerprints rather than a physical topology.  On
  ECL/Traffic the true sensor layout exists but is not in the CSVs shipped here;
  using correlation is the standard fallback and is labelled as such.

The row must therefore be cited as "FedGTP-style graph aggregation", never as
"FedGTP reproduced".
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_provider.federated_data import make_client_args

from .base import FedMethod
from .fedda import _fingerprint
from .fl_client import BaselineClient, average_state_dicts


class FedGTP(FedMethod):
    name = 'fedgtp'
    venue = '—'
    short_desc = ('Graph-regularised aggregation over a correlation adjacency '
                  'built from client fingerprints')

    def __init__(self, args, device):
        super().__init__(args, device)
        self.beta = float(getattr(args, 'fedgtp_beta', 0.5))
        self.desc: Dict[str, np.ndarray] = {}

    def setup_clients(self, loaders: Dict[str, Dict[str, DataLoader]]) -> None:
        cargs = make_client_args(self.args)
        for cid, ld in loaders.items():
            self.clients[cid] = BaselineClient(
                cid, cargs, self.device, ld['train'], ld['val'], ld['test'])
            self.desc[cid] = _fingerprint(ld['train'])

    def _adjacency(self) -> tuple:
        keys: List[str] = list(self.clients)
        x = np.stack([self.desc[c] for c in keys])
        x = (x - x.mean(0)) / (x.std(0) + 1e-6)
        # cosine similarity -> non-negative adjacency, self loops removed
        norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
        a = (x / norm) @ (x / norm).T
        a = np.clip(a, 0.0, None)
        np.fill_diagonal(a, 0.0)
        row = a.sum(1, keepdims=True)
        a = np.where(row > 1e-8, a / np.maximum(row, 1e-8), 0.0)
        return keys, a

    def aggregate(self, round_idx: int) -> None:
        keys, a = self._adjacency()
        states = {c: self.clients[c].state_dict() for c in keys}
        glob = average_state_dicts([states[c] for c in keys])
        for i, cid in enumerate(keys):
            mix = {k: (1.0 - self.beta) * glob[k] for k in glob}
            for j, other in enumerate(keys):
                if a[i, j] <= 0:
                    continue
                for k in glob:
                    mix[k] = mix[k] + self.beta * float(a[i, j]) * states[other][k]
            self.clients[cid].load_state_dict(mix)
        deg = int((a > 0).sum())
        print(f'[{self.name}] round {round_idx + 1}: beta={self.beta}, '
              f'{deg} directed edges over {len(keys)} clients')

    def predict(self, client_id, batch_x, batch_x_mark):
        return self.clients[client_id].predict(batch_x, batch_x_mark)
