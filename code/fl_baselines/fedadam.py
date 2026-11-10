"""FedAdam (ICLR 2021, Reddi et al. "Adaptive Federated Optimization").

The server treats the client-average displacement as a pseudo-gradient and runs
Adam on it::

    Delta = w_global - mean_i(w_i^local)
    m = b1*m + (1-b1)*Delta ;  v = b2*v + (1-b2)*Delta^2
    w_global <- w_global - lr_server * m_hat / (sqrt(v_hat) + tau)

**Read the numbers with the round count in mind.**  Adaptive server optimisers
are designed for tens to hundreds of rounds; our budget parity fixes the sync
count at one aggregation (two local phases).  With a single server step the
bias-corrected Adam direction is essentially ``sign(Delta)`` of magnitude
``lr_server`` in every coordinate, so the reported MSE mostly measures what a
one-shot sign step does to a 81k-parameter model, not the method's converged
behaviour.  ``--fedadam_lr`` is therefore deliberately small and the row is
reported as a lower bound.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import DataLoader

from data_provider.federated_data import make_client_args

from .base import FedMethod
from .fl_client import BaselineClient, average_state_dicts


class FedAdam(FedMethod):
    name = 'fedadam'
    venue = 'ICLR 2021'
    short_desc = 'Server-side Adam on the averaged client displacement'

    def setup_clients(self, loaders: Dict[str, Dict[str, DataLoader]]) -> None:
        cargs = make_client_args(self.args)
        self.lr_s = float(getattr(self.args, 'fedadam_lr', 1e-4))
        self.b1 = float(getattr(self.args, 'fedadam_beta1', 0.9))
        self.b2 = float(getattr(self.args, 'fedadam_beta2', 0.99))
        self.tau = float(getattr(self.args, 'fedadam_tau', 1e-3))
        for cid, ld in loaders.items():
            self.clients[cid] = BaselineClient(
                cid, cargs, self.device, ld['train'], ld['val'], ld['test']
            )
        self.global_state = average_state_dicts([c.state_dict() for c in self.clients.values()])
        # NB: ``v.float()`` returns *self* when v is already float32, so
        # ``v.float().zero_()`` silently wiped ``self.global_state`` in place
        # (observed: every weight reset to 0, MoE mask 0, MSE 60).  zeros_like
        # allocates a new tensor unconditionally.
        self.m = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in self.global_state.items()}
        self.v = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in self.global_state.items()}
        self.server_steps = 0

    def local_update(self, client_id: str, epochs: int = 1) -> None:
        # No explicit broadcast: ``aggregate`` leaves every client on the server
        # state, and round 1 starts from client-local inits for every baseline
        # in this matrix (see scaffold.py for the same note).
        self.clients[client_id].local_train(epochs=epochs)

    def aggregate(self, round_idx: int) -> None:
        avg = average_state_dicts([c.state_dict() for c in self.clients.values()])
        pkeys = next(iter(self.clients.values())).parameter_keys()
        self.server_steps += 1
        t = self.server_steps
        new_global = {}
        for k, w in self.global_state.items():
            if k not in pkeys:
                # bool / int64 bookkeeping buffers (expert_mask, call counters):
                # take the client average verbatim, never an optimiser step
                new_global[k] = avg[k].float()
                continue
            delta = w.float() - avg[k].float()
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * delta
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (delta * delta)
            m_hat = self.m[k] / (1 - self.b1 ** t)
            v_hat = self.v[k] / (1 - self.b2 ** t)
            new_global[k] = w.float() - self.lr_s * m_hat / (v_hat.clamp_min(1e-12).sqrt() + self.tau)
        self.global_state = new_global
        for c in self.clients.values():
            c.load_state_dict(self.global_state)
            c.global_anchor = {k: v.clone() for k, v in self.global_state.items()}

    def predict(self, client_id, batch_x, batch_x_mark):
        return self.clients[client_id].predict(batch_x, batch_x_mark)

    def summary(self) -> dict:
        out = super().summary()
        out.update({'server_lr': self.lr_s, 'beta1': self.b1, 'beta2': self.b2,
                    'tau': self.tau, 'server_steps': self.server_steps})
        return out
