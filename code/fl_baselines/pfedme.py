"""pFedMe (NeurIPS 2020) — personalised models via a Moreau-envelope objective.

Per client and round:

1. **inner**: ``K`` steps on ``F_i(theta) + (lambda/2)||theta - w||^2`` -- this is
   the same proximal objective Ditto uses, but here it plays the role of the
   *inner solver* of the Moreau envelope, not a separate model.
2. **outer**: one envelope-gradient step ``theta <- theta - eta * lambda * (theta - w)``,
   which is the exact gradient of the Moreau envelope ``F_i^lambda(w)``.
3. **server**: ``w <- mean_i theta_i`` (the global model is the average of the
   personalised models, not the average of ERM solutions).

``lambda`` is an absolute weight on a squared parameter distance, so its useful
range depends on the scale of the weights: the paper's 15 is calibrated for
convex problems where ``F`` dominates.  On a 81k-parameter transformer with MSE
around 0.3 the proximal term at ``lambda=15`` would freeze ``theta`` at ``w`` and
pFedMe would collapse onto FedAvg, so the default here is 1.0.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import DataLoader

from data_provider.federated_data import make_client_args

from .base import FedMethod
from .fl_client import BaselineClient, average_state_dicts


class pFedMe(FedMethod):
    name = 'pfedme'
    venue = 'NeurIPS 2020'
    short_desc = 'Personalised theta via Moreau envelope, global w = mean(theta)'

    def setup_clients(self, loaders: Dict[str, Dict[str, DataLoader]]) -> None:
        cargs = make_client_args(self.args)
        self.lam = float(getattr(self.args, 'pfedme_lambda', 1.0))
        self.eta = float(getattr(self.args, 'pfedme_eta', 0.1))
        for cid, ld in loaders.items():
            self.clients[cid] = BaselineClient(
                cid, cargs, self.device, ld['train'], ld['val'], ld['test']
            )
        self.global_state = average_state_dicts([c.state_dict() for c in self.clients.values()])
        for c in self.clients.values():
            c.load_state_dict(self.global_state)
            c.global_anchor = {k: v.clone() for k, v in self.global_state.items()}

    def local_update(self, client_id: str, epochs: int = 1) -> None:
        client = self.clients[client_id]
        anchor = {k: v.clone() for k, v in self.global_state.items()}
        client.global_anchor = anchor
        client.local_train(epochs=epochs, proximal_mu=self.lam)
        # envelope gradient step: theta -= eta * lambda * (theta - w)
        with torch.no_grad():
            for name, p in client.model.named_parameters():
                if name in anchor:
                    p.add_(self.eta * self.lam * (p.data - anchor[name].to(self.device)))

    def aggregate(self, round_idx: int) -> None:
        self.global_state = average_state_dicts([c.state_dict() for c in self.clients.values()])

    def predict(self, client_id, batch_x, batch_x_mark):
        return self.clients[client_id].predict(batch_x, batch_x_mark)

    def summary(self) -> dict:
        out = super().summary()
        out.update({'lambda': self.lam, 'eta': self.eta})
        return out
