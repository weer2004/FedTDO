"""FedPer (arXiv 2019) — shared representation, personalised prediction head.

Lower layers (patch embedding + MoE stack) are federated; the linear
projection head never leaves the client.  Only the shared layers are uploaded
and averaged, and only they are overwritten by the broadcast.

Personalisation budget note: the local model trains end-to-end for the full
step budget and the *head* is simply excluded from aggregation, which is what
the paper prescribes (as opposed to FedRep, which additionally splits the local
budget into a head phase and a representation phase).
"""

from __future__ import annotations

from typing import Dict

from torch.utils.data import DataLoader

from data_provider.federated_data import make_client_args

from .base import FedMethod
from .fl_client import BaselineClient, average_state_dicts


class FedPer(FedMethod):
    name = 'fedper'
    venue = 'arXiv 2019'
    short_desc = 'Federate the shared body, keep the prediction head local'

    def setup_clients(self, loaders: Dict[str, Dict[str, DataLoader]]) -> None:
        cargs = make_client_args(self.args)
        for cid, ld in loaders.items():
            self.clients[cid] = BaselineClient(
                cid, cargs, self.device, ld['train'], ld['val'], ld['test']
            )
        self.global_body = average_state_dicts(
            [c.body_state_dict() for c in self.clients.values()]
        )

    def aggregate(self, round_idx: int) -> None:
        self.global_body = average_state_dicts(
            [c.body_state_dict() for c in self.clients.values()]
        )
        for c in self.clients.values():
            c.load_state_dict(self.global_body, strict=False)

    def predict(self, client_id, batch_x, batch_x_mark):
        return self.clients[client_id].predict(batch_x, batch_x_mark)
