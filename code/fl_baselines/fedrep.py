"""FedRep (ICML 2021) — decoupled representation learning and local heads.

Two local phases per round:

1. **head phase** (``--fedrep_head_steps``, default 200): body frozen, only the
   projection head is optimised on the current representation.
2. **representation phase** (remaining budget): head frozen, body optimised with
   the head's loss gradient.

Only the body is uploaded and averaged.  Splitting one 800-step budget into
200 + 600 keeps total client work identical to every other baseline; the split
is the method-specific knob, so it is recorded in ``summary()``.
"""

from __future__ import annotations

from typing import Dict

from torch.utils.data import DataLoader

from data_provider.federated_data import make_client_args

from .base import FedMethod
from .fl_client import BaselineClient, average_state_dicts


class FedRep(FedMethod):
    name = 'fedrep'
    venue = 'ICML 2021'
    short_desc = 'Head-first / body-second decoupled local training, body shared'

    def setup_clients(self, loaders: Dict[str, Dict[str, DataLoader]]) -> None:
        cargs = make_client_args(self.args)
        nominal = int(getattr(self.args, 'fl_local_max_steps', 800) or 800)
        head = int(getattr(self.args, 'fedrep_head_steps', 0) or 0)
        self.head_steps = head if head > 0 else max(1, nominal // 4)
        self.body_steps = max(1, nominal - self.head_steps)
        for cid, ld in loaders.items():
            self.clients[cid] = BaselineClient(
                cid, cargs, self.device, ld['train'], ld['val'], ld['test']
            )
        self.global_body = average_state_dicts(
            [c.body_state_dict() for c in self.clients.values()]
        )
        for c in self.clients.values():
            c.load_state_dict(self.global_body, strict=False)

    def local_update(self, client_id: str, epochs: int = 1) -> None:
        client = self.clients[client_id]
        client.set_trainable('head')
        client.local_train(epochs=epochs, max_steps=self.head_steps)
        client.set_trainable('body')
        client.local_train(epochs=epochs, max_steps=self.body_steps)
        client.set_trainable('all')

    def aggregate(self, round_idx: int) -> None:
        self.global_body = average_state_dicts(
            [c.body_state_dict() for c in self.clients.values()]
        )
        for c in self.clients.values():
            c.load_state_dict(self.global_body, strict=False)

    def predict(self, client_id, batch_x, batch_x_mark):
        return self.clients[client_id].predict(batch_x, batch_x_mark)

    def summary(self) -> dict:
        out = super().summary()
        out.update({'head_steps': self.head_steps, 'body_steps': self.body_steps})
        return out
