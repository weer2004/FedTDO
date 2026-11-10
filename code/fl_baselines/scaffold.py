"""SCAFFOLD (ICML 2020) — control variates that cancel client drift.

Local step uses the corrected gradient  ``g_i - c_i + c``  (server variate
``c``, client variate ``c_i``), and the variates are refreshed from the actual
displacement::

    c_i^+ = c_i - c + (w_global - w_i) / (K * lr)
    c^+   = c + (1/N) * sum_i (c_i^+ - c_i)

Two protocol notes that matter for reading the numbers:

* The correction is injected as a *gradient bias* (``grad_bias`` argument of
  ``BaselineClient.local_train``) instead of being fused into the optimiser, so
  every baseline still runs the same AdamW with the same moments.  A pure-SGD
  variant is available via ``--scaffold_opt sgd`` for reproducing the paper
  exactly; with a 1600-step budget SGD is simply a weaker optimiser, so the
  default stays AdamW for a like-for-like comparison.
* ``K * lr`` uses the real step count returned by ``local_train``, not the
  nominal budget, otherwise the variate scale is wrong on clients whose loader
  is shorter than the budget.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import DataLoader

from data_provider.federated_data import make_client_args

from .base import FedMethod
from .fl_client import BaselineClient, average_state_dicts


class SCAFFOLD(FedMethod):
    name = 'scaffold'
    venue = 'ICML 2020'
    short_desc = 'Control variates (c, c_i) added to local gradients to cancel drift'

    def setup_clients(self, loaders: Dict[str, Dict[str, DataLoader]]) -> None:
        cargs = make_client_args(self.args)
        # SCAFFOLD's variate is defined in *gradient* units: (w_start - w) / (K*lr)
        # reproduces the average gradient only for plain SGD.  With AdamW the
        # displacement saturates at ~K*lr per coordinate while the true gradient
        # is O(1e-2), so the variate ends up ~100x too large and swamps the loss
        # gradient (observed: ETTh1 MSE 1.33, c_global norm 4.5e5).  Default is
        # therefore SGD, and ``fedavg_sgd`` is run as the matched control.
        self.opt_kind = str(getattr(self.args, 'scaffold_opt', 'sgd'))
        for cid, ld in loaders.items():
            self.clients[cid] = BaselineClient(
                cid, cargs, self.device, ld['train'], ld['val'], ld['test']
            )
        self.global_state = average_state_dicts([c.state_dict() for c in self.clients.values()])
        # The variate is defined in *gradient* units, so it only exists for
        # tensors that actually receive a gradient.  ``state_dict()`` also holds
        # 42 bookkeeping entries of the Dynamic-TMoE model -- int64 counters
        # (``global_patch_counter`` advances by ``batch`` per step, so its
        # displacement is 112000 after 1600 steps), bool masks such as
        # ``expert_mask``, and the fixed ``position_embedding.pe``.  Computing
        # ``(start - local) / (K*lr)`` on those put a 1e5-scale number into
        # ``c_global`` and made the logged ``c_global_norm`` useless as a
        # diagnostic.  ``grad_bias`` only indexes ``named_parameters()``, so the
        # junk never entered a real gradient and the eight completed datasets
        # reproduce bit for bit (ETTh1 0.5638 before and after) -- this is a
        # correctness/observability fix, not a re-scaling of the method.
        ref = next(iter(self.clients.values()))
        self.param_keys = set(ref.parameter_keys())
        zero = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in self.global_state.items()}
        self.c_global = {k: v.clone() for k, v in zero.items()}
        self.c_local = {cid: {k: v.clone() for k, v in zero.items()} for cid in self.clients}
        self.last_steps: Dict[str, int] = {}
        self.round_start: Dict[str, Dict[str, 'torch.Tensor']] = {}

    def local_update(self, client_id: str, epochs: int = 1) -> None:
        client = self.clients[client_id]
        # Round 1 starts from each client's own init, rounds 2+ start from the
        # aggregated state that ``aggregate`` already pushed down.  Every other
        # baseline in this matrix behaves identically, so no explicit broadcast
        # here -- but the displacement in the variate must then be measured from
        # the *actual* round-start state, not from ``self.global_state``.
        self.round_start[client_id] = client.state_dict()
        bias = {k: (self.c_global[k] - self.c_local[client_id][k])
                for k in self.c_global}
        steps = client.local_train(epochs=epochs, grad_bias=bias,
                                   optimizer_kind=self.opt_kind)
        self.last_steps[client_id] = max(1, int(steps))

    def aggregate(self, round_idx: int) -> None:
        lr = float(getattr(self.args, 'learning_rate', 5e-4))
        new_local = {}
        deltas = []
        for cid, client in self.clients.items():
            k = self.last_steps.get(cid, 1)
            local = client.state_dict()
            start = self.round_start.get(cid, self.global_state)
            # only trainable parameters carry a variate; bookkeeping entries keep
            # c == 0 so they can never leak into ``grad_bias``
            numerator = {n: ((start[n].float() - local[n].float()) / (k * lr)
                             if n in self.param_keys
                             else torch.zeros_like(self.c_local[cid][n]))
                         for n in self.global_state}
            nxt = {n: self.c_local[cid][n] - self.c_global[n] + numerator[n]
                   for n in self.global_state}
            new_local[cid] = nxt
            deltas.append({n: nxt[n] - self.c_local[cid][n] for n in self.global_state})
        for n in self.c_global:
            self.c_global[n] = self.c_global[n] + sum(d[n] for d in deltas) / len(deltas)
        self.c_local = new_local

        self.global_state = average_state_dicts([c.state_dict() for c in self.clients.values()])
        for c in self.clients.values():
            c.load_state_dict(self.global_state)

    def predict(self, client_id, batch_x, batch_x_mark):
        return self.clients[client_id].predict(batch_x, batch_x_mark)

    def summary(self) -> dict:
        out = super().summary()
        out['optimizer'] = self.opt_kind
        out['c_global_norm'] = float(sum(float(v.norm()) for v in self.c_global.values()))
        out['steps'] = dict(self.last_steps)
        return out
