"""FedNova (NeurIPS 2020) — normalised averaging for unequal local work.

Each client reports its own step count ``tau_i``.  The server averages the
*normalised* direction ``d_i = (w_global - w_i) / tau_i`` and rescales by a
reference step count ``tau_eff``::

    p_i = tau_i / sum_j tau_j ,   w <- w_global - tau_eff * sum_i p_i d_i

With ``tau_eff = mean(tau)`` this is algebraically identical to FedAvg, which is
the honest default and a useful sanity check: if the two rows differ on equal
step counts, the implementation is wrong.  The method only exists in the regime
it was written for — **unequal per-client work** — so ``--fednova_hetero 1``
draws each client's budget proportional to its loader length (clipped to
[0.5, 1.5] of the nominal budget and renormalised so the fleet total is
unchanged), which is exactly the "different local workload" bias the paper
targets.
"""

from __future__ import annotations

from typing import Dict

from torch.utils.data import DataLoader

from data_provider.federated_data import make_client_args

from .base import FedMethod
from .fl_client import BaselineClient, average_state_dicts


class FedNova(FedMethod):
    name = 'fednova'
    venue = 'NeurIPS 2020'
    short_desc = 'Normalised gradient averaging with per-client step counts tau_i'

    def setup_clients(self, loaders: Dict[str, Dict[str, DataLoader]]) -> None:
        cargs = make_client_args(self.args)
        self.hetero = int(getattr(self.args, 'fednova_hetero', 1))
        self.teff_mode = str(getattr(self.args, 'fednova_teff', 'fixed')).lower()
        for cid, ld in loaders.items():
            self.clients[cid] = BaselineClient(
                cid, cargs, self.device, ld['train'], ld['val'], ld['test']
            )
        self.global_state = average_state_dicts([c.state_dict() for c in self.clients.values()])
        self.budgets = self._budgets()
        self.tau: Dict[str, int] = {}
        self.budget_mode = 'equal'

    def _budgets(self) -> Dict[str, int]:
        """Per-client step budgets whose fleet total equals ``N * nominal``.

        Clients in this benchmark hold one channel of the same series, so their
        loaders are *always* the same length -- data-volume heterogeneity does
        not exist here.  ``hetero=1`` therefore falls back to a deterministic
        straggler spread in [0.5, 1.5] x nominal across clients (rank order),
        rescaled to preserve the total.  That is the system-heterogeneity regime
        FedNova was written for; ``hetero=0`` gives equal steps and degenerates
        to FedAvg, which is reported as a sanity check.
        """
        nominal = int(getattr(self.args, 'fl_local_max_steps', 800) or 800)
        ids = list(self.clients)
        if not self.hetero:
            self.budget_mode = 'equal'
            return {cid: nominal for cid in ids}
        sizes = []
        for cid in ids:
            try:
                sizes.append(max(1, len(self.clients[cid].train_loader)))
            except TypeError:
                sizes.append(1)
        homogeneous = max(sizes) <= 1.05 * min(sizes)
        if homogeneous:
            self.budget_mode = 'straggler'
            n = len(ids)
            raw = [nominal * (0.5 + (1.0 * i / max(1, n - 1))) for i in range(n)]
        else:
            self.budget_mode = 'volume'
            mean_size = sum(sizes) / len(sizes)
            raw = [min(1.5 * nominal, max(0.5 * nominal, nominal * (s / mean_size)))
                   for s in sizes]
        scale = (nominal * len(ids)) / sum(raw)
        return {cid: max(64, int(round(v * scale))) for cid, v in zip(ids, raw)}

    def local_update(self, client_id: str, epochs: int = 1) -> None:
        client = self.clients[client_id]
        # round 1 from own init, rounds 2+ from the state ``aggregate`` pushed
        # down -- identical round semantics for every baseline here
        steps = client.local_train(epochs=epochs, max_steps=self.budgets[client_id])
        self.tau[client_id] = max(1, int(steps))

    def aggregate(self, round_idx: int) -> None:
        total_tau = float(sum(self.tau.values()))
        n = len(self.tau)
        tau_eff = (total_tau / n if self.teff_mode == 'mean'
                   else float(getattr(self.args, 'fl_local_max_steps', 800) or 800))
        pkeys = next(iter(self.clients.values())).parameter_keys()
        states = {cid: c.state_dict() for cid, c in self.clients.items()}
        acc = None
        for cid, client in self.clients.items():
            k = float(self.tau[cid])
            local = states[cid]
            p = k / total_tau
            scaled = {}
            for name in self.global_state:
                if name not in pkeys:
                    continue
                scaled[name] = ((self.global_state[name].float() - local[name].float()) / k) \
                    * (p * tau_eff)
            acc = scaled if acc is None else {name: acc[name] + scaled[name] for name in acc}
        updated = {name: self.global_state[name].float() - acc[name] for name in acc}
        # Bookkeeping buffers (expert_mask, call counters, positional encoding)
        # are not parameters: they follow the plain client average exactly as in
        # FedAvg, otherwise the normalised update flips the MoE mask and the next
        # forward returns NaN.
        for name in self.global_state:
            if name in pkeys:
                continue
            acc_buf = None
            for local in states.values():
                t = local[name].float()
                acc_buf = t if acc_buf is None else acc_buf + t
            updated[name] = acc_buf / float(len(states))
        self.global_state = updated
        for c in self.clients.values():
            c.load_state_dict(self.global_state)

    def predict(self, client_id, batch_x, batch_x_mark):
        return self.clients[client_id].predict(batch_x, batch_x_mark)

    def summary(self) -> dict:
        out = super().summary()
        out.update({
            'hetero_steps': self.hetero,
            'budget_mode': self.budget_mode,
            'budgets': dict(self.budgets),
            'teff_mode': self.teff_mode,
            'tau': dict(self.tau),
            'tau_min': min(self.tau.values()) if self.tau else None,
            'tau_max': max(self.tau.values()) if self.tau else None,
        })
        return out
