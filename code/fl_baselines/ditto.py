"""Ditto (ICML 2022) — one global model plus one personalised model per client.

Per round, each client trains **two** models from the same starting global
weights:

* the *global* model on plain ERM -- this is the object that gets averaged;
* the *personalised* model on ERM + (lambda/2)||v - w_global||^2 -- this is the
  object that is evaluated and never uploaded.

Compute consequence, stated up front: each client performs 2 x 800 gradient
steps, i.e. twice the work of every single-model baseline at the same
per-model budget.  Halving each to 400 would instead undertrain both, so the
doubled cost is the faithful reading and is recorded in ``summary()``.
"""

from __future__ import annotations

from typing import Dict

from torch.utils.data import DataLoader

from data_provider.federated_data import make_client_args

from .base import FedMethod
from .fl_client import BaselineClient, average_state_dicts


class Ditto(FedMethod):
    name = 'ditto'
    venue = 'ICML 2022'
    short_desc = 'Global model (ERM, averaged) + personalised model (proximal, local)'

    def setup_clients(self, loaders: Dict[str, Dict[str, DataLoader]]) -> None:
        cargs = make_client_args(self.args)
        self.lam = float(getattr(self.args, 'ditto_lambda', 0.1))
        for cid, ld in loaders.items():
            self.clients[cid] = BaselineClient(
                cid, cargs, self.device, ld['train'], ld['val'], ld['test']
            )
        self.global_state = average_state_dicts([c.state_dict() for c in self.clients.values()])
        # Second model per client: the ERM copy that is actually federated.
        self.gclients: Dict[str, BaselineClient] = {}
        for cid, ld in loaders.items():
            g = BaselineClient(cid, cargs, self.device, ld['train'], ld['val'], ld['test'])
            g.load_state_dict(self.global_state)
            self.gclients[cid] = g
        for c in self.clients.values():
            c.load_state_dict(self.global_state)
            c.global_anchor = {k: v.clone() for k, v in self.global_state.items()}

    def train_offline(self) -> None:
        if self.step_mode() == 'epochs':
            self._train_fed_epochs()
            return
        rounds = int(getattr(self.args, 'fl_rounds', 2))
        for r in range(rounds):
            print(f'[{self.name}] round {r + 1}/{rounds}')
            for cid in self.clients:
                # (a) global-path ERM model
                g = self.gclients[cid]
                g.load_state_dict(self.global_state)
                g.local_train()
                # (b) personalised model, proximal to the *pre-round* global model
                p = self.clients[cid]
                p.global_anchor = {k: v.clone() for k, v in self.global_state.items()}
                p.local_train(proximal_mu=self.lam)
            self.aggregate(r)

    def _train_fed_epochs(self) -> None:
        """Epoch-mode path: both models advance on one global epoch counter.

        The base-class path cannot be reused because every tick has to train
        *two* models per client: the ERM copy that is averaged (``gclients``) and
        the proximal personal copy that is evaluated (``clients``).  The lr
        schedule is evaluated once per local epoch -- not once per client-epoch,
        which would decay 2*nc times too fast -- and both models receive the same
        scheduled lr so their trajectories stay comparable.

        The proximal anchor is refreshed at the round boundary only, matching the
        step-mode semantics: re-anchoring on every local epoch would turn a round
        into a chain of weakly coupled single-epoch fits.
        """
        rounds = max(1, int(getattr(self.args, 'fl_rounds', 2)))
        local_epochs = max(1, int(getattr(self.args, 'fl_local_epochs', 1)))
        patience = int(getattr(self.args, 'fl_patience', 15))
        total = rounds * local_epochs
        self.args.fl_total_epochs = total
        print(f'[{self.name}] epochs budget={total} '
              f'({rounds} rounds x {local_epochs} epochs x 2 models), lradj='
              f'{getattr(self.args, "fl_lradj", "type3")} patience={patience}')
        epoch, best, stale = 0, float('inf'), 0
        best_global, best_personal = None, None
        for r in range(rounds):
            anchor = {k: v.clone() for k, v in self.global_state.items()}
            for c in self.clients.values():
                c.global_anchor = anchor
            for cid in self.gclients:
                self.gclients[cid].load_state_dict(self.global_state)
            for _ in range(local_epochs):
                epoch += 1
                lr = self._lr_for_epoch(epoch)
                for cid in self.clients:
                    self.gclients[cid].local_train(epochs=1, lr=lr, no_step_cap=True)
                    self.clients[cid].local_train(
                        epochs=1, lr=lr, proximal_mu=self.lam, no_step_cap=True)
            self.aggregate(r)
            val = self._mean_val_loss()
            gval = self._mean_val_of(self.gclients)
            self.last_val_loss = val
            print(f'[{self.name}] round {r + 1}/{rounds} epoch {epoch}/{total} '
                  f'lr={lr:.2e} val={val} global_val={gval}')
            if val is None:
                continue
            if val < best:
                best, stale = val, 0
                best_global = {k: v.clone() for k, v in self.global_state.items()}
                best_personal = {cid: c.state_dict() for cid, c in self.clients.items()}
            else:
                stale += 1
                if patience and stale >= patience:
                    print(f'[{self.name}] early stop at epoch {epoch} '
                          f'(best val {best:.6f})')
                    break
        if best_personal is not None:
            self.global_state = best_global
            for cid, state in best_personal.items():
                self.clients[cid].load_state_dict(state)
            for cid, g in self.gclients.items():
                g.load_state_dict(self.global_state)
            self.best_val_loss = best

    def aggregate(self, round_idx: int) -> None:
        self.global_state = average_state_dicts([g.state_dict() for g in self.gclients.values()])

    def predict(self, client_id, batch_x, batch_x_mark):
        return self.clients[client_id].predict(batch_x, batch_x_mark)

    def summary(self) -> dict:
        out = super().summary()
        out.update({'lambda': self.lam, 'models_per_client': 2})
        return out
