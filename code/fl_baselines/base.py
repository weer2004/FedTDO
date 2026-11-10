"""
Common interface for FL baselines under the FedTDO experiment protocol.

All methods share:
  - channel-as-client univariate Dynamic-TMoE models
  - the same train/val/test loaders
  - the same offline rounds + online prequential eval metrics (MSE/MAE)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader


class FedMethod(ABC):
    """Abstract FL baseline. Subclasses implement aggregation / personalization."""

    name: str = 'base'
    venue: str = ''
    short_desc: str = ''

    def __init__(self, args, device: torch.device):
        self.args = args
        self.device = device
        self.clients = {}  # cid -> BaselineClient
        self._online_optims = {}  # cid -> optimiser state kept across online steps

    @abstractmethod
    def setup_clients(self, loaders: Dict[str, Dict[str, DataLoader]]) -> None:
        ...

    # ------------------------------------------------------------ lr schedule
    def step_mode(self) -> str:
        """``steps`` (legacy) or ``epochs`` (DTMoe-aligned protocol)."""
        return str(getattr(self.args, 'fl_step_mode', 'steps')).lower()

    def _lr_for_epoch(self, epoch: int) -> float:
        """Same formulas as ``exp_fedtdo._adjust_learning_rate`` and
        ``utils/tools.adjust_learning_rate``: 1-based ``epoch``."""
        base = float(self.args.learning_rate)
        schedule = str(getattr(self.args, 'fl_lradj', 'type3'))
        if schedule == 'type1':
            return base * (0.9 ** (epoch - 1))
        if schedule == 'type3':
            return base if epoch <= 3 else base * (0.8 ** (epoch - 3))
        if schedule == 'cosine':
            total = max(1, int(getattr(self.args, 'fl_total_epochs', 60) or 60))
            return base / 2 * (1 + math.cos(epoch / total * math.pi))
        if schedule == 'constant':
            return base
        raise ValueError(f'unknown fl_lradj {schedule!r}')

    def _client_val_loss(self, client) -> Optional[float]:
        loader = getattr(client, 'val_loader', None)
        if loader is None or len(loader) == 0:
            return None
        client.model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, _ in loader:
                batch_x = batch_x.float().to(client.device)
                batch_y = batch_y.float().to(client.device)
                batch_x_mark = batch_x_mark.float().to(client.device)
                f_dim = -1 if client.args.features == 'MS' else 0
                output = client.model(batch_x, batch_x_mark)
                output = output[:, -client.args.pred_len:, f_dim:]
                target = batch_y[:, -client.args.pred_len:, f_dim:]
                total += float(client.criterion(output, target).item())
                count += 1
        client.model.train()
        return total / max(count, 1)

    def _mean_val_of(self, clients: Dict[str, object]) -> Optional[float]:
        values = [v for v in (self._client_val_loss(c) for c in clients.values())
                  if v is not None]
        return sum(values) / len(values) if values else None

    def _mean_val_loss(self) -> Optional[float]:
        return self._mean_val_of(self.clients)

    def _push_lr(self, lr: float):
        """Temporarily install a scheduled lr and remove the step cap."""
        saved = (getattr(self.args, 'learning_rate', None),
                 getattr(self.args, 'fl_local_max_steps', 0))
        self.args.learning_rate = lr
        self.args.fl_local_max_steps = 0
        return saved

    def _pop_lr(self, saved) -> None:
        lr, steps = saved
        if lr is not None:
            self.args.learning_rate = lr
        self.args.fl_local_max_steps = steps

    # ----------------------------------------------------------------- budget
    def train_offline(self) -> None:
        """Several Fed rounds of local training + aggregation."""
        if self.step_mode() == 'epochs':
            self._train_fed_epochs()
            return
        rounds = int(getattr(self.args, 'fl_rounds', 5))
        local_epochs = int(getattr(self.args, 'fl_local_epochs', 1))
        for r in range(rounds):
            print(f'[{self.name}] round {r + 1}/{rounds}')
            for cid in self.clients:
                self.local_update(cid, epochs=local_epochs)
            self.aggregate(r)

    def _train_fed_epochs(self) -> None:
        """DTMoe-aligned federated training.

        Total budget is ``fl_rounds x fl_local_epochs`` epochs per client, with
        the lr schedule advancing on a *global* epoch counter (a round boundary
        must not restart the decay, otherwise every round trains at the warm-up
        lr).  Early stopping watches the mean validation loss over clients after
        each aggregation, and the best *global* snapshot is restored - the
        federated analogue of ``EarlyStopping`` in ``run.py``.
        """
        rounds = int(getattr(self.args, 'fl_rounds', 5))
        local_epochs = max(1, int(getattr(self.args, 'fl_local_epochs', 1)))
        patience = int(getattr(self.args, 'fl_patience', 15))
        total = rounds * local_epochs
        self.args.fl_total_epochs = total
        print(f'[{self.name}] epochs budget={total} '
              f'({rounds} rounds x {local_epochs} epochs), lradj='
              f'{getattr(self.args, "fl_lradj", "type3")} patience={patience}')
        epoch, best, stale, best_state = 0, float('inf'), 0, None
        for r in range(rounds):
            # One epoch tick per local epoch, not per client-epoch: incrementing
            # inside the client loop advances the schedule nc times too fast and
            # drove the lr to 8.9e-28 by round 3.
            for _ in range(local_epochs):
                epoch += 1
                lr = self._lr_for_epoch(epoch)
                for cid in self.clients:
                    self._local_update_epoch(cid, lr)
            self.aggregate(r)
            val = self._mean_val_loss()
            self.last_val_loss = val
            print(f'[{self.name}] round {r + 1}/{rounds} epoch {epoch}/{total} '
                  f'lr={self._lr_for_epoch(epoch):.2e} val={val}')
            if val is None:
                continue
            if val < best:
                best, stale = val, 0
                best_state = {cid: c.state_dict() for cid, c in self.clients.items()}
            else:
                stale += 1
                if patience and stale >= patience:
                    print(f'[{self.name}] early stop at epoch {epoch} (best val {best:.6f})')
                    break
        if best_state is not None:
            for cid, state in best_state.items():
                self.clients[cid].load_state_dict(state)
            self.best_val_loss = best

    def train_local_epochs(self) -> None:
        """Per-client early stopping with no aggregation (``local`` control).

        This is the epoch-mode counterpart of ``--baseline local``: identical
        model, identical data, identical budget to the FedTDO arm, but no
        sharing and no freeze.
        """
        total = max(1, int(getattr(self.args, 'fl_rounds', 5))
                    * max(1, int(getattr(self.args, 'fl_local_epochs', 1))))
        patience = int(getattr(self.args, 'fl_patience', 15))
        self.args.fl_total_epochs = total
        print(f'[{self.name}] local epochs budget={total}, '
              f'lradj={getattr(self.args, "fl_lradj", "type3")} patience={patience}')
        for cid, client in self.clients.items():
            best, stale, best_state, ran = float('inf'), 0, None, 0
            for epoch in range(1, total + 1):
                client.local_train(epochs=1, lr=self._lr_for_epoch(epoch),
                                   no_step_cap=True)
                ran = epoch
                val = self._client_val_loss(client)
                if val is None:
                    continue
                if val < best:
                    best, stale, best_state = val, 0, client.state_dict()
                else:
                    stale += 1
                    if patience and stale >= patience:
                        break
            if best_state is not None:
                client.load_state_dict(best_state)
            print(f'  {cid}: epochs_run={ran} best_val={best if best < float("inf") else float("nan"):.6f}')

    def local_update(self, client_id: str, epochs: int = 1) -> None:
        self.clients[client_id].local_train(epochs=epochs)

    def _local_update_epoch(self, client_id: str, lr: float) -> None:
        """One local epoch at a scheduled lr, through the method's own path.

        Calling ``client.local_train`` directly would skip every override of
        ``local_update`` -- proximal terms (FedProx/pFedMe), SCAFFOLD's control
        variates, FedNova's ``tau``, FedPKDA's prototypes, FedALA's per-layer lrs
        -- and the resulting rows would silently describe a different method
        (FedNova/FedPKDA crashed outright).  The scheduled lr is injected through
        ``args.learning_rate`` because that is what ``local_train`` reads, and it
        is restored before aggregation so server-side optimisers keep the CLI lr.
        """
        saved = self._push_lr(lr)
        try:
            self.local_update(client_id, epochs=1)
        finally:
            self._pop_lr(saved)

    def aggregate(self, round_idx: int) -> None:
        """Default no-op; FedAvg-style methods override."""
        return None

    @abstractmethod
    def predict(
        self,
        client_id: str,
        batch_x: torch.Tensor,
        batch_x_mark: torch.Tensor,
    ) -> torch.Tensor:
        """Return forecast [B, pred_len, C]."""
        ...

    def online_adapt(
        self,
        client_id: str,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        batch_x_mark: torch.Tensor,
        t: int,
        pred: torch.Tensor,
    ) -> Optional[dict]:
        """
        Optional delayed-feedback adaptation for online methods (e.g. Fed-POE).
        Called only after the complete H-step target for the stored forecast has
        matured; it may affect forecasts from the release origin onward.
        """
        return None

    def online_step(
        self,
        client_id: str,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        batch_x_mark: torch.Tensor,
        lr: float,
    ) -> None:
        """One gradient step on a stored full-horizon target after release.

        The caller waits until the final target timestamp is observable (origin
        t + pred_len or later) before invoking this method. The update can affect
        forecasts from that release origin onward, but cannot revise any earlier
        forecast. ``lr = 0`` reproduces the static model.
        """
        if lr <= 0:
            return
        client = self.clients[client_id]
        optimiser = self._online_optims.get(client_id)
        if optimiser is None:
            optimiser = torch.optim.SGD(
                [p for p in client.model.parameters() if p.requires_grad], lr=lr)
            self._online_optims[client_id] = optimiser
        else:
            for group in optimiser.param_groups:
                group['lr'] = lr
        client.model.train()
        x = batch_x.float().to(client.device)
        xm = batch_x_mark.float().to(client.device)
        f_dim = -1 if client.args.features == 'MS' else 0
        target = batch_y[:, -client.args.pred_len:, f_dim:].float().to(client.device)
        # ``online_evaluate`` runs under ``@torch.no_grad()``; the adaptation step
        # is the one place that needs a graph, so re-enable it locally.
        with torch.enable_grad():
            optimiser.zero_grad()
            output = client.model(x, xm)[:, -client.args.pred_len:, f_dim:]
            if str(getattr(self.args, 'fl_online_target', 'first')) == 'first':
                # Timestamp-level one-step feedback: only y(t+1) is revealed when
                # this update fires, and it is the only label used.
                output = output[:, :1]
                target = target[:, :1]
            loss = client.criterion(output, target)
            loss.backward()
            # Numerical guard, not an algorithmic change: a non-finite loss would
            # poison every later prediction and turn the cell into a NaN row.  A
            # skipped step leaves the served model exactly where it was.
            if torch.isfinite(loss):
                optimiser.step()
            else:
                optimiser.zero_grad(set_to_none=True)

    def client_ids(self) -> List[str]:
        return list(self.clients.keys())

    def summary(self) -> dict:
        return {'method': self.name, 'venue': self.venue, 'num_clients': len(self.clients)}
