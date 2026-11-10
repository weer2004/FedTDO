"""
FedKWAZ (NeurIPS 2025) adapted to homogeneous TS forecasting.

Original targets model+data dual heterogeneity via proxy mutual learning and
Knowledge Weak-Aware Zones. Here we keep a shared proxy model on the server and
bidirectional distill on high-discrepancy (mixup) windows — the KWAZ idea —
while all clients use the same Dynamic-TMoE architecture.
"""

from __future__ import annotations

import copy
from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader

from data_provider.federated_data import make_client_args

from .base import FedMethod
from .fl_client import BaselineClient, average_state_dicts


class FedKWAZ(FedMethod):
    name = 'fedkwaz'
    venue = 'NeurIPS 2025'
    short_desc = 'Knowledge Weak-Aware Zones: proxy mutual distillation on hard mixups'

    def setup_clients(self, loaders: Dict[str, Dict[str, DataLoader]]) -> None:
        cargs = make_client_args(self.args)
        self.distill_alpha = float(getattr(self.args, 'fedkwaz_alpha', 0.3))
        self.mix_ratio = float(getattr(self.args, 'fedkwaz_mix', 0.5))
        for cid, ld in loaders.items():
            self.clients[cid] = BaselineClient(
                cid, cargs, self.device, ld['train'], ld['val'], ld['test']
            )
        # Shared proxy (= global model)
        self.proxy = copy.deepcopy(next(iter(self.clients.values())).model)
        self.proxy_state = average_state_dicts([c.state_dict() for c in self.clients.values()])
        self.proxy.load_state_dict({k: v.to(self.device) for k, v in self.proxy_state.items()})

    def _kwaz_batch(self, client: BaselineClient):
        """Build a mixup batch that maximizes private↔proxy prediction discrepancy."""
        best = None
        best_gap = -1.0
        criterion = nn.MSELoss(reduction='none')
        self.proxy.eval()
        client.model.eval()
        for batch in client.train_loader:
            batch_x, batch_y, batch_x_mark, _ = batch
            B = batch_x.shape[0]
            if B < 2:
                continue
            # random mixup pairs
            perm = torch.randperm(B)
            lam = self.mix_ratio
            x = lam * batch_x + (1 - lam) * batch_x[perm]
            y = lam * batch_y + (1 - lam) * batch_y[perm]
            xm = batch_x_mark  # keep marks of first
            x = x.float().to(self.device)
            y = y.float().to(self.device)
            xm = xm.float().to(self.device)
            with torch.no_grad():
                p_out = client.model(x, xm)
                q_out = self.proxy(x, xm)
                f_dim = -1 if client.args.features == 'MS' else 0
                p_out = p_out[:, -client.args.pred_len:, f_dim:]
                q_out = q_out[:, -client.args.pred_len:, f_dim:]
                gap = float((p_out - q_out).pow(2).mean().item())
            if gap > best_gap:
                best_gap = gap
                best = (x, y, xm, p_out, q_out)
            break  # one batch scan is enough per round for speed
        return best, best_gap

    def local_update(self, client_id: str, epochs: int = 1) -> None:
        client = self.clients[client_id]
        # Stage 1: local supervised train + distill from proxy
        client.local_train(
            epochs=epochs,
            distill_teacher=self.proxy,
            distill_alpha=self.distill_alpha * 0.5,
        )
        # Stage 2: KWAZ refinement — emphasize high-discrepancy mixups
        kwaz, gap = self._kwaz_batch(client)
        if kwaz is None:
            return
        x, y, xm, _, q_out = kwaz
        client.model.train()
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, client.model.parameters()),
            lr=self.args.learning_rate,
        )
        f_dim = -1 if client.args.features == 'MS' else 0
        for _ in range(max(1, epochs)):
            opt.zero_grad()
            out = client.model(x, xm)[:, -client.args.pred_len:, f_dim:]
            tgt = y[:, -client.args.pred_len:, f_dim:]
            loss = client.criterion(out, tgt) + self.distill_alpha * client.criterion(out, q_out.detach())
            loss.backward()
            opt.step()

        # Bidirectional: also update proxy toward client on KWAZ (small step)
        self.proxy.train()
        popt = torch.optim.AdamW(self.proxy.parameters(), lr=self.args.learning_rate * 0.5)
        popt.zero_grad()
        pout = self.proxy(x, xm)[:, -client.args.pred_len:, f_dim:]
        with torch.no_grad():
            cout = client.model(x, xm)[:, -client.args.pred_len:, f_dim:]
        ploss = nn.MSELoss()(pout, cout)
        ploss.backward()
        popt.step()
        self.proxy.eval()

    def aggregate(self, round_idx: int) -> None:
        # Proxy ← FedAvg of clients; clients keep personalized params
        self.proxy_state = average_state_dicts([c.state_dict() for c in self.clients.values()])
        self.proxy.load_state_dict({k: v.to(self.device) for k, v in self.proxy_state.items()})

    def predict(self, client_id, batch_x, batch_x_mark):
        return self.clients[client_id].predict(batch_x, batch_x_mark)
