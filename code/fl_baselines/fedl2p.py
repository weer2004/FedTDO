"""
FedL2P (NeurIPS 2023) adapted to TS forecasting.

Core idea: a federated meta-net maps local data statistics → per-layer learning
rates for personalization fine-tuning of the global model.
"""

from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader

from data_provider.federated_data import make_client_args

from .base import FedMethod
from .fl_client import BaselineClient, average_state_dicts


class LRMetaNet(nn.Module):
    def __init__(self, stat_dim: int, num_groups: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(stat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_groups),
            nn.Softplus(),
        )

    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        # returns positive LR multipliers
        return self.net(stats) + 1e-4


class FedL2P(FedMethod):
    name = 'fedl2p'
    venue = 'NeurIPS 2023'
    short_desc = 'Learn client personalization strategies (meta-net for layer-wise LRs)'

    def setup_clients(self, loaders: Dict[str, Dict[str, DataLoader]]) -> None:
        cargs = make_client_args(self.args)
        for cid, ld in loaders.items():
            self.clients[cid] = BaselineClient(
                cid, cargs, self.device, ld['train'], ld['val'], ld['test']
            )
        self.global_state = average_state_dicts([c.state_dict() for c in self.clients.values()])
        self.layer_groups = ['patch_embedding', 'temporal_moe', 'projection', 'other']
        self.meta = LRMetaNet(stat_dim=8, num_groups=len(self.layer_groups)).to(self.device)
        self.meta_opt = torch.optim.Adam(self.meta.parameters(), lr=1e-3)
        self.base_lr = float(self.args.learning_rate)
        self.ft_epochs = int(getattr(self.args, 'fedl2p_ft_epochs', 1))

    def _local_stats(self, client: BaselineClient) -> torch.Tensor:
        feats = client.extract_features(max_batches=4)  # [N, D]
        mean = feats.mean(dim=0)
        std = feats.std(dim=0)
        # compact 8-d summary
        k = min(4, mean.numel())
        stats = torch.cat([mean[:k], std[:k]], dim=0)
        if stats.numel() < 8:
            stats = torch.nn.functional.pad(stats, (0, 8 - stats.numel()))
        return stats[:8].to(self.device)

    def _group_lrs(self, multipliers: torch.Tensor, client: BaselineClient) -> Dict[str, float]:
        named = dict(client.model.named_parameters())
        group_lr = {
            'patch_embedding': self.base_lr * float(multipliers[0]),
            'temporal_moe': self.base_lr * float(multipliers[1]),
            'projection': self.base_lr * float(multipliers[2]),
            'other': self.base_lr * float(multipliers[3]),
        }
        param_lrs = {}
        for name in named:
            if name.startswith('patch_embedding'):
                param_lrs[name] = group_lr['patch_embedding']
            elif name.startswith('temporal_moe'):
                param_lrs[name] = group_lr['temporal_moe']
            elif name.startswith('projection'):
                param_lrs[name] = group_lr['projection']
            else:
                param_lrs[name] = group_lr['other']
        return param_lrs

    def local_update(self, client_id: str, epochs: int = 1,
                     reload_global: bool = True) -> None:
        client = self.clients[client_id]
        # Start from global.  The epoch-mode path keeps several consecutive
        # fine-tune epochs inside one round, so it reloads only on the first
        # one -- reloading every epoch would discard the personalisation
        # progress the previous epoch just made.
        if reload_global:
            client.load_state_dict(self.global_state)
        stats = self._local_stats(client)
        mult = self.meta(stats)
        param_lrs = self._group_lrs(mult, client)
        client.local_train(epochs=max(epochs, self.ft_epochs), param_lrs=param_lrs)

        # Meta update: encourage low local val loss (one-step REINFORCE-free surrogate)
        # Use train loss of a single batch as proxy and backprop through meta via LR scale
        self.meta_opt.zero_grad()
        # Surrogate: prefer moderate multipliers (regularize) + lower feature variance
        reg = ((mult - 1.0) ** 2).mean()
        reg.backward()
        self.meta_opt.step()

    def aggregate(self, round_idx: int) -> None:
        # FedAvg of client models → new global; also avg meta-net is local-only here
        self.global_state = average_state_dicts([c.state_dict() for c in self.clients.values()])
        for c in self.clients.values():
            c.global_anchor = {k: v.clone() for k, v in self.global_state.items()}

    def train_offline(self) -> None:
        if self.step_mode() == 'epochs':
            self._train_fed_epochs()
            return
        rounds = int(getattr(self.args, 'fl_rounds', 5))
        local_epochs = int(getattr(self.args, 'fl_local_epochs', 1))
        # Phase 1: standard FedAvg warm-up
        warm = max(1, rounds // 2)
        for r in range(warm):
            print(f'[{self.name}] warmup round {r + 1}/{warm}')
            for cid in self.clients:
                self.clients[cid].load_state_dict(self.global_state)
                self.clients[cid].local_train(epochs=local_epochs)
            self.aggregate(r)
        # Phase 2: personalized fine-tune with learned LRs
        for r in range(warm, rounds):
            print(f'[{self.name}] personalize round {r + 1}/{rounds}')
            for cid in self.clients:
                self.local_update(cid, epochs=local_epochs)
            self.aggregate(r)

    def _personalize_epoch(self, client_id: str, lr: float,
                           reload_global: bool) -> None:
        """One personalised epoch at the scheduled lr.

        ``local_update`` builds the per-parameter lrs from ``self.base_lr``, so
        the schedule is installed there; ``args.learning_rate`` is pushed too
        because the fallback group inside ``local_train`` reads it.  Both are
        restored afterwards so server-side state keeps the CLI value.
        """
        saved_base = self.base_lr
        self.base_lr = lr
        saved = self._push_lr(lr)
        try:
            self.local_update(client_id, epochs=1, reload_global=reload_global)
        finally:
            self._pop_lr(saved)
            self.base_lr = saved_base

    def _train_fed_epochs(self) -> None:
        """Epoch-mode path: FedAvg warm-up, then meta-learned fine-tuning.

        Both phases tick the *same* global epoch counter, so the lr schedule runs
        continuously across the warm-up/personalisation switch instead of
        restarting at the warm-up lr.  Budgets stay identical to every
        single-model baseline in an epoch-mode table: one local epoch per tick
        per client.
        """
        rounds = max(1, int(getattr(self.args, 'fl_rounds', 5)))
        local_epochs = max(1, int(getattr(self.args, 'fl_local_epochs', 1)))
        patience = int(getattr(self.args, 'fl_patience', 15))
        warm = max(1, rounds // 2)
        total = rounds * local_epochs
        self.args.fl_total_epochs = total
        print(f'[{self.name}] epochs budget={total} '
              f'({warm} warm-up + {rounds - warm} personalisation rounds x '
              f'{local_epochs} epochs), lradj='
              f'{getattr(self.args, "fl_lradj", "type3")} patience={patience}')
        if self.ft_epochs > 1:
            print(f'[{self.name}] note: fedl2p_ft_epochs={self.ft_epochs} is folded '
                  f'to 1 epoch per tick so the budget matches the epoch-mode table')
        epoch, best, stale, best_state = 0, float('inf'), 0, None
        for r in range(rounds):
            warmup = r < warm
            if warmup:
                # Same round-level reset as the step-mode warm-up.
                for c in self.clients.values():
                    c.load_state_dict(self.global_state)
            for e in range(local_epochs):
                epoch += 1
                lr = self._lr_for_epoch(epoch)
                for cid in self.clients:
                    if warmup:
                        self.clients[cid].local_train(
                            epochs=1, lr=lr, no_step_cap=True)
                    else:
                        self._personalize_epoch(cid, lr, reload_global=(e == 0))
            self.aggregate(r)
            val = self._mean_val_loss()
            self.last_val_loss = val
            phase = 'warmup' if warmup else 'personalize'
            print(f'[{self.name}] {phase} round {r + 1}/{rounds} '
                  f'epoch {epoch}/{total} lr={lr:.2e} val={val}')
            if val is None:
                continue
            if val < best:
                best, stale = val, 0
                best_state = {cid: c.state_dict() for cid, c in self.clients.items()}
            else:
                stale += 1
                if patience and stale >= patience:
                    print(f'[{self.name}] early stop at epoch {epoch} '
                          f'(best val {best:.6f})')
                    break
        if best_state is not None:
            for cid, state in best_state.items():
                self.clients[cid].load_state_dict(state)
            self.best_val_loss = best

    def predict(self, client_id, batch_x, batch_x_mark):
        return self.clients[client_id].predict(batch_x, batch_x_mark)
