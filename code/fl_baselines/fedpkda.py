"""
FedPKDA (AAAI 2026) adapted to TS forecasting.

Core idea: clip + Laplacian noise on local feature prototypes; server builds
global prototypes; clients dynamically align representations to local and
privacy-preserving global prototypes while keeping personalized heads.
"""

from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn
from torch.utils.data import DataLoader

from data_provider.federated_data import make_client_args

from .base import FedMethod
from .fl_client import BaselineClient, average_state_dicts


class FedPKDA(FedMethod):
    name = 'fedpkda'
    venue = 'AAAI 2026'
    short_desc = 'Privacy-preserving prototype alignment + personalized heads'

    def setup_clients(self, loaders: Dict[str, Dict[str, DataLoader]]) -> None:
        cargs = make_client_args(self.args)
        self.noise_scale = float(getattr(self.args, 'fedpkda_noise', 0.1))
        # ``fedpkda_clip <= 0`` means "auto": set the clip radius from the observed
        # prototype norms once the clients are built (see ``setup_clients``).
        # A hard-coded radius of 1.0 shrinks every prototype to unit norm while the
        # encoder features live at ~10x that scale, which makes the alignment term
        # ~100x the prediction loss and destroys the encoder -- measured as 9/9 last
        # place and -94% vs local across the nine benchmarks.
        self.clip_c = float(getattr(self.args, 'fedpkda_clip', 0.0))
        self.align_lambda = float(getattr(self.args, 'fedpkda_align', 0.1))
        for cid, ld in loaders.items():
            self.clients[cid] = BaselineClient(
                cid, cargs, self.device, ld['train'], ld['val'], ld['test']
            )
        self.global_body = None
        self.global_proto: torch.Tensor = torch.zeros(self.args.d_model)
        self.local_protos: Dict[str, torch.Tensor] = {}
        if self.clip_c <= 0:
            self.clip_c = self._auto_clip()

    @torch.no_grad()
    def _auto_clip(self) -> float:
        """Median prototype norm, so the clip radius tracks the feature scale.

        The paper treats the clip as a privacy radius; it has to be expressed in
        the units of the representation it clips.  Using the median keeps the
        bound tight for half the clients without collapsing anyone to the origin.
        """
        norms = []
        for client in self.clients.values():
            feats = client.extract_features(max_batches=4)
            if feats.numel():
                norms.append(float(feats.mean(dim=0).norm()))
        return float(sorted(norms)[len(norms) // 2]) if norms else 1.0

    def _noisy_proto(self, client: BaselineClient) -> torch.Tensor:
        feats = client.extract_features(max_batches=6)
        proto = feats.mean(dim=0)
        # clip
        norm = proto.norm() + 1e-8
        proto = proto * min(1.0, self.clip_c / float(norm))
        # Laplacian noise
        noise = torch.distributions.Laplace(0.0, self.noise_scale).sample(proto.shape)
        return proto + noise

    def local_update(self, client_id: str, epochs: int = 1) -> None:
        client = self.clients[client_id]
        if self.global_body is not None:
            sd = client.state_dict()
            sd.update(self.global_body)
            client.load_state_dict(sd, strict=False)

        client.model.train()
        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, client.model.parameters()),
            lr=self.args.learning_rate,
        )
        local_proto = self.local_protos.get(client_id, self.global_proto).to(self.device)
        g_proto = self.global_proto.to(self.device)
        # Budget parity: this loop iterates the loader directly, so it has to honour
        # ``fl_local_max_steps`` explicitly -- one epoch is 377 steps on ETTh1 but
        # 1600+ on ECL, i.e. 2-5x off the 800-step budget every other baseline gets.
        budget = int(getattr(self.args, 'fl_local_max_steps', 0) or 0)
        steps = 0

        for _ in range(max(int(epochs), 1)):
            for batch in client.train_loader:
                batch_x, batch_y, batch_x_mark, _ = batch
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                opt.zero_grad()
                enc, _, _, _, _, _ = client.model._encode_patches(batch_x, batch_x_mark)
                feat = enc.mean(dim=1)  # [BN, D]
                out = client.model(batch_x, batch_x_mark)
                f_dim = -1 if client.args.features == 'MS' else 0
                out = out[:, -client.args.pred_len:, f_dim:]
                tgt = batch_y[:, -client.args.pred_len:, f_dim:]
                pred_loss = client.criterion(out, tgt)
                # Dynamic alignment to local & global prototypes
                align = ((feat.mean(0) - local_proto) ** 2).mean() + (
                    (feat.mean(0) - g_proto) ** 2
                ).mean()
                loss = pred_loss + self.align_lambda * align
                loss.backward()
                opt.step()
                steps += 1
                if budget and steps >= budget:
                    break
            if budget and steps >= budget:
                break

        self.local_protos[client_id] = self._noisy_proto(client)

    def aggregate(self, round_idx: int) -> None:
        self.global_body = average_state_dicts([c.body_state_dict() for c in self.clients.values()])
        # Global prototype: centroid of noisy local protos (Mahalanobis → Euclidean fallback)
        protos = torch.stack(list(self.local_protos.values()), dim=0)
        self.global_proto = protos.mean(dim=0)
        # Optional: reweight by inverse distance to centroid
        dist = torch.norm(protos - self.global_proto.unsqueeze(0), dim=1) + 1e-6
        w = (1.0 / dist)
        w = w / w.sum()
        self.global_proto = (w.unsqueeze(1) * protos).sum(dim=0)

        for c in self.clients.values():
            sd = c.state_dict()
            sd.update(self.global_body)
            c.load_state_dict(sd, strict=False)

    def predict(self, client_id, batch_x, batch_x_mark):
        return self.clients[client_id].predict(batch_x, batch_x_mark)
