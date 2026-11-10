"""FedDA — federated *dataset* learning: client clustering + local augmentation.

The idea the paper sells is that aggregation should not be global: clients whose
data look alike should exchange models, and each client should also enlarge its
own dataset so that the shared model sees more of the local distribution.

Implementation in this codebase
------------------------------
* **clustering** — every client is described by a 7-number fingerprint of its
  target distribution (mean, std, min, max, q25, q50, q75 over the training
  loader).  The server clusters the fingerprints with k-means and each client is
  aggregated only with its own cluster (size-weighted FedAvg).
* **augmentation** — local mixup (``lam ~ Beta(a, a)`` folded to ``>= 0.5``)
  inside :meth:`BaselineClient.local_train`, so the local step budget is
  unchanged and the comparison with every other row stays budget-matched.

Deviations
----------
The paper's augmentation operator is designed for its own data modality and is not
public for time series; mixup is the standard substitute and is labelled as such.
Clients are also clustered on raw data statistics rather than on learned
representations, which keeps the server blind to model internals.

This class shares the temporal backbone with every other row (``--backbone``), so
the only thing that differs is *who aggregates with whom* plus the local mixup.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_provider.federated_data import make_client_args

from .base import FedMethod
from .fl_client import BaselineClient, average_state_dicts


def _fingerprint(loader: DataLoader, max_batches: int = 20) -> np.ndarray:
    """Seven robust statistics of the target distribution."""
    vals = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        vals.append(batch[1].reshape(-1).float())
    if not vals:
        return np.zeros(7, dtype=np.float32)
    y = torch.cat(vals).numpy()
    q = np.quantile(y, [0.25, 0.5, 0.75])
    return np.array([y.mean(), y.std(), y.min(), y.max(), q[0], q[1], q[2]],
                    dtype=np.float32)


def _cluster(desc: Dict[str, np.ndarray], k: int, seed: int) -> Dict[str, int]:
    """Lloyd's k-means on standardised fingerprints.

    Written out instead of calling ``sklearn.cluster.KMeans`` because the CPU
    interpreter used for the baseline grid ships an sklearn build that imports
    ``np.float`` and therefore fails against its own numpy.
    """
    keys = list(desc)
    x = np.stack([desc[c] for c in keys]).astype(np.float64)
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    k = max(1, min(k, len(keys)))
    if k == 1:
        return {c: 0 for c in keys}

    rng = np.random.default_rng(seed)
    centers = [x[rng.integers(len(x))]]
    for _ in range(k - 1):                       # k-means++ seeding
        d = np.min(((x[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
        p = d / d.sum() if d.sum() > 1e-12 else np.full(len(x), 1.0 / len(x))
        centers.append(x[rng.choice(len(x), p=p)])
    C = np.array(centers, dtype=np.float64)

    labels = np.zeros(len(x), dtype=int)
    for _ in range(50):
        labels = np.argmin(((x[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
        newC = C.copy()
        for j in range(k):
            if (labels == j).any():
                newC[j] = x[labels == j].mean(0)
        if np.allclose(newC, C, atol=1e-9):
            break
        C = newC
    return {c: int(l) for c, l in zip(keys, labels)}


class FedDA(FedMethod):
    name = 'fedda'
    venue = '—'
    short_desc = ('Cluster clients on data statistics, aggregate within the cluster, '
                  'local mixup augmentation')

    def __init__(self, args, device):
        super().__init__(args, device)
        self.n_clusters = int(getattr(args, 'fedda_clusters', 0) or 0)
        self.mixup_alpha = float(getattr(args, 'fedda_mixup', 0.2))
        self.desc: Dict[str, np.ndarray] = {}
        self.labels: Dict[str, int] = {}

    def setup_clients(self, loaders: Dict[str, Dict[str, DataLoader]]) -> None:
        cargs = make_client_args(self.args)
        for cid, ld in loaders.items():
            self.clients[cid] = BaselineClient(
                cid, cargs, self.device, ld['train'], ld['val'], ld['test'])
            self.desc[cid] = _fingerprint(ld['train'])

    def local_update(self, client_id: str, epochs: int = 1) -> None:
        self.clients[client_id].local_train(epochs=epochs,
                                            mixup_alpha=self.mixup_alpha)

    def aggregate(self, round_idx: int) -> None:
        k = self.n_clusters or max(2, int(round(len(self.clients) ** 0.5)))
        self.labels = _cluster(self.desc, k, int(getattr(self.args, 'seed', 2021)))
        groups: Dict[int, List[str]] = {}
        for cid, lab in self.labels.items():
            groups.setdefault(lab, []).append(cid)

        for lab, members in sorted(groups.items()):
            states = [self.clients[c].state_dict() for c in members]
            avg = average_state_dicts(states)
            for c in members:
                self.clients[c].load_state_dict(avg)
        sizes = {lab: len(m) for lab, m in sorted(groups.items())}
        print(f'[{self.name}] round {round_idx + 1}: clusters {sizes}, '
              f'mixup alpha={self.mixup_alpha}')

    def predict(self, client_id, batch_x, batch_x_mark):
        return self.clients[client_id].predict(batch_x, batch_x_mark)
