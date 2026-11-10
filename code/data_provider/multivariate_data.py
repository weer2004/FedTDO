"""Dataset-level multivariate clients — the protocol FFTS / FeDaL actually use.

Why a second data path
----------------------
``federated_data.build_federated_clients_from_channels`` gives every client **one
channel of one dataset** (``enc_in = 1``).  That maximises statistical
heterogeneity, but it structurally disables half of the backbone list: with a
single variate there is nothing for iTransformer's variate attention, TimesNet's
2-D period tensor or TimeMixer's cross-scale channel mixing to model, and the
resulting numbers measure a degenerate form rather than the published model.

FFTS (AAAI'25) and FeDaL (ICLR'26) both train one model over **heterogeneous
datasets**, each client holding a whole multivariate dataset (``features M``,
``enc_in = C``).  This module implements that construction:

* one ``Dataset`` class over ``[N, C]`` arrays with TSLib's 7:1:2 split;
* ``StandardScaler`` fitted on the train slice only (per channel);
* a **fixed** channel budget ``C`` for every dataset, so one global model can be
  averaged across clients whose source datasets have 7 / 8 / 21 / 321 / 862
  columns.  The subset is drawn with a per-dataset seeded RNG, so every backbone
  and every method sees the identical sensors.

The channel budget is the one honest compromise: a global federated model needs a
single ``enc_in``, and the alternative (per-dataset ``enc_in``) makes the
parameter tensors un-averagable.  ``--paper_channels 7`` keeps ETT/ILI/Exchange at
their full channel count and subsamples the wide datasets, which is exactly the
"every client holds a multivariate series" regime the two papers assume.
"""

from __future__ import annotations

import os
import zlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


class Dataset_MultivariateClient(Dataset):
    """Sliding windows over a ``[N, C]`` block with train/val/test borders."""

    def __init__(
        self,
        values: np.ndarray,
        flag: str,
        seq_len: int,
        label_len: int,
        pred_len: int,
        scale: bool = True,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
    ):
        assert flag in ('train', 'val', 'test')
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.flag = flag

        n = len(values)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        borders = {
            'train': (0, n_train),
            # same overlap convention as Dataset_UnivariateClient: the val/test
            # slice keeps ``seq_len`` points of history so the first window is
            # complete
            'val': (n_train - seq_len, n_train + n_val),
            'test': (n_train + n_val - seq_len, n),
        }
        b1, b2 = borders[flag]

        if scale:
            self.scaler = StandardScaler()
            self.scaler.fit(values[:n_train])
            data = self.scaler.transform(values)
        else:
            self.scaler = None
            data = values
        self.data_x = data[b1:b2]
        self.data_y = self.data_x

    def __len__(self):
        return max(0, len(self.data_x) - self.seq_len - self.pred_len + 1)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        # the zoo models ignore x_mark; a 1-column zero stamp keeps the 4-tuple
        # contract that BaselineClient unpacks
        stamp = np.zeros((self.seq_len, 1), dtype=np.float32)
        y_stamp = np.zeros((self.label_len + self.pred_len, 1), dtype=np.float32)
        return (
            self.data_x[s_begin:s_end].astype(np.float32),
            self.data_y[r_begin:r_end].astype(np.float32),
            stamp,
            y_stamp,
        )


def pick_channels(n_cols: int, wanted: int, seed: int, name: str) -> List[int]:
    """Deterministic per-dataset channel subset (sorted)."""
    if wanted >= n_cols:
        return list(range(n_cols))
    # stable (process independent) hash of the dataset name, so two datasets with
    # the same column count do not get identical positions by coincidence --
    # Python's builtin hash() is salted per process and would make runs
    # irreproducible
    offset = zlib.crc32(str(name).encode('utf-8')) % 9973
    rng = np.random.default_rng(int(seed) + offset)
    picked = rng.choice(n_cols, size=int(wanted), replace=False)
    return sorted(int(i) for i in picked)


def build_paper_clients(
    specs: Dict[str, Tuple[str, str]],
    *,
    channels: int = 7,
    seq_len: int = 96,
    label_len: int = 48,
    pred_len: int = 96,
    batch_size: int = 32,
    eval_batch_size: int = 128,
    seed: int = 2021,
    num_workers: int = 0,
) -> Dict[str, Dict]:
    """``{dataset: {'train': loader, 'test': loader, 'channels': [...]}}``.

    ``specs`` maps dataset name -> ``(root_path, data_path)``; the caller owns the
    dataset list so a hold-out can simply be dropped from it.
    """
    bundles: Dict[str, Dict] = {}
    for name, (root, fname) in specs.items():
        path = os.path.join(root, fname)
        df = pd.read_csv(path)
        cols = [c for c in df.columns if c != 'date']
        idx = pick_channels(len(cols), channels, seed, name)
        values = df[[cols[i] for i in idx]].values.astype(np.float64)

        loaders = {}
        for flag, bs, shuffle in (('train', batch_size, True),
                                  ('val', eval_batch_size, False),
                                  ('test', eval_batch_size, False)):
            ds = Dataset_MultivariateClient(
                values, flag, seq_len=seq_len, label_len=label_len,
                pred_len=pred_len, scale=True)
            loaders[flag] = DataLoader(
                ds, batch_size=bs, shuffle=shuffle, num_workers=num_workers,
                drop_last=False, pin_memory=False)
        # the scaler is fitted per split but always on ``values[:n_train]``, so all
        # three splits share one normalisation — same as the univariate path
        bundles[name] = {
            'train': loaders['train'],
            'val': loaders['val'],
            'test': loaders['test'],
            'channels': [cols[i] for i in idx],
            'n_channels': len(idx),
            'n_train': len(loaders['train'].dataset),
            'n_test': len(loaders['test'].dataset),
        }
    return bundles
