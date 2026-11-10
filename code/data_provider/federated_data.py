"""
Federated client data construction for FedTDO.

v1 recommendation from the plan: univariate clients with identical model configs
so expert cores remain structurally compatible across clients.
"""

from __future__ import annotations

import copy
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from utils.timefeatures import time_features


class Dataset_UnivariateClient(Dataset):
    """Single-channel time series client with standard train/val/test borders."""

    def __init__(
        self,
        values: np.ndarray,
        stamps: np.ndarray,
        flag: str,
        seq_len: int,
        label_len: int,
        pred_len: int,
        scale: bool = True,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
    ):
        assert flag in ('train', 'val', 'test')
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.flag = flag

        n = len(values)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        borders = {
            'train': (0, n_train),
            'val': (n_train - seq_len, n_train + n_val),
            'test': (n_train + n_val - seq_len, n),
        }
        # ETT-style fixed borders when series is long enough (hourly ETT length ~17420)
        # Prefer ratio split for general custom / channel clients.
        b1, b2 = borders[flag]

        series = values.reshape(-1, 1).astype(np.float64)
        self.scaler = StandardScaler()
        if scale:
            train_slice = values[:n_train].reshape(-1, 1).astype(np.float64)
            self.scaler.fit(train_slice)
            series = self.scaler.transform(series)
        else:
            series = series.astype(np.float64)

        self.data_x = series[b1:b2]
        self.data_y = series[b1:b2]
        self.data_stamp = stamps[b1:b2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        return (
            self.data_x[s_begin:s_end],
            self.data_y[r_begin:r_end],
            self.data_stamp[s_begin:s_end],
            self.data_stamp[r_begin:r_end],
        )

    def __len__(self):
        return max(0, len(self.data_x) - self.seq_len - self.pred_len + 1)


class Dataset_MultiChannelClient(Dataset):
    """Client holding ``k`` channels, trained by one shared trunk (``enc_in = k``).

    Mirrors :class:`Dataset_UnivariateClient` (same 7:1:2 borders, scaler fitted
    on the train slice only) but keeps the ``k`` columns.  Every client gets the
    *same* budget ``k``, which is what keeps parameter tensors averagable across
    clients while restoring the cross-variate paths a single-variate client
    structurally cannot use.
    """

    def __init__(
        self,
        values: np.ndarray,
        stamps: np.ndarray,
        flag: str,
        seq_len: int,
        label_len: int,
        pred_len: int,
        scale: bool = True,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
    ):
        assert flag in ('train', 'val', 'test')
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.flag = flag

        n = len(values)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        borders = {
            'train': (0, n_train),
            'val': (n_train - seq_len, n_train + n_val),
            'test': (n_train + n_val - seq_len, n),
        }
        b1, b2 = borders[flag]

        series = np.asarray(values, dtype=np.float64)
        if series.ndim == 1:
            series = series.reshape(-1, 1)
        self.scaler = StandardScaler()
        if scale:
            self.scaler.fit(series[:n_train])
            series = self.scaler.transform(series)

        self.data_x = series[b1:b2]
        self.data_y = series[b1:b2]
        self.data_stamp = stamps[b1:b2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        return (
            self.data_x[s_begin:s_end],
            self.data_y[r_begin:r_end],
            self.data_stamp[s_begin:s_end],
            self.data_stamp[r_begin:r_end],
        )

    def __len__(self):
        return max(0, len(self.data_x) - self.seq_len - self.pred_len + 1)


def resolve_channels_per_client(args) -> int:
    """Channels each client holds.  ``1`` reproduces the univariate protocol."""
    raw = getattr(args, 'fedtdo_channels_per_client', None)
    if raw is None:
        raw = getattr(args, 'channels_per_client', 1)
    return max(1, int(raw or 1))


def _load_dataframe(root_path: str, data_path: str) -> pd.DataFrame:
    path = os.path.join(root_path, data_path)
    df = pd.read_csv(path)
    return df


def _make_stamps(df_raw: pd.DataFrame, freq: str, timeenc: int) -> np.ndarray:
    dates = pd.to_datetime(df_raw['date'].values)
    if timeenc == 0:
        stamp = pd.DataFrame({'date': dates})
        stamp['month'] = stamp.date.dt.month
        stamp['day'] = stamp.date.dt.day
        stamp['weekday'] = stamp.date.dt.weekday
        stamp['hour'] = stamp.date.dt.hour
        return stamp.drop(columns=['date']).values.astype(np.float32)
    data_stamp = time_features(dates, freq=freq)
    return data_stamp.transpose(1, 0).astype(np.float32)


def build_federated_clients_from_channels(
    args,
    *,
    num_clients: Optional[int] = None,
    channel_indices: Optional[List[int]] = None,
    batch_size: Optional[int] = None,
    channel_sampling: str = 'random',
    seed: Optional[int] = None,
    channels_per_client: int = 1,
) -> Dict[str, Dict[str, DataLoader]]:
    """
    Split multivariate CSV columns into FedTDO clients.

    ``channels_per_client`` (default 1) decides how many columns each client
    holds.  ``1`` is the original univariate protocol and is reproduced
    bit-for-bit; ``k > 1`` requires ``num_clients * k`` columns and gives every
    client the same ``enc_in = k``, so parameter tensors stay averagable while
    the cross-variate paths (relation layer, variate mixing) stop being
    degenerate.  Both FedTDO and every FL baseline call this function, so the
    changed split is automatically symmetric across the two tables.

    ``channel_sampling`` decides which columns become clients when the dataset
    has more channels than the protocol consumes:

    * ``random`` (default): a seeded random subset.  Taking ``cols[:N]`` on
      ECL / Traffic would silently pick physically adjacent meters or sensors,
      which understates heterogeneity and makes the migration problem easier
      than it really is.
    * ``head``: the old ``cols[:N]`` behaviour, kept for reproducing earlier
      ETT runs.

    Returns:
        {client_id: {'train': loader, 'val': loader, 'test': loader}}
    """
    df_raw = _load_dataframe(args.root_path, args.data_path)
    timeenc = 0 if args.embed != 'timeF' else 1
    stamps = _make_stamps(df_raw, args.freq, timeenc)

    cols = list(df_raw.columns[1:])  # drop date
    if channel_indices is None:
        channel_indices = list(range(len(cols)))

    bs = int(batch_size if batch_size is not None else args.batch_size)
    k = max(1, int(channels_per_client or 1))
    # The one-client-per-column path keeps its original selection semantics.
    # With k > 1 the budget is num_clients * k columns, chosen below, so this
    # pre-truncation must not run (it would cap the pool at num_clients).
    if k == 1 and num_clients is not None:
        wanted = int(num_clients)
        if wanted < len(channel_indices):
            if channel_sampling == 'head':
                channel_indices = channel_indices[:wanted]
            elif channel_sampling == 'random':
                rng = np.random.default_rng(
                    int(seed if seed is not None else getattr(args, 'seed', 2021))
                )
                picked = rng.choice(len(channel_indices), size=wanted, replace=False)
                channel_indices = sorted(channel_indices[i] for i in picked)
            else:
                raise ValueError(f'unknown channel_sampling {channel_sampling!r}')
        else:
            channel_indices = channel_indices[:wanted]

    if k == 1:
        groups = [[idx] for idx in channel_indices]
    else:
        client_count = (int(num_clients) if num_clients is not None
                        else max(1, len(channel_indices) // k))
        need = client_count * k
        if need > len(channel_indices):
            raise ValueError(
                f'num_clients({client_count}) x channels_per_client({k}) = {need} '
                f'needs {need} channels but only {len(channel_indices)} are available'
            )
        pool = list(channel_indices)
        if need < len(pool):
            if channel_sampling == 'head':
                pool = pool[:need]
            elif channel_sampling == 'random':
                rng = np.random.default_rng(
                    int(seed if seed is not None else getattr(args, 'seed', 2021))
                )
                picked = rng.choice(len(pool), size=need, replace=False)
                pool = sorted(pool[i] for i in picked)
            else:
                raise ValueError(f'unknown channel_sampling {channel_sampling!r}')
        groups = [pool[i:i + k] for i in range(0, need, k)]

    clients = {}
    for group in groups:
        if k == 1:
            idx = group[0]
            col = cols[idx]
            values = df_raw[col].values.astype(np.float64)
            cid = f'c{idx}_{col}'
            dataset_cls = Dataset_UnivariateClient
        else:
            values = df_raw[[cols[i] for i in group]].values.astype(np.float64)
            cid = f'c{group[0]}-{group[-1]}_{cols[group[0]]}_{k}ch'
            dataset_cls = Dataset_MultiChannelClient
        loaders = {}
        for flag in ('train', 'val', 'test'):
            ds = dataset_cls(
                values=values,
                stamps=stamps,
                flag=flag,
                seq_len=args.seq_len,
                label_len=args.label_len,
                pred_len=args.pred_len,
                scale=True,
            )
            shuffle = flag == 'train'
            nw = int(getattr(args, 'num_workers', 0) or 0)
            # Train needs prefetch; val/test keep a few workers to avoid 4×3×nw explosion
            if flag != 'train':
                nw = min(nw, 4) if nw > 0 else 0
            batch = bs
            if flag != 'train':
                # Short series (ILI has ~74 test windows) cannot fill 60 rounds at
                # batch 32, so the stream runs dry and the online stage collapses
                # into cold start.  The stream batch only controls how many windows
                # one round consumes; training keeps ``bs`` so BatchNorm stays sane.
                #
                # The val split is fed as an online stream only when
                # ``--fedtdo_eval_split val``.  Otherwise val is used for early
                # stopping once per downstream epoch, and shrinking it to the
                # stream batch turns a single batched pass into thousands of
                # single-window forwards per client per epoch -- at
                # ``--fedtdo_stream_batch 1`` that is a ~200x slowdown of the
                # whole run for no protocol benefit.
                eval_split = getattr(args, 'fedtdo_eval_split', 'test')
                if flag == eval_split:
                    stream_batch = int(getattr(args, 'fedtdo_stream_batch', 0) or 0)
                    if stream_batch > 0:
                        batch = stream_batch
            dl_kwargs = dict(
                batch_size=batch,
                shuffle=shuffle,
                num_workers=nw,
                drop_last=False,
                pin_memory=bool(getattr(args, 'use_gpu', False)),
            )
            if nw > 0:
                dl_kwargs['persistent_workers'] = True
                dl_kwargs['prefetch_factor'] = 4
            loaders[flag] = DataLoader(ds, **dl_kwargs)
        clients[cid] = loaders
        print(f'[FedTDO data] client={cid} train={len(loaders["train"].dataset)} '
              f'val={len(loaders["val"].dataset)} test={len(loaders["test"].dataset)}')
    return clients


def make_client_args(base_args, enc_in: Optional[int] = None):
    """Clone args with the client's own ``enc_in``/``c_out``.

    ``enc_in=None`` resolves from ``--channels_per_client`` /
    ``--fedtdo_channels_per_client`` (default 1 = the univariate protocol), so
    every FL baseline automatically trains on the same number of channels its
    loader hands it.  Pass an explicit value to override.
    """
    width = resolve_channels_per_client(base_args) if enc_in is None else int(enc_in)
    args = copy.copy(base_args)
    args.enc_in = width
    args.dec_in = width
    args.c_out = width
    args.features = 'S'
    args.channel_independence = 1
    # Transferable experts should not bind relation layers to n_vars
    args.use_relation_layer = getattr(base_args, 'fedtdo_use_relation_layer', 0)
    return args
