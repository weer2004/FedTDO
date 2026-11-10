"""Federated training / evaluation under the FFTS-FeDaL protocol.

What changes relative to ``Exp_FL_Baselines``
---------------------------------------------
==========================================  ==========================  =========================
                                            ``Exp_FL_Baselines``        here
==========================================  ==========================  =========================
client                                       one channel of one dataset  one multivariate dataset
model input / output                         ``features S``, ``enc_in=1``  ``features M``, ``enc_in=C``
heterogeneity                                statistical (per sensor)    dataset/domain level
evaluation                                   online, one sample at a time  batched sliding windows
metric aggregation                           mean over clients           mean over clients, per dataset
==========================================  ==========================  =========================

Everything else is deliberately shared with the existing baseline path: the same
``BaselineClient`` (so the local optimiser, step budget and loss are identical),
the same ``average_state_dicts`` aggregator, the same ``utils.metrics.metric``.
That is what makes a backbone-axis number here comparable with the FL-axis table
on the other side.

The federated algorithm is **FedAvg** (``--method fedavg``) or no communication at
all (``--method local``).  The point of this runner is the *backbone* axis, so the
aggregator is held fixed; the FL-method axis is already covered by
``Exp_FL_Baselines``.
"""

from __future__ import annotations

import copy
import csv
import json
import os
from typing import Dict, List

import numpy as np
import torch

from data_provider.multivariate_data import build_paper_clients
from fl_baselines.fl_client import (BaselineClient, average_state_dicts,
                                     weighted_average_state_dicts)
from utils.metrics import metric

# name -> (root_path, data_path) -- identical to scripts/xc_sweep.DATASETS so the
# two protocols read byte-identical files
DATASETS: Dict[str, tuple] = {
    'ETTh1': ('./dataset/ETT-small/', 'ETTh1.csv'),
    'ETTh2': ('./dataset/ETT-small/', 'ETTh2.csv'),
    'ETTm1': ('./dataset/ETT-small/', 'ETTm1.csv'),
    'ETTm2': ('./dataset/ETT-small/', 'ETTm2.csv'),
    'Weather': ('./dataset/weather/', 'weather.csv'),
    'Exchange': ('./dataset/exchange_rate/', 'exchange_rate.csv'),
    'ILI': ('./dataset/illness/', 'national_illness.csv'),
    'ECL': ('./dataset/electricity/', 'electricity.csv'),
    'Traffic': ('./dataset/traffic/', 'traffic.csv'),
}

MAIN_DATASETS = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Exchange', 'ILI',
                 'ECL', 'Traffic']


class Exp_PaperProtocol:
    def __init__(self, args):
        self.args = args
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required: CPU execution is disabled '
                               'project-wide.')
        gpu_index = min(int(getattr(args, 'gpu', 0) or 0),
                        torch.cuda.device_count() - 1)
        args.use_gpu = True
        args.gpu = gpu_index
        self.device = torch.device(f'cuda:{gpu_index}')
        print(f'[paper] device={self.device}')

        self.datasets = list(getattr(args, 'datasets', MAIN_DATASETS))
        self.holdout = getattr(args, 'holdout', None) or None
        train_names = [d for d in self.datasets if d != self.holdout]

        specs = {d: DATASETS[d] for d in self.datasets}
        bundles = build_paper_clients(
            specs,
            channels=int(getattr(args, 'paper_channels', 7)),
            seq_len=int(args.seq_len), label_len=int(args.label_len),
            pred_len=int(args.pred_len),
            batch_size=int(args.batch_size),
            eval_batch_size=int(getattr(args, 'eval_batch_size', 128)),
            seed=int(getattr(args, 'seed', 2021)),
            num_workers=int(getattr(args, 'num_workers', 0)),
        )
        self.bundles = bundles
        self.train_names = train_names

        # one client per training dataset; a hold-out dataset gets a client too,
        # but it never trains -- the global model is evaluated on it unchanged
        # (cross-dataset / zero-shot generalisation, the FeDaL setting)
        self.clients = {}
        for name in self.datasets:
            cargs = copy.deepcopy(args)
            cargs.features = 'M'
            cargs.enc_in = int(bundles[name]['n_channels'])
            cargs.c_out = cargs.enc_in
            cargs.dec_in = cargs.enc_in
            cargs.data = name
            cargs.data_path = DATASETS[name][1]
            cargs.root_path = DATASETS[name][0]
            # the dtmoe backbone reads a few fields straight off ``configs``
            # (``configs.task_name`` first of all); the standalone runner gets
            # them from its own parser, here they have to be spelled out
            # ``BaselineClient.local_train`` falls back to ``args.fl_local_max_steps``
            # whenever ``max_steps`` is 0, so the epoch budget has to switch the
            # *client's* copy of the flag off -- otherwise ``--local-mode epochs``
            # silently reproduces the 1600-step cell bit for bit
            if str(getattr(args, 'local_mode', 'steps')).lower() == 'epochs':
                cargs.fl_local_max_steps = 0
            cargs.task_name = getattr(args, 'task_name', 'long_term_forecast')
            cargs.model_id = getattr(args, 'model_id', 'paper_protocol')
            cargs.model = getattr(args, 'model', 'Dynamic_TMoE')
            cargs.des = getattr(args, 'des', 'pp')
            for key, default in (('cycle_length', 24), ('num_experts', 4),
                                 ('num_temporal_moe_layers', 1), ('use_relation_layer', 0),
                                 ('channel_independence', 0), ('expand', 2),
                                 ('d_conv', 4), ('activation', 'gelu'),
                                 ('num_rnn_layers', 1), ('num_drift_experts', 1),
                                 ('top_k', 2)):
                if not hasattr(cargs, key):
                    setattr(cargs, key, default)
            self.clients[name] = BaselineClient(
                name, cargs, self.device, bundles[name]['train'],
                bundles[name]['val'], bundles[name]['test'])

        self._check_averageable()
        self.method = str(getattr(args, 'method', 'fedavg')).lower()
        self.agg_weight = str(getattr(args, 'agg_weight', 'uniform')).lower()

    def _check_averageable(self) -> None:
        ref = None
        for name, client in self.clients.items():
            shapes = {k: tuple(v.shape) for k, v in client.state_dict().items()}
            if ref is None:
                ref = (name, shapes)
                continue
            if shapes != ref[1]:
                diff = {k for k in set(shapes) | set(ref[1])
                        if shapes.get(k) != ref[1].get(k)}
                raise RuntimeError(
                    f'client {name!r} state_dict does not match {ref[0]!r}: {sorted(diff)}. '
                    'Every client must share one architecture, i.e. the same '
                    '--paper_channels / d_model / pred_len.')

    # -- 2025 supervised methods ---------------------------------------------
    # Two backbones need something the plain ``forward(x)`` contract cannot give
    # them, so they are wired here rather than inside the model:
    #   * RAFT retrieves *training windows together with their futures*, and the
    #     model never sees a target -- the bank has to come from the dataset.
    #   * ST-MTM pre-trains with a masked-reconstruction objective before the
    #     supervised budget, which is a second loop.
    # Both hooks are no-ops for every other backbone.
    def _attach_retrieval_banks(self) -> None:
        from numpy.lib.stride_tricks import sliding_window_view

        cap = int(getattr(self.args, 'raft_bank_cap', 16384))
        for name in self.datasets:
            model = self.clients[name].model
            if not hasattr(model, 'set_retrieval_bank'):
                continue
            ds = self.bundles[name]['train']
            # ``build_paper_clients`` hands back DataLoaders, not Datasets
            ds = getattr(ds, 'dataset', ds)
            arr = np.ascontiguousarray(np.asarray(ds.data_x, dtype=np.float32))
            seq_len, pred_len = int(ds.seq_len), int(ds.pred_len)
            n = min(len(ds), cap)
            keys = sliding_window_view(arr, seq_len, axis=0)[:n]          # [n,C,L]
            vals = sliding_window_view(arr, pred_len, axis=0)[seq_len:seq_len + n]
            model.set_retrieval_bank(
                torch.from_numpy(np.ascontiguousarray(keys)),
                torch.from_numpy(np.ascontiguousarray(vals.transpose(0, 2, 1))),
            )
            print(f'[raft] {name}: bank {n} windows '
                  f'(topk={int(getattr(self.args, "raft_topk", 3))})')

    def _pretrain_ssl(self) -> None:
        steps = int(getattr(self.args, 'stmtm_pretrain_steps', 0))
        if steps <= 0:
            return
        for name in self.train_names:
            client = self.clients[name]
            model = client.model
            if not hasattr(model, 'ssl_loss'):
                continue
            opt = torch.optim.AdamW(
                model.parameters(),
                lr=float(getattr(self.args, 'learning_rate', 5e-4)))
            model.train()
            loader = client.train_loader
            it = iter(loader)
            loss = None
            for _ in range(steps):
                try:
                    batch = next(it)
                except StopIteration:
                    it = iter(loader)
                    batch = next(it)
                x = batch[0].float().to(self.device)
                loss = model.ssl_loss(x)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            print(f'[stmtm] {name}: {steps} SSL steps, '
                  f'last masked-recon loss {float(loss):.4f}')

    # -- training ------------------------------------------------------------
    def train(self) -> None:
        rounds = int(getattr(self.args, 'fl_rounds', 2))
        local_steps = int(getattr(self.args, 'fl_local_max_steps', 800))
        local_epochs = int(getattr(self.args, 'fl_local_epochs', 1))
        # FFTS / FeDaL budget local work in *epochs* (FFTS: 10 rounds x 5 epochs),
        # which makes a client's step count proportional to its dataset size.  The
        # step budget is kept as the default because it is the only thing that
        # makes this table comparable with the channel-as-client FL table.
        mode = str(getattr(self.args, 'local_mode', 'steps')).lower()
        names = self.train_names
        if not names:
            return
        self._attach_retrieval_banks()
        self._pretrain_ssl()
        for rnd in range(rounds):
            states, used = [], 0
            for name in names:
                client = self.clients[name]
                if mode == 'epochs':
                    used = client.local_train(epochs=local_epochs, max_steps=0)
                else:
                    used = client.local_train(max_steps=local_steps)
                states.append(client.state_dict())
            if self.method == 'local':
                print(f'[local] round {rnd + 1}/{rounds}: {used} steps/client, no aggregation')
                continue
            if self.agg_weight == 'size':
                # FeDaL/FFTS weight clients by the amount of data they hold; with
                # ILI at 676 training windows and Traffic at 12k the two choices
                # are far from equivalent
                weights = [self.bundles[n]['n_train'] for n in names]
                avg = weighted_average_state_dicts(states, weights)
            else:
                avg = average_state_dicts(states)
            for name in names:
                self.clients[name].load_state_dict(avg)
            print(f'[fedavg] round {rnd + 1}/{rounds}: {used} steps/client, aggregated '
                  f'{len(states)} clients')

    # -- evaluation ----------------------------------------------------------
    @torch.no_grad()
    def evaluate(self) -> Dict[str, Dict]:
        rows: Dict[str, Dict] = {}
        for name in self.datasets:
            client = self.clients[name]
            preds, trues = [], []
            for batch_x, batch_y, batch_x_mark, _ in client.test_loader:
                out = client.predict(batch_x, batch_x_mark)
                preds.append(out.detach().float().cpu().numpy())
                trues.append(batch_y[:, -client.args.pred_len:, 0:].float().cpu().numpy())
            preds_np = np.concatenate(preds, axis=0)
            trues_np = np.concatenate(trues, axis=0)
            mae, mse, rmse, _, _ = metric(preds_np, trues_np)
            rows[name] = {
                'mse': float(mse), 'mae': float(mae), 'rmse': float(rmse),
                'n_windows': int(preds_np.shape[0]),
                'n_channels': int(preds_np.shape[-1]),
                'trained': name in self.train_names,
                'channels': self.bundles[name]['channels'],
            }
            print(f'  [{name}] mse={mse:.4f} mae={mae:.4f} '
                  f'({preds_np.shape[0]} windows x {preds_np.shape[-1]} ch, '
                  f'{"train" if name in self.train_names else "HOLDOUT"})')
        return rows

    def run(self, setting: str) -> dict:
        if self.holdout:
            print(f'[paper] hold-out dataset {self.holdout!r} is evaluated zero-shot')
        self.train()
        rows = self.evaluate()
        trained = [r['mse'] for n, r in rows.items() if r['trained']]
        summary = {
            'setting': setting,
            'protocol': 'dataset-level multivariate (FFTS/FeDaL style)',
            'method': self.method,
            'backbone': str(getattr(self.args, 'backbone', 'dtmoe')),
            'datasets': self.datasets,
            'holdout': self.holdout,
            'seq_len': int(self.args.seq_len),
            'pred_len': int(self.args.pred_len),
            'paper_channels': int(getattr(self.args, 'paper_channels', 7)),
            'd_model': int(self.args.d_model),
            'fl_rounds': int(getattr(self.args, 'fl_rounds', 2)),
            'fl_local_max_steps': int(getattr(self.args, 'fl_local_max_steps', 800)),
            'learning_rate': float(self.args.learning_rate),
            'batch_size': int(self.args.batch_size),
            'seed': int(getattr(self.args, 'seed', 2021)),
            'params': sum(p.numel() for p in self.clients[self.datasets[0]].model.parameters()),
            'per_dataset': rows,
            'mean_mse_trained': float(np.mean(trained)) if trained else None,
            'mean_mae_trained': float(np.mean([r['mae'] for n, r in rows.items()
                                               if r['trained']])) if trained else None,
        }
        out_dir = os.path.join('./results_paper', setting)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        with open(os.path.join(out_dir, 'metrics.csv'), 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['dataset', 'mse', 'mae', 'rmse', 'n_windows', 'n_channels', 'trained'])
            for name, r in rows.items():
                w.writerow([name, f"{r['mse']:.6f}", f"{r['mae']:.6f}",
                            f"{r['rmse']:.6f}", r['n_windows'], r['n_channels'],
                            int(r['trained'])])
        print(f'[paper] saved -> {out_dir}')
        return summary
