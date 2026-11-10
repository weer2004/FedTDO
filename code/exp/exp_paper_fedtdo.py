"""FedTDO under the dataset-level multivariate protocol.

Why this file exists
--------------------
``Exp_FedTDO`` implements the full method (PatchEmbedding warm-up + FedAvg +
freeze + expert capsules + transfer controller + online counterfactual
decisions) but only on the *channel-as-client* partition, i.e. ``features S``
with ``enc_in = 1``.  ``Exp_PaperProtocol`` implements the FFTS/FeDaL
dataset-level protocol (``features M``, ``enc_in = C``) but only for FedAvg and
``local``.

A table that puts the TSLib backbones next to FedTDO therefore needs both halves
in one runner.  This subclass keeps every line of ``Exp_FedTDO``'s training,
transfer and decision logic and only replaces the partition: clients become
datasets, tensors keep all seven channels.

What is shared with the backbone table
--------------------------------------
* identical data: ``build_paper_clients`` (same channel picker, same 7:1:2
  chronological split, same scaler) as ``run_paper_protocol.py``;
* identical budget: ``fedtdo_warmup_steps + fedtdo_local_steps = 1600`` gradient
  steps per client, matching ``--fl_local_max_steps 800 --fl_rounds 2``;
* identical architecture hyper-parameters (``d_model 32``, ``patch_len 16``,
  ``stride 16``, ``num_experts 4``, ``top_k 2``) so the only difference from a
  ``fedavg`` row is the algorithm.

What is *not* shared
--------------------
The evaluation is online, one stream step at a time, with the same
``fedtdo_online_steps`` budget as the FL-axis table.  A backbone row is scored on
every test window; a FedTDO row is scored on the first ``online_steps`` batches
of each dataset's test stream.  The two are not the same estimator, so this
runner also reports the *same* windows under a plain batched pass
(``offline_metrics``) to make the comparison apples to apples.
"""

from __future__ import annotations

import copy
import os

import numpy as np
import torch

from data_provider.multivariate_data import build_paper_clients
from exp.exp_fedtdo import Exp_FedTDO
from utils.metrics import metric

DATASETS = {
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

MAIN_DATASETS = list(DATASETS)


class Exp_PaperFedTDO(Exp_FedTDO):
    """FedTDO with one multivariate dataset per client."""

    def __init__(self, args):
        super().__init__(args)
        self.datasets = list(getattr(args, 'datasets', MAIN_DATASETS))
        self.holdout = getattr(args, 'holdout', None) or None
        # ``Model`` reads a handful of fields off the config object; the
        # standalone parser supplies them, here they have to be spelled out.
        args.task_name = getattr(args, 'task_name', 'long_term_forecast')
        args.model = getattr(args, 'model', 'Dynamic_TMoE')
        args.des = getattr(args, 'des', 'pp')
        for key, default in (('cycle_length', 24), ('num_experts', 4),
                             ('num_temporal_moe_layers', 1), ('use_relation_layer', 0),
                             ('channel_independence', 0), ('expand', 2),
                             ('d_conv', 4), ('activation', 'gelu'),
                             ('num_rnn_layers', 1), ('num_drift_experts', 1),
                             ('top_k', 2)):
            if not hasattr(args, key):
                setattr(args, key, default)
        self._bundles = None

    # -- partition -----------------------------------------------------------
    def _prepare_partition(self):
        specs = {d: DATASETS[d] for d in self.datasets}
        bundles = build_paper_clients(
            specs,
            channels=int(getattr(self.args, 'paper_channels', 7)),
            seq_len=int(self.args.seq_len), label_len=int(self.args.label_len),
            pred_len=int(self.args.pred_len),
            batch_size=int(self.args.batch_size),
            eval_batch_size=int(getattr(self.args, 'eval_batch_size', 128)),
            seed=int(getattr(self.args, 'seed', 2021)),
            num_workers=int(getattr(self.args, 'num_workers', 0)),
        )
        self._bundles = bundles
        self.train_names = [d for d in self.datasets if d != self.holdout]

        model_args = copy.deepcopy(self.args)
        model_args.features = 'M'
        model_args.enc_in = int(bundles[self.datasets[0]]['n_channels'])
        model_args.dec_in = model_args.enc_in
        model_args.c_out = model_args.enc_in
        # every client must expose one capsule space, so a shared width is a
        # hard requirement of the protocol, not an accident
        for name in self.datasets:
            if int(bundles[name]['n_channels']) != model_args.enc_in:
                raise RuntimeError(
                    f'{name} has {bundles[name]["n_channels"]} channels but '
                    f'{self.datasets[0]} has {model_args.enc_in}; a shared expert '
                    'capsule needs one channel count across clients')
        self.model_args = model_args

        names = list(self.datasets)
        train_loaders = [bundles[n]['train'] for n in names]
        test_loaders = [bundles[n]['test'] for n in names]
        return names, train_loaders, test_loaders

    # -- extra evaluation ----------------------------------------------------
    @torch.no_grad()
    def offline_evaluate(self):
        """Batched pass over *every* test window, same estimator as the backbone table.

        The online loop only sees ``fedtdo_online_steps`` batches, so its MSE and
        a backbone row's MSE are different statistics.  This pass exists purely
        so the two can be put in one table without the reader having to know.
        """
        rows = {}
        for name, client in zip(self.datasets, self._clients):
            preds, trues = [], []
            for batch in client._offline_loader:
                batch_x, batch_y, batch_x_mark, _ = batch
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                out = client.model(batch_x, batch_x_mark)
                preds.append(out[:, -self.args.pred_len:, :].detach().float().cpu().numpy())
                trues.append(batch_y[:, -self.args.pred_len:, :].float().cpu().numpy())
            preds_np = np.concatenate(preds, axis=0)
            trues_np = np.concatenate(trues, axis=0)
            mae, mse, rmse, _, _ = metric(preds_np, trues_np)
            rows[name] = {'mse': float(mse), 'mae': float(mae), 'rmse': float(rmse),
                          'n_windows': int(preds_np.shape[0]),
                          'n_channels': int(preds_np.shape[-1])}
            print(f'  [offline {name}] mse={mse:.4f} mae={mae:.4f} '
                  f'({preds_np.shape[0]}x{preds_np.shape[-1]})')
        return rows

    def run(self, setting: str):
        # keep the client objects around so ``offline_evaluate`` can reuse the
        # exact models the online loop ended with
        original_build = self._build_clients
        holder = {}

        def build(names, loaders):
            clients = original_build(names, loaders)
            holder['clients'] = clients
            return clients

        self._build_clients = build
        try:
            result = super().run()
        finally:
            self._build_clients = original_build

        self._clients = holder['clients']
        for client, name in zip(self._clients, self.datasets):
            client._offline_loader = self._bundles[name]['test']
        offline = self.offline_evaluate()
        trained = [r['mse'] for n, r in offline.items() if n in self.train_names]
        result['offline_per_dataset'] = offline
        result['offline_mean_mse_trained'] = float(np.mean(trained)) if trained else None
        result['partition'] = 'dataset-level multivariate'
        result['protocol'] = 'dataset-level multivariate (FFTS/FeDaL style, FedTDO)'
        result['paper_channels'] = int(getattr(self.args, 'paper_channels', 7))
        result['datasets'] = self.datasets

        # ``super().run`` already wrote metrics.json under the arm directory;
        # rewrite it with the offline block included.  The online numbers must
        # not be silently replaced, so both stay in the file.
        import json
        output_dir = os.path.join(self.args.fedtdo_output_dir, self.args.model_id)
        with open(os.path.join(output_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        import csv
        with open(os.path.join(output_dir, 'per_dataset.csv'), 'w', newline='',
                  encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['dataset', 'offline_mse', 'offline_mae', 'n_windows',
                        'n_channels', 'online_mse', 'online_mae'])
            for name in self.datasets:
                off = offline.get(name, {})
                on = result['client_metrics'].get(name, {})
                w.writerow([name,
                            f"{off.get('mse', float('nan')):.6f}",
                            f"{off.get('mae', float('nan')):.6f}",
                            off.get('n_windows', ''), off.get('n_channels', ''),
                            f"{on.get('mse', float('nan')):.6f}",
                            f"{on.get('mae', float('nan')):.6f}"])
        print(f'[paper-fedtdo] offline mean (trained) = '
              f'{result["offline_mean_mse_trained"]:.4f}')
        return result
