"""
Unified FL-baseline experiment under the FedTDO protocol.

Same channel-as-client split, Dynamic-TMoE backbone, and online MSE/MAE
evaluation as Exp_FedTDO — so results are comparable.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from fl_baselines.registry import get_baseline, list_baselines
from data_provider.federated_data import (
    build_federated_clients_from_channels,
    resolve_channels_per_client,
)
from utils.metrics import metric
from utils.paper_recorder import PaperRecorder


class Exp_FL_Baselines:
    def __init__(self, args):
        self.args = args
        # Columns per client.  Must be applied *before* the method is built, so
        # every baseline's backbone gets the same ``enc_in`` as the data it is
        # trained on.  1 reproduces the univariate tables on disk.
        self.channels_per_client = resolve_channels_per_client(args)
        if self.channels_per_client > 1:
            args.enc_in = self.channels_per_client
            args.dec_in = self.channels_per_client
            args.c_out = self.channels_per_client
        # GPU is mandatory: a CPU arm silently slipped into the table once and its
        # numbers were not comparable with the CUDA rows.
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required: CPU execution is disabled '
                               'project-wide.')
        gpu_index = min(int(getattr(args, 'gpu', 0) or 0),
                        torch.cuda.device_count() - 1)
        args.use_gpu = True
        args.gpu = gpu_index
        self.device = torch.device(f'cuda:{gpu_index}')
        print(f'Use GPU: {self.device} ({torch.cuda.get_device_name(gpu_index)})')

        self.method_name = str(getattr(args, 'baseline', 'fedavg')).lower()
        self.method = get_baseline(self.method_name, args, self.device)

    def _make_recorder(self):
        """Paper-grade event recording; disabled unless ``--paper_record 1``."""
        enabled = int(getattr(self.args, 'paper_record', 0) or 0)
        root = str(getattr(self.args, 'paper_results_dir', './results/paper_fedtdo'))
        scalar_args = {
            key: value for key, value in vars(self.args).items()
            if isinstance(value, (str, int, float, bool)) or value is None
        }
        recorder = PaperRecorder(
            root=root, method=str(self.method_name),
            dataset=str(getattr(self.args, 'data', 'unknown')),
            seed=int(getattr(self.args, 'seed', 0)),
            run_kind='baseline', enabled=bool(enabled),
            extra_config=scalar_args)
        if recorder.enabled:
            recorder.set_config(
                num_clients=int(getattr(self.args, 'num_clients', 0) or 0),
                channels_per_client=int(self.channels_per_client),
                seq_len=int(getattr(self.args, 'seq_len', 0)),
                pred_len=int(getattr(self.args, 'pred_len', 0)),
                model_name=str(getattr(self.args, 'backbone', 'dtmoe')),
                online_enabled=bool(float(getattr(self.args, 'fl_online_lr', 0.0) or 0.0) > 0),
                online_lr=float(getattr(self.args, 'fl_online_lr', 0.0) or 0.0),
                online_steps=int(getattr(self.args, 'fedtdo_online_steps', 0) or 0),
                stream_batch=int(getattr(self.args, 'fedtdo_stream_batch', 0) or 0),
                label_delay=(1 if str(getattr(self.args,'fl_online_target','first'))=='first'
                                else int(getattr(self.args, 'pred_len', 1))),
                controller_extra_delay=max(0, int(getattr(self.args, 'fedtdo_feedback_delay', 0) or 0)),
                feedback_protocol=('one_step' if str(getattr(self.args,'fl_online_target','first'))=='first'
                                   else 'delayed_full_horizon'),
                transfer_enabled=False,
                share_modules='',
                optimizer='SGD(online)',
                gradient_steps_per_update=1,
            )
        return recorder

    def _build_loaders(self) -> Dict[str, Dict]:
        # Keep the client subset identical to the FedTDO runner: same seeded
        # random channel sampling, otherwise the two tables describe different
        # sets of sensors and cannot be compared.
        return build_federated_clients_from_channels(
            self.args,
            num_clients=int(getattr(self.args, 'num_clients', 4)),
            batch_size=getattr(self.args, 'batch_size', 32),
            channels_per_client=self.channels_per_client,
            channel_sampling=getattr(self.args, 'fedtdo_channel_sampling', 'random'),
            seed=getattr(self.args, 'seed', 2021),
        )

    def train(self):
        loaders = self._build_loaders()
        self.method.setup_clients(loaders)
        print(
            f'[Baseline] method={self.method.name} venue={self.method.venue} '
            f'clients={len(self.method.clients)} desc={self.method.short_desc}'
        )
        self.method.train_offline()

    @torch.no_grad()
    def online_evaluate(self, setting: str) -> dict:
        max_steps = int(getattr(self.args, 'fedtdo_online_steps',
                                getattr(self.args, 'fl_online_steps', 100)))
        client_metrics = {}
        all_logs: List[dict] = []
        recorder = self._make_recorder()

        for cid in self.method.client_ids():
            client = self.method.clients[cid]
            print(f'[{self.method.name}] online eval client={cid}')
            preds, trues = [], []
            # ``fedtdo_online_steps`` is a *batch* budget, exactly like the
            # ``served`` counter in Exp_FedTDO.  Counting samples here would
            # evaluate 32× fewer windows per client and make the two tables
            # incomparable.
            steps = 0      # zero-based forecast origin within this client's stream
            batches = 0
            pending = []
            # Reveal horizon.  ``first`` = timestamp-level feedback: only y(t+1)
            # is consumed, and it is consumed one window later, matching the
            # FedTDO arm exactly.  Anything else falls back to the full-horizon
            # delayed protocol, which is *not* comparable to it.
            if str(getattr(self.args, 'fl_online_target', 'first')) == 'first':
                horizon = 1
            else:
                horizon = int(getattr(self.args, 'pred_len', 1))
            feedback_delay = horizon + max(0, int(getattr(self.args, 'fedtdo_feedback_delay', 0) or 0))
            online_lr = float(getattr(self.args, 'fl_online_lr', 0.0) or 0.0)

            def release(item, available_idx):
                due_idx, old_origin, old_x, old_y, old_xm, old_pred, old_log = item
                update_started = time.perf_counter()
                if online_lr > 0:
                    self.method.online_step(cid, old_x, old_y, old_xm, online_lr)
                update_ms = (time.perf_counter() - update_started) * 1000.0
                adapt_info = self.method.online_adapt(
                    cid, old_x, old_y, old_xm, old_origin, old_pred)
                if adapt_info:
                    old_log['adapt'] = {
                        k: (v if not isinstance(v, torch.Tensor) else v.tolist())
                        for k, v in adapt_info.items()
                    }
                if recorder.enabled and (online_lr > 0 or adapt_info):
                    recorder.online_update(
                        client_id=cid, window_idx=old_origin + 1,
                        gradient_steps=1 if online_lr > 0 else 0,
                        learning_rate=online_lr,
                        batch_size=1, duration_ms=update_ms,
                        optimizer='SGD' if online_lr > 0 else 'method_adaptation',
                        update_type=('one_step' if str(getattr(self.args,'fl_online_target','first'))=='first' else 'delayed_full_horizon'),
                        target_start_idx=old_origin + 1,
                        target_end_idx=old_origin + horizon,
                        label_available_idx=old_origin + horizon,
                        update_time_idx=available_idx)
                    recorder.cost(client_id=cid, window_idx=old_origin + 1,
                                  operation='adaptation', duration_ms=update_ms)

            for batch in client.test_loader:
                if batches >= max_steps:
                    break
                batches += 1
                batch_x, batch_y, batch_x_mark, _ = batch
                B = batch_x.shape[0]
                for b in range(B):
                    x = batch_x[b:b + 1]
                    y = batch_y[b:b + 1]
                    xm = batch_x_mark[b:b + 1]
                    origin_idx = steps

                    # Release only forecasts whose complete H-step target has
                    # arrived.  This happens before the next prediction.
                    while pending and pending[0][0] <= origin_idx:
                        release(pending.pop(0), origin_idx)

                    predict_started = time.perf_counter()
                    pred = self.method.predict(cid, x, xm)
                    predict_ms = (time.perf_counter() - predict_started) * 1000.0

                    f_dim = -1 if client.args.features == 'MS' else 0
                    yt = y[:, -client.args.pred_len:, f_dim:].float()
                    pred_np = (pred.detach().cpu().numpy()
                               if isinstance(pred, torch.Tensor) else np.asarray(pred))
                    true_np = yt.detach().cpu().numpy()
                    preds.append(pred_np)
                    trues.append(true_np)

                    if recorder.enabled:
                        position = origin_idx + 1
                        recorder.window(
                            client_id=cid, window_idx=position,
                            y_true=true_np[0], y_pred=pred_np[0],
                            x_history=x[0].detach().cpu().numpy(),
                            label_reveal_idx=origin_idx + horizon,
                            target_start_idx=origin_idx + 1,
                            target_end_idx=origin_idx + horizon)
                        recorder.cost(client_id=cid, window_idx=position,
                                      operation='forecasting',
                                      duration_ms=predict_ms)

                    log = {
                        'client_id': cid,
                        't': origin_idx,
                        'method': self.method.name,
                        'instant_mse': float(np.mean((pred_np - true_np) ** 2)),
                    }
                    all_logs.append(log)
                    pending.append((origin_idx + feedback_delay, origin_idx,
                                    x.detach(), y.detach(), xm.detach(),
                                    pred.detach() if isinstance(pred, torch.Tensor) else pred,
                                    log))
                    steps += 1

            # The final H labels mature after the last forecast origin.  Drain
            # the queue only at their simulated availability times; no later
            # forecast can be affected by these tail updates.
            while pending:
                item = pending.pop(0)
                release(item, item[0])

            if preds:
                preds_np = np.concatenate(preds, axis=0)
                trues_np = np.concatenate(trues, axis=0)
                mae, mse, rmse, mape, mspe = metric(preds_np, trues_np)
                client_metrics[cid] = {
                    'mae': float(mae),
                    'mse': float(mse),
                    'rmse': float(rmse),
                }
                print(f'  {cid}: mse={mse:.6f} mae={mae:.6f}')

        mses = [v['mse'] for v in client_metrics.values()]
        maes = [v['mae'] for v in client_metrics.values()]
        # Client errors are heavy-tailed on Traffic (one sensor at +39 sigma drags
        # the mean by an order of magnitude), so the mean and the median are both
        # reported and they can disagree in sign across methods.
        summary = {
            'setting': setting,
            'method': self.method.name,
            'venue': self.method.venue,
            'mean_mse': float(np.mean(mses)) if mses else None,
            'mean_mae': float(np.mean(maes)) if maes else None,
            'median_mse': float(np.median(mses)) if mses else None,
            'median_mae': float(np.median(maes)) if maes else None,
            'std_mse': float(np.std(mses)) if mses else None,
            'max_mse': float(np.max(mses)) if mses else None,
            'min_mse': float(np.min(mses)) if mses else None,
            'num_clients': len(client_metrics),
            'clients': client_metrics,
            'method_summary': self.method.summary(),
            # Without this the two budgets are indistinguishable from disk, and a
            # step-mode row silently lands in an epoch-mode table.
            'train': {
                'step_mode': getattr(self.args, 'fl_step_mode', 'steps'),
                'rounds': getattr(self.args, 'fl_rounds', None),
                'local_epochs': getattr(self.args, 'fl_local_epochs', None),
                'local_max_steps': getattr(self.args, 'fl_local_max_steps', None),
                'patience': getattr(self.args, 'fl_patience', None),
                'lradj': getattr(self.args, 'fl_lradj', None),
                'best_val': getattr(self.method, 'best_val_loss', None),
            },
            'num_online_logs': len(all_logs),
            'available_baselines': list_baselines(),
        }

        if recorder.enabled:
            recorder.finish(summary={
                'overall_mse': summary.get('mean_mse'),
                'overall_mae': summary.get('mean_mae'),
                'per_client': {cid: {k: float(v) for k, v in vals.items()
                                     if isinstance(v, (int, float))}
                               for cid, vals in client_metrics.items()},
                'num_clients': len(client_metrics),
            })
        out_dir = os.path.join(getattr(self.args, 'results_dir', './results_baselines'), setting)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        with open(os.path.join(out_dir, 'online_logs.json'), 'w', encoding='utf-8') as f:
            json.dump(all_logs, f, indent=2, ensure_ascii=False)
        print(f'[Baseline] saved -> {out_dir}')
        return summary

    def run(self, setting: str) -> dict:
        self.train()
        return self.online_evaluate(setting)
