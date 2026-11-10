"""Causal federated online experiment for FedTDO.

The baseline experiment remains untouched.  This runner builds one of two
client partitions, synchronizes only the PatchEmbedding during warm-up,
publishes expert cores, and evaluates counterfactual experts with delayed
labels.

* ``--fedtdo_partition channel`` (default, plan §10.1): every network channel /
  sensor is a client, and each client keeps its own chronological
  train/val/test split.  This is the protocol the FL baselines use, so numbers
  are comparable.
* ``--fedtdo_partition time``: one series sliced into contiguous blocks, used
  for the controlled regime-shift experiments of plan §10.2.

``--fedtdo_control`` selects the encoder pre-training path:

* ``fedtdo`` (default): warm up PatchEmbedding locally, FedAvg it, freeze it,
  then train experts/head.  This is the method under test.
* ``isomorphic``: no warm-up, no FedAvg, no freezing - every parameter is
  trained for the same total number of steps as the FedTDO path.  The only
  difference from the method arm is therefore the shared frozen encoder, which
  is the only fair way to ask whether migration itself pays off.  Combine with
  ``--fedtdo_disable_transfer`` to obtain the isomorphic local control whose
  default path is what ``run_baselines.py --baseline local`` measures.
* ``local_encoder``: same local training as ``isomorphic`` but with transfer
  forced on.  The encoder is private to each client and the expert capsule is
  the only shared unit, which is the alternative sharing topology to the
  method arm (see ``SOLAR_CONTROL_REPORT.md``).
"""

from __future__ import annotations

import copy
import json
import math
import os
import time
from collections import deque

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset

from data_provider.data_factory import data_provider
from data_provider.federated_data import (
    build_federated_clients_from_channels,
    make_client_args,
    resolve_channels_per_client,
)
from federated import CostModel, FedTDOClient, FedTDOController, ExpertRegistry
from federated.server import FedTDOServer, module_slots
from models.Dynamic_TMoE.model import Model
from utils.metrics import metric
from utils.paper_recorder import PaperRecorder


class Exp_FedTDO:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.server = FedTDOServer(ExpertRegistry(max_experts=args.fedtdo_registry_size))
        self.partition = getattr(args, 'fedtdo_partition', 'channel')
        # Columns each client holds.  1 = the original univariate protocol; k>1
        # gives every client an identical ``enc_in = k`` block so the shared
        # trunk sees cross-variate structure without breaking parameter averaging.
        self.client_channels = resolve_channels_per_client(args)
        self.control = getattr(args, 'fedtdo_control', 'fedtdo')
        if self.control not in ('fedtdo', 'fedtdo_nofreeze', 'fedtdo_noavg',
                                'fedtdo_joint', 'fedtdo_localfreeze',
                                'isomorphic', 'local_encoder'):
            raise ValueError(f'unknown fedtdo_control {self.control!r}')
        self.transfer_enabled = not bool(getattr(args, 'fedtdo_disable_transfer', 0))
        if self.control == 'local_encoder':
            # Arm under test in this file's design note: the shared unit is the
            # expert capsule, not the encoder.  Transfer is the whole point, so
            # the disable flag is ignored here (use ``isomorphic`` for the
            # matched no-transfer control).
            self.transfer_enabled = True
        self.model_args = args
        self.online_steps = getattr(args, 'fedtdo_online_steps', None)
        # ``test`` (default) keeps every existing table comparable.  ``val`` scores
        # the same online stream on the held-out split, which is what a
        # hyper-parameter sweep has to do: selecting on the stream you report is
        # test-set tuning.
        self.eval_split = getattr(args, 'fedtdo_eval_split', 'test')
        if self.eval_split not in ('val', 'test'):
            raise ValueError(f'unknown fedtdo_eval_split {self.eval_split!r}')
        # Sharing topology.  ``share_modules`` decides *what* is federated;
        # ``share_prox`` decides *how hard*: 0 keeps the historical hard freeze,
        # >0 keeps the shared weights trainable and pulls them back with
        # 0.5*lambda*||theta - theta_shared||^2 (soft sharing).  Defaults
        # reproduce the previous behaviour exactly.
        raw_modules = str(getattr(self.args, 'fedtdo_share_modules',
                                  'patch_embedding') or 'patch_embedding')
        self.share_modules = tuple(part.strip() for part in raw_modules.split(',')
                                   if part.strip())
        self.share_prox = float(getattr(self.args, 'fedtdo_share_prox', 0.0) or 0.0)
        # ``sync_every`` > 0 re-averages the shared modules during the downstream
        # stage instead of freezing them once after warm-up.
        self.sync_every = int(getattr(self.args, 'fedtdo_sync_every', 0) or 0)
        # Online step size on the freshly revealed label (prequential learning).
        # The stream drifts: per-step error on ETTh1/pl96 is ~0.32 for t0-29 and
        # ~0.44 for t30-59, so a static model pays for the drift it never tracks.
        self.online_lr = float(getattr(self.args, 'fedtdo_online_lr', 0.0) or 0.0)
        self.shared_snapshot = None
        self.round_log = []

    def _dataset(self, flag):
        dataset, _ = data_provider(self.args, flag)
        return dataset

    def _shards(self, dataset):
        # Every shard remains temporally ordered.  We do not randomize samples
        # in online evaluation, so no future label can affect an earlier choice.
        total = len(dataset)
        edges = np.linspace(0, total, self.args.fedtdo_num_clients + 1, dtype=int)
        shards = []
        for index in range(self.args.fedtdo_num_clients):
            indices = list(range(edges[index], edges[index + 1]))
            if not indices:
                raise ValueError('fedtdo_num_clients exceeds available windows')
            shards.append(Subset(dataset, indices))
        return shards

    def _loader(self, shard, shuffle=False, batch_size=None):
        return DataLoader(
            shard, batch_size=batch_size or self.args.batch_size, shuffle=shuffle,
            num_workers=self.args.num_workers, drop_last=False,
        )

    def _loss_inputs(self, batch):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        f_dim = -1 if self.model_args.features == 'MS' else 0
        target = batch_y[:, -self.args.pred_len:, f_dim:]
        return batch_x, batch_x_mark, target

    def _adjust_learning_rate(self, optimizer, epoch):
        """DTMoe's ``utils/tools.adjust_learning_rate``, reimplemented here.

        ``epoch`` is 1-based, exactly as in ``run.py`` (called after each epoch
        with ``epoch + 1``).  FedTDO used to run a constant lr for a fixed number
        of steps, which is not the DTMoe schedule and made its numbers
        incomparable with the published tables.  ``type3`` is the DTMoe default:
        constant for the first three epochs, then 0.8x per epoch.
        """
        base = float(self.args.learning_rate)
        schedule = getattr(self.args, 'fedtdo_lradj', 'type3')
        if schedule == 'type1':
            lr = base * (0.9 ** (epoch - 1))
        elif schedule == 'type3':
            lr = base if epoch <= 3 else base * (0.8 ** (epoch - 3))
        elif schedule == 'cosine':
            total = max(1, int(getattr(self.args, 'fedtdo_epochs', 60)))
            lr = base / 2 * (1 + math.cos(epoch / total * math.pi))
        elif schedule == 'constant':
            lr = base
        else:
            raise ValueError(f'unknown fedtdo_lradj {schedule!r}')
        for group in optimizer.param_groups:
            group['lr'] = lr
        return lr

    def _epoch_train(self, model, loader, optimizer, criterion,
                     prox_snapshot=None, prox_mu=0.0):
        model.train()
        total, count = 0.0, 0
        for batch in loader:
            batch_x, batch_x_mark, target = self._loss_inputs(batch)
            optimizer.zero_grad()
            output = model(batch_x, batch_x_mark)[:, :, -target.shape[-1]:]
            loss = criterion(output, target)
            if prox_snapshot and prox_mu > 0:
                penalty = self.server.proximal_penalty(
                    model, prox_snapshot, self.share_modules)
                if penalty is not None:
                    loss = loss + 0.5 * float(prox_mu) * penalty
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            count += 1
        return total / max(count, 1)

    def _epoch_valid(self, model, loader, criterion):
        if loader is None or len(loader) == 0:
            return None
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for batch in loader:
                batch_x, batch_x_mark, target = self._loss_inputs(batch)
                output = model(batch_x, batch_x_mark)[:, :, -target.shape[-1]:]
                total += float(criterion(output, target).item())
                count += 1
        model.train()
        return total / max(count, 1)

    def _train_epochs(self, model, loader, parameters, epochs, val_loader=None,
                      patience=0, prox_snapshot=None, prox_mu=0.0):
        """Epoch-based training with DTMoe's lr schedule and early stopping.

        Returns ``{'epochs_run', 'best_val', 'lr_history', 'val_history'}``.  The
        best weights are restored at the end, mirroring ``EarlyStopping`` in
        ``run.py``, so a diverging tail cannot leak into the evaluated model.
        ``val_history`` is the per-epoch validation MSE: without it a run leaves
        no training curve on disk (only ``best_val``), so FedTDO cannot appear in
        the per-epoch figures that the FL baselines get from their round logs.
        """
        info = {'epochs_run': 0, 'best_val': None, 'lr_history': [],
                'val_history': []}
        parameters = [p for p in parameters if p.requires_grad]
        if epochs <= 0 or not parameters or len(loader) == 0:
            return info
        optimizer = optim.AdamW(parameters, lr=float(self.args.learning_rate))
        criterion = nn.MSELoss()
        best_val, stale, best_state = float('inf'), 0, None
        for epoch in range(1, int(epochs) + 1):
            lr = self._adjust_learning_rate(optimizer, epoch)
            info['lr_history'].append(lr)
            self._epoch_train(model, loader, optimizer, criterion,
                              prox_snapshot=prox_snapshot, prox_mu=prox_mu)
            info['epochs_run'] = epoch
            if val_loader is None:
                # Warm-up stage: no early stopping by design, so there is nothing
                # to validate against and no reason to pay for a second pass.
                continue
            val_loss = self._epoch_valid(model, val_loader, criterion)
            if val_loss is None:
                continue
            info['val_history'].append(float(val_loss))
            info['best_val'] = val_loss if info['best_val'] is None else min(
                info['best_val'], val_loss)
            if val_loss < best_val:
                best_val, stale = val_loss, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
                if patience and stale >= int(patience):
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        return info

    def _train_downstream_synced(self, models, train_loaders, val_loaders, epochs):
        """Downstream stage with periodic FedAvg of the shared modules.

        The frozen-encoder design locks one averaged embedding for the whole
        downstream stage; a single seed average cannot represent seven different
        channels, and the measurements on ETTh1/pl96 show that freeze alone costs
        ~3.3% MSE versus keeping the average but letting each client adapt it.
        Here the shared modules are re-averaged every ``sync_every`` epochs and
        stay trainable in between, which keeps the federated representation
        improving with what the clients actually learn while still letting each
        client drift towards its own channel.

        Early stopping stays per client and per epoch (same rule as every other
        FedTDO arm); each client restores its own best weights at the end, so a
        later synchronisation cannot overwrite a better local snapshot.
        """
        criterion = nn.MSELoss()
        optimizers = [optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=float(self.args.learning_rate))
                      for model in models]
        infos = [{'epochs_run': 0, 'best_val': None, 'lr_history': [],
                  'val_history': []} for _ in models]
        best = [float('inf')] * len(models)
        stale = [0] * len(models)
        best_state = [None] * len(models)
        active = [True] * len(models)
        for epoch in range(1, int(epochs) + 1):
            for index, model in enumerate(models):
                if not active[index]:
                    continue
                lr = self._adjust_learning_rate(optimizers[index], epoch)
                infos[index]['lr_history'].append(lr)
                self._epoch_train(model, train_loaders[index], optimizers[index],
                                  criterion, prox_snapshot=self.shared_snapshot,
                                  prox_mu=self.share_prox)
                infos[index]['epochs_run'] = epoch
            if self.sync_every > 0 and epoch % self.sync_every == 0:
                self.server.synchronize_modules(models, self.share_modules)
            for index, model in enumerate(models):
                if not active[index]:
                    continue
                val_loss = self._epoch_valid(model, val_loaders[index], criterion)
                if val_loss is None:
                    continue
                infos[index]['val_history'].append(float(val_loss))
                infos[index]['best_val'] = val_loss if infos[index]['best_val'] is None else min(
                    infos[index]['best_val'], val_loss)
                if val_loss < best[index]:
                    best[index], stale[index] = val_loss, 0
                    best_state[index] = copy.deepcopy(model.state_dict())
                else:
                    stale[index] += 1
                    if self.patience and stale[index] >= int(self.patience):
                        active[index] = False
            if not any(active):
                break
        for index, model in enumerate(models):
            if best_state[index] is not None:
                model.load_state_dict(best_state[index])
        return infos

    def _train_steps(self, model, loader, parameters, max_steps):
        if max_steps == 0:
            return
        parameters = [p for p in parameters if p.requires_grad]
        if not parameters:
            return
        if len(loader) == 0:
            return
        optimizer = optim.AdamW(parameters, lr=self.args.learning_rate)
        criterion = nn.MSELoss()
        model.train()
        # ``max_steps`` is a budget in gradient steps, so the loop has to cross
        # epoch boundaries.  Iterating the loader once silently capped every arm
        # at ``len(loader)`` steps: on Traffic (381 batches/epoch) an arm asking
        # for 1200 steps got 381, and the isomorphic arm asking for 1600 in a
        # single call got 381 while an arm calling this twice got 762.  That made
        # "two-stage training" look 8pp better than "one-stage" purely because it
        # performed two passes over the data.
        step = 0
        while step < max_steps:
            for batch in loader:
                if step >= max_steps:
                    break
                batch_x, batch_x_mark, target = self._loss_inputs(batch)
                optimizer.zero_grad()
                output = model(batch_x, batch_x_mark)[:, :, -target.shape[-1]:]
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                step += 1
            else:
                # A loader shorter than the remaining budget: keep looping.
                continue
            break

    # ------------------------------------------------------------------ setup
    def _prepare_partition(self):
        """Return client names, train / val / eval loaders for one protocol.

        ``val`` is only used for early stopping in the epoch-based protocol; the
        evaluation stream is ``test`` unless ``--fedtdo_eval_split val``.
        """
        if self.partition == 'time':
            train_shards = self._shards(self._dataset('train'))
            val_shards = self._shards(self._dataset('val'))
            eval_shards = self._shards(self._dataset(self.eval_split))
            names = [f'client-{index}' for index in range(len(train_shards))]
            train_loaders = [self._loader(shard, shuffle=True) for shard in train_shards]
            # Each client owns 1/nc of the evaluation split, so a 256 training batch
            # would leave two online rounds per client and the controller would never
            # see enough feedback.  Decouple the stream batch, exactly as the channel
            # partition does.
            eval_batch = int(getattr(self.args, 'fedtdo_stream_batch', 0) or 0) or None
            val_loaders = [self._loader(shard, shuffle=False, batch_size=eval_batch)
                           for shard in val_shards]
            eval_loaders = [self._loader(shard, shuffle=False, batch_size=eval_batch)
                            for shard in eval_shards]
            return names, train_loaders, val_loaders, eval_loaders

        if self.partition != 'channel':
            raise ValueError(f'unknown fedtdo_partition {self.partition!r}')

        bundles = build_federated_clients_from_channels(
            self.args, num_clients=self.args.fedtdo_num_clients,
            channels_per_client=self.client_channels,
            channel_sampling=getattr(self.args, 'fedtdo_channel_sampling', 'random'),
            seed=getattr(self.args, 'seed', 2021),
        )
        self.model_args = make_client_args(self.args, enc_in=self.client_channels)
        names = list(bundles.keys())
        train_loaders = [bundle['train'] for bundle in bundles.values()]
        val_loaders = [bundle['val'] for bundle in bundles.values()]
        eval_loaders = [bundle[self.eval_split] for bundle in bundles.values()]
        return names, train_loaders, val_loaders, eval_loaders

    def _build_clients(self, names, train_loaders, val_loaders):
        # Identical initialization plus explicit encoder FedAvg is important:
        # an expert capsule has a well-defined representation space only after
        # this shared encoder is frozen.
        base_model = Model(self.model_args).float().to(self.device)
        models = [copy.deepcopy(base_model).to(self.device) for _ in names]
        self.epoch_mode = getattr(self.args, 'fedtdo_step_mode', 'steps') == 'epochs'
        self.patience = int(getattr(self.args, 'fedtdo_patience', 0))
        self.train_info = []

        if self.control in ('isomorphic', 'local_encoder'):
            # Isomorphic control: same architecture, same budget, but no
            # cross-client encoder averaging and no freeze.  The local model is
            # trained exactly the way ``--baseline local`` trains it, so the
            # remaining difference to the method arm is the shared frozen
            # encoder - and, when the controller is enabled, migration.
            if self.epoch_mode:
                # Same total number of passes as the method arm, in one run, so
                # DTMoe's lr decay and early stopping apply to it unchanged.
                total_epochs = (int(self.args.fedtdo_encoder_rounds)
                                * int(self.args.fedtdo_warmup_epochs)
                                + int(self.args.fedtdo_epochs))
                for index, (model, loader, val_loader) in enumerate(
                        zip(models, train_loaders, val_loaders)):
                    info = self._train_epochs(model, loader, model.parameters(), total_epochs,
                                              val_loader=val_loader, patience=self.patience)
                    info['client'] = names[index]
                    self.train_info.append(info)
            else:
                total_steps = (int(self.args.fedtdo_encoder_rounds)
                               * int(self.args.fedtdo_warmup_steps)
                               + int(self.args.fedtdo_local_steps))
                for model, loader in zip(models, train_loaders):
                    self._train_steps(model, loader, model.parameters(), total_steps)
            # Each client now owns a different encoder, so a capsule's
            # representation space is only nominally shared.  Publishing one
            # common version is deliberately permissive: measuring what breaks
            # when the version gate is bypassed is the point of this arm.
            encoder_version = (f'{self.control}-v0' if self.control == 'isomorphic'
                               else 'local-encoder-v0')
            post_steps, post_epochs = 0, 0
        else:
            encoder_version = 'patch-encoder-v0'
            for _ in range(self.args.fedtdo_encoder_rounds):
                for model, loader in zip(models, train_loaders):
                    if self.control in ('fedtdo_joint', 'fedtdo_localfreeze'):
                        # Joint warm-up: the encoder sees the gradient of a
                        # *training* downstream instead of the gradient of a
                        # randomly initialised one.  Only the encoder is
                        # averaged and frozen afterwards, so the capsule space
                        # is unchanged - this isolates the optimisation defect,
                        # not the sharing topology.
                        parameters = model.parameters()
                    else:
                        # Warm up exactly the modules that will be shared; when
                        # the share spec is wider than the patch embedding
                        # (``--fedtdo_share_modules patch_embedding,temporal_moe_layers``)
                        # those layers need the same pre-training before FedAvg.
                        shared_parameters = []
                        for share_name in self.share_modules:
                            for module in module_slots(model, share_name):
                                shared_parameters.extend(module.parameters())
                        parameters = shared_parameters
                    if self.epoch_mode:
                        # DTMoe-aligned warm-up: whole epochs over the client's
                        # training split, with the same lr schedule.  No early
                        # stopping here - the shared encoder must be a product of
                        # a fixed number of federated rounds, not of a per-client
                        # stopping rule, or FedAvg averages models trained to
                        # different depths.
                        self._train_epochs(
                            model, loader, parameters,
                            int(self.args.fedtdo_warmup_epochs),
                        )
                    else:
                        self._train_steps(
                            model, loader, parameters,
                            self.args.fedtdo_warmup_steps,
                        )
                if self.control == 'fedtdo_localfreeze':
                    # Decisive control for ``fedtdo_joint``: identical warm-up
                    # (400 joint steps), identical freeze, identical 1200
                    # downstream steps - only the FedAvg average is removed.
                    # Whatever gap remains between the two arms is the value of
                    # aggregation; everything else is the value of freezing.
                    encoder_version = 'patch-encoder-v0-local'
                    continue
                if self.control == 'fedtdo_noavg':
                    # Diagnostic that separates the two halves of the warm-up:
                    # encoder-only training (staging) vs the FedAvg average on
                    # top of it.  No averaging happens, so every client keeps
                    # its own trunk; a single common version string keeps the
                    # capsules comparable, exactly as the isomorphic arm does.
                    continue
                sync_started = time.perf_counter()
                encoder_version = self.server.synchronize_modules(
                    models, self.share_modules)
                self._paper_encoder_sync(names, models,
                                         (time.perf_counter() - sync_started) * 1000.0)
            # ``fedtdo_nofreeze`` isolates the cost of the version gate: the
            # averaged encoder is only an initialisation, and local training may
            # keep adapting it.  Plan §7.2 requires freezing for a well-defined
            # capsule space, so this arm is a diagnostic, not the method.
            if self.control != 'fedtdo_nofreeze':
                if self.share_prox > 0:
                    # Soft sharing: instead of forbidding adaptation of the
                    # federated weights, pull each client back towards them with
                    # 0.5*lambda*||theta - theta_shared||^2.  A channel that can
                    # reuse the shared pattern still gets the benefit; one that
                    # cannot is no longer forced to live with a bad average.
                    self.shared_snapshot = self.server.snapshot_modules(
                        models[0], self.share_modules)
                elif self.sync_every <= 0:
                    # Periodic sharing keeps the modules trainable: the freeze is
                    # replaced by a re-average every ``sync_every`` epochs.
                    self.server.freeze_modules(models, self.share_modules)
            post_steps = self.args.fedtdo_local_steps
            post_epochs = int(self.args.fedtdo_epochs)

        clients = []
        sync_infos = None
        if self.epoch_mode and self.sync_every > 0 and post_epochs > 0:
            sync_infos = self._train_downstream_synced(
                models, train_loaders, val_loaders, post_epochs)
        for index, (name, model, loader) in enumerate(zip(names, models, train_loaders)):
            if self.epoch_mode:
                # Downstream stage under DTMoe's rules: full epochs, lr decay,
                # early stopping on this client's validation split, best weights
                # restored.  The encoder is already frozen, so the lr schedule
                # only moves the expert/router/head parameters that are left.
                # The isomorphic arm already consumed its whole budget above, so
                # post_epochs is 0 there and nothing is trained (or recorded)
                # twice.
                if post_epochs > 0:
                    if sync_infos is not None:
                        info = sync_infos[index]
                    else:
                        info = self._train_epochs(
                            model, loader,
                            [p for p in model.parameters() if p.requires_grad],
                            post_epochs, val_loader=val_loaders[index],
                            patience=self.patience,
                            prox_snapshot=self.shared_snapshot,
                            prox_mu=self.share_prox,
                        )
                    info['client'] = name
                    self.train_info.append(info)
            else:
                self._train_steps(
                    model, loader,
                    (parameter for parameter in model.parameters() if parameter.requires_grad),
                    post_steps,
                )
            # ``--fedtdo_disable_drift 1`` is the ablation switch: without it the
            # detector's contribution cannot be separated from the transfer
            # policy's, because every transfer decision consumes a drift score.
            if not int(getattr(self.args, 'fedtdo_disable_drift', 0) or 0):
                model.enable_drift_detection(
                    window_size=self.args.drift_window_size,
                    history_size=self.args.drift_history_size,
                    k_sigma=self.args.drift_k_sigma,
                )
            controller = FedTDOController(
                evaluation_budget=self.args.fedtdo_eval_budget,
                beta=self.args.fedtdo_beta,
                eta_age=self.args.fedtdo_eta_age,
                eta_drift=self.args.fedtdo_eta_drift,
                lambda_eval=self.args.fedtdo_lambda_eval,
                lambda_transfer=self.args.fedtdo_lambda_transfer,
                lambda_switch=self.args.fedtdo_lambda_switch,
                switch_margin=float(getattr(self.args, 'fedtdo_switch_margin', 0.0)),
                evidence_k=int(getattr(self.args, 'fedtdo_evidence_k', 8)),
                evidence_t=float(getattr(self.args, 'fedtdo_evidence_t', 1.0)),
                evidence_gate=str(getattr(self.args, 'fedtdo_evidence_gate', 'ranked')),
                transfer_threshold=self.args.fedtdo_transfer_threshold,
                min_observations=self.args.fedtdo_min_observations,
                initial_probes=self.args.fedtdo_initial_probes,
                probe_threshold=self.args.fedtdo_probe_threshold,
                delta=getattr(self.args, 'fedtdo_delta', 0.1),
                transfer_enabled=self.transfer_enabled,
                estimator=getattr(self.args, 'fedtdo_estimator', 'mean'),
                decision=getattr(self.args, 'fedtdo_decision', 'lcb'),
                trim_lower=getattr(self.args, 'fedtdo_trim_lower', 0.1),
                trim_upper=getattr(self.args, 'fedtdo_trim_upper', 0.9),
                upper_q=getattr(self.args, 'fedtdo_upper_q', 0.75),
                shrink_n0=getattr(self.args, 'fedtdo_shrink_n0', 0.0),
                shrink_group=getattr(self.args, 'fedtdo_shrink_group', 'global'),
                hierarchy=getattr(self.args, 'fedtdo_hierarchy', 'none'),
                shrink_n0_role=getattr(self.args, 'fedtdo_shrink_n0_role', 5.0),
                probe_floor=bool(getattr(self.args, 'fedtdo_probe_floor', 1)),
                shock=getattr(self.args, 'fedtdo_shock', 'raw'),
                decorr_rho=float(getattr(self.args, 'fedtdo_decorr_rho', 0.0)),
                baseline_k=float(getattr(self.args, 'fedtdo_baseline_k', 0.6)),
            )
            clients.append(FedTDOClient(
                deploy_topk=int(getattr(self.args, 'fedtdo_deploy_topk', 1)),
                include_self_experts=bool(int(getattr(self.args, 'fedtdo_include_self', 0))),
                client_id=name, model=model, registry=self.server.registry,
                controller=controller, cost_model=CostModel(
                    switch_cost=self.args.fedtdo_lambda_switch,
                ), transfer_layer=self.args.fedtdo_transfer_layer,
                encoder_version=encoder_version, mix_alpha=self.args.fedtdo_mix_alpha,
                feedback_delay=max(0, int(self.args.pred_len) - 1)
                                + max(0, int(self.args.fedtdo_feedback_delay)),
                output_start=(-1 if self.model_args.features == 'MS' else 0),
                alpha_calibration=bool(getattr(self.args, 'fedtdo_alpha_calibration', 1)),
                alpha_min_observations=int(getattr(self.args, 'fedtdo_alpha_min_obs', 3)),
                cache_limit=getattr(self.args, 'fedtdo_cache_limit', None),
                gain_mode=getattr(self.args, 'fedtdo_gain_mode', 'relative'),
                clip_reward=bool(getattr(self.args, 'fedtdo_clip_reward', 0)),
                reward_basis=getattr(self.args, 'fedtdo_reward_basis', 'mixed'),
            ))
        for client in clients:
            client.publish_experts()
        return clients

    # ------------------------------------------------------------------- run
    def run(self):
        names, train_loaders, val_loaders, eval_loaders = self._prepare_partition()
        # Built before the clients so the encoder-synchronisation bytes can be
        # recorded as they happen; disabled by default (``--paper_record 0``).
        self._recorder = self._make_recorder()
        clients = self._build_clients(names, train_loaders, val_loaders)
        streams = [iter(loader) for loader in eval_loaders]
        # A full H-step target is not observable until H forecast origins later.
        # Keep each client's pending window separate because streams are interleaved.
        _target_mode = str(getattr(self.args, 'fedtdo_online_target', 'full'))
        if _target_mode == 'first':
            _prefix = 1
        elif _target_mode == 'prefix':
            _prefix = max(1, min(int(getattr(self.args, 'fedtdo_online_prefix', 4)),
                                 int(self.args.pred_len)))
        else:
            _prefix = int(self.args.pred_len)
        feedback_delay = _prefix + max(0, int(self.args.fedtdo_feedback_delay))
        pending_by_client = {client.client_id: deque() for client in clients}
        predictions, targets = [], []
        # Per-client streams, so the table can be compared with the FL baselines
        # (which report the mean over clients rather than a pooled MSE).
        per_client = {client.client_id: {'pred': [], 'true': []} for client in clients}
        global_step = 0
        active = [True] * len(clients)
        served = [0] * len(clients)
        online_optims = {}
        window_idx = {client.client_id: 0 for client in clients}
        while any(active):
            for index, (client, stream) in enumerate(zip(clients, streams)):
                if not active[index]:
                    continue
                if self.online_steps is not None and served[index] >= int(self.online_steps):
                    active[index] = False
                    continue
                try:
                    batch = next(stream)
                except StopIteration:
                    active[index] = False
                    continue
                served[index] += 1
                global_step += 1
                batch_x, batch_x_mark, target = self._loss_inputs(batch)
                if batch_x.shape[0] != 1:
                    raise ValueError('Causal delayed-feedback evaluation requires fedtdo_stream_batch=1')
                origin_idx = served[index] - 1
                pending = pending_by_client[client.client_id]
                # Labels from origin (origin_idx - H) are now fully observable.
                while pending and pending[0][0] <= origin_idx:
                    due_idx, old_origin, old_x, old_xm, old_target, old_record = pending.popleft()
                    update_steps, update_ms = (self._online_step(
                        client, old_x, old_xm, old_target, online_optims)
                        if self.online_lr > 0 else (0, 0.0))
                    self._paper_online_update(
                        client, old_origin, update_steps, update_ms,
                        old_target.shape[0], due_idx)
                    stats = client.reveal_feedback(old_record.record_id, old_target)
                    self._paper_feedback(client, stats, due_idx, window_idx)
                    self._log_round(due_idx, client, old_record, stats)
                predict_started = time.perf_counter()
                record = client.predict(batch_x, batch_x_mark)
                predict_ms = (time.perf_counter() - predict_started) * 1000.0
                # No update is made from this window's target yet: its final
                # label arrives only at origin_idx + pred_len.
                steps, update_ms = 0, 0.0
                predictions.append(record.prediction.detach().cpu().numpy())
                targets.append(target.detach().cpu().numpy())
                per_client[client.client_id]['pred'].append(
                    record.prediction.detach().cpu().numpy())
                per_client[client.client_id]['true'].append(
                    target.detach().cpu().numpy())
                self._paper_window(client, batch_x, batch_x_mark, target,
                                   record, window_idx, steps, update_ms,
                                   predict_ms, global_step, origin_idx,
                                   origin_idx + int(self.args.pred_len))
                pending.append((origin_idx + feedback_delay, origin_idx,
                                batch_x.detach(), batch_x_mark.detach(),
                                target.detach(), record))
        # Drain only after the corresponding final labels have arrived.  These
        # tail updates cannot affect any forecast in the evaluated stream.
        for client in clients:
            pending = pending_by_client[client.client_id]
            while pending:
                due_idx, old_origin, old_x, old_xm, old_target, old_record = pending.popleft()
                update_steps, update_ms = (self._online_step(
                    client, old_x, old_xm, old_target, online_optims)
                    if self.online_lr > 0 else (0, 0.0))
                self._paper_online_update(
                    client, old_origin, update_steps, update_ms,
                    old_target.shape[0], due_idx)
                stats = client.reveal_feedback(old_record.record_id, old_target)
                self._paper_feedback(client, stats, due_idx, window_idx)
                self._log_round(due_idx, client, old_record, stats)

        preds = np.concatenate(predictions, axis=0)
        trues = np.concatenate(targets, axis=0)
        mae, mse, rmse, mape, mspe = metric(preds, trues)

        client_metrics = {}
        for client_id, store in per_client.items():
            if not store['pred']:
                continue
            client_preds = np.concatenate(store['pred'], axis=0)
            client_trues = np.concatenate(store['true'], axis=0)
            c_mae, c_mse, _, _, _ = metric(client_preds, client_trues)
            client_metrics[client_id] = {'mse': float(c_mse), 'mae': float(c_mae)}

        result = {
            'mae': float(mae), 'mse': float(mse), 'rmse': float(rmse),
            'mape': float(mape), 'mspe': float(mspe),
            'client_metrics': client_metrics,
            'mean_client_mse': float(np.mean([v['mse'] for v in client_metrics.values()]))
            if client_metrics else None,
            'mean_client_mae': float(np.mean([v['mae'] for v in client_metrics.values()]))
            if client_metrics else None,
            'partition': self.partition,
            'control': self.control,
            'transfer_enabled': self.transfer_enabled,
            'online_steps': int(self.online_steps) if self.online_steps is not None else None,
            'registry_experts': len(self.server.registry),
            'registry_upload_bytes': self.server.registry.bytes_uploaded,
            'clients': [client.summary() for client in clients],
            'train': {
                'step_mode': getattr(self.args, 'fedtdo_step_mode', 'steps'),
                'warmup_epochs': getattr(self.args, 'fedtdo_warmup_epochs', None),
                'epochs': getattr(self.args, 'fedtdo_epochs', None),
                'patience': getattr(self.args, 'fedtdo_patience', None),
                'lradj': getattr(self.args, 'fedtdo_lradj', None),
                'per_client': getattr(self, 'train_info', []),
            },
        }
        # Without this block two arms can only be told apart by their directory
        # name, which is how a contradictory table gets built.
        result['config'] = {
            key: getattr(self.args, key)
            for key in (
                'data', 'model_id', 'seq_len', 'pred_len', 'label_len', 'patch_len',
                'stride', 'd_model', 'learning_rate', 'batch_size', 'train_epochs',
                'seed', 'num_workers',
                'fedtdo_partition', 'fedtdo_num_clients', 'fedtdo_control',
                'fedtdo_adapter', 'fedtdo_disable_transfer', 'fedtdo_encoder_rounds',
                'fedtdo_warmup_steps', 'fedtdo_local_steps', 'fedtdo_online_steps',
                'fedtdo_stream_batch', 'fedtdo_eval_split',
                'fedtdo_step_mode', 'fedtdo_warmup_epochs', 'fedtdo_epochs',
                'fedtdo_patience', 'fedtdo_lradj', 'dropout',
                'fedtdo_eval_budget', 'fedtdo_estimator', 'fedtdo_decision',
                'fedtdo_shrink_n0', 'fedtdo_shrink_group', 'fedtdo_gain_mode',
                'fedtdo_hierarchy', 'fedtdo_shrink_n0_role', 'fedtdo_reward_basis',
                'fedtdo_probe_floor',
                'fedtdo_shock', 'fedtdo_decorr_rho', 'fedtdo_baseline_k',
                'fedtdo_clip_reward', 'fedtdo_mix_alpha', 'fedtdo_feedback_delay',
                'fedtdo_transfer_layer', 'fedtdo_alpha_calibration',
                # Mechanism knobs.  Without these the only way to tell two arms
                # apart was the directory name: ``evidence_k``/``evidence_t``,
                # ``switch_margin``, ``deploy_topk`` and ``include_self`` all used
                # to be written as ``None``, so a sweep could not be audited.
                'fedtdo_decision', 'fedtdo_switch_margin',
                'fedtdo_evidence_k', 'fedtdo_evidence_t',
                'fedtdo_evidence_gate',
                'fedtdo_deploy_topk', 'fedtdo_include_self',
                'fedtdo_share_modules', 'fedtdo_online_lr',
                'fedtdo_online_target', 'fedtdo_online_prefix',
                'fedtdo_online_substeps',
            )
            if hasattr(self.args, key)
        }
        result['actions'] = self._action_summary(clients)
        output_dir = os.path.join(self.args.fedtdo_output_dir, self.args.model_id)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'metrics.json'), 'w', encoding='utf-8') as handle:
            json.dump(result, handle, indent=2)
        with open(os.path.join(output_dir, 'rounds.json'), 'w', encoding='utf-8') as handle:
            json.dump(self.round_log, handle)
        np.save(os.path.join(output_dir, 'pred.npy'), preds)
        np.save(os.path.join(output_dir, 'true.npy'), trues)
        recorder = getattr(self, '_recorder', None)
        if recorder is not None:
            recorder.finish(summary={
                'overall_mse': float(mse), 'overall_mae': float(mae),
                'overall_rmse': float(rmse), 'overall_mape': float(mape),
                'overall_mspe': float(mspe),
                'per_client': {cid: {k: float(v) for k, v in vals.items()}
                               for cid, vals in client_metrics.items()},
                'num_clients': len(client_metrics),
            })
        print(json.dumps({key: result[key] for key in
                          ('mse', 'mae', 'partition', 'control', 'actions')}, indent=2))
        return result

    def _online_step(self, client, batch_x, batch_x_mark, target, cache):
        """Update on a stored H-step target after its final label has arrived.

        The caller releases this example at origin t+H (or later if an extra
        controller delay is configured), before forecasting at that origin.
        The update can affect that and subsequent forecasts, never earlier ones.
        Optimiser state is kept per client so momentum is not reset every step.
        """
        lr = float(self.online_lr)
        if lr <= 0:
            return 0, 0.0
        model = client.model
        optimiser = cache.get(client.client_id)
        if optimiser is None:
            optimiser = optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lr)
            cache[client.client_id] = optimiser
        else:
            for group in optimiser.param_groups:
                group['lr'] = lr
        was_training = model.training
        model.train()
        criterion = nn.MSELoss()
        # ``substeps`` splits the stream batch so the online stage can take one
        # step per sample (the granularity the FL baselines use) instead of one
        # per batch, which is 32x fewer updates on ETTh1.
        substeps = max(1, int(getattr(self.args, 'fedtdo_online_substeps', 1) or 1))
        size = max(1, batch_x.shape[0] // substeps)
        steps = 0
        started = time.perf_counter()
        for start in range(0, batch_x.shape[0], size):
            xs = batch_x[start:start + size]
            xms = batch_x_mark[start:start + size]
            ys = target[start:start + size]
            _mode = str(getattr(self.args, 'fedtdo_online_target', 'full'))
            if _mode in ('first', 'prefix'):
                _k = 1 if _mode == 'first' else max(
                    1, min(int(getattr(self.args, 'fedtdo_online_prefix', 4)),
                           int(self.args.pred_len)))
                # Masked partial feedback: only the steps that have actually been
                # revealed by the time this update fires.
                ys = ys[:, :_k]
            optimiser.zero_grad()
            output = model(xs, xms)[:, :, -ys.shape[-1]:]
            if _mode in ('first', 'prefix'):
                output = output[:, :_k]
            loss = criterion(output, ys)
            loss.backward()
            # Same numerical guard as the baselines: a non-finite batch must not
            # poison the served model.  Skipping keeps the model where it was.
            clip = float(getattr(self.args, 'fedtdo_grad_clip', 0.0) or 0.0)
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            if torch.isfinite(loss):
                optimiser.step()
                steps += 1
            else:
                self.nan_skipped_updates = getattr(self, 'nan_skipped_updates', 0) + 1
                optimiser.zero_grad(set_to_none=True)
        if not was_training:
            model.eval()
        return steps, (time.perf_counter() - started) * 1000.0

    # ------------------------------------------------------- paper recording
    def _make_recorder(self):
        """Build the paper recorder; a disabled recorder makes every call a no-op."""
        enabled = int(getattr(self.args, 'paper_record', 0) or 0)
        root = str(getattr(self.args, 'paper_results_dir', './results/paper_fedtdo'))
        scalar_args = {
            key: value for key, value in vars(self.args).items()
            if isinstance(value, (str, int, float, bool)) or value is None
        }
        recorder = PaperRecorder(
            root=root,
            method=str(self.control),
            dataset=str(getattr(self.args, 'data', 'unknown')),
            seed=int(getattr(self.args, 'seed', 0)),
            run_kind='fedtdo',
            enabled=bool(enabled),
            extra_config=scalar_args,
        )
        if recorder.enabled:
            recorder.set_config(
                num_clients=int(getattr(self.args, 'fedtdo_num_clients', 0) or 0),
                channels_per_client=int(getattr(self, 'client_channels', 1)),
                seq_len=int(getattr(self.args, 'seq_len', 0)),
                pred_len=int(getattr(self.args, 'pred_len', 0)),
                model_name='Dynamic_TMoE',
                online_enabled=bool(float(self.online_lr) > 0),
                online_update_policy=str(getattr(self.args, 'fedtdo_step_mode', '')),
                online_lr=float(self.online_lr),
                online_substeps=int(getattr(self.args, 'fedtdo_online_substeps', 1) or 1),
                online_steps=int(self.online_steps) if self.online_steps else None,
                stream_batch=int(getattr(self.args, 'fedtdo_stream_batch', 0) or 0),
                label_delay=int(self.args.pred_len),
                controller_extra_delay=max(0, int(getattr(self.args, 'fedtdo_feedback_delay', 0))),
                feedback_protocol='delayed_full_horizon',
                eval_budget=int(getattr(self.args, 'fedtdo_eval_budget', 0)),
                cache_limit=getattr(self.args, 'fedtdo_cache_limit', None),
                transfer_enabled=bool(self.transfer_enabled),
                share_modules=','.join(self.share_modules),
                optimizer='AdamW(train)/SGD(online)',
                gradient_steps_per_update=int(
                    getattr(self.args, 'fedtdo_online_substeps', 1) or 1),
            )
        return recorder

    def _paper_encoder_sync(self, names, models, duration_ms):
        """Bytes moved by one federated averaging of the shared modules."""
        recorder = getattr(self, '_recorder', None)
        if recorder is None or not recorder.enabled or not models:
            return
        payload = 0
        for share_name in self.share_modules:
            for module in module_slots(models[0], share_name):
                payload += sum(p.numel() * p.element_size() for p in module.parameters())
                payload += sum(b.numel() * b.element_size() for b in module.buffers())
        for client_id in names:
            recorder.communication(
                client_id=client_id, window_idx=0, round_idx=0,
                direction='upload', payload_type='shared_modules',
                n_bytes=payload, duration_ms=duration_ms)
            recorder.communication(
                client_id=client_id, window_idx=0, round_idx=0,
                direction='download', payload_type='shared_modules',
                n_bytes=payload, duration_ms=duration_ms)
            recorder.cost(client_id=client_id, window_idx=0,
                          operation='encoder_sync', duration_ms=duration_ms)

    def _paper_window(self, client, batch_x, batch_x_mark, target, record,
                      window_idx, steps, update_ms, predict_ms, global_step,
                      origin_idx, label_available_idx):
        """Record one stream batch: per-window metrics plus the events it caused."""
        recorder = getattr(self, '_recorder', None)
        if recorder is None or not recorder.enabled:
            return
        y_true = target.detach().cpu().numpy()
        y_pred = record.prediction.detach().cpu().numpy()
        x_hist = batch_x.detach().cpu().numpy()
        batch = y_true.shape[0]
        shadow_ms = float(getattr(client, 'last_shadow_ms', 0.0) or 0.0)
        for offset in range(batch):
            window_idx[client.client_id] += 1
            recorder.window(
                client_id=client.client_id,
                window_idx=window_idx[client.client_id],
                y_true=y_true[offset], y_pred=y_pred[offset],
                x_history=x_hist[offset],
                label_reveal_idx=int(label_available_idx),
                target_start_idx=int(origin_idx + 1),
                target_end_idx=int(origin_idx + int(self.args.pred_len)),
                regime_id='',
            )
        # A batch-level decision is attributed to the first window it served.
        decision_window = window_idx[client.client_id] - batch + 1
        decision = getattr(client, 'last_decision', None) or {}
        active_id = decision.get('active_id')
        for probe in getattr(client, 'last_probes', []) or []:
            recorder.candidate_evaluation(
                client_id=client.client_id, window_idx=decision_window,
                candidate_expert_id=probe['expert_id'],
                source_client_id=probe.get('source_id', ''),
                candidate_version=probe.get('version', ''),
                predicted_gain=probe.get('predicted_gain'),
                estimated_uncertainty=probe.get('uncertainty'),
                selected=(probe['expert_id'] == active_id),
                transferred=bool(decision.get('switched'))
                and probe['expert_id'] == active_id,
            )
        for expert_id, source_id, n_bytes in getattr(client, 'download_log', []) or []:
            recorder.communication(
                client_id=client.client_id, window_idx=decision_window,
                direction='download', payload_type='expert_capsule',
                n_bytes=int(n_bytes), expert_id=expert_id,
                source_client_id=str(source_id or ''),
            )
        if decision.get('switched') and active_id is not None:
            recorder.transfer(
                client_id=client.client_id, window_idx=decision_window,
                event_type='switch', expert_id=active_id,
                old_expert_id=str(decision.get('previous_id') or ''),
                new_expert_id=active_id,
                source_client_id='',
                estimated_gain=decision.get('estimated_gain'),
                decision_reason='controller_selected',
            )
        # Raw drift score of the served model.  No threshold is applied here:
        # whether the detector *alarmed* is internal to the model, and inventing
        # an alarm from a score would be exactly the inference the spec forbids.
        try:
            drift_score = float(client.model.online_drift_signal().get('score', 0.0) or 0.0)
        except (AttributeError, TypeError, ValueError):
            drift_score = None
        if drift_score is not None:
            recorder.regime(client_id=client.client_id, window_idx=decision_window,
                            event_type='drift_score', detection_score=drift_score)
        recorder.cost(client_id=client.client_id, window_idx=decision_window,
                      operation='forecasting',
                      duration_ms=max(0.0, predict_ms - shadow_ms))
        if shadow_ms:
            recorder.cost(client_id=client.client_id, window_idx=decision_window,
                          operation='shadow_eval', duration_ms=shadow_ms)
        if steps:
            recorder.online_update(
                client_id=client.client_id, window_idx=decision_window,
                gradient_steps=int(steps),
                learning_rate=float(self.online_lr),
                batch_size=int(batch_x.shape[0]),
                duration_ms=update_ms,
            )
            recorder.cost(client_id=client.client_id, window_idx=decision_window,
                          operation='adaptation', duration_ms=update_ms)

    def _paper_online_update(self, client, origin_idx, steps,
                             update_ms, batch_size, update_time_idx):
        """Record an SGD update against its original forecast after H labels mature."""
        recorder = getattr(self, '_recorder', None)
        if recorder is None or not recorder.enabled or not steps:
            return
        recorder.online_update(
            client_id=client.client_id, window_idx=int(origin_idx + 1),
            gradient_steps=int(steps), learning_rate=float(self.online_lr),
            batch_size=int(batch_size), duration_ms=float(update_ms),
            update_type='delayed_full_horizon',
            target_start_idx=int(origin_idx + 1),
            target_end_idx=int(origin_idx + int(self.args.pred_len)),
            label_available_idx=int(origin_idx + int(self.args.pred_len)),
            update_time_idx=int(update_time_idx),
        )
        recorder.cost(client_id=client.client_id, window_idx=int(origin_idx + 1),
                      operation='adaptation', duration_ms=float(update_ms))

    def _paper_feedback(self, client, stats, global_step, window_idx):
        """Attach realised gains to the evaluations that produced them."""
        recorder = getattr(self, '_recorder', None)
        if recorder is None or not recorder.enabled:
            return
        default_loss = float(stats['default_loss'])
        for expert_id, gain in (stats.get('gains') or {}).items():
            recorder.feedback(
                client_id=client.client_id, expert_id=expert_id,
                realized_gain=float(gain),
                reward=float(client._reward(gain, default_loss)),
                feedback_release_window=int(global_step),
                reward_basis=str(getattr(client, 'reward_basis', '')),
            )

    def _log_round(self, step, client, record, stats):
        default_loss = float(stats['default_loss'])
        # Per-expert rewards are logged so ``scripts/replay_selector.py`` can
        # rank decision rules offline against the exact rewards this round
        # produced, instead of paying for a full retrain per variant.
        rewards = {expert_id: client._reward(gain, default_loss)
                   for expert_id, gain in stats['gains'].items()}
        expert_gains = stats.get('expert_gains') or {}
        expert_rewards = {expert_id: client._reward(gain, default_loss)
                          for expert_id, gain in expert_gains.items()}
        # ``rewards`` is what the controller actually consumed this round: the
        # blend for ``--fedtdo_reward_basis mixed``, the expert's own forecast
        # for ``expert``.  ``expert_rewards`` is always the latter, so a run can
        # be replayed under either ruler without retraining.
        consumed = expert_rewards if client.reward_basis == 'expert' else rewards
        self.round_log.append({
            'step': int(step),
            'client': client.client_id,
            'active_expert': record.active_expert_id,
            'probes': len(record.probed_expert_ids),
            'probe_ids': list(record.probed_expert_ids),
            'priors': record.priors,
            'default_loss': default_loss,
            'deployed_loss': float(stats['deployed_loss']),
            'best_gain': float(max(stats['gains'].values())) if stats['gains'] else 0.0,
            # ``raw_gains`` is the true objective (MSE difference of the blend);
            # ``expert_raw_gains`` is the MSE difference of the expert alone.
            'raw_gains': {eid: float(gain) for eid, gain in stats['gains'].items()},
            'expert_raw_gains': {eid: float(gain) for eid, gain in expert_gains.items()},
            'gains': consumed,
            'mix_rewards': rewards,
            'expert_rewards': expert_rewards,
            'alphas': {expert_id: client.alpha_for(expert_id) for expert_id in rewards},
            'cumulative_drift': float(client.controller.cumulative_drift),
        })

    @staticmethod
    def _action_summary(clients):
        evaluated = sum(client.cost_model.summary().get('evaluations', 0) for client in clients)
        transfers = sum(client.cost_model.summary().get('transfers', 0) for client in clients)
        deployed = sum(
            1 for client in clients if client.controller.active_expert_id is not None
        )
        default_loss = sum(client.cumulative_default_loss for client in clients)
        deployed_loss = sum(client.cumulative_deployed_loss for client in clients)
        return {
            'evaluations': evaluated,
            'transfers': transfers,
            'clients_with_active_expert': deployed,
            'default_loss': default_loss,
            'deployed_loss': deployed_loss,
            'relative_gain': (default_loss - deployed_loss) / default_loss if default_loss else 0.0,
        }
