"""Pick a training budget for Traffic by measuring loss against steps.

Solar-Energy has 1147 batches per client epoch, Traffic only 381.  Copying the
Solar step counts would silently train Traffic for 4.2 epochs while Solar ran
1.05, which confounds every comparison that changes the backbone.  This script
trains one channel and records the test loss as a function of the number of
steps, so ``--fedtdo_local_steps`` can be chosen from data instead of symmetry.

    python scripts/probe_budget.py --channels 274 427 449
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_provider.federated_data import (  # noqa: E402
    Dataset_UnivariateClient, _load_dataframe, _make_stamps, make_client_args,
)
from models.Dynamic_TMoE.model import Model  # noqa: E402


def build_args(**overrides) -> Namespace:
    base = dict(
        task_name='long_term_forecast',
        data='Traffic', root_path='./dataset/traffic/', data_path='traffic.csv',
        freq='h', features='S', target='0', seq_len=96, label_len=24, pred_len=24,
        enc_in=1, dec_in=1, c_out=1, batch_size=32, num_workers=0, seed=2021,
        embed='timeF', timeenc=0, scale=True, patch_len=16, stride=16,
        d_model=32, d_ff=128, n_heads=8, e_layers=1, dropout=0.1,
        num_experts=4, num_drift_experts=3, num_temporal_moe_layers=2,
        top_k=2, use_moe=1, channel_independence=1, use_relation_layer=0,
        learning_rate=5e-4, fedtdo_use_relation_layer=0,
        cycle_length=24, revin_affine=1, enable_drift_detection=1,
        drift_window_size=576, drift_history_size=20, drift_k_sigma=2.0,
        train_epochs=10, finetune_patience=5,
    )
    base.update(overrides)
    return Namespace(**base)


def data_for_channel(args, column, flag):
    frame = _load_dataframe(args.root_path, args.data_path)
    stamps = _make_stamps(frame, args.freq, 0)
    values = frame[column].values.astype(np.float64)
    return Dataset_UnivariateClient(
        values=values, stamps=stamps, flag=flag,
        seq_len=args.seq_len, label_len=args.label_len, pred_len=args.pred_len,
        scale=True,
    )


def evaluate(model, loader) -> float:
    model.eval()
    criterion = nn.MSELoss()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            batch_x = batch_x.float()
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float()
            out = model(batch_x, batch_x_mark)
            truth = batch_y[:, -model.pred_len:, :]
            total += criterion(out, truth).item() * len(batch_x)
            count += len(batch_x)
    return total / max(1, count)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--channels', nargs='+', default=['274', '427', '449'])
    parser.add_argument('--max-steps', type=int, default=2400)
    parser.add_argument('--every', type=int, default=150)
    parser.add_argument('--seed', type=int, default=2021)
    args_cli = parser.parse_args()

    print(f'{"steps":>6}{"epochs":>8}{"loss":>10}')
    for column in args_cli.channels:
        torch.manual_seed(args_cli.seed)
        np.random.seed(args_cli.seed)
        args = build_args()
        client_args = make_client_args(args, enc_in=1)
        train_ds = data_for_channel(client_args, column, 'train')
        test_ds = data_for_channel(client_args, column, 'test')
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

        model = Model(copy.deepcopy(client_args))
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        criterion = nn.MSELoss()

        batches_per_epoch = len(train_loader)
        step = 0
        model.train()
        while step < args_cli.max_steps:
            for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
                if step >= args_cli.max_steps:
                    break
                batch_x = batch_x.float()
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float()
                optimizer.zero_grad()
                out = model(batch_x, batch_x_mark)
                loss = criterion(out, batch_y[:, -model.pred_len:, :])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                step += 1
                if step % args_cli.every == 0:
                    test_loss = evaluate(model, test_loader)
                    print(f'{step:>6}{step / batches_per_epoch:>8.2f}{test_loss:>10.4f}')
                    model.train()


if __name__ == '__main__':
    main()
