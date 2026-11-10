"""FFTS/FeDaL-style federated backbone matrix (dataset-level clients, ``features M``).

One run = one federation over N datasets (default: the 9 main ones), each client
holding a multivariate block of the same width.  The backbone is the only thing
that changes between runs, so the resulting table is the "which architecture
should the federated model be" axis the two papers leave open.

Usage::

    python run_paper_protocol.py --backbone patchtst --method fedavg
    python run_paper_protocol.py --backbone timesnet --pred_len 96 --gpu 0
    python run_paper_protocol.py --backbone autoformer --holdout ETTh1   # zero-shot
    python scripts/xc_paper_matrix.py --backbones patchtst timesnet --dry-run
"""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from exp.exp_paper_protocol import MAIN_DATASETS, Exp_PaperProtocol
from models.zoo import ZOO
from utils.tools import force_cuda

BACKBONES = ['dtmoe'] + sorted(ZOO)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='FFTS/FeDaL-protocol federated backbone matrix')
    p.add_argument('--backbone', type=str, default='patchtst', choices=BACKBONES)
    p.add_argument('--method', type=str, default='fedavg', choices=['fedavg', 'local'],
                   help='fedavg = parameter averaging; local = no communication')
    p.add_argument('--agg_weight', type=str, default='uniform',
                   choices=['uniform', 'size'],
                   help='FedAvg client weighting: uniform (paper default) or '
                        'proportional to training-set size (FeDaL/FFTS style)')
    p.add_argument('--datasets', nargs='+', default=MAIN_DATASETS, choices=MAIN_DATASETS)
    p.add_argument('--holdout', type=str, default=None, choices=MAIN_DATASETS,
                   help='exclude this dataset from federated training and evaluate it '
                        'zero-shot (cross-dataset generalisation)')

    # data / shapes
    p.add_argument('--seq_len', type=int, default=96)
    p.add_argument('--label_len', type=int, default=48)
    p.add_argument('--pred_len', type=int, default=96)
    p.add_argument('--paper_channels', type=int, default=7,
                   help='channel budget C shared by every client (a global federated '
                        'model needs one enc_in); wide datasets are subsampled')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--eval_batch_size', type=int, default=128)
    p.add_argument('--num_workers', type=int, default=0)

    # architecture (matches scripts/xc_sweep SHARED where it overlaps)
    p.add_argument('--d_model', type=int, default=32)
    p.add_argument('--d_ff', type=int, default=64)
    p.add_argument('--n_heads', type=int, default=4)
    p.add_argument('--e_layers', type=int, default=2)
    p.add_argument('--d_layers', type=int, default=1)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--factor', type=int, default=3)
    p.add_argument('--patch_len', type=int, default=16)
    p.add_argument('--stride', type=int, default=8)
    p.add_argument('--modes', type=int, default=32)
    p.add_argument('--top_k', type=int, default=3)
    p.add_argument('--down_sampling_layers', type=int, default=1)
    p.add_argument('--moving_avg', type=int, default=25)

    # 2025 method-specific knobs
    p.add_argument('--raft_topk', type=int, default=3,
                   help='RAFT: retrieved training windows per query (0 disables)')
    p.add_argument('--tfps_patterns', type=int, default=4,
                   help='TFPS: number of pattern subspaces / experts')
    p.add_argument('--tfps_tau', type=float, default=1.0,
                   help='TFPS: subspace assignment temperature')
    p.add_argument('--stmtm_kernel', type=int, default=25,
                   help='ST-MTM: moving-average kernel for the trend component')
    p.add_argument('--stmtm_mask_ratio', type=float, default=0.4,
                   help='ST-MTM: contiguous trend block mask ratio')
    p.add_argument('--stmtm_pretrain_steps', type=int, default=400,
                   help='ST-MTM: self-supervised steps before the supervised budget '
                        '(0 disables the two-stage schedule)')

    # optimisation / federation
    p.add_argument('--learning_rate', type=float, default=5e-4)
    p.add_argument('--fl_rounds', type=int, default=2)
    p.add_argument('--local_mode', type=str, default='steps',
                   choices=['steps', 'epochs'],
                   help="steps: fixed gradient-step budget (comparable with the "
                        "channel-as-client table); epochs: FFTS/FeDaL style, local "
                        "work proportional to dataset size")
    p.add_argument('--fl_local_epochs', type=int, default=2)
    p.add_argument('--fl_local_max_steps', type=int, default=800,
                   help='gradient steps per client per round (budget parity with the '
                        'channel-as-client grid: 2 x 800 = 1600 steps, 1 aggregation)')
    p.add_argument('--seed', type=int, default=2021)

    # runtime
    p.add_argument('--use_gpu', type=int, default=1)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--num_threads', type=int, default=4)
    p.add_argument('--tag', type=str, default='')
    p.add_argument('--des', type=str, default='pp')
    return p


if __name__ == '__main__':
    args = build_parser().parse_args()

    # first line of every log: which interpreter, which torch, which device.
    # ffts / timeffm need the ma_fmarl env and refuse to start elsewhere.
    from utils.env_banner import print_env_banner
    print_env_banner(tag=args.backbone, feature=args.backbone)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, int(args.num_threads)))
    force_cuda(args)
    if args.use_gpu:
        # LM-backed backbones (FFTS / Time-FFM) do ~600 GFLOP of forward+backward
        # per step; without TF32 one dataset takes ~16 min and the 9-cell grid
        # becomes 2.4 h.  TF32 is the default precision for Ampere+ matmuls.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tag = f'_{args.tag}' if args.tag else ''
    # non-default aggregation must not collide with the uniform-weight cell
    weight_tag = '' if args.agg_weight == 'uniform' else f'_w{args.agg_weight}'
    budget_tag = f'_ep{args.fl_local_epochs}' if args.local_mode == 'epochs' else ''
    setting = (f'PP_{args.method}_{args.backbone}_pl{args.pred_len}_dm{args.d_model}'
               f'_c{args.paper_channels}_r{args.fl_rounds}{weight_tag}{budget_tag}{tag}')
    print(f'[paper] setting={setting}')
    print(f"[paper] backbone={args.backbone} method={args.method} "
          f"datasets={len(args.datasets)} holdout={args.holdout}")

    exp = Exp_PaperProtocol(args)
    summary = exp.run(setting)
    print(f"[paper] mean MSE over trained clients = {summary['mean_mse_trained']}")
