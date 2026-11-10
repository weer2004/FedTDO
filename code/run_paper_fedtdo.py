"""Entry point: FedTDO on the dataset-level multivariate protocol.

Companion of ``run_paper_protocol.py`` (FedAvg / local on the same data).  Every
FedTDO knob keeps its ``run_fedtdo.py`` default; this file only adds the
protocol arguments and makes ``--data`` optional, because the client list comes
from ``--datasets``.
"""

from __future__ import annotations

import argparse
import json
import random

import numpy as np
import torch

from exp.exp_paper_fedtdo import MAIN_DATASETS, Exp_PaperFedTDO
from utils.tools import force_cuda
from run_fedtdo import parser as fedtdo_parser


def parser():
    p = fedtdo_parser()
    # the dataset list replaces --data/--root_path/--data_path
    for action in list(p._actions):
        if action.dest in ('data', 'root_path', 'data_path'):
            action.required = False
            action.default = None if action.dest == 'data' else ''
        if action.dest == 'model_id':
            # derived from the setting, never typed by hand
            action.required = False
            action.default = 'ppf'
        if action.dest == 'enc_in':
            # every client keeps ``--paper_channels`` channels; the value is
            # copied onto the model args inside the experiment
            action.required = False
            action.default = None
    p.add_argument('--datasets', nargs='+', default=MAIN_DATASETS,
                   choices=MAIN_DATASETS,
                   help='one client per dataset (FFTS/FeDaL protocol)')
    p.add_argument('--paper_channels', type=int, default=7,
                   help='channels kept per dataset; 0 keeps every native channel')
    p.add_argument('--eval_batch_size', type=int, default=128)
    p.add_argument('--tag', type=str, default='')
    p.add_argument('--arm', default='', choices=['', 'base', 'center', 'decorr'],
                   help='label only: the shock/rho flags of the arm still have to be '
                        'passed explicitly, exactly as scripts/xc_sweep.py does')
    return p


def setting_name(args) -> str:
    tag = f'_{args.tag}' if args.tag else ''
    return (f'PPF_{args.fedtdo_control}_{args.arm or "base"}_pl{args.pred_len}'
            f'_dm{args.d_model}_c{args.paper_channels}'
            f'_w{args.fedtdo_warmup_steps}+l{args.fedtdo_local_steps}{tag}')


if __name__ == '__main__':
    args = parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    force_cuda(args)
    if not hasattr(args, 'arm'):
        args.arm = ''
    name = setting_name(args)
    args.model_id = name
    print(f'[pp-fedtdo] arm={args.arm or "base"} control={args.fedtdo_control} '
          f'device={args.device} datasets={args.datasets}')
    summary = Exp_PaperFedTDO(args).run(name)
    print(json.dumps({'offline_mean_mse_trained': summary['offline_mean_mse_trained'],
                      'online_mse': summary['mse']}, indent=2))
