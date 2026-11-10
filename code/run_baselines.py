"""
Run FL baselines adapted for Dynamic-TMoE federated time-series forecasting.

Examples:
  python run_baselines.py --baseline fedavg --data ETTh1 --num_clients 4
  python run_baselines.py --baseline fedpoe --fl_rounds 5 --fl_online_steps 100
  python run_baselines.py --list_baselines
"""

import argparse
import random

import numpy as np
import torch

from fl_baselines.registry import list_baselines
from exp.exp_fl_baselines import Exp_FL_Baselines
from utils.tools import force_cuda


def build_parser():
    parser = argparse.ArgumentParser(description='FedTDO FL Baselines')
    parser.add_argument('--list_baselines', action='store_true', help='print registry and exit')

    parser.add_argument('--baseline', type=str, default='fedavg',
                        help='local|fedavg|fedala|scaffold|fedadam|fednova|fedprox|fedper|fedrep|'
                             'ditto|pfedme|fedl2p|pfedfda|fedpoe|pfedcs|fedkwaz|fedpkda|fedkme')
    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    parser.add_argument('--model_id', type=str, default='fl_baseline')
    parser.add_argument('--model', type=str, default='Dynamic_TMoE')
    parser.add_argument('--backbone', type=str, default='dtmoe',
                        choices=['dtmoe', 'patchtst', 'itransformer', 'transformer',
                                 'autoformer', 'fedformer', 'timesnet', 'timemixer', 'gru',
                                 'tfps', 'stmtm', 'raft', 'ffts', 'timeffm'],
                        help='per-client forecaster; dtmoe = the paper backbone')
    parser.add_argument('--des', type=str, default='exp')

    # data
    parser.add_argument('--data', type=str, default='ETTh1')
    parser.add_argument('--root_path', type=str, default='./dataset/ETT-small/')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv')
    parser.add_argument('--features', type=str, default='S')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--num_workers', type=int, default=32,
                        help='DataLoader workers')
    parser.add_argument('--num_threads', type=int, default=1,
                        help='PyTorch/OMP intra-op threads (use 1 when num_workers is high)')
    parser.add_argument('--augmentation_ratio', type=int, default=0)
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

    # sizes
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly')

    # model
    parser.add_argument('--enc_in', type=int, default=1)
    parser.add_argument('--dec_in', type=int, default=1)
    parser.add_argument('--c_out', type=int, default=1)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--expand', type=int, default=2)
    parser.add_argument('--d_conv', type=int, default=4)
    parser.add_argument('--top_k', type=int, default=2)
    parser.add_argument('--distil', action='store_false', default=True)
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--channel_independence', type=int, default=1)
    parser.add_argument('--patch_len', type=int, default=16)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--cycle_length', type=int, default=24)
    parser.add_argument('--use_relation_layer', type=int, default=0)
    parser.add_argument('--fedtdo_use_relation_layer', type=int, default=0)
    parser.add_argument('--num_temporal_moe_layers', type=int, default=1)
    parser.add_argument('--num_rnn_layers', type=int, default=1)
    parser.add_argument('--num_drift_experts', type=int, default=1)
    parser.add_argument('--num_experts', type=int, default=4)
    parser.add_argument('--baseline_enable_drift', type=int, default=0)
    parser.add_argument('--fedda_clusters', type=int, default=0,
                        help='FedDA: number of client clusters (0 = sqrt(n))')
    parser.add_argument('--fedda_mixup', type=float, default=0.2,
                        help='FedDA: mixup Beta concentration (0 disables)')
    parser.add_argument('--fedgtp_beta', type=float, default=0.5,
                        help='FedGTP: graph weight (0 = FedAvg, 1 = neighbours only)')

    # drift (optional)
    parser.add_argument('--enable_drift_detection', type=int, default=0)
    parser.add_argument('--drift_window_size', type=int, default=128)
    parser.add_argument('--drift_history_size', type=int, default=20)
    parser.add_argument('--drift_k_sigma', type=float, default=3.0)

    # optim / FL protocol
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--paper_record', type=int, default=0,
                        help='1 = write the paper-grade event tables under '
                             '--paper_results_dir (off by default)')
    parser.add_argument('--paper_results_dir', default='./results/paper_fedtdo')
    parser.add_argument('--results_dir', default='./results_baselines')
    parser.add_argument('--fl_online_target', default='first', choices=['full','first'])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--train_epochs', type=int, default=3)
    parser.add_argument('--num_clients', type=int, default=4)
    parser.add_argument('--channels_per_client', type=int, default=1,
                        help='columns each client holds (enc_in). Must match '
                             'run_fedtdo.py --fedtdo_channels_per_client, otherwise '
                             'the two tables describe different splits. Requires '
                             'num_clients * k channels')
    parser.add_argument('--fedtdo_channel_sampling', type=str, default='random',
                        choices=['random', 'head'],
                        help='client subset selection; must match run_fedtdo.py')
    parser.add_argument('--fl_local_max_steps', type=int, default=0,
                        help='cap gradient steps per client (0 = epoch budget). Use this '
                             'to match the FedTDO step budget exactly')
    parser.add_argument('--fl_rounds', type=int, default=5)
    parser.add_argument('--fl_local_epochs', type=int, default=1)
    # DTMoe-aligned budget. Default stays 'steps' so every existing
    # results_baselines/ cell keeps its meaning; 'epochs' switches to
    # whole-epoch training with lr decay and validation early stopping, which is
    # what a table next to run.py (or next to run_fedtdo.py --fedtdo_step_mode
    # epochs) requires. See PROTOCOL_DTMOE_ALIGNED.md.
    parser.add_argument('--fl_step_mode', default='steps', choices=['steps', 'epochs'])
    parser.add_argument('--fl_patience', type=int, default=15,
                        help='early-stopping patience in epochs mode')
    parser.add_argument('--fl_lradj', default='type3',
                        choices=['type1', 'type3', 'cosine', 'constant'])
    parser.add_argument('--fl_online_steps', type=int, default=100)
    parser.add_argument('--fl_online_lr', type=float, default=0.0,
                        help='lr of the SGD update on a stored full-horizon target after its final '
                             'label arrives (origin t + pred_len); 0 = static model')
    parser.add_argument('--fedtdo_online_steps', type=int, default=None,
                        help='alias of fl_online_steps for protocol parity with FedTDO')
    parser.add_argument('--fedtdo_stream_batch', type=int, default=0,
                        help='override the online-stream batch size only (0 = use --batch_size). '
                             'Must match run_fedtdo.py --fedtdo_stream_batch, otherwise ILI '
                             'evaluates a different number of windows than FedTDO does')

    # method-specific knobs
    parser.add_argument('--fedala_eta', type=float, default=1.0)
    parser.add_argument('--fedala_steps', type=int, default=5)
    parser.add_argument('--fedala_rand_percent', type=float, default=0.8)
    parser.add_argument('--fedl2p_ft_epochs', type=int, default=1)
    parser.add_argument('--pfedfda_shrink', type=float, default=0.5)
    parser.add_argument('--fedpoe_bank_size', type=int, default=5)
    parser.add_argument('--fedpoe_eta', type=float, default=0.2)
    parser.add_argument('--fedpoe_store_every', type=int, default=1)
    parser.add_argument('--pfedcs_top_m', type=int, default=2)
    parser.add_argument('--pfedcs_temp', type=float, default=1.0)
    parser.add_argument('--fedkwaz_alpha', type=float, default=0.3)
    parser.add_argument('--fedkwaz_mix', type=float, default=0.5)
    parser.add_argument('--fedpkda_noise', type=float, default=0.1)
    parser.add_argument('--fedpkda_clip', type=float, default=0.0,
                        help='prototype clip radius; <=0 means auto (median observed norm)')
    parser.add_argument('--fedpkda_align', type=float, default=0.1)
    parser.add_argument('--fedkme_rff_dim', type=int, default=64)
    parser.add_argument('--fedkme_sigma', type=float, default=1.0)

    # FedOpt / personalisation family
    parser.add_argument('--fedprox_mu', type=float, default=0.01,
                        help='proximal weight; 0 degenerates to FedAvg')
    parser.add_argument('--scaffold_opt', type=str, default='sgd', choices=['adamw', 'sgd'],
                        help='adamw keeps the optimiser identical across baselines; '
                             'sgd reproduces the paper exactly')
    parser.add_argument('--fedadam_lr', type=float, default=1e-4,
                        help='server-side Adam step size on the pseudo-gradient. '
                             'With one or two server steps the bias-corrected direction is '
                             '~10x lr per coordinate, so 0.005 diverged to NaN on ETTh1')
    parser.add_argument('--fedadam_beta1', type=float, default=0.9)
    parser.add_argument('--fedadam_beta2', type=float, default=0.99)
    parser.add_argument('--fedadam_tau', type=float, default=1e-3)
    parser.add_argument('--fednova_hetero', type=int, default=1,
                        help='1 = per-client step budgets proportional to loader size '
                             '(fleet total preserved); 0 = equal steps')
    parser.add_argument('--fednova_teff', type=str, default='fixed', choices=['fixed', 'mean'],
                        help='reference step count; mean makes this algebraically FedAvg')
    parser.add_argument('--fedrep_head_steps', type=int, default=0,
                        help='budget of the head phase; 0 = 25%% of fl_local_max_steps')
    parser.add_argument('--ditto_lambda', type=float, default=0.1)
    parser.add_argument('--pfedme_lambda', type=float, default=1.0)
    parser.add_argument('--pfedme_eta', type=float, default=0.1)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--gpu_type', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=2021)

    return parser


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()

    if args.list_baselines:
        for row in list_baselines():
            print(f"{row['name']:10s}  {row['venue']:16s}  {row['desc']}")
        raise SystemExit(0)

    if args.fedtdo_online_steps is None:
        args.fedtdo_online_steps = args.fl_online_steps

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    n_threads = max(1, int(args.num_threads))
    torch.set_num_threads(n_threads)
    try:
        torch.set_num_interop_threads(min(n_threads, 8))
    except RuntimeError:
        pass
    print(f'CPU threads: torch={torch.get_num_threads()} num_workers={args.num_workers}')

    force_cuda(args)
    print(f'Using GPU cuda:{args.gpu} ({torch.cuda.get_device_name(args.gpu)})')

    setting = (
        f'BL_{args.baseline}_{args.model_id}_{args.data}_nc{args.num_clients}_'
        f'sl{args.seq_len}_pl{args.pred_len}_dm{args.d_model}_r{args.fl_rounds}_{args.des}'
    )
    # A non-zero prequential learning rate must be part of the cell identity:
    # without it the online arm silently overwrites the static-evaluation cell of
    # the same name, and the driver's finished-cell check looks at a path that
    # will never exist.  Must match setting_name() in scripts/fl_baseline_compare.py.
    # Must match protocol_suffix() in scripts/fl_baseline_compare.py exactly,
    # otherwise the driver looks for a path that never exists and marks a
    # finished cell as FAILED (the run itself lands somewhere else).
    _lr = float(getattr(args, 'fl_online_lr', 0.0) or 0.0)
    _stream = int(getattr(args, 'fedtdo_stream_batch', 0) or 0) or 1
    if _lr > 0 or _stream != 1:
        setting += f'_olr{_lr:g}_sb{_stream}_os'
    print('Args:', {
        'baseline': args.baseline,
        'data': args.data,
        'num_clients': args.num_clients,
        'fl_rounds': args.fl_rounds,
        'fl_online_steps': args.fedtdo_online_steps,
        'd_model': args.d_model,
    })

    exp = Exp_FL_Baselines(args)
    summary = exp.run(setting)
    print('======= Baseline finished =======')
    print(f"{summary['method']} mean_mse={summary.get('mean_mse')} mean_mae={summary.get('mean_mae')}")
