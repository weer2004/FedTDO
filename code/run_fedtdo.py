"""Standalone entry point for the FedTDO causal federated experiment."""

import argparse
import random

import numpy as np
import torch

from exp.exp_fedtdo import Exp_FedTDO
from utils.tools import force_cuda


def parser():
    argument_parser = argparse.ArgumentParser(description='FedTDO federated online forecasting')
    argument_parser.add_argument('--model_id', required=True)
    argument_parser.add_argument('--data', required=True,
                                 choices=['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2',
                                          'Solar', 'ECL', 'Electricity', 'Traffic',
                                          'Weather', 'Exchange', 'ILI', 'custom'])
    argument_parser.add_argument('--root_path', required=True)
    argument_parser.add_argument('--data_path', required=True)
    argument_parser.add_argument('--features', default='M', choices=['M', 'S', 'MS'])
    argument_parser.add_argument('--target', default='OT')
    argument_parser.add_argument('--freq', default='h')
    argument_parser.add_argument('--embed', default='timeF')
    argument_parser.add_argument('--seasonal_patterns', default='Monthly')
    argument_parser.add_argument('--task_name', default='long_term_forecast')
    argument_parser.add_argument('--seq_len', type=int, default=96)
    argument_parser.add_argument('--label_len', type=int, default=48)
    argument_parser.add_argument('--pred_len', type=int, default=96)
    argument_parser.add_argument('--enc_in', type=int, required=True)
    argument_parser.add_argument('--d_model', type=int, default=128)
    argument_parser.add_argument('--dropout', type=float, default=0.1)
    argument_parser.add_argument('--patch_len', type=int, default=16)
    argument_parser.add_argument('--stride', type=int, default=16)
    argument_parser.add_argument('--cycle_length', type=int, default=1)
    argument_parser.add_argument('--channel_independence', type=int, default=1)
    argument_parser.add_argument('--use_relation_layer', type=int, default=0)
    argument_parser.add_argument('--num_temporal_moe_layers', type=int, default=1)
    argument_parser.add_argument('--num_experts', type=int, default=4)
    argument_parser.add_argument('--num_drift_experts', type=int, default=0)
    argument_parser.add_argument('--num_rnn_layers', type=int, default=1)
    argument_parser.add_argument('--top_k', type=int, default=2)
    argument_parser.add_argument('--revin_affine', action='store_true', default=False)
    argument_parser.add_argument('--enable_drift_detection', type=int, default=1)
    argument_parser.add_argument('--drift_window_size', type=int, default=64)
    argument_parser.add_argument('--drift_history_size', type=int, default=20)
    argument_parser.add_argument('--drift_k_sigma', type=float, default=3.0)
    argument_parser.add_argument('--batch_size', type=int, default=32)
    argument_parser.add_argument('--num_workers', type=int, default=0)
    argument_parser.add_argument('--learning_rate', type=float, default=1e-4)
    argument_parser.add_argument('--augmentation_ratio', type=int, default=0)
    argument_parser.add_argument('--use_gpu', action='store_true', default=False)
    argument_parser.add_argument('--gpu', type=int, default=0)
    argument_parser.add_argument('--seed', type=int, default=2021)

    argument_parser.add_argument('--fedtdo_num_clients', type=int, default=4)
    argument_parser.add_argument('--fedtdo_channels_per_client', type=int, default=1,
                                 help='columns each client holds (enc_in). 1 = the '
                                      'univariate protocol every existing table uses; '
                                      'k>1 needs num_clients*k channels and restores '
                                      'the cross-variate paths. Mirror this on '
                                      'run_baselines.py or the two tables diverge')
    argument_parser.add_argument('--fedtdo_partition', default='channel',
                                 choices=['channel', 'time'],
                                 help='channel = one client per sensor (plan §10.1, '
                                      'comparable with FL baselines); '
                                      'time = contiguous shards (plan §10.2)')
    argument_parser.add_argument('--fedtdo_channel_sampling', default='random',
                                 choices=['random', 'head'],
                                 help='how to pick clients when the dataset has more '
                                      'channels than --fedtdo_num_clients; head '
                                      'reproduces the old cols[:N] behaviour')
    argument_parser.add_argument('--fedtdo_control', default='fedtdo',
                                 choices=['fedtdo', 'fedtdo_nofreeze', 'fedtdo_noavg',
                                          'fedtdo_joint', 'fedtdo_localfreeze',
                                          'isomorphic', 'local_encoder'],
                                 help='fedtdo = PatchEmbedding warm-up + FedAvg + freeze; '
                                      'fedtdo_nofreeze = same but the averaged encoder stays '
                                      'trainable (diagnostic for the version gate); '
                                      'fedtdo_noavg = encoder-only warm-up and freeze but no '
                                      'FedAvg averaging, which splits the warm-up cost into '
                                      'its staging and averaging halves; '
                                      'fedtdo_joint = warm up every parameter (only the encoder '
                                      'is averaged and frozen), which tests whether the damage '
                                      'comes from training the encoder against an untrained '
                                      'downstream; '
                                      'fedtdo_localfreeze = same as fedtdo_joint but without '
                                      'FedAvg, which separates the value of aggregation from '
                                      'the value of freezing; '
                                      'isomorphic = identical backbone trained end-to-end '
                                      'for the same number of steps, no sharing and no '
                                      'freeze (fair control for "does migration itself pay"); '
                                      'local_encoder = per-client encoder, expert capsules '
                                      'are the only shared unit, transfer always on')
    argument_parser.add_argument('--fedtdo_disable_transfer', type=int, default=0,
                                 help='never probe or migrate; the controller only keeps '
                                      'local, which turns the run into an isomorphic local '
                                      'control with zero decision cost')
    argument_parser.add_argument('--fedtdo_use_relation_layer', type=int, default=0,
                                 help='give the experts a cross-variate relation layer with '
                                      'n_vars = channels_per_client. Only meaningful when '
                                      'channels_per_client > 1, where it is the one module '
                                      'that actually mixes channels; 0 keeps every existing '
                                      'channel-partition number reproducible')
    argument_parser.add_argument('--fedtdo_disable_drift', type=int, default=0,
                                 help='ablation switch: do not attach the drift detector. '
                                      'Without it the drift score cannot be separated from '
                                      'the transfer policy, because every migration decision '
                                      'consumes a drift term')
    argument_parser.add_argument('--fedtdo_adapter', default='none',
                                 choices=['none', 'private'],
                                 help='private calibration between the shared frozen encoder '
                                      'and the MoE stack. none = historical arm. private = '
                                      'a per-client affine (2*d_model parameters, identity '
                                      'at init) applied to encoder features; the trunk stays '
                                      'shared and frozen, so capsules remain portable while '
                                      'each client can undo the shared encoder mismatch '
                                      '(measured at 7.96%% of the default path)')
    argument_parser.add_argument('--fedtdo_share_modules', default='patch_embedding',
                                 help='comma-separated modules that are FedAvg-ed across '
                                      'clients: patch_embedding, temporal_moe_layers, '
                                      'projection (default = historical behaviour)')
    argument_parser.add_argument('--fedtdo_share_prox', type=float, default=0.0,
                                 help='0 keeps the historical hard freeze; >0 turns the '
                                      'shared modules into a soft constraint with '
                                      '0.5*lambda*||theta - theta_shared||^2, so a client '
                                      'whose channel cannot reuse the average can still '
                                      'adapt instead of being forced to keep it')
    argument_parser.add_argument('--fedtdo_sync_every', type=int, default=0,
                                 help='>0 re-averages the shared modules every N downstream '
                                      'epochs and keeps them trainable in between (periodic '
                                      'sharing). 0 keeps the historical one-shot average '
                                      'followed by a hard freeze')
    argument_parser.add_argument('--fedtdo_online_lr', type=float, default=0.0,
                                 help='lr of the one-step SGD taken on each revealed batch '
                                      'during the online stream (0 = static model). Matches '
                                      'run_baselines.py --fl_online_lr, which is what makes '
                                      'the late-stream drift trackable')
    argument_parser.add_argument('--fedtdo_online_target', default='full',
                                 choices=['full', 'first', 'prefix'],
                                 help="'full' = H-step objective revealed after H windows "
                                      "(causal but stale); 'first' = one-step objective "
                                      "revealed at the next window (causal and fresh)")
    argument_parser.add_argument('--fedtdo_online_prefix', type=int, default=4,
                                 help='with --fedtdo_online_target prefix: how many of '
                                      'the window''s steps must be revealed before it is '
                                      'used for an update (1 = the first mode, pred_len = '
                                      'the delayed full-horizon mode)')
    argument_parser.add_argument('--fedtdo_grad_clip', type=float, default=0.0,
                                 help='diagnostic only: clip online gradient norm to this '
                                      'value. 0 (default) leaves the update untouched')
    argument_parser.add_argument('--fedtdo_online_substeps', type=int, default=1,
                                 help='split each online stream batch into N sub-batches, '
                                      'i.e. N gradient steps per revealed batch. 32 matches '
                                      'the per-sample granularity of the FL baselines')
    argument_parser.add_argument('--fedtdo_encoder_rounds', type=int, default=1)
    argument_parser.add_argument('--fedtdo_warmup_steps', type=int, default=10,
                                 help='local PatchEmbedding warm-up steps per federated round; 0 disables')
    argument_parser.add_argument('--fedtdo_local_steps', type=int, default=50,
                                 help='local post-warmup training steps; 0 disables')
    # DTMoe-aligned training protocol (see PROTOCOL_DTMoe_ALIGNED.md).  The
    # step-based budget above is what every pre-existing result used; switching
    # to epochs makes a run comparable with the published long-term tables,
    # because it brings back DTMoe's lr decay and validation-based early
    # stopping.  ``step_mode`` is recorded in metrics.json, so the two families
    # of results can never be confused.
    argument_parser.add_argument('--fedtdo_step_mode', default='steps',
                                 choices=['steps', 'epochs'],
                                 help='steps: fixed gradient-step budget, constant lr. '
                                      'epochs: DTMoe protocol (whole epochs, lr decay, '
                                      'early stopping on val)')
    argument_parser.add_argument('--fedtdo_warmup_epochs', type=int, default=5,
                                 help='epochs of federated encoder warm-up per round '
                                      '(epochs mode only)')
    argument_parser.add_argument('--fedtdo_epochs', type=int, default=60,
                                 help='downstream epochs per client (epochs mode only); '
                                      'DTMoe uses --train_epochs')
    argument_parser.add_argument('--fedtdo_patience', type=int, default=15,
                                 help='early-stopping patience on val loss (epochs mode only)')
    argument_parser.add_argument('--fedtdo_lradj', default='type3',
                                 choices=['type1', 'type3', 'cosine', 'constant'],
                                 help='lr schedule, same names as run.py; type3 is the '
                                      'DTMoe default (constant 3 epochs, then x0.8)')
    argument_parser.add_argument('--fedtdo_transfer_layer', type=int, default=0)
    argument_parser.add_argument('--fedtdo_registry_size', type=int, default=256)
    argument_parser.add_argument('--fedtdo_eval_budget', type=int, default=2)
    argument_parser.add_argument('--fedtdo_initial_probes', type=int, default=1)
    argument_parser.add_argument('--fedtdo_feedback_delay', type=int, default=0)
    argument_parser.add_argument('--fedtdo_mix_alpha', type=float, default=0.5)
    argument_parser.add_argument('--fedtdo_beta', type=float, default=1.0)
    argument_parser.add_argument('--fedtdo_eta_age', type=float, default=0.01)
    argument_parser.add_argument('--fedtdo_eta_drift', type=float, default=0.1,
                                 help='per-unit-drift forgetting rate; 0.1 gives a '
                                      'half-life of ~7 cumulative drift units. The old '
                                      'default 1.0 was calibrated against a signal that '
                                      'was zeroed 70%% of the time and collapses the '
                                      'effective sample size once the signal is live')
    argument_parser.add_argument('--fedtdo_lambda_eval', type=float, default=0.0)
    argument_parser.add_argument('--fedtdo_lambda_transfer', type=float, default=0.0)
    argument_parser.add_argument('--fedtdo_lambda_switch', type=float, default=0.0)
    argument_parser.add_argument('--fedtdo_include_self', type=int, default=0,
                                 help='A3: let a client keep its own earlier snapshots in '
                                      'the candidate pool (temporal experts, zero bytes)')
    argument_parser.add_argument('--fedtdo_deploy_topk', type=int, default=1,
                                 help='average the calibrated deltas of the k most trusted '
                                      'experts into the deployed forecast. 1 = published '
                                      'single-expert blend')
    argument_parser.add_argument('--fedtdo_evidence_k', type=int, default=8,
                                 help='decision=evidence: number of recent paired gains used '
                                      'by the one-sided t test')
    argument_parser.add_argument('--fedtdo_evidence_t', type=float, default=1.0,
                                 help='decision=evidence: minimum t statistic to deploy')
    argument_parser.add_argument('--fedtdo_evidence_gate', default='ranked',
                                 choices=['leader', 'ranked'],
                                 help='decision=evidence: which candidate the t test is '
                                      'applied to. ranked = first passing candidate in '
                                      'score order (default); leader = top-scoring '
                                      'candidate only, else stay local (earlier '
                                      'behaviour)')
    argument_parser.add_argument('--fedtdo_switch_margin', type=float, default=0.0,
                                 help='deadband in reward units: a candidate must beat the '
                                      'incumbent expert by this much before switching. 0 = '
                                      'published behaviour')

    argument_parser.add_argument('--fedtdo_transfer_threshold', type=float, default=0.0)
    argument_parser.add_argument('--fedtdo_probe_threshold', type=float, default=0.0)
    argument_parser.add_argument('--fedtdo_probe_floor', type=int, default=1,
                                 help='always fill the evaluation budget; 0 restores the '
                                      'pre-fix behaviour where a threshold on the optimism '
                                      'score could switch exploration off')
    argument_parser.add_argument('--fedtdo_min_observations', type=int, default=1)
    argument_parser.add_argument('--fedtdo_shock', default='raw',
                                 choices=['raw', 'centered'],
                                 help='raw stores each arm reward as measured (the round shock is '
                                      'kept and averaged over); centered stores gain - round_mean, '
                                      'i.e. the paired reward, and reconstructs the level with '
                                      'baseline_hat(). Probes and the deployment decision then share '
                                      'one shock-free statistic. Measured within-arm lag-1 '
                                      'autocorrelation falls from 0.635 to 0.365')
    argument_parser.add_argument('--fedtdo_decorr_rho', type=float, default=0.0,
                                 help='design-effect coefficient for the confidence radius: '
                                      'n_eff = n / (1 + (n-1) rho). 0 keeps the iid radius; '
                                      'use the measured residual autocorrelation (0.365 after '
                                      'centering) to make the radius honest')
    argument_parser.add_argument('--fedtdo_baseline_k', type=float, default=0.6,
                                 help='weight of the last round mean when reconstructing the '
                                      'level under --fedtdo_shock centered (set to the measured '
                                      'lag-1 autocorrelation of the round shock)')
    argument_parser.add_argument('--fedtdo_delta', type=float, default=0.1,
                                 help='confidence delta of the log(2KT/delta) radius (plan §4.3)')
    argument_parser.add_argument('--fedtdo_gain_mode', default='relative',
                                 choices=['relative', 'raw'],
                                 help='relative (default) feeds (default_loss - candidate_loss) '
                                      '/ default_loss to the UCB, so the reward is dimensionless and '
                                      'the radius is comparable across regimes; raw reproduces the '
                                      'previous absolute-MSE behaviour')
    argument_parser.add_argument('--fedtdo_estimator', default='mean',
                                 choices=['mean', 'winsor', 'cvar', 'quantile'],
                                 help='gain statistic used by the deployment decision: mean '
                                      '(historical), winsor (winsorised mean), cvar (mean of '
                                      'the best k rounds), quantile (upper quantile / vote share). '
                                      'See '
                                      'federated/selector.py and scripts/replay_selector.py')
    argument_parser.add_argument('--fedtdo_decision', default='lcb',
                                 choices=['lcb', 'posterior', 'evidence'],
                                 help='lcb = deploy only if statistic - radius clears the threshold '
                                      '(historical, keeps the client local with ~4 observations per '
                                      'arm); posterior = compare the shrunk statistic itself with '
                                      'the threshold and use the radius for exploration only, as '
                                      'plan §5.2 specifies for the migration step')
    argument_parser.add_argument('--fedtdo_trim_lower', type=float, default=0.1,
                                 help='lower winsorising quantile of --fedtdo_estimator winsor')
    argument_parser.add_argument('--fedtdo_trim_upper', type=float, default=0.9,
                                 help='upper winsorising quantile of --fedtdo_estimator winsor')
    argument_parser.add_argument('--fedtdo_upper_q', type=float, default=0.75,
                                 help='tail split of --fedtdo_estimator cvar / quantile')
    argument_parser.add_argument('--fedtdo_shrink_group', default='global',
                                 choices=['global', 'role'],
                                 help='level the empirical-Bayes prior is pooled at: global uses '
                                      'every arm, role pools arms that occupy the same expert slot '
                                      'across source clients. Arms are not exchangeable - on '
                                      'Solar-Energy slot E1 averages +0.004 while E0/E2/E3 average '
                                      '-0.26/-0.46/-0.29 - so global pooling lands on a negative '
                                      'prior and suppresses every deployment')
    argument_parser.add_argument('--fedtdo_shrink_n0', type=float, default=0.0,
                                 help='empirical-Bayes pseudo-count pulling each arm toward the '
                                      'pooled reward mean; 0 disables. With ~4 observations per '
                                      'arm a positive n0 is what makes per-arm ranking possible')
    argument_parser.add_argument('--fedtdo_shrink_n0_role', type=float, default=5.0,
                                 help='pseudo-count with which a role pool is pulled toward the '
                                      'host pool in --fedtdo_hierarchy host_role')
    argument_parser.add_argument('--fedtdo_hierarchy', default='none',
                                 choices=['none', 'host_role'],
                                 help='none = one flat arm statistic; host_role = nested '
                                      'host -> role -> arm estimate (see '
                                      'scripts/probe_conditional.py for the variance '
                                      'decomposition that motivates it)')
    argument_parser.add_argument('--fedtdo_reward_basis', default='mixed',
                                 choices=['mixed', 'expert'],
                                 help='what the controller scores an arm on: mixed = the '
                                      'alpha-blended forecast it would deploy (historical), '
                                      'expert = the expert\'s own forecast. The mixing weight '
                                      'is fitted per arm from revealed labels, so the blended '
                                      'ruler also measures how often the arm was seen')
    argument_parser.add_argument('--fedtdo_clip_reward', type=int, default=0,
                                 help='clip the reward into [-1, 1] (historical). Off by '
                                      'default: clipping turns a -300%% night round into '
                                      '-100%% and erases the magnitude the decision needs')
    argument_parser.add_argument('--fedtdo_alpha_calibration', type=int, default=1,
                                 help='fit the local mix weight by least squares on revealed labels')
    argument_parser.add_argument('--fedtdo_alpha_min_obs', type=int, default=3,
                                 help='revealed batches required before the calibrated alpha is used')
    argument_parser.add_argument('--fedtdo_cache_limit', type=int, default=0,
                                 help='max cached capsules per client; 0 = unbounded')
    argument_parser.add_argument('--fedtdo_online_steps', type=int, default=None,
                                 help='cap the online steps per client (debug / regime loops)')
    argument_parser.add_argument('--fedtdo_stream_batch', type=int, default=0,
                                 help='batch size of the online test stream only (0 = same as '
                                      '--batch_size); needed when a short series cannot fill the '
                                      'round budget at the training batch size')
    argument_parser.add_argument('--fedtdo_eval_split', default='test',
                                 choices=['test', 'val'],
                                 help='which held-out split feeds the online stream. test keeps '
                                      'every table in this repo comparable; val is the leak-free '
                                      'choice when the run is used to *select* a hyper-parameter '
                                      '(e.g. the learning-rate sweep), because otherwise the '
                                      'stream that chooses lr is the stream that reports it')
    argument_parser.add_argument('--fedtdo_output_dir', default='./results/fedtdo')
    argument_parser.add_argument('--paper_record', type=int, default=0,
                                 help='1 = write the paper-grade event tables '
                                      '(per-window metrics, predictions, updates, '
                                      'candidate evaluations, transfers, '
                                      'communication and cost events). Off by '
                                      'default so existing runs are unaffected')
    argument_parser.add_argument('--paper_results_dir',
                                 default='./results/paper_fedtdo',
                                 help='root of the paper data tree; the run lands '
                                      'in raw/<dataset>/<method>/seed_<s>/run_<id>/')
    return argument_parser


if __name__ == '__main__':
    args = parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    force_cuda(args)
    print(f'Using GPU cuda:{args.gpu} ({torch.cuda.get_device_name(args.gpu)})')
    Exp_FedTDO(args).run()
