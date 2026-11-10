#!/usr/bin/env bash
# ETTh1 / channel-nc7 transfer + drift ablations (the missing numbers).
#
# Everything except the switched-off mechanism is the best known ETTh1 cell:
# online lr 1e-3, substeps 1, step_mode epochs, 60 epochs, warm-up 5.
#
#   abl_e3_notransfer : --fedtdo_disable_transfer 1   -> value of migration
#   abl_e3_nodrift    : --fedtdo_disable_drift 1      -> value of drift detection
#   abl_e3_neither    : both off                      -> joint value
#   abl_e3_k7         : nc=1, k=7                     -> all 7 channels in one
#                                                       client (grouping reference)
#
# Usage: GPU=0 nohup bash scripts/run_ett_transfer_drift_ablation.sh &
set -u
cd "$(dirname "$0")/.."

P=${PYTHON:-/mnt/sda1/yyj/miniconda3/envs/ma_fmarl/bin/python}
GPU=${GPU:-0}
LOGD=${LOGD:-./logs/ett_abl}
OUT=${OUT:-./results/fedtdo_abl}

mkdir -p "$LOGD" "$OUT"

COMMON=(--data ETTh1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --freq h
        --seq_len 96 --label_len 48 --pred_len 96 --patch_len 48 --stride 12
        --d_model 128 --dropout 0.4 --batch_size 256 --learning_rate 0.0012
        --num_workers 0 --seed 2021 --use_gpu --gpu "$GPU"
        --fedtdo_partition channel --fedtdo_step_mode epochs
        --fedtdo_epochs 60 --fedtdo_warmup_epochs 5 --fedtdo_encoder_rounds 1
        --fedtdo_patience 15 --fedtdo_lradj type3
        --fedtdo_online_steps 60 --fedtdo_online_lr 1e-3
        --fedtdo_stream_batch 32 --fedtdo_eval_budget 2
        --fedtdo_control fedtdo_nofreeze
        --fedtdo_share_modules patch_embedding,temporal_moe_layers,projection
        --fedtdo_output_dir "$OUT")

launch () {  # $1 = model_id, rest = extra flags
  local mid=$1; shift
  echo "[start $(date '+%H:%M:%S')] $mid"
  nice -n 10 "$P" -u run_fedtdo.py "${COMMON[@]}" --enc_in 1 \
    --fedtdo_num_clients 7 --model_id "$mid" "$@" \
    > "$LOGD/$mid.log" 2>&1 &
}

launch abl_e3_notransfer --fedtdo_disable_transfer 1
launch abl_e3_nodrift    --fedtdo_disable_drift 1
launch abl_e3_neither    --fedtdo_disable_transfer 1 --fedtdo_disable_drift 1

echo "[start $(date '+%H:%M:%S')] abl_e3_k7"
nice -n 10 "$P" -u run_fedtdo.py "${COMMON[@]}" --enc_in 7 \
  --fedtdo_num_clients 1 --fedtdo_channels_per_client 7 --model_id abl_e3_k7 \
  > "$LOGD/abl_e3_k7.log" 2>&1 &

wait
echo "[done $(date '+%H:%M:%S')]"
