#!/usr/bin/env bash
# Channel-grouping ladder on Weather (21 channels).
#
# Every cell consumes all 21 channels, so the only thing that changes across a
# column is how those channels are distributed over clients:
#   k=1  -> nc=21 univariate clients (the original protocol)
#   k=3  -> nc=7  clients holding 3 channels each
#   k=7  -> nc=3  clients holding 7 channels each
#   k=21 -> nc=1  single client holding everything (centralised reference)
#
# Budget: 12003 windows / batch 256 = 47 steps per epoch on ETTh1, and the
# ETTh1 protocol used 60 epochs -> 2820 steps per client.  Weather has 36696
# training windows = 144 steps per epoch, so 20 epochs reproduces the same
# per-client gradient-step budget.
#
# Usage:  GPU=1 MAX_PARALLEL=2 nohup bash scripts/run_weather_channel_group.sh &
set -u
cd "$(dirname "$0")/.."

P=${PYTHON:-/mnt/sda1/yyj/miniconda3/envs/ma_fmarl/bin/python}
GPU=${GPU:-1}
MAX_PARALLEL=${MAX_PARALLEL:-2}
TOTAL_CHANNELS=${TOTAL_CHANNELS:-21}
OUT=${OUT:-./results/fedtdo_k}
LOGD=${LOGD:-./logs/weather_k}
EPOCHS=${EPOCHS:-20}

mkdir -p "$LOGD" "$OUT"

COMMON=(--data Weather --root_path ./dataset/weather/ --data_path weather.csv --freq h
        --seq_len 96 --label_len 48 --pred_len 96 --patch_len 48 --stride 12
        --d_model 128 --dropout 0.4 --batch_size 256 --learning_rate 0.0012
        --num_workers 0 --seed 2021)

run_fedtdo () {  # $1 = k
  local k=$1 nc=$(( TOTAL_CHANNELS / k ))
  nice -n 15 "$P" -u run_fedtdo.py "${COMMON[@]}" --enc_in "$k" \
    --use_gpu --gpu "$GPU" \
    --model_id wf_k"$k" \
    --fedtdo_partition channel --fedtdo_num_clients "$nc" \
    --fedtdo_channels_per_client "$k" \
    --fedtdo_step_mode epochs --fedtdo_epochs "$EPOCHS" \
    --fedtdo_warmup_epochs 2 --fedtdo_encoder_rounds 1 \
    --fedtdo_patience 15 --fedtdo_lradj type3 \
    --fedtdo_online_steps 60 --fedtdo_online_lr 1e-3 --fedtdo_online_substeps 32 \
    --fedtdo_stream_batch 32 --fedtdo_eval_budget 2 \
    --fedtdo_control fedtdo_nofreeze \
    --fedtdo_share_modules patch_embedding,temporal_moe_layers,projection \
    --fedtdo_output_dir "$OUT"
}

run_base () {  # $1 = baseline, $2 = k
  local m=$1 k=$2 nc=$(( TOTAL_CHANNELS / k ))
  nice -n 15 "$P" -u run_baselines.py "${COMMON[@]}" --baseline "$m" --enc_in "$k" \
    --features M --gpu "$GPU" \
    --num_clients "$nc" --channels_per_client "$k" \
    --fl_rounds "$EPOCHS" --fl_step_mode epochs --fl_local_epochs 1 \
    --fl_patience 15 --fl_lradj type3 \
    --fl_online_steps 60 --fl_online_lr 1e-3 \
    --fedtdo_stream_batch 32 \
    --model_id wf_k"$k" --des ep"$EPOCHS" --num_threads 4
}

submit () {  # $1 = tag, rest = command
  local tag=$1; shift
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do sleep 20; done
  echo "[start $(date '+%H:%M:%S')] $tag"
  "$@" > "$LOGD/$tag.log" 2>&1 &
}

# Cheapest cells first so a partial sweep still contains the k=21 reference.
for k in 21 7 3 1; do
  submit "fedtdo_k${k}" run_fedtdo "$k"
  submit "local_k${k}"  run_base local  "$k"
  submit "fedavg_k${k}" run_base fedavg "$k"
done

wait
echo "[done $(date '+%H:%M:%S')] all cells finished"
