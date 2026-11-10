#!/bin/bash
# FedTDO on ETTh1 — aligned with scripts/run_all_baselines.sh protocol
# (4 clients, sl96/pl96/dm64, local_epochs≈5, online_steps=100)

# Keep intra-op threads at 1 when DataLoader uses many workers (avoid oversubscription)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

python -u run_fedtdo.py \
  --data ETTh1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id fedtdo_etth1 \
  --num_clients 4 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --d_model 64 \
  --patch_len 16 \
  --stride 8 \
  --cycle_length 24 \
  --num_temporal_moe_layers 1 \
  --num_drift_experts 2 \
  --batch_size 16 \
  --train_epochs 5 \
  --fedtdo_local_epochs 5 \
  --fedtdo_post_freeze_epochs 1 \
  --fedtdo_online_steps 100 \
  --fedtdo_eval_budget 2 \
  --fedtdo_mix_alpha 0.5 \
  --drift_window_size 64 \
  --learning_rate 0.0005 \
  --num_workers 32 \
  --num_threads 1 \
  --des aligned \
  --use_gpu True
