#!/bin/bash
# Sweep all FL baselines under a shared FedTDO-compatible protocol.
# Usage: bash ./scripts/run_all_baselines.sh [DATA_ROOT] [DATA_CSV]

ROOT=${1:-./dataset/ETT-small/}
CSV=${2:-ETTh1.csv}
DATA=${3:-ETTh1}

# Keep intra-op threads at 1 when DataLoader uses many workers (avoid oversubscription)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

METHODS=(local fedavg fedala fedl2p pfedfda fedpoe pfedcs fedkwaz fedpkda fedkme)

COMMON=(
  --data "$DATA"
  --root_path "$ROOT"
  --data_path "$CSV"
  --model_id fl_sweep
  --num_clients 4
  --seq_len 96
  --label_len 48
  --pred_len 96
  --d_model 64
  --patch_len 16
  --stride 8
  --cycle_length 24
  --num_temporal_moe_layers 1
  --batch_size 16
  --learning_rate 0.0005
  --fl_rounds 5
  --fl_local_epochs 1
  --fl_online_steps 100
  --num_workers 32
  --num_threads 1
  --des sweep
)

for m in "${METHODS[@]}"; do
  echo "========== Running baseline: $m =========="
  python -u run_baselines.py --baseline "$m" "${COMMON[@]}"
done

echo "All baselines finished. See ./results_baselines/"
