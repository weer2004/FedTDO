#!/usr/bin/env bash
# Multi-seed replication of the 32-client Traffic result (5 seeds x 3 arms).
#
# Seed 2021 was the first run and gave transfer +1.82% (paired).  That number
# is not a result yet: 78% of it came from a single client, so the only way to
# know whether it is a property of the mechanism or of one lucky draw is to
# redraw.  `--seed` moves *both* the channel subset and the initialisation, so
# each seed is an independent federation, not a re-run of the same one.
#
# Three arms per seed, all at 32 clients and 1600 steps per client:
#   joint_b9      method arm, eval_budget matched to the pool (see traffic32.sh)
#   notransfer    same training, transfer disabled -> paired control
#   local         pure local training via run_baselines.py
#
# Usage:
#   bash scripts/traffic32_seeds.sh                 # seeds 2022..2025
#   SEEDS="2022 2023" MAXJOBS=2 bash scripts/traffic32_seeds.sh
set -uo pipefail
cd "$(dirname "$0")/.."

PY="D:/software/Anaconda3/python.exe"
SEEDS="${SEEDS:-2022 2023 2024 2025}"
MAXJOBS="${MAXJOBS:-4}"
LOGDIR="results/fedtdo/_seed_logs"
mkdir -p "$LOGDIR"

COMMON=(
  --data Traffic --root_path ./dataset/traffic/ --data_path traffic.csv --freq h --enc_in 1
  --seq_len 96 --pred_len 24 --label_len 24 --patch_len 16 --stride 16
  --d_model 32 --learning_rate 5e-4 --batch_size 32
  --fedtdo_num_clients 32 --fedtdo_online_steps 60
)

launch() {
  local tag="$1"; shift
  echo "[launch] $tag"
  OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 "$@" >"$LOGDIR/$tag.log" 2>&1
  echo "[done]   $tag (exit $?)"
}

throttle() {
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do
    wait -n
  done
}

for seed in $SEEDS; do
  throttle
  ( launch "s${seed}_joint_b9" "$PY" run_fedtdo.py --model_id "traffic32_s${seed}_joint_b9" \
      "${COMMON[@]}" --seed "$seed" --fedtdo_eval_budget 9 \
      --fedtdo_control fedtdo_joint --fedtdo_warmup_steps 400 --fedtdo_local_steps 1200 ) &

  throttle
  ( launch "s${seed}_notransfer" "$PY" run_fedtdo.py --model_id "traffic32_s${seed}_notransfer" \
      "${COMMON[@]}" --seed "$seed" --fedtdo_eval_budget 9 \
      --fedtdo_control fedtdo_joint --fedtdo_warmup_steps 400 --fedtdo_local_steps 1200 \
      --fedtdo_disable_transfer 1 ) &

  throttle
  ( launch "s${seed}_local" "$PY" run_baselines.py --baseline local \
      --model_id "traffic32_s${seed}_local" \
      --data Traffic --root_path ./dataset/traffic/ --data_path traffic.csv --freq h --enc_in 1 \
      --seq_len 96 --pred_len 24 --label_len 24 --patch_len 16 --stride 16 \
      --d_model 32 --batch_size 32 --num_clients 32 --num_workers 0 --num_threads 5 \
      --fl_rounds 1 --fl_local_max_steps 1600 --fl_online_steps 60 --seed "$seed" ) &
done

wait
echo "[all done]"
