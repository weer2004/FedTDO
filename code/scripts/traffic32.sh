#!/usr/bin/env bash
# 32-client run on Traffic (plan §10.1 channel partition).
#
# Same protocol as the 8-client table in TRAFFIC_REPORT.md
# (400 warm-up + 1200 local steps per client, pred_len 24, d_model 32, p24),
# only --fedtdo_num_clients changes.  Channel selection is seeded random, so
# 8 clients are *not* a subset of the 32-client draw: every arm has to be
# re-run, including the isomorphic and no-transfer controls.
#
# eval_budget is the variable that must not be forgotten.  The controller can
# only rank an arm after seeing it, and the candidate pool grows with the
# client count (nc*num_experts - own 4).  At nc=8 the historical budget of 2
# gives 4.3 observations per arm; at nc=32 it gives 0.97, i.e. no arm is ever
# seen twice.  `_b9` keeps observations per arm constant; `_b2` keeps the old
# budget so the two can be compared directly.
set -euo pipefail
cd "$(dirname "$0")/.."

COMMON=(
  --data Traffic --root_path ./dataset/traffic/ --data_path traffic.csv --freq h --enc_in 1
  --seq_len 96 --pred_len 24 --label_len 24 --patch_len 16 --stride 16
  --d_model 32 --learning_rate 5e-4 --batch_size 32
  --fedtdo_num_clients 32 --fedtdo_online_steps 60 --fedtdo_eval_budget 9
)

PY="D:/software/Anaconda3/python.exe"

# A. method arm, exploration budget matched to the larger candidate pool
"$PY" run_fedtdo.py --model_id traffic32_joint_b9 "${COMMON[@]}" \
  --fedtdo_control fedtdo_joint --fedtdo_warmup_steps 400 --fedtdo_local_steps 1200 &

# B. matched no-transfer control (same topology, transfer disabled)
"$PY" run_fedtdo.py --model_id traffic32_notransfer "${COMMON[@]}" \
  --fedtdo_control fedtdo_joint --fedtdo_warmup_steps 400 --fedtdo_local_steps 1200 \
  --fedtdo_disable_transfer 1 &

# C. topology control: no shared frozen encoder, no FedAvg
"$PY" run_fedtdo.py --model_id traffic32_iso "${COMMON[@]}" \
  --fedtdo_control isomorphic --fedtdo_warmup_steps 400 --fedtdo_local_steps 1200 &

wait

# D. same as A but with the historical budget, isolates "pool grew" from
#    "exploration starved"
"$PY" run_fedtdo.py --model_id traffic32_joint_b2 --data Traffic \
  --root_path ./dataset/traffic/ --data_path traffic.csv --freq h --enc_in 1 \
  --seq_len 96 --pred_len 24 --label_len 24 --patch_len 16 --stride 16 \
  --d_model 32 --learning_rate 5e-4 --batch_size 32 \
  --fedtdo_num_clients 32 --fedtdo_online_steps 60 --fedtdo_eval_budget 2 \
  --fedtdo_control fedtdo_joint --fedtdo_warmup_steps 400 --fedtdo_local_steps 1200
