# Federated Baselines for FedTDO

All methods share the **same** protocol as FedTDO:

- Dynamic-TMoE backbone (univariate channel-as-client)
- Offline federated rounds → online prequential MSE/MAE
- No test-set peeking during training

These are **TS-adapted** implementations of the published ideas (not drop-in CV code).

| CLI key | Paper | Venue | Adaptation to Dynamic-TMoE |
|---------|-------|-------|----------------------------|
| `local` | — | control | Independent local training, no aggregation |
| `fedavg` | FedAvg | AISTATS 2017 | Full-model parameter averaging |
| `fedala` | FedALA | AAAI 2023 | Learn element-wise mix of global/local on MoE+head before local SGD; keep patch embedding from global |
| `fedl2p` | FedL2P | NeurIPS 2023 | Meta-net maps local feature stats → layer-group LRs for personalized fine-tuning |
| `pfedfda` | pFedFDA | NeurIPS 2024 | Shared body FedAvg; head personalization via local↔global feature-distribution shrinkage |
| `fedpoe` | Fed-POE (Mixture of Models) | NeurIPS 2024 | Online hedge ensemble of local model + historical federated checkpoints (**closest to FedTDO online setting**) |
| `pfedcs` | PFedCS | AAAI 2025 | Collaborate on prediction heads by parameter similarity; body shared globally |
| `fedkwaz` | FedKWAZ | NeurIPS 2025 | Proxy mutual distillation on high-discrepancy mixup windows (KWAZ under homogeneous MoE) |
| `fedpkda` | FedPKDA | AAAI 2026 | Noisy clipped feature prototypes + dynamic alignment; personalized heads |
| `fedkme` | Multi-task KME averaging | ICML 2026 | RFF kernel mean embeddings → collaboration weights → personalized weighted FedAvg |

## Run

```bash
# list methods
python run_baselines.py --list_baselines

# single method
python run_baselines.py \
  --baseline fedpoe \
  --data ETTh1 --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
  --num_clients 4 --fl_rounds 5 --fl_local_epochs 1 --fl_online_steps 100 \
  --d_model 64 --seq_len 96 --pred_len 96

# sweep (bash)
bash ./scripts/run_all_baselines.sh
```

Results: `./results_baselines/BL_<method>_.../summary.json`

## Fair comparison checklist

Use identical `--num_clients`, `--seq_len`, `--pred_len`, `--d_model`, `--fl_online_steps` (or `--fedtdo_online_steps`) when comparing against `run_fedtdo.py`.
