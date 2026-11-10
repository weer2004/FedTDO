# Sweep all FL baselines (Windows PowerShell)
# Usage: .\scripts\run_all_baselines.ps1

$Root = if ($args[0]) { $args[0] } else { "./dataset/ETT-small/" }
$Csv  = if ($args[1]) { $args[1] } else { "ETTh1.csv" }
$Data = if ($args[2]) { $args[2] } else { "ETTh1" }

$Methods = @(
  "local","fedavg","fedala","fedl2p","pfedfda",
  "fedpoe","pfedcs","fedkwaz","fedpkda","fedkme"
)

$Common = @(
  "--data", $Data,
  "--root_path", $Root,
  "--data_path", $Csv,
  "--model_id", "fl_sweep",
  "--num_clients", "4",
  "--seq_len", "96",
  "--label_len", "48",
  "--pred_len", "96",
  "--d_model", "64",
  "--patch_len", "16",
  "--stride", "8",
  "--cycle_length", "24",
  "--num_temporal_moe_layers", "1",
  "--batch_size", "16",
  "--learning_rate", "0.0005",
  "--fl_rounds", "5",
  "--fl_local_epochs", "1",
  "--fl_online_steps", "100",
  "--des", "sweep"
)

foreach ($m in $Methods) {
  Write-Host "========== Running baseline: $m =========="
  python -u run_baselines.py --baseline $m @Common
}

Write-Host "All baselines finished. See ./results_baselines/"
