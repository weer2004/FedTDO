"""Quantify how heterogeneous the federated clients actually are.

Two layers matter and they are NOT the same thing:

* raw scale heterogeneity -- killed by the per-client StandardScaler in
  ``Dataset_UnivariateClient``, so it never reaches the model;
* shape heterogeneity -- daily profile, weekday/weekend gap, burstiness,
  residual noise -- which survives standardisation and is what the shared
  encoder actually has to cope with.

Usage:
    python scripts/probe_heterogeneity.py --data dataset/traffic/traffic.csv \
        --num_clients 8 --seed 2021
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pick_channels(n_cols: int, num_clients: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    picked = rng.choice(n_cols, size=num_clients, replace=False)
    return sorted(int(i) for i in picked)


def profile_stats(series: np.ndarray, train_ratio: float = 0.7) -> dict:
    n = len(series)
    tr = series[: int(n * train_ratio)]
    mu, sd = float(tr.mean()), float(tr.std())
    z = (series - mu) / max(sd, 1e-9)

    hours = np.arange(len(z)) % 24
    # daily profile on the standardised series (shape only)
    prof = np.array([z[hours == h].mean() for h in range(24)])
    diurnal_amp = float(prof.max() - prof.min())
    # weekday vs weekend level gap
    dow = (np.arange(len(z)) // 24) % 7
    wk = z[dow < 5].mean() if (dow < 5).any() else 0.0
    we = z[dow >= 5].mean() if (dow >= 5).any() else 0.0
    # residual after removing the daily profile -> how much is unpredictable
    resid = z - prof[hours]
    return {
        "raw_mean": round(mu, 5),
        "raw_std": round(sd, 5),
        "raw_cv": round(sd / max(abs(mu), 1e-9), 3),
        "diurnal_amp_z": round(diurnal_amp, 3),
        "weekend_gap_z": round(float(we - wk), 3),
        "resid_std_z": round(float(resid.std()), 3),
        "peak_hour": int(np.argmax(prof)),
        "trough_hour": int(np.argmin(prof)),
        "zero_frac": round(float((tr <= 1e-6).mean()), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/traffic/traffic.csv")
    ap.add_argument("--num_clients", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2021)
    ap.add_argument("--out", default="results/heterogeneity_traffic.json")
    args = ap.parse_args()

    df = pd.read_csv(ROOT / args.data)
    cols = list(df.columns[1:])
    values = df[cols].values.astype(float)

    picked = pick_channels(len(cols), args.num_clients, args.seed)
    names = [f"c{i}" for i in picked]
    sel = values[:, picked]

    raw = pd.DataFrame([profile_stats(sel[:, k]) for k in range(sel.shape[1])],
                       index=names)

    # correlation between clients: raw vs per-client standardised
    n = len(sel)
    ntr = int(n * 0.7)
    z = (sel - sel[:ntr].mean(0)) / np.maximum(sel[:ntr].std(0), 1e-9)
    corr_raw = np.corrcoef(sel[:ntr].T)
    corr_z = np.corrcoef(z[:ntr].T)
    iu = np.triu_indices(len(names), k=1)

    summary = {
        "data": args.data,
        "n_channels_total": len(cols),
        "clients": names,
        "picked_indices": picked,
        "raw_corr_mean": round(float(corr_raw[iu].mean()), 3),
        "raw_corr_min": round(float(corr_raw[iu].min()), 3),
        "std_corr_mean": round(float(corr_z[iu].mean()), 3),
        "std_corr_min": round(float(corr_z[iu].min()), 3),
        "raw_mean_spread": [round(float(raw.raw_mean.min()), 5),
                            round(float(raw.raw_mean.max()), 5)],
        "raw_mean_ratio": round(float(raw.raw_mean.max() / max(raw.raw_mean.min(), 1e-9)), 1),
        "diurnal_amp_range": [round(float(raw.diurnal_amp_z.min()), 3),
                              round(float(raw.diurnal_amp_z.max()), 3)],
        "resid_std_range": [round(float(raw.resid_std_z.min()), 3),
                            round(float(raw.resid_std_z.max()), 3)],
    }

    print("=== per-client shape stats (standardised) ===")
    print(raw.to_string())
    print("\n=== heterogeneity summary ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary,
                               "per_client": raw.reset_index().rename(
                                   columns={"index": "client"}).to_dict("records")},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
