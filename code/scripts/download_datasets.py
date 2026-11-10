"""
Download all long-term forecasting datasets required by FedTDO / Dynamic TMoE.

Uses Hugging Face dataset `thuml/Time-Series-Library`.
If huggingface.co is unreachable (e.g. in CN), set:
  set HF_ENDPOINT=https://hf-mirror.com
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "thuml/Time-Series-Library"
NEEDED = [
    "ETT-small/ETTh1.csv",
    "ETT-small/ETTh2.csv",
    "ETT-small/ETTm1.csv",
    "ETT-small/ETTm2.csv",
    "electricity/electricity.csv",
    "traffic/traffic.csv",
    "weather/weather.csv",
    "exchange_rate/exchange_rate.csv",
    "illness/national_illness.csv",
]


def main():
    root = Path(__file__).resolve().parents[1] / "dataset"
    root.mkdir(parents=True, exist_ok=True)
    print("HF_ENDPOINT =", os.environ.get("HF_ENDPOINT", "(default huggingface.co)"))
    print("Saving to", root)

    for rel in NEEDED:
        print("Downloading", rel, "...", flush=True)
        local = hf_hub_download(
            repo_id=REPO,
            filename=rel,
            repo_type="dataset",
            local_dir=str(root),
        )
        size_mb = os.path.getsize(local) / 1e6
        print(f"  -> {local} ({size_mb:.2f} MB)", flush=True)

    print("Done. All required forecasting datasets are under ./dataset/")


if __name__ == "__main__":
    main()
