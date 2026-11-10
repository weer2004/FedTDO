"""Convert laiguokun Solar-Energy (solar_AL.txt) into the TSLib CSV layout.

Source : https://github.com/laiguokun/multivariate-time-series-data
         solar-energy/solar_AL.txt.gz  ->  52560 rows x 137 columns, no header,
         no timestamp column, 10-minute resolution, Alabama 2006.

Target : dataset/solar-energy/solar_energy.csv
         first column `date` (YYYY-MM-DD HH:MM:SS), remaining 137 columns `T1..T137`.

The DataLoader in data_provider/ only needs (date + numeric columns), so this
script is the whole integration surface for the new dataset.

Usage:
    python scripts/prepare_solar.py                # convert + report
    python scripts/prepare_solar.py --keep-txt     # keep the 172 MB intermediate
"""

from __future__ import annotations

import argparse
import gzip
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

N_STEPS = 52560          # 365 days * 144 slots
N_COLS = 137
FREQ = '10min'
START = '2006-01-01 00:00:00'


def load_raw(path: Path) -> pd.DataFrame:
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as handle:
        frame = pd.read_csv(handle, header=None)
    if frame.shape != (N_STEPS, N_COLS):
        raise ValueError(
            f'unexpected shape {frame.shape}, expected ({N_STEPS}, {N_COLS})'
        )
    return frame


def to_tslib(frame: pd.DataFrame) -> pd.DataFrame:
    index = pd.date_range(start=START, periods=len(frame), freq=FREQ)
    out = frame.copy()
    out.columns = [f'T{i + 1}' for i in range(out.shape[1])]
    out.insert(0, 'date', index.strftime('%Y-%m-%d %H:%M:%S'))
    return out


def report(frame: pd.DataFrame) -> None:
    values = frame.to_numpy(dtype=np.float64)
    n_nan = int(np.isnan(values).sum())
    zero_rows = int((values.sum(axis=1) == 0.0).sum())
    zero_share = float((values == 0.0).mean())
    print(f'  shape            : {frame.shape}')
    print(f'  value range      : min={np.nanmin(values):.3f} max={np.nanmax(values):.3f}')
    print(f'  NaN cells        : {n_nan}')
    print(f'  all-zero rows    : {zero_rows} ({zero_rows / len(values):.1%})  <- night hours')
    print(f'  zero cell share  : {zero_share:.1%}')
    live = (values > 0).sum(axis=0)
    print(f'  active steps/col : min={live.min()} median={int(np.median(live))} max={live.max()}')
    dead = int((live == 0).sum())
    if dead:
        print(f'  WARNING          : {dead} column(s) are all zero and should be dropped')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--src', default=None, help='solar_AL.txt[.gz]')
    parser.add_argument('--out', default=None, help='destination csv')
    parser.add_argument('--keep-txt', action='store_true',
                        help='keep the 172 MB decompressed intermediate')
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    data_dir = root / 'dataset' / 'solar-energy'
    src = Path(args.src) if args.src else data_dir / 'solar_AL.txt'
    if not src.exists() and (data_dir / 'solar_AL.txt.gz').exists():
        src = data_dir / 'solar_AL.txt.gz'
    if not src.exists():
        raise FileNotFoundError(f'raw file not found under {data_dir}')

    print(f'[solar] reading {src}')
    raw = load_raw(src)
    report(raw)

    frame = to_tslib(raw)
    out = Path(args.out) if args.out else data_dir / 'solar_energy.csv'
    frame.to_csv(out, index=False, float_format='%.4f')
    size_mb = out.stat().st_size / 1e6
    print(f'[solar] wrote {out}  ({size_mb:.1f} MB, {len(frame)} rows x {frame.shape[1]} cols)')

    if not args.keep_txt and src.suffix == '.txt':
        src.unlink()
        print(f'[solar] removed intermediate {src.name} (regenerate from .gz if needed)')


if __name__ == '__main__':
    main()
