#!/usr/bin/env python3
"""Validate timestamp causality in one recorded delayed-feedback run.

Usage:
    python scripts/validate_causal_protocol.py <run_directory>

The validator reads only the explicitly supplied run's recorder CSV files.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _integer(row: dict, field: str, file_name: str, row_number: int) -> int:
    value = row.get(field, "")
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{file_name}:{row_number}: missing/non-integer {field}={value!r}"
        )
    if parsed < 0:
        raise ValueError(f"{file_name}:{row_number}: invalid {field}={parsed}")
    return parsed


def validate_run(run_dir: Path) -> dict:
    windows_path = run_dir / "per_window_metrics.csv"
    updates_path = run_dir / "online_updates.csv"
    if not windows_path.is_file():
        raise FileNotFoundError(windows_path)

    window_count = 0
    with windows_path.open(newline="", encoding="utf-8") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            start = _integer(row, "target_start_idx", windows_path.name, row_number)
            end = _integer(row, "target_end_idx", windows_path.name, row_number)
            available = _integer(row, "label_reveal_idx", windows_path.name, row_number)
            if start > end:
                raise ValueError(
                    f"{windows_path.name}:{row_number}: target start {start} > end {end}"
                )
            if end > available:
                raise ValueError(
                    f"{windows_path.name}:{row_number}: target end {end} "
                    f"exceeds label availability {available}"
                )
            window_count += 1

    update_count = 0
    if updates_path.is_file():
        with updates_path.open(newline="", encoding="utf-8") as handle:
            for row_number, row in enumerate(csv.DictReader(handle), start=2):
                start = _integer(row, "target_start_idx", updates_path.name, row_number)
                end = _integer(row, "target_end_idx", updates_path.name, row_number)
                available = _integer(row, "label_available_idx", updates_path.name, row_number)
                update_time = _integer(row, "update_time_idx", updates_path.name, row_number)
                if start > end:
                    raise ValueError(
                        f"{updates_path.name}:{row_number}: target start {start} > end {end}"
                    )
                if end > available:
                    raise ValueError(
                        f"{updates_path.name}:{row_number}: update target ends at {end} "
                        f"but labels are available only at {available}"
                    )
                if update_time < available:
                    raise ValueError(
                        f"{updates_path.name}:{row_number}: update time {update_time} "
                        f"precedes label availability {available}"
                    )
                if row.get("update_type") != "delayed_full_horizon":
                    raise ValueError(
                        f"{updates_path.name}:{row_number}: unexpected update_type "
                        f"{row.get('update_type')!r}"
                    )
                update_count += 1

    return {
        "run_directory": str(run_dir),
        "forecast_windows_checked": window_count,
        "delayed_updates_checked": update_count,
        "causal": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    result = validate_run(args.run_directory)
    print(
        "PASS: causal delayed-feedback protocol "
        f"({result['forecast_windows_checked']} windows, "
        f"{result['delayed_updates_checked']} updates)"
    )


if __name__ == "__main__":
    main()
