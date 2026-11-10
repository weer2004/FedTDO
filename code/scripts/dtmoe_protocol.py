"""Extract the Dynamic TMoE protocol from the official ``scripts/*.sh`` files.

The shell scripts pass most hyper-parameters through variables
(``--batch_size $batch_size``), so a naive ``--key value`` regex reads ``0`` and
silently produces a wrong table.  This module expands the variable assignments
first, then parses each ``python -u run.py`` block.

The result is the ground truth for "run it the way DTMoe runs it":
9 benchmarks x 4 horizons = 36 configurations, each with its own d_model,
learning rate, dropout, patch length, stride and layer counts.  Appendix A
(Table 6) of the paper confirms the same split and horizon sets, including
Illness using {24, 36, 48, 60} instead of {96, 192, 336, 720}.

Usage::

    python scripts/dtmoe_protocol.py              # print the table
    python scripts/dtmoe_protocol.py --json       # emit dtmoe_protocol.json
    python scripts/dtmoe_protocol.py --cost       # cost estimate per config
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re

SCRIPTS = {
    'ETTh1': 'ETTh1.sh', 'ETTh2': 'ETTh2.sh', 'ETTm1': 'ETTm1.sh', 'ETTm2': 'ETTm2.sh',
    'Weather': 'weather.sh', 'Exchange': 'exchange.sh', 'Traffic': 'traffic.sh',
    'ECL': 'ECL.sh', 'ILI': 'ILI.sh',
}

INT_KEYS = ('pred_len', 'enc_in', 'dec_in', 'c_out', 'd_model', 'patch_len', 'stride',
            'batch_size', 'train_epochs', 'label_len', 'cycle_length',
            'num_temporal_moe_layers', 'num_rnn_layers', 'num_drift_experts',
            'drift_window_size', 'channel_independence', 'use_relation_layer',
            'enable_drift_detection', 'top_k')
FLOAT_KEYS = ('learning_rate', 'dropout', 'drift_k_sigma')

# Reference wall-clock measured on this machine (CPU, torch 2.2.2+cpu, under load):
# ETTh1 / pred_len 96 / 1 epoch = 47 s.  Scaling below is data- and width-driven.
CPU_SECONDS_PER_EPOCH_ETTH1 = 47.0


def expand_variables(text: str) -> str:
    """Replace ``$var`` and ``${var}`` with the ``var=value`` assignments above it."""
    env = {}
    for name, value in re.findall(r'^\s*(?:export\s+)?(\w+)=([^\s#]+)\s*$', text, re.M):
        env[name] = value
    def substitute(match):
        name = match.group(1) or match.group(2)
        return env.get(name, match.group(0))
    return re.sub(r'\$\{(\w+)\}|\$(\w+)', substitute, text)


def parse_script(path: pathlib.Path):
    text = expand_variables(path.read_text(encoding='utf-8', errors='ignore'))
    blocks = [b for b in text.split('python -u run.py') if '--pred_len' in b]
    runs = []
    for block in blocks:
        block = block.replace('\\\n', ' ').replace('\\', ' ')
        pairs = dict(re.findall(r'--(\w+)\s+([\w.\-]+)', block))
        row = {}
        for key, value in pairs.items():
            if key in INT_KEYS:
                try:
                    row[key] = int(float(value))
                except ValueError:
                    continue
            elif key in FLOAT_KEYS:
                try:
                    row[key] = float(value)
                except ValueError:
                    continue
            elif key in ('data', 'model', 'features', 'des'):
                row[key] = value
        runs.append(row)
    return runs


def estimate_seconds(run: dict) -> float:
    """Rough CPU cost, anchored on the ETTh1/pred=96 measurement.

    Three multipliers, each measured or structural:
      * width   -- parameter count scales with enc_in * d_model^2;
      * horizon -- patch count and decoder length grow with pred_len;
      * epochs  -- DTMoe's own train_epochs (5 for Exchange, 60 for ETT).
    """
    enc = max(1, run.get('enc_in', 7))
    d_model = run.get('d_model', 128)
    pred = run.get('pred_len', 96)
    epochs = run.get('train_epochs', 50)
    width = (enc / 7.0) * (d_model / 128.0) ** 2
    horizon = 1.0 + 0.6 * (pred / 96.0 - 1.0)
    base = CPU_SECONDS_PER_EPOCH_ETTH1 * epochs
    return base * max(width, 0.35) * max(horizon, 0.6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--cost', action='store_true')
    args = parser.parse_args()

    root = pathlib.Path(__file__).resolve().parents[1]
    table = {}
    for dataset, filename in SCRIPTS.items():
        path = root / 'scripts' / filename
        if not path.exists():
            print(f'missing script: {filename}')
            continue
        table[dataset] = parse_script(path)

    if args.json:
        out = root / 'dtmoe_protocol.json'
        out.write_text(json.dumps(table, indent=1), encoding='utf-8')
        print(f'wrote {out}')
        return

    header = (f'{"dataset":9s}{"pred":>5s}{"enc":>5s}{"dmod":>6s}{"patch":>6s}{"strd":>6s}'
              f'{"lr":>9s}{"drop":>6s}{"moe":>4s}{"rnn":>4s}{"bs":>5s}{"ep":>4s}')
    if args.cost:
        header += f'{"est_h":>7s}'
    print(header)
    total = 0.0
    for dataset, runs in table.items():
        for run in runs:
            seconds = estimate_seconds(run)
            total += seconds
            line = (f'{dataset:9s}{run.get("pred_len", 0):5d}{run.get("enc_in", 0):5d}'
                    f'{run.get("d_model", 0):6d}{run.get("patch_len", 0):6d}'
                    f'{run.get("stride", 0):6d}{run.get("learning_rate", 0):9.4f}'
                    f'{run.get("dropout", 0):6.2f}{run.get("num_temporal_moe_layers", 0):4d}'
                    f'{run.get("num_rnn_layers", 0):4d}{run.get("batch_size", 0):5d}'
                    f'{run.get("train_epochs", 0):4d}')
            if args.cost:
                line += f'{seconds / 3600:7.2f}'
            print(line)
    print(f'\nconfigurations: {sum(len(v) for v in table.values())}')
    if args.cost:
        print(f'DTMoe centralised single-model total: {total / 3600:.1f} h on this CPU')


if __name__ == '__main__':
    main()
