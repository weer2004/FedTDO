"""Measure the *wire* cost of every model in both protocols.

Why a separate probe
--------------------
The communication cost of a federated method is not ``param count * 4``:

* ``state_dict()`` also carries buffers, and ``Dynamic_TMoE`` registers a
  **160,000-element sinusoidal position table** plus an expert bank that is
  reachable through two paths (``layer.expert_pool`` and
  ``layer.gating_network.expert_pool`` are the *same* object), so the same
  tensors appear twice in ``state_dict()``.
* ``FedAvg``-style aggregation ships the whole ``state_dict()``, so the wire
  cost is the state size, not the parameter count.

This script builds each model at the exact setting used in the paper matrix and
records the numbers.  Fully frozen LM backbones (FFTS / Time-FFM) need the GPU
environment; run this with ``ENVIRONMENT.md``'s ``GPU_PYTHON``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

OUT = ROOT / 'results_paper' / 'cost_probe.json'


def _measure(model: torch.nn.Module) -> dict:
    params = [n for n, _ in model.named_parameters()]
    p_elems = sum(p.numel() for p in model.parameters())
    sd = model.state_dict()
    state_elems = sum(v.numel() for v in sd.values())

    # a tensor reachable through two module paths is serialised twice
    seen: dict = {}
    dup = 0
    for v in sd.values():
        key = (v.untyped_storage().data_ptr(), v.storage_offset(), v.shape)
        if key in seen:
            dup += v.numel()
        else:
            seen[key] = True

    # buffers that never change across clients (positional encodings) are pure
    # overhead on the wire
    const = sum(v.numel() for k, v in sd.items()
                if 'position_embedding' in k or k.endswith('.pe'))

    head = sum(p.numel() for n, p in model.named_parameters()
               if n.startswith('projection_layer'))
    return {
        'params': int(p_elems),
        'state_elems': int(state_elems),
        'duplicate_elems': int(dup),
        'constant_elems': int(const),
        'irreducible_elems': int(state_elems - dup - const),
        'head_elems': int(head),
        'body_elems': int(p_elems - head),
        'wire_bytes': int(state_elems * 4),
        'body_wire_bytes': int((state_elems - head) * 4),
    }


def channel_models() -> dict:
    """``channel`` protocol: one sensor per client, sl96 -> pl24, d_model 32."""
    from argparse import Namespace
    from models.zoo import build_model

    a = Namespace(
        seq_len=96, pred_len=24, label_len=24, enc_in=1, dec_in=1, c_out=1,
        d_model=32, n_heads=4, e_layers=2, d_layers=1, d_ff=128, dropout=0.1,
        factor=1, expand=2, d_conv=4, top_k=2, distil=True, activation='gelu',
        channel_independence=1, patch_len=16, stride=16, num_experts=4,
        cycle_length=24, features='S', task_name='long_term_forecast',
        model='Dynamic_TMoE', use_norm=1, embed='timeF', freq='h',
        baseline_enable_drift=0, learning_rate=5e-4, train_epochs=10,
    )
    out = {'dtmoe': _measure(build_model('dtmoe', a))}
    return out


def multivariate_models(backbones: list[str]) -> dict:
    """Dataset-level protocol: 7 channels, sl96 -> pl96, d_model 32."""
    from run_paper_protocol import build_parser
    from models.zoo import build_model

    args = build_parser().parse_args([])
    args.enc_in = args.paper_channels
    args.dec_in = args.paper_channels
    args.c_out = args.paper_channels
    # ``Dynamic_TMoE`` reads a few attributes the paper parser does not define
    for key, value in dict(task_name='long_term_forecast', cycle_length=24,
                           baseline_enable_drift=0).items():
        if not hasattr(args, key):
            setattr(args, key, value)
    out = {}
    for b in backbones:
        try:
            m = build_model(b, args)
        except Exception as exc:                      # noqa: BLE001
            out[b] = {'error': f'{type(exc).__name__}: {exc}'}
            print(f'  [skip] {b}: {type(exc).__name__}: {exc}')
            continue
        out[b] = _measure(m)
        print(f'  {b:<13} params {out[b]["params"]:>10,}  wire {out[b]["wire_bytes"]/2**20:8.2f} MiB')
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


def main() -> None:
    print(f'[cost] interpreter: {sys.executable}')
    print(f'[cost] torch {torch.__version__}  cuda {torch.cuda.is_available()}')
    payload = {
        'channel': channel_models(),
        'multivariate': multivariate_models([
            'patchtst', 'itransformer', 'transformer', 'gru', 'autoformer',
            'fedformer', 'timesnet', 'timemixer', 'dtmoe',
            'stmtm', 'raft', 'tfps', 'ffts', 'timeffm',
        ]),
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(f'[cost] wrote {OUT}')


if __name__ == '__main__':
    main()
