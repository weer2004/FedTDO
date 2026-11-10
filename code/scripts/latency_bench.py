"""Inference latency of the *client* model that FedTDO actually deploys.

Plan §10.5 asks for "实际推理延迟" (real inference latency).  The online loop
never timestamps its evaluations, so the number has to be rebuilt offline: this
script instantiates the same ``Dynamic_TMoE`` with the same ``make_client_args``
shape the clients use (``enc_in=1``, ``features='S'``) and times a forward pass.

Two numbers matter and they are not interchangeable:

``inference``   batch=1, one window.  What a single client pays per prediction.
``evaluation``  batch=eval_budget (default 32).  What a *candidate probe* costs;
                the controller evaluates ``fedtdo_eval_budget`` candidates per
                client per round, so per-candidate cost is this divided by batch.

Latency is noisy -- 20% run-to-run spread on a loaded CPU -- so the script
reports median / mean / p95 over many repeats and the caller should run it a few
times in separate processes and report the median of medians, not one average.

Usage::

    python scripts/latency_bench.py                 # human readable
    python scripts/latency_bench.py --json          # machine readable, one object
    python scripts/latency_bench.py --repeats 300 --batch 32
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_provider.federated_data import make_client_args  # noqa: E402
from models.Dynamic_TMoE.model import Model  # noqa: E402
from run_fedtdo import parser as fedtdo_parser  # noqa: E402


def build_args(device: str, threads: int):
    """Parser defaults + the config of the headline Traffic run.

    Using ``run_fedtdo.parser()`` instead of a hand-written namespace keeps this
    script honest: the moment a default changes, the measurement changes with it.
    """
    # ``run_fedtdo`` marks model_id / data / root_path / data_path / enc_in as
    # required, so they have to be supplied even though the benchmark only needs
    # the architecture fields.  Values mirror the headline Traffic run; the
    # measurement is unaffected because no data is loaded.
    args = fedtdo_parser().parse_args([
        '--model_id', 'latency_bench', '--data', 'Traffic',
        '--root_path', './dataset/traffic/', '--data_path', 'traffic.csv',
        '--enc_in', '1',
    ])
    reference = ROOT / 'results' / 'fedtdo' / 'traffic32_s2021_joint_b9_fix' / 'metrics.json'
    if reference.exists():
        config = json.loads(reference.read_text())['config']
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
    args.device = torch.device(device)
    torch.set_num_threads(threads)
    return args


def timeit(model, batch: int, repeats: int, warmup: int, device):
    """Median / mean / p95 of ``repeats`` forward passes at a fixed batch size."""
    x = torch.randn(batch, model.seq_len, 1, device=device)
    lengths = torch.arange(model.seq_len, dtype=torch.float32, device=device)
    x_mark = torch.stack([
        lengths % 24, lengths % 7, lengths % 30, lengths % 365,
    ], dim=-1).unsqueeze(0).repeat(batch, 1, 1)
    zeros = torch.zeros(batch, model.pred_len, 1, device=device)
    zeros_mark = torch.zeros(batch, model.pred_len, 4, device=device)

    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x, x_mark, zeros, zeros_mark)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        samples = []
        for _ in range(repeats):
            started = time.perf_counter()
            model(x, x_mark, zeros, zeros_mark)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            samples.append((time.perf_counter() - started) * 1000.0)

    ordered = sorted(samples)
    return dict(
        batch=batch,
        median_ms=st.median(ordered),
        mean_ms=st.fmean(ordered),
        p95_ms=ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        per_candidate_ms=st.median(ordered) / batch,
        repeats=repeats,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--repeats', type=int, default=200)
    ap.add_argument('--warmup', type=int, default=20)
    ap.add_argument('--batch', type=int, default=32,
                    help='batch used for the per-candidate number (eval_budget)')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    ap.add_argument('--json', action='store_true')
    args = ap.parse_args()

    if args.device == 'auto':
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device_name = args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu'

    base = build_args(device_name, args.threads)
    # The deployed client is univariate regardless of the dataset's channel count.
    client_args = make_client_args(base, enc_in=1)
    client_args.task_name = 'long_term_forecast'

    model = Model(client_args).float().to(base.device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    single = timeit(model, 1, args.repeats, args.warmup, base.device)
    batched = timeit(model, args.batch, max(20, args.repeats // 4), args.warmup, base.device)

    payload = dict(
        params=params,
        device=torch.cuda.get_device_name(0) if device_name == 'cuda' else 'CPU',
        torch=torch.__version__,
        threads=args.threads,
        shape=dict(seq_len=client_args.seq_len, pred_len=client_args.pred_len,
                   d_model=client_args.d_model, patch_len=client_args.patch_len,
                   stride=client_args.stride, enc_in=1),
        inference=single,
        evaluation=batched,
    )

    if args.json:
        print(json.dumps(payload, indent=1))
        return 0

    print(f"device        : {payload['device']}  (torch {payload['torch']}, {args.threads} threads)")
    print(f"params        : {params:,}")
    print(f"shape         : seq {client_args.seq_len} -> pred {client_args.pred_len}, "
          f"d_model {client_args.d_model}, patch {client_args.patch_len}/{client_args.stride}, enc_in 1")
    print(f"batch=1       : {single['median_ms']:.2f} ms median / "
          f"{single['mean_ms']:.2f} mean / {single['p95_ms']:.2f} p95  (n={single['repeats']})")
    print(f"batch={batched['batch']:<3d}     : {batched['median_ms']:.2f} ms median -> "
          f"{batched['per_candidate_ms']:.3f} ms per candidate  (n={batched['repeats']})")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
