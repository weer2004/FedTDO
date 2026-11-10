"""Which interpreter am I running under, and is that the right one?

This repository has **two** interpreters and the choice is not a matter of taste:

===========================  ==================================================
``D:/software/Anaconda3/``   torch 2.2.2+cpu.  Produced every number in
``python.exe``               ``results/`` and ``results_baselines/`` (the
                             ``channel``-protocol tables, FedTDO's three arms,
                             the backbone axis).  No ``transformers``.
``D:/software/Anaconda3/``   torch cu130, RTX 4060 8 GB, ``transformers`` +
``envs/ma_fmarl/python.exe`` the local GPT-2 checkpoint.  Required for
                             anything that touches a language model (FFTS,
                             Time-FFM) and for the dataset-level multivariate
                             protocol, which is GPU-bound.
===========================  ==================================================

The routing rule is hard-coded in :func:`scripts.xc_sweep.python_for` and in
:func:`needed_env` below -- nobody should have to remember it, and nobody should
have to be reminded of it.  Every entry point prints the banner as its first
line, so a wrong interpreter is visible in line 1 of the log instead of surfacing
as nine silent FAILs at the end.
"""

from __future__ import annotations

import os
import sys

CPU_PYTHON = 'D:/software/Anaconda3/python.exe'
GPU_PYTHON = 'D:/software/Anaconda3/envs/ma_fmarl/python.exe'

#: features that cannot run on the CPU interpreter
GPU_ONLY = ('ffts', 'timeffm', 'fedal', 'lm', 'gpt2')


def needed_env(feature: str = '') -> str:
    """``'cpu'`` or ``'gpu'`` for a feature name (backbone / method / label)."""
    override = os.environ.get('FEDTDO_ENV_OVERRIDE')
    if override in ('cpu', 'gpu'):
        return override
    return 'gpu' if str(feature or '').lower() in GPU_ONLY else 'cpu'


def print_env_banner(tag: str = '', feature: str = '') -> dict:
    """Print interpreter / torch / device, and refuse a GPU feature on CPU.

    Returns a small dict so the caller can reuse e.g. ``cuda_ok``.
    """
    import torch

    exe = sys.executable.replace('\\', '/')
    is_gpu_env = exe.lower().endswith('envs/ma_fmarl/python.exe')
    cuda_ok = bool(torch.cuda.is_available())
    device = torch.cuda.get_device_name(0) if cuda_ok else 'cpu'
    need = needed_env(feature)

    print(f'[env] tag={tag or "-"} feature={feature or "-"}')
    print(f'[env] interpreter={exe}  ({"ma_fmarl/GPU env" if is_gpu_env else "CPU env"})')
    print(f'[env] torch={torch.__version__} cuda={cuda_ok} device={device}')
    print(f'[env] rule: LM / multivariate-GPU work MUST use {GPU_PYTHON}; '
          f'the channel-protocol grid MUST use {CPU_PYTHON}')

    if need == 'gpu' and not cuda_ok:
        raise SystemExit(
            f'\n[fatal] feature {feature!r} needs the GPU environment, but this '
            f'process is {exe} (torch {torch.__version__}, cuda={cuda_ok}).\n'
            f'  run it with:  {GPU_PYTHON}\n'
            f'  or let the sweep pick it:  python scripts/xc_sweep.py ... '
            f'(python_for() routes ffts/timeffm to ma_fmarl automatically)\n'
            f'  to silence this check set FEDTDO_ENV_OVERRIDE=cpu (not recommended)'
        )
    return {'cuda_ok': cuda_ok, 'gpu_env': is_gpu_env, 'device': device}
