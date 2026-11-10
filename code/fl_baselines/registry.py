"""Baseline method registry."""

from __future__ import annotations

from typing import Dict, Type

from .base import FedMethod
from .ditto import Ditto
from .fedadam import FedAdam
from .fedala import FedALA
from .fedavg import FedAvg, FedAvgSGD
from .fedkme import FedKME
from .fedkwaz import FedKWAZ
from .fedl2p import FedL2P
from .fednova import FedNova
from .fedper import FedPer
from .fedpkda import FedPKDA
from .fedprox import FedProx
from .fedrep import FedRep
from .pfedme import pFedMe
from .scaffold import SCAFFOLD
from .fedpoe import FedPOE
from .local_only import LocalOnly
from .fedda import FedDA
from .fedgtp import FedGTP
from .pfedcs import PFedCS
from .pfedfda import PFedFDA

BASELINE_REGISTRY: Dict[str, Type[FedMethod]] = {
    'local': LocalOnly,
    'fedavg': FedAvg,
    'fedavg_sgd': FedAvgSGD,
    'fedala': FedALA,
    'fedl2p': FedL2P,
    'pfedfda': PFedFDA,
    'fedpoe': FedPOE,
    'pfedcs': PFedCS,
    'fedkwaz': FedKWAZ,
    'fedpkda': FedPKDA,
    'fedkme': FedKME,
    'fedda': FedDA,
    'fedgtp': FedGTP,
    # FedOpt-style optimisation / personalisation baselines
    'fedprox': FedProx,
    'scaffold': SCAFFOLD,
    'fedadam': FedAdam,
    'fednova': FedNova,
    'fedper': FedPer,
    'fedrep': FedRep,
    'ditto': Ditto,
    'pfedme': pFedMe,
}


def get_baseline(name: str, args, device) -> FedMethod:
    key = name.lower().strip()
    if key not in BASELINE_REGISTRY:
        raise KeyError(f'Unknown baseline {name!r}. Choose from: {list_baselines()}')
    return BASELINE_REGISTRY[key](args, device)


def list_baselines():
    rows = []
    for k, cls in BASELINE_REGISTRY.items():
        rows.append({
            'name': k,
            'venue': getattr(cls, 'venue', ''),
            'desc': getattr(cls, 'short_desc', ''),
        })
    return rows
