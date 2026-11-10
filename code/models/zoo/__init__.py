"""Backbone zoo for the FL baseline matrix.

Purpose: separate the two axes a reviewer will want to see.

* **FL axis** -- who talks to whom and what is averaged (``fl_baselines/``).
* **Backbone axis** -- what the per-client forecaster is (this package).

Everything here implements the same three-call contract that ``BaselineClient``
expects from ``models.Dynamic_TMoE.model.Model``:

1. ``forward(x, x_mark) -> [B, pred_len, C]``
2. ``_encode_patches(x, x_mark) -> 6-tuple`` whose first element is the token
   matrix ``[B, P, D]`` (used only by the prototype / feature-adaptation
   baselines; the zoo returns a real token matrix so those still work)
3. ``get_encoder_state_dict() -> dict`` (everything except the forecast head,
   used by FedL2P-style layer-wise personalisation)

Design notes that matter for comparability:

* every model applies **RevIN** (instance normalisation with de-normalisation at
  the output), which is what Dynamic_TMoE does internally -- without it the
  channel-as-client split sees eight different scales and the comparison is
  dominated by normalisation rather than by architecture;
* parameter counts are deliberately *not* matched.  At ``d_model=32`` the zoo is
  smaller than Dynamic_TMoE (81,373 params), so if a plain PatchTST wins it wins
  with fewer parameters, and the paper must say so;
* with ``enc_in = 1`` (the channel-as-client split of the main table) the variate
  axis collapses: ``iTransformer`` has exactly one token to attend over, its
  attention is the identity and it degenerates to a per-variate MLP; TimesNet's
  2-D period tensor loses a dimension; TimeMixer has no channels to mix.  Those
  four are only meaningful under the **dataset-level multivariate protocol**
  (``data_provider/multivariate_data.py``, ``enc_in = 7``), which is what
  ``PAPER_PROTOCOL_BACKBONES.md`` reports.

* the TSLib lineage (Autoformer / FEDformer / TimesNet / TimeMixer) is ported from
  the reference implementation in ``tslib_blocks.py``; the encoder-decoder pair
  goes through ``Seq2SeqForecaster`` so ``dec_in`` never contains future values.
"""

from __future__ import annotations

from .base import Seq2SeqForecaster, ZooForecaster
from .autoformer import Autoformer
from .fedformer import FEDformer
from .gru import GRUForecaster
from .itransformer import ITransformer
from .patchtst import PatchTST
from .timemixer import TimeMixer
from .timesnet import TimesNet
from .transformer import VanillaTransformer
# 2025 supervised forecasting methods the paper has to beat
from .raft import RAFT
from .stmtm import STMTM
from .tfps import TFPS
# LM-backed federated foundation-model backbones (frozen GPT-2 + adapters)
from .lm_backbones import FFTS, TimeFFM

ZOO = {
    'patchtst': PatchTST,
    'gru': GRUForecaster,
    'itransformer': ITransformer,
    'transformer': VanillaTransformer,
    # TSLib lineage, needed for the FFTS / FeDaL-style multivariate protocol
    'autoformer': Autoformer,
    'fedformer': FEDformer,
    'timesnet': TimesNet,
    'timemixer': TimeMixer,
    # 2025 SOTA rows of the centralised table
    'tfps': TFPS,        # NeurIPS 2025, pattern-specific experts
    'stmtm': STMTM,      # KDD 2025, seasonal-trend masked modelling
    'raft': RAFT,        # ICML 2025, retrieval augmented
    'ffts': FFTS,        # AAAI 2025, frozen GPT-2 + per-block adapters
    'timeffm': TimeFFM,  # NeurIPS 2024, frozen GPT-2 + multi-scale
}


def build_model(name: str, args):
    """Return the per-client forecaster for ``--backbone``.

    ``dtmoe`` (default) keeps the original Dynamic_TMoE so every number already
    in ``results_baselines/`` stays valid.
    """
    key = str(name or 'dtmoe').lower()
    if key in ('dtmoe', 'dynamic_tmoe', 'none', ''):
        from models.Dynamic_TMoE.model import Model
        return Model(args)
    if key not in ZOO:
        raise KeyError(f'unknown backbone {name!r}; choose from dtmoe, {sorted(ZOO)}')
    return ZOO[key](args)
