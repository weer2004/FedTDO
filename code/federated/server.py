"""Federated coordination that never receives raw client time series."""

from __future__ import annotations

import torch

from .expert_registry import ExpertRegistry


def module_slots(model, name: str):
    """Return the list of ``nn.Module`` objects a share name refers to.

    ``patch_embedding`` is a single module, ``temporal_moe_layers`` is a
    ``ModuleList`` and ``projection`` is either a ``ModuleList``
    (``projection_layers``) or a single ``Linear`` (``projection_layer``), so a
    share spec has to expand to a stable, indexable list on every client.
    """
    if name == 'patch_embedding':
        return [model.patch_embedding]
    if name == 'temporal_moe_layers':
        return list(getattr(model, 'temporal_moe_layers'))
    if name == 'projection':
        slots = []
        if hasattr(model, 'projection_layers'):
            slots.extend(list(model.projection_layers))
        if hasattr(model, 'projection_layer'):
            slots.append(model.projection_layer)
        return slots
    raise ValueError(f'unknown fedtdo share module {name!r}')


class FedTDOServer:
    def __init__(self, registry=None):
        self.registry = registry or ExpertRegistry()
        self.encoder_version = 0

    @torch.no_grad()
    def synchronize_modules(self, models, names=('patch_embedding',)):
        """FedAvg the listed modules across clients.

        Averaging more than the patch embedding is the alternative sharing
        topology: with one channel per client the *temporal pattern* layers can
        transfer while the amplitude/statistics-bearing embedding cannot, so the
        caller decides which side of that split to share.
        """
        if not models:
            return None
        for name in names:
            slots = module_slots(models[0], name)
            for index in range(len(slots)):
                reference = module_slots(models[0], name)[index].state_dict()
                averaged = {}
                for key, tensor in reference.items():
                    values = [module_slots(model, name)[index].state_dict()[key].detach()
                              for model in models]
                    if torch.is_floating_point(tensor):
                        averaged[key] = torch.stack(values, dim=0).mean(dim=0)
                    else:
                        averaged[key] = values[0]
                for model in models:
                    module_slots(model, name)[index].load_state_dict(averaged, strict=True)
        self.encoder_version += 1
        return f'shared-v{self.encoder_version}'

    @torch.no_grad()
    def synchronize_patch_encoder(self, models):
        """Backwards-compatible wrapper: share only the PatchEmbedding."""
        return self.synchronize_modules(models, ('patch_embedding',))

    @staticmethod
    def freeze_modules(models, names=('patch_embedding',)):
        for model in models:
            for name in names:
                for module in module_slots(model, name):
                    for parameter in module.parameters():
                        parameter.requires_grad_(False)

    @staticmethod
    def freeze_patch_encoder(models):
        FedTDOServer.freeze_modules(models, ('patch_embedding',))

    @staticmethod
    def snapshot_modules(model, names=('patch_embedding',)):
        """Detached CPU copy of the shared parameters, used by soft sharing."""
        snapshot = {}
        for name in names:
            snapshot[name] = [
                {key: value.detach().clone()
                 for key, value in module.named_parameters()}
                for module in module_slots(model, name)
            ]
        return snapshot

    @staticmethod
    def proximal_penalty(model, snapshot, names=('patch_embedding',)):
        """``0.5 * ||theta_shared - theta_local||^2`` over the shared modules.

        Soft sharing replaces the hard freeze with this term: a client may adapt
        the shared weights when its own channel needs it, but drifts away from
        the federated representation at a quadratic cost.
        """
        penalty = None
        if not snapshot:
            return None
        for name in names:
            refs = snapshot.get(name) or []
            for index, module in enumerate(module_slots(model, name)):
                if index >= len(refs):
                    continue
                reference = refs[index]
                for key, parameter in module.named_parameters():
                    target = reference.get(key)
                    if target is None:
                        continue
                    term = ((parameter - target.to(parameter.device)) ** 2).sum()
                    penalty = term if penalty is None else penalty + term
        return penalty
