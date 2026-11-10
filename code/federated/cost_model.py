"""Explicit accounting for evaluation, transfer, and switching resources."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter


@dataclass
class CostModel:
    eval_flop_scale: float = 1_000_000.0
    transfer_byte_scale: float = 1_000_000.0
    switch_cost: float = 0.0
    total_eval_flops: int = 0
    total_download_bytes: int = 0
    transfer_count: int = 0
    switch_count: int = 0
    event_counts: Counter = field(default_factory=Counter)

    def evaluation_cost(self, capsule, cached: bool = False) -> float:
        """Shadow-evaluation cost; the first evaluation also pays the download.

        Plan §5.1 charges a first-time probe with both the download and the
        forward pass, otherwise a freshly fetched expert looks free and the
        budget comparison against ``keep local`` is biased.
        """
        cost = capsule.estimated_flops / max(self.eval_flop_scale, 1.0)
        if not cached:
            cost += capsule.estimated_bytes / max(self.transfer_byte_scale, 1.0)
        return cost

    def transfer_cost(self, capsule, cached: bool) -> float:
        return 0.0 if cached else capsule.estimated_bytes / max(self.transfer_byte_scale, 1.0)

    def record_evaluation(self, capsule):
        self.total_eval_flops += capsule.estimated_flops
        self.event_counts['evaluations'] += 1

    def record_download(self, capsule):
        self.total_download_bytes += capsule.estimated_bytes
        self.event_counts['downloads'] += 1

    def record_transfer(self, switched: bool):
        self.transfer_count += 1
        self.event_counts['transfers'] += 1
        if switched:
            self.switch_count += 1
            self.event_counts['switches'] += 1

    def summary(self):
        return {
            'evaluation_flops': self.total_eval_flops,
            'download_bytes': self.total_download_bytes,
            'transfers': self.transfer_count,
            'switches': self.switch_count,
            **dict(self.event_counts),
        }
