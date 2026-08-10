"""Post-action verification loop.

After a human approves and applies an action, rerun the relevant collectors and
compare error counters to confirm the fix (or detect no-progress). Returns a verdict
so the ticket can be closed or escalated. Never takes actions itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..inspect.base import RegisterDump
from ..inspect.decoder import Decoder


@dataclass
class VerifyResult:
    verdict: str             # "resolved" | "no_change" | "worsened" | "inconclusive"
    before: dict = field(default_factory=dict)
    after: dict = field(default_factory=dict)
    delta: dict = field(default_factory=dict)


class Verifier:
    def __init__(self, decoder: Decoder | None = None) -> None:
        self.decoder = decoder or Decoder()

    def compare(self, before: list[RegisterDump], after: list[RegisterDump],
                metric_key: str = "ecc_error") -> VerifyResult:
        """Compare an error-count metric between two dumps of the same subsystem."""
        before_counts = {d.source.lower(): self._metric(d.raw, metric_key) for d in before}
        after_counts = {d.source.lower(): self._metric(d.raw, metric_key) for d in after}
        delta = {k: after_counts.get(k, 0) - before_counts.get(k, 0) for k in before_counts}
        trend = sum(delta.values())
        if trend < 0:
            verdict = "resolved"
        elif trend == 0:
            verdict = "inconclusive"
        else:
            verdict = "worsened"
        return VerifyResult(
            verdict=verdict,
            before=before_counts,
            after=after_counts,
            delta=delta,
        )

    @staticmethod
    def _metric(raw: str, key: str) -> int:
        return sum(1 for line in raw.splitlines() if key.lower() in line.lower())