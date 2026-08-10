"""Anomalous-register summarizer.

The full register dump is large and must NOT all go into the LLM context (context
budget). This reduces it to the anomalous/interesting values only, keeping the
full dump on disk for audit. Mirrors the spec rule: summarize anomalous registers,
send full dump to storage, pass summary to LLM.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..inspect.base import RegisterDump


@dataclass
class EvidenceSummary:
    interesting: list[str]      # human-readable lines worth sending to the LLM
    anomaly_count: int
    total: int


def summarize(dumps: list[RegisterDump], max_items: int = 50) -> EvidenceSummary:
    """Keep only registers that look anomalous (non-zero/likely error) up to max_items."""
    interesting: list[str] = []
    for dump in dumps:
        for line in dump.raw.splitlines():
            if _looks_anomalous(line):
                interesting.append(f"[{dump.subsystem}] {line.strip()}")
            if len(interesting) >= max_items:
                break
        if len(interesting) >= max_items:
            break
    return EvidenceSummary(
        interesting=interesting,
        anomaly_count=len(interesting),
        total=sum(len(d.raw.splitlines()) for d in dumps),
    )


_ANOMALY_HINTS = ("error", "fail", "uncorrectable", "corrected", "warning",
                  "critical", "overflow", "sensor", "non-zero", "threshold",
                  "ecc", "mce", "0x", "bad", "fault", "degraded")


def _looks_anomalous(line: str) -> bool:
    lowered = line.lower()
    return any(h in lowered for h in _ANOMALY_HINTS)