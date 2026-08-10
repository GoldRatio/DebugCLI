"""Classify initial symptoms into likely subsystem(s).

Heuristic keyword matching over symptom text and error counters. Produces an ordered
list of candidate subsystems with a ranking; the diagnostic engine uses this to pick
the first (safest) test. It reports a differential, not a single guess.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..inspect.model import PROFILE_COLLECTORS

_SYMPTOM_TO_SUBSYSTEM = {
    "memory": ["mce", "machine check", "edac", "ecc", "uncorrectable", "crash", "single-bit"],
    "cpu": ["therm", "thermal", "mce", "core", "cache", "fpu", "cpu"],
    "pcie": ["aer", "pci express", "link up", "linkdown", "uncorrectable error", "surprise removal"],
    "bmc": ["ipmi", "sensor", "fru", "watchdog", "bmc", "sel"],
    "storage": ["smartfail", "smart failed", "nvme", "raid", "media error", "disk", "scsi", "i/o error"],
}


@dataclass(frozen=True)
class SubsystemRanking:
    subsystem: str
    score: int


def classify(symptom_text: str) -> list[SubsystemRanking]:
    lowered = symptom_text.lower()
    scores: dict[str, int] = {}
    for subsystem, keywords in _SYMPTOM_TO_SUBSYSTEM.items():
        score = sum(1 for kw in keywords if kw in lowered)
        if score:
            scores[subsystem] = score
    if not scores:
        return [SubsystemRanking("generic", 0)]
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [SubsystemRanking(s, c) for s, c in ranked]


def addl_collectors_for(subsystem: str) -> list[str]:
    return PROFILE_COLLECTORS.get(subsystem, PROFILE_COLLECTORS["generic"])