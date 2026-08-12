"""Classify initial symptoms into likely subsystem(s).

Heuristic keyword matching over symptom text and error counters. Produces an ordered
list of candidate subsystems with a ranking; the diagnostic engine uses this to pick
the first (safest) test. It reports a differential, not a single guess.

Outcome-fed priors (Prompt 05) layer fleet history on top: each keyword hit's
static contribution is multiplied by a per-(keyword, subsystem) multiplier
learned from VERIFIED outcomes (``plan/priors_build.py``). Without priors the
behavior is byte-for-byte identical to the static table.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..inspect.model import PROFILE_COLLECTORS

_SYMPTOM_TO_SUBSYSTEM = {
    "memory": ["mce", "machine check", "edac", "ecc", "uncorrectable", "crash", "single-bit"],
    "cpu": ["therm", "thermal", "mce", "core", "cache", "fpu", "cpu"],
    "pcie": ["aer", "pci express", "link up", "linkdown", "uncorrectable error", "surprise removal"],
    "bmc": ["ipmi", "sensor", "fru", "watchdog", "bmc", "sel"],
    "storage": ["smartfail", "smart failed", "nvme", "raid", "media error", "disk", "scsi", "i/o error"],
}

# Multipliers outside this band are clamped so a small fleet never drowns the
# static heuristic table (applied at build time AND at classify time).
MULTIPLIER_MIN = 0.5
MULTIPLIER_MAX = 2.0


def clamp_multiplier(value: float) -> float:
    return min(MULTIPLIER_MAX, max(MULTIPLIER_MIN, value))


@dataclass(frozen=True)
class PriorModel:
    """Learned symptom-keyword -> subsystem likelihood multipliers.

    ``defaults`` is the static keyword table (unchanged); ``keyword_multipliers``
    maps each symptom keyword to per-subsystem multipliers (absent keyword or
    subsystem -> 1.0, i.e. the static behavior).
    """

    defaults: dict[str, list[str]] = field(
        default_factory=lambda: dict(_SYMPTOM_TO_SUBSYSTEM))
    keyword_multipliers: dict[str, dict[str, float]] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "keyword_multipliers",
                           self.keyword_multipliers or {})

    @property
    def empty(self) -> bool:
        return not self.keyword_multipliers

    def covers(self, keyword: str) -> bool:
        """True when the prior file has any signal for ``keyword``."""
        return keyword in self.keyword_multipliers

    def multiplier(self, keyword: str, subsystem: str) -> float:
        mult = self.keyword_multipliers.get(keyword, {}).get(subsystem, 1.0)
        return clamp_multiplier(mult)


@dataclass(frozen=True)
class SubsystemRanking:
    subsystem: str
    score: float


def classify(symptom_text: str,
             priors: PriorModel | None = None) -> list[SubsystemRanking]:
    """Keyword rank of likely subsystems; priors multiply each keyword hit.

    With ``priors`` None or empty, every keyword contributes exactly 1.0 and
    the result matches the pre-priors behavior (``score: int`` semantics are
    unchanged for whole-number scores).
    """
    lowered = symptom_text.lower()
    scores: dict[str, float] = {}
    for subsystem, keywords in _SYMPTOM_TO_SUBSYSTEM.items():
        score = 0.0
        for kw in keywords:
            if kw in lowered:
                if priors is not None and priors.covers(kw):
                    score += priors.multiplier(kw, subsystem)
                else:
                    score += 1.0
        if score:
            scores[subsystem] = score
    if not scores:
        return [SubsystemRanking("generic", 0)]
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [SubsystemRanking(s, c) for s, c in ranked]


def addl_collectors_for(subsystem: str) -> list[str]:
    return PROFILE_COLLECTORS.get(subsystem, PROFILE_COLLECTORS["generic"])