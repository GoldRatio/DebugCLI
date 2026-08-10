"""Profile: use the subsystem ranking to build an ordered, minimal collector plan."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..inspect.model import PROFILE_COLLECTORS
from .subsystem import SubsystemRanking


@dataclass(frozen=True)
class CollectionPlan:
    subsystem_order: list[SubsystemRanking]
    collectors: list[str] = field(default_factory=list)

    @property
    def primary_subsystem(self) -> str | None:
        return self.subsystem_order[0].subsystem if self.subsystem_order else None


def plan_collection(symptom_text: str) -> CollectionPlan:
    """Return the minimal ordered collector set for the classified subsystem(s).

    If a clear primary subsystem exists, collect only that profile's collectors and
    defer the rest (differential keeps them in ``subsystem_order`` for later). If
    ambiguous (generic), fall back to the broad set.
    """
    ranking = classify_subsystems(symptom_text)
    primary = ranking[0].subsystem if ranking else "generic"
    if primary == "generic" or primary not in PROFILE_COLLECTORS:
        collectors = PROFILE_COLLECTORS["generic"]
    else:
        collectors = PROFILE_COLLECTORS[primary]
    return CollectionPlan(subsystem_order=ranking, collectors=collectors)


def classify_subsystems(text: str) -> list[SubsystemRanking]:
    from .subsystem import classify
    return classify(text)