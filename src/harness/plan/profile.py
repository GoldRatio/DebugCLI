"""Profile: use the subsystem ranking to build an ordered, minimal collector plan.

Doc-guided planning extends the heuristic: when doc snippets are available they are
mined for read-only probe commands the docs explicitly name (e.g. ``i2cdump -y 8 0xb``
for amber-light boot state) and the accepted commands ride along on the plan as
``doc_probes`` so the engine can run them in the same collection pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..inspect.model import PROFILE_COLLECTORS
from .doc_guided import mine_probe_commands
from .subsystem import SubsystemRanking


@dataclass(frozen=True)
class CollectionPlan:
    subsystem_order: list[SubsystemRanking]
    collectors: list[str] = field(default_factory=list)
    doc_probes: list[str] = field(default_factory=list)  # doc-named read-only commands

    @property
    def primary_subsystem(self) -> str | None:
        return self.subsystem_order[0].subsystem if self.subsystem_order else None


def plan_collection(symptom_text: str, doc_snippets: list[str] | None = None) -> CollectionPlan:
    """Return the minimal ordered collector set for the classified subsystem(s).

    If a clear primary subsystem exists, collect only that profile's collectors and
    defer the rest (differential keeps them in ``subsystem_order`` for later). If
    ambiguous (generic), fall back to the broad set. ``doc_snippets`` are mined for
    probe commands the docs name; each is probe-gate validated before it joins the plan.
    """
    ranking = classify_subsystems(symptom_text)
    primary = ranking[0].subsystem if ranking else "generic"
    if primary == "generic" or primary not in PROFILE_COLLECTORS:
        collectors = PROFILE_COLLECTORS["generic"]
    else:
        collectors = PROFILE_COLLECTORS[primary]
    doc_probes = mine_probe_commands(doc_snippets) if doc_snippets else []
    return CollectionPlan(subsystem_order=ranking, collectors=collectors,
                          doc_probes=doc_probes)


def classify_subsystems(text: str) -> list[SubsystemRanking]:
    from .subsystem import classify
    return classify(text)