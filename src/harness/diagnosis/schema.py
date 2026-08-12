"""Pydantic schemas, versioned.

``Diagnosis`` is the machine-readable output sent to the operator/ticketing and
recorded to the audit log. Every output carries ``schema_version``; migrations are
backward compatible (only additive optional fields). No write intent is ever
expressed here -- actions are recommendations only, and ``risk``/``required_tool``
are mandatory so the human approval gate always has full context.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.1.0"


class Subsystem(str, Enum):
    CPU = "cpu"
    MEMORY = "memory"
    PCIE = "pcie"
    BMC = "bmc"
    STORAGE = "storage"
    KERNEL = "kernel"
    GENERIC = "generic"


class ServerState(str, Enum):
    """Structural verdict separating an ACTIVE fault from a fixed server.

    ``healthy`` means current-state evidence (live sensors, kernel state) is
    nominal even if the historical event log references past faults -- a
    server that was repaired and is now fully operational is ``healthy``.
    ``fault``/``degraded`` require current-state evidence of a live problem.
    """

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAULT = "fault"
    UNKNOWN = "unknown"


class Risk(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Reference(BaseModel):
    source: str            # e.g. Server_Arch_v2.3.pdf
    page: str | None = None
    detail: str | None = None


class Action(BaseModel):
    step: int
    action: str
    rationale: str = Field(description="must cite page/part references")
    risk: Risk
    required_tool: str
    impact: str = Field(description="e.g. requires reboot, live test may degrade")
    references: list[Reference] = Field(default_factory=list)


class ConfidenceBreakdown(BaseModel):
    retrieval_citation_support: float = 0.0
    evidence_fit: float = 0.0
    model_agreement: float = 0.0
    penalty: float = 0.0
    calibration_llm: str | None = None  # LLM ident whose calibration resolved model_agreement


CaseOutcomeValues = Literal["fixed", "partial", "not_fixed", "inconclusive", "unknown"]


class Diagnosis(BaseModel):
    schema_version: str = SCHEMA_VERSION
    state: ServerState = ServerState.UNKNOWN
    diagnosis: str
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_breakdown: ConfidenceBreakdown | None = None
    subsystems_considered: list[Subsystem] = Field(default_factory=list)
    actions: list[Action] = Field(default_factory=list)
    references: list[Reference] = Field(default_factory=list)
    evidence: list[Any] = Field(default_factory=list)  # raw decoded registers / summaries
    unknown_registers: list[str] = Field(default_factory=list)
    parts_discrepancies: list[str] = Field(default_factory=list)


class TurnResponse(BaseModel):
    """One agent turn in the multi-turn session: ask a question, request further
    read-only probes, or deliver the final Diagnosis.

    The agent never proposes commands -- ``subsystems``/``doc_topics`` are mapped
    by the harness to curated read-only collectors / the doc library.
    """

    kind: Literal["question", "probe", "diagnosis"]
    question: str | None = None
    subsystems: list[str] = Field(default_factory=list)
    doc_topics: list[str] = Field(default_factory=list)
    diagnosis: Diagnosis | None = None


OUTCOMES = ("fixed", "partial", "not_fixed", "inconclusive", "unknown")


class CaseOutcome(BaseModel):
    """Append-only record of ONE diagnosis run and its verified outcome.

    This is the fleet-learning loop's ground truth: only records with
    ``outcome`` in ("fixed", "partial", "not_fixed") count as verified
    (``inconclusive``/``unknown`` are unverified). ``llm_ident`` records WHICH
    model produced the diagnosis so calibration stays per-model (swap-safe).
    Written once via ``CaseStore``; nothing here is ever overwritten.
    """

    schema_version: int = 1
    run_id: str
    target_id: str
    model_key: str | None = None
    model_source: str | None = None
    symptom: str
    subsystem_primary: str | None = None
    actions_recommended: list[str] = Field(default_factory=list)
    actions_taken: list[str] = Field(default_factory=list)
    outcome: Literal["fixed", "partial", "not_fixed", "inconclusive", "unknown"] = "unknown"
    verification_verdict: str | None = None
    verification_delta: dict = Field(default_factory=dict)
    llm_ident: str = "stub"
    evidence_hash: str = ""
    created_at: str = ""
    # Prompt 07 contract (optional; replay builds deterministic evidence blocks):
    evidence_summary: list[str] = Field(default_factory=list)
    cited_titles: list[str] = Field(default_factory=list)
    # Prompt 06 contract (optional): the diagnosis confidence at report time;
    # the per-bin predicted value calibration bins against.
    confidence: float | None = None

    @property
    def verified(self) -> bool:
        """True when the outcome carries verified signal (not inconclusive/unknown)."""
        return self.outcome in ("fixed", "partial", "not_fixed")