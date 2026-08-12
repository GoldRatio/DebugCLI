"""Prompt 07: ``harness eval`` -- offline regression against the case library.

The safety net for every change to prompts, catalog, retrieval, or LLM model:
replays held-out VERIFIED cases through the CURRENT pipeline (retrieval ->
prompt -> LLM -> scorer) and reports verdict accuracy, calibration error, and
retrieval quality, then diffs against a stored baseline.

The holdout is deterministic (sha256 of the run id), and the holdout ids are
excluded from case retrieval, priors, and calibration during the eval so
nothing the eval measures was part of the learning loop it tests.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from ..inspect.model import from_alias
from .calibration import RESOLUTION, bin_index
from .case_library import CaseLibrary, render
from .case_store import CaseStore
from .prompt import build_prompt
from .schema import CaseOutcome
from .scorer import score_diagnosis
from .summarize import EvidenceSummary

REPLAY_TOP_K = 3          # snippet titles the replay sees (retrieval quality)
PRIOR_TOP_K = 5           # prior-case lines shown during the replay

# A verified case (fixed/partial/not_fixed) was a fault at diagnosis time, so a
# replay that still detects the fault (fault/degraded) is a verdict hit;
# healthy/unknown replays are misses (the current pipeline would now miss it).
_FAULT_STATES = {"fault", "degraded"}


def holdout_ids(cases: list[CaseOutcome], frac: float = 0.25) -> frozenset[str]:
    """Deterministic holdout partition: n < int(225 * frac) of 256 buckets.

    ``hashlib.sha256(run_id).hexdigest()`` first two hex digits -> int(d, 16);
    ``n < int(225 * frac)`` -> holdout (stable across runs, ~= frac of cases).
    """
    threshold = int(225 * frac)
    return frozenset(
        c.run_id for c in cases
        if int(hashlib.sha256(c.run_id.encode("utf-8")).hexdigest()[:2], 16)
        < threshold)


@dataclass
class ReplayResult:
    run_id: str
    subsystem: str | None
    state: str
    verdict_hit: bool
    confidence: float
    citation_support: float
    retrieval_recall: float
    retrieved_titles: list[str] = field(default_factory=list)
    cited_titles: list[str] = field(default_factory=list)


def replay_case(case: CaseOutcome, llm, rag, prior_cases: list[str]) -> ReplayResult:
    """Replay ONE holdout case through the CURRENT pipeline.

    The evidence block is rebuilt deterministically from the case record:
    symptom, model_key, and the stored decoded-register lines
    (``CaseOutcome.evidence_summary``). Retrieval is re-run over the CURRENT
    doc library with ``rag_platform_filter=model_key`` (top-3). The replay
    LLM's structured state must still detect the fault the case recorded.
    """
    model = from_alias(case.model_key) if case.model_key else None
    snippets = rag.retrieve(case.symptom, top_k=REPLAY_TOP_K,
                            platform=case.model_key)
    lines = [s.as_line() for s in snippets]
    titles = [s.title for s in snippets]
    prompt = build_prompt(
        model=model,
        decoded=[],
        summaries=EvidenceSummary(
            interesting=[], anomaly_count=0, total=0,
            notes=list(case.evidence_summary)),
        doc_snippets=lines,
        parts_refs=[],
        symptom=case.symptom,
        prior_cases=prior_cases,
    )
    scored = score_diagnosis(llm(prompt), retrieved_snippets=lines)
    state = scored.state.value
    cited = set(case.cited_titles)
    recall = (len(set(titles) & cited) / len(cited)) if cited else 0.0
    support = (scored.confidence_breakdown.retrieval_citation_support
               if scored.confidence_breakdown else 0.0)
    return ReplayResult(
        run_id=case.run_id,
        subsystem=case.subsystem_primary or "generic",
        state=state,
        verdict_hit=state in _FAULT_STATES,
        confidence=scored.confidence,
        citation_support=support,
        retrieval_recall=recall,
        retrieved_titles=titles,
        cited_titles=case.cited_titles,
    )


def _ece(results: list[ReplayResult], outcomes: dict[str, str]) -> float:
    """Expected calibration error over replayed confidences vs real outcomes.

    Reuses the calibration bin edges (``bin_index``/``BIN_EDGES``): for each
    bin, |observed resolution - mean predicted confidence|, weighted by
    bin population.
    """
    if not results:
        return 0.0
    per_bin: dict[int, list[tuple[float, float]]] = {}
    for r in results:
        per_bin.setdefault(bin_index(r.confidence), []).append(
            (r.confidence, RESOLUTION.get(outcomes[r.run_id], 0.5)))
    total = 0.0
    n = len(results)
    for pairs in per_bin.values():
        confs = [c for c, _ in pairs]
        rate = sum(o for _, o in pairs) / len(pairs)
        total += (len(pairs) / n) * abs(rate - sum(confs) / len(confs))
    return round(total, 4)


def evaluate(store: CaseStore, llm, rag, llm_ident: str, *,
             frac: float = 0.25,
             excluded: frozenset[str] | None = None,
             library: CaseLibrary | None = None) -> dict:
    """Full eval report: replay every holdout VERIFIED case, aggregate metrics.

    Returns the JSON-ready report dict. ``frac`` is the deterministic holdout
    fraction (default 0.25). ``excluded`` lets callers layer an
    extra exclusion set on top of the built-in holdout (e.g. a second eval
    split); the same set is always passed to ``CaseLibrary.similar`` so
    retrieval never leaks holdout cases. ``library`` injects a pre-built
    (possibly spying) case library; when None one is constructed with the
    holdout pre-excluded.
    """
    cases = store.all()
    verified = [c for c in cases if c.verified]
    holdout = holdout_ids(verified, frac) | frozenset(excluded or ())
    library = library or CaseLibrary(store, exclude_holdout=holdout)
    replayed: list[CaseOutcome] = []
    results: list[ReplayResult] = []
    for case in verified:
        if case.run_id not in holdout:
            continue
        replayed.append(case)
        prior = render(library.similar(
            case.symptom, case.model_key, top_k=PRIOR_TOP_K,
            exclude_holdout=holdout))
        results.append(replay_case(case, llm, rag, prior))
    return _report(verified, results, llm_ident, holdout)


def _report(verified: list[CaseOutcome], results: list[ReplayResult],
            llm_ident: str, holdout: frozenset[str]) -> dict:
    """Aggregate per-case metrics into the report dict (JSON-ready)."""
    from datetime import UTC, datetime

    outcomes = {c.run_id: c.outcome for c in verified}
    by_subsystem: dict[str, list[ReplayResult]] = {}
    for r in results:
        by_subsystem.setdefault(r.subsystem, []).append(r)

    def _metrics(group: list[ReplayResult]) -> dict:
        hits = sum(1 for r in group if r.verdict_hit)
        return {
            "verdict_accuracy": round(hits / len(group), 3) if group else 0.0,
            "ece": _ece(group, outcomes),
            "mean_citation_support": round(
                sum(r.citation_support for r in group) / len(group), 3) if group else 0.0,
            "mean_retrieval_recall": round(
                sum(r.retrieval_recall for r in group) / len(group), 3) if group else 0.0,
            "n": len(group),
        }

    overall = _metrics(results)
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "llm_ident": llm_ident,
        "n_verified": len(verified),
        "n_holdout": len(holdout),
        "n_replayed": len(results),
        "holdout_ids": sorted(holdout),
        "verdict_accuracy": overall["verdict_accuracy"],
        "ece": overall["ece"],
        "mean_citation_support": overall["mean_citation_support"],
        "mean_retrieval_recall": overall["mean_retrieval_recall"],
        "per_subsystem": {s: _metrics(g) for s, g in sorted(by_subsystem.items())},
        "cases": [
            {**{k: getattr(r, k) for k in
                 ("run_id", "subsystem", "state", "verdict_hit", "confidence",
                  "citation_support", "retrieval_recall",
                  "retrieved_titles", "cited_titles")}}
            for r in results],
    }