"""Prompt 05: learn symptom-keyword -> subsystem likelihood multipliers.

``build_priors`` turns VERIFIED outcome records (never unverified runs) into a
``PriorModel``: for each symptom keyword we count, per diagnosed subsystem,
the outcome-weighted prevalence (fixed=1.0, partial=0.5, not_fixed=-0.75,
inconclusive/unknown=0.25), Laplace-smoothed against the fleet-wide total,
then clamped to [0.5, 2.0]. Builds return None until at least ``min_verified``
outcome-recorded cases exist; callers then keep the static heuristic table
exactly as today.
"""

from __future__ import annotations

import json
from collections import defaultdict

from ..diagnosis.schema import CaseOutcome
from .subsystem import _SYMPTOM_TO_SUBSYSTEM, PriorModel, clamp_multiplier

# Outcome weights per prompt 05 (negative evidence discounts a keyword's pull).
_OUTCOME_WEIGHTS = {
    "fixed": 1.0,
    "partial": 0.5,
    "not_fixed": -0.75,
    "inconclusive": 0.25,
    "unknown": 0.25,
}


def _symptom_keywords(text: str) -> set[str]:
    """Keywords in ``text`` via the same substring matching ``classify`` uses."""
    lowered = text.lower()
    return {kw for keywords in _SYMPTOM_TO_SUBSYSTEM.values()
            for kw in keywords if kw in lowered}


def build_priors(cases: list[CaseOutcome],
                 min_verified: int = 10) -> PriorModel | None:
    """Build ``PriorModel`` multipliers from outcome-recorded cases.

    For each symptom keyword, the per-subsystem weighted prevalence
    ``count_s`` (weighted by outcome, restricted to cases whose primary
    subsystem is ``s``) is Laplace-smoothed against the fleet-wide weighted
    total: ``mult = (count_s + 1) / (count_all + 1) / (1 / n_subsystems)``,
    clamped into [0.5, 2.0] at build time. Keywords with a net-zero fleet-wide
    weight (unseen or washed-out evidence) are omitted entirely, so
    ``PriorModel.covers`` stays False and classify degrades to 1.0 per hit.

    Returns None until at least ``min_verified`` outcome-recorded cases exist
    fleet-wide (callers keep static behavior), or when no keyword has signal.
    """
    recorded = [c for c in cases if c.outcome in _OUTCOME_WEIGHTS]
    if len(recorded) < min_verified:
        return None

    n_subsystems = len(_SYMPTOM_TO_SUBSYSTEM)
    subsystem_hits: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float))
    total_hits: dict[str, float] = defaultdict(float)
    for case in recorded:
        weight = _OUTCOME_WEIGHTS[case.outcome]
        primary = case.subsystem_primary
        for keyword in _symptom_keywords(case.symptom):
            total_hits[keyword] += weight
            if primary:
                subsystem_hits[keyword][primary] += weight

    multipliers: dict[str, dict[str, float]] = {}
    for keyword, total in total_hits.items():
        if total <= 0.0:
            continue  # net-zero evidence -> no multiplier (static behavior)
        per: dict[str, float] = {}
        for subsystem in _SYMPTOM_TO_SUBSYSTEM:
            count_s = subsystem_hits[keyword].get(subsystem, 0.0)
            smoothed = (count_s + 1.0) / (total + 1.0) / (1.0 / n_subsystems)
            per[subsystem] = clamp_multiplier(smoothed)
        multipliers[keyword] = per
    if not multipliers:
        return None
    return PriorModel(keyword_multipliers=multipliers)


def dump_priors(priors: PriorModel) -> str:
    """JSON form of a ``PriorModel`` (defaults are static, not serialized)."""
    return json.dumps(
        {"keyword_multipliers": priors.keyword_multipliers},
        indent=2, sort_keys=True)


def load_priors(text: str) -> PriorModel:
    """Rebuild a ``PriorModel`` from ``dump_priors`` output."""
    data = json.loads(text)
    return PriorModel(
        keyword_multipliers=data.get("keyword_multipliers") or {})