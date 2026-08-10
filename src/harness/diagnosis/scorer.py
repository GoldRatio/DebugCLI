"""Confidence scoring -- transparent, reproducible (see spec section 4).

``confidence``  = 0.55 * retrieval_citation_support
                + 0.30 * evidence_fit
                + 0.15 * model_agreement
                - penalty
"""

from __future__ import annotations

import re

from .schema import ConfidenceBreakdown, Diagnosis

WEIGHTS = {"retrieval": 0.55, "evidence": 0.30, "agreement": 0.15}


def evidence_fit_from_dumps(diagnosis, dump_sets) -> float:
    """Evidence fit for runs whose probes are not register-decodable.

    BMC console probes (``ipmitool sensor/sel/fru``, BusyBox ``dmesg -r``) return
    raw text evidence the register catalog cannot decode, so the register-based
    fit would wrongly floor at 0.3. Instead of rewarding mere probe success, this
    measures CONSISTENCY between the diagnosis's structured verdict and the
    CURRENT STATE evidence (live sensor readings):

    - verdict ``fault``/``degraded`` AND live sensors show a fault   -> 1.0
    - verdict ``fault`` but every live sensor is ok                  -> 0.3
      (over-weighting historical SEL entries over current state)
    - verdict ``healthy`` and sensors are ok                         -> 1.0
    - verdict ``healthy`` but sensors show a fault                   -> 0.3
    - verdict ``degraded`` with sensors all ok                       -> 0.6
    - verdict ``unknown`` (or a plain-text claim when the model gave
      no verdict), or no sensor evidence at all                     -> fall back
      to the fraction of probes that succeeded and produced output
    """
    if not dump_sets:
        return 0.0
    sensor_anomaly, sensor_seen = _sensor_health(dump_sets)
    if sensor_seen:
        verdict = _verdict_of(diagnosis)
        if verdict in ("fault", "degraded"):
            return 1.0 if sensor_anomaly else (0.3 if verdict == "fault" else 0.6)
        if verdict == "healthy":
            return 1.0 if not sensor_anomaly else 0.3
    dumps = [d for ds in dump_sets.values() for d in ds]
    if not dumps:
        return 0.0
    useful = sum(1 for d in dumps if d.ok and (d.raw or "").strip())
    return round(useful / len(dumps), 3)


_PROBLEM_RE = re.compile(
    r"\b(fail(?:ure|ed)?s?|faults?|error|problem|defective|degraded|critical|mce|ecc)\b",
    re.IGNORECASE)
_HEALTHY_RE = re.compile(
    r"\b(no\s+issue|no\s+problem|no\s+active|no\s+fault|no\s+failure|healthy|"
    r"resolved|nominal|normal|all\s+ok|nothing\s+(?:wrong|wrongly))\b",
    re.IGNORECASE)


def _verdict_of(diagnosis) -> str | None:
    """Prefer the structured ``Diagnosis.state`` verdict; fall back to keyword
    classification of the diagnosis text (or a plain string)."""
    state = getattr(diagnosis, "state", None)
    if state is not None:
        value = str(getattr(state, "value", state))
        if value in ("healthy", "degraded", "fault"):
            return value
    text = getattr(diagnosis, "diagnosis", None) or diagnosis
    if isinstance(text, str):
        if _HEALTHY_RE.search(text):
            return "healthy"
        if _PROBLEM_RE.search(text):
            return "fault"
    return None


def _sensor_health(dump_sets) -> tuple[bool, bool]:
    """(any_non_ok_sensor, sensor_evidence_seen) from live sensor dumps."""
    from .summarize import _dump_kind, _parse_sensor_line
    seen = False
    for dumps in dump_sets.values():
        for dump in dumps:
            if not dump.ok or _dump_kind(dump) != "sensor":
                continue
            for line in dump.raw.splitlines():
                name, status = _parse_sensor_line(line)
                if name is None:
                    continue
                seen = True
                if status not in ("ok", "na"):
                    return True, True
    return False, seen


class Scorer:
    def score(self, *,
              retrieval_citation_support: float,
              evidence_fit: float,
              model_agreement: float,
              penalty: float = 0.0) -> ConfidenceBreakdown:
        return ConfidenceBreakdown(
            retrieval_citation_support=round(retrieval_citation_support, 3),
            evidence_fit=round(evidence_fit, 3),
            model_agreement=round(model_agreement, 3),
            penalty=round(penalty, 3),
        )


def apply_to(diagnosis: Diagnosis, breakdown: ConfidenceBreakdown,
             weights: dict[str, float] | None = None) -> Diagnosis:
    """Attach the breakdown and compute confidence from the effective weights.

    ``weights`` lets callers drop components that cannot be evaluated (e.g. the
    retrieval/citation component when the diagnosis cites no document, or when
    no doc snippets were retrieved); confidence is renormalized over the
    remaining components so an inapplicable component neither inflates nor
    deflates the score.
    """
    w = weights or WEIGHTS
    total = w["retrieval"] + w["evidence"] + w["agreement"]
    diagnosis.confidence_breakdown = breakdown
    if total <= 0:
        diagnosis.confidence = 0.0
    else:
        diagnosis.confidence = round(
            (breakdown.retrieval_citation_support * w["retrieval"]
             + breakdown.evidence_fit * w["evidence"]
             + breakdown.model_agreement * w["agreement"]) / total
            - breakdown.penalty,
            3,
        )
    diagnosis.confidence = max(0.0, min(1.0, diagnosis.confidence))
    return diagnosis


# Prompt sections are evidence, not documents: the model must not "cite" them,
# and the scorer must not treat them as unsupported citations.
_NON_DOC_SOURCES = frozenset({
    "evidence notes", "symptom", "anomalous evidence summary",
    "decoded registers", "relevant architecture snippets",
    "parts list references", "conversation so far", "system",
})


def _is_doc_ref(ref) -> bool:
    return ref.source.strip().lower() not in _NON_DOC_SOURCES


def score_diagnosis(diagnosis: Diagnosis, *,
                    retrieved_snippets: list[str] | None = None,
                    evidence_fit: float | None = None,
                    model_agreement: float = 0.5) -> Diagnosis:
    """Compute the transparent confidence breakdown from the actual run data.

    ``retrieval_citation_support`` = fraction of the diagnosis's cited references
    (top-level plus every action's references, excluding in-prompt evidence
    sections) whose (source, page) text actually appears in the retrieved snippet
    set, so the number is reproducible and forces the model to cite real
    documents. A reference without a page is supported when any snippet of that
    source was retrieved.

    When there are no document references to verify (a healthy/no-issue verdict
    typically cites none), or when no doc snippets were retrieved at all, the
    retrieval/citation component is dropped and confidence is renormalized over
    the remaining components (evidence_fit, model_agreement) -- an unverifiable
    component must neither inflate nor punish the score.

    ``evidence_fit`` defaults to the fraction of decoded registers that are known
    (not flagged unknown); 0.3 when the run produced no decoded registers at all.

    Penalties: unknown registers (0.1 each, capped at 0.3) and an empty action
    list (0.05).
    """
    support = 0.0
    weights = dict(WEIGHTS)
    if retrieved_snippets:
        joined = "\n".join(retrieved_snippets)
        refs = list(diagnosis.references)
        for action in diagnosis.actions:
            for ref in action.references:
                if not any(r.source == ref.source and r.page == ref.page
                           for r in refs):
                    refs.append(ref)
        refs = [r for r in refs if _is_doc_ref(r)]
        if refs:
            for ref in refs:
                if ref.page:
                    cited = (f"[{ref.source} p.{ref.page}]",
                             f"{ref.source} p.{ref.page}]")
                    supported = any(c in joined for c in cited)
                else:
                    supported = f"[{ref.source} p." in joined
                if supported:
                    support += 1.0
            support /= len(refs)
        else:
            # No document to verify: the citation component does not apply.
            weights["retrieval"] = 0.0
    else:
        # No doc library/retrieval this run: citations cannot be verified.
        weights["retrieval"] = 0.0

    if evidence_fit is None:
        evidence = [d for d in diagnosis.evidence if isinstance(d, dict)]
        if not evidence:
            evidence_fit = 0.3
        else:
            evidence_fit = sum(1 for d in evidence if not d.get("unknown")) / len(evidence)

    penalty = 0.0
    if diagnosis.unknown_registers:
        penalty += min(0.3, 0.1 * len(diagnosis.unknown_registers))
    if not diagnosis.actions:
        penalty += 0.05

    breakdown = Scorer().score(
        retrieval_citation_support=support,
        evidence_fit=evidence_fit,
        model_agreement=model_agreement,
        penalty=penalty,
    )
    return apply_to(diagnosis, breakdown, weights)