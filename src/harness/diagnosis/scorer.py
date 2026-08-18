"""Confidence scoring -- transparent, reproducible (see spec section 4).

``confidence``  = 0.55 * retrieval_citation_support
                + 0.30 * evidence_fit
                + 0.15 * model_agreement
                - penalty

Runs that decoded a power-rail FAILURE POINT (a rail fault register, not a root
cause) switch to the failure-point weights, which add a root-cause-certainty
component measuring how well the diagnosis's evidence discriminates the named
root cause from the failing rail's documented suspect set:

``confidence``  = 0.40 * retrieval_citation_support
                + 0.25 * evidence_fit
                + 0.10 * model_agreement
                + 0.25 * root_cause_certainty
                - penalty

``root_cause_certainty`` = 0.5 * suspect_coverage + 0.5 * discrimination, where
``suspect_coverage`` is the fraction of the documented suspects (rail loads +
supplying board) the diagnosis text actually addresses, and ``discrimination``
is 1.0 only when isolation probes ran AND the diagnosis engages their evidence
(cites an isolation doc page or proposes a discriminating step such as an
impedance measurement or a busbar removal). This is the incident lesson: a
diagnosis that jumps to one FRU off a bare failure point ("replace PDB" when a
shorted load pulls the rail down) must score LOW, while one that discriminates
among all documented suspects scores high.
"""

from __future__ import annotations

import re

from .schema import ConfidenceBreakdown, Diagnosis

WEIGHTS = {"retrieval": 0.55, "evidence": 0.30, "agreement": 0.15}

# Failure-point runs: the certainty component joins the formula (the other
# components rebalance); non-failure-point runs keep the classic weights.
FAILURE_POINT_WEIGHTS = {"retrieval": 0.40, "evidence": 0.25,
                         "agreement": 0.10, "certainty": 0.25}

# Diagnosis text that engages DISCRIMINATING evidence (an isolation measurement
# or comparison that separates one suspect from the others on the rail).
_DISCRIMINATOR_RE = re.compile(
    r"\b(measur\w+|impedance|isolat\w+|swap[- ]?test|remove the (?:inner|internal)"
    r" busbar|register dump|fpga dump|dump\w*|decod\w+|compar\w+|each load|"
    r"both biancas?|one at a time)\b",
    re.IGNORECASE)


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
              penalty: float = 0.0,
              root_cause_certainty: float | None = None) -> ConfidenceBreakdown:
        return ConfidenceBreakdown(
            retrieval_citation_support=round(retrieval_citation_support, 3),
            evidence_fit=round(evidence_fit, 3),
            model_agreement=round(model_agreement, 3),
            penalty=round(penalty, 3),
            root_cause_certainty=(round(root_cause_certainty, 3)
                                  if root_cause_certainty is not None else None),
        )


def _diagnosis_text(diagnosis: Diagnosis) -> str:
    """All free text the model produced: verdict + actions + rationales."""
    parts = [diagnosis.diagnosis or ""]
    for action in diagnosis.actions:
        parts.append(action.action or "")
        parts.append(action.rationale or "")
    return "\n".join(parts)


def root_cause_certainty(diagnosis: Diagnosis) -> float | None:
    """How well the evidence discriminates the root cause (0..1), or None.

    None when the run decoded no power-rail failure point (component not
    applicable). Otherwise:

    - ``suspect_coverage``: fraction of the documented suspects on the failing
      rail (topology loads + supplying board) that the diagnosis text actually
      addresses. A diagnosis that tunnel-visions on one FRU leaves the others
      unconsidered. When no documented suspects exist, coverage is a neutral
      0.5 placeholder.
    - ``discrimination``: 1.0 only when isolation probes ran AND the diagnosis
      engages their evidence -- it cites an isolation doc page or proposes a
      discriminating step (impedance measurement, busbar removal, comparing
      loads). Otherwise 0.0.

    ``certainty = 0.5 * coverage + 0.5 * discrimination``.
    """
    fp = getattr(diagnosis, "failure_point", None)
    if fp is None:
        return None
    text = _diagnosis_text(diagnosis).lower()

    suspects = [s for s in (fp.suspects or []) if s]
    if suspects:
        addressed = sum(1 for s in suspects if s.lower() in text)
        coverage = addressed / len(suspects)
    else:
        coverage = 0.5  # no documented suspect set: neutral, documented placeholder

    discrimination = 0.0
    if fp.isolation_ran:
        engaged = bool(_DISCRIMINATOR_RE.search(_diagnosis_text(diagnosis)))
        if not engaged and fp.isolation_refs:
            joined_refs = "\n".join(fp.isolation_refs)
            for ref in diagnosis.references:
                if not _is_doc_ref(ref):
                    continue
                if ref.page and (f"[{ref.source} p.{ref.page}]" in joined_refs
                                 or f"{ref.source} p.{ref.page}]" in joined_refs):
                    engaged = True
                    break
            if not engaged:
                for action in diagnosis.actions:
                    for ref in action.references:
                        if not _is_doc_ref(ref):
                            continue
                        if ref.page and (
                                f"[{ref.source} p.{ref.page}]" in joined_refs
                                or f"{ref.source} p.{ref.page}]" in joined_refs):
                            engaged = True
                            break
        discrimination = 1.0 if engaged else 0.0
    return round(0.5 * coverage + 0.5 * discrimination, 3)


def apply_to(diagnosis: Diagnosis, breakdown: ConfidenceBreakdown,
             weights: dict[str, float] | None = None) -> Diagnosis:
    """Attach the breakdown and compute confidence from the effective weights.

    ``weights`` lets callers drop components that cannot be evaluated (e.g. the
    retrieval/citation component when the diagnosis cites no document, or when
    no doc snippets were retrieved), or add the certainty component for
    failure-point runs; confidence is renormalized over the remaining
    components so an inapplicable component neither inflates nor deflates the
    score.
    """
    w = weights or WEIGHTS
    total = (w["retrieval"] + w["evidence"] + w["agreement"]
             + w.get("certainty", 0.0))
    diagnosis.confidence_breakdown = breakdown
    if total <= 0:
        diagnosis.confidence = 0.0
    else:
        certainty_term = 0.0
        certainty_val = getattr(breakdown, "root_cause_certainty", None)
        if certainty_val is not None and w.get("certainty"):
            certainty_term = certainty_val * w["certainty"]
        diagnosis.confidence = round(
            (breakdown.retrieval_citation_support * w["retrieval"]
             + breakdown.evidence_fit * w["evidence"]
             + breakdown.model_agreement * w["agreement"]
             + certainty_term) / total
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
    "prior verified cases",
})


def _is_doc_ref(ref) -> bool:
    return ref.source.strip().lower() not in _NON_DOC_SOURCES


def score_diagnosis(diagnosis: Diagnosis, *,
                    retrieved_snippets: list[str] | None = None,
                    evidence_fit: float | None = None,
                    model_agreement: float | None = None,
                    calibration_root=None,
                    llm_ident: str | None = None) -> Diagnosis:
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

    ``model_agreement`` (prompt 06): None resolves through the calibration store
    at ``calibration_root`` for ``llm_ident`` using the primary subsystem; any
    failure resolves to the historical 0.5 default (never an exception to
    callers). The bin is selected by the model's SELF-reported confidence (the
    value ``Diagnosis.confidence`` carries when the scorer runs, before it is
    overwritten) and the returned agreement is that bin's observed fix rate
    shrunk toward 0.5 -- so it is a correctness prior, not a calibration-error
    score. ``ConfidenceBreakdown.calibration_llm`` records which ident the
    agreement came from (None when uncalibrated), and
    ``ConfidenceBreakdown.self_reported_confidence`` preserves the raw bin key.

    Penalty: an empty action list (0.05). Unknown registers are reported in
    ``Diagnosis.unknown_registers`` but no longer dock the score -- an
    incomplete register catalog is a harness gap, not a diagnosis error.
    """
    # Capture the model's self-reported confidence BEFORE scoring: it selects
    # the calibration bin and is preserved in the breakdown for the case store.
    self_reported = diagnosis.confidence
    if model_agreement is None:
        model_agreement, calibration_llm = _resolved_agreement(
            diagnosis, calibration_root, llm_ident)
    else:
        calibration_llm = None
    support = 0.0
    # A decoded power-rail failure point switches the formula to the
    # failure-point weights: root-cause certainty joins the score.
    certainty = root_cause_certainty(diagnosis)
    weights = dict(FAILURE_POINT_WEIGHTS) if certainty is not None else dict(WEIGHTS)
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
    if not diagnosis.actions:
        penalty += 0.05

    breakdown = Scorer().score(
        retrieval_citation_support=support,
        evidence_fit=evidence_fit,
        model_agreement=model_agreement,
        penalty=penalty,
        root_cause_certainty=certainty,
    )
    breakdown.calibration_llm = calibration_llm
    breakdown.self_reported_confidence = self_reported
    return apply_to(diagnosis, breakdown, weights)


def _resolved_agreement(diagnosis: Diagnosis, calibration_root,
                        llm_ident: str | None) -> tuple[float, str | None]:
    """Calibrated ``model_agreement`` via the per-ident fix-rate store.

    Never raises: any failure (missing store, unknown ident, unreadable file)
    resolves to the historical 0.5 default with ``calibration_llm=None``.
    A calibration store is only consulted when its directory already exists
    (a diagnose run must never CREATE calibration data). The primary subsystem
    (``subsystems_considered[0]``) selects the subsystem histogram; unknown
    subsystems fall back to the aggregate bin.
    """
    if calibration_root is None:
        return 0.5, None
    try:
        from pathlib import Path
        if not Path(calibration_root).is_dir():
            return 0.5, None
        from .calibration import CalibrationStore, agreement_for
        ident = llm_ident or "stub"
        cal = CalibrationStore(calibration_root).load(ident)
        if cal is None:
            return 0.5, None
        primary = None
        if diagnosis.subsystems_considered:
            primary = diagnosis.subsystems_considered[0].value
        agreement = agreement_for(
            cal, diagnosis.confidence if diagnosis.confidence else 0.5, primary)
        return agreement, ident
    except Exception:  # noqa: BLE001 - calibration must never break scoring
        return 0.5, None