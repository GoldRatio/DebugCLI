"""Confidence scoring -- transparent, reproducible (see spec section 4).

``confidence``  = 0.55 * retrieval_citation_support
                + 0.30 * evidence_fit
                + 0.15 * model_agreement
                - penalty
"""

from __future__ import annotations

from .schema import ConfidenceBreakdown, Diagnosis

WEIGHTS = {"retrieval": 0.55, "evidence": 0.30, "agreement": 0.15}


def evidence_fit_from_dumps(dump_sets) -> float:
    """Evidence fit for runs whose probes are not register-decodable.

    BMC console probes (``ipmitool sensor/sel/fru``, BusyBox ``dmesg -r``) return
    raw text evidence the register catalog cannot decode, so the register-based
    fit would wrongly floor at 0.3. Here the fit is the fraction of probes that
    succeeded and produced output.
    """
    dumps = [d for ds in dump_sets.values() for d in ds]
    if not dumps:
        return 0.0
    useful = sum(1 for d in dumps if d.ok and (d.raw or "").strip())
    return round(useful / len(dumps), 3)


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


def apply_to(diagnosis: Diagnosis, breakdown: ConfidenceBreakdown) -> Diagnosis:
    diagnosis.confidence_breakdown = breakdown
    diagnosis.confidence = round(
        breakdown.retrieval_citation_support * WEIGHTS["retrieval"]
        + breakdown.evidence_fit * WEIGHTS["evidence"]
        + breakdown.model_agreement * WEIGHTS["agreement"]
        - breakdown.penalty,
        3,
    )
    diagnosis.confidence = max(0.0, min(1.0, diagnosis.confidence))
    return diagnosis


def score_diagnosis(diagnosis: Diagnosis, *,
                    retrieved_snippets: list[str] | None = None,
                    evidence_fit: float | None = None,
                    model_agreement: float = 0.5) -> Diagnosis:
    """Compute the transparent confidence breakdown from the actual run data.

    ``retrieval_citation_support`` = fraction of the diagnosis's cited references
    (top-level plus every action's references) whose (source, page) text actually
    appears in the retrieved snippet set, so the number is reproducible and forces
    the model to cite. A reference without a page is supported when any snippet of
    that source was retrieved.

    ``evidence_fit`` defaults to the fraction of decoded registers that are known
    (not flagged unknown); 0.3 when the run produced no decoded registers at all.

    Penalties: unknown registers (0.1 each, capped at 0.3) and an empty action
    list (0.05).
    """
    if retrieved_snippets:
        joined = "\n".join(retrieved_snippets)
        refs = list(diagnosis.references)
        for action in diagnosis.actions:
            for ref in action.references:
                if not any(r.source == ref.source and r.page == ref.page
                           for r in refs):
                    refs.append(ref)
        support = 0.0
        for ref in refs:
            if ref.page:
                cited = (f"[{ref.source} p.{ref.page}]",
                         f"{ref.source} p.{ref.page}]")
                supported = any(c in joined for c in cited)
            else:
                supported = f"[{ref.source} p." in joined
            if supported:
                support += 1.0
        support /= max(len(refs), 1)
    else:
        support = 0.0

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
    return apply_to(diagnosis, breakdown)