"""Past-case retrieval: verified fleet history ranked for prompt injection.

``CaseLibrary.similar`` scores past verified cases against the current symptom
(outcome-weighted, same-model boosted, BM25 over symptom + action text) so the
diagnostic prompt can show prior fixed machines as fleet history. Holdout ids
passed via ``exclude_holdout`` are NEVER returned (eval integrity).
"""

from __future__ import annotations

import math
from collections import Counter

from .case_store import CaseStore
from .schema import CaseOutcome

_OUTCOME_WEIGHTS = {"fixed": 1.0, "partial": 0.5, "inconclusive": 0.25,
                    "not_fixed": 0.1, "unknown": 0.5}
_SAME_MODEL_MULTIPLIER = 1.5
#: Per shared test-log failure identity (e.g. "P02002001@PCIe Test Fail").
#: A case whose log showed the same harness failure as the current run's log is
#: exactly on-topic, so it is boosted -- but still soft, not a hard filter, so
#: near-matches (same test, different code) keep surfacing.
_LOG_FAILURE_MULTIPLIER = 0.5
_K1, _B = 1.5, 0.75


def _tokenize(text: str) -> list[str]:
    return [t for t in text.lower().split() if len(t) > 2]


class CaseLibrary:
    def __init__(self, store: CaseStore,
                 exclude_holdout: frozenset[str] = frozenset()) -> None:
        self.store = store
        self.cases = [c for c in store.all() if c.run_id not in exclude_holdout]
        self._stats = _corpus_stats([_case_text(c) for c in self.cases])

    def similar(self, symptom: str, model_key: str | None = None, top_k: int = 5,
                outcome_min: float = 0.0,
                exclude_holdout: frozenset[str] | None = None) -> list[tuple[CaseOutcome, float]]:
        """Score cases against ``symptom``; best-first, outcome-weighted.

        ``exclude_holdout`` adds evaluation-time exclusion on top of the
        constructor set (eval mode). Cases scoring below ``outcome_min`` are
        excluded so a not-fixed case never sails through as authoritative.
        """
        excluded = frozenset(exclude_holdout or ()) | {
            c.run_id for c in self.cases if _OUTCOME_WEIGHTS.get(
                c.outcome, 0.0) < outcome_min}
        terms = _tokenize(symptom)
        scored: list[tuple[CaseOutcome, float]] = []
        for case in self.cases:
            if case.run_id in excluded:
                continue
            score = _bm25(terms, _case_text(case), self._stats)
            score *= _OUTCOME_WEIGHTS.get(case.outcome, 0.5)
            if model_key and case.model_key == model_key:
                score *= _SAME_MODEL_MULTIPLIER
            if case.test_log_failures:
                matched = sum(
                    1 for f in case.test_log_failures
                    if f.lower() in symptom.lower())
                if matched:
                    score *= (1.0 + _LOG_FAILURE_MULTIPLIER * matched)
            if score > 0.0:
                scored.append((case, round(score, 6)))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:top_k]


def _case_text(case: CaseOutcome) -> str:
    return " ".join([case.symptom, *case.actions_recommended, *case.actions_taken,
                     *case.evidence_summary, *case.test_log_failures])


def _corpus_stats(texts: list[str]) -> tuple[float, Counter]:
    lengths = [len(_tokenize(t)) for t in texts]
    avg = sum(lengths) / len(lengths) if lengths else 1.0
    doc_freq: Counter = Counter()
    for text in texts:
        for term in set(_tokenize(text)):
            doc_freq[term] += 1
    return avg, doc_freq


def _bm25(terms: list[str], text: str, stats) -> float:
    """BM25 over a plain text body (no Chunk objects needed)."""
    if not terms:
        return 0.0
    avg_len, doc_freq = stats
    tf = Counter(_tokenize(text))
    N = max(len(doc_freq), 1)
    doc_len = len(_tokenize(text))
    k1, b = _K1, _B
    score = 0.0
    for term in terms:
        if term not in tf:
            continue
        idf = math.log(1 + (N - doc_freq.get(term, 0) + 0.5)
                       / (doc_freq.get(term, 0) + 0.5))
        denom = tf[term] + k1 * (1 - b + b * doc_len / avg_len)
        score += idf * (tf[term] * (k1 + 1)) / denom
    return score


def render(records: list[tuple[CaseOutcome, float]]) -> list[str]:
    """One prompt line per case, exact format:

    ``[case {run_id}] model={model_key|'unknown'} outcome={outcome}: {symptom} -> {first action taken or 'n/a'}``
    """
    lines: list[str] = []
    for case, _score in records:
        model = case.model_key or "unknown"
        first_action = case.actions_taken[0] if case.actions_taken else "n/a"
        extra = ""
        if case.test_log_failures:
            extra = " [" + ", ".join(case.test_log_failures) + "]"
        lines.append(
            f"[case {case.run_id}] model={model} outcome={case.outcome}: "
            f"{case.symptom}{extra} -> {first_action}")
    return lines