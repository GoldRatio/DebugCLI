"""Prompt 07: harness eval -- holdout, replay, baseline diff (acceptance)."""

from pathlib import Path
from types import SimpleNamespace

from harness.diagnosis.case_library import CaseLibrary
from harness.diagnosis.case_store import CaseStore
from harness.diagnosis.eval import evaluate, holdout_ids
from harness.diagnosis.llm import StubLLM
from harness.diagnosis.schema import CaseOutcome
from harness.docs.retrieval.rag import CitedSnippet


def _case(run_id: str, *, subsystem: str = "memory", outcome: str = "fixed",
          symptom: str = "ECC errors on DIMM_A2", model_key: str = "poweredge_r650",
          cited=None, evidence=None, confidence: float = 0.8) -> CaseOutcome:
    return CaseOutcome(
        run_id=run_id,
        target_id="target-1",
        model_key=model_key,
        symptom=symptom,
        subsystem_primary=subsystem,
        outcome=outcome,
        actions_taken=["reseated DIMM A2"],
        llm_ident="stub",
        evidence_summary=list(evidence or ["MCE: memory error on DIMM_A2"]),
        cited_titles=list(cited or []),
        confidence=confidence,
    )


def _fill(store: CaseStore, cases: list[CaseOutcome]) -> None:
    for c in cases:
        store.save(c)


class FakeRag:
    """Deterministic retriever stand-in: returns the given snippet titles."""

    def __init__(self, titles: list[str]) -> None:
        self._titles = titles

    def retrieve(self, query: str, top_k: int = 3, platform=None):
        return [CitedSnippet(text=f"snippet {t} for {query}",
                             title=t, page=1)
                for t in self._titles[:top_k]]


class SpyLibrary(CaseLibrary):
    """Records every ``similar(...)`` call's kwargs (exclude_holdout spy)."""

    def __init__(self, store: CaseStore, exclude_holdout=None) -> None:
        super().__init__(store, exclude_holdout=frozenset(exclude_holdout or ()))
        self.similar_calls: list[dict] = []

    def similar(self, symptom, model_key=None, top_k=5, outcome_min=0.0,
                exclude_holdout=None):
        self.similar_calls.append(
            {"symptom": symptom, "model_key": model_key,
             "exclude_holdout": frozenset(exclude_holdout or ())})
        return super().similar(symptom, model_key=model_key, top_k=top_k,
                               outcome_min=outcome_min,
                               exclude_holdout=exclude_holdout)


# ---- 1. holdout partition ------------------------------------------------


def test_holdout_split_is_deterministic_and_stable():
    cases = [_case(f"run-{i:03d}") for i in range(40)]
    first = holdout_ids(cases, frac=0.25)
    second = holdout_ids(cases, frac=0.25)
    assert first == second
    assert 0 < len(first) < 40
    # same run id hashes to the same bucket, independent of list order
    assert holdout_ids(list(reversed(cases)), frac=0.25) == first


def test_holdout_frac_flag_propagates_through_evaluate():
    """--holdout-frac must change the split; frac=0 takes nothing."""
    cases = [_case(f"run-{i:03d}") for i in range(40)]
    quarter = holdout_ids(cases, frac=0.25)
    minimal = holdout_ids(cases, frac=0.01)
    assert len(minimal) < len(quarter)
    assert holdout_ids(cases, frac=0.0) == frozenset()
    # contract caps the split at int(225 * frac) of 256 buckets, so frac=1.0
    # tops out around 0.88 of cases -- but never shrinks the split
    top = holdout_ids(cases, frac=1.0)
    assert quarter <= top
    assert len(top) >= len(quarter)


def test_holdout_excludes_from_case_retrieval(tmp_path):
    """exclude_holdout propagates to CaseLibrary.similar during the eval."""
    cases = [_case(f"run-{i:03d}") for i in range(30)]
    store = CaseStore(tmp_path / "cases")
    _fill(store, cases)
    expected = frozenset(holdout_ids(cases, frac=0.25))
    spy = SpyLibrary(store)
    evaluate(store, StubLLM(), FakeRag(["A.pdf"]), "stub", library=spy)
    assert all(call["exclude_holdout"] == expected for call in spy.similar_calls)
    # a non-default frac flows into the same retrieval exclusion
    spy2 = SpyLibrary(store)
    small = frozenset(holdout_ids(cases, frac=0.05))
    evaluate(store, StubLLM(), FakeRag(["A.pdf"]), "stub",
             frac=0.05, library=spy2)
    assert all(call["exclude_holdout"] == small for call in spy2.similar_calls)


# ---- 2. replay with the stub LLM ----------------------------------------


def test_replay_with_stub_produces_report_of_right_shape(tmp_path):
    cases = [_case(f"run-{i:03d}", cited=["Server_Arch_v2.3.pdf"])
             for i in range(30)]
    store = CaseStore(tmp_path / "cases")
    _fill(store, cases)
    rag = FakeRag(["Server_Arch_v2.3.pdf", "Other.pdf"])
    report = evaluate(store, StubLLM(), rag, "stub")
    assert set(report) == {
        "created_at", "llm_ident", "n_verified", "n_holdout", "n_replayed",
        "holdout_ids", "verdict_accuracy", "ece", "mean_citation_support",
        "mean_retrieval_recall", "per_subsystem", "cases"}
    assert report["llm_ident"] == "stub"
    assert report["n_replayed"] == report["n_holdout"]
    assert report["n_verified"] == len(cases)
    assert 0 < report["n_replayed"] < len(cases)
    assert report["per_subsystem"]["memory"]["n"] == report["n_replayed"]
    # stub can't reason: unknown verdicts are misses -> accuracy 0
    assert report["verdict_accuracy"] == 0.0
    for row in report["cases"]:
        assert set(row) == {
            "run_id", "subsystem", "state", "verdict_hit", "confidence",
            "citation_support", "retrieval_recall", "retrieved_titles",
            "cited_titles"}
        assert row["state"] == "unknown"
        assert row["verdict_hit"] is False


def test_degraded_retrieval_yields_visible_recall_drop(tmp_path):
    cited = ["Server_Arch_v2.3.pdf", "Parts_Master_v1.4.pdf"]
    cases = [_case(f"run-{i:03d}", cited=cited) for i in range(30)]
    store = CaseStore(tmp_path / "cases")
    _fill(store, cases)

    good = evaluate(store, StubLLM(), FakeRag(cited), "stub")
    bad = evaluate(store, StubLLM(), FakeRag(["Unrelated_v9.pdf"]), "stub")
    # replay sees the same cases; only retrieval degraded
    assert good["n_replayed"] == bad["n_replayed"]
    assert good["mean_retrieval_recall"] > bad["mean_retrieval_recall"]
    assert bad["mean_retrieval_recall"] == 0.0


# ---- 3. baseline diff ----------------------------------------------------


def _eval_args(tmp_path, **kw) -> SimpleNamespace:
    base = {"cases": str(tmp_path / "cases"),
            "lib": str(tmp_path / "docs"),
            "llm": "stub",
            "holdout_frac": 0.25,
            "out": str(tmp_path / "eval_report.json"),
            "update_baseline": False,
            "tolerance": 0.05}
    base.update(kw)
    return SimpleNamespace(**base)


def _lib_dir(tmp_path) -> Path:
    from harness.docs.ingest.library import DocLibrary
    src = tmp_path / "src"
    src.mkdir()
    (src / "server_arch.md").write_text(
        "# Server Architecture\n\nECC errors on DIMM_A2 indicate a memory\n"
        "module fault requiring reseat.\n", encoding="utf-8")
    lib = tmp_path / "docs"
    DocLibrary(lib).add([src / "server_arch.md"])
    return lib


def test_eval_first_run_writes_baseline(tmp_path):
    from harness.operator.cli import run_eval
    store = CaseStore(tmp_path / "cases")
    _fill(store, [_case(f"run-{i:03d}") for i in range(30)])
    _lib_dir(tmp_path)
    assert run_eval(_eval_args(tmp_path)) == 0
    assert (tmp_path / "baseline.json").exists()


def test_tampered_baseline_exits_one_on_regression(tmp_path):
    from harness.operator.cli import run_eval
    store = CaseStore(tmp_path / "cases")
    _fill(store, [_case(f"run-{i:03d}") for i in range(30)])
    _lib_dir(tmp_path)
    args = _eval_args(tmp_path)
    assert run_eval(args) == 0  # first run writes the baseline
    # force a regression: the stub's true numbers are accuracy 0 / ECE ~ 1
    (tmp_path / "baseline.json").write_text(
        '{"verdict_accuracy": 1.0, "ece": 0.0, "n_replayed": 7}',
        encoding="utf-8")
    assert run_eval(_eval_args(tmp_path, tolerance=0.01)) == 1
    assert run_eval(_eval_args(tmp_path, update_baseline=True)) == 0


def test_eval_hard_errors_without_pipeline(tmp_path):
    from harness.operator.cli import run_eval
    store = CaseStore(tmp_path / "cases")
    _fill(store, [_case(f"run-{i:03d}") for i in range(30)])
    assert run_eval(_eval_args(tmp_path, lib=str(tmp_path / "missing_lib"))) == 1