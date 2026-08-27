"""03/04 integration: CaseOutcome recording, prior-case prompt injection, engine wiring.

The case store/library modules themselves are implemented by the reverse agent;
these tests cover the forward contracts that consume them: schema, Verifier.record,
prompt rendering, scorer guard, and engine hook-up.
"""

from harness.diagnosis.case_library import CaseLibrary, render
from harness.diagnosis.case_store import CaseStore, evidence_hash
from harness.diagnosis.engine import DiagnosticEngine, EngineContext
from harness.diagnosis.llm import StubLLM
from harness.diagnosis.prompt import SYSTEM_PREAMBLE, build_prompt
from harness.diagnosis.schema import CaseOutcome, Diagnosis, Reference
from harness.diagnosis.scorer import score_diagnosis
from harness.diagnosis.summarize import EvidenceSummary
from harness.diagnosis.verifier import Verifier
from harness.engine.allowlist import AllowPolicy, AllowRule
from harness.engine.runner import Runner
from harness.inspect.decoder import Decoder
from harness.inspect.model import from_alias


def _case(run_id: str, **kw) -> CaseOutcome:
    defaults = {
        "run_id": run_id, "target_id": "target-1",
        "symptom": "ECC errors on DIMM_A2",
        "model_key": "poweredge_r650", "outcome": "fixed",
        "actions_recommended": ["1. Reseat DIMM A2"],
        "actions_taken": ["reseated DIMM A2"],
        "llm_ident": "stub",
        "evidence_summary": ["MCE: memory error on DIMM_A2"],
    }
    defaults.update(kw)
    return CaseOutcome(**defaults)


def test_verifier_record_persists_and_is_worm(tmp_path):
    store = CaseStore(tmp_path / "cases")
    case = _case("run-1")
    Verifier().record(case, store)
    stored = store.get("run-1")
    assert stored is not None and stored.outcome == "fixed"
    assert stored.created_at  # store stamped the timestamp
    assert stored.evidence_hash == evidence_hash(case.evidence_summary)


def test_verifier_record_refuses_overwrite(tmp_path):
    store = CaseStore(tmp_path / "cases")
    Verifier().record(_case("run-1"), store)
    try:
        Verifier().record(_case("run-1", outcome="not_fixed"), store)
    except FileExistsError:
        pass
    else:
        raise AssertionError("second record must be refused (append-only)")
    assert store.get("run-1").outcome == "fixed"


def test_case_store_revise_replaces_and_audits(tmp_path):
    from harness.audit.auditlog import AuditLog
    from harness.audit.redact import Redactor
    log = AuditLog(tmp_path / "audit.jsonl", Redactor([]))
    store = CaseStore(tmp_path / "cases")
    store.record(_case("run-1"), audit=log, session_id="run-1")
    store.record(_case("run-1", outcome="not_fixed",
                       actions_taken=["replaced DIMM A2"]),
                 audit=log, session_id="run-1", revise=True)
    stored = store.get("run-1")
    assert stored.outcome == "not_fixed"
    assert stored.actions_taken == ["replaced DIMM A2"]
    events = log.read()
    assert [e.kind for e in events] == ["case_record", "case_revised"]
    revised = events[-1].payload
    assert revised["prev_outcome"] == "fixed"
    assert revised["outcome"] == "not_fixed"
    assert log.verify() == []


def test_case_store_revise_without_existing_is_plain_record(tmp_path):
    from harness.audit.auditlog import AuditLog
    from harness.audit.redact import Redactor
    log = AuditLog(tmp_path / "audit.jsonl", Redactor([]))
    store = CaseStore(tmp_path / "cases")
    store.record(_case("run-9"), audit=log, session_id="run-9", revise=True)
    assert [e.kind for e in log.read()] == ["case_record"]


def test_verifier_record_audit_links(tmp_path):
    from harness.audit.auditlog import AuditLog
    from harness.audit.redact import Redactor
    log = AuditLog(tmp_path / "audit.jsonl", Redactor([]))
    Verifier().record(_case("run-a"), CaseStore(tmp_path / "cases"),
                      audit=log, session_id="run-a")
    events = log.read()
    case_events = [e for e in events if e.kind == "case_record"]
    assert len(case_events) == 1
    assert case_events[0].payload["run_id"] == "run-a"
    assert log.verify() == []


def test_prompt_renders_prior_cases_section():
    lines = render([(_case("r1"), 1.0), (_case("r2", model_key=None,
                                               actions_taken=[]), 0.5)])
    p = build_prompt(
        model=from_alias("poweredge_r650"),
        decoded=[],
        summaries=EvidenceSummary(interesting=[], anomaly_count=0, total=0),
        doc_snippets=["[doc p.1] a snippet"],
        parts_refs=[],
        symptom="ECC",
        prior_cases=lines,
    )
    assert "## Prior Verified Cases (fleet history, NOT vendor documentation)" in p
    assert "[case r1] model=poweredge_r650 outcome=fixed:" in p
    assert "n/a" in p  # no action taken -> n/a
    assert "Prior Verified Cases are observed history" in SYSTEM_PREAMBLE
    assert "CONTRA" not in p  # no placeholder text


def test_scorer_does_not_count_prior_case_citations():
    diag = Diagnosis(
        state="fault", diagnosis="test",
        confidence=0.0,
        references=[Reference(source="Prior Verified Cases", page="1")],
    )
    scored = score_diagnosis(
        diag,
        retrieved_snippets=["[Prior Verified Cases p.1] ECC fixed by reseat"],
    )
    assert scored.confidence_breakdown.retrieval_citation_support == 0.0


class _FleetRunner(Runner):
    """Detection succeeds with Product Name -> poweredge_r650."""

    def __init__(self) -> None:
        super().__init__(AllowPolicy([AllowRule("/bin/dmidecode", ())]))
        self._count = 0

    def _exec(self, argv, timeout=30.0):
        from harness.engine.runner import CommandResult
        self._count += 1
        out = ("Product Name: PowerEdge R650\n"
               "BIOS Vendor: Dell\nBIOS Version: 2.3\n")
        return CommandResult(argv=argv, stdout=out, stderr="",
                             exit_code=0, elapsed_ms=1)


def test_engine_injects_prior_cases_at_diagnosis_time():
    calls = []
    prompts = []

    def case_library(symptom, model_key):
        calls.append((symptom, model_key))
        return ["[case r1] model=poweredge_r650 outcome=fixed: ECC -> reseated"]

    def llm(prompt):
        prompts.append(prompt)
        return StubLLM()(prompt)

    engine = DiagnosticEngine(EngineContext(
        runner=_FleetRunner(),
        decoder=Decoder(),
        collector_factory=lambda name, _r: None,
        llm=llm,
        docs_retriever=lambda q, _k: [],
        case_library=case_library,
    ))
    engine.run("ECC errors")
    assert calls and calls[0][1] == "poweredge_r650"
    assert any("Prior Verified Cases" in p for p in prompts)
    assert any("[case r1] model=poweredge_r650" in p for p in prompts)


def test_engine_wires_test_log_evidence_rag_and_case_query():
    prompts = []
    retrieved = []
    case_calls = []

    def llm(prompt):
        prompts.append(prompt)
        return StubLLM()(prompt)

    engine = DiagnosticEngine(EngineContext(
        runner=_FleetRunner(),
        decoder=Decoder(),
        collector_factory=lambda name, _r: None,
        llm=llm,
        docs_retriever=lambda q, _k: retrieved.append(q) or [f"[doc] {q}"],
        case_library=lambda q, _k: case_calls.append(q) or [],
        test_log_lines=["source=fat.log", "P02002001@PCIe Test Fail",
                        "test=pcie_cmp_chk"],
        test_log_queries=["PCIe Test Fail P02002001"],
        test_log_case_terms=["P02002001@PCIe Test Fail"],
    ))
    engine.run("PCIe compare failure")
    # dedicated prompt section with the failure identity
    assert any("## Factory Test Log Evidence" in p for p in prompts)
    assert any("P02002001@PCIe Test Fail" in p for p in prompts)
    # failure-derived retrieval + case-library queries used
    assert any("PCIe Test Fail P02002001" in q for q in retrieved)
    assert case_calls and "P02002001@PCIe Test Fail" in case_calls[0]


def test_case_library_ranking_outcome_and_model(tmp_path):
    store = CaseStore(tmp_path / "cases")
    Verifier().record(_case("fixed-1",
                            symptom="ECC errors DIMM", outcome="fixed"), store)
    Verifier().record(_case("failed-1",
                            symptom="ECC errors DIMM", outcome="not_fixed",
                            actions_taken=["reseated DIMM"]), store)
    lib = CaseLibrary(store)
    records = lib.similar("ECC errors DIMM", model_key="poweredge_r650", top_k=2)
    assert records[0][0].run_id == "fixed-1"
    cross_model = lib.similar("ECC errors DIMM", model_key="poweredge_r750",
                              top_k=2)
    # same-model boost still ranks same-key cases first even with fewer terms
    assert cross_model[0][0].run_id == "fixed-1"
    blocked = lib.similar("ECC errors DIMM", outcome_min=0.5, top_k=10)
    assert all(c.outcome != "not_fixed" for c, _ in blocked)


def _fat_log_case(run_id, **kw):
    return _case(run_id,
                 symptom="Factory test log failure",
                 test_log_failures=["P02002001@PCIe Test Fail", "pcie_cmp_chk"],
                 **kw)


def test_case_carries_test_log_failures(tmp_path):
    store = CaseStore(tmp_path / "cases")
    case = _fat_log_case("pcie-1", outcome="fixed",
                         actions_taken=["replaced NVMe backplane"])
    Verifier().record(case, store)
    stored = store.get("pcie-1")
    assert stored.test_log_failures == ["P02002001@PCIe Test Fail", "pcie_cmp_chk"]


def test_case_library_matches_by_failure_code(tmp_path):
    store = CaseStore(tmp_path / "cases")
    Verifier().record(_fat_log_case("pcie-1", outcome="fixed",
                                    actions_taken=["replaced backplane"]), store)
    Verifier().record(_case("ecc-1", outcome="fixed",
                            symptom="ECC errors"), store)
    lib = CaseLibrary(store)
    # A fresh run whose test log shows the same failure code surfaces the case
    # pre-probe, even with a generic symptom.
    records = lib.similar("Factory test log failure P02002001@PCIe Test Fail",
                          model_key="t6t", top_k=5)
    assert records[0][0].run_id == "pcie-1"
    # Near-match on the test name alone also surfaces it.
    near = lib.similar("Factory test log failure pcie_cmp_chk",
                       model_key="t6t", top_k=5)
    assert near[0][0].run_id == "pcie-1"


def test_render_shows_test_log_failure_identity():
    case = _fat_log_case("pcie-1", outcome="fixed",
                         actions_taken=["replaced NVMe backplane"])
    lines = render([(case, 1.0)])
    assert "P02002001@PCIe Test Fail" in lines[0]
    assert "replaced NVMe backplane" in lines[0]


def test_case_text_indexes_test_log_failures():
    from harness.diagnosis.case_library import _case_text
    text = _case_text(_fat_log_case("pcie-1"))
    assert "P02002001@PCIe Test Fail" in text
    assert "pcie_cmp_chk" in text