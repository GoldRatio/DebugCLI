"""Audit log hash-chain, session trace, and the scorer formula."""

import pytest

from harness.audit.auditlog import AuditLog
from harness.audit.redact import Redactor
from harness.audit.trace import SessionTrace
from harness.diagnosis.schema import Diagnosis, Reference
from harness.diagnosis.scorer import Scorer, apply_to, score_diagnosis


def test_audit_hash_chain_intact(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    log.append("s1", "cmd", {"argv": ["/bin/dmidecode"]})
    log.append("s1", "cmd", {"argv": ["/usr/bin/rdmsr", "-a"]})
    assert log.verify() == []


def test_audit_tamper_detected(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    log.append("s1", "cmd", {"argv": ["/bin/x"]})
    path = log.path
    # corrupt the first line
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[0] = '{"session_id": "s1", "seq": 1, "ts": "x", "kind": "cmd", "payload": {}, "prev_hash": "", "hash": "DEAD"}'
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert log.verify() != []


def test_redaction_scrubs_secret():
    r = Redactor(["hunter2secret"])
    assert "hunter2secret" not in r.redact("token=hunter2secret here")
    assert r.redact("nope") == "nope"


def test_redactor_scrubs_private_key_blocks(tmp_path):
    key = "-----BEGIN PRIVATE KEY-----\nblah\n-----END PRIVATE KEY-----"
    broad = Redactor()
    broad.add_secret(key[20:25])
    # Even without exact secret, KEY block scrubbed by static method
    assert "PRIVATE KEY" not in Redactor.scrub_ssh_keys(key)


def test_session_trace_records_commands():
    t = SessionTrace(session_id="run-9")
    t.record_command(["/bin/lsblk"], 0, 12)
    assert t.command_trace[0]["exit"] == 0


def test_scorer_formula_weights():
    b = Scorer().score(retrieval_citation_support=1.0, evidence_fit=1.0,
                       model_agreement=1.0, penalty=0.0)
    d = apply_to(Diagnosis(diagnosis="x", confidence=0.0), b)
    # 0.55 + 0.30 + 0.15 = 1.0
    assert d.confidence == 1.0


def test_scorer_penalty_deduction():
    b = Scorer().score(retrieval_citation_support=0.5, evidence_fit=0.5,
                       model_agreement=0.0, penalty=0.2)
    d = apply_to(Diagnosis(diagnosis="x", confidence=0.0), b)
    assert d.confidence <= 0.425 + 1e-9


def test_score_diagnosis_citation_support():
    diag = Diagnosis(diagnosis="x", confidence=0.0,
                     references=[Reference(source="Server_Arch_v2.3.pdf", page="78")])
    scored = score_diagnosis(diag, retrieved_snippets=[
        "[Server_Arch_v2.3.pdf p.78] IA32_MC0_STATUS: valid bit means error logged.",
    ])
    assert scored.confidence_breakdown.retrieval_citation_support == 1.0


def test_score_diagnosis_citation_not_retrieved():
    diag = Diagnosis(diagnosis="x", confidence=0.0,
                     references=[Reference(source="Server_Arch_v2.3.pdf", page="78")])
    scored = score_diagnosis(diag, retrieved_snippets=["[Other.pdf p.1] unrelated"])
    assert scored.confidence_breakdown.retrieval_citation_support == 0.0


def test_score_diagnosis_evidence_fit_from_known_registers():
    diag = Diagnosis(diagnosis="x", confidence=0.0,
                     evidence=[{"mnemonic": "IA32_MC0_STATUS", "unknown": False},
                               {"mnemonic": "IA32_MCG_STATUS", "unknown": True}])
    scored = score_diagnosis(diag, retrieved_snippets=None)
    assert scored.confidence_breakdown.evidence_fit == 0.5


def test_score_diagnosis_penalties():
    diag = Diagnosis(diagnosis="x", confidence=0.0, unknown_registers=["R1", "R2"],
                     evidence=[{"unknown": True}])
    scored = score_diagnosis(diag, retrieved_snippets=None)
    # 0.3 evidence_fit fallback * 0.30 + 0.15*0.5 - (0.2 + 0.05)
    assert scored.confidence_breakdown.penalty == pytest.approx(0.25)
    assert scored.confidence >= 0.0


def test_evidence_fit_from_dumps_measures_useful_probes():
    from harness.diagnosis.scorer import evidence_fit_from_dumps
    from harness.inspect.base import RegisterDump

    def dump(raw, ok=True):
        return RegisterDump(subsystem="bmc", source="s", raw=raw,
                            cmd_argv=["x"], ok=ok,
                            meta={"exit": 0 if ok else 1, "elapsed_ms": 1})

    sets = {"kernel": [
        dump("CPU0 Temp | 45.0 | ok"),       # usable
        dump(""),                             # empty output
        dump("failed", ok=False),            # failed probe
    ]}
    assert evidence_fit_from_dumps(sets) == pytest.approx(1 / 3, abs=0.001)
    assert evidence_fit_from_dumps({}) == 0.0


def test_evidence_fit_from_dumps_all_ok_is_one():
    from harness.diagnosis.scorer import evidence_fit_from_dumps
    from harness.inspect.base import RegisterDump

    sets = {"ipmi": [
        RegisterDump(subsystem="bmc", source="s", raw="sensor list output",
                     cmd_argv=["x"], ok=True,
                     meta={"exit": 0, "elapsed_ms": 1})
        for _ in range(3)
    ]}
    assert evidence_fit_from_dumps(sets) == 1.0