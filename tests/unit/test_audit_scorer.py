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


def test_score_diagnosis_no_doc_refs_renormalizes_weights():
    # A healthy verdict cites no documents: the citation component must not
    # deflate confidence; it is dropped and weights renormalized.
    diag = Diagnosis(diagnosis="Server healthy; all sensors nominal.",
                     confidence=0.0)
    scored = score_diagnosis(diag, retrieved_snippets=["[Other.pdf p.1] unrelated"],
                             evidence_fit=1.0, model_agreement=0.5)
    # (0.30*1.0 + 0.15*0.5) / 0.45 = 0.833 - 0.05 (empty actions) = 0.783
    assert scored.confidence == pytest.approx(0.783, abs=0.001)


def test_score_diagnosis_prompt_section_citations_are_not_doc_refs():
    # Citing "Evidence Notes" (an in-prompt section) is not a document citation
    # and must not be counted as an unsupported reference.
    diag = Diagnosis(diagnosis="x", confidence=0.0,
                     references=[Reference(source="Evidence Notes", page=None)])
    scored = score_diagnosis(diag, retrieved_snippets=["[Other.pdf p.1] unrelated"],
                             evidence_fit=1.0, model_agreement=0.5)
    assert scored.confidence == pytest.approx(0.783, abs=0.001)


def test_score_diagnosis_no_snippets_drops_citation_weight():
    # No doc library / no retrieval this run: citations cannot be verified,
    # so the citation component must not floor the confidence.
    diag = Diagnosis(diagnosis="x", confidence=0.0,
                     references=[Reference(source="Server_Arch_v2.3.pdf", page="78")])
    scored = score_diagnosis(diag, retrieved_snippets=None,
                             evidence_fit=1.0, model_agreement=0.5)
    assert scored.confidence == pytest.approx(0.783, abs=0.001)


def test_score_diagnosis_unsupported_doc_ref_still_penalized():
    # Real document references that fail verification keep the full weight.
    diag = Diagnosis(diagnosis="x", confidence=0.0,
                     references=[Reference(source="Server_Arch_v2.3.pdf", page="78")])
    scored = score_diagnosis(diag, retrieved_snippets=["[Other.pdf p.1] unrelated"],
                             evidence_fit=1.0, model_agreement=0.5)
    b = scored.confidence_breakdown
    # 0.55*0.0 + 0.30*1.0 + 0.15*0.5 = 0.375 - 0.05 (empty actions) = 0.325
    assert b.retrieval_citation_support == 0.0
    assert scored.confidence == pytest.approx(0.325, abs=0.001)


def test_score_diagnosis_evidence_fit_from_known_registers():
    diag = Diagnosis(diagnosis="x", confidence=0.0,
                     evidence=[{"mnemonic": "IA32_MC0_STATUS", "unknown": False},
                               {"mnemonic": "IA32_MCG_STATUS", "unknown": True}])
    scored = score_diagnosis(diag, retrieved_snippets=None)
    assert scored.confidence_breakdown.evidence_fit == 0.5


def test_score_diagnosis_unknown_registers_not_penalized():
    diag = Diagnosis(diagnosis="x", confidence=0.0, unknown_registers=["R1", "R2"],
                     evidence=[{"unknown": True}])
    scored = score_diagnosis(diag, retrieved_snippets=None)
    # Unknown registers are a catalog gap, not a diagnosis error: only the
    # empty-actions penalty (0.05) applies now.
    assert scored.confidence_breakdown.penalty == pytest.approx(0.05)
    assert scored.confidence >= 0.0


def test_evidence_fit_from_dumps_falls_back_to_probe_usefulness():
    from harness.diagnosis.scorer import evidence_fit_from_dumps
    from harness.inspect.base import RegisterDump

    def dump(raw, ok=True):
        return RegisterDump(subsystem="bmc", source="s", raw=raw,
                            cmd_argv=["x"], ok=ok,
                            meta={"exit": 0 if ok else 1, "elapsed_ms": 1})

    # No sensor-kind dumps -> claim unknown -> fall back to probe usefulness.
    sets = {"kernel": [
        dump("CPU0 Temp | 45.0 | ok"),       # usable
        dump(""),                             # empty output
        dump("failed", ok=False),            # failed probe
    ]}
    assert evidence_fit_from_dumps(None, sets) == pytest.approx(1 / 3, abs=0.001)
    assert evidence_fit_from_dumps(None, {}) == 0.0


def _sensor_dump(raw, kind="sensor"):
    from harness.inspect.base import RegisterDump
    return RegisterDump(subsystem="bmc", source=f"ipmitool {kind} list", raw=raw,
                        cmd_argv=["x"], ok=True,
                        meta={"exit": 0, "elapsed_ms": 1, "kind": kind})


SENSORS_OK = (
    "P12V_SCM_VOLT | 11.985 | Volts | ok | na | na\n"
    "HSC0_INPUT_VOLT | 51.345 | Volts | ok | na | na\n"
    "Power_Status | 0x0 | discrete | 0x0180 | na | na\n")
SENSOR_BAD = (
    "P12V_SCM_VOLT | 11.985 | Volts | ok | na | na\n"
    "CPU0_TEMP | 95.000 | degrees C | ucr | na | na\n")


def test_evidence_fit_claim_problem_with_all_ok_sensors_is_penalized():
    from harness.diagnosis.scorer import evidence_fit_from_dumps
    sets = {"ipmi": [_sensor_dump(SENSORS_OK)]}
    text = ("The server has experienced power supply failures per the SEL "
            "event log.")
    assert evidence_fit_from_dumps(text, sets) == pytest.approx(0.3)


def test_evidence_fit_claim_problem_with_sensor_fault_is_one():
    from harness.diagnosis.scorer import evidence_fit_from_dumps
    sets = {"ipmi": [_sensor_dump(SENSOR_BAD)]}
    assert evidence_fit_from_dumps("Power supply failure", sets) == 1.0


def test_evidence_fit_claim_healthy_matches_ok_sensors():
    from harness.diagnosis.scorer import evidence_fit_from_dumps
    sets = {"ipmi": [_sensor_dump(SENSORS_OK)]}
    text = ("No active issue found; all sensors nominal, server healthy.")
    assert evidence_fit_from_dumps(text, sets) == 1.0


def test_evidence_fit_claim_healthy_but_sensor_fault_is_penalized():
    from harness.diagnosis.scorer import evidence_fit_from_dumps
    sets = {"ipmi": [_sensor_dump(SENSOR_BAD)]}
    text = ("No active issue found; server is healthy.")
    assert evidence_fit_from_dumps(text, sets) == pytest.approx(0.3)


def _diagnosis(state=None, text="server diagnosed"):
    from harness.diagnosis.schema import Diagnosis, ServerState
    return Diagnosis(diagnosis=text, confidence=0.0,
                     state=ServerState(state) if state else ServerState.UNKNOWN)


def test_evidence_fit_structured_verdict_healthy_matches_ok_sensors():
    from harness.diagnosis.scorer import evidence_fit_from_dumps
    sets = {"ipmi": [_sensor_dump(SENSORS_OK)]}
    diag = _diagnosis("healthy",
                      "All live sensors nominal; SEL shows only historical "
                      "power supply entries from a past fault, server is fixed.")
    assert evidence_fit_from_dumps(diag, sets) == 1.0


def test_evidence_fit_structured_verdict_fault_contradicts_ok_sensors():
    from harness.diagnosis.scorer import evidence_fit_from_dumps
    sets = {"ipmi": [_sensor_dump(SENSORS_OK)]}
    diag = _diagnosis("fault", "Power supply failure detected in SEL.")
    assert evidence_fit_from_dumps(diag, sets) == pytest.approx(0.3)


def test_evidence_fit_structured_verdict_fault_confirms_sensor_fault():
    from harness.diagnosis.scorer import evidence_fit_from_dumps
    sets = {"ipmi": [_sensor_dump(SENSOR_BAD)]}
    diag = _diagnosis("fault", "CPU temperature above upper critical.")
    assert evidence_fit_from_dumps(diag, sets) == 1.0


def test_evidence_fit_structured_verdict_degraded_with_ok_sensors_is_partial():
    from harness.diagnosis.scorer import evidence_fit_from_dumps
    sets = {"ipmi": [_sensor_dump(SENSORS_OK)]}
    diag = _diagnosis("degraded", "Suspected marginal PSU not visible on sensors.")
    assert evidence_fit_from_dumps(diag, sets) == pytest.approx(0.6)


def test_evidence_fit_structured_verdict_healthy_but_sensor_fault_penalized():
    from harness.diagnosis.scorer import evidence_fit_from_dumps
    sets = {"ipmi": [_sensor_dump(SENSOR_BAD)]}
    diag = _diagnosis("healthy", "Server is healthy.")
    assert evidence_fit_from_dumps(diag, sets) == pytest.approx(0.3)


def test_evidence_fit_discrete_sensor_state_codes_are_not_faults():
    from harness.diagnosis.scorer import evidence_fit_from_dumps
    sets = {"ipmi": [_sensor_dump(SENSORS_OK)]}
    # 0x0180 state code must be treated as ok, not a fault
    assert evidence_fit_from_dumps("power supply failure", sets) == pytest.approx(0.3)