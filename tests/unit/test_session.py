"""Multi-turn session engine: question -> answer -> probe -> diagnosis loop."""

from typing import ClassVar

import pytest

from harness.diagnosis.engine import EngineContext
from harness.diagnosis.llm import StubLLM
from harness.diagnosis.schema import Action, Diagnosis, Risk
from harness.diagnosis.session import SessionEngine, SessionError
from harness.engine.allowlist import AllowPolicy, AllowRule
from harness.engine.runner import CommandResult, Runner
from harness.inspect.decoder import Decoder
from harness.inspect.registry import make_collector

FAKE_POLICY = AllowPolicy([
    AllowRule("/usr/bin/rdmsr", ("-a",)),
    AllowRule("/bin/dmesg", ("-l", "*")),
    AllowRule("/bin/dmidecode", ()),
    AllowRule("/usr/bin/lspci", ("-xxx",)),
])


class FakeRunner(Runner):
    OUTPUT: ClassVar[dict[str, str]] = {
        "/usr/bin/rdmsr -a": "IA32_MC0_STATUS = 0x8000000000000001\n",
        "/bin/dmesg -l err": "MCE: memory error on DIMM_A2\n",
        "/bin/dmesg -l err, crit, alert, emerg": "MCE: memory error on DIMM_A2\n",
        "/bin/dmidecode": "Product Name: model_x\nBIOS Vendor: Intel\nBIOS Version: 2.3\n",
        "/usr/bin/lspci -xxx": "00:1f.2 PCIe link down\n",
    }

    def __init__(self) -> None:
        super().__init__(FAKE_POLICY)

    def _exec(self, argv, timeout=30.0):
        return CommandResult(argv=list(argv), stdout=self.OUTPUT.get(" ".join(argv), ""),
                             stderr="", exit_code=0, elapsed_ms=1)


class ScriptedLLM:
    """Returns canned chat_json answers in order; records the messages it saw."""

    def __init__(self, *answers: dict):
        self.answers = list(answers)
        self.calls: list[list[dict]] = []

    def chat_json(self, messages: list[dict]) -> dict:
        self.calls.append(messages)
        if not self.answers:
            return {"kind": "diagnosis", "diagnosis": _stub_diagnosis().model_dump()}
        return self.answers.pop(0)


def _stub_diagnosis() -> Diagnosis:
    return Diagnosis(
        diagnosis="Memory ECC errors on DIMM_A2",
        confidence=0.0,
        actions=[Action(
            step=1,
            action="Reseat DIMM in slot A2",
            rationale="Memory ECC uncorrectable errors; architecture doc page 78",
            risk=Risk.LOW,
            required_tool="Physical access",
            impact="requires reboot",
        )],
    )


def _ctx(dump_sets_seen=None, snippets=False) -> EngineContext:
    return EngineContext(
        runner=FakeRunner(),
        decoder=Decoder(),
        collector_factory=make_collector,
        llm=_stub_diagnosis,  # unused by the session loop itself
        docs_retriever=(lambda q: [f"[doc p.3] {q} related section"]) if snippets else None,
        supervisor=lambda label: None,
        dump_callback=(lambda dumps: dump_sets_seen.update(dumps))
        if dump_sets_seen is not None else None,
    )


def test_session_question_answer_then_diagnosis():
    llm = ScriptedLLM(
        {"kind": "question", "question": "What previous repair actions were attempted?"},
        {"kind": "diagnosis", "diagnosis": _stub_diagnosis().model_dump()},
    )
    answers = []
    engine = SessionEngine(_ctx(), llm=llm,
                           human_input=lambda q: answers.append(q) or "DIMM A2 reseated, no change")
    diag = engine.run("MCE uncorrectable ECC error")

    assert diag.diagnosis == "Memory ECC errors on DIMM_A2"
    assert diag.evidence  # decoded registers attached
    assert answers == ["What previous repair actions were attempted?"]
    kinds = [t["kind"] for t in engine.transcript]
    assert kinds == ["question", "answer", "diagnosis"]
    # the human answer made it into the next LLM call
    assert any("reseated" in m["content"] for m in llm.calls[1])
    assert llm.calls[1][0]["role"] == "system"  # system preamble present


def test_session_probe_requests_collect_only_known_subsystems():
    dump_sets = {}
    llm = ScriptedLLM(
        {"kind": "probe", "subsystems": ["pcie", "bogus"], "doc_topics": ["PCIe link recovery"]},
        {"kind": "diagnosis", "diagnosis": _stub_diagnosis().model_dump()},
    )
    engine = SessionEngine(_ctx(dump_sets, snippets=True), llm=llm, max_turns=4)
    diag = engine.run("MCE uncorrectable ECC error")

    assert "pcie" in dump_sets  # requested subsystem collected
    assert "cpu_msr" in dump_sets  # initial plan for the memory symptom
    assert any("unknown subsystem 'bogus'" in t["content"] for t in engine.transcript)
    assert any("PCIe link recovery" in t["content"] for t in engine.transcript)
    # doc topic retrieved and surfaced in the next turn's evidence
    assert any("PCIe link recovery related section" in m["content"] for m in llm.calls[1])
    assert diag.actions[0].risk == Risk.LOW


def test_session_second_probe_of_same_subsystem_does_not_recollect():
    llm = ScriptedLLM(
        {"kind": "probe", "subsystems": ["memory"]},
        {"kind": "probe", "subsystems": ["memory"]},
        {"kind": "diagnosis", "diagnosis": _stub_diagnosis().model_dump()},
    )
    engine = SessionEngine(_ctx(), llm=llm, max_turns=4)
    engine.run("MCE uncorrectable ECC error")
    probe_results = [t["content"] for t in engine.transcript if t["kind"] == "probe"]
    assert "already held" in probe_results[1]


def test_session_stub_llm_asks_then_diagnoses():
    engine = SessionEngine(_ctx(), llm=StubLLM(),
                           human_input=lambda q: "we reseated the DIMM")
    diag = engine.run("MCE uncorrectable ECC error")
    assert diag.diagnosis  # stub text
    kinds = [t["kind"] for t in engine.transcript]
    assert kinds == ["question", "answer", "diagnosis"]
    assert any("reseated" in t["content"] for t in engine.transcript)


def test_session_malformed_and_question_loops_bounded_then_forced():
    class NeverDiagnoses:
        def chat_json(self, messages):
            return {"kind": "question", "question": "more info?"}

    engine = SessionEngine(_ctx(), llm=NeverDiagnoses(), max_turns=2)
    with pytest.raises(SessionError):
        engine.run("MCE uncorrectable ECC error")
    assert len(engine.transcript) == 4  # 2 questions + 2 "(no answer)"

    class Malformed:
        def chat_json(self, messages):
            return {"kind": "nonsense"}

    engine = SessionEngine(_ctx(), llm=Malformed(), max_turns=3)
    with pytest.raises(SessionError):
        engine.run("MCE uncorrectable ECC error")
    assert any("malformed" in n for n in engine.notes)


def test_session_requires_chat_json_llm():
    with pytest.raises(SessionError):
        SessionEngine(_ctx(), llm=object())


def test_session_context_seeded_before_first_turn():
    llm = ScriptedLLM({"kind": "diagnosis", "diagnosis": _stub_diagnosis().model_dump()})
    engine = SessionEngine(_ctx(), llm=llm)
    engine.run("MCE uncorrectable ECC error",
               initial_answers=["PSU was replaced last week, no change"])
    assert any(t["kind"] == "context" and "PSU was replaced" in t["content"]
               for t in engine.transcript)
    assert any("PSU was replaced" in m["content"] for m in llm.calls[0])
