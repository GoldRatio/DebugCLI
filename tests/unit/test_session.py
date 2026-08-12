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


class FakeConsoleRunner(Runner):
    """Console-target runner with canned BMC-shell outputs."""

    OUTPUT: ClassVar[dict[str, str]] = {
        "sudo -S ipmitool sensor list": (
            "Power_Status     | 0x0        | discrete   | 0x0180| na | na\n"
        ),
        "sudo -S ipmitool sel list": (
            "  1 | 08/10/26 | 21:30:38 UTC | System Event #0x07 | Timestamp Clock Sync | Asserted\n"
        ),
        "sudo -S ipmitool fru print": "Product Name          : C4A15\n",
        "sudo -S i2cdump -y 8 0xb": (
            "     0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f\n"
            "1b: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00\n"
            "a1: 05 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00\n"
        ),
        "dmesg -r": "<6>Booting Linux ... Machine model: C4A15 BMC\n",
    }

    def __init__(self) -> None:
        super().__init__(AllowPolicy([
            AllowRule("sudo -S ipmitool sensor list"),
            AllowRule("sudo -S ipmitool sel list"),
            AllowRule("sudo -S ipmitool fru print"),
            AllowRule("sudo -S i2cdump -y 8 0xb"),
            AllowRule("dmesg -r"),
        ]))
        self.is_console = True

    def _exec(self, argv, timeout=30.0):
        return CommandResult(argv=list(argv), stdout=self.OUTPUT.get(" ".join(argv), ""),
                             stderr="", exit_code=0, elapsed_ms=1)


def _console_factory(name, _runner):
    from harness.inspect.collectors.bmc_console import BmcConsoleCollector
    if name in ("cpu_msr", "kernel", "ipmi"):
        return BmcConsoleCollector(_runner, subsystem={
            "cpu_msr": "cpu", "kernel": "kernel", "ipmi": "ipmi",
        }[name])
    return None


def _console_ctx(runner) -> EngineContext:
    return EngineContext(
        runner=runner,
        decoder=Decoder(),
        collector_factory=_console_factory,
        llm=_stub_diagnosis,
        supervisor=lambda label: None,
    )


GB_HANGUP_I2C = (
    "[GB_HangUp_troubleshooting_v1.2.pdf p.1] For EVERY Amber Light issue, "
    'please dump "i2cdump -y 8 0xb" in the BMC to get the boot state.'
)


def test_session_initial_doc_probe_runs_and_decodes():
    runner = FakeConsoleRunner()
    ctx = _console_ctx(runner)
    ctx.docs_retriever = lambda q: [GB_HANGUP_I2C]
    llm = ScriptedLLM({"kind": "diagnosis", "diagnosis": _stub_diagnosis().model_dump()})
    engine = SessionEngine(ctx, llm=llm)
    diag = engine.run("amber light, server stuck no boot")
    assert any(c.ok and "i2cdump -y 8 0xb" in " ".join(c.argv) for c in runner.calls)
    mnemonics = {e["mnemonic"] for e in diag.evidence}
    assert "CPLD_1B_CRITICAL" in mnemonics
    assert "CPLD_A1_BOOT_STATE" in mnemonics


def test_session_doc_topic_mines_new_probes():
    runner = FakeConsoleRunner()
    runner.policy.add(AllowRule("sudo -S i2cdump -y 9 0xb"))
    runner.OUTPUT = dict(FakeConsoleRunner.OUTPUT)
    runner.OUTPUT["sudo -S i2cdump -y 9 0xb"] = (
        "     0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f\n"
        "1b: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00\n"
        "a1: 05 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00\n"
    )
    ctx = _console_ctx(runner)
    ctx.docs_retriever = lambda q: [
        ("[GB_HangUp_troubleshooting_v1.2.pdf p.1] "
         'dump "i2cdump -y 9 0xb" to get the boot state.'),
    ]
    llm = ScriptedLLM(
        {"kind": "probe", "subsystems": [], "doc_topics": ["amber light"]},
        {"kind": "diagnosis", "diagnosis": _stub_diagnosis().model_dump()},
    )
    engine = SessionEngine(ctx, llm=llm, max_turns=3)
    diag = engine.run("amber light, server stuck no boot")
    assert any(c.ok and "i2cdump -y 9 0xb" in " ".join(c.argv) for c in runner.calls)
    mnemonics = {e["mnemonic"] for e in diag.evidence}
    assert "CPLD_1B_CRITICAL" in mnemonics
