"""Plan + diagnosis orchestration smoke test with a fake runner + fake LLM."""

from typing import ClassVar

from harness.diagnosis.engine import DiagnosticEngine, EngineContext
from harness.diagnosis.schema import Action, Diagnosis, Reference, Risk
from harness.engine.allowlist import AllowPolicy, AllowRule
from harness.engine.runner import Runner
from harness.inspect.decoder import Decoder
from harness.inspect.registry import make_collector
from harness.plan.profile import plan_collection
from harness.plan.subsystem import classify

FAKE_POLICY = AllowPolicy([
    AllowRule("/usr/bin/rdmsr", ("-a",)),
    AllowRule("/bin/dmesg", ("-l", "*")),
    AllowRule("/bin/dmidecode", ()),
    AllowRule("/usr/bin/lspci", ("-xxx",)),
])


class FakeRunner(Runner):
    """Return canned output per argv without touching the host."""

    OUTPUT: ClassVar[dict[str, str]] = {
        "/usr/bin/rdmsr -a": "IA32_MC0_STATUS = 0x8000000000000001\n",
        "/bin/dmesg -l err": "MCE: memory error on DIMM_A2\n",
        "/bin/dmidecode": "Product Name: model_x\nBIOS Vendor: Intel\nBIOS Version: 2.3\n",
        "/usr/bin/lspci -xxx": "00:1f.2 PCIe link down\n",
    }

    def __init__(self) -> None:
        super().__init__(FAKE_POLICY)

    def _exec(self, argv, timeout=30.0):
        from harness.engine.runner import CommandResult
        key = " ".join(argv)
        out = self.OUTPUT.get(key, "")
        return CommandResult(argv=argv, stdout=out, stderr="", exit_code=0, elapsed_ms=1)


def test_classify_memory_symptom():
    ranked = classify("machine check uncorrectable ECC error in DIMM")
    assert ranked[0].subsystem == "memory"


def test_plan_minimal_collectors_for_memory():
    plan = plan_collection("MCE uncorrectable ECC")
    assert plan.primary_subsystem == "memory"
    assert "cpu_msr" in plan.collectors


def test_end_to_end_diagnosis():
    runner = FakeRunner()
    engine = DiagnosticEngine(EngineContext(
        runner=runner,
        decoder=Decoder(),
        collector_factory=make_collector,
        llm=_fake_llm,
        scorer=lambda d, dump_sets: d,
    ))
    d = engine.run("MCE uncorrectable ECC error")
    assert d.schema_version == "1.0.0"
    assert d.actions and d.actions[0].risk == Risk.LOW
    assert any("Memory ECC" in a.rationale for a in d.actions)
    assert d.references  # must cite docs


def test_engine_skips_none_collectors_and_calls_hooks():
    runner = FakeRunner()
    seen_steps = []
    dump_sets_seen = {}

    def factory(name, _runner):
        if name == "ipmi":
            return None  # BMC channel unavailable -> collector skipped
        return make_collector(name, _runner)

    def parts_validate(dumps):
        dump_sets_seen.update(dumps)
        from harness.diagnosis.parts_validate import PartsCheckResult
        return PartsCheckResult(matches=["ok"], discrepancies=[])

    engine = DiagnosticEngine(EngineContext(
        runner=runner,
        decoder=Decoder(),
        collector_factory=factory,
        llm=_fake_llm,
        parts_validate=parts_validate,
        supervisor=lambda label: seen_steps.append(label),
    ))
    d = engine.run("MCE uncorrectable ECC error")
    assert "ipmi" not in dump_sets_seen
    assert seen_steps == ["plan", "collect", "decode", "retrieve", "reason"]
    assert d.parts_discrepancies == []


def _fake_llm(_prompt: str) -> Diagnosis:
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
            references=[],
        )],
        references=[Reference(source="Server_Arch_v2.3.pdf", page="78")],
    )