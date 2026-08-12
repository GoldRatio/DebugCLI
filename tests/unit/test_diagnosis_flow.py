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
    assert d.schema_version == "1.1.0"
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
    assert seen_steps == ["retrieve", "plan", "collect", "decode", "reason"]
    assert d.parts_discrepancies == []


def test_engine_prompt_callback_receives_built_prompt():
    runner = FakeRunner()
    seen = []

    engine = DiagnosticEngine(EngineContext(
        runner=runner,
        decoder=Decoder(),
        collector_factory=make_collector,
        llm=_fake_llm,
        prompt_callback=seen.append,
    ))
    engine.run("MCE uncorrectable ECC error")
    assert len(seen) == 1
    prompt = seen[0]
    assert "MCE uncorrectable ECC error" in prompt       # symptom present
    assert "## Anomalous Evidence Summary" in prompt     # evidence block present
    assert "## Evidence Notes" in prompt                 # kind notes present


class FakeConsoleRunner(Runner):
    """Console-target runner: canned BMC-shell outputs keyed by command."""

    OUTPUT: ClassVar[dict[str, str]] = {
        "sudo -S ipmitool sensor list": (
            "FANS_SWB_PWM     | 32.000     | percent    | ok    | na | na\n"
            "Power_Status     | 0x0        | discrete   | 0x0180| na | na\n"
        ),
        "sudo -S ipmitool sel list": "  1 | 08/10/26 | 21:30:38 UTC | System Event #0x07 | Timestamp Clock Sync | Asserted\n",
        "dmesg -r": "<6>Booting Linux ... Machine model: Microsoft DC-SCM 1.2 C4A15 BMC Device Tree\n",
        "sudo -S ipmitool fru print": "Product Name          : C4A15\nProduct Manufacturer  : Microsoft\n",
        "sudo -S i2cdump -y 8 0xb": (
            "     0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f\n"
            "0b: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00\n"
            "1b: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00\n"
            "a1: 05 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00\n"
        ),
    }

    def __init__(self) -> None:
        from harness.engine.allowlist import AllowPolicy, AllowRule
        super().__init__(AllowPolicy([
            AllowRule("sudo -S ipmitool sensor list"),
            AllowRule("sudo -S ipmitool sel list"),
            AllowRule("sudo -S ipmitool fru print"),
            AllowRule("sudo -S i2cdump -y 8 0xb"),
            AllowRule("dmesg -r"),
        ]))
        self.is_console = True

    def _exec(self, argv, timeout=30.0):
        from harness.engine.runner import CommandResult
        key = " ".join(argv)
        out = self.OUTPUT.get(key, "")
        return CommandResult(argv=argv, stdout=out, stderr="", exit_code=0, elapsed_ms=1)


def test_console_cpu_profile_decodes_cpld_boot_state_evidence():
    from harness.inspect.collectors.bmc_console import BmcConsoleCollector

    runner = FakeConsoleRunner()

    def factory(name, _runner):
        if name in ("cpu_msr", "kernel", "ipmi"):
            return BmcConsoleCollector(_runner, subsystem={
                "cpu_msr": "cpu", "kernel": "kernel", "ipmi": "ipmi",
            }[name])
        return None

    prompt_seen = {}

    def llm(prompt):
        prompt_seen["text"] = prompt
        return _fake_llm(prompt)

    engine = DiagnosticEngine(EngineContext(
        runner=runner,
        decoder=Decoder(),
        collector_factory=factory,
        llm=llm,
    ))
    d = engine.run("register 0x1b shows the CPU is in a no boot state")
    # The CPLD boot-state dump is read and decoded into evidence.
    mnemonics = {e["mnemonic"] for e in d.evidence}
    assert "CPLD_1B_CRITICAL" in mnemonics
    assert "CPLD_A1_BOOT_STATE" in mnemonics
    # The prompt shows the decoded register with the no-boot meaning.
    assert "CPU in POST / no boot state" in prompt_seen["text"]
    assert "s5_pwron_wait" in prompt_seen["text"]


def test_console_doc_named_probe_runs_and_decodes():
    from harness.engine.allowlist import AllowRule
    from harness.inspect.collectors.bmc_console import BmcConsoleCollector

    runner = FakeConsoleRunner()
    # A doc-named command NOT in any collector profile (bus 9, unlike the profile's
    # bus 8): the doc-guided plan must add and run it on its own.
    runner.policy.add(AllowRule("sudo -S i2cdump -y 9 0xb"))
    runner.OUTPUT = dict(FakeConsoleRunner.OUTPUT)
    runner.OUTPUT["sudo -S i2cdump -y 9 0xb"] = (
        "     0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f\n"
        "1b: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00\n"
        "a1: 05 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00\n"
    )

    def factory(name, _runner):
        if name in ("cpu_msr", "kernel", "ipmi"):
            return BmcConsoleCollector(_runner, subsystem={
                "cpu_msr": "cpu", "kernel": "kernel", "ipmi": "ipmi",
            }[name])
        return None

    docs = [
        ("[GB_HangUp_troubleshooting_v1.2.pdf p.1] For EVERY Amber Light issue, "
         'please dump "i2cdump -y 9 0xb" in the BMC to get the boot state.'),
    ]
    engine = DiagnosticEngine(EngineContext(
        runner=runner,
        decoder=Decoder(),
        collector_factory=factory,
        llm=_fake_llm,
        docs_retriever=lambda q: docs,
    ))
    d = engine.run("amber light, server stuck in no boot")
    assert any(c.ok and "i2cdump -y 9 0xb" in " ".join(c.argv) for c in runner.calls)
    mnemonics = {e["mnemonic"] for e in d.evidence}
    assert "CPLD_1B_CRITICAL" in mnemonics
    assert "CPLD_A1_BOOT_STATE" in mnemonics


def test_doc_named_probe_denied_on_host_continues():
    """A doc-mined probe the host allowlist rejects must not crash the run."""
    docs = [
        ("[GB_HangUp_troubleshooting_v1.2.pdf p.1] For EVERY Amber Light issue, "
         'please dump "i2cdump -y 8 0xb" in the BMC to get the boot state.'),
    ]
    dump_sets_seen = {}

    engine = DiagnosticEngine(EngineContext(
        runner=FakeRunner(),
        decoder=Decoder(),
        collector_factory=make_collector,
        llm=_fake_llm,
        docs_retriever=lambda q: docs,
        dump_callback=lambda dumps: dump_sets_seen.update(dumps),
    ))
    d = engine.run("MCE uncorrectable ECC error")
    assert d.diagnosis  # pipeline completed despite the denied probe
    denied = dump_sets_seen["doc_guided"]
    assert denied and all(not x.ok for x in denied)  # denied probes surface as failed dumps


def test_console_cpld_chain_falls_back_to_i2ctransfer():
    """BMC without i2cdump: the chain runs i2ctransfer and decodes the CPLD."""
    from harness.engine.allowlist import AllowPolicy, AllowRule
    from harness.inspect.collectors.bmc_console import BmcConsoleCollector

    class _NoI2CdumpConsole(Runner):
        OUTPUT: ClassVar[dict[str, str]] = {
            "sudo -S ipmitool sensor list": "Power_Status | 0x0180 | discrete\n",
            "sudo -S i2cdump -y 8 0xb": "sudo: i2cdump: command not found\n",
            "i2cdump -y 8 0xb": "i2cdump: not found\n",
            "sudo -S i2ctransfer -y 8 w1@0xb 0x00 r256": (
                "00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 "
                "00 00 00 00 00 00 00 00 00 00 00 08 00 00 00 00 "
                + " ".join(["00"] * 128) + " "
                + "00 05 " + " ".join(["00"] * 94)
            ),
        }

        def __init__(self) -> None:
            super().__init__(AllowPolicy([
                AllowRule("sudo -S ipmitool fru print"),
                AllowRule("sudo -S ipmitool sensor list"),
                AllowRule("sudo -S ipmitool sel list"),
                AllowRule("dmesg -r"),
                AllowRule("sudo -S i2cdump -y 8 0xb"),
                AllowRule("i2cdump -y 8 0xb"),
                AllowRule("sudo -S i2ctransfer -y 8 w1@0xb 0x00 r256"),
            ]))
            self.is_console = True

        def _exec(self, argv, timeout=30.0):
            from harness.engine.runner import CommandResult
            key = " ".join(argv)
            out = self.OUTPUT.get(key, "")
            code = 127 if "not found" in out else 0
            return CommandResult(argv=argv, stdout=out, stderr="",
                                 exit_code=code, elapsed_ms=1)

    runner = _NoI2CdumpConsole()

    def factory(name, _runner):
        if name in ("cpu_msr", "kernel", "ipmi"):
            return BmcConsoleCollector(_runner, subsystem={
                "cpu_msr": "cpu", "kernel": "kernel", "ipmi": "ipmi",
            }[name])
        return None

    prompt_seen = {}

    def llm(prompt):
        prompt_seen["text"] = prompt
        return _fake_llm(prompt)

    engine = DiagnosticEngine(EngineContext(
        runner=runner,
        decoder=Decoder(),
        collector_factory=factory,
        llm=llm,
    ))
    d = engine.run("amber light, server stuck no boot")
    # The chain ran i2ctransfer and its block read decoded the CPLD registers.
    mnemonics = {e["mnemonic"] for e in d.evidence}
    assert "CPLD_1B_CRITICAL" in mnemonics
    assert "CPLD_A1_BOOT_STATE" in mnemonics
    # Failed i2cdump attempts are surfaced to the LLM, not hidden.
    assert "probe failed" in prompt_seen["text"]
    assert "command not found" in prompt_seen["text"]


def test_prompt_no_symptom_guidance_is_grounded():
    """The single-shot system prompt turns a generic/no-symptom run into an
    evidence-driven diagnosis instead of asking the operator for input."""
    from harness.diagnosis.prompt import SYSTEM_PREAMBLE, build_prompt
    from harness.diagnosis.summarize import EvidenceSummary

    assert "No-symptom runs" in SYSTEM_PREAMBLE
    prompt = build_prompt(
        model=None,
        decoded=[],
        summaries=EvidenceSummary(interesting=[], anomaly_count=0, total=0),
        doc_snippets=["(none)"],
        parts_refs=[],
        symptom="No specific symptom was reported by the operator.",
    )
    assert prompt.startswith(SYSTEM_PREAMBLE)
    assert "## Symptom" in prompt and "No specific symptom" in prompt


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