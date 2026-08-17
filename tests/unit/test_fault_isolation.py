"""Fault-isolation pass tests: decoded rail fault triggers a second, targeted
doc + probe round whose raw output reaches the prompt but never the catalog
decoder (a non-SWB device's bytes must not be misattributed to SWB registers).
"""

from typing import ClassVar

from harness.diagnosis.engine import DiagnosticEngine, EngineContext
from harness.diagnosis.schema import Diagnosis
from harness.engine.allowlist import AllowPolicy, AllowRule
from harness.engine.runner import Runner
from harness.inspect.base import DecodedField, RegisterDecode
from harness.inspect.decoder import Decoder

ISOLATION_SNIPPET = (
    "[GB_NVL72_Troubleshooting_v2.pdf p.14] To isolate a PDB 12V power fault, "
    "dump the PRIMARY Bianca FPGA registers: run \"i2ctransfer -y 2 w2@0x11 "
    "0x00 0x00 r256\"; for the SECONDARY Bianca use \"i2ctransfer -y 1 "
    "w2@0x11 0x00 0x00 r256\"."
)

# 256 bytes, all zero EXCEPT offset 0x63 = 0x80. The i2cdump rows only cover
# offsets 0x0b-0x2a and 0x93-0xa2, so 0x63 (CPLD_63_HANDSHAKE) can only enter
# the evidence if the isolation dump were wrongly routed to the catalog decoder.
_FPGA_BYTES = ("00 " * 0x63) + "80 " + ("00 " * (0x100 - 0x64))


class FaultConsoleRunner(Runner):
    """Console runner whose CPLD dump asserts a PDB 12V rail fault."""

    OUTPUT: ClassVar[dict[str, str]] = {
        "sudo -S ipmitool sensor list": (
            "P12V_CB2_VOLT   | 0.000     | Volts     | nr    | na | na\n"
            "FANS_SWB_PWM    | 32.000    | percent   | ok    | na | na\n"
            "Power_Status    | 0x0       | discrete  | 0x0180| na | na\n"
        ),
        "sudo -S ipmitool sel list": (
            "  1 | 08/10/26 | 21:30:38 UTC | Power Supply #0x75 | Failure detected | Asserted\n"
        ),
        "dmesg -r": "<6>Booting Linux ... Machine model: Microsoft DC-SCM 1.2 C4A15 BMC Device Tree\n",
        "sudo -S ipmitool fru print": (
            "Product Name          : C4A15\nProduct Manufacturer  : Microsoft\n"
        ),
        "sudo -S i2cdump -y 8 0xb": (
            "     0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f\n"
            "0b: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00\n"
            "1b: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00\n"
            "93: c0 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00\n"
            "94: 80 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00\n"
            "a1: 02 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00\n"
        ),
        "sudo -S i2ctransfer -y 2 w2@0x11 0x00 0x00 r256": (
            "PRIMARY BIANCA FPGA DUMP: " + _FPGA_BYTES
        ),
        "sudo -S i2ctransfer -y 1 w2@0x11 0x00 0x00 r256": (
            "SECONDARY BIANCA FPGA DUMP: " + _FPGA_BYTES
        ),
    }

    def __init__(self) -> None:
        super().__init__(AllowPolicy([
            AllowRule(cmd) for cmd in self.OUTPUT
        ]))
        self.is_console = True

    def _exec(self, argv, timeout=30.0):
        from harness.engine.runner import CommandResult
        key = " ".join(argv)
        out = self.OUTPUT.get(key, "")
        return CommandResult(argv=argv, stdout=out, stderr="", exit_code=0, elapsed_ms=1)

    def batch_execute(self, cmds: list[str]):
        return [self.execute([c]) for c in cmds]


class HealthyConsoleRunner(FaultConsoleRunner):
    OUTPUT: ClassVar[dict[str, str]] = {
        "sudo -S ipmitool sensor list": (
            "FANS_SWB_PWM    | 32.000    | percent   | ok    | na | na\n"
            "P12V_CB2_VOLT   | 12.100    | Volts     | ok    | na | na\n"
            "Power_Status    | 0x0       | discrete  | 0x0180| na | na\n"
        ),
        "sudo -S ipmitool sel list": (
            "  1 | 08/10/26 | 21:30:38 UTC | System Event #0x07 | Timestamp Clock Sync | Asserted\n"
        ),
        "dmesg -r": "<6>Booting Linux ... Machine model: Microsoft DC-SCM 1.2 C4A15 BMC Device Tree\n",
        "sudo -S ipmitool fru print": (
            "Product Name          : C4A15\nProduct Manufacturer  : Microsoft\n"
        ),
        "sudo -S i2cdump -y 8 0xb": (
            "     0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f\n"
            "0b: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00\n"
            "1b: 18 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00\n"
            "93: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00\n"
            "94: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00\n"
            "a1: 05 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00\n"
        ),
    }


def _console_factory(name, _runner):
    from harness.inspect.collectors.bmc_console import BmcConsoleCollector
    if name in ("cpu_msr", "kernel", "ipmi"):
        return BmcConsoleCollector(_runner, subsystem={
            "cpu_msr": "cpu", "kernel": "kernel", "ipmi": "ipmi",
        }[name])
    return None


def _make_retriever(seen_queries):
    def retriever(query, _model_key):
        seen_queries.append(query)
        if "power distribution board busbar" in query \
                or "register dump power sequence" in query:
            return [ISOLATION_SNIPPET]
        return []
    return retriever


def _fake_llm(_prompt: str) -> Diagnosis:
    from harness.diagnosis.schema import Action, Reference, Risk
    return Diagnosis(
        diagnosis="PDB 12V rail power-up fault",
        confidence=0.0,
        actions=[Action(
            step=1,
            action="Isolate the rail: check each 12V load (Bianca) before replacing the PDB",
            rationale="PDB 12V power-up fault decoded; isolation doc page 14",
            risk=Risk.LOW,
            required_tool="BMC console",
            impact="none",
            references=[Reference(source="GB_NVL72_Troubleshooting_v2.pdf", page="14")],
        )],
        references=[Reference(source="GB_NVL72_Troubleshooting_v2.pdf", page="14")],
    )


def test_isolation_pass_runs_second_doc_probe_round():
    runner = FaultConsoleRunner()
    seen_queries: list[str] = []
    dump_sets_seen: dict = {}
    prompt_seen: dict = {}
    used_snippets: dict = {"lines": []}
    progress: list[str] = []

    def llm(prompt):
        prompt_seen["text"] = prompt
        return _fake_llm(prompt)

    engine = DiagnosticEngine(EngineContext(
        runner=runner,
        decoder=Decoder(),
        collector_factory=_console_factory,
        llm=llm,
        docs_retriever=_make_retriever(seen_queries),
        dump_callback=lambda dumps: dump_sets_seen.update(dumps),
        prompt_callback=lambda content: prompt_seen.update(content=content),
        snippets_callback=lambda lines: used_snippets.update(lines=lines),
        progress=progress.append,
    ))
    d = engine.run("amber light, PDB 12V rail, server stuck in no boot")

    # The isolation round retrieved a SECOND doc query set (isolation vocab).
    assert any("power distribution board busbar" in q for q in seen_queries)
    assert any("register dump power sequence" in q for q in seen_queries)

    # The doc-named FPGA probes ran and their output was kept as raw evidence.
    iso = dump_sets_seen["doc_guided_isolation"]
    assert iso and all(x.ok for x in iso)
    assert len(iso) == 2
    assert any("i2ctransfer -y 2 w2@0x11 0x00 0x00 r256" in x.source for x in iso)
    assert any("i2ctransfer -y 1 w2@0x11 0x00 0x00 r256" in x.source for x in iso)
    assert any("i2ctransfer" in " ".join(c.argv) for c in runner.calls)

    # Isolation output is NOT catalog-decoded: the FPGA dump byte at offset 0x63
    # (0x80) would decode to CPLD_63_HANDSHAKE, an address the i2cdump rows do
    # not cover. It must not appear as evidence.
    mnemonics = {e["mnemonic"] for e in d.evidence}
    assert "CPLD_63_HANDSHAKE" not in mnemonics
    assert {"CPLD_1B_CRITICAL", "CPLD_A1_BOOT_STATE",
            "CPLD_93_VR_OR", "CPLD_94_PWRUP_FAULTS"} <= mnemonics

    # The prompt carries the isolation snippet AND the raw probe evidence.
    assert "## Isolation Probe Evidence" in prompt_seen["text"]
    assert "- sudo -S i2ctransfer -y 2 w2@0x11 0x00 0x00 r256" in prompt_seen["text"]
    assert "PRIMARY BIANCA FPGA DUMP" in prompt_seen["text"]
    assert ISOLATION_SNIPPET in prompt_seen["text"]

    # The scorer got the exact snippet union used in the prompt.
    assert any("Bianca" in s for s in used_snippets["lines"])

    # Progress surfaced the isolation round.
    assert any(line.startswith("isolate:") for line in progress)

    # The raw isolation dumps are never routed to the catalog decoder; the
    # decoded entries remain catalog-tagged (from the i2cdump CPLD block).
    assert all(e["catalog_version"] for e in d.evidence)


def test_healthy_run_skips_isolation_pass():
    runner = HealthyConsoleRunner()
    seen_queries: list[str] = []
    dump_sets_seen: dict = {}
    prompt_seen: dict = {}

    def llm(prompt):
        prompt_seen["text"] = prompt
        return _fake_llm(prompt)

    engine = DiagnosticEngine(EngineContext(
        runner=runner,
        decoder=Decoder(),
        collector_factory=_console_factory,
        llm=llm,
        docs_retriever=_make_retriever(seen_queries),
        dump_callback=lambda dumps: dump_sets_seen.update(dumps),
        prompt_callback=lambda content: prompt_seen.update(content=content),
    ))
    engine.run("amber light, server stuck in no boot")

    # No isolation-vocab retrieval happened; no isolation dumps; prompt says none.
    assert not any("busbar" in q for q in seen_queries)
    assert "doc_guided_isolation" not in dump_sets_seen
    assert "## Isolation Probe Evidence" in prompt_seen["text"]
    assert "(none)" in prompt_seen["text"]


def test_detect_fault_signature_units():
    from harness.plan.isolation import build_isolation_queries, detect_fault_signature

    fault = [
        RegisterDecode(
            mnemonic="CPLD_93_VR_OR", raw_hex="0xc0",
            decoded_fields=[
                DecodedField(name="vr_fail_all_or", msb=7, lsb=7,
                             raw_value="1"),
                DecodedField(name="pdb_vr_flt", msb=6, lsb=6, raw_value="1"),
            ]),
        RegisterDecode(
            mnemonic="CPLD_94_PWRUP_FAULTS", raw_hex="0x80",
            decoded_fields=[
                DecodedField(name="pdb_12v_pwrup_flt", msb=7, lsb=7,
                             raw_value="1"),
            ]),
    ]
    healthy = [
        RegisterDecode(
            mnemonic="CPLD_94_PWRUP_FAULTS", raw_hex="0x00",
            decoded_fields=[
                DecodedField(name="pdb_12v_pwrup_flt", msb=7, lsb=7,
                             raw_value="0"),
            ]),
    ]

    sig = detect_fault_signature(fault)
    assert sig is not None
    assert "pdb" in sig["rail_tokens"] and "12v" in sig["rail_tokens"]
    assert detect_fault_signature(healthy) is None

    queries = build_isolation_queries(sig)
    assert len(queries) == 2
    assert queries[0].startswith("pdb 12v rail power-up fault isolate")
    assert "busbar" in queries[0] and "impedance" in queries[0]
    assert queries[1].startswith("pdb 12v power fault isolate")


def test_sensor_anomaly_supplies_rail_hint():
    from harness.diagnosis.summarize import EvidenceSummary
    from harness.plan.isolation import detect_fault_signature

    sig = detect_fault_signature(
        [RegisterDecode(
            mnemonic="CPLD_94_PWRUP_FAULTS", raw_hex="0x80",
            decoded_fields=[
                DecodedField(name="pdb_12v_pwrup_flt", msb=7, lsb=7,
                             raw_value="1"),
            ])],
        summaries=EvidenceSummary(
            interesting=["[current] P12V_CB2_VOLT   | 0.000     | Volts     | nr"],
            anomaly_count=1, total=10),
    )
    assert sig is not None
    assert "p12v" in sig["rail_tokens"]
    assert "cb2" in sig["rail_tokens"]


def test_parts_ask_invoked_on_rail_fault():
    """An opted-in parts-ask fires with the rail key on a decoded rail fault,
    and the operator's answers reach the prompt."""
    runner = FaultConsoleRunner()
    seen_queries: list[str] = []
    asked_rail: dict = {}
    prompt_seen: dict = {}

    def llm(prompt):
        prompt_seen["text"] = prompt
        return _fake_llm(prompt)

    def parts_ask(rail):
        asked_rail["rail"] = rail
        return [{"slot": "right_bianca", "fru": "Bianca Board",
                 "pn": "PN-1234", "sn": "SN-5678"}]

    engine = DiagnosticEngine(EngineContext(
        runner=runner,
        decoder=Decoder(),
        collector_factory=_console_factory,
        llm=llm,
        docs_retriever=_make_retriever(seen_queries),
        parts_ask=parts_ask,
        prompt_callback=lambda content: prompt_seen.update(content=content),
    ))
    engine.run("amber light, PDB 12V rail, server stuck in no boot")

    assert "pdb" in (asked_rail.get("rail") or "")
    assert "12v" in (asked_rail.get("rail") or "")
    assert "## Operator-Supplied Parts" in prompt_seen["text"]
    assert "right_bianca: Bianca Board (PN-1234), SN SN-5678" in prompt_seen["text"]


def test_power_topology_section_renders_on_rail_fault():
    """A wired topology hook maps the decoded rail fault to its documented loads,
    and the prompt renders the suspect set with page provenance."""
    runner = FaultConsoleRunner()
    prompt_seen: dict = {}

    def llm(prompt):
        prompt_seen["text"] = prompt
        return _fake_llm(prompt)

    def topology(sig, model_key):
        from harness.docs.parts.topology import loads_for_rail
        return loads_for_rail(sig, model_key)

    engine = DiagnosticEngine(EngineContext(
        runner=runner,
        decoder=Decoder(),
        collector_factory=_console_factory,
        llm=llm,
        docs_retriever=_make_retriever([]),
        topology=topology,
        prompt_callback=lambda content: prompt_seen.update(content=content),
    ))
    engine.run("amber light, PDB 12V rail, server stuck in no boot")

    assert "## Power-Topology (documented rail loads)" in prompt_seen["text"]
    # the sensor hint (P12V_CB2_VOLT nr) + pdb_12v_pwrup_flt resolve to P12V_CB2
    assert "- P12V_CB2 (12V DC, platform samoa" in prompt_seen["text"]
    assert "OSFP Board" in prompt_seen["text"]
    assert "p.86" in prompt_seen["text"]


def test_power_topology_absent_without_hook():
    """Without a topology hook the prompt has no Power-Topology section."""
    runner = FaultConsoleRunner()
    prompt_seen: dict = {}

    def llm(prompt):
        prompt_seen["text"] = prompt
        return _fake_llm(prompt)

    engine = DiagnosticEngine(EngineContext(
        runner=runner,
        decoder=Decoder(),
        collector_factory=_console_factory,
        llm=llm,
        docs_retriever=_make_retriever([]),
        prompt_callback=lambda content: prompt_seen.update(content=content),
    ))
    engine.run("amber light, PDB 12V rail, server stuck in no boot")
    assert "## Power-Topology" not in prompt_seen["text"]


def test_parts_ask_not_called_when_unconfigured():
    """Without parts_ask configured the prompt has no operator-parts section."""
    runner = FaultConsoleRunner()
    prompt_seen: dict = {}

    def llm(prompt):
        prompt_seen["text"] = prompt
        return _fake_llm(prompt)

    engine = DiagnosticEngine(EngineContext(
        runner=runner,
        decoder=Decoder(),
        collector_factory=_console_factory,
        llm=llm,
        docs_retriever=_make_retriever([]),
        prompt_callback=lambda content: prompt_seen.update(content=content),
    ))
    engine.run("amber light, PDB 12V rail, server stuck in no boot")
    assert "## Operator-Supplied Parts" not in prompt_seen["text"]


def test_parts_ask_skipped_on_healthy_run():
    """A healthy/standby run never asks for parts even when opted in."""
    runner = HealthyConsoleRunner()
    called: list[str] = []

    engine = DiagnosticEngine(EngineContext(
        runner=runner,
        decoder=Decoder(),
        collector_factory=_console_factory,
        llm=_fake_llm,
        docs_retriever=_make_retriever([]),
        parts_ask=lambda rail: called.append(rail) or [],
    ))
    engine.run("amber light, server stuck in no boot")
    assert called == []
