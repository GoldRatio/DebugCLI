"""CLI end-to-end: parser, full diagnose run with a fake session + fake LLM,
audit integrity, approval recording, verify against a baseline, lint, console."""

import json
from types import SimpleNamespace

import pytest

from harness.audit.auditlog import AuditLog
from harness.config.inventory_lint import load_inventory
from harness.config.vault import MemorySecretStore
from harness.diagnosis.llm import GeminiLLM, OpenAICompatLLM, StubLLM
from harness.diagnosis.schema import Action, Diagnosis, Risk
from harness.engine.allowlist import AllowPolicy, AllowRule
from harness.engine.runner import CommandResult, Runner
from harness.operator import cli as cli_mod
from harness.operator.cli import (
    _resolve_llm,
    build_parser,
    run_console,
    run_diagnose,
    run_docs,
    run_lint,
    run_verify,
)

FAKE_POLICY = AllowPolicy([
    AllowRule("/usr/bin/rdmsr", ("-a",)),
    AllowRule("/bin/dmesg", ("-l", "*")),
    AllowRule("/bin/dmidecode", ()),
    AllowRule("/usr/bin/lspci", ("-xxx",)),
])

OUTPUT = {
    "/usr/bin/rdmsr -a": "IA32_MC0_STATUS = 0x8000000000000001\n",
    "/bin/dmesg -l err": "MCE: memory error on DIMM_A2\n",
    "/bin/dmesg -l err, crit, alert, emerg": "MCE: memory error on DIMM_A2\n",
    "/bin/dmidecode": "Product Name: model_x\nBIOS Vendor: Intel\nBIOS Version: 2.3\n",
    "/usr/bin/lspci -xxx": "00:1f.2 PCIe link down\n",
    "sudo -S ipmitool fru print": "Product Manufacturer: Quanta\nProduct Name: model_x\n",
    "sudo -S ipmitool sensor list": "CPU0 Temp | 45.000 | degrees C | ok\n",
    "sudo -S ipmitool sel list": " 5 | 08/07/2026 | 21:45:00 | Memory #0x01 | Uncorrectable ECC\n",
    "dmesg -r": "raw kernel ring buffer lines\n",
}


class FakeSession(Runner):
    """Stand-in for SSHSession: canned output per argv, records calls."""

    def __init__(self) -> None:
        super().__init__(FAKE_POLICY)

    def _exec(self, argv, timeout=30.0):
        return CommandResult(argv=list(argv), stdout=OUTPUT.get(" ".join(argv), ""),
                             stderr="", exit_code=0, elapsed_ms=1)


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
        references=[],
    )


_INVENTORY = (
    "trust_level: lab\n"
    "hosts:\n"
    "  - name: h1\n"
    "    address: 10.0.0.10\n"
    "    model: model_x\n"
    "    ssh:\n"
    "      user: diagbot\n"
    "      identity_vault_path: secret/harness/diagbot/id_ed25519\n"
    "      known_hosts_path: config/known_hosts\n"
    "    bmc:\n"
    "      address: 10.0.0.11\n"
    "      username: bmc-ro\n"
    "      password_vault_path: secret/harness/bmc/bmc-ro\n"
    "    console:\n"
    "      address: 192.168.202.51\n"
    "      user: log\n"
    "      identity_vault_path: secret/harness/rackmgr/id_ed25519\n"
    "      known_hosts_path: config/rackmgr_known_hosts\n"
    "      rack: 03\n"
    "      cable: 12\n"
    "      trust_level: lab\n"
    "      port: 2200\n"
    "      sudo_vault_path: secret/harness/bmc/sudo\n"
)


def _inventory(tmp_path) -> str:
    path = tmp_path / "inventory.yaml"
    path.write_text(_INVENTORY, encoding="utf-8")
    return str(path)


def _diagnose_args(tmp_path, **kw):
    argv = ["diagnose", "--inventory", _inventory(tmp_path),
            "--host", "h1", "--symptom", "MCE uncorrectable ECC error",
            "--out-dir", str(tmp_path / "runs"), "--llm", "stub"]
    if kw.get("approve_all"):
        argv.append("--approve-all")
    if kw.get("parts_csv"):
        argv += ["--parts-csv", str(kw["parts_csv"])]
    return build_parser().parse_args(argv)


def test_parser_defaults_to_menu_command():
    # bare `harness` (no subcommand) -> interactive menu, not an argparse error
    args = build_parser().parse_args([])
    assert args.command is None


def test_run_lint_ok(tmp_path):
    inv = tmp_path / "inventory.yaml"
    inv.write_text("trust_level: lab\nhosts: []\n", encoding="utf-8")
    assert run_lint(build_parser().parse_args(["lint", "--inventory", str(inv)])) == 0


def test_diagnose_end_to_end(tmp_path, capsys):
    args = _diagnose_args(tmp_path, approve_all=True)
    diag = run_diagnose(args, overrides={"session": FakeSession(), "llm": _fake_llm})

    assert diag.schema_version == "1.0.0"
    assert diag.actions[0].risk == Risk.LOW
    assert diag.parts_discrepancies == []

    run_dir = next((tmp_path / "runs").iterdir())

    diagnosis_json = json.loads((run_dir / "diagnosis.json").read_text(encoding="utf-8"))
    assert diagnosis_json["actions"][0]["action"] == "Reseat DIMM in slot A2"

    trace = json.loads((run_dir / "trace.json").read_text(encoding="utf-8"))
    assert trace["command_trace"]  # dmidecode/rdmsr/dmesg recorded with hashes
    assert trace["command_trace"][0]["stdout_sha"]

    dumps = json.loads((run_dir / "dumps.json").read_text(encoding="utf-8"))
    assert dumps  # raw dumps persisted as the verify baseline

    audit = AuditLog(run_dir / "audit.jsonl")
    assert audit.verify() == []  # hash chain intact
    kinds = [e.kind for e in audit.read()]
    assert "run_start" in kinds and "cmd" in kinds and "diagnosis" in kinds
    assert "approval" in kinds  # approve_all -> recorded decision

    captured = capsys.readouterr().out
    assert "repair action list" in captured
    assert "confidence:" in captured


def test_diagnose_without_approval_records_denied(tmp_path):
    args = _diagnose_args(tmp_path)
    run_diagnose(args, overrides={"session": FakeSession(), "llm": _fake_llm})
    run_dir = next((tmp_path / "runs").iterdir())
    audit = AuditLog(run_dir / "audit.jsonl")
    approvals = [e.payload for e in audit.read() if e.kind == "approval"]
    assert approvals and approvals[0]["approved"] is False
    assert approvals[0]["note"] == "not prompted"


def test_diagnose_with_parts_csv(tmp_path):
    parts = tmp_path / "parts.csv"
    parts.write_text("slot,fru,pn,sn\nDIMM_A2,12345,SVR-X,SN-9001\n", encoding="utf-8")
    args = _diagnose_args(tmp_path, parts_csv=str(parts))
    diag = run_diagnose(args, overrides={"session": FakeSession(), "llm": _fake_llm})
    # ipmi skipped (no BMC channel) -> FRU cross-check reports "skipped", not a defect
    assert diag.parts_discrepancies == []


def test_verify_against_baseline(tmp_path, capsys):
    baseline = tmp_path / "dumps.json"
    baseline.write_text(json.dumps([{
        "subsystem": "cpu", "source": "/bin/dmesg -l err", "raw": "ecc_error 5\n",
        "cmd_argv": ["/bin/dmesg", "-l", "err"], "ok": True,
    }]), encoding="utf-8")
    args = build_parser().parse_args(
        ["verify", "--inventory", _inventory(tmp_path), "--host", "h1",
         "--symptom", "MCE uncorrectable ECC error", "--baseline", str(baseline)])
    assert args.command == "verify" and args.metric == "ecc"
    # live dmesg output has no "ecc" counter -> count dropped -> resolved
    code = run_verify(args, overrides={"session": FakeSession()})
    assert code == 0
    assert "verdict: resolved" in capsys.readouterr().out


class _FakeSerialConsole:
    def __init__(self, domain, store):
        self.domain = domain
        self.store = store

    def run_probes(self, probes):
        assert self.domain.port == 2200            # inventory port kept
        assert self.domain.sudo_vault_path == "secret/harness/bmc/sudo"
        assert "0penBmc" in self.store.get("secret/harness/bmc/sudo").decode()
        return SimpleNamespace(
            output="i2c 0x51: ff\n[stderr]\n0penBmc leaked into output\n",
            probe_count=1, elapsed_ms=7)

    def close(self):
        pass


def test_run_console_audit_and_redaction(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli_mod, "SerialConsole", _FakeSerialConsole)
    secret_dir = tmp_path / "secrets"
    (secret_dir / "secret" / "harness" / "bmc").mkdir(parents=True)
    (secret_dir / "secret" / "harness" / "bmc" / "sudo").write_text("0penBmc\n", encoding="utf-8")
    out_dir = tmp_path / "runs"

    args = build_parser().parse_args([
        "console", "--inventory", _inventory(tmp_path), "--host", "h1",
        "--probe", "sudo -S i2cdump -y -f 0 0x51",
        "--secret-dir", str(secret_dir), "--out-dir", str(out_dir)])
    assert run_console(args) == 0

    out = capsys.readouterr().out
    assert "i2c 0x51: ff" in out
    assert "0penBmc" not in out                       # secret redacted from stdout

    run_dir = next(out_dir.iterdir())
    console_txt = (run_dir / "console.txt").read_text(encoding="utf-8")
    assert "0penBmc" not in console_txt                # and from the saved output

    audit = AuditLog(run_dir / "audit.jsonl")
    assert audit.verify() == []                        # hash chain intact
    kinds = [e.kind for e in audit.read()]
    assert "console_start" in kinds and "cmd" in kinds
    assert not any("0penBmc" in json.dumps(e.payload) for e in audit.read())

    trace = json.loads((run_dir / "trace.json").read_text(encoding="utf-8"))
    assert trace["command_trace"][0]["argv"] == ["console", "sudo -S i2cdump -y -f 0 0x51"]


def test_run_console_port_override(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_mod, "SerialConsole", _FakeSerialConsole)
    secret_dir = tmp_path / "secrets"
    (secret_dir / "secret" / "harness" / "bmc").mkdir(parents=True)
    (secret_dir / "secret" / "harness" / "bmc" / "sudo").write_text("0penBmc\n", encoding="utf-8")

    args = build_parser().parse_args([
        "console", "--inventory", _inventory(tmp_path), "--host", "h1",
        "--probe", "lspci -xxx",
        "--port", "2200",
        "--secret-dir", str(secret_dir)])
    assert run_console(args) == 0


def test_run_console_missing_sudo_secret(tmp_path, monkeypatch, capsys):
    # explicit empty store: auto-discovery would pick up a lab secrets/ dir
    empty = tmp_path / "empty-store"
    args = build_parser().parse_args([
        "console", "--inventory", _inventory(tmp_path), "--host", "h1",
        "--probe", "lspci -xxx", "--secret-dir", str(empty)])
    assert run_console(args) == 2
    assert "missing from vault" in capsys.readouterr().err


def test_run_console_selects_rack_and_cable_per_launch(tmp_path, monkeypatch):
    seen = {}

    class _FakeSerialConsole:
        def __init__(self, domain, store):
            seen["domain"] = domain
            self.store = store

        def run_probes(self, probes):
            assert seen["domain"].rack == "Q71"
            assert seen["domain"].cable == "8"
            assert seen["domain"].address == "10.9.9.99"
            assert seen["domain"].port == 2200  # inventory default kept
            return SimpleNamespace(output="lspci: ok\n", probe_count=1, elapsed_ms=1)

        def close(self):
            pass

    monkeypatch.setattr(cli_mod, "SerialConsole", _FakeSerialConsole)
    secret_dir = tmp_path / "secrets"
    (secret_dir / "secret" / "harness" / "bmc").mkdir(parents=True)
    (secret_dir / "secret" / "harness" / "bmc" / "sudo").write_text("0penBmc\n", encoding="utf-8")

    args = build_parser().parse_args([
        "console", "--inventory", _inventory(tmp_path), "--host", "h1",
        "--probe", "lspci -xxx", "--rack", "Q71", "--cable", "8",
        "--console-address", "10.9.9.99", "--secret-dir", str(secret_dir)])
    assert run_console(args) == 0


def test_console_override_rejects_rack_injection(tmp_path, monkeypatch):
    from harness.engine.sol import SerialProbeDenied
    args = build_parser().parse_args([
        "diagnose", "--inventory", _inventory(tmp_path), "--host", "h1",
        "--symptom", "MCE uncorrectable ECC error", "--console",
        "--rack", "03; ls", "--llm", "stub"])
    with pytest.raises(SerialProbeDenied):
        run_diagnose(args, overrides={})


def test_diagnose_console_path_runs_pipeline(tmp_path):
    from harness.engine.runner import CommandResult

    class _FakeConsoleRunner:
        is_console = True

        def __init__(self):
            self.calls = []

        def execute(self, argv, timeout=300.0):
            result = CommandResult(
                argv=list(argv),
                stdout=OUTPUT.get(" ".join(argv), ""),
                stderr="", exit_code=0, elapsed_ms=1)
            self.calls.append(result)
            return result

    console_runner = _FakeConsoleRunner()
    args = build_parser().parse_args([
        "diagnose", "--inventory", _inventory(tmp_path), "--host", "h1",
        "--symptom", "MCE uncorrectable ECC error", "--console",
        "--out-dir", str(tmp_path / "runs"), "--llm", "stub", "--approve-all"])
    diag = run_diagnose(args, overrides={"console_runner": console_runner})

    assert diag.schema_version == "1.0.0"
    calls = [" ".join(c.argv) for c in console_runner.calls]
    # BMC console path: FRU model detection + BMC-shell probes, no host-OS tools
    assert "sudo -S ipmitool fru print" in calls
    assert "sudo -S ipmitool sensor list" in calls
    assert "sudo -S ipmitool sel list" in calls
    assert not any("/usr/bin/rdmsr" in c or "smartctl" in c for c in calls)
    # BMC probes returned output: dump-based evidence fit, not the 0.3 floor
    assert diag.confidence_breakdown.evidence_fit == 1.0


def test_diagnose_console_generic_plan_dedupes_bmc_probes(tmp_path):
    console_runner = _FakeConsoleRunner()
    args = build_parser().parse_args([
        "diagnose", "--inventory", _console_defaults_inventory(tmp_path),
        "--rack", "Q61", "--cable", "8",
        "--symptom", "Repair seems to be done",
        "--out-dir", str(tmp_path / "runs"), "--llm", "stub", "--approve-all"])
    run_diagnose(args, overrides={"console_runner": console_runner})

    calls = [" ".join(c.argv) for c in console_runner.calls]
    # generic plan (cpu+ipmi+kernel): overlapping sensor/sel/fru probes run once
    assert calls == [
        "sudo -S ipmitool fru print",       # model detection
        "sudo -S ipmitool sensor list",     # cpu
        "sudo -S ipmitool sel list",        # kernel
        "dmesg -r",                         # kernel
    ]

    run_dir = next((tmp_path / "runs").iterdir())
    audit = AuditLog(run_dir / "audit.jsonl")
    assert audit.verify() == []
    kinds = [e.kind for e in audit.read()]
    assert "cmd" in kinds  # console probes recorded in the WORM audit

    model_event = next(e.payload for e in audit.read() if e.kind == "model_detected")
    assert model_event["product_name"] == "model_x"  # FRU-detected, not "unknown"

    trace = json.loads((run_dir / "trace.json").read_text(encoding="utf-8"))
    assert trace["command_trace"]


def test_diagnose_console_requires_console_block(tmp_path):
    no_console_inv = tmp_path / "inventory.yaml"
    no_console_inv.write_text(
        "trust_level: lab\n"
        "hosts:\n"
        "  - name: h1\n"
        "    address: 10.0.0.10\n"
        "    model: model_x\n"
        "    ssh:\n"
        "      user: diagbot\n"
        "      identity_vault_path: secret/harness/diagbot/id_ed25519\n"
        "      known_hosts_path: config/known_hosts\n"
        "    bmc:\n"
        "      address: 10.0.0.11\n"
        "      username: bmc-ro\n"
        "      password_vault_path: secret/harness/bmc/bmc-ro\n",
        encoding="utf-8")
    args = build_parser().parse_args([
        "diagnose", "--inventory", str(no_console_inv), "--host", "h1",
        "--symptom", "MCE uncorrectable ECC error", "--console", "--llm", "stub"])
    with pytest.raises(RuntimeError, match="no console block"):
        run_diagnose(args, overrides={"console_runner": object()})


class _FakePdfParser:
    def parse(self, path):
        from harness.docs.ingest.pdf_parser import PageText
        return [PageText(page=1, text="memory ECC uncorrectable DIMM error IA32_MC0_STATUS")]


def _docs_lib(tmp_path, monkeypatch) -> str:
    from harness.docs.ingest import library as lib_mod
    monkeypatch.setattr(lib_mod, "PdfParser", _FakePdfParser)
    lib = tmp_path / "docs_lib"
    pdf = tmp_path / "arch.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    assert run_docs(build_parser().parse_args(["docs", "--lib", str(lib), "add", str(pdf)])) == 0
    return str(lib)


def test_docs_cli_add_ls_reindex_rm(tmp_path, monkeypatch, capsys):
    lib = _docs_lib(tmp_path, monkeypatch)
    assert (tmp_path / "docs_lib" / "pdfs" / "arch.pdf").exists()  # uploaded copy

    assert run_docs(build_parser().parse_args(["docs", "--lib", lib, "ls"])) == 0
    assert "arch.pdf" in capsys.readouterr().out
    assert run_docs(build_parser().parse_args(["docs", "--lib", lib, "reindex"])) == 0

    assert run_docs(build_parser().parse_args(["docs", "--lib", lib, "rm", "arch.pdf"])) == 0
    assert not (tmp_path / "docs_lib" / "pdfs" / "arch.pdf").exists()
    assert run_docs(build_parser().parse_args(["docs", "--lib", lib, "ls"])) == 0
    assert "empty library" in capsys.readouterr().out


def test_docs_cli_rm_missing_returns_error(tmp_path, monkeypatch, capsys):
    lib = _docs_lib(tmp_path, monkeypatch)
    code = run_docs(build_parser().parse_args(["docs", "--lib", lib, "rm", "nope.pdf"]))
    assert code == 2
    assert "not in library" in capsys.readouterr().err


def test_diagnose_uses_docs_lib_for_rag(tmp_path, monkeypatch):
    lib = _docs_lib(tmp_path, monkeypatch)
    args = build_parser().parse_args([
        "diagnose", "--inventory", _inventory(tmp_path), "--host", "h1",
        "--symptom", "MCE uncorrectable ECC error", "--docs-lib", lib,
        "--out-dir", str(tmp_path / "runs"), "--llm", "stub", "--approve-all"])
    diag = run_diagnose(args, overrides={"session": FakeSession()})
    assert diag.schema_version == "1.0.0"
    assert diag.confidence_breakdown is not None  # scorer ran over retrieved snippets


def test_diagnose_interactive_session(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _q: "DIMM was reseated, no change")
    args = build_parser().parse_args([
        "diagnose", "--inventory", _inventory(tmp_path), "--host", "h1",
        "--symptom", "MCE uncorrectable ECC error", "--out-dir", str(tmp_path / "runs"),
        "--llm", "stub", "--interactive", "--approve-all"])
    diag = run_diagnose(args, overrides={"session": FakeSession()})
    assert diag.schema_version == "1.0.0"

    run_dir = next((tmp_path / "runs").iterdir())
    transcript = json.loads((run_dir / "transcript.json").read_text(encoding="utf-8"))
    kinds = [t["kind"] for t in transcript]
    assert kinds == ["question", "answer", "diagnosis"]
    assert any("reseated" in t["content"] for t in transcript)

    audit = AuditLog(run_dir / "audit.jsonl")
    assert audit.verify() == []
    kinds = [e.kind for e in audit.read()]
    assert "turn" in kinds and "diagnosis" in kinds
    run_start = next(e.payload for e in audit.read() if e.kind == "run_start")
    assert run_start["mode"] == "session"

    out = capsys.readouterr().out
    assert "repair action list" in out


def test_diagnose_context_file_seeds_session(tmp_path):
    ctx_file = tmp_path / "context.txt"
    ctx_file.write_text("Replaced PSU last week; fans OK", encoding="utf-8")
    args = build_parser().parse_args([
        "diagnose", "--inventory", _inventory(tmp_path), "--host", "h1",
        "--symptom", "MCE uncorrectable ECC error", "--out-dir", str(tmp_path / "runs"),
        "--llm", "stub", "--context-file", str(ctx_file), "--approve-all"])
    diag = run_diagnose(args, overrides={"session": FakeSession()})
    assert diag.schema_version == "1.0.0"

    run_dir = next((tmp_path / "runs").iterdir())
    transcript = json.loads((run_dir / "transcript.json").read_text(encoding="utf-8"))
    assert any(t["kind"] == "context" and "Replaced PSU" in t["content"]
               for t in transcript)
    # stub asks one question and gets "(no answer)" non-interactively, then diagnoses
    assert any(t["kind"] == "question" for t in transcript)
    assert any(t["kind"] == "answer" and t["content"] == "(no answer)" for t in transcript)


def test_diagnose_single_shot_has_no_transcript(tmp_path):
    args = _diagnose_args(tmp_path, approve_all=True)
    run_diagnose(args, overrides={"session": FakeSession(), "llm": _fake_llm})
    run_dir = next((tmp_path / "runs").iterdir())
    assert not (run_dir / "transcript.json").exists()


def test_resolve_llm_flag_precedence(tmp_path):
    inv = load_inventory(_inventory(tmp_path))  # no llm block in inventory
    store = MemorySecretStore()
    assert isinstance(_resolve_llm(SimpleNamespace(llm=None), inv, store),
                      OpenAICompatLLM)              # default -> openai
    assert isinstance(_resolve_llm(SimpleNamespace(llm="gemini"), inv, store),
                      GeminiLLM)                    # flag -> gemini
    assert isinstance(_resolve_llm(SimpleNamespace(llm="stub"), inv, store),
                      StubLLM)                      # flag -> stub


def test_resolve_llm_from_inventory_block(tmp_path):
    inv_path = tmp_path / "inv.yaml"
    inv_path.write_text(
        "trust_level: lab\n"
        "llm:\n"
        "  provider: gemini\n"
        "  model: gemini-2.5-flash\n"
        "  api_key_vault_path: secret/harness/llm/gemini-key\n"
        "hosts: []\n", encoding="utf-8")
    inv = load_inventory(inv_path)
    store = MemorySecretStore({"secret/harness/llm/gemini-key": b"gk-secret\n"})
    llm = _resolve_llm(SimpleNamespace(llm=None), inv, store)
    assert isinstance(llm, GeminiLLM)
    assert llm.model == "gemini-2.5-flash"
    assert llm.api_key == "gk-secret"          # resolved through the secret store

    missing = MemorySecretStore()              # key absent -> env fallback, no crash
    assert _resolve_llm(SimpleNamespace(llm=None), inv, missing).api_key is None


def test_diagnose_llm_from_inventory(tmp_path):
    path = tmp_path / "inventory.yaml"
    path.write_text(_INVENTORY.replace(
        "trust_level: lab\nhosts:\n",
        "trust_level: lab\nllm:\n  provider: stub\nhosts:\n"), encoding="utf-8")
    args = build_parser().parse_args([
        "diagnose", "--inventory", str(path), "--host", "h1",
        "--symptom", "MCE uncorrectable ECC error",
        "--out-dir", str(tmp_path / "runs"), "--approve-all"])
    assert args.llm is None                     # not passed on the CLI
    diag = run_diagnose(args, overrides={"session": FakeSession()})
    assert "No LLM configured" in diag.diagnosis  # inventory stub was picked up


def test_diagnose_flag_overrides_inventory_llm(tmp_path):
    path = tmp_path / "inventory.yaml"
    path.write_text(_INVENTORY.replace(
        "trust_level: lab\nhosts:\n",
        "trust_level: lab\nllm:\n  provider: gemini\nhosts:\n"), encoding="utf-8")
    args = build_parser().parse_args([
        "diagnose", "--inventory", str(path), "--host", "h1",
        "--symptom", "MCE uncorrectable ECC error",
        "--out-dir", str(tmp_path / "runs"), "--llm", "stub", "--approve-all"])
    diag = run_diagnose(args, overrides={"session": FakeSession()})
    assert "No LLM configured" in diag.diagnosis  # --llm stub beat the inventory


# ---- zero-YAML targeting end-to-end ----

_CONSOLE_DEFAULTS_INVENTORY = (
    "trust_level: lab\n"
    "llm:\n"
    "  provider: stub\n"
    "console_defaults:\n"
    "  address: 192.168.202.51\n"
    "  user: log\n"
    "  identity_vault_path: secret/harness/rackmgr/id_ed25519\n"
    "  known_hosts_path: config/rackmgr_known_hosts\n"
    "  tool: jumpin\n"
    "  trust_level: lab\n"
    "  port: 2200\n"
    "  sudo_vault_path: secret/harness/bmc/sudo\n"
    "hosts: []\n"
)


def _console_defaults_inventory(tmp_path) -> str:
    path = tmp_path / "inventory.yaml"
    path.write_text(_CONSOLE_DEFAULTS_INVENTORY, encoding="utf-8")
    return str(path)


class _FakeConsoleRunner:
    """Stand-in for the serial console runner: canned output per argv."""

    is_console = True

    def __init__(self):
        self.calls = []

    def execute(self, argv, timeout=300.0):
        result = CommandResult(
            argv=list(argv),
            stdout=OUTPUT.get(" ".join(argv), ""),
            stderr="", exit_code=0, elapsed_ms=1)
        self.calls.append(result)
        return result


def test_diagnose_rack_cable_uses_console_defaults(tmp_path):
    console_runner = _FakeConsoleRunner()
    args = build_parser().parse_args([
        "diagnose", "--inventory", _console_defaults_inventory(tmp_path),
        "--rack", "Q61", "--cable", "8",
        "--symptom", "MCE uncorrectable ECC error",
        "--out-dir", str(tmp_path / "runs"), "--llm", "stub", "--approve-all"])
    diag = run_diagnose(args, overrides={"console_runner": console_runner})

    assert diag.schema_version == "1.0.0"
    assert "sudo -S ipmitool fru print" in [" ".join(c.argv) for c in console_runner.calls]

    run_dir = next((tmp_path / "runs").iterdir())
    audit = AuditLog(run_dir / "audit.jsonl")
    assert audit.verify() == []
    run_start = next(e.payload for e in audit.read() if e.kind == "run_start")
    assert run_start["target"] == "console"
    assert run_start["rack"] == "Q61" and run_start["cable"] == "8"
    assert run_start["host"] == "Q61-cable8"


def test_diagnose_rack_cable_without_console_defaults_raises(tmp_path):
    inv = tmp_path / "inventory.yaml"
    inv.write_text("trust_level: lab\nhosts: []\n", encoding="utf-8")
    from harness.targets import TargetError
    args = build_parser().parse_args([
        "diagnose", "--inventory", str(inv), "--rack", "Q61", "--cable", "8",
        "--symptom", "x", "--llm", "stub"])
    with pytest.raises(TargetError, match="console_defaults"):
        run_diagnose(args, overrides={})


def test_diagnose_by_address_uses_ssh_identity_from_store(tmp_path):
    store = MemorySecretStore({
        "secret/harness/ssh/10.0.0.50": b"-----BEGIN OPENSSH PRIVATE KEY-----\nAAAA\n"
                                        b"-----END OPENSSH PRIVATE KEY-----\n"})
    args = build_parser().parse_args([
        "diagnose", "--inventory", _console_defaults_inventory(tmp_path),
        "--address", "10.0.0.50",
        "--symptom", "MCE uncorrectable ECC error",
        "--out-dir", str(tmp_path / "runs"), "--llm", "stub", "--approve-all"])
    diag = run_diagnose(args, overrides={"session": FakeSession(), "store": store})

    assert diag.schema_version == "1.0.0"
    run_dir = next((tmp_path / "runs").iterdir())
    run_start = next(e.payload for e in
                     AuditLog(run_dir / "audit.jsonl").read() if e.kind == "run_start")
    assert run_start["target"] == "ssh"
    assert run_start["ip"] == "10.0.0.50"
    assert run_start["host"] == "10.0.0.50"


def test_diagnose_by_address_missing_identity_raises_with_hint(tmp_path):
    from harness.targets import TargetError
    args = build_parser().parse_args([
        "diagnose", "--inventory", _console_defaults_inventory(tmp_path),
        "--address", "10.0.0.50", "--symptom", "x", "--llm", "stub"])
    with pytest.raises(TargetError, match="no SSH identity.*add-ssh"):
        run_diagnose(args, overrides={"store": MemorySecretStore()})


def test_diagnose_alias_targets_console(tmp_path):
    targets_file = tmp_path / "targets.yaml"
    targets_file.write_text("targets:\n- {alias: d1, rack: Q61, cable: 8}\n",
                            encoding="utf-8")
    console_runner = _FakeConsoleRunner()
    args = build_parser().parse_args([
        "diagnose", "--inventory", _console_defaults_inventory(tmp_path),
        "--target", "d1", "--targets-file", str(targets_file),
        "--symptom", "MCE uncorrectable ECC error",
        "--out-dir", str(tmp_path / "runs"), "--llm", "stub", "--approve-all"])
    diag = run_diagnose(args, overrides={"console_runner": console_runner})
    assert diag.schema_version == "1.0.0"

    run_dir = next((tmp_path / "runs").iterdir())
    run_start = next(e.payload for e in
                     AuditLog(run_dir / "audit.jsonl").read() if e.kind == "run_start")
    assert run_start["target"] == "console"
    assert run_start["rack"] == "Q61" and run_start["cable"] == "8"


def test_diagnose_requires_a_target_when_no_default(tmp_path):
    from harness.targets import TargetError
    args = build_parser().parse_args([
        "diagnose", "--inventory", _console_defaults_inventory(tmp_path),
        "--symptom", "x", "--llm", "stub"])
    with pytest.raises(TargetError, match="no target given"):
        run_diagnose(args, overrides={"store": MemorySecretStore()})


def test_targets_subcommand_add_ls_rm(tmp_path, capsys):
    targets_file = tmp_path / "targets.yaml"
    args = build_parser().parse_args([
        "targets", "--targets-file", str(targets_file),
        "add", "d1", "--rack", "Q61", "--cable", "8"])
    assert cli_mod.run_targets(args) == 0

    args = build_parser().parse_args(
        ["targets", "--targets-file", str(targets_file), "ls"])
    assert cli_mod.run_targets(args) == 0
    assert "d1" in capsys.readouterr().out

    args = build_parser().parse_args(
        ["targets", "--targets-file", str(targets_file), "rm", "d1"])
    assert cli_mod.run_targets(args) == 0
    capsys.readouterr()
    args = build_parser().parse_args(
        ["targets", "--targets-file", str(targets_file), "ls"])
    assert cli_mod.run_targets(args) == 0
    assert "no targets registered" in capsys.readouterr().out


def test_targets_subcommand_rm_missing(tmp_path, capsys):
    args = build_parser().parse_args(
        ["targets", "--targets-file", str(tmp_path / "t.yaml"), "rm", "nope"])
    assert cli_mod.run_targets(args) == 1
    assert "unknown target alias" in capsys.readouterr().err


def test_targets_add_requires_rack_cable_or_address(tmp_path, capsys):
    args = build_parser().parse_args(
        ["targets", "--targets-file", str(tmp_path / "t.yaml"), "add", "d1"])
    assert cli_mod.run_targets(args) == 2
    assert "needs --rack/--cable" in capsys.readouterr().err
