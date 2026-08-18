"""FAT single-test driver + session-engine integration.

Unit tests only: the driver is exercised against a scripted interactive shell
(no SSH), and the session engine against the scripted shell + canned LLM turns.
"""

import json
from typing import ClassVar

import pytest

from harness.diagnosis.engine import EngineContext
from harness.diagnosis.schema import Action, Diagnosis, Risk
from harness.diagnosis.session import SessionEngine
from harness.engine.allowlist import AllowPolicy, AllowRule
from harness.engine.interactive import normalize_terminal
from harness.engine.runner import CommandResult, Runner
from harness.engine.single_test import (
    SCRIPT_PATH,
    SingleTestDriver,
    SingleTestError,
    _parse_menu,
    _verdict,
    validate_server_number,
)
from harness.inspect.decoder import Decoder
from harness.inspect.registry import make_collector

MAIN_MENU = """\
=== Single RTP L10 Main Menu ===
 1) INFO
 2) FAT
 3) STRESS
Select: """

FAT_MENU = """\
=== FAT Single Tests ===
 1) CPU_REG_TEST
 2) PCIE_LINK_TEST
 3) DIMM_TRAIN_TEST
Select: """

TEST_PASS = (
    "Running PCIE_LINK_TEST...\n"
    "  link width negotiated: x8\n"
    "Test Result: PASS\n"
    + FAT_MENU
)

TEST_FAIL = (
    "Running CPU_REG_TEST...\n"
    "  register readback mismatch\n"
    "Test Result: FAIL\n"
    + FAT_MENU
)

LAUNCH = f"{SCRIPT_PATH} 3"


class ScriptedShell:
    """Fake InteractiveShell protocol: a script of (expected_send, response).

    Each ``send_line`` consumes one script step (response is queued for the next
    ``read_until_quiet``); ``expected_send=None`` skips the match check.
    """

    def __init__(self, script):
        self.script = list(script)
        self.sent: list[str] = []
        self.closed = False
        self._queue: list[str] = []

    def open(self, banner_timeout=10.0) -> str:
        return ""

    def send_line(self, text: str) -> None:
        self.sent.append(text)
        if not self.script:
            raise AssertionError(f"unexpected send {text!r} (script exhausted)")
        expected, response = self.script.pop(0)
        if expected is not None and expected != text:
            raise AssertionError(f"expected send {expected!r}, got {text!r}")
        if response:
            self._queue.append(response)

    def send_raw(self, data: str) -> None:
        self.sent.append(data)

    def read_until_quiet(self, quiet_s, timeout) -> str:
        out = "".join(self._queue)
        self._queue = []
        return out

    def close(self) -> None:
        self.closed = True

    @property
    def alive(self) -> bool:
        return not self.closed


def _driver(script, **kw) -> tuple[SingleTestDriver, ScriptedShell]:
    shell = ScriptedShell(script)
    driver = SingleTestDriver(3, shell_factory=lambda: shell, **kw)
    return driver, shell


# ---- parsing / validation ----

def test_parse_menu_entries_and_dedupe():
    entries = _parse_menu(" 1) A\n 2) B\n 1) C\n3. D\n  4)  \nnot a line\n")
    assert [(e.number, e.label) for e in entries] == [(1, "A"), (2, "B"), (3, "D")]


def test_verdict():
    assert _verdict("Test Result: PASS") == "pass"
    assert _verdict("FAIL on lane 3") == "fail"
    assert _verdict("FAIL then PASS") == "fail"  # fail wins
    assert _verdict("no verdict markers") is None


def test_validate_server_number():
    assert validate_server_number("12") == 12
    assert validate_server_number(7) == 7
    for bad in ("0", "-1", "2; rm -rf /", "abc", "", "3 4", "1.5"):
        with pytest.raises(SingleTestError):
            validate_server_number(bad)


def test_ansi_normalization():
    assert normalize_terminal("\x1b[31m 1) A\x1b[0m\r\n\x1b[?25l") == " 1) A\n"


# ---- discover ----

def test_discover_navigates_to_fat_and_lists_tests():
    driver, shell = _driver([(LAUNCH, MAIN_MENU), ("2", FAT_MENU)])
    tests = driver.discover()

    assert [t.label for t in tests] == ["CPU_REG_TEST", "PCIE_LINK_TEST", "DIMM_TRAIN_TEST"]
    assert [t.number for t in tests] == [1, 2, 3]
    assert shell.sent[0] == LAUNCH
    # every menu selection after launch is a bare digit
    assert all(s.isdigit() for s in shell.sent[1:])
    assert driver.discovered


def test_discover_fat_absent_raises_with_excerpt():
    menu = "=== Menu ===\n 1) INFO\n 2) STRESS\nSelect: "
    driver, _shell = _driver([(LAUNCH, menu)])
    with pytest.raises(SingleTestError) as exc:
        driver.discover()
    assert "no FAT entry" in str(exc.value)
    assert "INFO" in str(exc.value)


def test_launch_sends_validated_server_number():
    shell = ScriptedShell([(f"{SCRIPT_PATH} 42", MAIN_MENU), ("2", FAT_MENU)])
    driver = SingleTestDriver(42, shell_factory=lambda: shell)
    driver.discover()
    assert shell.sent[0] == f"{SCRIPT_PATH} 42"


def test_bad_server_number_at_construction():
    with pytest.raises(SingleTestError):
        SingleTestDriver("2; rm -rf /", shell_factory=lambda: object())


# ---- run_test ----

def test_run_test_exact_label_and_verdict():
    driver, shell = _driver([
        (LAUNCH, MAIN_MENU), ("2", FAT_MENU),
        ("2", TEST_PASS),
    ])
    result = driver.run_test("PCIE_LINK_TEST")

    assert result.test == "PCIE_LINK_TEST"
    assert result.verdict == "pass"
    assert result.elapsed_s >= 0
    assert "link width" in result.output
    assert shell.sent[-1] == "2"


def test_run_test_fail_verdict():
    driver, _shell = _driver([
        (LAUNCH, MAIN_MENU), ("2", FAT_MENU),
        ("1", TEST_FAIL),
    ])
    result = driver.run_test("CPU_REG_TEST")
    assert result.verdict == "fail"


def test_run_test_unknown_label_lists_available():
    driver, _shell = _driver([(LAUNCH, MAIN_MENU), ("2", FAT_MENU)])
    with pytest.raises(SingleTestError) as exc:
        driver.run_test("BOGUS_TEST")
    assert "unknown single test" in str(exc.value)
    assert "CPU_REG_TEST" in str(exc.value)


def test_run_test_autodiscovers_without_prior_list():
    driver, shell = _driver([
        (LAUNCH, MAIN_MENU), ("2", FAT_MENU),
        ("2", TEST_PASS),
    ])
    result = driver.run_test("PCIE_LINK_TEST")
    assert result.verdict == "pass"
    assert shell.sent[0] == LAUNCH


def test_run_test_self_heals_when_menu_loops_to_main():
    # after the first run the menu returns to the MAIN menu, not the FAT list
    driver, shell = _driver([
        (LAUNCH, MAIN_MENU), ("2", FAT_MENU),
        ("2", MAIN_MENU),          # test 1 loops back to MAIN
        ("2", FAT_MENU),           # driver re-selects FAT
        ("1", TEST_FAIL),          # test 2 runs
    ])
    driver.run_test("PCIE_LINK_TEST")
    second = driver.run_test("CPU_REG_TEST")
    assert second.verdict == "fail"
    assert shell.sent.count("2") == 3  # FAT select + test select + re-select FAT


def test_run_test_after_shell_closed_relaunches():
    first = ScriptedShell([(LAUNCH, MAIN_MENU), ("2", FAT_MENU)])
    second = ScriptedShell([(LAUNCH, MAIN_MENU), ("2", FAT_MENU), ("1", TEST_FAIL)])
    shells = [first, second]
    driver = SingleTestDriver(3, shell_factory=lambda: shells.pop(0))
    driver.discover()
    first.closed = True  # the first session dies mid-run
    result = driver.run_test("CPU_REG_TEST")
    assert result.verdict == "fail"


def test_run_test_empty_menu_raises():
    driver, _shell = _driver([(LAUNCH, " 1) INFO\n 2) FAT\nSelect: "),
                              ("2", "=== FAT ===\nSelect: ")])
    with pytest.raises(SingleTestError) as exc:
        driver.run_test("PCIE_LINK_TEST")
    assert "no FAT single tests" in str(exc.value)


# ---- session engine integration ----

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


def _ctx(driver=None) -> EngineContext:
    return EngineContext(
        runner=FakeRunner(),
        decoder=Decoder(),
        collector_factory=make_collector,
        llm=_stub_diagnosis,
        supervisor=lambda label: None,
        single_test_driver=driver,
    )


def test_session_list_then_run_then_diagnosis():
    driver, _shell = _driver([(LAUNCH, MAIN_MENU), ("2", FAT_MENU),
                              ("2", TEST_PASS)])
    llm = ScriptedLLM(
        {"kind": "test", "single_test": {"action": "list"}},
        {"kind": "test", "single_test": {"action": "run", "test": "PCIE_LINK_TEST"}},
        {"kind": "diagnosis", "diagnosis": _stub_diagnosis().model_dump()},
    )
    engine = SessionEngine(_ctx(driver), llm=llm, max_turns=4)
    diag = engine.run("MCE uncorrectable ECC error")

    assert diag.diagnosis == "Memory ECC errors on DIMM_A2"
    kinds = [t["kind"] for t in engine.transcript]
    assert kinds == ["test", "test", "diagnosis"]
    list_msg = engine.transcript[0]["content"]
    assert "FAT single tests available" in list_msg
    assert "PCIE_LINK_TEST" in list_msg
    run_msg = engine.transcript[1]["content"]
    assert "[test result] PCIE_LINK_TEST: pass" in run_msg
    # the test list + result are visible in the next LLM call
    assert "PCIE_LINK_TEST" in llm.calls[2][-1]["content"]
    assert "FAT Single Tests" in llm.calls[2][-1]["content"]


def test_session_tool_messages_use_test_prefix():
    driver, _shell = _driver([(LAUNCH, MAIN_MENU), ("2", FAT_MENU),
                              ("2", TEST_PASS)])
    llm = ScriptedLLM(
        {"kind": "test", "single_test": {"action": "run", "test": "PCIE_LINK_TEST"}},
        {"kind": "diagnosis", "diagnosis": _stub_diagnosis().model_dump()},
    )
    engine = SessionEngine(_ctx(driver), llm=llm, max_turns=3)
    engine.run("MCE uncorrectable ECC error")
    rendered = [m["content"] for m in llm.calls[1]
                if m["role"] == "user"]
    assert any("[test result] PCIE_LINK_TEST: pass" in c for c in rendered)


def test_session_driver_unavailable_is_soft():
    llm = ScriptedLLM(
        {"kind": "test", "single_test": {"action": "list"}},
        {"kind": "diagnosis", "diagnosis": _stub_diagnosis().model_dump()},
    )
    engine = SessionEngine(_ctx(driver=None), llm=llm, max_turns=3)
    engine.run("MCE uncorrectable ECC error")
    assert engine.transcript[0]["content"] == (
        "single tests unavailable (requires --server-number and an SSH target)")
    assert engine.transcript[0]["kind"] == "test"


def test_session_non_gb_platform_gated(monkeypatch):
    from harness.diagnosis import session as session_mod

    driver, _shell = _driver([(LAUNCH, MAIN_MENU), ("2", FAT_MENU)])
    monkeypatch.setattr(session_mod, "family_for", lambda key: "spray")
    llm = ScriptedLLM(
        {"kind": "test", "single_test": {"action": "list"}},
        {"kind": "diagnosis", "diagnosis": _stub_diagnosis().model_dump()},
    )
    engine = SessionEngine(_ctx(driver), llm=llm, max_turns=3)
    engine.run("MCE uncorrectable ECC error")
    contents = [t["content"] for t in engine.transcript]
    assert any("single tests unavailable on platform 'spray'" in c
               for c in contents)


def test_session_malformed_test_turn_skipped():
    driver, _shell = _driver([(LAUNCH, MAIN_MENU), ("2", FAT_MENU)])
    llm = ScriptedLLM(
        {"kind": "test"},  # no single_test payload
        {"kind": "diagnosis", "diagnosis": _stub_diagnosis().model_dump()},
    )
    engine = SessionEngine(_ctx(driver), llm=llm, max_turns=3)
    engine.run("MCE uncorrectable ECC error")
    assert any("[note] malformed agent response" in t["content"]
               for t in engine.transcript)


# ---- CLI wiring ----

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
)


def test_run_diagnose_session_uses_override_driver(tmp_path):
    from harness.operator.cli import build_parser, run_diagnose

    inv = tmp_path / "inventory.yaml"
    inv.write_text(_INVENTORY, encoding="utf-8")

    driver, _shell = _driver([(LAUNCH, MAIN_MENU), ("2", FAT_MENU),
                              ("2", TEST_PASS)])
    llm = ScriptedLLM(
        {"kind": "test", "single_test": {"action": "run", "test": "PCIE_LINK_TEST"}},
        {"kind": "diagnosis", "diagnosis": _stub_diagnosis().model_dump()},
    )
    args = build_parser().parse_args([
        "diagnose", "--inventory", str(inv), "--host", "h1",
        "--symptom", "MCE uncorrectable ECC error",
        "--out-dir", str(tmp_path / "runs"), "--llm", "stub",
        "--interactive", "--server-number", "3",
    ])
    diag = run_diagnose(args, overrides={
        "session": FakeRunner(),
        "llm": llm,
        "human_input": lambda q: "",
        "single_test_driver": driver,
    })

    assert diag.diagnosis == "Memory ECC errors on DIMM_A2"
    run_dir = next((tmp_path / "runs").iterdir())
    transcript = json.loads(
        (run_dir / "transcript.json").read_text(encoding="utf-8"))
    test_msgs = [t for t in transcript if t.get("kind") == "test"]
    assert any("[test result] PCIE_LINK_TEST: pass" in t["content"]
               for t in test_msgs)