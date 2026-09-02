"""FAT single-test driver unit tests.

The driver is exercised against a scripted interactive shell (no SSH). It is
kept intact for a future debug-REPL ``test`` tool; nothing wires it into the
one-shot pipeline anymore.
"""

import pytest

from harness.engine.interactive import normalize_terminal
from harness.engine.single_test import (
    SCRIPT_PATH,
    SingleTestDriver,
    SingleTestError,
    _parse_menu,
    _verdict,
    validate_server_number,
)

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
