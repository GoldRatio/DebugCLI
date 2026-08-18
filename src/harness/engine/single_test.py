"""FAT single-test driver over an interactive SSH shell.

The GB models expose a vendor single-test menu: ``/mnt/smbfs/single_rtp_l10.sh
ServerNumber`` pops a number-driven menu; selecting ``FAT`` (Functional
Acceptance Tests) reveals a list of single tests, each of which runs in well
under a minute and then returns to the menu (so several can be run in one
session). ``SingleTestDriver`` walks that menu for a diagnostic session: it
discovers the FAT test list and runs a specific test by exact label.

Safety rails (mirroring the deny-by-default posture everywhere else in the
harness):

* the script path is a fixed constant;
* the server number must be a positive integer (a validated ``ServerNumber``);
* the ONLY things ever written to the wire are the launch line, bare digit
  strings for menu selections, and exit/control sequences.

The harness never sends free text, so a hostile or malformed menu cannot be
injected into -- it can only be reported.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

SCRIPT_PATH = "/mnt/smbfs/single_rtp_l10.sh"
CATEGORY = "FAT"

_DIGITS = re.compile(r"\d+")
_MENU_ENTRY = re.compile(
    r"^\s*(\d{1,3})[.)][ \t]*(\S[^\r\n]*?)[ \t]*$", re.MULTILINE)
_FAIL_RE = re.compile(r"\b(FAIL(?:ED|URE)?)\b")
_PASS_RE = re.compile(r"\b(PASS(?:ED)?|SUCCESS(?:FUL)?)\b")

_DEFAULT_RUN_TIMEOUT = 300.0
_DEFAULT_QUIET_S = 3.0


class SingleTestError(RuntimeError):
    pass


@dataclass(frozen=True)
class SingleTest:
    number: int
    label: str


@dataclass
class SingleTestResult:
    test: str
    output: str
    verdict: str | None  # "pass" | "fail" | None (unrecognized)
    elapsed_s: float


def validate_server_number(value) -> int:
    """Return a validated positive integer server number or raise."""
    text = str(value).strip()
    if not _DIGITS.fullmatch(text):
        raise SingleTestError(
            f"server number must be a positive integer, got {value!r}")
    number = int(text)
    if number <= 0:
        raise SingleTestError(f"server number must be positive, got {number}")
    return number


def _parse_menu(text: str) -> list[SingleTest]:
    """Numbered entries ``N) Label`` / ``N. Label``, deduped by number."""
    seen: dict[int, str] = {}
    for number_s, label in _MENU_ENTRY.findall(text):
        number = int(number_s)
        clean = label.strip().strip(":-").strip()
        if clean and number not in seen:
            seen[number] = clean
    return [SingleTest(number=n, label=l) for n, l in seen.items()]


def _verdict(text: str) -> str | None:
    if _FAIL_RE.search(text):
        return "fail"
    if _PASS_RE.search(text):
        return "pass"
    return None


def _tail(text: str, limit: int = 2000) -> str:
    return text[-limit:] if len(text) > limit else text


class SingleTestDriver:
    """Walk the vendor FAT single-test menu on one target.

    ``shell_factory`` must return an open object exposing ``send_line``,
    ``send_raw``, ``read_until_quiet(quiet_s, timeout)``, ``close`` and an
    ``alive`` attribute -- the production factory wraps ``InteractiveShell``.
    """

    def __init__(
        self,
        server_number,
        shell_factory: Callable[[], object],
        *,
        run_timeout: float = _DEFAULT_RUN_TIMEOUT,
        menu_quiet_s: float = _DEFAULT_QUIET_S,
        progress: Callable[[str], None] | None = None,
        artifact_dir=None,
    ) -> None:
        self.server_number = validate_server_number(server_number)
        self._shell_factory = shell_factory
        self._run_timeout = run_timeout
        self._menu_quiet_s = menu_quiet_s
        self._progress = progress
        self._artifact_dir = artifact_dir
        self._shell: object | None = None
        self._fat_number: int | None = None
        self._tests: list[SingleTest] = []
        self._last_menu_text = ""
        self.runs: list[SingleTestResult] = []

    # ---- public ----

    @property
    def tests(self) -> list[SingleTest]:
        return list(self._tests)

    @property
    def discovered(self) -> bool:
        return bool(self._tests)

    def discover(self) -> list[SingleTest]:
        """Launch the script and navigate to the FAT single-test list."""
        self._launch()
        return self.tests

    def run_test(self, label: str) -> SingleTestResult:
        """Run a single test by exact label; returns the captured output."""
        label = label.strip()
        if not self.discovered:
            self._launch()
        if not self._tests:
            raise SingleTestError("no FAT single tests discovered in the menu")
        match = next((t for t in self._tests
                      if t.label.strip().lower() == label.lower()), None)
        if match is None:
            available = ", ".join(t.label for t in self._tests)
            raise SingleTestError(
                f"unknown single test {label!r}; available FAT tests: "
                f"{available or '(none)'}")
        self._ensure_fat_menu()
        self._emit(f"single test: {match.label}")
        start = time.monotonic()
        self._send_number(match.number)
        output = self._shell.read_until_quiet(
            self._menu_quiet_s, self._run_timeout)
        result = SingleTestResult(
            test=match.label,
            output=output,
            verdict=_verdict(output),
            elapsed_s=time.monotonic() - start,
        )
        self.runs.append(result)
        self._remember_menu(output)
        self._append_artifact(f"\n== run {match.label} ({result.verdict}) ==\n{output}")
        return result

    def close(self) -> None:
        """Best-effort teardown: Ctrl-C out of the menu, then close the shell."""
        shell = self._shell
        if shell is None:
            return
        with suppress(Exception):
            if getattr(shell, "alive", False):
                shell.send_raw("\x03")
                shell.send_raw("\x03")
                shell.send_line("exit")
        with suppress(Exception):
            shell.close()
        self._shell = None

    # ---- internals ----

    def _emit(self, text: str) -> None:
        if self._progress is not None:
            self._progress(text)

    def _append_artifact(self, text: str) -> None:
        if self._artifact_dir is None:
            return
        try:
            path = Path(self._artifact_dir) / "single_test_transcript.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(text + "\n")
        except OSError:
            pass  # artifacts are best-effort; the result is still in the transcript

    def _send_number(self, number: int) -> None:
        if not _DIGITS.fullmatch(str(number)):
            raise SingleTestError(f"menu selections must be digits, got {number!r}")
        self._shell.send_line(str(number))

    def _launch(self) -> None:
        self.close()
        shell = self._shell_factory()
        self._shell = shell
        # The launch line is built only from a fixed constant and a validated
        # positive integer, so no shell metacharacters can reach the wire.
        launch = f"{SCRIPT_PATH} {self.server_number}"
        self._append_artifact(f"\n== launch {launch} ==")
        self._emit(f"single tests: launching {SCRIPT_PATH} {self.server_number}")
        shell.send_line(launch)
        main_menu = shell.read_until_quiet(self._menu_quiet_s, self._run_timeout)
        self._append_artifact(main_menu)
        self._remember_menu(main_menu)

        entries = _parse_menu(main_menu)
        fat = next((e for e in entries
                    if CATEGORY in e.label.upper()), None)
        if fat is None:
            raise SingleTestError(
                f"no FAT entry in single-test menu; entries: "
                f"{', '.join(e.label for e in entries) or '(none)'} "
                f"| menu tail: {_tail(main_menu)!r}")
        self._fat_number = fat.number
        self._send_number(fat.number)
        submenu = shell.read_until_quiet(self._menu_quiet_s, self._run_timeout)
        self._append_artifact(f"\n== FAT submenu ({fat.label}) ==\n{submenu}")
        self._remember_menu(submenu)
        self._tests = _parse_menu(submenu)

    def _ensure_fat_menu(self) -> None:
        """Re-establish position at the FAT submenu (it may loop back anywhere)."""
        if not getattr(self._shell, "alive", False):
            self._launch()
            return
        current = _parse_menu(self._last_menu_text)
        known_labels = {t.label.strip().lower() for t in self._tests}
        if any(e.label.strip().lower() in known_labels for e in current):
            return  # already at the FAT submenu
        if any(CATEGORY in e.label.upper() for e in current):
            # looped back to the MAIN menu: re-select FAT
            fat = next(e for e in current if CATEGORY in e.label.upper())
            self._send_number(fat.number)
            submenu = self._shell.read_until_quiet(
                self._menu_quiet_s, self._run_timeout)
            self._remember_menu(submenu)
            self._append_artifact(f"\n== re-select {fat.label} ==\n{submenu}")
            return
        self._launch()

    def _remember_menu(self, text: str) -> None:
        self._last_menu_text = _tail(text, 4000)