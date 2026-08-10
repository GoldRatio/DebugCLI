"""Execution results and the read-only command runner.

``Runner`` is the single funnel for executing commands against the target. It
enforces (1) the allowlist and (2) the hard read-only security gate for every
invocation, then captures stdout, stderr, exit code, and timing.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass

from .allowlist import AllowPolicy
from .security_gate import ReadOnlyViolation
from .security_gate import check as security_check


@dataclass
class CommandResult:
    argv: list[str]
    stdout: str
    stderr: str
    exit_code: int
    elapsed_ms: int

    @property
    def ok(self) -> bool:
        return self.exit_code == 0

    @property
    def stdout_sha(self) -> str:
        import hashlib
        return hashlib.sha256(self.stdout.encode()).hexdigest()


class Runner:
    """Local-exec read-only runner.

    This base runs via ``subprocess.run`` (used for local-lab / trusted-lab where
    the harness itself is on the target, or for test doubles). A remote ``SSHRunner``
    subclass overrides ``_exec`` to run parameterized over the session.
    """

    #: True for runners that drive a BMC serial console (node BusyBox shell)
    #: instead of a host OS. Collectors use it to pick BMC-appropriate probes.
    is_console: bool = False

    def __init__(self, policy: AllowPolicy, force_read_only: bool = True) -> None:
        self.policy = policy
        self.force_read_only = force_read_only
        self.calls: list[CommandResult] = []

    def execute(self, argv: list[str], timeout: float = 30.0) -> CommandResult:
        if self.force_read_only:
            security_check(argv)  # hard no-write guarantee, always on
        if not self.policy.allows(argv):
            raise ReadOnlyViolation(argv, "not in allowlist")
        start = time.monotonic()
        result = self._exec(argv, timeout)
        result.elapsed_ms = int((time.monotonic() - start) * 1000)
        self.calls.append(result)
        return result

    def _exec(self, argv: list[str], timeout: float) -> CommandResult:
        proc = subprocess.run(argv, capture_output=True, text=True,
                              timeout=timeout, shell=False, check=False)
        return CommandResult(
            argv=list(argv),
            stdout=proc.stdout,
            stderr=proc.stderr,
            exit_code=proc.returncode,
            elapsed_ms=0,
        )