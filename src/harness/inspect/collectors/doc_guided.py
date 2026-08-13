"""Doc-guided probe collector: run read-only commands mined from troubleshooting docs.

The profile heuristic picks collector profiles but cannot know which registers a
failure mode needs. Doc snippets name the exact commands (``i2cdump -y 8 0xb``,
``ipmitool fru print``, ...); this collector executes the mined commands through the
read-only runner, mirroring the BMC console collector (skip probes already run this
session, tag dumps with an evidence kind, never bypass the runner).
"""

from __future__ import annotations

from ...engine.runner import CommandResult
from ...engine.security_gate import ReadOnlyViolation
from ..base import RegisterDump
from . import Collector

# BMC BusyBox shell needs `sudo -S` for privileged tools; plain `dmesg` does not.
_PRIVILEGED = ("i2cdump", "i2cget", "i2cdetect", "ipmitool")


def _kind_for(cmd: str) -> str:
    if "sel " in cmd:
        return "sel"
    if "sensor " in cmd or "sdr" in cmd:
        return "sensor"
    if "fru print" in cmd:
        return "fru"
    if "dmesg" in cmd:
        return "dmesg"
    if "i2cdump" in cmd or "i2cget" in cmd or "i2cdetect" in cmd:
        return "i2c"
    return "other"


class DocGuidedProbeCollector(Collector):
    """Run doc-mined probes, skipping any already executed successfully this run."""

    subsystem = "doc"

    def __init__(self, runner, commands: list[str]) -> None:
        super().__init__(runner)
        self.commands = commands

    def collect(self, **kwargs) -> list[RegisterDump]:
        # Reuse results the runner already recorded (the plan-level pre-batch
        # runs doc-named probes in the SAME one-session batch): a probe that
        # already ran must become evidence, not be skipped. Only commands not
        # yet attempted execute now.
        prior = {" ".join(c.argv): c for c in getattr(self.runner, "calls", [])}
        is_console = bool(getattr(self.runner, "is_console", False))
        dumps = []
        for cmd in self.commands:
            final = cmd
            prog = cmd.split()[0].split("/")[-1]
            if (is_console and prog in _PRIVILEGED
                    and not cmd.startswith("sudo -S ")):
                final = "sudo -S " + cmd
            if final in prior:
                result = prior[final]
            else:
                try:
                    result = self.runner.execute([final])
                except ReadOnlyViolation as exc:
                    # Host runner: probe not in the allowlist. Record it as a failed
                    # dump (like ConsoleRunner does for denied probes) so the pipeline
                    # continues instead of crashing.
                    result = CommandResult(argv=[final], stdout="", stderr=str(exc),
                                           exit_code=2, elapsed_ms=0)
            dumps.append(RegisterDump(
                subsystem=self.subsystem,
                source=final,
                raw=result.stdout,
                cmd_argv=[final],
                ok=result.ok,
                meta={"exit": result.exit_code, "elapsed_ms": result.elapsed_ms,
                      "kind": _kind_for(final)},
            ))
        return dumps
