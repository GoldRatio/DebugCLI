"""CPU MSR collector: rdmsr / turbostat / cpuid (read-only)."""

from __future__ import annotations

from ..base import RegisterDump
from . import Collector


class CpuMsrCollector(Collector):
    subsystem = "cpu"

    def collect(self, **kwargs) -> list[RegisterDump]:
        dumps = []
        for argv in (["/usr/bin/rdmsr", "-a"], ["/bin/dmesg", "-l", "err"]):
            result = self.runner.execute(argv)
            dumps.append(RegisterDump(
                subsystem=self.subsystem,
                source=" ".join(argv),
                raw=result.stdout,
                cmd_argv=list(argv),
                ok=result.ok,
                meta={"exit": result.exit_code, "elapsed_ms": result.elapsed_ms},
            ))
        return dumps