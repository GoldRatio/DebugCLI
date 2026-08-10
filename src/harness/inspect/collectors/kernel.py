"""Kernel / panic log collector: dmesg, pstore, sysrq (read-only)."""

from __future__ import annotations

from ..base import RegisterDump
from . import Collector


class KernelCollector(Collector):
    subsystem = "kernel"

    def collect(self, **kwargs) -> list[RegisterDump]:
        result = self.runner.execute(["/bin/dmesg", "-l", "err, crit, alert, emerg"])
        return [RegisterDump(
            subsystem=self.subsystem,
            source="dmesg -l",
            raw=result.stdout,
            cmd_argv=["/bin/dmesg", "-l"],
            ok=result.ok,
            meta={"exit": result.exit_code, "elapsed_ms": result.elapsed_ms},
        )]