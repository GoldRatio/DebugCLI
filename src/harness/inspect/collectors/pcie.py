"""PCIe config space collector (read-only): lspci -xxx / sysfs."""

from __future__ import annotations

from ..base import RegisterDump
from . import Collector


class PcieCollector(Collector):
    subsystem = "pcie"

    def collect(self, **kwargs) -> list[RegisterDump]:
        result = self.runner.execute(["/usr/bin/lspci", "-xxx"])
        return [RegisterDump(
            subsystem=self.subsystem,
            source="lspci -xxx",
            raw=result.stdout,
            cmd_argv=["/usr/bin/lspci", "-xxx"],
            ok=result.ok,
            meta={"exit": result.exit_code, "elapsed_ms": result.elapsed_ms},
        )]