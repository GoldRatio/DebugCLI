"""Storage collector: smartctl -a/-x, nvme (read-only forms)."""

from __future__ import annotations

from ..base import RegisterDump
from . import Collector


class StorageCollector(Collector):
    subsystem = "storage"

    def collect(self, devices: list[str] | None = None, **kwargs) -> list[RegisterDump]:
        devices = devices or ["/dev/sda"]
        dumps = []
        for dev in devices:
            for flag in ("-a", "-x"):
                argv = ["/bin/smartctl", flag, dev]
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