"""IPMI / BMC collector: sensor, sel list, fru print.

Runs over the SEPARATE BMC credential domain (never the OS identity). Commands are
read-only ipmitool queries and pass through the same allowlist runner; the runner
here should be pointed at the BMC channel, not the OS SSH channel.
"""

from __future__ import annotations

from ..base import RegisterDump
from . import Collector

_BMC_ARGS: list[tuple[list[str], str]] = [
    (["/usr/sbin/ipmitool", "sensor"], "sensor"),   # live sensor readings = current state
    (["/usr/sbin/ipmitool", "sel", "list"], "sel"),  # System Event Log = HISTORICAL records
    (["/usr/sbin/ipmitool", "fru", "print"], "fru"),
]


class IpmiCollector(Collector):
    subsystem = "bmc"

    def collect(self, **kwargs) -> list[RegisterDump]:
        dumps = []
        for argv, kind in _BMC_ARGS:
            result = self.runner.execute(argv)
            dumps.append(RegisterDump(
                subsystem=self.subsystem,
                source=" ".join(argv),
                raw=result.stdout,
                cmd_argv=list(argv),
                ok=result.ok,
                meta={"exit": result.exit_code, "elapsed_ms": result.elapsed_ms,
                      "kind": kind},
            ))
        return dumps