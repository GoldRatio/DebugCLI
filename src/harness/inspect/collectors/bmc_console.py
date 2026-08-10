"""BMC console collectors: read-only probes that exist on the node's BMC shell.

``start serial session -i <cable> -p 2200`` drops into the BMC's BusyBox Linux
shell, NOT the host OS -- so ``rdmsr``/``smartctl``/``lspci``/``dmidecode`` are
absent. The read-only evidence available there is ``ipmitool`` (sensor/sel/fru),
the BusyBox ``dmesg -r`` ring buffer, and the I2C bus. Probes that need privilege
are wrapped in ``sudo -S`` so the console script feeds the vault password at the
``[sudo] password for`` prompt.
"""

from __future__ import annotations

from ..base import RegisterDump
from . import Collector

_BMC_PROBES: dict[str, list[str]] = {
    "cpu": ["sudo -S ipmitool sensor list"],
    "kernel": ["sudo -S ipmitool sel list", "dmesg -r"],
    "ipmi": [
        "sudo -S ipmitool sensor list",
        "sudo -S ipmitool sel list",
        "sudo -S ipmitool fru print",
    ],
}


class BmcConsoleCollector(Collector):
    """Run BMC-shell probes for one subsystem over a console runner."""

    subsystem = "bmc"

    def __init__(self, runner, subsystem: str = "ipmi") -> None:
        super().__init__(runner)
        self.subsystem = subsystem

    def collect(self, **kwargs) -> list[RegisterDump]:
        # Skip probes this run already executed successfully (the generic plan
        # overlaps: sensor list in cpu+ipmi, sel list in kernel+ipmi, and
        # detect_model already ran fru print). One console round-trip ~30s.
        done = {" ".join(c.argv) for c in getattr(self.runner, "calls", []) if c.ok}
        dumps = []
        for cmd in _BMC_PROBES[self.subsystem]:
            if cmd in done:
                continue
            result = self.runner.execute([cmd])
            dumps.append(RegisterDump(
                subsystem=self.subsystem,
                source=cmd,
                raw=result.stdout,
                cmd_argv=[cmd],
                ok=result.ok,
                meta={"exit": result.exit_code, "elapsed_ms": result.elapsed_ms},
            ))
        return dumps
