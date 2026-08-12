"""BMC console collectors: read-only probes that exist on the node's BMC shell.

``start serial session -i <cable> -p 2200`` drops into the BMC's BusyBox Linux
shell, NOT the host OS -- so ``rdmsr``/``smartctl``/``lspci``/``dmidecode`` are
absent. The read-only evidence available there is ``ipmitool`` (sensor/sel/fru),
the BusyBox ``dmesg -r`` ring buffer, and the I2C bus. Probes that need privilege
are wrapped in ``sudo -S`` so the console script feeds the vault password at the
``[sudo] password for`` prompt.

BMC images differ in which I2C tools they ship (``i2cdump``/``i2ctransfer``/...),
so the SWB CPLD boot-state dump is a fallback chain: try the documented
``i2cdump -y 8 0xb`` first, then ``i2ctransfer`` block read. Each candidate must
actually produce register rows; a candidate whose tool is missing (``command not
found``) or whose read fails is recorded honestly and the next one is tried.
"""

from __future__ import annotations

import re

from ..base import RegisterDump
from ..decoder import _after_echo
from . import Collector

_BMC_PROBES: dict[str, list[tuple[str, str]]] = {
    "cpu": [
        ("sudo -S ipmitool sensor list", "sensor"),
        # SWB CPLD boot-state block: bus 8 = BMC MM8, 7-bit addr 0xb (8-bit
        # 0x16). GB_HangUp_troubleshooting_v1.2.pdf p.1: "For EVERY Amber Light
        # issue, dump i2cdump -y 8 0xb to get the boot state"; the C4A14 SWB
        # CPLD spec maps register 0x1b (CPU_BOOT_DONE, RUN_POWER_PG ...) and
        # 0xa1 (power-sequence FSM). Read-only; allowlisted in sol.py. The
        # fallback chain (below) covers BMCs without i2cdump.
        ("sudo -S i2cdump -y 8 0xb", "i2c"),
        ("i2cdump -y 8 0xb", "i2c"),                                    # no-sudo fallback
        ("sudo -S i2ctransfer -y 8 w1@0xb 0x00 r256", "i2c"),           # i2c-tools v4+
        ("i2ctransfer -y 8 w1@0xb 0x00 r256", "i2c"),                   # no-sudo fallback
    ],
    "kernel": [("sudo -S ipmitool sel list", "sel"), ("dmesg -r", "dmesg")],
    "ipmi": [
        ("sudo -S ipmitool sensor list", "sensor"),
        ("sudo -S ipmitool sel list", "sel"),
        ("sudo -S ipmitool fru print", "fru"),
    ],
}

_I2C_ROW_RE = re.compile(r"^\s*[0-9a-fA-F]{2}\s*:")
_HEX_BYTE_RE = re.compile(r"(?:0x)?([0-9a-fA-F]{2})\b")


def _has_register_output(cmd: str, stdout: str) -> bool:
    """True when the probe output really contains register data.

    Guards the CPLD chain: ``i2cdump`` must show ``<offset>:`` rows and
    ``i2ctransfer`` must show a read-data block after the command echo. A tool
    that did not run, or a read that returned nothing, is not a successful dump.
    """
    if "not found" in stdout or "No such file" in stdout:
        return False
    if "i2cdump" in cmd:
        return any(_I2C_ROW_RE.match(line) for line in stdout.splitlines())
    if "i2ctransfer" in cmd:
        body = _after_echo(stdout, cmd)
        return len(_HEX_BYTE_RE.findall(body)) >= 16
    return True


def _kind_for(cmd: str) -> str:
    if "sel list" in cmd:
        return "sel"
    if "sensor list" in cmd:
        return "sensor"
    if "fru print" in cmd:
        return "fru"
    if "dmesg" in cmd:
        return "dmesg"
    if "i2cdump" in cmd or "i2ctransfer" in cmd or "i2cget" in cmd:
        return "i2c"
    return "other"


class BmcConsoleCollector(Collector):
    """Run BMC-shell probes for one subsystem over a console runner."""

    subsystem = "bmc"

    def __init__(self, runner, subsystem: str = "ipmi") -> None:
        super().__init__(runner)
        self.subsystem = subsystem

    def candidate_probes(self) -> list[str]:
        """Probes this collector needs this pass: non-i2c probes plus the FIRST
        CPLD-chain candidate. Chain fallbacks run individually only when the
        first candidate fails, so a healthy BMC costs one console session for
        the whole plan."""
        non_i2c = [cmd for cmd, kind in _BMC_PROBES[self.subsystem]
                   if kind != "i2c"]
        first_i2c = next((cmd for cmd, kind in _BMC_PROBES[self.subsystem]
                          if kind == "i2c"), None)
        return non_i2c + ([first_i2c] if first_i2c is not None else [])

    def collect(self, **kwargs) -> list[RegisterDump]:
        # Skip probes this run already executed successfully (the generic plan
        # overlaps: sensor list in cpu+ipmi, sel list in kernel+ipmi, and
        # detect_model already ran fru print). One console round-trip ~30s.
        done = {" ".join(c.argv) for c in getattr(self.runner, "calls", []) if c.ok}
        pending = [(cmd, kind) for cmd, kind in _BMC_PROBES[self.subsystem]
                   if cmd not in done]
        if not pending:
            return []

        def _dump(cmd: str, kind: str, result) -> RegisterDump:
            return RegisterDump(
                subsystem=self.subsystem,
                source=cmd,
                raw=result.stdout,
                cmd_argv=[cmd],
                ok=result.ok,
                meta={"exit": result.exit_code, "elapsed_ms": result.elapsed_ms,
                      "kind": kind},
            )

        # CPLD dump chain: stop at the first candidate that produced real
        # register rows; failed candidates stay in the record as evidence.
        batch = [p for p in pending if p[1] != "i2c"]
        chain = [p for p in pending if p[1] == "i2c"]
        if batch and hasattr(self.runner, "batch_execute"):
            results = self.runner.batch_execute([c for c, _ in batch])
            dumps = [_dump(c, k, r) for (c, k), r in zip(batch, results)]
        else:
            dumps = []
            for cmd, kind in batch:
                result = self.runner.execute([cmd])
                dumps.append(_dump(cmd, kind, result))
        for cmd, kind in chain:
            if kind == "i2c" and any(
                    d.meta.get("kind") == "i2c" and d.ok
                    and _has_register_output(d.source, d.raw) for d in dumps):
                break
            result = self.runner.execute([cmd])
            dumps.append(_dump(cmd, kind, result))
            if kind == "i2c" and result.ok and _has_register_output(cmd, result.stdout):
                break
        return dumps
