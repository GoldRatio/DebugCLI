"""BMC/IPMI channel: ipmitool over LAN with the SEPARATE BMC credential domain.

The BMC domain never reuses the OS identity. Commands run through the same
``Runner`` contract (allowlist + hard read-only gate) so the single-choke-point
invariant holds here too; only the BMC allowlist templates differ because the
LAN form pins ``-H <host> -U <user> -E``.

The password is passed via the ``IPMI_PASSWORD`` environment variable
(``ipmitool -E``), never as an argv token, so it cannot appear in a process
listing or in the command trace.
"""

from __future__ import annotations

import os
import subprocess

from .allowlist import AllowPolicy, AllowRule
from .runner import CommandResult, Runner


def bmc_policy() -> AllowPolicy:
    """Exact read-only ipmitool LAN templates (``-E`` reads pw from env)."""
    return AllowPolicy([
        AllowRule("/usr/bin/ipmitool", ("-I", "lanplus", "-H", "*", "-U", "*", "-E", "sensor")),
        AllowRule("/usr/bin/ipmitool", ("-I", "lanplus", "-H", "*", "-U", "*", "-E", "sel", "list")),
        AllowRule("/usr/bin/ipmitool", ("-I", "lanplus", "-H", "*", "-U", "*", "-E", "fru", "print")),
    ])


class BmcRunner(Runner):
    """Local-exec runner pointed at the BMC channel (ipmitool over LAN)."""

    def __init__(self, address: str, username: str, password: str,
                 policy: AllowPolicy | None = None, force_read_only: bool = True) -> None:
        super().__init__(policy or bmc_policy(), force_read_only=force_read_only)
        self.address = address
        self.username = username
        self._password = password

    def _exec(self, argv: list[str], timeout: float) -> CommandResult:
        env = dict(os.environ)
        env["IPMI_PASSWORD"] = self._password
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout,
            shell=False, env=env, check=False,
        )
        return CommandResult(
            argv=list(argv),
            stdout=proc.stdout,
            stderr=proc.stderr,
            exit_code=proc.returncode,
            elapsed_ms=0,
        )
