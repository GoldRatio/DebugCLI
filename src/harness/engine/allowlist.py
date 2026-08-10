"""Command allowlist: exact argv templates.

An allowlist entry pins the full argv shape for a read-only invocation: program
path is literal, option flags are literal, and only positions marked ``"*"`` accept
a value (e.g. a device path). This is the sudoers-aligned model the spec calls
"exact argv templates". Arguments never cross a shell -- execution is parameterized.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib


@dataclass(frozen=True)
class AllowRule:
    """A pinned read-only invocation template: exact tokens, ``"*"`` for a single value."""

    program: str                      # exact program path, e.g. /bin/smartctl
    args: tuple[str, ...] = ()        # exact arg tokens; "*" matches any single positional

    def matches(self, argv: list[str]) -> bool:
        if not argv:
            return False
        if argv[0] != self.program:
            return False
        if len(argv) - 1 != len(self.args):
            return False
        for got, expected in zip(argv[1:], self.args):
            if expected != "*" and got != expected:
                return False
        return True


class AllowPolicy:
    def __init__(self, rules: list[AllowRule] | None = None) -> None:
        self._rules: list[AllowRule] = list(rules or [])

    def add(self, rule: AllowRule) -> None:
        self._rules.append(rule)

    def allows(self, argv: list[str]) -> bool:
        return any(r.matches(argv) for r in self._rules)

    def rules(self) -> list[AllowRule]:
        return list(self._rules)

    @lru_cache(maxsize=1)
    def fingerprint(self) -> str:
        body = "".join(f"{r.program}|{' '.join(r.args)}\n" for r in self._rules)
        return hashlib.sha256(body.encode()).hexdigest()


# Read-only probes only. Note the "*" positions: a single device path, register, etc.
# Destructive programs (dd, mkfs, shutdown, setpci write forms, flashrom, ...) are
# ABSENT here by design and additionally hard-blocked in security_gate.
_default_rules = [
    AllowRule("/bin/smartctl", ("-a", "*")),
    AllowRule("/bin/smartctl", ("-x", "*")),
    AllowRule("/usr/sbin/ipmitool", ("sensor",)),
    AllowRule("/usr/sbin/ipmitool", ("sel", "list")),
    AllowRule("/usr/sbin/ipmitool", ("fru", "print")),
    AllowRule("/usr/bin/rdmsr", ("-a",)),
    AllowRule("/sbin/modprobe", ("msr",)),
    AllowRule("/usr/bin/lspci", ("-xxx",)),
    AllowRule("/bin/dmidecode", ()),
    AllowRule("/bin/dmesg", ("-l", "*")),
    AllowRule("/bin/lsblk", ()),
]


def default_policy() -> AllowPolicy:
    return AllowPolicy(_default_rules)