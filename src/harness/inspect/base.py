"""Typed register-dump models shared across collectors, decoder, and diagnosis.

``RegisterDump`` is the output of a collector: raw, timestamped, with the command
that produced it. ``RegisterDecode`` is the decoder output for one register:
human-readable fields with bit ranges and meanings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class RegisterDump:
    subsystem: str
    source: str                      # e.g. "rdmsr -a", "ipmitool sensor"
    raw: str
    cmd_argv: list[str]
    ts: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    ok: bool = True
    meta: dict = field(default_factory=dict)

    @property
    def register_name(self) -> str | None:
        # best-effort: extract an MSR/register mnemonic from source line
        for line in self.raw.splitlines()[:1]:
            return line.split()[0] if line.split() else None
        return None


@dataclass
class DecodedField:
    name: str
    msb: int | None
    lsb: int | None
    raw_value: str
    meaning: str | None = None


@dataclass
class RegisterDecode:
    mnemonic: str
    raw_hex: str
    decoded_fields: list[DecodedField] = field(default_factory=list)
    catalog_version: str | None = None
    page_ref: str | None = None
    unknown: bool = False            # True when catalog has no entry -> flag, don't guess
    platforms: list[str] = field(default_factory=list)  # catalog scope, empty = any