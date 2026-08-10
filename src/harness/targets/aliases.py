"""Short target aliases (``harness targets``).

An alias maps a human-friendly name to a console target ``(rack, cable)`` and/or
an SSH address. The file holds identifiers and addresses ONLY -- never
credentials -- so it is safe to keep next to the inventory and passes the
no-inline-secret invariant.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

# Identifiers/addresses are drawn from a restricted charset: nothing here can
# ever look like (or smuggle) a credential blob.
_SAFE_VALUE = re.compile(r"^[A-Za-z0-9._:-]+$")
_IP4 = re.compile(r"^(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)$")
_RACK = re.compile(r"^[A-Za-z0-9_-]+$")
_CABLE = re.compile(r"^[A-Za-z0-9_-]+$")


class AliasError(RuntimeError):
    pass


@dataclass(frozen=True)
class TargetAlias:
    alias: str
    rack: str | None = None
    cable: str | None = None
    address: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.alias, str):
            raise AliasError(f"invalid alias {self.alias!r}: must be a string")
        # YAML booleans/ints arrive as non-str; coerce cable/rack numbers like
        # `cable: 8` to "8" before validation.
        for name in ("rack", "cable", "address"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (str, int)):
                raise AliasError(f"invalid {name} {value!r}")
            if isinstance(value, int):
                object.__setattr__(self, name, str(value))
        if not _SAFE_VALUE.match(self.alias):
            raise AliasError(f"invalid alias {self.alias!r}: allowed [A-Za-z0-9._:-]")
        if self.rack is not None and not _RACK.match(self.rack):
            raise AliasError(f"invalid rack {self.rack!r}")
        if self.cable is not None and not _CABLE.match(self.cable):
            raise AliasError(f"invalid cable {self.cable!r}")
        if self.address is not None and not _IP4.match(self.address):
            raise AliasError(f"invalid address {self.address!r}")
        if self.rack is None and self.cable is None and self.address is None:
            raise AliasError(f"alias {self.alias!r} needs --rack/--cable and/or --address")
        if (self.rack is None) != (self.cable is None):
            raise AliasError(f"alias {self.alias!r}: --rack and --cable must be given together")


def load_targets(path: str | Path) -> dict[str, TargetAlias]:
    """Load aliases; returns {} when the file is absent or empty."""
    p = Path(path)
    if not p.exists():
        return {}
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    out: dict[str, TargetAlias] = {}
    for entry in raw.get("targets", []) or []:
        alias = TargetAlias(
            alias=entry["alias"],
            rack=entry.get("rack"),
            cable=entry.get("cable"),
            address=entry.get("address"),
        )
        out[alias.alias] = alias
    return out


def save_targets(path: str | Path, aliases: dict[str, TargetAlias]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    body = {"targets": [
        {"alias": a.alias, "rack": a.rack, "cable": a.cable, "address": a.address}
        for a in sorted(aliases.values(), key=lambda a: a.alias)
    ]}
    p.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
