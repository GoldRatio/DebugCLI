"""Loader + validation for the curated register catalog.

Enforces the sign-off / review invariant: a catalog version that is not reviewed
cannot be used for decoding. Loaded definitions are frozen in memory so nothing
downstream can mutate the authoritative decode tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

import yaml


class CatalogUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class BitField:
    name: str
    msb: int | None
    lsb: int | None
    values: dict[str, str] | None = None


@dataclass(frozen=True)
class RegisterDef:
    mnemonic: str
    addr: str | None
    canonical_source: str | None
    page_ref: str | None
    bit_fields: tuple[BitField, ...]


@dataclass(frozen=True)
class RegisterCatalog:
    schema_version: str
    owner: str
    reviewed: bool
    registers: dict[str, RegisterDef]

    def lookup(self, mnemonic: str) -> RegisterDef | None:
        return self.registers.get(mnemonic.upper())

    def lookup_addr(self, addr: str) -> RegisterDef | None:
        """Find a register by its I2C/CPU address (e.g. "0x1b"), case-insensitive."""
        want = addr.upper()
        for defn in self.registers.values():
            if defn.addr and defn.addr.upper() == want:
                return defn
        return None


def load_catalog(path: str | Path | None = None) -> RegisterCatalog:
    if path is None:
        raw = files("harness.inspect.catalog").joinpath("register_catalog.yaml").read_text(encoding="utf-8")
    else:
        raw = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise CatalogUnavailable("catalog must be a mapping")

    signoff = data.get("signoff") or {}
    reviewed = bool(signoff.get("reviewed")) and data.get("schema_version") is not None
    if not reviewed:
        raise CatalogUnavailable("register catalog is not reviewed/signed-off; refusing to decode")

    registers: dict[str, RegisterDef] = {}
    for mnemonic, reg in (data.get("registers") or {}).items():
        fields = tuple(
            BitField(name=f["name"], msb=f.get("msb"), lsb=f.get("lsb"), values=f.get("values"))
            for f in (reg.get("bit_fields") or [])
        )
        registers[mnemonic.upper()] = RegisterDef(
            mnemonic=mnemonic,
            addr=reg.get("addr"),
            canonical_source=reg.get("canonical_source"),
            page_ref=reg.get("page_ref"),
            bit_fields=fields,
        )
    return RegisterCatalog(
        schema_version=str(data["schema_version"]),
        owner=str(data.get("owner", "unknown")),
        reviewed=reviewed,
        registers=registers,
    )


@lru_cache(maxsize=1)
def default_catalog() -> RegisterCatalog:
    return load_catalog()