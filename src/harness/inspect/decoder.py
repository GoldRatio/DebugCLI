"""Decoder: raw_hex -> typed, human-readable fields via the curated catalog.

Deterministic and LLM-free. If a register/stepping is not in the catalog, the
decoder flags ``unknown=True`` and returns no guessed fields -- the diagnostic
engine must then say "unknown register; manual lookup" rather than hallucinate.
"""

from __future__ import annotations

import re

from .base import DecodedField, RegisterDecode
from .catalog.catalog_loader import RegisterCatalog, default_catalog


class Decoder:
    def __init__(self, catalog: RegisterCatalog | None = None) -> None:
        self.catalog = catalog or default_catalog()

    def decode(self, mnemonic: str, raw_hex: str) -> RegisterDecode:
        mnemonic = mnemonic.upper()
        defn = self.catalog.lookup(mnemonic)
        if defn is None:
            return RegisterDecode(mnemonic=mnemonic, raw_hex=raw_hex, unknown=True)

        value = int(raw_hex, 16)
        fields = [
            self._decode_field(value, self._int(f.msb), self._int(f.lsb), f.name, f.values)
            for f in defn.bit_fields
        ]
        return RegisterDecode(
            mnemonic=mnemonic,
            raw_hex=raw_hex,
            decoded_fields=fields,
            catalog_version=self.catalog.schema_version,
            page_ref=defn.page_ref,
            unknown=False,
        )

    def decode_many(self, raw_lines: str, mnemonic_pattern: str = r"([A-Za-z0-9_\.]+)\s+=\s+0x([0-9a-fA-F]+)") -> list[RegisterDecode]:
        out: list[RegisterDecode] = []
        for match in re.finditer(mnemonic_pattern, raw_lines):
            out.append(self.decode(match.group(1), match.group(2)))
        return out

    @staticmethod
    def _int(v: int | None) -> int | None:
        return v

    @staticmethod
    def _decode_field(value: int, msb: int | None, lsb: int | None, name: str,
                      values: dict[str, str] | None) -> DecodedField:
        if msb is not None and lsb is not None:
            mask_width = msb - lsb + 1
            field_value = (value >> lsb) & ((1 << mask_width) - 1)
        else:
            field_value = value
        raw_str = str(field_value)
        meaning = (values or {}).get(str(field_value))
        return DecodedField(name=name, msb=msb, lsb=lsb, raw_value=raw_str, meaning=meaning)