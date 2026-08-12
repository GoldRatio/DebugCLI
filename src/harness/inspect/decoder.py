"""Decoder: raw_hex -> typed, human-readable fields via the curated catalog.

Deterministic and LLM-free. If a register/stepping is not in the catalog, the
decoder flags ``unknown=True`` and returns no guessed fields -- the diagnostic
engine must then say "unknown register; manual lookup" rather than hallucinate.
"""

from __future__ import annotations

import re

from .base import DecodedField, RegisterDecode
from .catalog.catalog_loader import RegisterCatalog, default_catalog

# i2ctransfer/i2cget: data printed AFTER the shell echo of the command line.
# The console banner/echo lines precede it; the next shell prompt ends it.
_PROMPT_RE = re.compile(r"^\s*\S*@\S*[:#\$~]\s*$")
_HEX_BYTE_RE = re.compile(r"(?:0x)?([0-9a-fA-F]{2})\b")


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
            platforms=list(defn.platforms),
        )

    def decode_many(self, raw_lines: str, mnemonic_pattern: str = r"([A-Za-z0-9_\.]+)\s+=\s+0x([0-9a-fA-F]+)") -> list[RegisterDecode]:
        out: list[RegisterDecode] = []
        for match in re.finditer(mnemonic_pattern, raw_lines):
            out.append(self.decode(match.group(1), match.group(2)))
        return out

    # i2cdump row: "0b: 05 00 00 ..." -- a 2-hex-digit offset then 16 byte tokens.
    _I2C_ROW_RE = re.compile(r"^\s*([0-9a-fA-F]{2})\s*:\s*(.*?)\s*$")

    def decode_i2c_dump(self, raw: str) -> list[RegisterDecode]:
        """Decode an ``i2cdump -y <bus> <addr>`` output into catalog registers.

        Only offsets that have a curated catalog entry (matched by address, e.g.
        0x1b/0xa1 of the SWB CPLD boot-state block) are surfaced; all other bytes
        stay raw on disk for audit. This keeps the LLM context small and never
        guesses the meaning of an unknown offset.
        """
        out: list[RegisterDecode] = []
        seen: set[str] = set()
        for offset, value in self._parse_i2c_dump(raw):
            defn = self.catalog.lookup_addr(f"0x{offset:02X}")
            if defn is None or defn.mnemonic in seen:
                continue
            seen.add(defn.mnemonic)
            fields = [
                self._decode_field(value, self._int(f.msb), self._int(f.lsb),
                                   f.name, f.values)
                for f in defn.bit_fields
            ]
            out.append(RegisterDecode(
                mnemonic=defn.mnemonic,
                raw_hex=f"0x{value:02x}",
                decoded_fields=fields,
                catalog_version=self.catalog.schema_version,
                page_ref=defn.page_ref,
                unknown=False,
                platforms=list(defn.platforms),
            ))
        return out

    @classmethod
    def _parse_i2c_dump(cls, raw: str):
        for line in raw.splitlines():
            m = cls._I2C_ROW_RE.match(line)
            if m is None:
                continue
            try:
                base = int(m.group(1), 16)
            except ValueError:
                continue
            for i, token in enumerate(re.findall(r"[0-9a-fA-F]{2}", m.group(2))):
                try:
                    yield base + i, int(token, 16)
                except ValueError:
                    continue

    def decode_i2c_transfer(self, raw: str, source: str = "") -> list[RegisterDecode]:
        """Decode an ``i2ctransfer ... r<n>`` block read into catalog registers.

        Byte ``i`` of the read data maps to register offset ``i`` (flat register
        map, e.g. the SWB CPLD). Only cataloged offsets are surfaced; the console
        banner and command echo are skipped.
        """
        out: list[RegisterDecode] = []
        seen: set[str] = set()
        body = _after_echo(raw, source) if source else raw
        for offset, token in enumerate(_HEX_BYTE_RE.findall(body)):
            try:
                value = int(token, 16)
            except ValueError:
                continue
            defn = self.catalog.lookup_addr(f"0x{offset:02X}")
            if defn is None or defn.mnemonic in seen:
                continue
            seen.add(defn.mnemonic)
            fields = [
                self._decode_field(value, self._int(f.msb), self._int(f.lsb),
                                   f.name, f.values)
                for f in defn.bit_fields
            ]
            out.append(RegisterDecode(
                mnemonic=defn.mnemonic,
                raw_hex=f"0x{value:02x}",
                decoded_fields=fields,
                catalog_version=self.catalog.schema_version,
                page_ref=defn.page_ref,
                unknown=False,
                platforms=list(defn.platforms),
            ))
        return out

    def decode_i2c_get(self, raw: str, source: str = "") -> list[RegisterDecode]:
        """Decode a single ``i2cget -y <bus> <addr> <reg>`` read.

        The register offset is the last ``0x`` token in the command; the value is
        the hex byte printed after the command echo.
        """
        hexes = re.findall(r"0x([0-9a-fA-F]+)", source)
        if not hexes:
            return []
        try:
            reg = int(hexes[-1], 16)
        except ValueError:
            return []
        defn = self.catalog.lookup_addr(f"0x{reg:02X}")
        if defn is None:
            return []
        body = _after_echo(raw, source) if source else raw
        m = _HEX_BYTE_RE.search(body)
        if m is None:
            return []
        value = int(m.group(1), 16)
        fields = [
            self._decode_field(value, self._int(f.msb), self._int(f.lsb),
                               f.name, f.values)
            for f in defn.bit_fields
        ]
        return [RegisterDecode(
            mnemonic=defn.mnemonic,
            raw_hex=f"0x{value:02x}",
            decoded_fields=fields,
            catalog_version=self.catalog.schema_version,
            page_ref=defn.page_ref,
            unknown=False,
            platforms=list(defn.platforms),
        )]

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


def _after_echo(raw: str, hint: str) -> str:
    """Text after the console line echoing ``hint``, cut at the next prompt.

    Console output wraps every probe with an expect banner and a shell echo of
    the command line; ``hint`` (the command text) marks the echo. Everything
    after it up to the next ``user@host:~$``-style prompt is the probe's own
    output.
    """
    lines = raw.splitlines()
    for i, line in enumerate(lines):
        if hint and hint in line:
            body: list[str] = []
            for rest in lines[i + 1:]:
                if _PROMPT_RE.match(rest):
                    break
                body.append(rest)
            return "\n".join(body)
    return raw