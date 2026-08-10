"""Physical parts list -> slot -> FRU/PN/SN knowledge graph.

Ingests a CSV/Excel parts list into a mapping ``{slot: {"fru":..., "pn":..., "sn":...}}``
so the diagnostic engine can recommend a specific replaceable part. This module has
NO ``inspect`` dependency (live-system validation lives in ``diagnosis.parts_validate``).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PartEntry:
    slot: str
    fru: str | None = None
    pn: str | None = None
    sn: str | None = None
    line: int | None = None

    def as_dict(self) -> dict:
        out = {"slot": self.slot}
        if self.fru:
            out["fru"] = self.fru
        if self.pn:
            out["pn"] = self.pn
        if self.sn:
            out["sn"] = self.sn
        if self.line is not None:
            out["line"] = self.line
        return out


@dataclass
class PartsGraph:
    entries: list[PartEntry] = field(default_factory=list)

    def by_slot(self) -> dict[str, dict]:
        return {e.slot: e.as_dict() for e in self.entries}


def load_parts_csv(path: str | Path, *,
                   slot_col: str = "slot", fru_col: str | None = "fru",
                   pn_col: str | None = "pn", sn_col: str | None = "sn") -> PartsGraph:
    entries: list[PartEntry] = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        for line_no, row in enumerate(reader, start=2):
            slot = (row.get(slot_col) or "").strip()
            if not slot:
                continue
            entries.append(PartEntry(
                slot=slot,
                fru=(row.get(fru_col) or "").strip() if fru_col else None,
                pn=(row.get(pn_col) or "").strip() if pn_col else None,
                sn=(row.get(sn_col) or "").strip() if sn_col else None,
                line=line_no,
            ))
    return PartsGraph(entries=entries)