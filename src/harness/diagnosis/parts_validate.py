"""Parts validation adapter: docs.parts_graph <-> live system snapshot.

Cross-references physical identifiers (slot label, PCIe slot, drive bay) from the
parts graph against what the live system reports via FRU/dmidecode. Flags
discrepancies so the harness never recommends replacing a part that is already
known-mismatched or outdated. This lives in ``diagnosis`` (not ``docs`` or
``inspect``) to avoid a ``docs`` -> ``inspect`` dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PartsCheckResult:
    matches: list[str] = field(default_factory=list)
    discrepancies: list[str] = field(default_factory=list)
    ok: bool = True

    def __bool__(self) -> bool:
        return self.ok


class PartsValidator:
    """Validates that live system serials/FRUs agree with the parts list mapping.

    ``parts_graph`` is expected to be a mapping ``{slot: {"fru": ..., "pn": ..., "sn": ...}}``
    and ``system_snapshot`` a mapping ``{slot: {...}}`` captured by the appropriate
    collector (e.g., ``ipmitool fru print`` decoded).
    """

    def __init__(self) -> None:
        pass

    def validate(self, parts_graph: dict, system_snapshot: dict,
                 check_keys: tuple[str, ...] = ("sn", "fru", "pn")) -> PartsCheckResult:
        matches: list[str] = []
        discrepancies: list[str] = []
        for slot, sys_entry in system_snapshot.items():
            listed = parts_graph.get(slot)
            for key in check_keys:
                sys_val = sys_entry.get(key)
                list_val = listed.get(key) if listed else None
                if sys_val is None:
                    continue
                if list_val is None:
                    discrepancies.append(f"{slot}.{key}: present in live system but missing from parts list")
                elif str(sys_val).strip().lower() != str(list_val).strip().lower():
                    discrepancies.append(f"{slot}.{key}: live {sys_val!r} != parts list {list_val!r}")
                else:
                    matches.append(f"{slot}.{key}: consistent")
        return PartsCheckResult(matches=matches, discrepancies=discrepancies, ok=not discrepancies)

    # ---- value-based cross-check against raw `ipmitool fru print` text ----

    def fru_snapshot(self, fru_text: str) -> dict[str, dict]:
        """Parse FRU print text into ``{part_number: {"pn": ..., "sn": ...}}``.

        Serial lines are associated with the most recent part-number line, which
        mirrors the standard ``ipmitool fru print`` layout (Product/Board sections
        list Part Number immediately before Serial).
        """
        entries: dict[str, dict] = {}
        current_pn: str | None = None
        for line in fru_text.splitlines():
            if ":" not in line:
                continue
            label, _, value = line.partition(":")
            value = value.strip()
            kind = _FRU_LABELS.get(label.strip().lower())
            if kind is None or not value:
                continue
            if kind == "pn":
                current_pn = value
                entries.setdefault(value, {"pn": value})
            elif kind == "sn" and current_pn is not None:
                entries[current_pn]["sn"] = value
        return entries

    def validate_fru(self, parts_graph: dict, fru_text: str) -> PartsCheckResult:
        """Cross-check installed FRU part numbers/serials against the parts list.

        A part installed but absent from the parts list, or a serial mismatch, is
        a discrepancy (the harness must never recommend replacing a part that is
        already known-mismatched). With no FRU data the check is skipped (ok).
        """
        installed = self.fru_snapshot(fru_text)
        if not installed:
            return PartsCheckResult(matches=["(no FRU part/serial data; parts cross-check skipped)"], ok=True)

        listed = {str(e.get("pn", "")).strip().lower(): e for e in parts_graph.values() if e.get("pn")}
        matches: list[str] = []
        discrepancies: list[str] = []
        for pn, info in installed.items():
            entry = listed.get(pn.strip().lower())
            if entry is None:
                discrepancies.append(f"installed part {pn!r} not present in parts list")
                continue
            sn = info.get("sn")
            if sn is not None:
                listed_sn = str(entry.get("sn", "")).strip().lower()
                if listed_sn and sn.strip().lower() != listed_sn:
                    discrepancies.append(f"serial mismatch for {pn!r}: live {sn!r} != parts list {listed_sn!r}")
                    continue
            matches.append(f"{pn}: installed part present in parts list")
        return PartsCheckResult(matches=matches, discrepancies=discrepancies, ok=not discrepancies)


_FRU_LABELS = {
    "product part number": "pn",
    "product serial": "sn",
    "board part number": "pn",
    "board serial": "sn",
}