"""Per-target operator-supplied parts (instance data) store.

The doc-derived parts TOPOLOGY tells the engine which loads sit on a rail; the
INSTANCE data (which FRU/PN/SN populates each slot, and on which target) is
fleet knowledge only the operator has. When a diagnosis needs it (a decoded
rail fault under an opt-in ``ask-parts`` run), the operator is prompted and the
answers are persisted per target so future runs reuse them silently. The store
is best-effort: unreadable/corrupt files degrade to an empty store, never an
error.
"""

from __future__ import annotations

import json
from pathlib import Path


class InstancePartsStore:
    """A JSON file per target keyed by ``{slot: {fru, pn, sn, rail, ...}}``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path_for(self, target: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in target)
        return self.root / f"{safe}.json"

    def load(self, target: str) -> dict[str, dict]:
        p = self.path_for(target)
        if not p.exists():
            return {}
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        parts = data.get("parts", {}) if isinstance(data, dict) else {}
        return {str(k): dict(v) for k, v in parts.items()}

    def merge(self, target: str, entries: list[dict]) -> None:
        """Upsert one or more ``{slot, fru, pn, sn, rail, ...}`` entries."""
        parts = self.load(target)
        for entry in entries:
            slot = str(entry.get("slot") or "").strip()
            if not slot:
                continue
            parts[slot] = {k: v for k, v in entry.items() if k != "slot"}
        self._save(target, parts)

    def _save(self, target: str, parts: dict[str, dict]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.path_for(target)
        payload = {"target": target, "parts": parts}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def merge_store_into_parts(parts_graph: dict[str, dict] | None,
                           stored: dict[str, dict]) -> dict[str, dict]:
    """Instance-store entries merged under the explicit CSV/parts graph.

    ``parts_graph`` (from ``--parts-csv``) is authoritative per-field (a CSV
    ``fru``/``pn``/``sn`` wins); stored fields the CSV does not carry (e.g.
    ``rail``) survive the merge.
    """
    merged = {k: dict(v) for k, v in (stored or {}).items()}
    for k, v in (parts_graph or {}).items():
        merged.setdefault(k, {})
        merged[k].update(dict(v))
    return merged