"""Case store: hash-linked home of verified diagnosis outcomes.

Every recorded run becomes a ``CaseOutcome`` JSON file under ``<root>/<run_id>.json``
written atomically. By default the store is append-only -- an existing file is
NEVER overwritten (WORM invariant -- the outcome is stored once). An operator
correction (a mislabeled fix/outcome) may explicitly replace the record via
``save(overwrite=True)`` / ``record(revise=True)``, which also appends a
``case_revised`` audit event carrying the previous outcome so the change is
traceable. ``index.json`` is a rebuildable manifest (mtime-ordered, tolerates
missing/stale index by rescanning). ``save`` refuses any record whose fields
reference the secret store or key material (defense in depth).
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from ..audit.auditlog import AuditLog
from .schema import CaseOutcome

_FORBIDDEN = ("secret_dir", "key file", "private key", ".pem", "id_ed25519",
              "-----BEGIN")


def _secret_refs(record: CaseOutcome) -> list[str]:
    """Field values that look like secret storage references (defense in depth)."""
    bad: list[str] = []
    if record.llm_ident and any(t in record.llm_ident.lower() for t in _FORBIDDEN):
        bad.append("llm_ident")
    for label, values in (("actions_recommended", record.actions_recommended),
                          ("actions_taken", record.actions_taken),
                          ("evidence_summary", record.evidence_summary)):
        for v in values:
            low = str(v).lower()
            if any(t in low for t in _FORBIDDEN):
                bad.append(label)
                break
    return bad


def evidence_hash(evidence_summary: list[str]) -> str:
    """Deterministic sha256 over the canonical JSON of the evidence block.

    Sorting keys only (not the list order -- lines keep their array order, and
    the same lines in the same order hash identically across runs).
    """
    canonical = json.dumps({"evidence": evidence_summary}, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class CaseStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"

    def save(self, record: CaseOutcome, *, overwrite: bool = False) -> Path:
        """Persist an outcome record (append-only unless ``overwrite=True``).

        Refuses to overwrite an existing ``{run_id}.json`` unless the caller
        explicitly opts in (label correction), and rejects records carrying
        secret references. Audit-hash linkage is the caller's job
        (``CaseStore.record``); ``save`` itself never mutates old bytes.
        """
        if _secret_refs(record):
            raise ValueError(
                f"case record {record.run_id!r} contains a secret/key reference; "
                "rejected")
        target = self.root / f"{record.run_id}.json"
        if target.exists() and not overwrite:
            raise FileExistsError(
                f"case {record.run_id!r} already recorded (append-only: an outcome "
                "is stored once)")
        if not record.created_at:
            record.created_at = datetime.now(UTC).isoformat()
        tmp = target.with_suffix(".tmp")
        tmp.write_text(record.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(target)
        self._rebuild_index()
        return target

    def record(self, record: CaseOutcome, *, audit: AuditLog | None = None,
               session_id: str | None = None, revise: bool = False) -> Path:
        """Persist AND audit-link a case (hash covers the record's own bytes).

        The audit entry carries the record's ``evidence_hash`` + outcome so the
        WORM chain and the case store are independently verifiable; a mismatch
        between stored bytes and the audit hash surfaces as a chain error.
        ``revise=True`` replaces an existing record (operator correction) and
        audits a ``case_revised`` event with the previous outcome; when nothing
        exists yet it behaves like a plain first record.
        """
        previous: CaseOutcome | None = None
        target = self.root / f"{record.run_id}.json"
        if revise and target.exists():
            try:
                previous = CaseOutcome.model_validate_json(
                    target.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 - unreadable old record: still revise
                previous = None
        path = self.save(record, overwrite=revise)
        if audit is not None:
            if previous is not None:
                audit.append(
                    session_id or record.run_id, "case_revised",
                    {"run_id": record.run_id, "prev_outcome": previous.outcome,
                     "prev_actions_taken": previous.actions_taken,
                     "outcome": record.outcome,
                     "evidence_hash": record.evidence_hash,
                     "case_file": str(path.name)},
                )
            else:
                audit.append(
                    session_id or record.run_id, "case_record",
                    {"run_id": record.run_id, "evidence_hash": record.evidence_hash,
                     "outcome": record.outcome, "case_file": str(path.name)},
                )
        return path

    def get(self, run_id: str) -> CaseOutcome | None:
        path = self.root / f"{run_id}.json"
        if not path.exists():
            return None
        return CaseOutcome.model_validate_json(path.read_text(encoding="utf-8"))

    def delete(self, run_id: str) -> bool:
        """Remove a case record and rebuild the index.

        The ONE deliberate exception to the append-only rule: used only by
        explicit run deletion (runs menu / ``harness runs delete``), never by
        the learning loop itself. Returns True when a record was removed.
        """
        path = self.root / f"{run_id}.json"
        if not path.exists():
            return False
        path.unlink()
        self._rebuild_index()
        return True

    def all(self) -> list[CaseOutcome]:
        """All records, ordered by index manifest when present else file mtime."""
        ordered = self._indexed_order()
        return [self.get(rid) for rid in ordered if self.get(rid) is not None]

    def _indexed_order(self) -> list[str]:
        """Run ids from the manifest, or from a directory rescan when the index
        is missing/stale -- the index is always rebuildable, never authoritative."""
        if self.index_path.exists():
            try:
                manifest = json.loads(self.index_path.read_text(encoding="utf-8"))
                return [rid for rid in sorted(manifest, key=lambda r: manifest[r]
                                              .get("created_at", ""))]
            except (json.JSONDecodeError, AttributeError):
                pass  # fall through to a rescan
        files = sorted(self.root.glob("*.json"),
                       key=lambda p: p.stat().st_mtime)
        return [p.stem for p in files if p.stem != "index"]

    def _rebuild_index(self) -> None:
        manifest: dict[str, dict] = {}
        for path in sorted(self.root.glob("*.json")):
            if path.stem == "index":
                continue
            record = self._try_read(path)  # corrupt file: skip, keep the rest
            if record is None:
                continue
            manifest[record.run_id] = {
                "model_key": record.model_key,
                "outcome": record.outcome,
                "created_at": record.created_at,
            }
        tmp = self.index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True),
                       encoding="utf-8")
        tmp.replace(self.index_path)

    @staticmethod
    def _try_read(path: Path) -> CaseOutcome | None:
        try:
            return CaseOutcome.model_validate_json(
                path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - tolerate one corrupt file
            return None