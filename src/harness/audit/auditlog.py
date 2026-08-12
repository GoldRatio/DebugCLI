"""Append-only, hash-chained audit log (WORM backing).

Every event is appended with a hash of the previous event, forming a chain. Tampering
with any historical entry is detectable because the subsequent hash no longer matches.
Writes are line-append only; there is no update API.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .redact import Redactor


@dataclass(frozen=True)
class AuditEvent:
    session_id: str
    seq: int
    ts: str
    kind: str
    payload: dict
    prev_hash: str
    hash: str
    raw: str  # original (pre-hash) JSON line, used for replay; redacted separately


class AuditLog:
    def __init__(self, path: str | Path, redactor: Redactor | None = None) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._redactor = redactor or Redactor()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seq = 0
        self._last_hash = self._tail_hash()

    def _tail_hash(self) -> str:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return hashlib.sha256(b"genesis").hexdigest()
        with self.path.open("rb") as fh:
            last = fh.readlines()[-1]
        return json.loads(last)["hash"]

    def append(self, session_id: str, kind: str, payload: dict) -> AuditEvent:
        with self._lock:
            self._seq += 1
            ts = datetime.now(UTC).isoformat()
            body = {"session_id": session_id, "seq": self._seq, "ts": ts, "kind": kind,
                    "payload": payload, "prev_hash": self._last_hash}
            # Hash over the canonical body (no hash field), exactly what verify recomputes.
            body_json = json.dumps(body, sort_keys=True)
            event_hash = hashlib.sha256(body_json.encode("utf-8")).hexdigest()
            line = dict(body, hash=event_hash)
            stored = self._redactor.redact(json.dumps(line, sort_keys=True))
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(stored + "\n")
            self._last_hash = event_hash
            return AuditEvent(**line, raw=stored)

    def verify(self) -> list[str]:
        """Return list of chain-consistency errors (empty == tamper-free)."""
        errors: list[str] = []
        prev = hashlib.sha256(b"genesis").hexdigest()
        with self.path.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh, start=1):
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"line {i}: malformed JSON: {exc}")
                    continue
                if event.get("prev_hash") != prev:
                    errors.append(f"line {i}: prev_hash mismatch")
                doc = json.dumps({"session_id": event["session_id"], "seq": event["seq"],
                                  "ts": event["ts"], "kind": event["kind"],
                                  "payload": event["payload"], "prev_hash": event["prev_hash"]},
                                 sort_keys=True)
                if hashlib.sha256(doc.encode()).hexdigest() != event.get("hash"):
                    errors.append(f"line {i}: content hash mismatch")
                prev = event.get("hash")
        return errors

    def read(self) -> list[AuditEvent]:
        out: list[AuditEvent] = []
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                event = json.loads(line)
                out.append(AuditEvent(**event, raw=line.strip()))
        return out
