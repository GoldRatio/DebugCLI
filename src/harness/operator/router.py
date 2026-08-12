"""Natural-language -> harness command routing for the interactive session.

The operator chats in plain language; the router turns each message into one
``SessionCommand``. When an LLM with ``chat_json`` is available it picks the
intent; otherwise (or on malformed output) a conservative keyword fallback keeps
the session usable. Routing only ever selects *existing* harness actions -- the
router cannot invent commands, and every action stays read-only.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel

ROUTER_SYSTEM = """You are the command router for a READ-ONLY server debugging harness.
The operator is chatting with you. Choose exactly ONE intent and answer with a single
JSON object -- no prose:

- "diagnose": the operator describes a symptom/error and wants a fresh read-only
  diagnosis. Fields: symptom (required, the symptom text), host (optional, must be
  one of the available hosts or omitted for the active target). Target fields
  (all optional, for targeting a server WITHOUT a YAML entry): rack (e.g. "Q61"),
  cable (e.g. "8"), ip (IPv4 address). When a target is named, keep the symptom
  free of the rack/cable/IP phrase.
- "probe": the operator wants further read-only probes or doc lookups inside the
  current session. Fields: subsystems (from {memory, cpu, pcie, bmc, storage, kernel,
  generic}), doc_topics (short topic strings).
- "docs": a documentation/manual lookup only. Fields: query (required).
- "verify": re-check whether a previous state changed. Fields: metric (optional,
  default "ecc"), baseline (optional, omitted = latest run).
- "status": the operator asks what is running or what was done so far.
- "reply": purely conversational messages only. Field: text (required).

The operator expects ACTION: mentions of a symptom, of a previous diagnosis's
quality (low confidence, missing/absent repair actions), or of what part should be
replaced or repaired are "diagnose" requests -- re-run the read-only diagnosis with
the full message as the symptom. Never answer with advice instead of acting.

Never fabricate hardware findings, never propose commands or writes. Respond with the
JSON object only."""

SUBSYSTEMS = ("memory", "cpu", "pcie", "bmc", "storage", "kernel", "generic")

# "Q61" / "q 61" / "rack q71" -> rack id; "cable 8" / "cable#8" -> cable id.
_RACK_RE = re.compile(r"\bq\s*[-_ ]?(\d{1,4})\b", re.IGNORECASE)
_CABLE_RE = re.compile(r"\bcable\s*#?\s*(\d{1,4})\b", re.IGNORECASE)
_IP_RE = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b")


class SessionCommand(BaseModel):
    """One routed operator message."""

    intent: Literal["diagnose", "probe", "docs", "verify", "status", "reply"]
    symptom: str | None = None
    host: str | None = None
    rack: str | None = None
    cable: str | None = None
    ip: str | None = None
    alias: str | None = None
    subsystems: list[str] = []
    doc_topics: list[str] = []
    query: str | None = None
    metric: str = "ecc"
    baseline: str | None = None
    text: str | None = None


def extract_target(text: str) -> tuple[str, dict]:
    """Pull a target mention out of natural language.

    Returns ``(cleaned_text, target)`` where ``target`` holds any of
    ``rack``/``cable``/``ip`` found. The phrase is removed from the text so the
    diagnosis prompt carries only the symptom.
    """
    target: dict = {}
    cleaned = text
    m = _IP_RE.search(cleaned)
    if m:
        target["ip"] = m.group(0)
        cleaned = cleaned.replace(m.group(0), " ").strip()
    m = _RACK_RE.search(cleaned)
    if m:
        target["rack"] = f"Q{int(m.group(1))}"
        cleaned = re.sub(r"\bq\s*[-_ ]?" + m.group(1) + r"\b", " ",
                         cleaned, flags=re.IGNORECASE).strip()
    m = _CABLE_RE.search(cleaned)
    if m:
        target["cable"] = str(int(m.group(1)))
        cleaned = re.sub(r"\bcable\s*#?\s*" + m.group(1) + r"\b", " ",
                         cleaned, flags=re.IGNORECASE).strip()
    return re.sub(r"\s+", " ", cleaned).strip(), target


def route_message(text: str, llm, history: list[dict] | None = None,
                  host_names: tuple[str, ...] = ()) -> SessionCommand:
    """Route one operator message; falls back to keyword routing."""
    chat_json = getattr(llm, "chat_json", None)
    if callable(chat_json):
        try:
            messages = [{"role": "system", "content": ROUTER_SYSTEM}]
            messages.extend((history or [])[-6:])
            messages.append({"role": "user", "content": text})
            cmd = SessionCommand.model_validate(chat_json(messages))
            if cmd.intent == "diagnose" and not cmd.symptom:
                raise ValueError("diagnose intent requires a symptom")
            if cmd.intent == "docs" and not cmd.query:
                raise ValueError("docs intent requires a query")
            if cmd.host and host_names and cmd.host not in host_names:
                cmd.host = None  # unknown host -> active target, never an error
            if not any((cmd.rack, cmd.cable, cmd.ip)):
                # LLM may leave the phrase in the symptom; pull it out for the
                # diagnosis prompt either way.
                cleaned, target = extract_target(cmd.symptom or "")
                if target:
                    cmd.symptom = cleaned or cmd.symptom
                    cmd.rack = cmd.rack or target.get("rack")
                    cmd.cable = cmd.cable or target.get("cable")
                    cmd.ip = cmd.ip or target.get("ip")
            return cmd
        except Exception:  # noqa: BLE001, S110 - malformed router output falls back
            pass
    return _keyword_route(text)


def _keyword_route(text: str) -> SessionCommand:
    """Deterministic fallback so the session works without an LLM/router model."""
    low = text.lower()

    if any(k in low for k in ("status", "busy", "doing", "progress")):
        return SessionCommand(intent="status")
    if any(k in low for k in ("verify", "compare", "did it change", "still there")):
        return SessionCommand(intent="verify")
    if any(k in low for k in ("probe", "read", "dump", "register", "fetch")):
        wanted = [s for s in SUBSYSTEMS if s in low]
        return SessionCommand(intent="probe", subsystems=wanted[:1])
    if any(k in low for k in ("doc", "manual", "pdf", "lookup", "reference")):
        return SessionCommand(intent="docs", query=text)
    if any(k in low for k in ("diagnos", "why", "error", "crash", "panic", "fail",
                              "fault", "warn", "check", "inspect", "look at",
                              "confidence", "percent", "previous run", "repair",
                              "replace", "redo", "rerun")):
        cleaned, target = extract_target(text)
        return SessionCommand(intent="diagnose", symptom=cleaned or text, **target)
    return SessionCommand(intent="reply", text=text)
