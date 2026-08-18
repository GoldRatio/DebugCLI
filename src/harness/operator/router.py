"""Deterministic keyword fallback routing for the interactive session.

The session REPL is driven by the conversation agent (``operator.chat_agent``):
the LLM decides what to say and which read-only tool to call. When the
conversation LLM is unavailable or answers garbage, ``_keyword_route`` keeps
the session usable: it maps each operator message to one ``SessionCommand``
which ``chat_agent.fallback_turn`` converts into the same agent-turn shape.
Routing only ever selects *existing* harness actions -- it cannot invent
commands, and every action stays read-only.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel

SUBSYSTEMS = ("memory", "cpu", "pcie", "bmc", "storage", "kernel", "generic")

# "Q61" / "q 61" / "rack q71" -> rack id; "cable 8" / "cable#8" -> cable id.
_RACK_RE = re.compile(r"\bq\s*[-_ ]?(\d{1,4})\b", re.IGNORECASE)
_CABLE_RE = re.compile(r"\bcable\s*#?\s*(\d{1,4})\b", re.IGNORECASE)
_IP_RE = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b")


class SessionCommand(BaseModel):
    """One keyword-routed operator message (the agent-LLM fallback shape)."""

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


def _keyword_route(text: str) -> SessionCommand:
    """Deterministic fallback so the session works without a conversation LLM."""
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
