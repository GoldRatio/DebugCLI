"""Conversation agent for the session REPL: a decide-and-act tool loop.

The chat REPL is agent-driven: for each operator message the conversation LLM
sees the dialog plus an evidence digest of the latest diagnosis run and answers
with one ``ChatTurn`` -- short text for the operator (``say``; ALWAYS printed
before anything runs) and at most one tool call (diagnose / probe / docs /
verify).  Tool results feed back into the conversation so the agent can chain
steps (diagnose -> probe -> docs -> verify) until it can answer, exactly like a
coding agent that explains, acts, observes, and continues.

Read-only safety is unchanged: the agent only ever picks WHICH existing
harness action to run; it never emits command text, and every execution path
still enforces the allowlist + ``security_check`` gates.

When the conversation LLM is unavailable or answers garbage, ``fallback_turn``
maps the deterministic keyword router's ``SessionCommand`` (see
``operator.router``) to the same ``ChatTurn`` shape so the session stays usable
without a conversation model.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from ..diagnosis.schema import Diagnosis
from .router import SUBSYSTEMS, SessionCommand, extract_target

CHAT_TOOLS = ("diagnose", "probe", "docs", "verify", "run")

CHAT_SYSTEM = """You are the conversational agent of a READ-ONLY server-debugging
harness. You chat with the operator like a senior debugging partner: explain
what you are doing, call the harness's read-only tools to gather evidence, and
answer questions grounded in that evidence.

Hard rules:
- The harness is force_read_only: every tool below is a read-only pipeline.
  Never claim to have run a write/repair action; never invent commands,
  registers, or results. If evidence is missing, say so and call the tool that
  can collect it.
- Ground every claim about the current server in the Latest Run Evidence
  section or in tool results from this conversation. Distinguish what was
  measured from what you infer. IPMI SEL entries are HISTORICAL events, not
  proof of a current fault; weight live sensors over stale log records.
- The operator expects ACTION: a symptom report, a follow-up on a weak
  diagnosis (low confidence, missing repair actions), or a request to re-check
  or replace a part is a "diagnose" call with the operator's words as the
  symptom -- do not answer with advice alone. EXCEPTION: when the operator
  asks about an EXISTING run ("the run on ...", a harness_runs path, "what did
  the last run find"), use "run" to load that run's recorded diagnosis and
  evidence -- do NOT launch a fresh diagnosis to answer a question about a past
  run.

Tools (at most one per response):
- "diagnose": run the full read-only pipeline (collect -> decode -> docs ->
  LLM -> scored diagnosis) on one target. Fields: symptom (required, a crisp
  symptom phrase -- no questions, no run paths, no hedging like "I'm
  wondering"), host/rack/cable/ip/alias (ONLY when the operator names a
  DIFFERENT target; omit for the active target).
- "probe": quick read-only evidence, no diagnosis turn. Fields: subsystems
  (subset of [memory, cpu, pcie, bmc, storage, kernel, generic]), doc_topics
  (short topic strings; the harness mines safe probe commands from the manual).
- "docs": manual / architecture lookup only. Fields: query (required).
- "verify": re-collect one metric and compare against a baseline run. Fields:
  metric (default "ecc"), baseline (omit = latest run).
- "run": load a PAST run's recorded diagnosis + evidence digest by run id or
  directory (READ-ONLY, local file read; never re-runs anything). Use it
  whenever the operator references an existing run and asks what it found or
  how it relates to something. Fields: run (the hex run id or run directory
  path, e.g. "harness_runs/bcf28c2bc94448919445af3b2e66fdcc").
- "none": pure conversation -- answer, summarize, recommend next steps, or ask
  ONE concise clarifying question.

Always fill "say" -- it is shown to the operator before anything runs. When
calling a tool, say briefly what you are about to do and why (1-2 sentences).
You will see each tool's result before your next decision, and you may chain
tools until you can answer; stop (tool "none") as soon as you can. Prefer more
evidence over guessing."""

CHAT_CONTRACT = """Respond with STRICT JSON only (no markdown fences, no prose):
{"say": "...", "tool": "diagnose"|"probe"|"docs"|"verify"|"run"|"none", <tool fields>}
- diagnose fields: symptom (required -- a crisp symptom phrase, NOT a question
  and NOT a run path), host, rack, cable, ip, alias (optional).
- probe fields: subsystems, doc_topics.
- docs fields: query (required).
- verify fields: metric, baseline.
- run fields: run (required -- hex run id or run directory path).
"say" is always required and is printed to the operator first."""


class ChatTurn(BaseModel):
    """One conversation-agent decision: say something, then maybe call a tool.

    The agent never proposes commands -- ``subsystems``/``doc_topics`` are
    mapped by the harness to curated read-only collectors, and every other
    tool reuses an existing read-only pipeline.
    """

    say: str = ""
    tool: Literal["diagnose", "probe", "docs", "verify", "run", "none"] = "none"
    symptom: str | None = None
    host: str | None = None
    rack: str | None = None
    cable: str | None = None
    ip: str | None = None
    alias: str | None = None
    subsystems: list[str] = Field(default_factory=list)
    doc_topics: list[str] = Field(default_factory=list)
    query: str | None = None
    metric: str = "ecc"
    baseline: str | None = None
    run: str | None = None


def decide(llm, messages: list[dict],
           host_names: tuple[str, ...] = ()) -> ChatTurn | None:
    """One agent decision from the conversation LLM.

    Returns ``None`` when the LLM is unavailable or its output is unusable
    (missing/malformed JSON, old router format, semantic violations like a
    diagnose without a symptom) -- the caller then falls back to keyword
    routing so the session stays usable.
    """
    chat_json = getattr(llm, "chat_json", None)
    if not callable(chat_json):
        return None
    try:
        raw = chat_json(messages)
    except Exception:  # noqa: BLE001 - any LLM failure falls back to keywords
        return None
    if not isinstance(raw, dict) or "tool" not in raw:
        return None
    try:
        turn = ChatTurn.model_validate(raw)
    except Exception:  # noqa: BLE001 - malformed output falls back
        return None
    if turn.tool == "diagnose" and not (turn.symptom or "").strip():
        return None
    if turn.tool == "docs" and not (turn.query or "").strip():
        return None
    if turn.tool == "probe" and not turn.subsystems and not turn.doc_topics:
        return None
    if turn.tool == "run" and not (turn.run or "").strip():
        return None
    if turn.host and host_names and turn.host not in host_names:
        turn.host = None  # unknown host -> active target, never an error
    if turn.tool == "diagnose" and not any((turn.rack, turn.cable, turn.ip)):
        # The LLM may leave the target phrase inside the symptom; pull it out
        # so the diagnosis prompt carries only the symptom.
        cleaned, target = extract_target(turn.symptom or "")
        if target:
            turn.symptom = cleaned or turn.symptom
            turn.rack = turn.rack or target.get("rack")
            turn.cable = turn.cable or target.get("cable")
            turn.ip = turn.ip or target.get("ip")
    return turn


def fallback_turn(cmd: SessionCommand, text: str) -> ChatTurn:
    """Map a keyword-routed ``SessionCommand`` to the same ``ChatTurn`` shape
    (used when the conversation LLM is unavailable or answered garbage)."""
    if cmd.intent == "diagnose":
        return ChatTurn(say="Running a read-only diagnosis on the symptom.",
                        tool="diagnose", symptom=cmd.symptom or text,
                        host=cmd.host, rack=cmd.rack, cable=cmd.cable,
                        ip=cmd.ip, alias=cmd.alias)
    if cmd.intent == "probe":
        return ChatTurn(say="Running the requested read-only probes.",
                        tool="probe", subsystems=cmd.subsystems,
                        doc_topics=cmd.doc_topics)
    if cmd.intent == "docs":
        return ChatTurn(say="Looking that up in the manuals.",
                        tool="docs", query=cmd.query or text)
    if cmd.intent == "verify":
        return ChatTurn(say="Re-collecting the metric and comparing against "
                            "the baseline.", tool="verify", metric=cmd.metric,
                        baseline=cmd.baseline)
    return ChatTurn(say="(no conversational reply available; describe a "
                        "symptom to diagnose, or see /help)", tool="none")


# transcript kind -> message prefix; kinds without a prefix pass verbatim
_ENTRY_PREFIX = {
    "answer": "operator answer: ",
    "context": "operator context: ",
    "result": "[tool result] ",
    "error": "[error] ",
    "action": "[calling tool] ",
}

_MAX_ENTRY_CHARS = 800
_TRANSCRIPT_WINDOW = 12


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def build_messages(*, transcript: list[dict], user_text: str,
                   evidence_digest: str,
                   host_names: tuple[str, ...] = (),
                   target_label: str = "",
                   pending: list[str] | None = None) -> list[dict]:
    """Chat messages for one agent decision: system prompt + dialog window +
    a final user message carrying the current state (evidence digest, queued
    notes, the operator's line, and the output contract)."""
    messages: list[dict] = [{"role": "system", "content": CHAT_SYSTEM}]
    for entry in list(transcript)[-_TRANSCRIPT_WINDOW:]:
        role = "assistant" if entry.get("role") == "agent" else "user"
        prefix = _ENTRY_PREFIX.get(entry.get("kind", ""), "")
        content = _clip(f"{prefix}{entry.get('content', '')}", _MAX_ENTRY_CHARS)
        if content:
            messages.append({"role": role, "content": content})
    blocks: list[str] = []
    if host_names:
        blocks.append("## Available Hosts\n" + ", ".join(host_names))
    if target_label:
        blocks.append(f"## Active Target\n{target_label}")
    blocks.append(evidence_digest or
                  "## Latest Run Evidence\n(no diagnosis run yet in this "
                  "session)")
    if pending:
        blocks.append("## Operator Notes (typed while busy, read them now)\n"
                      + "\n".join(f"- {_clip(p, 300)}" for p in pending))
    blocks.append("## Operator Message\n" + user_text)
    runs = _referenced_runs(user_text)
    if runs:
        blocks.append("## Referenced Runs (the operator pointed at existing "
                      "runs)\n" + ", ".join(f"`{r}`" for r in runs)
                      + "\nUse the 'run' tool to load one of these instead of "
                      "launching a new diagnosis.")
    blocks.append(CHAT_CONTRACT)
    messages.append({"role": "user", "content": "\n\n".join(blocks)})
    return messages


def _referenced_runs(user_text: str) -> list[str]:
    """Run ids (32-hex) named in the operator's message, in order.

    Matches both ``harness_runs/<hex>`` / ``harness_runs\\<hex>`` references and
    a bare 32-hex word. The list feeds a hint to the agent so past-run questions
    are routed to the ``run`` tool instead of a fresh diagnosis.
    """
    found: list[str] = []
    for match in re.finditer(r"(?:harness_runs[\\/])?([0-9a-f]{32})", user_text,
                             re.IGNORECASE):
        run_id = match.group(1).lower()
        if run_id not in found:
            found.append(run_id)
    return found


def _register_lines(evidence: list) -> list[str]:
    lines: list[str] = []
    for ev in evidence[:24]:
        if isinstance(ev, dict) and ev.get("mnemonic"):
            fields = ", ".join(
                f"{f.get('name')}={f.get('raw_value')}"
                + (f"({f.get('meaning')})" if f.get("meaning") else "")
                for f in ev.get("decoded_fields") or []
                if isinstance(f, dict))
            line = f"{ev.get('mnemonic')} = {ev.get('raw_hex')}"
            if fields:
                line += f"  [{fields}]"
            if ev.get("unknown"):
                line += "  (unknown register)"
            lines.append(line)
        else:
            lines.append(_clip(ev, 120))
    return lines


def build_evidence(diagnosis: Diagnosis, run_dir: str | None = None) -> str:
    """Compact digest of the latest scored diagnosis, injected into every
    agent decision so follow-up questions are grounded in real evidence."""
    lines = ["## Latest Run Evidence"]
    if run_dir:
        lines.append(f"run_dir: {run_dir}")
    state = getattr(diagnosis.state, "value", diagnosis.state)
    lines.append(f"state: {state}  confidence: {diagnosis.confidence:.2f}")
    lines.append(f"verdict: {_clip(diagnosis.diagnosis, 600)}")
    if diagnosis.actions:
        lines.append("recommended actions:")
        for a in diagnosis.actions[:4]:
            lines.append(f"  {a.step}. {_clip(a.action, 160)} "
                         f"-- {_clip(a.rationale, 220)} (risk: {a.risk.value})")
    if diagnosis.references:
        refs = [f"{r.source}" + (f" {r.page}" if r.page else "")
                for r in diagnosis.references[:6]]
        lines.append("references: " + "; ".join(refs))
    if diagnosis.unknown_registers:
        lines.append("unknown registers (flagged, meanings NOT decoded): "
                     + ", ".join(diagnosis.unknown_registers[:10]))
    if diagnosis.parts_discrepancies:
        lines.append("parts discrepancies: "
                     + "; ".join(_clip(d, 160)
                                 for d in diagnosis.parts_discrepancies[:4]))
    if diagnosis.failure_point is not None:
        fp = diagnosis.failure_point
        lines.append(f"failure point (NOT a root cause): rail={fp.rail_tokens} "
                     f"reasons={'; '.join(_clip(r, 120) for r in fp.reasons[:4])} "
                     f"suspects={', '.join(fp.suspects[:6])}")
    registers = _register_lines(diagnosis.evidence)
    if registers:
        lines.append(f"decoded registers ({len(registers)} shown):")
        lines += [f"- {r}" for r in registers]
    return "\n".join(lines)


__all__ = [
    "CHAT_CONTRACT",
    "CHAT_SYSTEM",
    "CHAT_TOOLS",
    "SUBSYSTEMS",
    "ChatTurn",
    "build_evidence",
    "build_messages",
    "decide",
    "fallback_turn",
]
