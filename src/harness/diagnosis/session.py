"""Multi-turn diagnostic session.

A single ``diagnose --symptom "..."`` prompt is not the typical flow: the agent
generally has to traverse the server through many actions -- successive read-only
probes, doc lookups, and clarifying questions (e.g. "what was attempted in the
last repair?") -- before it can produce a diagnosable conclusion.

``SessionEngine`` implements that loop. Each turn the agent may:

- ask the operator ONE question (answers are fed back, e.g. previous repair actions),
- request more read-only probes (subsystem names mapped to curated collectors) and
  more architecture-doc retrieval (doc topics),
- or deliver the final ``Diagnosis`` (schema-validated, scored, audited).

The agent never supplies command text: probes are mapped to the read-only collector
registry / the doc library by the harness. ``max_turns`` bounds the loop and the
supervisor enforces the step/wall-clock budgets.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field

from ..inspect.base import RegisterDecode, RegisterDump
from ..inspect.model import PROFILE_COLLECTORS, detect_model
from ..plan.doc_guided import mine_probe_commands
from ..plan.profile import plan_collection
from .engine import EngineContext, _to_subsystems, decode_dumps
from .llm import LLMError
from .prompt import TURN_CONTRACT, TURN_SYSTEM_PREAMBLE, build_turn_evidence
from .schema import Diagnosis, TurnResponse
from .summarize import EvidenceSummary, summarize


class SessionError(RuntimeError):
    pass


def _no_answer(_question: str) -> str:
    """Non-interactive answer: the agent is told no information is available."""
    return ""


@dataclass
class SessionEngine:
    """Turn loop. ``llm`` must expose ``chat_json(messages) -> dict``."""

    ctx: EngineContext
    llm: object
    human_input: Callable[[str], str] = _no_answer
    max_turns: int = 6
    transcript: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not callable(getattr(self.llm, "chat_json", None)):
            raise SessionError(
                "session mode requires an LLM with chat_json (use --llm openai)")
        self._dumps: dict[str, list[RegisterDump]] = {}
        self._decoded: list[RegisterDecode] = []
        self._summaries: EvidenceSummary = EvidenceSummary(interesting=[], anomaly_count=0, total=0)
        self._topic_snippets: list[str] = []
        self._base_snippets: list[str] = []
        self._model = None
        self._plan = None
        self._prompt_seq = 0

    def run(self, symptom: str, initial_answers: list[str] | tuple[str, ...] = ()) -> Diagnosis:
        _step = self.ctx.supervisor or (lambda _label: None)
        _emit = self.ctx.progress or (lambda _text: None)
        for answer in initial_answers:
            if answer:
                self.transcript.append({"role": "user", "content": answer, "kind": "context"})

        for _turn in range(1, self.max_turns + 1):
            _emit(f"turn {_turn}/{self.max_turns}: agent working")
            _step("session_turn")
            try:
                raw = self.llm.chat_json(self._turn_messages(symptom))
            except LLMError as exc:
                self._note(f"agent call failed: {exc}; continuing")
                continue
            resp = self._parse_turn(raw)
            if resp is None:
                self._note("malformed agent response (expected question|probe|diagnosis JSON)")
                continue
            if resp.kind == "question":
                question = resp.question or "(agent asked a question without text)"
                self.transcript.append({"role": "agent", "content": question, "kind": "question"})
                _emit(f"agent asks: {question}")
                answer = self.human_input(question)
                self.transcript.append({"role": "user", "content": answer or "(no answer)",
                                        "kind": "answer"})
            elif resp.kind == "probe":
                result = self._apply_probe(resp)
                _emit(result or "probe: nothing new")
                self.transcript.append({
                    "role": "tool", "content": result or "(nothing new)",
                    "kind": "probe",
                })
            else:  # diagnosis
                diag = self._finalize(resp.diagnosis, symptom)
                self.transcript.append({"role": "agent", "content": diag.diagnosis,
                                        "kind": "diagnosis"})
                _emit("diagnosis delivered")
                return diag
        return self._forced_diagnosis(symptom)

    # ---- evidence gathering ----

    def _ensure_initial(self, symptom: str) -> None:
        if self._plan is not None:
            return
        from ..inspect.model import DetectedModel
        self._base_snippets = (self.ctx.docs_retriever(symptom)
                               if self.ctx.docs_retriever else [])
        self._plan = plan_collection(symptom, self._base_snippets)
        # Console runners: run the whole plan's probes in ONE serial session
        # (~35s each) and let the collectors dedupe against the probe cache.
        from .engine import prebatch_console_plan
        planned = [(name, self.ctx.collector_factory(name, self.ctx.runner))
                   for name in self._plan.collectors]
        prebatch_console_plan(self.ctx, self._plan, planned)
        self._model = detect_model(self.ctx.runner) or DetectedModel(
            product_name="unknown", bios_vendor="unknown", bios_version=None, raw="")
        self._collect_all(self._plan.collectors)
        self._collect_doc_probes(self._plan.doc_probes)

    def _collect_all(self, names: list[str]) -> list[str]:
        added: list[str] = []
        for name in names:
            if name in self._dumps:
                continue
            collector = self.ctx.collector_factory(name, self.ctx.runner)
            if collector is None:
                self._note(f"collector {name!r} unavailable; skipped")
                continue
            self._dumps[name] = collector.collect()
            added.append(name)
        if added:
            self._refresh()
        return added

    def _collect_doc_probes(self, commands: list[str]) -> list[str]:
        """Run doc-named probes (deduped against already-run commands).

        Returns the sources of probes actually executed this call.
        """
        if not commands:
            return []
        from ..inspect.collectors.doc_guided import DocGuidedProbeCollector
        collector = DocGuidedProbeCollector(self.ctx.runner, commands)
        new_dumps = collector.collect()
        merged = self._dumps.get("doc_guided", []) + new_dumps
        seen: set[str] = set()
        out = []
        for dump in merged:
            if dump.source in seen:
                continue
            seen.add(dump.source)
            out.append(dump)
        self._dumps["doc_guided"] = out
        self._refresh()
        return [d.source for d in new_dumps]

    def _apply_probe(self, resp: TurnResponse) -> str:
        lines: list[str] = []
        for subsystem in resp.subsystems:
            if subsystem not in PROFILE_COLLECTORS:
                self._note(f"agent requested unknown subsystem {subsystem!r}; skipped")
                lines.append(f"unknown subsystem {subsystem!r} (skipped)")
                continue
            added = self._collect_all(PROFILE_COLLECTORS[subsystem])
            lines.append(f"{subsystem}: collected {', '.join(added) or 'already held'}")
        for topic in resp.doc_topics:
            if not self.ctx.docs_retriever:
                lines.append(f"doc topic {topic!r}: no doc library (skipped)")
                continue
            snippets = self.ctx.docs_retriever(topic)
            self._topic_snippets.extend(snippets)
            lines.append(f"doc topic {topic!r}: {len(snippets)} snippet(s) retrieved")
            probes = mine_probe_commands(snippets)
            if probes:
                added = self._collect_doc_probes(probes)
                lines.append(f"doc-named probes: {', '.join(added) or 'already run'}")
        return "; ".join(lines)

    def _refresh(self) -> None:
        all_dumps = [d for dumps in self._dumps.values() for d in dumps]
        self._decoded = decode_dumps(self.ctx.decoder, all_dumps)
        self._summaries = summarize(all_dumps)

    def _snippets(self, symptom: str) -> list[str]:
        base = self._base_snippets
        if not base and self.ctx.docs_retriever:
            base = self.ctx.docs_retriever(symptom)
        seen = set(base)
        for s in self._topic_snippets:
            if s not in seen:
                base.append(s)
                seen.add(s)
        return base

    # ---- messages ----

    def _turn_messages(self, symptom: str) -> list[dict]:
        self._ensure_initial(symptom)
        messages: list[dict] = [{"role": "system", "content": TURN_SYSTEM_PREAMBLE}]
        for entry in self.transcript:
            role = "assistant" if entry["role"] == "agent" else "user"
            content = entry["content"]
            if entry["role"] == "tool":
                content = f"[probe result] {content}"
            messages.append({"role": role, "content": content})
        parts = self.ctx.parts_refs() if self.ctx.parts_refs else []
        conversation = [f"{t['role']}: {t['content']}" for t in self.transcript]
        messages.append({"role": "user", "content": build_turn_evidence(
            model=self._model,
            symptom=symptom,
            decoded=self._decoded,
            summaries=self._summaries,
            doc_snippets=self._snippets(symptom),
            parts_refs=parts,
            conversation=conversation,
        )})
        if self.ctx.prompt_callback is not None:
            self._prompt_seq += 1
            self.ctx.prompt_callback(json.dumps(
                {"turn": self._prompt_seq, "messages": messages}, indent=2))
        return messages

    def _forced_diagnosis(self, symptom: str) -> Diagnosis:
        """Turns exhausted: force a final diagnosis from the agent."""
        messages = self._turn_messages(symptom)
        messages[-1] = {"role": "user", "content": messages[-1]["content"] + (
            "\n\nYou have exhausted the turn budget. Return the final diagnosis now: "
            f'{{"kind": "diagnosis", "diagnosis": {{...}}}} {TURN_CONTRACT}')}
        try:
            raw = self.llm.chat_json(messages)
        except LLMError as exc:
            raise SessionError(f"no diagnosis before turn budget exhausted: {exc}") from exc
        resp = self._parse_turn(raw)
        if resp is None or resp.kind != "diagnosis" or resp.diagnosis is None:
            raise SessionError("agent did not produce a diagnosis within the turn budget")
        return self._finalize(resp.diagnosis, symptom)

    def _finalize(self, diag: Diagnosis, symptom: str) -> Diagnosis:
        assert self._plan is not None
        diag.subsystems_considered = _to_subsystems(self._plan.subsystem_order)
        diag.evidence = [d.__dict__ for d in self._decoded]
        diag.unknown_registers = [d.mnemonic for d in self._decoded if d.unknown]
        if self.ctx.dump_callback is not None:
            self.ctx.dump_callback(self._dumps)
        if self.ctx.parts_validate is not None:
            diag.parts_discrepancies = self.ctx.parts_validate(self._dumps).discrepancies
        if self.ctx.scorer is not None:
            diag = self.ctx.scorer(diag, self._dumps)
        return diag

    @staticmethod
    def _parse_turn(raw: dict) -> TurnResponse | None:
        try:
            resp = TurnResponse.model_validate(raw)
        except Exception:  # noqa: BLE001 - any malformed turn is skipped, not fatal
            return None
        if resp.kind == "question" and not resp.question:
            return None
        if resp.kind == "probe" and not resp.subsystems and not resp.doc_topics:
            return None
        if resp.kind == "diagnosis" and resp.diagnosis is None:
            return None
        return resp

    def _note(self, text: str) -> None:
        self.notes.append(text)
        self.transcript.append({"role": "tool", "content": f"[note] {text}", "kind": "note"})
