"""Diagnostic orchestrator.

Sequences: plan (symptom -> minimal collectors) -> collect (via read-only runner)
-> decode (curated catalog) -> summarize -> retrieve docs -> validate parts ->
prompt -> LLM -> validate schema -> score -> return Diagnosis.

The LLM adapter is injected so the harness can run against a local/on-prem model
(proprietary-data safe). Everything is deterministic except the final reasoning
step, which is schema-validated and citation-enforced.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..engine.runner import Runner
from ..inspect.base import RegisterDump
from ..inspect.collectors import Collector
from ..inspect.decoder import Decoder
from ..inspect.model import DetectedModel, detect_model
from ..plan.profile import plan_collection
from .parts_validate import PartsCheckResult
from .prompt import build_prompt
from .schema import Diagnosis


@dataclass
class EngineContext:
    runner: Runner
    decoder: Decoder
    collector_factory: Callable[[str, Runner], Collector | None]  # name -> collector (None = skip)
    llm: Callable[[str], Diagnosis]                              # prompt -> validated Diagnosis
    docs_retriever: Callable[[str], list[str]] | None = None
    parts_refs: Callable[[], list[str]] | None = None
    parts_validate: Callable[[dict[str, list[RegisterDump]]], PartsCheckResult] | None = None
    scorer: Callable[[Diagnosis, dict[str, list[RegisterDump]]], Diagnosis] | None = None
    supervisor: Callable[[str], None] | None = None
    dump_callback: Callable[[dict[str, list[RegisterDump]]], None] | None = None
    progress: Callable[[str], None] | None = None  # human-readable step events (UI streaming)
    model_hook: Callable[[DetectedModel | None], None] | None = None  # audit of detected model


class DiagnosticEngine:
    def __init__(self, ctx: EngineContext) -> None:
        self.ctx = ctx

    def run(self, symptom: str) -> Diagnosis:
        _step = self.ctx.supervisor or (lambda _label: None)
        _emit = self.ctx.progress or (lambda _text: None)

        plan = plan_collection(symptom)
        _step("plan")
        _emit(f"plan: {len(plan.collectors)} collector(s), "
              f"primary subsystem: {plan.primary_subsystem or 'generic'}")

        model = detect_model(self.ctx.runner)
        if self.ctx.model_hook is not None:
            self.ctx.model_hook(model)

        # 1. Collect the minimal targetted set for the primary subsystem.
        dump_sets: dict[str, list[RegisterDump]] = {}
        for collector_name in plan.collectors:
            collector = self.ctx.collector_factory(collector_name, self.ctx.runner)
            if collector is None:
                continue
            _emit(f"collect: {collector_name}")
            dump_sets[collector_name] = collector.collect()
        _step("collect")

        # 2. Decode registers referenced by the dumps.
        decoded = self._decode_all(dump_sets)
        _step("decode")
        _emit(f"decode: {len(decoded)} register(s) decoded")

        # 3. Summarize anomalies for the LLM context budget.
        from .summarize import summarize
        all_dumps = [d for dumps in dump_sets.values() for d in dumps]
        summaries = summarize(all_dumps)

        # 4. Retrieve relevant docs + parts refs.
        snippets = self.ctx.docs_retriever(symptom) if self.ctx.docs_retriever else []
        parts = self.ctx.parts_refs() if self.ctx.parts_refs else []
        _step("retrieve")
        _emit(f"retrieve: {len(snippets)} doc snippet(s)")

        # 5. Build prompt and call the LLM.
        prompt = build_prompt(
            model=model,
            decoded=decoded,
            summaries=summaries,
            doc_snippets=snippets,
            parts_refs=parts,
            symptom=symptom,
        )
        _step("reason")
        _emit("reason: agent reasoning over evidence")
        diagnosis = self.ctx.llm(prompt)

        # 6. Attach structural evidence + score.
        diagnosis.subsystems_considered = _to_subsystems(plan.subsystem_order)
        diagnosis.evidence = [d.__dict__ for d in decoded]
        diagnosis.unknown_registers = [d.mnemonic for d in decoded if d.unknown]
        if self.ctx.dump_callback is not None:
            self.ctx.dump_callback(dump_sets)
        if self.ctx.parts_validate is not None:
            diagnosis.parts_discrepancies = self.ctx.parts_validate(dump_sets).discrepancies
        if self.ctx.scorer is not None:
            diagnosis = self.ctx.scorer(diagnosis, dump_sets)
        _emit(f"score: confidence {diagnosis.confidence:.2f}")
        return diagnosis

    def _decode_all(self, dump_sets: dict[str, list[RegisterDump]]):
        decoded = []
        for dumps in dump_sets.values():
            for dump in dumps:
                if not dump.ok:
                    continue
                decoded.extend(self.ctx.decoder.decode_many(dump.raw))
        return decoded


def _to_subsystems(ranking) -> list:
    from .schema import Subsystem
    out = []
    for r in ranking:
        try:
            out.append(Subsystem(r.subsystem))
        except ValueError:
            out.append(Subsystem.GENERIC)
    return out