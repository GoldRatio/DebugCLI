"""Diagnostic orchestrator.

Sequences: plan (symptom -> minimal collectors, plus doc-named probes) -> collect
(via read-only runner) -> decode (curated catalog) -> summarize -> retrieve docs ->
validate parts -> prompt -> LLM -> validate schema -> score -> return Diagnosis.

Docs are retrieved BEFORE collection so doc-named probe commands (e.g. ``i2cdump -y
8 0xb`` for amber-light boot state) join the plan; the same snippets feed the prompt.

The LLM adapter is injected so the harness can run against a local/on-prem model
(proprietary-data safe). Everything is deterministic except the final reasoning
step, which is schema-validated and citation-enforced.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..engine.runner import Runner
from ..inspect.base import RegisterDecode, RegisterDump
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
    prompt_callback: Callable[[str], None] | None = None  # audit: exact prompt(s) sent to the LLM
    progress: Callable[[str], None] | None = None  # human-readable step events (UI streaming)
    model_hook: Callable[[DetectedModel | None], None] | None = None  # audit of detected model


def prebatch_console_plan(ctx: EngineContext, plan, collectors) -> None:
    """Run every planned console probe in ONE serial session (~35s each).

    ``collectors`` are ``(name, collector_or_None)`` pairs from the factory.
    Probes any collector would issue (non-i2c + the first CPLD-chain candidate)
    plus the model probe and the doc-named probes are batch-executed and cached
    on the runner; individual collector/probe requests then dedupe against the
    cache instead of opening a new session.
    """
    if not (getattr(ctx.runner, "is_console", False)
            and hasattr(ctx.runner, "batch_execute")):
        return
    cmds: list[str] = ["sudo -S ipmitool fru print"]  # model probe
    for _name, collector in collectors:
        if collector is not None:
            cmds += list(getattr(collector, "candidate_probes", list)())
    if plan.doc_probes:
        from ..inspect.collectors.doc_guided import _PRIVILEGED
        for probe in plan.doc_probes:
            prog = probe.split()[0].split("/")[-1]
            if prog in _PRIVILEGED and not probe.startswith("sudo -S "):
                probe = "sudo -S " + probe
            cmds.append(probe)
    ordered: list[str] = []
    ok_done = {" ".join(c.argv)
               for c in getattr(ctx.runner, "calls", []) if c.ok}
    for cmd in cmds:
        if cmd not in ordered and cmd not in ok_done:
            ordered.append(cmd)
    if ordered:
        emit = ctx.progress or (lambda _text: None)
        emit(f"collect: {len(ordered)} probe(s) in one console session")
        ctx.runner.batch_execute(ordered)


class DiagnosticEngine:
    def __init__(self, ctx: EngineContext) -> None:
        self.ctx = ctx

    def run(self, symptom: str) -> Diagnosis:
        _step = self.ctx.supervisor or (lambda _label: None)
        _emit = self.ctx.progress or (lambda _text: None)

        # 0. Retrieve docs FIRST so doc-named probes join the plan (the heuristic
        # cannot know which registers a failure mode needs). Reused for the prompt.
        snippets = self.ctx.docs_retriever(symptom) if self.ctx.docs_retriever else []
        _step("retrieve")
        _emit(f"retrieve: {len(snippets)} doc snippet(s)")

        plan = plan_collection(symptom, snippets)
        _step("plan")
        _emit(f"plan: {len(plan.collectors)} collector(s), "
              f"{len(plan.doc_probes)} doc-named probe(s), "
              f"primary subsystem: {plan.primary_subsystem or 'generic'}")

        # Build the collectors once so a console runner can pre-batch every
        # probe into ONE serial session (~35s each otherwise).
        collectors: list[tuple[str, object]] = [
            (name, collector) for name in plan.collectors
            if (collector := self.ctx.collector_factory(name, self.ctx.runner))
            is not None
        ]
        prebatch_console_plan(self.ctx, plan, collectors)

        model = detect_model(self.ctx.runner)
        if self.ctx.model_hook is not None:
            self.ctx.model_hook(model)

        # 1. Collect the minimal targetted set for the primary subsystem.
        dump_sets: dict[str, list[RegisterDump]] = {}
        for collector_name, collector in collectors:
            _emit(f"collect: {collector_name}")
            dump_sets[collector_name] = collector.collect()
        # 1b. Run the doc-named probes (deduped against already-run commands).
        if plan.doc_probes:
            from ..inspect.collectors.doc_guided import DocGuidedProbeCollector
            _emit(f"collect: {len(plan.doc_probes)} doc-named probe(s)")
            dump_sets["doc_guided"] = DocGuidedProbeCollector(
                self.ctx.runner, plan.doc_probes).collect()
        _step("collect")

        # 2. Decode registers referenced by the dumps.
        decoded = self._decode_all(dump_sets)
        _step("decode")
        _emit(f"decode: {len(decoded)} register(s) decoded")

        # 3. Summarize anomalies for the LLM context budget.
        from .summarize import summarize
        all_dumps = [d for dumps in dump_sets.values() for d in dumps]
        summaries = summarize(all_dumps)

        # 4. Parts refs (snippets already retrieved at step 0).
        parts = self.ctx.parts_refs() if self.ctx.parts_refs else []

        # 5. Build prompt and call the LLM.
        prompt = build_prompt(
            model=model,
            decoded=decoded,
            summaries=summaries,
            doc_snippets=snippets,
            parts_refs=parts,
            symptom=symptom,
        )
        if self.ctx.prompt_callback is not None:
            self.ctx.prompt_callback(prompt)
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
        return decode_dumps(
            self.ctx.decoder,
            [d for dumps in dump_sets.values() for d in dumps],
        )


def decode_dumps(decoder: Decoder, dumps: list[RegisterDump]) -> list[RegisterDecode]:
    """Decode every successful dump, dispatching by probe tool."""
    decoded: list[RegisterDecode] = []
    for dump in dumps:
        if not dump.ok:
            continue
        if "i2cdump" in dump.source:
            decoded.extend(decoder.decode_i2c_dump(dump.raw))
        elif "i2ctransfer" in dump.source:
            decoded.extend(decoder.decode_i2c_transfer(dump.raw, dump.source))
        elif "i2cget" in dump.source:
            decoded.extend(decoder.decode_i2c_get(dump.raw, dump.source))
        else:
            decoded.extend(decoder.decode_many(dump.raw))
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