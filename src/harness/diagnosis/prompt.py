"""Prompt builder (strict no-write, citation-enforced).

The prompt is assembled deterministically so the LLM always sees the safety rules,
the decoded evidence, and the instruction to cite exact pages. The no-write rule is
repeated in a prominent position and is also enforced structurally at output time
by the schema (actions are non-destructive recommendations).
"""

from __future__ import annotations

import json

from ..inspect.base import RegisterDecode
from ..inspect.model import DetectedModel
from .schema import Diagnosis, Reference
from .summarize import EvidenceSummary

SYSTEM_PREAMBLE = """You are a server-diagnostic assistant. You MAY ONLY suggest
non-destructive, verifiable actions. ABSOLUTE RULE: never recommend writing to a
hardware register, flashing firmware, or running any destructive/downgrade command.
Your output must be grounded: every assertion about register meaning or repair must
cite the source (document title + page) or a parts-list reference. Citations must
name an actual document (title + page) from the Relevant Architecture Snippets or
Parts List References -- the evidence sections of this prompt (Evidence Notes,
Anomalous Evidence Summary, etc.) are NOT citable sources. If any register
is flagged unknown, say so and suggest a manual lookup -- do NOT guess its meaning.

Evidence kind rules:
- IPMI SEL entries are a HISTORICAL event log: each entry records a PAST event with
  its own timestamp and is NOT proof of a current fault. Weight live sensor
  readings (current state) over stale SEL records.
- An asserted event with no matching deassert is not automatically an active fault;
  entries that recur at every power-on may be expected boot sequencing.
- State clearly in the diagnosis whether a fault is ACTIVE (current sensors/state
  disagree with nominal) versus HISTORICAL (only past log entries reference it).
- Set the structured "state" field explicitly: "healthy" when live sensors/state
  are nominal and only historical log entries reference past faults (the server is
  fixed); "fault" ONLY when current-state evidence shows a live problem;
  "degraded" for an active non-fatal issue; "unknown" only when evidence is
  insufficient. The "diagnosis" text must match the chosen state.

No-symptom runs: when the Symptom section is generic or absent, do not ask for
more information or refuse -- this is a scheduled evidence-only diagnosis (the
operator skipped the symptom question on purpose). Diagnose strictly from the
live evidence in this prompt: read the decoded boot-state registers (e.g. the
SWB CPLD boot-state block: CPU/reset flags at 0x1b, power-sequence FSM at 0xa1),
weigh current sensors over the SEL history, and state the most likely fault class
and repair path -- or a clean verdict when nothing is anomalous. If a non-dumped
register is flagged UNKNOWN, propose the read-only probe that would capture it
rather than guessing its value.

Prior Verified Cases are observed history from THIS fleet. Use them to weigh
likelihoods, never as documentation: they MUST NOT be cited as a Reference
source. If a prior case contradicts the vendor snippets, the snippets and
catalog win.

The System section is hardware-detected FACT with its source recorded. If you
believe it is wrong based on the snippets, say so explicitly instead of
silently assuming a different model."""

def _model_key(model: DetectedModel | None) -> str | None:
    if model is None or model.product_name == "unknown":
        return None
    return model.model_key.lower()


def _decode_lines(decoded: list[RegisterDecode], model_key: str | None) -> list[str]:
    """Register lines with catalog provenance; platform-mismatched decodes are
    flagged so the LLM distrusts them instead of silently assuming scope."""
    out: list[str] = []
    for d in decoded:
        mismatch = ""
        if d.platforms and model_key is not None \
                and model_key not in {p.lower() for p in d.platforms}:
            mismatch = " [PLATFORM MISMATCH - verify]"
        out.append(f"- {d.mnemonic} = {d.raw_hex}"
                   + (f" (UNKNOWN, manual lookup){mismatch}" if d.unknown
                      else f" [catalog {d.catalog_version}]{mismatch}"))
        for f in d.decoded_fields:
            out.append(f"    {f.name}: value={f.raw_value}"
                       + (f" -> {f.meaning}" if f.meaning else ""))
    return out


def _snippets_section(doc_snippets: list[str], model_key: str | None) -> list[str]:
    lines = ["## Relevant Architecture Snippets"]
    lines.extend(doc_snippets or ["(none)"])
    if doc_snippets:
        lines.append(f"(retrieved for platform {model_key})" if model_key
                     else "(retrieved with no platform filter)")
    return lines


def _prior_cases_section(prior_cases: list[str]) -> list[str]:
    lines = ["## Prior Verified Cases (fleet history, NOT vendor documentation)"]
    lines.extend(prior_cases or ["(none yet)"])
    return lines


def build_prompt(*,
                 model: DetectedModel | None,
                 decoded: list[RegisterDecode],
                 summaries: EvidenceSummary,
                 doc_snippets: list[str],
                 parts_refs: list[str],
                 symptom: str,
                 prior_cases: list[str] | None = None) -> str:
    model_key = _model_key(model)
    lines = [
        SYSTEM_PREAMBLE,
        "",
        "## System",
        f"model={model.product_name if model else 'unknown'}",
        f"model_source={model.source if model else 'unknown'}",
    ]
    if model is None:
        lines.append("model_note=detection failed; treat register/bit meanings as unverified")
    lines.extend([
        f"bios_vendor={model.bios_vendor if model else 'unknown'}",
        f"bios_version={model.bios_version if model else 'unknown'}",
        f"rag_platform_filter={model_key or 'none'}",
        "",
        "## Symptom",
        symptom,
        "",
        "## Decoded Registers",
    ])
    lines.extend(_decode_lines(decoded, model_key))
    lines.extend(["", "## Evidence Notes"])
    lines.extend(summaries.notes or ["(none)"])
    lines.extend(["", "## Anomalous Evidence Summary"])
    lines.extend(summaries.interesting or ["(none)"])
    lines.extend(_snippets_section(doc_snippets, model_key))
    lines.extend(["", "## Parts List References"])
    lines.extend(parts_refs or ["(none)"])
    lines.extend(["", *_prior_cases_section(prior_cases or [])])
    lines.extend([
        "",
        "Produce a Diagnosis. Every Action.rationale must cite a Reference with source and page.",
        "",
        ("Output MUST be STRICT JSON matching this exact schema (no markdown "
         "fences, no extra fields, no prose):"),
        json.dumps(Diagnosis.model_json_schema(), indent=2),
    ])
    return "\n".join(lines)


def ref(source: str, page: str | None = None, detail: str | None = None) -> Reference:
    return Reference(source=source, page=page, detail=detail)


TURN_SYSTEM_PREAMBLE = """You are a server-diagnostic assistant working through a
server one probe at a time. You MAY ONLY request read-only probes and ask clarifying
questions; you never receive command text and you never write to any register or
device. ABSOLUTE RULE: never recommend a raw write, firmware flash, or destructive
command -- the harness is force_read_only.

You work iteratively: the harness collects evidence, shows it to you, and you decide
the next step. You generally need SEVERAL rounds of probing, questions, and doc
lookups before you can produce a diagnosis -- do not rush.

When you ask a question, keep it to ONE concise question so the operator can answer
in a sentence (e.g. "Were there previous repair actions attempted, and what was
replaced?"). The operator may answer briefly or not at all; never block forever on
missing answers.

Evidence kind rules:
- IPMI SEL entries are a HISTORICAL event log: each entry records a PAST event with
  its own timestamp and is NOT proof of a current fault. Weight live sensor
  readings (current state) over stale SEL records.
- An asserted event with no matching deassert is not automatically an active fault;
  entries that recur at every power-on may be expected boot sequencing.
- State clearly in the final diagnosis whether a fault is ACTIVE (current
  sensors/state disagree with nominal) versus HISTORICAL (only past log entries
  reference it).
- Set the structured "state" field explicitly: "healthy" when live sensors/state
  are nominal and only historical log entries reference past faults (the server is
  fixed); "fault" ONLY when current-state evidence shows a live problem;
  "degraded" for an active non-fatal issue; "unknown" only when evidence is
  insufficient. The "diagnosis" text must match the chosen state.

The System section is hardware-detected FACT with its source recorded. If you
believe it is wrong based on the snippets, say so explicitly instead of
silently assuming a different model.

Prior Verified Cases are observed history from THIS fleet. Use them to weigh
likelihoods, never as documentation: they MUST NOT be cited as a Reference
source. If a prior case contradicts the vendor snippets, the snippets and
catalog win."""

TURN_CONTRACT = """Respond with STRICT JSON, exactly one of:
{"kind": "question", "question": "..."}
{"kind": "probe", "subsystems": ["memory"|"cpu"|"pcie"|"bmc"|"storage"|"kernel"], "doc_topics": ["..." ]}
{"kind": "diagnosis", "diagnosis": <full Diagnosis object as defined in the harness schema>}
- "question": you need one piece of human knowledge (previous repair actions,
  environmental history, what changed recently).
- "probe": you want the harness to run more read-only collectors (subsystems must
  come from the allowed list above) and/or retrieve more architecture doc sections
  (doc_topics). The harness maps these to curated read-only commands.
- "diagnosis": only when the evidence is sufficient. Diagnosis.actions are
  recommendations only; every rationale must cite a source+page or parts reference.
When evidence is thin and no human answer is available, prefer more probes/docs over
guessing."""


def build_turn_evidence(*,
                        model: DetectedModel | None,
                        symptom: str,
                        decoded: list[RegisterDecode],
                        summaries: EvidenceSummary,
                        doc_snippets: list[str],
                        parts_refs: list[str],
                        conversation: list[str],
                        prior_cases: list[str] | None = None) -> str:
    """User-message block for one session turn: current evidence + conversation."""
    model_key = _model_key(model)
    lines = [
        "## System",
        f"model={model.product_name if model else 'unknown'}",
        f"model_source={model.source if model else 'unknown'}",
    ]
    if model is None:
        lines.append("model_note=detection failed; treat register/bit meanings as unverified")
    lines.extend([
        f"bios_vendor={model.bios_vendor if model else 'unknown'}",
        f"bios_version={model.bios_version if model else 'unknown'}",
        f"rag_platform_filter={model_key or 'none'}",
        "",
        "## Symptom",
        symptom,
        "",
        "## Decoded Registers",
    ])
    lines.extend(_decode_lines(decoded, model_key))
    lines.extend(["", "## Evidence Notes"])
    lines.extend(summaries.notes or ["(none)"])
    lines.extend(["", "## Anomalous Evidence Summary"])
    lines.extend(summaries.interesting or ["(none)"])
    lines.extend(_snippets_section(doc_snippets, model_key))
    lines.extend(["", "## Parts List References"])
    lines.extend(parts_refs or ["(none)"])
    lines.extend(["", *_prior_cases_section(prior_cases or [])])
    lines.extend(["", "## Conversation So Far"])
    lines.extend(conversation or ["(none yet)"])
    lines.extend(["", TURN_CONTRACT])
    return "\n".join(lines)