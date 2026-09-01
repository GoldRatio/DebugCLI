"""CLI entry point: inventory -> target -> session -> plan -> collect -> decode ->
RAG -> LLM -> score -> approval -> audit -> JSON/trace output.

Subcommands:

- ``menu``    interactive launcher (also the default when no subcommand is
              given): pick the inventory, pick a target, then run chat /
              diagnose / verify / console / docs / targets / secrets / lint
              without remembering any flags
- ``lint``    validate an inventory (rejects inline secrets)
- ``docs``      manage the RAG document library: ``add`` uploads PDFs, ``ls``/``rm``
                list/remove them, ``reindex`` retries failed or dropped files
- ``diagnose``  read-only diagnosis of one host. Default is a single pass
                (symptom -> collect -> RAG -> LLM -> actions). With ``--interactive``
                (or ``--context``/``--context-file``) it becomes a multi-turn session:
                the agent may ask the operator questions (e.g. previous repair
                actions) and request further read-only probes / doc lookups over
                several turns before producing the diagnosis. ``--docs-lib`` points
                at the library managed by ``docs``. Targeting needs NO per-server
                YAML: ``--host <name>`` (inventory) or ``--rack R --cable N``
                (fleet console_defaults) or ``--address <ip>`` (SSH, identity from
                the secret store) or ``--target <alias>`` (harness targets).
                ``--console`` dials the serial console instead of SSH, and
                ``--console-address``/``--rack``/``--cable``/``--port``/
                ``--sudo-vault-path`` override the console target per launch.
- ``console``   run read-only probes over the serial console (lab/qa only); per
                launch select the rack manager console and cable with
                ``--console-address``/``--rack``/``--cable``/``--port``/
                ``--sudo-vault-path``
- ``verify``    re-run the collectors and compare against a stored baseline
                (post-repair "did it change" check)
- ``session``   interactive Claude-Code-style chat: describe symptoms in natural
                language and the agent diagnoses read-only in the background
                while you keep typing. Messages may name a target ("Diagnose the
                server in Q61 Cable 8" / "Diagnose 10.0.0.10"). Slash commands
                (``/help``, ``/use``, ``/status``, ``/stop``, ``/resume``, ...) and
                queued context are supported; see ``operator.repl``.
- ``secrets``   NON-AGENT credential CLI (never through the prompt): ``add-ssh``
                (key from a FILE only), ``set-password`` (interactive), ``list``,
                ``rm``, ``check``. Writes only into the secret store; every
                registration is audit-logged (redacted).
- ``targets``   short aliases for dynamic targets: ``add <alias> --rack R
                --cable N [--address ip]``, ``ls``, ``rm`` (path-only file).

The run directory holds: ``diagnosis.json`` (machine-readable action list),
``trace.json`` (session trace with command hashes), ``dumps/`` (raw register
dumps), ``dumps.json`` (baseline for ``verify``), ``transcript.json`` (the
multi-turn conversation, when in session mode), and ``audit.jsonl`` (WORM
hash-chained log, secrets redacted).
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import re
import shlex
import sys
from collections.abc import Callable
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import yaml

from ..audit.auditlog import AuditLog
from ..audit.redact import Redactor
from ..audit.trace import SessionTrace
from ..config.inventory_lint import load_inventory
from ..config.models import ConsoleDomain, Host, Inventory
from ..config.vault import DirSecretStore, MemorySecretStore, SecretStore
from ..diagnosis.engine import DiagnosticEngine, EngineContext
from ..diagnosis.llm import (
    GeminiLLM,
    LLMError,
    LocalLLM,
    OpenAICompatLLM,
    StubLLM,
    list_models,
    probe_chat,
)
from ..diagnosis.parts_validate import PartsValidator
from ..diagnosis.schema import Diagnosis
from ..diagnosis.scorer import evidence_fit_from_dumps, score_diagnosis
from ..diagnosis.session import SessionEngine
from ..diagnosis.verifier import Verifier
from ..engine.allowlist import default_policy
from ..engine.bmc import BmcRunner
from ..engine.interactive import InteractiveShell
from ..engine.redfish import RedfishClient
from ..engine.runner import CommandResult, Runner
from ..engine.session import SSHSession
from ..engine.single_test import SingleTestDriver, SingleTestError
from ..engine.sol import (
    ConsoleRunner,
    SerialConsole,
    SerialConsoleError,
    _absolutize_bmc_i2c_tools,
    validate_serial_probe,
)
from ..engine.tunnel import LLMForward, TunnelError, parse_tunnel_spec
from ..inspect.base import RegisterDump
from ..inspect.collectors.bmc_console import BmcConsoleCollector
from ..inspect.collectors.ipmi import IpmiCollector
from ..inspect.collectors.redfish import RedfishCollector
from ..inspect.decoder import Decoder
from ..inspect.registry import make_collector
from ..operator.gate import ApprovalDecision, ApprovalGate
from ..operator.supervisor import Escalation, RunSupervisor
from ..plan.profile import plan_collection
from ..platforms import family_for
from ..targets import TargetError, TargetSpec, resolve_target
from ..targets.aliases import AliasError, TargetAlias, load_targets, save_targets
from . import ui
from .credential_gate import apply_ssh_context

# ---- helpers ----

# Menu-driven diagnosis runs without asking for a symptom prompt: target first,
# then diagnose straight from the live evidence (see SYSTEM_PREAMBLE in
# diagnosis/prompt.py for how the LLM handles a generic symptom).
_MENU_DIAGNOSE_SYMPTOM = (
    "No specific symptom was reported by the operator. Diagnose this server "
    "from the live evidence in this prompt (boot state, sensors, SEL, kernel "
    "log). Report the current state and the most likely fault class or verdict."
)

def _make_store(args, *, prompt: bool | None = None) -> SecretStore:
    """Resolve the lab secret store; when running interactively, wrap it so a
    missing vault path prompts the OPERATOR on demand (never the agent/LLM)
    instead of failing the run (see ``operator.credential_gate``)."""
    secret_dir = getattr(args, "secret_dir", None)
    if not secret_dir:
        # auto-discover the well-known lab store, mirroring inventory discovery
        for candidate in ("secrets", ".secrets"):
            if Path(candidate).is_dir():
                secret_dir = candidate
                break
    interactive = (prompt if prompt is not None
                   else (sys.stdin.isatty()
                         and os.environ.get("HARNESS_NO_PROMPT") != "1"))
    if not secret_dir:
        if not interactive:
            return MemorySecretStore()
        # zero-config start: on-demand credentials must PERSIST between runs
        secret_dir = "secrets"
    base: SecretStore = DirSecretStore(secret_dir)
    if not interactive:
        return base
    from .credential_gate import CredentialPrompter, OnDemandSecretStore
    return OnDemandSecretStore(base, CredentialPrompter(base))


def _resolve_target_from_args(args, inv: Inventory, store: SecretStore):
    """Runtime spec -> Target: --host (named) | --rack/--cable (console_defaults)
    | --address <ip> (SSH via store identity) | --target <alias>."""
    spec = TargetSpec(
        name=getattr(args, "host", None),
        rack=getattr(args, "rack", None),
        cable=getattr(args, "cable", None),
        ip=getattr(args, "address", None),
        alias=getattr(args, "target", None),
    )
    return resolve_target(
        spec, inv, store,
        targets_path=getattr(args, "targets_file", None),
        ssh_user=getattr(args, "ssh_user", "diagbot"),
        identity_vault_path=getattr(args, "identity_vault_path", None),
        known_hosts_path=getattr(args, "known_hosts_path", "config/known_hosts"),
    )


def _resolve_profile(args, inv: Inventory) -> object:
    """Resolve the effective LLM profile under the shared precedence.

    ``--llm-model <ident>`` > ``--llm <provider>`` > persisted current model
    (``config/models.yaml``) > inventory ``llm`` block > default. An explicit
    ``--llm-url`` overrides the endpoint of whichever profile resolves (and
    ``--llm-tunnel`` rewrites ``args.llm_url`` to the forward's local URL in
    ``_prepare_llm_endpoint``). Both ``_resolve_llm`` and ``_llm_ident_for``
    go through here so the live adapter and its calibration ident can never
    drift apart.
    """
    from ..config.model_catalog import ModelCatalog

    catalog = ModelCatalog.load(inv=inv)
    return catalog.resolve(provider=getattr(args, "llm", None),
                           model_id=getattr(args, "llm_model", None),
                           url=getattr(args, "llm_url", None))


def _resolve_llm(args, inv: Inventory, store: SecretStore) -> object:
    """LLM backend resolution (see ``_resolve_profile``). The api key, when
    vault-path configured, is resolved through the secret store; otherwise env
    fallbacks apply (e.g. GEMINI_API_KEY)."""
    return _resolve_profile(args, inv).build(store)


def _llm_ident_for(args, inv: Inventory) -> str:
    """Stable identity of the resolved LLM backend (calibration key).

    Matches ``_resolve_llm`` resolution order; e.g. ``stub``,
    ``openai/gpt-4o``, ``local/Qwen2.5-7B-Instruct`` or
    ``gemini/gemini-2.5-flash``. The ident is what the per-model calibration
    store is keyed on -- it must not change for the same configured model, or
    calibration silently goes inactive (0.5).
    """
    return _resolve_profile(args, inv).ident


def _preflight_llm_url(url: str) -> None:
    """Fast reachability probe for an explicit ``--llm-url`` before any
    collecting starts: a diagnosis costs register dumps; a dead endpoint must
    fail in milliseconds, not after the pipeline is up."""
    try:
        list_models(url, timeout=10.0)
    except LLMError as exc:
        raise RuntimeError(
            f"LLM endpoint {url} unreachable ({exc}). Run "
            f"`harness llm check --url {url}` for staged diagnostics.") from exc


def _prepare_llm_endpoint(args, inv: Inventory, store: SecretStore):
    """Per-run LLM endpoint setup (``--llm-tunnel`` / ``--llm-url`` / profile).

    Explicit flags win. With neither flag given, the resolved model profile
    (``config/models.yaml`` current) supplies the endpoint: a profile
    ``tunnel`` opens an ``LLMForward`` through the inventory's rack manager
    and rewrites ``args.llm_url`` to the forward's local URL so every adapter
    resolution binds to it; a profile ``url`` is preflighted like an explicit
    ``--llm-url``. Returns the open forward (or None); it is tracked in
    ``_ACTIVE_LLM_FORWARDS`` so ``_close_llm_forwards`` can tear it down
    deterministically at run end, and an atexit hook guarantees no hop
    outlives the process.
    """
    tunnel_spec = getattr(args, "llm_tunnel", None)
    url = getattr(args, "llm_url", None)
    profile_url = None
    if not tunnel_spec and not url:
        profile = _resolve_profile(args, inv)
        tunnel_spec = getattr(profile, "tunnel", None)
        profile_url = getattr(profile, "url", None)
    if not tunnel_spec:
        effective_url = url or profile_url
        if effective_url:
            _preflight_llm_url(effective_url)
        return None
    target_host, target_port = parse_tunnel_spec(tunnel_spec)
    from .llm_discover import llm_bastion_domain, llm_console_domain

    domain = llm_console_domain(inv, getattr(args, "rack", None) or "",
                                getattr(args, "cable", None) or "")
    if domain is None:
        raise RuntimeError(
            "--llm-tunnel requires a fleet llm_console (or console_defaults) "
            "block in the inventory (the rack-manager SSH hop is the only "
            "path to nodes)")
    forward = LLMForward(target_host, target_port, domain, store,
                         timeout=float(getattr(args, "llm_timeout", 30.0)),
                         bastion=llm_bastion_domain(
                             inv, getattr(args, "rack", None) or "",
                             getattr(args, "cable", None) or ""))
    try:
        args.llm_url = forward.start()
    except TunnelError as exc:
        forward.close()
        raise _tunnel_failure(exc, f"{target_host}:{target_port}") from exc
    _ACTIVE_LLM_FORWARDS.append(forward)
    return forward


_ACTIVE_LLM_FORWARDS: list = []


def _close_llm_forwards() -> None:
    """Tear down every open rack-manager forward (idempotent per instance)."""
    while _ACTIVE_LLM_FORWARDS:
        try:
            _ACTIVE_LLM_FORWARDS.pop().close()
        except Exception:  # noqa: BLE001, S110 - shutdown must never raise
            pass


atexit.register(_close_llm_forwards)


def _tunnel_failure(exc: TunnelError, target: str) -> RuntimeError:
    """Stage-specific operator guidance for a failed rack-manager forward."""
    if exc.stage == "auth":
        return RuntimeError(
            f"rack-manager SSH hop failed: {exc} (check llm_console/console_defaults "
            "identity / known_hosts; `harness llm check --tunnel ...` "
            "re-tests each stage)")
    if exc.stage == "forward":
        relay_port = 18000
        return RuntimeError(
            f"rack manager refused or could not route a channel to {target} "
            f"(sshd forwarding disabled, or no route to the node): {exc}. "
            "Fallback (reverse tunnel from the node): console onto the node "
            "and run `ssh -fN -R 127.0.0.1:"
            f"{relay_port}:127.0.0.1:<node-port> <relay-reachable-from-node>`, "
            f"then pass --llm-url http://127.0.0.1:{relay_port}/v1 instead of "
            "--llm-tunnel")
    return RuntimeError(str(exc))


def _console_overrides(domain: ConsoleDomain, args) -> ConsoleDomain:
    """Per-launch console selection: rack manager console (address), rack id and
    cable are chosen at each launch and layered over the inventory defaults.
    Rack/cable are identifier-validated (injection-safe into the expect script)."""
    from ..engine.sol import validate_identifier
    kw: dict = {}
    addr = getattr(args, "console_address", None)
    rack = getattr(args, "rack", None)
    cable = getattr(args, "cable", None)
    if addr:
        kw["address"] = addr
    if rack:
        kw["rack"] = validate_identifier(rack, "rack")
    if cable:
        kw["cable"] = validate_identifier(cable, "cable")
    if getattr(args, "port", None) is not None:
        kw["port"] = args.port
    if getattr(args, "sudo_vault_path", None) is not None:
        kw["sudo_vault_path"] = args.sudo_vault_path
    return replace(domain, **kw) if kw else domain


def _console_sudo_secret(store: SecretStore, domain: ConsoleDomain, secrets: list[str]) -> None:
    """Best-effort: add the BMC sudo password to the redaction set. The console
    resolves the secret itself at probe time; here we only ensure the output is
    redacted from stdout/audit."""
    if domain.sudo_vault_path is None:
        return
    try:
        secrets.append(store.get(domain.sudo_vault_path).decode().rstrip("\r\n"))
    except KeyError:
        pass


def _build_redfish_collector(domain: ConsoleDomain, store: SecretStore,
                             secrets: list[str], progress) -> RedfishCollector | None:
    """Redfish collector for the debug step: rack-level read-only evidence
    (event logs, service conditions) fetched over HTTPS from the rack manager
    -- no serial session, no server jumpin.

    Enabled by the console block's ``redfish_password_vault_path``; a missing
    vault secret disables it with a visible note (staged, never fatal). The
    password joins the redaction set before any request is made.
    """
    if domain.redfish_password_vault_path is None:
        return None
    try:
        secrets.append(store.get(domain.redfish_password_vault_path)
                       .decode().rstrip("\r\n"))
    except KeyError:
        if progress is not None:
            progress(f"redfish disabled: secret missing from vault "
                     f"({domain.redfish_password_vault_path!r})")
        return None
    return RedfishCollector(
        RedfishClient(domain, domain.cable, store))


def _build_retriever(docs_dir: str | None):
    """Build a RAG retriever over a directory of supported documents (PDF /
    markdown / text / CSV). Returns None when absent."""
    if not docs_dir:
        return None
    try:
        from ..docs.ingest.chunk import CharTokenizer, Chunker
        from ..docs.ingest.library import SUPPORTED_SUFFIXES, parse_pages
        from ..docs.retrieval.hybrid_search import HybridRetriever
        from ..docs.retrieval.rag import RagPipeline
    except ImportError:
        print("  [docs] extras not installed; install harness[docs] for --docs-dir",
              file=sys.stderr)
        return None
    chunks = []
    for pdf in sorted(Path(docs_dir).iterdir()):
        if not pdf.is_file() or pdf.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        try:
            parsed = parse_pages(pdf)
        except Exception as exc:  # noqa: BLE001 - unreadable docs skipped, never fatal
            print(f"  [docs] skip {pdf.name}: {exc}", file=sys.stderr)
            continue
        chunks.extend(Chunker(CharTokenizer()).chunk_pages(parsed, title=pdf.name))
    if not chunks:
        print(f"  [docs] no readable documents in {docs_dir!r}; proceeding without RAG",
              file=sys.stderr)
        return None
    rag = RagPipeline(HybridRetriever(chunks))
    return lambda query, model_key: rag.lines(query, top_k=5, platform=model_key)


def _build_docs_retriever(args):
    """RAG retriever: use --docs-lib (or the default harness_docs/ library when it
    exists), else the legacy ad-hoc --docs-dir. Always prints whether RAG is
    active, so a run is never silently unreferenced.

    Returns ``(retriever, docs_lib_used)``; both None when no docs are available.
    """
    lib_dir = getattr(args, "docs_lib", None)
    if lib_dir is None and Path("harness_docs").is_dir():
        lib_dir = "harness_docs"
    docs_dir = getattr(args, "docs_dir", None)
    if lib_dir and Path(lib_dir).is_dir():
        try:
            from ..docs.ingest.library import DocLibrary
            library = DocLibrary(lib_dir)
            rag = library.build_retriever()
        except Exception as exc:  # noqa: BLE001 - never let docs block a diagnosis
            print(f"  [docs] doc library error: {exc}; proceeding without RAG",
                  file=sys.stderr)
            return None, None
        if rag is None:
            print(f"  [docs] doc library {lib_dir!r} is empty; run 'harness docs add' "
                  "to upload PDFs", file=sys.stderr)
            return None, None
        total = sum(e.chunks for e in library.entries())
        print(f"  [docs] using library {lib_dir!r}: {total} chunk(s)", file=sys.stderr)
        return (lambda query, model_key: rag.lines(query, top_k=5,
                                                   platform=model_key)), lib_dir
    if docs_dir:
        return _build_retriever(docs_dir), None
    print("  [docs] proceeding without RAG (run 'harness docs add' or pass --docs-lib)",
          file=sys.stderr)
    return None, None


def _load_context(args) -> list[str]:
    """Human-supplied context: --context strings plus --context-file contents."""
    answers: list[str] = list(getattr(args, "context", None) or [])
    for path in getattr(args, "context_file", None) or []:
        try:
            answers.append(Path(path).read_text(encoding="utf-8").strip())
        except OSError as exc:
            print(f"  [context] skip {path}: {exc}", file=sys.stderr)
    return [a for a in answers if a]


def _strip_surrounding_quotes(value: str) -> str:
    """Trim one level of surrounding quotes a user may have pasted into a
    prompt (e.g. ``"C:\\path\\log.log"`` or ``'C:\\path\\log.log'``)."""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def _load_test_logs(args) -> list:
    """Load operator-supplied test logs via the log-source seam (files today, a
    website fetcher later). Unreadable/unparseable logs are reported and
    skipped -- a bad log must never block a diagnosis."""
    from ..testlog import FileLogSource, LogSourceError

    reports = []
    for path in getattr(args, "test_log", None) or []:
        try:
            reports.append(FileLogSource(_strip_surrounding_quotes(path)).load())
        except (OSError, LogSourceError) as exc:
            print(f"  [test-log] skip {path}: {exc}", file=sys.stderr)
    return reports


def _session_no_answer(_question: str) -> str:
    """Non-interactive session: the agent is told no answer is available."""
    return ""


def _save_transcript(out: Path, transcript: list[dict], log: AuditLog,
                     trace: SessionTrace, secrets: list[str]) -> None:
    """Persist the session transcript and mirror it into the audit log (redacted)."""
    redactor = Redactor(secrets)
    redacted = []
    for entry in transcript:
        content = entry["content"]
        redacted.append(dict(entry, content=redactor.redact(content)))
        log.append(trace.session_id, "turn", {
            "role": entry.get("role"), "kind": entry.get("kind"),
            "content": redactor.redact(content)[:500],
        })
    (out / "transcript.json").write_text(
        json.dumps(redacted, indent=2), encoding="utf-8")


def _load_parts_graph(parts_csv: str | None):
    if not parts_csv:
        return None
    from ..docs.parts.parts_graph import load_parts_csv
    return load_parts_csv(parts_csv).by_slot()


def _parts_check(parts_graph: dict, dump_sets) -> object:
    fru_text = ""
    for dump in dump_sets.get("ipmi", []):
        if "fru print" in dump.source:
            fru_text = dump.raw
            break
    return PartsValidator().validate_fru(parts_graph, fru_text)


def _parse_parts_line(line: str, rail: str) -> dict | None:
    """``slot, FRU, PN, SN`` (comma-separated) -> instance-part entry."""
    cells = [c.strip() for c in (line or "").split(",")]
    if not cells or not cells[0]:
        return None
    entry = {"slot": cells[0], "rail": rail}
    if len(cells) > 1 and cells[1]:
        entry["fru"] = cells[1]
    if len(cells) > 2 and cells[2]:
        entry["pn"] = cells[2]
    if len(cells) > 3 and cells[3]:
        entry["sn"] = cells[3]
    return entry


def _make_parts_ask(store, target_label: str, progress: Callable[[str], None],
                    repl_input) -> Callable[[str], list[dict]]:
    """Isolation-pass parts-ask callback: prompt the operator, persist answers.

    ``repl_input`` is the REPL's human-input bridge when running inside a
    session (its ``_make_answer_fn``); the bridge renders nothing itself, so the
    question is shown through ``progress``. Otherwise a TTY ``input`` is used.
    Automated runs (no bridge, no TTY) never block: they return [] and leave the
    gap on record. The documented rail->loads topology (from ``topology.py``)
    pre-fills the question's candidate slots so the operator confirms/extends
    instead of enumerating from memory.
    """

    def _candidate_loads(rail: str) -> list[str]:
        from ..docs.parts.topology import loads_for_rail
        names: list[str] = []
        for edge in loads_for_rail(rail, None):
            for load in edge.get("loads") or []:
                name = load.get("name")
                conn = load.get("connection")
                if name and name not in names:
                    names.append(name + (f" ({conn})" if conn else ""))
        return names

    def ask(rail: str) -> list[dict]:
        if repl_input is not None:
            ask_ui = repl_input
        elif sys.stdin.isatty():
            ask_ui = input
        else:
            progress(f"ask-parts: no parts on record for {target_label!r}; "
                     f"skipped (no interactive TTY)")
            return []
        candidates = _candidate_loads(rail)
        progress(f"ask-parts: no parts on record for {target_label!r}; "
                 f"prompting for the {rail!r} rail loads")
        entries: list[dict] = []
        while True:
            question = (
                f"[parts] {rail} fault decoded -- which components are on this rail?"
                + (f"\n  documented loads: {', '.join(candidates)}" if candidates else "")
                + "\n  enter one component as: slot, FRU, PN, SN   "
                "('done' to finish, 'skip' to skip)")
            if repl_input is not None:
                progress(question)
            answer = ask_ui(question)
            text = (answer or "").strip()
            if not text or text.lower() in ("done", "skip", "none"):
                break
            entry = _parse_parts_line(text, rail)
            if entry is None:
                progress("  (unrecognized; expected 'slot, FRU, PN, SN')")
                continue
            entries.append(entry)
            progress(f"  recorded {entry['slot']}")
        if entries:
            store.merge(target_label, entries)
            progress(f"  saved {len(entries)} part(s) for {target_label!r} "
                     f"({store.path_for(target_label)})")
        return entries

    return ask


def _save_dumps(out: Path, dump_sets) -> None:
    dumps_dir = out / "dumps"
    dumps_dir.mkdir(exist_ok=True)
    all_dumps: list[RegisterDump] = []
    for subsystem, dumps in dump_sets.items():
        for i, dump in enumerate(dumps):
            all_dumps.append(dump)
            (dumps_dir / f"{subsystem}_{i}.txt").write_text(
                f"# source: {dump.source}\n# ok: {dump.ok}\n\n{dump.raw}",
                encoding="utf-8",
            )
    (out / "dumps.json").write_text(
        json.dumps([asdict(d) for d in all_dumps], indent=2), encoding="utf-8")


def _save_prompt(out: Path, content: str, session_mode: bool) -> None:
    """Persist the exact prompt(s) sent to the LLM for end-to-end audit.

    Single-shot runs write ``prompt.txt``; session runs append one JSON object
    per turn to ``prompt_turns.jsonl`` (turn number + full message list, so the
    conversation and evidence view are reproducible turn by turn).
    """
    if session_mode:
        with (out / "prompt_turns.jsonl").open("a", encoding="utf-8") as f:
            f.write(content + "\n")
    else:
        # Single-shot runs can emit more than one prompt (auto follow-up round):
        # append, separator-delimited, so the audit keeps every prompt verbatim.
        with (out / "prompt.txt").open("a", encoding="utf-8") as f:
            f.write(content)
            f.write("\n\n==================== END PROMPT ====================\n\n")


def _seat_pending_case(out: Path, diagnosis: Diagnosis, target_label: str,
                       ident: str, symptom: str, session_id: str,
                       test_log_failures: list[str] | None = None) -> None:
    """Seed ``pending_case.json`` (outcome="unknown") in a run dir.

    ``harness report`` fills this base instead of inventing a record from
    artifacts, so actions_taken/outcome are added to (never overwrite) what the
    run itself observed. Deterministic pieces the learning loop needs are
    pre-computed here: evidence_hash over the collected dumps, the decoded
    register lines (~ evidence block), and the cited doc titles. ``--test-log``
    runs additionally record the failure signatures so a future run with the
    same harness failure surfaces this case pre-probe.
    """
    from ..diagnosis.schema import CaseOutcome

    evidence_lines = [
        f"- {d.get('mnemonic', '?')} = {d.get('raw_hex', '?')}"
        for d in diagnosis.evidence if isinstance(d, dict)]
    cited = list(dict.fromkeys(
        r.source for r in diagnosis.references
        if r.source))
    for action in diagnosis.actions:
        for r in action.references:
            if r.source:
                cited.append(r.source)
    cited = [s for s in dict.fromkeys(cited) if _is_doc_source(s)]
    pending = CaseOutcome(
        run_id=session_id,
        target_id=target_label,
        model_key=None,  # filled at report time from audit; unknown here is honest
        model_source=None,
        symptom=symptom,
        subsystem_primary=(
            diagnosis.subsystems_considered[0].value
            if diagnosis.subsystems_considered else None),
        actions_recommended=[f"{a.step}. {a.action}" for a in diagnosis.actions],
        outcome="unknown",
        llm_ident=ident,
        evidence_hash=_evidence_hash(out / "dumps.json"),
        evidence_summary=evidence_lines,
        cited_titles=cited,
        confidence=diagnosis.confidence,
        self_reported_confidence=(
            diagnosis.confidence_breakdown.self_reported_confidence
            if diagnosis.confidence_breakdown is not None else None),
        test_log_failures=list(test_log_failures or []),
    )
    (out / "pending_case.json").write_text(
        pending.model_dump_json(indent=2), encoding="utf-8")


def _is_doc_source(source: str) -> bool:
    from ..diagnosis.scorer import _NON_DOC_SOURCES
    return source.strip().lower() not in _NON_DOC_SOURCES


def _evidence_hash(dumps_path: Path) -> str:
    """sha256 over the canonical JSON of the collected RegisterDumps."""
    import hashlib

    if not dumps_path.exists():
        return ""
    try:
        dumps = json.loads(dumps_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ""
    canonical = json.dumps(dumps, sort_keys=True, indent=2)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _audit_commands(log: AuditLog, trace: SessionTrace, runner) -> None:
    for call in runner.calls:
        trace.record_command(call.argv, call.exit_code, call.elapsed_ms,
                             stdout_sha=call.stdout_sha)
        log.append(trace.session_id, "cmd", {
            "argv": call.argv, "exit": call.exit_code,
            "elapsed_ms": call.elapsed_ms, "stdout_sha": call.stdout_sha,
        })


def _confidence_bar(value: float) -> str:
    """Tiny ASCII confidence meter, e.g. ``[#######---]``."""
    filled = max(0, min(10, round(float(value) * 10)))
    return "[" + "#" * filled + "-" * (10 - filled) + "]"


def _print_diagnosis(diagnosis: Diagnosis, out: Path, session_id: str) -> None:
    """Human verdict block. Styling is value-driven (green healthy / red
    fault, risk-colored action tags); asserted substrings stay contiguous."""
    state_value = diagnosis.state.value
    state_style = (ui.good if state_value == "healthy"
                   else ui.warn if state_value == "degraded" else ui.bad)
    print(ui.heading(f"\n==== Diagnosis [{session_id}] ===="))
    print(state_style(f"state: {state_value}"))
    print(diagnosis.diagnosis)
    bar = ui.dim(_confidence_bar(diagnosis.confidence))
    print(f"{ui.bold(f'confidence: {diagnosis.confidence}')}  {bar}")
    if diagnosis.confidence_breakdown is not None:
        b = diagnosis.confidence_breakdown
        parts = [f"retrieval_citation_support={b.retrieval_citation_support}",
                 f"evidence_fit={b.evidence_fit}",
                 f"model_agreement={b.model_agreement}"]
        if b.root_cause_certainty is not None:
            parts.append(f"root_cause_certainty={b.root_cause_certainty}")
        parts.append(f"penalty={b.penalty}")
        print(ui.dim("  " + " ".join(parts)))
    if diagnosis.failure_point is not None:
        fp = diagnosis.failure_point
        suspects = ", ".join(fp.suspects) if fp.suspects else "(no documented suspect set)"
        probe_note = ("isolation probes ran" if fp.isolation_ran
                      else "no isolation evidence collected")
        print(ui.warn(f"failure point (NOT a root cause yet): rail '{fp.rail_tokens}' "
                      f"suspects: {suspects}; {probe_note}"))
    if diagnosis.unknown_registers:
        print(ui.dim(
            f"unknown registers (manual lookup): {', '.join(diagnosis.unknown_registers)}"))
    if diagnosis.parts_discrepancies:
        print(ui.bold("parts discrepancies:"))
        for d in diagnosis.parts_discrepancies:
            print(f"  - {d}")
    risk_style = {"none": lambda s: s, "low": ui.good,
                  "medium": ui.warn, "high": ui.bad}
    print(ui.bold("repair action list (recommendations only; no raw writes "
                  "are ever made):"))
    for action in diagnosis.actions:
        tag = risk_style.get(action.risk.value, lambda s: s)(action.risk.value)
        print(f"  {action.step}. [{tag}] {action.action} "
              f"(tool: {action.required_tool}, impact: {action.impact})")
        print(ui.dim(f"     {action.rationale}"))
    print(f"\n{ui.dim(f'outputs: {out}')}")


# ---- subcommands ----

def run_lint(args) -> int:
    inv = load_inventory(args.inventory)
    print(f"OK: {len(inv.hosts)} host(s): {', '.join(sorted(inv.host_names))}")
    return 0


def run_docs(args) -> int:
    """Manage the RAG document library: upload (add), list, remove, reindex."""
    from ..docs.ingest.library import DocLibrary
    lib = DocLibrary(args.lib)
    action = args.docs_action
    if action == "add":
        platform = getattr(args, "platform", None)
        for line in lib.add(args.files, platform=platform):
            print(line)
        if platform:
            print(f"  tagged {len(args.files)} document(s) as platform={platform}")
    elif action == "reindex":
        for line in lib.reindex():
            print(line)
    elif action == "retag":
        for line in lib.retag(args.names, platform=getattr(args, "platform", None)):
            print(line)
    elif action == "rm":
        try:
            print(lib.remove(args.name))
        except KeyError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    elif action == "ls":
        entries = lib.entries()
        if not entries:
            print(f"(empty library at {args.lib!r})")
            return 0
        for e in entries:
            platform = f"  platform={e.platform}" if e.platform else ""
            suffix = f"  ERROR: {e.error}" if e.error else ""
            print(f"{e.name}  sha256={e.sha256[:12]}  {e.chunks} chunk(s)  "
                  f"{e.ingested_at}{platform}{suffix}")
    return 0


def _progress_printer() -> Callable[[str], None]:
    """Live pipeline trace writer: goes to stderr so the diagnosis verdict
    remains the only stdout output."""

    def _emit(text: str) -> None:
        print(f"  {text}", file=sys.stderr, flush=True)

    return _emit


def _probe_line(res: CommandResult) -> str:
    """One-line summary of an executed probe: command, outcome, elapsed."""
    cmd = " ".join(res.argv)
    outcome = "ok" if res.ok else f"exit {res.exit_code}"
    elapsed = f"{res.elapsed_ms / 1000:.1f}s" if res.elapsed_ms else ""
    return f"{cmd} -> {outcome} {elapsed}"


def _audit_model(log: AuditLog, trace: SessionTrace, model, drifted: bool,
                 progress: Callable[[str], None]) -> None:
    """Audit the detected model and surface alias drift to the operator."""
    if model is None:
        log.append(trace.session_id, "model_detected", {"model": None,
                                                        "drifted_from_hint": False})
        return
    log.append(trace.session_id, "model_detected", {
        "model": model.model_key, "product_name": model.product_name,
        "bios_vendor": model.bios_vendor, "bios_version": model.bios_version,
        "source": model.source, "drifted_from_hint": drifted,
    })
    if drifted:
        progress(f"model={model.model_key} (alias drift: detected value wins)")


def _build_case_retriever(out_dir: str | Path):
    """Verified-fleet-history hook: symptom, model_key -> prior-case prompt lines.

    Returns None when no case records exist yet (a fresh fleet learns nothing
    before its first outcome is recorded -- by design).
    """
    try:
        from ..diagnosis.case_library import CaseLibrary, render
        from ..diagnosis.case_store import CaseStore
    except ImportError:
        return None
    store = CaseStore(Path(out_dir) / "cases")
    if not store.all():
        return None
    lib = CaseLibrary(store)
    return lambda symptom, model_key: render(
        lib.similar(symptom, model_key, top_k=5))


def run_diagnose(args, overrides: dict | None = None) -> Diagnosis:
    """Full read-only diagnosis; returns the scored, audited Diagnosis.

    Single-shot mode (default) or multi-turn session mode (--interactive /
    --context / --context-file): in session mode the agent may ask questions and
    request further read-only probes across several turns before diagnosing.
    """
    overrides = overrides or {}
    inv = load_inventory(args.inventory)
    store = overrides.get("store") or _make_store(args)
    # LLM endpoint setup (tunnel / preflight) before anything expensive: a bad
    # --llm-url / --llm-tunnel must fail in milliseconds, not after collection.
    llm_forward = None
    if overrides.get("llm_forward") is None:
        llm_forward = _prepare_llm_endpoint(args, inv, store)
    target = _resolve_target_from_args(args, inv, store)
    apply_ssh_context(store, target, ssh_user=getattr(args, "ssh_user", "diagbot"))
    host: Host = target.host

    trace = SessionTrace()
    out = Path(args.out_dir) / trace.session_id
    out.mkdir(parents=True, exist_ok=True)

    # Operator-supplied test logs: loaded BEFORE the retriever/audit so the
    # derived symptom, run_start event, doc retrieval, and case seeding all see
    # the same failure identity.
    test_log_reports = _load_test_logs(args)
    symptom = args.symptom or ""
    if not symptom:
        first = test_log_reports[0] if test_log_reports else None
        if first is not None and first.failures:
            symptom = (f"Factory test log failure: "
                       f"{first.failures[0].signature}")
        elif first is not None and first.raw_excerpt:
            symptom = "Factory test log failure (see test-log evidence)"
    if not symptom:
        print("  error: a --symptom or --test-log is required to diagnose",
              file=sys.stderr)
        raise SystemExit(2)

    bmc_password: str | None = None
    secrets: list[str] = []
    try:
        bmc_password = store.get(host.bmc.password_vault_path).decode()
        secrets.append(bmc_password)
    except KeyError:
        bmc_password = None  # BMC channel unavailable; ipmi collector skipped

    max_turns = int(getattr(args, "max_turns", 6))
    human_input = overrides.get("human_input")
    session_mode = bool(getattr(args, "interactive", False) or
                        getattr(args, "context", None) or
                        getattr(args, "context_file", None) or
                        human_input is not None)
    if human_input is None:
        human_input = input if getattr(args, "interactive", False) else _session_no_answer

    retriever = overrides.get("retriever")
    docs_lib_used = None
    if retriever is None:
        retriever, docs_lib_used = _build_docs_retriever(args)

    # Optional operator fallback when model detection fails: only in
    # interactive runs; one-shot/automated paths never block on the question.
    model_ask = None
    if getattr(args, "interactive", False):
        def _ask_model() -> str | None:
            try:
                answer = input(
                    "Model not detected. Enter product name "
                    "(Enter to skip): ").strip()
            except (EOFError, KeyboardInterrupt):
                return None
            return answer or None
        model_ask = _ask_model

    log = AuditLog(out / "audit.jsonl", Redactor(secrets))
    log.append(trace.session_id, "run_start", {
        "host": target.label, "symptom": symptom, "trust_level": target.trust_level,
        "model": host.model, "collector_profile": host.collector_profile,
        "mode": "session" if session_mode else "single", "max_turns": max_turns,
        "target": target.kind,
        **({"rack": target.console.rack, "cable": target.console.cable}
           if target.console is not None else {}),
        **({"ip": target.ip} if target.ip is not None else {}),
        **({"docs_lib": docs_lib_used} if docs_lib_used else {}),
    })
    if test_log_reports:
        log.append(trace.session_id, "test_log_loaded", {
            "count": len(test_log_reports),
            "logs": [
                {"source": r.source, "model": r.model, "serial": r.serial,
                 "station": r.station, "stage": r.test_stage,
                 "failures": [f.signature for f in r.failures],
                 "parsed": bool(r.failures) or bool(r.raw_excerpt)}
                for r in test_log_reports
            ],
        })

    use_console = bool(getattr(args, "console", False)) or target.kind == "console"
    if use_console:
        if host.console is None:
            raise RuntimeError(
                f"target {target.label!r} has no console block (use --rack/--cable "
                "with a fleet console_defaults block, or a named host with a "
                "console block)")
        domain = _console_overrides(host.console, args)
        _console_sudo_secret(store, domain, secrets)
        console_runner = overrides.get("console_runner")
        if console_runner is None:
            console_runner = ConsoleRunner(SerialConsole(domain, store))
        runner: Runner = console_runner
    else:
        session = overrides.get("session")
        if session is None:
            session = SSHSession(host, default_policy(), store)
            session.open()
        runner = session

    # Live pipeline trace: step events (plan/collect/decode/reason) plus one
    # line per executed probe (command, outcome, elapsed). Defaults to stderr.
    if "progress" in overrides:
        progress: Callable[[str], None] = overrides["progress"]
    else:
        progress = _progress_printer()
    if hasattr(runner, "on_probe"):
        runner.on_probe = lambda res: progress(f"probe {_probe_line(res)}")

    # Rack-level Redfish evidence (parallel to the console probes): enabled by
    # the console block's redfish vault path, independent of the serial path.
    redfish_collector = overrides.get("redfish_collector")
    if redfish_collector is None and use_console:
        redfish_collector = _build_redfish_collector(domain, store, secrets,
                                                     progress)

    # Preserve the raw test logs in the run dir so every diagnosis is
    # reproducible from its own harness_runs/<id>/ directory.
    if test_log_reports:
        tl_out = out / "test_logs"
        tl_out.mkdir(parents=True, exist_ok=True)
        for report in test_log_reports:
            src = Path(report.source)
            try:
                if src.exists():
                    import shutil
                    shutil.copy2(src, tl_out / src.name)
            except OSError as exc:
                progress(f"test-log {report.source!r} not copied: {exc}")
        test_log_lines: list[str] | None = [
            line for r in test_log_reports for line in r.summary_lines()]
        test_log_queries: list[str] | None = [
            q for r in test_log_reports for q in r.rag_queries()]
        test_log_case_terms: list[str] | None = [
            t for r in test_log_reports for t in r.case_terms()]
    else:
        test_log_lines = test_log_queries = test_log_case_terms = None

    def _open_interactive(client) -> InteractiveShell:
        shell = InteractiveShell(client)
        shell.open()
        return shell

    single_test_driver = overrides.get("single_test_driver")
    owns_driver = False
    server_number = getattr(args, "server_number", None)
    if single_test_driver is None and server_number and session_mode \
            and not use_console and host.ssh is not None:
        hint = (target.model_hint or host.model or "").strip()
        fam = family_for(hint) if hint else None
        if fam not in (None, "samoa", "nvl72"):
            progress(f"single tests skipped: hint {hint!r} is not a GB platform")
        else:
            client = getattr(runner, "client", None)
            if client is not None:
                try:
                    single_test_driver = SingleTestDriver(
                        server_number,
                        shell_factory=lambda c=client: _open_interactive(c),
                        progress=progress,
                        artifact_dir=out,
                    )
                    owns_driver = True
                    log.append(trace.session_id, "single_test_enabled", {
                        "server_number": server_number,
                        "target": target.label,
                        "platform_hint": hint or None,
                    })
                except SingleTestError as exc:
                    progress(f"single tests unavailable: {exc}")

    bmc_runner = overrides.get("bmc_runner")
    if bmc_runner is None and bmc_password is not None:
        bmc_runner = BmcRunner(host.bmc.address, host.bmc.username, bmc_password)

    supervisor = RunSupervisor(max_steps=8 + max_turns,
                               wall_s=float(getattr(args, "wall_s", 900.0)))
    cancel_event = overrides.get("cancel_event")

    def _step(label: str) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise Escalation("run cancelled by operator")
        supervisor.check(label)

    parts_graph = _load_parts_graph(getattr(args, "parts_csv", None))
    # Operator-answered instance parts (prompted under --ask-parts) persist per
    # target and are reused by future runs; the explicit CSV wins on a slot.
    from ..docs.parts.instance_store import InstancePartsStore, merge_store_into_parts
    parts_store = InstancePartsStore(getattr(args, "parts_dir", "config/parts"))
    stored_parts = parts_store.load(target.label)
    parts_graph = merge_store_into_parts(parts_graph, stored_parts)
    parts_ask = (_make_parts_ask(parts_store, target.label, progress,
                                 overrides.get("human_input"))
                 if getattr(args, "ask_parts", False) else None)

    # Documented power-topology edges (rail -> loads) for the fault-isolation
    # round: the decoded rail fault is mapped back to the loads it feeds so the
    # prompt enumerates the whole suspect set. Keyed to the detected model when
    # known, else every platform is returned with its own label.
    from ..docs.parts.topology import loads_for_rail
    topology_hook = lambda sig, model_key: loads_for_rail(sig, model_key)

    def collector_factory(name, _runner):
        if name == "redfish":
            return redfish_collector
        if _runner.is_console:
            # BMC BusyBox shell: no rdmsr/smartctl/lspci/dmidecode. Map the
            # host-OS subsystems to BMC-shell probes; skip host-only ones.
            if name in ("pcie", "storage"):
                return None
            if name in ("cpu_msr", "kernel", "ipmi"):
                return BmcConsoleCollector(_runner, subsystem={
                    "cpu_msr": "cpu", "kernel": "kernel", "ipmi": "ipmi",
                }[name])
        if name == "ipmi":
            return IpmiCollector(bmc_runner) if bmc_runner is not None else None
        return make_collector(name, _runner)

    llm = overrides.get("llm") or _resolve_llm(args, inv, store)
    calibration_root = Path(args.out_dir) / "calibration"
    ident = _llm_ident_for(args, inv)

    # Snippets the engine actually put in the prompt (first-pass + isolation
    # round). The scorer reads these so LLM citations to isolation snippets are
    # counted as supported instead of re-retrieving the symptom only.
    used_snippets: dict[str, list[str]] = {"lines": []}

    ctx = EngineContext(
        runner=runner,
        decoder=Decoder(),
        collector_factory=collector_factory,
        llm=llm,
        docs_retriever=retriever,
        snippets_callback=lambda lines: used_snippets.update(lines=lines),
        parts_ask=parts_ask,
        topology=topology_hook,
        parts_refs=(
            lambda: [f"{k}: {v}" for k, v in parts_graph.items()]
        ) if parts_graph else None,
        parts_validate=(
            lambda dumps: _parts_check(parts_graph, dumps)
        ) if parts_graph else None,
        scorer=(
            lambda d, dump_sets: score_diagnosis(
                d,
                retrieved_snippets=(
                    used_snippets["lines"]
                    or (retriever(symptom, None) if retriever else None)
                ),
                evidence_fit=(
                    None if d.evidence else evidence_fit_from_dumps(d, dump_sets)
                ),
                calibration_root=calibration_root,
                llm_ident=ident,
            )
        ),
        supervisor=_step,
        dump_callback=lambda dumps: _save_dumps(out, dumps),
        prompt_callback=lambda content: _save_prompt(out, content, session_mode),
        progress=progress,
        model_hook=(
            lambda model, drifted: _audit_model(log, trace, model, drifted,
                                                progress)
        ),
        model_hint=(
            target.model_hint
            or (host.model if host.model not in ("", "unknown") else None)
        ),
        model_ask=model_ask,
        llm_ident=lambda: ident,
        calibration_root=str(calibration_root),
        priors=_load_priors(args.out_dir),
        single_test_driver=single_test_driver,
        test_log_lines=test_log_lines,
        test_log_queries=test_log_queries,
        test_log_case_terms=test_log_case_terms,
        extra_collectors=("redfish",) if redfish_collector is not None else (),
    )
    if session_mode:
        engine = SessionEngine(
            ctx,
            llm=llm,
            human_input=human_input,
            max_turns=max_turns,
        )
        try:
            diagnosis = engine.run(symptom, initial_answers=_load_context(args))
        finally:
            if owns_driver and single_test_driver is not None:
                single_test_driver.close()
        _save_transcript(out, engine.transcript, log, trace, secrets)
    else:
        diagnosis = DiagnosticEngine(ctx).run(symptom)

    _audit_commands(log, trace, runner)
    if bmc_runner is not None:
        _audit_commands(log, trace, bmc_runner)
    trace.link_raw_log(str(out / "audit.jsonl"))
    (out / "trace.json").write_text(json.dumps(asdict(trace), indent=2), encoding="utf-8")

    log.append(trace.session_id, "diagnosis", diagnosis.model_dump(mode="json"))
    (out / "diagnosis.json").write_text(diagnosis.model_dump_json(indent=2), encoding="utf-8")

    gate = ApprovalGate()
    for action in diagnosis.actions:
        if args.approval:
            decision = gate.prompt(action, trace.session_id)
        elif args.approve_all:
            decision = ApprovalDecision(action=action, approved=True)
        else:
            decision = ApprovalDecision(action=action, approved=False, note="not prompted")
        gate.record(decision, trace.session_id, log)

    _seat_pending_case(out, diagnosis, target.label, ident, symptom,
                       trace.session_id,
                       test_log_failures=(test_log_case_terms or []))
    _write_run_meta(out, target, host, trace.session_id)
    print(f"\npending case: {out / 'pending_case.json'}")
    print("close the learning loop after the repair with:")
    print(f"  harness label --run {trace.session_id}")
    print(f"  (or: harness report --run {trace.session_id} --outcome fixed "
          f"[--taken \"...\"])")

    _print_diagnosis(diagnosis, out, trace.session_id)
    if _should_prompt_label(args):
        _prompt_label_after_run(out, out.parent / "cases")
    if llm_forward is not None:
        llm_forward.close()          # the hop lives exactly as long as the run
    if redfish_collector is not None:
        redfish_collector.client.close()   # tunnel (if any) dies with the run
    return diagnosis


def run_console(args) -> int:
    inv = load_inventory(args.inventory)
    store = _make_store(args)
    target = _resolve_target_from_args(args, inv, store)
    apply_ssh_context(store, target, ssh_user=getattr(args, "ssh_user", "diagbot"))
    host: Host = target.host
    if host.console is None:
        print(f"target {target.label!r} has no console path "
              f"(use --rack/--cable with console_defaults, or a named host "
              f"with a console block)", file=sys.stderr)
        return 2
    domain = _console_overrides(host.console, args)

    secrets: list[str] = []
    sudo_password: str | None = None
    if domain.sudo_vault_path is not None:
        try:
            sudo_password = store.get(domain.sudo_vault_path).decode().rstrip("\r\n")
            secrets.append(sudo_password)
        except KeyError:
            print(f"sudo password missing from vault: {domain.sudo_vault_path!r}",
                  file=sys.stderr)
            return 2

    out = Path(args.out_dir) / SessionTrace().session_id if args.out_dir else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
    trace = SessionTrace()
    log = AuditLog(out / "audit.jsonl", Redactor(secrets)) if out is not None else None
    if log is not None:
        log.append(trace.session_id, "console_start", {
            "host": target.label, "target": target.kind, "rack": domain.rack,
            "cable": domain.cable, "port": domain.port,
            "trust_level": domain.trust_level, "probes": list(args.probe),
        })

    console = SerialConsole(domain, store)
    try:
        wire = [_absolutize_bmc_i2c_tools(c) for c in args.probe]
        result = console.run_probes(wire)
        redactor = Redactor(secrets)
        rendered = redactor.redact(result.output) if secrets else result.output
        for line in rendered.splitlines():
            print(line)
        if log is not None:
            for cmd, wire_cmd in zip(args.probe, wire):
                if wire_cmd != cmd:
                    validate_serial_probe(wire_cmd)  # wire invariant: gate-checked
                log.append(trace.session_id, "cmd", {
                    "argv": shlex.split(wire_cmd), "exit": 0,
                    "probe_count": result.probe_count,
                })
            trace.record_command(["console", *args.probe], 0, result.elapsed_ms,
                                 stdout_sha=hashlib.sha256(rendered.encode()).hexdigest())
            trace.link_raw_log(str(out / "audit.jsonl"))
            (out / "trace.json").write_text(
                json.dumps(asdict(trace), indent=2), encoding="utf-8")
            (out / "console.txt").write_text(rendered, encoding="utf-8")
    finally:
        console.close()
    return 0


def run_verify(args, overrides: dict | None = None) -> int:
    overrides = overrides or {}
    inv = load_inventory(args.inventory)
    store = overrides.get("store") or _make_store(args)
    target = _resolve_target_from_args(args, inv, store)
    apply_ssh_context(store, target, ssh_user=getattr(args, "ssh_user", "diagbot"))
    host: Host = target.host
    baseline = [RegisterDump(**d) for d in json.loads(
        Path(args.baseline).read_text(encoding="utf-8"))]

    session = overrides.get("session")
    if session is None:
        session = SSHSession(host, default_policy(), store)
        session.open()
    try:
        plan = plan_collection(args.symptom)
        after: list[RegisterDump] = []
        for name in plan.collectors:
            if name == "ipmi":  # BMC channel not opened here; covered by diagnose
                continue
            after.extend(make_collector(name, session).collect())
    finally:
        if overrides.get("session") is None:
            session.close()

    result = Verifier().compare(baseline, after, metric_key=args.metric)
    print(f"verdict: {result.verdict}")
    print(f"before:  {result.before}")
    print(f"after:   {result.after}")
    print(f"delta:   {result.delta}")
    return 0 if result.verdict in ("resolved", "no_change") else 1


def run_session(args, overrides: dict | None = None) -> int:
    """Interactive chat session: natural language in, read-only debugging out.

    See ``operator.repl`` for the REPL itself; this indirection keeps the
    lazy import so ``repl`` may import helpers from this module freely.
    """
    from .repl import run_session as _run_session
    return _run_session(args, overrides or {})


def run_targets(args) -> int:
    """Manage short target aliases (path-only file, passes lint semantics)."""
    path = Path(args.targets_file)
    aliases = load_targets(path)
    if args.target_action == "add":
        try:
            entry = TargetAlias(alias=args.alias, rack=args.rack,
                                cable=args.cable, address=args.address,
                                model=getattr(args, "model", None))
        except AliasError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        aliases[entry.alias] = entry
        save_targets(path, aliases)
        where = entry.address or f"{entry.rack}-cable{entry.cable}"
        print(f"added target {entry.alias!r} -> {where} ({path})")
        if entry.model:
            print(f"  model: {entry.model} (fallback when detection fails)")
        return 0
    if args.target_action == "ls":
        if not aliases:
            print(f"(no targets registered; file: {path})")
            return 0
        for alias in sorted(aliases):
            e = aliases[alias]
            where = e.address or (f"rack {e.rack}, cable {e.cable}" if e.rack else "-")
            extra = f"  model={e.model}" if e.model else ""
            print(f"{alias}  {where}{extra}")
        return 0
    if args.target_action == "rm":
        if args.alias not in aliases:
            print(f"error: unknown target alias {args.alias!r}", file=sys.stderr)
            return 1
        del aliases[args.alias]
        save_targets(path, aliases)
        print(f"removed target {args.alias!r}")
        return 0
    return 2


def _secrets_entry(args) -> int:
    from .secrets_cli import run_secrets
    return run_secrets(args)


def run_setup_cmd(args) -> int:
    """Entry for ``harness setup`` (additive, non-agent credential flow)."""
    from .setup_cli import SetupError, run_setup
    try:
        return run_setup(args, overrides={})
    except SetupError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def run_calibrate(args) -> int:
    """Rebuild per-LLM-ident fix-rate calibration from verified case outcomes.

    ``harness calibrate --cases <dir> [--llm <ident>] [--out <dir>]`` -- loads
    the case store, builds a histogram per ident (or the given one) and saves it
    to the calibration store (default <out-dir>/calibration). Each model
    calibrates from ITS OWN recorded outcomes, so a model swap never inherits
    another model's calibration.
    """
    from ..diagnosis.calibration import CalibrationStore, build_calibration
    from ..diagnosis.case_store import CaseStore

    store = CaseStore(args.cases)
    cases = store.all()
    if not cases:
        print("no case records found; run 'harness report' after diagnoses",
              file=sys.stderr)
        return 1
    idents = [args.llm] if args.llm else sorted({c.llm_ident for c in cases})
    out = CalibrationStore(args.out)
    built = 0
    for ident in idents:
        cal = build_calibration(cases, ident)
        if cal is None:
            print(f"{ident}: insufficient verified cases for {ident!r} "
                  f"(need >=3) -- calibration stays inactive (0.5 fallback)")
            continue
        out.save(cal)
        built += 1
        print(f"{ident}: {cal.total_samples()} verified case(s), "
              f"created {cal.created_at}")
        _print_calibration_tables(cal)
    if built == 0:
        print("no calibration written (insufficient verified outcomes)",
              file=sys.stderr)
        return 1
    print(f"calibration store: {out.root}")
    return 0


def _print_calibration_tables(cal) -> None:
    """Per-ident bin tables: (subsystem, bin_max, observed_rate, n)."""
    header = f"  {'subsystem':<10} {'bin_max':<8} {'rate':<6} n"
    print(header)
    rows = [(None, *entry) for entry in cal.aggregate]
    for subsystem, entries in sorted(cal.subsystem_bins.items()):
        rows += [(subsystem, *entry) for entry in entries]
    for subsystem, bin_max, rate, n in rows:
        name = subsystem if subsystem else "(aggregate)"
        print(f"  {name:<10} {bin_max:<8.2f} {rate:<6.3f} {n}")
        if subsystem and n == 0:
            print("  insufficient data - calibration falls back to aggregate",
                  file=sys.stderr)


def run_report(args) -> int:
    """Turn a diagnosis run into a verified case record (prompt 03 contract).

    ``harness report --run <run_id> --outcome {fixed,partial,not_fixed,
    inconclusive} [--taken ...] [--cases <dir>]`` fills the run's
    ``pending_case.json`` (seeded at the end of every diagnosis) with the
    operator-reported outcome and the actions that were actually taken, then
    persists it via the ``CaseStore``. A second report for the same run is
    rejected unless ``--revise`` is passed (operator correction: the record
    is replaced and the change is audited as ``case_revised``).
    ``--status`` prints any existing record for the run.
    For the interactive version (prefilled fix, runs menu integration) use
    ``harness label`` instead.
    """
    from ..diagnosis.case_store import CaseStore
    from ..diagnosis.schema import CaseOutcome

    runs_root = Path(args.out_dir)
    run_dir = runs_root / args.run
    pending = run_dir / "pending_case.json"
    store = CaseStore(args.cases)

    if args.status:
        record = store.get(args.run)
        if record is None:
            print(f"no case record for run {args.run!r}", file=sys.stderr)
            return 1
        print(record.model_dump_json(indent=2))
        return 0

    if not pending.exists():
        print(f"no pending_case.json in {run_dir} (was this run a diagnosis?)",
              file=sys.stderr)
        return 1
    base = CaseOutcome.model_validate_json(pending.read_text(encoding="utf-8"))

    model_key, model_source = _model_from_audit(run_dir / "audit.jsonl")
    record = CaseOutcome(
        **{**base.model_dump(),
           "model_key": model_key or base.model_key,
           "model_source": model_source or base.model_source,
           "outcome": args.outcome,
           "actions_taken": list(args.taken),
           "verification_verdict": args.verdict,
           "created_at": "",  # CaseStore stamps it
        })
    try:
        path = store.record(record, audit=AuditLog(run_dir / "audit.jsonl"),
                            revise=bool(getattr(args, "revise", False)))
    except (FileExistsError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"recorded case {record.run_id}: outcome={record.outcome} "
          f"llm={record.llm_ident} evidence_hash={record.evidence_hash[:12]}")
    print(f"  {path}")
    return 0


def run_label(args) -> int:
    """Label a run with the verified outcome + the correct fix (learning-loop
    ground truth).

    ``harness label --run <run_id|path> [--outcome ...] [--taken ...]
    [--revise]`` -- with no outcome/fix flags it prompts interactively,
    prefilling the fix with the run's top recommendation (Enter accepts it,
    type over it when the recommendation was wrong). Old runs without
    ``pending_case.json`` are synthesized from diagnosis.json + audit.
    ``--status`` prints any existing record for the run.
    """
    from ..diagnosis.case_store import CaseStore

    run_arg = Path(args.run)
    run_dir = run_arg if run_arg.is_dir() else Path(args.out_dir) / args.run
    if not run_dir.is_dir():
        print(f"no such run directory: {run_dir}", file=sys.stderr)
        return 1
    if args.status:
        record = CaseStore(args.cases).get(run_dir.name)
        if record is None:
            print(f"no case record for run {run_dir.name!r}", file=sys.stderr)
            return 1
        print(record.model_dump_json(indent=2))
        return 0
    return _label_run(run_dir, Path(args.cases), outcome=args.outcome,
                      taken=list(args.taken) or None, verdict=args.verdict,
                      revise=args.revise)


def _model_from_audit(audit_path: Path) -> tuple[str | None, str | None]:
    """Model facts from the run's audit log (model_detected event), if any."""
    if not audit_path.exists():
        return None, None
    try:
        for line in reversed(audit_path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("kind") == "model_detected":
                payload = event["payload"] or {}
                return payload.get("model"), payload.get("source")
    except (json.JSONDecodeError, OSError):
        return None, None
    return None, None


def _load_priors(out_dir: str | Path) -> object | None:
    """Load ``priors.json`` from the run out-dir when present.

    Failures to load are warnings, never errors: priors are an optimization,
    and a corrupt/missing file must not block a diagnosis.
    """
    path = Path(out_dir) / "priors.json"
    if not path.exists():
        return None
    try:
        from ..plan.subsystem import PriorModel
        data = json.loads(path.read_text(encoding="utf-8"))
        priors = PriorModel(keyword_multipliers=data.get("keyword_multipliers"))
    except Exception as exc:  # noqa: BLE001 - corrupt priors degrade to static table
        print(f"  [priors] {path}: {exc}; using the static heuristic table",
              file=sys.stderr)
        return None
    if priors.empty:
        return None
    return priors


def run_priors_update(args) -> int:
    """Build outcome-fed subsystem priors from VERIFIED case outcomes.

    ``harness priors update --cases <dir> [--out priors.json]`` -- reads the
    case store, Laplace-smooths per-keyword subsystem multipliers over
    verified outcomes only, and writes the prior file. Prints the
    ``min_verified`` gate state so callers see when priors are inactive.
    """
    from ..diagnosis.case_store import CaseStore
    from ..plan.priors_build import build_priors

    store = CaseStore(args.cases)
    cases = store.all()
    gate = len({c.run_id for c in cases
                if c.outcome in ("fixed", "partial", "not_fixed",
                                 "inconclusive")})
    priors = build_priors(cases, min_verified=args.min_verified)
    if priors is None:
        print(f"{gate}/{args.min_verified} outcome-recorded cases, "
              "priors inactive (static heuristic table in use)")
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "min_verified": args.min_verified,
        "verified_cases": gate,
        "keyword_multipliers": priors.keyword_multipliers,
    }, indent=2, sort_keys=True), encoding="utf-8")
    print(f"{gate}/{args.min_verified} outcome-recorded cases, "
          f"priors active for {len(priors.keyword_multipliers)} keyword(s): {out}")
    return 0


def _eval_llm(args) -> tuple:
    """LLM backend for eval replay: ``--llm {stub,openai,local,gemini}``.

    Eval needs no inventory; the adapter is constructed directly so a
    misconfigured pipeline reports a hard error instead of a silent pass.
    ``--llm-url`` (or ``HARNESS_LLM_URL``) pins the endpoint; the ident keys
    the replay into per-model calibration (e.g. ``local/Qwen2.5-7B-Instruct``).
    """
    provider = getattr(args, "llm", None) or "stub"
    url = getattr(args, "llm_url", None)
    if provider == "stub":
        return StubLLM(), "stub"
    model = os.environ.get("HARNESS_LLM_MODEL")
    if provider == "gemini":
        model = model or "gemini-2.5-flash"
        return GeminiLLM(model=model), f"gemini/{model}"
    if provider == "local":
        model = model or "harness-diag"
        return LocalLLM(model=model, url=url), f"local/{model}"
    model = model or "harness-diag"
    return OpenAICompatLLM(model=model, url=url), f"openai/{model}"


def run_eval(args) -> int:
    """Offline regression: replay held-out verified cases through the CURRENT
    pipeline (retrieval -> prompt -> LLM -> scorer) and diff against the
    stored baseline.

    ``harness eval --cases <dir> [--lib <dir>] [--llm {stub,openai,gemini}]
    [--holdout-frac 0.25] [--out eval_report.json] [--update-baseline]
    [--tolerance 0.05]``

    The holdout is deterministic (sha256 of run_id) and excluded from case
    retrieval, priors, and calibration during the eval. Exits 1 when verdict
    accuracy or ECE regresses beyond ``--tolerance``; ``--update-baseline``
    rewrites the baseline deliberately.
    """
    from ..diagnosis.case_store import CaseStore
    from ..diagnosis.eval import evaluate
    from ..docs.ingest.library import DocLibrary

    store = CaseStore(args.cases)
    cases = store.all()
    verified = [c for c in cases if c.verified]
    if not verified:
        print("no verified cases to evaluate; run 'harness report' after "
              "diagnoses first", file=sys.stderr)
        return 1

    lib_dir = getattr(args, "lib", None)
    if lib_dir is None and Path("harness_docs").is_dir():
        lib_dir = "harness_docs"
    if lib_dir is None:
        print("eval: no doc library -- construct one with 'harness docs add' "
              "or pass --lib <dir> (hard error, not a silent pass)",
              file=sys.stderr)
        return 1
    try:
        rag = DocLibrary(lib_dir).build_retriever()
    except Exception as exc:  # noqa: BLE001 - pipeline construction failures are hard errors
        print(f"eval: library error: {exc}", file=sys.stderr)
        return 1
    if rag is None:
        print(f"eval: library {lib_dir!r} is empty; cannot construct the "
              "retrieval pipeline (hard error, not a silent pass)",
              file=sys.stderr)
        return 1

    llm, ident = _eval_llm(args)
    try:
        report = evaluate(store, llm, rag, ident,
                          frac=args.holdout_frac)
    except Exception as exc:  # noqa: BLE001 - any replay failure is a hard error
        print(f"eval: replay failed: {exc}", file=sys.stderr)
        return 1
    if not report["cases"]:
        print("eval: holdout split produced no replayed cases "
              f"(--holdout-frac {args.holdout_frac})", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    baseline = out.with_name("baseline.json")
    _print_eval_report(report)

    summary = {"verdict_accuracy": report["verdict_accuracy"],
               "ece": report["ece"],
               "n_replayed": report["n_replayed"]}
    if args.update_baseline or not baseline.exists():
        baseline.write_text(json.dumps(summary, indent=2, sort_keys=True),
                            encoding="utf-8")
        print(f"baseline written: {baseline}")
        return 0

    try:
        saved = json.loads(baseline.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        print(f"eval: baseline {baseline} unreadable; run "
              "--update-baseline to rewrite", file=sys.stderr)
        return 1
    tol = args.tolerance
    regressions = []
    if report["verdict_accuracy"] < saved["verdict_accuracy"] - tol:
        regressions.append(
            f"verdict accuracy {report['verdict_accuracy']:.3f} regressed "
            f"from {saved['verdict_accuracy']:.3f} (tolerance {tol})")
    if report["ece"] > saved["ece"] + tol:
        regressions.append(
            f"ECE {report['ece']:.3f} regressed from {saved['ece']:.3f} "
            f"(tolerance {tol})")
    if regressions:
        for line in regressions:
            print(f"  regression: {line}")
        print("eval FAILED -- rerun with --update-baseline to accept the new "
              "numbers", file=sys.stderr)
        return 1
    print("no regressions vs baseline")
    return 0


def _print_eval_report(report: dict) -> None:
    """Human table from the eval report dict (per-subsystem + overall)."""
    print(f"eval: {report['n_replayed']} holdout case(s) replayed with "
          f"{report['llm_ident']}")
    header = f"  {'subsystem':<10} {'acc':<6} {'ece':<6} {'cite':<6} {'recall':<7} n"
    print(header)
    for subsystem, metrics in sorted(report["per_subsystem"].items()):
        print(f"  {subsystem:<10} {metrics['verdict_accuracy']:<6.3f} "
              f"{metrics['ece']:<6.3f} {metrics['mean_citation_support']:<6.3f} "
              f"{metrics['mean_retrieval_recall']:<7.3f} {metrics['n']}")
    print(f"  {'(overall)':<10} {report['verdict_accuracy']:<6.3f} "
          f"{report['ece']:<6.3f} {report['mean_citation_support']:<6.3f} "
          f"{report['mean_retrieval_recall']:<7.3f} {report['n_replayed']}")


# ---- interactive menu (bare `harness` / `harness menu`) ----

_IP4_RE = re.compile(
    r"^(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)$")

_BACK = "back"

# Top-level menu: the two daily flows plus runs, everything else one hop down.
# Action KEYS are stable (tests and argv building depend on them); only the
# display labels are user-facing copy.
_MAIN_ACTIONS: list[tuple[str, str]] = [
    ("chat", "Chat session        describe symptoms in plain English; the agent debugs"),
    ("diagnose", "Debug a target       one-shot read-only diagnosis of one machine"),
    ("runs", "Inspect past runs    verdicts, commands, evidence from earlier diagnoses"),
    ("advanced", "Advanced             verify - console - model - docs - targets - secrets - setup - lint"),
    ("quit", "Quit"),
]

_ADVANCED_ACTIONS: list[tuple[str, str]] = [
    ("verify", "verify         compare a run against a baseline (post-repair check)"),
    ("console", "console        read-only probes over the serial console (lab/qa)"),
    ("model", "model          pick the LLM model for reasoning (remembered)"),
    ("docs", "docs           manage the RAG document library"),
    ("targets", "targets        manage short target aliases"),
    ("secrets", "secrets        register credentials (non-agent, never in prompts)"),
    ("setup", "setup          one-time wizard: API key, SSH key, inventory (first machine setup)"),
    ("lint", "lint           validate this inventory"),
    (_BACK, "Back to main menu"),
]

_WIZARD_FLAGS: dict[str, str] = {
    "--secret-dir": "secret_dir",
    "--docs-lib": "docs_lib",
    "--docs-dir": "docs_dir",
    "--parts-csv": "parts_csv",
    "--parts-dir": "parts_dir",
    "--out-dir": "out_dir",
    "--session-dir": "session_dir",
    "--targets-file": "targets_file",
    "--test-log": "test_log",
}

_ALLOWED_FLAGS: dict[str, tuple[str, ...]] = {
    "diagnose": ("--secret-dir", "--docs-lib", "--docs-dir", "--parts-csv",
                 "--parts-dir", "--out-dir", "--targets-file", "--test-log"),
    "session": ("--secret-dir", "--docs-lib", "--docs-dir", "--parts-csv",
                "--parts-dir", "--out-dir", "--session-dir", "--targets-file",
                "--test-log"),
    "console": ("--secret-dir", "--out-dir", "--targets-file"),
    "verify": ("--secret-dir", "--targets-file"),
    "setup": ("--secret-dir",),
}


def _discover_inventory() -> list[Path]:
    """Inventory candidates: well-known names plus ``config/*.yaml``. A file
    only counts when it loads, passes lint AND declares a top-level ``hosts:``
    or ``console_defaults:`` key -- even when empty, so a minimal inventory
    created by ``harness setup`` (``hosts: []``) is discoverable while
    ``config/targets.yaml`` etc. never look like an inventory."""
    candidates: list[Path] = []
    seen: set[str] = set()
    names: list[Path] = []
    config_dir = Path("config")
    if config_dir.is_dir():
        names += sorted(config_dir.glob("*.yaml")) + sorted(config_dir.glob("*.yml"))
    names += [Path("inventory.yaml"), Path("inventory.yml")]
    for p in names:
        try:
            key = str(p.resolve())
        except OSError:
            key = str(p)
        if key in seen or not p.is_file():
            continue
        seen.add(key)
        try:
            raw = yaml.safe_load(p.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or not (
                    "hosts" in raw or "console_defaults" in raw):
                continue
            load_inventory(p)  # lint: unreadable/lint-failing files are skipped
        except Exception:  # noqa: BLE001, S112 - unreadable/lint-failing files are skipped
            continue
        candidates.append(p)
    return candidates


def _pick_inventory(args) -> Path | None:
    from .menu import select

    explicit = getattr(args, "inventory", None)
    if explicit:
        return Path(explicit)
    found = _discover_inventory()
    if not found:
        interactive = (sys.stdin.isatty()
                       and os.environ.get("HARNESS_NO_PROMPT") != "1")
        if interactive:
            print("no inventory found: launching `harness setup` to create one "
                  "(LLM key, SSH key, inventory.yaml)...", file=sys.stderr)
            try:
                _run_wizard_sub(["setup"] + _wizard_flags(args, "setup"))
            except KeyboardInterrupt:
                print("  setup cancelled", file=sys.stderr)
                return None
            found = _discover_inventory()
        if not found:
            print("no inventory found: pass --inventory <file> or put one under config/",
                  file=sys.stderr)
            return None
    if len(found) == 1:
        print(ui.accent(f"  inventory: {found[0]}"))
        return found[0]
    idx = select("Inventory", [str(p) for p in found])
    return found[idx] if idx is not None else None


def _ssh_target_args(args) -> dict:
    return {
        "ssh_user": getattr(args, "ssh_user", "diagbot"),
        "identity_vault_path": getattr(args, "identity_vault_path", None),
        "known_hosts_path": getattr(args, "known_hosts_path", "config/known_hosts"),
    }


def _pick_target(inv, store, args, *, console: bool) -> TargetSpec | None:
    """Menu to pick a target: named host / alias / rack+cable / IP / none.

    Resolves eagerly so a doomed target (missing console_defaults, unregistered
    identity) is reported before any run is launched.
    """
    from .menu import ask_text, select

    targets_file = getattr(args, "targets_file", None) or "config/targets.yaml"
    hosts = inv.hosts if not console else [h for h in inv.hosts if h.console is not None]
    choices: list[tuple[str, TargetSpec | None]] = [
        (f"{h.name}  ({h.trust_level}, {h.model})", TargetSpec(name=h.name))
        for h in sorted(hosts, key=lambda h: h.name)
    ]
    aliases = load_targets(targets_file)
    for alias in sorted(aliases):
        entry = aliases[alias]
        where = entry.address or f"{entry.rack}-cable{entry.cable}"
        choices.append((f"alias {alias}  ({where})", TargetSpec(alias=alias)))
    rack_idx = len(choices)
    choices.append(("Rack + cable (console target, zero-YAML)", None))
    ip_idx = len(choices)
    choices.append(("IP address (SSH by address)", None))
    none_idx = len(choices)
    choices.append(("(none - name a target later)", None))

    idx = select("Target", [label for label, _ in choices])
    if idx is None or idx == none_idx:
        return None
    if idx == rack_idx:
        rack = ask_text("Rack id (e.g. Q61)").strip()
        cable = ask_text("Cable number").strip()
        if not rack or not cable:
            print("  cancelled: rack and cable are both required", file=sys.stderr)
            return None
        spec = TargetSpec(rack=rack, cable=cable)
    elif idx == ip_idx:
        ip = ask_text("IP address").strip()
        if not _IP4_RE.match(ip):
            print(f"  cancelled: {ip!r} is not an IPv4 address", file=sys.stderr)
            return None
        spec = TargetSpec(ip=ip)
    else:
        spec = choices[idx][1]
    try:
        target = resolve_target(spec, inv, store, targets_path=targets_file,
                                **_ssh_target_args(args))
    except TargetError as exc:
        print(f"  x target: {exc}", file=sys.stderr)
        return None
    print(f"  target: {target.label} ({target.kind})")
    return spec


def _target_argv(spec: TargetSpec) -> list[str]:
    argv: list[str] = []
    if spec.name is not None:
        argv += ["--host", spec.name]
    elif spec.rack is not None and spec.cable is not None:
        argv += ["--rack", spec.rack, "--cable", spec.cable]
    elif spec.ip is not None:
        argv += ["--address", spec.ip]
    elif spec.alias is not None:
        argv += ["--target", spec.alias]
    return argv


def _wizard_flags(args, subcommand: str) -> list[str]:
    """Wizard-chosen flags that the target subcommand accepts (unknown flags
    would make argparse error, so only pass what each subcommand takes)."""
    argv: list[str] = []
    for flag in _ALLOWED_FLAGS.get(subcommand, ()):
        value = getattr(args, _WIZARD_FLAGS[flag], None)
        if value:
            values = value if isinstance(value, list) else [value]
            for v in values:
                argv += [flag, str(v)]
    if subcommand in ("diagnose", "session"):
        llm = getattr(args, "llm", None)
        if llm:
            argv += ["--llm", llm]
        if getattr(args, "console", False):
            argv.append("--console")
        if getattr(args, "ask_parts", False):
            argv.append("--ask-parts")
    return argv


def _run_wizard_sub(argv: list[str]) -> int:
    """Parse and run a subcommand synthesized by the menu; returns an exit
    code. Errors are printed and the menu keeps running."""
    try:
        args = build_parser().parse_args(argv)
    except SystemExit:
        return 2
    try:
        result = args.func(args)
        return result if isinstance(result, int) else 0
    except SystemExit as exc:  # e.g. secrets without --secret-dir
        if str(exc):
            print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - back to the menu, never crash it
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _baselines(out_dir: str | Path) -> list[Path]:
    root = Path(out_dir)
    if not root.is_dir():
        return []
    return sorted(
        (p for p in root.iterdir()
         if p.is_dir() and (p / "dumps.json").is_file()),
        key=lambda p: p.stat().st_mtime, reverse=True)


def _menu_model(inv, store) -> int:
    """Pick the LLM reasoning model from the catalog (remembered in
    ``config/models.yaml`` so the next run uses it). Selecting an
    unconfigured built-in (``local/harness-diag`` etc.) or the ``+`` row
    launches the guided setup; every non-stub pick is probed against the
    endpoint's served list."""
    from dataclasses import replace as _dc_replace

    from ..config.model_catalog import ModelCatalog, picker_rows
    from .menu import ask_model_profile, check_profile, select

    catalog = ModelCatalog.load(inv=inv)
    labels, profiles, add_idx = picker_rows(catalog)
    idx = select("LLM model (reasoning backend)", labels)
    if idx is None:
        return 0
    if idx == add_idx:
        profile = ask_model_profile(inv=inv, store=store)
        if profile is None:
            return 0
        catalog.add(profile)
    else:
        profile = profiles[idx]
        if profile.needs_setup:
            guided = ask_model_profile(provider=profile.provider, inv=inv,
                                       store=store)
            if guided is None:
                return 0
            catalog.add(guided)
            profile = guided
        else:
            catalog.choose(profile)
    suggestion = check_profile(profile, inv=inv, store=store)
    if suggestion and suggestion != profile.model:
        profile = _dc_replace(profile, model=suggestion)
        catalog.choose(profile)
    catalog.save()
    print(f"  model: {catalog.current.ident} (remembered for the next run)")
    return 0


def _menu_docs(args) -> int:
    from ..docs.ingest.library import DocLibrary
    from .menu import ask_text, select

    lib = getattr(args, "docs_lib", None) or "harness_docs"
    actions = [("add", "add PDF(s)"), ("list", "list documents"),
               ("rm", "remove a document"), ("reindex", "re-index all PDFs"),
               (_BACK, _BACK)]
    while True:
        idx = select("Doc library", [label for _, label in actions])
        if idx is None:
            return 0
        key = actions[idx][0]
        if key == _BACK:
            return 0
        if key == "add":
            raw = ask_text("PDF path(s), space-separated").strip()
            files = shlex.split(raw) if raw else []
            if files:
                _run_wizard_sub(["docs", "--lib", lib, "add", *files])
        elif key == "list":
            _run_wizard_sub(["docs", "--lib", lib, "ls"])
        elif key == "rm":
            names = [e.name for e in DocLibrary(lib).entries()]
            if not names:
                print("  (empty library)")
                continue
            pick = select("Remove document", names)
            if pick is not None:
                _run_wizard_sub(["docs", "--lib", lib, "rm", names[pick]])
        else:
            _run_wizard_sub(["docs", "--lib", lib, "reindex"])
    return 0


def _menu_targets(args) -> int:
    from .menu import ask_text, select

    path = getattr(args, "targets_file", None) or "config/targets.yaml"
    actions = [("add", "add an alias"), ("ls", "list aliases"),
               ("rm", "remove an alias"), (_BACK, _BACK)]
    while True:
        idx = select("Target aliases", [label for _, label in actions])
        if idx is None:
            return 0
        key = actions[idx][0]
        if key == _BACK:
            return 0
        if key == "add":
            alias = ask_text("Alias (letters/digits/._:-)").strip()
            rack = ask_text("Rack (optional)").strip() or None
            cable = ask_text("Cable (optional)").strip() or None
            ip = ask_text("IP address (optional)").strip() or None
            if not alias:
                print("  cancelled: alias is required")
                continue
            argv = ["targets", "--targets-file", path, "add", alias]
            if rack:
                argv += ["--rack", rack]
            if cable:
                argv += ["--cable", cable]
            if ip:
                argv += ["--address", ip]
            _run_wizard_sub(argv)
        elif key == "ls":
            _run_wizard_sub(["targets", "--targets-file", path, "ls"])
        else:
            aliases = load_targets(path)
            if not aliases:
                print("  no aliases registered")
                continue
            names = sorted(aliases)
            pick = select("Remove", names)
            if pick is not None:
                _run_wizard_sub(["targets", "--targets-file", path, "rm",
                                 names[pick]])
    return 0


def _menu_secrets(args) -> int:
    from .menu import ask_text, select

    secret_dir = getattr(args, "secret_dir", None)
    if secret_dir is None:
        print("  secrets need --secret-dir <dir> (file-backed lab store)",
              file=sys.stderr)
        return 1
    actions = [("add-ssh", "add SSH key (from a file)"),
               ("set-password", "set a password (never echoed)"),
               ("list", "list registered vault paths"),
               ("rm", "remove a secret"), ("check", "verify a vault path"),
               (_BACK, _BACK)]
    while True:
        idx = select("Credentials (non-agent)", [label for _, label in actions])
        if idx is None:
            return 0
        key = actions[idx][0]
        if key == _BACK:
            return 0
        if key == "add-ssh":
            name = ask_text("Name (use the target IP for --address targeting)").strip()
            key_file = ask_text("Private key FILE path").strip()
            if not name or not key_file:
                print("  cancelled: name and key file are required")
                continue
            _run_wizard_sub(["secrets", "add-ssh", name, "--key-file", key_file,
                             "--secret-dir", secret_dir])
        elif key == "set-password":
            name = ask_text("Name").strip()
            if not name:
                print("  cancelled: name is required")
                continue
            _run_wizard_sub(["secrets", "set-password", name,
                             "--secret-dir", secret_dir])
        elif key == "list":
            _run_wizard_sub(["secrets", "list", "--secret-dir", secret_dir])
        elif key == "rm":
            keys = sorted(DirSecretStore(secret_dir).keys())
            if not keys:
                print("  no secrets registered")
                continue
            pick = select("Remove", keys)
            if pick is not None:
                _run_wizard_sub(["secrets", "rm", keys[pick].rsplit("/", 1)[-1],
                                 "--secret-dir", secret_dir])
        else:
            name = ask_text("Name").strip()
            if not name:
                print("  cancelled: name is required")
                continue
            _run_wizard_sub(["secrets", "check", name,
                             "--secret-dir", secret_dir])
    return 0


# ---- past-run inspection (runs menu) ----

#: A directory must contain at least one of these to be a diagnosable run
#: (``dumps/`` alone also qualifies). Reserved harness_runs subdirectories are
#: never runs regardless of contents.
_RUN_ARTIFACTS = ("audit.jsonl", "diagnosis.json", "trace.json",
                  "pending_case.json", "run_meta.json", "prompt.txt",
                  "prompt_turns.jsonl", "dumps.json")
_RUN_RESERVED_DIRS = frozenset({"sessions", "cases", "secrets", "calibration"})

#: ``Product Serial : 4CX1234`` from ``ipmitool fru print`` collector output.
_FRU_SERIAL_RE = re.compile(r"Product Serial\s*:\s*(\S+)")


def _is_run_dir(path: Path) -> bool:
    if path.name in _RUN_RESERVED_DIRS or not path.is_dir():
        return False
    return any((path / m).exists() for m in _RUN_ARTIFACTS) \
        or (path / "dumps").is_dir()


def _run_start_event(run_dir: Path) -> tuple[dict, str | None]:
    """(payload, ts) of the run's first audit event (run_start/console_start).

    Tolerates missing/corrupt logs: returns ({}, None) instead of raising.
    """
    path = run_dir / "audit.jsonl"
    if not path.exists():
        return {}, None
    try:
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                event = json.loads(line)
                payload = event.get("payload")
                return (payload if isinstance(payload, dict) else {},
                        event.get("ts"))
    except (json.JSONDecodeError, OSError):
        return {}, None
    return {}, None


def _read_run_meta(run_dir: Path) -> dict:
    """The small display-metadata file written at run end (may not exist)."""
    path = run_dir / "run_meta.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _run_verdict(run_dir: Path) -> tuple[str | None, float | None]:
    """(state, confidence) from the run's diagnosis.json, when readable."""
    path = run_dir / "diagnosis.json"
    if not path.exists():
        return None, None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None, None
    state = data.get("state")
    conf = data.get("confidence")
    return (state if isinstance(state, str) else None,
            float(conf) if isinstance(conf, (int, float)) else None)


#: BMCs report these placeholders when no serial is programmed: not a serial.
_FRU_SERIAL_PLACEHOLDERS = frozenset({"n/a", "na", "none", "unknown", "0123456789"})


def _clean_serial(value: str | None) -> str | None:
    """The FRU serial, or None when it is a BMC placeholder."""
    if not value:
        return None
    return None if value.strip().lower() in _FRU_SERIAL_PLACEHOLDERS \
        else value.strip()


def _serial_from_dumps(run_dir: Path) -> str | None:
    """Machine serial from the FRU dump (fallback for runs written before
    ``run_meta.json`` existed); scans ipmi dumps first, then the rest."""
    dumps_dir = run_dir / "dumps"
    if not dumps_dir.is_dir():
        return None
    files = sorted(dumps_dir.glob("ipmi*.txt")) + sorted(
        p for p in dumps_dir.glob("*.txt") if not p.name.startswith("ipmi"))
    for path in files:
        try:
            match = _FRU_SERIAL_RE.search(
                path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
        if match:
            serial = _clean_serial(match.group(1))
            if serial:
                return serial
    return None


def _case_summary(run_dir: Path, cases_dir: Path) -> tuple[str | None, str | None]:
    """(outcome, first action taken) from the run's labeled case record."""
    path = cases_dir / f"{run_dir.name}.json"
    if not path.exists():
        return None, None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None, None
    outcome = data.get("outcome")
    taken = data.get("actions_taken") or []
    first = str(taken[0]) if taken else None
    return (outcome if isinstance(outcome, str) else None, first)


def _summarize_run(run_dir: Path, cases_dir: Path | None = None) -> str:
    """Compact identifying one-liner for the runs picker, e.g.::

        2026-08-11 21:08 | Q63-cable2 (rack Q63 cable 2) | SN:4CX1234 | FAULT 92% | fixed: reseated CPU B

    Fields drop out when unknown; a run with no metadata at all degenerates
    to its directory name. The picker's type-to-filter matches any substring,
    so rack/cable/serial/date tokens are all searchable.
    """
    if cases_dir is None:
        cases_dir = run_dir.parent / "cases"
    start, ts = _run_start_event(run_dir)
    meta = _read_run_meta(run_dir)
    when = str(ts or meta.get("ts") or "")[:16].replace("T", " ")
    host = meta.get("host") or start.get("host")
    rack = meta.get("rack") or start.get("rack")
    cable = meta.get("cable") or start.get("cable")
    where = host or None
    if where and rack is not None and cable is not None:
        where = f"{where} (rack {rack} cable {cable})"
    serial = _clean_serial(meta.get("serial")) or _serial_from_dumps(run_dir)
    state, confidence = _run_verdict(run_dir)
    verdict = None
    if state:
        verdict = state.upper()
        if confidence is not None:
            verdict = f"{verdict} {round(confidence * 100)}%"
    outcome, fix = _case_summary(run_dir, cases_dir)
    fix_bit = (f"{outcome}: {fix}" if fix
               else f"outcome: {outcome}" if outcome else None)
    bits = [b for b in (when, where,
                        f"SN:{serial}" if serial else None,
                        verdict, fix_bit) if b]
    return " | ".join(bits) if bits else run_dir.name


def _print_run_header(run_dir: Path, cases_dir: Path) -> None:
    """Context block shown after picking a run in the inspector."""
    start, ts = _run_start_event(run_dir)
    meta = _read_run_meta(run_dir)
    state, confidence = _run_verdict(run_dir)
    outcome, fix = _case_summary(run_dir, cases_dir)
    serial = _clean_serial(meta.get("serial")) or _serial_from_dumps(run_dir)
    rack = meta.get("rack") or start.get("rack")
    cable = meta.get("cable") or start.get("cable")
    verdict = None
    if state:
        verdict = state.upper()
        if confidence is not None:
            verdict = f"{verdict} {round(confidence * 100)}%"
    print(f"---- run {run_dir.name} ----")
    rows = [
        ("dir", str(run_dir)),
        ("date", ts or meta.get("ts")),
        ("host", meta.get("host") or start.get("host")),
        ("rack/cable", f"{rack} / {cable}"
         if rack is not None and cable is not None else None),
        ("serial", serial),
        ("model", meta.get("model") or start.get("model")),
        ("symptom", start.get("symptom")),
        ("verdict", verdict),
        ("fix", fix),
        ("outcome", outcome),
    ]
    for label, value in rows:
        if value:
            print(f"  {label:<9} {value}")


def _write_run_meta(out: Path, target, host, session_id: str) -> None:
    """Write ``run_meta.json`` (display metadata) into a finished run dir.

    Captures the machine serial parsed from the FRU collector output so the
    runs menu never has to re-parse the full dumps just to show one label.
    """
    serial: str | None = None
    dumps_dir = out / "dumps"
    if dumps_dir.is_dir():
        for path in sorted(dumps_dir.glob("*.txt")):
            try:
                match = _FRU_SERIAL_RE.search(
                    path.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                continue
            if match:
                serial = _clean_serial(match.group(1))
                if serial:
                    break
    meta: dict = {
        "session_id": session_id,
        "host": target.label,
        "model": getattr(host, "model", None),
        "ts": datetime.now(UTC).isoformat(),
    }
    console = getattr(target, "console", None)
    if console is not None:
        meta["rack"] = console.rack
        meta["cable"] = console.cable
    if serial:
        meta["serial"] = serial
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2),
                                       encoding="utf-8")


def _print_artifact(path: Path) -> None:
    if not path.exists():
        print(f"  (missing: {path})")
        return
    print(f"---- {path} ----")
    print(path.read_text(encoding="utf-8"))


def _menu_runs(args) -> int:
    """Inspect a previous run end to end: verdict, command pathway, the exact
    prompt(s) sent to the LLM, the raw collector evidence -- and label the
    correct fix so the learning loop can consume the run.

    Runs are listed with identifying one-liners (date, host, rack/cable,
    serial, verdict, fix) instead of raw hash paths; type-to-filter matches
    any substring of those.
    """
    from .menu import select

    out_dir = Path(getattr(args, "out_dir", "harness_runs"))
    cases_dir = out_dir / "cases"
    runs = sorted((p for p in out_dir.glob("*") if _is_run_dir(p)),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        print(f"  no runs yet under {out_dir}")
        return 0
    while True:
        pick = select("Run (most recent first)",
                      [_summarize_run(p, cases_dir) for p in runs])
        if pick is None:
            return 0
        run_dir = runs[pick]
        _print_run_header(run_dir, cases_dir)
        views = [
            ("verdict", "verdict  - diagnosis.json (state, text, confidence, actions)"),
            ("commands", "commands - trace.json (every command run, in order)"),
            ("prompt", "prompt   - the exact prompt(s) sent to the LLM"),
            ("dumps", "dumps    - raw collector evidence files"),
            ("fix", "fix      - recommended vs. labeled correct fix (learning loop)"),
            ("label", "label    - record/revise the correct fix + outcome"),
            (_BACK, _BACK),
        ]
        while True:
            idx = select("Inspect", [label for _, label in views])
            if idx is None:
                return 0
            key = views[idx][0]
            if key == _BACK:
                return 0
            if key == "verdict":
                _print_artifact(run_dir / "diagnosis.json")
            elif key == "commands":
                _print_artifact(run_dir / "trace.json")
            elif key == "prompt":
                _print_run_prompt(run_dir)
            elif key == "dumps":
                _print_run_dumps(run_dir)
            elif key == "fix":
                _print_run_fix(run_dir, cases_dir)
            elif key == "label":
                _label_run(run_dir, cases_dir)


def _print_run_prompt(run_dir: Path) -> None:
    from .menu import select

    single = run_dir / "prompt.txt"
    turns = run_dir / "prompt_turns.jsonl"
    if single.exists():
        _print_artifact(single)
        return
    if not turns.exists():
        print("  (no prompt artifact in this run)")
        return
    entries = [json.loads(line) for line in
               turns.read_text(encoding="utf-8").splitlines() if line.strip()]
    options = [f"turn {e.get('turn', i + 1)}" for i, e in enumerate(entries)] + ["all turns"]
    pick = select("Prompt", options)
    if pick is None:
        return
    if pick == len(options) - 1:
        for entry in entries:
            print(f"---- turn {entry.get('turn', '?')} ----")
            print(json.dumps(entry.get("messages", entry), indent=2))
    else:
        print(json.dumps(entries[pick].get("messages", entries[pick]), indent=2))


def _print_run_dumps(run_dir: Path) -> None:
    from .menu import select

    dumps_dir = run_dir / "dumps"
    files = sorted(dumps_dir.glob("*.txt")) if dumps_dir.is_dir() else []
    if not files:
        print("  (no dump files in this run)")
        return
    pick = select("Dump", [p.name for p in files])
    if pick is None:
        return
    _print_artifact(files[pick])


def _print_run_fix(run_dir: Path, cases_dir: Path) -> None:
    """Learning-loop view: what the LLM recommended vs. what actually fixed it."""
    recommended: list[str] = []
    pending = run_dir / "pending_case.json"
    if pending.exists():
        try:
            data = json.loads(pending.read_text(encoding="utf-8"))
            recommended = [str(a) for a in data.get("actions_recommended") or []]
        except (json.JSONDecodeError, OSError):
            recommended = []
    if not recommended:
        diag = run_dir / "diagnosis.json"
        if diag.exists():
            try:
                data = json.loads(diag.read_text(encoding="utf-8"))
                recommended = [
                    f"{a.get('step', '?')}. {a.get('action', '')}"
                    for a in data.get("actions", []) if isinstance(a, dict)]
            except (json.JSONDecodeError, OSError):
                recommended = []
    print("---- recommended (what the LLM said) ----")
    if not recommended:
        print("  (no recommended actions recorded in this run)")
    for line in recommended:
        print(f"  {line}")
    print("---- labeled (what actually fixed it) ----")
    outcome, fix = _case_summary(run_dir, cases_dir)
    if not outcome and not fix:
        print("  (not labeled yet: use the 'label' view to record the correct fix)")
        return
    if fix:
        print(f"  {fix}")
    if outcome:
        print(f"  outcome: {outcome}")
    case_path = cases_dir / f"{run_dir.name}.json"
    if case_path.exists():
        try:
            case = json.loads(case_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            case = None
        if case:
            for line in (case.get("actions_taken") or [])[1:]:
                print(f"  {line}")
            if case.get("verification_verdict"):
                print(f"  verification: {case['verification_verdict']}")


def _synthesize_case(run_dir: Path):
    """Minimal ``CaseOutcome`` for old runs without ``pending_case.json``.

    Rebuilds the deterministic pieces from the run's own artifacts
    (diagnosis.json + audit run_start); returns None when there is nothing
    to synthesize from.
    """
    from ..diagnosis.schema import CaseOutcome

    start, _ts = _run_start_event(run_dir)
    diag: dict = {}
    diag_path = run_dir / "diagnosis.json"
    if diag_path.exists():
        try:
            loaded = json.loads(diag_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                diag = loaded
        except (json.JSONDecodeError, OSError):
            diag = {}
    if not diag and not start:
        return None
    model_key, model_source = _model_from_audit(run_dir / "audit.jsonl")
    subsystems = diag.get("subsystems_considered") or []
    evidence_hash = ""
    if (run_dir / "dumps.json").exists():
        try:
            evidence_hash = _evidence_hash(run_dir / "dumps.json")
        except OSError:
            evidence_hash = ""
    breakdown = diag.get("confidence_breakdown") or {}
    return CaseOutcome(
        run_id=run_dir.name,
        target_id=str(start.get("host") or run_dir.name),
        model_key=model_key,
        model_source=model_source,
        symptom=str(start.get("symptom") or "(unknown)"),
        subsystem_primary=(subsystems[0] if subsystems else None),
        actions_recommended=[
            f"{a.get('step', '?')}. {a.get('action', '')}"
            for a in diag.get("actions", []) if isinstance(a, dict)],
        llm_ident="unknown",
        evidence_hash=evidence_hash,
        evidence_summary=[
            f"- {d.get('mnemonic', '?')} = {d.get('raw_hex', '?')}"
            for d in diag.get("evidence", []) if isinstance(d, dict)],
        confidence=(diag.get("confidence")
                    if isinstance(diag.get("confidence"), (int, float)) else None),
        self_reported_confidence=(
            breakdown.get("self_reported_confidence")
            if isinstance(breakdown.get("self_reported_confidence"),
                          (int, float)) else None),
    )


#: Verified outcomes an operator can label a run with (unknown = not labeled).
_LABEL_OUTCOMES = ("fixed", "partial", "not_fixed", "inconclusive")

_DEFER_HINT = "(defer - label it later from Inspect past runs)"


def _label_run(run_dir: Path, cases_dir: Path, *, outcome: str | None = None,
               taken: list[str] | None = None, verdict: str | None = None,
               revise: bool = False, allow_defer: bool = False) -> int:
    """Record the verified outcome + correct fix for a run: the learning
    loop's ground truth (feeds case retrieval, priors and calibration).

    Shared by the runs menu, the post-diagnosis prompt and ``harness label``.
    Interactive prompts fill whatever the caller left None; a blank outcome
    with ``allow_defer=True`` defers labeling (the run keeps its seeded
    ``pending_case.json`` and can be labeled any time from the runs menu).
    Returns a CLI-style exit code.
    """
    from ..diagnosis.case_store import CaseStore
    from ..diagnosis.schema import CaseOutcome
    from .menu import ask_text, confirm, select

    pending = run_dir / "pending_case.json"
    base: CaseOutcome | None = None
    if pending.exists():
        try:
            base = CaseOutcome.model_validate_json(
                pending.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - corrupt seed: synthesize instead
            base = None
    if base is None:
        base = _synthesize_case(run_dir)
    if base is None:
        print(f"  cannot label {run_dir.name}: no pending_case.json, "
              "diagnosis.json or audit.jsonl to seed the record from")
        return 1

    interactive = sys.stdin.isatty() and os.environ.get("HARNESS_NO_PROMPT") != "1"
    store = CaseStore(cases_dir)
    existing = store.get(run_dir.name)
    if existing is not None and existing.outcome != "unknown" and not revise:
        if not interactive:
            print(f"error: case {run_dir.name!r} already recorded "
                  f"(outcome={existing.outcome}); pass --revise to correct it",
                  file=sys.stderr)
            return 1
        print(f"  already labeled: outcome={existing.outcome} "
              f"fix={'; '.join(existing.actions_taken) or '(none)'}")
        if not confirm("Revise this label? (overwrites the record)"):
            print("  kept the existing label")
            return 0
        revise = True

    if outcome is None:
        if not interactive:
            print("error: --outcome is required (fixed | partial | not_fixed | "
                  "inconclusive)", file=sys.stderr)
            return 1
        options = list(_LABEL_OUTCOMES) + ([_DEFER_HINT] if allow_defer else [])
        pick = select("Outcome", options)
        if pick is None:
            if allow_defer:
                return 0
            return 0
        if allow_defer and pick == len(_LABEL_OUTCOMES):
            print("  deferred: label later via 'Inspect past runs' > label")
            return 0
        outcome = _LABEL_OUTCOMES[pick] if pick < len(_LABEL_OUTCOMES) else None
        if outcome is None:
            print(f"  unknown outcome; pick one of {', '.join(_LABEL_OUTCOMES)}")
            return _label_run(run_dir, cases_dir, outcome=None, taken=taken,
                              verdict=verdict, revise=revise,
                              allow_defer=allow_defer)
    elif outcome not in _LABEL_OUTCOMES:
        if interactive:
            print(f"  unknown outcome {outcome!r}; pick one")
            return _label_run(run_dir, cases_dir, outcome=None, taken=taken,
                              verdict=verdict, revise=revise,
                              allow_defer=allow_defer)
        print(f"error: unknown outcome {outcome!r} "
              f"(fixed | partial | not_fixed | inconclusive)", file=sys.stderr)
        return 1

    if taken is None:
        taken = []
        if interactive:
            prefill = ""
            for action in base.actions_recommended:
                prefill = action
                break
            fix = ask_text("Correct fix (Enter = top recommendation was right)",
                           default=prefill).strip()
            if fix:
                taken.append(fix)
            while confirm("Add another fix line?"):
                extra = ask_text("Fix detail").strip()
                if not extra:
                    break
                taken.append(extra)
    if not taken:
        print("  (no fix text recorded: the case will carry the outcome only)")

    record = CaseOutcome(
        **{**base.model_dump(),
           "run_id": run_dir.name,
           "outcome": outcome,
           "actions_taken": list(taken),
           "verification_verdict": verdict,
           "created_at": "",  # CaseStore stamps it
           })
    try:
        audit_log = AuditLog(run_dir / "audit.jsonl")
    except (json.JSONDecodeError, KeyError, OSError):
        # corrupt/truncated audit chain on an old run: label without the
        # audit linkage rather than refusing the ground-truth record
        audit_log = None
        print("  [label] warning: audit.jsonl unreadable; recording the case "
              "without audit linkage")
    try:
        path = store.record(record, audit=audit_log,
                            session_id=record.run_id, revise=revise)
    except (FileExistsError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"recorded case {record.run_id}: outcome={record.outcome} "
          f"fix={'; '.join(record.actions_taken) or '(none)'}")
    print(f"  {path}")
    print("rebuild the learned priors from labeled cases with: "
          "harness priors update")
    return 0


def _should_prompt_label(args) -> bool:
    """The post-diagnosis labeling prompt is opt-in so background chat runs
    (REPL worker threads) never block on stdin: the interactive menu passes
    ``--label-prompt`` and direct ``--interactive`` runs imply it."""
    if not sys.stdin.isatty() or os.environ.get("HARNESS_NO_PROMPT") == "1":
        return False
    return bool(getattr(args, "label_prompt", False)
                or getattr(args, "interactive", False))


def _prompt_label_after_run(run_dir: Path, cases_dir: Path) -> None:
    """Offer to label the fix right after a diagnosis completes; blank defers
    (the seeded pending_case.json stays, the run shows as un-labeled in the
    runs menu and can be labeled any time)."""
    from .menu import ask_text

    print()
    answer = ask_text(
        "Label the correct fix now? fixed/partial/not_fixed/inconclusive "
        "(Enter = defer)").strip().lower()
    if not answer:
        print("  deferred: label later via 'Inspect past runs' > label")
        return
    _label_run(run_dir, cases_dir, outcome=answer)


def run_menu(args) -> int:
    """Interactive launcher: pick the inventory, then drive every subcommand
    from one prompt. This is the default command when ``harness`` is run
    with no arguments."""
    from .menu import ask_text, select

    inv_path = _pick_inventory(args)
    if inv_path is None:
        return 2
    inv = load_inventory(inv_path)
    store = _make_store(args)
    if not store.keys() and not getattr(inv, "llm", None):
        print("  hint: nothing registered yet -- pick `setup` to register the "
              "LLM API key and SSH identity up front, or just start: the "
              "agent will ask you for each credential the moment it needs it "
              "(never through the model)", file=sys.stderr)
    console_default = bool(getattr(args, "console", False))
    ui.enable_vt()
    width = max(44, min(ui.terminal_width(), 78))
    print(ui.rule(width))
    console_tag = ui.dim("  console mode") if console_default else ""
    print(f"  {ui.title('harness menu')}   "
          f"{ui.dim(f'inventory: {inv_path} - {len(inv.hosts)} host(s)')}{console_tag}")
    print(ui.rule(width))
    print(ui.dim("  most days: Chat or Debug - everything else lives under Advanced"))

    def _run_advanced() -> None:
        """The housekeeping submenu; returns to the main menu on Back/Esc."""
        while True:
            choice = select("Advanced", [label for _, label in _ADVANCED_ACTIONS])
            if choice is None:
                return
            key = _ADVANCED_ACTIONS[choice][0]
            if key == _BACK:
                return
            try:
                if key == "verify":
                    spec = _pick_target(inv, store, args, console=console_default)
                    if spec is None:
                        continue
                    baselines = _baselines(getattr(args, "out_dir", "harness_runs"))
                    if not baselines:
                        print("  no baseline yet: run a diagnosis first")
                        continue
                    pick = select("Baseline (most recent first)",
                                  [str(p) for p in baselines])
                    if pick is None:
                        continue
                    argv = ["verify", "--inventory", str(inv_path),
                            "--symptom", "(menu verify)",
                            "--baseline", str(baselines[pick])]
                    argv += _target_argv(spec)
                    argv += _wizard_flags(args, "verify")
                    _run_wizard_sub(argv)
                elif key == "console":
                    spec = _pick_target(inv, store, args, console=True)
                    if spec is None:
                        continue
                    probe = ask_text("Probe command (read-only)").strip()
                    if not probe:
                        print("  cancelled: no probe command")
                        continue
                    argv = ["console", "--inventory", str(inv_path),
                            "--probe", probe]
                    argv += _target_argv(spec)
                    argv += _wizard_flags(args, "console")
                    _run_wizard_sub(argv)
                elif key == "model":
                    _menu_model(inv, store)
                elif key == "docs":
                    _menu_docs(args)
                elif key == "targets":
                    _menu_targets(args)
                elif key == "secrets":
                    _menu_secrets(args)
                elif key == "setup":
                    _run_wizard_sub(["setup", "--inventory", str(inv_path)]
                                    + _wizard_flags(args, "setup"))
                elif key == "lint":
                    _run_wizard_sub(["lint", "--inventory", str(inv_path)])
            except KeyboardInterrupt:
                print("  (interrupted)")

    while True:
        choice = select("What do you want to do?",
                        [label for _, label in _MAIN_ACTIONS])
        if choice is None:
            return 0
        key = _MAIN_ACTIONS[choice][0]
        if key == "quit":
            return 0
        try:
            if key == "advanced":
                _run_advanced()
            elif key == "chat":
                spec = _pick_target(inv, store, args, console=console_default)
                argv = ["session", "--inventory", str(inv_path)]
                if spec is not None:
                    argv += _target_argv(spec)
                argv += _wizard_flags(args, "session")
                _run_wizard_sub(argv)
            elif key == "diagnose":
                spec = _pick_target(inv, store, args, console=console_default)
                if spec is None:
                    continue
                symptom = ask_text(
                    "Symptom (Enter to derive from the test log / live "
                    "evidence)").strip()
                test_log = ""
                while True:
                    candidate = _strip_surrounding_quotes(ask_text(
                        "Test-harness/FAT log path (Enter to skip)"))
                    if not candidate:
                        break
                    if Path(candidate).is_file():
                        test_log = candidate
                        break
                    print(f"  no such file: {candidate}")
                if symptom:
                    argv = ["diagnose", "--inventory", str(inv_path),
                            "--symptom", symptom]
                elif test_log:
                    argv = ["diagnose", "--inventory", str(inv_path)]
                else:
                    print("  (no symptom: diagnosing from live evidence)")
                    argv = ["diagnose", "--inventory", str(inv_path),
                            "--symptom", _MENU_DIAGNOSE_SYMPTOM]
                if test_log:
                    argv += ["--test-log", test_log]
                argv += _target_argv(spec)
                argv += _wizard_flags(args, "diagnose")
                argv.append("--label-prompt")
                _run_wizard_sub(argv)
            elif key == "runs":
                _menu_runs(args)
        except KeyboardInterrupt:
            print("  (interrupted)")
    return 0


# ---- LLM endpoint preflight ----

def _print_stages(stages: list[tuple[str, bool, str]]) -> None:
    for name, ok, detail in stages:
        print(f"  [{'ok ' if ok else 'FAIL'}] {name:<8} {detail}")


def run_llm_check(args) -> int:
    """Staged connectivity probe for an OpenAI-compatible LLM endpoint.

    ``harness llm check [--url URL | --tunnel HOST:PORT --inventory PATH]
    [--model ID] [--timeout S]``

    Tunnel mode exercises each leg of the rack-manager hop (ssh auth ->
    direct-tcpip forward -> HTTP) and, when forwarding is refused, prints the
    reverse-tunnel fallback recipe. Direct mode probes reachability and lists
    served models. Both modes then send a minimal chat/completions probe in
    the run's exact wire shape so server-side request validation (a 400 the
    run would otherwise only hit at the ``reason`` step) fails the check
    here. Exit 0 = usable; 1 = failed stage; 2 = usage/config error.
    """
    url = getattr(args, "url", None)
    tunnel_spec = getattr(args, "tunnel", None)
    timeout = float(getattr(args, "timeout", 10.0))
    want_model = getattr(args, "model", None)
    forward = None
    stages: list[tuple[str, bool, str]] = []
    try:
        if tunnel_spec:
            target_host, target_port = parse_tunnel_spec(tunnel_spec)
            inv_path = getattr(args, "inventory", None)
            if not inv_path:
                print("llm check: --tunnel needs --inventory (the rack-manager "
                      "hop comes from its llm_console/console_defaults block)",
                      file=sys.stderr)
                return 2
            inv = load_inventory(inv_path)
            from .llm_discover import llm_bastion_domain, llm_console_domain

            domain = llm_console_domain(inv, getattr(args, "rack", None) or "")
            if domain is None:
                print("llm check: inventory has no llm_console or "
                      "console_defaults block (nothing to tunnel through)",
                      file=sys.stderr)
                return 2
            store = _make_store(args)
            print(f"rack manager: {domain.address_for_rack()}; "
                  f"target: {target_host}:{target_port}")
            forward = LLMForward(target_host, target_port,
                                 domain, store, timeout=timeout,
                                 bastion=llm_bastion_domain(
                                     inv, getattr(args, "rack", None) or ""))
            try:
                url = forward.start()
                stages.append(("ssh", True,
                               f"authenticated to {domain.address_for_rack()}"))
                stages.append(("forward", True,
                               f"{target_host}:{target_port} via {url}"))
            except TunnelError as exc:
                stages.append((exc.stage if exc.stage != "bind" else "local",
                               False, str(exc)))
                _print_stages(stages)
                print(_tunnel_failure(exc, f"{target_host}:{target_port}"),
                      file=sys.stderr)
                return 1
        elif not url:
            url = os.environ.get("HARNESS_LLM_URL") or "http://127.0.0.1:8000/v1"
        try:
            ids = list_models(url, timeout=timeout)
        except LLMError as exc:
            stages.append(("http", False, str(exc)))
            _print_stages(stages)
            return 1
        stages.append(("http", True, f"{len(ids)} model(s) served at {url}"))
        if want_model:
            served = any(m == want_model or m.endswith("/" + want_model)
                         for m in ids)
            if not served:
                reason = f"requested model {want_model!r} is not in the served list"
                stages.append(("model", False, reason))
                _print_stages(stages)
                for model_id in ids:
                    print(f"  - {model_id}")
                print(f"warning: requested model {want_model!r} is not in the "
                      "served list", file=sys.stderr)
                return 1
        probe_model = want_model or (ids[0] if ids else None)
        try:
            chat_model = probe_chat(url, model=probe_model, timeout=timeout)
        except LLMError as exc:
            stages.append(("chat", False, str(exc)))
            _print_stages(stages)
            print("  the endpoint answered /models but refused a minimal "
                  "chat/completions request in the run's wire shape",
                  file=sys.stderr)
            return 1
        stages.append(("chat", True, f"chat/completions ok (served {chat_model})"))
        _print_stages(stages)
        for model_id in ids:
            print(f"  - {model_id}")
        return 0
    finally:
        if forward is not None:
            forward.close()


def run_llm_pin_host(args) -> int:
    """Fetch the per-rack manager's SSH host key through the bastion and pin
    it into ``llm_console.known_hosts_path`` (the ``ssh-keyscan`` pattern,
    no credentials). Exit 0 = pinned; 1 = failed; 2 = usage error."""
    from .llm_discover import pin_llm_host_key

    rack = getattr(args, "rack", None)
    cable = getattr(args, "cable", None)
    if not (rack and cable):
        print("llm pin-host: --rack and --cable are required", file=sys.stderr)
        return 2
    inv = load_inventory(args.inventory)
    store = _make_store(args)
    try:
        summary = pin_llm_host_key(rack, cable, inv, store)
    except Exception as exc:  # noqa: BLE001 - staged CLI error
        print(f"llm pin-host: {exc}", file=sys.stderr)
        return 1
    print(f"  pinned: {summary}")
    print("  re-run `harness llm discover` (or the model wizard) to proceed")
    return 0


def run_llm_discover(args) -> int:
    """Node-side vLLM endpoint discovery on the rack/cable target.

    ``harness llm discover --inventory PATH --rack R --cable N`` runs a batch
    of read-only probes over the jumpin console (``hostname -I``, ``ss``,
    ``sudo -S docker ps`` when the sudo password is configured) and prints
    the candidates the model wizard's tunnel step uses: the node's
    rackmgr-side addresses, listening ports, and container port mappings.
    Exit 0 = usable candidates; 1 = probes ran but nothing parsed; 2 =
    usage/targeting error.
    """
    from ..targets.resolver import TargetError, TargetSpec, resolve_target
    from .llm_discover import discover

    rack = getattr(args, "rack", None)
    cable = getattr(args, "cable", None)
    if not (rack and cable):
        print("llm discover: --rack and --cable are required (the golden "
              "server hosting the model)", file=sys.stderr)
        return 2
    inv = load_inventory(args.inventory)
    store = _make_store(args)
    try:
        target = resolve_target(TargetSpec(rack=rack, cable=cable), inv, store)
        result = discover(rack, cable, inv, store)
    except TargetError as exc:
        print(f"llm discover: {exc}", file=sys.stderr)
        return 2
    print(f"target: {target.label} (rack manager {target.console.address_for_rack()})")
    print(f"  addresses : {', '.join(result.addresses) or '(none)'}")
    for name, port in result.containers.items():
        print(f"  container : {name} -> {port}")
    print(f"  listening : {', '.join(str(p) for p in result.ports) or '(none)'}")
    for note in result.notes:
        print(f"  note      : {note}")
    ports = result.suggested_ports()
    suggested = [(a, p) for a in result.addresses for p in ports]
    if suggested:
        print("suggested next step:")
        for addr, port in suggested:
            print(f"  harness llm check --tunnel {addr}:{port} "
                  f"--inventory {args.inventory}")
        print("  (or pick 'tunnel' in the model wizard -- it asks rack/cable "
              "and fills the rest)")
        return 0
    print("no endpoint candidates; configure the tunnel manually in the "
          "model wizard", file=sys.stderr)
    return 1


# ---- argparse ----

def _add_target_args(p: argparse.ArgumentParser, *, ssh: bool) -> None:
    """Targeting flags shared by diagnose/console/verify/session."""
    p.add_argument("--host", default=None, help="named host from the inventory")
    p.add_argument("--rack", default=None,
                   help="rack id for console targeting, e.g. Q61 (needs --cable; "
                        "uses the fleet console_defaults block -- no per-server YAML)")
    p.add_argument("--cable", default=None,
                   help="cable id for console targeting, e.g. 8 (needs --rack)")
    p.add_argument("--address", default=None,
                   help="direct SSH targeting by IPv4; identity + known_hosts "
                        "resolved from the secret store (harness secrets add-ssh)")
    p.add_argument("--target", default=None,
                   help="target alias (see 'harness targets add')")
    p.add_argument("--targets-file", default="config/targets.yaml",
                   help="target aliases file (default: config/targets.yaml)")
    if ssh:
        p.add_argument("--ssh-user", default="diagbot",
                       help="SSH user for --address targeting (default: diagbot)")
        p.add_argument("--identity-vault-path", default=None,
                       help="SSH identity vault path for --address targeting "
                            "(default: secret/harness/ssh/<ip>, fallback "
                            "secret/harness/diagbot/id_ed25519)")
        p.add_argument("--known-hosts-path", default="config/known_hosts",
                       help="known_hosts file for --address targeting")


def _add_llm_args(p: argparse.ArgumentParser, *, tunnel: bool = True) -> None:
    """LLM endpoint flags shared by menu/diagnose/session (eval: no tunnel --
    it has no inventory to hop through)."""
    p.add_argument("--llm", choices=("openai", "gemini", "local", "stub"),
                   default=None,
                   help="LLM backend: openai-compatible endpoint (default) | "
                        "gemini | local (vLLM/Ollama on a lab server) | stub "
                        "(no reasoning); defaults to the inventory 'llm' block")
    p.add_argument("--llm-url", default=None,
                   help="OpenAI-compatible base URL, e.g. http://10.0.0.42:8000/v1; "
                        "overrides inventory / catalog / HARNESS_LLM_URL")
    if tunnel:
        p.add_argument("--llm-tunnel", default=None, metavar="HOST:PORT",
                       help="serve the LLM behind the rack-manager hop: forward "
                            "HOST:PORT over the console_defaults ssh connection "
                            "for this run (implies provider 'local'); falls back "
                            "with a reverse-tunnel recipe when refused")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="harness",
        description=(
            "Read-only server diagnosis: inspect registers -> decode -> RAG -> LLM -> "
            "prioritized repair actions for human approval. No raw writes, ever.\n"
            "Run `harness` with no arguments for the interactive menu."
        ),
    )
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("menu",
                       help="interactive menu (the default when no subcommand is given)")
    p.add_argument("--inventory", default=None,
                   help="inventory file (auto-discovered under config/ when omitted)")
    p.add_argument("--secret-dir", default=None,
                   help="local dir mapping vault paths to files (lab use)")
    p.add_argument("--docs-lib", default=None,
                   help="RAG document library (default: harness_docs/ if it exists)")
    p.add_argument("--docs-dir", default=None,
                   help="directory of architecture PDFs (legacy; prefer --docs-lib)")
    p.add_argument("--parts-csv", default=None,
                   help="parts list CSV (columns: slot,fru,pn,sn)")
    p.add_argument("--ask-parts", action="store_true",
                   help="interactive: prompt for and store missing per-slot instance parts")
    p.add_argument("--parts-dir", default="config/parts",
                   help="where answered instance parts are persisted, one file per target")
    p.add_argument("--console", action="store_true",
                   help="prefer the serial console for targets (lab/qa only)")
    p.add_argument("--out-dir", default="harness_runs")
    p.add_argument("--session-dir", default="harness_runs/sessions")
    _add_llm_args(p)
    p.add_argument("--targets-file", default="config/targets.yaml",
                   help="target aliases file (default: config/targets.yaml)")
    p.set_defaults(func=run_menu)

    p = sub.add_parser("lint", help="validate an inventory file (no inline secrets)")
    p.add_argument("--inventory", required=True)
    p.set_defaults(func=run_lint)

    p = sub.add_parser("docs", help="manage the RAG document library (upload PDFs / markdown / text / CSV)")
    p.add_argument("--lib", default="harness_docs",
                   help="library directory (default: harness_docs)")
    docs_sub = p.add_subparsers(dest="docs_action", required=True)
    a = docs_sub.add_parser("add", help="upload document(s) into the library and index them")
    a.add_argument("files", nargs="+", help="file(s) to upload (pdf, md, txt, csv)")
    a.add_argument("--platform", default=None,
                   help="canonical model key to tag the documents with (retrieval "
                        "is filtered to the detected server model)")
    r = docs_sub.add_parser("rm", help="remove a document from the library")
    r.add_argument("name", help="filename as shown by 'docs ls'")
    docs_sub.add_parser("ls", help="list indexed documents")
    docs_sub.add_parser("reindex", help="re-index all documents (retries failures, "
                                        "picks up files dropped into the library)")
    t = docs_sub.add_parser("retag", help="set/clear the platform tag on indexed documents")
    t.add_argument("names", nargs="+", help="document name(s) as shown by 'docs ls'")
    t.add_argument("--platform", default=None,
                   help="canonical platform key(s) to tag with (comma-separated "
                        "for multiple); omit to clear the tag")
    p.set_defaults(func=run_docs)

    p = sub.add_parser("diagnose", help="run a read-only diagnosis on one host")
    p.add_argument("--inventory", required=True)
    _add_target_args(p, ssh=True)
    p.add_argument("--symptom", default="",
                   help="symptom to diagnose; may be omitted when --test-log is "
                        "given (default derives from the log's first failure)")
    p.add_argument("--test-log", action="append", default=None,
                   help="harness/FAT run log whose failures seed this diagnosis "
                        "(repeatable; parsed for error codes/test names, feeds the "
                        "agent evidence, doc retrieval, and the learning loop)")
    p.add_argument("--secret-dir", help="local dir mapping vault paths to files (lab use)")
    p.add_argument("--parts-csv", help="parts list CSV (columns: slot,fru,pn,sn)")
    p.add_argument("--docs-lib", default=None,
                   help="RAG document library managed by 'harness docs' "
                        "(default: harness_docs/ if it exists)")
    p.add_argument("--docs-dir", help="directory of architecture PDFs used for RAG "
                                      "(legacy ad-hoc; prefer --docs-lib)")
    p.add_argument("--interactive", action="store_true",
                   help="multi-turn session: the agent may ask questions (e.g. previous "
                        "repair actions) and request further read-only probes before "
                        "diagnosing")
    p.add_argument("--label-prompt", action="store_true",
                   help="after the diagnosis, prompt to label the correct fix "
                        "(Enter defers; the interactive menu passes this)")
    p.add_argument("--context", action="append", default=None,
                   help="extra human-supplied context, e.g. 'DIMM was reseated, no "
                        "change' (repeatable; implies session mode)")
    p.add_argument("--context-file", action="append", default=None,
                   help="file whose contents are added as human-supplied context "
                        "(repeatable; implies session mode)")
    p.add_argument("--max-turns", type=int, default=6,
                   help="session turn budget before a forced diagnosis (default: 6)")
    p.add_argument("--server-number", type=int, default=None,
                   help="server number for the vendor FAT single-test menu "
                        "(GB targets, SSH/session mode only)")
    p.add_argument("--console", action="store_true",
                   help="run probes over the serial console (rack manager + cable) "
                        "instead of SSH (lab/qa only)")
    p.add_argument("--console-address", default=None,
                   help="override the rack manager console address (inventory default)")
    p.add_argument("--port", type=int, default=None,
                   help="override the BMC access port (inventory default: 2200)")
    p.add_argument("--sudo-vault-path", default=None,
                   help="override the vault path of the BMC sudo password")
    p.add_argument("--out-dir", default="harness_runs")
    _add_llm_args(p)
    p.add_argument("--llm-model", default=None,
                   help="specific LLM model, e.g. gemini/gemini-2.5-pro, "
                        "local/Qwen2.5-7B-Instruct or gpt-4o; overrides --llm / "
                        "inventory / the remembered model (config/models.yaml)")
    p.add_argument("--approval", action="store_true", help="prompt y/N for each action")
    p.add_argument("--approve-all", action="store_true", help="record every action approved")
    p.add_argument("--wall-s", type=float, default=900.0, help="supervisor wall-clock budget")
    p.add_argument("--ask-parts", action="store_true",
                   help="interactive: when a rail fault is decoded, prompt for the "
                        "operator's per-slot instance parts and store them for future runs")
    p.add_argument("--parts-dir", default="config/parts",
                   help="where answered instance parts are persisted, one file per target")
    p.set_defaults(func=run_diagnose)

    p = sub.add_parser("console", help="run read-only probes over the serial console (lab/qa only)")
    p.add_argument("--inventory", required=True)
    _add_target_args(p, ssh=False)
    p.add_argument("--probe", action="append", required=True,
                   help="read-only probe command (repeatable)")
    p.add_argument("--secret-dir", help="local dir mapping vault paths to files (lab use)")
    p.add_argument("--console-address", default=None,
                   help="override the rack manager console address (inventory default)")
    p.add_argument("--port", type=int, default=None,
                   help="override the BMC access port (inventory default: 2200)")
    p.add_argument("--sudo-vault-path", default=None,
                   help="override the vault path of the BMC sudo password")
    p.add_argument("--out-dir", default=None,
                   help="save console.txt + audit.jsonl + trace.json to a run dir")
    p.set_defaults(func=run_console)

    p = sub.add_parser("verify", help="re-run collectors and compare to a baseline dump set")
    p.add_argument("--inventory", required=True)
    _add_target_args(p, ssh=True)
    p.add_argument("--symptom", required=True)
    p.add_argument("--baseline", required=True, help="dumps.json from a previous diagnose run")
    p.add_argument("--secret-dir", help="local dir mapping vault paths to files (lab use)")
    p.add_argument("--metric", default="ecc", help="error-counter keyword to compare")
    p.set_defaults(func=run_verify)

    p = sub.add_parser("session",
                       help="interactive chat session (Claude-Code-style REPL): "
                            "type symptoms in natural language, the agent diagnoses "
                            "read-only in the background")
    p.add_argument("--inventory", required=True)
    _add_target_args(p, ssh=True)
    p.add_argument("--secret-dir", help="local dir mapping vault paths to files (lab use)")
    p.add_argument("--docs-lib", default=None,
                   help="RAG document library managed by 'harness docs'")
    p.add_argument("--docs-dir", default=None,
                   help="directory of architecture PDFs used for RAG "
                       "(legacy ad-hoc; prefer --docs-lib)")
    p.add_argument("--parts-csv", help="parts list CSV (columns: slot,fru,pn,sn)")
    p.add_argument("--ask-parts", action="store_true",
                   help="interactive: when a rail fault is decoded, prompt for the "
                        "operator's per-slot instance parts and store them for future runs")
    p.add_argument("--parts-dir", default="config/parts",
                   help="where answered instance parts are persisted, one file per target")
    p.add_argument("--console", action="store_true",
                   help="run probes over the serial console (lab/qa only)")
    p.add_argument("--out-dir", default="harness_runs",
                   help="where each run writes diagnosis.json / trace.json / dumps")
    p.add_argument("--session-dir", default="harness_runs/sessions",
                   help="where chat sessions are saved (see /history, /resume)")
    p.add_argument("--resume", default=None,
                   help="resume a saved session directory at startup")
    p.add_argument("--max-turns", type=int, default=6,
                   help="agent turn budget per diagnosis run (default: 6)")
    p.add_argument("--server-number", type=int, default=None,
                   help="server number for the vendor FAT single-test menu "
                        "(GB targets, SSH targets only)")
    _add_llm_args(p)
    p.add_argument("--llm-model", default=None,
                   help="specific LLM model for conversation + reasoning, e.g. "
                        "gemini/gemini-2.5-pro, local/Qwen2.5-7B-Instruct or "
                        "gpt-4o; overrides --llm / inventory / the remembered "
                        "model (config/models.yaml)")
    p.set_defaults(func=run_session)

    p = sub.add_parser("secrets",
                       help="NON-AGENT credential CLI: register SSH keys / passwords "
                            "into the secret store (never through the prompt)")
    from .secrets_cli import build_secrets_parser as _build_secrets_parser
    _build_secrets_parser(p)
    p.set_defaults(func=_secrets_entry)

    p = sub.add_parser(
        "setup",
        help="one-time interactive wizard: LLM API key, SSH key (+generate), "
             "BMC creds, inventory, vault-path verification")
    p.add_argument("--inventory", default=None,
                   help="existing inventory (default: discover / create inventory.yaml)")
    p.add_argument("--secret-dir", default=None,
                   help="file-backed secret store dir (default: secrets)")
    p.set_defaults(func=run_setup_cmd)

    p = sub.add_parser("targets", help="manage short target aliases (path-only file)")
    p.add_argument("--targets-file", default="config/targets.yaml",
                   help="target aliases file (default: config/targets.yaml)")
    t_sub = p.add_subparsers(dest="target_action", required=True)
    t_add = t_sub.add_parser("add", help="add an alias (rack/cable and/or address)")
    t_add.add_argument("alias")
    t_add.add_argument("--rack", default=None)
    t_add.add_argument("--cable", default=None)
    t_add.add_argument("--address", default=None)
    t_add.add_argument("--model", default=None,
                       help="canonical model key (fallback when detection fails)")
    t_sub.add_parser("ls", help="list aliases")
    t_rm = t_sub.add_parser("rm", help="remove an alias")
    t_rm.add_argument("alias")
    p.set_defaults(func=run_targets)

    p = sub.add_parser(
        "calibrate",
        help="rebuild per-LLM fix-rate calibration from verified case outcomes")
    p.add_argument("--cases", default="harness_runs/cases",
                   help="case store directory (default: harness_runs/cases)")
    p.add_argument("--llm", default=None,
                   help="ident to calibrate (default: every ident present in the store)")
    p.add_argument("--out", default="harness_runs/calibration",
                   help="calibration store directory (default: harness_runs/calibration)")
    p.set_defaults(func=run_calibrate)

    p = sub.add_parser(
        "llm", help="LLM endpoint utilities: preflight reachability / tunnel probe")
    llm_sub = p.add_subparsers(dest="llm_action", required=True)
    c = llm_sub.add_parser(
        "check",
        help="staged connectivity probe: ssh -> forward -> GET /models")
    c.add_argument("--url", default=None,
                   help="OpenAI-compatible base URL (default: HARNESS_LLM_URL "
                        "or http://127.0.0.1:8000/v1)")
    c.add_argument("--tunnel", default=None, metavar="HOST:PORT",
                   help="probe a forward through the inventory's rack manager "
                        "(needs --inventory); reports which stage fails and "
                        "the fallback when forwarding is refused")
    c.add_argument("--inventory", default=None,
                   help="inventory whose llm_console/console_defaults supplies "
                        "the tunnel hop")
    c.add_argument("--rack", default=None,
                   help="rack id for per-rack manager addressing "
                        "(llm_console.rack_addresses)")
    c.add_argument("--model", default=None,
                   help="expected model id; warns when not served")
    c.add_argument("--timeout", type=float, default=10.0)
    c.add_argument("--secret-dir", default=None,
                   help="local dir mapping vault paths to files (lab use)")
    c.set_defaults(func=run_llm_check)
    c = llm_sub.add_parser(
        "discover",
        help="find the vLLM endpoint on the rack/cable target "
             "(read-only console probes)")
    c.add_argument("--inventory", required=True)
    _add_target_args(c, ssh=False)
    c.add_argument("--secret-dir", default=None,
                   help="local dir mapping vault paths to files (lab use)")
    c.set_defaults(func=run_llm_discover)
    c = llm_sub.add_parser(
        "pin-host",
        help="pin a per-rack manager's host key via the bastion "
             "(ssh-keyscan pattern)")
    c.add_argument("--inventory", required=True)
    c.add_argument("--rack", required=True)
    c.add_argument("--cable", required=True)
    c.add_argument("--secret-dir", default=None,
                   help="local dir mapping vault paths to files (lab use)")
    c.set_defaults(func=run_llm_pin_host)

    p = sub.add_parser(
        "report",
        help="record a diagnosis run's verified outcome (learning-loop ground truth)")
    p.add_argument("--run", required=True, help="harness run directory id")
    p.add_argument("--outcome",
                   choices=("fixed", "partial", "not_fixed", "inconclusive"),
                   help="verified outcome after the repair attempt")
    p.add_argument("--taken", action="append", default=[],
                   help="action that was actually taken (repeatable)")
    p.add_argument("--verdict", default=None,
                   help="verification verdict, e.g. resolved / no_change")
    p.add_argument("--cases", default="harness_runs/cases",
                   help="case store directory (default: harness_runs/cases)")
    p.add_argument("--out-dir", default="harness_runs",
                   help="run directory root (default: harness_runs)")
    p.add_argument("--status", action="store_true",
                   help="print the recorded case for the run, if any")
    p.add_argument("--revise", action="store_true",
                   help="replace an existing record (operator correction; "
                        "audited as case_revised)")
    p.set_defaults(func=run_report)

    p = sub.add_parser(
        "label",
        help="label a run with the correct fix + outcome (interactive; "
             "learning-loop ground truth)")
    p.add_argument("--run", required=True,
                   help="harness run directory id (or a path to the run dir)")
    p.add_argument("--outcome",
                   choices=("fixed", "partial", "not_fixed", "inconclusive"),
                   help="verified outcome (omit to prompt interactively)")
    p.add_argument("--taken", action="append", default=[],
                   help="the correct fix / action actually taken (repeatable; "
                        "omit to prompt with the top recommendation prefilled)")
    p.add_argument("--verdict", default=None,
                   help="verification verdict, e.g. resolved / no_change")
    p.add_argument("--cases", default="harness_runs/cases",
                   help="case store directory (default: harness_runs/cases)")
    p.add_argument("--out-dir", default="harness_runs",
                   help="run directory root (default: harness_runs)")
    p.add_argument("--revise", action="store_true",
                   help="replace an existing record (operator correction; "
                        "audited as case_revised)")
    p.add_argument("--status", action="store_true",
                   help="print the recorded case for the run, if any")
    p.set_defaults(func=run_label)

    p = sub.add_parser(
        "priors",
        help="build outcome-fed subsystem priors from verified case outcomes")
    priors_sub = p.add_subparsers(dest="priors_action", required=True)
    priors_update = priors_sub.add_parser(
        "update", help="rebuild priors.json from the case store")
    priors_update.add_argument("--cases", default="harness_runs/cases",
                               help="case store directory (default: harness_runs/cases)")
    priors_update.add_argument("--out", default="priors.json",
                               help="prior file path (default: priors.json)")
    priors_update.add_argument("--min-verified", type=int, default=10,
                               help="minimum outcome-recorded cases before priors activate")
    priors_update.set_defaults(func=run_priors_update)

    p = sub.add_parser(
        "eval",
        help="offline regression: replay held-out verified cases through the CURRENT pipeline")
    p.add_argument("--cases", default="harness_runs/cases",
                   help="case store directory (default: harness_runs/cases)")
    p.add_argument("--lib", default=None,
                   help="doc library directory (default: harness_docs when present)")
    p.add_argument("--llm", choices=("stub", "openai", "local", "gemini"), default="stub",
                   help="LLM backend for the replay (default: stub)")
    p.add_argument("--llm-url", default=None,
                   help="OpenAI-compatible base URL for openai/local replay "
                        "(default: HARNESS_LLM_URL)")
    p.add_argument("--holdout-frac", type=float, default=0.25,
                   help="deterministic holdout fraction (default: 0.25)")
    p.add_argument("--out", default="eval_report.json",
                   help="report output path (default: eval_report.json)")
    p.add_argument("--update-baseline", action="store_true",
                   help="rewrite baseline.json to the current numbers")
    p.add_argument("--tolerance", type=float, default=0.05,
                   help="max absolute verdict-accuracy/ECE regression (default: 0.05)")
    p.set_defaults(func=run_eval)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command is None:
        return run_menu(args)
    try:
        result = args.func(args)
        return result if isinstance(result, int) else 0
    except SerialConsoleError as exc:
        print(f"console error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - surface any pipeline failure cleanly
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
