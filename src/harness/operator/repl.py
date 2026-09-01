"""Interactive session REPL: an agent chat over read-only debugging tools.

``harness session`` starts a Claude-Code-style REPL: you describe symptoms in
plain English and a conversation agent (``operator.chat_agent``) explains what
it is about to do, then calls the harness's read-only tools (diagnose / probe /
docs / verify) *in the background* while you keep typing. Tool results feed
back into the conversation, so the agent can chain steps and answer follow-up
questions grounded in the evidence it gathered.

Design:

- ``BackgroundTask`` runs the whole agent turn (decide -> say -> tool ->
  observe -> decide ...) in a daemon thread. Every worker print is captured and
  streamed back through an event queue, so the terminal stays under the REPL's
  control. The agent's LLM call is itself inside the task: a slow model never
  freezes the prompt, and the agent's own progress and statements stream in as
  they happen (use /status on demand to see what is running).
- ``LineReader`` (in ``operator.menu``) puts the tty in raw mode (termios on
  POSIX, msvcrt on Windows) so the REPL can redraw the in-progress input line
  around background output. When stdin is not a tty it falls back to blocking
  ``input()``.
- Messages typed while a diagnosis is running are queued and seeded as context
  (``--context``) into the next run -- the agent reads them once it winds down.
- Session-mode agent questions block the worker on an answer event; the REPL
  delivers the next typed line to the agent.

Every turn and every run is persisted to ``--session-dir`` as ``session.json``
(host, transcript, run dirs) so a session can be resumed with ``/resume``.
"""

from __future__ import annotations

import contextlib
import io
import json
import queue
import shlex
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TextIO

from ..config.inventory_lint import load_inventory
from ..config.vault import SecretStore
from ..diagnosis.schema import Diagnosis
from ..inspect.base import RegisterDump
from ..operator.supervisor import Escalation
from ..targets import TargetError, TargetSpec, resolve_target
from . import ui
from .chat_agent import ChatTurn, build_evidence, build_messages, decide, fallback_turn
from .credential_gate import apply_ssh_context
from .menu import LineReader as _LineReader
from .router import _keyword_route


@dataclass
class Session:
    inv_path: str
    inv: object
    host: object
    store: SecretStore
    out_dir: Path
    session_dir: Path
    llm: object
    router_llm: object
    llm_mode: str
    docs_lib: str | None
    docs_dir: str | None
    parts_csv: str | None
    secret_dir: str | None
    console: bool
    max_turns: int
    server_number: int | None = None
    llm_ident: str = ""
    llm_url: str | None = None
    llm_tunnel: str | None = None
    ask_parts: bool = False
    overrides: dict = field(default_factory=dict)
    target: TargetSpec = field(default_factory=TargetSpec)
    target_label: str = ""
    targets_file: str | None = None
    ssh_user: str = "diagbot"
    identity_vault_path: str | None = None
    known_hosts_path: str = "config/known_hosts"
    transcript: list[dict] = field(default_factory=list)
    evidence: str = ""       # digest of the latest diagnosis (chat grounding)
    pending: list[str] = field(default_factory=list)
    test_logs: list[str] = field(default_factory=list)  # queued --test-log paths
    runs: list[Path] = field(default_factory=list)
    last_run: Path | None = None
    reader: object | None = None
    task: BackgroundTask | None = None
    answer_value: str = ""
    answer_ready: threading.Event = field(default_factory=threading.Event)
    awaiting: bool = False
    answered: bool = False
    cred_pending: str | None = None
    cred_answer: bytes | None = None
    cred_event: threading.Event = field(default_factory=threading.Event)
    quit: bool = False


# ---- background task + captured output ----

class _Capture(TextIO):
    """Redirects *worker* stdout/stderr into the event queue, line-buffered.

    ``sys.stdout`` is process-global, so the REPL's own prints would be
    swallowed (and re-captured into the queue, feeding back forever) unless the
    capture is owner-thread-aware: only the thread that created the capture is
    redirected; everyone else passes through to the real stream.
    """

    def __init__(self, emit: Callable[[str], None], real: TextIO) -> None:
        self._emit = emit
        self._real = real
        self._owner = threading.get_ident()
        self._buf = ""

    def write(self, s: str) -> int:
        if threading.get_ident() != self._owner:
            return self._real.write(s)
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._emit(line)
        return len(s)

    def flush(self) -> None:
        self._real.flush()

    def fileno(self) -> int:
        """Delegate to the real stream.

        ``typing.TextIO.fileno()`` (the inherited no-op) returns ``None``, which
        breaks stdlib callers that pass the fd on -- argparse's help color
        detection calls ``os.isatty(stream.fileno())`` on every worker-thread
        ``build_parser()`` (the captured stdout is process-global), and
        ``os.isatty(None)`` raises ``TypeError: 'NoneType' object cannot be
        interpreted as an integer``. Delegating gives the real fd (or raises
        ``io.UnsupportedOperation`` for fd-less streams, which the caller's
        ``OSError`` fallback handles via ``isatty()``).
        """
        return self._real.fileno()

    def isatty(self) -> bool:
        return self._real.isatty()

    @property
    def encoding(self) -> str:
        return "utf-8"


@contextlib.contextmanager
def _redirect_stdout(emit: Callable[[str], None]):
    old, sys.stdout = sys.stdout, _Capture(emit, sys.stdout)
    try:
        yield
    finally:
        sys.stdout = old


@contextlib.contextmanager
def _redirect_stderr(emit: Callable[[str], None]):
    old, sys.stderr = sys.stderr, _Capture(emit, sys.stderr)
    try:
        yield
    finally:
        sys.stderr = old


@dataclass
class BackgroundTask:
    label: str
    fn: Callable[[Callable[[str], None], threading.Event], object]
    cancel: threading.Event = field(default_factory=threading.Event)
    events: queue.Queue = field(default_factory=queue.Queue)
    started: float = field(default_factory=time.monotonic)
    _thread: threading.Thread | None = field(default=None, init=False)

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name=f"harness-{self.label}", daemon=True)
        self._thread.start()

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def _run(self) -> None:
        def progress(text: str) -> None:
            self.events.put(("progress", text))

        try:
            with _redirect_stdout(lambda l: self.events.put(("out", l))), \
                 _redirect_stderr(lambda l: self.events.put(("err", l))):
                result = self.fn(progress, self.cancel)
            self.events.put(("done", result))
        except Escalation as exc:
            self.events.put(("error", f"cancelled: {exc}"))
        except KeyboardInterrupt:
            self.events.put(("error", "interrupted"))
        except Exception as exc:  # noqa: BLE001 - surfaced as an error event
            import os
            if os.environ.get("REPL_DEBUG"):
                import traceback as _tb
                with open(os.environ["REPL_DEBUG"], "a", encoding="utf-8") as _f:
                    _tb.print_exc(file=_f)
            self.events.put(("error", f"{type(exc).__name__}: {exc}"))


# ---- task factories ----

def _target_argv(session: Session) -> list[str]:
    """CLI targeting args for the active target (zero-YAML targeting in session)."""
    argv: list[str] = []
    t = session.target
    if t.name is not None:
        argv += ["--host", t.name]
    elif t.rack is not None and t.cable is not None:
        argv += ["--rack", t.rack, "--cable", t.cable]
    elif t.ip is not None:
        argv += ["--address", t.ip]
    elif t.alias is not None:
        argv += ["--target", t.alias]
    elif session.host is not None:
        argv += ["--host", session.host.name]  # fallback: first-inventory default
    if session.targets_file:
        argv += ["--targets-file", session.targets_file]
    return argv


def _set_target(session: Session, spec: TargetSpec) -> bool:
    """Resolve a runtime spec into the session's active target. Prints errors
    (e.g. missing console_defaults / unregistered SSH identity) and returns
    False on failure, so the REPL never launches a doomed run."""
    try:
        target = resolve_target(
            spec, session.inv, session.store,
            targets_path=session.targets_file,
            ssh_user=session.ssh_user,
            identity_vault_path=session.identity_vault_path,
            known_hosts_path=session.known_hosts_path,
        )
    except TargetError as exc:
        _print_line(session, ui.bad(f"  x target: {exc}"))
        return False
    apply_ssh_context(session.store, target, ssh_user=session.ssh_user)
    session.host = target.host
    session.target = spec
    session.target_label = target.label
    _print_line(session, ui.good(f"  active target: {target.label} ({target.kind})"))
    return True


def _diagnose_argv(session: Session, symptom: str, host_name: str | None) -> list[str]:
    """Build the ``diagnose`` argv for one run, seeding queued context."""
    argv = ["diagnose", "--inventory", session.inv_path,
            "--symptom", symptom,
            "--out-dir", str(session.out_dir),
            "--llm", session.llm_mode,
            "--max-turns", str(session.max_turns)]
    if session.llm_ident:
        argv += ["--llm-model", session.llm_ident]
    if session.llm_url:
        argv += ["--llm-url", session.llm_url]
    if session.llm_tunnel:
        # The spawned run owns its own short-lived forward (a paramiko
        # transport cannot cross the process boundary).
        argv += ["--llm-tunnel", session.llm_tunnel]
    argv += _target_argv(session)
    if session.secret_dir:
        argv += ["--secret-dir", session.secret_dir]
    if session.docs_lib:
        argv += ["--docs-lib", session.docs_lib]
    if session.docs_dir:
        argv += ["--docs-dir", session.docs_dir]
    if session.parts_csv:
        argv += ["--parts-csv", session.parts_csv]
    if session.ask_parts:
        argv.append("--ask-parts")
    if session.console:
        argv.append("--console")
    if session.server_number:
        argv += ["--server-number", str(session.server_number)]
    for context_line in session.pending:
        argv += ["--context", context_line]
    session.pending = []
    for test_log in session.test_logs:
        argv += ["--test-log", test_log]
    session.test_logs = []
    return argv


def _make_answer_fn(session: Session) -> Callable[[str], str]:
    """Block the worker until the REPL delivers the operator's typed answer."""

    def answer(_question: str) -> str:
        session.awaiting = True
        session.answered = False
        session.answer_ready.clear()
        while not session.answer_ready.wait(0.1):
            if session.task is not None and session.task.cancel.is_set():
                session.awaiting = False
                raise Escalation("run cancelled by operator")
        session.awaiting = False
        return session.answer_value

    return answer


# ---- agent tools (called inline by the conversation agent's loop) ----

def _apply_turn_target(session: Session, turn: ChatTurn) -> bool:
    """If the agent's turn names a target, switch the active target first.
    Returns False when the named target could not be resolved (the error was
    already printed), so the agent does not launch a doomed run."""
    spec = None
    if turn.host is not None and turn.host in session.inv.host_names:
        spec = TargetSpec(name=turn.host)
    elif turn.ip is not None:
        spec = TargetSpec(ip=turn.ip)
    elif turn.rack is not None and turn.cable is not None:
        spec = TargetSpec(rack=turn.rack, cable=turn.cable)
    elif turn.alias is not None:
        spec = TargetSpec(alias=turn.alias)
    if spec is not None and spec != session.target:
        return _set_target(session, spec)
    return True


def _tool_diagnose(session: Session, turn: ChatTurn,
                   progress: Callable[[str], None],
                   cancel: threading.Event):
    """Run the full read-only diagnose pipeline as one agent tool call."""
    if not _apply_turn_target(session, turn):
        return None  # unresolved target: error already printed
    argv = _diagnose_argv(session, turn.symptom or "", turn.host)
    from ..operator.cli import build_parser, run_diagnose
    args = build_parser().parse_args(argv)
    return run_diagnose(args, overrides={
        **session.overrides,
        "store": session.store,
        "llm": session.llm,
        "progress": progress,
        "cancel_event": cancel,
        "human_input": _make_answer_fn(session),
    })


def _tool_verify(session: Session, turn: ChatTurn,
                 progress: Callable[[str], None]) -> str:
    """Re-collect a metric and compare against a baseline run; the verdict
    lines are streamed via ``progress`` AND returned so the agent reasons
    over them on its next decision."""
    baseline = turn.baseline
    if not baseline and session.last_run is not None:
        dumps = session.last_run / "dumps.json"
        if dumps.exists():
            baseline = str(dumps)
    if not baseline:
        return "(no baseline yet: run a diagnosis first, or pass a baseline path)"
    argv = ["verify", "--inventory", session.inv_path,
            "--symptom", "(session verify)",
            "--baseline", baseline]
    argv += _target_argv(session)
    if session.secret_dir:
        argv += ["--secret-dir", session.secret_dir]
    if turn.metric:
        argv += ["--metric", turn.metric]

    from ..operator.cli import build_parser, run_verify

    def run():
        args = build_parser().parse_args(argv)
        return run_verify(args, overrides={**session.overrides,
                                           "store": session.store})

    # Tee the verdict output: lines stream to the operator via progress while
    # being collected for the agent's next decision. Thread-owner-aware like
    # the task-level capture, so the REPL's own prints always pass through.
    lines: list[str] = []
    tee = _Capture(lambda l: (lines.append(l), progress(l)), sys.stdout)
    old, sys.stdout = sys.stdout, tee
    try:
        run()
    finally:
        sys.stdout = old
    return "\n".join(lines) or "(verify produced no output)"


def _tool_docs(session: Session, turn: ChatTurn) -> str:
    if not session.docs_lib and not session.docs_dir:
        return ("(no doc library configured; set --docs-lib / --docs-dir or "
                "use `harness docs add`)")
    from types import SimpleNamespace

    from ..operator.cli import _build_docs_retriever
    retriever, _lib = _build_docs_retriever(SimpleNamespace(
        docs_lib=session.docs_lib, docs_dir=session.docs_dir))
    if retriever is None:
        return "(no doc library; use `harness docs add` to upload PDFs)"
    lines = retriever(turn.query or "", None)
    if not lines:
        return f"(no snippets matched {turn.query!r})"
    return "\n".join(f"- {line}" for line in lines)


def _tool_run(session: Session, turn: ChatTurn) -> str:
    """Load a PAST run's recorded diagnosis + evidence digest (local, read-only).

    Resolves the reference against the session's run root and its own run
    list, then rebuilds the evidence digest so follow-ups are grounded in that
    run. Returns a summary line for the transcript; unresolved or unreadable
    runs come back as an observation string, not an exception.
    """
    ref = (turn.run or "").strip().rstrip("/\\")
    if not ref:
        return "(run reference missing)"
    candidates = [Path(ref)]
    candidates.append(session.out_dir / ref)
    candidates.append(session.out_dir / ref.split("/")[-1].split("\\")[-1])
    run_dir = None
    for cand in candidates:
        if cand.is_dir():
            run_dir = cand
            break
    if run_dir is None:
        for run in session.runs:
            if run.name == ref or str(run).endswith(ref):
                run_dir = run
                break
    if run_dir is None:
        return (f"(run {ref!r} not found under {session.out_dir}; /runs lists "
                "the runs this session knows about)")
    diagnosis_path = run_dir / "diagnosis.json"
    if not diagnosis_path.exists():
        return (f"(run {run_dir.name} exists but has no recorded diagnosis.json; "
                "it may have failed before scoring -- /runs shows the session's runs)")
    try:
        diag = Diagnosis.model_validate(json.loads(
            diagnosis_path.read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        return f"(could not read {diagnosis_path}: {exc})"
    session.last_run = run_dir
    if run_dir not in session.runs:
        session.runs.append(run_dir)
    session.evidence = build_evidence(diag, str(run_dir))
    summary = (f"loaded run {run_dir.name}: state="
               f"{getattr(diag.state, 'value', diag.state)} "
               f"confidence={diag.confidence:.2f} -- {diag.diagnosis}")
    session.transcript.append(
        {"role": "agent", "kind": "diagnosis",
         "content": f"{diag.diagnosis} (confidence {diag.confidence:.2f})"})
    return summary


def _render_decode(d) -> str:
    """One line for a decoded register: ``MNEMONIC = 0x..  [field=value ...]``."""
    fields = ", ".join(
        f"{f.name}={f.raw_value}" + (f"({f.meaning})" if f.meaning else "")
        for f in getattr(d, "decoded_fields", []))
    line = f"{d.mnemonic} = {d.raw_hex}"
    if fields:
        line += f"  [{fields}]"
    if getattr(d, "unknown", False):
        line += "  (unknown register)"
    return line


def _run_probes(session: Session, subsystems: list[str], doc_topics: list[str],
                progress: Callable[[str], None],
                cancel: threading.Event) -> str:
    """Collect + decode read-only probes for the active target; no LLM turn.

    Mirrors the session engine's probe mapping (subsystem -> PROFILE_COLLECTORS,
    doc topics -> doc-named probe commands) but stands alone: the operator asked
    for evidence, so this runs the collectors and reports what came back.
    """
    from types import SimpleNamespace

    from ..diagnosis.engine import decode_dumps, prebatch_console_plan
    from ..engine.allowlist import default_policy
    from ..engine.bmc import BmcRunner
    from ..engine.redfish import RedfishClient, RedfishError
    from ..engine.session import SSHSession
    from ..engine.sol import ConsoleRunner, SerialConsole
    from ..inspect.collectors.bmc_console import BmcConsoleCollector
    from ..inspect.collectors.doc_guided import DocGuidedProbeCollector
    from ..inspect.collectors.ipmi import IpmiCollector
    from ..inspect.collectors.redfish import RedfishCollector
    from ..inspect.decoder import Decoder
    from ..inspect.model import PROFILE_COLLECTORS
    from ..inspect.registry import make_collector
    from ..operator.cli import _build_docs_retriever
    from ..plan.doc_guided import mine_probe_commands

    if not subsystems and not doc_topics:
        return "nothing to probe (say which subsystem, e.g. 'probe memory')"

    try:
        target = resolve_target(
            session.target, session.inv, session.store,
            targets_path=session.targets_file,
            ssh_user=session.ssh_user,
            identity_vault_path=session.identity_vault_path,
            known_hosts_path=session.known_hosts_path,
        )
    except TargetError as exc:
        return f"probe failed: {exc}"
    host = target.host

    use_console = session.console or target.kind == "console"
    if use_console and host.console is None:
        return f"probe failed: target {target.label!r} has no console block"

    runner = None
    owned_session = None
    try:
        if use_console:
            runner = session.overrides.get("console_runner")
            if runner is None:
                runner = ConsoleRunner(SerialConsole(host.console, session.store))
        else:
            runner = session.overrides.get("session")
            if runner is None:
                owned_session = SSHSession(host, default_policy(), session.store)
                owned_session.open()
                runner = owned_session

        bmc_runner = None
        if host.bmc is not None and host.bmc.password_vault_path:
            try:
                bmc_password = session.store.get(
                    host.bmc.password_vault_path).decode()
            except KeyError:
                bmc_password = None
            if bmc_password:
                bmc_runner = BmcRunner(host.bmc.address, host.bmc.username,
                                       bmc_password)

        redfish_collector = None
        if use_console and host.console.redfish_password_vault_path is not None:
            try:
                redfish_collector = RedfishCollector(
                    RedfishClient(host.console, host.console.cable,
                                  session.store))
            except RedfishError as exc:
                progress(f"redfish unavailable: {exc}")

        def collector_factory(name, _runner):
            if name == "redfish":
                return redfish_collector
            if _runner.is_console:
                if name in ("pcie", "storage"):
                    return None
                if name in ("cpu_msr", "kernel", "ipmi"):
                    return BmcConsoleCollector(_runner, subsystem={
                        "cpu_msr": "cpu", "kernel": "kernel", "ipmi": "ipmi",
                    }[name])
            if name == "ipmi":
                return IpmiCollector(bmc_runner) if bmc_runner is not None else None
            return make_collector(name, _runner)

        wanted: list[str] = []
        for subsystem in subsystems:
            if subsystem not in PROFILE_COLLECTORS:
                continue
            for name in PROFILE_COLLECTORS[subsystem]:
                if name not in wanted:
                    wanted.append(name)

        lines: list[str] = []
        doc_commands: list[str] = []
        if doc_topics:
            retriever, _lib = _build_docs_retriever(SimpleNamespace(
                docs_lib=session.docs_lib, docs_dir=session.docs_dir))
            if retriever is not None:
                for topic in doc_topics:
                    snippets = retriever(topic, None)
                    doc_commands.extend(mine_probe_commands(snippets))
            else:
                lines.append("doc topic(s) skipped: no doc library configured")

        if use_console and hasattr(runner, "batch_execute"):
            # one serial console session for the whole probe set
            prebatch_console_plan(
                SimpleNamespace(runner=runner, progress=progress),
                SimpleNamespace(doc_probes=doc_commands),
                [(name, collector_factory(name, runner)) for name in wanted],
            )

        dumps: dict[str, list[RegisterDump]] = {}
        for subsystem in subsystems:
            if subsystem not in PROFILE_COLLECTORS:
                lines.append(f"unknown subsystem {subsystem!r} (skipped)")
                continue
            if cancel.is_set():
                return "\n".join(lines + ["cancelled by operator"])
            names = PROFILE_COLLECTORS[subsystem]
            progress(f"probe {subsystem}: {', '.join(names)}")
            for name in names:
                if name in dumps:
                    continue
                collector = collector_factory(name, runner)
                if collector is None:
                    lines.append(f"{subsystem}/{name}: unavailable (skipped)")
                    continue
                collected = collector.collect()
                dumps[name] = collected
                ok = sum(1 for d in collected if d.ok)
                lines.append(f"{subsystem}/{name}: {len(collected)} dump(s), {ok} ok")

        if doc_commands:
            progress(f"probe doc-named: {', '.join(doc_commands)}")
            dumps["doc_guided"] = DocGuidedProbeCollector(
                runner, doc_commands).collect()
            ok = sum(1 for d in dumps["doc_guided"] if d.ok)
            lines.append(f"doc-guided: {len(dumps['doc_guided'])} dump(s), {ok} ok")

        if redfish_collector is not None:
            progress("probe redfish: event_log, service_conditions")
            dumps["redfish"] = redfish_collector.collect()
            ok = sum(1 for d in dumps["redfish"] if d.ok)
            lines.append(f"redfish: {len(dumps['redfish'])} dump(s), {ok} ok")

        decoded = decode_dumps(Decoder(),
                               [d for ds in dumps.values() for d in ds])
        if decoded:
            lines.append(f"decoded registers ({len(decoded)}):")
            lines += [f"  {_render_decode(d)}" for d in decoded]
        elif not lines:
            lines.append("no probes ran")
        return "\n".join(lines)
    finally:
        if owned_session is not None:
            owned_session.close()


# ---- rendering ----

_ERROR_PREFIXES = ("x ", "no such file", "unknown command", "unknown host",
                   "bad arguments")


def _auto_style(text: str) -> str:
    """Give unstyled REPL feedback a consistent look by convention: `x ...`
    errors go red, usage/help hints go dim. Already-styled lines (they carry
    an ESC sequence) pass through untouched -- never double-wrap."""
    if "\x1b[" in text:
        return text
    stripped = text.strip()
    if any(stripped.startswith(p) for p in _ERROR_PREFIXES):
        return ui.bad(text)
    if stripped.startswith(("usage:", "(")):
        return ui.dim(text)
    return text


def _print_line(session: Session, text: str) -> None:
    reader = session.reader
    clear = getattr(reader, "clear_line", None)
    if clear is not None:
        clear()
    print(_auto_style(text), flush=True)
    refresh = getattr(reader, "refresh_line", None)
    if refresh is None:
        refresh = getattr(reader, "redraw", None)
    if refresh is not None:
        refresh()


def _render_event(session: Session, kind: str, payload: object) -> None:
    """Render one worker event, styled by KIND so the stream reads at a
    glance: accent = the agent speaking / launching, dim bullet = progress
    ticks, red = stderr and failures, plain = captured tool output."""
    if kind == "progress":
        text = str(payload)
        if text.startswith("agent: "):
            _print_line(session, ui.accent(text))
        elif text.startswith("thinking:"):
            _print_line(session, ui.dim(text))
        elif "failed" in text:
            _print_line(session, ui.bad(f"  {text}"))
        elif text.startswith("started in the background:"):
            _print_line(session, ui.accent(f"  {text}"))
        else:
            _print_line(session, ui.dim(f"  {ui.GLYPH_BULLET} {text}"))
    elif kind == "out":
        for line in str(payload).splitlines():
            _print_line(session, line)
    elif kind == "err":
        for line in str(payload).splitlines():
            _print_line(session, ui.bad(f"  [stderr] {line}"))


def _drain_events(session: Session) -> tuple[object | None, str | None]:
    """Pull all pending worker events; returns (result, error) terminators."""
    task = session.task
    if task is None:
        return None, None
    result: object | None = None
    error: str | None = None
    while True:
        try:
            kind, payload = task.events.get_nowait()
        except queue.Empty:
            break
        if kind == "done":
            result = payload
        elif kind == "error":
            error = payload
        else:
            _render_event(session, kind, payload)
    return result, error


def _credential_bridge(session: Session) -> Callable[[str], bytes | None]:
    """Bridge from the worker thread into the REPL main loop: the worker sets
    ``cred_pending`` and waits; the loop prompts on the main thread and fires
    ``cred_event`` to release the worker (see ``_handle_credential``)."""
    def bridge(vault_path: str) -> bytes | None:
        session.cred_pending = vault_path
        session.cred_event.clear()
        session.cred_event.wait()
        session.cred_pending = None
        return session.cred_answer
    return bridge


def _handle_credential(session: Session) -> None:
    """Prompt the operator on the MAIN thread for a credential the worker needs
    (see ``OnDemandSecretStore`` / ``CredentialPrompter`` in credential_gate).
    The answer bytes travel through the event bridge -- never the LLM."""
    vault_path = session.cred_pending
    if vault_path is None:
        return
    prompter = getattr(session.store, "prompter", None)
    if prompter is None:
        session.cred_answer = None
        session.cred_event.set()
        return
    _print_line(session, f"  credential needed: {vault_path}")
    try:
        session.cred_answer = prompter.prompt_now(vault_path)
    except Exception:  # noqa: BLE001 - declined credential stays None, never crash the REPL
        session.cred_answer = None
    finally:
        session.cred_event.set()


def _find_newest_run(out_dir: Path, session_dir: Path) -> Path | None:
    try:
        dirs = [p for p in out_dir.iterdir()
                if p.is_dir() and p != session_dir and (p / "diagnosis.json").exists()]
    except OSError:
        return None
    return max(dirs, key=lambda p: p.stat().st_mtime) if dirs else None


def _finish_task(session: Session, result: object | None,
                 error: str | None) -> None:
    task = session.task
    session.task = None
    if error:
        _print_line(session, ui.bad(f"  x {task.label}: {error}"))
        session.transcript.append({"role": "tool", "kind": "error", "content": error})
        return
    if result is not None:
        _print_line(session, ui.good(f"  + done: {result}"))
        # The final say / tool result was already recorded in the transcript
        # when it was produced; only record genuinely new content.
        last = session.transcript[-1] if session.transcript else {}
        if not (last.get("kind") in ("say", "result")
                and last.get("content") == str(result)):
            session.transcript.append(
                {"role": "tool", "kind": "result", "content": str(result)})
    if session.pending:
        _print_line(session, f"  {len(session.pending)} queued message(s) seed the next run")


# ---- line handling ----

def _status_line(session: Session) -> str:
    if session.task is not None and session.task.alive:
        return ui.accent(
            f"  running in the background: {session.task.label} "
            f"({session.task.elapsed():.0f}s) -- you can keep typing; "
            f"/stop cancels")
    if session.pending:
        return ui.dim(f"  idle; {len(session.pending)} queued message(s) for the next run")
    if session.last_run is not None:
        return ui.dim(f"  idle; last run: {session.last_run}")
    return ui.dim("  idle")


# ---- the conversation agent ----

_AGENT_MAX_TOOLS = 4  # tool calls per operator message (the loop then stops)


def _say(session: Session, progress: Callable[[str], None], text: str) -> None:
    """Show the agent's reasoning and record it (always BEFORE a tool runs)."""
    progress(f"agent: {text}")
    session.transcript.append({"role": "agent", "kind": "say", "content": text})


def _action_text(turn: ChatTurn) -> str:
    if turn.tool == "diagnose":
        return f"diagnose: {turn.symptom}"
    if turn.tool == "probe":
        return (f"probe: {', '.join(turn.subsystems)}"
                + (f" docs: {', '.join(turn.doc_topics)}" if turn.doc_topics else ""))
    if turn.tool == "docs":
        return f"docs: {turn.query}"
    if turn.tool == "verify":
        return f"verify: {turn.metric}"
    if turn.tool == "run":
        return f"run: {turn.run}"
    return "none"


def _run_agent(session: Session, line: str,
               progress: Callable[[str], None],
               cancel: threading.Event):
    """One operator message -> agent decisions -> read-only tools -> answer.

    Loop: decide (LLM; keyword fallback on the first decision when the LLM is
    unavailable/garbage) -> print ``say`` -> run at most one tool -> append the
    result to the transcript -> decide again. Tool results and the evidence
    digest of any diagnosis feed back into every decision, so the agent can
    chain steps and answer follow-ups grounded in what it gathered. Returns a
    short final string for ``_finish_task`` (None when everything the operator
    needs was already printed/streamed).
    """
    context_lines = list(session.pending)  # queued notes feed THIS turn
    first = True
    tool_calls = 0
    final: str | None = None
    while True:
        if cancel.is_set():
            raise Escalation("run cancelled by operator")
        if first:
            progress("thinking: the agent is considering your message ...")
        messages = build_messages(
            transcript=session.transcript, user_text=line,
            evidence_digest=session.evidence,
            host_names=tuple(session.inv.host_names),
            target_label=session.target_label,
            pending=context_lines if first else [])
        turn = decide(session.router_llm, messages,
                      tuple(session.inv.host_names))
        if turn is None:
            if not first:
                break  # mid-chain LLM failure: stop with what we have
            cmd = _keyword_route(line)
            if cmd.intent == "status":
                progress(_status_line(session).strip())
                return None
            if cmd.intent == "reply":
                return fallback_turn(cmd, line).say
            turn = fallback_turn(cmd, line)
        first = False
        if turn.say:
            _say(session, progress, turn.say)
        if turn.tool in ("", "none"):
            return turn.say or final
        if tool_calls >= _AGENT_MAX_TOOLS:
            progress("tool-call budget reached for this message; send another "
                     "message to continue")
            return final or turn.say
        tool_calls += 1
        session.transcript.append(
            {"role": "agent", "kind": "action", "content": _action_text(turn)})
        try:
            result = _run_one_tool(session, turn, progress, cancel)
        except Escalation:
            raise  # operator /stop or a supervisor gate still aborts the turn
        except Exception as exc:  # noqa: BLE001 - tool failure is an observation
            msg = f"{turn.tool} failed: {exc}"
            progress(msg)
            session.transcript.append(
                {"role": "tool", "kind": "result", "content": msg})
            final = msg
            continue  # the agent sees the failure and decides what to do next
        if result is None:  # printed-only turn (e.g. unresolved target)
            break
        if isinstance(result, tuple):
            final, diag = result
            run_dir = _find_newest_run(session.out_dir, session.session_dir)
            if run_dir is not None:
                session.last_run = run_dir
                session.runs.append(run_dir)
            session.evidence = build_evidence(diag, str(run_dir) if run_dir else None)
        else:
            session.transcript.append(
                {"role": "tool", "kind": "result", "content": result})
            final = result
    return final


def _run_one_tool(session: Session, turn: ChatTurn,
                  progress: Callable[[str], None],
                  cancel: threading.Event):
    """Execute one agent tool call; returns the result string for the transcript
    (a ``(summary, Diagnosis)`` tuple for the diagnose tool)."""
    if turn.tool == "diagnose":
        progress(f"starting read-only diagnosis: {turn.symptom}")
        t0 = time.monotonic()
        diag = _tool_diagnose(session, turn, progress, cancel)
        if diag is None:  # unresolved target: error already printed
            return None
        progress(f"diagnosis complete ({time.monotonic() - t0:.0f}s)")
        summary = f"diagnosis complete (confidence {diag.confidence:.2f})"
        session.transcript.append({
            "role": "agent", "kind": "diagnosis",
            "content": f"{diag.diagnosis} (confidence {diag.confidence:.2f})",
        })
        return summary, diag
    if turn.tool == "probe":
        label = ", ".join(turn.subsystems + turn.doc_topics) or "nothing"
        progress(f"probing: {label}")
        return _run_probes(session, turn.subsystems, turn.doc_topics,
                           progress, cancel)
    if turn.tool == "docs":
        progress(f"docs lookup: {turn.query}")
        return _tool_docs(session, turn)
    if turn.tool == "verify":
        progress(f"verify {turn.metric} against "
                 f"{turn.baseline or 'the latest run'}")
        return _tool_verify(session, turn, progress)
    if turn.tool == "run":
        progress(f"loading run: {turn.run}")
        return _tool_run(session, turn)
    return "(no tool)"


def _agent_task(session: Session, line: str) -> BackgroundTask:
    """Background task wrapper around one full agent turn (decide + tools)."""
    label = session.target_label or (
        session.host.name if session.host is not None else "agent")

    def fn(progress: Callable[[str], None], cancel: threading.Event):
        return _run_agent(session, line, progress, cancel)

    return BackgroundTask(label=label, fn=fn)


def _handle_idle_line(session: Session, line: str) -> bool:
    """Hand one operator message to the agent; starts a background task.
    False = quit."""
    if not line.strip():
        return True
    if line.startswith("/"):
        _handle_slash(session, line)
        return not session.quit
    session.transcript.append({"role": "user", "kind": "message", "content": line})
    _save_session(session)
    # Same race-avoidance as before: the task is started before it is
    # published as ``session.task``; the launch announcement is queued on the
    # NEW task (whose events the REPL drains).
    task = _agent_task(session, line)
    task.start()
    session.task = task
    task.events.put(("progress", (
        f"started in the background: {task.label} -- keep typing "
        f"(messages queue), /status for progress, /stop to cancel")))
    return not session.quit


def _handle_busy_line(session: Session, line: str) -> None:
    """A line typed while a task runs: answer the agent or queue it."""
    if not line.strip():
        return
    if line.startswith("/"):
        _handle_slash(session, line)
        return
    if session.awaiting and not session.answered:
        session.answer_value = line
        session.answer_ready.set()
        session.answered = True
        session.transcript.append({"role": "user", "kind": "answer", "content": line})
        _print_line(session, ui.good("  (answer sent to agent)"))
        return
    session.pending.append(line)
    session.transcript.append({"role": "user", "kind": "message", "content": line})
    _print_line(session, ui.dim("  (queued; the agent sees this after the current run)"))


def _parse_use_arg(session: Session, arg: str) -> TargetSpec | None:
    """``/use`` argument -> TargetSpec: IP | rack+cable | named host | alias."""
    from .router import extract_target
    _, target = extract_target(arg)
    if target.get("ip"):
        return TargetSpec(ip=target["ip"])
    if target.get("rack") and target.get("cable"):
        return TargetSpec(rack=target["rack"], cable=target["cable"])
    if arg in session.inv.host_names:
        return TargetSpec(name=arg)
    return TargetSpec(alias=arg)  # unknown named -> alias (resolver validates)


# ---- slash commands ----

def _run_sync(session: Session, fn: Callable[[], int]) -> None:
    """Run a CLI function synchronously, capturing its output and streaming it
    through the REPL renderer so the prompt line is redrawn around it."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            code = fn()
        except SystemExit as exc:  # e.g. argparse errors inside the target CLI
            if str(exc):
                print(f"error: {exc}", file=sys.stderr)
            code = 1
        except Exception as exc:  # noqa: BLE001 - never crash the REPL
            print(f"error: {exc}", file=sys.stderr)
            code = 1
    for line in buf.getvalue().splitlines():
        _print_line(session, line)
    if code != 0:
        _print_line(session, f"  (exit {code})")


def _slash_targets(session: Session, arg: str) -> None:
    """``/targets ls | add <alias> [--rack R --cable N] [--address ip] | rm <alias>``
    -- same aliases machinery as the CLI, driven from inside the session."""
    from ..operator.cli import build_parser, run_targets

    path = session.targets_file or "config/targets.yaml"
    parts = shlex.split(arg)
    usage = ("  usage: /targets ls | add <alias> [--rack R --cable N] "
             "[--address ip] | rm <alias>")
    if not parts or parts[0] not in ("ls", "add", "rm"):
        _print_line(session, usage)
        return
    argv = ["targets", "--targets-file", path]
    if parts[0] == "add":
        if len(parts) < 2:
            _print_line(session, usage)
            return
        argv += ["add", parts[1]]
        i = 2
        while i < len(parts):
            flag, value = parts[i], parts[i + 1] if i + 1 < len(parts) else None
            if flag in ("--rack", "--cable", "--address") and value is not None:
                argv += [flag, value]
                i += 2
            else:
                _print_line(session, f"  x unknown flag {parts[i]!r}; {usage}")
                return
    elif parts[0] == "rm":
        if len(parts) != 2:
            _print_line(session, usage)
            return
        argv += ["rm", parts[1]]
    else:
        argv += ["ls"]
    try:
        args = build_parser().parse_args(argv)
    except SystemExit:
        _print_line(session, f"  x bad arguments: {arg!r}; {usage}")
        return
    _run_sync(session, lambda: run_targets(args))


def _slash_docs(session: Session, arg: str) -> None:
    """``/docs ls | add <pdf...> | rm <name> | reindex`` -- manage the RAG
    document library without leaving the session."""
    from ..docs.ingest.library import DocLibrary

    lib = DocLibrary(session.docs_lib or "harness_docs")
    parts = arg.split(None, 1)
    action = parts[0] if parts else "ls"
    rest = parts[1] if len(parts) > 1 else ""
    try:
        if action == "ls":
            entries = lib.entries()
            if not entries:
                _print_line(session, f"  (empty doc library at {lib.root!r})")
                return
            for e in entries:
                suffix = f"  ERROR: {e.error}" if e.error else ""
                _print_line(session, f"  {e.name}  {e.chunks} chunk(s){suffix}")
        elif action == "add":
            files = shlex.split(rest)
            if not files:
                _print_line(session, "  usage: /docs add <pdf> [pdf ...]")
                return
            for line in lib.add(files):
                _print_line(session, f"  {line}")
        elif action == "rm":
            if not rest:
                _print_line(session, "  usage: /docs rm <name>")
                return
            _print_line(session, f"  {lib.remove(rest)}")
        elif action == "reindex":
            for line in lib.reindex():
                _print_line(session, f"  {line}")
        else:
            _print_line(session, "  usage: /docs ls | add <pdf...> | rm <name> | reindex")
    except KeyError as exc:
        _print_line(session, f"  x {exc.args[0] if exc.args else exc}")


def _slash_lint(session: Session) -> None:
    """``/lint`` -- validate the session's inventory (no inline secrets)."""
    from ..operator.cli import build_parser, run_lint

    args = build_parser().parse_args(["lint", "--inventory", session.inv_path])
    _run_sync(session, lambda: run_lint(args))


def _set_session_model(session: Session, catalog, profile) -> None:
    """Switch the session's reasoning + routing adapters to ``profile`` and
    remember it in ``config/models.yaml`` for the next run. The profile's own
    endpoint wins; a profile tunnel opens (or reuses) a rack-manager forward
    for the in-session adapters and is kept so spawned runs open their own
    hop; otherwise a session-pinned ``--llm-url`` is merged as before. A
    failed forward aborts the switch (the previous model stays active)."""
    from dataclasses import replace as _dc_replace

    if profile.tunnel:
        if session.llm_tunnel == profile.tunnel and session.llm_url:
            effective = _dc_replace(profile, url=session.llm_url)
        else:
            from ..engine.tunnel import LLMForward, TunnelError, parse_tunnel_spec
            from .cli import _ACTIVE_LLM_FORWARDS
            from .llm_discover import llm_bastion_domain, llm_console_domain

            console = llm_console_domain(session.inv,
                                         session.target.rack or "",
                                         session.target.cable or "")
            if console is None:
                _print_line(session, ui.warn(
                    "  x profile tunnel needs a fleet llm_console (or "
                    "console_defaults) block in the inventory"))
                return
            forward = LLMForward(*parse_tunnel_spec(profile.tunnel), console,
                                 session.store,
                                 bastion=llm_bastion_domain(
                                     session.inv, session.target.rack or "",
                                     session.target.cable or ""))
            try:
                url = forward.start()
            except TunnelError as exc:
                forward.close()
                _print_line(session, ui.warn(
                    f"  x rack-manager forward failed ({exc.stage}): {exc} "
                    "(switch aborted; previous model stays active)"))
                return
            _ACTIVE_LLM_FORWARDS.append(forward)
            session.llm_tunnel = profile.tunnel
            session.llm_url = url
            effective = _dc_replace(profile, url=url)
    else:
        effective = (_dc_replace(profile, url=session.llm_url)
                     if not profile.url and session.llm_url else profile)
        if profile.url:
            session.llm_url = profile.url
            session.llm_tunnel = None
    session.llm = effective.build(session.store)
    session.router_llm = effective.build(session.store)
    session.llm_ident = profile.ident
    session.llm_mode = "stub" if profile.ident == "stub" else profile.ident.split("/")[0]
    catalog.choose(profile)
    catalog.save()
    _print_line(session, ui.good(
        f"  model: {profile.ident} (active for the next run; saved)"))
    _save_session(session)


def _slash_model(session: Session, arg: str) -> None:
    """``/model`` / ``/model <ident>`` -- pick the LLM reasoning model from the
    catalog (arrow-key picker, type-to-filter), switch straight to a named
    ident, or run the guided setup for an unconfigured built-in / the ``+``
    row. The choice is applied to the next run and remembered; the picked
    endpoint is probed against the served list (a mismatch offers a one-key
    switch)."""
    from dataclasses import replace as _dc_replace

    from ..config.model_catalog import ModelCatalog, picker_rows
    from .menu import ask_model_profile, check_profile, select

    if session.task is not None and session.task.alive:
        _print_line(session, ui.warn("  model switching is disabled while a run is "
                                     "active (wait or /stop)"))
        return
    catalog = ModelCatalog.load(inv=session.inv)
    arg = arg.strip()
    if arg:
        profile = catalog.resolve(model_id=arg)
    else:
        labels, profiles, add_idx = picker_rows(catalog)
        idx = select("LLM model (reasoning backend)", labels, reader=session.reader)
        if idx is None:
            return
        if idx == add_idx:
            profile = ask_model_profile(reader=session.reader, inv=session.inv,
                                        store=session.store,
                                        target_label=session.target_label or None)
            if profile is None:
                return
            catalog.add(profile)
        else:
            profile = profiles[idx]
            if profile.needs_setup:
                guided = ask_model_profile(provider=profile.provider,
                                           reader=session.reader,
                                           inv=session.inv, store=session.store,
                                           target_label=session.target_label or None)
                if guided is None:
                    return
                catalog.add(guided)
                profile = guided
    _set_session_model(session, catalog, profile)
    suggestion = check_profile(profile, url=session.llm_url,
                               reader=session.reader,
                               rack=session.target.rack or "",
                               cable=session.target.cable or "")
    if suggestion and suggestion != profile.model:
        profile = _dc_replace(profile, model=suggestion)
        _set_session_model(session, catalog, profile)


_HELP = """\
/help      this list
/hosts     list inventory hosts
/use <h|rack cable n|ip|alias>   switch the active target
/model [ident]  pick the LLM model (arrow keys, type to filter); guided setup
                for a new endpoint; or set one directly
/context   queue a note for the next run
/testlog <path>  queue a harness/FAT test log for the next run (failures become evidence)
/status    what is running / what was done
/stop      cancel the running task
/runs      run directories of this session
/history   saved sessions
/resume    load a saved session dir
/lint      validate the inventory
/targets   ls | add <alias> [--rack R --cable N] [--address ip] | rm <alias>
/docs      ls | add <pdf...> | rm <name> | reindex  (RAG library)
/askparts  on | off  prompt for and store missing per-slot parts on rail faults
/quit      exit

Anything else goes to the agent in natural language, e.g.
  "the DIMM error is back on h1"
  "probe the memory controller"
  "look up the DIMM reference in the manual"
  "what do you recommend given the last run?"

The agent explains what it is about to do, then runs read-only tools in the
background (diagnose / probe / docs / verify) and reports back with what it
found; it may chain several tools for one message. Long actions run IN THE
BACKGROUND: the agent keeps working while you keep typing, and its progress and
statements stream in as they happen ("/status" shows what is running; lines
typed while busy are queued and the agent reads them on its next turn
("/context" queues a note explicitly)."""


def _print_help() -> None:
    print(_HELP)


def _handle_slash(session: Session, line: str) -> None:
    parts = line.split(None, 1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""
    if cmd in ("/quit", "/exit", "/q"):
        session.quit = True
        if session.task is not None:
            session.task.cancel.set()
    elif cmd == "/help":
        _print_help()
    elif cmd == "/hosts":
        for host in sorted(session.inv.hosts, key=lambda h: h.name):
            is_active = host.name == session.target_label
            mark = ui.accent("  <- active") if is_active else ""
            name = ui.good(host.name) if is_active else host.name
            _print_line(session, f"  {name}  {host.trust_level}  {host.model}{mark}")
        if session.target_label and session.target_label not in session.inv.host_names:
            _print_line(session,
                        f"  {session.target_label}  (dynamic target, no YAML)")
    elif cmd == "/use":
        if not arg:
            _print_line(session, "  usage: /use <host> | <rack> cable <n> | <ip> | <alias>")
        else:
            spec = _parse_use_arg(session, arg)
            if spec is None:
                _print_line(session, f"  unknown host {arg!r}; see /hosts")
            else:
                _set_target(session, spec)
    elif cmd in ("/model", "/models"):
        _slash_model(session, arg)
    elif cmd == "/context":
        if not arg:
            _print_line(session, "  usage: /context <note>")
        else:
            session.pending.append(arg)
            session.transcript.append({"role": "user", "kind": "context", "content": arg})
            _print_line(session, "  context queued for the next run")
    elif cmd in ("/testlog", "/tl"):
        if not arg:
            _print_line(session, "  usage: /testlog <path-to-harness-FAT-log>")
        else:
            from .cli import _strip_surrounding_quotes
            path = Path(_strip_surrounding_quotes(arg))
            if not path.exists():
                _print_line(session, f"  no such file: {arg}")
            else:
                session.test_logs.append(str(path))
                _print_line(session, f"  test log queued for the next run: {path}")
    elif cmd == "/stop":
        if session.task is not None and session.task.alive:
            session.task.cancel.set()
            _print_line(session, "  stop requested; the run winds down at the next checkpoint")
        else:
            _print_line(session, "  nothing running")
    elif cmd == "/status":
        _print_line(session, _status_line(session))
    elif cmd == "/runs":
        if not session.runs:
            _print_line(session, "  no runs yet in this session")
        for run in reversed(session.runs):
            _print_line(session, f"  {run}")
    elif cmd == "/history":
        _list_sessions(session)
    elif cmd == "/resume":
        if not arg:
            _print_line(session, "  usage: /resume <session-dir>  (see /history)")
        else:
            _load_session(session, Path(arg))
    elif cmd == "/lint":
        _slash_lint(session)
    elif cmd == "/targets":
        _slash_targets(session, arg)
    elif cmd == "/docs":
        _slash_docs(session, arg)
    elif cmd == "/askparts":
        if arg in ("on", "1", "yes", "true"):
            session.ask_parts = True
            _print_line(session, "  ask-parts ON: diagnoses prompt for and store "
                                 "missing per-slot parts (kept on this session)")
            _save_session(session)
        elif arg in ("off", "0", "no", "false"):
            session.ask_parts = False
            _print_line(session, "  ask-parts OFF")
            _save_session(session)
        else:
            state = "ON" if session.ask_parts else "OFF"
            _print_line(session, f"  ask-parts is {state}; usage: /askparts on|off")
    else:
        _print_line(session, f"  unknown command {cmd!r}; /help for the list")


# ---- persistence ----

def _save_session(session: Session) -> None:
    try:
        session.session_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "host": session.host.name if session.host is not None else None,
            "target_label": session.target_label,
            "target": {k: v for k, v in asdict(session.target).items()
                       if v is not None},
            "llm_mode": session.llm_mode,
            "llm_ident": session.llm_ident,
            "ask_parts": session.ask_parts,
            "transcript": list(session.transcript),  # snapshot: worker appends
            "evidence": session.evidence,
            "runs": [str(p) for p in session.runs],
        }
        (session.session_dir / "session.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        pass  # persistence is best-effort


def _load_session(session: Session, path: Path) -> None:
    payload_path = path / "session.json"
    if not payload_path.exists():
        _print_line(session, f"  no session.json in {path}")
        return
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    session.transcript = payload.get("transcript", [])
    session.pending = []
    session.runs = [Path(p) for p in payload.get("runs", [])]
    session.evidence = payload.get("evidence") or ""
    if not session.evidence:
        # Old sessions (pre-agent) did not persist a digest: rebuild one from
        # the newest run that still has a diagnosis.json so follow-up
        # questions are grounded even right after /resume.
        for run in reversed(session.runs):
            diagnosis_path = run / "diagnosis.json"
            if not diagnosis_path.exists():
                continue
            try:
                diag = Diagnosis.model_validate(json.loads(
                    diagnosis_path.read_text(encoding="utf-8")))
                session.evidence = build_evidence(diag, str(run))
            except (OSError, ValueError):
                continue
            break
    session.llm_mode = payload.get("llm_mode") or session.llm_mode
    session.llm_ident = payload.get("llm_ident") or session.llm_ident
    session.ask_parts = bool(payload.get("ask_parts", False))
    if session.llm_ident and session.llm_ident != "stub":
        from ..config.model_catalog import ModelCatalog

        profile = ModelCatalog.load(inv=session.inv).resolve(
            model_id=session.llm_ident, url=session.llm_url)
        session.llm = profile.build(session.store)
        session.router_llm = profile.build(session.store)
    host = payload.get("host")
    if host and host in session.inv.host_names:
        session.host = session.inv.get(host)
    target_raw = payload.get("target") or {}
    if target_raw:
        session.target = TargetSpec(**target_raw)
        session.target_label = payload.get("target_label") or (
            session.host.name if session.host is not None else "(target)")
        try:
            resolve_target(session.target, session.inv, session.store,
                           targets_path=session.targets_file,
                           ssh_user=session.ssh_user,
                           identity_vault_path=session.identity_vault_path,
                           known_hosts_path=session.known_hosts_path)
        except TargetError as exc:
            session.target_label = payload.get("target_label") or "(target)"
            _print_line(session, f"  x target no longer resolvable: {exc}")
    _print_line(session, ui.good(f"  resumed: target={session.target_label}, "
                                 f"{len(session.transcript)} transcript entr(ies)"))
    for entry in session.transcript[-3:]:
        _print_line(session, ui.dim(
            f"    [{entry.get('role')}] {str(entry.get('content', ''))[:120]}"))


def _list_sessions(session: Session) -> None:
    root = session.session_dir.parent
    if not root.exists():
        _print_line(session, "  no saved sessions yet")
        return
    dirs = sorted((p for p in root.iterdir() if p.is_dir()), reverse=True)
    shown = 0
    for path in dirs:
        payload_path = path / "session.json"
        if not payload_path.exists():
            continue
        try:
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
        except OSError:
            continue
        first = next((e.get("content") for e in payload.get("transcript", [])
                      if e.get("kind") == "message"), "")
        _print_line(session, f"  {path}  host={payload.get('host')}  "
                             f"'{str(first)[:60]}'")
        shown += 1
    if not shown:
        _print_line(session, "  no saved sessions yet")


# ---- entry point ----

def run_session(args, overrides: dict | None = None) -> int:
    """Start the interactive chat session. Returns the process exit code."""
    overrides = overrides or {}
    inv = load_inventory(args.inventory)
    if not inv.hosts and inv.console_defaults is None:
        print(f"error: inventory {args.inventory!r} has no hosts", file=sys.stderr)
        return 2
    hosts = sorted(inv.hosts, key=lambda h: h.name)

    from ..operator.cli import (
        _llm_ident_for,
        _make_store,
        _prepare_llm_endpoint,
        _resolve_llm,
    )

    store = overrides.get("store") or _make_store(args)
    # Tunnel / preflight before any adapter is built (same contract as
    # diagnose): a dead --llm-url must fail here, not mid-conversation.
    forward = overrides.get("llm_forward")
    if forward is None:
        forward = _prepare_llm_endpoint(args, inv, store)
    llm = overrides.get("llm") or _resolve_llm(args, inv, store)
    router_llm = overrides.get("router_llm") or _resolve_llm(args, inv, store)
    llm_ident = _llm_ident_for(args, inv)
    llm_mode = "stub" if llm_ident == "stub" else llm_ident.split("/")[0]

    initial_spec = TargetSpec(
        name=args.host if args.host else (hosts[0].name if hosts
                                          and not (args.address or args.target
                                                   or args.rack or args.cable) else None),
        rack=args.rack,
        cable=args.cable,
        ip=args.address,
        alias=args.target,
    )
    if not any((initial_spec.name, initial_spec.rack, initial_spec.cable,
                initial_spec.ip, initial_spec.alias)):
        if hosts:
            initial_spec = TargetSpec(name=hosts[0].name)
        else:
            initial_spec = TargetSpec()  # zero-YAML: resolved by the first message
    target = None
    if any((initial_spec.name, initial_spec.rack, initial_spec.cable,
            initial_spec.ip, initial_spec.alias)):
        try:
            target = resolve_target(
                initial_spec, inv, store,
                targets_path=args.targets_file,
                ssh_user=args.ssh_user,
                identity_vault_path=args.identity_vault_path,
                known_hosts_path=args.known_hosts_path,
            )
        except TargetError as exc:
            print(f"error: cannot resolve initial target: {exc}", file=sys.stderr)
            return 2
    if target is not None:
        apply_ssh_context(store, target, ssh_user=getattr(args, "ssh_user", "diagbot"))

    session = Session(
        inv_path=args.inventory,
        inv=inv,
        host=target.host if target is not None else None,
        store=store,
        out_dir=Path(args.out_dir),
        session_dir=Path(args.session_dir) / (
            f"{target.label}-{int(time.time())}" if target is not None
            else f"session-{int(time.time())}"),
        llm=llm,
        router_llm=router_llm,
        llm_mode=llm_mode,
        llm_ident=llm_ident,
        llm_url=getattr(args, "llm_url", None),
        llm_tunnel=getattr(args, "llm_tunnel", None),
        docs_lib=getattr(args, "docs_lib", None),
        docs_dir=getattr(args, "docs_dir", None),
        parts_csv=getattr(args, "parts_csv", None),
        ask_parts=bool(getattr(args, "ask_parts", False)),
        secret_dir=getattr(args, "secret_dir", None),
        console=bool(getattr(args, "console", False)),
        max_turns=int(getattr(args, "max_turns", 6)),
        server_number=getattr(args, "server_number", None),
        overrides=overrides,
        target=initial_spec,
        target_label=target.label if target is not None else "",
        targets_file=getattr(args, "targets_file", None),
        ssh_user=getattr(args, "ssh_user", "diagbot"),
        identity_vault_path=getattr(args, "identity_vault_path", None),
        known_hosts_path=getattr(args, "known_hosts_path", "config/known_hosts"),
    )
    if getattr(args, "resume", None):
        _load_session(session, Path(args.resume))

    on_session = overrides.get("on_session")
    if on_session is not None:
        on_session(session)

    reader = overrides.get("reader") or _LineReader()
    session.reader = reader
    prompter = getattr(store, "prompter", None)
    if prompter is not None:
        prompter.set_bridge(_credential_bridge(session))

    ui.enable_vt()
    width = max(44, min(ui.terminal_width(), 78))
    print()
    print(ui.rule(width))
    print(f"  {ui.title('harness session')}   "
          f"{ui.dim(f'inventory: {args.inventory}')}")
    print(ui.rule(width))
    if target is not None:
        active = ui.good(session.target_label) + " " + ui.dim(f"({target.kind})")
    else:
        active = ui.dim("(none yet - name a rack/cable/IP)")
    print(ui.kv("target", active))
    print(ui.kv("llm", session.llm_ident or session.llm_mode))
    if hosts:
        print(ui.kv("hosts", ", ".join(
            f"{h.name}{ui.dim(f'({h.trust_level})')}" for h in hosts)))
    print()
    print(ui.dim("  Type /help for commands, or describe a symptom. /quit exits."))
    _save_session(session)

    try:
        while not session.quit:
            try:
                # A task exists the moment a line is handled (the agent task's
                # thread may still be starting). Polling with a timeout
                # whenever ANY task object exists avoids a blocking idle poll in
                # the window before the launched task's thread becomes alive,
                # which would swallow the operator's next line prematurely.
                busy = session.task is not None
                line = reader.poll(0.1 if busy else None)
            except EOFError:
                break
            result, error = _drain_events(session)
            if result is not None or error is not None:
                _finish_task(session, result, error)
                _save_session(session)
            if session.cred_pending is not None:
                if session.task is not None and session.task.alive:
                    _handle_credential(session)
                else:
                    session.cred_pending = None
                    session.cred_event.set()
            task = session.task
            if (task is not None and not task.alive and task.events.empty()
                    and result is None and error is None):
                # Printed-only turn (status / unresolved target / no baseline):
                # the agent task is finished, its queue fully drained -- return
                # to a blocking idle poll so /quit and the next message are
                # picked up promptly.
                session.task = None
                task = None
            if line is None:
                continue
            if session.task is not None and session.task.alive:
                _handle_busy_line(session, line)
            else:
                if not _handle_idle_line(session, line):
                    break
    except KeyboardInterrupt:
        if session.task is not None and session.task.alive:
            session.task.cancel.set()
            _print_line(session, "(stop requested; the run winds down at the next checkpoint)")
        else:
            print()
    finally:
        reader.close()
        _save_session(session)
        if forward is not None:
            forward.close()
    return 0
