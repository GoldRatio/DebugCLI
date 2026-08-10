"""Interactive session REPL: chat with the harness in natural language.

``harness session`` starts a Claude-Code-style REPL: you describe symptoms in
plain English and the agent runs the read-only pipeline (collect -> decode ->
RAG -> LLM -> scored diagnosis) *in the background* while you keep typing.

Design:

- ``BackgroundTask`` runs long actions (routing / diagnose / verify / docs
  lookup / conversation) in a daemon thread. Every worker print is captured and
  streamed back through an event queue, so the terminal stays under the REPL's
  control. NL routing is itself a background task: a slow router LLM never
  freezes the prompt, and a status line ticks every few seconds so it is always
  visible that the agent is working in the background.
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
from ..operator.supervisor import Escalation
from ..targets import TargetError, TargetSpec, resolve_target
from .menu import LineReader as _LineReader
from .router import SessionCommand, route_message

REPLY_SYSTEM = (
    "You are the assistant of a READ-ONLY server debugging harness. Answer the "
    "operator concisely about the current session. You have no evidence beyond "
    "what the harness gathered; never invent registers, commands, or repairs. "
    "Offer concrete next steps the operator can trigger (diagnose, probe, docs, "
    "verify). Respond with a JSON object: {\"text\": \"...\"}."
)


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
    overrides: dict = field(default_factory=dict)
    target: TargetSpec = field(default_factory=TargetSpec)
    target_label: str = ""
    targets_file: str | None = None
    ssh_user: str = "diagbot"
    identity_vault_path: str | None = None
    known_hosts_path: str = "config/known_hosts"
    transcript: list[dict] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)
    runs: list[Path] = field(default_factory=list)
    last_run: Path | None = None
    reader: object | None = None
    task: BackgroundTask | None = None
    answer_value: str = ""
    answer_ready: threading.Event = field(default_factory=threading.Event)
    awaiting: bool = False
    answered: bool = False
    quit: bool = False
    last_tick: float = field(default_factory=time.monotonic)


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
        _print_line(session, f"  x target: {exc}")
        return False
    session.host = target.host
    session.target = spec
    session.target_label = target.label
    _print_line(session, f"  active target: {target.label} ({target.kind})")
    return True


def _diagnose_argv(session: Session, symptom: str, host_name: str | None) -> list[str]:
    """Build the ``diagnose`` argv for one run, seeding queued context."""
    argv = ["diagnose", "--inventory", session.inv_path,
            "--symptom", symptom,
            "--out-dir", str(session.out_dir),
            "--llm", session.llm_mode,
            "--max-turns", str(session.max_turns)]
    argv += _target_argv(session)
    if session.secret_dir:
        argv += ["--secret-dir", session.secret_dir]
    if session.docs_lib:
        argv += ["--docs-lib", session.docs_lib]
    if session.docs_dir:
        argv += ["--docs-dir", session.docs_dir]
    if session.parts_csv:
        argv += ["--parts-csv", session.parts_csv]
    if session.console:
        argv.append("--console")
    for context_line in session.pending:
        argv += ["--context", context_line]
    session.pending = []
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


def _diagnose_task(session: Session, symptom: str,
                   host_name: str | None) -> BackgroundTask:
    argv = _diagnose_argv(session, symptom, host_name)
    label = session.target_label or host_name or (
        session.host.name if session.host is not None else "(unnamed target)")

    def fn(progress: Callable[[str], None], cancel: threading.Event):
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

    return BackgroundTask(label=label, fn=fn)


def _verify_task(session: Session, metric: str,
                 baseline: str | None) -> BackgroundTask:
    if not baseline and session.last_run is not None:
        dumps = session.last_run / "dumps.json"
        if dumps.exists():
            baseline = str(dumps)
    argv = ["verify", "--inventory", session.inv_path,
            "--symptom", "(session verify)",
            "--baseline", baseline or ""]
    argv += _target_argv(session)
    if session.secret_dir:
        argv += ["--secret-dir", session.secret_dir]
    if metric:
        argv += ["--metric", metric]

    def fn(progress: Callable[[str], None], cancel: threading.Event):
        from ..operator.cli import build_parser, run_verify
        args = build_parser().parse_args(argv)
        return run_verify(args, overrides={
            **session.overrides, "store": session.store})

    return BackgroundTask(label="verify", fn=fn)


def _docs_task(session: Session, query: str) -> BackgroundTask:
    def fn(progress: Callable[[str], None], cancel: threading.Event):
        from types import SimpleNamespace

        from ..operator.cli import _build_docs_retriever
        retriever, _lib = _build_docs_retriever(SimpleNamespace(
            docs_lib=session.docs_lib, docs_dir=session.docs_dir))
        if retriever is None:
            return "(no doc library; use `harness docs add` to upload PDFs)"
        lines = retriever(query)
        if not lines:
            return f"(no snippets matched {query!r})"
        return "\n".join(f"- {line}" for line in lines)

    return BackgroundTask(label="docs lookup", fn=fn)


def _reply_task(session: Session, text: str) -> BackgroundTask:
    def fn(progress: Callable[[str], None], cancel: threading.Event):
        if not callable(getattr(session.llm, "chat_json", None)):
            return ("(no LLM configured for conversation; describe a symptom to "
                    "diagnose, or see /help)")
        messages = [{"role": "system", "content": REPLY_SYSTEM}]
        for entry in session.transcript[-6:]:
            role = "assistant" if entry.get("role") == "agent" else "user"
            messages.append({"role": role, "content": entry["content"]})
        messages.append({"role": "user", "content": text})
        raw = session.llm.chat_json(messages)
        return raw.get("text") or raw.get("content") or "(no reply)"

    return BackgroundTask(label="reply", fn=fn)


# ---- rendering ----

def _print_line(session: Session, text: str) -> None:
    reader = session.reader
    clear = getattr(reader, "clear_line", None)
    if clear is not None:
        clear()
    print(text, flush=True)
    refresh = getattr(reader, "refresh_line", None)
    if refresh is None:
        refresh = getattr(reader, "redraw", None)
    if refresh is not None:
        refresh()


def _render_event(session: Session, kind: str, payload: object) -> None:
    if kind == "progress":
        _print_line(session, f"  {payload}")
    elif kind == "out":
        for line in str(payload).splitlines():
            _print_line(session, line)
    elif kind == "err":
        for line in str(payload).splitlines():
            _print_line(session, f"  [stderr] {line}")


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
        _print_line(session, f"  x {task.label}: {error}")
        session.transcript.append({"role": "tool", "kind": "error", "content": error})
        return
    if isinstance(result, Diagnosis):
        run_dir = _find_newest_run(session.out_dir, session.session_dir)
        if run_dir is not None:
            session.last_run = run_dir
            session.runs.append(run_dir)
        _print_line(session, f"  done: diagnosis complete ({task.elapsed():.0f}s) in {run_dir}")
        session.transcript.append({
            "role": "agent", "kind": "diagnosis",
            "content": f"{result.diagnosis} (confidence {result.confidence:.2f})",
        })
    else:
        _print_line(session, f"  done: {result}")
        session.transcript.append({"role": "tool", "kind": "result", "content": str(result)})
    if session.pending:
        _print_line(session, f"  {len(session.pending)} queued message(s) seed the next run")


# ---- line handling ----

def _status_line(session: Session) -> str:
    if session.task is not None and session.task.alive:
        return (f"  running in the background: {session.task.label} "
                f"({session.task.elapsed():.0f}s) -- you can keep typing; "
                f"/stop cancels")
    if session.pending:
        return f"  idle; {len(session.pending)} queued message(s) for the next run"
    if session.last_run is not None:
        return f"  idle; last run: {session.last_run}"
    return "  idle"


def _launch(session: Session, cmd: SessionCommand, raw_text: str) -> BackgroundTask | None:
    """Start the background task for a routed command; returns it (None when
    the command only printed, e.g. status / a missing-baseline verify).

    The launch announcement is queued on the NEW task (whose events the REPL
    drains) so it is rendered even though the worker that started it is a
    different thread whose own queue is already abandoned.
    """
    if cmd.intent == "diagnose":
        session.task = _diagnose_task(session, cmd.symptom or raw_text, cmd.host)
    elif cmd.intent == "verify":
        if not cmd.baseline and (session.last_run is None or
                                 not (session.last_run / "dumps.json").exists()):
            _print_line(session, "  no baseline yet: run a diagnosis first "
                                 "(or pass a baseline path)")
            return None
        session.task = _verify_task(session, cmd.metric, cmd.baseline)
    elif cmd.intent == "docs":
        if not session.docs_lib and not session.docs_dir:
            _print_line(session, "  no doc library (set --docs-lib / --docs-dir)")
            return None
        session.task = _docs_task(session, cmd.query or raw_text)
    elif cmd.intent == "status":
        _print_line(session, _status_line(session))
        return None
    else:
        session.task = _reply_task(session, cmd.text or raw_text)
    session.task.start()
    session.task.events.put(("progress",
        f"started in the background: {session.task.label} -- keep typing "
        f"(messages queue), /status for progress, /stop to cancel"))
    return session.task


def _apply_cmd_target(session: Session, cmd: SessionCommand) -> bool:
    """If the routed message names a target, switch the active target first.
    Returns False when the named target could not be resolved (the error was
    already printed), so the caller does not launch a doomed run."""
    spec = None
    if cmd.host is not None and cmd.host in session.inv.host_names:
        spec = TargetSpec(name=cmd.host)
    elif cmd.ip is not None:
        spec = TargetSpec(ip=cmd.ip)
    elif cmd.rack is not None and cmd.cable is not None:
        spec = TargetSpec(rack=cmd.rack, cable=cmd.cable)
    elif cmd.alias is not None:
        spec = TargetSpec(alias=cmd.alias)
    if spec is not None and spec != session.target:
        return _set_target(session, spec)
    return True


def _route_task(session: Session, line: str) -> BackgroundTask:
    """Route one NL message in the background, then hand the command to
    ``_launch``.

    Routing calls the router LLM, which can take many seconds (or time out).
    Running it on the worker keeps the REPL responsive: the operator sees the
    "thinking" progress and can keep typing (lines are queued) while the agent
    decides what to do.

    Prints made by the worker (target errors, "no baseline yet", status notes)
    are captured into THIS task's queue, which the REPL drains; the worker must
    never clear ``session.task`` itself, or the main loop would skip this queue
    and the message would be lost.
    """

    def fn(progress: Callable[[str], None], cancel: threading.Event):
        progress("thinking: routing your message to the agent ...")
        cmd = route_message(line, session.router_llm, session.transcript[-8:],
                            tuple(session.inv.host_names))
        if cancel.is_set():
            return None
        if not _apply_cmd_target(session, cmd):
            return None  # unresolved target: error already printed
        _launch(session, cmd, line)
        return None  # printed-only commands leave a done(None) the REPL drains

    return BackgroundTask(label="routing to agent", fn=fn)


def _handle_idle_line(session: Session, line: str) -> bool:
    """Route one operator message; may start a background task. False = quit."""
    if not line.strip():
        return True
    if line.startswith("/"):
        _handle_slash(session, line)
        return not session.quit
    session.transcript.append({"role": "user", "kind": "message", "content": line})
    _save_session(session)
    session.task = _route_task(session, line)
    session.task.start()
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
        _print_line(session, "  (answer sent to agent)")
        return
    session.pending.append(line)
    session.transcript.append({"role": "user", "kind": "message", "content": line})
    _print_line(session, "  (queued; the agent sees this after the current run)")


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


_HELP = """\
/help      this list
/hosts     list inventory hosts
/use <h|rack cable n|ip|alias>   switch the active target
/context   queue a note for the next run
/status    what is running / what was done
/stop      cancel the running task
/runs      run directories of this session
/history   saved sessions
/resume    load a saved session dir
/lint      validate the inventory
/targets   ls | add <alias> [--rack R --cable N] [--address ip] | rm <alias>
/docs      ls | add <pdf...> | rm <name> | reindex  (RAG library)
/quit      exit

Anything else is routed to the agent in natural language, e.g.
  "the DIMM error is back on h1"
  "probe the memory controller"
  "look up the DIMM reference in the manual"
  "verify whether the counter changed since the last run"

Long actions run IN THE BACKGROUND: the agent's progress and a status line
every few seconds show it is working while you keep typing; lines typed then
are queued and seed the next run ("/context" does the same explicitly)."""


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
            mark = " (active)" if host.name == session.target_label else ""
            _print_line(session, f"  {host.name}  {host.trust_level}  {host.model}{mark}")
        if session.target_label and session.target_label not in session.inv.host_names:
            _print_line(session, f"  {session.target_label}  (dynamic target, no YAML)")
    elif cmd == "/use":
        if not arg:
            _print_line(session, "  usage: /use <host> | <rack> cable <n> | <ip> | <alias>")
        else:
            spec = _parse_use_arg(session, arg)
            if spec is None:
                _print_line(session, f"  unknown host {arg!r}; see /hosts")
            else:
                _set_target(session, spec)
    elif cmd == "/context":
        if not arg:
            _print_line(session, "  usage: /context <note>")
        else:
            session.pending.append(arg)
            session.transcript.append({"role": "user", "kind": "context", "content": arg})
            _print_line(session, "  context queued for the next run")
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
            "transcript": session.transcript,
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
    _print_line(session, f"  resumed: target={session.target_label}, "
                         f"{len(session.transcript)} transcript entr(ies)")
    for entry in session.transcript[-3:]:
        _print_line(session, f"    [{entry.get('role')}] {str(entry.get('content', ''))[:120]}")


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

    from ..operator.cli import _make_store, _resolve_llm

    store = overrides.get("store") or _make_store(args)
    llm = overrides.get("llm") or _resolve_llm(args, inv, store)
    router_llm = overrides.get("router_llm") or _resolve_llm(args, inv, store)
    llm_mode = args.llm or (inv.llm.provider if inv.llm else None) or "openai"

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
        docs_lib=getattr(args, "docs_lib", None),
        docs_dir=getattr(args, "docs_dir", None),
        parts_csv=getattr(args, "parts_csv", None),
        secret_dir=getattr(args, "secret_dir", None),
        console=bool(getattr(args, "console", False)),
        max_turns=int(getattr(args, "max_turns", 6)),
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

    print(f"harness session | inventory: {args.inventory}")
    print(f"hosts: {', '.join(f'{h.name}({h.trust_level})' for h in hosts)}")
    active = (f"active target: {session.target_label} ({target.kind})"
              if target is not None else "active target: (none yet - name a rack/cable/IP)")
    print(f"{active} | llm: {session.llm_mode}")
    print("Type /help for commands, or describe a symptom. /quit exits.")
    _save_session(session)

    try:
        while not session.quit:
            try:
                # A task exists the moment a line is handled (the routing worker
                # may still be assigning the real task). Polling with a timeout
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
            task = session.task
            if (task is not None and not task.alive and task.events.empty()
                    and result is None and error is None):
                # Printed-only command (status / unresolved target / missing
                # baseline): its routing task is finished, its queue fully
                # drained -- return to a blocking idle poll so /quit and the
                # next message are picked up promptly.
                session.task = None
                task = None
            if (task is not None and task.alive
                    and time.monotonic() - session.last_tick >= 5.0):
                session.last_tick = time.monotonic()
                _print_line(session, _status_line(session))
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
    return 0
