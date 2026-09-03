"""Interactive session REPL: NL routing, background tasks, scripted end-to-end."""

import json
import queue
import threading
import time
from pathlib import Path

import pytest

from harness.audit.auditlog import AuditLog
from harness.config.inventory_lint import load_inventory
from harness.config.vault import MemorySecretStore
from harness.diagnosis.llm import StubLLM
from harness.engine.allowlist import AllowPolicy, AllowRule
from harness.engine.runner import CommandResult, Runner
from harness.operator.chat_agent import ChatTurn
from harness.operator.cli import build_parser
from harness.operator.repl import (
    BackgroundTask,
    Session,
    _diagnose_argv,
    _handle_busy_line,
    _tool_file,
    run_session,
)
from harness.operator.router import _keyword_route

FAKE_POLICY = AllowPolicy([
    AllowRule("/usr/bin/rdmsr", ("-a",)),
    AllowRule("/bin/dmesg", ("-l", "*")),
    AllowRule("/bin/dmidecode", ()),
    AllowRule("/usr/bin/lspci", ("-xxx",)),
    AllowRule("/bin/smartctl", ("-a", "*")),
    AllowRule("/bin/smartctl", ("-x", "*")),
])

OUTPUT = {
    "/usr/bin/rdmsr -a": "IA32_MC0_STATUS = 0x8000000000000001\n",
    "/bin/dmesg -l err": "MCE: memory error on DIMM_A2\n",
    "/bin/dmesg -l err, crit, alert, emerg": "MCE: memory error on DIMM_A2\n",
    "/bin/dmidecode": "Product Name: model_x\nBIOS Vendor: Intel\nBIOS Version: 2.3\n",
    "/usr/bin/lspci -xxx": "00:1f.2 PCIe link down\n",
}


class FakeSession(Runner):
    """Stand-in for SSHSession: canned output per argv, records calls."""

    def __init__(self) -> None:
        super().__init__(FAKE_POLICY)

    def _exec(self, argv, timeout=30.0):
        return CommandResult(argv=list(argv), stdout=OUTPUT.get(" ".join(argv), ""),
                             stderr="", exit_code=0, elapsed_ms=1)


_INVENTORY = (
    "trust_level: lab\n"
    "hosts:\n"
    "  - name: h1\n"
    "    address: 10.0.0.10\n"
    "    model: model_x\n"
    "    ssh:\n"
    "      user: diagbot\n"
    "      identity_vault_path: secret/harness/diagbot/id_ed25519\n"
    "      known_hosts_path: config/known_hosts\n"
    "    bmc:\n"
    "      address: 10.0.0.11\n"
    "      username: bmc-ro\n"
    "      password_vault_path: secret/harness/bmc/bmc-ro\n"
)


def _inventory(tmp_path) -> str:
    path = tmp_path / "inventory.yaml"
    path.write_text(_INVENTORY, encoding="utf-8")
    return str(path)


def _make_session(tmp_path, overrides=None) -> Session:
    inv_path = Path(_inventory(tmp_path))
    inv = load_inventory(inv_path)
    return Session(
        mode="debug",
        inv_path=str(inv_path),
        inv=inv,
        host=inv.get("h1"),
        store=MemorySecretStore(),
        out_dir=tmp_path / "runs",
        session_dir=tmp_path / "sessions" / "s1",
        llm=StubLLM(),
        router_llm=StubLLM(),
        llm_mode="stub",
        docs_lib=None,
        docs_dir=None,
        parts_csv=None,
        secret_dir=None,
        console=False,
        max_tools=0,
        overrides=overrides or {},
    )


def _session_args(tmp_path, **kw):
    argv = ["debug", "--inventory", _inventory(tmp_path), "--host", "h1",
            "--out-dir", str(tmp_path / "runs"),
            "--session-dir", str(tmp_path / "sessions"),
            "--llm", "stub"]
    if kw.get("host"):
        argv += ["--host", kw["host"]]
    return build_parser().parse_args(argv)


class _ScriptedReader:
    """Deterministic stand-in for _LineReader.

    Idle polls (timeout=None) pop the next scripted line (EOFError when empty);
    busy polls yield a line only while ``yield_when()`` is true, so the test can
    say "answer the agent as soon as it asks" without racing the worker thread.
    ``keys`` script the raw key events that the arrow-key picker (``/model``)
    reads via ``read_key``; when empty it cancels the picker (esc).
    """

    def __init__(self, lines, keys=(), yield_when=None):
        self.lines = list(lines)
        self.keys = list(keys)
        self.yield_when = yield_when or (lambda: False)

    @property
    def raw(self):
        return True

    def poll(self, timeout):
        if timeout is None:
            if not self.lines:
                raise EOFError
            return self.lines.pop(0)
        if self.lines and self.yield_when():
            return self.lines.pop(0)
        return None

    def read_key(self, timeout=None):
        if self.keys:
            return self.keys.pop(0)
        return "esc"

    def clear_line(self):
        pass

    def refresh_line(self):
        pass

    def redraw(self):
        pass

    def close(self):
        pass


# ---- keyword fallback ----

def test_keyword_route_intents():
    assert _keyword_route("the DIMM error came back").intent == "diagnose"
    assert _keyword_route("probe the memory subsystem").intent == "probe"
    assert _keyword_route("check registers on the cpu").intent == "probe"
    assert _keyword_route("look up the DIMM reference in the manual").intent == "docs"
    assert _keyword_route("verify whether the counter changed").intent == "verify"
    assert _keyword_route("what is the harness doing").intent == "status"
    assert _keyword_route("hello there").intent == "reply"


class _AgentLLM:
    """Conversation-LLM returning a ChatTurn-shaped payload."""

    def __init__(self, raw):
        self.raw = raw
        self.messages = None

    def chat_json(self, messages):
        self.messages = messages
        return self.raw


class _NoChatLLM:
    """Router without a chat_json: forces the keyword-fallback path, which is
    what natural-language-targeting / intent tests want to exercise."""

    def __call__(self, prompt):
        raise AssertionError("fallback router must not be called as an LLM")


# ---- session helpers ----

def test_diagnose_argv_carries_pending_context(tmp_path):
    session = _make_session(tmp_path)
    session.pending = ["Replaced PSU last week"]
    argv = _diagnose_argv(session, "ECC error", None)
    assert "--host" in argv and argv[argv.index("--host") + 1] == "h1"
    assert "--context" in argv and "Replaced PSU last week" in argv
    assert session.pending == []  # context seeded, queue drained


def test_diagnose_argv_carries_test_logs(tmp_path):
    session = _make_session(tmp_path)
    session.test_logs = [str(tmp_path / "fat.log")]
    argv = _diagnose_argv(session, "ECC error", None)
    assert "--test-log" in argv
    assert argv[argv.index("--test-log") + 1] == str(tmp_path / "fat.log")
    assert session.test_logs == []  # queued, then drained


def test_testlog_slash_command_queues_path(tmp_path):
    from harness.operator.repl import _handle_slash

    session = _make_session(tmp_path)
    log_file = tmp_path / "fat.log"
    log_file.write_text("ERROR:2026-08-11 22:02:01 || [FAIL][P02002001@Test]\n",
                        encoding="utf-8")
    _handle_slash(session, f"/testlog {log_file}")
    assert session.test_logs == [str(log_file)]
    # missing path is rejected, queue untouched
    _handle_slash(session, "/testlog does/not/exist.log")
    assert session.test_logs == [str(log_file)]
    # no arg prints usage
    _handle_slash(session, "/testlog")
    assert session.test_logs == [str(log_file)]


def test_diagnose_argv_askparts_toggle(tmp_path):
    session = _make_session(tmp_path)
    assert "--ask-parts" not in _diagnose_argv(session, "ECC error", None)
    session.ask_parts = True
    argv = _diagnose_argv(session, "ECC error", None)
    assert "--ask-parts" in argv


def test_askparts_slash_command_toggles(tmp_path, capsys):
    from harness.operator.repl import _handle_slash

    session = _make_session(tmp_path)
    _handle_slash(session, "/askparts on")
    assert session.ask_parts is True
    _handle_slash(session, "/askparts off")
    assert session.ask_parts is False
    _handle_slash(session, "/askparts bogus")
    assert session.ask_parts is False  # unknown arg leaves state untouched


def test_busy_line_queued_when_not_awaiting(tmp_path):
    session = _make_session(tmp_path)
    _handle_busy_line(session, "also check storage")
    assert session.pending == ["also check storage"]


def test_busy_line_answers_when_awaiting(tmp_path):
    session = _make_session(tmp_path)
    session.awaiting = True
    _handle_busy_line(session, "reseated already")
    assert session.answer_value == "reseated already"
    assert session.answer_ready.is_set()
    assert session.answered is True
    # a second line while the same question is pending must not clobber the answer
    _handle_busy_line(session, "more text")
    assert session.answer_value == "reseated already"
    assert session.pending == ["more text"]


# ---- background tasks ----

def _drain(task: BackgroundTask, timeout=5.0):
    events = []
    deadline = time.monotonic() + timeout
    while task.alive or not task.events.empty():
        try:
            events.append(task.events.get_nowait())
        except queue.Empty:
            if time.monotonic() > deadline:
                raise TimeoutError("task did not finish")
            time.sleep(0.005)
    return events


def test_background_task_streams_progress_and_stdout():
    def fn(progress, cancel):
        progress("collecting memory registers")
        print("dmesg: MCE on DIMM_A2")
        return "ok"

    task = BackgroundTask("test", fn)
    task.start()
    events = _drain(task)
    assert ("progress", "collecting memory registers") in events
    assert ("out", "dmesg: MCE on DIMM_A2") in events
    assert ("done", "ok") in events
    assert not task.alive


def test_background_task_surfaces_errors():
    def fn(progress, cancel):
        raise RuntimeError("boom")

    task = BackgroundTask("test", fn)
    task.start()
    events = _drain(task)
    assert any(k == "error" and "boom" in str(p) for k, p in events)


def test_background_task_builds_argparse_with_captured_stdout(monkeypatch):
    """A worker task may call ``build_parser()`` while stdout is captured: the
    capture's ``fileno()`` must delegate to the real stream, or Python 3.14's
    argparse help-color detection calls ``os.isatty(None)`` and raises
    ``TypeError: 'NoneType' object cannot be interpreted as an integer``
    (the chat-session diagnose/verify tasks build their argv this way).

    ``can_colorize`` reaches ``os.isatty`` only when the Windows console advertises
    virtual-terminal support, so force it to take that path regardless of the
    test runner's console mode.
    """
    monkeypatch.setattr("nt._supports_virtual_terminal", lambda: True)

    def fn(progress, cancel):
        args = build_parser().parse_args(
            ["diagnose", "--inventory", "config/rack.yaml",
             "--symptom", "x", "--out-dir", "harness_runs"])
        return f"ok {args.symptom}"

    task = BackgroundTask("test", fn)
    task.start()
    events = _drain(task)
    assert ("done", "ok x") in events
    assert not any(k == "error" for k, _ in events)


def test_capture_fileno_delegates_to_real_stream():
    import io as _io

    from harness.operator.repl import _Capture

    # fd-less stream: delegation raises exactly what the real stream raises
    with pytest.raises(_io.UnsupportedOperation):
        _Capture(lambda line: None, _io.StringIO()).fileno()

    # real fd: returns the underlying stream's fd
    class _Real:
        def fileno(self):
            return 7

    assert _Capture(lambda line: None, _Real()).fileno() == 7



def test_background_task_obeys_cancel():
    def fn(progress, cancel):
        progress("working")
        while not cancel.is_set():
            time.sleep(0.01)
        raise RuntimeError("wind-down")

    task = BackgroundTask("test", fn)
    task.start()
    deadline = time.monotonic() + 2.0
    while not any(k == "progress" for k, _ in _peek(task)):
        if time.monotonic() > deadline:
            raise TimeoutError("task never started")
        time.sleep(0.01)
    task.cancel.set()
    events = _drain(task)
    assert any(k == "error" for k, _ in events)


def _peek(task: BackgroundTask):
    events = []
    while not task.events.empty():
        events.append(task.events.get_nowait())
    return events


# ---- end-to-end REPL ----

def test_session_repl_end_to_end(tmp_path, capsys):
    holder = {}

    def on_session(session):
        holder["session"] = session

    def yield_when():
        return (holder["session"].awaiting
                and not holder["session"].answered)

    class DiagnoseOnce:
        """First message -> diagnose tool; follow-up -> conversational answer
        grounded in the queued note (fed as Operator Notes)."""

        def __init__(self):
            self.calls: list[list[dict]] = []

        def chat_json(self, messages):
            self.calls.append(messages)
            if len(self.calls) == 1:
                return {"say": "Running a read-only diagnosis.",
                        "tool": "diagnose", "symptom": "ECC error"}
            return {"say": "Given the note that the DIMM was reseated, "
                           "I would re-check the error counters.",
                    "tool": "none"}

    router = DiagnoseOnce()
    reader = _ScriptedReader([
        "diagnose the ECC error",
        "DIMM was reseated already, no change",
        "/quit",
    ], yield_when=yield_when)
    args = _session_args(tmp_path)

    code = run_session(args, overrides={
        "reader": reader,
        "on_session": on_session,
        "session": FakeSession(),
        "llm": StubLLM(),
        "router_llm": router,
    })
    assert code == 0

    out = capsys.readouterr().out
    assert "harness debug" in out
    assert "==== Diagnosis" in out                   # full report streamed from the worker
    assert "diagnosis complete" in out
    assert "re-check the error counters" in out      # follow-up answer
    # the queued note reached the agent's next decision
    assert any("reseated" in m.get("content", "")
               for call in router.calls[1:] for m in call)

    session_dir = next((tmp_path / "sessions").iterdir())
    payload = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    kinds = [e["kind"] for e in payload["transcript"]]
    assert "message" in kinds and "diagnosis" in kinds
    assert any("reseated" in e["content"] for e in payload["transcript"])

    run_dir = next((tmp_path / "runs").iterdir())
    assert (run_dir / "diagnosis.json").exists()
    assert (run_dir / "audit.jsonl").exists()


def test_session_repl_quit_directly(tmp_path, capsys):
    reader = _ScriptedReader(["/quit"])
    code = run_session(_session_args(tmp_path), overrides={"reader": reader})
    assert code == 0
    assert "Type /help" in capsys.readouterr().out


def test_session_banner_layout(tmp_path, capsys):
    """Startup gets a distinct header block (rules + labeled fields) instead
    of four same-weight lines; plain-mode output keeps the key content."""
    reader = _ScriptedReader(["/quit"])
    code = run_session(_session_args(tmp_path), overrides={"reader": reader})
    assert code == 0
    out = capsys.readouterr().out
    assert "harness debug" in out
    assert "h1 (named)" in out                 # target field
    assert "stub" in out                       # llm field
    assert "Type /help for commands, or describe a symptom. /quit exits." in out


def test_agent_runs_in_background_and_announces_launch(tmp_path, capsys):
    """A slow conversation LLM must NOT freeze the REPL: the operator keeps
    typing (lines queue up), a status tick shows the agent working in the
    background, and the turn is announced and finishes once the LLM responds.

    Regression: routing used to block the main thread synchronously, so after
    submitting a message the session sat silently (no run, no status) until
    the operator gave up and quit."""
    holder = {}
    gate = threading.Event()
    threading.Timer(1.0, gate.set).start()  # release the LLM shortly after start

    class BlockingLLM:
        def chat_json(self, messages):
            gate.wait(5.0)
            return {"say": "hi"}  # old-format/no-tool output -> keyword fallback

    class Reader(_ScriptedReader):
        def __init__(self):
            super().__init__(["hello there", "/quit"])
            self.queued = False

        def poll(self, timeout):
            if timeout is None:
                if not self.lines:
                    raise EOFError
                return self.lines.pop(0)
            task = holder["s"].task
            if task is not None and task.alive and not self.queued:
                self.queued = True
                return "second line while the agent thinks"
            return None

    def on_session(session):
        holder["s"] = session

    code = run_session(_session_args(tmp_path), overrides={
        "reader": Reader(),
        "on_session": on_session,
        "llm": StubLLM(),
        "router_llm": BlockingLLM(),
    })
    assert code == 0

    out = capsys.readouterr().out
    assert "thinking: the agent is considering your message" in out
    assert "started in the background: h1" in out
    assert "done: (no conversational reply available" in out

    session = holder["s"]
    assert session.pending == ["second line while the agent thinks"]


def test_session_background_does_not_tick_status_lines(tmp_path, capsys):
    """While a diagnose runs, the REPL does NOT re-print a status line every
    few seconds -- the launch announcement and the agent's own statements are
    the only background output."""
    holder = {}

    def on_session(session):
        holder["s"] = session

    reader = _ScriptedReader([
        "diagnose the ECC error",
        "DIMM was reseated already, no change",
        "/quit",
    ], yield_when=lambda: (holder["s"].awaiting and not holder["s"].answered))
    code = run_session(_session_args(tmp_path), overrides={
        "reader": reader,
        "on_session": on_session,
        "session": FakeSession(),
        "llm": StubLLM(),
        "router_llm": _NoChatLLM(),
    })
    assert code == 0
    out = capsys.readouterr().out
    assert "started in the background: h1" in out
    assert "diagnosis complete" in out
    assert "running in the background" not in out
    assert "you can keep typing" not in out


def test_session_repl_hosts_and_use(tmp_path, capsys):
    reader = _ScriptedReader(["/hosts", "/quit"])
    code = run_session(_session_args(tmp_path), overrides={"reader": reader})
    assert code == 0
    out = capsys.readouterr().out
    assert "h1" in out and "active" in out


def test_session_repl_verify_without_baseline(tmp_path, capsys):
    reader = _ScriptedReader(["verify whether the counter changed", "/quit"])
    code = run_session(_session_args(tmp_path), overrides={
        "reader": reader, "llm": StubLLM(), "router_llm": _NoChatLLM()})
    assert code == 0
    out = capsys.readouterr().out
    assert "no baseline yet" in out


def test_session_repl_docs_without_library(tmp_path, capsys):
    reader = _ScriptedReader(["look up the DIMM in the manual", "/quit"])
    run_session(_session_args(tmp_path), overrides={
        "reader": reader, "llm": StubLLM(), "router_llm": _NoChatLLM()})
    assert "no doc library" in capsys.readouterr().out


# ---- probe intent wiring ----

def test_session_repl_probe_runs_and_decodes(tmp_path, capsys):
    """'probe' must run real read-only collectors on the active target and
    report the decoded registers -- not fall through to a chat reply."""
    holder = {}

    def on_session(session):
        holder["session"] = session

    reader = _ScriptedReader(["probe the memory controller", "/quit"])
    code = run_session(_session_args(tmp_path), overrides={
        "reader": reader,
        "on_session": on_session,
        "session": FakeSession(),
        "llm": StubLLM(),
        "router_llm": _AgentLLM({"say": "Checking the memory subsystem.",
                                 "tool": "probe", "subsystems": ["memory"]}),
    })
    assert code == 0
    out = capsys.readouterr().out
    assert "started in the background: h1" in out
    assert "agent: Checking the memory subsystem." in out
    assert "probe memory: cpu_msr, kernel" in out
    assert "memory/cpu_msr: 2 dump(s), 2 ok" in out
    assert "memory/kernel: 1 dump(s), 1 ok" in out
    assert "decoded registers" in out
    assert "IA32_MC0_STATUS" in out

    session = holder["session"]
    kinds = [e["kind"] for e in session.transcript]
    assert "result" in kinds  # probe evidence recorded as a tool result, not a chat reply


def test_session_repl_probe_unknown_subsystem_reported(tmp_path, capsys):
    reader = _ScriptedReader(["probe something odd", "/quit"])
    code = run_session(_session_args(tmp_path), overrides={
        "reader": reader,
        "session": FakeSession(),
        "llm": StubLLM(),
        "router_llm": _AgentLLM({"say": "Probing the odd subsystem.",
                                 "tool": "probe", "subsystems": ["bogus"]}),
    })
    assert code == 0
    assert "unknown subsystem 'bogus' (skipped)" in capsys.readouterr().out


def test_session_repl_probe_doc_topic_without_library(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)  # isolate from the repo's harness_docs/ library
    reader = _ScriptedReader(["probe the boot state from the manual", "/quit"])
    code = run_session(_session_args(tmp_path), overrides={
        "reader": reader,
        "session": FakeSession(),
        "llm": StubLLM(),
        "router_llm": _AgentLLM({"say": "Mining the manual.",
                                 "tool": "probe", "doc_topics": ["boot state"]}),
    })
    assert code == 0
    assert "doc topic(s) skipped: no doc library configured" in capsys.readouterr().out


def test_session_repl_probe_console_after_use(tmp_path, capsys):
    """Probing over the serial console runs the BMC-shell collectors on the
    selected rack/cable target (no SSH, no host OS probes)."""
    holder = {}

    def on_session(session):
        holder["session"] = session

    reader = _ScriptedReader([
        "/use Q61 Cable 8",
        "probe the memory controller",
        "/quit",
    ])
    console_runner = _FakeConsoleRunner()
    code = run_session(_console_defaults_session_args(tmp_path), overrides={
        "reader": reader,
        "on_session": on_session,
        "console_runner": console_runner,
        "llm": StubLLM(),
        "router_llm": _AgentLLM({"say": "Probing memory on the console target.",
                                 "tool": "probe", "subsystems": ["memory"]}),
    })
    assert code == 0
    session = holder["session"]
    assert session.target.rack == "Q61" and session.target.cable == "8"

    out = capsys.readouterr().out
    assert "started in the background: Q61-cable8" in out
    assert "probe memory: cpu_msr, kernel" in out
    assert "dump(s)" in out

    calls = [" ".join(c.argv) for c in console_runner.calls]
    assert "sudo -S ipmitool sensor list" in calls
    assert not any("/bin/dmidecode" in c for c in calls)


# ---- agentic loop ----

def test_agent_chains_probe_then_conversational(tmp_path, capsys):
    """The agent explains what it is about to do BEFORE a tool runs, sees the
    tool result, then answers conversationally -- one background task per
    message, no separate routing hop."""
    holder = {}

    def on_session(session):
        holder["session"] = session

    class Chained:
        def __init__(self):
            self.calls = 0

        def chat_json(self, messages):
            self.calls += 1
            if self.calls == 1:
                return {"say": "Collecting memory evidence first.",
                        "tool": "probe", "subsystems": ["memory"]}
            return {"say": "Decoded registers show a corrected MCA on DIMM_A2; "
                           "I would reseat it.", "tool": "none"}

    reader = _ScriptedReader(["probe the memory controller", "/quit"])
    code = run_session(_session_args(tmp_path), overrides={
        "reader": reader,
        "on_session": on_session,
        "session": FakeSession(),
        "llm": StubLLM(),
        "router_llm": Chained(),
    })
    assert code == 0

    out = capsys.readouterr().out
    # announce-before-act ordering: the say prints before the tool's progress
    assert out.index("Collecting memory evidence first.") < \
        out.index("probe memory: cpu_msr, kernel")
    assert "Decoded registers show a corrected MCA" in out

    session = holder["session"]
    kinds = [e["kind"] for e in session.transcript]
    assert "say" in kinds and "action" in kinds and "result" in kinds


def test_agent_follow_up_grounded_in_evidence(tmp_path, capsys):
    """After a diagnosis run the agent's next decision sees the evidence
    digest, so a follow-up ("what do you recommend?") is grounded in the run
    rather than answered blind."""
    holder = {}

    def on_session(session):
        holder["session"] = session

    def yield_when():
        return (holder["session"].awaiting and not holder["session"].answered)

    class FollowUp:
        def __init__(self):
            self.calls = 0
            self.seen = []

        def chat_json(self, messages):
            self.calls += 1
            self.seen.append(messages)
            if self.calls == 1:
                return {"say": "Running a full diagnosis on the ECC error.",
                        "tool": "diagnose", "symptom": "ECC error on DIMM_A2"}
            return {"say": "Based on the evidence I would reseat DIMM_A2 first.",
                    "tool": "none"}

    reader = _ScriptedReader([
        "diagnose the ECC error",
        "DIMM was reseated already, no change",
        "/quit",
    ], yield_when=yield_when)
    follow_up = FollowUp()
    code = run_session(_session_args(tmp_path), overrides={
        "reader": reader,
        "on_session": on_session,
        "session": FakeSession(),
        "llm": StubLLM(),
        "router_llm": follow_up,
    })
    assert code == 0

    session = holder["session"]
    assert "## Latest Run Evidence" in session.evidence  # digest persisted
    # the post-diagnosis decision was grounded in the digest + transcript
    assert "Latest Run Evidence" in follow_up.seen[-1][-1]["content"]
    assert "diagnosis complete" in capsys.readouterr().out


def test_agent_answers_past_run_question_without_diagnosing(tmp_path, capsys):
    """A question about an existing run uses the 'run' tool -- the agent loads
    the recorded diagnosis + evidence instead of launching a fresh diagnosis."""
    run_id = "bcf28c2bc94448919445af3b2e66fdcc"
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "diagnosis.json").write_text(json.dumps({
        "state": "fault",
        "diagnosis": "Bianca IO board CPLD logged a corrected warning on "
                     "CPLD_97_WARNING.",
        "confidence": 0.71,
        "subsystems_considered": ["memory", "bmc"],
        "actions": [],
        "references": [],
        "evidence": [{
            "mnemonic": "CPLD_97_WARNING", "raw_hex": "0x1",
            "decoded_fields": [{"name": "warn", "raw_value": "1",
                                "meaning": "warning set"}],
            "unknown": False,
        }],
        "unknown_registers": [],
    }), encoding="utf-8")

    holder = {}

    def on_session(session):
        holder["session"] = session

    class RunLLM:
        def __init__(self):
            self.calls = 0
            self.seen = []

        def chat_json(self, messages):
            self.calls += 1
            self.seen.append(messages)
            if self.calls == 1:
                return {"say": "Loading that run's recorded diagnosis.",
                        "tool": "run", "run": run_id}
            return {"say": "Yes -- that run logged a Bianca CPLD warning on "
                           "CPLD_97_WARNING; it is consistent with a Bianca "
                           "error.", "tool": "none"}

    reader = _ScriptedReader([
        f"is the run on harness_runs\\{run_id} related to a bianca error?",
        "/quit"])
    run_llm = RunLLM()
    code = run_session(_session_args(tmp_path), overrides={
        "reader": reader,
        "on_session": on_session,
        "session": FakeSession(),
        "llm": StubLLM(),
        "router_llm": run_llm,
    })
    assert code == 0

    out = capsys.readouterr().out
    assert "loading run" in out
    assert "starting read-only diagnosis" not in out  # never re-diagnosed
    assert "Yes -- that run logged a Bianca CPLD warning" in out

    session = holder["session"]
    assert session.evidence and "CPLD_97_WARNING" in session.evidence
    assert session.runs and session.runs[-1].name == run_id
    assert "harness_runs" in run_llm.seen[0][-1]["content"] or \
        run_id in run_llm.seen[0][-1]["content"]


def test_agent_recovers_from_tool_failure_as_observation(tmp_path, capsys, monkeypatch):
    """A failing tool is an observation, not a dead end: the agent sees the
    error text, answers, and the session does not print 'x ... SessionError'."""
    import harness.operator.repl as repl_mod

    holder = {}

    def on_session(session):
        holder["session"] = session

    def boom(session, subsystems, doc_topics, progress, cancel):
        raise RuntimeError("collector exploded")

    class ToolLLM:
        def __init__(self):
            self.calls = 0

        def chat_json(self, messages):
            self.calls += 1
            if self.calls == 1:
                return {"say": "Running the probe.",
                        "tool": "probe", "subsystems": ["memory"]}
            return {"say": "The probe failed; I can only go on the symptom for "
                           "now.", "tool": "none"}

    reader = _ScriptedReader(["probe the memory controller", "/quit"])
    tool_llm = ToolLLM()
    monkeypatch.setattr(repl_mod, "_run_probes", boom)
    code = run_session(_session_args(tmp_path), overrides={
        "reader": reader,
        "on_session": on_session,
        "session": FakeSession(),
        "llm": StubLLM(),
        "router_llm": tool_llm,
    })
    assert code == 0

    out = capsys.readouterr().out
    assert "probe failed: collector exploded" in out
    assert "The probe failed; I can only go on the symptom for now." in out
    assert "SessionError" not in out


def test_session_parser_requires_inventory():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["session"])


# ---- slash commands for housekeeping (lint / targets / docs) ----

def test_session_slash_lint(tmp_path, capsys):
    reader = _ScriptedReader(["/lint", "/quit"])
    code = run_session(_session_args(tmp_path), overrides={
        "reader": reader, "llm": StubLLM(), "router_llm": StubLLM()})
    assert code == 0
    out = capsys.readouterr().out
    assert "OK: 1 host(s): h1" in out


def test_session_slash_model_by_ident(tmp_path, monkeypatch, capsys):
    """``/model <ident>`` switches the session's reasoning + routing adapters
    and remembers the pick in config/models.yaml for the next run."""
    monkeypatch.chdir(tmp_path)  # keep config/models.yaml inside tmp_path
    holder = {}

    def on_session(session):
        holder["session"] = session

    reader = _ScriptedReader(["/model gemini/gemini-2.5-flash", "/quit"])
    code = run_session(_session_args(tmp_path), overrides={
        "reader": reader,
        "on_session": on_session,
        "llm": StubLLM(),
        "router_llm": StubLLM(),
    })
    assert code == 0
    out = capsys.readouterr().out
    assert "model: gemini/gemini-2.5-flash (active for the next run; saved)" in out

    session = holder["session"]
    assert session.llm_ident == "gemini/gemini-2.5-flash"
    assert session.llm_mode == "gemini"
    from harness.diagnosis.llm import GeminiLLM
    assert isinstance(session.llm, GeminiLLM)

    saved = json.loads((tmp_path / "config" / "models.yaml").read_text(
        encoding="utf-8"))
    assert saved["current"]["provider"] == "gemini"
    assert saved["current"]["model"] == "gemini-2.5-flash"
    assert "llm_ident" in json.loads(
        (next((tmp_path / "sessions").iterdir()) / "session.json").read_text(
            encoding="utf-8"))


def test_session_slash_model_picker(tmp_path, monkeypatch, capsys):
    """``/model`` with no argument opens the arrow-key picker; the chosen model
    is applied and remembered."""
    monkeypatch.chdir(tmp_path)
    holder = {}

    def on_session(session):
        holder["session"] = session

    # defaults order: openai/harness-diag, gemini-2.5-flash, stub, +add
    reader = _ScriptedReader(["/model", "/quit"], keys=["down", "enter"])
    code = run_session(_session_args(tmp_path), overrides={
        "reader": reader,
        "on_session": on_session,
        "llm": StubLLM(),
        "router_llm": StubLLM(),
    })
    assert code == 0
    out = capsys.readouterr().out
    assert "? LLM model (reasoning backend)" in out
    assert "model: gemini/gemini-2.5-flash (active for the next run; saved)" in out
    assert holder["session"].llm_ident == "gemini/gemini-2.5-flash"


def test_session_slash_model_busy_is_rejected(tmp_path, capsys):
    """Switching models mid-run would freeze the REPL picker; it is refused."""
    session = _make_session(tmp_path)

    class _Running:
        alive = True

    session.task = _Running()
    from harness.operator.repl import _slash_model
    _slash_model(session, "gemini/gemini-2.5-flash")
    assert "disabled while a run is active" in capsys.readouterr().out


def test_session_slash_model_unconfigured_builtin_launches_wizard(
        tmp_path, monkeypatch, capsys):
    """``/model`` on the silent-default local row runs the guided setup; the
    configured profile becomes the session's active adapter and is saved."""
    monkeypatch.chdir(tmp_path)
    from harness.config.model_catalog import ModelProfile
    from harness.diagnosis.llm import LocalLLM
    from harness.operator import menu as menu_mod

    holder = {}

    def on_session(session):
        holder["session"] = session

    def fake_wizard(*, provider=None, **kw):
        assert provider == "local"
        assert kw.get("target_label") == "h1"   # context for the golden-server hop
        return ModelProfile(provider="local", model="Qwen2.5-7B-Instruct",
                            url="http://10.0.0.42:8000/v1")

    monkeypatch.setattr(menu_mod, "ask_model_profile", fake_wizard)
    monkeypatch.setattr(menu_mod, "check_profile", lambda *a, **kw: None)

    # defaults order: openai, gemini, local, stub, +add -> pick index 2
    reader = _ScriptedReader(["/model", "/quit"], keys=["down", "down", "enter"])
    code = run_session(_session_args(tmp_path), overrides={
        "reader": reader,
        "on_session": on_session,
        "llm": StubLLM(),
        "router_llm": StubLLM(),
    })
    assert code == 0
    out = capsys.readouterr().out
    assert "model: local/Qwen2.5-7B-Instruct (active for the next run; saved)" in out
    session = holder["session"]
    assert session.llm_ident == "local/Qwen2.5-7B-Instruct"
    assert isinstance(session.llm, LocalLLM)
    assert session.llm.url == "http://10.0.0.42:8000/v1"
    saved = json.loads((tmp_path / "config" / "models.yaml").read_text(
        encoding="utf-8"))
    assert saved["current"]["model"] == "Qwen2.5-7B-Instruct"


def test_set_session_model_tunnel_opens_forward(tmp_path, monkeypatch):
    """A tunnel profile opens a rack-manager forward for the in-session
    adapters, binds them to its local URL, and keeps the spec so spawned
    runs open their own hop."""
    monkeypatch.chdir(tmp_path)
    import harness.engine.tunnel as tunnel_mod
    from harness.config.model_catalog import ModelCatalog, ModelProfile
    from harness.operator import repl as repl_mod

    inv_path = tmp_path / "inv_console.yaml"
    inv_path.write_text(
        "trust_level: lab\n"
        "console_defaults:\n"
        "  address: 192.168.202.51\n"
        "  user: log\n"
        "  identity_vault_path: secret/harness/rackmgr/id_ed25519\n"
        "  known_hosts_path: config/rackmgr_known_hosts\n"
        "  tool: jumpin\n"
        "  trust_level: lab\n"
        "  port: 2200\n"
        "  sudo_vault_path: secret/harness/bmc/sudo\n"
        "hosts: []\n", encoding="utf-8")
    session = _make_session(tmp_path)
    session.inv = load_inventory(inv_path)
    session.inv_path = str(inv_path)

    recorded = {}

    class _StubForward:
        def __init__(self, host, port, domain, store, timeout=30.0, bastion=None):
            recorded["target"] = (host, port)

        def start(self):
            return "http://127.0.0.1:28300/v1"

        def close(self):
            recorded["closed"] = True

    monkeypatch.setattr(tunnel_mod, "LLMForward", _StubForward)

    catalog = ModelCatalog.load(tmp_path / "nope.yaml", inv=session.inv)
    profile = ModelProfile(provider="local", model="Qwen2.5-7B-Instruct",
                           tunnel="10.0.0.42:8000")
    repl_mod._set_session_model(session, catalog, profile)

    assert recorded["target"] == ("10.0.0.42", 8000)
    assert session.llm_tunnel == "10.0.0.42:8000"
    assert session.llm_url == "http://127.0.0.1:28300/v1"
    assert session.llm.url == "http://127.0.0.1:28300/v1"
    saved = json.loads((tmp_path / "config" / "models.yaml").read_text(
        encoding="utf-8"))
    assert saved["current"]["tunnel"] == "10.0.0.42:8000"


def test_set_session_model_tunnel_failure_keeps_current_model(
        tmp_path, monkeypatch, capsys):
    """A failed forward aborts the switch: the previous model stays active."""
    monkeypatch.chdir(tmp_path)
    import harness.engine.tunnel as tunnel_mod
    from harness.config.model_catalog import ModelCatalog, ModelProfile
    from harness.operator import repl as repl_mod

    inv_path = tmp_path / "inv_console.yaml"
    inv_path.write_text(
        "trust_level: lab\n"
        "console_defaults:\n"
        "  address: 192.168.202.51\n"
        "  user: log\n"
        "  identity_vault_path: secret/harness/rackmgr/id_ed25519\n"
        "  known_hosts_path: config/rackmgr_known_hosts\n"
        "  tool: jumpin\n"
        "  trust_level: lab\n"
        "  port: 2200\n"
        "  sudo_vault_path: secret/harness/bmc/sudo\n"
        "hosts: []\n", encoding="utf-8")
    session = _make_session(tmp_path)
    session.inv = load_inventory(inv_path)
    session.inv_path = str(inv_path)

    class _DeadForward:
        def __init__(self, *a, **kw):
            pass

        def start(self):
            raise tunnel_mod.TunnelError("auth", "ssh to rack manager failed")

        def close(self):
            pass

    monkeypatch.setattr(tunnel_mod, "LLMForward", _DeadForward)

    catalog = ModelCatalog.load(tmp_path / "nope.yaml")
    profile = ModelProfile(provider="local", model="Qwen2.5-7B-Instruct",
                           tunnel="10.0.0.42:8000")
    repl_mod._set_session_model(session, catalog, profile)

    out = capsys.readouterr().out
    assert "switch aborted" in out
    assert isinstance(session.llm, StubLLM)       # unchanged
    assert session.llm_tunnel is None
    assert not (tmp_path / "config" / "models.yaml").exists()  # nothing saved


def test_set_session_model_tunnel_needs_console_defaults(tmp_path, capsys):
    """A tunnel profile in an inventory without console_defaults is refused
    (the rack-manager SSH hop is the only path to nodes)."""
    from harness.config.model_catalog import ModelCatalog, ModelProfile
    from harness.operator import repl as repl_mod

    session = _make_session(tmp_path)
    catalog = ModelCatalog.load(tmp_path / "nope.yaml")
    profile = ModelProfile(provider="local", model="Qwen2.5-7B-Instruct",
                           tunnel="10.0.0.42:8000")
    repl_mod._set_session_model(session, catalog, profile)
    assert "console_defaults" in capsys.readouterr().out
    assert isinstance(session.llm, StubLLM)       # unchanged


def test_session_slash_model_served_mismatch_switch(tmp_path, monkeypatch,
                                                    capsys):
    """When the picked model id is not served, the probe's one-key suggestion
    replaces it (e.g. harness-diag -> the actually served Qwen id)."""
    monkeypatch.chdir(tmp_path)
    from harness.operator import menu as menu_mod

    holder = {}

    def on_session(session):
        holder["session"] = session

    suggestions = iter(["Qwen2.5-7B-Instruct"])

    def fake_check(profile, **kw):
        return next(suggestions)

    monkeypatch.setattr(menu_mod, "check_profile", fake_check)

    reader = _ScriptedReader(["/model local/harness-diag", "/quit"])
    code = run_session(_session_args(tmp_path), overrides={
        "reader": reader,
        "on_session": on_session,
        "llm": StubLLM(),
        "router_llm": StubLLM(),
    })
    assert code == 0
    out = capsys.readouterr().out
    assert out.count("model: local/Qwen2.5-7B-Instruct (active for the next "
                     "run; saved)") == 1
    assert holder["session"].llm_ident == "local/Qwen2.5-7B-Instruct"


def test_session_slash_targets_round_trip(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)  # keep config/targets.yaml inside tmp_path
    reader = _ScriptedReader([
        "/targets ls",
        "/targets add t1 --rack Q1 --cable 1",
        "/targets ls",
        "/targets rm t1",
        "/targets ls",
        "/quit",
    ])
    code = run_session(_session_args(tmp_path), overrides={
        "reader": reader, "llm": StubLLM(), "router_llm": StubLLM()})
    assert code == 0
    out = capsys.readouterr().out
    assert "no targets registered" in out
    assert "added target 't1' -> Q1-cable1" in out
    assert "t1  rack Q1, cable 1" in out
    assert "removed target 't1'" in out
    assert "no targets registered" in out


def test_session_slash_targets_bad_flag_usage(tmp_path, capsys):
    reader = _ScriptedReader(["/targets add t1 --bogus x", "/quit"])
    code = run_session(_session_args(tmp_path), overrides={
        "reader": reader, "llm": StubLLM(), "router_llm": StubLLM()})
    assert code == 0
    assert "unknown flag" in capsys.readouterr().out


def test_session_slash_docs_ls_empty(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)  # resolve harness_docs/ to an empty dir
    reader = _ScriptedReader(["/docs ls", "/quit"])
    code = run_session(_session_args(tmp_path), overrides={
        "reader": reader, "llm": StubLLM(), "router_llm": StubLLM()})
    assert code == 0
    assert "empty doc library" in capsys.readouterr().out


def test_session_slash_docs_add_missing_file(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    reader = _ScriptedReader(["/docs add missing.pdf", "/docs rm nope.pdf", "/quit"])
    code = run_session(_session_args(tmp_path), overrides={
        "reader": reader, "llm": StubLLM(), "router_llm": StubLLM()})
    assert code == 0
    out = capsys.readouterr().out
    assert "skip missing.pdf: not a file" in out
    assert "x not in library: 'nope.pdf'" in out


# ---- zero-YAML message-driven targeting ----

_CONSOLE_DEFAULTS_INVENTORY = (
    "trust_level: lab\n"
    "llm:\n"
    "  provider: stub\n"
    "console_defaults:\n"
    "  address: 192.168.202.51\n"
    "  user: log\n"
    "  identity_vault_path: secret/harness/rackmgr/id_ed25519\n"
    "  known_hosts_path: config/rackmgr_known_hosts\n"
    "  tool: jumpin\n"
    "  trust_level: lab\n"
    "  port: 2200\n"
    "  sudo_vault_path: secret/harness/bmc/sudo\n"
    "hosts: []\n"
)


def _console_defaults_session_args(tmp_path, **kw):
    path = tmp_path / "inventory.yaml"
    path.write_text(_CONSOLE_DEFAULTS_INVENTORY, encoding="utf-8")
    argv = ["session", "--inventory", str(path),
            "--out-dir", str(tmp_path / "runs"),
            "--session-dir", str(tmp_path / "sessions"),
            "--llm", "stub"]
    return build_parser().parse_args(argv)


class _FakeConsoleRunner:
    """Stand-in for the serial console runner: canned output per argv."""

    is_console = True

    def __init__(self):
        self.calls = []

    def execute(self, argv, timeout=300.0):
        result = CommandResult(
            argv=list(argv),
            stdout=OUTPUT.get(" ".join(argv), ""),
            stderr="", exit_code=0, elapsed_ms=1)
        self.calls.append(result)
        return result


def test_session_message_names_rack_cable_target(tmp_path, capsys):
    holder = {}

    def on_session(session):
        holder["session"] = session

    def yield_when():
        return (holder["session"].awaiting
                and not holder["session"].answered)

    reader = _ScriptedReader([
        "Diagnose the server in Q61 Cable 8",
        "DIMM was reseated already, no change",
        "/quit",
    ], yield_when=yield_when)
    console_runner = _FakeConsoleRunner()
    code = run_session(_console_defaults_session_args(tmp_path), overrides={
        "reader": reader,
        "on_session": on_session,
        "console_runner": console_runner,
        "llm": StubLLM(),
        "router_llm": _NoChatLLM(),  # keyword fallback exercises targeting
    })
    assert code == 0

    session = holder["session"]
    assert session.target.rack == "Q61"
    assert session.target.cable == "8"
    assert session.target_label == "Q61-cable8"

    out = capsys.readouterr().out
    assert "active target: Q61-cable8 (console)" in out
    assert "diagnosis complete" in out
    calls = [" ".join(c.argv) for c in console_runner.calls]
    assert "sudo -S ipmitool fru print" in calls     # FRU model detection on BMC console
    assert not any("/bin/dmidecode" in c for c in calls)

    run_dir = next((tmp_path / "runs").iterdir())
    audit = AuditLog(run_dir / "audit.jsonl")
    run_start = next(e.payload for e in audit.read() if e.kind == "run_start")
    assert run_start["target"] == "console"
    assert run_start["rack"] == "Q61" and run_start["cable"] == "8"


def test_session_message_names_ip_target(tmp_path, capsys):
    holder = {}

    def on_session(session):
        holder["session"] = session

    def yield_when():
        return (holder["session"].awaiting
                and not holder["session"].answered)

    store = MemorySecretStore({
        "secret/harness/ssh/10.0.0.50": b"-----BEGIN OPENSSH PRIVATE KEY-----\nAAAA\n"
                                        b"-----END OPENSSH PRIVATE KEY-----\n"})
    reader = _ScriptedReader([
        "Diagnose 10.0.0.50",
        "DIMM was reseated already, no change",
        "/quit",
    ], yield_when=yield_when)
    code = run_session(_console_defaults_session_args(tmp_path), overrides={
        "reader": reader,
        "on_session": on_session,
        "session": FakeSession(),
        "store": store,
        "llm": StubLLM(),
        "router_llm": _NoChatLLM(),  # keyword fallback exercises targeting
    })
    assert code == 0

    session = holder["session"]
    assert session.target.ip == "10.0.0.50"
    assert session.target_label == "10.0.0.50"
    assert "active target: 10.0.0.50 (ssh)" in capsys.readouterr().out

    run_dir = next((tmp_path / "runs").iterdir())
    audit = AuditLog(run_dir / "audit.jsonl")
    run_start = next(e.payload for e in audit.read() if e.kind == "run_start")
    assert run_start["target"] == "ssh"
    assert run_start["ip"] == "10.0.0.50"


def test_session_use_accepts_rack_cable(tmp_path, capsys):
    reader = _ScriptedReader(["/use Q61 Cable 8", "/hosts", "/quit"])
    code = run_session(_console_defaults_session_args(tmp_path), overrides={
        "reader": reader, "llm": StubLLM(), "router_llm": StubLLM()})
    assert code == 0
    out = capsys.readouterr().out
    assert "Q61-cable8  (dynamic target, no YAML)" in out


def test_session_message_unknown_target_shows_error_not_crash(tmp_path, capsys):
    holder = {}

    def on_session(session):
        holder["session"] = session

    def yield_when():
        return (holder["session"].awaiting
                and not holder["session"].answered)

    reader = _ScriptedReader([
        "Diagnose the server in Q99 Cable 99",
        "DIMM was reseated already, no change",
        "/quit",
    ], yield_when=yield_when)
    code = run_session(_session_args(tmp_path), overrides={
        "reader": reader,
        "on_session": on_session,
        "session": FakeSession(),
        "llm": StubLLM(),
        "router_llm": _NoChatLLM(),  # keyword fallback exercises targeting
    })
    assert code == 0
    out = capsys.readouterr().out
    assert "x target:" in out
    assert "console_defaults" in out
    # the active target was NOT switched: the default named host stays active
    # (startup banner labels it `target`; _set_target echoes `active target:`)
    assert "h1 (named)" in out


# ---- file tool ----

def test_tool_file_missing_path_is_observation(tmp_path):
    session = _make_session(tmp_path)
    out = _tool_file(session, ChatTurn(tool="file", path="does/not/exist.log"))
    assert "(no such file" in out


def test_tool_file_missing_path_field_is_observation(tmp_path):
    session = _make_session(tmp_path)
    assert "(file path missing)" in _tool_file(session, ChatTurn(tool="file"))


def test_tool_file_directory_is_observation(tmp_path):
    session = _make_session(tmp_path)
    out = _tool_file(session, ChatTurn(tool="file", path=str(tmp_path)))
    assert "(not a file" in out


def test_tool_file_reads_text_content(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("DIMM A2 has been swapped twice\nPOST hangs after MRC init\n",
                 encoding="utf-8")
    session = _make_session(tmp_path)
    out = _tool_file(session, ChatTurn(tool="file", path=str(f)))
    assert "notes.txt" in out
    assert "2 lines; content:" in out
    assert "swapped twice" in out


FAT_LOG = (
    "INFO :2026-08-11 22:01:34 || Model     : T6T\n"
    "INFO :2026-08-11 22:01:34 || Station   : FAT test start\n"
    "ERROR:2026-08-11 22:02:01 || [FAIL][P02002001@PCIe Test Fail]\n"
    "ERROR:2026-08-11 22:02:01 || PCIe compare check test Failed!\n"
)


def test_tool_file_parses_fat_log_and_matches_prior_cases(tmp_path):
    """A referenced FAT log gets structured parsing AND prior verified fixes
    from the case store (fleet learning) in the same observation."""
    from harness.diagnosis.case_store import CaseStore
    from harness.diagnosis.schema import CaseOutcome

    cases = tmp_path / "cases"
    case = CaseOutcome(
        run_id="0" * 32, target_id="other", symptom="PCIe link down in FAT",
        actions_recommended=["1. Reseat the GPU riser"],
        actions_taken=["1. Reseated the GPU riser"],
        outcome="fixed", llm_ident="stub",
        evidence_summary=[], cited_titles=[],
        test_log_failures=["P02002001@PCIe Test Fail"],
    )
    CaseStore(cases).save(case)

    log = tmp_path / "fat.log"
    log.write_text(FAT_LOG, encoding="utf-8")
    session = _make_session(tmp_path)
    session.cases_dir = str(cases)
    out = _tool_file(session, ChatTurn(tool="file", path=str(log)))
    assert "parsed as a factory test log" in out
    assert "P02002001@PCIe Test Fail" in out
    assert "prior verified fixes" in out
    assert "Reseated the GPU riser" in out


def test_tool_file_fat_log_without_cases_omits_learning(tmp_path):
    log = tmp_path / "fat.log"
    log.write_text(FAT_LOG, encoding="utf-8")
    session = _make_session(tmp_path)  # empty case store
    session.cases_dir = str(tmp_path / "cases")
    out = _tool_file(session, ChatTurn(tool="file", path=str(log)))
    assert "parsed as a factory test log" in out
    assert "prior verified fixes" not in out


def test_agent_file_tool_wired_in_repl(tmp_path, capsys):
    """The chat/debug agent can answer from a referenced file in one turn."""
    log = tmp_path / "fat.log"
    log.write_text(FAT_LOG, encoding="utf-8")
    reader = _ScriptedReader([f"read {log} and tell me what failed", "/quit"])
    code = run_session(_session_args(tmp_path), overrides={
        "reader": reader,
        "session": FakeSession(),
        "llm": StubLLM(),
        "router_llm": _AgentLLM({"say": "Reading the referenced log.",
                                 "tool": "file", "path": str(log)}),
    })
    assert code == 0
    out = capsys.readouterr().out
    assert "reading file" in out
    assert "parsed as a factory test log" in out


# ---- chat mode REPL ----

def _chat_args(tmp_path):
    return build_parser().parse_args([
        "chat",
        "--out-dir", str(tmp_path / "runs"),
        "--session-dir", str(tmp_path / "sessions"),
        "--llm", "stub"])


def test_chat_repl_starts_without_inventory(tmp_path, capsys):
    """`harness chat` needs no inventory and never resolves a target."""
    reader = _ScriptedReader(["/quit"])
    code = run_session(_chat_args(tmp_path), overrides={"reader": reader})
    assert code == 0
    out = capsys.readouterr().out
    assert "harness chat" in out
    assert "no target connected" in out
    assert "inventory" not in out


def test_chat_repl_blocks_debug_slash_commands(tmp_path, capsys):
    reader = _ScriptedReader(["/use h1", "/hosts", "/quit"])
    run_session(_chat_args(tmp_path), overrides={"reader": reader})
    out = capsys.readouterr().out
    assert "debug-only" in out
    assert "no target is connected" in out
    assert "h1  lab" not in out


def test_chat_repl_agent_reads_file_and_no_target_tools(tmp_path, capsys):
    """In chat mode the agent can read a referenced file; a diagnose request
    via the keyword fallback is refused (no target)."""
    log = tmp_path / "fat.log"
    log.write_text("plain note about a fan alarm\n", encoding="utf-8")
    reader = _ScriptedReader(["diagnose the ECC error on 10.0.0.9", "/quit"])
    run_session(_chat_args(tmp_path), overrides={
        "reader": reader, "llm": StubLLM(), "router_llm": _NoChatLLM()})
    out = capsys.readouterr().out
    assert "chat-only" in out          # fallback refuses the diagnose intent
    assert "cannot connect to a machine" in out


def test_chat_repl_agent_uses_file_tool(tmp_path, capsys):
    log = tmp_path / "fan.log"
    log.write_text("FAN1_CTRL fault logged at boot\n", encoding="utf-8")
    reader = _ScriptedReader([f"check {log}", "/quit"])
    run_session(_chat_args(tmp_path), overrides={
        "reader": reader, "llm": StubLLM(),
        "router_llm": _AgentLLM({"say": "Reading the file now.",
                                 "tool": "file", "path": str(log)})})
    out = capsys.readouterr().out
    assert "reading file" in out
    assert "FAN1_CTRL fault logged at boot" in out


def test_chat_session_persists_mode(tmp_path):
    """session.json records mode=chat; /resume in chat mode ignores targets."""
    holder = {}

    def on_session(session):
        holder["s"] = session

    reader = _ScriptedReader(["hello there", "/quit"])
    run_session(_chat_args(tmp_path), overrides={
        "reader": reader, "on_session": on_session,
        "llm": StubLLM(), "router_llm": _NoChatLLM()})
    session_dir = next((tmp_path / "sessions").iterdir())
    payload = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    assert payload["mode"] == "chat"
    assert "hello there" in [e.get("content") for e in payload["transcript"]]


# ---- chat mode: tunnel-backed models (rack-manager hop) ----

_TUNNEL_MODELS = {
    "current": {"provider": "local", "model": "Qwen/Qwen3.8-27B",
                "tunnel": "10.0.126.15:8000"},
}

_HOP_INVENTORY = (
    "trust_level: lab\n"
    "console_defaults:\n"
    "  address: 192.168.202.51\n"
    "  user: log\n"
    "  identity_vault_path: secret/harness/rackmgr/id_ed25519\n"
    "hosts: []\n"
)


def _chat_hop_env(tmp_path, monkeypatch) -> None:
    """cwd with a persisted tunnel-backed current model (ModelCatalog reads
    ``config/models.yaml`` from the working directory)."""
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "config"
    config.mkdir()
    (config / "models.yaml").write_text(json.dumps(_TUNNEL_MODELS), encoding="utf-8")


def _chat_args_in(tmp_path, *extra):
    return build_parser().parse_args([
        "chat",
        "--out-dir", str(tmp_path / "runs"),
        "--session-dir", str(tmp_path / "sessions"),
        *extra,
    ])


class _FakeForward:
    """Forward stand-in: _repl_loop closes it when the session ends."""

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_chat_tunnel_profile_loads_inventory_for_hop(tmp_path, monkeypatch, capsys):
    """A tunnel-backed remembered model: chat loads the inventory (explicit
    --inventory) so the rack-manager hop can open, while the session itself
    stays target-less."""
    _chat_hop_env(tmp_path, monkeypatch)
    inv_path = tmp_path / "inv.yaml"
    inv_path.write_text(_HOP_INVENTORY, encoding="utf-8")
    captured: dict = {}

    def fake_prepare(args, inv, store):
        captured["inv"] = inv
        return _FakeForward()

    monkeypatch.setattr("harness.operator.cli._prepare_llm_endpoint", fake_prepare)
    code = run_session(_chat_args_in(tmp_path, "--inventory", str(inv_path)),
                       overrides={"reader": _ScriptedReader(["/quit"]),
                                  "llm": StubLLM(), "router_llm": _NoChatLLM()})
    assert code == 0
    assert captured["inv"].console_defaults.address == "192.168.202.51"
    assert "no target connected" in capsys.readouterr().out


def test_chat_tunnel_profile_without_inventory_prints_guidance(
        tmp_path, monkeypatch, capsys):
    """No inventory anywhere: chat fails fast with actionable guidance, not
    the raw --llm-tunnel inventory error."""
    _chat_hop_env(tmp_path, monkeypatch)
    code = run_session(_chat_args_in(tmp_path),
                       overrides={"reader": _ScriptedReader(["/quit"])})
    assert code == 2
    err = capsys.readouterr().err
    assert "--inventory" in err
    assert "harness debug" in err
    assert "rack-manager tunnel" in err


def test_chat_tunnel_profile_ambiguous_inventory_lists_candidates(
        tmp_path, monkeypatch, capsys):
    """Several inventories under config/: chat lists them and asks for an
    explicit --inventory instead of guessing."""
    _chat_hop_env(tmp_path, monkeypatch)
    for name in ("a.yaml", "b.yaml"):
        (tmp_path / "config" / name).write_text("trust_level: lab\nhosts: []\n",
                                                encoding="utf-8")
    code = run_session(_chat_args_in(tmp_path),
                       overrides={"reader": _ScriptedReader(["/quit"])})
    assert code == 2
    err = capsys.readouterr().err
    assert "a.yaml" in err and "b.yaml" in err
    assert "--inventory" in err


def test_chat_tunnel_inventory_without_console_block_guidance(
        tmp_path, monkeypatch, capsys):
    """An inventory without llm_console/console_defaults cannot supply the
    rack-manager hop: fail fast with guidance."""
    _chat_hop_env(tmp_path, monkeypatch)
    inv_path = tmp_path / "inv.yaml"
    inv_path.write_text("trust_level: lab\nhosts: []\n", encoding="utf-8")
    code = run_session(_chat_args_in(tmp_path, "--inventory", str(inv_path)),
                       overrides={"reader": _ScriptedReader(["/quit"])})
    assert code == 2
    assert "no llm_console (or console_defaults)" in capsys.readouterr().err


def test_chat_llm_url_overrides_tunnel_profile_without_hop(
        tmp_path, monkeypatch, capsys):
    """--llm-url is authoritative: the profile tunnel is dropped, no inventory
    is loaded, and the session stays target-less."""
    _chat_hop_env(tmp_path, monkeypatch)
    monkeypatch.setattr("harness.operator.cli.list_models", lambda *a, **k: [])
    captured: dict = {}

    def fake_prepare(args, inv, store):
        captured["inv"] = inv
        return _FakeForward()

    monkeypatch.setattr("harness.operator.cli._prepare_llm_endpoint", fake_prepare)
    code = run_session(
        _chat_args_in(tmp_path, "--llm-url", "http://127.0.0.1:8000/v1"),
        overrides={"reader": _ScriptedReader(["/quit"]), "llm": StubLLM(),
                   "router_llm": _NoChatLLM()})
    assert code == 0
    from harness.operator.repl import _NoInventory
    assert isinstance(captured["inv"], _NoInventory)
