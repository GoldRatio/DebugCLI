"""Interactive menu: key translation, selection logic, inventory discovery,
wizard dispatch, and the bare-`harness` default."""

from pathlib import Path
from types import SimpleNamespace

from harness.config.inventory_lint import load_inventory
from harness.config.vault import MemorySecretStore
from harness.operator import cli as cli_mod
from harness.operator.cli import (
    _baselines,
    _discover_inventory,
    _menu_runs,
    _pick_inventory,
    _run_wizard_sub,
    run_menu,
)
from harness.operator.menu import LineReader, ask_text, decode_escape, select

_INVENTORY = (
    "trust_level: lab\n"
    "llm:\n"
    "  provider: stub\n"
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


def _write_inventory(tmp_path, name="inventory.yaml", body=_INVENTORY):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


class _KeyReader:
    """Raw-mode stand-in for LineReader: scripts key events."""

    def __init__(self, keys):
        self.keys = list(keys)
        self.raw = True

    def read_key(self, timeout):
        if not self.keys:
            raise KeyboardInterrupt  # unscripted input cancels the menu
        return self.keys.pop(0)


# ---- key decoding ----

def test_decode_escape():
    assert decode_escape(b"[A") == "up"
    assert decode_escape(b"[B") == "down"
    assert decode_escape(b"[C") == "right"
    assert decode_escape(b"[D") == "left"
    assert decode_escape(b"[H") == "up"      # home key
    assert decode_escape(b"[F") == "down"    # end key
    assert decode_escape(b"") is None
    assert decode_escape(b"[Z") is None      # unknown sequence -> lone ESC


def test_line_reader_token_mapping():
    assert LineReader._token("\r") == "enter"
    assert LineReader._token("\n") == "enter"
    assert LineReader._token("\x08") == "backspace"
    assert LineReader._token("\x7f") == "backspace"
    assert LineReader._token("\x03") == "ctrl_c"
    assert LineReader._token("\x04") == "ctrl_d"
    assert LineReader._token("\t") == "tab"
    assert LineReader._token("x") == "x"


# ---- selection logic ----

def test_select_raw_arrows_enter(capsys):
    idx = select("Pick", ["a", "b", "c"],
                 reader=_KeyReader(["down", "down", "enter"]))
    assert idx == 2
    out = capsys.readouterr().out
    assert "? Pick: c" in out


def test_select_raw_wraps_around(capsys):
    idx = select("Pick", ["a", "b"],
                 reader=_KeyReader(["up", "enter"]))  # up wraps to last
    assert idx == 1


def test_select_raw_type_to_filter(capsys):
    idx = select("Pick", ["dog house", "cat", "cog"],
                 reader=_KeyReader(["d", "o", "g", "enter"]))
    assert idx == 0  # "dog" filters down to "dog house"


def test_select_raw_backspace_restores(capsys):
    options = ["cat", "cog", "dog"]
    idx = select("Pick", options,
                 reader=_KeyReader(["c", "a", "backspace", "backspace",
                                    "c", "enter"]))
    assert idx == 0  # "ca" -> [cat]; backspace x2 -> all; "c" -> [cat, cog]
    assert options[idx] == "cat"


def test_select_raw_no_matches_blocks_enter(capsys):
    reader = _KeyReader(["z", "z", "enter", "backspace", "backspace", "enter"])
    idx = select("Pick", ["cat", "dog"], reader=reader)
    assert idx == 0  # "zz" matches nothing; enter blocked; clear -> all


def test_select_raw_backspace_redraw_stays_anchored(capsys):
    # Heights: initial=4 ("cat","dog","cog"), "c"=4 (2 matches+footer),
    # "ca"=3 ([cat]+footer), backspace -> "c"=4. The backspace redraw must move
    # up the PREVIOUS block height (3), not the new one (4), or the menu drifts
    # upward one row per keystroke.
    idx = select("Pick", ["cat", "dog", "cog"],
                 reader=_KeyReader(["c", "a", "backspace", "enter"]))
    assert idx == 0  # "cat" selected
    out = capsys.readouterr().out
    assert out.count("\x1b[3A") == 1  # only the backspace redraw moves up 3
    assert out.count("\x1b[4A") == 3  # "c", "ca" redraws + enter clear


def test_select_raw_esc_cancels(capsys):
    assert select("Pick", ["a"], reader=_KeyReader(["esc"])) is None


def test_select_numbered_fallback(monkeypatch, capsys):
    answers = iter(["x", "2"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    assert select("Pick", ["a", "b", "c"]) == 1  # "x" rejected, "2" picked


def test_select_numbered_cancel(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _prompt: "q")
    assert select("Pick", ["a", "b"]) is None


# ---- inventory discovery ----

def test_discover_inventory_filters_non_inventories(tmp_path, monkeypatch):
    _write_inventory(tmp_path, "config/rack.yaml", _CONSOLE_DEFAULTS_INVENTORY)
    _write_inventory(tmp_path, "inventory.yaml", _INVENTORY)
    _write_inventory(tmp_path, "config/targets.yaml", "targets:\n  - alias: x\n    rack: Q1\n    cable: 1\n")
    _write_inventory(tmp_path, "config/broken.yaml", "::: not yaml\n")
    monkeypatch.chdir(tmp_path)

    found = _discover_inventory()
    assert {p.name for p in found} == {"rack.yaml", "inventory.yaml"}


def test_pick_inventory_explicit_wins(tmp_path):
    inv = _write_inventory(tmp_path)
    args = SimpleNamespace(inventory=str(inv))
    assert _pick_inventory(args) == inv


def test_pick_inventory_single_candidate(tmp_path, monkeypatch, capsys):
    _write_inventory(tmp_path, "config/rack.yaml", _CONSOLE_DEFAULTS_INVENTORY)
    monkeypatch.chdir(tmp_path)
    assert _pick_inventory(SimpleNamespace(inventory=None)) == Path("config/rack.yaml")
    assert "inventory:" in capsys.readouterr().out


def test_pick_inventory_none_found(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert _pick_inventory(SimpleNamespace(inventory=None)) is None
    assert "no inventory found" in capsys.readouterr().err


# ---- wizard flow ----

def _menu_args(**kw):
    base = {
        "inventory": None, "secret_dir": None, "docs_lib": None, "docs_dir": None,
        "parts_csv": None, "console": False, "out_dir": "harness_runs",
        "session_dir": "harness_runs/sessions", "llm": None,
        "targets_file": "config/targets.yaml",
    }
    base.update(kw)
    return SimpleNamespace(**base)


def _scripted_select(seq):
    """menu.select that returns scripted indices, then the last option."""
    seq = iter(seq)

    def _select(title, options, **kw):
        try:
            return next(seq)
        except StopIteration:
            return len(options) - 1

    return _select


def test_run_menu_lint_then_quit(tmp_path, monkeypatch, capsys):
    _write_inventory(tmp_path)
    monkeypatch.chdir(tmp_path)
    from harness.operator import menu as menu_mod
    # main menu: pick lint, then quit (indices resolved by action key so menu
    # reordering never breaks this test)
    keys = [k for k, _ in cli_mod._MAIN_ACTIONS]
    monkeypatch.setattr(menu_mod, "select",
                        _scripted_select([keys.index("lint"), keys.index("quit")]))

    assert run_menu(_menu_args()) == 0
    out = capsys.readouterr().out
    assert "harness menu" in out
    assert "OK: 1 host(s): h1" in out


def test_run_menu_diagnose_builds_argv(tmp_path, monkeypatch, capsys):
    _write_inventory(tmp_path)
    monkeypatch.chdir(tmp_path)
    from harness.operator import menu as menu_mod
    built = []

    def fake_sub(argv):
        built.append(argv)
        return 0

    monkeypatch.setattr(menu_mod, "select", _scripted_select([1, 0, 9]))
    # The symptom prompt is optional: an empty answer falls back to the
    # default evidence-driven symptom.
    monkeypatch.setattr(menu_mod, "ask_text", lambda prompt, **kw: "")
    monkeypatch.setattr(cli_mod, "_run_wizard_sub", fake_sub)

    assert run_menu(_menu_args()) == 0
    assert built and built[0][:4] == ["diagnose", "--inventory", "inventory.yaml",
                                      "--symptom"]
    assert built[0][built[0].index("--host") + 1] == "h1"
    # empty symptom -> the default evidence-driven symptom is supplied
    assert built[0][built[0].index("--symptom") + 1] == cli_mod._MENU_DIAGNOSE_SYMPTOM
    assert "diagnosing from live evidence" in capsys.readouterr().out


def test_run_menu_diagnose_uses_typed_symptom(tmp_path, monkeypatch):
    _write_inventory(tmp_path)
    monkeypatch.chdir(tmp_path)
    from harness.operator import menu as menu_mod
    built = []

    def fake_sub(argv):
        built.append(argv)
        return 0

    monkeypatch.setattr(menu_mod, "select", _scripted_select([1, 0, 9]))
    monkeypatch.setattr(menu_mod, "ask_text",
                        lambda prompt, **kw: "amber light on power rail")
    monkeypatch.setattr(cli_mod, "_run_wizard_sub", fake_sub)

    assert run_menu(_menu_args()) == 0
    idx = built[0].index("--symptom")
    assert built[0][idx + 1] == "amber light on power rail"


# ---- runs inspection menu ----

def _fake_run_dir(base, run_id, files: dict):
    run = base / "harness_runs" / run_id
    run.mkdir(parents=True, exist_ok=True)
    for rel, body in files.items():
        path = run / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    return base


def test_menu_runs_verdict_view(tmp_path, monkeypatch, capsys):
    from harness.operator import menu as menu_mod
    base = _fake_run_dir(tmp_path, "abc123", {
        "diagnosis.json": '{"state": "healthy", "confidence": 0.8}'})
    args = SimpleNamespace(out_dir=str(base / "harness_runs"))
    # 0 = run, 0 = verdict, 4 = back
    monkeypatch.setattr(menu_mod, "select", _scripted_select([0, 0, 4]))
    assert _menu_runs(args) == 0
    out = capsys.readouterr().out
    assert "healthy" in out


def test_menu_runs_prompt_turns_view(tmp_path, monkeypatch, capsys):
    from harness.operator import menu as menu_mod
    base = _fake_run_dir(tmp_path, "def456", {
        "prompt_turns.jsonl": '{"turn": 1, "messages": [{"role": "user", '
                              '"content": "evidence block"}]}\n'})
    args = SimpleNamespace(out_dir=str(base / "harness_runs"))
    # 0 = run, 2 = prompt, 0 = turn 1, 4 = back
    monkeypatch.setattr(menu_mod, "select", _scripted_select([0, 2, 0, 4]))
    assert _menu_runs(args) == 0
    assert "evidence block" in capsys.readouterr().out


def test_menu_runs_dumps_view(tmp_path, monkeypatch, capsys):
    from harness.operator import menu as menu_mod
    base = _fake_run_dir(tmp_path, "ghi789", {
        "dumps/ipmi_0.txt": "sensor data here"})
    args = SimpleNamespace(out_dir=str(base / "harness_runs"))
    # 0 = run, 3 = dumps, 0 = first dump file, 4 = back
    monkeypatch.setattr(menu_mod, "select", _scripted_select([0, 3, 0, 4]))
    assert _menu_runs(args) == 0
    assert "sensor data here" in capsys.readouterr().out


def test_menu_runs_empty(tmp_path, monkeypatch, capsys):
    from harness.operator import menu as menu_mod
    args = SimpleNamespace(out_dir=str(tmp_path / "harness_runs"))
    monkeypatch.setattr(menu_mod, "select", _scripted_select([0]))
    assert _menu_runs(args) == 0
    assert "no runs yet" in capsys.readouterr().out


def test_menu_runs_missing_artifact(tmp_path, monkeypatch, capsys):
    from harness.operator import menu as menu_mod
    base = _fake_run_dir(tmp_path, "jkl012", {})
    args = SimpleNamespace(out_dir=str(base / "harness_runs"))
    # 0 = run, 2 = prompt (missing artifact), 4 = back
    monkeypatch.setattr(menu_mod, "select", _scripted_select([0, 2, 4]))
    assert _menu_runs(args) == 0
    assert "no prompt artifact" in capsys.readouterr().out


def test_pick_target_named_host(tmp_path, monkeypatch, capsys):
    inv = load_inventory(_write_inventory(tmp_path))
    from harness.operator import menu as menu_mod
    monkeypatch.setattr(menu_mod, "select", lambda title, options, **kw: 0)
    spec = cli_mod._pick_target(inv, MemorySecretStore(), _menu_args(),
                                console=False)
    assert spec.name == "h1"
    assert "target: h1 (named)" in capsys.readouterr().out


def test_pick_target_rack_cable_requires_both(tmp_path, monkeypatch, capsys):
    inv = load_inventory(_write_inventory(tmp_path, body=_CONSOLE_DEFAULTS_INVENTORY))
    from harness.operator import menu as menu_mod
    options_list = []

    def fake_select(title, options, **kw):
        options_list.append(options)
        # "Rack + cable" is the first option after the (empty) host list
        return options.index("Rack + cable (console target, zero-YAML)")

    monkeypatch.setattr(menu_mod, "select", fake_select)
    monkeypatch.setattr(menu_mod, "ask_text",
                        lambda prompt, **kw: "" if "Rack" in prompt else "8")
    spec = cli_mod._pick_target(inv, MemorySecretStore(), _menu_args(),
                                console=True)
    assert spec is None  # rack was empty -> cancelled
    assert "rack and cable are both required" in capsys.readouterr().err


def test_pick_target_rack_cable_ok(tmp_path, monkeypatch, capsys):
    inv = load_inventory(_write_inventory(tmp_path, body=_CONSOLE_DEFAULTS_INVENTORY))
    from harness.operator import menu as menu_mod

    def fake_select(title, options, **kw):
        return options.index("Rack + cable (console target, zero-YAML)")

    monkeypatch.setattr(menu_mod, "select", fake_select)
    monkeypatch.setattr(menu_mod, "ask_text",
                        lambda prompt, **kw: "Q61" if "Rack" in prompt else "8")
    spec = cli_mod._pick_target(inv, MemorySecretStore(), _menu_args(),
                                console=True)
    assert spec.rack == "Q61" and spec.cable == "8"
    assert "target: Q61-cable8 (console)" in capsys.readouterr().out


def test_pick_target_bad_ip_rejected(tmp_path, monkeypatch, capsys):
    inv = load_inventory(_write_inventory(tmp_path))
    from harness.operator import menu as menu_mod
    labels = []

    def fake_select(title, options, **kw):
        labels.append(options)
        return options.index("IP address (SSH by address)")

    monkeypatch.setattr(menu_mod, "select", fake_select)
    monkeypatch.setattr(menu_mod, "ask_text", lambda prompt, **kw: "not-an-ip")
    spec = cli_mod._pick_target(inv, MemorySecretStore(), _menu_args(),
                                console=False)
    assert spec is None
    assert "not an IPv4 address" in capsys.readouterr().err


def test_baselines_only_dirs_with_dumps(tmp_path):
    a = tmp_path / "a"
    a.mkdir()
    (a / "dumps.json").write_text("[]", encoding="utf-8")
    b = tmp_path / "b"
    b.mkdir()
    lst = _baselines(tmp_path)
    assert [p.name for p in lst] == ["a"]


def test_run_wizard_sub_lint(tmp_path, capsys):
    inv = _write_inventory(tmp_path)
    assert _run_wizard_sub(["lint", "--inventory", str(inv)]) == 0
    assert "OK:" in capsys.readouterr().out


def test_run_wizard_sub_surfaces_errors(tmp_path, capsys):
    _write_inventory(tmp_path)
    assert _run_wizard_sub(["lint", "--inventory", str(tmp_path / "missing.yaml")]) == 1
    assert "error:" in capsys.readouterr().err


def test_main_no_command_routes_to_menu(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    captured = {}

    def fake_run_menu(args):
        captured["args"] = args
        return 0

    monkeypatch.setattr(cli_mod, "run_menu", fake_run_menu)
    assert cli_mod.main([]) == 0
    assert captured["args"].command is None


# ---- typed-line editing (backspace must redraw with the ACTIVE prompt) ----

def test_poll_backspace_redraws_with_active_prompt(capsys):
    """The redraw after a backspace must reuse the prompt passed to poll, not
    the LineReader default ("harness> ") -- regression: ask_text prompts were
    overwritten with the default prompt on backspace."""
    reader = LineReader()
    reader._raw = True
    reader.read_key = _KeyReader(["Q", "backspace", "Q", "enter"]).read_key
    line = reader.poll(None, prompt="? Rack id (e.g. Q61) ")
    out = capsys.readouterr().out
    assert line == "Q"
    assert "? Rack id (e.g. Q61) " in out
    assert "\x1b[K? Rack id (e.g. Q61) " in out  # redraw clears with active prompt
    assert "\rharness> " not in out


def test_ask_text_backspace_fix_keeps_prompt(capsys, monkeypatch):
    reader = LineReader()
    reader._raw = True
    reader.read_key = _KeyReader(["Q", "Q", "backspace", "1", "enter"]).read_key
    from harness.operator import menu as menu_mod

    monkeypatch.setattr(menu_mod, "LineReader", lambda: reader)
    result = ask_text("Rack id (e.g. Q61)")
    out = capsys.readouterr().out
    assert result == "Q1"
    assert "\x1b[K? Rack id (e.g. Q61) " in out
    assert "\rharness> " not in out


def test_poll_backspace_redraws_wrapped_line(capsys, monkeypatch):
    """When the typed line wraps past the terminal width, the backspace redraw
    must climb back across the wrapped rows and clear exactly that block --
    not re-anchor "\r" on the wrapped row, which copies prompt+text onto a
    fresh row per keystroke."""
    from harness.operator import menu as menu_mod

    monkeypatch.setattr(menu_mod, "_terminal_width", lambda: 12)
    reader = LineReader()
    reader._raw = True
    reader.read_key = _KeyReader(
        list("abcdefghij") + ["k", "backspace", "backspace", "enter"]).read_key
    line = reader.poll(None, prompt="> ")
    out = capsys.readouterr().out
    assert line == "abcdefghi"
    assert "\x1b[1A" in out          # climbed across the wrap
    assert "\x1b[2A" not in out      # never overshoots
    assert out.endswith("> abcdefghi\n")


def test_poll_busy_timeouts_do_not_reprint_prompt(capsys):
    """A background run busy-polls ``poll(0.1)`` at 10Hz with an empty buffer;
    the prompt must not be re-echoed on every timeout -- regression: the REPL
    flooded the terminal with ``harness> harness> ...`` while a task ran."""
    reader = LineReader()
    reader._raw = True
    reader.read_key = lambda timeout: None  # busy poll: always times out
    assert reader.poll(0.1) is None
    assert reader.poll(0.1) is None
    assert reader.poll(0.1) is None
    out = capsys.readouterr().out
    assert out.count("harness> ") == 1
