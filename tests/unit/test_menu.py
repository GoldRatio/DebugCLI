"""Interactive menu: key translation, selection logic, inventory discovery,
wizard dispatch, and the bare-`harness` default."""

import json
import sys
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


# ---- line editor (cursor movement, history, paste, widths) ----

def _editor(keys):
    reader = LineReader()
    reader._raw = True
    reader.read_key = _KeyReader(list(keys)).read_key
    return reader


def test_poll_cursor_left_inserts_middle(capsys):
    reader = _editor(["a", "b", "left", "X", "enter"])
    assert reader.poll(None) == "aXb"


def test_poll_home_end(capsys):
    reader = _editor(["a", "b", "c", "home", "Z", "end", "Y", "enter"])
    assert reader.poll(None) == "ZabcY"


def test_poll_forward_delete(capsys):
    reader = _editor(["a", "b", "c", "home", "delete", "enter"])
    assert reader.poll(None) == "bc"


def test_poll_backspace_midline_keeps_tail(capsys):
    reader = _editor(["a", "b", "c", "left", "backspace", "enter"])
    assert reader.poll(None) == "ac"


def test_poll_ctrl_u_clears_line(capsys):
    reader = _editor(["a", "b", "c", "ctrl_u", "d", "enter"])
    assert reader.poll(None) == "d"


def test_poll_ctrl_k_kills_to_end(capsys):
    reader = _editor(["a", "b", "c", "home", "ctrl_k", "z", "enter"])
    assert reader.poll(None) == "z"


def test_poll_ctrl_w_deletes_word_back(capsys):
    reader = _editor(list("foo bar") + ["ctrl_w", "enter"])
    assert reader.poll(None) == "foo"


def test_poll_history_recall(tmp_path, capsys):
    reader = LineReader()
    reader._raw = True
    script = _KeyReader(["h", "i", "enter"])
    reader.read_key = script.read_key
    assert reader.poll(None) == "hi"
    script.keys += ["up", "enter"]
    assert reader.poll(None) == "hi"          # up recalls the last line
    script.keys += ["down", "enter"]
    assert reader.poll(None) == ""            # down past newest -> empty draft


def test_poll_history_nav_skips_empty_lines():
    reader = LineReader()
    reader._raw = True
    script = _KeyReader(["enter"])            # empty line: NOT recorded
    reader.read_key = script.read_key
    assert reader.poll(None) == ""
    script.keys += ["x", "enter"]
    assert reader.poll(None) == "x"
    script.keys += ["up", "up", "enter"]      # only one entry -> stays on it
    assert reader.poll(None) == "x"


def test_editor_utf8_multibyte_decoded(monkeypatch):
    """POSIX path must assemble continuation bytes instead of per-byte
    errors="replace" (which degraded every non-ASCII keystroke)."""
    reader = LineReader()
    reader._raw = True
    data = "\u00e9".encode("utf-8")  # b"\xc3\xa9"
    rest = iter([data[1:]])
    monkeypatch.setattr(reader, "_read_bytes_posix",
                        lambda count, timeout: next(rest))
    assert reader._decode_utf8(data[:1]) == "\u00e9"


def test_read_key_nt_joins_surrogate_pairs(monkeypatch):
    """Astral chars arrive as UTF-16 surrogate pairs from getwch; joining
    them keeps emoji from landing in the buffer as lone surrogates."""
    import msvcrt

    seq = iter(["\ud83d", "\ude00"])
    monkeypatch.setattr(msvcrt, "kbhit", lambda: True)
    monkeypatch.setattr(msvcrt, "getwch", lambda: next(seq))
    reader = LineReader()
    assert reader.read_key(None) == "\U0001f600"


def test_read_key_nt_drops_stray_surrogates(monkeypatch):
    import msvcrt

    monkeypatch.setattr(msvcrt, "kbhit", lambda: True)
    monkeypatch.setattr(msvcrt, "getwch", lambda: "\udc00")
    reader = LineReader()
    assert reader.read_key(None) == ""


def test_poll_bracketed_paste_flattens_newlines(capsys):
    """Pasted text arrives wrapped in paste_start/paste_end; embedded
    newlines become spaces and submit as ONE line."""
    reader = _editor(["paste_start", "l", "i", "enter", "n", "e",
                      "paste_end", "enter"])
    assert reader.poll(None) == "li ne"


def test_poll_paste_inserts_at_cursor_midline(capsys):
    reader = _editor(["a", "c", "left", "paste_start", "b", "paste_end", "enter"])
    assert reader.poll(None) == "abc"


def test_poll_wide_chars_track_display_columns(capsys, monkeypatch):
    """Wrap math counts COLUMNS: two wide chars fill a 6-col terminal exactly;
    the deferred-wrap cursor means NO climb is emitted there (the old code
    climbed a row too far and drifted upward at exact boundaries)."""
    monkeypatch.setattr("harness.operator.menu._terminal_width", lambda: 6)
    reader = _editor(["\u65e5", "\u672c", "backspace", "\u672c", "backspace",
                      "enter"])
    line = reader.poll(None, prompt="> ")
    out = capsys.readouterr().out
    assert line == "\u65e5"                    # 日本 -> 日 -> 日本 -> 日
    assert "\x1b[1A" not in out                # never climbs past row 0 at this width
    assert "\x1b[2A" not in out


def test_poll_wrapped_line_climbs_across_rows(capsys, monkeypatch):
    monkeypatch.setattr("harness.operator.menu._terminal_width", lambda: 6)
    reader = _editor(["a", "b", "c", "d", "\u65e5", "backspace", "backspace",
                      "backspace", "enter"])
    line = reader.poll(None, prompt="> ")
    out = capsys.readouterr().out
    assert line == "ab"                        # cols: 2+4=6 wide char wraps once
    assert "\x1b[1A" in out


# ---- key decoding ----

def test_decode_escape():
    assert decode_escape(b"[A") == "up"
    assert decode_escape(b"[B") == "down"
    assert decode_escape(b"[C") == "right"
    assert decode_escape(b"[D") == "left"
    assert decode_escape(b"[H") == "home"
    assert decode_escape(b"[F") == "end"
    assert decode_escape(b"[3~") == "delete"
    assert decode_escape(b"[200~") == "paste_start"   # bracketed paste
    assert decode_escape(b"[201~") == "paste_end"
    assert decode_escape(b"OA") == "up"               # SS3 application mode
    assert decode_escape(b"") is None
    assert decode_escape(b"x") is None
    assert decode_escape(b"[Z") is None      # unknown sequence -> lone ESC
    assert decode_escape(b"[99~") is None


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
    # Heights (title + options + status footer): initial=5 ("cat","dog","cog"),
    # "c"=4 (2 matches+footer), "ca"=3 ([cat]+footer), backspace -> "c"=4. The
    # backspace redraw must move up the PREVIOUS block height (3), not the new
    # one (4), or the menu drifts upward one row per keystroke.
    idx = select("Pick", ["cat", "dog", "cog"],
                 reader=_KeyReader(["c", "a", "backspace", "enter"]))
    assert idx == 0  # "cat" selected
    out = capsys.readouterr().out
    assert out.count("\x1b[3A") == 1  # only the backspace redraw moves up 3
    assert out.count("\x1b[4A") == 2  # "ca" redraw + enter clear
    assert out.count("\x1b[5A") == 1  # first "c" redraw climbs the initial 5


def test_select_raw_esc_cancels(capsys):
    assert select("Pick", ["a"], reader=_KeyReader(["esc"])) is None


def test_select_numbered_fallback(monkeypatch, capsys):
    answers = iter(["x", "2"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    assert select("Pick", ["a", "b", "c"]) == 1  # "x" rejected, "2" picked


def test_select_numbered_cancel(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _prompt: "q")
    assert select("Pick", ["a", "b"]) is None


# ---- viewport: long lists scroll instead of overflowing the screen ----

def test_select_raw_scrolls_long_lists(monkeypatch, capsys):
    from harness.operator import menu as menu_mod

    monkeypatch.setattr(menu_mod, "_view_limit", lambda total: 3)
    idx = select("Pick", ["alpha", "bravo", "charlie", "delta", "echo"],
                 reader=_KeyReader(["down", "down", "down", "enter"]))
    assert idx == 3                            # delta selected
    out = capsys.readouterr().out
    assert "delta" in out
    assert "^2 more" in out                    # scrolled: items above the window
    # the redraw block never grows past title+viewport+footer rows
    assert "\x1b[6A" not in out and "\x1b[7A" not in out


def test_select_raw_clamps_long_options_to_width(monkeypatch, capsys):
    import re

    from harness.operator import menu as menu_mod

    monkeypatch.setattr(menu_mod, "_terminal_width", lambda: 40)
    long = "x" * 200
    idx = select("Pick some extremely long titled thing that also wraps",
                 [long], reader=_KeyReader(["enter"]))
    assert idx == 0
    raw_lines = capsys.readouterr().out.splitlines()
    assert any("\u2026" in ln for ln in raw_lines)       # ellipsis marker shown
    # measure VISIBLE width: clear sequences share a line with the next frame
    visible = [re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", ln) for ln in raw_lines]
    assert max(len(ln) for ln in visible) <= 42          # no physical wrapping


# ---- inventory discovery ----

def test_discover_inventory_filters_non_inventories(tmp_path, monkeypatch):
    _write_inventory(tmp_path, "config/rack.yaml", _CONSOLE_DEFAULTS_INVENTORY)
    _write_inventory(tmp_path, "inventory.yaml", _INVENTORY)
    _write_inventory(tmp_path, "config/targets.yaml", "targets:\n  - alias: x\n    rack: Q1\n    cable: 1\n")
    _write_inventory(tmp_path, "config/broken.yaml", "::: not yaml\n")
    monkeypatch.chdir(tmp_path)

    found = _discover_inventory()
    assert {p.name for p in found} == {"rack.yaml", "inventory.yaml"}


def test_discover_inventory_finds_empty_host_inventory(tmp_path, monkeypatch):
    """A minimal inventory with hosts: [] (e.g. created by `harness setup`)
    must be discoverable -- zero-YAML --address/--rack targeting needs it."""
    minimal = ("trust_level: lab\n"
               "llm:\n"
               "  provider: gemini\n"
               "  api_key_vault_path: secret/harness/llm/gemini-key\n"
               "hosts: []\n")
    _write_inventory(tmp_path, "inventory.yaml", minimal)
    monkeypatch.chdir(tmp_path)

    assert [p.name for p in _discover_inventory()] == ["inventory.yaml"]


def test_discover_inventory_excludes_plain_llm_file(tmp_path, monkeypatch):
    """A YAML with an llm block but neither hosts: nor console_defaults: is
    not an inventory (e.g. config/models.yaml must never be picked)."""
    _write_inventory(tmp_path, "config/models.yaml",
                     "llm:\n  provider: stub\n")
    monkeypatch.chdir(tmp_path)

    assert _discover_inventory() == []


def test_pick_inventory_auto_launches_setup_when_none_found(tmp_path,
                                                            monkeypatch,
                                                            capsys):
    """Fresh install, interactive: no inventory -> auto-run `harness setup`,
    then re-discover the created minimal inventory."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    built = []

    def fake_sub(argv):
        built.append(argv)
        (tmp_path / "inventory.yaml").write_text(
            "trust_level: lab\n"
            "llm:\n  provider: gemini\n  api_key_vault_path: "
            "secret/harness/llm/gemini-key\n"
            "hosts: []\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(cli_mod, "_run_wizard_sub", fake_sub)

    inv = _pick_inventory(SimpleNamespace(inventory=None, secret_dir=None,
                                          docs_lib=None, docs_dir=None,
                                          parts_csv=None, parts_dir=None,
                                          out_dir="harness_runs",
                                          session_dir="harness_runs/sessions",
                                          llm=None, console=False,
                                          ask_parts=False,
                                          targets_file="config/targets.yaml"))
    assert inv == Path("inventory.yaml")
    assert built == [["setup"]]
    out = capsys.readouterr().out
    assert "inventory: inventory.yaml" in out


def test_pick_inventory_no_autolaunch_noninteractive(tmp_path, monkeypatch,
                                                     capsys):
    """Non-interactive (CI/automation) keeps the plain error + exit path."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    built = []
    monkeypatch.setattr(cli_mod, "_run_wizard_sub",
                        lambda argv: built.append(argv) or 0)

    assert _pick_inventory(SimpleNamespace(inventory=None)) is None
    assert built == []
    assert "no inventory found" in capsys.readouterr().err


def test_pick_inventory_setup_cancel_returns_none(tmp_path, monkeypatch,
                                                  capsys):
    """Ctrl-C during auto-launched setup backs out cleanly."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    def fake_sub(argv):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli_mod, "_run_wizard_sub", fake_sub)

    assert _pick_inventory(SimpleNamespace(inventory=None)) is None
    assert "setup cancelled" in capsys.readouterr().err


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
    # lint lives in the ADVANCED submenu now: main -> advanced -> lint ->
    # back -> quit (indices resolved by action key so reordering never
    # breaks this test)
    main_keys = [k for k, _ in cli_mod._MAIN_ACTIONS]
    adv_keys = [k for k, _ in cli_mod._ADVANCED_ACTIONS]
    monkeypatch.setattr(menu_mod, "select", _scripted_select([
        main_keys.index("advanced"),
        adv_keys.index("lint"),
        adv_keys.index(cli_mod._BACK),
        main_keys.index("quit"),
    ]))

    assert run_menu(_menu_args()) == 0
    out = capsys.readouterr().out
    assert "harness menu" in out
    assert "OK: 1 host(s): h1" in out


def test_main_and_advanced_menu_shapes():
    """The simplified top level is exactly the daily flows + Advanced;
    every demoted tool stays reachable in the submenu."""
    assert [k for k, _ in cli_mod._MAIN_ACTIONS] == [
        "chat", "diagnose", "runs", "advanced", "quit"]
    adv_keys = [k for k, _ in cli_mod._ADVANCED_ACTIONS]
    assert adv_keys == ["verify", "console", "model", "docs", "targets",
                        "secrets", "learning", "setup", "lint", cli_mod._BACK]
    # labels are user-facing copy: no raw command names as the whole label
    assert "Debug a target" in dict(cli_mod._MAIN_ACTIONS)["diagnose"]


def test_run_menu_advanced_round_trip(tmp_path, monkeypatch, capsys):
    """Advanced opens, Back returns to the main menu, then Quit exits."""
    _write_inventory(tmp_path)
    monkeypatch.chdir(tmp_path)
    from harness.operator import menu as menu_mod

    main_keys = [k for k, _ in cli_mod._MAIN_ACTIONS]
    adv_keys = [k for k, _ in cli_mod._ADVANCED_ACTIONS]
    monkeypatch.setattr(menu_mod, "select", _scripted_select([
        main_keys.index("advanced"),
        adv_keys.index(cli_mod._BACK),
        main_keys.index("quit"),
    ]))

    assert run_menu(_menu_args()) == 0
    out = capsys.readouterr().out
    assert "harness menu" in out


def test_run_menu_diagnose_builds_debug_argv(tmp_path, monkeypatch, capsys):
    """\"Debug a target\" launches the interactive debug REPL on the picked
    target; no symptom is prompted (the agent takes it as the first message)."""
    _write_inventory(tmp_path)
    monkeypatch.chdir(tmp_path)
    from harness.operator import menu as menu_mod
    built = []

    def fake_sub(argv):
        built.append(argv)
        return 0

    # diagnose -> pick h1 (target list index 0) -> quit
    main_keys = [k for k, _ in cli_mod._MAIN_ACTIONS]
    monkeypatch.setattr(menu_mod, "select",
                        _scripted_select([main_keys.index("diagnose"), 0,
                                          main_keys.index("quit")]))
    asked = []
    monkeypatch.setattr(menu_mod, "ask_text",
                        lambda prompt, **kw: asked.append(prompt) or "")
    monkeypatch.setattr(cli_mod, "_run_wizard_sub", fake_sub)

    assert run_menu(_menu_args()) == 0
    assert built and built[0][0] == "debug"
    assert built[0][1:3] == ["--inventory", "inventory.yaml"]
    assert built[0][built[0].index("--host") + 1] == "h1"
    assert "--symptom" not in built[0]
    assert not asked  # no symptom / test-log prompting in the debug flow


def test_run_menu_chat_builds_argv_without_target(tmp_path, monkeypatch):
    """The Chat entry is target-less: no _pick_target, no target argv. The
    menu's inventory rides along only so tunnel-backed models can open the
    LLM hop."""
    _write_inventory(tmp_path)
    monkeypatch.chdir(tmp_path)
    from harness.operator import menu as menu_mod
    built = []

    def fake_sub(argv):
        built.append(argv)
        return 0

    main_keys = [k for k, _ in cli_mod._MAIN_ACTIONS]
    monkeypatch.setattr(menu_mod, "select",
                        _scripted_select([main_keys.index("chat"),
                                          main_keys.index("quit")]))
    monkeypatch.setattr(cli_mod, "_run_wizard_sub", fake_sub)

    assert run_menu(_menu_args()) == 0
    assert built and built[0][0] == "chat"
    assert "--host" not in built[0]
    assert "--rack" not in built[0] and "--cable" not in built[0]
    assert "--address" not in built[0] and "--target" not in built[0]
    assert built[0][built[0].index("--inventory") + 1] == "inventory.yaml"


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
    # 0 = run, 0 = verdict, 7 = back
    monkeypatch.setattr(menu_mod, "select", _scripted_select([0, 0, 7]))
    assert _menu_runs(args) == 0
    out = capsys.readouterr().out
    assert "healthy" in out


def test_menu_runs_prompt_turns_view(tmp_path, monkeypatch, capsys):
    from harness.operator import menu as menu_mod
    base = _fake_run_dir(tmp_path, "def456", {
        "prompt_turns.jsonl": '{"turn": 1, "messages": [{"role": "user", '
                              '"content": "evidence block"}]}\n'})
    args = SimpleNamespace(out_dir=str(base / "harness_runs"))
    # 0 = run, 2 = prompt, 0 = turn 1, 7 = back
    monkeypatch.setattr(menu_mod, "select", _scripted_select([0, 2, 0, 7]))
    assert _menu_runs(args) == 0
    assert "evidence block" in capsys.readouterr().out


def test_menu_runs_dumps_view(tmp_path, monkeypatch, capsys):
    from harness.operator import menu as menu_mod
    base = _fake_run_dir(tmp_path, "ghi789", {
        "dumps/ipmi_0.txt": "sensor data here"})
    args = SimpleNamespace(out_dir=str(base / "harness_runs"))
    # 0 = run, 3 = dumps, 0 = first dump file, 7 = back
    monkeypatch.setattr(menu_mod, "select", _scripted_select([0, 3, 0, 7]))
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
    base = _fake_run_dir(tmp_path, "jkl012", {
        "pending_case.json": '{"run_id": "jkl012", "target_id": "x", '
                             '"symptom": "?"}'})
    args = SimpleNamespace(out_dir=str(base / "harness_runs"))
    # 0 = run, 2 = prompt (missing artifact), 4 = fix, 7 = back
    monkeypatch.setattr(menu_mod, "select", _scripted_select([0, 2, 4, 7]))
    assert _menu_runs(args) == 0
    assert "no prompt artifact" in capsys.readouterr().out


# ---- runs menu: delete ----

def _run_with_case(base, run_id="abc123"):
    """A run dir plus its verified case in the (created) case store."""
    from harness.diagnosis.case_store import CaseStore
    from harness.diagnosis.schema import CaseOutcome

    run = base / "harness_runs" / run_id
    run.mkdir(parents=True, exist_ok=True)
    (run / "diagnosis.json").write_text('{"state": "healthy"}', encoding="utf-8")
    cases = base / "harness_runs" / "cases"
    CaseStore(cases).record(CaseOutcome(
        run_id=run_id, target_id="t", symptom="s", model_key="m",
        outcome="fixed", actions_recommended=[], actions_taken=[],
        llm_ident="stub", evidence_summary=[]))
    return run, cases


def test_menu_runs_delete_keeps_case_by_default(tmp_path, monkeypatch, capsys):
    from harness.operator import menu as menu_mod
    run, cases = _run_with_case(tmp_path)
    args = SimpleNamespace(out_dir=str(tmp_path / "harness_runs"))
    # 0 = run, 6 = delete; confirms: yes (delete run), no (keep case)
    monkeypatch.setattr(menu_mod, "select", _scripted_select([0, 6]))
    answers = iter([True, False])
    monkeypatch.setattr(menu_mod, "confirm",
                        lambda prompt, default=False: next(answers))
    assert _menu_runs(args) == 0
    out = capsys.readouterr().out
    assert "deleted run abc123" in out
    assert not run.exists()
    assert (cases / "abc123.json").exists()


def test_menu_runs_delete_also_drops_case(tmp_path, monkeypatch, capsys):
    from harness.operator import menu as menu_mod
    _run, cases = _run_with_case(tmp_path)
    args = SimpleNamespace(out_dir=str(tmp_path / "harness_runs"))
    # 0 = run, 6 = delete; confirms: yes (delete run), yes (drop case)
    monkeypatch.setattr(menu_mod, "select", _scripted_select([0, 6]))
    answers = iter([True, True])
    monkeypatch.setattr(menu_mod, "confirm",
                        lambda prompt, default=False: next(answers))
    assert _menu_runs(args) == 0
    out = capsys.readouterr().out
    assert "verified case deleted" in out
    assert not (cases / "abc123.json").exists()


def test_menu_runs_delete_cancelled(tmp_path, monkeypatch, capsys):
    from harness.operator import menu as menu_mod
    run, cases = _run_with_case(tmp_path)
    args = SimpleNamespace(out_dir=str(tmp_path / "harness_runs"))
    # 0 = run, 6 = delete (confirm no), 7 = back
    monkeypatch.setattr(menu_mod, "select", _scripted_select([0, 6, 7]))
    monkeypatch.setattr(menu_mod, "confirm", lambda prompt, default=False: False)
    assert _menu_runs(args) == 0
    out = capsys.readouterr().out
    assert "cancelled" in out
    assert run.exists()
    assert (cases / "abc123.json").exists()


# ---- run listing labels / run metadata ----

def test_is_run_dir_filters_reserved_and_requires_artifacts(tmp_path):
    base = tmp_path / "harness_runs"
    for name in ("sessions", "cases", "secrets", "calibration"):
        (base / name).mkdir(parents=True)
        (base / name / "audit.jsonl").write_text("{}\n", encoding="utf-8")
    (base / "abc").mkdir()
    (base / "abc" / "audit.jsonl").write_text("{}\n", encoding="utf-8")
    (base / "dumps_only").mkdir()
    (base / "dumps_only" / "dumps").mkdir()
    (base / "empty").mkdir()
    assert cli_mod._is_run_dir(base / "abc")
    assert cli_mod._is_run_dir(base / "dumps_only")
    for name in ("sessions", "cases", "secrets", "calibration"):
        assert not cli_mod._is_run_dir(base / name), name
    assert not cli_mod._is_run_dir(base / "empty")


def test_summarize_run_full_metadata_and_fallback(tmp_path):
    base = tmp_path / "harness_runs"
    run = base / "abc"
    run.mkdir(parents=True)
    (run / "audit.jsonl").write_text(json.dumps({
        "kind": "run_start", "ts": "2026-08-11T21:08:41+00:00",
        "payload": {"host": "Q63-cable2", "rack": "Q63", "cable": "2",
                    "symptom": "no post"}}) + "\n", encoding="utf-8")
    (run / "diagnosis.json").write_text(
        '{"state": "fault", "confidence": 0.92}', encoding="utf-8")
    (run / "run_meta.json").write_text('{"serial": "4CX1234"}',
                                       encoding="utf-8")
    cases = base / "cases"
    cases.mkdir()
    (cases / "abc.json").write_text(json.dumps({
        "outcome": "fixed", "actions_taken": ["reseated CPU B"]}),
        encoding="utf-8")

    label = cli_mod._summarize_run(run, cases)
    assert "2026-08-11 21:08" in label
    assert "Q63-cable2 (rack Q63 cable 2)" in label
    assert "SN:4CX1234" in label
    assert "FAULT 92%" in label
    assert "fixed: reseated CPU B" in label

    bare = base / "xyz"
    bare.mkdir()
    assert cli_mod._summarize_run(bare, cases) == "xyz"


def test_summarize_run_serial_falls_back_to_fru_dump(tmp_path):
    base = tmp_path / "harness_runs"
    run = base / "abc"
    (run / "dumps").mkdir(parents=True)
    (run / "dumps" / "ipmi_0.txt").write_text(
        "Product Name: PowerEdge R650\nProduct Serial :  4CX9ZZ7\n",
        encoding="utf-8")
    assert "SN:4CX9ZZ7" in cli_mod._summarize_run(run, base / "cases")


def test_write_run_meta_captures_target_and_serial(tmp_path):
    out = tmp_path / "run1"
    (out / "dumps").mkdir(parents=True)
    (out / "dumps" / "ipmi_0.txt").write_text("Product Serial : 4CX1234\n",
                                              encoding="utf-8")
    target = SimpleNamespace(
        label="Q63-cable2", console=SimpleNamespace(rack="Q63", cable="2"))
    cli_mod._write_run_meta(out, target, SimpleNamespace(model="r650"), "sess1")
    meta = json.loads((out / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["serial"] == "4CX1234"
    assert meta["host"] == "Q63-cable2"
    assert meta["rack"] == "Q63" and meta["cable"] == "2"
    assert meta["model"] == "r650"
    assert meta["session_id"] == "sess1"


def test_synthesize_case_from_diagnosis_and_audit(tmp_path):
    run = tmp_path / "abc"
    run.mkdir()
    (run / "audit.jsonl").write_text(json.dumps({
        "kind": "run_start", "ts": "2026-08-11T21:08:41+00:00",
        "payload": {"host": "Q63-cable2", "symptom": "no post"}}) + "\n",
        encoding="utf-8")
    (run / "diagnosis.json").write_text(json.dumps({
        "state": "fault", "confidence": 0.9,
        "subsystems_considered": ["cpu"],
        "actions": [{"step": 1, "action": "Reseat CPU"}],
        "evidence": [{"mnemonic": "MSR_0", "raw_hex": "0x0"}],
    }), encoding="utf-8")

    case = cli_mod._synthesize_case(run)
    assert case is not None
    assert case.run_id == "abc"
    assert case.symptom == "no post"
    assert case.actions_recommended == ["1. Reseat CPU"]
    assert case.subsystem_primary == "cpu"
    assert case.outcome == "unknown"
    assert cli_mod._synthesize_case(tmp_path / "nothing-here") is None


def test_print_run_fix_recommended_vs_labeled(tmp_path, capsys):
    run = tmp_path / "abc"
    run.mkdir()
    (run / "pending_case.json").write_text(json.dumps({
        "actions_recommended": ["1. Reseat CPU B"]}), encoding="utf-8")
    cases = tmp_path / "cases"
    cases.mkdir()
    (cases / "abc.json").write_text(json.dumps({
        "outcome": "fixed",
        "actions_taken": ["replaced CPU B", "updated BIOS"]}),
        encoding="utf-8")

    cli_mod._print_run_fix(run, cases)
    out = capsys.readouterr().out
    assert "1. Reseat CPU B" in out
    assert "replaced CPU B" in out
    assert "updated BIOS" in out
    assert "outcome: fixed" in out

    bare = tmp_path / "xyz"
    bare.mkdir()
    cli_mod._print_run_fix(bare, cases)
    assert "not labeled yet" in capsys.readouterr().out


# ---- labeling (learning-loop ground truth) ----

def _seed_run(base, run_id):
    from harness.audit.auditlog import AuditLog
    run = base / run_id
    run.mkdir(parents=True)
    (run / "pending_case.json").write_text(json.dumps({
        "run_id": run_id, "target_id": "Q63-cable2", "symptom": "no post",
        "actions_recommended": ["1. Reseat CPU B"], "llm_ident": "stub",
        "outcome": "unknown"}), encoding="utf-8")
    AuditLog(run / "audit.jsonl").append(
        run_id, "run_start", {"host": "Q63-cable2", "symptom": "no post",
                              "rack": "Q63", "cable": "2"})
    return run


def test_label_run_records_outcome_and_fix(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    run = _seed_run(tmp_path / "runs", "abc")
    cases = tmp_path / "cases"

    assert cli_mod._label_run(run, cases, outcome="fixed",
                              taken=["replaced CPU B"]) == 0
    case = json.loads((cases / "abc.json").read_text(encoding="utf-8"))
    assert case["outcome"] == "fixed"
    assert case["actions_taken"] == ["replaced CPU B"]
    assert case["symptom"] == "no post"


def test_label_run_refuses_second_record_without_revise(tmp_path, monkeypatch,
                                                        capsys):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    run = _seed_run(tmp_path / "runs", "abc")
    cases = tmp_path / "cases"
    assert cli_mod._label_run(run, cases, outcome="fixed",
                              taken=["replaced CPU B"]) == 0
    rc = cli_mod._label_run(run, cases, outcome="partial", taken=["x"])
    assert rc == 1
    assert "already recorded" in capsys.readouterr().err
    case = json.loads((cases / "abc.json").read_text(encoding="utf-8"))
    assert case["outcome"] == "fixed"  # unchanged


def test_label_run_revise_replaces_and_audits(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    run = _seed_run(tmp_path / "runs", "abc")
    cases = tmp_path / "cases"
    assert cli_mod._label_run(run, cases, outcome="fixed",
                              taken=["replaced CPU B"]) == 0
    assert cli_mod._label_run(run, cases, outcome="partial",
                              taken=["reseated CPU B"], revise=True) == 0
    case = json.loads((cases / "abc.json").read_text(encoding="utf-8"))
    assert case["outcome"] == "partial"
    kinds = [json.loads(line)["kind"]
             for line in (run / "audit.jsonl").read_text(encoding="utf-8")
             .splitlines() if line.strip()]
    assert kinds == ["run_start", "case_record", "case_revised"]


def test_label_run_synthesizes_for_old_runs(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    run = tmp_path / "runs" / "old1"
    run.mkdir(parents=True)
    from harness.audit.auditlog import AuditLog
    AuditLog(run / "audit.jsonl").append(
        "old1", "run_start", {"host": "Q63-cable2", "symptom": "no post"})
    (run / "diagnosis.json").write_text(json.dumps({
        "state": "fault", "confidence": 0.9,
        "subsystems_considered": ["cpu"],
        "actions": [{"step": 1, "action": "Reseat CPU"}]}),
        encoding="utf-8")
    cases = tmp_path / "cases"
    assert cli_mod._label_run(run, cases, outcome="fixed",
                              taken=["replaced CPU"]) == 0
    case = json.loads((cases / "old1.json").read_text(encoding="utf-8"))
    assert case["actions_recommended"] == ["1. Reseat CPU"]
    assert case["outcome"] == "fixed"


def test_label_run_defers_on_blank_outcome(tmp_path, monkeypatch, capsys):
    from harness.operator import menu as menu_mod
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    # the picker offers the four outcomes + a defer row; pick the defer row
    monkeypatch.setattr(menu_mod, "select",
                        lambda title, options, **kw: len(options) - 1)
    run = _seed_run(tmp_path / "runs", "abc")
    cases = tmp_path / "cases"
    assert cli_mod._label_run(run, cases, allow_defer=True) == 0
    assert "deferred" in capsys.readouterr().out
    assert not (cases / "abc.json").exists()


def test_should_prompt_label_gating(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    assert cli_mod._should_prompt_label(
        SimpleNamespace(label_prompt=True))
    assert not cli_mod._should_prompt_label(
        SimpleNamespace(label_prompt=False))
    assert not cli_mod._should_prompt_label(
        SimpleNamespace(label_prompt=False, interactive=False))
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert not cli_mod._should_prompt_label(
        SimpleNamespace(label_prompt=True, interactive=False))


def test_prompt_label_after_run_defers_on_blank(tmp_path, monkeypatch, capsys):
    from harness.operator import menu as menu_mod
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr(menu_mod, "ask_text", lambda prompt, **kw: "")
    run = _seed_run(tmp_path / "runs", "abc")
    cli_mod._prompt_label_after_run(run, tmp_path / "cases")
    assert "deferred" in capsys.readouterr().out
    assert not (tmp_path / "cases" / "abc.json").exists()


def test_run_label_cli_status_and_non_interactive(tmp_path, capsys):
    base = tmp_path / "harness_runs"
    _seed_run(base, "abc")
    cases = str(tmp_path / "cases")
    parser = cli_mod.build_parser()

    argv = ["label", "--run", "abc", "--out-dir", str(base), "--cases", cases]
    assert cli_mod.run_label(parser.parse_args(argv + ["--status"])) == 1

    assert cli_mod.run_label(parser.parse_args(
        argv + ["--outcome", "fixed", "--taken", "replaced CPU B"])) == 0
    case = json.loads((tmp_path / "cases" / "abc.json").read_text(encoding="utf-8"))
    assert case["outcome"] == "fixed"

    capsys.readouterr()
    assert cli_mod.run_label(parser.parse_args(argv + ["--status"])) == 0
    assert '"outcome": "fixed"' in capsys.readouterr().out


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
    assert "\x1b[J? Rack id (e.g. Q61) " in out  # redraw clears with active prompt
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
    assert "\x1b[J? Rack id (e.g. Q61) " in out
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


# ---- guided model setup (wizard + post-pick probe) ----

def test_ask_model_profile_discovers_served_model(monkeypatch):
    """Direct endpoint, reachable: the wizard lists the served model ids and
    the operator picks one -- no typing model names."""
    import harness.diagnosis.llm as llm_mod
    from harness.operator import menu as menu_mod
    from harness.operator.menu import ask_model_profile

    monkeypatch.setattr(menu_mod, "select", _scripted_select([
        2,   # provider: local
        0,   # transport: direct
        0,   # served model id: the only one listed
    ]))
    monkeypatch.setattr(menu_mod, "ask_text", lambda prompt, **kw: "")
    monkeypatch.setattr(llm_mod, "list_models",
                        lambda url, api_key=None, timeout=10.0:
                        ["Qwen2.5-7B-Instruct"])

    profile = ask_model_profile()
    assert profile.provider == "local"
    assert profile.model == "Qwen2.5-7B-Instruct"
    assert profile.url == "http://127.0.0.1:8000/v1"   # Enter took the default
    assert profile.tunnel is None


def test_ask_model_profile_unreachable_saves_anyway(monkeypatch, capsys):
    """Probe failure is a warning, not a wall: the profile is saved with a
    manually typed model id."""
    import harness.diagnosis.llm as llm_mod
    from harness.operator import menu as menu_mod
    from harness.operator.menu import ask_model_profile

    def dead(url, api_key=None, timeout=10.0):
        raise llm_mod.LLMError("LLM unreachable: connection refused")

    monkeypatch.setattr(menu_mod, "select", _scripted_select([
        0,   # provider: openai
        0,   # transport: direct
    ]))
    answers = iter(["http://10.0.0.42:8000/v1", "my-model", ""])
    monkeypatch.setattr(menu_mod, "ask_text", lambda prompt, **kw: next(answers))
    monkeypatch.setattr(llm_mod, "list_models", dead)

    profile = ask_model_profile()
    assert profile.provider == "openai"
    assert profile.model == "my-model"
    assert profile.url == "http://10.0.0.42:8000/v1"
    assert profile.api_key_vault_path is None           # Enter = env fallback
    captured = capsys.readouterr()
    assert "endpoint unreachable" in captured.err + captured.out


def test_ask_model_profile_tunnel_saved_unprobed(monkeypatch, capsys):
    """Golden server, no inventory in reach: the wizard asks rack/cable, then
    falls back to manual port/HOST entry; the spec is saved unprobed."""
    from harness.operator import menu as menu_mod
    from harness.operator.menu import ask_model_profile

    monkeypatch.setattr(menu_mod, "select", _scripted_select([
        2,   # provider: local
        1,   # transport: tunnel (model on the golden server)
    ]))
    answers = iter(["",              # endpoint URL: Enter = default
                    "Q61", "8",      # rack/cable of the golden server
                    "",              # vLLM port: Enter = 8000
                    "10.0.0.42",     # node HOST as the rackmgr addresses it
                    "Qwen2.5-7B-Instruct"])
    monkeypatch.setattr(menu_mod, "ask_text", lambda prompt, **kw: next(answers))

    profile = ask_model_profile()
    assert profile.tunnel == "10.0.0.42:8000"     # port default applied
    assert profile.url is None                    # tunnel profiles carry no URL
    assert profile.model == "Qwen2.5-7B-Instruct"
    captured = capsys.readouterr()
    assert "unprobed" in captured.err + captured.out


def test_ask_model_profile_label_prefills_rack_cable(monkeypatch):
    """A console session target label (Q61-cable8) prefills the rack/cable
    asks -- Enter twice and the wizard moves on to port/HOST."""
    from harness.operator import menu as menu_mod
    from harness.operator.menu import ask_model_profile

    monkeypatch.setattr(menu_mod, "select", _scripted_select([1]))  # tunnel
    asked = []
    answers = iter(["",              # endpoint URL: Enter = default
                    "",              # rack: Enter = prefilled Q61
                    "",              # cable: Enter = prefilled 8
                    "",              # vLLM port: Enter = 8000
                    "10.0.0.42",     # node HOST
                    "Qwen2.5-7B-Instruct"])

    def record_ask(prompt, **kw):
        asked.append(prompt)
        return next(answers)

    monkeypatch.setattr(menu_mod, "ask_text", record_ask)

    profile = ask_model_profile(provider="local", target_label="Q61-cable8")
    assert profile.tunnel == "10.0.0.42:8000"     # completed via prefills
    prompts = " | ".join(asked)
    assert "Enter = Q61" in prompts and "Enter = 8" in prompts


def test_ask_model_profile_transport_labels_pin_golden_server_copy(monkeypatch):
    """The tunnel option is framed as 'model runs on the golden server
    (rack/cable debug target)' -- regression guard against reusing
    rack/cable for anything but debug-target addressing."""
    from harness.operator import menu as menu_mod
    from harness.operator.menu import ask_model_profile

    seen = {}

    def capturing_select(title, options, **kw):
        seen[title] = list(options)  # cancel implicitly (None)

    monkeypatch.setattr(menu_mod, "select", capturing_select)
    monkeypatch.setattr(menu_mod, "ask_text", lambda prompt, **kw: "")
    ask_model_profile(provider="local")
    transport = seen["Transport"]
    assert transport[0] == "direct -- endpoint reachable from this workstation"
    assert transport[1].startswith(
        "tunnel -- model runs on the golden server (the rack/cable debug target)")


def test_ask_model_profile_discovery_fills_port_and_host(
        tmp_path, monkeypatch, capsys):
    """With inv+store the wizard probes the golden server over the console
    and the operator picks the discovered candidates from arrow-key lists."""
    import harness.diagnosis.llm as llm_mod
    import harness.engine.tunnel as tunnel_mod
    from harness.operator import menu as menu_mod
    from harness.operator.llm_discover import DiscoveryResult
    from harness.operator.menu import ask_model_profile

    monkeypatch.setattr("harness.operator.llm_discover.discover",
                        lambda *a, **kw: DiscoveryResult(
                            addresses=["10.0.0.42"],
                            ports=[8000],
                            containers={"vllm-qwen": 8000}))

    class _WorkingForward:
        def __init__(self, host, port, console, store, bastion=None):
            pass

        def start(self):
            return "http://127.0.0.1:28400/v1"

        def close(self):
            pass

    monkeypatch.setattr(tunnel_mod, "LLMForward", _WorkingForward)
    monkeypatch.setattr(menu_mod, "select", _scripted_select([
        2,   # provider: local
        1,   # transport: tunnel
        0,   # vLLM port: the discovered one
        0,   # node HOST: the discovered one
        0,   # served model id
    ]))
    answers = iter(["", "Q61", "8", "yemankyaw"])
    monkeypatch.setattr(menu_mod, "ask_text", lambda prompt, **kw: next(answers))
    _patch_node_capture(monkeypatch)
    monkeypatch.setattr(llm_mod, "list_models",
                        lambda url, api_key=None, timeout=10.0:
                        ["Qwen2.5-7B-Instruct"])

    inv = load_inventory(_write_inventory(
        tmp_path, "inv_cd.yaml", body=_CONSOLE_DEFAULTS_INVENTORY))
    profile = ask_model_profile(inv=inv, store=MemorySecretStore())
    assert profile.tunnel == "10.0.0.42:8000"
    assert profile.model == "Qwen2.5-7B-Instruct"
    captured = capsys.readouterr()
    assert "hop: rack manager 192.168.202.51" in captured.err + captured.out


def test_ask_model_profile_discovery_failure_falls_back_manual(
        tmp_path, monkeypatch, capsys):
    """Console discovery failure is a warning: the wizard continues with
    manual port/HOST entry."""
    import harness.engine.tunnel as tunnel_mod
    from harness.operator import menu as menu_mod
    from harness.operator.menu import ask_model_profile
    from harness.targets.resolver import TargetError

    def boom(*a, **kw):
        raise TargetError("console targeting blocked")

    monkeypatch.setattr("harness.operator.llm_discover.discover", boom)
    monkeypatch.setattr(menu_mod, "select", _scripted_select([2, 1, 0]))
    answers = iter(["", "Q61", "8", "yemankyaw", "", "10.0.0.42"])
    monkeypatch.setattr(menu_mod, "ask_text", lambda prompt, **kw: next(answers))
    _patch_node_capture(monkeypatch)
    monkeypatch.setattr("harness.diagnosis.llm.list_models",
                        lambda url, api_key=None, timeout=10.0:
                        ["Qwen2.5-7B-Instruct"])

    class _WorkingForward:
        def __init__(self, host, port, console, store, bastion=None):
            pass

        def start(self):
            return "http://127.0.0.1:28401/v1"

        def close(self):
            pass

    monkeypatch.setattr(tunnel_mod, "LLMForward", _WorkingForward)

    inv = load_inventory(_write_inventory(
        tmp_path, "inv_cd.yaml", body=_CONSOLE_DEFAULTS_INVENTORY))
    profile = ask_model_profile(inv=inv, store=MemorySecretStore())
    assert profile.tunnel == "10.0.0.42:8000"
    assert profile.model == "Qwen2.5-7B-Instruct"
    captured = capsys.readouterr()
    assert "discovery failed" in captured.err + captured.out


def _patch_node_capture(monkeypatch, password=b"pw\n"):
    """The wizard prompts for the node user + sudo password at EVERY setup;
    tests stub the getpass capture (the flow still runs unconditionally)."""
    monkeypatch.setattr(
        "harness.operator.credential_gate.CredentialPrompter.prompt_now",
        lambda self, path: password)


def test_ask_model_profile_node_capture_prompt_and_persist(tmp_path, monkeypatch,
                                                           capsys):
    """Node user + sudo password are prompted at EVERY model setup (never
    silently reused); the captured password lands at the per-rack node-sudo
    vault path and feeds the docker probe's sudo handshake."""
    import harness.engine.tunnel as tunnel_mod
    from harness.config.vault import MemorySecretStore
    from harness.operator import menu as menu_mod
    from harness.operator.llm_discover import DiscoveryResult
    from harness.operator.menu import ask_model_profile

    inv = load_inventory(_write_inventory(tmp_path, "inv_cd.yaml", body=(
        "trust_level: lab\n"
        "llm_console: {address: 192.168.202.51, user: log, "
        "identity_vault_path: secret/harness/rackmgr/id_ed25519, "
        "known_hosts_path: config/rackmgr_known_hosts, "
        "tool: jumpin, trust_level: lab, port: 22, "
        "prompts: [RScmCli#, '~$']}\n"
        "hosts: []\n")))

    monkeypatch.setattr("harness.operator.llm_discover.discover",
                        lambda *a, **kw: DiscoveryResult(
                            addresses=["10.0.0.42"], ports=[8000],
                            containers={"vllm": 8000}))

    class _WorkingForward:
        def __init__(self, host, port, console, store, bastion=None):
            pass

        def start(self):
            return "http://127.0.0.1:28410/v1"

        def close(self):
            pass

    monkeypatch.setattr(tunnel_mod, "LLMForward", _WorkingForward)
    monkeypatch.setattr(menu_mod, "select", _scripted_select([
        2,   # provider: local
        1,   # transport: tunnel
        0,   # vLLM port: discovered
        0,   # node HOST: discovered
        0,   # served model id
    ]))
    answers = iter(["", "Q71", "8", "yemankyaw"])
    monkeypatch.setattr(menu_mod, "ask_text", lambda prompt, **kw: next(answers))
    monkeypatch.setattr("harness.diagnosis.llm.list_models",
                        lambda url, api_key=None, timeout=10.0:
                        ["Qwen2.5-7B-Instruct"])
    captures = []
    monkeypatch.setattr(
        "harness.operator.credential_gate.CredentialPrompter.prompt_now",
        lambda self, path: (captures.append(path), b"s3cret\n")[1])

    store = MemorySecretStore()
    profile = ask_model_profile(inv=inv, store=store)
    assert profile.tunnel == "10.0.0.42:8000"
    assert captures == ["secret/harness/llm/node-sudo-71"]
    assert store.get("secret/harness/llm/node-sudo-71") == b"s3cret\n"
    out = capsys.readouterr().out
    assert "probing Q71-cable8 over the console..." in out


def test_ask_model_profile_node_capture_decline_aborts(tmp_path, monkeypatch,
                                                       capsys):
    """A declined sudo capture aborts the wizard up front with the vault
    path named -- no mid-hop surprises."""
    from harness.config.vault import MemorySecretStore
    from harness.operator import menu as menu_mod
    from harness.operator.menu import ask_model_profile

    inv = load_inventory(_write_inventory(tmp_path, "inv_cd.yaml", body=(
        "trust_level: lab\n"
        "llm_console: {address: 192.168.202.51, user: log, "
        "identity_vault_path: secret/harness/rackmgr/id_ed25519, "
        "known_hosts_path: config/rackmgr_known_hosts, "
        "tool: jumpin, trust_level: lab, port: 22, "
        "prompts: [RScmCli#, '~$']}\n"
        "hosts: []\n")))

    monkeypatch.setattr(menu_mod, "select", _scripted_select([2, 1]))
    answers = iter(["", "Q71", "8", "yemankyaw"])
    monkeypatch.setattr(menu_mod, "ask_text", lambda prompt, **kw: next(answers))

    class _DeclinePrompter:
        def __init__(self, store):
            pass

        def prompt_now(self, path):
            raise KeyError(path)

    monkeypatch.setattr(
        "harness.operator.credential_gate.CredentialPrompter",
        _DeclinePrompter)

    assert ask_model_profile(inv=inv, store=MemorySecretStore()) is None
    captured = capsys.readouterr()
    assert "node sudo password skipped" in captured.err
    assert "secret/harness/llm/node-sudo-71" in captured.err


def test_ask_model_profile_relay_fallback_on_forward_refusal(
        tmp_path, monkeypatch, capsys):
    """Golden server behind a jumpin-only rackmgr: a forward-stage refusal
    prints the reverse-tunnel relay recipe and offers to save the relay URL
    as the endpoint (no tunnel in the saved profile)."""
    import harness.diagnosis.llm as llm_mod
    import harness.engine.tunnel as tunnel_mod
    from harness.operator import menu as menu_mod
    from harness.operator.llm_discover import DiscoveryResult
    from harness.operator.menu import ask_model_profile

    class _RefusedForward:
        def __init__(self, host, port, console, store, bastion=None):
            pass

        def start(self):
            raise tunnel_mod.TunnelError("forward", "channel open failed")

        def close(self):
            pass

    monkeypatch.setattr(tunnel_mod, "LLMForward", _RefusedForward)
    monkeypatch.setattr("harness.operator.llm_discover.discover",
                        lambda *a, **kw: DiscoveryResult(
                            addresses=["10.0.0.42"], ports=[8000],
                            containers={"vllm": 8000}))
    monkeypatch.setattr(menu_mod, "select", _scripted_select([
        2,   # provider: local
        1,   # transport: tunnel
        0,   # vLLM port: discovered
        0,   # node HOST: discovered
        0,   # served model id (from the relay endpoint)
    ]))
    answers = iter(["", "Q61", "8", "yemankyaw", "y"])
    monkeypatch.setattr(menu_mod, "ask_text", lambda prompt, **kw: next(answers))
    _patch_node_capture(monkeypatch)
    monkeypatch.setattr(llm_mod, "list_models",
                        lambda url, api_key=None, timeout=10.0:
                        ["Qwen2.5-7B-Instruct"])

    inv = load_inventory(_write_inventory(
        tmp_path, "inv_cd.yaml", body=_CONSOLE_DEFAULTS_INVENTORY))
    profile = ask_model_profile(inv=inv, store=MemorySecretStore())

    assert profile.url == "http://127.0.0.1:18000/v1"   # relay, not a tunnel
    assert profile.tunnel is None
    assert profile.model == "Qwen2.5-7B-Instruct"       # picked from the list
    captured = capsys.readouterr()
    assert "ssh -fN -R 127.0.0.1:18000:127.0.0.1:8000" in captured.err + captured.out
    assert "tunnel hop failed (forward)" in captured.err + captured.out


def test_ask_model_profile_auth_failure_keeps_tunnel(tmp_path, monkeypatch,
                                                     capsys):
    """A rackmgr SSH auth failure is staged (no relay offer): the tunnel spec
    is kept and the model id typed manually."""
    import harness.engine.tunnel as tunnel_mod
    from harness.operator import menu as menu_mod
    from harness.operator.llm_discover import DiscoveryResult
    from harness.operator.menu import ask_model_profile

    class _AuthFailedForward:
        def __init__(self, host, port, console, store, bastion=None):
            pass

        def start(self):
            raise tunnel_mod.TunnelError("auth", "ssh to rack manager failed")

        def close(self):
            pass

    monkeypatch.setattr(tunnel_mod, "LLMForward", _AuthFailedForward)
    monkeypatch.setattr("harness.operator.llm_discover.discover",
                        lambda *a, **kw: DiscoveryResult(
                            addresses=["10.0.0.42"], ports=[8000]))
    monkeypatch.setattr(menu_mod, "select", _scripted_select([2, 1, 0, 0]))
    answers = iter(["", "Q61", "8", "yemankyaw", "Qwen2.5-7B-Instruct"])
    monkeypatch.setattr(menu_mod, "ask_text", lambda prompt, **kw: next(answers))
    _patch_node_capture(monkeypatch)

    inv = load_inventory(_write_inventory(
        tmp_path, "inv_cd.yaml", body=_CONSOLE_DEFAULTS_INVENTORY))
    profile = ask_model_profile(inv=inv, store=MemorySecretStore())

    assert profile.tunnel == "10.0.0.42:8000"          # kept, not swapped
    assert profile.model == "Qwen2.5-7B-Instruct"
    captured = capsys.readouterr()
    assert "tunnel hop failed (auth)" in captured.err + captured.out
    assert "relay" not in (captured.err + captured.out).lower()


def test_ask_model_profile_invalid_tunnel_cancels(monkeypatch, capsys):
    from harness.operator import menu as menu_mod
    from harness.operator.menu import ask_model_profile

    monkeypatch.setattr(menu_mod, "select", _scripted_select([2, 1]))
    answers = iter(["", "Q61", "8", "", "bad host"])   # HOST with a space
    monkeypatch.setattr(menu_mod, "ask_text", lambda prompt, **kw: next(answers))

    assert ask_model_profile() is None
    assert "invalid tunnel target" in capsys.readouterr().err


def test_ask_model_profile_warns_manager_address_as_host(tmp_path, monkeypatch,
                                                         capsys):
    """A HOST that is actually the rack manager's address gets called out --
    the tunnel must target the golden server's own address."""
    import harness.engine.tunnel as tunnel_mod
    from harness.config.vault import MemorySecretStore
    from harness.operator import menu as menu_mod
    from harness.operator.llm_discover import DiscoveryResult
    from harness.operator.menu import ask_model_profile

    class _WorkingForward:
        def __init__(self, host, port, console, store, bastion=None):
            pass

        def start(self):
            return "http://127.0.0.1:28420/v1"

        def close(self):
            pass

    monkeypatch.setattr(tunnel_mod, "LLMForward", _WorkingForward)
    monkeypatch.setattr("harness.operator.llm_discover.discover",
                        lambda *a, **kw: DiscoveryResult(ports=[8000]))
    _patch_node_capture(monkeypatch)
    monkeypatch.setattr(menu_mod, "select", _scripted_select([
        2,   # provider: local
        1,   # transport: tunnel
        0,   # vLLM port: discovered
        0,   # served model id
    ]))
    answers = iter(["", "Q61", "8", "yemankyaw", "192.168.202.51"])
    monkeypatch.setattr(menu_mod, "ask_text", lambda prompt, **kw: next(answers))
    monkeypatch.setattr("harness.diagnosis.llm.list_models",
                        lambda url, api_key=None, timeout=10.0:
                        ["Qwen2.5-7B-Instruct"])

    inv = load_inventory(_write_inventory(
        tmp_path, "inv_cd.yaml", body=_CONSOLE_DEFAULTS_INVENTORY))
    profile = ask_model_profile(inv=inv, store=MemorySecretStore())
    assert profile.tunnel == "192.168.202.51:8000"   # saved anyway (warn only)
    captured = capsys.readouterr()
    assert "rack manager's address" in captured.err + captured.out


def test_ask_model_profile_surfaces_forward_refusal(tmp_path, monkeypatch,
                                                    capsys):
    """A refused tunnel target (e.g. the manager's IP entered as HOST) is
    reported with the recorded channel error -- no traceback, no silent
    'unreachable'."""
    import harness.diagnosis.llm as llm_mod
    import harness.engine.tunnel as tunnel_mod
    from harness.config.vault import MemorySecretStore
    from harness.operator import menu as menu_mod
    from harness.operator.llm_discover import DiscoveryResult
    from harness.operator.menu import ask_model_profile

    class _RefusedTargetForward:
        def __init__(self, host, port, console, store, bastion=None):
            self.forward_error = "Connection refused: Connect failed"

        def start(self):
            return "http://127.0.0.1:28420/v1"   # listener binds; channel fails

        def close(self):
            pass

    monkeypatch.setattr(tunnel_mod, "LLMForward", _RefusedTargetForward)
    monkeypatch.setattr("harness.operator.llm_discover.discover",
                        lambda *a, **kw: DiscoveryResult(
                            addresses=["10.0.0.42"], ports=[8000]))
    _patch_node_capture(monkeypatch)
    monkeypatch.setattr(menu_mod, "select", _scripted_select([
        2,   # provider: local
        1,   # transport: tunnel
        0,   # vLLM port: discovered
        0,   # node HOST: discovered
        0,   # served model id
    ]))
    answers = iter(["", "Q61", "8", "yemankyaw", "Qwen2.5-7B-Instruct"])
    monkeypatch.setattr(menu_mod, "ask_text", lambda prompt, **kw: next(answers))
    monkeypatch.setattr(llm_mod, "list_models",
                        lambda url, api_key=None, timeout=10.0:
                        (_ for _ in ()).throw(
                            llm_mod.LLMError("LLM connection failed: reset")))

    inv = load_inventory(_write_inventory(
        tmp_path, "inv_cd.yaml", body=_CONSOLE_DEFAULTS_INVENTORY))
    profile = ask_model_profile(inv=inv, store=MemorySecretStore())
    assert profile.tunnel == "10.0.0.42:8000"   # saved with the warning
    assert profile.model == "Qwen2.5-7B-Instruct"
    captured = capsys.readouterr()
    assert "tunnel target refused: Connection refused" in captured.err + captured.out


def test_menu_model_guided_setup_on_unconfigured_builtin(tmp_path, monkeypatch,
                                                         capsys):
    """Selecting the silent-default ``local/harness-diag`` row launches the
    guided setup and saves the configured profile as the current model."""
    monkeypatch.chdir(tmp_path)
    from harness.config.model_catalog import ModelCatalog, ModelProfile
    from harness.operator import menu as menu_mod

    # no inventory llm block: the picker shows exactly the well-known defaults
    plain = _INVENTORY.replace("llm:\n  provider: stub\n", "")
    inv = load_inventory(_write_inventory(tmp_path, body=plain))

    def fake_wizard(*, provider=None, **kw):
        assert provider == "local"
        return ModelProfile(provider="local", model="Qwen2.5-7B-Instruct",
                            tunnel="10.0.0.42:8000")

    monkeypatch.setattr(menu_mod, "ask_model_profile", fake_wizard)
    monkeypatch.setattr(menu_mod, "check_profile", lambda *a, **kw: None)
    monkeypatch.setattr(menu_mod, "select", _scripted_select([2]))  # local row

    assert cli_mod._menu_model(inv, MemorySecretStore()) == 0
    out = capsys.readouterr().out
    assert "model: local/Qwen2.5-7B-Instruct" in out
    saved = json.loads((tmp_path / "config" / "models.yaml").read_text(
        encoding="utf-8"))
    assert saved["current"]["tunnel"] == "10.0.0.42:8000"
    assert ModelCatalog.load(tmp_path / "config" / "models.yaml").current.tunnel


def test_check_profile_offers_switch_on_model_mismatch(monkeypatch, capsys):
    """The probe catches a model id the endpoint does not serve (vLLM would
    404 every request) and offers the served ids as a one-key switch."""
    import harness.diagnosis.llm as llm_mod
    from harness.config.model_catalog import ModelProfile
    from harness.operator import menu as menu_mod
    from harness.operator.menu import check_profile

    monkeypatch.setattr(llm_mod, "list_models",
                        lambda url, api_key=None, timeout=10.0:
                        ["Qwen2.5-7B-Instruct"])
    monkeypatch.setattr(menu_mod, "select", _scripted_select([0]))
    profile = ModelProfile(provider="local", model="harness-diag",
                           url="http://127.0.0.1:8000/v1")
    assert check_profile(profile) == "Qwen2.5-7B-Instruct"

    out = capsys.readouterr().out
    assert "not served" in out


def test_check_profile_quiet_when_served_or_skipped(monkeypatch, capsys):
    import harness.diagnosis.llm as llm_mod
    from harness.config.model_catalog import ModelProfile
    from harness.operator.menu import check_profile

    monkeypatch.setattr(llm_mod, "list_models",
                        lambda url, api_key=None, timeout=10.0:
                        ["Qwen/Qwen2.5-7B-Instruct"])
    ok = ModelProfile(provider="local", model="Qwen2.5-7B-Instruct",
                      url="http://127.0.0.1:8000/v1")
    assert check_profile(ok) is None                  # suffix match on the id
    assert check_profile(ModelProfile(provider="gemini")) is None   # skipped
    assert check_profile(ModelProfile(provider="stub")) is None
    dead = ModelProfile(provider="local", model="m",
                        url="http://127.0.0.1:8000/v1")
    monkeypatch.setattr(llm_mod, "list_models",
                        lambda url, api_key=None, timeout=10.0: (_ for _ in ()).throw(
                            llm_mod.LLMError("LLM unreachable")))
    assert check_profile(dead) is None                # warning only, no raise
    captured = capsys.readouterr()
    assert "unreachable" in captured.out + captured.err
