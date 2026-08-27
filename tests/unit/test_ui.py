"""ui module: styling gates, display-width math, and the restyled verdict."""

from harness.diagnosis.schema import Action, Diagnosis
from harness.operator import ui

# ---- styling gate ----

def test_styling_disabled_without_tty(monkeypatch):
    # capsys stdout is not a tty -> everything degrades to plain text
    assert ui.enabled() is False
    assert ui.bold("x") == "x"
    assert ui.dim("y") == "y"
    assert ui.heading("z") == "z"


def test_styling_honors_no_color(monkeypatch):
    import sys

    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)
    assert ui.enabled()                       # sanity: tty forces styling on
    monkeypatch.setenv("NO_COLOR", "1")
    assert not ui.enabled()
    assert ui.bold("x") == "x"
    monkeypatch.delenv("NO_COLOR")
    monkeypatch.setenv("HARNESS_NO_COLOR", "1")
    assert not ui.enabled()


def test_styling_wraps_when_enabled(monkeypatch):
    monkeypatch.setattr(ui, "enabled", lambda: True)
    out = ui.bad("boom")
    assert out.startswith("\x1b[31m") and out.endswith("\x1b[0m")
    assert "boom" in out


def test_empty_text_never_wrapped(monkeypatch):
    monkeypatch.setattr(ui, "enabled", lambda: True)
    assert ui.bold("") == ""


# ---- display width / clipping ----

def test_char_width_ascii_and_wide():
    assert ui.char_width("a") == 1
    assert ui.char_width("\u65e5") == 2  # CJK: east-asian wide
    assert ui.char_width("\u0301") == 0  # combining mark
    assert ui.char_width("\t") == 4


def test_disp_width_mixed():
    assert ui.disp_width("> ab") == 4
    assert ui.disp_width("\u65e5\u672c") == 4  # two wide chars = 4 columns


def test_clip_short_text_untouched():
    assert ui.clip("hello", 10) == "hello"


def test_clip_truncates_with_ellipsis():
    out = ui.clip("abcdefghij", 5)
    assert len(out) == 5
    assert out.endswith("\u2026")
    assert out.startswith("abcd")


def test_clip_counts_wide_chars_as_two_columns():
    out = ui.clip("\u65e5\u672c\u8a9e\u3067\u3059", 5)  # five wide chars = 10 cols
    assert ui.disp_width(out) <= 5


def test_clip_zero_and_negative():
    assert ui.clip("abc", 0) == ""
    assert ui.clip("", 10) == ""


def test_rule_respects_explicit_width():
    assert ui.rule(10) == "-" * 10


# ---- restyled diagnosis report (plain mode: asserted substrings intact) ----

def _diagnosis(**kw):
    base = {
        "state": "fault",
        "diagnosis": "DIMM_A2 has corrected errors.",
        "confidence": 0.72,
        "actions": [Action(step=1, action="reseat DIMM_A2",
                           rationale="see manual p.12", risk="medium",
                           required_tool="hands",
                           impact="requires brief downtime")],
    }
    base.update(kw)
    return Diagnosis(**base)


def test_report_plain_mode_keeps_substrings(capsys):
    from harness.operator.cli import _print_diagnosis

    _print_diagnosis(_diagnosis(), out=None, session_id="abc123")  # type: ignore[arg-type]
    out = capsys.readouterr().out
    assert "==== Diagnosis [abc123] ====" in out
    assert "state: fault" in out
    assert "confidence: 0.72" in out
    assert "[#######---]" in out                      # confidence bar (0.72 -> 7)
    assert "repair action list" in out
    assert "[medium]" in out                          # risk tag readable in plain mode


def test_report_confidence_bar_bounds(capsys):
    from harness.operator.cli import _print_diagnosis

    _print_diagnosis(_diagnosis(state="healthy", confidence=1.0),
                     out=None, session_id="s")
    assert "[##########]" in capsys.readouterr().out
    _print_diagnosis(_diagnosis(confidence=0.04), out=None, session_id="s2")
    assert "[----------]" in capsys.readouterr().out


def test_report_failure_point_warn_styled(capsys):
    from harness.diagnosis.schema import FailurePoint
    from harness.operator.cli import _print_diagnosis

    fp = FailurePoint(rail_tokens="P3V3", suspects=["DIMM_A2"],
                      isolation_ran=True)
    _print_diagnosis(_diagnosis(failure_point=fp), out=None, session_id="s")
    out = capsys.readouterr().out
    assert "failure point" in out and "P3V3" in out
