"""Terminal styling helpers for the operator UI (banners, events, menus).

Every helper degrades to plain text when stdout is not a tty or when
``NO_COLOR`` / ``HARNESS_NO_COLOR`` is set, so piped output (CI, log files)
stays clean. During background REPL tasks ``sys.stdout`` is the thread-aware
capture wrapper whose ``isatty()`` delegates to the real stream
(``operator.repl._Capture``), so styled output still reaches real terminals
from worker threads.

Also home to the display-width math shared with the line editor and menus:
wrapping/redraw must count terminal COLUMNS (east-asian wide chars count 2,
combining marks 0), not Python code points.

Glyphs are deliberately conservative (present in cp437/Consolas-era fonts):
``·`` bullets, ``»`` labels, ``>`` menu cursor -- differentiation comes from
color and weight first, so a font without an exotic glyph never matters.
"""

from __future__ import annotations

import os
import shutil
import sys
import unicodedata

# ---- ANSI escape plumbing -------------------------------------------------

RESET = "\x1b[0m"

_BOLD = "1"
_DIM = "2"
_FG_CYAN = "36"
_FG_GREEN = "32"
_FG_YELLOW = "33"
_FG_RED = "31"
_FG_MAGENTA = "35"


def enabled() -> bool:
    """True when ANSI styling should be emitted right now."""
    if os.environ.get("NO_COLOR") or os.environ.get("HARNESS_NO_COLOR"):
        return False
    try:
        return bool(sys.stdout.isatty())
    except (ValueError, OSError, AttributeError):
        return False


def enable_vt() -> None:
    """Best-effort Windows console VT enablement (no-op elsewhere)."""
    from .menu import _enable_vt

    _enable_vt()


def _wrap(text: str, *codes: str) -> str:
    if not codes or not enabled() or not text:
        return text
    return f"\x1b[{';'.join(codes)}m{text}{RESET}"


def bold(text: str) -> str:
    return _wrap(text, _BOLD)


def dim(text: str) -> str:
    return _wrap(text, _DIM)


def accent(text: str) -> str:
    return _wrap(text, _FG_CYAN)


def good(text: str) -> str:
    return _wrap(text, _FG_GREEN)


def warn(text: str) -> str:
    return _wrap(text, _FG_YELLOW)


def bad(text: str) -> str:
    return _wrap(text, _FG_RED)


def heading(text: str) -> str:
    return _wrap(text, _BOLD, _FG_CYAN)


def selected(text: str) -> str:
    """The picker's highlighted row (bold cyan, single escape pair)."""
    return _wrap(text, _BOLD, _FG_CYAN)


def title(text: str) -> str:
    return _wrap(text, _BOLD, _FG_MAGENTA)


# ---- status glyphs ----------------------------------------------------------

GLYPH_BULLET = "·"   # progress / secondary lines
GLYPH_POINT = ">"    # selected row / prompts
GLYPH_LABEL = "»"    # banner field labels


# ---- layout helpers --------------------------------------------------------

def rule(width: int | None = None) -> str:
    """A dim horizontal rule across the terminal (or ``width`` columns)."""
    if width is None:
        width = max(20, min(terminal_width(), 78))
    return dim("-" * width)


def kv(label: str, value: str, *, label_width: int = 9) -> str:
    """One banner row: accent label, padded, then the value."""
    pad = " " * max(1, label_width - len(label))
    return f"  {accent(label)}{pad}{value}"


# ---- display width / clipping ----------------------------------------------

def char_width(ch: str) -> int:
    """Terminal columns occupied by one character."""
    if ch == "\t":
        return 4
    if unicodedata.combining(ch):
        return 0
    if ch.isprintable() and unicodedata.east_asian_width(ch) in ("W", "F"):
        return 2
    return 1 if ch.isprintable() else 0


def disp_width(text: str) -> int:
    """Terminal columns a string occupies when printed at column 0."""
    return sum(char_width(ch) for ch in text)


def clip(text: str, width: int) -> str:
    """Truncate ``text`` to ``width`` columns, ellipsis on cut.

    Menu blocks count their LINES to redraw; a row longer than the terminal
    would silently wrap into two physical rows and leave stale fragments
    behind after the next repaint.
    """
    if width <= 0:
        return ""
    if disp_width(text) <= width:
        return text
    keep = max(0, width - 1)  # reserve one column for the ellipsis
    cols = 0
    out: list[str] = []
    for ch in text:
        w = char_width(ch)
        if cols + w > keep:
            break
        out.append(ch)
        cols += w
    return "".join(out) + "…"


def terminal_width() -> int:
    try:
        return max(1, shutil.get_terminal_size((80, 24)).columns)
    except Exception:  # noqa: BLE001 - cosmetic path, never fatal
        return 80


def terminal_height() -> int:
    try:
        return max(1, shutil.get_terminal_size((80, 24)).lines)
    except Exception:  # noqa: BLE001 - cosmetic path, never fatal
        return 24
