"""Interactive selection menus for the operator CLI (Claude-Code-style).

The operator launches ``harness`` (or ``harness menu``) and drives everything
from inside: pick an inventory, pick a target, pick an action, describe the
symptom in plain text. This module provides:

- ``LineReader``: the raw-mode key reader (moved from ``operator.repl`` and
  extended with arrow-key / escape-sequence handling). ``poll()`` reads whole
  lines for the chat REPL; ``read_key()`` reads single key events for menus.
  Falls back to blocking ``input()`` when stdin is not a tty.
- ``select()``: arrow-key picker with type-to-filter; numbered fallback when
  the terminal is not interactive.
- ``ask_text()`` / ``confirm()``: small prompts on top of the same reader.

Menus render with ANSI clear/rewrite and enable VT processing on Windows
consoles (best-effort; if that fails the numbered fallback still works).
"""

from __future__ import annotations

import os
import sys
import time

# Windows scan codes (msvcrt.getwch after an \x00/\xe0 prefix).
_NT_SCAN = {"H": "up", "P": "down", "K": "left", "M": "right"}

# POSIX escape-sequence suffixes (bytes after ESC '[').
_POSIX_ESC = {
    b"[A": "up", b"[B": "down", b"[C": "right", b"[D": "left",
    b"[H": "up", b"[F": "down",  # home/end keys read as up/down
    b"[1~": "home", b"[4~": "end", b"[3~": "delete",
}


def decode_escape(rest: bytes) -> str | None:
    """Decode the bytes that followed ESC into a key token (or None if it is
    not a recognized sequence -- a lone ESC reads as "esc")."""
    if len(rest) < 2 or not rest.startswith(b"["):
        return None
    return _POSIX_ESC.get(rest[:2]) or _POSIX_ESC.get(rest)


def _enable_vt() -> None:
    """Enable ANSI VT processing on a Windows console (best-effort; menus
    still work in numbered mode when this fails)."""
    if os.name != "nt":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:  # noqa: BLE001, S110 - VT is an optimization, never fatal
        pass


def _terminal_width() -> int:
    """Current terminal width in columns (best-effort; 80 when unknown)."""
    try:
        import shutil
        return max(1, shutil.get_terminal_size((80, 24)).columns)
    except Exception:  # noqa: BLE001 - cosmetic path, never fatal
        return 80


class LineReader:
    """Raw-mode line/key reader (POSIX termios + Windows msvcrt).

    ``poll(timeout)`` returns a complete line (the chat REPL); ``read_key``
    returns single key events ("up"/"down"/"enter"/"esc"/... or the char
    itself) for menus. When stdin is not a tty both fall back to blocking
    ``input()``.
    """

    def __init__(self, prompt: str = "harness> ") -> None:
        self.prompt = prompt
        self._active = prompt  # prompt of the line being edited (poll can
        # override it; redraw/clear_line must use the ACTIVE one, not the
        # default, or backspacing rewrites with the wrong prompt)
        self._buf = ""
        self._raw = False
        self._fd = None
        self._restore = None
        self._max_width = len(prompt)
        self._prompt_printed = False  # echoed the active prompt for this line;
        # poll() re-fires at 10Hz while a task runs, so this must gate re-echoing
        self._cursor_rows = 0  # physical rows below the line start the cursor
        # sits on (0 = the prompt row); the line may WRAP past the terminal
        # width, and redraw must climb back across those rows, not just "\r"
        if not sys.stdin.isatty():
            self._raw = False
        elif os.name == "nt":
            try:
                import msvcrt  # noqa: F401 - presence check only
                self._raw = True
            except ImportError:
                self._raw = False
        else:
            self._raw = self._setup_posix()
        if self._raw:
            _enable_vt()  # ANSI clear/redraw must work for the REPL, not just menus

    @property
    def raw(self) -> bool:
        """True when the tty is in raw mode and key events are available."""
        return self._raw

    def _setup_posix(self) -> bool:
        try:
            import termios
            self._fd = sys.stdin.fileno()
            self._attrs = termios.tcgetattr(self._fd)
            new = termios.tcgetattr(self._fd)
            new[3] &= ~(termios.ICANON | termios.ECHO)
            termios.tcsetattr(self._fd, termios.TCSANOW, new)
            self._restore = lambda: termios.tcsetattr(
                self._fd, termios.TCSADRAIN, self._attrs)
            return True
        except (ImportError, OSError, ValueError):
            return False

    def close(self) -> None:
        if self._restore is not None:
            self._restore()

    # ---- key-level input ----

    def read_key(self, timeout: float | None) -> str | None:
        """One key event within ``timeout`` seconds (None = block).

        Returns a single character, or one of: "enter", "up", "down", "left",
        "right", "tab", "esc", "backspace", "ctrl_c", "ctrl_d". None = timeout.
        """
        if os.name == "nt":
            return self._read_key_nt(timeout)
        return self._read_key_posix(timeout)

    def _read_bytes_posix(self, count: int, timeout: float | None) -> bytes:
        import select
        deadline = None if timeout is None else time.monotonic() + timeout
        data = b""
        while len(data) < count:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
            else:
                remaining = None
            wait = 0.02 if remaining is None else min(0.02, remaining)
            ready, _, _ = select.select([sys.stdin], [], [], wait)
            if not ready:
                if deadline is not None:
                    continue  # loop re-checks the deadline
                continue
            try:
                chunk = os.read(self._fd, count - len(data))
            except OSError:
                raise EOFError
            if not chunk:
                raise EOFError
            data += chunk
        return data

    def _read_key_posix(self, timeout: float | None) -> str | None:
        first = self._read_bytes_posix(1, timeout)
        if not first:
            return None
        ch = first.decode("utf-8", errors="replace")
        if ch == "\x1b":
            rest = self._read_bytes_posix(3, 0.05)
            return decode_escape(rest) or "esc"
        return self._token(ch)

    def _read_key_nt(self, timeout: float | None) -> str | None:
        import msvcrt
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
            else:
                remaining = None
            while not msvcrt.kbhit():
                if deadline is not None and time.monotonic() >= deadline:
                    return None
                time.sleep(0.02)
            ch = msvcrt.getwch()
            if ch in ("\x00", "\xe0"):  # function/arrow key: scan code follows
                scan = msvcrt.getwch()
                return _NT_SCAN.get(scan, "esc")  # unmapped keys read as esc
            return self._token(ch)

    @staticmethod
    def _token(ch: str) -> str:
        if ch in ("\r", "\n"):
            return "enter"
        if ch == "\x03":
            return "ctrl_c"
        if ch == "\x04":
            return "ctrl_d"
        if ch in ("\x08", "\x7f"):
            return "backspace"
        if ch == "\t":
            return "tab"
        if ch == "\x1b":
            return "esc"
        return ch

    # ---- line-level input (chat REPL) ----

    def poll(self, timeout: float | None, prompt: str | None = None) -> str | None:
        """Return a complete line within ``timeout`` seconds (None = block)."""
        if not self._raw:
            return input(prompt or self.prompt)
        self._active = prompt or self.prompt
        if not self._buf and not self._prompt_printed:
            self._max_width = len(self._active)
            self._cursor_rows = len(self._active) // _terminal_width()
            print(self._active, end="", flush=True)
            self._prompt_printed = True
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if timeout is None:
                remaining = None
            else:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
            key = self.read_key(remaining)
            if key is None:
                return None
            if key in ("up", "down", "left", "right", "tab", "esc"):
                continue  # navigation keys are ignored while typing a line
            if key == "enter":
                line, self._buf = self._buf, ""
                self._max_width = len(self.prompt)
                self._cursor_rows = 0
                self._prompt_printed = False
                print()
                return line
            if key == "ctrl_c":
                raise KeyboardInterrupt
            if key == "ctrl_d":
                raise EOFError
            if key == "backspace":
                if self._buf:
                    self._buf = self._buf[:-1]
                    self.redraw(len(self._active) + len(self._buf) + 1)
                continue
            self._buf += key
            n = len(self._active) + len(self._buf)
            self._max_width = max(self._max_width, n)
            self._cursor_rows = n // _terminal_width()
            print(key, end="", flush=True)

    def clear_line(self) -> None:
        """Erase every physical row the echoed input line occupies.

        A long line WRAPS past the terminal width, so a plain ``\\r`` + spaces
        only re-anchors the last row and leaves stale copies of the wrapped
        rows on screen (each background event then re-copies the input line).
        ``_cursor_rows`` tracks how far below the line start the cursor sits;
        climb back that far, clear each row down to the cursor, then return to
        the first row so the caller can print there.
        """
        if not self._raw:
            return
        rows = self._cursor_rows
        up = rows
        seq = (f"\x1b[{up}A" if up else "")
        seq += "\r\x1b[K" + "\x1b[B\x1b[K" * rows
        seq += (f"\x1b[{up}A" if up else "") + "\r"
        print(seq, end="", flush=True)

    def refresh_line(self) -> None:
        """Re-echo the active prompt + buffer at the CURRENT cursor position.

        Used by the chat REPL after printing a background event line on the
        cleared input rows: writes the input line fresh below the event and
        updates the wrap tracking for the next ``clear_line``.
        """
        if not self._raw:
            return
        n = len(self._active) + len(self._buf)
        self._max_width = max(self._max_width, n)
        pad = self._max_width - n
        print(f"\r{self._active}{self._buf}{' ' * pad}\r{self._active}{self._buf}",
              end="", flush=True)
        self._cursor_rows = n // _terminal_width()
        self._prompt_printed = True

    def redraw(self, n_before: int | None = None) -> None:
        """Re-render the active prompt line in place, crossing line wraps.

        ``n_before`` is the length (prompt+text) before the edit; the cursor
        is ``n_before // width`` rows below the line start, so we climb back
        exactly that far, clear exactly those rows plus the current one
        (``\\x1b[K`` per row -- never erase below the block), then rewrite.
        """
        if not self._raw:
            return
        width = _terminal_width()
        if n_before is not None:
            self._cursor_rows = n_before // width
        up = self._cursor_rows
        n = len(self._active) + len(self._buf)
        self._max_width = max(self._max_width, n)
        seq = (f"\x1b[{up}A" if up else "") + "\r"
        seq += "\x1b[K" + "\x1b[B\x1b[K" * up
        if up:
            seq += f"\x1b[{up}A"
        self._cursor_rows = n // width
        print(seq + f"{self._active}{self._buf}", end="", flush=True)
        self._prompt_printed = True


# ---- selection menu ----

def select(title: str, options: list[str], *, reader: LineReader | None = None,
           default: int = 0, allow_cancel: bool = True) -> int | None:
    """Pick one option; returns its index or None when cancelled.

    Uses an arrow-key menu with type-to-filter when the terminal supports raw
    mode; otherwise a numbered ``input()`` fallback.
    """
    reader = reader or LineReader()
    if not options:
        return None
    if not reader.raw:
        return _select_numbered(title, options, default, allow_cancel)
    return _select_raw(reader, title, options, default, allow_cancel)


def _select_numbered(title: str, options: list[str], default: int,
                     allow_cancel: bool) -> int | None:
    print(f"? {title}")
    for i, opt in enumerate(options):
        mark = "  (default)" if i == default else ""
        print(f"  [{i + 1}] {opt}{mark}")
    hint = (f"  enter 1-{len(options)} [Enter={default + 1}], q=cancel: "
            if allow_cancel else f"  enter 1-{len(options)}: ")
    while True:
        try:
            raw = input(hint).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return None
        if allow_cancel and raw in ("q", "quit", "cancel"):
            return None
        if not raw:
            if allow_cancel:
                return default if 0 <= default < len(options) else None
            continue
        try:
            n = int(raw)
        except ValueError:
            print(f"  enter 1-{len(options)}")
            continue
        if 1 <= n <= len(options):
            return n - 1
        print(f"  enter 1-{len(options)}")


def _render_rows(title: str, options: list[str], filtered: list[int],
                 cursor: int, query: str) -> list[str]:
    lines = [f"? {title}  (up/down, type to filter, Enter, Esc)"]
    if filtered:
        for i, idx in enumerate(filtered):
            marker = "> " if i == cursor else "  "
            lines.append(f"{marker}{options[idx]}")
    else:
        lines.append("  (no matches)")
    if query:
        lines.append(f"filter: {query!r} {len(filtered)}/{len(options)}")
    return lines


def _clear_rows(rows: int) -> None:
    """Erase the menu block (plus any wrapped lines below it) so the next
    menu/render starts from a clean screen region."""
    if rows <= 0:
        return
    print(f"\x1b[{rows}A\x1b[2K\x1b[J", end="")


def _select_raw(reader: LineReader, title: str, options: list[str],
                default: int, allow_cancel: bool) -> int | None:
    _enable_vt()
    cursor = default if 0 <= default < len(options) else 0
    query = ""
    last_rows = 0
    first = True
    while True:
        filtered = [i for i, o in enumerate(options) if query.lower() in o.lower()]
        if filtered and cursor >= len(filtered):
            cursor = 0
        rows = _render_rows(title, options, filtered, cursor, query)
        if first:
            for line in rows:
                print(line)
            first = False
        else:
            # The cursor sits exactly `last_rows` rows below the previous block
            # start; move up THAT many and erase everything below (also covers
            # physical rows left behind when a line wrapped), then rewrite.
            print(f"\x1b[{last_rows}A\x1b[2K\x1b[J", end="")
            for line in rows:
                print(line)
        last_rows = len(rows)
        try:
            key = reader.read_key(None)
        except KeyboardInterrupt:
            _clear_rows(last_rows)
            return None
        if key == "up":
            cursor = (cursor - 1) % len(filtered) if filtered else 0
        elif key == "down":
            cursor = (cursor + 1) % len(filtered) if filtered else 0
        elif key == "backspace":
            query = query[:-1]
            cursor = 0
        elif key in ("esc", "ctrl_c"):
            _clear_rows(last_rows)
            return None
        elif key == "enter":
            if not filtered:
                continue  # nothing selected; keep the menu open
            _clear_rows(last_rows)
            print(f"? {title}: {options[filtered[cursor]]}")
            return filtered[cursor]
        elif len(key) == 1 and key.isprintable():
            query += key
            cursor = 0


# ---- small prompts ----

def ask_text(prompt: str, *, reader: LineReader | None = None,
             default: str | None = None) -> str:
    """Prompt for one line of text (returns default when input is empty)."""
    reader = reader or LineReader()
    try:
        line = reader.poll(None, prompt=f"? {prompt} ")
    except (KeyboardInterrupt, EOFError):
        return default or ""
    line = line.strip()
    return line or default or ""


def confirm(prompt: str, *, default: bool = False) -> bool:
    """y/N-style confirmation (empty input takes the default)."""
    reader = LineReader()
    suffix = " [y/N] " if not default else " [Y/n] "
    try:
        line = reader.poll(None, prompt=f"? {prompt}{suffix}")
    except (KeyboardInterrupt, EOFError):
        return default
    line = line.strip().lower()
    if not line:
        return default
    return line in ("y", "yes")


def ask_model_profile(*, reader: LineReader | None = None):
    """Three short prompts building a custom LLM ``ModelProfile``.

    Used by the ``+ add a custom model`` row in the model picker (menu and
    ``/model`` in the REPL). Provider must be ``openai`` or ``gemini``; the URL
    and API-key vault path are optional and fall back to the provider defaults.
    Returns None when cancelled.
    """
    from ..config.model_catalog import ModelProfile

    provider = ask_text("Provider (openai | gemini)", reader=reader).strip().lower()
    if provider not in ("openai", "gemini"):
        print(f"  x unknown provider {provider!r} (openai | gemini)", file=sys.stderr)
        return None
    model = ask_text("Model id (e.g. gpt-4o)", reader=reader).strip()
    if not model:
        return None
    url = ask_text("Endpoint URL (Enter = provider default)", reader=reader).strip() or None
    vault = ask_text("API key vault path (Enter = env fallback)",
                     reader=reader).strip() or None
    return ModelProfile(provider=provider, model=model, url=url,
                        api_key_vault_path=vault)
