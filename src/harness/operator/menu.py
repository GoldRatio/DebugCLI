"""Interactive selection menus + raw-mode line editor for the operator CLI.

The operator launches ``harness`` (or ``harness menu``) and drives everything
from inside: pick an inventory, pick a target, pick an action, describe the
symptom in plain text. This module provides:

- ``LineReader``: the raw-mode key reader (POSIX termios, Windows msvcrt).
  ``poll()`` is a full single-line editor for the chat REPL: cursor movement
  (arrows / Home / End / Delete), Emacs bindings (Ctrl-A/E/B/F/U/K/W/L),
  up/down history recall, incremental UTF-8 decoding (multi-byte input never
  degrades), Windows surrogate-pair joining, bracketed-paste support, and
  wrap-aware repaint math measured in DISPLAY COLUMNS (wide chars count 2),
  not code points. Falls back to blocking ``input()`` off-tty.
- ``select()``: arrow-key picker with type-to-filter and a scrolling viewport
  bounded by the terminal height (long lists never climb into scrollback);
  numbered fallback when the terminal is not interactive.
- ``ask_text()`` / ``confirm()``: small prompts on top of the same reader.

Menus render with ANSI clear/rewrite and enable VT processing on Windows
consoles (best-effort; if that fails the numbered fallback still works).
"""

from __future__ import annotations

import os
import re
import sys
import time

from . import ui as _ui

# Windows scan codes (msvcrt.getwch after an \x00/\xe0 prefix).
_NT_SCAN = {
    "H": "up", "P": "down", "K": "left", "M": "right",
    "G": "home", "O": "end", "S": "delete",
}

# POSIX escape-sequence suffixes (bytes between ESC '[' and the final byte).
_POSIX_CSI = {
    b"A": "up", b"B": "down", b"C": "right", b"D": "left",
    b"H": "home", b"F": "end",
}
_POSIX_SS3 = {  # application-mode arrow keys: ESC O <letter>
    b"A": "up", b"B": "down", b"C": "right", b"D": "left",
}
_POSIX_TILDE = {  # CSI <num> ~ forms
    b"1": "home", b"4": "end", b"3": "delete",
    b"200": "paste_start", b"201": "paste_end",
}


def decode_escape(rest: bytes) -> str | None:
    """Decode the bytes that followed ESC into a key token (or None if it is
    not a recognized sequence -- a lone ESC reads as "esc").

    Accepts the classic two-byte forms (``[A``), tilde forms of any digit
    length (``[200~``), and SS3 application-mode arrows (``OA``).
    """
    if len(rest) < 2:
        return None
    if rest.startswith(b"["):
        body, final = rest[1:-1], rest[-1:]
        if not body and final in _POSIX_CSI:
            return _POSIX_CSI[final]
        if final == b"~":
            return _POSIX_TILDE.get(body)
        return None
    if rest.startswith(b"O"):
        return _POSIX_SS3.get(rest[1:])
    return None


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
    """Current terminal width in columns (best-effort; 80 when unknown).

    Module-level indirection so tests can monkeypatch the width the editor
    and menus see.
    """
    return _ui.terminal_width()


class LineReader:
    """Raw-mode line/key reader (POSIX termios + Windows msvcrt).

    ``poll(timeout)`` edits and returns one complete line (the chat REPL);
    ``read_key`` returns single key events ("up"/"down"/"enter"/"esc"/... or
    the char itself) for menus. When stdin is not a tty both fall back to
    blocking ``input()``.
    """

    def __init__(self, prompt: str = "harness> ") -> None:
        self.prompt = prompt
        self._active = prompt  # prompt of the line being edited (poll can
        # override it; repaints must use the ACTIVE one, not the default,
        # or backspacing rewrites with the wrong prompt)
        self._buf = ""
        self._pos = 0          # cursor position INSIDE the buffer
        self._raw = False
        self._fd = None
        self._restore = None
        self._block_cols = 0   # display columns of prompt+buffer as last
        # echoed; the physical cursor sits at (cols-1)//width rows below the
        # line start (deferred-wrap aware -- terminals leave the cursor ON
        # the last column until the NEXT char actually wraps).
        self._prompt_printed = False  # echoed the active prompt for this line;
        # poll() re-fires at 10Hz while a task runs, so this must gate re-echoing
        self._history: list[str] = []  # in-memory recall (not persisted)
        self._hist_idx = -1    # index while walking history (-1 = live draft)
        self._draft = ""       # buffer being typed before history navigation
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
        "right", "home", "end", "delete", "tab", "esc", "backspace",
        "ctrl_c", "ctrl_d", "ctrl_<letter>", "paste_start", "paste_end".
        None = timeout.
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

    def _decode_utf8(self, first: bytes) -> str:
        """Decode one (possibly multi-byte) UTF-8 character.

        Reads continuation bytes as needed instead of decoding each byte
        separately -- per-byte ``errors="replace"`` turned every non-ASCII
        keystroke into replacement characters.
        """
        data = first
        for _ in range(3):  # UTF-8 sequences are at most 4 bytes
            try:
                return data.decode("utf-8")
            except UnicodeDecodeError:
                nxt = self._read_bytes_posix(1, None)  # continuation imminent
                if not nxt:
                    break
                data += nxt
        return data.decode("utf-8", errors="replace")

    def _read_csi(self) -> bytes | None:
        """Read a full CSI/SS3 sequence: everything up to the final byte
        (0x40-0x7E). Variable-length forms like bracketed paste (``[200~``)
        need this -- reading a fixed 3 bytes left ``0~`` behind as garbage."""
        data = b""
        while True:
            try:
                chunk = self._read_bytes_posix(1, 0.05)
            except EOFError:
                return None
            if not chunk:
                return None  # incomplete sequence: treat as lone ESC
            data += chunk
            if 0x40 <= data[-1] <= 0x7E:
                return data

    def _read_key_posix(self, timeout: float | None) -> str | None:
        first = self._read_bytes_posix(1, timeout)
        if not first:
            return None
        if first == b"\x1b":
            rest = self._read_csi()
            if rest is None:
                return "esc"
            return decode_escape(rest) or "esc"
        return self._token(self._decode_utf8(first))

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
            if "\ud800" <= ch <= "\udbff":  # astral char: high surrogate first
                if msvcrt.kbhit():
                    lo = msvcrt.getwch()
                    if "\udc00" <= lo <= "\udfff":
                        return chr(0x10000 + ((ord(ch) - 0xD800) << 10)
                                   + (ord(lo) - 0xDC00))
                return ""  # lone high surrogate: unusable, drop silently
            if "\udc00" <= ch <= "\udfff":
                return ""  # stray low surrogate
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
        if ch == "\x01":
            return "ctrl_a"
        if ch == "\x02":
            return "ctrl_b"
        if ch == "\x05":
            return "ctrl_e"
        if ch == "\x06":
            return "ctrl_f"
        if ch == "\x0b":
            return "ctrl_k"
        if ch == "\x0c":
            return "ctrl_l"
        if ch == "\x15":
            return "ctrl_u"
        if ch == "\x17":
            return "ctrl_w"
        return ch

    # ---- line-level input (chat REPL) ----

    def poll(self, timeout: float | None, prompt: str | None = None) -> str | None:
        """Return a complete line within ``timeout`` seconds (None = block)."""
        if not self._raw:
            return input(prompt or self.prompt)
        self._active = prompt or self.prompt
        if not self._buf and not self._prompt_printed:
            self._begin_line()
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
            if key in ("esc", "tab", ""):
                continue  # navigation/no-op keys are ignored while typing
            if key == "enter":
                line, self._buf, self._pos = self._buf, "", 0
                if line.strip() and (not self._history
                                     or self._history[-1] != line):
                    self._history.append(line)
                self._hist_idx = -1
                self._draft = ""
                self._end_line()
                return line
            if key == "up":
                self._history_nav(1)
                continue
            if key == "down":
                self._history_nav(-1)
                continue
            if key == "left":
                if self._pos > 0:
                    self._pos -= 1
                    print("\x1b[D", end="", flush=True)
                continue
            if key == "right":
                if self._pos < len(self._buf):
                    self._pos += 1
                    print("\x1b[C", end="", flush=True)
                continue
            if key in ("home", "ctrl_a"):
                self._move_home_end(0)
                continue
            if key in ("end", "ctrl_e"):
                self._move_home_end(len(self._buf))
                continue
            if key == "ctrl_b":
                if self._pos > 0:
                    self._pos -= 1
                    print("\x1b[D", end="", flush=True)
                continue
            if key == "ctrl_f":
                if self._pos < len(self._buf):
                    self._pos += 1
                    print("\x1b[C", end="", flush=True)
                continue
            if key == "backspace":
                if self._pos > 0:
                    old = self._block_cols
                    self._buf = self._buf[:self._pos - 1] + self._buf[self._pos:]
                    self._pos -= 1
                    self._repaint(old)
                continue
            if key == "delete":
                if self._pos < len(self._buf):
                    old = self._block_cols
                    self._buf = self._buf[:self._pos] + self._buf[self._pos + 1:]
                    self._repaint(old)
                continue
            if key == "ctrl_u":
                self._replace_line("")
                continue
            if key == "ctrl_k":
                old = self._block_cols
                self._buf = self._buf[:self._pos]
                self._repaint(old)
                continue
            if key == "ctrl_w":
                head = self._buf[:self._pos].rstrip()
                cut = max(head.rfind(" "), 0)
                old = self._block_cols
                self._buf = head[:cut] + self._buf[self._pos:]
                self._pos = cut
                self._repaint(old)
                continue
            if key == "ctrl_l":
                print("\x1b[2J\x1b[H", end="")
                self._block_cols = 0
                self._repaint(0)
                continue
            if key == "paste_start":
                self._consume_paste()
                continue
            if key == "ctrl_c":
                self._abort_line("^C")
                raise KeyboardInterrupt
            if key == "ctrl_d":
                raise EOFError
            # printable text (possibly several code points: paste w/o brackets)
            self._insert(key)

    # ---- editor internals ----

    def _begin_line(self) -> None:
        self._buf = ""
        self._pos = 0
        self._hist_idx = -1
        self._draft = ""
        self._block_cols = 0
        self._prompt_printed = True
        if os.name == "posix":
            sys.stdout.write("\x1b[?2004h")  # bracketed paste on
        print(self._active, end="", flush=True)
        self._block_cols = _ui.disp_width(self._active)

    def _end_line(self) -> None:
        if os.name == "posix":
            sys.stdout.write("\x1b[?2004l")  # bracketed paste off
        print()

    def _abort_line(self, echo: str) -> None:
        """Discard the in-progress line (Ctrl-C) and start fresh next poll."""
        if echo:
            print(echo)
        self._buf = ""
        self._pos = 0
        self._hist_idx = -1
        self._draft = ""
        self._block_cols = 0
        self._prompt_printed = False

    def _sanitize(self, text: str) -> str:
        """Strip control characters from inserted text; tabs become spaces."""
        out = []
        for ch in text:
            if ch == "\t":
                out.append("    ")
            elif ch.isprintable():
                out.append(ch)
        return "".join(out)

    def _insert(self, text: str) -> None:
        text = self._sanitize(text)
        if not text:
            return
        if self._pos == len(self._buf):
            # fast path: appending at the end, the terminal echoes for us
            self._buf += text
            self._pos = len(self._buf)
            self._block_cols += _ui.disp_width(text)
            print(text, end="", flush=True)
        else:
            old = self._block_cols
            self._buf = self._buf[:self._pos] + text + self._buf[self._pos:]
            self._pos += len(text)
            self._repaint(old)

    def _consume_paste(self) -> None:
        """Accumulate a bracketed paste as ONE insertion: multi-line pastes
        must not submit early on embedded newlines, and the chunk lands in
        the buffer verbatim instead of keystroke-by-keystroke."""
        chunk: list[str] = []
        while True:
            key = self.read_key(None)
            if key is None or key in ("paste_end", "ctrl_c", "ctrl_d", "esc"):
                break
            if key == "enter":
                chunk.append("\n")  # newlines INSIDE the paste are content
                continue
            chunk.append(key)
        text = "".join(chunk)
        text = " ".join(text.splitlines())  # single-line editor: flatten
        self._insert(text)

    def _history_nav(self, step: int) -> None:
        if not self._history:
            return
        if self._hist_idx == -1 and step > 0:
            self._draft = self._buf  # leaving the live draft; remember it
        new_idx = self._hist_idx + step
        if new_idx >= len(self._history):
            return
        if new_idx < 0:  # stepped back past the newest entry: restore draft
            self._hist_idx = -1
            self._replace_line(self._draft)
            return
        self._hist_idx = new_idx
        self._replace_line(self._history[new_idx])

    def _replace_line(self, text: str) -> None:
        old = self._block_cols
        self._buf = text
        self._pos = len(text)
        self._repaint(old)

    def _move_home_end(self, target: int) -> None:
        old = self._block_cols
        self._pos = target
        self._repaint(old)

    # ---- repaint plumbing ----

    def _erase_block(self, cols: int) -> None:
        """Erase every physical row the echoed input occupies.

        ``cols`` is the tracked display width of the block; the PHYSICAL
        cursor sits ``(max(cols,1)-1)//width`` rows below the line start
        (a terminal defers wrapping until the next character), so climb that
        far, then ``\\x1b[J`` clears the whole block below -- immune to the
        off-by-one at exact width multiples that made ``\\x1b[K``-per-row
        repaints drift upward.
        """
        width = _terminal_width()
        up = max(cols - 1, 0) // width if cols else 0
        seq = (f"\x1b[{up}A" if up else "") + "\r\x1b[J"
        print(seq, end="", flush=True)

    def _repaint(self, old_cols: int) -> None:
        """Redraw prompt+buffer from the block start (see ``_erase_block``)."""
        self._erase_block(old_cols)
        print(f"{self._active}{self._buf}", end="", flush=True)
        self._block_cols = _ui.disp_width(self._active) + _ui.disp_width(self._buf)
        self._prompt_printed = True

    def clear_line(self) -> None:
        """Erase the echoed input block so the caller can print there."""
        if not self._raw:
            return
        self._erase_block(self._block_cols)

    def refresh_line(self) -> None:
        """Re-echo the active prompt + buffer at the CURRENT cursor position.

        Used by the chat REPL after printing a background event line on the
        cleared input rows: writes the input line fresh below the event and
        updates the column tracking for the next ``clear_line``.
        """
        if not self._raw:
            return
        print(f"{self._active}{self._buf}", end="", flush=True)
        self._block_cols = _ui.disp_width(self._active) + _ui.disp_width(self._buf)
        self._prompt_printed = True

    def redraw(self, n_before: int | None = None) -> None:
        """Re-render the active prompt line in place (legacy entry point).

        ``n_before`` (columns occupied before the edit) defaults to the
        tracked value.
        """
        if not self._raw:
            return
        self._repaint(self._block_cols if n_before is None else n_before)


# ---- selection menu ----

def select(title: str, options: list[str], *, reader: LineReader | None = None,
           default: int = 0, allow_cancel: bool = True) -> int | None:
    """Pick one option; returns its index or None when cancelled.

    Uses an arrow-key menu with type-to-filter and a scrolling viewport when
    the terminal supports raw mode; otherwise a numbered ``input()`` fallback.
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


def _view_limit(total: int) -> int:
    """How many option rows fit on screen: terminal-height driven, clamped
    so short menus look normal and huge ones never overflow the window."""
    available = max(4, _ui.terminal_height() - 6)
    return max(3, min(total, available))


def _render_rows(title: str, options: list[str], filtered: list[int],
                 cursor: int, query: str) -> list[str]:
    """One frame of the picker: title, a VIEWPORT of options, status footer.

    Every line is width-clamped so a long option can never wrap into a second
    physical row -- the redraw math assumes one screen row per rendered line.
    """
    width = _terminal_width()
    lines = [_ui.heading(
        _ui.clip(f"? {title}  (up/down, type to filter, Enter, Esc)", width))]
    total = len(filtered)
    limit = _view_limit(total)
    start = min(max(0, cursor - limit // 2), max(0, total - limit))
    window = filtered[start:start + limit]
    if not window:
        lines.append(_ui.dim("  (no matches)"))
    for i, idx in enumerate(window):
        pos = start + i
        label = _ui.clip(options[idx], width - 2)
        if pos == cursor:
            lines.append(_ui.selected(f"{_ui.GLYPH_POINT} {label}"))
        else:
            lines.append(f"  {label}")
    # status footer: position, scroll indicators, active filter
    bits = [f"{min(cursor + 1, total) if total else 0}/{total}"]
    above, below = start, max(0, total - (start + limit))
    if above:
        bits.append(f"^{above} more")
    if below:
        bits.append(f"v{below} more")
    if query:
        bits.append(f"filter: {query!r}")
    lines.append(_ui.dim(_ui.clip("  " + "  ".join(bits), width)))
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
            # start; move up THAT many and erase everything below, then rewrite.
            print(f"\x1b[{last_rows}A\x1b[2K\x1b[J", end="")
            for line in rows:
                print(line)
        last_rows = len(rows)
        try:
            key = reader.read_key(None)
        except KeyboardInterrupt:
            _clear_rows(last_rows)
            return None
        if key in ("up", "home"):
            cursor = (cursor - 1) % len(filtered) if filtered else 0
        elif key in ("down", "end"):
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
            chosen = _ui.clip(f"? {title}: {options[filtered[cursor]]}",
                              _terminal_width())
            print(_ui.accent(chosen))
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


_TARGET_LABEL_RE = re.compile(r"^(Q?\d+)-cable(\d+)$", re.IGNORECASE)


def _rack_cable_from_label(label: str | None) -> tuple[str, str] | None:
    """Session target labels like ``Q61-cable8`` prefill the rack/cable asks."""
    if not label:
        return None
    m = _TARGET_LABEL_RE.match(label.strip())
    return (m.group(1), m.group(2)) if m else None


def _golden_server_tunnel(*, reader: LineReader | None = None, inv=None,
                          store=None, target_label: str | None = None,
                        ) -> tuple[str, str, str] | None:
    """Ask which golden server hosts the model (rack/cable -- the same
    addressing as the debug target), capture the node login credentials
    fresh, probe the node over the jumpin console, and compose the tunnel
    ``HOST:PORT``. Discovered candidates are picked from arrow-key lists;
    manual entry is the fallback. Returns ``(spec, rack, cable, node_user)``
    or None on cancel."""
    from ..engine.sol import SerialProbeDenied
    from ..engine.tunnel import parse_tunnel_spec
    from ..targets.resolver import TargetError
    from .llm_discover import discover, llm_console

    prefill = _rack_cable_from_label(target_label)
    rack = ask_text("Rack of the golden server"
                    + (f" (Enter = {prefill[0]})" if prefill else " (e.g. Q61)"),
                    reader=reader).strip() or (prefill[0] if prefill else "")
    if not rack:
        return None
    cable = ask_text("Cable"
                     + (f" (Enter = {prefill[1]})" if prefill else " (e.g. 8)"),
                     reader=reader).strip() or (prefill[1] if prefill else "")
    if not cable:
        return None

    discovered = None
    node_user = ""
    if inv is not None and store is not None:
        # Node login capture -- prompted at EVERY model setup (the node user
        # and its sudo password are per-host and change over time). The
        # docker probe authenticates with this password via the sudo
        # handshake, and the login handshake reuses it (sudo and login share
        # the user's own password). The capture happens here where the
        # terminal is sane.
        node_user = ask_text("Node user on the LLM server",
                             reader=reader).strip()
        if not node_user:
            return None
        from .credential_gate import CredentialPrompter
        from .llm_discover import _node_sudo_path

        node_sudo_path = _node_sudo_path(rack)
        try:
            material = CredentialPrompter(store).prompt_now(node_sudo_path)
        except KeyError:
            print(_ui.warn(f"  ! node sudo password skipped -- {node_sudo_path} "
                           "stays unregistered; the docker probe will fail"),
                  file=sys.stderr)
            return None
        store.put(node_sudo_path, material)
        print(_ui.dim(f"  probing {rack}-cable{cable} over the console..."))
        try:
            discovered = discover(rack, cable, inv, store, node_user=node_user)
        except (TargetError, SerialProbeDenied) as exc:
            print(_ui.warn(f"  ! discovery failed: {exc}"), file=sys.stderr)
            print(_ui.dim("    continuing with manual entry"), file=sys.stderr)
        if discovered is not None and any(
                "not in pinned known_hosts" in n for n in discovered.notes):
            from .llm_discover import pin_llm_host_key

            if ask_text("Pin the rack manager's host key via the bastion "
                        "now? [y/N]", reader=reader).strip().lower() in (
                            "y", "yes"):
                try:
                    summary = pin_llm_host_key(rack, cable, inv, store)
                    print(_ui.good(f"  host key pinned: {summary}"))
                    discovered = discover(rack, cable, inv, store)
                except Exception as exc:  # noqa: BLE001 - staged, never fatal
                    print(_ui.warn(f"  ! pinning failed: {exc}"), file=sys.stderr)
                    print(_ui.dim("    continuing with manual entry"),
                          file=sys.stderr)
    host = port = None
    if discovered is not None:
        for note in discovered.notes:
            print(_ui.dim(f"    - {note}"))
        ports = discovered.suggested_ports()
        if ports:
            opts = [f"{p} [docker]" if p in discovered.containers.values()
                    else str(p) for p in ports]
            pidx = select("vLLM port", [*opts, "enter manually"], reader=reader)
            if pidx is None:
                return None
            port = str(ports[pidx]) if pidx < len(ports) else None
        addrs = discovered.addresses
        if addrs:
            aidx = select("Node HOST", [*addrs, "enter manually"], reader=reader)
            if aidx is None:
                return None
            host = addrs[aidx] if aidx < len(addrs) else None
    console = llm_console(inv) if inv is not None else None
    if console is not None:
        print(_ui.dim(f"  hop: rack manager {console.address_for_rack(rack)}; "
                      "HOST must be reachable from there"))
    if port is None:
        port = ask_text("vLLM port (Enter = 8000)", reader=reader).strip()
    if host is None:
        host = ask_text("Node HOST as the rack manager addresses it "
                        "(the node's own address, e.g. from `hostname -I`)",
                        reader=reader).strip()
    if not host:
        return None
    if console is not None and host in (
            console.address_for_rack(rack), console.address):
        print(_ui.warn("  ! that HOST is the rack manager's address -- the "
                       "tunnel must target the golden server's OWN address "
                       "(run `hostname -I` on the node)"), file=sys.stderr)
    spec = f"{host}:{port or '8000'}"
    try:
        parse_tunnel_spec(spec)
    except ValueError as exc:
        print(f"  x {exc}", file=sys.stderr)
        return None
    return spec, rack, cable, node_user


def ask_model_profile(*, reader: LineReader | None = None, provider: str | None = None,
                      inv=None, store=None, target_label: str | None = None):
    """Guided setup for a custom LLM ``ModelProfile``.

    Used by the ``+ add / configure a model`` row in the model picker (menu
    and ``/model`` in the REPL) and when an unconfigured built-in
    (``local/harness-diag`` / ``openai/harness-diag``) is selected -- the
    provider is then preselected and only the missing pieces are asked.

    Asks: provider (arrow-key select), endpoint URL (default
    ``http://127.0.0.1:8000/v1``; skipped for gemini), then transport:

    - ``direct``: the endpoint is reachable from this workstation.
    - ``tunnel``: the model runs on the golden server -- the rack/cable debug
      target. The wizard asks rack/cable, probes the node over the jumpin
      console (addresses, listening ports, docker port mappings) and lets the
      operator pick the tunnel ``HOST:PORT`` from the candidates; manual
      entry is the fallback. ``target_label`` (the active debug target)
      prefills the rack/cable asks when it matches.

    The wizard probes ``GET /models`` -- direct endpoints with a plain
    request, tunnel endpoints through an ``LLMForward`` when ``inv``/``store``
    are supplied. A reachable endpoint lets the operator pick the served model
    id from a list (vLLM rejects any other name). A tunnel refused at the
    ``forward`` stage prints the reverse-tunnel relay recipe and offers to
    save the relay URL instead; other failures are warnings and the profile is
    saved anyway with a manual model id. Provider must be ``openai``,
    ``gemini`` or ``local``. Returns None when cancelled.
    """
    from ..config.model_catalog import ModelProfile
    from ..diagnosis.llm import LLMError, list_models
    from ..engine.tunnel import LLMForward, TunnelError, parse_tunnel_spec
    from .llm_discover import llm_bastion_domain, llm_console_domain

    if provider is None:
        idx = select("Provider", ["openai (OpenAI-compatible endpoint)",
                                  "gemini (Google Gemini)",
                                  "local (vLLM / llama.cpp / Ollama)"],
                     reader=reader)
        if idx is None:
            return None
        provider = ("openai", "gemini", "local")[idx]
    provider = provider.strip().lower()
    if provider not in ("openai", "gemini", "local"):
        print(f"  x unknown provider {provider!r} (openai | gemini | local)",
              file=sys.stderr)
        return None

    url = None
    tunnel = None
    if provider != "gemini":
        raw = ask_text("Endpoint URL (Enter = http://127.0.0.1:8000/v1)",
                       reader=reader).strip()
        url = raw or "http://127.0.0.1:8000/v1"
        tidx = select("Transport", [
            "direct -- endpoint reachable from this workstation",
            ("tunnel -- model runs on the golden server (the rack/cable "
             "debug target); HTTP via the rack-manager hop")],
            reader=reader)
        if tidx is None:
            return None
        if tidx == 1:
            golden = _golden_server_tunnel(reader=reader, inv=inv, store=store,
                                           target_label=target_label)
            if golden is None:
                return None
            tunnel, golden_rack, golden_cable, _node_user = golden

    ids: list[str] = []
    forward = None
    probe_url = None
    try:
        if tunnel:
            console = (llm_console_domain(inv, golden_rack, golden_cable)
                       if inv is not None else None)
            if inv is not None and store is not None and console is not None:
                host, port = parse_tunnel_spec(tunnel)
                forward = LLMForward(host, port, console, store,
                                     bastion=llm_bastion_domain(
                                         inv, golden_rack, golden_cable))
                try:
                    probe_url = forward.start()
                except TunnelError as exc:
                    forward.close()
                    forward = None
                    print(_ui.warn(f"  ! tunnel hop failed ({exc.stage}): {exc}"),
                          file=sys.stderr)
                    if exc.stage == "forward":
                        # rackmgr refuses / cannot route to the node (jumpin-only
                        # fleet): the sanctioned fallback is a reverse tunnel the
                        # operator runs from the node console, then the harness
                        # calls the relay like any local endpoint.
                        print(_ui.dim(
                            "    fallback: console onto the node and run\n"
                            f"      ssh -fN -R 127.0.0.1:18000:127.0.0.1:{port} "
                            "<relay-reachable-from-node>\n"
                            "    the workstation then reaches the model at "
                            "http://127.0.0.1:18000/v1"))
                        answer = ask_text(
                            "Save the relay URL as the endpoint instead? [y/N]",
                            reader=reader).strip().lower()
                        if answer in ("y", "yes"):
                            url = "http://127.0.0.1:18000/v1"
                            tunnel = None
                            probe_url = url
                        else:
                            print(_ui.dim(
                                f"    `harness llm check --tunnel {tunnel} "
                                "--inventory <path>` re-tests each stage"),
                                file=sys.stderr)
                    else:
                        print(_ui.dim(f"    `harness llm check --tunnel {tunnel} "
                                      "--inventory <path>` re-tests each stage"),
                                      file=sys.stderr)
            else:
                print(_ui.dim(f"  ! tunnel saved unprobed (`harness llm check "
                              f"--tunnel {tunnel} --inventory <path>` probes it)"),
                      file=sys.stderr)
        elif url:
            probe_url = url
        if probe_url:
            try:
                ids = list_models(probe_url, timeout=10.0)
                print(_ui.good(f"  endpoint reachable: {len(ids)} model(s) served"))
            except LLMError as exc:
                refused = getattr(forward, "forward_error", None) if forward else None
                print(_ui.warn(f"  ! endpoint unreachable: {exc}"), file=sys.stderr)
                if refused:
                    print(_ui.dim(f"    tunnel target refused: {refused} -- the "
                                  "HOST:PORT must point at the golden server's "
                                  "vLLM port as reachable from the manager "
                                  "(the node's own address from `hostname -I`, "
                                  "not the manager's)"), file=sys.stderr)
                print(_ui.dim(f"    `harness llm check --url {probe_url}` "
                              "stages the failure; saving anyway"),
                      file=sys.stderr)
    finally:
        if forward is not None:
            forward.close()

    model = ""
    if ids:
        midx = select("Served model id", [*ids, "+ type a model id manually"],
                      reader=reader)
        if midx is None:
            return None
        if midx < len(ids):
            model = ids[midx]
    if not model:
        model = ask_text("Model id (must match the server's served name)",
                         reader=reader).strip()
        if not model:
            return None
    vault = None
    if provider != "local":
        vault = ask_text("API key vault path (Enter = env fallback, "
                         "e.g. GEMINI_API_KEY)", reader=reader).strip() or None
    # A tunnel profile carries no direct URL: the hop owns the endpoint, and
    # persisting the placeholder would just be misleading in models.yaml.
    return ModelProfile(provider=provider, model=model,
                        url=None if tunnel else url,
                        api_key_vault_path=vault, tunnel=tunnel)


def check_profile(profile, *, url: str | None = None, inv=None, store=None,
                  reader: LineReader | None = None, rack: str = "",
                  cable: str = "") -> str | None:
    """Non-blocking post-pick sanity probe for a selected ``ModelProfile``.

    Confirms the endpoint answers ``GET /models`` and that the profile's model
    id is actually served (vLLM rejects any other name). Direct endpoints are
    probed as-is; tunnel endpoints through an ``LLMForward`` when
    ``inv``/``store`` are supplied (rack/cable select the per-rack manager);
    an explicit ``url`` override wins (the REPL passes its live forward URL).
    Gemini is skipped -- its ``/models`` needs the API key and a failing
    diagnosis surfaces that anyway. Every failure is a printed warning only; a
    run is never blocked. Returns a replacement model id when the operator
    picks one from the served list, else None.
    """
    from ..diagnosis.llm import LLMError, list_models
    from ..engine.tunnel import LLMForward, TunnelError, parse_tunnel_spec
    from .llm_discover import llm_bastion_domain, llm_console_domain

    if profile is None or profile.provider in ("stub", "gemini"):
        return None
    forward = None
    try:
        probe_url = url or profile.url
        if not probe_url and profile.tunnel:
            console = llm_console_domain(inv, rack, cable) if inv is not None else None
            if inv is None or store is None or console is None:
                print(_ui.dim(f"  (tunnel {profile.tunnel} not probed here -- "
                             "harness llm check --tunnel ... --inventory <path> "
                             "tests it)"))
                return None
            forward = LLMForward(*parse_tunnel_spec(profile.tunnel), console,
                                 store, bastion=llm_bastion_domain(inv, rack, cable))
            try:
                probe_url = forward.start()
            except TunnelError as exc:
                forward.close()
                print(_ui.warn(f"  ! tunnel hop failed ({exc.stage}): {exc}"))
                print(_ui.dim("    the run will fail fast; harness llm check "
                             "--tunnel ... stages each leg"))
                return None
        if not probe_url:
            return None  # env-default endpoint: nothing meaningful to probe
        try:
            ids = list_models(probe_url, timeout=10.0)
        except LLMError as exc:
            refused = getattr(forward, "forward_error", None)
            print(_ui.warn(f"  ! endpoint {probe_url} unreachable: {exc}"))
            if refused:
                print(_ui.dim(f"    tunnel target refused: {refused} -- the "
                              "HOST must be the golden server's own address "
                              "(from `hostname -I`), not the manager's"))
            print(_ui.dim(f"    `harness llm check --url {probe_url}` stages "
                         "the failure"))
            return None
        if any(m == profile.model or m.endswith("/" + profile.model)
               for m in ids):
            print(_ui.good(f"  endpoint ok: {len(ids)} model(s) served, "
                          f"{profile.model} available"))
            return None
        print(_ui.warn(f"  ! model {profile.model!r} is not served at {probe_url}"))
        if not ids:
            return None
        midx = select("Switch to a served model id?", [*ids, "keep it anyway"],
                      reader=reader)
        if midx is None or midx == len(ids):
            return None
        return ids[midx]
    finally:
        if forward is not None:
            forward.close()
