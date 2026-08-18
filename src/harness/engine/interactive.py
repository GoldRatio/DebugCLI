"""Interactive SSH shell channel for driving target-side menus.

The probe paths (``SSHSession`` / ``SerialConsole``) run one-shot validated
commands; some vendor tooling is an interactive, menu-driven script instead
(e.g. the FAT single-test menu). ``InteractiveShell`` opens a persistent
``invoke_shell`` channel on an ALREADY-PINNED paramiko transport (host keys and
identity are handled by the owning ``SSHSession``) and provides line-oriented
send / read-until primitives with ANSI-normalized output, so a driver can walk
a number-driven menu exactly the way an operator would.
"""

from __future__ import annotations

import re
import time
from contextlib import suppress


class InteractiveShellError(RuntimeError):
    pass


# CSI sequences (colors, cursor moves, erase), two-char ESC sequences, BEL and
# backspace: all stripped so menu text can be matched with plain regexes.
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b[@-_]|[\x07\x08]")


def normalize_terminal(text: str) -> str:
    """Strip terminal control sequences and normalize line endings."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return _ANSI_RE.sub("", text)


class InteractiveShell:
    """Persistent shell channel over an open paramiko ``SSHClient``.

    The client's host key is already pinned (it comes from ``SSHSession``);
    this class never authenticates on its own. Output is accumulated into a
    readable-but-unread pending buffer; ``read_until``/``read_until_quiet``
    consume it and return the captured chunk.
    """

    def __init__(self, client, *, chunk_s: float = 0.2) -> None:
        self._client = client
        self._chunk_s = chunk_s
        self._chan = None
        self._pending = ""
        self.log: list[str] = []  # every captured chunk, for artifacts/audit

    # ---- lifecycle ----

    def open(self, *, banner_timeout: float = 10.0) -> str:
        """Open the channel and wait for the first output (banner/prompt)."""
        if self._chan is not None:
            raise InteractiveShellError("shell already open")
        try:
            self._chan = self._client.invoke_shell(term="xterm", width=250, height=50)
        except Exception as exc:
            raise InteractiveShellError(f"invoke_shell failed: {exc}") from exc
        deadline = time.monotonic() + banner_timeout
        while not self._pending and time.monotonic() < deadline:
            self._pump()
            if not self._pending:
                time.sleep(self._chunk_s)
        return self._pending

    def close(self) -> None:
        if self._chan is not None:
            with suppress(Exception):
                self._chan.close()
            self._chan = None

    @property
    def alive(self) -> bool:
        chan = self._chan
        return chan is not None and not chan.closed

    # ---- output ----

    def _pump(self) -> None:
        """Move whatever the channel has into the pending buffer."""
        if self._chan is None:
            return
        while self._chan.recv_ready():
            raw = self._chan.recv(65536)
            if not raw:
                break
            chunk = normalize_terminal(raw.decode(errors="replace"))
            if chunk:
                self._pending += chunk
                self.log.append(chunk)

    def read_until(self, patterns: list[str], timeout: float) -> str:
        """Read until any regex ``pattern`` matches the pending stream."""
        compiled = [re.compile(p) for p in patterns]
        deadline = time.monotonic() + timeout
        captured = ""
        while True:
            self._pump()
            if self._pending:
                captured += self._pending
                self._pending = ""
            if any(c.search(captured) for c in compiled):
                return captured
            if time.monotonic() >= deadline:
                raise InteractiveShellError(
                    f"timed out after {timeout}s waiting for {patterns!r}; "
                    f"last output: {captured[-400:]!r}")
            if not self.alive:
                raise InteractiveShellError(
                    f"channel closed while waiting for {patterns!r}; "
                    f"last output: {captured[-400:]!r}")
            time.sleep(self._chunk_s)

    def read_until_quiet(self, quiet_s: float, timeout: float) -> str:
        """Read until no new data arrives for ``quiet_s`` (menus wait for input).

        The most robust completion signal for an unknown menu format: a menu
        blocks on stdin, so the output stream goes quiet once it is fully
        painted. ``timeout`` bounds the total wait.
        """
        deadline = time.monotonic() + timeout
        captured = ""
        quiet_deadline = time.monotonic() + quiet_s
        while True:
            self._pump()
            if self._pending:
                captured += self._pending
                self._pending = ""
                quiet_deadline = time.monotonic() + quiet_s
            if captured and time.monotonic() >= quiet_deadline:
                return captured
            if time.monotonic() >= deadline:
                return captured  # best effort: return whatever arrived
            if not self.alive:
                return captured
            time.sleep(min(self._chunk_s, quiet_s / 4 or 0.05))

    # ---- input ----

    def send_line(self, text: str) -> None:
        self.send_raw(text + "\r")

    def send_raw(self, data: str) -> None:
        if self._chan is None or not self.alive:
            raise InteractiveShellError("shell not open")
        try:
            self._chan.sendall(data.encode())
        except Exception as exc:
            raise InteractiveShellError(f"send failed: {exc}") from exc