"""Log-source seam: where a test-harness log comes from.

``LogSource`` is the one interface the pipeline depends on; ``FileLogSource``
reads local files today, and a website fetcher (auth via the secret store) can
slot in later as a second implementation without touching ``cli.py`` or the
engines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .model import TestLogReport
from .parse import parse_test_log


class LogSourceError(RuntimeError):
    """Raised when a log source cannot produce a usable report."""


class LogSource(Protocol):
    def load(self) -> TestLogReport:
        """Return a parsed, redacted ``TestLogReport`` for one log."""


class FileLogSource:
    """Read a local test-harness log file (UTF-8, tolerant of bad bytes)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> TestLogReport:
        try:
            text = self.path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise LogSourceError(f"cannot read {self.path}: {exc}") from exc
        return parse_test_log(text, source=str(self.path))