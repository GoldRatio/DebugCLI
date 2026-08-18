"""Operator-supplied test-harness log evidence (--test-log).

Parses factory/harness run logs (e.g. Quanta FAT ``.log`` files) into a
structured ``TestLogReport`` (metadata + deduped failures with error codes like
``P02002001@PCIe Test Fail``) so the diagnosis pipeline can show the failures to
the agent, seed doc retrieval, and record them on the fleet learning loop.

The load path goes through a ``LogSource`` seam (``provider.py``): files today,
a website fetcher later, without touching the pipeline.
"""

from .model import TestLogFailure, TestLogReport
from .parse import parse_test_log
from .provider import FileLogSource, LogSource, LogSourceError

__all__ = [
    "FileLogSource",
    "LogSource",
    "LogSourceError",
    "TestLogFailure",
    "TestLogReport",
    "parse_test_log",
]