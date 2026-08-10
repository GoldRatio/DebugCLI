"""Session trace linking: summary <-> full command trace <-> raw log.

One ``session_id`` is the join key across every artifact. Read the spec for the
conflict between full auditability and LLM context budget: the *summary* is what
flows to the LLM, the *raw* log and *command trace* are the full-fidelity records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4


@dataclass
class SessionTrace:
    session_id: str = field(default_factory=lambda: uuid4().hex)
    summary: dict = field(default_factory=dict)
    command_trace: list[dict] = field(default_factory=list)
    raw_log_path: str | None = None

    def record_command(self, argv: list[str], exit_code: int, elapsed_ms: int,
                       stdout_sha: str | None = None) -> None:
        self.command_trace.append({
            "argv": argv,
            "exit": exit_code,
            "elapsed_ms": elapsed_ms,
            "stdout_sha": stdout_sha,
        })

    def link_raw_log(self, path: str) -> None:
        self.raw_log_path = path