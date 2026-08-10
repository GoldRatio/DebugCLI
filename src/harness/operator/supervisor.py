"""Per-step supervision: wall-clock + step budgets with escalation.

The engine calls ``check(label)`` between phases. Exceeding the step budget or
the wall-clock budget raises ``Escalation``, which the CLI turns into a hard stop
and an audit record -- the harness must never loop or crawl on a target.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


class Escalation(RuntimeError):
    pass


@dataclass
class RunSupervisor:
    max_steps: int = 8
    wall_s: float = 900.0
    _started: float = field(default_factory=time.monotonic)
    _steps: int = 0

    def check(self, label: str) -> None:
        self._steps += 1
        if self._steps > self.max_steps:
            raise Escalation(f"step budget exceeded at {label!r} ({self.max_steps} max)")
        elapsed = time.monotonic() - self._started
        if elapsed > self.wall_s:
            raise Escalation(f"wall-clock budget exceeded at {label!r} ({self.wall_s:.0f}s)")

    def elapsed(self) -> float:
        return time.monotonic() - self._started
