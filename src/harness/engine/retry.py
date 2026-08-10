"""Retry policy + idempotent-command registry (network-flakiness handling).

Read-only probes are safe to retry. Retries are bounded, backoff is jittered, and
only commands marked idempotent are retried (a probe is idempotent by construction,
but we still only retry when the target explicitly allows it).
"""

from __future__ import annotations

from dataclasses import dataclass
import random
import time

from .runner import CommandResult, Runner


@dataclass
class RetryPolicy:
    max_attempts: int = 3
    base_delay_s: float = 0.5
    max_delay_s: float = 5.0
    jitter: float = 0.3


class RetryExecutor:
    """Wraps a Runner and retries transient failures for idempotent probes."""

    _IDEMPOTENT = frozenset({
        "dmidecode", "lspci", "ipmitool", "rdmsr", "smartctl", "nvme",
        "dmesg", "lsblk", "lsmod",
    })

    def __init__(self, runner: "Runner", policy: RetryPolicy | None = None) -> None:
        self.runner = runner
        self.policy = policy or RetryPolicy()

    def execute(self, argv: list[str], timeout: float = 30.0) -> CommandResult:
        attempt = 0
        last: CommandResult | None = None
        can_retry = self.runner.force_read_only and self._idempotent(argv)
        while True:
            attempt += 1
            last = self.runner.execute(argv, timeout=timeout)
            if last.ok or attempt >= self.policy.max_attempts or not can_retry:
                return last
            delay = min(self.policy.max_delay_s,
                        self.policy.base_delay_s * (2 ** attempt) + random.uniform(0, self.policy.jitter))
            time.sleep(delay)

    def _idempotent(self, argv: list[str]) -> bool:
        return argv[0].split("/")[-1] in self._IDEMPOTENT