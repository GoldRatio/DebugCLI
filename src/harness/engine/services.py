"""Multi-service console routing: one run, several node console service ports.

``console_defaults.services`` maps service names (``bmc``, ``host``, ...) to
per-port configuration; :class:`MultiConsoleRunner` fans probe requests out to
one :class:`~engine.sol.ConsoleRunner` per service. Routing is DETERMINISTIC
and neutral to the agent: each service owns a program table (``ipmitool``/
``i2c*`` on the BMC access port, ``lspci``/``smartctl``/... on host SOL) and the
first configured service whose table lists the command's program serves it.
Unmatched commands go to the fallback service (first configured). The agent
never picks ports -- it never sees them.

Security/semantics are unchanged per service: every command still passes the
hard read-only gate and the console probe spec inside each per-service
``ConsoleRunner``, each service keeps its own probe cache/call log, and a
command routed to a service that lacks the tool degrades to an honest exit-127
"not found" result (the collectors already tolerate this) rather than a wrong
answer.
"""

from __future__ import annotations

import shlex


def _program_of_cmd(cmd: str) -> str | None:
    try:
        parts = shlex.split(cmd)
    except ValueError:
        return None
    if len(parts) >= 2 and parts[0] == "sudo" and parts[1] == "-S":
        parts = parts[2:]
    if not parts:
        return None
    base = parts[0].rsplit("/", 1)[-1]
    return base or None


class MultiConsoleRunner:
    """``Runner``-shaped facade over one ConsoleRunner per console service.

    ``execute``/``batch_execute`` route per command by program table; the
    per-service runners keep their own caches and call logs. ``calls`` is the
    ordered union across services (what the engine dedupes and the audit
    walks).
    """

    is_console = True
    force_read_only = True

    def __init__(self, runners: dict[str, object],
                 programs: dict[str, tuple[str, ...]]) -> None:
        if not runners:
            raise ValueError("MultiConsoleRunner needs at least one service")
        self._runners = runners
        # service -> frozenset of programs, checked in mapping insertion order
        self._programs: dict[str, frozenset[str]] = {
            name: frozenset(p) for name, p in programs.items() if p}
        self._fallback = ("default" if "default" in runners
                          else next(iter(runners)))
        self._on_probe = None

    @property
    def on_probe(self) -> object:
        return self._on_probe

    @on_probe.setter
    def on_probe(self, listener) -> None:
        self._on_probe = listener
        for runner in self._runners.values():
            runner.on_probe = listener

    @property
    def calls(self) -> list[object]:
        """Ordered union of every service's call log (the engine dedupes and
        the audit walks this)."""
        out: list[object] = []
        for runner in self._runners.values():
            out.extend(runner.calls)
        return out

    def route(self, argv: list[str]) -> str:
        return self.route_cmd(" ".join(argv) if argv else "")

    def route_cmd(self, cmd: str) -> str:
        prog = _program_of_cmd(cmd)
        if prog is not None:
            for service, progs in self._programs.items():
                if prog in progs and service in self._runners:
                    return service
        return self._fallback

    def execute(self, argv: list[str], timeout: float = 300.0):
        return self._runners[self.route(argv)].execute(argv, timeout=timeout)

    def batch_execute(self, cmds: list[str], timeout: float = 300.0):
        """Run several probes: grouped per service, one console session per
        service, results returned in the ORIGINAL command order."""
        if not cmds:
            return []
        groups: dict[str, list[int]] = {}
        for i, cmd in enumerate(cmds):
            groups.setdefault(self.route_cmd(cmd), []).append(i)
        results: dict[int, object] = {}
        for service, indexes in groups.items():
            group = [cmds[i] for i in indexes]
            done = self._runners[service].batch_execute(group, timeout=timeout)
            for i, res in zip(indexes, done):
                results[i] = res
        return [results[i] for i in range(len(cmds))]


def service_programs(domains: dict) -> dict[str, tuple[str, ...]]:
    """Effective program table per service from resolved ConsoleDomains."""
    return {name: getattr(dom, "programs", None) or ()
            for name, dom in domains.items()}


__all__ = ["MultiConsoleRunner", "_program_of_cmd", "service_programs"]
