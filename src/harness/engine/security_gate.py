"""Security gate: read-only validator consulted BY the runner.

Located in ``engine`` (owns security enforcement) so it cannot be bypassed by a
future command path. Collectors and the planner never call this directly -- they
only go through ``runner.execute``, which applies it. ``check`` is also exposed
for tests and for the audit layer to re-assert the invariant at record time.
"""

from __future__ import annotations


class ReadOnlyViolation(Exception):
    """Raised when an argv implies write intent (or is otherwise denied).

    Deliberately NOT a frozen dataclass: Exception subclasses must allow Python to
    attach ``__traceback__``; a frozen ``__setattr__`` breaks the exception machinery.
    """

    def __init__(self, argv: list[str], reason: str) -> None:
        self.argv = argv
        self.reason = reason
        super().__init__(reason)

    def __str__(self) -> str:
        return f"read-only violation ({self.reason}): {self.argv!r}"


# Destructive-program names that are NEVER permitted, even if a template matched,
# plus a set of hard-banned write-ish flag substrings.
BANNED_PROGRAMS = frozenset({
    "dd", "mkfs", "mkfs.*", "mpart", "shutdown", "reboot", "poweroff",
    "flashrom", "setpci", "nvram", "fwupd", "wipefs", "fdisk", "parted",
    "echo", "tee", "bash", "sh", "python", "perl", "ruby",
})

# Any argv containing these is denied regardless of allowlist template.
BANNED_TOKENS = frozenset({
    "-w",  # setpci write / generic write flag
    "--write",
    ">", "<", ">>", "|", ";", "&", "$(", "`",  # shell metacharacters
})


def check(argv: list[str]) -> None:
    """Raise ``ReadOnlyViolation`` if ``argv`` implies any write intent."""
    if not argv:
        raise ReadOnlyViolation(argv, "empty argv")

    prog = argv[0]
    core = prog.split("/")[-1]
    if core in BANNED_PROGRAMS or any(fnmatch_like(core, p) for p in BANNED_PROGRAMS):
        raise ReadOnlyViolation(argv, f"banned program {core!r}")

    for token in argv[1:]:
        if token in BANNED_TOKENS or token.startswith("--write"):
            raise ReadOnlyViolation(argv, f"banned token {token!r}")


def fnmatch_like(core: str, pattern: str) -> bool:
    if "*" not in pattern:
        return core == pattern
    import fnmatch
    return fnmatch.fnmatch(core, pattern)