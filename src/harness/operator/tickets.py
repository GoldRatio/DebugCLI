"""Ticketing adapter: send a Diagnosis's actions for human approval/execution.

Stub interface so the operator surface is defined even before a backing system
(Jira/ServiceNow/email) is wired. Nothing here executes actions.

Prompt 05: ``record_outcome`` records what actually happened to a ticket so the
case store is populated from the approval flow too. The ABC default raises
``NotImplementedError`` (backing system has no learning-loop write); the
harness wraps the call so a missing implementation never crashes the flow.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..diagnosis.schema import Diagnosis


class Ticketing(ABC):
    @abstractmethod
    def submit(self, diagnosis: Diagnosis) -> str: ...
    @abstractmethod
    def status(self, ticket_id: str) -> str: ...

    def record_outcome(self, ticket_id: str, outcome: str,
                       actions_taken: list[str]) -> str:
        """Record what actually happened; ``raise NotImplementedError`` unless
        the backing system closes the loop (harness handles the absence)."""
        raise NotImplementedError(
            f"ticketing backend has no record_outcome (ticket {ticket_id})")


class NoOpTicketing(Ticketing):
    """Prints the action list; returns a fake ticket id. For local/lab."""

    def submit(self, diagnosis: Diagnosis) -> str:
        ticket = f"DIAG-{abs(hash(diagnosis.diagnosis)) % 100000:05d}"
        print(f"[tickets] Submitted {ticket}:")
        for action in diagnosis.actions:
            print(f"  {action.step}. {action.action} [{action.risk.value}] -> {action.required_tool}")
        return ticket

    def status(self, ticket_id: str) -> str:
        return "open"

    def record_outcome(self, ticket_id: str, outcome: str,
                       actions_taken: list[str]) -> str:
        """Local path: print the ``harness report`` command that closes the
        case-loop for ``ticket_id`` and hand back the case id."""
        taken = " ".join(f"--taken \"{a}\"" for a in actions_taken)
        print(f"[tickets] Recorded {ticket_id}: outcome={outcome}")
        print(f"  harness report --run {ticket_id} --outcome {outcome} {taken}"
              .rstrip())
        return ticket_id


def record_outcome_safe(ticketing: Ticketing, ticket_id: str, outcome: str,
                        actions_taken: list[str]) -> str:
    """Call ``record_outcome`` without crashing when the backend lacks it.

    The prompt 05 contract: the ABC default raises ``NotImplementedError`` and
    the HARNESS wraps the call so a backing system without the learning-loop
    write keeps the flow alive (returns a status line regardless).
    """
    try:
        return ticketing.record_outcome(ticket_id, outcome, actions_taken)
    except NotImplementedError:
        return (f"ticketing backend has no record_outcome; "
                f"close manually: harness report --run {ticket_id} "
                f"--outcome {outcome}")