"""Ticketing adapter: send a Diagnosis's actions for human approval/execution.

Stub interface so the operator surface is defined even before a backing system
(Jira/ServiceNow/email) is wired. Nothing here executes actions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..diagnosis.schema import Diagnosis


class Ticketing(ABC):
    @abstractmethod
    def submit(self, diagnosis: Diagnosis) -> str: ...
    @abstractmethod
    def status(self, ticket_id: str) -> str: ...


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