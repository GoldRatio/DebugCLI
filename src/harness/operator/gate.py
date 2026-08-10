"""Human approval gate -- no automated execution by default.

Every action requires explicit human approval before it may be handed to a ticketing
/execution system. Failed or skipped approvals are recorded in the audit log. This
gate is the ONLY bridge between recommendations and any real-world effect.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..diagnosis.schema import Action


@dataclass
class ApprovalDecision:
    action: Action
    approved: bool
    note: str | None = None


class ApprovalGate:
    """Interactive prompt for each action. A falsy answer is treated as DENIED.

    In a managed deployment this is replaced by a ticketing adapter
    (``operator.tickets``) but the contract is the same: every action is either
    approved or not, and the decision is recorded.
    """

    def prompt(self, action: Action, session_id: str) -> ApprovalDecision:
        print(f"\n-- Action {action.step}: {action.action}")
        print(f"   rationale: {action.rationale}")
        print(f"   impact   : {action.impact}")
        print(f"   risk     : {action.risk.value}")
        answer = input("Approve? [y/N] ").strip().lower()
        return ApprovalDecision(action=action, approved=answer in ("y", "yes"))

    def record(self, decision: ApprovalDecision, session_id: str,
               log=None) -> None:
        """Append the decision to the audit log (WORM) when one is provided.

        The default is a no-op so the gate stays usable without a log; the CLI
        always passes the run's AuditLog so every approval/denial is recorded.
        """
        if log is not None:
            log.append(session_id, "approval", {
                "step": decision.action.step,
                "approved": decision.approved,
                "action": decision.action.action,
                "note": decision.note,
            })