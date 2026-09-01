"""Redfish collector: rack-level read-only evidence, no server jumpin.

The rack manager serves a per-node Redfish API on its own HTTPS port
(``engine.redfish`` for the transport). The operator's rack tooling fetches
two things from it for any diagnosis -- the node's event log (sorted by
``Created``) and its service conditions -- and this collector brings exactly
those into the debug step as ordinary ``RegisterDump`` evidence. No serial
session is opened for it, and the evidence arrives even when the console hop
is the thing that is broken.

Every fetch is recorded honestly: a non-200 status or an unreachable service
produces a failed dump in the record (like a denied probe), never an
exception, so one broken endpoint cannot stall the diagnosis pipeline.
"""

from __future__ import annotations

import json

from ...engine.redfish import RedfishError, RedfishGet
from ..base import RegisterDump
from . import Collector

# The two endpoints rack-side diagnostics actually use. Deliberately small:
# each response rides into the LLM context, so breadth costs prompt budget.
_REDFISH_ENDPOINTS: tuple[tuple[str, str], ...] = (
    ("/Systems/HGX_Baseboard_0/LogServices/EventLog/Entries", "event_log"),
    ("/ServiceConditions", "service_conditions"),
)


def _sort_event_log(body: str) -> str:
    """Sort the event-log ``Members`` by ``Created`` (get_event_logs parity).

    Anything that does not parse as the expected shape is returned untouched:
    the raw body stays on disk for audit either way.
    """
    try:
        doc = json.loads(body)
    except ValueError:
        return body
    members = doc.get("Members") if isinstance(doc, dict) else None
    if not isinstance(members, list):
        return body
    doc["Members"] = sorted(
        members,
        key=lambda m: str(m.get("Created", "")) if isinstance(m, dict) else "")
    return json.dumps(doc, indent=2)


class RedfishCollector(Collector):
    """Fetch the Redfish evidence set for one node over a ``RedfishClient``."""

    subsystem = "redfish"

    def __init__(self, client) -> None:
        super().__init__(runner=None)
        self.client = client

    def collect(self, **kwargs) -> list[RegisterDump]:
        dumps: list[RegisterDump] = []
        try:
            pre = self.client.preflight()
        except RedfishError as exc:
            return [_staged("<preflight>", "preflight",
                            f"preflight failed ({exc.stage}): {exc}")]
        dumps.append(_from_get("<preflight>", "preflight", pre))

        for path, kind in _REDFISH_ENDPOINTS:
            try:
                result = self.client.get(path)
            except RedfishError as exc:
                dumps.append(_staged(path, kind,
                                     f"fetch failed ({exc.stage}): {exc}"))
                continue
            if kind == "event_log" and result.status == 200:
                body, truncated = _sort_event_log(result.body), result.truncated
            else:
                body, truncated = result.body, result.truncated
            dumps.append(_from_get(path, kind, result, body, truncated))
        return dumps


def _from_get(path: str, kind: str, result: RedfishGet,
              body: str | None = None, truncated: bool = False) -> RegisterDump:
    return RegisterDump(
        subsystem="redfish",
        source=f"GET {path}" if path else "GET <service root>",
        raw=body if body is not None else result.body,
        cmd_argv=[f"GET {path}" if path else "GET /"],
        ok=result.status == 200,
        meta={"kind": kind, "status": result.status,
              "elapsed_ms": result.elapsed_ms,
              "transport": result.transport,
              "truncated": truncated or result.truncated},
    )


def _staged(path: str, kind: str, message: str) -> RegisterDump:
    return RegisterDump(
        subsystem="redfish",
        source=f"GET {path}" if path else "GET <service root>",
        raw=message,
        cmd_argv=[f"GET {path}" if path else "GET /"],
        ok=False,
        meta={"kind": kind, "status": None},
    )
