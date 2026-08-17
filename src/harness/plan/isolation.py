"""Fault-isolation pass: turn a decoded rail fault into a deeper debug round.

A latched power-rail fault register (``*_pwrup_flt`` / ``*_rtime_flt`` /
``*_vr_flt`` asserted, or a VR fault OR) is a FAILURE POINT, not a root cause:
a rail that failed to come up is downstream of every load it feeds, so a shorted
load, a connector/busbar, or the supplying board can all produce the same
register. This module detects that signature from the decoded evidence and
builds a targeted retrieval query -- from the fault's OWN rail tokens plus a
generic isolation vocabulary -- so the doc library surfaces the documented
isolation procedure and the read-only probes that discriminate which component
on the rail failed.

The logic is data-driven: the query is derived from the decoded field names,
never from a hardcoded per-FRU check. If the library grows other platforms,
the same vocabulary retrieves their power-isolation pages as long as they use
the same terms ("power fault", "isolate", "measure", "impedance", "busbar").

Only LATCHED fault registers trigger the pass. ``run_power_pg`` /
``run_power_fault_l`` are deliberately NOT triggers on their own: a powered-off
or standby system legitimately shows RUN power absent, and those signals stay in
the decoded evidence for the LLM without forcing an isolation round.
"""

from __future__ import annotations

import re

from ..inspect.base import RegisterDecode

# Field-name suffixes that mark a decoded rail fault (failure point).
_RAIL_FAULT_SUFFIXES = ("_pwrup_flt", "_rtime_flt", "_vr_flt")

# Tokens stripped when deriving rail tokens from a register field name.
_STOP_TOKENS = {
    "flt", "fault", "faults", "pwrup", "rtime", "fail", "all", "or", "vr",
    "cb", "mod", "rail", "n", "l", "en", "status",
}

# Live-sensor rail hints: a non-ok sensor whose name matches one of these
# contributes rail tokens too (e.g. "P12V_CB2_VOLT" -> 12v/cb2/volt).
_SENSOR_RAIL_HINTS = (
    "volt", "pwr", "12v", "50v", "48v", "5v", "3v3", "1v", "0v8",
)

# Generic isolation vocabulary appended to the rail tokens. Drawn from the
# documented power-isolation procedures; deliberately not platform-specific.
_ISOLATION_VOCAB = [
    ("rail power-up fault isolate power distribution board busbar measure "
     "impedance load"),
    "power fault isolate failure FPGA register dump power sequence",
]

_LIVE_SENSOR_RE = re.compile(r"\[current\]\s*(\S+)\s*\|")


def _field_value(field) -> str:
    return str(getattr(field, "raw_value", "") or "").strip().lower()


def _is_asserted(value: str) -> bool:
    return value not in ("", "0", "0x0", "00", "0x00")


def _rail_tokens_from_field(name: str, suffix: str) -> list[str]:
    stem = name.removesuffix(suffix)
    return [t for t in stem.split("_")
            if len(t) >= 2 and t not in _STOP_TOKENS]


def detect_fault_signature(
        decoded: list[RegisterDecode],
        summaries=None) -> dict | None:
    """Return a rail-fault signature or None when no isolatable fault is present.

    ``decoded`` is the catalog-decoded register list; ``summaries`` (an
    ``EvidenceSummary``) contributes live-sensor rail anomalies. A signature is
    ``{"rail_tokens": "...", "reasons": [...]}``; returning None skips the
    isolation pass entirely.
    """
    rail_tokens: list[str] = []
    reasons: list[str] = []
    seen: set[str] = set()

    def _add_token(token: str) -> None:
        token = token.strip().lower()
        if len(token) >= 2 and token not in _STOP_TOKENS and token not in seen:
            seen.add(token)
            rail_tokens.append(token)

    for reg in decoded:
        for field in reg.decoded_fields:
            name = (field.name or "").lower()
            value = _field_value(field)
            asserted = _is_asserted(value)
            for suffix in _RAIL_FAULT_SUFFIXES:
                if name.endswith(suffix) and asserted:
                    for tok in _rail_tokens_from_field(name, suffix):
                        _add_token(tok)
                    reasons.append(f"{reg.mnemonic}.{field.name}")
            if name == "vr_fail_all_or" and asserted:
                reasons.append(f"{reg.mnemonic}.{field.name} (VR fault OR)")

    if summaries is not None:
        for line in getattr(summaries, "interesting", None) or []:
            if not line.startswith("[current]"):
                continue
            lowered = line.lower()
            if not any(hint in lowered for hint in _SENSOR_RAIL_HINTS):
                continue
            m = _LIVE_SENSOR_RE.match(line)
            if m is None:
                continue
            for tok in re.split(r"[_\s.]+", m.group(1).lower()):
                _add_token(tok)

    if not reasons:
        return None
    return {"rail_tokens": " ".join(rail_tokens) or "power rail",
            "reasons": reasons}


def build_isolation_queries(signature: dict) -> list[str]:
    """Data-driven isolation retrieval queries for a detected rail fault."""
    rail = (signature or {}).get("rail_tokens") or "power rail"
    return [f"{rail} {vocab}".strip() for vocab in _ISOLATION_VOCAB]
