"""Anomalous-register summarizer.

The full register dump is large and must NOT all go into the LLM context (context
budget). This reduces it to the anomalous/interesting values only, keeping the
full dump on disk for audit. Mirrors the spec rule: summarize anomalous registers,
send full dump to storage, pass summary to LLM.

Evidence kind matters: a register dump is either CURRENT STATE (live sensor
readings, MSR values, kernel ring buffer) or a HISTORICAL record (the IPMI SEL is
a past-event log whose entries may predate repairs and are never deasserted).
The summarizer tags SEL lines with their recorded timestamp and emits a
current-state health note so the LLM can weigh live sensors over stale entries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..inspect.base import RegisterDump

# SEL entry shape: "  12 | 08/10/26 | 17:12:16 UTC | Power Supply #0x75 | Failure detected | Asserted"
_SEL_TS_RE = re.compile(
    r"^\s*\S+\s*\|\s*(\d{2}/\d{2}/\d{2})\s*\|\s*(\d{2}:\d{2}:\d{2})\s+UTC")

# ipmitool sensor status codes that mean a real out-of-range/threshold trip.
_SENSOR_BAD = {"cr", "lnr", "lcr", "lnc", "ucr", "unc", "unr", "nr"}


@dataclass
class EvidenceSummary:
    interesting: list[str]      # human-readable lines worth sending to the LLM
    anomaly_count: int
    total: int
    notes: list[str] = field(default_factory=list)  # evidence-kind context lines
    current_health: str = "unknown"  # "ok" | "anomaly" | "unknown" from live sensors


def summarize(dumps: list[RegisterDump], max_items: int = 50) -> EvidenceSummary:
    """Keep registers that look anomalous, tagged by evidence kind, up to max_items.

    Health notes are computed from ALL dumps regardless of the ``max_items`` cap,
    so a verbose historical SEL cannot starve the live-sensor state assessment.
    """
    interesting: list[str] = []
    notes: list[str] = []
    sel_entries = 0
    sel_asserted_no_deassert = 0
    sel_fault_lines = 0
    sensor_statuses: dict[str, str] = {}

    for dump in dumps:
        if not dump.ok:
            continue
        kind = _dump_kind(dump)
        for line in dump.raw.splitlines():
            if kind == "sel":
                ts = _SEL_TS_RE.match(line)
                sel_entries += 1 if ts else 0
                if _SEL_FAULT_RE.search(line):
                    sel_fault_lines += 1
                    if "Deasserted" not in line:
                        sel_asserted_no_deassert += 1
                if len(interesting) < max_items and _looks_anomalous(line):
                    tag = f"[sel-historical {ts.group(1)} {ts.group(2)} UTC]" if ts \
                        else "[sel-historical]"
                    interesting.append(f"{tag} {line.strip()}")
            elif kind == "sensor":
                name, status = _parse_sensor_line(line)
                if name and status:
                    sensor_statuses[name] = status
                    if status not in ("ok", "na") and len(interesting) < max_items:
                        interesting.append(f"[current] {line.strip()}")
            else:
                if len(interesting) < max_items and _looks_anomalous(line):
                    interesting.append(f"[current] {line.strip()}")

    if sel_entries:
        notes.append(
            f"SEL is a HISTORICAL event log ({sel_entries} timestamped entries; "
            f"{sel_fault_lines} fault-marked). Entries record PAST events and are "
            f"NOT proof of a current fault; prefer live sensor state.")
        if sel_asserted_no_deassert:
            notes.append(
                f"{sel_asserted_no_deassert} fault entry(ies) are asserted without a "
                f"matching deassert; if they recur at every power-on they may be "
                f"expected boot sequencing rather than an active failure.")

    # Probes that did not run (missing tool, denied, non-zero) must be visible to
    # the LLM: an absent register dump is evidence of a gap, not of a healthy value.
    failed = [_probe_failure_note(d) for d in dumps if not d.ok]
    notes.extend(failed[:5])

    non_ok = [f"{n}={s}" for n, s in sensor_statuses.items()
              if s not in ("ok", "na")]
    if sensor_statuses:
        health = "anomaly" if non_ok else "ok"
        notes.append(
            f"Current live sensors: {health} "
            f"({len(sensor_statuses)} sensors; "
            + ("non-ok: " + ", ".join(non_ok) if non_ok else "all ok/na") + ").")
    else:
        health = "unknown"

    return EvidenceSummary(
        interesting=interesting,
        anomaly_count=len(interesting),
        total=sum(len(d.raw.splitlines()) for d in dumps),
        notes=notes,
        current_health=health,
    )


def _dump_kind(dump: RegisterDump) -> str:
    kind = (dump.meta or {}).get("kind")
    if kind in ("sensor", "sel", "fru", "dmesg", "i2c"):
        return kind
    source = dump.source
    if "sel list" in source:
        return "sel"
    if "fru print" in source:
        return "fru"
    if "sensor list" in source or ("sensor" in source and "sdr" not in source):
        return "sensor"
    if "i2cdump" in source or "i2cget" in source:
        return "i2c"
    if "dmesg" in source or "rdmsr" in source or "msr" in source:
        return "dmesg"
    return "other"


_SEL_FAULT_RE = re.compile(r"\b(fail(?:ure)?s?|faults?|error|critical)\b", re.IGNORECASE)

_FAILURE_HINTS = ("command not found", "No such file", "not found", "denied",
                  "Permission denied", "error", "failed")


def _probe_failure_note(dump: RegisterDump) -> str:
    """Short reason a probe dump failed, from its raw output (console banners)."""
    exit_code = (dump.meta or {}).get("exit")
    for hint in _FAILURE_HINTS:
        m = re.search(hint, dump.raw, re.IGNORECASE)
        if m:
            return f"probe failed: {dump.source} ({m.group(0)}; exit {exit_code})"
    return f"probe failed: {dump.source} (exit {exit_code})"

_ANOMALY_HINTS = ("error", "fail", "uncorrectable", "corrected", "warning",
                  "critical", "overflow", "sensor", "non-zero", "threshold",
                  "ecc", "mce", "0x", "bad", "fault", "degraded")


def _looks_anomalous(line: str) -> bool:
    lowered = line.lower()
    return any(h in lowered for h in _ANOMALY_HINTS)


def _parse_sensor_line(line: str) -> tuple[str | None, str | None]:
    """Extract (sensor name, status) from an ipmitool sensor line.

    Shape: "CPU0_Temp  | 45.000  | degrees C | ok  | na | na | ..." -- the status
    is the fourth field. Discrete sensors carry a hex state code instead of a
    textual status (e.g. "0x0180"); those are treated as readings, not faults.
    Non-sensor rows (headers, banners, prompts, SEL entries) return (None, None).
    """
    if "|" not in line or _SEL_TS_RE.match(line):
        return None, None
    parts = [p.strip() for p in line.split("|")]
    if len(parts) < 4:
        return None, None
    name, _reading, _units, status = parts[:4]
    if not name or not status or name.lower() in ("sensor", "sensor name", "reading"):
        return None, None
    if status.lower().startswith("0x"):
        return name, "ok"   # discrete state code; cannot assert a fault from it
    if status.lower() in _SENSOR_BAD:
        return name, status
    if status.lower() in ("ok", "na"):
        return name, status
    return None, None       # unrecognized status column -> not a sensor row
