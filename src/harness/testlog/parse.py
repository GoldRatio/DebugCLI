"""Parser for factory/test-harness run logs (Quanta FAT format and friends).

The reference format (see ``harness_docs/pdfs/quanta_qmf_fat_*.log``):

    ERROR:2026-08-11 22:02:01 || [FAIL][P02002001@PCIe Test Fail]
    ERROR:2026-08-11 22:02:01 || PCIe compare check test Failed! ...
    ...
    DEBUG:2026-08-11 22:02:01 || [END] pcie_cmp_chk
    DEBUG:2026-08-11 22:02:01 || FAIL: pcie_cmp_chk
    DEBUG:2026-08-11 22:02:01 || AssertionError: pcie_cmp_chk failed with code 1

Header metadata comes from ``INFO`` lines (``Model : T6T``, ``Chassis SN:``,
``Station:``, ``Test Stage:``, ``Build Site:``). Coded ``[FAIL][<code>@<desc>]``
entries are deduped across repeated test runs; uncoded ``[FAIL] ...`` lines are
kept as generic failures. When nothing structured matches, the tail of the file
is kept as a raw excerpt so the agent still gets evidence.

Security: log content is untrusted input (it can contain cleartext passwords,
e.g. ``sshpass -p '...'``). Every line that could reach a prompt, audit event,
or case record is redacted before it leaves this module.
"""

from __future__ import annotations

import re

from .model import TestLogFailure, TestLogReport

#: Max failures kept from one log (prompt/case budget).
_MAX_FAILURES = 10
#: Max context lines captured per failure.
_MAX_CONTEXT = 30
#: Tail kept as raw-excerpt fallback.
_EXCERPT_CHARS = 4000

# level:timestamp || message  (levels are padded to a fixed width before ':')
_LINE_RE = re.compile(
    r"^(?P<level>ERROR|INFO|DEBUG|WARN|NOTICE|CRITICAL)\s*:"
    r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \|\| (?P<msg>.*)$")

_FAIL_CODE_RE = re.compile(
    r"^\[FAIL\]\[(?P<code>[^\]]+)@(?P<desc>[^\]]+)\](?P<rest>.*)$")
_FAIL_GENERIC_RE = re.compile(r"^\[FAIL\](?P<rest>.*)$")
_FAIL_TEST_RE = re.compile(r"^FAIL:\s*(?P<test>\S+)")
_END_TEST_RE = re.compile(r"^\[END\]\s*(?P<test>\S+)")

_REDACT_PATTERNS = [
    re.compile(r"(?i)(sshpass\s+-p\s+['\"]?)[^\s'\"]+"),
    re.compile(r"(?i)(password|passwd)\s*[=:]\s*['\"]?[^\s'\"]+"),
]


def _redact(text: str) -> str:
    for pattern in _REDACT_PATTERNS:
        text = pattern.sub(lambda m: f"{m.group(1)}[REDACTED]", text)
    return text


def _grab_field(lines: list[str], key: str) -> str | None:
    """Extract a header field like ``Model     : T6T`` from INFO lines."""
    pattern = re.compile(r"\b" + re.escape(key) + r"\s*:\s*(.+)")
    for line in lines:
        m = _LINE_RE.match(line)
        if not m:
            continue
        mm = pattern.search(m.group("msg"))
        if mm:
            return mm.group(1).strip()
    return None


def _records(text: str) -> list[tuple[str, str, str]]:
    """(level, timestamp, message) triples; unparsed lines keep level ''."""
    out: list[tuple[str, str, str]] = []
    for line in text.splitlines():
        m = _LINE_RE.match(line)
        if m:
            out.append((m.group("level"), m.group("ts"), m.group("msg").strip()))
        else:
            out.append(("", "", line.strip()))
    return out


def parse_test_log(text: str, source: str = "<log>") -> TestLogReport:
    """Parse a test-harness log into a structured ``TestLogReport``.

    Never raises on odd content: unrecognized formats degrade to a raw excerpt
    so the pipeline always has evidence to show.
    """
    lines = text.splitlines()
    report = TestLogReport(
        source=source,
        model=_grab_field(lines, "Model"),
        serial=_grab_field(lines, "Chassis SN"),
        station=_grab_field(lines, "Station"),
        test_stage=_grab_field(lines, "Test Stage"),
        build_site=_grab_field(lines, "Build Site"),
        program_version=_grab_field(lines, "Test Program Version"),
    )
    records = _records(text)

    failures: dict[tuple[str, str], TestLogFailure] = {}
    for i, (level, ts, msg) in enumerate(records):
        m = _FAIL_CODE_RE.match(msg)
        if not m:
            continue
        code, desc = m.group("code").strip(), m.group("desc").strip()
        key = (code, desc)
        existing = failures.get(key)
        if existing is not None:
            existing.occurrences += 1
            continue
        failure = TestLogFailure(code=code, description=desc, first_seen=ts or None)
        context: list[str] = []
        for j in range(i + 1, len(records)):
            lv, _t, mm = records[j]
            if lv == "ERROR" and _FAIL_CODE_RE.match(mm):
                break
            if mm.startswith("[START]"):
                break
            if not mm:
                continue
            context.append(_redact(mm))
            if len(context) >= _MAX_CONTEXT:
                break
        failure.context = context
        for line in context:
            tm = _FAIL_TEST_RE.match(line)
            if tm:
                failure.test_name = tm.group("test")
                break
            em = _END_TEST_RE.match(line)
            if em:
                failure.test_name = em.group("test")
                break
        failures[key] = failure
        if len(failures) >= _MAX_FAILURES:
            break

    if not failures:
        # Uncoded [FAIL] lines (no code@desc bracket).
        for _level, ts, msg in records:
            gm = _FAIL_GENERIC_RE.match(msg)
            if not gm:
                continue
            desc = gm.group("rest").strip() or "harness test failure"
            key = ("", desc)
            if key not in failures:
                failures[key] = TestLogFailure(
                    description=desc, first_seen=ts or None,
                    context=[_redact(msg)],
                )
            if len(failures) >= _MAX_FAILURES:
                break

    report.failures = list(failures.values())
    if not report.failures:
        report.raw_excerpt = _redact(text[-_EXCERPT_CHARS:])
    return report