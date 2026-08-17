"""Doc-guided planning: mine probe commands from retrieved doc snippets.

Subsystem classification is keyword-heuristic and cannot know which registers a
failure mode needs. The troubleshooting docs often name the exact probe command
(e.g. GB_HangUp p.1: "For EVERY Amber Light issue, dump ``i2cdump -y 8 0xb`` in
the BMC to get the boot state"). This module scans retrieved snippets for those
commands and returns the ones that pass the deny-by-default read-only probe gate
(``validate_serial_probe``) -- a doc typo can never smuggle a write past the gate.
"""

from __future__ import annotations

import re

from ..engine.sol import SerialProbeDenied, validate_serial_probe

# Command forms the docs quote; each pattern captures the whole command. The
# probe spec (not this regex) is the authoritative gate; patterns are deliberately
# narrow so prose that merely mentions a tool name is not enough.
_PROBE_PATTERNS = [
    # i2cdump [-y|-f|-a] <bus> 0x<addr> [-r 0x<lo>-0x<hi>]
    re.compile(
        r"\bi2cdump(?:\s+-[yfa]+)?\s+\d+\s+0x[0-9a-fA-F]+"
        r"(?:\s+-r\s+0x[0-9a-fA-F]+(?:-0x[0-9a-fA-F]+)?)?"),
    # i2cget [-y|-f|-a] <bus> 0x<addr> [0x<reg>]
    re.compile(r"\bi2cget(?:\s+-[yfa]+)?\s+\d+\s+0x[0-9a-fA-F]+(?:\s+0x[0-9a-fA-F]+)?"),
    # i2cdetect [-y|-a|-l|-q|-r] <bus>
    re.compile(r"\bi2cdetect(?:\s+-[yfalqr]+)?\s+\d+"),
    # i2ctransfer block read: `i2ctransfer -y <bus> [w<n>@0x..] 0x.. ... r<n>`
    # (the FPGA register dumps e.g. `i2ctransfer -y 2 w2@0x11 0x00 0x00 r256`).
    re.compile(
        r"\bi2ctransfer(?:\s+-[yfv]+)?\s+\d+"
        r"(?:\s+(?:[rw]\d+(?:@0x[0-9a-fA-F]{1,2})?|0x[0-9a-fA-F]{1,2}))+"),
    # ipmitool read-only forms: sdr [list|elist]; sensor list/elist; sel list/info;
    # fru print. Destructive forms ("sel clear", "sensor set") are not expressible.
    re.compile(r"\bipmitool\s+(?:sdr(?:\s+(?:list|elist))?|sensor\s+(?:list|elist)"
               r"|sel\s+(?:list|info)|fru\s+print)"),
    # dmesg with read-only flags (bare "dmesg" prose is not a probe)
    re.compile(r"\bdmesg(?:\s+-[xrtTEdu]+)+"),
]


def mine_probe_commands(snippets: list[str]) -> list[str]:
    """Return read-only probe commands named in doc snippets, in doc order.

    Every candidate must pass ``validate_serial_probe``; anything the probe gate
    rejects (writes, non-allowlisted tools, shell metacharacters) is dropped.
    """
    found: list[str] = []
    for snippet in snippets:
        text = snippet
        if text.startswith("["):
            _, _, text = text.partition("]")
        for pattern in _PROBE_PATTERNS:
            for match in pattern.finditer(text):
                candidate = match.group(0).strip("\"'")
                if candidate in found:
                    continue
                try:
                    validate_serial_probe(candidate)
                except SerialProbeDenied:
                    continue
                found.append(candidate)
    return found
