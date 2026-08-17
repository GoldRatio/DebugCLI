"""Documented power-topology edges: which loads sit on which rail.

A decoded rail fault (see ``plan.isolation``) names a failing rail but not its
loads. This module maps a rail back to the components it feeds so the diagnosis
can enumerate the suspect set instead of jumping to a single FRU. The edges are
EXTRACTED from the vendor architecture docs (page-referenced below), never
invented and never derived from a specific failure case -- the diagnosis stays
data-driven.

Two platforms are modeled today, each a distinct ``PowerTopology``:

* Samoa (C4A14 node, PDB 50V->12V) -- M1337855 System Spec, PDB connectors
  section 9.3 (pp. 85-86), power distribution pp. 41-42, busbar/FRU p. 55,
  electrical spec p. 99.
* NVL72 (rack node) -- GB_NVL72 troubleshooting, power issues pp. 17-19 and
  FPGA register dump p. 14.

Rails are matched from fault tokens by CONTAINMENT of the rail's significant
token set (e.g. input ``pdb 12v cb2 volt`` matches rail ``P12V_CB2`` whose
tokens are {12v, cb2}). This is robust to the rail tokens the isolation pass
strips (volt/flt/mod are noise to it but meaningful here when supplied as full
field names). When only an aggregate token is present (bare ``12v``) a generic
aggregate rail lists every documented 12V load rather than guessing a sub-rail.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ...platforms import family_for

SAMOA_SRC = "M1337855 Samoa System Spec"
NVL72_SRC = "GB_NVL72 Troubleshooting"

# Tokens that never disambiguate a rail (voltage units, status noise).
_STOP = {
    "v", "volt", "volts", "voltage", "pg", "pwrup", "rtime", "flt", "fault",
    "faults", "fail", "status", "en", "n", "l", "rail", "input", "output",
}

_TOKEN_RE = re.compile(r"[^a-z0-9]+")


def _tokens(text: str) -> set[str]:
    """Lowercased alphanumeric tokens of a rail name/alias/field name."""
    return {t for t in _TOKEN_RE.split(text.lower()) if t and len(t) >= 1}


@dataclass
class TopologyLoad:
    """One documented consumer on a rail, with its connector/connection detail."""

    name: str
    connection: str | None = None
    refs: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        out = {"name": self.name}
        if self.connection:
            out["connection"] = self.connection
        if self.refs:
            out["refs"] = list(self.refs)
        return out


@dataclass
class Rail:
    """A power rail with its alias token variants and documented loads."""

    key: str
    aliases: tuple[str, ...]
    voltage: str
    loads: list[TopologyLoad]
    refs: list[str]
    aggregate: bool = False  # matches any rail token but only used as a fallback

    def _match_token_sets(self) -> list[set[str]]:
        """One token set per key/alias spelling; a rail matches when ANY of its
        spellings is fully contained in the input tokens."""
        return [{t for t in _tokens(alias) if t not in _STOP}
                for alias in (self.key, *self.aliases)]

    def as_dict(self) -> dict:
        return {
            "platform": self.platform,  # type: ignore[attr-defined]
            "rail": self.key,
            "voltage": self.voltage,
            "refs": list(self.refs),
            "loads": [load.as_dict() for load in self.loads],
        }


@dataclass
class PowerTopology:
    """One platform's documented power distribution (rail -> loads)."""

    platform: str
    platform_aliases: tuple[str, ...]
    rails: list[Rail] = field(default_factory=list)

    def _post_init(self) -> None:
        for rail in self.rails:
            rail.platform = self.platform  # type: ignore[attr-defined]


def _samoa() -> PowerTopology:
    p = PowerTopology(
        platform="samoa",
        platform_aliases=("samoa", "c4a14", "c4a15", "c4a14_server",
                          "m1337855", "grace_hopper", "gb200"),
    )
    p.rails = [
        Rail(
            key="P50V_STBY",
            aliases=("50v_stby", "50v stby", "50v standby"),
            voltage="50V DC (PDB input)",
            loads=[
                TopologyLoad(
                    "Standby rails + non-GB200 boards (DC-SCM/HMC)",
                    connection="50V HSC standby (OC alert 10.1A)",
                    refs=[f"{SAMOA_SRC} p.41"]),
            ],
            refs=[f"{SAMOA_SRC} pp.41,99"],
        ),
        Rail(
            key="P50V_CB1",
            aliases=("50v_cb1",),
            voltage="50V DC (PDB input)",
            loads=[
                TopologyLoad(
                    "1st GB200 (Bianca) power train",
                    connection="50V HSC CB1 (OC alert 69.5A) -> 50V-12V VR#1 "
                               "(up to 4000W) powering the 1st GB200 and its fans",
                    refs=[f"{SAMOA_SRC} p.41-42"]),
            ],
            refs=[f"{SAMOA_SRC} pp.41-42"],
        ),
        Rail(
            key="P50V_CB2",
            aliases=("50v_cb2",),
            voltage="50V DC (PDB input)",
            loads=[
                TopologyLoad(
                    "2nd GB200 (Bianca) power train",
                    connection="50V HSC CB2 (OC alert 69.5A) -> 50V-12V VR#2 "
                               "(up to 4000W) powering the 2nd GB200 and its fans",
                    refs=[f"{SAMOA_SRC} p.41-42"]),
            ],
            refs=[f"{SAMOA_SRC} pp.41-42"],
        ),
        Rail(
            key="P12V_STBY",
            aliases=("12v_stby", "12v stby"),
            voltage="12V DC",
            loads=[
                TopologyLoad(
                    "1st Bianca (GB200)",
                    connection="PDB connector J4 (HMHA010-L2A32-9H), P12V_STBY",
                    refs=[f"{SAMOA_SRC} p.86"]),
                TopologyLoad(
                    "2nd Bianca (GB200)",
                    connection="PDB connector J5 (HMHA010-L2A32-9H), P12V_STBY",
                    refs=[f"{SAMOA_SRC} p.86"]),
                TopologyLoad(
                    "SWB (switch board)",
                    connection="PDB connector J6 (HMHA033-L2A11-9H), P12V_STBY",
                    refs=[f"{SAMOA_SRC} p.86"]),
            ],
            refs=[f"{SAMOA_SRC} p.86"],
        ),
        Rail(
            key="P12V_CB1",
            aliases=("12v_cb1",),
            voltage="12V DC",
            loads=[
                TopologyLoad(
                    "OSFP Board",
                    connection="PDB connector J8 (HMHA010-L2A32-9H), P12V_CB1",
                    refs=[f"{SAMOA_SRC} p.86"]),
            ],
            refs=[f"{SAMOA_SRC} p.86"],
        ),
        Rail(
            key="P12V_CB2",
            aliases=("12v_cb2",),
            voltage="12V DC",
            loads=[
                TopologyLoad(
                    "OSFP Board",
                    connection="PDB connector J9 (HMHA010-L2A32-9H), P12V_CB2",
                    refs=[f"{SAMOA_SRC} p.86"]),
            ],
            refs=[f"{SAMOA_SRC} p.86"],
        ),
        Rail(
            key="P12V_BUS",
            aliases=("12v bus", "p12v bus", "fan bus", "12v busbar", "busbar"),
            voltage="12V DC (fan/bus distribution)",
            loads=[
                TopologyLoad(
                    "GB200 fans",
                    connection="12V power bus bar from PDB; each fan has its own "
                               "12V eFuse on the GB200 (up to 8x 12V 40mm dual-rotor)",
                    refs=[f"{SAMOA_SRC} p.42"]),
                TopologyLoad(
                    "PDB Busbar L",
                    connection="FRU: PDB Busbar L",
                    refs=[f"{SAMOA_SRC} p.55"]),
                TopologyLoad(
                    "PDB Busbar R",
                    connection="FRU: PDB Busbar R",
                    refs=[f"{SAMOA_SRC} p.55"]),
            ],
            refs=[f"{SAMOA_SRC} pp.42,55"],
        ),
        Rail(
            key="12V_ALL",
            aliases=("12v",),
            voltage="12V DC",
            loads=[
                TopologyLoad(
                    "All 12V loads",
                    connection="PDB 12V rails feed Biancas (J4/J5), SWB (J6), "
                               "OSFP boards (J8/J9), and the fan busbar",
                    refs=[f"{SAMOA_SRC} p.86"]),
            ],
            refs=[f"{SAMOA_SRC} p.86"],
            aggregate=True,
        ),
    ]
    p._post_init()
    return p


def _nvl72() -> PowerTopology:
    p = PowerTopology(
        platform="nvl72",
        platform_aliases=("nvl72", "nvl", "gb_nvl72", "nvl72_rack",
                          "nvl rack", "gb200_nvl72"),
    )
    p.rails = [
        Rail(
            key="12V_MOD_0",
            aliases=("pwr_fail_12v_mod_0", "12v_mod_0", "12v mod 0",
                     "12v module 0"),
            voltage="12V DC",
            loads=[
                TopologyLoad(
                    "Primary Bianca HPM",
                    connection="12V_IN fed via the internal busbar; sideband "
                               "cable between Bianca HPM and PDB",
                    refs=[f"{NVL72_SRC} pp.17,19"]),
                TopologyLoad(
                    "Primary Bianca FPGA register bus",
                    connection="i2c bus 2 @0x11 (i2ctransfer dump)",
                    refs=[f"{NVL72_SRC} p.14"]),
            ],
            refs=[f"{NVL72_SRC} pp.14,17,19"],
        ),
        Rail(
            key="12V_MOD_1",
            aliases=("pwr_fail_12v_mod_1", "12v_mod_1", "12v mod 1",
                     "12v module 1"),
            voltage="12V DC",
            loads=[
                TopologyLoad(
                    "Secondary Bianca HPM",
                    connection="12V_IN fed via the internal busbar; sideband "
                               "cable between Bianca HPM and PDB",
                    refs=[f"{NVL72_SRC} pp.17,19"]),
                TopologyLoad(
                    "Secondary Bianca FPGA register bus",
                    connection="i2c bus 1 @0x11 (i2ctransfer dump)",
                    refs=[f"{NVL72_SRC} p.14"]),
            ],
            refs=[f"{NVL72_SRC} pp.14,17,19"],
        ),
        Rail(
            key="12V_INTERNAL_BUSBAR",
            aliases=("internal busbar", "busbar", "inner busbar", "12v busbar"),
            voltage="12V DC (distribution)",
            loads=[
                TopologyLoad(
                    "Both Bianca HPMs (12V_IN_1 / 12V_IN_2)",
                    connection="measure impedance of both 12V_IN rails to GND on "
                               "both Biancas; remove the inner busbar to isolate "
                               "PDB vs Bianca",
                    refs=[f"{NVL72_SRC} p.19"]),
            ],
            refs=[f"{NVL72_SRC} pp.18-19"],
        ),
        Rail(
            key="12V_ALL",
            aliases=("12v",),
            voltage="12V DC",
            loads=[
                TopologyLoad(
                    "All 12V loads",
                    connection="PDB 12V rails reach both Bianca HPMs through the "
                               "internal busbar",
                    refs=[f"{NVL72_SRC} pp.18-19"]),
            ],
            refs=[f"{NVL72_SRC} pp.18-19"],
            aggregate=True,
        ),
    ]
    p._post_init()
    return p


_TOPOLOGIES: list[PowerTopology] = [_samoa(), _nvl72()]


def _input_tokens(tokens: str | dict) -> set[str]:
    """Fault tokens for matching: from the rail-token string and, when the full
    fault signature is given, the decoded field names (their noise is stripped
    by the isolation pass, but the full names carry rail identity)."""
    if isinstance(tokens, dict):
        parts = [str(tokens.get("rail_tokens") or "")]
        parts += [str(r) for r in (tokens.get("reasons") or [])]
        raw = " ".join(parts)
    else:
        raw = str(tokens or "")
    return _tokens(raw)


def _platform_topology(platform: str | None) -> list[PowerTopology]:
    if not platform:
        return list(_TOPOLOGIES)
    family = family_for(platform)
    if family:
        for topo in _TOPOLOGIES:
            if topo.platform == family:
                return [topo]
        return list(_TOPOLOGIES)
    # Fall back to legacy alias spellings not yet in the shared taxonomy.
    key = str(platform).strip().lower()
    for topo in _TOPOLOGIES:
        if key in topo.platform_aliases:
            return [topo]
    return list(_TOPOLOGIES)


def loads_for_rail(tokens: str | dict, platform: str | None = None) -> list[dict]:
    """Rail -> documented loads for the fault tokens.

    ``tokens`` is either the isolation signature dict (``rail_tokens`` +
    ``reasons``) or a plain rail-token string. ``platform`` (a canonical model
    key or alias) restricts to one platform's topology; an unrecognized
    platform falls back to all topologies, each result carrying its own
    ``platform`` label so the caller never misattributes a rail.

    A rail matches when its significant token set is CONTAINED in the input
    tokens. Matches are ranked most-specific first; the generic aggregate rail
    is only returned when nothing more specific matched.
    """
    inp = _input_tokens(tokens)
    if not inp:
        return []
    matched: list[tuple[int, bool, dict]] = []
    for topo in _platform_topology(platform):
        for rail in topo.rails:
            best = 0
            for sig_toks in rail._match_token_sets():
                if sig_toks and sig_toks <= inp:
                    best = max(best, len(sig_toks))
            if best:
                matched.append((best, rail.aggregate, rail.as_dict()))
    if not matched:
        return []
    specific = [m for m in matched if not m[1]]
    ranked = sorted(specific if specific else matched, key=lambda m: m[0],
                    reverse=True)
    return [entry for _score, _agg, entry in ranked]
