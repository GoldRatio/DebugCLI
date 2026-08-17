"""Shared platform taxonomy for the fleet's documented hardware.

One source of truth for which model keys / product names belong to which
documented platform, used consistently by:

* model detection (``inspect.model.MODEL_ALIASES`` canonical keys),
* RAG retrieval (``docs.retrieval.hybrid_search`` platform filter),
* power topology (``docs.parts.topology`` platform restriction).

``PLATFORM_FAMILIES`` maps a canonical platform key to every alias (product
name, board name, rack name) that should resolve to it. ``family_for`` returns
the canonical platform a key belongs to (None when unknown), and
``family_members`` expands a canonical platform to all of its keys so filters
match any alias of the same platform.
"""

from __future__ import annotations

PLATFORM_FAMILIES: dict[str, tuple[str, ...]] = {
    # C4A14/C4A15 Grace-Blackwell compute tray (Microsoft "Samoa" node spec).
    # The SWB CPLD register spec and the node PDB/power spec are this platform.
    "samoa": (
        "samoa", "c4a14", "c4a15", "c4a14_server", "c4a14 server",
        "m1337855", "grace_hopper", "gb200", "gb200 superchip",
    ),
    # NVL72 rack (Quanta rack-level guide; its compute trays are the Samoa node).
    "nvl72": (
        "nvl72", "nvl", "gb_nvl72", "nvl72_rack", "nvl72 rack",
        "nvl rack", "gb200_nvl72", "gb200 nvl72",
    ),
}

_LOOKUP: dict[str, str] = {
    alias.lower(): family
    for family, aliases in PLATFORM_FAMILIES.items()
    for alias in (family, *aliases)
}


def family_for(key: str | None) -> str | None:
    """Canonical platform family for ``key`` (a model key / alias), else None."""
    if not key:
        return None
    return _LOOKUP.get(str(key).strip().lower())


def family_members(family: str) -> set[str]:
    """Every key/alias that belongs to a canonical platform family."""
    family = family.lower()
    members = {family}
    if family in PLATFORM_FAMILIES:
        members.update(PLATFORM_FAMILIES[family])
    return members
