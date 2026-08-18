"""Shared platform taxonomy: family resolution used by detection, RAG and topology."""

from harness.platforms import family_for, family_members


def test_family_for_resolves_known_aliases():
    assert family_for("samoa") == "samoa"
    assert family_for("c4a15") == "samoa"
    assert family_for("C4A14 Server") == "samoa"
    assert family_for("m1337855") == "samoa"
    assert family_for("nvl72") == "nvl72"
    assert family_for("GB_NVL72") == "nvl72"
    assert family_for("nvl72 rack") == "nvl72"


def test_family_for_unknown_is_none():
    assert family_for("poweredge_r650") is None
    assert family_for("model_x") is None
    assert family_for(None) is None
    assert family_for("") is None


def test_family_members_covers_family_and_aliases():
    members = family_members("samoa")
    assert "samoa" in members and "c4a14" in members and "c4a15" in members
    nvl = family_members("nvl72")
    assert "nvl72" in nvl and "gb_nvl72" in nvl
    # families do not bleed into each other
    assert not (members & nvl)


def test_all_families_resolve_to_self():
    from harness.platforms import PLATFORM_FAMILIES
    for family in PLATFORM_FAMILIES:
        assert family_for(family) == family
        assert family in family_members(family)