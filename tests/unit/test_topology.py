"""Power-topology tests: the documented rail -> loads edges (from the vendor
architecture docs) match fault tokens deterministically and never guess a rail
beyond what the tokens support.
"""

import json

from harness.docs.parts.topology import loads_for_rail


def test_matches_p12v_cb2_from_sensor_and_field_tokens():
    """Real incident shape: 'pdb p12v cb2 volt' resolves to the P12V_CB2 rail."""
    edges = loads_for_rail("pdb p12v cb2 volt")
    assert len(edges) == 1
    assert edges[0]["rail"] == "P12V_CB2"
    assert edges[0]["platform"] == "samoa"
    assert edges[0]["loads"][0]["name"] == "OSFP Board"
    assert "J9" in edges[0]["loads"][0]["connection"]
    assert all(edge["refs"] for edge in edges)  # every edge is page-provenanced


def test_matches_full_signature_reasons():
    """The full fault signature (rail_tokens + decoded field names) carries the
    rail identity even when the stripped rail tokens are only '12v'."""
    edges = loads_for_rail({
        "rail_tokens": "12v",
        "reasons": ["CPLD_94_PWRUP_FAULTS.pwr_fail_12v_mod_0"],
    })
    assert len(edges) == 1
    assert edges[0]["rail"] == "12V_MOD_0"
    assert edges[0]["platform"] == "nvl72"
    assert any("Primary Bianca HPM" in load["name"] for load in edges[0]["loads"])


def test_aggregate_only_when_nothing_specific():
    """A bare 12v token yields the documented 12V aggregate loads, never a
    guessed sub-rail, and only when no specific rail matched."""
    edges = loads_for_rail("pdb 12v rail")
    assert edges and all(e["rail"] == "12V_ALL" for e in edges)
    # with a distinguishing token the specific rail wins and aggregate is dropped
    edges = loads_for_rail("pdb 12v cb2 volt")
    assert all(e["rail"] != "12V_ALL" for e in edges)


def test_platform_restriction():
    """An explicit platform restricts to its topology; other platforms return [].
    An unrecognized platform falls back to all topologies (never mislabels)."""
    assert [e["rail"] for e in loads_for_rail("pdb p12v cb2 volt", "samoa")] \
        == ["P12V_CB2"]
    assert loads_for_rail("pdb p12v cb2 volt", "nvl72") == []
    # unknown platform -> labeled results from every topology, never empty
    unknown = loads_for_rail("pdb 12v rail", "nvl72")
    assert unknown and all(e["platform"] == "nvl72" for e in unknown)
    assert loads_for_rail("pdb p12v cb2 volt", "c4a15")[0]["platform"] == "samoa"


def test_no_guess_on_unrelated_tokens():
    assert loads_for_rail("qsfp retimer flap") == []
    assert loads_for_rail("") == []
    assert loads_for_rail("dimm ecc") == []


def test_stby_and_bus_distinct():
    edges = loads_for_rail("p12v stby fault")
    assert [e["rail"] for e in edges] == ["P12V_STBY"]
    bus = loads_for_rail("12v bus rail fan")
    assert [e["rail"] for e in bus] == ["P12V_BUS"]


def test_edges_are_stable_json():
    """The emitted edges round-trip through JSON (the prompt renders dicts)."""
    edges = loads_for_rail({"rail_tokens": "12v mod 1",
                            "reasons": ["CPLD.pwr_fail_12v_mod_1"]})
    blob = json.dumps(edges)
    assert json.loads(blob) == edges
    assert blob and edges[0]["rail"] == "12V_MOD_1"