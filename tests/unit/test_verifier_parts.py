"""Verifier and parts-validation unit tests."""

from harness.diagnosis.parts_validate import PartsValidator
from harness.diagnosis.verifier import Verifier
from harness.inspect.base import RegisterDump


def _dump(raw: str, source: str = "ipmitool sensor") -> RegisterDump:
    return RegisterDump(subsystem="bmc", source=source, raw=raw, cmd_argv=["x"])


def test_verifier_detects_resolution():
    v = Verifier()
    before = [_dump("ecc_error 10\necc_error 5")]
    after = [_dump("ecc_error 2")]
    result = v.compare(before, after)
    assert result.verdict == "resolved"
    assert result.delta["ipmitool sensor"] == -1


def test_verifier_detects_no_change():
    v = Verifier()
    before = [_dump("ecc_error 3")]
    after = [_dump("ecc_error 3")]
    assert v.compare(before, after).verdict == "inconclusive"


def test_parts_validate_flags_discrepancy():
    graph = {"DIMM_A2": {"sn": "SN1", "pn": "PN1"}}
    system = {"DIMM_A2": {"sn": "SN1", "pn": "PN9"}}
    result = PartsValidator().validate(graph, system)
    assert not result.ok
    assert any("DIMM_A2.pn" in d for d in result.discrepancies)


def test_parts_validate_consistent():
    graph = {"DIMM_A2": {"sn": "SN1"}}
    system = {"DIMM_A2": {"sn": "SN1"}}
    result = PartsValidator().validate(graph, system)
    assert result.ok


FRU_OK = """Chassis Part Number  : CH-1
Chassis Serial       : CS-1
Board Part Number    : BPN-7
Board Serial         : BS-7
Product Part Number  : SVR-X
Product Serial       : SN-9001
"""


def test_fru_snapshot_parses_pn_serial_pairs():
    snap = PartsValidator().fru_snapshot(FRU_OK)
    assert snap["SVR-X"]["sn"] == "SN-9001"
    assert snap["BPN-7"]["sn"] == "BS-7"


def test_validate_fru_matches_parts_list():
    graph = {"slot0": {"pn": "SVR-X", "sn": "SN-9001"},
             "board": {"pn": "BPN-7", "sn": "BS-7"}}
    result = PartsValidator().validate_fru(graph, FRU_OK)
    assert result.ok
    assert any("SVR-X" in m for m in result.matches)


def test_validate_fru_flags_serial_mismatch():
    graph = {"slot0": {"pn": "SVR-X", "sn": "SN-OTHER"}}
    result = PartsValidator().validate_fru(graph, FRU_OK)
    assert not result.ok
    assert any("serial mismatch" in d for d in result.discrepancies)


def test_validate_fru_flags_installed_part_not_in_list():
    graph = {"slot0": {"pn": "SVR-Y"}}
    result = PartsValidator().validate_fru(graph, FRU_OK)
    assert not result.ok
    assert any("not present in parts list" in d for d in result.discrepancies)


def test_validate_fru_skips_when_no_fru_data():
    result = PartsValidator().validate_fru({"a": {"pn": "X"}}, "")
    assert result.ok
    assert "skipped" in result.matches[0]