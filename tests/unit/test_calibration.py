"""Prompt 06: outcome-calibrated confidence (per-LLM-ident fix-rate bins)."""

from pathlib import Path

import pytest

from harness.diagnosis.calibration import (
    Calibration,
    CalibrationStore,
    agreement_for,
    bin_index,
    build_calibration,
)
from harness.diagnosis.case_store import CaseStore
from harness.diagnosis.schema import CaseOutcome, Diagnosis, ServerState
from harness.diagnosis.scorer import score_diagnosis


def _case(run_id, outcome, subsystem=None, confidence=0.5, ident="openai/gpt-4o",
          self_reported=None):
    return CaseOutcome(
        run_id=run_id, target_id="t1", symptom=f"symptom {run_id}",
        outcome=outcome, subsystem_primary=subsystem, llm_ident=ident,
        confidence=confidence, self_reported_confidence=self_reported,
    )


def test_bin_index_edges():
    assert bin_index(0.0) == 0
    assert bin_index(0.19) == 0
    assert bin_index(0.2) == 0  # value <= bin_max -> first (0.0-0.2] bin
    assert bin_index(0.21) == 1
    assert bin_index(0.99) == 4
    assert bin_index(1.0) == 4
    assert bin_index(-1.0) == 0
    assert bin_index(5.0) == 4


def test_build_calibration_returns_none_below_min_samples():
    cases = [_case("a", "fixed"), _case("b", "fixed")]
    assert build_calibration(cases, "openai/gpt-4o", min_per_bin=3) is None


def test_build_calibration_fix_rates_with_partial_half():
    # 3 fixed + 1 partial in the 0.4-0.6 confidence bin -> (3*1 + 0.5)/4 = 0.875
    cases = [
        _case("a", "fixed", confidence=0.45),
        _case("b", "fixed", confidence=0.55),
        _case("c", "fixed", confidence=0.5),
        _case("d", "partial", confidence=0.5),
    ]
    cal = build_calibration(cases, "openai/gpt-4o")
    assert cal is not None
    assert cal.aggregate[2] == pytest.approx((0.6, 0.875, 4))


def test_build_calibration_only_counts_matching_ident():
    cases = [
        _case("a", "fixed", ident="openai/gpt-4o"),
        _case("b", "fixed", ident="gemini-2.5-flash"),
        _case("c", "fixed", ident="openai/gpt-4o"),
        _case("d", "fixed", ident="openai/gpt-4o"),
    ]
    cal = build_calibration(cases, "openai/gpt-4o")
    assert cal is not None
    assert cal.total_samples() == 3
    # the other ident calibrates itself from its OWN cases
    assert build_calibration(cases, "gemini-2.5-flash") is None


def test_sparse_subsystem_bin_falls_back_to_aggregate():
    # aggregate: memory-heavy population; subsystem "storage" has ONE sample
    # in a bin -> its rate must track the aggregate bin, not its lone case.
    cases = [
        _case("a", "fixed", subsystem="memory", confidence=0.25),
        _case("b", "fixed", subsystem="memory", confidence=0.25),
        _case("c", "fixed", subsystem="memory", confidence=0.25),
        _case("d", "fixed", subsystem="memory", confidence=0.25),
        _case("e", "not_fixed", subsystem="storage", confidence=0.25),
    ]
    cal = build_calibration(cases, "openai/gpt-4o")
    assert cal is not None
    # bin index for 0.25 -> (0.2, 0.4] -> bin 1; 4 fixed + 1 not_fixed = 0.8
    agg_rate = cal.aggregate[1][1]
    assert agg_rate == pytest.approx(0.8)
    storage_bins = dict(cal.subsystem_bins)["storage"]
    # sparse (n=1 < 3) -> inherits aggregate rate 0.8 (not its own 0.0)
    assert storage_bins[1][1] == pytest.approx(0.8)
    assert storage_bins[1][2] == 1


def test_agreement_for_none_is_half():
    assert agreement_for(None, 0.9) == 0.5


def test_agreement_for_empty_calibration_is_half():
    empty = Calibration(llm_ident="x", aggregate=[], subsystem_bins={})
    assert agreement_for(empty, 0.9) == 0.5


def test_agreement_for_is_shrunk_fix_rate_posterior():
    # predicted 0.9, observed fix rate 0.4 (n=5): (5*0.4 + 5*0.5)/10 = 0.45
    cal = build_calibration(
        [_case(f"c{i}", "not_fixed", confidence=0.9,
               subsystem="memory") for i in range(3)]
        + [_case(f"f{i}", "fixed", confidence=0.9,
                 subsystem="memory") for i in range(2)],
        "openai/gpt-4o")
    assert cal is not None
    # observed rate for the 0.8-1.0 bin: 2/5 = 0.4 -> shrunk toward 0.5
    agreement = agreement_for(cal, 0.9, "memory")
    assert agreement == pytest.approx(0.45, abs=0.01)


def test_agreement_for_high_fix_rate_bin():
    # Bin (0.8-1.0] with observed rate 0.9 (n=5): (5*0.9 + 5*0.5)/10 = 0.7
    cases = [*[_case(f"f{i}", "fixed", confidence=0.9) for i in range(4)],
             _case("p", "partial", confidence=0.9)]
    cal = build_calibration(cases, "openai/gpt-4o")
    assert cal is not None
    agreement = agreement_for(cal, 0.9)
    assert agreement == pytest.approx(0.7, abs=0.01)


def test_agreement_for_clamped():
    # 5 not_fixed at 0.9 -> observed rate 0.0 (n=5): (0 + 5*0.5)/10 = 0.25
    cal = build_calibration([_case(f"c{i}", "not_fixed", confidence=0.9)
                             for i in range(5)], "openai/gpt-4o")
    assert cal is not None
    agreement = agreement_for(cal, 0.9, None)
    assert 0.0 <= agreement <= 1.0
    assert agreement == pytest.approx(0.25, abs=0.01)


def test_agreement_for_unknown_subsystem_falls_back_to_aggregate():
    cal = build_calibration([_case(f"c{i}", "fixed", confidence=0.9)
                             for i in range(5)], "openai/gpt-4o")
    assert cal is not None
    # subsystem absent from the histogram -> aggregate top bin (fix rate 1.0,
    # n=5): (5*1.0 + 5*0.5)/10 = 0.75
    assert agreement_for(cal, 1.0, "nonexistent") == pytest.approx(0.75, abs=0.01)


def test_build_calibration_bins_on_self_reported_confidence():
    # self-report 0.9 (top bin) differs from scored confidence 0.4 (bin 1); the
    # histogram must bin on the SELF-REPORT, the same key agreement_for uses.
    cases = [_case(f"c{i}", "not_fixed", confidence=0.4, self_reported=0.9)
             for i in range(5)]
    cal = build_calibration(cases, "openai/gpt-4o")
    assert cal is not None
    assert cal.aggregate[4][1] == pytest.approx(0.0)  # top bin: all not_fixed
    assert cal.aggregate[4][2] == 5
    # scored-confidence bin 1 (0.2-0.4] has no samples -> neutral rate
    assert cal.aggregate[1][1] == pytest.approx(0.5)
    assert cal.aggregate[1][2] == 0


def test_build_calibration_legacy_records_fall_back_to_scored_confidence():
    # No self-report recorded (legacy record): bin on scored confidence 0.9.
    cases = [_case(f"c{i}", "not_fixed", confidence=0.9) for i in range(5)]
    cal = build_calibration(cases, "openai/gpt-4o")
    assert cal is not None
    assert cal.aggregate[4][1] == pytest.approx(0.0)
    assert cal.aggregate[4][2] == 5


def test_calibration_store_round_trip(tmp_path):
    store = CalibrationStore(tmp_path / "cal")
    cases = [_case(f"c{i}", "fixed", confidence=0.5) for i in range(5)]
    cal = build_calibration(cases, "openai/gpt-4o")
    assert cal is not None
    store.save(cal)
    loaded = store.load("openai/gpt-4o")
    assert loaded is not None
    assert loaded.llm_ident == "openai/gpt-4o"
    assert loaded.aggregate == cal.aggregate
    assert loaded.total_samples() == 5
    assert store.load("other-model") is None
    assert [c.llm_ident for c in store.all()] == ["openai/gpt-4o"]


def _calibrated_score_diagnosis(calibration_root: Path | None):
    diag = Diagnosis(
        diagnosis="memory ECC",
        state=ServerState.FAULT,
        confidence=0.5,
        subsystems_considered=[],
    )
    return score_diagnosis(
        diag,
        retrieved_snippets=None,
        evidence_fit=0.5,
        calibration_root=calibration_root,
        llm_ident="openai/gpt-4o",
    )


def test_scorer_uses_calibration_when_root_present(tmp_path):
    # skewed calibration: top bin fix rate 0.0 (n=5) -> posterior (0 + 2.5)/10
    cases = [_case(f"c{i}", "not_fixed", confidence=0.9) for i in range(5)]
    store = CalibrationStore(tmp_path / "cal")
    store.save(build_calibration(cases, "openai/gpt-4o"))

    diag = Diagnosis(diagnosis="x", confidence=0.9)
    scored = score_diagnosis(
        diag, retrieved_snippets=None, evidence_fit=0.5,
        calibration_root=tmp_path / "cal", llm_ident="openai/gpt-4o")
    assert scored.confidence_breakdown.calibration_llm == "openai/gpt-4o"
    assert scored.confidence_breakdown.model_agreement == pytest.approx(0.25, abs=0.01)
    # the self-report used as the bin key is preserved for the case store
    assert scored.confidence_breakdown.self_reported_confidence == 0.9

    # empty/missing root keeps the 0.5 default and no calibration_llm
    plain = _calibrated_score_diagnosis(tmp_path / "does-not-exist")
    assert plain.confidence_breakdown.model_agreement == 0.5
    assert plain.confidence_breakdown.calibration_llm is None
    assert plain.confidence_breakdown.self_reported_confidence == 0.5


def test_scorer_existing_model_agreement_wins_over_calibration(tmp_path):
    # explicit model_agreement still works (backward compat with prompt 03 tests)
    diag = Diagnosis(diagnosis="x", confidence=0.5)
    scored = score_diagnosis(
        diag, retrieved_snippets=None, evidence_fit=0.5,
        model_agreement=0.25, calibration_root=tmp_path / "cal",
        llm_ident="openai/gpt-4o")
    assert scored.confidence_breakdown.model_agreement == 0.25
    assert scored.confidence_breakdown.calibration_llm is None


def test_calibrate_cli_writes_per_ident_files(tmp_path, capsys, monkeypatch):
    from harness.operator.cli import build_parser, run_calibrate

    store = CaseStore(tmp_path / "cases")
    store.save(_case("c1", "fixed", ident="openai/gpt-4o"))
    store.save(_case("c2", "fixed", ident="openai/gpt-4o"))
    store.save(_case("c3", "fixed", ident="openai/gpt-4o"))
    store.save(_case("c4", "fixed", ident="stub"))
    store.save(_case("c5", "fixed", ident="stub"))
    store.save(_case("c6", "fixed", ident="stub"))

    args = build_parser().parse_args([
        "calibrate", "--cases", str(tmp_path / "cases"),
        "--out", str(tmp_path / "cal")])
    assert run_calibrate(args) == 0

    out = capsys.readouterr().out
    assert "openai/gpt-4o" in out
    assert "stub" in out
    cal_dir = tmp_path / "cal"
    assert list(cal_dir.glob("*.json"))  # per-ident files written
    loaded = CalibrationStore(cal_dir).load("stub")
    assert loaded is not None and loaded.total_samples() == 3


def test_calibrate_cli_insufficient_cases(tmp_path, capsys):
    from harness.operator.cli import build_parser, run_calibrate

    store = CaseStore(tmp_path / "cases")
    store.save(_case("c1", "fixed"))
    store.save(_case("c2", "fixed"))
    args = build_parser().parse_args([
        "calibrate", "--cases", str(tmp_path / "cases"),
        "--out", str(tmp_path / "cal")])
    assert run_calibrate(args) == 1  # below min_per_bin -> no calibration