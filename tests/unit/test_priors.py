"""Prompt 05: outcome-fed priors + ticket outcome flow.

Covers ``plan.subsystem`` (classify with/without priors, clamping),
``plan.priors_build`` (gate, Laplace smoothing, dampening), the ``priors
update`` CLI path, and the ``record_outcome`` ticket flow (NoOp delegates to
the report path; a backend without the method must not crash the flow).
"""

import json

import pytest

from src.harness.diagnosis.case_store import CaseStore
from src.harness.diagnosis.schema import CaseOutcome
from src.harness.operator import cli
from src.harness.operator.tickets import NoOpTicketing, Ticketing, record_outcome_safe
from src.harness.plan.priors_build import build_priors, dump_priors, load_priors
from src.harness.plan.subsystem import MULTIPLIER_MAX, MULTIPLIER_MIN, PriorModel, classify


def _case(run_id: str, symptom: str, outcome: str,
          subsystem: str | None = None) -> CaseOutcome:
    return CaseOutcome(run_id=run_id, target_id="t1", symptom=symptom,
                       subsystem_primary=subsystem, outcome=outcome)


def test_classify_identical_without_priors():
    static = classify("mce error on cpu core 2")
    assert classify("mce error on cpu core 2", priors=None) == static
    assert classify("mce error on cpu core 2", priors=PriorModel()) == static
    assert classify("mce error on cpu core 2",
                    priors=PriorModel(keyword_multipliers={})) == static


def test_classify_applies_and_clamps_multipliers():
    priors = PriorModel(keyword_multipliers={
        "mce": {"memory": 2.0, "cpu": 0.5},
    })
    ranked = classify("mce", priors)
    assert ranked[0].subsystem == "memory"  # static tie, priors break it
    assert ranked[0].score == pytest.approx(2.0)

    out_of_band = PriorModel(keyword_multipliers={
        "mce": {"memory": 9.0, "cpu": 0.01},
    })
    assert out_of_band.multiplier("mce", "memory") == MULTIPLIER_MAX
    assert out_of_band.multiplier("mce", "cpu") == MULTIPLIER_MIN
    # unknown keyword/subsystem -> static 1.0
    assert out_of_band.multiplier("crypto", "memory") == 1.0


def test_build_priors_none_under_min_verified():
    cases = [_case(f"r{i}", "mce ecc error", "fixed", "memory")
             for i in range(3)]
    assert build_priors(cases, min_verified=10) is None


def test_build_priors_boosts_memory_above_static_ranking():
    cases = [_case(f"r{i}", "mce ecc error", "fixed", "memory")
             for i in range(12)]
    priors = build_priors(cases, min_verified=10)
    assert priors is not None
    assert priors.covers("mce")
    ranked = classify("mce", priors)
    # static table ties memory/cpu on "mce"; fleet history breaks the tie
    assert ranked[0].subsystem == "memory"
    assert classified_score(priors, "mce", "memory") > 1.0


def test_build_priors_dampens_storage_below_one():
    not_fixed = [_case(f"r{i}", "i/o error", "not_fixed", "storage")
                 for i in range(8)]
    fixed_elsewhere = [_case(f"s{i}", "i/o error", "fixed", "memory")
                       for i in range(7)]
    priors = build_priors(not_fixed + fixed_elsewhere, min_verified=10)
    assert priors is not None
    assert priors.multiplier("i/o error", "storage") < 1.0


def test_dump_load_round_trip():
    priors = build_priors(
        [_case(f"r{i}", "ecc sel error", "fixed", "memory") for i in range(11)],
        min_verified=10)
    assert priors is not None
    restored = load_priors(dump_priors(priors))
    assert restored.keyword_multipliers == priors.keyword_multipliers
    assert classify("ecc sel error", restored) == classify("ecc sel error", priors)


def _seed_store(root, cases):
    store = CaseStore(root)
    for c in cases:
        store.save(c)
    return store


def test_priors_update_cli_round_trip(tmp_path):
    store_dir = tmp_path / "cases"
    _seed_store(store_dir, [_case(f"r{i}", "mce error", "fixed", "memory")
                             for i in range(11)])
    out_path = tmp_path / "priors.json"

    class Args:
        cases = str(store_dir)
        out = str(out_path)
        min_verified = 10

    assert cli.run_priors_update(Args) == 0
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["verified_cases"] == 11
    priors = load_priors(json.dumps(data))
    assert classify("mce")[0].subsystem == "memory"
    assert priors.covers("mce")


def test_priors_update_inactive_below_gate(tmp_path):
    store_dir = tmp_path / "cases"
    _seed_store(store_dir, [_case(f"r{i}", "mce error", "fixed", "memory")
                             for i in range(5)])
    out_path = tmp_path / "priors.json"

    class Args:
        cases = str(store_dir)
        out = str(out_path)
        min_verified = 10

    assert cli.run_priors_update(Args) == 1  # "5/10 verified cases, priors inactive"
    assert not out_path.exists()


def test_noop_record_outcome_closes_loop_via_report(tmp_path, capsys):
    ticket = NoOpTicketing().record_outcome("DIAG-42", "fixed", ["replace dimm"])
    out = capsys.readouterr()
    assert "harness report --run DIAG-42 --outcome fixed" in out.out
    assert "DIAG-42" in ticket

    # emulating the one-liner: run the real report path, record lands in store
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "DIAG-42"
    run_dir.mkdir(parents=True)
    (run_dir / "pending_case.json").write_text(
        CaseOutcome(run_id="DIAG-42", target_id="t1",
                    symptom="mce").model_dump_json(), encoding="utf-8")
    store_dir = tmp_path / "cases"
    result = cli.run_report(_report_args(store_dir, runs_root))
    assert result == 0
    stored = CaseStore(store_dir).get("DIAG-42")
    assert stored is not None
    assert stored.outcome == "fixed"
    assert stored.actions_taken == ["replace dimm"]


class _NoOutcomeBackend(Ticketing):
    def submit(self, diagnosis) -> str:
        return "DIAG-1"

    def status(self, ticket_id: str) -> str:
        return "open"


def test_missing_record_outcome_does_not_crash_flow():
    backend = _NoOutcomeBackend()
    line = record_outcome_safe(backend, "DIAG-1", "fixed", ["reboot"])
    assert "harness report --run DIAG-1 --outcome fixed" in line
    with pytest.raises(NotImplementedError):
        backend.record_outcome("DIAG-1", "fixed", ["reboot"])


def _report_args(store_dir, runs_root):
    class Args:
        run = "DIAG-42"
        outcome = "fixed"
        taken: tuple[str, ...] = ("replace dimm",)
        cases = str(store_dir)
        out_dir = str(runs_root)
        status = False
        verdict = None

    return Args()


def classified_score(priors, keyword, subsystem) -> float:
    return priors.multiplier(keyword, subsystem)