"""Learning transfer: bundle export/import of cases + derived + run artifacts.

Covers the device-to-device contract: roundtrip fidelity, WORM-safe skip /
--revise semantics, schema + secret-ref enforcement on import, manifest
integrity (tamper evidence), run-id filtering, zip-slip safety and dry-run.
"""

import hashlib
import json
import zipfile

import pytest

from harness.diagnosis.calibration import Calibration, CalibrationStore
from harness.diagnosis.case_store import CaseStore
from harness.diagnosis.schema import CaseOutcome
from harness.operator.cli import build_parser
from harness.operator.learning_transfer import (
    BundleError,
    export_bundle,
    import_bundle,
)


def _case(run_id: str, **kw) -> CaseOutcome:
    defaults = {
        "run_id": run_id, "target_id": "target-1",
        "symptom": "ECC errors on DIMM_A2",
        "model_key": "poweredge_r650", "outcome": "fixed",
        "actions_recommended": ["1. Reseat DIMM A2"],
        "actions_taken": ["reseated DIMM A2"],
        "llm_ident": "stub",
        "evidence_summary": ["MCE: memory error on DIMM_A2"],
    }
    defaults.update(kw)
    return CaseOutcome(**defaults)


def _device(tmp_path) -> "object":
    """A populated 'source device' run root (cases + runs + derived)."""
    out = tmp_path / "device-a"
    for run_id, state in (("run-1", "degraded"), ("run-2", "healthy")):
        run_dir = out / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "diagnosis.json").write_text(
            json.dumps({"state": state, "confidence": 0.7}), encoding="utf-8")
        (run_dir / "audit.jsonl").write_text("", encoding="utf-8")
    (out / "sessions").mkdir()
    (out / "sessions" / "chat.json").write_text("{}", encoding="utf-8")
    CaseStore(out / "cases").record(_case("run-1"))
    CalibrationStore(out / "calibration").save(
        Calibration(llm_ident="stub", aggregate=[(0.2, 0.5, 3)],
                    created_at="2026-01-01T00:00:00+00:00"))
    (out / "priors.json").write_text(json.dumps(
        {"min_verified": 10, "verified_cases": 1, "keyword_multipliers": {}}),
        encoding="utf-8")
    return out


def _make_bundle(path, entries: dict[str, bytes], bundle_version: int = 1) -> None:
    """Hand-built bundle (for corrupt/tampered/foreign-manifest tests)."""
    manifest = {
        "bundle_version": bundle_version,
        "created_at": "2026-01-01T00:00:00+00:00",
        "counts": {},
        "files": {n: hashlib.sha256(d).hexdigest()
                  for n, d in sorted(entries.items())},
    }
    with zipfile.ZipFile(path, "w") as zf:
        for n, d in sorted(entries.items()):
            zf.writestr(n, d)
        zf.writestr("manifest.json",
                    json.dumps(manifest, indent=2, sort_keys=True))


def _rezip_with_tampered_entry(path, target: str) -> None:
    """Rewrite the zip with one entry's bytes changed, manifest untouched."""
    with zipfile.ZipFile(path) as zf:
        entries = {n: zf.read(n) for n in zf.namelist()}
    entries[target] = entries[target] + b"tampered"
    with zipfile.ZipFile(path, "w") as zf:
        for n, d in entries.items():
            zf.writestr(n, d)


def test_export_import_roundtrip(tmp_path):
    out = _device(tmp_path)
    bundle = tmp_path / "share.zip"
    result = export_bundle(bundle, out)
    assert result.bundle == bundle
    assert result.cases == ["run-1"]
    assert result.runs == ["run-1", "run-2"]
    assert result.calibration == ["stub"]
    assert result.priors

    target = tmp_path / "device-b"
    imported = import_bundle(bundle, target)
    assert imported.ok
    assert imported.cases_imported == ["run-1"]
    assert imported.runs_imported == ["run-1", "run-2"]

    source = CaseStore(out / "cases").get("run-1")
    copy = CaseStore(target / "cases").get("run-1")
    assert copy is not None
    assert copy.model_dump() == source.model_dump()
    assert (target / "run-1" / "diagnosis.json").read_text(encoding="utf-8") \
        == (out / "run-1" / "diagnosis.json").read_text(encoding="utf-8")
    assert (target / "run-2" / "audit.jsonl").exists()
    assert (target / "calibration" / "stub.json").exists()
    assert json.loads((target / "priors.json").read_text(encoding="utf-8"))[
        "verified_cases"] == 1
    assert not (target / "sessions").exists()
    # the index was rebuilt on the target too
    assert "run-1" in json.loads(
        (target / "cases" / "index.json").read_text(encoding="utf-8"))


def test_import_skips_existing_then_revise(tmp_path):
    out = _device(tmp_path)
    bundle = tmp_path / "share.zip"
    export_bundle(bundle, out)

    target = tmp_path / "device-b"
    target.mkdir()
    # a pre-existing (conflicting) device: same run ids, different verdicts
    (target / "run-1").mkdir()
    (target / "run-1" / "diagnosis.json").write_text('{"state": "healthy"}',
                                                     encoding="utf-8")
    CaseStore(target / "cases").record(_case("run-1", outcome="not_fixed"))

    skipped = import_bundle(bundle, target)
    assert skipped.ok
    assert skipped.cases_skipped == ["run-1"]
    assert skipped.runs_skipped == ["run-1"]       # known run: untouched
    assert skipped.runs_imported == ["run-2"]      # new run: arrives
    assert skipped.calibration_imported == ["stub"]
    assert skipped.priors == "imported"
    assert CaseStore(target / "cases").get("run-1").outcome == "not_fixed"
    assert json.loads((target / "run-1" / "diagnosis.json").read_text(
        encoding="utf-8"))["state"] == "healthy"

    revised = import_bundle(bundle, target, revise=True)
    assert revised.cases_imported == ["run-1"]  # replaced by the bundle's
    assert CaseStore(target / "cases").get("run-1").outcome == "fixed"
    assert json.loads((target / "run-1" / "diagnosis.json").read_text(
        encoding="utf-8"))["state"] == "degraded"
    assert revised.runs_imported == ["run-1", "run-2"]  # both replaced
    assert revised.runs_skipped == []
    assert revised.calibration_imported == ["stub"]  # replaced
    assert revised.priors == "imported"  # replaced


def test_import_rejects_secret_refs(tmp_path):
    target = tmp_path / "device"
    bundle = tmp_path / "bad.zip"
    payload = _case("run-9", actions_taken=["paste -----BEGIN PRIVATE KEY-----"],
                    evidence_hash="x").model_dump_json().encode("utf-8")
    _make_bundle(bundle, {"cases/run-9.json": payload})
    result = import_bundle(bundle, target)
    assert not result.ok
    assert "run-9" in result.cases_failed
    assert CaseStore(target / "cases").get("run-9") is None


def test_import_partial_failure(tmp_path):
    target = tmp_path / "device"
    bundle = tmp_path / "mixed.zip"
    _make_bundle(bundle, {
        "cases/bad.json": b"{not json",
        "cases/run-1.json": _case("run-1").model_dump_json().encode("utf-8"),
    })
    result = import_bundle(bundle, target)
    assert not result.ok
    assert set(result.cases_failed) == {"bad"}
    assert result.cases_imported == ["run-1"]
    assert CaseStore(target / "cases").get("run-1") is not None


def test_tampered_bundle_rejected(tmp_path):
    out = _device(tmp_path)
    bundle = tmp_path / "share.zip"
    export_bundle(bundle, out)
    _rezip_with_tampered_entry(bundle, "cases/run-1.json")
    with pytest.raises(BundleError, match="integrity"):
        import_bundle(bundle, tmp_path / "device-b")


def test_missing_manifest_and_bad_version_rejected(tmp_path):
    plain = tmp_path / "plain.zip"
    with zipfile.ZipFile(plain, "w") as zf:
        zf.writestr("cases/run-1.json", _case("run-1").model_dump_json())
    with pytest.raises(BundleError, match="manifest"):
        import_bundle(plain, tmp_path / "d1")

    foreign = tmp_path / "foreign.zip"
    _make_bundle(foreign, {"cases/run-1.json": b"{}"}, bundle_version=99)
    with pytest.raises(BundleError, match="bundle_version"):
        import_bundle(foreign, tmp_path / "d2")


def test_missing_manifest_file_rejected(tmp_path):
    bundle = tmp_path / "incomplete.zip"
    _make_bundle(bundle, {"cases/run-1.json": b"{}"})
    with zipfile.ZipFile(bundle) as zf:
        entries = {n: zf.read(n) for n in zf.namelist() if n != "cases/run-1.json"}
    with zipfile.ZipFile(bundle, "w") as zf:
        for n, d in entries.items():
            zf.writestr(n, d)
    with pytest.raises(BundleError, match="missing"):
        import_bundle(bundle, tmp_path / "d3")


def test_export_run_filter_and_unknown_ids(tmp_path):
    out = _device(tmp_path)
    bundle = tmp_path / "one.zip"
    result = export_bundle(bundle, out, run_ids=["run-1", "nope"])
    assert result.cases == ["run-1"]
    assert result.runs == ["run-1"]
    assert result.unknown_runs == ["nope"]
    with zipfile.ZipFile(bundle) as zf:
        names = zf.namelist()
    assert any(n.startswith("runs/run-1/") for n in names)
    assert not any(n.startswith("runs/run-2/") for n in names)
    assert not any(n.startswith("runs/nope/") for n in names)


def test_import_dry_run_writes_nothing(tmp_path):
    out = _device(tmp_path)
    bundle = tmp_path / "share.zip"
    export_bundle(bundle, out)
    target = tmp_path / "device-b"
    result = import_bundle(bundle, target, dry_run=True)
    assert result.ok
    assert result.cases_imported == ["run-1"]
    assert result.runs_imported == ["run-1", "run-2"]
    assert result.priors == "imported"
    assert not (target / "cases" / "run-1.json").exists()
    assert not (target / "run-1" / "diagnosis.json").exists()
    assert not (target / "priors.json").exists()


def test_zip_slip_entries_are_dropped(tmp_path):
    target = tmp_path / "device"
    bundle = tmp_path / "evil.zip"
    _make_bundle(bundle, {"runs/../../evil.txt": b"nope"})
    result = import_bundle(bundle, target)
    assert result.runs_imported == []
    assert not (tmp_path / "evil.txt").exists()
    assert not (target / "evil.txt").exists()


def test_no_cases_dir_exports_runs_only(tmp_path):
    out = tmp_path / "device"
    (out / "run-1").mkdir(parents=True)
    (out / "run-1" / "trace.json").write_text("{}", encoding="utf-8")
    bundle = tmp_path / "runs.zip"
    result = export_bundle(bundle, out)
    assert result.cases == []
    assert result.runs == ["run-1"]
    assert not result.priors
    imported = import_bundle(bundle, tmp_path / "device-b")
    assert imported.ok
    assert (tmp_path / "device-b" / "run-1" / "trace.json").exists()


def test_parser_learning_subcommands():
    args = build_parser().parse_args(
        ["learning", "export", "--out", "b.zip", "--runs", "r1,r2", "--list"])
    assert args.out == "b.zip"
    assert args.runs == "r1,r2"
    assert args.list
    assert callable(args.func)
    args = build_parser().parse_args(
        ["learning", "import", "b.zip", "--revise", "--dry-run"])
    assert args.bundle == "b.zip"
    assert args.revise and args.dry_run
    assert callable(args.func)
