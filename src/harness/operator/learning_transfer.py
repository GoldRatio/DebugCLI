"""Learning transfer: share the learning factor between devices.

Everything the learning loop produced on one device -- verified case outcomes
(the fixes), the derived priors + per-LLM calibration, and the full run
artifact dirs they came from -- packs into ONE zip bundle that another device
can import. The bundle is deliberately unredacted: it may contain hostnames,
serials and evidence text, so it is for your own devices, not for sharing
externally.

Layout inside the bundle::

    manifest.json                  # bundle_version, created_at, counts,
                                   # per-file sha256 (tamper evidence)
    cases/<run_id>.json            # CaseOutcome records (re-serialized,
                                   # schema-checked at export time)
    calibration/<llm_ident>.json   # per-model confidence calibration
    priors.json                    # symptom-keyword subsystem multipliers
    runs/<run_id>/...              # full run dirs (diagnosis, trace, audit,
                                   # dumps, prompts)

Import is conservative: existing run ids are SKIPPED (the case store is
append-only WORM) unless ``revise=True``, which routes an operator-intended
replacement through ``CaseStore.record(revise=True)`` so the change is audited
as ``case_revised``. Every case is validated against the pydantic schema and
re-checked for secret references by ``CaseStore.save`` on the way in. The
manifest's per-file sha256 is verified before anything is written, so a
tampered bundle is never partially applied; imported cases carry no local
audit linkage (the bundle carries the original run's ``audit.jsonl``).
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

BUNDLE_VERSION = 1
MANIFEST_NAME = "manifest.json"
CASES_PREFIX = "cases/"
CALIBRATION_PREFIX = "calibration/"
RUNS_PREFIX = "runs/"
PRIORS_NAME = "priors.json"

#: A directory must contain at least one of these to be a diagnosable run
#: (mirrors the operator CLI's run-dir detection; defined here so transfer
#: never needs to import cli.py).
RUN_ARTIFACTS = ("audit.jsonl", "diagnosis.json", "trace.json",
                 "pending_case.json", "run_meta.json", "prompt.txt",
                 "prompt_turns.jsonl", "dumps.json")
RUN_RESERVED_DIRS = frozenset({"sessions", "cases", "secrets", "calibration"})

_TMP_SUFFIX = ".tmp"


class BundleError(Exception):
    """Malformed or tampered bundle (missing/bad manifest, hash mismatch)."""


def is_run_dir(path: Path) -> bool:
    """True when ``path`` looks like a diagnosable run directory."""
    if path.name in RUN_RESERVED_DIRS or not path.is_dir():
        return False
    return any((path / m).exists() for m in RUN_ARTIFACTS) \
        or (path / "dumps").is_dir()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + _TMP_SUFFIX)
    tmp.write_bytes(data)
    tmp.replace(path)


def _safe_parts(rel: str) -> list[str] | None:
    """Path parts of a bundle-relative arcname, or None when unsafe.

    Zip-slip guard: rejects absolute paths, ``..`` segments, drive letters
    and backslash separators.
    """
    rel = rel.replace("\\", "/")
    parts = [p for p in rel.split("/") if p not in ("", ".")]
    if not parts or rel.startswith(("/", "~")) \
            or any(":" in p or p == ".." for p in parts):
        return None
    return parts


# ---- export ----

@dataclass
class ExportResult:
    bundle: Path | None = None
    cases: list[str] = field(default_factory=list)
    runs: list[str] = field(default_factory=list)
    calibration: list[str] = field(default_factory=list)
    priors: bool = False
    unknown_runs: list[str] = field(default_factory=list)


def export_bundle(out: Path, out_dir: str | Path, *,
                  cases_dir: str | Path | None = None,
                  run_ids: list[str] | None = None, include_runs: bool = True,
                  include_calibration: bool = True, include_priors: bool = True,
                  dry_run: bool = False) -> ExportResult:
    """Collect everything learnable under ``out_dir`` into one zip bundle.

    ``run_ids`` filters both the case records and the run dirs to the given
    ids (ids matching neither are reported in ``unknown_runs``). With
    ``dry_run`` nothing is written; the result still describes the bundle.
    """
    from ..diagnosis.case_store import CaseStore

    root = Path(out_dir)
    cases_root = Path(cases_dir) if cases_dir else root / "cases"
    result = ExportResult(bundle=None if dry_run else Path(out))
    entries: dict[str, bytes] = {}

    store = CaseStore(cases_root) if cases_root.is_dir() else None
    wanted = set(run_ids or [])
    for record in (store.all() if store else []):
        if wanted and record.run_id not in wanted:
            continue
        entries[CASES_PREFIX + f"{record.run_id}.json"] = \
            record.model_dump_json(indent=2).encode("utf-8")
        result.cases.append(record.run_id)

    if include_runs:
        for run_dir in sorted((p for p in root.glob("*") if is_run_dir(p)),
                              key=lambda p: p.name):
            if wanted and run_dir.name not in wanted:
                continue
            for path in sorted(run_dir.rglob("*")):
                if not path.is_file() or path.suffix == _TMP_SUFFIX:
                    continue
                rel = path.relative_to(run_dir).as_posix()
                entries[RUNS_PREFIX + f"{run_dir.name}/{rel}"] = path.read_bytes()
            result.runs.append(run_dir.name)

    if include_calibration:
        cal_dir = root / "calibration"
        if cal_dir.is_dir():
            for path in sorted(cal_dir.glob("*.json")):
                if path.suffix == _TMP_SUFFIX:
                    continue
                entries[CALIBRATION_PREFIX + path.name] = path.read_bytes()
                result.calibration.append(path.stem)

    if include_priors:
        priors = root / PRIORS_NAME
        if priors.is_file():
            entries[PRIORS_NAME] = priors.read_bytes()
            result.priors = True

    if wanted:
        result.unknown_runs = sorted(wanted - set(result.cases) - set(result.runs))

    if dry_run:
        return result

    manifest = {
        "bundle_version": BUNDLE_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "counts": {
            "cases": len(result.cases),
            "runs": len(result.runs),
            "calibration": len(result.calibration),
            "priors": result.priors,
        },
        "files": {name: _sha256(data) for name, data in sorted(entries.items())},
    }
    entries[MANIFEST_NAME] = json.dumps(manifest, indent=2, sort_keys=True) \
        .encode("utf-8")
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in sorted(entries.items()):
            zf.writestr(name, data)
    return result


# ---- import ----

@dataclass
class ImportResult:
    cases_imported: list[str] = field(default_factory=list)
    cases_skipped: list[str] = field(default_factory=list)
    cases_failed: dict[str, str] = field(default_factory=dict)
    runs_imported: list[str] = field(default_factory=list)
    runs_skipped: list[str] = field(default_factory=list)
    calibration_imported: list[str] = field(default_factory=list)
    calibration_skipped: list[str] = field(default_factory=list)
    priors: str = "absent"  # imported | skipped | absent

    @property
    def ok(self) -> bool:
        return not self.cases_failed

    def summary(self) -> str:
        return (f"cases: {len(self.cases_imported)} imported, "
                f"{len(self.cases_skipped)} skipped, "
                f"{len(self.cases_failed)} failed | "
                f"runs: {len(self.runs_imported)} imported, "
                f"{len(self.runs_skipped)} skipped | "
                f"calibration: {len(self.calibration_imported)} imported, "
                f"{len(self.calibration_skipped)} skipped | "
                f"priors: {self.priors}")


def import_bundle(bundle: Path, out_dir: str | Path, *,
                  cases_dir: str | Path | None = None, revise: bool = False,
                  dry_run: bool = False) -> ImportResult:
    """Apply a learning bundle to this device's ``out_dir``.

    Cases whose run id already exists are skipped unless ``revise``; the same
    rule covers run dirs, calibration files and priors. ``dry_run`` evaluates
    everything (including integrity + schema checks) without writing.
    Raises ``BundleError`` for a missing/foreign manifest or any sha256
    mismatch -- a tampered bundle is never partially applied.
    """
    from ..diagnosis.case_store import CaseStore
    from ..diagnosis.schema import CaseOutcome

    root = Path(out_dir)
    cases_root = Path(cases_dir) if cases_dir else root / "cases"
    result = ImportResult()

    if not bundle.is_file():
        raise BundleError(f"no such bundle: {bundle}")
    with zipfile.ZipFile(bundle) as zf:
        names = set(zf.namelist())
        if MANIFEST_NAME not in names:
            raise BundleError(
                f"not a harness learning bundle: {MANIFEST_NAME} missing")
        try:
            manifest = json.loads(zf.read(MANIFEST_NAME))
        except json.JSONDecodeError as exc:
            raise BundleError(f"unreadable manifest: {exc}") from exc
        if not isinstance(manifest, dict) \
                or manifest.get("bundle_version") != BUNDLE_VERSION:
            raise BundleError(
                f"unsupported bundle_version "
                f"{manifest.get('bundle_version')!r} (expected {BUNDLE_VERSION})")
        expected: dict[str, str] = manifest.get("files") or {}
        for name, digest in expected.items():
            if name not in names:
                raise BundleError(f"bundle file missing: {name}")
            if _sha256(zf.read(name)) != digest:
                raise BundleError(f"integrity check failed for {name} "
                                  "(bundle modified after export)")
        payloads = {name: zf.read(name) for name in expected}

    store = CaseStore(cases_root)
    for name in sorted(k for k in expected if k.startswith(CASES_PREFIX)):
        stem = name[len(CASES_PREFIX):].removesuffix(".json")
        record = None
        error = ""
        try:
            record = CaseOutcome.model_validate_json(payloads[name].decode("utf-8"))
            if stem != record.run_id:
                error = (f"case file {stem!r} does not match its run_id "
                         f"{record.run_id!r}")
                record = None
        except Exception as exc:  # noqa: BLE001 - one bad record must not stop the rest
            error = f"invalid case record: {exc}"
        if record is None:
            result.cases_failed[stem] = error
            continue
        exists = (cases_root / f"{record.run_id}.json").exists()
        if exists and not revise:
            result.cases_skipped.append(record.run_id)
            continue
        if not dry_run:
            try:
                store.record(record, revise=revise)
            except (FileExistsError, ValueError) as exc:
                result.cases_failed[record.run_id] = str(exc)
                continue
        result.cases_imported.append(record.run_id)  # new, or replaced (revise)

    runs: dict[str, dict[tuple[str, ...], bytes]] = {}
    for name in sorted(k for k in expected if k.startswith(RUNS_PREFIX)):
        parts = _safe_parts(name)
        if not parts or len(parts) < 3 or parts[0] != "runs":
            continue  # unsafe arcname: dropped, never escapes the out-dir
        runs.setdefault(parts[1], {})[tuple(parts[2:])] = payloads[name]
    for run_id, files in sorted(runs.items()):
        run_dir = root / run_id
        if run_dir.exists() and not revise:
            result.runs_skipped.append(run_id)
            continue
        safe = True
        if not dry_run:
            for rel, data in files.items():
                _atomic_write(run_dir.joinpath(*rel), data)
        if safe:
            result.runs_imported.append(run_id)  # new, or replaced (revise)

    for name in sorted(k for k in expected if k.startswith(CALIBRATION_PREFIX)):
        dest = root / CALIBRATION_PREFIX / name[len(CALIBRATION_PREFIX):]
        if dest.exists() and not revise:
            result.calibration_skipped.append(dest.stem)
            continue
        if not dry_run:
            _atomic_write(dest, payloads[name])
        result.calibration_imported.append(dest.stem)  # new, or replaced (revise)

    if PRIORS_NAME in expected:
        dest = root / PRIORS_NAME
        if dest.exists() and not revise:
            result.priors = "skipped"
        else:
            if not dry_run:
                _atomic_write(dest, payloads[PRIORS_NAME])
            result.priors = "imported"

    return result
