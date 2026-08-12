"""Outcome-calibrated confidence: per-LLM-ident, per-subsystem fix-rate bins.

``model_agreement`` used to be a static 0.5; this makes it a measured store.
Every verified case (outcome fixed/partial/not_fixed) contributes its predicted
confidence and its observed outcome to a bin histogram keyed by LLM identity, so
each model calibrates ITSELF (model swaps stay safe). ``agreement_for`` then maps
a raw predicted confidence to an agreement value the scorer uses as
``model_agreement`` -- degrading to 0.5 on absent/thin data instead of
exaggerating.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from .schema import CaseOutcome

# Bin edges (including both endpoints): [0.0, 0.2, ..., 1.0].
BIN_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
BIN_MAXES = tuple(BIN_EDGES[1:])

# Outcome → numeric resolution weight (prompt 03 contract + priors/calibration).
RESOLUTION = {"fixed": 1.0, "partial": 0.5, "not_fixed": 0.0,
              "inconclusive": 0.25, "unknown": 0.5}


def bin_index(predicted: float) -> int:
    """Bin of a predicted confidence under ``BIN_EDGES`` (0-based, 1.0 -> last)."""
    value = max(0.0, min(1.0, float(predicted)))
    for i, hi in enumerate(BIN_MAXES):
        if value <= hi:
            return i
    return len(BIN_MAXES) - 1


@dataclass
class Calibration:
    """Per-ident fix-rate histogram keyed by predicted-confidence bin.

    Each entry is ``(bin_max, observed_fix_rate, n_samples)``; sparse bins
    (n < min_per_bin at build time) carry the parent aggregate bin's rate.
    """

    llm_ident: str
    subsystem_bins: dict[str, list[tuple[float, float, int]]] = field(default_factory=dict)
    aggregate: list[tuple[float, float, int]] = field(default_factory=list)
    created_at: str = ""

    def total_samples(self) -> int:
        return sum(n for _bm, _rate, n in self.aggregate)


def build_calibration(cases: list[CaseOutcome], llm_ident: str,
                      min_per_bin: int = 3) -> Calibration | None:
    """Build a calibration histogram for ONE model identity.

    Only cases whose ``outcome`` carries verified signal (fixed/partial/
    not_fixed) and whose ``llm_ident`` matches count. Bins with fewer than
    ``min_per_bin`` samples inherit the aggregate bin's observed rate (graceful
    degradation, never exaggeration). Returns None when fewer than
    ``min_per_bin`` usable samples exist at all (callers fall back to 0.5).
    """
    usable = [c for c in cases
              if c.llm_ident == llm_ident
              and c.outcome in RESOLUTION
              and c.confidence is not None]
    if len(usable) < min_per_bin:
        return None

    def _bin_rates(group: list[CaseOutcome]) -> list[tuple[float, float, int]]:
        per_bin: dict[int, list[float]] = {i: [] for i in range(len(BIN_MAXES))}
        for case in group:
            per_bin[bin_index(case.confidence or 0.5)].append(
                RESOLUTION[case.outcome])
        out: list[tuple[float, float, int]] = []
        for i, hi in enumerate(BIN_MAXES):
            values = per_bin[i]
            n = len(values)
            rate = (sum(values) / n) if n else 0.5
            out.append((hi, round(rate, 3), n))
        return out

    aggregate = _bin_rates(usable)
    subsystem_bins: dict[str, list[tuple[float, float, int]]] = {}
    for subsystem in sorted({c.subsystem_primary for c in usable
                             if c.subsystem_primary}):
        group = [c for c in usable if c.subsystem_primary == subsystem]
        rates = _bin_rates(group)
        # Sparse bins inherit the aggregate bin's observed rate.
        rates = [(hi, agg_rate if n < min_per_bin else rate, n)
                 for (hi, rate, n), (_, agg_rate, _n_agg) in zip(rates, aggregate)]
        subsystem_bins[subsystem] = rates

    return Calibration(
        llm_ident=llm_ident,
        subsystem_bins=subsystem_bins,
        aggregate=aggregate,
        created_at=datetime.now(UTC).isoformat(),
    )


class CalibrationStore:
    """Per-ident calibration files under ``root`` (default <out-dir>/calibration).

    File layout: ``<root>/{sanitized_llm_ident}.json``. Old-model files are
    never deleted (rollback path: calibration is per ident and retained).
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _filename(ident: str) -> str:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in ident)
        return f"{safe}.json"

    def save(self, cal: Calibration) -> Path:
        path = self.root / self._filename(cal.llm_ident)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(_dump_cal(cal), encoding="utf-8")
        tmp.replace(path)
        return path

    def load(self, llm_ident: str) -> Calibration | None:
        path = self.root / self._filename(llm_ident)
        if not path.exists():
            return None
        try:
            return _load_cal(path.read_text(encoding="utf-8"), llm_ident)
        except Exception:  # noqa: BLE001 - broken calibration must never crash scoring
            return None

    def all(self) -> list[Calibration]:
        out: list[Calibration] = []
        for path in sorted(self.root.glob("*.json")):
            cal = self._try_load(path)  # broken file: skip, never crash evals
            if cal is not None:
                out.append(cal)
        return out

    def _try_load(self, path: Path) -> Calibration | None:
        try:
            return _load_cal(path.read_text(encoding="utf-8"), path.stem)
        except Exception:  # noqa: BLE001 - broken calibration must never crash scoring
            return None


def _load_cal(text: str, llm_ident: str) -> Calibration:
    data = json.loads(text)
    return Calibration(
        llm_ident=data.get("llm_ident", llm_ident),
        subsystem_bins={k: [tuple(t) for t in v]
                        for k, v in data.get("subsystem_bins", {}).items()},
        aggregate=[tuple(t) for t in data.get("aggregate", [])],
        created_at=data.get("created_at", ""),
    )


def _dump_cal(cal: Calibration) -> str:
    payload = {
        "llm_ident": cal.llm_ident,
        "subsystem_bins": cal.subsystem_bins,
        "aggregate": cal.aggregate,
        "created_at": cal.created_at,
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def agreement_for(cal: Calibration | None, predicted: float,
                  subsystem: str | None = None) -> float:
    """Predicted confidence → agreement value in [0,1]; 0.5 when uncalibrated.

    ``1 - abs(predicted - observed_fix_rate)`` from the subsystem histogram,
    falling back to the aggregate histogram when the subsystem is unknown or
    has no data. Zero total samples anywhere → 0.5 (never exaggerate).
    """
    if cal is None:
        return 0.5
    if cal.total_samples() == 0:
        return 0.5
    idx = bin_index(predicted)
    bins = cal.subsystem_bins.get(subsystem) if subsystem else None
    if not bins:
        bins = cal.aggregate
    if idx >= len(bins):
        return 0.5
    _hi, observed_rate, n = bins[idx]
    if n == 0:
        # Subsystem bin exists but has no samples: try the aggregate bin.
        if len(cal.aggregate) > idx and cal.aggregate[idx][2] > 0:
            observed_rate = cal.aggregate[idx][1]
        else:
            return 0.5
    value = 1.0 - abs(max(0.0, min(1.0, float(predicted))) - observed_rate)
    return round(max(0.0, min(1.0, value)), 3)