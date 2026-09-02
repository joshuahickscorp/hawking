"""Dominance scoreboard: useful-work cells, never an invented number.

V-D compares on capability, accepted TPS, tails, energy, and the rest of the
gene card. A cell nobody measured is UNMEASURED with `value: null`, not 0.
`require_measured` raises rather than returning a stand-in.

The board is a view of caller-supplied receipts. It does not take a hardware
measurement. Each cell keeps the evidence tier of its source; tiers are never
merged or upgraded.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

EVIDENCE_TIER_BOARD = "STATIC"
SCHEMA = "hawking.product.scoreboard.v1"
ALLOWED_TIERS = (
    "STATIC",
    "FUNCTIONAL_SIM",
    "COST_MODEL",
    "CYCLE_APPROX",
    "HARDWARE_MEASURED",
)
DOMINANCE_METRICS = (
    "capability",
    "accepted_tps",
    "ttft",
    "tpot",
    "tails",
    "memory",
    "energy",
    "verified_tasks_per_hour",
    "reliability",
    "recovery",
    "portability",
    "developer_burden",
)


class UnmeasuredError(ValueError):
    """Scoreboard refuses to report a value that was not measured."""


def _unmeasured(metric: str, *, reason: str = "not measured") -> dict[str, Any]:
    return {
        "metric": metric,
        "value": None,
        "state": "UNMEASURED",
        "evidence_tier": None,
        "source": None,
        "reason": reason,
    }


def _measured(metric: str, value: Any, *, tier: str, source: str) -> dict[str, Any]:
    return {
        "metric": metric,
        "value": value,
        "state": "MEASURED",
        "evidence_tier": tier,
        "source": source,
        "reason": None,
    }


def _is_bool(value: Any) -> bool:
    return isinstance(value, bool)


def _cell_from_row(metric: str, row: Any, *, source: str) -> Optional[dict[str, Any]]:
    """Parse one metric. A bare number without an evidence tier is not measured."""
    if row is None:
        return _unmeasured(metric, reason="null in source receipt")
    if _is_bool(row):
        return _unmeasured(metric, reason="boolean is not a measurement")
    if isinstance(row, (int, float)):
        return _unmeasured(
            metric,
            reason="numeric value without evidence_tier; refusing to invent a tier",
        )
    if not isinstance(row, Mapping):
        return _unmeasured(metric, reason=f"unsupported cell type {type(row).__name__}")
    tier = row.get("evidence_tier")
    value = row.get("value")
    if value is None:
        return _unmeasured(
            metric,
            reason=str(row.get("reason") or "not measured"),
        )
    if _is_bool(value):
        return _unmeasured(metric, reason="boolean is not a measurement")
    if tier not in ALLOWED_TIERS:
        return _unmeasured(
            metric,
            reason=f"missing or unknown evidence_tier {tier!r}; refusing to invent a tier",
        )
    return _measured(metric, value, tier=str(tier), source=source)


def extract_metrics(doc: Mapping[str, Any], *, source: str) -> dict[str, dict[str, Any]]:
    """Read only explicitly named dominance metrics. Do not alias or infer."""
    block = doc.get("metrics")
    if not isinstance(block, Mapping):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for name in DOMINANCE_METRICS:
        if name not in block:
            continue
        cell = _cell_from_row(name, block[name], source=source)
        if cell is not None:
            out[name] = cell
    return out


def load_scoreboard(paths: Iterable[str | Path]) -> dict[str, Any]:
    """Build a board from receipts. Conflicting measured values are refused."""
    cells = {name: _unmeasured(name) for name in DOMINANCE_METRICS}
    seen: dict[str, dict[str, Any]] = {}
    loaded: list[str] = []
    for raw in paths:
        path = Path(raw)
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise UnmeasuredError(f"receipt unreadable: {path}: {exc}") from exc
        if not isinstance(doc, dict):
            raise UnmeasuredError(f"receipt root must be an object: {path}")
        loaded.append(str(path))
        extracted = extract_metrics(doc, source=str(path))
        for name, cell in extracted.items():
            if cell["state"] != "MEASURED":
                if name not in seen:
                    cells[name] = cell
                continue
            prev = seen.get(name)
            if prev is not None:
                if prev["value"] != cell["value"] or prev["evidence_tier"] != cell["evidence_tier"]:
                    raise UnmeasuredError(
                        f"refusing to merge {name} from {prev['source']} and {cell['source']}"
                    )
                continue
            seen[name] = cell
            cells[name] = cell
    return {
        "schema": SCHEMA,
        "evidence_tier": EVIDENCE_TIER_BOARD,
        "metrics": cells,
        "receipts": loaded,
        "unmeasured_are_null": True,
        "tiers_not_merged": True,
        "gpu_authority": False,
    }


def require_measured(board: Mapping[str, Any], metric: str) -> dict[str, Any]:
    """Return one MEASURED cell, or refuse.

    An UNMEASURED cell, a missing metric, a null value, or a missing evidence
    tier is an error. This function never returns 0 as a stand-in.
    """
    metrics = board.get("metrics")
    if not isinstance(metrics, Mapping):
        raise UnmeasuredError("scoreboard has no metrics; refusing to report a value")
    if metric not in DOMINANCE_METRICS:
        raise UnmeasuredError(
            f"{metric} is not a dominance metric; refusing to report a value"
        )
    cell = metrics.get(metric)
    if not isinstance(cell, Mapping):
        raise UnmeasuredError(f"{metric} is not measured; refusing to report a value")
    if cell.get("state") != "MEASURED":
        raise UnmeasuredError(
            f"{metric} is {cell.get('state') or 'UNMEASURED'}; refusing to report a value"
        )
    if cell.get("value") is None:
        raise UnmeasuredError(
            f"{metric} has no measured value; refusing to invent one"
        )
    if cell.get("evidence_tier") not in ALLOWED_TIERS:
        raise UnmeasuredError(
            f"{metric} has no evidence tier; refusing to invent one"
        )
    return dict(cell)


def qualify(
    board: Mapping[str, Any],
    required: Iterable[str],
) -> dict[str, Any]:
    """Refuse promotion when any required metric is unmeasured. Calls require_measured."""
    missing: list[str] = []
    measured: list[str] = []
    for metric in required:
        try:
            require_measured(board, metric)
        except UnmeasuredError:
            missing.append(metric)
        else:
            measured.append(metric)
    if missing:
        raise UnmeasuredError(
            "qualification refused; unmeasured: " + ", ".join(missing)
        )
    return {
        "schema": "hawking.product.qualify.v1",
        "ok": True,
        "required": list(required),
        "measured": measured,
        "evidence_tier": EVIDENCE_TIER_BOARD,
    }
