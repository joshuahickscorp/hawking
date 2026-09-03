"""Shared helpers for the VMCP acceptance lane.

Receipts live under receipts/acceptance/. This module does not write
receipts/headless or receipts/future, and it does not import tools.audit
or tools.roadmap.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from tools.roadmap import lineage

REPO = Path(__file__).resolve().parents[3]
# The canonical roadmap is external and can vanish; tools.roadmap.lineage
# falls back to the digest-verified in-repo copy rather than a placeholder.
ROADMAP = lineage.roadmap_path()
RECEIPT_DIR = REPO / "receipts" / "acceptance"
SCHEMA = "hawking.acceptance.gate.v1"
INDEX_SCHEMA = "hawking.acceptance.index.v1"
VISIONMCP_SRC_DEFAULT = Path("/Users/scammermike/Downloads/hawking/visionmcp/src")

# Catalog acceptance spans (civilization/CAPABILITY_GRAPH.json). Not re-derived.
GATES: dict[str, dict[str, Any]] = {
    "VMCP_STATE_LATTICE": {"start": 7706, "end": 7738},
    "VMCP_DEEP_DIGEST": {"start": 7706, "end": 7738},
    "VMCP_TRUTH_LEDGER": {"start": 7706, "end": 7738},
    "VMCP_RECEIPT_LAW": {"start": 7770, "end": 7790},
    "VMCP_TOOL_DOCTOR": {"start": 7741, "end": 7767},
    "VMCP_FILE_CLASSIFIER": {"start": 7793, "end": 7806},
    "VMCP_WEB_CAPTURE": {"start": 7827, "end": 7840},
    "VMCP_VISUAL_DIFF": {"start": 7841, "end": 7854},
    "VMCP_SPATIAL_VALIDATE": {"start": 7855, "end": 7865},
    "VMCP_PTY_CAPTURE": {"start": 7866, "end": 7881},
    "VMCP_COMPACT_SURFACE": {"start": 7954, "end": 7980},
    "VMCP_AGENTOS_INTEGRATION": {"start": 7628, "end": 7630},
    "AGENTOS_BEHAVIOR_LAB": {"start": 7882, "end": 7910},
    "AGENTOS_DETERMINISTIC_OFFLOAD": {"start": 7485, "end": 7507},
}

E4_RECEIPT_FIELDS: tuple[str, ...] = (
    "tool",
    "version",
    "invocation",
    "input_ids",
    "input_hashes",
    "output_ids",
    "output_hashes",
    "started_at",
    "elapsed_ms",
    "status",
    "limitations",
    "verifier",
    "canary",
)

E2_LATTICE: tuple[str, ...] = (
    "DEEP_DIGEST",
    "ASSET_LATTICE",
    "DECODE_LATTICE",
    "ENTITY_GENOME",
    "RENDER_GENOME",
    "SPATIAL_GENOME",
    "REPAIR_VECTOR",
    "DIRECTOR_STATE",
    "PERFORMANCE_LEDGER",
    "TRUTH_LEDGER",
)

E3_REQUIRED: tuple[str, ...] = (
    "file classifier",
    "hashing",
    "archive/compression",
    "browser/CDP",
    "HTML/DOM capture",
    "CSS parser",
    "source-map parser",
    "visual diff",
    "image handling",
    "OBJ/GLTF parser",
    "spatial validator",
    "independent renderer/viewer",
    "PTY capture",
    "process inspection",
    "profiling hooks",
)

E5_EYE: tuple[str, ...] = (
    "file classification",
    "magic/header identification",
    "hash/size",
    "container type",
    "section inventory",
    "string inventory when appropriate",
    "imports/exports when supported",
    "archive inventory",
    "WASM identification/validation",
    "embedded-resource inventory",
)

E10_FIELDS: tuple[str, ...] = (
    "process identity",
    "argv",
    "cwd",
    "allowlisted environment metadata",
    "terminal text",
    "timestamps",
    "input/output event boundaries",
    "exit code/signal",
    "resize/layout",
    "tool/subprocess events where observable",
    "diff/test markers",
    "screenshot state where useful",
)

E14_RESPONSE: tuple[str, ...] = (
    "status",
    "deep_digest",
    "artifacts",
    "evidence",
    "residuals",
    "next_actions",
    "performance_ms",
    "note",
)

NINE_ACTS: tuple[str, ...] = (
    "see",
    "hold",
    "open",
    "know",
    "make",
    "check",
    "fix",
    "keep",
    "prove",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def quote_roadmap(start: int, end: int) -> str:
    if not ROADMAP.is_file():
        raise FileNotFoundError(f"roadmap not at {ROADMAP}")
    lines = ROADMAP.read_text(encoding="utf-8").splitlines()
    if start < 1 or end > len(lines) or start > end:
        raise ValueError(f"span {start}-{end} out of range (file has {len(lines)} lines)")
    return "\n".join(lines[start - 1 : end])


def ensure_visionmcp() -> Path:
    env = os.environ.get("VISIONMCP_SRC")
    candidates = []
    if env:
        candidates.append(Path(env).expanduser())
    candidates.extend(
        [
            REPO / "visionmcp" / "src",
            VISIONMCP_SRC_DEFAULT,
            Path.home() / ".searcher-donors" / "visionmcp" / "src",
        ]
    )
    for src in candidates:
        if (src / "visionmcp" / "__init__.py").is_file():
            resolved = src.resolve()
            inserted = str(resolved)
            if inserted not in sys.path:
                sys.path.insert(0, inserted)
            os.environ.setdefault("VISIONMCP_SRC", inserted)
            return resolved
    raise FileNotFoundError(
        "VISIONMCP_SRC not importable. Expected visionmcp 0.8.0a2 src/ "
        "(parent of the visionmcp package)."
    )


def call(
    symbol: str,
    fn: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """Call `fn` and record that the named symbol was invoked (not imported)."""
    started = time.perf_counter()
    raised = False
    error: str | None = None
    value: Any = None
    try:
        value = fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 — gate runners must record, not crash
        raised = True
        error = f"{type(exc).__name__}: {exc}"
        value = None
    return {
        "symbol": symbol,
        "kind": "call",
        "raised": raised,
        "error": error,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "value": value,
    }


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def jsonable(value: Any, *, limit: int = 2500) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v, limit=limit) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v, limit=limit) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (bytes, bytearray)):
        return {"_bytes": len(value), "_sha256": sha256_bytes(bytes(value))}
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str) and len(value) > limit:
            return value[:limit] + f"... <truncated {len(value) - limit} chars>"
        return value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return jsonable(value.to_dict(), limit=limit)
    return repr(value)


def which(names: Iterable[str]) -> str | None:
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    return None


def receipt_is_complete(row: Mapping[str, Any]) -> bool:
    """E.4: every listed field present. A missing field is not a pass."""
    for key in E4_RECEIPT_FIELDS:
        if key not in row:
            return False
        if row[key] is None:
            return False
    return True


def gate_receipt(
    *,
    gate: str,
    verdict: str,
    evidence_tier: str,
    invoked: list[dict[str, Any]],
    checks: list[dict[str, Any]],
    measured: Mapping[str, Any] | None,
    output: Mapping[str, Any] | None,
    command: list[str],
    blocker: Mapping[str, Any] | None,
    elapsed_ms: float,
) -> dict[str, Any]:
    if verdict not in {"ACCEPTED", "BLOCKED"}:
        raise ValueError(f"verdict must be ACCEPTED or BLOCKED, got {verdict!r}")
    if evidence_tier not in {
        "STATIC",
        "FUNCTIONAL_SIM",
        "COST_MODEL",
        "CYCLE_APPROX",
        "HARDWARE_MEASURED",
    }:
        raise ValueError(f"unknown evidence_tier {evidence_tier!r}")
    meta = GATES[gate]
    quoted = quote_roadmap(meta["start"], meta["end"])
    failed = [c for c in checks if not c.get("ok")]
    if verdict == "ACCEPTED" and failed:
        raise ValueError(f"{gate}: ACCEPTED with failed checks {failed}")
    if verdict == "BLOCKED" and not blocker:
        raise ValueError(f"{gate}: BLOCKED without an exact missing input")
    calls = [row for row in invoked if row.get("kind") == "call"]
    return {
        "schema": SCHEMA,
        "gate": gate,
        "verdict": verdict,
        "criterion": {
            "file": str(ROADMAP),
            "start_line": meta["start"],
            "end_line": meta["end"],
            "quoted": quoted,
            "weakened": False,
        },
        "command": list(command),
        "invoked_symbols": [
            {
                "symbol": row["symbol"],
                "kind": "call",
                "raised": bool(row.get("raised")),
                "error": row.get("error"),
                "elapsed_ms": row.get("elapsed_ms"),
            }
            for row in calls
        ],
        "checks": jsonable(checks),
        "measured": jsonable(measured or {}),
        "output": jsonable(output or {}),
        "blocker": jsonable(blocker) if blocker else None,
        "evidence_tier": evidence_tier,
        "gpu_authority": False,
        "criterion_weakened": False,
        "generated_at": utc_now(),
        "elapsed_ms": round(float(elapsed_ms), 3),
        "recorded_by": "tools/acceptance/vmcp/run.py",
        "claim_boundary": (
            "FUNCTIONAL_SIM / STATIC acceptance evidence. No claim that VisionMCP "
            "web/3d extras, Blender, Chrome, FPGA, DGX or eGPU ran unless a check "
            "records that binary on PATH. An empty collection is not evidence of absence."
        ),
    }


def write_json(path: Path, doc: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
