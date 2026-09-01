"""Audit the cross-architecture accelerator expansion without running hardware."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hcli.persist import atomic_write_json
from hcli.physical_graph import compile_physical_graph
from tools.accelerator.accelerator_runner import (
    build_compiled_queue,
    validate_compiled_queue,
)
from tools.accelerator.architecture_atlas import (
    EVIDENCE_CLASSES,
    PLANNING_BENCH,
    PRIMITIVES,
    SOURCE_SCHOOLS,
    STATUSES,
    build_atlas,
    validate_atlas,
)
from tools.accelerator.repatriation_effects import build_effects, validate_effects


SCHEMA = "hawking.accelerator.repatriation_audit.v1"
DEFAULT_OUT = Path("receipts/headless/ACCELERATOR_REPATRIATION_AUDIT.json")


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _check(
    check_id: str,
    requirement: str,
    passed: bool,
    evidence: Sequence[str],
    observed: Any = None,
) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "requirement": requirement,
        "passed": bool(passed),
        "evidence": list(evidence),
        "observed": observed,
    }


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _source_paths(root: Path, atlas: Mapping[str, Any]) -> list[str]:
    paths: set[str] = set()
    for entry in atlas.get("entries") or []:
        if not isinstance(entry, Mapping):
            continue
        for raw in entry.get("source_evidence") or []:
            path = Path(str(raw)).expanduser()
            if not path.is_absolute():
                path = root / path
            paths.add(str(path.resolve(strict=False)))
    return sorted(paths)


def validate_audit(audit: Mapping[str, Any]) -> dict[str, Any]:
    if audit.get("schema") != SCHEMA:
        raise ValueError(f"schema must be {SCHEMA}")
    checks = audit.get("checks")
    if not isinstance(checks, list) or not checks:
        raise ValueError("audit checks are missing")
    if not all(isinstance(row, Mapping) and isinstance(row.get("passed"), bool) for row in checks):
        raise ValueError("audit checks must carry boolean passed fields")
    expected = _hash({key: value for key, value in audit.items() if key != "fingerprint"})
    if audit.get("fingerprint") != expected:
        raise ValueError("audit fingerprint does not match canonical body")
    if audit.get("passed") is not all(row["passed"] for row in checks):
        raise ValueError("audit passed flag disagrees with checks")
    return {
        "schema": "hawking.accelerator.repatriation_audit_validation.v1",
        "passed": True,
        "check_count": len(checks),
        "failed_checks": [row["check_id"] for row in checks if not row["passed"]],
        "claim_boundary": "audit proves structural and provenance invariants, not physical performance",
    }


def build_audit(*, repo_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve() if repo_root else REPO
    atlas_path = root / "receipts" / "headless" / "ACCELERATOR_ARCHITECTURE_ATLAS.json"
    queue_path = root / "receipts" / "headless" / "ACCELERATOR_REPATRIATION_QUEUE.json"
    effects_path = root / "receipts" / "headless" / "ACCELERATOR_REPATRIATION_EFFECTS.json"
    atlas = _load_json(atlas_path)
    queue = _load_json(queue_path)
    effects = _load_json(effects_path)
    atlas_result = validate_atlas(atlas)
    queue_result = validate_compiled_queue(queue)
    effects_result = validate_effects(effects)
    entries = atlas.get("entries") if isinstance(atlas.get("entries"), list) else []
    entry_ids = {str(row.get("behavior_id")) for row in entries if isinstance(row, Mapping)}
    specs = queue.get("specs") if isinstance(queue.get("specs"), list) else []
    spec_models = {
        str((row.get("target") or {}).get("model_identity"))
        for row in specs
        if isinstance(row, Mapping) and isinstance(row.get("target"), Mapping)
    }
    evidence_paths = _source_paths(root, atlas)
    missing_evidence = [path for path in evidence_paths if not Path(path).is_file()]
    coverage = atlas.get("source_technique_coverage")
    coverage_rows = coverage if isinstance(coverage, list) else []
    coverage_schools = {str(row.get("source_school")) for row in coverage_rows if isinstance(row, Mapping)}
    queue_models = {
        str((row.get("target") or {}).get("model"))
        for row in (atlas.get("experiment_queue", {}).get("experiments") or [])
        if isinstance(row, Mapping) and isinstance(row.get("target"), Mapping)
    }
    hwir_ids = {
        str(row.get("behavior_id"))
        for row in (atlas.get("hwir_hypotheses") or [])
        if isinstance(row, Mapping)
    }
    asic_rows = atlas.get("asic_candidate_ledger", {}).get("entries") or []
    primitives = {
        str(row.get("hawking_primitive"))
        for row in entries
        if isinstance(row, Mapping)
    }
    evidence_labels = {
        str(row.get("evidence_class"))
        for row in entries
        if isinstance(row, Mapping)
    }
    statuses = {
        str(row.get("status"))
        for row in entries
        if isinstance(row, Mapping)
    }
    graph = compile_physical_graph(
        {"model_id": "Qwen3.8-27B", "organs": []},
        architecture_atlas=atlas,
        backend="metal",
    )
    rep_policy = graph.get("execution_policy", {}).get("architecture_repatriation", {})
    checks = [
        _check(
            "atlas-validator",
            "The canonical atlas passes its own schema, effect, queue, HWIR, and ASIC invariants.",
            atlas_result.get("passed") is True,
            [str(atlas_path)],
            atlas_result,
        ),
        _check(
            "effects-ledger",
            "Every imported behavior binds its source school, invariant, Hawking primitive, implementation, scope, evidence, measured-result boundary, and falsifier without promoting a generic law.",
            effects_result.get("passed") is True
            and {str(row.get("behavior_id")) for row in effects.get("entries") or [] if isinstance(row, Mapping)} == entry_ids
            and effects.get("transfer_policy", {}).get("current_physical_law_count") == 0,
            [str(effects_path), str(atlas_path), str(queue_path)],
            effects_result,
        ),
        _check(
            "source-schools",
            "Every bounded source architecture school is represented.",
            set(SOURCE_SCHOOLS).issubset(set(atlas.get("source_schools") or []))
            and set(SOURCE_SCHOOLS).issubset(coverage_schools),
            [str(atlas_path)],
            {"required": len(SOURCE_SCHOOLS), "covered": len(coverage_schools)},
        ),
        _check(
            "technique-coverage",
            "Named source techniques are collapsed into Hawking physical behaviors rather than product ports.",
            len(coverage_rows) >= 40
            and all(
                isinstance(row, Mapping)
                and str(row.get("behavior_id")) in entry_ids
                and str(row.get("source_school")) in set(SOURCE_SCHOOLS)
                for row in coverage_rows
            ),
            [str(atlas_path)],
            {"technique_rows": len(coverage_rows)},
        ),
        _check(
            "taxonomy-and-primitives",
            "The full behavior taxonomy and backend-neutral primitive vocabulary are present.",
            len(set(atlas.get("behavior_taxonomy") or [])) >= 21
            and set(PRIMITIVES).issubset(set(atlas.get("backend_neutral_primitives") or []))
            and primitives.issubset(set(PRIMITIVES)),
            [str(atlas_path), "hcli/physical_graph.py"],
            {"taxonomy": len(atlas.get("behavior_taxonomy") or []), "primitives": len(atlas.get("backend_neutral_primitives") or [])},
        ),
        _check(
            "evidence-provenance",
            "Every behavior has an allowed evidence label, status, falsifier, and present source evidence path.",
            evidence_labels.issubset(set(EVIDENCE_CLASSES))
            and statuses.issubset(set(STATUSES))
            and not missing_evidence
            and all(isinstance(row, Mapping) and str(row.get("cheapest_falsifier") or "").strip() for row in entries),
            [str(atlas_path), *evidence_paths],
            {"missing_paths": missing_evidence, "evidence_labels": sorted(evidence_labels)},
        ),
        _check(
            "qwen27-flash-funnel",
            "Qwen27 is the rapid laboratory and Flash is the adversarial transfer frontier.",
            {"Qwen27", "Flash"}.issubset(queue_models)
            and {"Qwen3.8-27B", "Qwen3.8-Flash-Next"}.issubset(spec_models),
            [str(atlas_path), str(queue_path)],
            {"atlas_models": sorted(queue_models), "spec_models": sorted(spec_models)},
        ),
        _check(
            "experiment-contract",
            "Each compiled spec carries model/NX/NR, organ/range, backend/lowering, state/session, receipt, metrics, and shell-free detached execution.",
            bool(specs)
            and all(
                isinstance(row, Mapping)
                and all(row.get(key) for key in ("model_identity", "nx_identity", "nr_identity", "organ", "organ_range", "backend", "kernel_lowering", "verification_mode", "benchmark_mode", "state_session_inputs", "output_receipt_path", "metrics"))
                and isinstance(row.get("command"), list)
                and (row.get("runner") or {}).get("detached") is True
                and (row.get("runner") or {}).get("shell") is False
                for row in specs
            ),
            [str(queue_path), "tools/accelerator/accelerator_runner.py"],
            {"specs": len(specs)},
        ),
        _check(
            "verification-funnel",
            "The queue preserves structural, diagnostic, protected, and promotion boundaries.",
            queue_result.get("passed") is True
            and all(
                (row.get("verification_mode") == "structural_then_diagnostic_then_protected")
                for row in specs
                if isinstance(row, Mapping)
            )
            and not (queue.get("funnel", {}).get("promotion") or []),
            [str(queue_path)],
            queue.get("funnel"),
        ),
        _check(
            "hwir-automatic-feed",
            "Every atlas behavior emits a derived HWIR hypothesis without a board/timing claim.",
            hwir_ids == entry_ids
            and all(str(row.get("label") or "").startswith("[D]") for row in atlas.get("hwir_hypotheses") or [] if isinstance(row, Mapping)),
            [str(atlas_path), "hcli/agentos/fpga_preboard.py"],
            {"entries": len(entry_ids), "hwir": len(hwir_ids)},
        ),
        _check(
            "asic-watchlist",
            "ASIC candidates remain watchlist-only until repeated cross-model/hardware survival is proven.",
            bool(asic_rows)
            and all(isinstance(row, Mapping) and row.get("status") == "WATCHLIST" and row.get("asic_candidate") is False for row in asic_rows),
            [str(atlas_path)],
            {"watchlist": len(asic_rows)},
        ),
        _check(
            "physical-graph-law",
            "PhysicalGraph makes layout, stationarity, memory-tier identity, movement cost, and no-device-prestige authority explicit.",
            rep_policy.get("memory_tier_is_executable_identity") is True
            and rep_policy.get("stationarity_is_explicit") is True
            and rep_policy.get("move_or_recompute_is_explicit") == "costed_dependency_query"
            and rep_policy.get("device_count_is_not_speed_authority") is True
            and graph.get("representation", {}).get("layout_algebra", {}).get("selection_is_parameterized") is True,
            [str(atlas_path), "hcli/physical_graph.py"],
            {"selected_behaviors": graph.get("architecture_repatriation", {}).get("selected_behavior_ids", [])},
        ),
        _check(
            "artifact-identity",
            "Generated canonical artifacts agree with their builders and carry explicit non-performance claim boundaries.",
            atlas.get("fingerprint") == build_atlas(repo_root=root).get("fingerprint")
            and queue.get("fingerprint") == build_compiled_queue(repo_root=root).get("fingerprint")
            and effects.get("fingerprint") == build_effects(repo_root=root).get("fingerprint")
            and "performance" in str(atlas.get("claim_boundary", "")).lower()
            and all(
                token in str(queue.get("claim_boundary", "")).lower()
                for token in ("physical", "timing", "capability")
            ),
            [str(atlas_path), str(queue_path), str(effects_path)],
            {
                "atlas_fingerprint": atlas.get("fingerprint"),
                "queue_fingerprint": queue.get("fingerprint"),
                "effects_fingerprint": effects.get("fingerprint"),
            },
        ),
    ]
    body: dict[str, Any] = {
        "schema": SCHEMA,
        "version": 1,
        "bench": dict(PLANNING_BENCH),
        "passed": all(row["passed"] for row in checks),
        "checks": checks,
        "artifact_paths": {
            "atlas": str(atlas_path),
            "queue": str(queue_path),
            "effects": str(effects_path),
        },
        "claim_boundary": "This audit proves structural/provenance/funnel invariants only; it does not claim physical performance, capability, FPGA timing, or ASIC suitability.",
    }
    body["fingerprint"] = _hash(body)
    return body


def emit_audit(*, repo_root: str | Path | None = None, output: str | Path | None = None) -> Path:
    root = Path(repo_root).expanduser().resolve() if repo_root else REPO
    destination = Path(output).expanduser() if output else root / DEFAULT_OUT
    if not destination.is_absolute():
        destination = root / destination
    body = build_audit(repo_root=root)
    validate_audit(body)
    atomic_write_json(destination, body)
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--emit", default=None)
    parser.add_argument("--validate", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.validate:
        print(json.dumps(validate_audit(_load_json(Path(args.validate).expanduser())), indent=2, sort_keys=True))
        return 0
    destination = emit_audit(repo_root=args.repo_root, output=args.emit)
    body = _load_json(destination)
    print(json.dumps({"status": "PASSED" if body["passed"] else "FAILED", "path": str(destination), "fingerprint": body["fingerprint"], "checks": len(body["checks"])}, sort_keys=True))
    return 0 if body["passed"] else 1


__all__ = ["DEFAULT_OUT", "SCHEMA", "build_audit", "emit_audit", "main", "validate_audit"]


if __name__ == "__main__":
    raise SystemExit(main())
