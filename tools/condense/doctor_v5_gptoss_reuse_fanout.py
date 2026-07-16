#!/usr/bin/env python3.12
"""Exact unbound GPT-OSS preprocess reuse across ten rates and four branches.

One immutable, range-receipted source traversal may feed every planned output
for its source unit.  Scientific evidence is never shared: each rate/branch
output has a unique namespace, branch dependency chain, artifact instance,
attestation, and receipt.  The live Doctor queue and runtime specs do not import
this module.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import stat
import sys
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import doctor_v5_gptoss_mxfp4 as mxfp4
import doctor_v5_gptoss_parallel_scaffold as parallel


FANOUT_PLAN_SCHEMA = "hawking.doctor_v5_gptoss_reuse_fanout_plan.v1"
BRANCH_RECEIPT_SCHEMA = "hawking.doctor_v5_gptoss_branch_output_receipt.v1"
FANOUT_MERGE_SCHEMA = "hawking.doctor_v5_gptoss_branch_merge_manifest.v1"
DEFAULT_FANOUT_PLAN = parallel.DEFAULT_OUTPUT_ROOT / "reuse_fanout_plan.json"
EXPECTED_JOBS = parallel.EXPECTED_SOURCE_UNITS * len(parallel.RATES) * len(
    parallel.BRANCHES
)
SHA_RE = re.compile(r"[0-9a-f]{64}")
MAX_JSON_BYTES = 256 * 1024 * 1024
BRANCH_ORDER = tuple(row[0] for row in parallel.BRANCHES)


class FanoutError(RuntimeError):
    """A shared-preprocess or isolated-evidence contract was violated."""


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _hash_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _without(doc: dict[str, Any], field: str) -> dict[str, Any]:
    return {key: value for key, value in doc.items() if key != field}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        info = path.stat()
        if path.is_symlink() or not stat.S_ISREG(info.st_mode) \
                or info.st_size > MAX_JSON_BYTES:
            raise FanoutError(f"invalid JSON file: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FanoutError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FanoutError(f"JSON root is not an object: {path}")
    return value


def _write_json(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mxfp4._atomic_json(path, doc)


def _job_id(source_unit_id: str, rate_id: str, branch: str) -> str:
    return f"{source_unit_id}/rate={rate_id}/branch={branch}"


def build_fanout_plan(
    work_plan: dict[str, Any], pending_wiring: dict[str, Any], *,
    created_at: str | None = None,
) -> dict[str, Any]:
    work_errors = parallel.validate_work_plan(work_plan)
    wiring_errors = parallel.validate_pending_wiring(pending_wiring)
    if work_errors or wiring_errors:
        raise FanoutError("invalid parent scaffold: " + "; ".join(
            work_errors + wiring_errors
        ))
    if pending_wiring["work_plan"]["work_plan_sha256"] \
            != work_plan["work_plan_sha256"]:
        raise FanoutError("pending wiring and work plan differ")
    cells = {
        (row["rate_id"], row["branch"]): row
        for row in pending_wiring["cell_bindings"]
    }
    jobs: list[dict[str, Any]] = []
    for source in work_plan["source_units"]:
        source_id = source["unit_id"]
        for rate_id in parallel.RATES:
            prior: list[str] = []
            for branch in BRANCH_ORDER:
                cell = cells[(rate_id, branch)]
                job_id = _job_id(source_id, rate_id, branch)
                namespace_authority = {
                    "work_plan_sha256": work_plan["work_plan_sha256"],
                    "pending_wiring_sha256": pending_wiring["pending_wiring_sha256"],
                    "cell_id": cell["cell_id"],
                    "cell_identity_sha256": cell["cell_identity_sha256"],
                    "source_unit_id": source_id, "rate_id": rate_id,
                    "branch": branch, "adapter_id": cell["adapter_id"],
                }
                jobs.append({
                    "job_id": job_id, "cell_id": cell["cell_id"],
                    "cell_identity_sha256": cell["cell_identity_sha256"],
                    "cell_spec_sha256": cell["cell_spec_sha256"],
                    "source_unit_id": source_id,
                    "source_binding_sha256": source["source_binding_sha256"],
                    "rate_id": rate_id, "branch": branch,
                    "command": cell["command"], "adapter_id": cell["adapter_id"],
                    "preprocess_reuse": {
                        "permitted": True,
                        "scope": "same_source_unit_exact_canonical_values_only",
                        "source_traversal_receipt_required": True,
                    },
                    "job_dependencies": list(prior),
                    "evidence_namespace_sha256": _hash_value(namespace_authority),
                    "evidence_reuse_permitted": False,
                    "output_artifact_instance_must_be_unique": True,
                    "status": "pending_unbound_execution",
                })
                prior.append(job_id)
    jobs.sort(key=lambda row: row["job_id"])
    doc: dict[str, Any] = {
        "schema": FANOUT_PLAN_SCHEMA, "created_at": created_at or _now(),
        "status": "unbound-reuse-plan-only",
        "work_plan_sha256": work_plan["work_plan_sha256"],
        "pending_wiring_sha256": pending_wiring["pending_wiring_sha256"],
        "matrix": {
            "source_preprocess_units": parallel.EXPECTED_SOURCE_UNITS,
            "rates": len(parallel.RATES), "branches": len(BRANCH_ORDER),
            "isolated_output_jobs": len(jobs),
        },
        "reuse_contract": {
            "one_source_traversal_per_source_unit": True,
            "maximum_consumers_per_traversal": len(parallel.RATES) * len(BRANCH_ORDER),
            "canonical_values_may_be_shared": True,
            "scientific_evidence_may_be_shared": False,
            "artifact_instances_may_be_shared": False,
            "branch_dependency_receipt_hashes_required": True,
            "source_staging_gc_requires_all_consumers_terminal": True,
        },
        "jobs": jobs,
        "execution_gate": {
            "executable": False, "reviewed_for_live_campaign": False,
            "source_traversal_receipts_complete": False,
            "all_branch_adapters_reviewed": False,
            "runtime_and_quality_parity_passed": False,
        },
        "quality_claims_permitted": False, "live_mutation_permitted": False,
    }
    doc["fanout_plan_sha256"] = _hash_value(doc)
    errors = validate_fanout_plan(doc, work_plan, pending_wiring)
    if errors:
        raise FanoutError("generated fanout plan invalid: " + "; ".join(errors))
    return doc


def validate_fanout_plan(
    doc: Any, work_plan: dict[str, Any], pending_wiring: dict[str, Any],
) -> list[str]:
    if not isinstance(doc, dict) or doc.get("schema") != FANOUT_PLAN_SCHEMA:
        return ["fanout-plan schema mismatch"]
    errors: list[str] = []
    if doc.get("fanout_plan_sha256") != _hash_value(_without(doc, "fanout_plan_sha256")):
        errors.append("fanout-plan hash mismatch")
    if doc.get("work_plan_sha256") != work_plan.get("work_plan_sha256") \
            or doc.get("pending_wiring_sha256") != pending_wiring.get(
                "pending_wiring_sha256"
            ):
        errors.append("fanout plan parent binding differs")
    if doc.get("status") != "unbound-reuse-plan-only" \
            or doc.get("execution_gate", {}).get("executable") is not False \
            or doc.get("quality_claims_permitted") is not False \
            or doc.get("live_mutation_permitted") is not False:
        errors.append("fanout plan is not fail closed")
    jobs = doc.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != EXPECTED_JOBS:
        errors.append(f"fanout plan does not contain exactly {EXPECTED_JOBS} jobs")
        jobs = []
    source_by_id = {row["unit_id"]: row for row in work_plan.get("source_units", [])}
    cell_by_key = {(row["rate_id"], row["branch"]): row
                   for row in pending_wiring.get("cell_bindings", [])}
    observed_ids: set[str] = set()
    namespaces: set[str] = set()
    for job in jobs:
        job_id = job.get("job_id") if isinstance(job, dict) else None
        source = source_by_id.get(job.get("source_unit_id")) if isinstance(job, dict) else None
        cell = cell_by_key.get((job.get("rate_id"), job.get("branch"))) \
            if isinstance(job, dict) else None
        if not isinstance(job_id, str) or job_id in observed_ids or source is None \
                or cell is None or job_id != _job_id(
                    source["unit_id"], job["rate_id"], job["branch"]
                ) or job.get("cell_id") != cell["cell_id"] \
                or job.get("adapter_id") != cell["adapter_id"] \
                or job.get("source_binding_sha256") != source["source_binding_sha256"] \
                or job.get("status") != "pending_unbound_execution" \
                or job.get("evidence_reuse_permitted") is not False:
            errors.append(f"fanout job identity invalid: {job_id}")
            continue
        observed_ids.add(job_id)
        namespace = job.get("evidence_namespace_sha256")
        if not isinstance(namespace, str) or SHA_RE.fullmatch(namespace) is None \
                or namespace in namespaces:
            errors.append(f"fanout evidence namespace invalid or aliased: {job_id}")
        namespaces.add(namespace)
        namespace_authority = {
            "work_plan_sha256": work_plan["work_plan_sha256"],
            "pending_wiring_sha256": pending_wiring["pending_wiring_sha256"],
            "cell_id": cell["cell_id"],
            "cell_identity_sha256": cell["cell_identity_sha256"],
            "source_unit_id": source["unit_id"], "rate_id": job["rate_id"],
            "branch": job["branch"], "adapter_id": cell["adapter_id"],
        }
        if namespace != _hash_value(namespace_authority) \
                or job.get("cell_identity_sha256") != cell["cell_identity_sha256"] \
                or job.get("cell_spec_sha256") != cell["cell_spec_sha256"] \
                or job.get("command") != cell["command"]:
            errors.append(f"fanout evidence authority differs: {job_id}")
        branch_index = BRANCH_ORDER.index(job["branch"])
        expected_dependencies = [
            _job_id(source["unit_id"], job["rate_id"], branch)
            for branch in BRANCH_ORDER[:branch_index]
        ]
        if job.get("job_dependencies") != expected_dependencies:
            errors.append(f"fanout dependency chain differs: {job_id}")
    expected_ids = {
        _job_id(source_id, rate_id, branch)
        for source_id in source_by_id for rate_id in parallel.RATES
        for branch in BRANCH_ORDER
    }
    if observed_ids != expected_ids:
        errors.append("fanout exact source x rate x branch coverage differs")
    if doc.get("matrix") != {
        "source_preprocess_units": parallel.EXPECTED_SOURCE_UNITS,
        "rates": 10, "branches": 4, "isolated_output_jobs": EXPECTED_JOBS,
    }:
        errors.append("fanout matrix cardinality is invalid")
    reuse = doc.get("reuse_contract")
    if not isinstance(reuse, dict) \
            or reuse.get("one_source_traversal_per_source_unit") is not True \
            or reuse.get("scientific_evidence_may_be_shared") is not False \
            or reuse.get("artifact_instances_may_be_shared") is not False:
        errors.append("fanout reuse/isolation contract is invalid")
    return errors


def build_branch_receipt(
    plan: dict[str, Any], work_plan: dict[str, Any], *, job_id: str,
    source_traversal_receipt: dict[str, Any],
    dependency_receipts: list[dict[str, str]], output_artifact: dict[str, Any],
    method_evidence: dict[str, Any], attestation_root_sha256: str,
) -> dict[str, Any]:
    job = {row["job_id"]: row for row in plan.get("jobs", [])}.get(job_id)
    if job is None:
        raise FanoutError(f"job is absent from fanout plan: {job_id}")
    traversal_errors = parallel.validate_source_traversal_receipt(
        work_plan, source_traversal_receipt
    )
    if traversal_errors or source_traversal_receipt.get("source_unit_id") \
            != job["source_unit_id"]:
        raise FanoutError("source traversal is invalid for branch job: "
                          + "; ".join(traversal_errors))
    expected_dependencies = job["job_dependencies"]
    if not isinstance(dependency_receipts, list) \
            or [row.get("job_id") for row in dependency_receipts
                if isinstance(row, dict)] != expected_dependencies \
            or any(not isinstance(row.get("receipt_sha256"), str)
                   or SHA_RE.fullmatch(row["receipt_sha256"]) is None
                   for row in dependency_receipts if isinstance(row, dict)):
        raise FanoutError("branch dependency receipt references differ")
    for name, artifact in (("output", output_artifact), ("method", method_evidence)):
        if not isinstance(artifact, dict) or not isinstance(artifact.get("sha256"), str) \
                or SHA_RE.fullmatch(artifact["sha256"]) is None \
                or isinstance(artifact.get("bytes"), bool) \
                or not isinstance(artifact.get("bytes"), int) or artifact["bytes"] < 0 \
                or not isinstance(artifact.get("artifact_instance_id"), str) \
                or not artifact["artifact_instance_id"]:
            raise FanoutError(f"{name} artifact identity is invalid")
    if not isinstance(attestation_root_sha256, str) \
            or SHA_RE.fullmatch(attestation_root_sha256) is None:
        raise FanoutError("branch archive attestation root is invalid")
    doc: dict[str, Any] = {
        "schema": BRANCH_RECEIPT_SCHEMA, "created_at": _now(), "status": "complete",
        "fanout_plan_sha256": plan["fanout_plan_sha256"],
        "job_id": job_id, "cell_id": job["cell_id"],
        "source_unit_id": job["source_unit_id"], "rate_id": job["rate_id"],
        "branch": job["branch"], "adapter_id": job["adapter_id"],
        "evidence_namespace_sha256": job["evidence_namespace_sha256"],
        "source_traversal_receipt_sha256": source_traversal_receipt["receipt_sha256"],
        "dependency_receipts": dependency_receipts,
        "output_artifact": output_artifact, "method_evidence": method_evidence,
        "attestation_root_sha256": attestation_root_sha256,
        "claims": {"structural_output_complete": True, "quality": False,
                   "campaign_cell_complete": False},
        "source_files_deleted": False,
    }
    doc["receipt_sha256"] = _hash_value(doc)
    errors = validate_branch_receipt(plan, doc)
    if errors:
        raise FanoutError("generated branch receipt invalid: " + "; ".join(errors))
    return doc


def validate_branch_receipt(plan: dict[str, Any], receipt: Any) -> list[str]:
    if not isinstance(receipt, dict) or receipt.get("schema") != BRANCH_RECEIPT_SCHEMA:
        return ["branch receipt schema mismatch"]
    errors: list[str] = []
    if receipt.get("receipt_sha256") != _hash_value(_without(receipt, "receipt_sha256")):
        errors.append("branch receipt hash mismatch")
    job = {row["job_id"]: row for row in plan.get("jobs", [])}.get(
        receipt.get("job_id")
    )
    if job is None:
        errors.append("branch receipt job is absent from plan")
    elif receipt.get("fanout_plan_sha256") != plan.get("fanout_plan_sha256") \
            or any(receipt.get(field) != job[field] for field in (
                "cell_id", "source_unit_id", "rate_id", "branch", "adapter_id",
                "evidence_namespace_sha256",
            )):
        errors.append("branch receipt does not bind its exact job")
    dependencies = receipt.get("dependency_receipts")
    if job is not None and (
            not isinstance(dependencies, list)
            or [row.get("job_id") for row in dependencies if isinstance(row, dict)]
            != job["job_dependencies"]):
        errors.append("branch receipt dependency list differs")
    elif isinstance(dependencies, list) and any(
            not isinstance(row.get("receipt_sha256"), str)
            or SHA_RE.fullmatch(row["receipt_sha256"]) is None
            for row in dependencies if isinstance(row, dict)):
        errors.append("branch dependency receipt hash is invalid")
    for field in ("source_traversal_receipt_sha256", "attestation_root_sha256"):
        value = receipt.get(field)
        if not isinstance(value, str) or SHA_RE.fullmatch(value) is None:
            errors.append(f"branch receipt {field} is invalid")
    for field in ("output_artifact", "method_evidence"):
        row = receipt.get(field)
        if not isinstance(row, dict) or not isinstance(row.get("sha256"), str) \
                or SHA_RE.fullmatch(row["sha256"]) is None \
                or not isinstance(row.get("bytes"), int) \
                or not isinstance(row.get("artifact_instance_id"), str) \
                or not row["artifact_instance_id"]:
            errors.append(f"branch receipt {field} is invalid")
    if receipt.get("claims") != {
            "structural_output_complete": True, "quality": False,
            "campaign_cell_complete": False,
            } or receipt.get("source_files_deleted") is not False:
        errors.append("branch receipt claims/lifecycle boundary is invalid")
    return errors


def build_merge_manifest(
    plan: dict[str, Any], receipts: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    by_job: dict[str, dict[str, Any]] = {}
    receipt_hashes: set[str] = set()
    output_instances: set[str] = set()
    evidence_instances: set[str] = set()
    traversal_by_source: dict[str, str] = {}
    for receipt in receipts:
        errors = validate_branch_receipt(plan, receipt)
        if errors:
            raise FanoutError("invalid branch receipt: " + "; ".join(errors))
        job_id = receipt["job_id"]
        output_instance = receipt["output_artifact"]["artifact_instance_id"]
        evidence_instance = receipt["method_evidence"]["artifact_instance_id"]
        if job_id in by_job or receipt["receipt_sha256"] in receipt_hashes \
                or output_instance in output_instances \
                or evidence_instance in evidence_instances:
            raise FanoutError("cross-job receipt or artifact evidence alias detected")
        source_id = receipt["source_unit_id"]
        traversal_sha = receipt["source_traversal_receipt_sha256"]
        previous = traversal_by_source.setdefault(source_id, traversal_sha)
        if previous != traversal_sha:
            raise FanoutError("one source unit used multiple preprocess traversals")
        by_job[job_id] = receipt
        receipt_hashes.add(receipt["receipt_sha256"])
        output_instances.add(output_instance)
        evidence_instances.add(evidence_instance)
    expected = {row["job_id"] for row in plan.get("jobs", [])}
    if set(by_job) != expected:
        raise FanoutError(
            f"branch receipt coverage differs: missing={len(expected - set(by_job))} "
            f"extra={len(set(by_job) - expected)}"
        )
    for job_id, receipt in by_job.items():
        for dependency in receipt["dependency_receipts"]:
            observed = by_job[dependency["job_id"]]["receipt_sha256"]
            if dependency["receipt_sha256"] != observed:
                raise FanoutError(f"branch dependency receipt hash differs: {job_id}")
    cells: list[dict[str, Any]] = []
    for cell_id in sorted({row["cell_id"] for row in plan["jobs"]}):
        components = [
            {"job_id": job_id, "receipt_sha256": row["receipt_sha256"],
             "output_artifact": row["output_artifact"],
             "method_evidence": row["method_evidence"]}
            for job_id, row in sorted(by_job.items()) if row["cell_id"] == cell_id
        ]
        cells.append({"cell_id": cell_id, "component_count": len(components),
                      "components_sha256": _hash_value(components),
                      "components": components})
    doc: dict[str, Any] = {
        "schema": FANOUT_MERGE_SCHEMA, "created_at": _now(),
        "status": "structural-components-complete-campaign-evidence-deferred",
        "fanout_plan_sha256": plan["fanout_plan_sha256"],
        "source_traversal_count": len(traversal_by_source),
        "branch_receipt_count": len(by_job), "cells": cells,
        "campaign_cell_completion_claimed": False,
        "quality_claims_permitted": False, "source_files_deleted": False,
    }
    doc["merge_manifest_sha256"] = _hash_value(doc)
    return doc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--work-plan", type=Path, default=parallel.DEFAULT_WORK_PLAN)
    build.add_argument("--pending-wiring", type=Path,
                       default=parallel.DEFAULT_PENDING_WIRING)
    build.add_argument("--output", type=Path, default=DEFAULT_FANOUT_PLAN)
    verify = sub.add_parser("verify")
    verify.add_argument("--work-plan", type=Path, default=parallel.DEFAULT_WORK_PLAN)
    verify.add_argument("--pending-wiring", type=Path,
                        default=parallel.DEFAULT_PENDING_WIRING)
    verify.add_argument("--fanout-plan", type=Path, default=DEFAULT_FANOUT_PLAN)
    args = parser.parse_args(argv)
    try:
        work, wiring = _read_json(args.work_plan), _read_json(args.pending_wiring)
        if args.command == "build":
            plan = build_fanout_plan(work, wiring)
            _write_json(args.output, plan)
            print(json.dumps({
                "status": "ok", "output": str(args.output.resolve()),
                "fanout_plan_sha256": plan["fanout_plan_sha256"],
                "source_preprocess_units": parallel.EXPECTED_SOURCE_UNITS,
                "isolated_output_jobs": EXPECTED_JOBS, "execution_permitted": False,
            }, indent=2, sort_keys=True))
            return 0
        plan = _read_json(args.fanout_plan)
        errors = validate_fanout_plan(plan, work, wiring)
        print(json.dumps({"status": "ok" if not errors else "invalid",
                          "errors": errors}, indent=2, sort_keys=True))
        return 0 if not errors else 2
    except (FanoutError, OSError, KeyError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "refused", "error": str(exc)}, sort_keys=True),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
