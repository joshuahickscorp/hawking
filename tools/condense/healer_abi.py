#!/usr/bin/env python3.12
"""Versioned correction-program ABI for Hawking's Condensation Doctor v2.

This module is deliberately stdlib-only and execution-free.  It defines and
validates the identity of a treatment program before an expensive experiment is
allowed to exist.  A program is a DAG of base rewrites, static corrections,
token-gated refinements, state operators, and system-level verification steps.

The ABI is backend-neutral: Apple CPU, Metal, CUDA, and distributed runtimes may
implement the same operator, but their evidence and measurements never alias.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


PROGRAM_SCHEMA = "hawking.healer_program.v2"
ARTIFACT_SCHEMA = "hawking.healer_artifact.v2"
OBSERVATION_SCHEMA = "hawking.healer_observation.v2"
CELL_SCHEMA = "hawking.healer_cell.v2"
CHECKPOINT_SCHEMA = "hawking.healer_checkpoint.v2"
POLICY_VERSION = "doctor-v2.2026-07-12.1"

OPERATOR_KINDS = {
    "analyze",
    "base_transform",
    "base_codec",
    "base_rewrite",
    "static_correction",
    "gated_correction",
    "train",
    "package",
    "evaluate",
    "state_codec",
    "runtime_policy",
    "retrieval",
    "verifier",
}
PHASES = {"offline", "load", "prefill", "decode", "post_decode"}
IMPLEMENTATION_STATES = {
    "measured",
    "oracle",
    "prototype",
    "runtime_gated",
    "research",
    "unimplemented",
}
BACKENDS = {"apple_cpu", "metal", "cuda", "distributed", "future_specialized"}
PROOF_STATES = (
    "planned",
    "reconstruction_oracle",
    "packed_artifact",
    "native_runtime_parity",
    "resident_capability",
    "capability_efficiency_promoted",
)


class AbiError(ValueError):
    """A fail-closed ABI validation error."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def hash_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def atomic_json(path: Path, value: Any) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _finite(value: Any, *, positive: bool = False, nonnegative: bool = False) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        return False
    if positive and float(value) <= 0:
        return False
    if nonnegative and float(value) < 0:
        return False
    return True


def _identity_payload(program: dict[str, Any]) -> dict[str, Any]:
    payload = dict(program)
    payload.pop("program_sha256", None)
    payload.pop("generated_at", None)
    return payload


def stamp_program(program: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(program))
    out["program_sha256"] = hash_value(_identity_payload(out))
    return out


def stamp_cell(cell: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(cell))
    payload = dict(out)
    payload.pop("cell_sha256", None)
    payload.pop("generated_at", None)
    out["cell_sha256"] = hash_value(payload)
    return out


def _cycle(nodes: dict[str, dict[str, Any]]) -> list[str] | None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str, trail: list[str]) -> list[str] | None:
        if node_id in visiting:
            start = trail.index(node_id) if node_id in trail else 0
            return trail[start:] + [node_id]
        if node_id in visited:
            return None
        visiting.add(node_id)
        for dep in nodes[node_id].get("depends_on", []):
            found = visit(dep, trail + [node_id])
            if found:
                return found
        visiting.remove(node_id)
        visited.add(node_id)
        return None

    for node_id in nodes:
        found = visit(node_id, [])
        if found:
            return found
    return None


def validate_program(program: Any, *, allow_planned: bool = True) -> list[str]:
    errors: list[str] = []
    if not isinstance(program, dict):
        return ["program must be an object"]
    if program.get("schema") != PROGRAM_SCHEMA:
        errors.append(f"schema must be {PROGRAM_SCHEMA}")
    if program.get("policy_version") != POLICY_VERSION:
        errors.append(f"policy_version must be {POLICY_VERSION}")
    if program.get("mode") not in {"planned", "executable"}:
        errors.append("mode must be planned or executable")
    if not allow_planned and program.get("mode") != "executable":
        errors.append("planned program is not executable")

    model = program.get("model")
    if not isinstance(model, dict):
        errors.append("model binding must be an object")
    else:
        if not isinstance(model.get("label"), str) or not model["label"].strip():
            errors.append("model.label missing")
        if not _finite(model.get("params_b"), positive=True):
            errors.append("model.params_b must be positive finite")
        active = model.get("active_b")
        if active is not None and (not _finite(active, positive=True) or (
            _finite(model.get("params_b"), positive=True) and float(active) > float(model["params_b"])
        )):
            errors.append("model.active_b must be positive and <= params_b")
        for field in ("parent_revision_sha256", "config_sha256", "tokenizer_sha256"):
            value = model.get(field)
            if program.get("mode") == "executable" and not is_sha256(value):
                errors.append(f"executable program requires model.{field}")
            elif program.get("mode") == "planned" and value not in {None, "required"} and not is_sha256(value):
                errors.append(f"planned model.{field} must be required, null, or SHA-256")

    target = program.get("target")
    if not isinstance(target, dict) or not _finite(target.get("physical_bpw_ceiling") if isinstance(target, dict) else None, positive=True):
        errors.append("target.physical_bpw_ceiling must be positive finite")
    elif not _finite(target.get("resident_bytes_ceiling"), positive=True):
        errors.append("target.resident_bytes_ceiling must be positive finite")

    runtime_identity = program.get("runtime_identity")
    runtime_fields = (
        "backend_kernel_sha256", "drafter_artifact_sha256", "verifier_path_sha256",
        "kv_policy_sha256", "cache_namespace_sha256", "adaptive_policy_sha256",
    )
    if not isinstance(runtime_identity, dict):
        errors.append("runtime_identity missing")
    else:
        for field in runtime_fields:
            value = runtime_identity.get(field)
            if program.get("mode") == "executable" and not is_sha256(value):
                errors.append(f"executable program requires runtime_identity.{field}")
            elif program.get("mode") == "planned" and value not in {None, "required"} and not is_sha256(value):
                errors.append(f"planned runtime_identity.{field} invalid")

    nodes_raw = program.get("operators")
    if not isinstance(nodes_raw, list) or not nodes_raw:
        errors.append("operators must be a non-empty list")
        nodes_raw = []
    nodes: dict[str, dict[str, Any]] = {}
    for idx, node in enumerate(nodes_raw):
        prefix = f"operators[{idx}]"
        if not isinstance(node, dict):
            errors.append(f"{prefix} must be an object")
            continue
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id:
            errors.append(f"{prefix}.id missing")
            continue
        if node_id in nodes:
            errors.append(f"duplicate operator id {node_id}")
        nodes[node_id] = node
        if node.get("kind") not in OPERATOR_KINDS:
            errors.append(f"{node_id}: unknown operator kind")
        if node.get("phase") not in PHASES:
            errors.append(f"{node_id}: unknown phase")
        if node.get("implementation_status") not in IMPLEMENTATION_STATES:
            errors.append(f"{node_id}: invalid implementation_status")
        if not isinstance(node.get("mechanism"), str) or not node["mechanism"]:
            errors.append(f"{node_id}: mechanism missing")
        if not isinstance(node.get("mechanism_version"), str) or not node["mechanism_version"]:
            errors.append(f"{node_id}: mechanism_version missing")
        deps = node.get("depends_on")
        if not isinstance(deps, list) or any(not isinstance(dep, str) for dep in deps):
            errors.append(f"{node_id}: depends_on must be a string list")
        support = node.get("backend_support")
        if not isinstance(support, dict) or set(support) != BACKENDS:
            errors.append(f"{node_id}: backend_support must name every backend")
        elif any(v not in {"measured", "prototype", "gated", "unsupported", "research"} for v in support.values()):
            errors.append(f"{node_id}: invalid backend support state")
        cost = node.get("cost_contract")
        if not isinstance(cost, dict) or cost.get("actual_bytes_authoritative") is not True:
            errors.append(f"{node_id}: cost_contract must make actual bytes authoritative")
        elif node.get("kind") in {
            "gated_correction", "state_codec", "runtime_policy", "retrieval", "verifier", "evaluate"
        }:
            required = cost.get("dynamic_required_metrics")
            if required != ["mean", "p95", "worst"]:
                errors.append(f"{node_id}: dynamic operator must require mean/p95/worst costs")
        executor = node.get("executor")
        if not isinstance(executor, dict) or not isinstance(executor.get("wired"), bool):
            errors.append(f"{node_id}: executor.wired Boolean required")
        elif executor.get("wired"):
            if program.get("mode") != "executable":
                errors.append(f"{node_id}: wired executor requires executable program mode")
            if node.get("implementation_status") in {"research", "unimplemented"}:
                errors.append(f"{node_id}: research/unimplemented executor cannot be wired")
            if not is_sha256(executor.get("source_sha256")):
                errors.append(f"{node_id}: wired executor requires source_sha256")
            if not isinstance(executor.get("argv"), list) or not executor["argv"]:
                errors.append(f"{node_id}: wired executor requires argv")

    for node_id, node in nodes.items():
        for dep in node.get("depends_on", []):
            if dep not in nodes:
                errors.append(f"{node_id}: unknown dependency {dep}")
    if nodes:
        found = _cycle(nodes)
        if found:
            errors.append("operator DAG cycle: " + " -> ".join(found))

    evidence = program.get("evidence_contract")
    if not isinstance(evidence, dict):
        errors.append("evidence_contract missing")
    else:
        if evidence.get("proof_states") != list(PROOF_STATES):
            errors.append("evidence_contract proof-state order mismatch")
        if evidence.get("selection_set_independent") is not True:
            errors.append("selection set must be independent from final evaluation")
        if evidence.get("negative_results_retained") is not True:
            errors.append("negative results must be retained")

    expected = program.get("program_sha256")
    if not is_sha256(expected):
        errors.append("program_sha256 missing or invalid")
    elif expected != hash_value(_identity_payload(program)):
        errors.append("program_sha256 does not match canonical identity")
    return errors


def validate_artifact(artifact: Any, program: dict[str, Any]) -> list[str]:
    errors = validate_program(program, allow_planned=False)
    if errors:
        return ["bound program invalid: " + error for error in errors]
    if not isinstance(artifact, dict):
        return ["artifact must be an object"]
    if artifact.get("schema") != ARTIFACT_SCHEMA:
        errors.append(f"artifact schema must be {ARTIFACT_SCHEMA}")
    if artifact.get("program_sha256") != program.get("program_sha256"):
        errors.append("artifact is not bound to the exact program")
    if not is_sha256(artifact.get("packed_base_sha256")):
        errors.append("packed_base_sha256 missing")
    if artifact.get("all_tensor_ownership_complete") is not True:
        errors.append("all tensor ownership must be complete")
    if artifact.get("dense_parent_fallback") is not False:
        errors.append("dense parent fallback must be false")
    files = artifact.get("files")
    if not isinstance(files, list) or not files:
        errors.append("artifact files missing")
    else:
        total = 0
        for idx, row in enumerate(files):
            if not isinstance(row, dict) or not is_sha256(row.get("sha256")) or not isinstance(row.get("bytes"), int) or row["bytes"] < 0:
                errors.append(f"artifact files[{idx}] invalid")
            else:
                total += row["bytes"]
        if isinstance(artifact.get("physical_model_bytes"), int) and artifact["physical_model_bytes"] != total:
            errors.append("physical_model_bytes does not equal exact file-byte sum")
    for field in ("physical_model_bytes", "resident_peak_bytes"):
        if not isinstance(artifact.get(field), int) or artifact[field] <= 0:
            errors.append(f"artifact {field} must be a positive integer")
    dynamic = artifact.get("dynamic_costs")
    if not isinstance(dynamic, dict) or any(not _finite(dynamic.get(k), nonnegative=True) for k in ("mean_bytes_per_token", "p95_bytes_per_token", "worst_bytes_per_token")):
        errors.append("artifact dynamic mean/p95/worst byte costs required")
    elif not (dynamic["mean_bytes_per_token"] <= dynamic["p95_bytes_per_token"] <= dynamic["worst_bytes_per_token"]):
        errors.append("artifact dynamic byte costs must be monotonic")
    return errors


def validate_cell(cell: Any, program: dict[str, Any]) -> list[str]:
    """Validate one immutable, backend-specific unit of campaign work."""
    errors: list[str] = []
    if not isinstance(cell, dict):
        return ["cell must be an object"]
    if cell.get("schema") != CELL_SCHEMA:
        errors.append(f"cell schema must be {CELL_SCHEMA}")
    if cell.get("program_sha256") != program.get("program_sha256"):
        errors.append("cell is not bound to the exact program")
    if cell.get("backend") not in BACKENDS:
        errors.append("cell backend invalid")
    if cell.get("proof_state") not in PROOF_STATES:
        errors.append("cell proof_state invalid")
    if not isinstance(cell.get("fidelity"), str) or not cell["fidelity"]:
        errors.append("cell fidelity missing")
    if not isinstance(cell.get("seed"), int):
        errors.append("cell seed missing")
    for field in ("calibration_sha256", "selection_sha256", "final_eval_sha256", "worker_source_sha256"):
        if not is_sha256(cell.get(field)):
            errors.append(f"cell {field} missing")
    estimate = cell.get("resource_estimate")
    if not isinstance(estimate, dict) or any(
        not _finite(estimate.get(field), nonnegative=True)
        for field in ("peak_memory_bytes", "scratch_bytes", "source_read_bytes", "estimated_seconds")
    ):
        errors.append("cell resource_estimate incomplete")
    checkpoint = cell.get("checkpoint_contract")
    required_checkpoint = {
        "operator_state", "optimizer", "microstep", "gradient_accumulation",
        "rng", "sampler_cursor", "teacher_cache_identity", "source_shard_offset",
        "partial_output_hashes", "resume_command_identity",
    }
    if not isinstance(checkpoint, dict) or set(checkpoint.get("required_state", [])) != required_checkpoint:
        errors.append("cell checkpoint contract does not require exact-resume state")
    expected = cell.get("cell_sha256")
    payload = dict(cell)
    payload.pop("cell_sha256", None)
    payload.pop("generated_at", None)
    if not is_sha256(expected) or expected != hash_value(payload):
        errors.append("cell_sha256 missing or mismatched")
    return errors


def validate_checkpoint(checkpoint: Any, cell: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(checkpoint, dict):
        return ["checkpoint must be an object"]
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        errors.append(f"checkpoint schema must be {CHECKPOINT_SCHEMA}")
    if checkpoint.get("cell_sha256") != cell.get("cell_sha256"):
        errors.append("checkpoint is not bound to the exact cell")
    required_hashes = (
        "operator_state_sha256", "optimizer_sha256", "rng_sha256", "sampler_sha256",
        "teacher_cache_sha256", "partial_outputs_sha256", "resume_command_sha256",
    )
    for field in required_hashes:
        if not is_sha256(checkpoint.get(field)):
            errors.append(f"checkpoint {field} missing")
    for field in ("microstep", "gradient_accumulation_phase", "source_shard_index", "source_byte_offset"):
        if not isinstance(checkpoint.get(field), int) or checkpoint[field] < 0:
            errors.append(f"checkpoint {field} invalid")
    if checkpoint.get("fsync_complete") is not True:
        errors.append("checkpoint fsync_complete must be true")
    return errors


def validate_observation(observation: Any, cell: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(observation, dict):
        return ["observation must be an object"]
    if observation.get("schema") != OBSERVATION_SCHEMA:
        errors.append(f"observation schema must be {OBSERVATION_SCHEMA}")
    if observation.get("cell_sha256") != cell.get("cell_sha256"):
        errors.append("observation is not bound to the exact cell")
    if observation.get("status") not in {
        "succeeded", "complete_negative", "failed_retryable", "failed_terminal",
        "invalidated", "superseded",
    }:
        errors.append("observation status invalid")
    if observation.get("proof_state") not in PROOF_STATES:
        errors.append("observation proof_state invalid")
    quality = observation.get("quality")
    if not isinstance(quality, dict) or not isinstance(quality.get("capability_vector"), dict) \
            or not _finite(quality.get("uncertainty"), nonnegative=True):
        errors.append("observation quality vector/uncertainty missing")
    costs = observation.get("costs")
    cost_fields = (
        "physical_model_bytes", "resident_peak_bytes", "mean_bytes_per_token",
        "p95_bytes_per_token", "worst_bytes_per_token", "joules_per_accepted_token",
        "p50_latency_ms", "p95_latency_ms", "accepted_tokens", "rejected_tokens",
    )
    if not isinstance(costs, dict) or any(not _finite(costs.get(field), nonnegative=True) for field in cost_fields):
        errors.append("observation cost vector incomplete")
    elif not (costs["mean_bytes_per_token"] <= costs["p95_bytes_per_token"] <= costs["worst_bytes_per_token"]):
        errors.append("observation dynamic byte costs must be monotonic")
    safety = observation.get("safety")
    if not isinstance(safety, dict) or safety.get("memory_pressure") != "normal" \
            or safety.get("swap_used_bytes") != 0:
        errors.append("observation requires normal pressure and zero swap")
    outputs = observation.get("output_sha256s")
    if not isinstance(outputs, list) or any(not is_sha256(value) for value in outputs):
        errors.append("observation output hashes invalid")
    if observation.get("negative_result_retained") is not True:
        errors.append("observation must retain negative results")
    expected = observation.get("observation_sha256")
    payload = dict(observation)
    payload.pop("observation_sha256", None)
    payload.pop("generated_at", None)
    if not is_sha256(expected) or expected != hash_value(payload):
        errors.append("observation_sha256 missing or mismatched")
    return errors


def make_planned_program(*, label: str, params_b: float, active_b: float | None,
                         physical_bpw: float, operators: list[dict[str, Any]]) -> dict[str, Any]:
    program = {
        "schema": PROGRAM_SCHEMA,
        "policy_version": POLICY_VERSION,
        "mode": "planned",
        "model": {
            "label": label,
            "params_b": params_b,
            "active_b": active_b,
            "parent_revision_sha256": "required",
            "config_sha256": "required",
            "tokenizer_sha256": "required",
        },
        "target": {
            "physical_bpw_ceiling": physical_bpw,
            "resident_bytes_ceiling": 64_000_000_000,
            "process_peak_bytes_ceiling": 78_000_000_000,
            "latency_slo_ms": None,
        },
        "runtime_identity": {
            "backend_kernel_sha256": "required",
            "drafter_artifact_sha256": "required",
            "verifier_path_sha256": "required",
            "kv_policy_sha256": "required",
            "cache_namespace_sha256": "required",
            "adaptive_policy_sha256": "required",
        },
        "operators": operators,
        "evidence_contract": {
            "proof_states": list(PROOF_STATES),
            "selection_set_independent": True,
            "negative_results_retained": True,
            "minimum_seeds": 3,
            "multiwindow_minimum": 4,
            "capability_tripwire_required": True,
            "same_box_efficiency_required": True,
        },
        "rate_contract": {
            "actual_file_bytes_authoritative": True,
            "bill": [
                "base", "pass_through", "scales", "codebooks", "indices", "routers",
                "corrections", "state", "metadata", "alignment",
            ],
            "dynamic_bill": ["mean", "p95", "worst"],
        },
    }
    return stamp_program(program)


def _sample_node(node_id: str, kind: str, deps: list[str], *, dynamic: bool = False) -> dict[str, Any]:
    return {
        "id": node_id,
        "kind": kind,
        "phase": "decode" if dynamic else "offline",
        "mechanism": node_id,
        "mechanism_version": "planned.v1",
        "implementation_status": "research" if dynamic else "oracle",
        "depends_on": deps,
        "parameters": {},
        "backend_support": {
            "apple_cpu": "research", "metal": "gated", "cuda": "research",
            "distributed": "research", "future_specialized": "research",
        },
        "cost_contract": {
            "actual_bytes_authoritative": True,
            "dynamic_required_metrics": ["mean", "p95", "worst"] if dynamic else [],
        },
        "executor": {"wired": False, "argv": [], "source_sha256": None},
    }


def selftest() -> int:
    program = make_planned_program(
        label="selftest-1.6T", params_b=1600.0, active_b=49.0, physical_bpw=0.25,
        operators=[
            _sample_node("binary-base", "base_codec", []),
            _sample_node("token-gate", "gated_correction", ["binary-base"], dynamic=True),
            _sample_node("verifier", "verifier", ["token-gate"], dynamic=True),
        ],
    )
    assert validate_program(program) == []
    digest = "a" * 64
    cell = stamp_cell({
        "schema": CELL_SCHEMA,
        "program_sha256": program["program_sha256"],
        "backend": "apple_cpu", "proof_state": "planned", "fidelity": "F0", "seed": 17,
        "calibration_sha256": digest, "selection_sha256": digest,
        "final_eval_sha256": digest, "worker_source_sha256": digest,
        "resource_estimate": {
            "peak_memory_bytes": 1, "scratch_bytes": 1,
            "source_read_bytes": 1, "estimated_seconds": 1,
        },
        "checkpoint_contract": {"required_state": sorted({
            "operator_state", "optimizer", "microstep", "gradient_accumulation", "rng",
            "sampler_cursor", "teacher_cache_identity", "source_shard_offset",
            "partial_output_hashes", "resume_command_identity",
        })},
    })
    assert validate_cell(cell, program) == []
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA, "cell_sha256": cell["cell_sha256"],
        "operator_state_sha256": digest, "optimizer_sha256": digest,
        "rng_sha256": digest, "sampler_sha256": digest, "teacher_cache_sha256": digest,
        "partial_outputs_sha256": digest, "resume_command_sha256": digest,
        "microstep": 0, "gradient_accumulation_phase": 0,
        "source_shard_index": 0, "source_byte_offset": 0, "fsync_complete": True,
    }
    assert validate_checkpoint(checkpoint, cell) == []
    observation = {
        "schema": OBSERVATION_SCHEMA, "cell_sha256": cell["cell_sha256"],
        "status": "complete_negative", "proof_state": "reconstruction_oracle",
        "quality": {"capability_vector": {"selftest": 0.0}, "uncertainty": 0.0},
        "costs": {
            "physical_model_bytes": 1, "resident_peak_bytes": 1,
            "mean_bytes_per_token": 1, "p95_bytes_per_token": 1,
            "worst_bytes_per_token": 1, "joules_per_accepted_token": 0,
            "p50_latency_ms": 0, "p95_latency_ms": 0,
            "accepted_tokens": 0, "rejected_tokens": 0,
        },
        "safety": {"memory_pressure": "normal", "swap_used_bytes": 0},
        "output_sha256s": [digest], "negative_result_retained": True,
    }
    observation["observation_sha256"] = hash_value(observation)
    assert validate_observation(observation, cell) == []
    changed = json.loads(json.dumps(program))
    changed["target"]["physical_bpw_ceiling"] = 0.33
    assert "program_sha256 does not match canonical identity" in validate_program(changed)
    cyclic = json.loads(json.dumps(program))
    cyclic["operators"][0]["depends_on"] = ["verifier"]
    cyclic = stamp_program(cyclic)
    assert any("cycle" in err for err in validate_program(cyclic))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "program.json"
        atomic_json(path, program)
        assert validate_program(json.loads(path.read_text())) == []
    print("healer_abi.py selftest OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("selftest")
    validate = sub.add_parser("validate")
    validate.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.command == "selftest":
        return selftest()
    doc = json.loads(args.path.read_text())
    errors = validate_program(doc)
    print(json.dumps({"ok": not errors, "errors": errors}, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
