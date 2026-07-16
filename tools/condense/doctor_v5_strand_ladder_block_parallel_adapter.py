#!/usr/bin/env python3.12
"""Pending-only control adapter for the block-parallel Qwen encoder."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import doctor_v5_accel_loader as _accel
import doctor_v5_source_seal as _source_seal


HERE = Path(__file__).resolve().parent
BASE_PATH = HERE / "doctor_v5_strand_ladder_adapter.py"
BASE_SHA256 = "cf3c236a90eeae89a576e1da60951206a19fce29e842528df92621d57934d332"
WORKER_PATH = HERE / "doctor_v5_strand_ladder_block_parallel_worker.py"
QUANTIZER = HERE.parents[1] / "build/strand-block-parallel/release/quantize-model-block-parallel"
ADAPTER_VERSION = "2-block-parallel"
TRANSITION_SCHEMA = "hawking.doctor_v5_inner_resume_transition.v1"
TRANSITION_NAME = "resume_transition.json"
TRANSITION_SCOPE = frozenset({("14B", "3"), ("14B", "4")})

_BASE = _accel.load_frozen("doctor_v5_strand_ladder_adapter_frozen", BASE_PATH,
                           BASE_SHA256)
_BASE.__file__ = str(Path(__file__).resolve())
_BASE.WORKER_PATH = WORKER_PATH
_BASE.QUANTIZER = QUANTIZER
_BASE.ADAPTER_VERSION = ADAPTER_VERSION
_ORIGINAL_BUILD_SPEC = _BASE.build_spec
_ORIGINAL_LOAD_MODULE = _BASE._load_module
_ORIGINAL_BUILD_INTERNAL = _BASE._build_internal
_ORIGINAL_OUTER_CHECKPOINT = _BASE._outer_checkpoint
_ORIGINAL_RETAINED_ARTIFACTS = _BASE._retained_artifacts
_LOADED_ABI: Any | None = None
_LOADED_WORKER: Any | None = None


def _load_module(name: str, path: Path) -> Any:
    """Install seal reuse before the outer ABI hashes request inputs."""
    global _LOADED_ABI, _LOADED_WORKER
    module = _ORIGINAL_LOAD_MODULE(name, path)
    resolved = Path(path).resolve(strict=False)
    if resolved == _BASE.ABI_PATH.resolve(strict=False):
        _source_seal.install_hash_reuse(module, attribute="_sha_file")
        _LOADED_ABI = module
    elif resolved == WORKER_PATH.resolve(strict=False):
        _LOADED_WORKER = module
    return module


_BASE._load_module = _load_module


def _transition_path(internal_root: Path) -> Path:
    return internal_root / TRANSITION_NAME


def _transition_artifact(internal_root: Path,
                         outer_sha256: str | None = None) -> dict[str, Any] | None:
    path = _transition_path(internal_root)
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise _BASE.AdapterError(f"resume transition artifact is invalid: {path}")
    document = _BASE._load_json(path)
    required = {
        "schema", "created_at", "scope", "new_outer_request_sha256",
        "prior_outer_request_prefix", "program_semantics_sha256",
        "candidate_inner_canonical_sha256", "preserved_inner_request",
        "verified_checkpoint", "verification", "quality_claims_permitted",
        "source_deletion_permitted", "transition_sha256",
    }
    scope = document.get("scope")
    preserved = document.get("preserved_inner_request")
    checkpoint = document.get("verified_checkpoint")
    expected_verification = {
        "only_request_id_changed": True,
        "old_inner_request_fully_validated": True,
        "checkpoint_request_plan_prefix_validated": True,
        "all_recorded_completed_artifacts_validated": True,
        "checkpoint_identity_weakened": False,
    }
    expected_inner_path = (internal_root / "request.json").resolve(strict=False)
    expected_checkpoint_path = (internal_root / "checkpoint.json").resolve(strict=False)
    if set(document) != required or document.get("schema") != TRANSITION_SCHEMA \
            or not isinstance(document.get("created_at"), str) \
            or not isinstance(scope, dict) or set(scope) != {"label", "rate_id"} \
            or (scope.get("label"), scope.get("rate_id")) not in TRANSITION_SCOPE \
            or not _source_seal._is_sha(document.get("new_outer_request_sha256")) \
            or not isinstance(document.get("prior_outer_request_prefix"), str) \
            or len(document["prior_outer_request_prefix"]) != 16 \
            or any(character not in "0123456789abcdef"
                   for character in document["prior_outer_request_prefix"]) \
            or not _source_seal._is_sha(document.get("program_semantics_sha256")) \
            or not _source_seal._is_sha(document.get("candidate_inner_canonical_sha256")) \
            or not isinstance(preserved, dict) or set(preserved) != {
                "path", "sha256", "bytes", "request_id"} \
            or Path(preserved.get("path", "")).resolve(strict=False) != expected_inner_path \
            or not _source_seal._is_sha(preserved.get("sha256")) \
            or not _source_seal._is_int(preserved.get("bytes")) \
            or not isinstance(preserved.get("request_id"), str) \
            or not preserved["request_id"].endswith(document["prior_outer_request_prefix"]) \
            or not isinstance(checkpoint, dict) or set(checkpoint) != {
                "path", "sha256", "bytes", "request_sha256", "status",
                "completed_units", "completed_artifact_count",
                "completed_artifact_binding_sha256"} \
            or Path(checkpoint.get("path", "")).resolve(strict=False) \
                != expected_checkpoint_path \
            or not _source_seal._is_sha(checkpoint.get("sha256")) \
            or not _source_seal._is_int(checkpoint.get("bytes")) \
            or checkpoint.get("request_sha256") != preserved.get("sha256") \
            or checkpoint.get("status") not in {"running", "checkpointed-stop", "complete"} \
            or not isinstance(checkpoint.get("completed_units"), list) \
            or not checkpoint["completed_units"] \
            or len(checkpoint["completed_units"]) != len(set(checkpoint["completed_units"])) \
            or not _source_seal._is_int(checkpoint.get("completed_artifact_count")) \
            or not _source_seal._is_sha(
                checkpoint.get("completed_artifact_binding_sha256")
            ) or document.get("verification") != expected_verification \
            or document.get("quality_claims_permitted") is not False \
            or document.get("source_deletion_permitted") is not False \
            or document.get("transition_sha256") != _source_seal._hash_value(
                _source_seal._without(document, "transition_sha256")
            ) or (outer_sha256 is not None
                  and document.get("new_outer_request_sha256") != outer_sha256):
        raise _BASE.AdapterError("resume transition receipt identity is invalid")
    inner_sha, inner_size = _BASE._hash_file(expected_inner_path)
    inner = _BASE._load_json(expected_inner_path)
    if inner_sha != preserved["sha256"] or inner_size != preserved["bytes"] \
            or inner.get("request_id") != preserved["request_id"] \
            or _source_seal._hash_value(
                {name: value for name, value in inner.items() if name != "request_id"}
            ) != document["program_semantics_sha256"]:
        raise _BASE.AdapterError("preserved inner request differs from transition receipt")
    current_checkpoint = _BASE._load_json(expected_checkpoint_path)
    current_units = current_checkpoint.get("completed_units")
    anchored_units = checkpoint["completed_units"]
    if current_checkpoint.get("request_sha256") != preserved["sha256"] \
            or not isinstance(current_units, list) \
            or current_units[:len(anchored_units)] != anchored_units \
            or not isinstance(current_checkpoint.get("units"), dict):
        raise _BASE.AdapterError("current checkpoint regressed from transition anchor")
    anchored_evidence = {
        "units": {unit: current_checkpoint["units"].get(unit) for unit in anchored_units}
    }
    references = sorted(
        _BASE._checkpoint_artifact_rows(anchored_evidence),
        key=lambda row: (row["path"], row["sha256"], row["bytes"]),
    )
    if len(references) != checkpoint["completed_artifact_count"] \
            or _source_seal._hash_value(references) \
                != checkpoint["completed_artifact_binding_sha256"]:
        raise _BASE.AdapterError("checkpoint transition-anchor evidence changed")
    digest, size = _BASE._hash_file(path)
    return {"role": "resume_transition", "path": str(path.resolve()),
            "sha256": digest, "bytes": size}


def _build_internal(request: dict[str, Any], spec: dict[str, Any],
                    parameter: dict[str, Any], internal_root: Path) -> dict[str, Any]:
    """Preserve a verified 14B 3/4-bpw inner request across seal-only rebinding."""
    candidate = _ORIGINAL_BUILD_INTERNAL(request, spec, parameter, internal_root)
    inner_path = internal_root / "request.json"
    if not inner_path.exists():
        return candidate
    if inner_path.is_symlink() or not inner_path.is_file():
        raise _BASE.AdapterError("existing inner request is not a regular file")
    existing = _BASE._load_json(inner_path)
    if existing == candidate:
        return existing
    scope = (candidate.get("label"), candidate.get("codec", {}).get("rate_id"))
    branch = candidate.get("campaign_binding", {}).get("branch")
    if scope not in TRANSITION_SCOPE or branch != "codec_control":
        raise _BASE.AdapterError("inner request differs outside the reviewed transition scope")
    if {name: value for name, value in existing.items() if name != "request_id"} != \
            {name: value for name, value in candidate.items() if name != "request_id"}:
        raise _BASE.AdapterError(
            "inner request differs beyond the sole permitted request_id transition"
        )
    old_request_id = existing.get("request_id")
    new_request_id = candidate.get("request_id")
    outer_sha = request.get("request_sha256")
    if not isinstance(old_request_id, str) or not isinstance(new_request_id, str) \
            or not isinstance(outer_sha, str) or len(outer_sha) != 64 \
            or not new_request_id.endswith(outer_sha[:16]) \
            or old_request_id == new_request_id:
        raise _BASE.AdapterError("inner request_id transition binding is invalid")
    prior_prefix = old_request_id.rsplit("-", 1)[-1]
    if len(prior_prefix) != 16 or any(c not in "0123456789abcdef" for c in prior_prefix):
        raise _BASE.AdapterError("prior outer request prefix is invalid")
    if _LOADED_WORKER is None or _LOADED_ABI is None:
        raise _BASE.AdapterError("transition validators were not loaded in execution order")

    # The worker validator proves every old request binding.  The checkpoint
    # validator then checks the exact request SHA, plan prefix, and every live
    # artifact recorded by completed units before the old request is retained.
    old_request, old_sha, shards = _LOADED_WORKER._validate_request(inner_path)
    if old_request != existing:
        raise _BASE.AdapterError("worker parsed inner request differs from preserved JSON")
    checkpoint_path = internal_root / "checkpoint.json"
    if not checkpoint_path.is_file() or checkpoint_path.is_symlink():
        raise _BASE.AdapterError("verified checkpoint is required for inner transition")
    stats = [
        _LOADED_WORKER._tensor_stats(row["path"], existing["codec"]["tensor_scope"])
        for row in shards
    ]
    plan = _LOADED_WORKER._plan(existing, stats)
    paths = _LOADED_WORKER._paths(internal_root, len(shards))
    checkpoint = _LOADED_WORKER._checkpoint(
        checkpoint_path, old_sha, plan, paths, stats
    )
    completed = checkpoint.get("completed_units")
    if not isinstance(completed, list) or not completed:
        raise _BASE.AdapterError("inner transition requires nonempty durable progress")
    checkpoint_sha, checkpoint_size = _BASE._hash_file(checkpoint_path)
    references = sorted(
        _BASE._checkpoint_artifact_rows(checkpoint),
        key=lambda row: (row["path"], row["sha256"], row["bytes"]),
    )
    inner_sha, inner_size = _BASE._hash_file(inner_path)
    if inner_sha != old_sha or checkpoint.get("request_sha256") != old_sha:
        raise _BASE.AdapterError("inner request/checkpoint SHA binding changed")
    transition_path = _transition_path(internal_root)
    if transition_path.exists():
        # A retry preserves the original transition anchor.  The current
        # checkpoint was fully revalidated above and may be a strict extension
        # of that anchor after additional durable units completed.
        _transition_artifact(internal_root, outer_sha)
        recorded = _BASE._load_json(transition_path)
        if recorded.get("candidate_inner_canonical_sha256") \
                != _source_seal._hash_value(candidate) \
                or recorded.get("program_semantics_sha256") \
                != _source_seal._hash_value(
                    {name: value for name, value in candidate.items()
                     if name != "request_id"}
                ):
            raise _BASE.AdapterError("resume transition candidate differs on retry")
        return existing
    receipt = {
        "schema": TRANSITION_SCHEMA,
        "created_at": _source_seal._now(),
        "scope": {"label": scope[0], "rate_id": scope[1]},
        "new_outer_request_sha256": outer_sha,
        "prior_outer_request_prefix": prior_prefix,
        "program_semantics_sha256": _source_seal._hash_value(
            {name: value for name, value in candidate.items() if name != "request_id"}
        ),
        "candidate_inner_canonical_sha256": _source_seal._hash_value(candidate),
        "preserved_inner_request": {
            "path": str(inner_path.resolve()), "sha256": inner_sha,
            "bytes": inner_size, "request_id": old_request_id,
        },
        "verified_checkpoint": {
            "path": str(checkpoint_path.resolve()), "sha256": checkpoint_sha,
            "bytes": checkpoint_size, "request_sha256": old_sha,
            "status": checkpoint.get("status"),
            "completed_units": list(completed),
            "completed_artifact_count": len(references),
            "completed_artifact_binding_sha256": _source_seal._hash_value(references),
        },
        "verification": {
            "only_request_id_changed": True,
            "old_inner_request_fully_validated": True,
            "checkpoint_request_plan_prefix_validated": True,
            "all_recorded_completed_artifacts_validated": True,
            "checkpoint_identity_weakened": False,
        },
        "quality_claims_permitted": False,
        "source_deletion_permitted": False,
    }
    receipt["transition_sha256"] = _source_seal._hash_value(receipt)
    _LOADED_ABI.atomic_json(transition_path, receipt)
    _transition_artifact(internal_root, outer_sha)
    return existing


def _outer_checkpoint(abi: Any, request: dict[str, Any], registry: dict[str, Any],
                      internal_root: Path, status: str) -> dict[str, Any]:
    document = _ORIGINAL_OUTER_CHECKPOINT(
        abi, request, registry, internal_root, status
    )
    transition = _transition_artifact(internal_root, request.get("request_sha256"))
    if transition is None:
        return document
    document["artifact_hashes"] = sorted(
        [row for row in document["artifact_hashes"] if row.get("role") != transition["role"]]
        + [transition],
        key=lambda row: row["role"],
    )
    document["checkpoint_sha256"] = abi._hash_value(
        abi._without(document, "checkpoint_sha256")
    )
    errors = abi.validate_checkpoint(document, request, registry)
    if errors:
        raise _BASE.AdapterError(
            "transition-bound outer checkpoint is invalid: " + "; ".join(errors)
        )
    return document


def _retained_artifacts(internal_root: Path) -> list[dict[str, Any]]:
    rows = _ORIGINAL_RETAINED_ARTIFACTS(internal_root)
    transition = _transition_artifact(internal_root)
    if transition is not None:
        rows = [row for row in rows if row.get("role") != transition["role"]]
        rows.append(transition)
        rows.sort(key=lambda row: row["role"])
    return rows


_BASE._build_internal = _build_internal
_BASE._outer_checkpoint = _outer_checkpoint
_BASE._retained_artifacts = _retained_artifacts


def build_spec(**kwargs: Any) -> dict[str, Any]:
    seal_path = _source_seal.default_path(kwargs["label"])
    if not seal_path.is_file() or seal_path.is_symlink():
        raise _accel.AccelerationBindingError(
            f"hash-bound source seal is required before accelerated wiring: {seal_path}"
        )
    document = _ORIGINAL_BUILD_SPEC(**kwargs)
    document = _accel.bind_extra_inputs(document, (
        _accel.input_row("acceleration_loader", Path(_accel.__file__)),
        _accel.input_row("source_seal_module", Path(_source_seal.__file__)),
        _accel.input_row("source_seal", seal_path),
        _accel.input_row("frozen_adapter_base", BASE_PATH),
        _accel.input_row("frozen_worker_base", HERE / "doctor_v5_strand_ladder_worker.py"),
    ))
    abi = _BASE._load_module("doctor_v5_block_parallel_spec_writer", _BASE.ABI_PATH)
    abi.atomic_json(_BASE._workspace_path(str(kwargs["output_path"]), must_exist=False),
                    document)
    return document


_BASE.build_spec = build_spec
_accel.export_module(_BASE, globals(), keep={"build_spec", "WORKER_PATH", "QUANTIZER",
                                             "ADAPTER_VERSION", "_load_module",
                                             "_build_internal", "_outer_checkpoint",
                                             "_retained_artifacts"})


if __name__ == "__main__":
    raise SystemExit(_BASE.main())
