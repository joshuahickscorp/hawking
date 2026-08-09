"""Fail-closed source-token-only outer launcher for Qwen80 L0 all-ten MoE.

This successor is intentionally incompatible with the historical fixture-router
launcher.  It has two explicit modes:

* ``--preflight-only`` creates a sealed source-token outer preflight, runs the
  distinct child in its CPU-only ``preflight`` mode, and seals the combined
  proof.  It cannot accept a lease or a capture directory for a Metal child.
* ``--execute-one-shot`` is a future, explicit Metal path.  It consumes that
  exact combined proof plus a fresh source-token lease, starts exactly one
  child, reaps it, and writes its terminal receipt last.

Neither mode accepts a legacy router receipt, router outer receipt, fixture
route plan, or legacy fixed-ABI argument.  This module never turns a component
receipt into a layer, token, decoder, generation, HCLI, TPS, TG, or tournament
claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.operators import ascension_qwen80_source_token_all_ten_true_moe_outer_preflight as outer_preflight
from lab.operators import ascension_qwen80_source_token_l0_route_plan as route_plan
from lab.receipts import seal, verify


SCHEMA = "hawking.ascension.qwen80_source_token_true_input_all_ten_moe_graph_outer_launcher.v1"
PREFLIGHT_PROOF_SCHEMA = "hawking.ascension.qwen80_source_token_true_input_all_ten_moe_graph_preflight_proof.v1"
PREFLIGHT_PROOF_STATUS = (
    "PREFLIGHTED_CURRENT_ADMITTED_QWEN80_SOURCE_TOKEN_TRUE_INPUT_ALL_TEN_MOE_"
    "OUTER_AND_CHILD_CPU_ONLY_NOT_LEASED_OR_EXECUTED"
)

ACTIVE_FILENAME = "active.json"
CHILD_FILENAME = "child.json"
TERMINAL_FILENAME = "outer-terminal-receipt.json"
OUTER_STDOUT = "outer.stdout.log"
OUTER_STDERR = "outer.stderr.log"
INNER_CAPTURE = "inner"
OUTER_LAUNCH_AUTHORITY_FILENAME = "outer-launch-authority.json"

OUTER_LAUNCH_AUTHORITY_SCHEMA = "hawking.ascension.qwen80_source_token_true_input_all_ten_moe_graph_outer_launch_authority.v1"
OUTER_LAUNCH_AUTHORITY_STATUS = (
    "AUTHORIZED_QWEN80_SOURCE_TOKEN_TRUE_INPUT_ALL_TEN_MOE_OUTER_REAPED_ONE_SHOT_METAL_CHILD"
)

PREFLIGHT_OUTER_FILENAME = "outer-preflight.json"
PREFLIGHT_CHILD_STDOUT = "child-preflight.stdout.json"
PREFLIGHT_CHILD_STDERR = "child-preflight.stderr.log"
PREFLIGHT_PROOF_FILENAME = "preflight-proof.json"

EXPECTED_PROBE_BASENAME = "ascension_qwen80_source_token_all_ten_true_moe_graph_device"
EXPECTED_CHILD_SCHEMA = "hawking.ascension.qwen80_source_token_all_ten_true_moe_graph_device.v1"
EXPECTED_CHILD_PREFLIGHT_STATUS = "PREPARED_QWEN80_SOURCE_TOKEN_ALL_TEN_TRUE_MOE_DEVICE_CHILD_NOT_LEASED_OR_EXECUTED"
EXPECTED_CHILD_METAL_STATUS = (
    "EARNED_CURRENT_ADMITTED_QWEN80_SOURCE_TOKEN_LAYER0_TRUE_INPUT_ALL_TEN_ROUTE_"
    "SHARED_SECOND_RESIDUAL_STRICT_MATH_METAL_COMPONENT_NOT_COMPLETE_LAYER_OR_TOKEN"
)

LEASE_SCHEMA = "hawking.ascension.qwen80_source_token_true_input_all_ten_moe_graph_quiet_metal_lease.v1"
LEASE_STATUS = "GRANTED_QWEN80_SOURCE_TOKEN_TRUE_INPUT_ALL_TEN_MOE_GRAPH_NON_TIMED_DEVICE_PARITY_LEASE"
LEASE_COMPONENT = "qwen80_source_token_true_input_all_ten_moe_graph"

PREFIX_DISPATCHES = 9
SUFFIX_DISPATCHES = 14
TOTAL_DISPATCHES = PREFIX_DISPATCHES + SUFFIX_DISPATCHES
SOURCE_TOKEN_ID = 1
TOP_K = 10
MAX_PREFLIGHT_STREAM_BYTES = 1_000_000
MAX_PREFLIGHT_SECONDS = 120.0

SHADER_SOURCE = REPO_ROOT / "crates/hawking-core/shaders/qwen80_all_ten_routed_expert_wave.metal"
METAL_REGISTRY_SOURCE = REPO_ROOT / "crates/hawking-core/src/metal/mod.rs"
METAL_REGISTRY_INCLUDE = 'include_str!("../../shaders/qwen80_all_ten_routed_expert_wave.metal")'

LEGACY_ARGUMENTS = frozenset(
    {
        "--router-receipt",
        "--router-outer-receipt",
        "--route-plan",
        "--fixed-abi-contract",
    }
)


class SourceTokenTrueInputAllTenMoeLauncherError(RuntimeError):
    """The source-token component capture lacks an exact antecedent."""


@dataclass(frozen=True)
class BaseInputs:
    manifest: Path
    admission_current: Path
    source_token_route_authority: Path
    first_residual_receipt: Path
    typed_bridge_receipt: Path
    fixed_suffix_contract: Path


@dataclass(frozen=True)
class PreflightContext:
    outer_preflight: dict[str, Any]
    outer_preflight_evidence: dict[str, Any]
    outer_preflight_seal_sha256: str
    manifest: dict[str, Any]
    admission_current: dict[str, Any]
    source_authority: dict[str, Any]
    first_residual: dict[str, Any]
    typed_bridge: dict[str, Any]
    fixed_suffix: dict[str, Any]
    manifest_seal_sha256: str
    admission_pointer_seal_sha256: str
    admission_receipt_seal_sha256: str
    source_authority_seal_sha256: str
    first_residual_seal_sha256: str
    first_residual_output_sha256: str
    typed_bridge_seal_sha256: str
    route_ids: tuple[int, ...]
    route_weights: tuple[float, ...]
    probe_binary: dict[str, Any]
    shader_source: dict[str, Any]
    metal_registry: dict[str, Any]


@dataclass(frozen=True)
class ProofContext:
    preflight: PreflightContext
    proof: dict[str, Any]
    proof_evidence: dict[str, Any]
    proof_seal_sha256: str


@dataclass(frozen=True)
class LaunchConfig:
    base: BaseInputs
    probe_bin: Path
    preflight_proof: Path
    lease_receipt: Path
    capture_dir: Path
    workers: int
    timeout_seconds: float


@dataclass(frozen=True)
class LaunchContext:
    proof: ProofContext
    lease_receipt: dict[str, Any]
    lease_seal_sha256: str
    lease_id: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_regular(path: Path, label: str, *, executable: bool = False) -> Path:
    try:
        return route_plan._canonical_regular(path, label, executable=executable)
    except route_plan.SourceTokenRoutePlanError as exc:
        raise SourceTokenTrueInputAllTenMoeLauncherError(str(exc)) from exc


def _file_evidence(path: Path, label: str, *, executable: bool = False) -> dict[str, Any]:
    try:
        return route_plan._file_evidence(path, label, executable=executable)
    except route_plan.SourceTokenRoutePlanError as exc:
        raise SourceTokenTrueInputAllTenMoeLauncherError(str(exc)) from exc


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        return route_plan._read_json(path, label)
    except route_plan.SourceTokenRoutePlanError as exc:
        raise SourceTokenTrueInputAllTenMoeLauncherError(str(exc)) from exc


def _sealed_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    try:
        return route_plan._sealed_json(path, label)
    except route_plan.SourceTokenRoutePlanError as exc:
        raise SourceTokenTrueInputAllTenMoeLauncherError(str(exc)) from exc


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SourceTokenTrueInputAllTenMoeLauncherError(f"{label} must be an object")
    return dict(value)


def _require_sha(value: object, label: str) -> str:
    try:
        return route_plan._require_sha256(value, label)
    except route_plan.SourceTokenRoutePlanError as exc:
        raise SourceTokenTrueInputAllTenMoeLauncherError(str(exc)) from exc


def _evidence_matches(value: object, expected: Mapping[str, Any], label: str) -> None:
    try:
        route_plan._evidence_matches(value, expected, label)
    except route_plan.SourceTokenRoutePlanError as exc:
        raise SourceTokenTrueInputAllTenMoeLauncherError(str(exc)) from exc


def _historical_pointer_matches(value: object, current: Mapping[str, Any], label: str) -> None:
    """Allow a current-pointer reseal while retaining its canonical location.

    The admitted manifest and immutable admission receipt remain exact.  The
    versioned current pointer records timestamps/housekeeping and is allowed to
    have been resealed after a predecessor proof was emitted.
    """
    historical = _mapping(value, label)
    if historical.get("present") is not True or historical.get("path") != current.get("path"):
        raise SourceTokenTrueInputAllTenMoeLauncherError(f"{label} path/presence drifted")
    if not isinstance(historical.get("bytes"), int) or historical["bytes"] <= 0:
        raise SourceTokenTrueInputAllTenMoeLauncherError(f"{label} historical byte count is invalid")
    _require_sha(historical.get("sha256"), f"{label} historical SHA")


def _write_new(path: Path, document: Mapping[str, Any]) -> None:
    try:
        route_plan.write_new(path, document)
    except route_plan.SourceTokenRoutePlanError as exc:
        raise SourceTokenTrueInputAllTenMoeLauncherError(str(exc)) from exc


def _write_raw_new(path: Path, payload: bytes) -> None:
    """Durably create a bounded child stream without replacing prior evidence."""
    if path.exists():
        raise SourceTokenTrueInputAllTenMoeLauncherError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as exc:
        raise SourceTokenTrueInputAllTenMoeLauncherError(f"refusing to overwrite {path}") from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _new_capture_dir(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise SourceTokenTrueInputAllTenMoeLauncherError(f"{label} must be absolute")
    if path.exists():
        raise SourceTokenTrueInputAllTenMoeLauncherError(f"{label} must be new; replay needs its terminal proof")
    parent = path.parent
    try:
        parent_metadata = parent.lstat()
    except OSError as exc:
        raise SourceTokenTrueInputAllTenMoeLauncherError(f"cannot stat {label} parent {parent}: {exc}") from exc
    if stat.S_ISLNK(parent_metadata.st_mode) or not stat.S_ISDIR(parent_metadata.st_mode):
        raise SourceTokenTrueInputAllTenMoeLauncherError(f"{label} parent must be an existing non-symlink directory")
    try:
        path.mkdir(mode=0o750)
    except OSError as exc:
        raise SourceTokenTrueInputAllTenMoeLauncherError(f"cannot create {label}: {exc}") from exc
    return path.resolve(strict=True)


def _validate_new_outer_capture_path(path: Path) -> Path:
    if not path.is_absolute():
        raise SourceTokenTrueInputAllTenMoeLauncherError("--capture-dir must be absolute")
    if path.exists():
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise SourceTokenTrueInputAllTenMoeLauncherError(f"cannot stat --capture-dir: {exc}") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise SourceTokenTrueInputAllTenMoeLauncherError(
                "existing --capture-dir must be a non-symlink directory"
            )
        return path.resolve(strict=True)
    parent = path.parent
    try:
        parent_metadata = parent.lstat()
    except OSError as exc:
        raise SourceTokenTrueInputAllTenMoeLauncherError(f"cannot stat --capture-dir parent: {exc}") from exc
    if stat.S_ISLNK(parent_metadata.st_mode) or not stat.S_ISDIR(parent_metadata.st_mode):
        raise SourceTokenTrueInputAllTenMoeLauncherError(
            "--capture-dir parent must be an existing non-symlink directory"
        )
    return path


def _route_ids(value: object, label: str) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) != TOP_K:
        raise SourceTokenTrueInputAllTenMoeLauncherError(f"{label} must contain exactly ten expert IDs")
    if any(isinstance(item, bool) or not isinstance(item, int) or not 0 <= item < 512 for item in value):
        raise SourceTokenTrueInputAllTenMoeLauncherError(f"{label} contains an invalid expert ID")
    if len(set(value)) != TOP_K:
        raise SourceTokenTrueInputAllTenMoeLauncherError(f"{label} contains duplicate experts")
    return tuple(value)


def _route_weights(value: object, label: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != TOP_K:
        raise SourceTokenTrueInputAllTenMoeLauncherError(f"{label} must contain exactly ten weights")
    try:
        weights = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise SourceTokenTrueInputAllTenMoeLauncherError(f"{label} contains a non-numeric weight") from exc
    if any(not math.isfinite(item) or item <= 0.0 for item in weights) or not math.isclose(
        sum(weights), 1.0, abs_tol=2.0e-5, rel_tol=0.0
    ):
        raise SourceTokenTrueInputAllTenMoeLauncherError(f"{label} is not a normalized positive route")
    return weights


def _same_route_weights(observed: object, expected: Sequence[float], label: str) -> None:
    weights = _route_weights(observed, label)
    if len(weights) != len(expected) or any(abs(left - right) > 1.0e-6 for left, right in zip(weights, expected)):
        raise SourceTokenTrueInputAllTenMoeLauncherError(f"{label} drifted")


def _assert_exact_mapping(value: object, expected: Mapping[str, Any], label: str) -> None:
    observed = _mapping(value, label)
    if observed != dict(expected):
        raise SourceTokenTrueInputAllTenMoeLauncherError(f"{label} drifted")


def _probe_evidence(path: Path) -> dict[str, Any]:
    probe = _canonical_regular(path, "--probe-bin", executable=True)
    if probe.name != EXPECTED_PROBE_BASENAME:
        raise SourceTokenTrueInputAllTenMoeLauncherError(
            f"--probe-bin must be {EXPECTED_PROBE_BASENAME}, got {probe.name!r}"
        )
    return _file_evidence(probe, "--probe-bin", executable=True)


def _implementation_evidence(probe_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    probe = _probe_evidence(probe_path)
    shader = _file_evidence(SHADER_SOURCE, "registered all-ten shader source")
    registry = _file_evidence(METAL_REGISTRY_SOURCE, "Metal shader registry")
    try:
        registry_text = METAL_REGISTRY_SOURCE.read_text(encoding="utf-8")
    except OSError as exc:
        raise SourceTokenTrueInputAllTenMoeLauncherError(f"cannot read Metal shader registry: {exc}") from exc
    if METAL_REGISTRY_INCLUDE not in registry_text:
        raise SourceTokenTrueInputAllTenMoeLauncherError(
            "registered all-ten shader source is absent from metal/mod.rs"
        )
    return probe, shader, {**registry, "registered": True, "registry_append_required": False}


def _build_outer_preflight(base: BaseInputs) -> dict[str, Any]:
    try:
        document = outer_preflight.build_preflight(
            manifest_path=base.manifest,
            admission_path=base.admission_current,
            source_authority_path=base.source_token_route_authority,
            first_residual_path=base.first_residual_receipt,
            typed_bridge_path=base.typed_bridge_receipt,
            fixed_suffix_path=base.fixed_suffix_contract,
        )
        verify(document, label="generated source-token outer preflight")
    except (outer_preflight.SourceTokenOuterPreflightError, ValueError) as exc:
        raise SourceTokenTrueInputAllTenMoeLauncherError(str(exc)) from exc
    if document.get("schema") != outer_preflight.SCHEMA or document.get("status") != outer_preflight.STATUS:
        raise SourceTokenTrueInputAllTenMoeLauncherError("generated source-token outer preflight schema/status drifted")
    return document


def _first_residual_output(path: Path, prefix_seal: str) -> str:
    document, observed_seal = _sealed_json(path, "--first-residual-receipt")
    if observed_seal != prefix_seal:
        raise SourceTokenTrueInputAllTenMoeLauncherError("first-residual receipt seal drifted")
    output = _mapping(document.get("first_residual_output"), "first-residual output")
    if (
        output.get("layer") != 0
        or output.get("linear_state_slot") != 0
        or output.get("elements") != 2_048
        or output.get("same_command_graph_required") is not True
    ):
        raise SourceTokenTrueInputAllTenMoeLauncherError("first-residual same-TCB geometry/state drifted")
    return _require_sha(output.get("sha256"), "first-residual output SHA")


def _context_from_outer_preflight(
    *,
    document: Mapping[str, Any],
    outer_evidence: Mapping[str, Any],
    outer_seal: str,
    base: BaseInputs,
    probe_path: Path,
) -> PreflightContext:
    try:
        verify(document, label="source-token outer preflight")
    except ValueError as exc:
        raise SourceTokenTrueInputAllTenMoeLauncherError(str(exc)) from exc
    if document.get("schema") != outer_preflight.SCHEMA or document.get("status") != outer_preflight.STATUS:
        raise SourceTokenTrueInputAllTenMoeLauncherError("source-token outer preflight schema/status drifted")
    source = _mapping(document.get("source_binding"), "source-token outer preflight source_binding")
    manifest = _file_evidence(base.manifest, "--manifest")
    admission = _file_evidence(base.admission_current, "--admission-current")
    authority = _file_evidence(base.source_token_route_authority, "--source-token-route-authority")
    prefix = _file_evidence(base.first_residual_receipt, "--first-residual-receipt")
    typed = _file_evidence(base.typed_bridge_receipt, "--typed-bridge-receipt")
    fixed = _file_evidence(base.fixed_suffix_contract, "--fixed-suffix-contract")
    for field, expected in (
        ("manifest", manifest),
        ("source_token_route_authority", authority),
        ("first_residual_receipt", prefix),
        ("typed_bridge_receipt", typed),
        ("fixed_suffix_contract", fixed),
    ):
        _evidence_matches(source.get(field), expected, f"source-token outer preflight {field}")
    _historical_pointer_matches(
        source.get("admission_current"), admission, "source-token outer preflight admission current"
    )
    manifest_seal = _require_sha(source.get("manifest_seal_sha256"), "outer preflight manifest seal")
    # A persisted outer preflight keeps the pointer seal it observed.  Read
    # today's pointer for any future inner receipt, while allowing harmless
    # pointer reseals that preserve the immutable selected admission chain.
    _require_sha(source.get("admission_pointer_seal_sha256"), "outer preflight historical admission pointer seal")
    current_admission, pointer_seal = _sealed_json(base.admission_current, "--admission-current")
    if (
        current_admission.get("schema") != route_plan.ADMISSION_SCHEMA
        or current_admission.get("status") != route_plan.ADMISSION_STATUS
    ):
        raise SourceTokenTrueInputAllTenMoeLauncherError("current admission pointer schema/status drifted")
    admission_receipt_seal = _require_sha(
        source.get("admission_receipt_seal_sha256"), "outer preflight admission receipt seal"
    )
    authority_seal = _require_sha(
        source.get("source_token_route_authority_seal_sha256"), "outer preflight source authority seal"
    )
    prefix_seal = _require_sha(
        source.get("first_residual_receipt_seal_sha256"), "outer preflight first-residual seal"
    )
    typed_seal = _require_sha(
        source.get("typed_bridge_receipt_seal_sha256"), "outer preflight typed bridge seal"
    )
    route = _mapping(document.get("source_token_route"), "source-token outer preflight route")
    if (
        route.get("layer") != 0
        or route.get("token_id") != SOURCE_TOKEN_ID
        or route.get("zero_l0_state_required") is not True
        or route.get("same_command_graph_required") is not True
        or route.get("all_ten_unique") is not True
    ):
        raise SourceTokenTrueInputAllTenMoeLauncherError("source-token outer preflight route geometry/lineage drifted")
    route_ids = _route_ids(route.get("route_ids"), "source-token outer preflight route IDs")
    route_weights = _route_weights(route.get("normalized_weights"), "source-token outer preflight route weights")
    next_child = _mapping(document.get("next_child_contract"), "source-token outer preflight child contract")
    if (
        next_child.get("legacy_router_receipt_or_fixture_plan_accepted") is not False
        or next_child.get("requires_source_token_authority_and_typed_bridge") is not True
        or next_child.get("requires_same_tcb_prefix_lineage") is not True
        or next_child.get("requires_fresh_component_only_quiet_lease") is not True
        or next_child.get("requires_outer_reaped_receipt_last_capture") is not True
    ):
        raise SourceTokenTrueInputAllTenMoeLauncherError("source-token outer preflight child contract drifted")
    boundary = _mapping(document.get("claim_boundary"), "source-token outer preflight claim boundary")
    if (
        boundary.get("artifact_scan_performed_by_preflight") is not False
        or boundary.get("metal_device_or_dispatch_performed") is not False
        or boundary.get("lease_issued") is not False
        or boundary.get("watcher_server_registry_or_hcli_changed") is not False
    ):
        raise SourceTokenTrueInputAllTenMoeLauncherError("source-token outer preflight was promoted beyond CPU authority")
    probe, shader, registry = _implementation_evidence(probe_path)
    return PreflightContext(
        outer_preflight=dict(document),
        outer_preflight_evidence=dict(outer_evidence),
        outer_preflight_seal_sha256=outer_seal,
        manifest=manifest,
        admission_current=admission,
        source_authority=authority,
        first_residual=prefix,
        typed_bridge=typed,
        fixed_suffix=fixed,
        manifest_seal_sha256=manifest_seal,
        admission_pointer_seal_sha256=pointer_seal,
        admission_receipt_seal_sha256=admission_receipt_seal,
        source_authority_seal_sha256=authority_seal,
        first_residual_seal_sha256=prefix_seal,
        first_residual_output_sha256=_first_residual_output(base.first_residual_receipt, prefix_seal),
        typed_bridge_seal_sha256=typed_seal,
        route_ids=route_ids,
        route_weights=route_weights,
        probe_binary=probe,
        shader_source=shader,
        metal_registry=registry,
    )


def _same_source_binding(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    fields = (
        "manifest",
        "admission_receipt",
        "source_token_route_authority",
        "first_residual_receipt",
        "typed_bridge_receipt",
        "fixed_suffix_contract",
        "manifest_seal_sha256",
        "admission_receipt_seal_sha256",
        "source_token_route_authority_seal_sha256",
        "first_residual_receipt_seal_sha256",
        "typed_bridge_receipt_seal_sha256",
    )
    if not all(left.get(field) == right.get(field) for field in fields):
        return False
    try:
        left_pointer = _mapping(left.get("admission_current"), "left admission pointer")
        right_pointer = _mapping(right.get("admission_current"), "right admission pointer")
        if (
            left_pointer.get("present") is not True
            or right_pointer.get("present") is not True
            or left_pointer.get("path") != right_pointer.get("path")
        ):
            return False
        _require_sha(left_pointer.get("sha256"), "left admission pointer SHA")
        _require_sha(right_pointer.get("sha256"), "right admission pointer SHA")
        _require_sha(left.get("admission_pointer_seal_sha256"), "left admission pointer seal")
        _require_sha(right.get("admission_pointer_seal_sha256"), "right admission pointer seal")
        return True
    except SourceTokenTrueInputAllTenMoeLauncherError:
        return False


def _binding_with_seal(evidence: Mapping[str, Any], seal_sha256: str) -> dict[str, Any]:
    return {**dict(evidence), "seal_sha256": seal_sha256}


def _child_sealed_binding(evidence: Mapping[str, Any], seal_sha256: str) -> dict[str, Any]:
    """The Rust child deliberately records only path/raw digest/seal triples."""
    return {
        "path": evidence["path"],
        "document_sha256": evidence["sha256"],
        "seal_sha256": seal_sha256,
    }


def _child_fixed_binding(evidence: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": evidence["path"],
        "document_sha256": evidence["sha256"],
        "schema": outer_preflight.FIXED_ABI_SCHEMA,
        "status": outer_preflight.FIXED_ABI_STATUS,
    }


def _validate_child_preflight_document(document: Mapping[str, Any], context: PreflightContext) -> None:
    if (
        document.get("schema") != EXPECTED_CHILD_SCHEMA
        or document.get("status") != EXPECTED_CHILD_PREFLIGHT_STATUS
        or document.get("mode") != "preflight"
    ):
        raise SourceTokenTrueInputAllTenMoeLauncherError("source-token child CPU preflight schema/status/mode drifted")
    _assert_exact_mapping(
        document.get("outer_preflight_binding"),
        _child_sealed_binding(context.outer_preflight_evidence, context.outer_preflight_seal_sha256),
        "source-token child outer preflight binding",
    )
    route_binding = _mapping(document.get("source_token_route_authority_binding"), "source-token child route binding")
    expected_route_binding = _child_sealed_binding(context.source_authority, context.source_authority_seal_sha256)
    for field, expected in expected_route_binding.items():
        if route_binding.get(field) != expected:
            raise SourceTokenTrueInputAllTenMoeLauncherError("source-token child route authority identity drifted")
    if tuple(route_binding.get("route_ids", [])) != context.route_ids:
        raise SourceTokenTrueInputAllTenMoeLauncherError("source-token child route IDs drifted")
    _same_route_weights(route_binding.get("normalized_weights"), context.route_weights, "source-token child route weights")
    _assert_exact_mapping(
        document.get("typed_bridge_binding"),
        _child_sealed_binding(context.typed_bridge, context.typed_bridge_seal_sha256),
        "source-token child typed bridge binding",
    )
    expected_prefix = _child_sealed_binding(context.first_residual, context.first_residual_seal_sha256)
    expected_prefix["output_sha256"] = context.first_residual_output_sha256
    _assert_exact_mapping(
        document.get("first_residual_antecedent"), expected_prefix, "source-token child first-residual antecedent"
    )
    expected_fixed = _child_fixed_binding(context.fixed_suffix)
    _assert_exact_mapping(
        document.get("fixed_suffix_contract_binding"), expected_fixed, "source-token child fixed suffix binding"
    )
    command_graph = _mapping(document.get("same_command_graph_contract"), "source-token child command graph")
    if (
        command_graph.get("source_token_id") != SOURCE_TOKEN_ID
        or command_graph.get("zero_l0_state_required") is not True
        or command_graph.get("prefix_dispatches") != PREFIX_DISPATCHES
        or command_graph.get("suffix_dispatches") != SUFFIX_DISPATCHES
        or command_graph.get("total_dispatches") != TOTAL_DISPATCHES
        or command_graph.get("route_guard_required") is not True
        or command_graph.get("all_ten_route_shared_routed_sum_second_residual_readbacks_required") is not True
    ):
        raise SourceTokenTrueInputAllTenMoeLauncherError("source-token child 9+14 same-TCB contract drifted")
    boundary = _mapping(document.get("claim_boundary"), "source-token child preflight claim boundary")
    if (
        boundary.get("metal_device_or_dispatch_performed") is not False
        or boundary.get("lease_issued") is not False
        or boundary.get("legacy_fixture_router_or_plan_accepted") is not False
        or boundary.get("no_complete_layer_token_decoder_generation_hcli_tps_tg_or_tournament_claim") is not True
    ):
        raise SourceTokenTrueInputAllTenMoeLauncherError("source-token child CPU preflight claim boundary drifted")


def _terminate_group(child: subprocess.Popen[bytes]) -> int | None:
    if child.poll() is not None:
        return child.returncode
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        return child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return child.wait(timeout=10)


def _terminal(returncode: int | None, *, timed_out: bool, spawn_error: str | None = None) -> dict[str, Any]:
    terminal: dict[str, Any] = {
        "reaped": returncode is not None,
        "timed_out": timed_out,
        "returncode": returncode,
        "exit_code": returncode if isinstance(returncode, int) and returncode >= 0 else None,
        "signal": -returncode if isinstance(returncode, int) and returncode < 0 else None,
    }
    if spawn_error is not None:
        terminal["spawn_error"] = spawn_error
        terminal["reaped"] = False
    return terminal


def _bounded_file_bytes(path: Path, label: str) -> bytes:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise SourceTokenTrueInputAllTenMoeLauncherError(f"cannot stat {label}: {exc}") from exc
    if size > MAX_PREFLIGHT_STREAM_BYTES:
        raise SourceTokenTrueInputAllTenMoeLauncherError(
            f"{label} exceeded bounded {MAX_PREFLIGHT_STREAM_BYTES}-byte capture"
        )
    try:
        return path.read_bytes()
    except OSError as exc:
        raise SourceTokenTrueInputAllTenMoeLauncherError(f"cannot read {label}: {exc}") from exc


def _run_child_cpu_preflight(capture: Path, context: PreflightContext) -> tuple[dict[str, Any], dict[str, Any]]:
    command = [
        str(context.probe_binary["path"]),
        "--outer-preflight",
        str(capture / PREFLIGHT_OUTER_FILENAME),
        "--mode",
        "preflight",
        "--workers",
        "1",
    ]
    stdout_path = capture / PREFLIGHT_CHILD_STDOUT
    stderr_path = capture / PREFLIGHT_CHILD_STDERR
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        try:
            child = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
                close_fds=True,
            )
        except OSError as exc:
            terminal = _terminal(None, timed_out=False, spawn_error=f"{type(exc).__name__}: {exc}")
        else:
            try:
                terminal = _terminal(child.wait(timeout=MAX_PREFLIGHT_SECONDS), timed_out=False)
            except subprocess.TimeoutExpired:
                terminal = _terminal(_terminate_group(child), timed_out=True)
    stdout_raw = _bounded_file_bytes(stdout_path, "source-token child preflight stdout")
    stderr_raw = _bounded_file_bytes(stderr_path, "source-token child preflight stderr")
    if terminal.get("spawn_error") or terminal.get("timed_out") or terminal.get("exit_code") != 0:
        raise SourceTokenTrueInputAllTenMoeLauncherError(
            f"source-token child CPU preflight failed terminal={terminal}"
        )
    if stderr_raw.strip():
        raise SourceTokenTrueInputAllTenMoeLauncherError("source-token child CPU preflight wrote stderr")
    try:
        parsed = json.loads(stdout_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceTokenTrueInputAllTenMoeLauncherError("source-token child CPU preflight stdout is not JSON") from exc
    if not isinstance(parsed, Mapping):
        raise SourceTokenTrueInputAllTenMoeLauncherError("source-token child CPU preflight stdout is not an object")
    _validate_child_preflight_document(dict(parsed), context)
    return terminal, dict(parsed)


def _proof_document(
    *,
    capture: Path,
    context: PreflightContext,
    command_terminal: Mapping[str, Any],
    child_document: Mapping[str, Any],
) -> dict[str, Any]:
    outer_path = capture / PREFLIGHT_OUTER_FILENAME
    return seal(
        {
            "schema": PREFLIGHT_PROOF_SCHEMA,
            "status": PREFLIGHT_PROOF_STATUS,
            "recorded_at": _utc_now(),
            "source_binding": {
                "probe_binary": context.probe_binary,
                "outer_preflight": _binding_with_seal(
                    context.outer_preflight_evidence, context.outer_preflight_seal_sha256
                ),
                "manifest": context.manifest,
                "admission_current": context.admission_current,
                "source_token_route_authority": _binding_with_seal(
                    context.source_authority, context.source_authority_seal_sha256
                ),
                "first_residual_receipt": {
                    **_binding_with_seal(context.first_residual, context.first_residual_seal_sha256),
                    "output_sha256": context.first_residual_output_sha256,
                },
                "typed_bridge_receipt": _binding_with_seal(
                    context.typed_bridge, context.typed_bridge_seal_sha256
                ),
                "fixed_suffix_contract": {
                    **context.fixed_suffix,
                    "schema": outer_preflight.FIXED_ABI_SCHEMA,
                    "status": outer_preflight.FIXED_ABI_STATUS,
                },
                "artifact_identity": {
                    "manifest_seal_sha256": context.manifest_seal_sha256,
                    "admission_pointer_seal_sha256": context.admission_pointer_seal_sha256,
                    "admission_receipt_seal_sha256": context.admission_receipt_seal_sha256,
                },
                "implementation_binding": {
                    "source_token_id": SOURCE_TOKEN_ID,
                    "prefix_dispatches": PREFIX_DISPATCHES,
                    "suffix_dispatches": SUFFIX_DISPATCHES,
                    "total_dispatches": TOTAL_DISPATCHES,
                    "same_command_buffer_fence_required": True,
                    "registered_all_ten_shader_source": context.shader_source,
                    "metal_registry": context.metal_registry,
                },
            },
            "child_preflight": {
                "command": [
                    str(context.probe_binary["path"]),
                    "--outer-preflight",
                    str(outer_path),
                    "--mode",
                    "preflight",
                    "--workers",
                    "1",
                ],
                "terminal": dict(command_terminal),
                "stdout": _file_evidence(capture / PREFLIGHT_CHILD_STDOUT, "child preflight stdout"),
                "stderr": _file_evidence(capture / PREFLIGHT_CHILD_STDERR, "child preflight stderr"),
                "parsed": dict(child_document),
            },
            "claim_boundary": {
                "cpu_outer_and_child_preflight_only": True,
                "metal_device_or_dispatch_performed": False,
                "lease_issued": False,
                "legacy_router_receipt_or_fixture_route_plan_accepted": False,
                "no_complete_layer_token_decoder_generation_hcli_tps_tg_or_tournament_claim": True,
            },
        }
    )


def run_preflight_only(*, base: BaseInputs, probe_bin: Path, capture_dir: Path) -> dict[str, Any]:
    """Create one CPU-only outer/child proof; no lease or Metal path is possible."""
    if capture_dir.exists():
        proof_path = capture_dir / PREFLIGHT_PROOF_FILENAME
        if not proof_path.is_file():
            raise SourceTokenTrueInputAllTenMoeLauncherError(
                "preflight capture exists without a terminal proof; refusing a second child"
            )
        return validate_preflight_proof(proof_path=proof_path, base=base, probe_bin=probe_bin).proof
    capture = _new_capture_dir(capture_dir, "--preflight-capture-dir")
    generated = _build_outer_preflight(base)
    outer_path = capture / PREFLIGHT_OUTER_FILENAME
    _write_new(outer_path, generated)
    outer_evidence = _file_evidence(outer_path, "generated source-token outer preflight")
    context = _context_from_outer_preflight(
        document=generated,
        outer_evidence=outer_evidence,
        outer_seal=_require_sha(generated.get("seal_sha256"), "generated outer preflight seal"),
        base=base,
        probe_path=probe_bin,
    )
    terminal, child_document = _run_child_cpu_preflight(capture, context)
    proof = _proof_document(
        capture=capture,
        context=context,
        command_terminal=terminal,
        child_document=child_document,
    )
    _write_new(capture / PREFLIGHT_PROOF_FILENAME, proof)
    return proof


def _read_child_preflight_stdout(path: Path) -> dict[str, Any]:
    raw = _bounded_file_bytes(path, "sealed child preflight stdout")
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceTokenTrueInputAllTenMoeLauncherError("sealed child preflight stdout is not JSON") from exc
    if not isinstance(document, Mapping):
        raise SourceTokenTrueInputAllTenMoeLauncherError("sealed child preflight stdout is not an object")
    return dict(document)


def validate_preflight_proof(*, proof_path: Path, base: BaseInputs, probe_bin: Path) -> ProofContext:
    """Revalidate a persisted CPU proof against the current exact authorities."""
    proof, proof_seal = _sealed_json(proof_path, "--preflight-proof")
    proof_evidence = _file_evidence(proof_path, "--preflight-proof")
    if proof.get("schema") != PREFLIGHT_PROOF_SCHEMA or proof.get("status") != PREFLIGHT_PROOF_STATUS:
        raise SourceTokenTrueInputAllTenMoeLauncherError("source-token preflight proof schema/status drifted")
    source = _mapping(proof.get("source_binding"), "source-token preflight proof source_binding")
    probe, shader, registry = _implementation_evidence(probe_bin)
    _evidence_matches(source.get("probe_binary"), probe, "source-token preflight proof probe")
    outer_binding = _mapping(source.get("outer_preflight"), "source-token preflight proof outer preflight")
    outer_path = _canonical_regular(Path(str(outer_binding.get("path"))), "source-token proof outer preflight")
    outer_document, outer_seal = _sealed_json(outer_path, "source-token proof outer preflight")
    outer_evidence = _file_evidence(outer_path, "source-token proof outer preflight")
    _assert_exact_mapping(
        outer_binding,
        _binding_with_seal(outer_evidence, outer_seal),
        "source-token preflight proof outer preflight binding",
    )
    # Build the equivalent authority again.  This is file/CPU-only and rejects
    # pointer or immutable lineage drift rather than trusting a stale proof.
    current = _build_outer_preflight(base)
    if not _same_source_binding(
        _mapping(outer_document.get("source_binding"), "persisted outer source binding"),
        _mapping(current.get("source_binding"), "current outer source binding"),
    ):
        raise SourceTokenTrueInputAllTenMoeLauncherError("preflight proof no longer matches current source authority")
    context = _context_from_outer_preflight(
        document=outer_document,
        outer_evidence=outer_evidence,
        outer_seal=outer_seal,
        base=base,
        probe_path=probe_bin,
    )
    _evidence_matches(source.get("manifest"), context.manifest, "source-token preflight proof manifest")
    _historical_pointer_matches(
        source.get("admission_current"), context.admission_current, "source-token preflight proof admission current"
    )
    _assert_exact_mapping(
        source.get("source_token_route_authority"),
        _binding_with_seal(context.source_authority, context.source_authority_seal_sha256),
        "source-token preflight proof route authority",
    )
    expected_prefix = _binding_with_seal(context.first_residual, context.first_residual_seal_sha256)
    expected_prefix["output_sha256"] = context.first_residual_output_sha256
    _assert_exact_mapping(source.get("first_residual_receipt"), expected_prefix, "source-token preflight proof prefix")
    _assert_exact_mapping(
        source.get("typed_bridge_receipt"),
        _binding_with_seal(context.typed_bridge, context.typed_bridge_seal_sha256),
        "source-token preflight proof typed bridge",
    )
    expected_fixed = {
        **context.fixed_suffix,
        "schema": outer_preflight.FIXED_ABI_SCHEMA,
        "status": outer_preflight.FIXED_ABI_STATUS,
    }
    _assert_exact_mapping(source.get("fixed_suffix_contract"), expected_fixed, "source-token preflight proof fixed suffix")
    artifact = _mapping(source.get("artifact_identity"), "source-token preflight proof artifact identity")
    if (
        artifact.get("manifest_seal_sha256") != context.manifest_seal_sha256
        or artifact.get("admission_receipt_seal_sha256") != context.admission_receipt_seal_sha256
    ):
        raise SourceTokenTrueInputAllTenMoeLauncherError("source-token preflight proof artifact identity drifted")
    _require_sha(artifact.get("admission_pointer_seal_sha256"), "source-token preflight proof historical pointer seal")
    implementation = _mapping(source.get("implementation_binding"), "source-token preflight proof implementation")
    if (
        implementation.get("source_token_id") != SOURCE_TOKEN_ID
        or implementation.get("prefix_dispatches") != PREFIX_DISPATCHES
        or implementation.get("suffix_dispatches") != SUFFIX_DISPATCHES
        or implementation.get("total_dispatches") != TOTAL_DISPATCHES
        or implementation.get("same_command_buffer_fence_required") is not True
    ):
        raise SourceTokenTrueInputAllTenMoeLauncherError("source-token preflight proof 9+14 implementation drifted")
    _assert_exact_mapping(
        implementation.get("registered_all_ten_shader_source"), shader, "source-token preflight proof shader"
    )
    _assert_exact_mapping(implementation.get("metal_registry"), registry, "source-token preflight proof registry")
    child = _mapping(proof.get("child_preflight"), "source-token preflight proof child")
    expected_command = [
        str(probe["path"]),
        "--outer-preflight",
        str(outer_path),
        "--mode",
        "preflight",
        "--workers",
        "1",
    ]
    if child.get("command") != expected_command:
        raise SourceTokenTrueInputAllTenMoeLauncherError("source-token preflight proof child command drifted")
    terminal = _mapping(child.get("terminal"), "source-token preflight proof child terminal")
    if (
        terminal.get("reaped") is not True
        or terminal.get("timed_out") is not False
        or terminal.get("exit_code") != 0
        or terminal.get("signal") is not None
        or terminal.get("spawn_error") is not None
    ):
        raise SourceTokenTrueInputAllTenMoeLauncherError("source-token preflight proof child was not cleanly reaped")
    stdout = _mapping(child.get("stdout"), "source-token preflight proof stdout")
    stderr = _mapping(child.get("stderr"), "source-token preflight proof stderr")
    stdout_path = _canonical_regular(Path(str(stdout.get("path"))), "source-token proof child stdout")
    stderr_path = _canonical_regular(Path(str(stderr.get("path"))), "source-token proof child stderr")
    _evidence_matches(stdout, _file_evidence(stdout_path, "source-token proof child stdout"), "source-token proof stdout")
    _evidence_matches(stderr, _file_evidence(stderr_path, "source-token proof child stderr"), "source-token proof stderr")
    if _bounded_file_bytes(stderr_path, "source-token proof child stderr").strip():
        raise SourceTokenTrueInputAllTenMoeLauncherError("source-token preflight proof child stderr is non-empty")
    parsed = _read_child_preflight_stdout(stdout_path)
    if parsed != _mapping(child.get("parsed"), "source-token preflight proof parsed child"):
        raise SourceTokenTrueInputAllTenMoeLauncherError("source-token preflight proof child stdout/parsed drifted")
    _validate_child_preflight_document(parsed, context)
    boundary = _mapping(proof.get("claim_boundary"), "source-token preflight proof claim boundary")
    if (
        boundary.get("cpu_outer_and_child_preflight_only") is not True
        or boundary.get("metal_device_or_dispatch_performed") is not False
        or boundary.get("lease_issued") is not False
        or boundary.get("legacy_router_receipt_or_fixture_route_plan_accepted") is not False
        or boundary.get("no_complete_layer_token_decoder_generation_hcli_tps_tg_or_tournament_claim") is not True
    ):
        raise SourceTokenTrueInputAllTenMoeLauncherError("source-token preflight proof claim boundary drifted")
    return ProofContext(
        preflight=context,
        proof=proof,
        proof_evidence=proof_evidence,
        proof_seal_sha256=proof_seal,
    )


def _exact_lease_binding(value: object, evidence: Mapping[str, Any], seal_sha256: str, label: str) -> None:
    _assert_exact_mapping(value, _child_sealed_binding(evidence, seal_sha256), label)


def _bind_lease(path: Path, proof: ProofContext) -> tuple[dict[str, Any], str, str]:
    document, lease_seal = _sealed_json(path, "--lease-receipt")
    evidence = _file_evidence(path, "--lease-receipt")
    if document.get("schema") != LEASE_SCHEMA or document.get("status") != LEASE_STATUS:
        raise SourceTokenTrueInputAllTenMoeLauncherError("source-token lease schema/status drifted")
    lease_id = _require_sha(document.get("lease_id"), "source-token lease ID")
    context = proof.preflight
    artifact = _mapping(document.get("artifact_binding"), "source-token lease artifact binding")
    if artifact != {
        "manifest_document_sha256": context.manifest["sha256"],
        "manifest_seal_sha256": context.manifest_seal_sha256,
        "admission_receipt_seal_sha256": context.admission_receipt_seal_sha256,
    }:
        raise SourceTokenTrueInputAllTenMoeLauncherError("source-token lease artifact identity drifted")
    _exact_lease_binding(
        document.get("outer_preflight_binding"),
        context.outer_preflight_evidence,
        context.outer_preflight_seal_sha256,
        "source-token lease outer preflight binding",
    )
    _exact_lease_binding(
        document.get("source_token_route_authority_binding"),
        context.source_authority,
        context.source_authority_seal_sha256,
        "source-token lease route authority binding",
    )
    _exact_lease_binding(
        document.get("typed_bridge_binding"),
        context.typed_bridge,
        context.typed_bridge_seal_sha256,
        "source-token lease typed bridge binding",
    )
    expected_prefix = _child_sealed_binding(context.first_residual, context.first_residual_seal_sha256)
    expected_prefix["output_sha256"] = context.first_residual_output_sha256
    _assert_exact_mapping(document.get("first_residual_antecedent"), expected_prefix, "source-token lease prefix")
    expected_fixed = _child_fixed_binding(context.fixed_suffix)
    _assert_exact_mapping(document.get("fixed_suffix_contract_binding"), expected_fixed, "source-token lease fixed suffix")
    _exact_lease_binding(
        document.get("child_preflight_proof_binding"),
        proof.proof_evidence,
        proof.proof_seal_sha256,
        "source-token lease child preflight proof binding",
    )
    implementation = _mapping(document.get("implementation_binding"), "source-token lease implementation")
    if (
        implementation.get("source_token_id") != SOURCE_TOKEN_ID
        or implementation.get("prefix_dispatches") != PREFIX_DISPATCHES
        or implementation.get("suffix_dispatches") != SUFFIX_DISPATCHES
        or implementation.get("total_dispatches") != TOTAL_DISPATCHES
        or implementation.get("same_command_buffer_fence_required") is not True
    ):
        raise SourceTokenTrueInputAllTenMoeLauncherError("source-token lease 9+14 implementation binding drifted")
    _assert_exact_mapping(implementation.get("probe_binary"), context.probe_binary, "source-token lease probe")
    _assert_exact_mapping(
        implementation.get("registered_all_ten_shader_source"), context.shader_source, "source-token lease shader"
    )
    _assert_exact_mapping(implementation.get("metal_registry"), context.metal_registry, "source-token lease registry")
    policy = _mapping(document.get("execution_policy"), "source-token lease execution policy")
    if (
        policy.get("component") != LEASE_COMPONENT
        or policy.get("quiet_qwen80_device_lease") is not True
        or policy.get("strict_math") is not True
        or policy.get("timing_or_benchmarking_allowed") is not False
        or policy.get("complete_layer_or_token_allowed") is not False
        or policy.get("tps_or_tg_claim_allowed") is not False
    ):
        raise SourceTokenTrueInputAllTenMoeLauncherError("source-token lease component scope/policy drifted")
    lifecycle = _mapping(document.get("lifecycle"), "source-token lease lifecycle")
    for field in (
        "fresh_for_this_exact_launch",
        "outer_reaped_capture_required",
        "lease_released_after_first_terminal_child",
        "automatic_retry_prohibited",
    ):
        if lifecycle.get(field) is not True:
            raise SourceTokenTrueInputAllTenMoeLauncherError(f"source-token lease lifecycle {field} drifted")
    watcher = _mapping(document.get("watcher_coordination"), "source-token lease watcher coordination")
    if (
        watcher.get("watcher_hold_must_remain_active") is not True
        or watcher.get("watcher_restart_or_transition_authorized") is not False
    ):
        raise SourceTokenTrueInputAllTenMoeLauncherError("source-token lease watcher hold coordination drifted")
    return evidence, lease_seal, lease_id


def _launch_identity(config: LaunchConfig, context: LaunchContext) -> str:
    payload = {
        "schema": SCHEMA,
        "probe_binary": context.proof.preflight.probe_binary,
        "preflight_proof": context.proof.proof_evidence,
        "preflight_proof_seal_sha256": context.proof.proof_seal_sha256,
        "outer_preflight": context.proof.preflight.outer_preflight_evidence,
        "outer_preflight_seal_sha256": context.proof.preflight.outer_preflight_seal_sha256,
        "lease": context.lease_receipt,
        "lease_seal_sha256": context.lease_seal_sha256,
        "workers": config.workers,
        "timeout_seconds": config.timeout_seconds,
    }
    return _sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _validate_launch_config(config: LaunchConfig) -> LaunchContext:
    if not 1 <= config.workers <= 4:
        raise SourceTokenTrueInputAllTenMoeLauncherError("--workers must be 1..4")
    if config.timeout_seconds <= 0.0:
        raise SourceTokenTrueInputAllTenMoeLauncherError("--timeout-seconds must be positive")
    _validate_new_outer_capture_path(config.capture_dir)
    proof = validate_preflight_proof(
        proof_path=config.preflight_proof,
        base=config.base,
        probe_bin=config.probe_bin,
    )
    lease, lease_seal, lease_id = _bind_lease(config.lease_receipt, proof)
    return LaunchContext(
        proof=proof,
        lease_receipt=lease,
        lease_seal_sha256=lease_seal,
        lease_id=lease_id,
    )


def _outer_launch_authority_document(
    config: LaunchConfig,
    context: LaunchContext,
    *,
    identity: str,
    capture: Path,
) -> dict[str, Any]:
    """Seal the exact one-shot/reaper topology before the child is spawned."""
    preflight = context.proof.preflight
    return seal(
        {
            "schema": OUTER_LAUNCH_AUTHORITY_SCHEMA,
            "status": OUTER_LAUNCH_AUTHORITY_STATUS,
            "recorded_at": _utc_now(),
            "launch_identity_sha256": identity,
            "lease_id": context.lease_id,
            "lease_receipt": context.lease_receipt,
            "lease_receipt_seal_sha256": context.lease_seal_sha256,
            "outer_preflight": preflight.outer_preflight_evidence,
            "outer_preflight_seal_sha256": preflight.outer_preflight_seal_sha256,
            "preflight_proof": _binding_with_seal(
                context.proof.proof_evidence, context.proof.proof_seal_sha256
            ),
            "child_preflight_proof_binding": _child_sealed_binding(
                context.proof.proof_evidence, context.proof.proof_seal_sha256
            ),
            "probe_binary": preflight.probe_binary,
            "planned_outer_capture_dir": str(capture),
            "planned_inner_capture_dir": str(capture / INNER_CAPTURE),
            "workers": config.workers,
            "execution_policy": {
                "component": LEASE_COMPONENT,
                "quiet_qwen80_device_lease": True,
                "strict_math": True,
                "timing_or_benchmarking_allowed": False,
                "complete_layer_or_token_allowed": False,
                "tps_or_tg_claim_allowed": False,
                "workers": config.workers,
                "timeout_seconds": config.timeout_seconds,
            },
            "lifecycle": {
                "fresh_for_this_exact_launch": True,
                "outer_reaped_capture_required": True,
                "lease_released_after_first_terminal_child": True,
                "automatic_retry_prohibited": True,
            },
            "watcher_coordination": {
                "watcher_hold_must_remain_active": True,
                "watcher_restart_or_transition_authorized": False,
            },
            "source_binding": {
                "probe_binary": preflight.probe_binary,
                "preflight_proof": _binding_with_seal(context.proof.proof_evidence, context.proof.proof_seal_sha256),
                "child_preflight_proof_binding": _child_sealed_binding(context.proof.proof_evidence, context.proof.proof_seal_sha256),
                "outer_preflight": _child_sealed_binding(
                    preflight.outer_preflight_evidence, preflight.outer_preflight_seal_sha256
                ),
                "lease_receipt": _child_sealed_binding(context.lease_receipt, context.lease_seal_sha256),
                "artifact_identity": {
                    "manifest_document_sha256": preflight.manifest["sha256"],
                    "manifest_seal_sha256": preflight.manifest_seal_sha256,
                    "admission_receipt_seal_sha256": preflight.admission_receipt_seal_sha256,
                },
                "implementation_binding": {
                    "source_token_id": SOURCE_TOKEN_ID,
                    "prefix_dispatches": PREFIX_DISPATCHES,
                    "suffix_dispatches": SUFFIX_DISPATCHES,
                    "total_dispatches": TOTAL_DISPATCHES,
                    "same_command_buffer_fence_required": True,
                    "registered_all_ten_shader_source": preflight.shader_source,
                    "metal_registry": preflight.metal_registry,
                },
            },
            "planned_capture": {
                "outer_capture_dir": str(capture),
                "inner_capture_dir": str(capture / INNER_CAPTURE),
                "inner_capture_is_new_direct_child_of_outer": True,
                "outer_capture_directory_created_before_authority": True,
                "workers": config.workers,
            },
            "outer_reaper": {
                "outer_pid": os.getpid(),
                "outer_starts_exactly_one_child": True,
                "outer_reaps_child_before_terminal_receipt": True,
                "terminal_receipt_written_last": True,
                "automatic_retry_prohibited": True,
                "lease_reuse_prohibited_after_first_terminal_child": True,
                "watcher_hold_must_remain_active": True,
                "watcher_restart_or_transition_authorized": False,
            },
            "claim_boundary": {
                "authority_only_before_child_spawn": True,
                "metal_device_or_dispatch_performed": False,
                "lease_issued_by_this_authority": False,
                "no_complete_layer_token_decoder_generation_hcli_tps_tg_or_tournament_claim": True,
            },
        }
    )


def _child_command(
    config: LaunchConfig,
    context: LaunchContext,
    outer_launch_authority: Mapping[str, Any],
    outer_capture: Path,
) -> list[str]:
    return [
        str(context.proof.preflight.probe_binary["path"]),
        "--outer-preflight",
        str(context.proof.preflight.outer_preflight_evidence["path"]),
        "--outer-launch-authority",
        str(outer_launch_authority["path"]),
        "--mode",
        "metal",
        "--lease-receipt",
        str(context.lease_receipt["path"]),
        "--outer-capture-dir",
        str(outer_capture),
        "--capture-dir",
        str(outer_capture / INNER_CAPTURE),
        "--workers",
        str(config.workers),
    ]


def _validate_inner_receipt(
    receipt: Mapping[str, Any], context: LaunchContext, outer_launch_authority: Mapping[str, Any]
) -> None:
    preflight = context.proof.preflight
    if (
        receipt.get("schema") != EXPECTED_CHILD_SCHEMA
        or receipt.get("status") != EXPECTED_CHILD_METAL_STATUS
        or receipt.get("mode") != "metal"
        or receipt.get("metal_device_or_dispatch_performed") is not True
        or receipt.get("component_only") is not True
        or receipt.get("complete_layer_or_token_performed") is not False
    ):
        raise SourceTokenTrueInputAllTenMoeLauncherError("source-token inner schema/status/scope drifted")
    artifact = _mapping(receipt.get("artifact_binding"), "source-token inner artifact binding")
    if (
        artifact.get("manifest_document_sha256") != preflight.manifest["sha256"]
        or artifact.get("manifest_seal_sha256") != preflight.manifest_seal_sha256
        or artifact.get("admission_receipt_seal_sha256") != preflight.admission_receipt_seal_sha256
        or artifact.get("layer") != 0
        or artifact.get("linear_state_slot") != 0
    ):
        raise SourceTokenTrueInputAllTenMoeLauncherError("source-token inner artifact authority drifted")
    _require_sha(
        artifact.get("admission_pointer_seal_sha256"), "source-token inner historical/current admission pointer seal"
    )
    _assert_exact_mapping(
        receipt.get("outer_preflight_binding"),
        _child_sealed_binding(preflight.outer_preflight_evidence, preflight.outer_preflight_seal_sha256),
        "source-token inner outer preflight binding",
    )
    _assert_exact_mapping(
        receipt.get("outer_launch_authority_binding"),
        _child_sealed_binding(outer_launch_authority, _require_sha(outer_launch_authority["seal_sha256"], "outer launch authority seal")),
        "source-token inner outer launch authority binding",
    )
    route = _mapping(receipt.get("source_token_route_authority_binding"), "source-token inner route authority")
    for field, expected in _child_sealed_binding(preflight.source_authority, preflight.source_authority_seal_sha256).items():
        if route.get(field) != expected:
            raise SourceTokenTrueInputAllTenMoeLauncherError("source-token inner route authority identity drifted")
    if tuple(route.get("route_ids", [])) != preflight.route_ids:
        raise SourceTokenTrueInputAllTenMoeLauncherError("source-token inner route IDs drifted")
    _same_route_weights(route.get("normalized_weights"), preflight.route_weights, "source-token inner route weights")
    expected_typed = _child_sealed_binding(preflight.typed_bridge, preflight.typed_bridge_seal_sha256)
    expected_typed.update(
        {"schema": outer_preflight.SOURCE_TYPED_BRIDGE_SCHEMA, "status": outer_preflight.SOURCE_TYPED_BRIDGE_STATUS}
    )
    _assert_exact_mapping(receipt.get("typed_bridge_binding"), expected_typed, "source-token inner typed bridge")
    expected_prefix = _child_sealed_binding(preflight.first_residual, preflight.first_residual_seal_sha256)
    expected_prefix["output_sha256"] = preflight.first_residual_output_sha256
    _assert_exact_mapping(receipt.get("first_residual_antecedent"), expected_prefix, "source-token inner prefix")
    expected_fixed = _child_fixed_binding(preflight.fixed_suffix)
    _assert_exact_mapping(receipt.get("fixed_suffix_contract_binding"), expected_fixed, "source-token inner fixed suffix")
    graph = _mapping(receipt.get("same_command_graph"), "source-token inner same command graph")
    if (
        graph.get("source_token_id") != SOURCE_TOKEN_ID
        or graph.get("same_command_graph_required") is not True
        or graph.get("same_command_graph_retained") is not True
        or graph.get("prefix_dispatches") != PREFIX_DISPATCHES
        or graph.get("suffix_dispatches") != SUFFIX_DISPATCHES
        or graph.get("total_dispatches") != TOTAL_DISPATCHES
        or graph.get("command_buffer_fenced_once_after_prefix_and_suffix") is not True
        or graph.get("first_residual_matches_sealed_prefix_antecedent") is not True
    ):
        raise SourceTokenTrueInputAllTenMoeLauncherError("source-token inner 9+14 same-TCB receipt drifted")
    guard = _mapping(receipt.get("route_guard_readback"), "source-token inner route guard")
    if (
        guard.get("value") != 1
        or guard.get("passed") is not True
        or tuple(guard.get("observed_ids", [])) != preflight.route_ids
        or tuple(guard.get("expected_ids", [])) != preflight.route_ids
    ):
        raise SourceTokenTrueInputAllTenMoeLauncherError("source-token inner route guard failed")
    _same_route_weights(guard.get("observed_weights"), preflight.route_weights, "source-token inner observed guard weights")
    _same_route_weights(guard.get("expected_weights"), preflight.route_weights, "source-token inner expected guard weights")
    parity = _mapping(receipt.get("readback_parity"), "source-token inner readback parity")
    witnesses = parity.get("all_ten_route_witnesses")
    if not isinstance(witnesses, list) or len(witnesses) != TOP_K:
        raise SourceTokenTrueInputAllTenMoeLauncherError("source-token inner lacks ten route witnesses")
    for index, witness_value in enumerate(witnesses):
        witness = _mapping(witness_value, f"source-token inner route witness {index}")
        if (
            witness.get("wave_index") != index
            or witness.get("expert_id") != preflight.route_ids[index]
            or witness.get("elements") != 2_048
        ):
            raise SourceTokenTrueInputAllTenMoeLauncherError(f"source-token inner route witness {index} drifted")
        weight = witness.get("normalized_weight")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or abs(float(weight) - preflight.route_weights[index]) > 1.0e-6:
            raise SourceTokenTrueInputAllTenMoeLauncherError(f"source-token inner route witness {index} weight drifted")
        _require_sha(witness.get("output_sha256"), f"source-token inner route witness {index} output SHA")
        error = witness.get("max_abs_error")
        if isinstance(error, bool) or not isinstance(error, (int, float)) or not math.isfinite(float(error)) or float(error) < 0.0:
            raise SourceTokenTrueInputAllTenMoeLauncherError(f"source-token inner route witness {index} error drifted")
    for field, nested_field in (
        ("postnorm_max_abs_error", "postnorm"),
        ("router_logits_max_abs_error", "router_logits"),
        ("shared_expert_max_abs_error", "shared_expert"),
        ("routed_sum_max_abs_error", "routed_sum"),
        ("second_residual_max_abs_error", "second_residual"),
    ):
        value = parity.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0.0:
            raise SourceTokenTrueInputAllTenMoeLauncherError(f"source-token inner {field} drifted")
        nested = _mapping(parity.get(nested_field), f"source-token inner {nested_field} parity")
        nested_value = nested.get("max_abs_error")
        if (
            isinstance(nested_value, bool)
            or not isinstance(nested_value, (int, float))
            or not math.isfinite(float(nested_value))
            or float(nested_value) < 0.0
            or float(nested_value) != float(value)
        ):
            raise SourceTokenTrueInputAllTenMoeLauncherError(
                f"source-token inner {field} does not exactly match {nested_field}.max_abs_error"
            )
    policy = _mapping(receipt.get("metal_execution_policy"), "source-token inner execution policy")
    if (
        policy.get("strict_math_required") is not True
        or policy.get("timing_or_benchmarking_allowed") is not False
        or policy.get("complete_layer_or_token_allowed") is not False
        or policy.get("tps_or_tg_claim_allowed") is not False
    ):
        raise SourceTokenTrueInputAllTenMoeLauncherError("source-token inner execution policy drifted")
    expected_lease = _child_sealed_binding(context.lease_receipt, context.lease_seal_sha256)
    expected_lease["lease_id"] = context.lease_id
    _assert_exact_mapping(policy.get("lease_binding"), expected_lease, "source-token inner lease binding")
    _assert_exact_mapping(
        policy.get("outer_launch_authority_binding"),
        _child_sealed_binding(
            outer_launch_authority,
            _require_sha(outer_launch_authority["seal_sha256"], "outer launch authority seal"),
        ),
        "source-token inner outer launch authority binding",
    )
    durable = _mapping(receipt.get("durable_capture"), "source-token inner durable capture")
    if (
        durable.get("receipt_written_last_is_completion_marker") is not True
        or durable.get("outer_reaped_capture_required") is not True
        or durable.get("replay_guarded") is not True
    ):
        raise SourceTokenTrueInputAllTenMoeLauncherError("source-token inner durable capture contract drifted")
    outer_reaper = _mapping(durable.get("outer_reaper_binding"), "source-token inner outer reaper binding")
    if outer_reaper.get("lease_id") != context.lease_id:
        raise SourceTokenTrueInputAllTenMoeLauncherError("source-token inner outer reaper lease ID drifted")
    _assert_exact_mapping(
        outer_reaper.get("outer_launch_authority"),
        _child_sealed_binding(
            outer_launch_authority,
            _require_sha(outer_launch_authority["seal_sha256"], "outer launch authority seal"),
        ),
        "source-token inner outer reaper authority binding",
    )
    boundary = _mapping(receipt.get("claim_boundary"), "source-token inner claim boundary")
    if (
        boundary.get("source_token_l0_true_moe_component_only") is not True
        or boundary.get("no_complete_layer_token_decoder_generation_hcli_tps_tg_or_tournament_claim") is not True
        or boundary.get("no_watcher_or_server_started") is not True
    ):
        raise SourceTokenTrueInputAllTenMoeLauncherError("source-token inner claim boundary drifted")


def _inner_evidence(
    config: LaunchConfig, context: LaunchContext, outer_launch_authority: Mapping[str, Any]
) -> dict[str, Any]:
    receipt_path = config.capture_dir / INNER_CAPTURE / "receipt.json"
    result: dict[str, Any] = {
        "capture_dir": str(config.capture_dir / INNER_CAPTURE),
        "receipt": {"path": str(receipt_path), "present": receipt_path.is_file()},
    }
    if not receipt_path.is_file():
        return result
    try:
        receipt = _read_json(receipt_path, "source-token inner receipt")
        result.update(_file_evidence(receipt_path, "source-token inner receipt"))
        result["schema"] = receipt.get("schema")
        result["status"] = receipt.get("status")
        _validate_inner_receipt(receipt, context, outer_launch_authority)
    except SourceTokenTrueInputAllTenMoeLauncherError as exc:
        result["binding_valid"] = False
        result["binding_error"] = str(exc)
    else:
        result["binding_valid"] = True
    return result


def _terminal_status(terminal: Mapping[str, Any], inner: Mapping[str, Any]) -> str:
    if terminal.get("spawn_error"):
        return "REFUSED_QWEN80_SOURCE_TOKEN_TRUE_INPUT_ALL_TEN_MOE_OUTER_CHILD_SPAWN_ERROR"
    if terminal.get("timed_out"):
        return "REFUSED_QWEN80_SOURCE_TOKEN_TRUE_INPUT_ALL_TEN_MOE_OUTER_CHILD_TIMEOUT"
    if terminal.get("signal") is not None:
        return "REFUSED_QWEN80_SOURCE_TOKEN_TRUE_INPUT_ALL_TEN_MOE_OUTER_CHILD_SIGNAL"
    if terminal.get("exit_code") != 0:
        return "REFUSED_QWEN80_SOURCE_TOKEN_TRUE_INPUT_ALL_TEN_MOE_OUTER_CHILD_NONZERO"
    if inner.get("binding_valid") is not True:
        return "REFUSED_QWEN80_SOURCE_TOKEN_TRUE_INPUT_ALL_TEN_MOE_OUTER_ZERO_EXIT_WITHOUT_STRICT_INNER_RECEIPT"
    return "CAPTURED_QWEN80_SOURCE_TOKEN_TRUE_INPUT_ALL_TEN_MOE_OUTER_TERMINAL_COMPONENT_ONLY"


def _sync_evidence(path: Path) -> dict[str, Any]:
    with path.open("ab") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    return _file_evidence(path, f"outer stream {path.name}")


def _terminal_receipt(
    config: LaunchConfig,
    context: LaunchContext,
    *,
    identity: str,
    command: Sequence[str],
    child_pid: int | None,
    started_at: str,
    terminal: Mapping[str, Any],
    capture_error: str | None,
    outer_launch_authority: Mapping[str, Any],
) -> dict[str, Any]:
    inner = _inner_evidence(config, context, outer_launch_authority)
    preflight = context.proof.preflight
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": _terminal_status(terminal, inner),
        "recorded_at": _utc_now(),
        "one_shot": {
            "automatic_retry_disabled": True,
            "same_capture_dir_never_starts_a_second_child": True,
            "terminal_receipt_written_last": True,
            "lease_reuse_prohibited_after_terminal": True,
        },
        "launch_identity_sha256": identity,
        "source_binding": {
            "probe_binary": preflight.probe_binary,
            "preflight_proof": _binding_with_seal(context.proof.proof_evidence, context.proof.proof_seal_sha256),
            "outer_preflight": _binding_with_seal(
                preflight.outer_preflight_evidence, preflight.outer_preflight_seal_sha256
            ),
            "source_token_route_authority": _binding_with_seal(
                preflight.source_authority, preflight.source_authority_seal_sha256
            ),
            "first_residual_receipt": {
                **_binding_with_seal(preflight.first_residual, preflight.first_residual_seal_sha256),
                "output_sha256": preflight.first_residual_output_sha256,
            },
            "typed_bridge_receipt": _binding_with_seal(preflight.typed_bridge, preflight.typed_bridge_seal_sha256),
            "fixed_suffix_contract": {
                **preflight.fixed_suffix,
                "schema": outer_preflight.FIXED_ABI_SCHEMA,
                "status": outer_preflight.FIXED_ABI_STATUS,
            },
            "lease_receipt": _binding_with_seal(context.lease_receipt, context.lease_seal_sha256),
            "outer_launch_authority": _binding_with_seal(
                outer_launch_authority,
                _require_sha(outer_launch_authority["seal_sha256"], "outer launch authority seal"),
            ),
            "artifact_identity": {
                "manifest_document_sha256": preflight.manifest["sha256"],
                "manifest_seal_sha256": preflight.manifest_seal_sha256,
                "admission_pointer_seal_sha256": preflight.admission_pointer_seal_sha256,
                "admission_receipt_seal_sha256": preflight.admission_receipt_seal_sha256,
            },
            "implementation_binding": {
                "source_token_id": SOURCE_TOKEN_ID,
                "prefix_dispatches": PREFIX_DISPATCHES,
                "suffix_dispatches": SUFFIX_DISPATCHES,
                "total_dispatches": TOTAL_DISPATCHES,
                "same_command_buffer_fence_required": True,
                "registered_all_ten_shader_source": preflight.shader_source,
                "metal_registry": preflight.metal_registry,
            },
            "workers": config.workers,
        },
        "child": {
            "pid": child_pid,
            "started_at": started_at,
            "finished_at": _utc_now(),
            "command": list(command),
            "terminal": dict(terminal),
        },
        "outer_capture": {
            "directory": str(config.capture_dir),
            "stdout": _sync_evidence(config.capture_dir / OUTER_STDOUT),
            "stderr": _sync_evidence(config.capture_dir / OUTER_STDERR),
        },
        "inner_probe_capture": inner,
        "claim_boundary": {
            "outer_terminal_capture_only": True,
            "requires_source_token_authority_same_tcb_and_all_readback_parity": True,
            "not_a_complete_layer_token_decoder_generation_hcli_tps_tg_or_tournament_result": True,
            "watcher_or_server_transition_not_authorized": True,
        },
    }
    if capture_error is not None:
        payload["capture_error"] = capture_error
    return seal(payload)


def _replay(config: LaunchConfig, identity: str) -> dict[str, Any]:
    terminal_path = config.capture_dir / TERMINAL_FILENAME
    if not terminal_path.is_file():
        raise SourceTokenTrueInputAllTenMoeLauncherError("capture directory exists without terminal receipt")
    document, _ = _sealed_json(terminal_path, "source-token outer terminal receipt")
    if document.get("schema") != SCHEMA or document.get("launch_identity_sha256") != identity:
        raise SourceTokenTrueInputAllTenMoeLauncherError("capture directory belongs to another launch identity")
    return document


def run_attempt(config: LaunchConfig) -> dict[str, Any]:
    """Run one explicitly leased child and reap it before the outer receipt."""
    context = _validate_launch_config(config)
    identity = _launch_identity(config, context)
    if config.capture_dir.exists():
        return _replay(config, identity)
    capture = _new_capture_dir(config.capture_dir, "--capture-dir")
    launch_authority_document = _outer_launch_authority_document(
        config, context, identity=identity, capture=capture
    )
    launch_authority_path = capture / OUTER_LAUNCH_AUTHORITY_FILENAME
    _write_new(launch_authority_path, launch_authority_document)
    launch_authority_evidence = _file_evidence(launch_authority_path, "source-token outer launch authority")
    launch_authority = {
        **launch_authority_evidence,
        "seal_sha256": _require_sha(launch_authority_document.get("seal_sha256"), "outer launch authority seal"),
    }
    command = _child_command(config, context, launch_authority, capture)
    started_at = _utc_now()
    _write_new(
        capture / ACTIVE_FILENAME,
        seal(
            {
                "schema": SCHEMA,
                "status": "STARTED_QWEN80_SOURCE_TOKEN_TRUE_INPUT_ALL_TEN_MOE_OUTER_ONE_SHOT",
                "recorded_at": started_at,
                "launch_identity_sha256": identity,
                "command": command,
                "claim_boundary": {"component_only": True, "automatic_retry_disabled": True},
            }
        ),
    )
    child_pid: int | None = None
    capture_error: str | None = None
    with (capture / OUTER_STDOUT).open("xb") as stdout, (capture / OUTER_STDERR).open("xb") as stderr:
        try:
            child = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
                close_fds=True,
            )
        except OSError as exc:
            terminal = _terminal(None, timed_out=False, spawn_error=f"{type(exc).__name__}: {exc}")
        else:
            child_pid = child.pid
            try:
                _write_new(
                    capture / CHILD_FILENAME,
                    seal(
                        {
                            "schema": SCHEMA,
                            "status": "RUNNING_QWEN80_SOURCE_TOKEN_TRUE_INPUT_ALL_TEN_MOE_OUTER_ONE_SHOT",
                            "recorded_at": _utc_now(),
                            "launch_identity_sha256": identity,
                            "pid": child_pid,
                            "parent_pid": os.getpid(),
                            "command": command,
                            "mode": "metal",
                            "strict_component_lease_required": True,
                        }
                    ),
                )
            except SourceTokenTrueInputAllTenMoeLauncherError as exc:
                capture_error = str(exc)
                terminal = _terminal(_terminate_group(child), timed_out=False)
            else:
                try:
                    terminal = _terminal(child.wait(timeout=config.timeout_seconds), timed_out=False)
                except subprocess.TimeoutExpired:
                    terminal = _terminal(_terminate_group(child), timed_out=True)
    receipt = _terminal_receipt(
        config,
        context,
        identity=identity,
        command=command,
        child_pid=child_pid,
        started_at=started_at,
        terminal=terminal,
        capture_error=capture_error,
        outer_launch_authority=launch_authority,
    )
    _write_new(capture / TERMINAL_FILENAME, receipt)
    return receipt


def _reject_legacy_arguments(arguments: Sequence[str]) -> None:
    for argument in arguments:
        flag = argument.split("=", 1)[0]
        if flag in LEGACY_ARGUMENTS:
            raise SourceTokenTrueInputAllTenMoeLauncherError(
                f"{flag} is prohibited: this launcher accepts only source-token route authority, never legacy router/fixture provenance"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--execute-one-shot", action="store_true")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--admission-current", type=Path, required=True)
    parser.add_argument("--source-token-route-authority", type=Path, required=True)
    parser.add_argument("--first-residual-receipt", type=Path, required=True)
    parser.add_argument("--typed-bridge-receipt", type=Path, required=True)
    parser.add_argument("--fixed-suffix-contract", type=Path, required=True)
    parser.add_argument("--probe-bin", type=Path, required=True)
    parser.add_argument("--preflight-capture-dir", type=Path)
    parser.add_argument("--preflight-proof", type=Path)
    parser.add_argument("--lease-receipt", type=Path)
    parser.add_argument("--capture-dir", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    return parser


def _base_from_args(args: argparse.Namespace) -> BaseInputs:
    return BaseInputs(
        manifest=args.manifest,
        admission_current=args.admission_current,
        source_token_route_authority=args.source_token_route_authority,
        first_residual_receipt=args.first_residual_receipt,
        typed_bridge_receipt=args.typed_bridge_receipt,
        fixed_suffix_contract=args.fixed_suffix_contract,
    )


def _preflight_only_args(args: argparse.Namespace) -> tuple[BaseInputs, Path, Path]:
    if args.preflight_capture_dir is None:
        raise SourceTokenTrueInputAllTenMoeLauncherError("--preflight-only requires --preflight-capture-dir")
    if any(value is not None for value in (args.preflight_proof, args.lease_receipt, args.capture_dir)):
        raise SourceTokenTrueInputAllTenMoeLauncherError(
            "--preflight-only refuses --preflight-proof, --lease-receipt, and --capture-dir"
        )
    if args.workers != 2 or args.timeout_seconds != 7200.0:
        raise SourceTokenTrueInputAllTenMoeLauncherError(
            "--preflight-only has a fixed one-worker bounded child; --workers/--timeout-seconds are metal-only"
        )
    return _base_from_args(args), args.probe_bin, args.preflight_capture_dir


def _launch_from_args(args: argparse.Namespace) -> LaunchConfig:
    if args.preflight_capture_dir is not None:
        raise SourceTokenTrueInputAllTenMoeLauncherError(
            "--execute-one-shot refuses --preflight-capture-dir; it must consume an existing sealed proof"
        )
    if args.preflight_proof is None or args.lease_receipt is None or args.capture_dir is None:
        raise SourceTokenTrueInputAllTenMoeLauncherError(
            "--execute-one-shot requires --preflight-proof, --lease-receipt, and --capture-dir"
        )
    return LaunchConfig(
        base=_base_from_args(args),
        probe_bin=args.probe_bin,
        preflight_proof=args.preflight_proof,
        lease_receipt=args.lease_receipt,
        capture_dir=args.capture_dir,
        workers=args.workers,
        timeout_seconds=args.timeout_seconds,
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        _reject_legacy_arguments(arguments)
        args = build_parser().parse_args(arguments)
        if args.preflight_only:
            base, probe, capture = _preflight_only_args(args)
            proof = run_preflight_only(base=base, probe_bin=probe, capture_dir=capture)
            print(
                json.dumps(
                    {
                        "status": proof["status"],
                        "proof": str(capture / PREFLIGHT_PROOF_FILENAME),
                        "seal_sha256": proof["seal_sha256"],
                    },
                    sort_keys=True,
                )
            )
            return 0
        config = _launch_from_args(args)
        receipt = run_attempt(config)
    except SourceTokenTrueInputAllTenMoeLauncherError as exc:
        print(
            json.dumps(
                {
                    "status": "REFUSED_QWEN80_SOURCE_TOKEN_TRUE_INPUT_ALL_TEN_MOE_OUTER_LAUNCHER",
                    "error": str(exc),
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["status"].startswith("CAPTURED_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
