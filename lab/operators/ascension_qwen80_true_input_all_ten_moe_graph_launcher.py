"""One-shot outer capture for the future Qwen80 L0 true-input all-ten MoE graph.

This is deliberately a launch contract, not a runtime.  It binds a future
device child to one current admitted Qwen80 artifact, sealed router authority,
the descriptor-selected ten compact bodies, a typed same-command-graph
DeltaNet first-residual bridge, and a fresh component-only quiet Metal lease.
It reaps exactly one child and publishes the outer terminal receipt last.

No code in this module opens a Metal context, registers a shader, starts a
watcher, or turns component evidence into a layer/token/HCLI/TPS/TG result.
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

from lab.receipts import seal, verify
SCHEMA = "hawking.ascension.qwen80_true_input_all_ten_moe_graph_outer_launcher.v1"
ACTIVE_FILENAME = "active.json"
CHILD_FILENAME = "child.json"
TERMINAL_FILENAME = "outer-terminal-receipt.json"
OUTER_STDOUT = "outer.stdout.log"
OUTER_STDERR = "outer.stderr.log"
INNER_CAPTURE = "inner"

EXPECTED_PROBE_BASENAME = "ascension_qwen80_all_ten_true_moe_graph_device"
EXPECTED_INNER_SCHEMA = "hawking.ascension.qwen80_all_ten_true_moe_graph_device.v1"
EXPECTED_INNER_STATUS = (
    "EARNED_CURRENT_ADMITTED_QWEN80_LAYER0_TRUE_INPUT_ALL_TEN_ROUTE_SHARED_"
    "SECOND_RESIDUAL_STRICT_MATH_METAL_COMPONENT_NOT_COMPLETE_LAYER_OR_TOKEN"
)

MANIFEST_SCHEMA = "hawking.ascension.qwen80_complete_binary_gravity.v1"
ADMISSION_SCHEMA = "hawking.ascension.qwen_complete_binary_gravity_admission_current_pointer.v1"
ADMISSION_STATUS = "CURRENT_COMPLETE_BINARY_ADMISSION_RECEIPT_SELECTED"
ROUTER_OUTER_SCHEMA = "hawking.ascension.qwen80_direct_packed_postnorm_router_top10_outer_launcher.v1"
ROUTER_OUTER_STATUS = "CAPTURED_QWEN80_POSTNORM_ROUTER_TOP10_OUTER_TERMINAL_COMPONENT_ONLY"
ROUTER_INNER_SCHEMA = "hawking.ascension.qwen80_direct_packed_postnorm_router_top10.v1"
ROUTER_INNER_STATUS = (
    "EARNED_CURRENT_ADMITTED_QWEN80_DIRECT_PACKED_POSTNORM_ROUTER_TOP10_"
    "STRICT_MATH_METAL_COMPONENT_NOT_COMPLETE_LAYER_OR_TOKEN"
)
ROUTE_PLAN_SCHEMA = "hawking.ascension.qwen80_all_ten_routed_expert_binding_plan.v1"
ROUTE_PLAN_STATUS = "SOURCE_BOUND_ALL_TEN_ROUTED_EXPERT_PLAN_READY_NOT_EXECUTED_NOT_COMBINED"
FIRST_RESIDUAL_SCHEMA = "hawking.ascension.qwen80_first_residual_outer_capture.v1"
FIRST_RESIDUAL_STATUS = "CAPTURED_QWEN80_FIRST_RESIDUAL_STRICT_MATH_COMPONENT_ONLY"
BRIDGE_SCHEMA = "hawking.ascension.qwen80_all_ten_true_moe_source_bridge.v1"
BRIDGE_STATUS = "SEALED_CURRENT_ADMITTED_QWEN80_ALL_TEN_TRUE_MOE_SOURCE_BRIDGE_READY_FOR_DEVICE_LEASE"
FIXED_ABI_SCHEMA = "hawking.ascension.qwen80_l0_true_moe_fixed_payload_contract.v1"
FIXED_ABI_STATUS = "PREPARED_QWEN80_L0_TRUE_MOE_FIXED_SUFFIX_PAYLOAD_PLAN_NOT_EXECUTED"
LEASE_SCHEMA = "hawking.ascension.qwen80_true_input_all_ten_moe_graph_quiet_metal_lease.v1"
LEASE_STATUS = "GRANTED_QWEN80_TRUE_INPUT_ALL_TEN_MOE_GRAPH_NON_TIMED_DEVICE_PARITY_LEASE"
LEASE_COMPONENT = "qwen80_true_input_all_ten_moe_graph"
TOP_K = 10
HIDDEN = 2048
ROUTER_WEIGHT_TOLERANCE = 2.0e-5
FIXED_ABI_MODEL_ID = "Qwen3-Coder-Next-80B"
FIXED_ABI_MODEL_KEY = "qwen80"
FIXED_ABI_SOURCE_REPOSITORY = "Qwen/Qwen3-Coder-Next"
FIXED_ABI_SOURCE_REVISION = "a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb"
FIXED_ABI_SOURCE_CONFIG_SHA256 = "a7b8098d3b05777f12bb5677a26bf1240a1bb09def1b06b29e6be86cae2e84f8"
FIXED_ABI_SOURCE_SHARD = "model-00001-of-00040.safetensors"
FIXED_ABI_SOURCE_SHARD_SHA256 = "8e9a517133bfbdc6806cf8b61793055a260efeb68e6e019fd90e4bbb1b665d0a"
FIXED_ABI_SOURCE_BODY_AUDIT_SEAL = "c572b2270b623b8677c374b43c89ddd729de135c25721488bb874b184ff8c3d4"
FIXED_ABI_SOURCE_REVALIDATION_SEAL = "541b16fca1d4805ecba356face97b4e8de1accdeb21e98ee0c13b70ab0746c45"
FIXED_ABI_KERNELS = (
    "qwen80_postnorm_router_top10_rmsnorm",
    "qwen80_postnorm_router_top10_matvec",
    "qwen80_postnorm_router_top10_select",
    "qwen80_all_ten_routed_wave_route_guard",
    "qwen80_all_ten_routed_wave_gate_up",
    "qwen80_all_ten_routed_wave_swiglu",
    "qwen80_all_ten_routed_wave_down_weighted",
    "qwen80_shared_expert_wave_gate_up",
    "qwen80_shared_expert_wave_swiglu",
    "qwen80_shared_expert_wave_down",
    "qwen80_shared_expert_wave_scalar_gate",
    "qwen80_shared_expert_wave_apply_sigmoid_gate",
    "qwen80_moe_wave_aggregate_second_residual_route_sum",
    "qwen80_moe_wave_aggregate_second_residual_add_shared_residual",
)


class TrueInputAllTenMoeGraphLauncherError(RuntimeError):
    """The future one-shot component capture cannot safely start."""


@dataclass(frozen=True)
class LaunchConfig:
    probe_bin: Path
    manifest: Path
    admission_current: Path
    router_receipt: Path
    router_outer_receipt: Path
    route_plan: Path
    first_residual_receipt: Path
    typed_bridge_receipt: Path
    fixed_abi_contract: Path
    lease_receipt: Path | None
    capture_dir: Path
    workers: int
    timeout_seconds: float


@dataclass(frozen=True)
class LaunchContext:
    probe_binary: dict[str, Any]
    manifest: dict[str, Any]
    manifest_seal_sha256: str
    admission_current: dict[str, Any]
    admission_pointer_seal_sha256: str
    admission_receipt_seal_sha256: str
    router_receipt: dict[str, Any]
    router_outer_receipt: dict[str, Any]
    router_outer_seal_sha256: str
    route_plan: dict[str, Any]
    route_ids: tuple[int, ...]
    route_weights: tuple[float, ...]
    first_residual_receipt: dict[str, Any]
    first_residual_seal_sha256: str
    first_residual_output_sha256: str
    typed_bridge_receipt: dict[str, Any]
    typed_bridge_seal_sha256: str
    fixed_abi_contract: dict[str, Any]
    lease_receipt: dict[str, Any]
    lease_seal_sha256: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _require_sha256(value: object, label: str) -> str:
    if not _is_sha256(value):
        raise TrueInputAllTenMoeGraphLauncherError(f"{label} must be a lowercase SHA-256")
    return str(value)


def _canonical_regular(path: Path, label: str, *, executable: bool = False) -> Path:
    if not path.is_absolute():
        raise TrueInputAllTenMoeGraphLauncherError(f"{label} must be absolute: {path}")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise TrueInputAllTenMoeGraphLauncherError(f"cannot stat {label} {path}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise TrueInputAllTenMoeGraphLauncherError(f"{label} must be a regular non-symlink file")
    if executable and not os.access(path, os.X_OK):
        raise TrueInputAllTenMoeGraphLauncherError(f"{label} must be executable")
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise TrueInputAllTenMoeGraphLauncherError(f"cannot canonicalize {label}: {exc}") from exc


def _file_evidence(path: Path, label: str, *, executable: bool = False) -> dict[str, Any]:
    clean = _canonical_regular(path, label, executable=executable)
    return {
        "path": str(clean),
        "present": True,
        "bytes": clean.stat().st_size,
        "sha256": _file_sha256(clean),
    }


def _read_json(path: Path, label: str) -> dict[str, Any]:
    clean = _canonical_regular(path, label)
    try:
        value = json.loads(clean.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrueInputAllTenMoeGraphLauncherError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise TrueInputAllTenMoeGraphLauncherError(f"{label} must be a JSON object")
    return dict(value)


def _sealed_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    value = _read_json(path, label)
    try:
        verify(value, label=str(path))
    except ValueError as exc:
        raise TrueInputAllTenMoeGraphLauncherError(f"{label} is not sealed: {exc}") from exc
    return value, _require_sha256(value.get("seal_sha256"), f"{label}.seal_sha256")


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TrueInputAllTenMoeGraphLauncherError(f"{label} must be an object")
    return dict(value)


def _evidence_matches(evidence: object, expected: Mapping[str, Any], label: str) -> None:
    row = _mapping(evidence, label)
    if row.get("present") is not True:
        raise TrueInputAllTenMoeGraphLauncherError(f"{label} does not attest a present file")
    observed = _canonical_regular(Path(str(row.get("path"))), f"{label}.path")
    if observed != Path(str(expected["path"])):
        raise TrueInputAllTenMoeGraphLauncherError(f"{label} path drifted")
    if row.get("bytes") != expected["bytes"] or row.get("sha256") != expected["sha256"]:
        raise TrueInputAllTenMoeGraphLauncherError(f"{label} byte/digest drifted")


def _atomic_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise TrueInputAllTenMoeGraphLauncherError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as exc:
        raise TrueInputAllTenMoeGraphLauncherError(f"refusing to overwrite {path}") from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _bind_manifest(path: Path) -> tuple[dict[str, Any], str]:
    document, seal_sha256 = _sealed_json(path, "--manifest")
    if document.get("schema") != MANIFEST_SCHEMA:
        raise TrueInputAllTenMoeGraphLauncherError("--manifest schema drifted")
    return _file_evidence(path, "--manifest"), seal_sha256


def _bind_admission(
    path: Path, manifest: Mapping[str, Any], manifest_seal: str
) -> tuple[dict[str, Any], str, str]:
    document, pointer_seal = _sealed_json(path, "--admission-current")
    if document.get("schema") != ADMISSION_SCHEMA or document.get("status") != ADMISSION_STATUS:
        raise TrueInputAllTenMoeGraphLauncherError("--admission-current schema/status drifted")
    selected = _mapping(document.get("complete_manifest"), "admission complete_manifest")
    if _canonical_regular(Path(str(selected.get("path"))), "admission complete_manifest.path") != Path(
        str(manifest["path"])
    ):
        raise TrueInputAllTenMoeGraphLauncherError("admission selects another manifest")
    if selected.get("document_sha256") != manifest["sha256"] or selected.get("seal_sha256") != manifest_seal:
        raise TrueInputAllTenMoeGraphLauncherError("admission manifest identity drifted")
    receipt = _mapping(document.get("admission_receipt"), "admission receipt")
    return (
        _file_evidence(path, "--admission-current"),
        pointer_seal,
        _require_sha256(receipt.get("seal_sha256"), "admission receipt seal"),
    )


def _route_ids_and_weights(plan: Mapping[str, Any]) -> tuple[tuple[int, ...], tuple[float, ...]]:
    if plan.get("schema") != ROUTE_PLAN_SCHEMA or plan.get("status") != ROUTE_PLAN_STATUS:
        raise TrueInputAllTenMoeGraphLauncherError("--route-plan schema/status drifted")
    if plan.get("layer") != 0:
        raise TrueInputAllTenMoeGraphLauncherError("--route-plan is not layer zero")
    router = _mapping(plan.get("router_evidence"), "route plan router_evidence")
    ids = router.get("source_stable_route_ids")
    weights = router.get("source_stable_normalized_weights")
    if not isinstance(ids, list) or not isinstance(weights, list) or len(ids) != TOP_K or len(weights) != TOP_K:
        raise TrueInputAllTenMoeGraphLauncherError("route plan must bind exactly ten router IDs/weights")
    if any(isinstance(item, bool) or not isinstance(item, int) or not 0 <= item < 512 for item in ids):
        raise TrueInputAllTenMoeGraphLauncherError("route plan IDs are invalid")
    if len(set(ids)) != TOP_K:
        raise TrueInputAllTenMoeGraphLauncherError("route plan IDs are not unique")
    try:
        numeric_weights = tuple(float(value) for value in weights)
    except (TypeError, ValueError) as exc:
        raise TrueInputAllTenMoeGraphLauncherError("route plan weights are not numeric") from exc
    if any(not math.isfinite(value) or value <= 0.0 for value in numeric_weights) or not math.isclose(
        sum(numeric_weights), 1.0, rel_tol=0.0, abs_tol=ROUTER_WEIGHT_TOLERANCE
    ):
        raise TrueInputAllTenMoeGraphLauncherError("route plan weights are not normalized")
    waves = plan.get("deterministic_waves")
    if not isinstance(waves, list) or len(waves) != TOP_K:
        raise TrueInputAllTenMoeGraphLauncherError("route plan must contain exactly ten waves")
    for index, wave_value in enumerate(waves):
        wave = _mapping(wave_value, f"route plan wave {index}")
        if wave.get("wave_index") != index or wave.get("expert_id") != ids[index]:
            raise TrueInputAllTenMoeGraphLauncherError("route plan wave order/expert drifted")
        weight = wave.get("normalized_weight")
        if not isinstance(weight, (int, float)) or isinstance(weight, bool) or float(weight) != numeric_weights[index]:
            raise TrueInputAllTenMoeGraphLauncherError("route plan wave weight drifted")
        for projection in ("gate", "up", "down"):
            row = _mapping(wave.get(projection), f"route plan wave {index} {projection}")
            _require_sha256(row.get("artifact_sha256"), f"route plan wave {index} {projection} SHA")
    return tuple(ids), numeric_weights


def _bind_router(
    *,
    manifest: Mapping[str, Any],
    manifest_seal: str,
    admission: Mapping[str, Any],
    admission_receipt_seal: str,
    router_path: Path,
    router_outer_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    inner_evidence = _file_evidence(router_path, "--router-receipt")
    outer_evidence = _file_evidence(router_outer_path, "--router-outer-receipt")
    outer, outer_seal = _sealed_json(router_outer_path, "--router-outer-receipt")
    if outer.get("schema") != ROUTER_OUTER_SCHEMA or outer.get("status") != ROUTER_OUTER_STATUS:
        raise TrueInputAllTenMoeGraphLauncherError("router outer schema/status drifted")
    source = _mapping(outer.get("source_binding"), "router outer source_binding")
    _evidence_matches(source.get("manifest"), manifest, "router outer manifest")
    historical_admission = _mapping(source.get("admission_current"), "router outer admission")
    if _canonical_regular(Path(str(historical_admission.get("path"))), "router outer admission.path") != Path(
        str(admission["path"])
    ):
        raise TrueInputAllTenMoeGraphLauncherError("router outer admission path drifted")
    inner_summary = _mapping(outer.get("inner_probe_capture"), "router outer inner")
    if (
        inner_summary.get("present") is not True
        or inner_summary.get("path") != inner_evidence["path"]
        or inner_summary.get("sha256") != inner_evidence["sha256"]
        or inner_summary.get("schema") != ROUTER_INNER_SCHEMA
        or inner_summary.get("status") != ROUTER_INNER_STATUS
        or inner_summary.get("mode") != "metal"
        or inner_summary.get("metal_performed") is not True
    ):
        raise TrueInputAllTenMoeGraphLauncherError("router outer does not bind supplied strict-Metal inner")
    inner = _read_json(router_path, "--router-receipt")
    if (
        inner.get("schema") != ROUTER_INNER_SCHEMA
        or inner.get("status") != ROUTER_INNER_STATUS
        or inner.get("mode") != "metal"
        or inner.get("component_only") is not True
        or inner.get("metal_device_or_dispatch_performed") is not True
    ):
        raise TrueInputAllTenMoeGraphLauncherError("router inner is not strict-Metal component evidence")
    binding = _mapping(inner.get("artifact_binding"), "router inner artifact_binding")
    if (
        _canonical_regular(Path(str(binding.get("manifest_path"))), "router manifest path")
        != Path(str(manifest["path"]))
        or binding.get("manifest_document_sha256") != manifest["sha256"]
        or binding.get("manifest_seal_sha256") != manifest_seal
        or _canonical_regular(Path(str(binding.get("admission_current_path"))), "router admission path")
        != Path(str(admission["path"]))
        or binding.get("admission_receipt_seal_sha256") != admission_receipt_seal
    ):
        raise TrueInputAllTenMoeGraphLauncherError("router inner artifact authority drifted")
    return inner_evidence, outer_evidence, outer_seal


def _bind_first_residual(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    manifest_seal: str,
    admission: Mapping[str, Any],
    admission_receipt_seal: str,
) -> tuple[dict[str, Any], str, str]:
    document, receipt_seal = _sealed_json(path, "--first-residual-receipt")
    if document.get("schema") != FIRST_RESIDUAL_SCHEMA or document.get("status") != FIRST_RESIDUAL_STATUS:
        raise TrueInputAllTenMoeGraphLauncherError("first-residual receipt schema/status drifted")
    source = _mapping(document.get("source_binding"), "first-residual source_binding")
    _evidence_matches(source.get("manifest"), manifest, "first-residual manifest")
    # `*_ADMISSION_CURRENT.json` is a versioned mutable pointer.  A watcher
    # may reseal it without changing the immutable selected manifest or the
    # admitted receipt.  Keep the historical raw pointer evidence in the
    # antecedent receipt, but bind a later launch to the *current* pointer
    # path plus the stable selected-manifest/admission-receipt authority.  A
    # changed path, manifest, or receipt seal is still a hard refusal.
    historical_admission = _mapping(source.get("admission_current"), "first-residual admission")
    if historical_admission.get("present") is not True:
        raise TrueInputAllTenMoeGraphLauncherError("first-residual admission is not historical file evidence")
    if _canonical_regular(
        Path(str(historical_admission.get("path"))), "first-residual admission.path"
    ) != Path(str(admission["path"])):
        raise TrueInputAllTenMoeGraphLauncherError("first-residual admission path drifted")
    _require_sha256(
        source.get("admission_pointer_seal_sha256"), "first-residual historical admission pointer seal"
    )
    if (
        source.get("manifest_seal_sha256") != manifest_seal
        or source.get("admission_receipt_seal_sha256") != admission_receipt_seal
    ):
        raise TrueInputAllTenMoeGraphLauncherError("first-residual artifact authority drifted")
    output = _mapping(document.get("first_residual_output"), "first-residual output")
    if (
        output.get("layer") != 0
        or output.get("linear_state_slot") != 0
        or output.get("elements") != HIDDEN
        or output.get("same_command_graph_required") is not True
    ):
        raise TrueInputAllTenMoeGraphLauncherError("first-residual source/state geometry drifted")
    return _file_evidence(path, "--first-residual-receipt"), receipt_seal, _require_sha256(
        output.get("sha256"), "first-residual output SHA"
    )


def _bind_typed_bridge(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    manifest_seal: str,
    admission: Mapping[str, Any],
    admission_receipt_seal: str,
    route_plan: Mapping[str, Any],
    first_residual: Mapping[str, Any],
    first_residual_seal: str,
    first_residual_output_sha256: str,
) -> tuple[dict[str, Any], str]:
    document, bridge_seal = _sealed_json(path, "--typed-bridge-receipt")
    if document.get("schema") != BRIDGE_SCHEMA or document.get("status") != BRIDGE_STATUS:
        raise TrueInputAllTenMoeGraphLauncherError("typed bridge schema/status drifted")
    source = _mapping(document.get("source_binding"), "typed bridge source_binding")
    _evidence_matches(source.get("manifest"), manifest, "typed bridge manifest")
    # Like the first-residual antecedent, this bridge may have been sealed
    # while the versioned admission-current pointer was subsequently resealed.
    # Keep that historical raw evidence in the bridge, but bind this launch to
    # the current pointer *path* plus the immutable selected manifest and
    # admission-receipt seals.  A changed path, manifest, or receipt remains a
    # hard refusal; a harmless recorded-at/pointer reseal is not.
    historical_admission = _mapping(source.get("admission_current"), "typed bridge admission")
    if historical_admission.get("present") is not True:
        raise TrueInputAllTenMoeGraphLauncherError(
            "typed bridge admission is not historical file evidence"
        )
    if _canonical_regular(
        Path(str(historical_admission.get("path"))), "typed bridge admission.path"
    ) != Path(str(admission["path"])):
        raise TrueInputAllTenMoeGraphLauncherError("typed bridge admission path drifted")
    _require_sha256(
        source.get("admission_pointer_seal_sha256"),
        "typed bridge historical admission pointer seal",
    )
    _evidence_matches(source.get("route_plan"), route_plan, "typed bridge route plan")
    _evidence_matches(source.get("first_residual_receipt"), first_residual, "typed bridge first residual")
    if source.get("manifest_seal_sha256") != manifest_seal or source.get("admission_receipt_seal_sha256") != admission_receipt_seal:
        raise TrueInputAllTenMoeGraphLauncherError("typed bridge artifact authority drifted")
    bridge = _mapping(document.get("typed_bridge"), "typed bridge")
    if (
        bridge.get("layer") != 0
        or bridge.get("route_count") != TOP_K
        or bridge.get("first_residual_elements") != HIDDEN
        or bridge.get("same_command_graph_required") is not True
        or bridge.get("first_residual_output_sha256") != first_residual_output_sha256
        or bridge.get("first_residual_receipt_seal_sha256") != first_residual_seal
    ):
        raise TrueInputAllTenMoeGraphLauncherError("typed bridge first-residual identity drifted")
    sections = bridge.get("compact_section_sha256")
    if not isinstance(sections, Mapping) or set(sections) != {
        "gate_scales", "gate_signs", "up_scales", "up_signs", "down_scales", "down_signs"
    }:
        raise TrueInputAllTenMoeGraphLauncherError("typed bridge compact section inventory drifted")
    for label, value in sections.items():
        _require_sha256(value, f"typed bridge compact section {label}")
    return _file_evidence(path, "--typed-bridge-receipt"), bridge_seal


def _bind_fixed_abi_contract(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    manifest_seal: str,
    admission_receipt_seal: str,
) -> dict[str, Any]:
    """Bind the static suffix ABI as immutable input, never as device evidence.

    The contract intentionally remains unsealed and `NOT_EXECUTED`; the future
    sealed quiet lease anchors its exact raw file evidence before an outer child
    may use it.  Treating this plan itself as a sealed execution receipt would
    be a false promotion.
    """

    document = _read_json(path, "--fixed-abi-contract")
    if document.get("seal_sha256") is not None:
        raise TrueInputAllTenMoeGraphLauncherError(
            "fixed ABI contract must remain an unsealed static plan, not execution evidence"
        )
    if document.get("schema") != FIXED_ABI_SCHEMA or document.get("status") != FIXED_ABI_STATUS:
        raise TrueInputAllTenMoeGraphLauncherError("fixed ABI contract schema/status drifted")
    source = _mapping(document.get("source_binding"), "fixed ABI source_binding")
    expected_source = {
        "model_id": FIXED_ABI_MODEL_ID,
        "model_key": FIXED_ABI_MODEL_KEY,
        "source_repository": FIXED_ABI_SOURCE_REPOSITORY,
        "source_revision": FIXED_ABI_SOURCE_REVISION,
        "source_config_sha256": FIXED_ABI_SOURCE_CONFIG_SHA256,
        "source_shard": FIXED_ABI_SOURCE_SHARD,
        "source_shard_sha256": FIXED_ABI_SOURCE_SHARD_SHA256,
        "manifest_schema": MANIFEST_SCHEMA,
        "manifest_seal_sha256": manifest_seal,
        "manifest_document_sha256": manifest["sha256"],
        "admission_receipt_seal_sha256": admission_receipt_seal,
        "source_body_audit_seal_sha256": FIXED_ABI_SOURCE_BODY_AUDIT_SEAL,
        "source_revalidation_seal_sha256": FIXED_ABI_SOURCE_REVALIDATION_SEAL,
    }
    if source != expected_source:
        raise TrueInputAllTenMoeGraphLauncherError("fixed ABI source/artifact authority drifted")
    geometry = _mapping(document.get("geometry"), "fixed ABI geometry")
    if geometry != {
        "layer": 0,
        "hidden": HIDDEN,
        "intermediate": 512,
        "experts": 512,
        "top_k": TOP_K,
        "group_size": 128,
        "rms_epsilon": "1e-6",
    }:
        raise TrueInputAllTenMoeGraphLauncherError("fixed ABI geometry drifted")
    authority = _mapping(document.get("external_authority"), "fixed ABI external_authority")
    expected_authority = {
        "route_plan_schema": ROUTE_PLAN_SCHEMA,
        "route_plan_status": ROUTE_PLAN_STATUS,
        "first_residual_schema": FIRST_RESIDUAL_SCHEMA,
        "first_residual_status": FIRST_RESIDUAL_STATUS,
        "typed_bridge_schema": BRIDGE_SCHEMA,
        "typed_bridge_status": BRIDGE_STATUS,
    }
    if any(authority.get(key) != value for key, value in expected_authority.items()):
        raise TrueInputAllTenMoeGraphLauncherError("fixed ABI external authority drifted")
    if (
        authority.get("route_payloads_materialized_here") is not False
        or authority.get("first_residual_materialized_here") is not False
        or authority.get("expected_topk_witness_materialized_here") is not False
        or authority.get("route_tensor_sha256s_materialized_here") is not False
    ):
        raise TrueInputAllTenMoeGraphLauncherError("fixed ABI illegally materializes external route/state authority")
    dispatches = document.get("fixed_14_dispatch_abi")
    if not isinstance(dispatches, list) or tuple(
        row.get("kernel") if isinstance(row, Mapping) else None for row in dispatches
    ) != FIXED_ABI_KERNELS or any(
        not isinstance(row, Mapping) or row.get("ordinal") != index
        for index, row in enumerate(dispatches, start=1)
    ):
        raise TrueInputAllTenMoeGraphLauncherError("fixed ABI 14-dispatch order drifted")
    boundary = _mapping(document.get("claim_boundary"), "fixed ABI claim_boundary")
    if (
        boundary.get("artifact_scan_or_payload_open_performed") is not False
        or boundary.get("metal_context_or_dispatch_performed") is not False
        or boundary.get("runtime_watcher_server_registry_or_hcli_changed") is not False
        or boundary.get("token_or_tps_claim") is not False
        or boundary.get("execution_status") != "PREPARED_NOT_EXECUTED"
    ):
        raise TrueInputAllTenMoeGraphLauncherError("fixed ABI plan was promoted beyond static preparation")
    # The static contract owns its complete buffer/payload grammar.  The outer
    # launcher pins its whole raw byte identity below; its future quiet lease
    # is the sealed object that authorizes this exact, non-executed revision.
    return _file_evidence(path, "--fixed-abi-contract")


def _bind_lease(
    path: Path | None,
    *,
    manifest: Mapping[str, Any],
    manifest_seal: str,
    admission_receipt_seal: str,
    typed_bridge: Mapping[str, Any],
    typed_bridge_seal: str,
    fixed_abi_contract: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    if path is None:
        raise TrueInputAllTenMoeGraphLauncherError("--lease-receipt is required before any child starts")
    document, lease_seal = _sealed_json(path, "--lease-receipt")
    if document.get("schema") != LEASE_SCHEMA or document.get("status") != LEASE_STATUS:
        raise TrueInputAllTenMoeGraphLauncherError("lease schema/status does not authorize this component")
    if not isinstance(document.get("lease_id"), str) or not document["lease_id"]:
        raise TrueInputAllTenMoeGraphLauncherError("lease lacks lease_id")
    lifecycle = _mapping(document.get("lifecycle"), "lease lifecycle")
    if (
        lifecycle.get("fresh_for_this_exact_launch") is not True
        or lifecycle.get("automatic_retry_prohibited") is not True
        or lifecycle.get("outer_reaped_capture_required") is not True
    ):
        raise TrueInputAllTenMoeGraphLauncherError("lease is not fresh/one-shot/outer-reaped")
    policy = _mapping(document.get("execution_policy"), "lease execution_policy")
    if (
        policy.get("component") != LEASE_COMPONENT
        or policy.get("quiet_qwen80_device_lease") is not True
        or policy.get("strict_math") is not True
        or policy.get("timing_or_benchmarking_allowed") is not False
        or policy.get("complete_layer_or_token_allowed") is not False
        or policy.get("tps_or_tg_claim_allowed") is not False
    ):
        raise TrueInputAllTenMoeGraphLauncherError("lease policy is not strict non-timed component-only")
    artifact = _mapping(document.get("artifact_binding"), "lease artifact_binding")
    if (
        artifact.get("manifest_document_sha256") != manifest["sha256"]
        or artifact.get("manifest_seal_sha256") != manifest_seal
        or artifact.get("admission_receipt_seal_sha256") != admission_receipt_seal
    ):
        raise TrueInputAllTenMoeGraphLauncherError("lease artifact authority drifted")
    bridge_binding = _mapping(document.get("typed_bridge_binding"), "lease typed_bridge_binding")
    if (
        bridge_binding.get("path") != typed_bridge["path"]
        or bridge_binding.get("document_sha256") != typed_bridge["sha256"]
        or bridge_binding.get("schema") != BRIDGE_SCHEMA
        or bridge_binding.get("status") != BRIDGE_STATUS
        or bridge_binding.get("seal_sha256") != typed_bridge_seal
    ):
        raise TrueInputAllTenMoeGraphLauncherError("lease typed bridge identity drifted")
    fixed_abi_binding = _mapping(document.get("fixed_abi_contract_binding"), "lease fixed ABI binding")
    if (
        fixed_abi_binding.get("path") != fixed_abi_contract["path"]
        or fixed_abi_binding.get("document_sha256") != fixed_abi_contract["sha256"]
        or fixed_abi_binding.get("schema") != FIXED_ABI_SCHEMA
        or fixed_abi_binding.get("status") != FIXED_ABI_STATUS
    ):
        raise TrueInputAllTenMoeGraphLauncherError("lease fixed ABI identity drifted")
    return _file_evidence(path, "--lease-receipt"), lease_seal


def _validate_config(config: LaunchConfig) -> LaunchContext:
    probe = _canonical_regular(config.probe_bin, "--probe-bin", executable=True)
    if probe.name != EXPECTED_PROBE_BASENAME:
        raise TrueInputAllTenMoeGraphLauncherError(
            f"--probe-bin must be {EXPECTED_PROBE_BASENAME}, got {probe.name!r}"
        )
    if config.workers < 1 or config.workers > 4:
        raise TrueInputAllTenMoeGraphLauncherError("--workers must be 1..4")
    if not config.timeout_seconds > 0:
        raise TrueInputAllTenMoeGraphLauncherError("--timeout-seconds must be positive")
    if not config.capture_dir.is_absolute() or not config.capture_dir.parent.is_dir():
        raise TrueInputAllTenMoeGraphLauncherError("--capture-dir must be absolute with existing parent")
    manifest, manifest_seal = _bind_manifest(config.manifest)
    admission, pointer_seal, admission_receipt_seal = _bind_admission(
        config.admission_current, manifest, manifest_seal
    )
    router, router_outer, router_outer_seal = _bind_router(
        manifest=manifest,
        manifest_seal=manifest_seal,
        admission=admission,
        admission_receipt_seal=admission_receipt_seal,
        router_path=config.router_receipt,
        router_outer_path=config.router_outer_receipt,
    )
    route_plan = _file_evidence(config.route_plan, "--route-plan")
    route_ids, route_weights = _route_ids_and_weights(_read_json(config.route_plan, "--route-plan"))
    first_residual, first_residual_seal, output_sha = _bind_first_residual(
        config.first_residual_receipt,
        manifest=manifest,
        manifest_seal=manifest_seal,
        admission=admission,
        admission_receipt_seal=admission_receipt_seal,
    )
    bridge, bridge_seal = _bind_typed_bridge(
        config.typed_bridge_receipt,
        manifest=manifest,
        manifest_seal=manifest_seal,
        admission=admission,
        admission_receipt_seal=admission_receipt_seal,
        route_plan=route_plan,
        first_residual=first_residual,
        first_residual_seal=first_residual_seal,
        first_residual_output_sha256=output_sha,
    )
    fixed_abi = _bind_fixed_abi_contract(
        config.fixed_abi_contract,
        manifest=manifest,
        manifest_seal=manifest_seal,
        admission_receipt_seal=admission_receipt_seal,
    )
    lease, lease_seal = _bind_lease(
        config.lease_receipt,
        manifest=manifest,
        manifest_seal=manifest_seal,
        admission_receipt_seal=admission_receipt_seal,
        typed_bridge=bridge,
        typed_bridge_seal=bridge_seal,
        fixed_abi_contract=fixed_abi,
    )
    return LaunchContext(
        probe_binary=_file_evidence(config.probe_bin, "--probe-bin", executable=True),
        manifest=manifest,
        manifest_seal_sha256=manifest_seal,
        admission_current=admission,
        admission_pointer_seal_sha256=pointer_seal,
        admission_receipt_seal_sha256=admission_receipt_seal,
        router_receipt=router,
        router_outer_receipt=router_outer,
        router_outer_seal_sha256=router_outer_seal,
        route_plan=route_plan,
        route_ids=route_ids,
        route_weights=route_weights,
        first_residual_receipt=first_residual,
        first_residual_seal_sha256=first_residual_seal,
        first_residual_output_sha256=output_sha,
        typed_bridge_receipt=bridge,
        typed_bridge_seal_sha256=bridge_seal,
        fixed_abi_contract=fixed_abi,
        lease_receipt=lease,
        lease_seal_sha256=lease_seal,
    )


def _launch_identity(config: LaunchConfig, context: LaunchContext) -> str:
    payload = {
        "schema": SCHEMA,
        "probe_binary": context.probe_binary,
        "manifest": context.manifest,
        "manifest_seal_sha256": context.manifest_seal_sha256,
        "admission_current": context.admission_current,
        "admission_pointer_seal_sha256": context.admission_pointer_seal_sha256,
        "admission_receipt_seal_sha256": context.admission_receipt_seal_sha256,
        "router_receipt": context.router_receipt,
        "router_outer_receipt": context.router_outer_receipt,
        "router_outer_seal_sha256": context.router_outer_seal_sha256,
        "route_plan": context.route_plan,
        "first_residual_receipt": context.first_residual_receipt,
        "first_residual_seal_sha256": context.first_residual_seal_sha256,
        "typed_bridge_receipt": context.typed_bridge_receipt,
        "typed_bridge_seal_sha256": context.typed_bridge_seal_sha256,
        "fixed_abi_contract": context.fixed_abi_contract,
        "lease_receipt": context.lease_receipt,
        "lease_seal_sha256": context.lease_seal_sha256,
        "workers": config.workers,
        "timeout_seconds": config.timeout_seconds,
    }
    return _sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())


def _child_command(config: LaunchConfig, inner_capture: Path) -> list[str]:
    assert config.lease_receipt is not None
    return [
        str(_canonical_regular(config.probe_bin, "--probe-bin", executable=True)),
        "--manifest", str(_canonical_regular(config.manifest, "--manifest")),
        "--admission-current", str(_canonical_regular(config.admission_current, "--admission-current")),
        "--router-receipt", str(_canonical_regular(config.router_receipt, "--router-receipt")),
        "--router-outer-receipt", str(_canonical_regular(config.router_outer_receipt, "--router-outer-receipt")),
        "--route-plan", str(_canonical_regular(config.route_plan, "--route-plan")),
        "--first-residual-receipt", str(_canonical_regular(config.first_residual_receipt, "--first-residual-receipt")),
        "--typed-bridge-receipt", str(_canonical_regular(config.typed_bridge_receipt, "--typed-bridge-receipt")),
        "--fixed-abi-contract", str(_canonical_regular(config.fixed_abi_contract, "--fixed-abi-contract")),
        "--lease-receipt", str(_canonical_regular(config.lease_receipt, "--lease-receipt")),
        "--outer-capture-dir", str(config.capture_dir),
        "--capture-dir", str(inner_capture),
        "--mode", "metal",
        "--workers", str(config.workers),
    ]


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
    result: dict[str, Any] = {
        "reaped": returncode is not None,
        "timed_out": timed_out,
        "returncode": returncode,
        "exit_code": returncode if isinstance(returncode, int) and returncode >= 0 else None,
        "signal": -returncode if isinstance(returncode, int) and returncode < 0 else None,
    }
    if spawn_error is not None:
        result["spawn_error"] = spawn_error
        result["reaped"] = False
    return result


def _validate_inner(receipt: Mapping[str, Any], context: LaunchContext) -> None:
    if (
        receipt.get("schema") != EXPECTED_INNER_SCHEMA
        or receipt.get("status") != EXPECTED_INNER_STATUS
        or receipt.get("mode") != "metal"
        or receipt.get("metal_device_or_dispatch_performed") is not True
        or receipt.get("component_only") is not True
        or receipt.get("complete_layer_or_token_performed") is not False
    ):
        raise TrueInputAllTenMoeGraphLauncherError("inner schema/status/scope boundary drifted")
    durable = _mapping(receipt.get("durable_capture"), "inner durable_capture")
    if (
        durable.get("receipt_written_last_is_completion_marker") is not True
        or durable.get("outer_reaped_capture_required") is not True
        or durable.get("replay_guarded") is not True
    ):
        raise TrueInputAllTenMoeGraphLauncherError("inner does not attest receipt-last/replay guard")
    artifact = _mapping(receipt.get("artifact_binding"), "inner artifact_binding")
    if (
        artifact.get("manifest_document_sha256") != context.manifest["sha256"]
        or artifact.get("manifest_seal_sha256") != context.manifest_seal_sha256
        or artifact.get("admission_pointer_seal_sha256") != context.admission_pointer_seal_sha256
        or artifact.get("admission_receipt_seal_sha256") != context.admission_receipt_seal_sha256
    ):
        raise TrueInputAllTenMoeGraphLauncherError("inner artifact authority drifted")
    bridge = _mapping(receipt.get("typed_bridge_binding"), "inner typed_bridge_binding")
    if (
        bridge.get("receipt_path") != context.typed_bridge_receipt["path"]
        or bridge.get("receipt_document_sha256") != context.typed_bridge_receipt["sha256"]
        or bridge.get("seal_sha256") != context.typed_bridge_seal_sha256
        or bridge.get("schema") != BRIDGE_SCHEMA
        or bridge.get("status") != BRIDGE_STATUS
        or bridge.get("first_residual_output_sha256") != context.first_residual_output_sha256
    ):
        raise TrueInputAllTenMoeGraphLauncherError("inner typed bridge identity drifted")
    fixed_abi = _mapping(receipt.get("fixed_abi_contract_binding"), "inner fixed ABI binding")
    if (
        fixed_abi.get("path") != context.fixed_abi_contract["path"]
        or fixed_abi.get("document_sha256") != context.fixed_abi_contract["sha256"]
        or fixed_abi.get("schema") != FIXED_ABI_SCHEMA
        or fixed_abi.get("status") != FIXED_ABI_STATUS
    ):
        raise TrueInputAllTenMoeGraphLauncherError("inner fixed ABI identity drifted")
    route_plan = _mapping(receipt.get("route_plan_binding"), "inner route_plan_binding")
    if route_plan.get("path") != context.route_plan["path"] or route_plan.get("sha256") != context.route_plan["sha256"]:
        raise TrueInputAllTenMoeGraphLauncherError("inner route-plan authority drifted")
    guard = _mapping(receipt.get("route_guard_readback"), "inner route_guard_readback")
    observed_ids = guard.get("observed_ids")
    expected_ids = guard.get("expected_ids")
    observed_weights = guard.get("observed_weights")
    expected_weights = guard.get("expected_weights")
    if (
        guard.get("value") != 1
        or guard.get("passed") is not True
        or observed_ids != list(context.route_ids)
        or expected_ids != list(context.route_ids)
        or not isinstance(observed_weights, list)
        or not isinstance(expected_weights, list)
        or len(observed_weights) != TOP_K
        or len(expected_weights) != TOP_K
    ):
        raise TrueInputAllTenMoeGraphLauncherError("inner route guard failed or route identity drifted")
    for index, (observed, expected, authoritative) in enumerate(
        zip(observed_weights, expected_weights, context.route_weights)
    ):
        if (
            isinstance(observed, bool)
            or isinstance(expected, bool)
            or not isinstance(observed, (int, float))
            or not isinstance(expected, (int, float))
            or not math.isfinite(float(observed))
            or not math.isfinite(float(expected))
            or abs(float(observed) - authoritative) > ROUTER_WEIGHT_TOLERANCE
            or abs(float(expected) - authoritative) > ROUTER_WEIGHT_TOLERANCE
        ):
            raise TrueInputAllTenMoeGraphLauncherError(f"inner route guard weight {index} drifted")
    parity = _mapping(receipt.get("readback_parity"), "inner readback_parity")
    if (
        parity.get("all_ten_route_witnesses") != TOP_K
        or parity.get("all_ten_route_cpu_device_parity_passed") is not True
        or parity.get("shared_expert_cpu_device_parity_passed") is not True
        or parity.get("routed_sum_cpu_device_parity_passed") is not True
        or parity.get("second_residual_cpu_device_parity_passed") is not True
    ):
        raise TrueInputAllTenMoeGraphLauncherError("inner all-ten/shared/second-residual parity is incomplete")
    policy = _mapping(receipt.get("metal_execution_policy"), "inner metal_execution_policy")
    lease = _mapping(policy.get("lease_binding"), "inner lease_binding")
    if (
        policy.get("strict_math_required") is not True
        or policy.get("timing_or_benchmarking_allowed") is not False
        or policy.get("complete_layer_or_token_allowed") is not False
        or policy.get("tps_or_tg_claim_allowed") is not False
        or lease.get("receipt_path") != context.lease_receipt["path"]
        or lease.get("receipt_document_sha256") != context.lease_receipt["sha256"]
        or lease.get("seal_sha256") != context.lease_seal_sha256
        or lease.get("schema") != LEASE_SCHEMA
        or lease.get("status") != LEASE_STATUS
    ):
        raise TrueInputAllTenMoeGraphLauncherError("inner fresh component lease binding drifted")


def _inner_evidence(config: LaunchConfig, context: LaunchContext) -> dict[str, Any]:
    capture = config.capture_dir / INNER_CAPTURE
    receipt_path = capture / "receipt.json"
    result: dict[str, Any] = {
        "capture_dir": str(capture),
        "receipt": {"path": str(receipt_path), "present": receipt_path.is_file()},
    }
    if not receipt_path.is_file():
        return result
    try:
        receipt = _read_json(receipt_path, "inner receipt")
        result.update(_file_evidence(receipt_path, "inner receipt"))
        result["schema"] = receipt.get("schema")
        result["status"] = receipt.get("status")
        _validate_inner(receipt, context)
    except TrueInputAllTenMoeGraphLauncherError as exc:
        result["binding_valid"] = False
        result["binding_error"] = str(exc)
    else:
        result["binding_valid"] = True
    return result


def _terminal_status(terminal: Mapping[str, Any], inner: Mapping[str, Any]) -> str:
    if terminal.get("spawn_error"):
        return "REFUSED_QWEN80_TRUE_INPUT_ALL_TEN_MOE_GRAPH_OUTER_CHILD_SPAWN_ERROR"
    if terminal.get("timed_out"):
        return "REFUSED_QWEN80_TRUE_INPUT_ALL_TEN_MOE_GRAPH_OUTER_CHILD_TIMEOUT"
    if terminal.get("signal") is not None:
        return "REFUSED_QWEN80_TRUE_INPUT_ALL_TEN_MOE_GRAPH_OUTER_CHILD_SIGNAL"
    if terminal.get("exit_code") != 0:
        return "REFUSED_QWEN80_TRUE_INPUT_ALL_TEN_MOE_GRAPH_OUTER_CHILD_NONZERO"
    if inner.get("binding_valid") is not True:
        return "REFUSED_QWEN80_TRUE_INPUT_ALL_TEN_MOE_GRAPH_OUTER_ZERO_EXIT_WITHOUT_STRICT_INNER_RECEIPT"
    return "CAPTURED_QWEN80_TRUE_INPUT_ALL_TEN_MOE_GRAPH_OUTER_TERMINAL_COMPONENT_ONLY"


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
) -> dict[str, Any]:
    inner = _inner_evidence(config, context)
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": _terminal_status(terminal, inner),
        "recorded_at": _utc_now(),
        "one_shot": {
            "automatic_retry_disabled": True,
            "same_capture_dir_never_starts_a_second_child": True,
            "terminal_receipt_written_last": True,
        },
        "launch_identity_sha256": identity,
        "source_binding": {
            "probe_binary": context.probe_binary,
            "manifest": context.manifest,
            "manifest_seal_sha256": context.manifest_seal_sha256,
            "admission_current": context.admission_current,
            "admission_pointer_seal_sha256": context.admission_pointer_seal_sha256,
            "admission_receipt_seal_sha256": context.admission_receipt_seal_sha256,
            "router_receipt": context.router_receipt,
            "router_outer_receipt": context.router_outer_receipt,
            "router_outer_receipt_seal_sha256": context.router_outer_seal_sha256,
            "route_plan": context.route_plan,
            "first_residual_receipt": context.first_residual_receipt,
            "first_residual_receipt_seal_sha256": context.first_residual_seal_sha256,
            "typed_bridge_receipt": context.typed_bridge_receipt,
            "typed_bridge_receipt_seal_sha256": context.typed_bridge_seal_sha256,
            "fixed_abi_contract": context.fixed_abi_contract,
            "lease_receipt": context.lease_receipt,
            "lease_receipt_seal_sha256": context.lease_seal_sha256,
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
            "requires_route_guard_and_all_readback_parity": True,
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
        raise TrueInputAllTenMoeGraphLauncherError("capture directory exists without terminal receipt")
    receipt, _ = _sealed_json(terminal_path, "outer terminal receipt")
    if receipt.get("schema") != SCHEMA or receipt.get("launch_identity_sha256") != identity:
        raise TrueInputAllTenMoeGraphLauncherError("capture directory belongs to another launch identity")
    return receipt


def run_attempt(config: LaunchConfig) -> dict[str, Any]:
    """Run one future device child, or sealed-replay exactly its terminal record."""

    context = _validate_config(config)
    identity = _launch_identity(config, context)
    if config.capture_dir.exists():
        return _replay(config, identity)
    try:
        config.capture_dir.mkdir(mode=0o750)
    except FileExistsError:
        return _replay(config, identity)
    command = _child_command(config, config.capture_dir / INNER_CAPTURE)
    started_at = _utc_now()
    _atomic_json_new(
        config.capture_dir / ACTIVE_FILENAME,
        seal({
            "schema": SCHEMA,
            "status": "STARTED_QWEN80_TRUE_INPUT_ALL_TEN_MOE_GRAPH_OUTER_ONE_SHOT",
            "recorded_at": started_at,
            "launch_identity_sha256": identity,
            "command": command,
            "claim_boundary": {"component_only": True, "automatic_retry_disabled": True},
        }),
    )
    child_pid: int | None = None
    capture_error: str | None = None
    with (config.capture_dir / OUTER_STDOUT).open("xb") as stdout, (
        config.capture_dir / OUTER_STDERR
    ).open("xb") as stderr:
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
                _atomic_json_new(
                    config.capture_dir / CHILD_FILENAME,
                    seal({
                        "schema": SCHEMA,
                        "status": "RUNNING_QWEN80_TRUE_INPUT_ALL_TEN_MOE_GRAPH_OUTER_ONE_SHOT",
                        "recorded_at": _utc_now(),
                        "launch_identity_sha256": identity,
                        "pid": child_pid,
                        "parent_pid": os.getpid(),
                        "command": command,
                        "mode": "metal",
                        "strict_component_lease_required": True,
                    }),
                )
            except TrueInputAllTenMoeGraphLauncherError as exc:
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
    )
    _atomic_json_new(config.capture_dir / TERMINAL_FILENAME, receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-bin", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--admission-current", type=Path, required=True)
    parser.add_argument("--router-receipt", type=Path, required=True)
    parser.add_argument("--router-outer-receipt", type=Path, required=True)
    parser.add_argument("--route-plan", type=Path, required=True)
    parser.add_argument("--first-residual-receipt", type=Path, required=True)
    parser.add_argument("--typed-bridge-receipt", type=Path, required=True)
    parser.add_argument("--fixed-abi-contract", type=Path, required=True)
    parser.add_argument("--lease-receipt", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parsed = build_parser().parse_args(argv)
    config = LaunchConfig(
        probe_bin=parsed.probe_bin,
        manifest=parsed.manifest,
        admission_current=parsed.admission_current,
        router_receipt=parsed.router_receipt,
        router_outer_receipt=parsed.router_outer_receipt,
        route_plan=parsed.route_plan,
        first_residual_receipt=parsed.first_residual_receipt,
        typed_bridge_receipt=parsed.typed_bridge_receipt,
        fixed_abi_contract=parsed.fixed_abi_contract,
        lease_receipt=parsed.lease_receipt,
        capture_dir=parsed.capture_dir,
        workers=parsed.workers,
        timeout_seconds=parsed.timeout_seconds,
    )
    try:
        receipt = run_attempt(config)
    except TrueInputAllTenMoeGraphLauncherError as exc:
        print(json.dumps({"status": "REFUSED_QWEN80_TRUE_INPUT_ALL_TEN_MOE_GRAPH_OUTER_LAUNCHER_ERROR", "error": str(exc)}))
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["status"].startswith("CAPTURED_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
