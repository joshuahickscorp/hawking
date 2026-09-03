"""Detached Qwen80 complete-runtime and TG3 dependency watchdogs.

The Qwen3-Coder-Next source body and its all-tensor binary baseline are real
physical inputs, but neither is a runtime by itself.  These restart-safe lanes
keep the exact hand-off path durable while the physical compiler progresses:

* the runtime lane binds the sealed source, complete-artifact progress, and
  Qwen-Next DeltaNet Metal component to the missing full hybrid decoder; and
* the TG lane rejects component timing and waits for a genuine complete-token,
  HCLI-capable, no-fallback runtime before it can emit any TG receipt.

They never generate tokens, benchmark throughput, or promote a contestant.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import stat
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from lab.receipts import verify


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "workspace/campaign/records/runs/qwen-80b/Qwen3-Coder-Next"
PHYSICAL_ROOT = REPO_ROOT / "workspace/campaign/records/ascension-sandbox/physical"
QWEN80_ROOT = PHYSICAL_ROOT / "qwen80"
ACQUISITION_ROOT = PHYSICAL_ROOT / "qwen80-acquisition"
COMPLETE_ROOT = QWEN80_ROOT / "complete-gravity"
RUNTIME_ROOT = QWEN80_ROOT / "complete-runtime"
TG3_ROOT = QWEN80_ROOT / "tg3"
SOURCE_AUDIT = ACQUISITION_ROOT / "QWEN80_SOURCE_BODY_AUDIT_CANDIDATE.json"
COMPLETE_MANIFEST = COMPLETE_ROOT / "QWEN80_COMPLETE_BINARY_GRAVITY_CANDIDATE.json"
COMPLETE_STATUS = COMPLETE_ROOT / "QWEN80_COMPLETE_GRAVITY_STATUS.json"
ADMISSION_RECEIPT = COMPLETE_ROOT / "QWEN80_COMPLETE_BINARY_GRAVITY_ADMISSION_RECEIPT.json"
ADMISSION_CURRENT_POINTER = (
    COMPLETE_ROOT / "QWEN80_COMPLETE_BINARY_GRAVITY_ADMISSION_CURRENT.json"
)
DELTANET_PROBE = PHYSICAL_ROOT / "kernel/QWEN_NEXT_GATED_DELTANET_METAL_COMPONENT_PROBE.json"
COMPLETE_BINARY_READER = REPO_ROOT / "crates/hawking-core/src/model/qwen_complete_binary.rs"
QWEN80_NATIVE_RUNTIME = REPO_ROOT / "crates/hawking-core/src/model/qwen80_complete_runtime.rs"
RUNTIME_PREFLIGHT_BINARY = (
    REPO_ROOT
    / "workspace/ops/build/rust/debug/examples/ascension_qwen80_complete_runtime_preflight"
)
RUNTIME_PREFLIGHT_RESULT = RUNTIME_ROOT / "QWEN80_COMPLETE_NATIVE_RUNTIME_PREFLIGHT.json"
SCHEMA = "hawking.ascension.qwen80_bootstrap_lanes.v1"
MODEL_ID = "Qwen3-Coder-Next-80B"
ARCHITECTURE = "Qwen3NextForCausalLM"
ADMISSION_SCHEMA = "hawking.ascension.qwen_complete_binary_gravity_admission_receipt.v1"
ADMISSION_STATUS = "EARNED_COMPLETE_BINARY_ARTIFACT_ADMITTED_NOT_RUNTIME_OR_CAPABILITY_QUALIFIED"
ADMISSION_CURRENT_POINTER_SCHEMA = (
    "hawking.ascension.qwen_complete_binary_gravity_admission_current_pointer.v1"
)
ADMISSION_CURRENT_POINTER_STATUS = "CURRENT_COMPLETE_BINARY_ADMISSION_RECEIPT_SELECTED"
PREFLIGHT_SCHEMA = "hawking.ascension.qwen80_complete_runtime_preflight_result.v1"
PREFLIGHT_CATALOG_STATUS = (
    "EARNED_QWEN80_COMPLETE_ARTIFACT_CATALOG_BOUND_NATIVE_HYBRID_DECODER_PENDING"
)
PREFLIGHT_STATE_STATUS = (
    "EARNED_QWEN80_COMPLETE_ARTIFACT_NATIVE_STATE_BOUND_HYBRID_DECODER_PENDING"
)
PREFLIGHT_DIRECT_PACKED_LINEAR_STAGE_STATUS = (
    "EARNED_QWEN80_ADMITTED_DIRECT_PACKED_FIRST_LINEAR_DELTANET_ROUTER_EXPERT_STAGE_"
    "NOT_FULL_LAYER_OR_TOKEN"
)
GPU_COORDINATION_HOLD_SCHEMA = "hawking.ascension.qwen80.watcher_gpu_coordination_hold.v1"
GPU_COORDINATION_HOLD_GLOB = "QWEN80_WATCHER_GPU_COORDINATION_HOLD_*.json"
GPU_COORDINATION_HELD_PREFIX = "HELD_QWEN80_"
GPU_COORDINATION_RELEASED_STATUS = "RELEASED_QWEN80_WATCHER_GPU_COORDINATION_HOLD"


def _preflight_status_for_mode(mode: str) -> str:
    statuses = {
        "catalog": PREFLIGHT_CATALOG_STATUS,
        "state": PREFLIGHT_STATE_STATUS,
        "direct-packed-linear-stage": PREFLIGHT_DIRECT_PACKED_LINEAR_STAGE_STATUS,
    }
    try:
        return statuses[mode]
    except KeyError as exc:
        raise Qwen80BootstrapError(f"unsupported native runtime preflight mode {mode!r}") from exc


def _preflight_result_key_for_mode(mode: str) -> str:
    """Map CLI modes to their sealed runtime-handoff result fields.

    The direct stage's CLI spelling intentionally contains hyphens, while the
    durable JSON handoff uses an underscore key.  Keeping this mapping explicit
    prevents a successful direct stage from being needlessly re-executed on
    every watcher pass.
    """

    keys = {
        "catalog": "catalog",
        "state": "state",
        "direct-packed-linear-stage": "direct_packed_linear_stage",
    }
    try:
        return keys[mode]
    except KeyError as exc:
        raise Qwen80BootstrapError(f"unsupported native runtime preflight mode {mode!r}") from exc


class Qwen80BootstrapError(RuntimeError):
    """A source-bound Qwen80 bootstrap lane cannot continue safely."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o640)
        os.replace(temporary, path)
        os.chmod(path, 0o640)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return dict(value) if isinstance(value, Mapping) else None


def _sha256(value: Any) -> str:
    payload = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _status(path: Path, *, lane: str, phase: str, **fields: Any) -> None:
    previous = _read_json(path) or {}
    _atomic_json(
        path,
        {
            "schema": SCHEMA,
            "recorded_at": _utc_now(),
            "pid": os.getpid(),
            "heartbeat": int(previous.get("heartbeat", 0)) + 1,
            "lane": lane,
            "phase": phase,
            **fields,
            "claim_boundary": {
                "raw_bf16_source_is_authority_teacher_not_tournament_participant": True,
                "complete_binary_candidate_is_not_a_runtime_or_density_qualification": True,
                "component_kernel_evidence_is_not_model_tps": True,
                "does_not_claim_full_token_tg3_or_manager_qualification": True,
            },
        },
    )


def _weight_map() -> dict[str, str]:
    document = _read_json(MODEL_DIR / "model.safetensors.index.json")
    weights = document.get("weight_map") if isinstance(document, Mapping) else None
    if not isinstance(weights, Mapping) or not weights:
        raise Qwen80BootstrapError("local Qwen80 safetensors index is unavailable")
    return {str(name): str(shard) for name, shard in weights.items()}


def _source_summary(weight_map: Mapping[str, str]) -> dict[str, Any]:
    layers = sorted(
        {
            int(parts[2])
            for name in weight_map
            if name.startswith("model.layers.")
            for parts in [name.split(".")]
            if len(parts) > 3 and parts[2].isdigit()
        }
    )
    missing_shards = sorted(
        shard for shard in set(weight_map.values()) if not (MODEL_DIR / shard).is_file()
    )
    return {
        "tensor_count": len(weight_map),
        "tensor_manifest_sha256": _sha256(dict(sorted(weight_map.items()))),
        "source_shard_count": len(set(weight_map.values())),
        "missing_source_shards": missing_shards,
        "layer_count": len(layers),
        "layers": layers,
        "gated_deltanet_tensor_count": sum(".linear_attn." in name for name in weight_map),
        "gated_attention_tensor_count": sum(".self_attn." in name for name in weight_map),
        "router_tensor_count": sum(name.endswith(".mlp.gate.weight") for name in weight_map),
        "routed_expert_tensor_count": sum(".mlp.experts." in name for name in weight_map),
        "shared_expert_tensor_count": sum(".mlp.shared_expert." in name for name in weight_map),
    }


def _verified_source_audit() -> dict[str, Any]:
    document = _read_json(SOURCE_AUDIT)
    if document is None:
        raise Qwen80BootstrapError(f"missing Qwen80 source audit: {SOURCE_AUDIT}")
    checked = verify(document, label=str(SOURCE_AUDIT))
    source = checked.get("source") if isinstance(checked.get("source"), Mapping) else {}
    if source.get("repository") != "Qwen/Qwen3-Coder-Next":
        raise Qwen80BootstrapError("source audit does not bind Qwen3-Coder-Next")
    return checked


def _candidate_progress() -> dict[str, Any]:
    status = _read_json(COMPLETE_STATUS) or {}
    progress = status.get("progress") if isinstance(status.get("progress"), Mapping) else {}
    manifest = _read_json(COMPLETE_MANIFEST)
    manifest_status = None
    manifest_seal = None
    if manifest is not None:
        try:
            checked = verify(manifest, label=str(COMPLETE_MANIFEST))
            manifest_status = checked.get("status")
            manifest_seal = checked.get("seal_sha256")
        except Exception as exc:
            manifest_status = f"INVALID_{type(exc).__name__}"
    return {
        "root": str(COMPLETE_ROOT),
        "status_phase": status.get("phase"),
        "progress": dict(progress),
        "manifest_path": str(COMPLETE_MANIFEST),
        "manifest_status": manifest_status or "NOT_YET_COMPLETE",
        "manifest_seal_sha256": manifest_seal,
        "all_tensor_artifact_complete": manifest_status
        == "CANDIDATE_COMPLETE_BINARY_ARTIFACT_LOW_FIDELITY_UNQUALIFIED",
    }


def _deltanet_component() -> dict[str, Any]:
    document = _read_json(DELTANET_PROBE)
    if document is None:
        return {"status": "ABSENT", "path": str(DELTANET_PROBE)}
    expected = "PASS_DIRECT_METAL_QWEN_NEXT_GATED_DELTANET_RECURRENCE_COMPONENT_NOT_FULL_MODEL_NOT_TPS_GATE"
    geometry = document.get("official_qwen_next_geometry") if isinstance(document.get("official_qwen_next_geometry"), Mapping) else {}
    exact = (
        document.get("status") == expected
        and geometry.get("heads") == 32
        and geometry.get("key_head_dim") == 128
        and geometry.get("value_head_dim") == 128
    )
    return {
        "status": "BYTE_HASH_BOUND_EXACT_RECURRENCE_COMPONENT" if exact else "INVALID_OR_UNEXPECTED",
        "path": str(DELTANET_PROBE),
        "sha256": _sha256(DELTANET_PROBE.read_bytes()),
        "geometry": dict(geometry),
        "claim_boundary": "recurrence state operator only; projection, convolution, attention, MoE, token loop, and TPS remain unimplemented",
    }


def _complete_binary_reader() -> dict[str, Any]:
    """Byte-bind the common all-tensor physical format reader as a primitive."""

    if not COMPLETE_BINARY_READER.is_file():
        return {"status": "ABSENT", "path": str(COMPLETE_BINARY_READER)}
    raw = COMPLETE_BINARY_READER.read_bytes()
    return {
        "status": "RUST_NATIVE_READER_SOURCE_PRESENT",
        "path": str(COMPLETE_BINARY_READER),
        "sha256": _sha256(raw),
        "claim_boundary": "strict tensor parser and diagnostic reconstruction only; no Qwen-Next hybrid layer catalog, Metal direct decode, model loader, or TPS result",
    }


def _regular_file_sha256(path: Path, label: str) -> str:
    """Hash one executable without allowing a symlink to change the handoff."""

    try:
        before = os.lstat(path)
    except OSError as exc:
        raise Qwen80BootstrapError(f"cannot stat {label}: {exc}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise Qwen80BootstrapError(f"{label} must be a regular non-symlink file: {path}")
    try:
        raw = path.read_bytes()
        after = os.lstat(path)
    except OSError as exc:
        raise Qwen80BootstrapError(f"cannot read {label}: {exc}") from exc
    if (
        stat.S_ISLNK(after.st_mode)
        or not stat.S_ISREG(after.st_mode)
        or before.st_size != after.st_size
        or len(raw) != before.st_size
    ):
        raise Qwen80BootstrapError(f"{label} changed while being read: {path}")
    return _sha256(raw)


def _active_qwen30_gpu_coordination_hold(
    binding: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return only the latest current-artifact Qwen30 coordination hold.

    The runtime watcher has no cross-process Metal lease around its bounded
    state/direct preflights.  Coordination is therefore an append-only,
    fail-closed control record: while the latest valid record says Qwen30 owns
    the window, this watcher may retain/reuse its catalog checkpoint but must
    not create a state or direct-Metal child.  A later matching release record
    is required to resume; stale or malformed records never authorize work.
    """

    candidates: list[tuple[datetime, str, dict[str, Any]]] = []
    for path in RUNTIME_ROOT.glob(GPU_COORDINATION_HOLD_GLOB):
        try:
            before = os.lstat(path)
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
                continue
            raw = path.read_bytes()
            after = os.lstat(path)
        except OSError:
            continue
        if (
            stat.S_ISLNK(after.st_mode)
            or not stat.S_ISREG(after.st_mode)
            or before.st_size != after.st_size
            or len(raw) != before.st_size
        ):
            continue
        try:
            document = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(document, Mapping):
            continue
        document = dict(document)
        recorded_at = document.get("recorded_at")
        status = document.get("status")
        source = document.get("source_binding")
        if (
            document.get("schema") != GPU_COORDINATION_HOLD_SCHEMA
            or not isinstance(recorded_at, str)
            or not recorded_at.endswith("Z")
            or not isinstance(status, str)
            or not isinstance(source, Mapping)
            or source.get("manifest_path") != binding.get("manifest_path")
            or source.get("manifest_seal_sha256") != binding.get("manifest_seal_sha256")
            or source.get("source_body_audit_seal_sha256")
            != binding.get("source_audit_seal_sha256")
            or source.get("admission_receipt_seal_sha256") != binding.get("receipt_seal_sha256")
            or source.get("source_revision") != binding.get("source_revision")
        ):
            continue
        try:
            timestamp = datetime.fromisoformat(recorded_at.replace("Z", "+00:00"))
        except ValueError:
            continue
        document["path"] = str(path)
        document["document_sha256"] = hashlib.sha256(raw).hexdigest()
        candidates.append((timestamp, path.name, document))
    if not candidates:
        return None
    _, _, latest = max(candidates, key=lambda item: (item[0], item[1]))
    if latest.get("status") == GPU_COORDINATION_RELEASED_STATUS:
        return None
    if not str(latest.get("status", "")).startswith(GPU_COORDINATION_HELD_PREFIX):
        return None
    coordination = latest.get("coordination")
    activity = coordination.get("qwen30_activity") if isinstance(coordination, Mapping) else None
    if not isinstance(activity, str) or "qwen30" not in activity.lower():
        return None
    return {
        "status": "ACTIVE_QWEN30_GPU_COORDINATION_HOLD",
        "record_path": latest["path"],
        "recorded_at": latest["recorded_at"],
        "record_sha256": latest["document_sha256"],
        "hold_status": latest["status"],
        "qwen30_activity": activity,
    }


def _admission_binding(candidate: Mapping[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Bind only the receipt selected for the live terminal manifest.

    The fixed public receipt is immutable historical evidence.  Qwen80 must
    not treat a valid older receipt as admission for a later resealed pack, so
    the runtime follows the sealed current-pointer contract introduced by the
    admission worker.  Both pointer and target receipt have to agree with the
    current terminal manifest before the native catalog executable can run.
    """

    pointer_document = _read_json(ADMISSION_CURRENT_POINTER)
    if pointer_document is None:
        return None, {
            "status": "WAITING_FOR_CURRENT_TERMINAL_NATIVE_ADMISSION_POINTER",
            "pointer_path": str(ADMISSION_CURRENT_POINTER),
            "historical_receipt_path": str(ADMISSION_RECEIPT),
        }
    try:
        pointer = verify(pointer_document, label=str(ADMISSION_CURRENT_POINTER))
    except Exception as exc:
        return None, {
            "status": f"INVALID_ADMISSION_CURRENT_POINTER_{type(exc).__name__}",
            "pointer_path": str(ADMISSION_CURRENT_POINTER),
        }
    pointer_model = pointer.get("model") if isinstance(pointer.get("model"), Mapping) else {}
    pointer_manifest = (
        pointer.get("complete_manifest")
        if isinstance(pointer.get("complete_manifest"), Mapping)
        else {}
    )
    receipt_binding = (
        pointer.get("admission_receipt")
        if isinstance(pointer.get("admission_receipt"), Mapping)
        else {}
    )
    if (
        pointer.get("schema") != ADMISSION_CURRENT_POINTER_SCHEMA
        or pointer.get("status") != ADMISSION_CURRENT_POINTER_STATUS
        or pointer.get("pointer_version") != 1
        or pointer_model.get("key") != "qwen80"
        or pointer_model.get("id") != MODEL_ID
        or pointer_model.get("repository") != "Qwen/Qwen3-Coder-Next"
        or not isinstance(pointer_model.get("revision"), str)
        or not pointer_model["revision"]
    ):
        return None, {
            "status": "INVALID_ADMISSION_CURRENT_POINTER_CONTRACT",
            "pointer_path": str(ADMISSION_CURRENT_POINTER),
        }
    if (
        pointer_manifest.get("path") != str(COMPLETE_MANIFEST)
        or pointer_manifest.get("seal_sha256") != candidate.get("manifest_seal_sha256")
        or not isinstance(pointer_manifest.get("document_sha256"), str)
        or len(pointer_manifest["document_sha256"]) != 64
        or not isinstance(pointer.get("admission_request_path"), str)
        or not isinstance(pointer.get("admission_request_seal_sha256"), str)
        or len(pointer["admission_request_seal_sha256"]) != 64
    ):
        return None, {
            "status": "CURRENT_ADMISSION_POINTER_DOES_NOT_BIND_TERMINAL_MANIFEST",
            "pointer_path": str(ADMISSION_CURRENT_POINTER),
            "terminal_manifest_seal_sha256": candidate.get("manifest_seal_sha256"),
        }
    receipt_path_value = receipt_binding.get("path")
    if not isinstance(receipt_path_value, str) or not receipt_path_value:
        return None, {
            "status": "INVALID_ADMISSION_CURRENT_POINTER_RECEIPT_PATH",
            "pointer_path": str(ADMISSION_CURRENT_POINTER),
        }
    receipt_path = Path(receipt_path_value)
    history_root = COMPLETE_ROOT / "complete-admission" / "receipts"
    try:
        resolved_receipt = receipt_path.resolve(strict=True)
        resolved_history_root = history_root.resolve(strict=True)
    except OSError:
        return None, {
            "status": "WAITING_FOR_CURRENT_TERMINAL_ADMISSION_RECEIPT",
            "pointer_path": str(ADMISSION_CURRENT_POINTER),
            "receipt_path": receipt_path_value,
        }
    try:
        resolved_receipt.relative_to(resolved_history_root)
    except ValueError:
        return None, {
            "status": "INVALID_ADMISSION_CURRENT_POINTER_RECEIPT_SCOPE",
            "pointer_path": str(ADMISSION_CURRENT_POINTER),
            "receipt_path": receipt_path_value,
        }
    document = _read_json(resolved_receipt)
    if document is None:
        return None, {
            "status": "WAITING_FOR_CURRENT_TERMINAL_ADMISSION_RECEIPT",
            "pointer_path": str(ADMISSION_CURRENT_POINTER),
            "receipt_path": str(resolved_receipt),
        }
    try:
        checked = verify(document, label=str(resolved_receipt))
        receipt_document_sha256 = _regular_file_sha256(
            resolved_receipt, "current Qwen80 native admission receipt"
        )
    except Exception as exc:
        return None, {
            "status": f"INVALID_CURRENT_TERMINAL_ADMISSION_RECEIPT_{type(exc).__name__}",
            "pointer_path": str(ADMISSION_CURRENT_POINTER),
            "receipt_path": str(resolved_receipt),
        }
    if (
        receipt_binding.get("document_sha256") != receipt_document_sha256
        or receipt_binding.get("seal_sha256") != checked.get("seal_sha256")
        or pointer.get("admission_request_path") != checked.get("admission_request_path")
        or pointer.get("admission_request_seal_sha256")
        != checked.get("admission_request_seal_sha256")
    ):
        return None, {
            "status": "CURRENT_ADMISSION_POINTER_AND_RECEIPT_DISAGREE",
            "pointer_path": str(ADMISSION_CURRENT_POINTER),
            "receipt_path": str(resolved_receipt),
        }
    model = checked.get("model") if isinstance(checked.get("model"), Mapping) else {}
    manifest = (
        checked.get("complete_manifest")
        if isinstance(checked.get("complete_manifest"), Mapping)
        else {}
    )
    revalidation = (
        checked.get("current_source_revalidation")
        if isinstance(checked.get("current_source_revalidation"), Mapping)
        else {}
    )
    if checked.get("schema") != ADMISSION_SCHEMA or checked.get("status") != ADMISSION_STATUS:
        return None, {
            "status": "INVALID_ADMISSION_RECEIPT_CONTRACT",
            "receipt_path": str(resolved_receipt),
        }
    if (
        model.get("key") != "qwen80"
        or model.get("id") != MODEL_ID
        or model.get("repository") != "Qwen/Qwen3-Coder-Next"
        or not isinstance(model.get("revision"), str)
        or not model["revision"]
    ):
        return None, {
            "status": "INVALID_ADMISSION_RECEIPT_MODEL_BINDING",
            "receipt_path": str(resolved_receipt),
        }
    if (
        manifest.get("path") != str(COMPLETE_MANIFEST)
        or manifest.get("document_sha256") != pointer_manifest.get("document_sha256")
        or manifest.get("seal_sha256") != candidate.get("manifest_seal_sha256")
        or revalidation.get("revision") != model.get("revision")
        or not isinstance(revalidation.get("source_audit_seal_sha256"), str)
        or len(revalidation["source_audit_seal_sha256"]) != 64
    ):
        return None, {
            "status": "INVALID_ADMISSION_RECEIPT_ARTIFACT_BINDING",
            "receipt_path": str(resolved_receipt),
        }
    return {
        "receipt_path": str(resolved_receipt),
        "receipt_seal_sha256": checked.get("seal_sha256"),
        "current_pointer_path": str(ADMISSION_CURRENT_POINTER),
        "current_pointer_seal_sha256": pointer.get("seal_sha256"),
        "manifest_path": str(COMPLETE_MANIFEST),
        "manifest_seal_sha256": manifest["seal_sha256"],
        "source_audit_seal_sha256": revalidation["source_audit_seal_sha256"],
        "source_revision": model["revision"],
    }, {"status": "CURRENT_TERMINAL_ADMISSION_RECEIPT_BOUND"}


def _run_native_runtime_preflight(
    *,
    mode: str,
    binding: Mapping[str, Any],
    timeout_seconds: float = 1800.0,
) -> dict[str, Any]:
    """Execute the Rust preflight through the exact admitted binary contract."""

    expected_status = _preflight_status_for_mode(mode)
    if timeout_seconds <= 0:
        raise Qwen80BootstrapError("native runtime preflight timeout must be positive")
    if not RUNTIME_PREFLIGHT_BINARY.is_file():
        return {
            "status": "WAITING_FOR_QWEN80_NATIVE_RUNTIME_PREFLIGHT_BINARY",
            "binary_path": str(RUNTIME_PREFLIGHT_BINARY),
        }
    try:
        before_sha256 = _regular_file_sha256(
            RUNTIME_PREFLIGHT_BINARY, "Qwen80 native runtime preflight executable"
        )
    except Qwen80BootstrapError as exc:
        return {
            "status": "INVALID_QWEN80_NATIVE_RUNTIME_PREFLIGHT_BINARY",
            "binary_path": str(RUNTIME_PREFLIGHT_BINARY),
            "detail": str(exc),
        }
    if not os.access(RUNTIME_PREFLIGHT_BINARY, os.X_OK):
        return {
            "status": "NONEXECUTABLE_QWEN80_NATIVE_RUNTIME_PREFLIGHT_BINARY",
            "binary_path": str(RUNTIME_PREFLIGHT_BINARY),
        }
    command = [
        str(RUNTIME_PREFLIGHT_BINARY),
        "--manifest",
        str(binding["manifest_path"]),
        "--expected-manifest-seal-sha256",
        str(binding["manifest_seal_sha256"]),
        "--expected-source-audit-seal-sha256",
        str(binding["source_audit_seal_sha256"]),
        "--expected-source-revision",
        str(binding["source_revision"]),
        "--mode",
        mode,
        "--max-seq-len",
        "256",
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "QWEN80_NATIVE_RUNTIME_PREFLIGHT_TIMEOUT",
            "mode": mode,
            "timeout_seconds": timeout_seconds,
        }
    except OSError as exc:
        return {
            "status": "QWEN80_NATIVE_RUNTIME_PREFLIGHT_EXECUTION_ERROR",
            "mode": mode,
            "detail": str(exc),
        }
    try:
        after_sha256 = _regular_file_sha256(
            RUNTIME_PREFLIGHT_BINARY, "Qwen80 native runtime preflight executable"
        )
    except Qwen80BootstrapError as exc:
        return {
            "status": "QWEN80_NATIVE_RUNTIME_PREFLIGHT_BINARY_CHANGED",
            "mode": mode,
            "detail": str(exc),
        }
    if before_sha256 != after_sha256:
        return {
            "status": "QWEN80_NATIVE_RUNTIME_PREFLIGHT_BINARY_CHANGED",
            "mode": mode,
        }
    if completed.returncode != 0:
        return {
            "status": "QWEN80_NATIVE_RUNTIME_PREFLIGHT_REFUSED",
            "mode": mode,
            "exit_code": completed.returncode,
            "detail": (completed.stderr or completed.stdout or "no diagnostic").strip()[:1000],
            "binary_sha256": before_sha256,
        }
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {
            "status": "QWEN80_NATIVE_RUNTIME_PREFLIGHT_INVALID_OUTPUT",
            "mode": mode,
            "binary_sha256": before_sha256,
        }
    model = result.get("model") if isinstance(result, Mapping) else None
    if (
        not isinstance(result, Mapping)
        or result.get("schema") != PREFLIGHT_SCHEMA
        or result.get("status") != expected_status
        or not isinstance(model, Mapping)
        or model.get("key") != "qwen80"
        or model.get("revision") != binding["source_revision"]
        or result.get("manifest_path") != binding["manifest_path"]
        or result.get("manifest_seal_sha256") != binding["manifest_seal_sha256"]
    ):
        return {
            "status": "QWEN80_NATIVE_RUNTIME_PREFLIGHT_INVALID_CONTRACT",
            "mode": mode,
            "binary_sha256": before_sha256,
        }
    return {
        "status": expected_status,
        "mode": mode,
        "binary_path": str(RUNTIME_PREFLIGHT_BINARY),
        "binary_sha256": before_sha256,
        "result": dict(result),
    }


def _cached_or_run_native_runtime_preflight(
    *,
    mode: str,
    binding: Mapping[str, Any],
    existing: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Reuse only a result bound to this receipt and current exact binary."""

    current_binary_sha256: str | None = None
    if RUNTIME_PREFLIGHT_BINARY.is_file():
        try:
            current_binary_sha256 = _regular_file_sha256(
                RUNTIME_PREFLIGHT_BINARY, "Qwen80 native runtime preflight executable"
            )
        except Qwen80BootstrapError:
            current_binary_sha256 = None
    prior = (
        existing.get(_preflight_result_key_for_mode(mode))
        if isinstance(existing, Mapping)
        else None
    )
    expected_status = _preflight_status_for_mode(mode)
    if (
        isinstance(prior, Mapping)
        and prior.get("status") == expected_status
        and prior.get("binary_sha256") == current_binary_sha256
        and existing.get("admission_receipt_seal_sha256") == binding.get("receipt_seal_sha256")
        and existing.get("manifest_seal_sha256") == binding.get("manifest_seal_sha256")
    ):
        return dict(prior)
    return _run_native_runtime_preflight(mode=mode, binding=binding)


def run_runtime_cycle() -> None:
    status_path = RUNTIME_ROOT / "QWEN80_COMPLETE_RUNTIME_STATUS.json"
    _status(status_path, lane="QWEN80_COMPLETE_HYBRID_RUNTIME", phase="SOURCE_AND_PHYSICAL_ASSEMBLY_AUDIT_RUNNING")
    audit = _verified_source_audit()
    source = _source_summary(_weight_map())
    candidate = _candidate_progress()
    kernel = _deltanet_component()
    native_runtime_source = {
        "path": str(QWEN80_NATIVE_RUNTIME),
        "sha256": _sha256(QWEN80_NATIVE_RUNTIME.read_bytes())
        if QWEN80_NATIVE_RUNTIME.is_file()
        else None,
        "status": "SOURCE_PRESENT" if QWEN80_NATIVE_RUNTIME.is_file() else "SOURCE_MISSING",
    }
    admission, admission_state = _admission_binding(candidate)
    transition = "WAITING_FOR_COMPLETE_BINARY_PACK"
    runtime_preflight: dict[str, Any] | None = None
    catalog_preflight: dict[str, Any] | None = None
    state_preflight: dict[str, Any] | None = None
    direct_packed_linear_stage: dict[str, Any] | None = None
    gpu_coordination_hold: dict[str, Any] | None = None
    report_status = "WAITING_FOR_COMPLETE_BINARY_PACK_BEFORE_NATIVE_RUNTIME_PREFLIGHT"
    phase = "WAITING_FOR_COMPLETE_BINARY_PACK"
    if candidate.get("all_tensor_artifact_complete") is True and admission is None:
        transition = "WAITING_FOR_NATIVE_COMPLETE_ARTIFACT_ADMISSION"
        report_status = "COMPLETE_PACK_WAITING_FOR_INDEPENDENT_NATIVE_ADMISSION"
        phase = "WAITING_FOR_NATIVE_COMPLETE_ARTIFACT_ADMISSION"
    elif candidate.get("all_tensor_artifact_complete") is True and admission is not None:
        transition = "ADMITTED_ARTIFACT_TO_NATIVE_CATALOG_TO_NATIVE_STATE_TO_FULL_HYBRID_DECODER"
        existing = _read_json(RUNTIME_PREFLIGHT_RESULT)
        _status(
            status_path,
            lane="QWEN80_COMPLETE_HYBRID_RUNTIME",
            phase="NATIVE_COMPLETE_ARTIFACT_CATALOG_PREFLIGHT_RUNNING",
            admission=admission_state,
            automatic_transition=transition,
            current_artifact=str(RUNTIME_PREFLIGHT_RESULT),
        )
        catalog_preflight = _cached_or_run_native_runtime_preflight(
            mode="catalog", binding=admission, existing=existing
        )
        runtime_preflight = {
            "schema": "hawking.ascension.qwen80_native_runtime_handoff.v1",
            "recorded_at": _utc_now(),
            "admission_receipt_seal_sha256": admission["receipt_seal_sha256"],
            "manifest_seal_sha256": admission["manifest_seal_sha256"],
            "catalog": catalog_preflight,
        }
        # Persist each earned/bounded stage before the next strict process is
        # launched. If a detached worker is interrupted, its next cycle can
        # reuse only this exact binary/receipt-bound checkpoint rather than
        # showing a stale heartbeat or claiming a downstream stage.
        _atomic_json(RUNTIME_PREFLIGHT_RESULT, runtime_preflight)
        if catalog_preflight.get("status") == PREFLIGHT_CATALOG_STATUS:
            gpu_coordination_hold = _active_qwen30_gpu_coordination_hold(admission)
            if gpu_coordination_hold is not None:
                runtime_preflight["gpu_coordination_hold"] = gpu_coordination_hold
                _atomic_json(RUNTIME_PREFLIGHT_RESULT, runtime_preflight)
                report_status = (
                    "ADMITTED_COMPLETE_ARTIFACT_NATIVE_CATALOG_BOUND_WAITING_FOR_"
                    "COORDINATED_GPU_LEASE"
                )
                phase = "WAITING_FOR_COORDINATED_GPU_LEASE"
                _status(
                    status_path,
                    lane="QWEN80_COMPLETE_HYBRID_RUNTIME",
                    phase=phase,
                    admission=admission_state,
                    automatic_transition=transition,
                    gpu_coordination_hold=gpu_coordination_hold,
                    native_runtime_preflight={"catalog_status": catalog_preflight.get("status")},
                    current_artifact=str(RUNTIME_PREFLIGHT_RESULT),
                )
            else:
                _status(
                    status_path,
                    lane="QWEN80_COMPLETE_HYBRID_RUNTIME",
                    phase="NATIVE_CATALOG_BOUND_HYBRID_STATE_PREFLIGHT_RUNNING",
                    admission=admission_state,
                    automatic_transition=transition,
                    native_runtime_preflight={"catalog_status": catalog_preflight.get("status")},
                    current_artifact=str(RUNTIME_PREFLIGHT_RESULT),
                )
                state_preflight = _cached_or_run_native_runtime_preflight(
                    mode="state", binding=admission, existing=existing
                )
                runtime_preflight["state"] = state_preflight
                _atomic_json(RUNTIME_PREFLIGHT_RESULT, runtime_preflight)
                if state_preflight.get("status") == PREFLIGHT_STATE_STATUS:
                    _status(
                        status_path,
                        lane="QWEN80_COMPLETE_HYBRID_RUNTIME",
                        phase="NATIVE_CATALOG_AND_STATE_BOUND_DIRECT_PACKED_FIRST_LINEAR_STAGE_RUNNING",
                        admission=admission_state,
                        automatic_transition=transition,
                        native_runtime_preflight={
                            "catalog_status": catalog_preflight.get("status"),
                            "state_status": state_preflight.get("status"),
                        },
                        current_artifact=str(RUNTIME_PREFLIGHT_RESULT),
                    )
                    direct_packed_linear_stage = _cached_or_run_native_runtime_preflight(
                        mode="direct-packed-linear-stage", binding=admission, existing=existing
                    )
                    runtime_preflight["direct_packed_linear_stage"] = direct_packed_linear_stage
                    _atomic_json(RUNTIME_PREFLIGHT_RESULT, runtime_preflight)
                    if (
                        direct_packed_linear_stage.get("status")
                        == PREFLIGHT_DIRECT_PACKED_LINEAR_STAGE_STATUS
                    ):
                        report_status = (
                            "ADMITTED_COMPLETE_ARTIFACT_NATIVE_CATALOG_STATE_AND_DIRECT_PACKED_"
                            "FIRST_LINEAR_STAGE_BOUND_FULL_HYBRID_DECODER_PENDING"
                        )
                        phase = "NATIVE_CATALOG_STATE_AND_DIRECT_PACKED_FIRST_LINEAR_STAGE_BOUND"
                    else:
                        report_status = (
                            "ADMITTED_COMPLETE_ARTIFACT_NATIVE_CATALOG_AND_STATE_BOUND_"
                            "DIRECT_PACKED_FIRST_LINEAR_STAGE_PENDING_OR_REFUSED"
                        )
                        phase = "NATIVE_CATALOG_AND_STATE_BOUND_DIRECT_PACKED_FIRST_LINEAR_STAGE_PENDING"
                else:
                    report_status = "ADMITTED_COMPLETE_ARTIFACT_CATALOG_BOUND_NATIVE_STATE_PENDING"
                    phase = "NATIVE_CATALOG_BOUND_HYBRID_STATE_PENDING"
        else:
            report_status = "ADMITTED_COMPLETE_ARTIFACT_NATIVE_CATALOG_PREFLIGHT_PENDING_OR_REFUSED"
            phase = "ADMITTED_COMPLETE_ARTIFACT_NATIVE_CATALOG_PREFLIGHT_PENDING"
    report = {
        "schema": "hawking.ascension.qwen80_complete_hybrid_assembly.v1",
        "status": report_status,
        "recorded_at": _utc_now(),
        "model": {"id": MODEL_ID, "architecture": ARCHITECTURE},
        "source": source,
        "source_body_audit_seal_sha256": audit.get("seal_sha256"),
        "physical_complete_binary_candidate": candidate,
        "native_complete_artifact_admission": admission_state,
        "native_complete_artifact_binding": admission,
        "native_complete_binary_reader": _complete_binary_reader(),
        "native_complete_runtime_source": native_runtime_source,
        "native_runtime_preflight_path": str(RUNTIME_PREFLIGHT_RESULT),
        "native_runtime_preflight": runtime_preflight,
        "direct_metal_deltanet_component": kernel,
        "full_artifact_contract": {
            "all_tensors_required": True,
            "complete_physical_bpw_required_at_most": 1.5,
            "forbids_component_only_bpw": True,
            "requires_native_packed_metal_execution": True,
            "requires_exact_hybrid_layer_schedule": True,
            "requires_hcli_then_tg3": True,
        },
        "native_runtime_implementation_backlog": [
            "gated_deltanet_projection_convolution_norm_and_recurrent_state_execution",
            "gated_attention_rope_kv_cache_and_hybrid_layer_schedule_execution",
            "512_expert_top10_router_shared_expert_and_native_packed_projection_waves",
            "complete_token_command_graph_sampling_and_no_fallback_receipt",
            "native_direct_HCLI_generation_endpoint_with_prompt_dependent_parity_and_rollback",
        ],
        "claim_boundary": {
            "this_is_an_assembly_handoff_not_a_complete_runtime": True,
            "does_not_load_raw_model_body_into_tournament_runtime": True,
            "does_not_overwrite_or_substitute_for_a_future_full_token_runtime_receipt": True,
            "does_not_clear_density_100_tps_or_tg3": True,
        },
    }
    _atomic_json(RUNTIME_ROOT / "QWEN80_COMPLETE_HYBRID_ASSEMBLY_CANDIDATE.json", report)
    _status(
        status_path,
        lane="QWEN80_COMPLETE_HYBRID_RUNTIME",
        phase=phase,
        source=source,
        physical_candidate=candidate,
        admission=admission_state,
        automatic_transition=transition,
        native_runtime_preflight=(
            {
                "catalog_status": catalog_preflight.get("status")
                if isinstance(catalog_preflight, Mapping)
                else None,
                "state_status": state_preflight.get("status")
                if isinstance(state_preflight, Mapping)
                else None,
                "direct_packed_linear_stage_status": direct_packed_linear_stage.get("status")
                if isinstance(direct_packed_linear_stage, Mapping)
                else None,
            }
            if runtime_preflight is not None
            else None
        ),
        deltanet_component_available=kernel.get("status") == "BYTE_HASH_BOUND_EXACT_RECURRENCE_COMPONENT",
        current_artifact=str(RUNTIME_ROOT / "QWEN80_COMPLETE_HYBRID_ASSEMBLY_CANDIDATE.json"),
    )


def run_tg3_cycle() -> None:
    status_path = TG3_ROOT / "QWEN80_TG3_ASCENT_STATUS.json"
    _status(status_path, lane="QWEN80_METAL_TG3", phase="COMPLETE_TOKEN_GATE_AUDIT_RUNNING")
    runtime = _read_json(RUNTIME_ROOT / "QWEN80_COMPLETE_HYBRID_ASSEMBLY_CANDIDATE.json")
    report = {
        "schema": "hawking.ascension.qwen80_tg3_ascent.v1",
        "status": "FULL_TOKEN_TG3_BLOCKED_NATIVE_QWEN80_HYBRID_DECODER_UNIMPLEMENTED",
        "recorded_at": _utc_now(),
        "complete_hybrid_assembly_input": runtime,
        "direct_metal_deltanet_component": _deltanet_component(),
        "required_pre_tournament_kernel_gate": {
            "each_candidate_requires_exact_model_custom_kernel": True,
            "operational_base_true_tps_minimum": 100.0,
            "is_not_a_component_timing_measurement": True,
        },
        "required_clean_benchmark": {
            "run_model_alone": True,
            "base_true_tps_minimum": 333.0,
            "no_fallback": True,
            "complete_token_timing": True,
            "hcli_prompt_dependent_generation": True,
        },
        "blocking_native_operators": [
            "full_qwen3next_complete_binary_gravity_loader",
            "hybrid_deltanet_and_attention_48_layer_execution",
            "all_512_expert_top10_and_shared_expert_native_packed_execution",
            "complete_token_command_graph_and_sampling",
            "HCLI_generation_runtime",
        ],
        "claim_boundary": {
            "deltanet_component_timing_is_not_model_tps": True,
            "no_100_tps_or_tg3_receipt_emitted": True,
        },
    }
    _atomic_json(TG3_ROOT / "QWEN80_TG3_ASCENT_CANDIDATE.json", report)
    _status(
        status_path,
        lane="QWEN80_METAL_TG3",
        phase="WAITING_FOR_NATIVE_COMPLETE_TOKEN_HYBRID_RUNTIME",
        current_artifact=str(TG3_ROOT / "QWEN80_TG3_ASCENT_CANDIDATE.json"),
    )


def watch(kind: str, *, idle_seconds: float) -> int:
    if idle_seconds <= 0:
        raise Qwen80BootstrapError("idle_seconds must be positive")
    stopping = False

    def stop(_signal: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True

    old_term = signal.signal(signal.SIGTERM, stop)
    old_int = signal.signal(signal.SIGINT, stop)
    try:
        while not stopping:
            if kind == "runtime":
                run_runtime_cycle()
            else:
                run_tg3_cycle()
            if not stopping:
                time.sleep(idle_seconds)
    finally:
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGINT, old_int)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for kind in ("runtime", "tg3"):
        item = commands.add_parser(kind, help=f"run one Qwen80 {kind} bootstrap cycle")
        item.add_argument("--watch", action="store_true")
        item.add_argument("--idle-seconds", type=float, default=300.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.watch:
        return watch(args.command, idle_seconds=args.idle_seconds)
    if args.command == "runtime":
        run_runtime_cycle()
    else:
        run_tg3_cycle()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
