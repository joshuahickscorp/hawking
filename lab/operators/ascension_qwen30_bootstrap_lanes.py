"""Dedicated Qwen30 complete-artifact and TG3 bootstrap lanes.

These lanes consume the real local Qwen30 source audit and the bounded Gravity
frontier.  They make the missing work explicit and durable without treating a
source manifest or a router component probe as a runnable model, a complete
BPW result, or a TG3 receipt.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import signal
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from lab.receipts import seal, verify


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "workspace/campaign/records/runs/qwen-30b/Qwen3-Coder-30B-A3B-Instruct"
PHYSICAL_ROOT = REPO_ROOT / "workspace/campaign/records/ascension-sandbox/physical"
QWEN30_ROOT = PHYSICAL_ROOT / "qwen30"
COMPLETE_ROOT = QWEN30_ROOT / "complete-gravity"
RUNTIME_ROOT = QWEN30_ROOT / "complete-runtime"
TG3_ROOT = QWEN30_ROOT / "tg3"
QWEN30_GQA_PROBE = PHYSICAL_ROOT / "kernel" / "QWEN30_GQA_METAL_COMPONENT_PROBE.json"
COMPLETE_BINARY_READER = REPO_ROOT / "crates/hawking-core/src/model/qwen_complete_binary.rs"
QWEN30_NATIVE_RUNTIME_EXECUTABLE = (
    REPO_ROOT / "workspace/ops/build/rust/debug/examples/ascension_qwen30_complete_native_runtime"
)
QWEN30_NATIVE_HTTP_SERVER = (
    REPO_ROOT / "workspace/ops/build/rust/debug/examples/ascension_qwen30_native_http_server"
)
QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_HTTP_SERVER_SOURCE = (
    REPO_ROOT
    / "workspace/ops/build/rust-qwen30-paired-production/debug/examples/"
    "ascension_qwen30_native_http_server"
)
QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_HTTP_ADAPTER_DEPLOYMENT = (
    RUNTIME_ROOT / "QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_HTTP_ADAPTER_DEPLOYMENT.json"
)
QWEN30_NATIVE_HTTP_SERVER_HISTORY = RUNTIME_ROOT / "native-http-server-history"
# This target directory is intentionally separate from the detached control
# runtime and its HTTP adapter.  Candidate parity can therefore compile and
# execute a changed binary without replacing the scalar production control or
# causing the watcher to reinterpret an old control receipt as new evidence.
QWEN30_GATEUP_FUSED_CANDIDATE_EXECUTABLE = (
    REPO_ROOT
    / "workspace/ops/build/rust-qwen30-gateup-parity/debug/examples/ascension_qwen30_complete_native_runtime"
)
QWEN30_ADMISSION_RECEIPT = COMPLETE_ROOT / "QWEN30_COMPLETE_BINARY_GRAVITY_ADMISSION_RECEIPT.json"
QWEN30_SOURCE_IDENTITY = QWEN30_ROOT / "evolution" / "SOURCE_CONTENT_IDENTITY.json"
QWEN30_SOURCE_REVALIDATION = COMPLETE_ROOT / "QWEN30_CURRENT_SOURCE_SHARD_REVALIDATION.json"
QWEN30_NATIVE_ACTIVE = RUNTIME_ROOT / "QWEN30_NATIVE_RUNTIME_ACTIVE.json"
QWEN30_NATIVE_LAST_PROCESS = RUNTIME_ROOT / "QWEN30_NATIVE_RUNTIME_LAST_PROCESS.json"
QWEN30_NATIVE_HTTP_ACTIVE = RUNTIME_ROOT / "QWEN30_NATIVE_HTTP_ADAPTER_ACTIVE.json"
QWEN30_NATIVE_HTTP_LAST_PROCESS = RUNTIME_ROOT / "QWEN30_NATIVE_HTTP_ADAPTER_LAST_PROCESS.json"
QWEN30_NATIVE_HTTP_STATUS = RUNTIME_ROOT / "QWEN30_NATIVE_HTTP_ADAPTER_STATUS.json"
QWEN30_NATIVE_HTTP_TRANSPORT_SMOKE = (
    RUNTIME_ROOT / "QWEN30_NATIVE_HTTP_ADAPTER_TRANSPORT_SMOKE_RECEIPT.json"
)
QWEN30_NATIVE_HTTP_TRANSPORT_SMOKE_HISTORY = RUNTIME_ROOT / "transport-smoke-history"
QWEN30_NATIVE_HTTP_CHAT_SMOKE = (
    RUNTIME_ROOT / "QWEN30_HCLI_UNQUALIFIED_CHAT_SSE_TRANSPORT_RECEIPT.json"
)
QWEN30_NATIVE_HTTP_CHAT_SMOKE_HISTORY = RUNTIME_ROOT / "chat-sse-smoke-history"
QWEN30_NATIVE_PREFLIGHT = RUNTIME_ROOT / "QWEN30_COMPLETE_NATIVE_RUNTIME_PREFLIGHT.json"
QWEN30_NATIVE_FULL_TOKEN = RUNTIME_ROOT / "QWEN30_DIRECT_PACKED_NATIVE_FULL_TOKEN_RESULT.json"
QWEN30_NATIVE_PROMPT_A = RUNTIME_ROOT / "QWEN30_DIRECT_PACKED_NATIVE_PROMPT_A_RESULT.json"
QWEN30_NATIVE_PROMPT_B = RUNTIME_ROOT / "QWEN30_DIRECT_PACKED_NATIVE_PROMPT_B_RESULT.json"
QWEN30_NATIVE_PROFILE_TOKEN = RUNTIME_ROOT / "QWEN30_DIRECT_PACKED_NATIVE_KERNEL_PROFILE_RESULT.json"
QWEN30_ROUTE_MAJOR_INPUT_OFFSET_METAL_REGRESSION = (
    RUNTIME_ROOT / "QWEN30_ROUTE_MAJOR_INPUT_OFFSET_METAL_REGRESSION_RECEIPT.json"
)
QWEN30_ROUTE_MAJOR_DEFECT_HISTORY = RUNTIME_ROOT / "route-major-defect-history"
QWEN30_ROUTE_MAJOR_DEFECT_HISTORY_MANIFEST = (
    QWEN30_ROUTE_MAJOR_DEFECT_HISTORY / "QWEN30_ROUTE_MAJOR_DEFECT_HISTORY_MANIFEST.json"
)
QWEN30_NATIVE_PROFILE_PARTIAL_NEGATIVE = (
    RUNTIME_ROOT / "QWEN30_DIRECT_PACKED_NATIVE_KERNEL_PROFILE_REJECTED_PARTIAL_GPU_TRACE.json"
)
QWEN30_SIMDGROUP_COMPONENT_PARITY = RUNTIME_ROOT / "QWEN30_PACKED_BINARY_SIMDGROUP_COMPONENT_PARITY.json"
QWEN30_NATIVE_SIMDGROUP_CANDIDATE_TOKEN = (
    RUNTIME_ROOT / "QWEN30_DIRECT_PACKED_NATIVE_SIMDGROUP_CANDIDATE_TOKEN_RESULT.json"
)
QWEN30_NATIVE_SIMDGROUP_CANDIDATE_PROMPT_A = (
    RUNTIME_ROOT / "QWEN30_DIRECT_PACKED_NATIVE_SIMDGROUP_CANDIDATE_PROMPT_A_RESULT.json"
)
QWEN30_NATIVE_SIMDGROUP_CANDIDATE_PROMPT_B = (
    RUNTIME_ROOT / "QWEN30_DIRECT_PACKED_NATIVE_SIMDGROUP_CANDIDATE_PROMPT_B_RESULT.json"
)
QWEN30_SIMDGROUP_TEMPLATE_PARITY = (
    RUNTIME_ROOT / "QWEN30_PACKED_BINARY_SIMDGROUP_TEMPLATE_PARITY_RECEIPT.json"
)
QWEN30_SIMDGROUP_TEMPLATE_PARITY_HISTORY = RUNTIME_ROOT / "simdgroup-template-parity-history"
QWEN30_SIMDGROUP_TEMPLATE_PARITY_ACTIVE = (
    RUNTIME_ROOT / "QWEN30_PACKED_BINARY_SIMDGROUP_TEMPLATE_PARITY_ACTIVE.json"
)
QWEN30_SIMDGROUP_TEMPLATE_PARITY_LAST = (
    RUNTIME_ROOT / "QWEN30_PACKED_BINARY_SIMDGROUP_TEMPLATE_PARITY_LAST_PROCESS.json"
)
QWEN30_GATEUP_FUSED_COMPONENT_RAW = (
    QWEN30_ROOT
    / "complete-token-profiler"
    / "QWEN30_DIRECT_PACKED_GATE_UP_SWIGLU_FUSED_COMPONENT_RAW_RESULT.json"
)
QWEN30_NATIVE_GATEUP_FUSED_CANDIDATE_PROMPT_A = (
    RUNTIME_ROOT / "QWEN30_DIRECT_PACKED_GATE_UP_SWIGLU_FUSED_CANDIDATE_PROMPT_A_RESULT.json"
)
QWEN30_NATIVE_GATEUP_FUSED_CANDIDATE_PROMPT_B = (
    RUNTIME_ROOT / "QWEN30_DIRECT_PACKED_GATE_UP_SWIGLU_FUSED_CANDIDATE_PROMPT_B_RESULT.json"
)
QWEN30_GATEUP_FUSED_TEMPLATE_PARITY = (
    RUNTIME_ROOT / "QWEN30_DIRECT_PACKED_GATE_UP_SWIGLU_FUSED_TEMPLATE_PARITY_RECEIPT.json"
)
QWEN30_GATEUP_FUSED_TEMPLATE_PARITY_ACTIVE = (
    RUNTIME_ROOT / "QWEN30_DIRECT_PACKED_GATE_UP_SWIGLU_FUSED_TEMPLATE_PARITY_ACTIVE.json"
)
QWEN30_GATEUP_FUSED_TEMPLATE_PARITY_LAST = (
    RUNTIME_ROOT / "QWEN30_DIRECT_PACKED_GATE_UP_SWIGLU_FUSED_TEMPLATE_PARITY_LAST_PROCESS.json"
)
QWEN30_GATEUP_PAIRED_SCALAR_ORDER_CPU_PARITY = (
    QWEN30_ROOT
    / "complete-token-profiler"
    / "QWEN30_DIRECT_PACKED_GATE_UP_PAIRED_SCALAR_ORDER_CPU_PARITY_RECEIPT.json"
)
QWEN30_GATEUP_PAIRED_SCALAR_ORDER_TEMPLATE_PARITY = (
    RUNTIME_ROOT
    / "QWEN30_DIRECT_PACKED_GATE_UP_PAIRED_SCALAR_ORDER_TEMPLATE_PARITY_RECEIPT.json"
)
QWEN30_GATEUP_PAIRED_SCALAR_ORDER_PRAGMA_SUCCESSOR_TEMPLATE_PARITY = (
    RUNTIME_ROOT
    / "QWEN30_DIRECT_PACKED_GATE_UP_PAIRED_SCALAR_ORDER_MSL_PRAGMA_SUCCESSOR_TEMPLATE_PARITY_RECEIPT.json"
)
QWEN_FAMILY_GPU_LEASE_ROOT = PHYSICAL_ROOT / "qwen-family" / "dual-gravity"
QWEN_FAMILY_GPU_LEASE_LOCK = QWEN_FAMILY_GPU_LEASE_ROOT / ".gpu-lease.lock"
QWEN_FAMILY_GPU_LEASE_STATUS = QWEN_FAMILY_GPU_LEASE_ROOT / "GPU_LEASE_STATUS.json"
QWEN30_EXACT_FULL_TOKEN_RUNTIME_RECEIPT = (
    RUNTIME_ROOT / "QWEN30_EXACT_FULL_TOKEN_RUNTIME_RECEIPT.json"
)
QWEN30_EXACT_FULL_TOKEN_RUNTIME_HISTORY = RUNTIME_ROOT / "runtime-receipt-history"
QWEN30_EXACT_FULL_TOKEN_RUNTIME_SUPERSESSION = (
    RUNTIME_ROOT / "QWEN30_EXACT_FULL_TOKEN_RUNTIME_SUPERSESSION.json"
)
QWEN30_RUNTIME_SUPERSESSION_HISTORY = RUNTIME_ROOT / "runtime-supersession-history"
QWEN30_RUNTIME_EXECUTABLE_TRANSITION_HISTORY = RUNTIME_ROOT / "runtime-transition-history"
QWEN30_VALIDATED_PAYLOAD_CATALOG_REQUALIFICATION_EPOCH = (
    RUNTIME_ROOT / "QWEN30_VALIDATED_PAYLOAD_CATALOG_REQUALIFICATION_EPOCH.json"
)
QWEN30_VALIDATED_PAYLOAD_CATALOG_REQUALIFICATION_HISTORY = (
    QWEN30_RUNTIME_EXECUTABLE_TRANSITION_HISTORY / "validated-payload-catalog-requalification"
)
QWEN30_HCLI_HANDOFF = RUNTIME_ROOT / "QWEN30_NATIVE_RUNTIME_HCLI_HANDOFF.json"
SCHEMA = "hawking.ascension.qwen30_bootstrap_lanes.v1"
PHYSICAL_RUNTIME_SCHEMA = "hawking.ascension.physical_exact_full_token_runtime.v1"
PHYSICAL_RUNTIME_STATUS = "PASS_EXACT_NATIVE_FULL_TOKEN_RUNTIME"
QWEN30_NATIVE_HTTP_BIND = "127.0.0.1:18430"
QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_RUNTIME_SOURCE = (
    REPO_ROOT
    / "workspace/ops/build/rust-qwen30-paired-production/debug/examples/"
    "ascension_qwen30_complete_native_runtime"
)
QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_DEPLOYMENT = (
    RUNTIME_ROOT / "QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_RUNTIME_DEPLOYMENT.json"
)
QWEN30_RUNTIME_EXECUTABLE_HISTORY = RUNTIME_ROOT / "runtime-executable-history"

# Kept in the launchd watcher process so child exit status can be reaped and
# written into a durable stage receipt. The on-disk active record covers a
# launchd restart without guessing that an inherited process is complete.
_ACTIVE_NATIVE_PROCESS: subprocess.Popen[bytes] | None = None
_ACTIVE_NATIVE_HTTP_PROCESS: subprocess.Popen[bytes] | None = None


class Qwen30BootstrapError(RuntimeError):
    """A source-bound bootstrap lane cannot continue safely."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _mapping(value: Any) -> dict[str, Any]:
    """Return a safely-owned JSON object view for transport parsing."""

    return dict(value) if isinstance(value, Mapping) else {}


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


def _atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
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
    raw = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _file_sha256(path: Path) -> str:
    """Hash an executable before allowing a sealed failed stage to be retried.

    A binding change naturally warrants a new run.  A runtime implementation
    change can also warrant one, but a 10-second watcher loop must never turn
    a deterministic crash into an unbounded retry storm.  The executable hash
    gives the next materially changed binary one clean retry while preserving
    the prior failure as evidence.
    """

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise Qwen30BootstrapError(f"native Qwen30 runtime executable cannot be hashed: {exc}") from exc
    return digest.hexdigest()


def _status(path: Path, *, lane: str, phase: str, **fields: Any) -> None:
    prior = _read_json(path) or {}
    _atomic_json(
        path,
        {
            "schema": SCHEMA,
            "recorded_at": _utc_now(),
            "pid": os.getpid(),
            "heartbeat": int(prior.get("heartbeat", 0)) + 1,
            "lane": lane,
            "phase": phase,
            **fields,
            "claim_boundary": {
                "raw_bf16_source_is_authority_teacher_not_tournament_participant": True,
                "does_not_claim_generation_capability_hcli_clean_tps_tg_or_tournament_qualification": True,
                "does_not_qualify_manager": True,
            },
        },
    )


def _weight_index() -> dict[str, str]:
    index = _read_json(MODEL_DIR / "model.safetensors.index.json")
    weights = index.get("weight_map") if isinstance(index, Mapping) else None
    if not isinstance(weights, Mapping):
        raise Qwen30BootstrapError("local Qwen30 safetensors index is unavailable")
    return {str(name): str(shard) for name, shard in weights.items()}


def _source_summary(weight_map: Mapping[str, str]) -> dict[str, Any]:
    layers = sorted({int(name.split(".")[2]) for name in weight_map if name.startswith("model.layers.")})
    mlp = sum(".mlp." in name for name in weight_map)
    router = sum(name.endswith(".mlp.gate.weight") for name in weight_map)
    experts = sum(".mlp.experts." in name for name in weight_map)
    missing_shards = sorted({shard for shard in set(weight_map.values()) if not (MODEL_DIR / shard).is_file()})
    return {
        "tensor_count": len(weight_map),
        "tensor_manifest_sha256": _sha256(weight_map),
        "source_shard_count": len(set(weight_map.values())),
        "missing_source_shards": missing_shards,
        "layer_count": len(layers),
        "layers": layers,
        "mlp_tensor_count": mlp,
        "router_tensor_count": router,
        "routed_expert_tensor_count": experts,
    }


def _complete_binary_candidate() -> dict[str, Any]:
    """Bind the physical all-tensor builder without mistaking it for a runtime."""

    status_path = COMPLETE_ROOT / "QWEN30_COMPLETE_GRAVITY_STATUS.json"
    manifest_path = COMPLETE_ROOT / "QWEN30_COMPLETE_BINARY_GRAVITY_CANDIDATE.json"
    status = _read_json(status_path) or {}
    progress = status.get("progress") if isinstance(status.get("progress"), Mapping) else {}
    manifest = _read_json(manifest_path)
    manifest_status = "NOT_YET_COMPLETE"
    manifest_seal = None
    if manifest is not None:
        try:
            checked = verify(manifest, label=str(manifest_path))
            manifest_status = str(checked.get("status") or "UNKNOWN")
            manifest_seal = checked.get("seal_sha256")
        except Exception as exc:
            manifest_status = f"INVALID_{type(exc).__name__}"
    return {
        "root": str(COMPLETE_ROOT),
        "status_phase": status.get("phase"),
        "progress": dict(progress),
        "manifest_path": str(manifest_path),
        "manifest_status": manifest_status,
        "manifest_seal_sha256": manifest_seal,
        "all_tensor_artifact_complete": manifest_status
        == "CANDIDATE_COMPLETE_BINARY_ARTIFACT_LOW_FIDELITY_UNQUALIFIED",
        "claim_boundary": "physical candidate progress is an input only; native decode, density qualification, TPS, and TG remain separate",
    }


def _qwen30_gqa_component() -> dict[str, Any]:
    """Admit only the exact source-bound GQA component receipt as a dependency."""

    document = _read_json(QWEN30_GQA_PROBE)
    if document is None:
        return {"status": "ABSENT", "path": str(QWEN30_GQA_PROBE)}
    geometry = document.get("official_qwen30_geometry") if isinstance(document.get("official_qwen30_geometry"), Mapping) else {}
    exact = (
        document.get("status")
        == "PASS_DIRECT_METAL_QWEN30_GQA_ATTENTION_COMPONENT_NOT_FULL_MODEL_NOT_TPS_GATE"
        and geometry.get("query_heads") == 32
        and geometry.get("kv_heads") == 4
        and geometry.get("head_dim") == 128
    )
    return {
        "status": "BYTE_HASH_BOUND_EXACT_GQA_COMPONENT" if exact else "INVALID_OR_UNEXPECTED",
        "path": str(QWEN30_GQA_PROBE),
        "sha256": hashlib.sha256(QWEN30_GQA_PROBE.read_bytes()).hexdigest(),
        "geometry": dict(geometry),
        "claim_boundary": "attention component only; QKV projection, RoPE, cache write, MoE, full decoder, and TPS remain separate",
    }


def _complete_binary_reader() -> dict[str, Any]:
    """Byte-bind the shared native read primitive without overstating a runtime."""

    if not COMPLETE_BINARY_READER.is_file():
        return {"status": "ABSENT", "path": str(COMPLETE_BINARY_READER)}
    raw = COMPLETE_BINARY_READER.read_bytes()
    return {
        "status": "RUST_NATIVE_READER_SOURCE_PRESENT",
        "path": str(COMPLETE_BINARY_READER),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "claim_boundary": "parser and f32 diagnostic reconstruction only; no Qwen layer tensor catalog, Metal direct decode, model loader, or TPS result",
    }


def _required_mapping(document: Mapping[str, Any], field: str, *, label: str) -> Mapping[str, Any]:
    value = document.get(field)
    if not isinstance(value, Mapping):
        raise Qwen30BootstrapError(f"{label} has no object field {field!r}")
    return value


def _required_text(document: Mapping[str, Any], field: str, *, label: str) -> str:
    value = document.get(field)
    if not isinstance(value, str) or not value:
        raise Qwen30BootstrapError(f"{label} has no non-empty string field {field!r}")
    return value


def _native_runtime_binding() -> dict[str, Any]:
    """Read the protected admission receipt, not a self-selected manifest."""

    receipt = _read_json(QWEN30_ADMISSION_RECEIPT)
    if receipt is None:
        raise Qwen30BootstrapError(f"Qwen30 admission receipt is absent: {QWEN30_ADMISSION_RECEIPT}")
    try:
        checked = verify(receipt, label=str(QWEN30_ADMISSION_RECEIPT))
    except Exception as exc:  # receipt package owns the exact error taxonomy
        raise Qwen30BootstrapError(f"Qwen30 admission receipt does not verify: {exc}") from exc
    if checked.get("status") != "EARNED_COMPLETE_BINARY_ARTIFACT_ADMITTED_NOT_RUNTIME_OR_CAPABILITY_QUALIFIED":
        raise Qwen30BootstrapError("Qwen30 admission receipt has an unexpected non-admitted status")
    manifest = _required_mapping(checked, "complete_manifest", label="Qwen30 admission receipt")
    current = _required_mapping(checked, "current_source_revalidation", label="Qwen30 admission receipt")
    manifest_path = Path(_required_text(manifest, "path", label="complete manifest binding"))
    expected_manifest = (COMPLETE_ROOT / "QWEN30_COMPLETE_BINARY_GRAVITY_CANDIDATE.json").resolve()
    try:
        observed_manifest = manifest_path.resolve(strict=True)
    except OSError as exc:
        raise Qwen30BootstrapError(f"admitted Qwen30 manifest is inaccessible: {exc}") from exc
    if observed_manifest != expected_manifest:
        raise Qwen30BootstrapError(
            f"admission receipt manifest {observed_manifest} is not the current protected Qwen30 candidate"
        )
    if manifest.get("status") != "CANDIDATE_COMPLETE_BINARY_ARTIFACT_LOW_FIDELITY_UNQUALIFIED":
        raise Qwen30BootstrapError("admission receipt does not bind a complete Qwen30 binary candidate")
    manifest_seal = _required_text(manifest, "seal_sha256", label="complete manifest binding")
    source_audit_seal = _required_text(current, "source_audit_seal_sha256", label="source revalidation")
    source_revision = _required_text(current, "revision", label="source revalidation")
    source_index = Path(_required_text(current, "index_path", label="source revalidation"))
    try:
        source_index = source_index.resolve(strict=True)
    except OSError as exc:
        raise Qwen30BootstrapError(f"admitted Qwen30 source index is inaccessible: {exc}") from exc
    config_path = source_index.parent / "config.json"
    config = _read_json(config_path)
    if config is None:
        raise Qwen30BootstrapError(f"admitted Qwen30 source config is unavailable: {config_path}")
    bos_token_id = config.get("bos_token_id")
    vocab_size = config.get("vocab_size")
    if not isinstance(bos_token_id, int) or not isinstance(vocab_size, int) or not 0 <= bos_token_id < vocab_size:
        raise Qwen30BootstrapError("admitted Qwen30 source config has no valid bos_token_id/vocab_size")
    return {
        "admission_receipt_path": str(QWEN30_ADMISSION_RECEIPT),
        "admission_receipt_seal_sha256": checked.get("seal_sha256"),
        "manifest_path": str(observed_manifest),
        "manifest_seal_sha256": manifest_seal,
        "source_audit_seal_sha256": source_audit_seal,
        "source_revision": source_revision,
        "source_config_path": str(config_path),
        "bos_token_id": bos_token_id,
        "model_vocab_size": vocab_size,
    }


def _native_stage_paths(stage: str) -> tuple[Path, Path, Path]:
    result = {
        "preflight": QWEN30_NATIVE_PREFLIGHT,
        "full-token": QWEN30_NATIVE_FULL_TOKEN,
        "prompt-a": QWEN30_NATIVE_PROMPT_A,
        "prompt-b": QWEN30_NATIVE_PROMPT_B,
        "profile-token": QWEN30_NATIVE_PROFILE_TOKEN,
        "simdgroup-candidate-token": QWEN30_NATIVE_SIMDGROUP_CANDIDATE_TOKEN,
        "simdgroup-candidate-prompt-a": QWEN30_NATIVE_SIMDGROUP_CANDIDATE_PROMPT_A,
        "simdgroup-candidate-prompt-b": QWEN30_NATIVE_SIMDGROUP_CANDIDATE_PROMPT_B,
    }.get(stage)
    if result is None:
        raise Qwen30BootstrapError(f"unsupported native Qwen30 stage {stage!r}")
    stem = result.stem
    return (
        result,
        RUNTIME_ROOT / f"{stem}.stdout.log",
        RUNTIME_ROOT / f"{stem}.stderr.log",
    )


def _native_stage_expected_status(stage: str) -> str:
    statuses = {
        "preflight": "EARNED_QWEN30_DIRECT_PACKED_NATIVE_RUNTIME_PREFLIGHT_NOT_TOKEN_EXECUTION",
        "full-token": "EARNED_QWEN30_DIRECT_PACKED_NATIVE_METAL_FULL_TOKEN_EXECUTED_UNQUALIFIED",
        "prompt-a": "EARNED_QWEN30_DIRECT_PACKED_NATIVE_GREEDY_AUTOREGRESSIVE_EXECUTED_UNQUALIFIED",
        "prompt-b": "EARNED_QWEN30_DIRECT_PACKED_NATIVE_GREEDY_AUTOREGRESSIVE_EXECUTED_UNQUALIFIED",
        "profile-token": "EARNED_QWEN30_DIRECT_PACKED_NATIVE_METAL_FULL_TOKEN_EXECUTED_UNQUALIFIED",
        "simdgroup-candidate-token": "EARNED_QWEN30_DIRECT_PACKED_NATIVE_METAL_FULL_TOKEN_EXECUTED_UNQUALIFIED",
        "simdgroup-candidate-prompt-a": "EARNED_QWEN30_DIRECT_PACKED_NATIVE_GREEDY_AUTOREGRESSIVE_EXECUTED_UNQUALIFIED",
        "simdgroup-candidate-prompt-b": "EARNED_QWEN30_DIRECT_PACKED_NATIVE_GREEDY_AUTOREGRESSIVE_EXECUTED_UNQUALIFIED",
    }
    try:
        return statuses[stage]
    except KeyError as exc:
        raise Qwen30BootstrapError(f"unsupported native Qwen30 stage {stage!r}") from exc


def _native_stage_expected_matvec_kernel(stage: str) -> str | None:
    if stage == "preflight":
        return None
    if stage in {
        "simdgroup-candidate-token",
        "simdgroup-candidate-prompt-a",
        "simdgroup-candidate-prompt-b",
    }:
        return "simdgroup_eight_rows_per_threadgroup_candidate"
    if stage in {"full-token", "prompt-a", "prompt-b", "profile-token"}:
        # Older immutable control receipts were emitted before the field was
        # added, so their absence is interpreted only as this explicit
        # scalar-control default—not as permission to accept an unknown mode.
        return "scalar_one_thread_per_row_control"
    raise Qwen30BootstrapError(f"unsupported native Qwen30 stage {stage!r}")


def _profile_token_has_complete_gpu_coverage(document: Mapping[str, Any]) -> bool:
    """Require exact all-dispatch gpu_prod coverage before optimization use.

    First-token native execution lazily decodes four RMS vectors per layer
    plus the final norm: 48 * 4 + 1 extra Metal dispatches beyond the token
    graph's own counter. A partial counter-buffer trace is useful negative
    science, but it is not enough to name the complete-token bottleneck.
    """

    execution = document.get("execution")
    step = execution.get("step") if isinstance(execution, Mapping) else None
    profiler = document.get("profiler")
    graph_dispatches = step.get("metal_dispatches") if isinstance(step, Mapping) else None
    if not isinstance(graph_dispatches, int) or not isinstance(profiler, Mapping):
        return False
    expected = graph_dispatches + (48 * 4 + 1)
    return (
        profiler.get("tcb_trace_mode_requested") == "gpu_prod"
        and profiler.get("expected_complete_token_dispatch_samples") == expected
        and profiler.get("dispatch_sample_count") == expected
        and profiler.get("gpu_timing_sample_count") == expected
        and profiler.get("complete_token_gpu_profile_coverage_earned") is True
    )


def _preserve_incomplete_profile_negative(
    binding: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Seal an incomplete gpu_prod trace before a repaired retry replaces it."""

    document = _read_json(QWEN30_NATIVE_PROFILE_TOKEN)
    if not isinstance(document, Mapping) or _profile_token_has_complete_gpu_coverage(document):
        return None
    runtime_binding = document.get("runtime_binding")
    if not isinstance(runtime_binding, Mapping) or (
        runtime_binding.get("manifest_seal_sha256") != binding.get("manifest_seal_sha256")
        or runtime_binding.get("source_revision") != binding.get("source_revision")
    ):
        return None
    try:
        raw = QWEN30_NATIVE_PROFILE_TOKEN.read_bytes()
    except OSError:
        return None
    execution = document.get("execution")
    step = execution.get("step") if isinstance(execution, Mapping) else None
    profiler = document.get("profiler")
    graph_dispatches = step.get("metal_dispatches") if isinstance(step, Mapping) else None
    expected = graph_dispatches + (48 * 4 + 1) if isinstance(graph_dispatches, int) else None
    observed = profiler.get("dispatch_sample_count") if isinstance(profiler, Mapping) else None
    existing = _read_json(QWEN30_NATIVE_PROFILE_PARTIAL_NEGATIVE)
    if isinstance(existing, Mapping) and existing.get("raw_profile_sha256") == _sha256(raw):
        return dict(existing)
    receipt = {
        "schema": "hawking.ascension.qwen30_complete_native_profile_negative.v1",
        "recorded_at": _utc_now(),
        "status": "REJECTED_QWEN30_PARTIAL_GPU_PRODUCTION_TRACE_REOPENED",
        "reason": "gpu_prod counter-sample coverage was incomplete; no complete-token bottleneck verdict may use it",
        "binding": dict(binding),
        "profile_path_before_retry": str(QWEN30_NATIVE_PROFILE_TOKEN),
        "raw_profile_sha256": _sha256(raw),
        "expected_complete_token_dispatch_samples": expected,
        "observed_dispatch_sample_count": observed,
        "observed_gpu_timing_sample_count": (
            profiler.get("gpu_timing_sample_count") if isinstance(profiler, Mapping) else None
        ),
        "raw_partial_profile": document,
        "claim_boundary": {
            "raw_ordered_partial_trace_preserved_for_negative_science": True,
            "not_a_complete_token_profile_or_dominant_bottleneck_receipt": True,
            "does_not_claim_tps_generation_hcli_tg_capability_or_tournament_qualification": True,
        },
    }
    _atomic_json(QWEN30_NATIVE_PROFILE_PARTIAL_NEGATIVE, receipt)
    return receipt


def _native_result_matches(stage: str, binding: Mapping[str, Any]) -> dict[str, Any] | None:
    result_path, _, _ = _native_stage_paths(stage)
    document = _read_json(result_path)
    if document is None or document.get("schema") != "hawking.ascension.qwen30_complete_native_runtime_result.v1":
        return None
    if document.get("status") != _native_stage_expected_status(stage):
        return None
    if not _same_current_runtime_binary(document):
        return None
    if stage == "preflight":
        observed = document.get("preflight")
    else:
        observed = document.get("runtime_binding")
    if not isinstance(observed, Mapping):
        return None
    if (
        observed.get("manifest_seal_sha256") != binding["manifest_seal_sha256"]
        or observed.get("source_revision") != binding["source_revision"]
    ):
        return None
    if stage == "preflight":
        if not _preflight_has_complete_immutable_payload_catalog(observed):
            return None
    elif not _runtime_has_complete_immutable_payload_catalog(observed):
        return None
    expected_kernel = _native_stage_expected_matvec_kernel(stage)
    if expected_kernel is not None:
        observed_kernel = observed.get("packed_matvec_kernel", "scalar_one_thread_per_row_control")
        if observed_kernel != expected_kernel:
            return None
    if stage != "preflight":
        try:
            expected_gate_up = _effective_qwen30_gate_up_swiglu_receipt_name()
        except Qwen30BootstrapError:
            return None
        if observed.get("gate_up_swiglu_kernel") != expected_gate_up:
            return None
    if stage == "profile-token" and not _profile_token_has_complete_gpu_coverage(document):
        return None
    if stage in {
        "prompt-a",
        "prompt-b",
        "simdgroup-candidate-prompt-a",
        "simdgroup-candidate-prompt-b",
    } and not _source_user_template_was_applied(document):
        return None
    return document


def _native_stage_command(stage: str, binding: Mapping[str, Any]) -> list[str]:
    command = [
        str(QWEN30_NATIVE_RUNTIME_EXECUTABLE),
        "--manifest",
        str(binding["manifest_path"]),
        "--expected-manifest-seal-sha256",
        str(binding["manifest_seal_sha256"]),
        "--expected-source-audit-seal-sha256",
        str(binding["source_audit_seal_sha256"]),
        "--expected-source-revision",
        str(binding["source_revision"]),
    ]
    gate_up_swiglu_kernel = _effective_qwen30_gate_up_swiglu_cli()
    if stage == "preflight":
        return command + [
            "--mode",
            "preflight",
            "--gate-up-swiglu-kernel",
            gate_up_swiglu_kernel,
        ]
    if stage in {"full-token", "profile-token", "simdgroup-candidate-token"}:
        kernel = (
            "simdgroup-candidate"
            if stage == "simdgroup-candidate-token"
            else "control"
        )
        return command + [
            "--mode",
            "forward-token",
            "--token-id",
            str(binding["bos_token_id"]),
            "--trace-dispatch",
            "--packed-matvec-kernel",
            kernel,
            "--gate-up-swiglu-kernel",
            gate_up_swiglu_kernel,
        ]
    prompts = {
        "prompt-a": "Reply with the single word native.",
        "prompt-b": "Write a one-line Python function named add.",
        "simdgroup-candidate-prompt-a": "Reply with the single word native.",
        "simdgroup-candidate-prompt-b": "Write a one-line Python function named add.",
    }
    try:
        prompt = prompts[stage]
    except KeyError as exc:
        raise Qwen30BootstrapError(f"unsupported native Qwen30 stage {stage!r}") from exc
    return command + [
        "--mode",
        "generate-greedy",
        "--prompt",
        prompt,
        "--prompt-template",
        "source-user-chat",
        "--packed-matvec-kernel",
        "simdgroup-candidate"
        if stage.startswith("simdgroup-candidate-")
        else "control",
        "--gate-up-swiglu-kernel",
        gate_up_swiglu_kernel,
        "--max-new-tokens",
        "2",
        "--max-seq-len",
        "256",
    ]


def _launch_native_stage(stage: str, binding: Mapping[str, Any]) -> dict[str, Any]:
    """Launch one bounded direct-packed stage under the detached watcher."""

    global _ACTIVE_NATIVE_PROCESS
    if not QWEN30_NATIVE_RUNTIME_EXECUTABLE.is_file():
        raise Qwen30BootstrapError(
            f"native Qwen30 runtime executable is unavailable: {QWEN30_NATIVE_RUNTIME_EXECUTABLE}"
        )
    result_path, stdout_path, stderr_path = _native_stage_paths(stage)
    executable_sha256 = _file_sha256(QWEN30_NATIVE_RUNTIME_EXECUTABLE)
    command = _native_stage_command(stage, binding)
    environment = os.environ.copy()
    diagnostic_environment: dict[str, str] = {}
    if stage == "profile-token":
        # Same direct pack and token graph as the baseline receipt, but this
        # bounded diagnostic invocation requests production-CB GPU timestamp
        # attribution.  It is deliberately a separate artifact/result and
        # remains unsuitable for clean TPS due to instrumentation overhead.
        environment["HAWKING_TCB_TRACE"] = "gpu_prod"
        diagnostic_environment["HAWKING_TCB_TRACE"] = "gpu_prod"
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
                env=environment,
            )
    except OSError as exc:
        raise Qwen30BootstrapError(f"could not launch native Qwen30 {stage} stage: {exc}") from exc
    _ACTIVE_NATIVE_PROCESS = process
    record = {
        "schema": "hawking.ascension.qwen30_native_runtime_process.v1",
        "phase": "RUNNING",
        "stage": stage,
        "pid": process.pid,
        "started_at": _utc_now(),
        "result_path": str(result_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "command": command,
        "diagnostic_environment": diagnostic_environment,
        "runtime_executable_sha256": executable_sha256,
        "binding": dict(binding),
        "claim_boundary": {
            "launch_is_not_a_successful_runtime_or_generation_receipt": True,
            "raw_bf16_source_is_not_passed_to_the_native_process": True,
            "not_clean_tps_hcli_tg_or_tournament_qualification": True,
        },
    }
    _atomic_json(QWEN30_NATIVE_ACTIVE, record)
    return record


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    # A stage inherited after a launchd watcher reload can be re-parented to
    # PID 1.  On macOS, `kill(pid, 0)` remains successful for a short-lived
    # zombie, which would otherwise leave the watcher believing a completed
    # native stage is still running forever.  Treat a visible zombie as
    # terminal; its stdout is then parsed and binding-checked exactly as a
    # recovered post-reload stage result.
    try:
        observed = subprocess.run(
            ["/bin/ps", "-o", "stat=", "-p", str(pid)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        observed = None
    if observed is not None and observed.returncode == 0:
        state = observed.stdout.strip().split(maxsplit=1)
        if state and state[0].startswith("Z"):
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _tail(path: Path, *, maximum_bytes: int = 4096) -> str:
    try:
        raw = path.read_bytes()
    except OSError:
        return ""
    return raw[-maximum_bytes:].decode("utf-8", errors="replace")


def _parse_native_stdout(path: Path) -> dict[str, Any]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise Qwen30BootstrapError(f"native runtime stdout is unavailable: {exc}") from exc
    for line in reversed(lines):
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, Mapping):
            return dict(parsed)
    raise Qwen30BootstrapError("native runtime exited zero without a machine-readable JSON result")


def _settle_native_process(record: Mapping[str, Any], returncode: int | None) -> dict[str, Any]:
    stage = str(record.get("stage") or "")
    result_path, stdout_path, stderr_path = _native_stage_paths(stage)
    process_result: dict[str, Any] = {
        "schema": "hawking.ascension.qwen30_native_runtime_process.v1",
        "phase": "EXITED",
        "stage": stage,
        "pid": record.get("pid"),
        "started_at": record.get("started_at"),
        "finished_at": _utc_now(),
        "returncode": returncode,
        "result_path": str(result_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "binding": record.get("binding"),
    }
    if returncode in {0, None}:
        try:
            result = _parse_native_stdout(stdout_path)
            binding = record.get("binding")
            if not isinstance(binding, Mapping) or not _native_result_matches_document(stage, result, binding):
                raise Qwen30BootstrapError("native result did not match the launched stage/binding")
            _atomic_json(result_path, result)
            process_result["outcome"] = (
                "EARNED_STAGE_RESULT_WRITTEN"
                if returncode == 0
                else "EARNED_STAGE_RESULT_RECOVERED_AFTER_WATCHER_RESTART"
            )
            process_result["native_result_status"] = result.get("status")
        except Qwen30BootstrapError as exc:
            process_result["outcome"] = (
                "ZERO_EXIT_WITH_INVALID_STAGE_RESULT"
                if returncode == 0
                else "TERMINAL_WITHOUT_VALID_STAGE_RESULT_AFTER_WATCHER_RESTART"
            )
            process_result["error"] = str(exc)
    else:
        process_result["outcome"] = "NATIVE_STAGE_FAILED"
        process_result["stderr_tail"] = _tail(stderr_path)
    _atomic_json(QWEN30_NATIVE_LAST_PROCESS, process_result)
    _atomic_json(QWEN30_NATIVE_ACTIVE, {**process_result, "phase": "TERMINAL"})
    return process_result


def _native_result_matches_document(stage: str, document: Mapping[str, Any], binding: Mapping[str, Any]) -> bool:
    if document.get("schema") != "hawking.ascension.qwen30_complete_native_runtime_result.v1":
        return False
    if document.get("status") != _native_stage_expected_status(stage):
        return False
    if not _same_current_runtime_binary(document):
        return False
    observed = document.get("preflight") if stage == "preflight" else document.get("runtime_binding")
    if not isinstance(observed, Mapping) or not (
        observed.get("manifest_seal_sha256") == binding.get("manifest_seal_sha256")
        and observed.get("source_revision") == binding.get("source_revision")
    ):
        return False
    if stage == "preflight":
        if not _preflight_has_complete_immutable_payload_catalog(observed):
            return False
    elif not _runtime_has_complete_immutable_payload_catalog(observed):
        return False
    expected_kernel = _native_stage_expected_matvec_kernel(stage)
    if expected_kernel is not None and observed.get(
        "packed_matvec_kernel", "scalar_one_thread_per_row_control"
    ) != expected_kernel:
        return False
    if stage != "preflight":
        try:
            expected_gate_up = _effective_qwen30_gate_up_swiglu_receipt_name()
        except Qwen30BootstrapError:
            return False
        if observed.get("gate_up_swiglu_kernel") != expected_gate_up:
            return False
    if stage == "profile-token" and not _profile_token_has_complete_gpu_coverage(document):
        return False
    if stage in {
        "prompt-a",
        "prompt-b",
        "simdgroup-candidate-prompt-a",
        "simdgroup-candidate-prompt-b",
    } and not _source_user_template_was_applied(document):
        return False
    return True


def _sealed_document(path: Path, *, label: str) -> dict[str, Any] | None:
    """Load one upstream sealed authority without treating a read as proof.

    The native executable's JSON is intentionally unsealed: it is a direct
    process observation.  This helper is only for the protected source and
    admission receipts used to bind a later physical-runtime receipt.
    """

    document = _read_json(path)
    if not isinstance(document, Mapping):
        return None
    try:
        return dict(verify(document, label=label))
    except Exception:
        return None


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _same_current_runtime_binary(document: Mapping[str, Any]) -> bool:
    """Do not promote a result generated by an older executable.

    Material runtime changes reopen all direct execution evidence.  Older
    results remain useful controls, but cannot silently certify the rewritten
    binary that an HCLI adapter or a later kernel candidate would use.
    """

    observed = document.get("runtime_executable_sha256")
    if not _is_sha256(observed):
        return False
    try:
        return observed == _file_sha256(QWEN30_NATIVE_RUNTIME_EXECUTABLE)
    except Qwen30BootstrapError:
        return False


def _preflight_has_complete_immutable_payload_catalog(preflight: Mapping[str, Any]) -> bool:
    """Require full verified direct payload admission, not a lazy subset.

    The preflight process proves the protected admission scan itself.  It exits
    afterward, so its snapshots are intentionally process-local; each native
    decoder process must prove its own retained immutable catalog below.
    """

    return (
        preflight.get("verified_payload_count") == 18_867
        and preflight.get("complete_verified_payload_cache_at_admission") is True
        and preflight.get("preflight_payload_snapshots_are_process_local") is True
    )


def _runtime_has_complete_immutable_payload_catalog(runtime_binding: Mapping[str, Any]) -> bool:
    """Require process-local verified direct payload handles on token paths."""

    catalog = runtime_binding.get("immutable_complete_payload_catalog")
    return isinstance(catalog, Mapping) and (
        catalog.get("validated_during_process_admission") is True
        and catalog.get("verified_payload_count") == 18_867
        and catalog.get("expected_complete_tensor_count") == 18_867
        and catalog.get("complete_verified_payload_cache") is True
        and catalog.get("payload_access_path")
        == "immutable_admission_verified_direct_snapshot"
        and catalog.get("per_token_payload_sha256_rescan") is False
        and catalog.get("full_artifact_revalidation_required_on_process_restart") is True
    )


def _valid_runtime_binding(document: Mapping[str, Any], binding: Mapping[str, Any]) -> bool:
    runtime_binding = document.get("runtime_binding")
    if not isinstance(runtime_binding, Mapping):
        return False
    return (
        runtime_binding.get("manifest_seal_sha256") == binding.get("manifest_seal_sha256")
        and runtime_binding.get("source_revision") == binding.get("source_revision")
        and runtime_binding.get("architecture") == "Qwen3MoeForCausalLM"
        and runtime_binding.get("layers") == 48
        and runtime_binding.get("metal_only") is True
        and runtime_binding.get("raw_bf16_loader_not_opened") is True
        and runtime_binding.get("model_alone") is True
        and runtime_binding.get("no_host_model_math_fallback") is True
        and runtime_binding.get("raw_bf16_teacher_not_runtime_participant") is True
        and _runtime_has_complete_immutable_payload_catalog(runtime_binding)
    )


def _source_user_template_was_applied(document: Mapping[str, Any]) -> bool:
    execution = document.get("execution")
    template = execution.get("prompt_template") if isinstance(execution, Mapping) else None
    return isinstance(template, Mapping) and (
        template.get("mode") == "source_user_chat_template"
        and template.get("source_template_bound") is True
        and template.get("applied_to_prompt") is True
        and _is_sha256(template.get("source_template_sha256"))
        and _is_sha256(template.get("tokenizer_config_sha256"))
    )


def _positive_forward_count(document: Mapping[str, Any]) -> int | None:
    execution = document.get("execution")
    value = execution.get("full_model_forward_count") if isinstance(execution, Mapping) else None
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


QWEN30_ROUTE_MAJOR_MOE_ACTIVATION_INPUT_OFFSET_DEFECT = (
    "REVOKED_ROUTE_MAJOR_MOE_ACTIVATION_INPUT_OFFSET_DEFECT"
)
QWEN30_VALIDATED_PAYLOAD_CATALOG_RUNTIME_TRANSITION = (
    "REVOKED_RUNTIME_EXECUTABLE_TRANSITION_TO_VALIDATED_ONCE_PAYLOAD_CATALOG"
)
QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_RUNTIME_TRANSITION = (
    "REVOKED_RUNTIME_EXECUTABLE_TRANSITION_TO_PAIRED_SCALAR_ORDER_PRODUCTION_NO_PARITY"
)
QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_KERNEL = (
    "paired_direct_packed_gate_up_swiglu_scalar_order_production_no_parity"
)
QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_KERNEL_CLI = (
    "paired-scalar-order-production-no-parity"
)
QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_HTTP_KERNEL_ID = (
    "qwen30_paired_scalar_order_no_parity_v1"
)
QWEN30_SCALAR_CONTROL_HTTP_KERNEL_ID = "qwen30_packed_binary_scalar_control_v1"
QWEN30_RUNTIME_SUPERSESSION_SCHEMA = (
    "hawking.ascension.physical_exact_full_token_runtime_supersession.v1"
)
QWEN30_VALIDATED_PAYLOAD_CATALOG_REQUALIFICATION_SCHEMA = (
    "hawking.ascension.qwen30_validated_payload_catalog_requalification_epoch.v1"
)
QWEN30_VALIDATED_PAYLOAD_CATALOG_REQUALIFICATION_STATUS = (
    "OPEN_FRESH_POST_TRANSITION_CACHE_BACKED_NATIVE_REQUALIFICATION"
)


def _qwen30_runtime_supersession_history_paths() -> list[Path]:
    """Return the active sidecar plus preserved historical sidecars.

    The generic gatekeeper consumes only the active sidecar. Runtime-local
    execution still remembers a prior architecture-defect revocation after a
    later, non-defect executable transition replaces that active pointer.
    """

    paths = [QWEN30_EXACT_FULL_TOKEN_RUNTIME_SUPERSESSION]
    if QWEN30_RUNTIME_SUPERSESSION_HISTORY.is_dir():
        paths.extend(
            sorted(
                QWEN30_RUNTIME_SUPERSESSION_HISTORY.glob(
                    "QWEN30_EXACT_FULL_TOKEN_RUNTIME_SUPERSESSION_*.json"
                )
            )
        )
    return paths


def _qwen30_route_offset_runtime_revocation() -> dict[str, Any] | None:
    """Return the active source-bound Qwen30 route-offset revocation.

    A supersession binds a *specific* defective executable SHA.  It prevents
    stale results from that executable from being rebuilt into a PASS receipt,
    while allowing a later corrected binary to earn a wholly new receipt from
    freshly generated evidence.
    """

    for path in _qwen30_runtime_supersession_history_paths():
        document = _sealed_document(path, label=str(path))
        if not isinstance(document, Mapping):
            continue
        target = document.get("revoked_runtime")
        binding = document.get("binding")
        if not isinstance(target, Mapping) or not isinstance(binding, Mapping):
            continue
        if not (
            document.get("schema") == QWEN30_RUNTIME_SUPERSESSION_SCHEMA
            and document.get("status") == QWEN30_ROUTE_MAJOR_MOE_ACTIVATION_INPUT_OFFSET_DEFECT
            and binding.get("canonical_runtime_receipt_path")
            == str(QWEN30_EXACT_FULL_TOKEN_RUNTIME_RECEIPT)
            and _is_sha256(binding.get("superseded_runtime_receipt_seal_sha256"))
            and _is_sha256(binding.get("defective_runtime_executable_sha256"))
            and isinstance(binding.get("archived_runtime_receipt_path"), str)
            and _is_sha256(binding.get("archived_runtime_receipt_document_sha256"))
            and target.get("canonical_receipt_path") == str(QWEN30_EXACT_FULL_TOKEN_RUNTIME_RECEIPT)
            and _is_sha256(target.get("canonical_receipt_seal_sha256"))
            and _is_sha256(target.get("runtime_executable_sha256"))
            and binding.get("superseded_runtime_receipt_seal_sha256")
            == target.get("canonical_receipt_seal_sha256")
            and binding.get("defective_runtime_executable_sha256")
            == target.get("runtime_executable_sha256")
            and isinstance(document.get("historical_pass_archive_path"), str)
        ):
            continue
        return dict(document)
    return None


def _qwen30_runtime_revocation_applies_to_sha(
    revocation: Mapping[str, Any] | None, executable_sha256: Any
) -> bool:
    target = revocation.get("revoked_runtime") if isinstance(revocation, Mapping) else None
    return (
        isinstance(target, Mapping)
        and _is_sha256(executable_sha256)
        and target.get("runtime_executable_sha256") == executable_sha256
    )


def _archive_route_major_defect_observations(
    revocation: Mapping[str, Any],
) -> dict[str, Any]:
    """Preserve raw controls invalidated by the route-major MoE defect.

    The canonical PASS receipt has its own sealed history.  The raw preflight,
    all-layer, prompt, profile, and endpoint observations are distinct
    diagnostic controls, however, and the corrected watcher will legitimately
    overwrite their canonical stage paths.  Archive their exact JSON bytes
    before that happens so the defect remains reproducible without allowing
    any of these observations to regain gate authority.
    """

    target = revocation.get("revoked_runtime")
    binding = revocation.get("binding")
    if not isinstance(target, Mapping) or not isinstance(binding, Mapping):
        raise Qwen30BootstrapError("route-major revocation lacks a valid target binding")
    defective_runtime_sha = target.get("runtime_executable_sha256")
    defective_receipt_seal = target.get("canonical_receipt_seal_sha256")
    if not _is_sha256(defective_runtime_sha) or not _is_sha256(defective_receipt_seal):
        raise Qwen30BootstrapError("route-major revocation target hashes are invalid")

    existing = _sealed_document(
        QWEN30_ROUTE_MAJOR_DEFECT_HISTORY_MANIFEST,
        label=str(QWEN30_ROUTE_MAJOR_DEFECT_HISTORY_MANIFEST),
    )
    if (
        isinstance(existing, Mapping)
        and existing.get("schema")
        == "hawking.ascension.qwen30_route_major_defect_observation_archive.v1"
        and existing.get("status")
        == "PRESERVED_REVOKED_ROUTE_MAJOR_MOE_RUNTIME_RAW_CONTROLS"
        and isinstance(existing.get("binding"), Mapping)
        and existing["binding"].get("defective_runtime_executable_sha256")
        == defective_runtime_sha
        and existing["binding"].get("superseded_runtime_receipt_seal_sha256")
        == defective_receipt_seal
    ):
        archives = existing.get("archived_observations")
        if isinstance(archives, list) and all(
            isinstance(item, Mapping)
            and isinstance(item.get("archive_path"), str)
            and _is_sha256(item.get("archive_sha256"))
            and Path(item["archive_path"]).is_file()
            and _file_sha256(Path(item["archive_path"])) == item["archive_sha256"]
            for item in archives
        ):
            return dict(existing)

    # Transport smoke carries the superseded runtime receipt seal rather than
    # the model executable SHA.  It is still a revoked control because the
    # endpoint called that same defective direct-packed runtime.
    observations = (
        ("preflight", QWEN30_NATIVE_PREFLIGHT),
        ("full_token", QWEN30_NATIVE_FULL_TOKEN),
        ("prompt_a", QWEN30_NATIVE_PROMPT_A),
        ("prompt_b", QWEN30_NATIVE_PROMPT_B),
        ("profile_token", QWEN30_NATIVE_PROFILE_TOKEN),
        ("http_health_context_transport", QWEN30_NATIVE_HTTP_TRANSPORT_SMOKE),
        ("hcli_chat_sse_transport", QWEN30_NATIVE_HTTP_CHAT_SMOKE),
    )
    archived: list[dict[str, Any]] = []
    for label, source_path in observations:
        if not source_path.is_file():
            continue
        try:
            raw = source_path.read_text(encoding="utf-8")
            document = json.loads(raw)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Qwen30BootstrapError(
                f"cannot preserve revoked {label} observation {source_path}: {exc}"
            ) from exc
        if not isinstance(document, Mapping):
            raise Qwen30BootstrapError(
                f"cannot preserve non-object revoked {label} observation {source_path}"
            )
        archive_path = QWEN30_ROUTE_MAJOR_DEFECT_HISTORY / (
            f"{source_path.stem}__runtime_{defective_runtime_sha}{source_path.suffix}"
        )
        if archive_path.exists():
            if _file_sha256(archive_path) != _sha256(raw.encode("utf-8")):
                raise Qwen30BootstrapError(
                    f"existing revoked {label} archive differs from the current control"
                )
        else:
            _atomic_text(archive_path, raw)
        archived.append(
            {
                "label": label,
                "source_path": str(source_path),
                "source_sha256": _sha256(raw.encode("utf-8")),
                "archive_path": str(archive_path),
                "archive_sha256": _file_sha256(archive_path),
                "status": document.get("status"),
                "runtime_executable_sha256": document.get("runtime_executable_sha256"),
                "runtime_receipt_seal_sha256": (
                    document.get("binding", {}).get("runtime_receipt_seal_sha256")
                    if isinstance(document.get("binding"), Mapping)
                    else None
                ),
            }
        )
    payload = {
        "schema": "hawking.ascension.qwen30_route_major_defect_observation_archive.v1",
        "status": "PRESERVED_REVOKED_ROUTE_MAJOR_MOE_RUNTIME_RAW_CONTROLS",
        "recorded_at": _utc_now(),
        "binding": {
            "model_id": target.get("model_id"),
            "defective_runtime_executable_sha256": defective_runtime_sha,
            "superseded_runtime_receipt_seal_sha256": defective_receipt_seal,
            "runtime_supersession_path": str(QWEN30_EXACT_FULL_TOKEN_RUNTIME_SUPERSESSION),
            "runtime_supersession_seal_sha256": revocation.get("seal_sha256"),
        },
        "archived_observations": archived,
        "claim_boundary": {
            "archives_are_negative_science_and_debugging_controls_only": True,
            "no_archived_control_is_a_current_runtime_hcli_tps_or_capability_authority": True,
            "corrected_stage_paths_must_be_re-earned_on_a_new_runtime_executable": True,
        },
    }
    sealed = seal(payload)
    _atomic_json(QWEN30_ROUTE_MAJOR_DEFECT_HISTORY_MANIFEST, sealed)
    return sealed


def _revoke_qwen30_route_offset_runtime_receipt() -> dict[str, Any]:
    """Archive a defective PASS and replace its canonical authority with revoke.

    The historical pass is copied as an independently verifiable sealed record
    before the canonical filename is replaced by a sealed, non-PASS document.
    Consumers that only know the canonical path therefore fail closed, and
    consumers which bind an old seal have a public supersession authority.
    """

    prior = _sealed_document(
        QWEN30_EXACT_FULL_TOKEN_RUNTIME_RECEIPT,
        label=str(QWEN30_EXACT_FULL_TOKEN_RUNTIME_RECEIPT),
    )
    existing_revocation = _qwen30_route_offset_runtime_revocation()
    if prior is None:
        if existing_revocation is not None:
            return existing_revocation
        raise Qwen30BootstrapError(
            "cannot revoke Qwen30 exact runtime: canonical receipt is absent or seal-invalid"
        )
    if prior.get("status") == QWEN30_ROUTE_MAJOR_MOE_ACTIVATION_INPUT_OFFSET_DEFECT:
        if existing_revocation is not None:
            return existing_revocation
        raise Qwen30BootstrapError(
            "canonical Qwen30 receipt is already revoked but its supersession authority is missing"
        )
    if not (
        prior.get("schema") == PHYSICAL_RUNTIME_SCHEMA
        and prior.get("status") == PHYSICAL_RUNTIME_STATUS
        and isinstance(prior.get("binding"), Mapping)
        and _is_sha256(prior.get("seal_sha256"))
        and _is_sha256(prior["binding"].get("runtime_executable_sha256"))
    ):
        raise Qwen30BootstrapError(
            "canonical Qwen30 receipt is not the expected sealed native-runtime PASS"
        )
    prior_seal = prior["seal_sha256"]
    prior_runtime_sha = prior["binding"]["runtime_executable_sha256"]
    history_path = QWEN30_EXACT_FULL_TOKEN_RUNTIME_HISTORY / (
        f"QWEN30_EXACT_FULL_TOKEN_RUNTIME_RECEIPT_{prior_seal}.json"
    )
    if history_path.exists():
        archived = _sealed_document(history_path, label=str(history_path))
        if not isinstance(archived, Mapping) or archived.get("seal_sha256") != prior_seal:
            raise Qwen30BootstrapError(
                "existing Qwen30 runtime history archive does not preserve the prior seal"
            )
    else:
        _atomic_json(history_path, prior)
        archived = _sealed_document(history_path, label=str(history_path))
        if not isinstance(archived, Mapping) or archived.get("seal_sha256") != prior_seal:
            raise Qwen30BootstrapError(
                "failed to preserve a verifiable historical Qwen30 runtime receipt"
            )
    supersession_payload = {
        "schema": QWEN30_RUNTIME_SUPERSESSION_SCHEMA,
        "status": QWEN30_ROUTE_MAJOR_MOE_ACTIVATION_INPUT_OFFSET_DEFECT,
        "recorded_at": _utc_now(),
        # Flat binding is the public generic-consumer contract. Keep the
        # richer `revoked_runtime` object below for runtime-local diagnostics,
        # but do not make a gatekeeper infer field names from it.
        "binding": {
            "model_id": prior["binding"].get("model_id"),
            "canonical_runtime_receipt_path": str(QWEN30_EXACT_FULL_TOKEN_RUNTIME_RECEIPT),
            "superseded_runtime_receipt_seal_sha256": prior_seal,
            "defective_runtime_executable_sha256": prior_runtime_sha,
            "archived_runtime_receipt_path": str(history_path),
            "archived_runtime_receipt_document_sha256": _file_sha256(history_path),
        },
        "revoked_runtime": {
            "canonical_receipt_path": str(QWEN30_EXACT_FULL_TOKEN_RUNTIME_RECEIPT),
            "canonical_receipt_seal_sha256": prior_seal,
            "runtime_executable_sha256": prior_runtime_sha,
            "model_id": prior["binding"].get("model_id"),
            "complete_manifest_seal_sha256": prior["binding"].get(
                "complete_manifest_seal_sha256"
            ),
        },
        "historical_pass_archive_path": str(history_path),
        "historical_pass_archive_sha256": _file_sha256(history_path),
        "defect": {
            "class": "ROUTE_MAJOR_MOE_ACTIVATION_INPUT_OFFSET_OMITTED",
            "location": "Qwen30CompleteNativeRuntime::forward_token_greedy routed expert down projection",
            "observed_wiring": (
                "gate/up/SwiGLU writes each route into expert_activation[mid_offset], "
                "but the down projection bound expert_activation at input offset 0 for every route"
            ),
            "effect": (
                "routed expert slots 1..7 consumed route-0 activation rather than their "
                "own selected route-major activation; the prior all-layer result is not "
                "architecture-valid Qwen30 MoE execution"
            ),
            "host_or_bf16_fallback_involved": False,
        },
        "invalidates": {
            "canonical_native_runtime_pass": True,
            "all_old_full_token_prompt_and_profile_controls_bound_to_runtime_sha": True,
            "native_http_adapter_and_transport_handoff_bound_to_runtime_sha": True,
            "any_hcli_tps_tg_capability_or_tournament_consumer_of_that_sha": True,
        },
        "required_before_reissue": [
            "corrected direct-packed down-projection input-offset implementation",
            "fresh source-bound preflight on a new executable sha256",
            "fresh corrected scalar full-token execution",
            "fresh corrected source-template prompt A and B generation",
            "fresh complete-token GPU profile",
            "fresh native HTTP health/context and chat transport observations",
        ],
        "consumer_contract": {
            "fail_closed_if_canonical_status_is_not_pass": True,
            "fail_closed_if_this_supersession_revokes_the_bound_receipt_seal_or_runtime_executable_sha256": True,
            "historical_archive_is_for_negative_science_only_not_a_gate_authority": True,
        },
        "claim_boundary": {
            "revocation_is_not_a_new_runtime_or_quality_receipt": True,
            "does_not_claim_corrected_execution_or_any_tps": True,
        },
    }
    supersession = seal(supersession_payload)
    _atomic_json(QWEN30_EXACT_FULL_TOKEN_RUNTIME_SUPERSESSION, supersession)
    canonical_revocation = seal(
        {
            "schema": PHYSICAL_RUNTIME_SCHEMA,
            "status": QWEN30_ROUTE_MAJOR_MOE_ACTIVATION_INPUT_OFFSET_DEFECT,
            "recorded_at": _utc_now(),
            "supersession_path": str(QWEN30_EXACT_FULL_TOKEN_RUNTIME_SUPERSESSION),
            "supersession_seal_sha256": supersession["seal_sha256"],
            "historical_pass_archive_path": str(history_path),
            "revoked_runtime": dict(supersession_payload["revoked_runtime"]),
            "consumer_contract": dict(supersession_payload["consumer_contract"]),
            "claim_boundary": {
                "this_canonical_filename_is_now_an_explicit_non_pass_revocation": True,
                "old_pass_is_preserved_only_at_the_historical_archive_path": True,
                "no_runtime_hcli_tps_tg_capability_or_tournament_pass_is_claimed": True,
            },
        }
    )
    _atomic_json(QWEN30_EXACT_FULL_TOKEN_RUNTIME_RECEIPT, canonical_revocation)
    return supersession


def _archive_active_qwen30_runtime_supersession() -> dict[str, Any] | None:
    """Preserve the active generic sidecar before an executable transition.

    The generic consumers have one active supersession filename.  Historical
    architecture-defect sidecars remain runtime-local negative science when a
    later executable identity transition must take over that active pointer.
    """

    prior = _sealed_document(
        QWEN30_EXACT_FULL_TOKEN_RUNTIME_SUPERSESSION,
        label=str(QWEN30_EXACT_FULL_TOKEN_RUNTIME_SUPERSESSION),
    )
    if not isinstance(prior, Mapping):
        return None
    prior_seal = prior.get("seal_sha256")
    if not _is_sha256(prior_seal):
        raise Qwen30BootstrapError("active Qwen30 runtime supersession lacks a valid seal")
    archive_path = QWEN30_RUNTIME_SUPERSESSION_HISTORY / (
        f"QWEN30_EXACT_FULL_TOKEN_RUNTIME_SUPERSESSION_{prior_seal}.json"
    )
    if archive_path.exists():
        archived = _sealed_document(archive_path, label=str(archive_path))
        if not isinstance(archived, Mapping) or archived.get("seal_sha256") != prior_seal:
            raise Qwen30BootstrapError(
                "existing runtime supersession archive does not preserve the active sidecar"
            )
    else:
        _atomic_json(archive_path, prior)
        archived = _sealed_document(archive_path, label=str(archive_path))
        if not isinstance(archived, Mapping) or archived.get("seal_sha256") != prior_seal:
            raise Qwen30BootstrapError("failed to preserve active runtime supersession")
    return {
        "path": str(archive_path),
        "seal_sha256": prior_seal,
        "document_sha256": _file_sha256(archive_path),
        "status": prior.get("status"),
    }


def _archive_qwen30_runtime_transition_controls(
    *,
    superseded_runtime_executable_sha256: str,
    superseded_runtime_receipt_seal_sha256: str,
) -> dict[str, Any]:
    """Preserve valid old-binary controls before fresh cache-backed output.

    Unlike a defect revocation, these are retained as historical control
    evidence. They are never eligible to certify the new executable because
    all normal stage matching binds the exact executable digest.
    """

    if not (
        _is_sha256(superseded_runtime_executable_sha256)
        and _is_sha256(superseded_runtime_receipt_seal_sha256)
    ):
        raise Qwen30BootstrapError("runtime transition archive received invalid source hashes")
    controls = {
        "preflight": QWEN30_NATIVE_PREFLIGHT,
        "full_token": QWEN30_NATIVE_FULL_TOKEN,
        "prompt_a": QWEN30_NATIVE_PROMPT_A,
        "prompt_b": QWEN30_NATIVE_PROMPT_B,
        "profile_token": QWEN30_NATIVE_PROFILE_TOKEN,
        "simdgroup_component_parity": QWEN30_SIMDGROUP_COMPONENT_PARITY,
        "simdgroup_candidate_token": QWEN30_NATIVE_SIMDGROUP_CANDIDATE_TOKEN,
        "simdgroup_candidate_prompt_a": QWEN30_NATIVE_SIMDGROUP_CANDIDATE_PROMPT_A,
        "simdgroup_candidate_prompt_b": QWEN30_NATIVE_SIMDGROUP_CANDIDATE_PROMPT_B,
        "simdgroup_template_parity": QWEN30_SIMDGROUP_TEMPLATE_PARITY,
        "gateup_fused_template_parity": QWEN30_GATEUP_FUSED_TEMPLATE_PARITY,
        "gateup_fused_candidate_prompt_a": QWEN30_NATIVE_GATEUP_FUSED_CANDIDATE_PROMPT_A,
        "gateup_fused_candidate_prompt_b": QWEN30_NATIVE_GATEUP_FUSED_CANDIDATE_PROMPT_B,
        "native_http_adapter_status": QWEN30_NATIVE_HTTP_STATUS,
        "native_http_transport_smoke": QWEN30_NATIVE_HTTP_TRANSPORT_SMOKE,
        "native_http_chat_sse_smoke": QWEN30_NATIVE_HTTP_CHAT_SMOKE,
        "native_hcli_handoff": QWEN30_HCLI_HANDOFF,
    }
    archived: list[dict[str, Any]] = []
    for label, source_path in controls.items():
        try:
            raw = source_path.read_text(encoding="utf-8")
            document = json.loads(raw)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(document, Mapping):
            continue
        archive_path = QWEN30_RUNTIME_EXECUTABLE_TRANSITION_HISTORY / (
            f"{source_path.stem}__runtime_{superseded_runtime_executable_sha256}{source_path.suffix}"
        )
        raw_sha256 = _sha256(raw.encode("utf-8"))
        if archive_path.exists():
            if _file_sha256(archive_path) != raw_sha256:
                raise Qwen30BootstrapError(
                    f"existing transition archive for {label} differs from current historical control"
                )
        else:
            _atomic_text(archive_path, raw)
        archived.append(
            {
                "label": label,
                "source_path": str(source_path),
                "source_sha256": raw_sha256,
                "archive_path": str(archive_path),
                "archive_sha256": _file_sha256(archive_path),
                "status": document.get("status"),
                "runtime_executable_sha256": document.get("runtime_executable_sha256"),
            }
        )
    payload = {
        "schema": "hawking.ascension.qwen30_runtime_executable_transition_control_archive.v1",
        "status": "PRESERVED_SUPERSEDED_RUNTIME_CONTROLS_NOT_CURRENT_AUTHORITY",
        "recorded_at": _utc_now(),
        "binding": {
            "model_id": "Qwen3-Coder-30B-A3B-Instruct",
            "superseded_runtime_executable_sha256": superseded_runtime_executable_sha256,
            "superseded_runtime_receipt_seal_sha256": superseded_runtime_receipt_seal_sha256,
            "canonical_runtime_receipt_path": str(QWEN30_EXACT_FULL_TOKEN_RUNTIME_RECEIPT),
        },
        "archived_controls": archived,
        "claim_boundary": {
            "historical_controls_remain_valid_only_for_their_old_binary": True,
            "no_archived_control_can_certify_the_cache_backed_executable": True,
            "not_a_runtime_hcli_tps_tg_capability_or_tournament_receipt": True,
        },
    }
    sealed = seal(payload)
    manifest_path = QWEN30_RUNTIME_EXECUTABLE_TRANSITION_HISTORY / (
        "QWEN30_RUNTIME_EXECUTABLE_TRANSITION_CONTROL_ARCHIVE_"
        f"{superseded_runtime_receipt_seal_sha256}.json"
    )
    _atomic_json(manifest_path, sealed)
    return {
        "manifest_path": str(manifest_path),
        "manifest_seal_sha256": sealed["seal_sha256"],
        "archived_control_count": len(archived),
    }


def _copy_executable_atomically(source: Path, destination: Path) -> None:
    """Install one already-hashed executable without exposing a partial file."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as target, source.open("rb") as origin:
            shutil.copyfileobj(origin, target, length=1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        shutil.copystat(source, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _archive_qwen30_runtime_executable(executable_sha256: str) -> dict[str, Any]:
    """Keep the superseded live executable as a byte-verifiable control."""

    if not _is_sha256(executable_sha256):
        raise Qwen30BootstrapError("runtime executable archive requires a SHA-256")
    archive = QWEN30_RUNTIME_EXECUTABLE_HISTORY / (
        f"ascension_qwen30_complete_native_runtime_{executable_sha256}"
    )
    if archive.exists():
        if _file_sha256(archive) != executable_sha256:
            raise Qwen30BootstrapError("existing runtime executable archive has a mismatched SHA-256")
    else:
        _copy_executable_atomically(QWEN30_NATIVE_RUNTIME_EXECUTABLE, archive)
        if _file_sha256(archive) != executable_sha256:
            raise Qwen30BootstrapError("runtime executable archive verification failed")
    return {"path": str(archive), "sha256": executable_sha256}


def _paired_scalar_order_production_deployment() -> dict[str, Any] | None:
    """Return the live no-parity deployment only when every binding is current.

    A deployment record is intentionally not a runtime receipt.  It merely
    prevents the detached watcher from silently falling back to the scalar
    control after the scalar-order production transition has been sealed.
    """

    document = _sealed_document(
        QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_DEPLOYMENT,
        label=str(QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_DEPLOYMENT),
    )
    if not isinstance(document, Mapping):
        return None
    binding = document.get("binding")
    if not (
        document.get("schema")
        == "hawking.ascension.qwen30_paired_scalar_order_production_runtime_deployment.v1"
        and document.get("status")
        == "DEPLOYED_AWAITING_FRESH_NATIVE_RUNTIME_REQUALIFICATION"
        and isinstance(binding, Mapping)
        and binding.get("production_gate_up_swiglu_kernel")
        == QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_KERNEL_CLI
        and binding.get("production_kernel_receipt_id")
        == QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_KERNEL
        and _is_sha256(binding.get("replacement_runtime_executable_sha256"))
        and binding.get("runtime_executable_path") == str(QWEN30_NATIVE_RUNTIME_EXECUTABLE)
        and binding.get("candidate_template_parity_receipt_seal_sha256")
        and binding.get("candidate_cpu_parity_receipt_seal_sha256")
        and binding.get("scalar_control_runtime_receipt_seal_sha256")
    ):
        return None
    try:
        current_sha = _file_sha256(QWEN30_NATIVE_RUNTIME_EXECUTABLE)
    except Qwen30BootstrapError:
        return None
    if current_sha != binding.get("replacement_runtime_executable_sha256"):
        return None
    return dict(document)


def _paired_scalar_order_production_requested() -> bool:
    return QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_DEPLOYMENT.exists()


def _effective_qwen30_gate_up_swiglu_cli() -> str:
    deployment = _paired_scalar_order_production_deployment()
    if deployment is not None:
        return QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_KERNEL_CLI
    if _paired_scalar_order_production_requested():
        raise Qwen30BootstrapError(
            "paired scalar-order production deployment is present but not bound to the live runtime; refusing scalar fallback"
        )
    return "control"


def _effective_qwen30_gate_up_swiglu_receipt_name() -> str:
    return (
        QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_KERNEL
        if _effective_qwen30_gate_up_swiglu_cli()
        == QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_KERNEL_CLI
        else "three_dispatch_direct_packed_gate_up_swiglu_control"
    )


def _native_http_adapter_kernel_contract() -> dict[str, Any]:
    """Return the exact server metadata allowed for the live runtime epoch.

    The HTTP process must never select a kernel by default.  This contract is
    shared by deployment, process reconciliation, and transport smoke so an
    old scalar listener cannot be relabelled as a current production endpoint.
    """

    cli = _effective_qwen30_gate_up_swiglu_cli()
    if cli == QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_KERNEL_CLI:
        return {
            "gate_up_swiglu_kernel_cli": cli,
            "gate_up_swiglu_kernel": QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_KERNEL,
            "kernel_id": QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_HTTP_KERNEL_ID,
            "custom_kernel_used": True,
            "production_no_parity": True,
        }
    if cli == "control":
        return {
            "gate_up_swiglu_kernel_cli": cli,
            "gate_up_swiglu_kernel": "three_dispatch_direct_packed_gate_up_swiglu_control",
            "kernel_id": QWEN30_SCALAR_CONTROL_HTTP_KERNEL_ID,
            "custom_kernel_used": False,
            "production_no_parity": False,
        }
    raise Qwen30BootstrapError(f"unsupported native HTTP gate/up kernel CLI {cli!r}")


def _native_http_adapter_runtime_binding(
    binding: Mapping[str, Any],
    runtime_receipt: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Build the non-forgeable runtime/kernel binding for an HTTP process."""

    if not isinstance(runtime_receipt, Mapping) or (
        runtime_receipt.get("schema") != PHYSICAL_RUNTIME_SCHEMA
        or runtime_receipt.get("status") != PHYSICAL_RUNTIME_STATUS
        or not _is_sha256(runtime_receipt.get("seal_sha256"))
    ):
        return None
    receipt_binding = runtime_receipt.get("binding")
    receipt_runtime = runtime_receipt.get("runtime")
    if not isinstance(receipt_binding, Mapping) or not isinstance(receipt_runtime, Mapping):
        return None
    contract = _native_http_adapter_kernel_contract()
    try:
        current_runtime_sha256 = _file_sha256(QWEN30_NATIVE_RUNTIME_EXECUTABLE)
    except Qwen30BootstrapError:
        return None
    if not (
        receipt_binding.get("complete_manifest_seal_sha256")
        == binding.get("manifest_seal_sha256")
        and receipt_binding.get("runtime_executable_sha256") == current_runtime_sha256
        and receipt_runtime.get("gate_up_swiglu_kernel")
        == contract["gate_up_swiglu_kernel"]
        and receipt_runtime.get("custom_kernel_used")
        is contract["custom_kernel_used"]
    ):
        return None
    runtime_deployment = _paired_scalar_order_production_deployment()
    if contract["production_no_parity"]:
        if not isinstance(runtime_deployment, Mapping):
            return None
        deployed = runtime_deployment.get("binding")
        if not isinstance(deployed, Mapping) or (
            deployed.get("replacement_runtime_executable_sha256") != current_runtime_sha256
            or deployed.get("production_gate_up_swiglu_kernel")
            != contract["gate_up_swiglu_kernel_cli"]
            or deployed.get("production_kernel_receipt_id")
            != contract["gate_up_swiglu_kernel"]
        ):
            return None
    return {
        # The endpoint has to bind both levels of authority: the current
        # canonical full-runtime receipt and the admission receipt it derives
        # from.  The strict HCLI/TPS sidecar independently checks the latter
        # against the canonical runtime, so omitting it here would make a
        # correct production server look detached from its admitted artifact.
        "admission_receipt_seal_sha256": binding.get(
            "admission_receipt_seal_sha256"
        ),
        "manifest_seal_sha256": binding.get("manifest_seal_sha256"),
        "source_audit_seal_sha256": binding.get("source_audit_seal_sha256"),
        "source_revision": binding.get("source_revision"),
        "canonical_runtime_receipt_path": str(QWEN30_EXACT_FULL_TOKEN_RUNTIME_RECEIPT),
        "canonical_runtime_receipt_seal_sha256": runtime_receipt.get("seal_sha256"),
        "runtime_executable_sha256": current_runtime_sha256,
        **contract,
        "production_runtime_deployment_path": (
            str(QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_DEPLOYMENT)
            if contract["production_no_parity"]
            else None
        ),
        "production_runtime_deployment_seal_sha256": (
            runtime_deployment.get("seal_sha256")
            if isinstance(runtime_deployment, Mapping)
            else None
        ),
    }


def _archive_native_http_receipt(path: Path, history_root: Path) -> dict[str, Any] | None:
    """Preserve a sealed current transport observation before a rebind."""

    existing = _sealed_document(path, label=str(path))
    if not isinstance(existing, Mapping) or not _is_sha256(existing.get("seal_sha256")):
        return None
    archive = history_root / f"{path.stem}_{existing['seal_sha256']}{path.suffix}"
    if archive.exists():
        archived = _sealed_document(archive, label=str(archive))
        if not isinstance(archived, Mapping) or archived.get("seal_sha256") != existing.get("seal_sha256"):
            raise Qwen30BootstrapError(
                f"existing native HTTP history archive does not preserve {path.name}"
            )
    else:
        _atomic_json(archive, existing)
    return {
        "path": str(archive),
        "seal_sha256": existing.get("seal_sha256"),
        "document_sha256": _file_sha256(archive),
        "status": existing.get("status"),
    }


def _paired_scalar_order_production_http_adapter_deployment(
    binding: Mapping[str, Any],
    runtime_receipt: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Accept only the server deployment bound to the active production epoch."""

    runtime_binding = _native_http_adapter_runtime_binding(binding, runtime_receipt)
    if not isinstance(runtime_binding, Mapping) or not runtime_binding.get("production_no_parity"):
        return None
    document = _sealed_document(
        QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_HTTP_ADAPTER_DEPLOYMENT,
        label=str(QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_HTTP_ADAPTER_DEPLOYMENT),
    )
    if not isinstance(document, Mapping) or (
        document.get("schema")
        != "hawking.ascension.qwen30_paired_scalar_order_production_http_adapter_deployment.v1"
        or document.get("status")
        != "DEPLOYED_PRODUCTION_NO_PARITY_HTTP_ADAPTER_AWAITING_LIVE_TRANSPORT_SMOKE"
    ):
        return None
    observed = document.get("binding")
    if not isinstance(observed, Mapping):
        return None
    expected_fields = (
        "manifest_seal_sha256",
        "source_audit_seal_sha256",
        "source_revision",
        "canonical_runtime_receipt_path",
        "canonical_runtime_receipt_seal_sha256",
        "runtime_executable_sha256",
        "gate_up_swiglu_kernel_cli",
        "gate_up_swiglu_kernel",
        "kernel_id",
        "custom_kernel_used",
        "production_runtime_deployment_path",
        "production_runtime_deployment_seal_sha256",
    )
    if any(observed.get(field) != runtime_binding.get(field) for field in expected_fields):
        return None
    if not (
        observed.get("source_server_binary_path")
        == str(QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_HTTP_SERVER_SOURCE)
        and observed.get("active_server_binary_path") == str(QWEN30_NATIVE_HTTP_SERVER)
        and _is_sha256(observed.get("source_server_binary_sha256"))
        and _is_sha256(observed.get("active_server_binary_sha256"))
    ):
        return None
    try:
        source_sha256 = _file_sha256(QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_HTTP_SERVER_SOURCE)
        active_sha256 = _file_sha256(QWEN30_NATIVE_HTTP_SERVER)
    except Qwen30BootstrapError:
        return None
    if (
        observed.get("source_server_binary_sha256") != source_sha256
        or observed.get("active_server_binary_sha256") != active_sha256
        or source_sha256 != active_sha256
    ):
        return None
    return dict(document)


def deploy_qwen30_paired_scalar_order_production_http_adapter() -> dict[str, Any]:
    """Atomically install only the bound no-parity adapter binary.

    Deployment is intentionally separate from endpoint launch.  It archives
    old scalar transport controls and refuses to overwrite any live listener,
    then leaves reconciliation to start the production endpoint with the
    explicit kernel CLI.
    """

    binding = _native_runtime_binding()
    runtime_receipt = _sealed_document(
        QWEN30_EXACT_FULL_TOKEN_RUNTIME_RECEIPT,
        label=str(QWEN30_EXACT_FULL_TOKEN_RUNTIME_RECEIPT),
    )
    runtime_binding = _native_http_adapter_runtime_binding(binding, runtime_receipt)
    if not isinstance(runtime_binding, Mapping) or not runtime_binding.get("production_no_parity"):
        raise Qwen30BootstrapError(
            "production HTTP deployment requires the current exact no-parity runtime receipt"
        )
    if not QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_HTTP_SERVER_SOURCE.is_file():
        raise Qwen30BootstrapError(
            "isolated production native HTTP server binary is absent: "
            f"{QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_HTTP_SERVER_SOURCE}"
        )
    active = _read_json(QWEN30_NATIVE_HTTP_ACTIVE)
    if isinstance(active, Mapping) and active.get("phase") == "RUNNING":
        pid = active.get("pid")
        if isinstance(pid, int) and _pid_is_alive(pid):
            raise Qwen30BootstrapError(
                "refusing to replace a native HTTP executable while its recorded listener is alive"
            )
    if _native_http_adapter_health() is not None:
        raise Qwen30BootstrapError(
            "refusing to replace a native HTTP executable while a healthy loopback adapter is listening"
        )
    source_sha256 = _file_sha256(QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_HTTP_SERVER_SOURCE)
    existing = _paired_scalar_order_production_http_adapter_deployment(binding, runtime_receipt)
    if existing is not None:
        return existing

    old_server_archive: dict[str, Any] | None = None
    if QWEN30_NATIVE_HTTP_SERVER.is_file():
        old_server_sha256 = _file_sha256(QWEN30_NATIVE_HTTP_SERVER)
        if old_server_sha256 != source_sha256:
            archive = QWEN30_NATIVE_HTTP_SERVER_HISTORY / (
                f"ascension_qwen30_native_http_server_{old_server_sha256}"
            )
            if archive.exists():
                if _file_sha256(archive) != old_server_sha256:
                    raise Qwen30BootstrapError(
                        "existing native HTTP server history archive has a mismatched SHA-256"
                    )
            else:
                _copy_executable_atomically(QWEN30_NATIVE_HTTP_SERVER, archive)
                if _file_sha256(archive) != old_server_sha256:
                    raise Qwen30BootstrapError(
                        "native HTTP server archive verification failed"
                    )
            old_server_archive = {"path": str(archive), "sha256": old_server_sha256}

    historical_transport = _archive_native_http_receipt(
        QWEN30_NATIVE_HTTP_TRANSPORT_SMOKE,
        QWEN30_NATIVE_HTTP_TRANSPORT_SMOKE_HISTORY,
    )
    historical_chat = _archive_native_http_receipt(
        QWEN30_NATIVE_HTTP_CHAT_SMOKE,
        QWEN30_NATIVE_HTTP_CHAT_SMOKE_HISTORY,
    )
    _copy_executable_atomically(
        QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_HTTP_SERVER_SOURCE,
        QWEN30_NATIVE_HTTP_SERVER,
    )
    active_sha256 = _file_sha256(QWEN30_NATIVE_HTTP_SERVER)
    if active_sha256 != source_sha256:
        raise Qwen30BootstrapError(
            "atomic production native HTTP adapter deployment changed the source binary digest"
        )
    payload = {
        "schema": "hawking.ascension.qwen30_paired_scalar_order_production_http_adapter_deployment.v1",
        "status": "DEPLOYED_PRODUCTION_NO_PARITY_HTTP_ADAPTER_AWAITING_LIVE_TRANSPORT_SMOKE",
        "recorded_at": _utc_now(),
        "binding": {
            **dict(runtime_binding),
            "source_server_binary_path": str(QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_HTTP_SERVER_SOURCE),
            "source_server_binary_sha256": source_sha256,
            "active_server_binary_path": str(QWEN30_NATIVE_HTTP_SERVER),
            "active_server_binary_sha256": active_sha256,
        },
        "historical_controls": {
            "superseded_active_server": old_server_archive,
            "transport_smoke": historical_transport,
            "chat_sse_smoke": historical_chat,
        },
        "claim_boundary": {
            "deployment_is_not_listener_readiness_or_prompt_generation": True,
            "does_not_claim_hcli_coherence_capability_clean_tps_tg_or_tournament": True,
            "requires_live_health_context_and_chat_sse_observation_after_deployment": True,
            "native_direct_packed_metal_only": True,
        },
    }
    sealed = seal(payload)
    _atomic_json(QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_HTTP_ADAPTER_DEPLOYMENT, sealed)
    _write_native_http_adapter_status(
        "NATIVE_HTTP_ADAPTER_PRODUCTION_DEPLOYED_AWAITING_LAUNCH",
        binding=runtime_binding,
        deployment_path=str(QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_HTTP_ADAPTER_DEPLOYMENT),
        deployment_seal_sha256=sealed.get("seal_sha256"),
        server_binary_path=str(QWEN30_NATIVE_HTTP_SERVER),
        server_binary_sha256=active_sha256,
    )
    return sealed


def transition_qwen30_runtime_to_paired_scalar_order_production() -> dict[str, Any]:
    """Seal and deploy the exact no-parity candidate as a fresh runtime epoch.

    The c496 template parity and c8 CPU discriminator are only authority to
    *start* this requalification.  This function archives the scalar control,
    writes the generic fail-closed supersession, then atomically installs the
    isolated production executable.  It never emits a runtime/HCLI/TPS pass.
    """

    active = _read_json(QWEN30_NATIVE_ACTIVE)
    adapter = _read_json(QWEN30_NATIVE_HTTP_ACTIVE)
    if isinstance(active, Mapping) and active.get("phase") == "RUNNING":
        raise Qwen30BootstrapError("cannot transition Qwen30 while a native stage is running")
    if isinstance(adapter, Mapping) and adapter.get("phase") == "RUNNING":
        raise Qwen30BootstrapError("cannot transition Qwen30 while a native HTTP adapter is running")
    if not QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_RUNTIME_SOURCE.is_file():
        raise Qwen30BootstrapError(
            f"isolated paired scalar-order production runtime is absent: {QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_RUNTIME_SOURCE}"
        )
    replacement_sha = _file_sha256(QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_RUNTIME_SOURCE)
    prior = _sealed_document(
        QWEN30_EXACT_FULL_TOKEN_RUNTIME_RECEIPT,
        label=str(QWEN30_EXACT_FULL_TOKEN_RUNTIME_RECEIPT),
    )
    if not isinstance(prior, Mapping):
        raise Qwen30BootstrapError("canonical scalar runtime receipt is absent or unsealed")
    if prior.get("status") == QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_RUNTIME_TRANSITION:
        deployment = _paired_scalar_order_production_deployment()
        if deployment is None:
            raise Qwen30BootstrapError("existing production transition lacks a valid live deployment")
        return deployment
    prior_binding = prior.get("binding")
    if not (
        prior.get("schema") == PHYSICAL_RUNTIME_SCHEMA
        and prior.get("status") == PHYSICAL_RUNTIME_STATUS
        and isinstance(prior_binding, Mapping)
        and _is_sha256(prior.get("seal_sha256"))
        and _is_sha256(prior_binding.get("runtime_executable_sha256"))
        and _is_sha256(prior_binding.get("complete_manifest_seal_sha256"))
    ):
        raise Qwen30BootstrapError("canonical scalar runtime is not an exact native full-token PASS")
    prior_sha = str(prior_binding["runtime_executable_sha256"])
    if _file_sha256(QWEN30_NATIVE_RUNTIME_EXECUTABLE) != prior_sha:
        raise Qwen30BootstrapError("active runtime executable does not match the canonical scalar PASS")
    if replacement_sha == prior_sha:
        raise Qwen30BootstrapError("production no-parity executable must differ from scalar control")
    try:
        from lab.operators.ascension_qwen30_paired_scalar_order_parity import (
            production_no_parity_requalification_binding,
        )

        handoff = production_no_parity_requalification_binding()
    except Exception as exc:
        raise Qwen30BootstrapError(
            f"sealed scalar-order parity handoff is not eligible for production requalification: {exc}"
        ) from exc
    if not (
        handoff.get("scalar_control_runtime_receipt_seal_sha256") == prior.get("seal_sha256")
        and handoff.get("scalar_control_runtime_executable_sha256") == prior_sha
        and handoff.get("production_gate_up_swiglu_kernel")
        == QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_KERNEL_CLI
        and handoff.get("production_kernel_receipt_id")
        == QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_KERNEL
    ):
        raise Qwen30BootstrapError("production handoff is not bound to the live scalar authority")
    prior_seal = str(prior["seal_sha256"])
    history_path = QWEN30_EXACT_FULL_TOKEN_RUNTIME_HISTORY / (
        f"QWEN30_EXACT_FULL_TOKEN_RUNTIME_RECEIPT_{prior_seal}.json"
    )
    if history_path.exists():
        archived = _sealed_document(history_path, label=str(history_path))
        if not isinstance(archived, Mapping) or archived.get("seal_sha256") != prior_seal:
            raise Qwen30BootstrapError("existing scalar PASS archive is malformed")
    else:
        _atomic_json(history_path, prior)
    executable_archive = _archive_qwen30_runtime_executable(prior_sha)
    previous_supersession = _archive_active_qwen30_runtime_supersession()
    transition_controls = _archive_qwen30_runtime_transition_controls(
        superseded_runtime_executable_sha256=prior_sha,
        superseded_runtime_receipt_seal_sha256=prior_seal,
    )
    supersession_payload = {
        "schema": QWEN30_RUNTIME_SUPERSESSION_SCHEMA,
        "status": QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_RUNTIME_TRANSITION,
        "recorded_at": _utc_now(),
        "binding": {
            "model_id": prior_binding.get("model_id"),
            "canonical_runtime_receipt_path": str(QWEN30_EXACT_FULL_TOKEN_RUNTIME_RECEIPT),
            "superseded_runtime_receipt_seal_sha256": prior_seal,
            "defective_runtime_executable_sha256": prior_sha,
            "archived_runtime_receipt_path": str(history_path),
            "archived_runtime_receipt_document_sha256": _file_sha256(history_path),
        },
        "revoked_runtime": {
            "canonical_receipt_path": str(QWEN30_EXACT_FULL_TOKEN_RUNTIME_RECEIPT),
            "canonical_receipt_seal_sha256": prior_seal,
            "runtime_executable_sha256": prior_sha,
            "model_id": prior_binding.get("model_id"),
            "complete_manifest_seal_sha256": prior_binding.get("complete_manifest_seal_sha256"),
        },
        "historical_pass_archive_path": str(history_path),
        "historical_pass_archive_sha256": _file_sha256(history_path),
        "defect": {
            "class": "RUNTIME_EXECUTABLE_IDENTITY_TRANSITION_TO_PAIRED_SCALAR_ORDER_PRODUCTION_NO_PARITY",
            "location": "Qwen30 routed-expert gate/up/SwiGLU command topology",
            "old_runtime_numerical_correctness_defect_asserted": False,
            "effect": "the scalar control remains historical evidence only; no downstream gate may use it after production-kernel deployment",
            "replacement_runtime_executable_sha256": replacement_sha,
            "replacement_runtime_source_path": str(QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_RUNTIME_SOURCE),
            "superseded_runtime_executable_archive": executable_archive,
            "production_no_parity_handoff": handoff,
            "previous_runtime_supersession_history": previous_supersession,
        },
        "invalidates": {
            "canonical_native_runtime_pass": True,
            "all_old_full_token_prompt_and_profile_controls_bound_to_runtime_sha": True,
            "native_http_adapter_and_transport_handoff_bound_to_runtime_sha": True,
            "any_hcli_tps_tg_capability_or_tournament_consumer_of_that_sha": True,
        },
        "required_before_reissue": [
            "fresh all-artifact admission verification in the replacement process",
            "fresh no-parity full-token and two source-template generations",
            "fresh quiet complete-token host-stage GPU profile",
            "separate HCLI transport and clean-performance evidence after runtime reissue",
        ],
        "consumer_contract": {
            "fail_closed_if_canonical_status_is_not_pass": True,
            "fail_closed_if_this_supersession_revokes_the_bound_receipt_seal_or_runtime_executable_sha256": True,
            "historical_archive_is_for_negative_science_only_not_a_gate_authority": True,
        },
        "claim_boundary": {
            "transition_is_not_a_new_runtime_or_quality_receipt": True,
            "template_parity_does_not_select_serve_or_benchmark_the_kernel": True,
            "hcli_tps_tg_capability_and_tournament_remain_closed": True,
            "transition_control_archive": transition_controls,
        },
    }
    supersession = seal(supersession_payload)
    _atomic_json(QWEN30_EXACT_FULL_TOKEN_RUNTIME_SUPERSESSION, supersession)
    canonical_transition = seal(
        {
            "schema": PHYSICAL_RUNTIME_SCHEMA,
            "status": QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_RUNTIME_TRANSITION,
            "recorded_at": _utc_now(),
            "supersession_path": str(QWEN30_EXACT_FULL_TOKEN_RUNTIME_SUPERSESSION),
            "supersession_seal_sha256": supersession["seal_sha256"],
            "revoked_runtime": dict(supersession_payload["revoked_runtime"]),
            "transition": {
                "replacement_runtime_executable_sha256": replacement_sha,
                "production_no_parity_handoff": handoff,
            },
            "claim_boundary": {
                "canonical_filename_is_non_pass_until_fresh_no_parity_chain_earns": True,
                "no_hcli_tps_tg_capability_or_tournament_pass_is_claimed": True,
            },
        }
    )
    _atomic_json(QWEN30_EXACT_FULL_TOKEN_RUNTIME_RECEIPT, canonical_transition)
    _copy_executable_atomically(
        QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_RUNTIME_SOURCE,
        QWEN30_NATIVE_RUNTIME_EXECUTABLE,
    )
    if _file_sha256(QWEN30_NATIVE_RUNTIME_EXECUTABLE) != replacement_sha:
        raise Qwen30BootstrapError("deployed production runtime SHA does not match the sealed replacement")
    deployment = seal(
        {
            "schema": "hawking.ascension.qwen30_paired_scalar_order_production_runtime_deployment.v1",
            "status": "DEPLOYED_AWAITING_FRESH_NATIVE_RUNTIME_REQUALIFICATION",
            "recorded_at": _utc_now(),
            "binding": {
                "runtime_executable_path": str(QWEN30_NATIVE_RUNTIME_EXECUTABLE),
                "replacement_runtime_executable_sha256": replacement_sha,
                "source_runtime_path": str(QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_RUNTIME_SOURCE),
                "source_runtime_sha256": replacement_sha,
                "runtime_supersession_path": str(QWEN30_EXACT_FULL_TOKEN_RUNTIME_SUPERSESSION),
                "runtime_supersession_seal_sha256": supersession["seal_sha256"],
                **handoff,
            },
            "claim_boundary": {
                "deployment_is_not_a_native_runtime_hcli_tps_tg_capability_or_tournament_receipt": True,
                "must_reearn_preflight_full_token_template_a_template_b_and_host_stage_profile": True,
                "adapter_hcli_and_clean_benchmark_remain_closed": True,
            },
        }
    )
    _atomic_json(QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_DEPLOYMENT, deployment)
    _status(
        RUNTIME_ROOT / "QWEN30_COMPLETE_RUNTIME_STATUS.json",
        lane="B_QWEN30_COMPLETE_RUNTIME",
        phase="QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_DEPLOYED_AWAITING_FRESH_PREFLIGHT",
        runtime_supersession_path=str(QWEN30_EXACT_FULL_TOKEN_RUNTIME_SUPERSESSION),
        runtime_supersession_seal_sha256=supersession["seal_sha256"],
        deployment_path=str(QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_DEPLOYMENT),
        deployment_seal_sha256=deployment["seal_sha256"],
        replacement_runtime_executable_sha256=replacement_sha,
        hcli_server_state="CLOSED",
    )
    return deployment


def transition_qwen30_runtime_to_validated_payload_catalog() -> dict[str, Any]:
    """Retire the old runtime receipt before a cache-backed executable runs.

    This is a current-authority transition, not a claim that the corrected
    route-major scalar runtime was numerically defective. Generic consumers
    use the same strict supersession schema as a defect revocation, so the old
    binary cannot retain HCLI/TPS/gate authority while its immutable-payload
    successor has not yet re-earned the complete chain.
    """

    current_executable_sha256 = _file_sha256(QWEN30_NATIVE_RUNTIME_EXECUTABLE)
    prior = _sealed_document(
        QWEN30_EXACT_FULL_TOKEN_RUNTIME_RECEIPT,
        label=str(QWEN30_EXACT_FULL_TOKEN_RUNTIME_RECEIPT),
    )
    if not isinstance(prior, Mapping):
        raise Qwen30BootstrapError("cannot transition Qwen30 runtime: canonical receipt is absent or unsealed")
    prior_binding = prior.get("binding")
    if prior.get("status") == QWEN30_VALIDATED_PAYLOAD_CATALOG_RUNTIME_TRANSITION:
        active = _sealed_document(
            QWEN30_EXACT_FULL_TOKEN_RUNTIME_SUPERSESSION,
            label=str(QWEN30_EXACT_FULL_TOKEN_RUNTIME_SUPERSESSION),
        )
        if (
            isinstance(active, Mapping)
            and active.get("status") == QWEN30_VALIDATED_PAYLOAD_CATALOG_RUNTIME_TRANSITION
        ):
            return dict(active)
        raise Qwen30BootstrapError(
            "canonical runtime is already in cache transition without its active supersession"
        )
    if not (
        prior.get("schema") == PHYSICAL_RUNTIME_SCHEMA
        and prior.get("status") == PHYSICAL_RUNTIME_STATUS
        and isinstance(prior_binding, Mapping)
        and _is_sha256(prior.get("seal_sha256"))
        and _is_sha256(prior_binding.get("runtime_executable_sha256"))
        and _is_sha256(prior_binding.get("complete_manifest_seal_sha256"))
    ):
        raise Qwen30BootstrapError(
            "canonical Qwen30 receipt is not a sealed exact native-runtime PASS eligible for transition"
        )
    prior_seal = str(prior["seal_sha256"])
    prior_runtime_sha = str(prior_binding["runtime_executable_sha256"])
    if current_executable_sha256 == prior_runtime_sha:
        raise Qwen30BootstrapError(
            "refusing runtime transition until the cache-backed executable SHA differs from current receipt"
        )
    history_path = QWEN30_EXACT_FULL_TOKEN_RUNTIME_HISTORY / (
        f"QWEN30_EXACT_FULL_TOKEN_RUNTIME_RECEIPT_{prior_seal}.json"
    )
    if history_path.exists():
        archived = _sealed_document(history_path, label=str(history_path))
        if not isinstance(archived, Mapping) or archived.get("seal_sha256") != prior_seal:
            raise Qwen30BootstrapError("existing runtime history archive does not preserve prior PASS")
    else:
        _atomic_json(history_path, prior)
        archived = _sealed_document(history_path, label=str(history_path))
        if not isinstance(archived, Mapping) or archived.get("seal_sha256") != prior_seal:
            raise Qwen30BootstrapError("failed to preserve prior runtime PASS in immutable history")
    previous_supersession = _archive_active_qwen30_runtime_supersession()
    transition_controls = _archive_qwen30_runtime_transition_controls(
        superseded_runtime_executable_sha256=prior_runtime_sha,
        superseded_runtime_receipt_seal_sha256=prior_seal,
    )
    supersession_payload = {
        "schema": QWEN30_RUNTIME_SUPERSESSION_SCHEMA,
        "status": QWEN30_VALIDATED_PAYLOAD_CATALOG_RUNTIME_TRANSITION,
        "recorded_at": _utc_now(),
        # Names are fixed by the generic v1 consumer contract. The rich
        # transition description below makes clear that this is an identity
        # supersession, not an assertion that the old scalar arithmetic failed.
        "binding": {
            "model_id": prior_binding.get("model_id"),
            "canonical_runtime_receipt_path": str(QWEN30_EXACT_FULL_TOKEN_RUNTIME_RECEIPT),
            "superseded_runtime_receipt_seal_sha256": prior_seal,
            "defective_runtime_executable_sha256": prior_runtime_sha,
            "archived_runtime_receipt_path": str(history_path),
            "archived_runtime_receipt_document_sha256": _file_sha256(history_path),
        },
        "revoked_runtime": {
            "canonical_receipt_path": str(QWEN30_EXACT_FULL_TOKEN_RUNTIME_RECEIPT),
            "canonical_receipt_seal_sha256": prior_seal,
            "runtime_executable_sha256": prior_runtime_sha,
            "model_id": prior_binding.get("model_id"),
            "complete_manifest_seal_sha256": prior_binding.get("complete_manifest_seal_sha256"),
        },
        "historical_pass_archive_path": str(history_path),
        "historical_pass_archive_sha256": _file_sha256(history_path),
        "defect": {
            "class": "RUNTIME_EXECUTABLE_IDENTITY_TRANSITION_TO_VALIDATED_ONCE_DIRECT_PAYLOAD_CATALOG",
            "location": "CompleteBinaryArtifact admission cache -> Qwen30CompleteNativeRuntime::packed_tensor",
            "old_runtime_numerical_correctness_defect_asserted": False,
            "effect": (
                "the old executable remains valid historical scalar-control evidence, but its "
                "per-access payload re-read/SHA path cannot certify the new immutable-payload "
                "runtime or any downstream HCLI/TPS evidence"
            ),
            "replacement_runtime_executable_sha256": current_executable_sha256,
            "previous_architecture_defect_supersession_history": previous_supersession,
        },
        "invalidates": {
            "canonical_native_runtime_pass": True,
            "all_old_full_token_prompt_and_profile_controls_bound_to_runtime_sha": True,
            "native_http_adapter_and_transport_handoff_bound_to_runtime_sha": True,
            "any_hcli_tps_tg_capability_or_tournament_consumer_of_that_sha": True,
        },
        "required_before_reissue": [
            "fresh complete all-artifact SHA/header verification at cache-backed process admission",
            "fresh cache-backed scalar preflight on the replacement executable SHA",
            "fresh cache-backed scalar full-token execution",
            "fresh cache-backed source-template prompt A and B generation",
            "fresh cache-backed complete-token GPU profile",
            "fresh source-bound SIMD and fused-gate/up candidate decisions before any server selection",
            "fresh native HTTP health/context and chat transport observations after runtime reissue",
        ],
        "consumer_contract": {
            "fail_closed_if_canonical_status_is_not_pass": True,
            "fail_closed_if_this_supersession_revokes_the_bound_receipt_seal_or_runtime_executable_sha256": True,
            "historical_archive_is_for_negative_science_only_not_a_gate_authority": True,
        },
        "claim_boundary": {
            "transition_is_not_a_new_runtime_or_quality_receipt": True,
            "old_runtime_is_preserved_as_historical_control_not_current_authority": True,
            "does_not_claim_cache_backed_execution_hcli_tps_tg_capability_or_tournament": True,
            "transition_control_archive": transition_controls,
        },
    }
    supersession = seal(supersession_payload)
    _atomic_json(QWEN30_EXACT_FULL_TOKEN_RUNTIME_SUPERSESSION, supersession)
    canonical_transition = seal(
        {
            "schema": PHYSICAL_RUNTIME_SCHEMA,
            "status": QWEN30_VALIDATED_PAYLOAD_CATALOG_RUNTIME_TRANSITION,
            "recorded_at": _utc_now(),
            "supersession_path": str(QWEN30_EXACT_FULL_TOKEN_RUNTIME_SUPERSESSION),
            "supersession_seal_sha256": supersession["seal_sha256"],
            "historical_pass_archive_path": str(history_path),
            "historical_pass_archive_sha256": _file_sha256(history_path),
            "revoked_runtime": dict(supersession_payload["revoked_runtime"]),
            "transition": {
                "replacement_runtime_executable_sha256": current_executable_sha256,
                "validated_once_immutable_payload_catalog_required": True,
                "old_runtime_numerical_correctness_defect_asserted": False,
                "transition_controls": transition_controls,
            },
            "consumer_contract": dict(supersession_payload["consumer_contract"]),
            "claim_boundary": {
                "canonical_filename_is_non_pass_until_requalification": True,
                "old_pass_is_preserved_only_at_history_path": True,
                "no_runtime_hcli_tps_tg_capability_or_tournament_pass_is_claimed": True,
            },
        }
    )
    _atomic_json(QWEN30_EXACT_FULL_TOKEN_RUNTIME_RECEIPT, canonical_transition)
    _status(
        RUNTIME_ROOT / "QWEN30_COMPLETE_RUNTIME_STATUS.json",
        lane="B_QWEN30_COMPLETE_RUNTIME",
        phase="QWEN30_RUNTIME_TRANSITIONED_TO_VALIDATED_ONCE_PAYLOAD_CATALOG_AWAITING_FRESH_CHAIN",
        runtime_supersession_path=str(QWEN30_EXACT_FULL_TOKEN_RUNTIME_SUPERSESSION),
        runtime_supersession_seal_sha256=supersession["seal_sha256"],
        historical_runtime_pass_path=str(history_path),
        superseded_runtime_executable_sha256=prior_runtime_sha,
        replacement_runtime_executable_sha256=current_executable_sha256,
        transition_control_archive=transition_controls,
    )
    return supersession


def begin_qwen30_validated_payload_catalog_requalification() -> dict[str, Any]:
    """Open an auditable, fresh cache-backed admission sequence.

    The first cache-backed preflight began while the executable-identity
    transition was being sealed.  Its direct observation is useful control
    evidence, but it must not silently satisfy the explicit *post-transition*
    admission requirement.  Preserve that exact JSON first, then remove only
    its mutable canonical stage path so the detached watcher has to run the
    full strict admission scan again.
    """

    active = _read_json(QWEN30_NATIVE_ACTIVE)
    if isinstance(active, Mapping) and active.get("phase") == "RUNNING":
        raise Qwen30BootstrapError(
            "cannot open a fresh cache-backed requalification while a native Qwen30 stage is running"
        )
    canonical = _sealed_document(
        QWEN30_EXACT_FULL_TOKEN_RUNTIME_RECEIPT,
        label=str(QWEN30_EXACT_FULL_TOKEN_RUNTIME_RECEIPT),
    )
    supersession = _sealed_document(
        QWEN30_EXACT_FULL_TOKEN_RUNTIME_SUPERSESSION,
        label=str(QWEN30_EXACT_FULL_TOKEN_RUNTIME_SUPERSESSION),
    )
    if not isinstance(canonical, Mapping) or not isinstance(supersession, Mapping):
        raise Qwen30BootstrapError("cache-backed requalification requires sealed canonical transition records")
    if not (
        canonical.get("schema") == PHYSICAL_RUNTIME_SCHEMA
        and canonical.get("status") == QWEN30_VALIDATED_PAYLOAD_CATALOG_RUNTIME_TRANSITION
        and canonical.get("supersession_path") == str(QWEN30_EXACT_FULL_TOKEN_RUNTIME_SUPERSESSION)
        and canonical.get("supersession_seal_sha256") == supersession.get("seal_sha256")
        and supersession.get("schema") == QWEN30_RUNTIME_SUPERSESSION_SCHEMA
        and supersession.get("status") == QWEN30_VALIDATED_PAYLOAD_CATALOG_RUNTIME_TRANSITION
        and _is_sha256(supersession.get("seal_sha256"))
    ):
        raise Qwen30BootstrapError(
            "canonical runtime authority is not the expected sealed validated-payload-catalog transition"
        )
    defect = supersession.get("defect")
    if not isinstance(defect, Mapping) or not _is_sha256(
        defect.get("replacement_runtime_executable_sha256")
    ):
        raise Qwen30BootstrapError("validated-payload-catalog transition lacks replacement executable binding")
    replacement_sha = str(defect["replacement_runtime_executable_sha256"])
    if _file_sha256(QWEN30_NATIVE_RUNTIME_EXECUTABLE) != replacement_sha:
        raise Qwen30BootstrapError(
            "refusing fresh cache-backed requalification: runtime executable no longer matches transition replacement SHA"
        )

    existing = _sealed_document(
        QWEN30_VALIDATED_PAYLOAD_CATALOG_REQUALIFICATION_EPOCH,
        label=str(QWEN30_VALIDATED_PAYLOAD_CATALOG_REQUALIFICATION_EPOCH),
    )
    if isinstance(existing, Mapping):
        binding = existing.get("binding")
        if (
            existing.get("schema") == QWEN30_VALIDATED_PAYLOAD_CATALOG_REQUALIFICATION_SCHEMA
            and existing.get("status") == QWEN30_VALIDATED_PAYLOAD_CATALOG_REQUALIFICATION_STATUS
            and isinstance(binding, Mapping)
            and binding.get("runtime_supersession_seal_sha256") == supersession.get("seal_sha256")
            and binding.get("replacement_runtime_executable_sha256") == replacement_sha
        ):
            return dict(existing)
        raise Qwen30BootstrapError("existing cache-backed requalification epoch is malformed or binds another runtime")

    archived_controls: list[dict[str, Any]] = []
    current_preflight = _read_json(QWEN30_NATIVE_PREFLIGHT)
    if isinstance(current_preflight, Mapping) and (
        current_preflight.get("runtime_executable_sha256") == replacement_sha
    ):
        try:
            raw_preflight = QWEN30_NATIVE_PREFLIGHT.read_text(encoding="utf-8")
        except OSError as exc:
            raise Qwen30BootstrapError(
                f"cannot preserve unsequenced cache-backed preflight: {exc}"
            ) from exc
        raw_sha256 = _sha256(raw_preflight.encode("utf-8"))
        archive_path = QWEN30_VALIDATED_PAYLOAD_CATALOG_REQUALIFICATION_HISTORY / (
            f"{QWEN30_NATIVE_PREFLIGHT.stem}__runtime_{replacement_sha}"
            f"__before_epoch_{supersession['seal_sha256']}{QWEN30_NATIVE_PREFLIGHT.suffix}"
        )
        if archive_path.exists():
            if _file_sha256(archive_path) != raw_sha256:
                raise Qwen30BootstrapError(
                    "existing unsequenced cache-backed preflight archive differs from canonical control"
                )
        else:
            _atomic_text(archive_path, raw_preflight)
        if _file_sha256(archive_path) != raw_sha256:
            raise Qwen30BootstrapError("unsequenced cache-backed preflight archive verification failed")
        QWEN30_NATIVE_PREFLIGHT.unlink()
        archived_controls.append(
            {
                "label": "preflight_started_before_transition_seal",
                "canonical_path": str(QWEN30_NATIVE_PREFLIGHT),
                "archive_path": str(archive_path),
                "archive_sha256": _file_sha256(archive_path),
                "runtime_executable_sha256": replacement_sha,
                "status": current_preflight.get("status"),
            }
        )

    payload = {
        "schema": QWEN30_VALIDATED_PAYLOAD_CATALOG_REQUALIFICATION_SCHEMA,
        "status": QWEN30_VALIDATED_PAYLOAD_CATALOG_REQUALIFICATION_STATUS,
        "recorded_at": _utc_now(),
        "binding": {
            "model_id": "Qwen3-Coder-30B-A3B-Instruct",
            "canonical_runtime_receipt_path": str(QWEN30_EXACT_FULL_TOKEN_RUNTIME_RECEIPT),
            "canonical_transition_receipt_seal_sha256": canonical.get("seal_sha256"),
            "runtime_supersession_path": str(QWEN30_EXACT_FULL_TOKEN_RUNTIME_SUPERSESSION),
            "runtime_supersession_seal_sha256": supersession.get("seal_sha256"),
            "replacement_runtime_executable_sha256": replacement_sha,
        },
        "preserved_unsequenced_controls": archived_controls,
        "required_stage_order": [
            "fresh complete all-artifact SHA/header verification at cache-backed process admission",
            "fresh cache-backed scalar preflight",
            "fresh cache-backed scalar full-token execution",
            "fresh cache-backed source-template prompt A",
            "fresh cache-backed source-template prompt B",
            "fresh cache-backed complete-token GPU profile",
        ],
        "claim_boundary": {
            "pretransition_or_unsequenced_cache_preflight_is_historical_control_only": True,
            "canonical_runtime_receipt_remains_non_pass_until_all_fresh_stages_earn": True,
            "does_not_claim_native_runtime_generation_hcli_tps_tg_capability_or_tournament": True,
        },
    }
    sealed = seal(payload)
    _atomic_json(QWEN30_VALIDATED_PAYLOAD_CATALOG_REQUALIFICATION_EPOCH, sealed)
    _status(
        RUNTIME_ROOT / "QWEN30_COMPLETE_RUNTIME_STATUS.json",
        lane="B_QWEN30_COMPLETE_RUNTIME",
        phase="QWEN30_VALIDATED_PAYLOAD_CATALOG_REQUALIFICATION_OPEN_AWAITING_FRESH_PREFLIGHT",
        requalification_epoch_path=str(QWEN30_VALIDATED_PAYLOAD_CATALOG_REQUALIFICATION_EPOCH),
        requalification_epoch_seal_sha256=sealed["seal_sha256"],
        preserved_unsequenced_controls=archived_controls,
        replacement_runtime_executable_sha256=replacement_sha,
        required_stage_order=payload["required_stage_order"],
    )
    return sealed


def _shutdown_qwen30_native_http_adapter_for_route_offset_revocation(
    revocation: Mapping[str, Any],
) -> dict[str, Any]:
    """Terminate only the verified old scalar HTTP server and record why."""

    active = _read_json(QWEN30_NATIVE_HTTP_ACTIVE)
    target = revocation.get("revoked_runtime")
    revoked_sha = target.get("runtime_executable_sha256") if isinstance(target, Mapping) else None
    action: dict[str, Any] = {
        "recorded_at": _utc_now(),
        "revocation_path": str(QWEN30_EXACT_FULL_TOKEN_RUNTIME_SUPERSESSION),
        "revoked_runtime_executable_sha256": revoked_sha,
        "active_record_path": str(QWEN30_NATIVE_HTTP_ACTIVE),
        "action": "NO_ACTIVE_SERVER_RECORD",
        "pid": None,
    }
    if isinstance(active, Mapping) and active.get("phase") == "RUNNING":
        pid = active.get("pid")
        action["pid"] = pid
        active_binding = active.get("binding")
        active_matches_revoked_artifact = (
            isinstance(active_binding, Mapping)
            and isinstance(target, Mapping)
            and active_binding.get("manifest_seal_sha256")
            == target.get("complete_manifest_seal_sha256")
        )
        action["active_matches_revoked_artifact"] = active_matches_revoked_artifact
        if not active_matches_revoked_artifact:
            action["action"] = "ACTIVE_SERVER_BINDING_MISMATCH_NOT_TERMINATED"
        elif isinstance(pid, int) and _pid_is_alive(pid):
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError as exc:
                action["action"] = "TERM_FAILED"
                action["error"] = str(exc)
            else:
                deadline = time.monotonic() + 10.0
                while _pid_is_alive(pid) and time.monotonic() < deadline:
                    time.sleep(0.1)
                action["action"] = (
                    "TERM_CONFIRMED" if not _pid_is_alive(pid) else "TERM_SENT_PROCESS_STILL_ALIVE"
                )
        else:
            action["action"] = "SERVER_ALREADY_NOT_ALIVE"
        terminal = {
            **dict(active),
            "phase": "TERMINAL",
            "finished_at": _utc_now(),
            "outcome": "TERMINATED_ROUTE_MAJOR_MOE_ACTIVATION_INPUT_OFFSET_REVOCATION",
            "runtime_supersession_path": str(QWEN30_EXACT_FULL_TOKEN_RUNTIME_SUPERSESSION),
            "runtime_supersession_seal_sha256": revocation.get("seal_sha256"),
            "shutdown_action": action,
        }
        _atomic_json(QWEN30_NATIVE_HTTP_ACTIVE, terminal)
        _atomic_json(QWEN30_NATIVE_HTTP_LAST_PROCESS, terminal)
    _write_native_http_adapter_status(
        "NATIVE_HTTP_ADAPTER_SHUT_DOWN_RUNTIME_RECEIPT_REVOKED",
        binding=_native_runtime_binding(),
        shutdown=action,
        runtime_supersession_path=str(QWEN30_EXACT_FULL_TOKEN_RUNTIME_SUPERSESSION),
        runtime_supersession_seal_sha256=revocation.get("seal_sha256"),
        endpoint_url=None,
    )
    _atomic_json(
        QWEN30_HCLI_HANDOFF,
        {
            "schema": "hawking.ascension.qwen30_native_runtime_hcli_handoff.v1",
            "recorded_at": _utc_now(),
            "status": "REVOKED_ROUTE_MAJOR_MOE_ACTIVATION_INPUT_OFFSET_DEFECT",
            "binding": {"revoked_runtime_executable_sha256": revoked_sha},
            "runtime_supersession_path": str(QWEN30_EXACT_FULL_TOKEN_RUNTIME_SUPERSESSION),
            "runtime_supersession_seal_sha256": revocation.get("seal_sha256"),
            "server_shutdown": action,
            "claim_boundary": {
                "prior_native_http_and_chat_transport_are_not_hcli_authority_after_runtime_revocation": True,
                "corrected_runtime_transport_must_be_observed_again": True,
                "not_a_clean_tps_tg_capability_or_tournament_receipt": True,
            },
        },
    )
    return action


def _build_exact_full_token_runtime_receipt(
    binding: Mapping[str, Any],
    *,
    preflight: Mapping[str, Any] | None,
    full_token: Mapping[str, Any] | None,
    prompt_a: Mapping[str, Any] | None,
    prompt_b: Mapping[str, Any] | None,
    profile_token: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Return a sealed gatekeeper receipt only for exact current evidence.

    This intentionally has no ``BLOCKED`` variant: the physical gatekeeper
    treats the filename as a pass authority, so an incomplete condition leaves
    it absent rather than publishing a pass-shaped scaffold.  The watcher
    status remains the place for detailed pending reasons.
    """

    observations = (preflight, full_token, prompt_a, prompt_b, profile_token)
    if not all(isinstance(item, Mapping) for item in observations):
        return None
    assert preflight is not None and full_token is not None and prompt_a is not None and prompt_b is not None
    assert profile_token is not None
    if _qwen30_runtime_revocation_applies_to_sha(
        _qwen30_route_offset_runtime_revocation(),
        full_token.get("runtime_executable_sha256"),
    ):
        # The old process outputs remain useful controls in their own files,
        # but must never recreate the revoked canonical PASS for this exact
        # defective executable.
        return None
    if not all(_same_current_runtime_binary(item) for item in observations):
        return None
    if not _valid_runtime_binding(full_token, binding) or not _valid_runtime_binding(profile_token, binding):
        return None
    if not _valid_runtime_binding(prompt_a, binding) or not _valid_runtime_binding(prompt_b, binding):
        return None
    preflight_facts = preflight.get("preflight")
    if not isinstance(preflight_facts, Mapping) or not (
        preflight_facts.get("manifest_seal_sha256") == binding.get("manifest_seal_sha256")
        and preflight_facts.get("source_revision") == binding.get("source_revision")
        and preflight_facts.get("tensor_count") == 18_867
        and isinstance(preflight_facts.get("tensor_payload_bytes"), int)
        and preflight_facts.get("tensor_payload_bytes", 0) > 0
        and _is_sha256(preflight_facts.get("tokenizer_sha256"))
        and _is_sha256(preflight_facts.get("source_user_chat_template_sha256"))
        and _is_sha256(preflight_facts.get("tokenizer_config_sha256"))
        and preflight_facts.get("source_user_chat_template_bound") is True
        and preflight_facts.get("complete_exact_tensor_catalog_bound") is True
        and _preflight_has_complete_immutable_payload_catalog(preflight_facts)
    ):
        return None
    full_execution = full_token.get("execution")
    profile_execution = profile_token.get("execution")
    if not isinstance(full_execution, Mapping) or not isinstance(profile_execution, Mapping):
        return None
    if not (
        full_execution.get("all_48_layers_executed") is True
        and full_execution.get("final_norm_lm_head_device_argmax_executed") is True
        and profile_execution.get("all_48_layers_executed") is True
        and profile_execution.get("final_norm_lm_head_device_argmax_executed") is True
        and _profile_token_has_complete_gpu_coverage(profile_token)
    ):
        return None
    prompt_counts = [_positive_forward_count(prompt_a), _positive_forward_count(prompt_b)]
    if any(value is None for value in prompt_counts):
        return None
    for document in (prompt_a, prompt_b):
        execution = document.get("execution")
        if not isinstance(execution, Mapping) or not (
            execution.get("all_48_layers_executed_for_each_forward") is True
            and execution.get("final_norm_lm_head_device_argmax_executed") is True
            and execution.get("autoregressive_feedback_executed") is True
            and _source_user_template_was_applied(document)
        ):
            return None
    # Two separate template-bound executions are not sufficient if the
    # decoder ignores their distinct inputs.  This is intentionally only a
    # prompt-dependence gate (not a coherence or capability judgement): at
    # least one generated token must differ and the applied source-tokenized
    # prompts must themselves differ.  A compact artifact that collapses both
    # prompts to an identical continuation remains an unqualified runtime.
    prompt_a_input_ids = _prompt_ids(prompt_a)
    prompt_b_input_ids = _prompt_ids(prompt_b)
    prompt_a_completion_ids = _completion_ids(prompt_a)
    prompt_b_completion_ids = _completion_ids(prompt_b)
    if (
        prompt_a_input_ids is None
        or prompt_b_input_ids is None
        or prompt_a_completion_ids is None
        or prompt_b_completion_ids is None
        or prompt_a_input_ids == prompt_b_input_ids
        or prompt_a_completion_ids == prompt_b_completion_ids
    ):
        return None

    source_identity = _sealed_document(QWEN30_SOURCE_IDENTITY, label=str(QWEN30_SOURCE_IDENTITY))
    revalidation = _sealed_document(QWEN30_SOURCE_REVALIDATION, label=str(QWEN30_SOURCE_REVALIDATION))
    admission = _sealed_document(QWEN30_ADMISSION_RECEIPT, label=str(QWEN30_ADMISSION_RECEIPT))
    if not isinstance(source_identity, Mapping) or not isinstance(revalidation, Mapping) or not isinstance(admission, Mapping):
        return None
    if not (
        source_identity.get("schema") == "hawking.ascension.qwen_source_content_identity.v1"
        and source_identity.get("status") == "IMMUTABLE_SOURCE_CONTENT_IDENTITY_BOUND"
        and _is_sha256(source_identity.get("content_identity_sha256"))
        and _is_sha256(source_identity.get("seal_sha256"))
        and revalidation.get("schema") == "hawking.ascension.complete_binary_source_revalidation.v1"
        and revalidation.get("status") == "EARNED_CURRENT_SOURCE_SHARDS_REVALIDATED"
        and _is_sha256(revalidation.get("seal_sha256"))
        and admission.get("status")
        == "EARNED_COMPLETE_BINARY_ARTIFACT_ADMITTED_NOT_RUNTIME_OR_CAPABILITY_QUALIFIED"
        and admission.get("seal_sha256") == binding.get("admission_receipt_seal_sha256")
    ):
        return None

    measured_token_count = 2 + sum(value for value in prompt_counts if value is not None)
    full_runtime_binding = full_token.get("runtime_binding")
    if not isinstance(full_runtime_binding, Mapping):
        return None
    gate_up_swiglu_kernel = full_runtime_binding.get("gate_up_swiglu_kernel")
    if not isinstance(gate_up_swiglu_kernel, str) or not gate_up_swiglu_kernel:
        return None
    profile_profiler = profile_token.get("profiler")
    if not isinstance(profile_profiler, Mapping):
        return None
    profile_expected_dispatches = profile_profiler.get("expected_complete_token_dispatch_samples")
    profile_observed_dispatches = profile_profiler.get("gpu_timing_sample_count")
    if not (
        isinstance(profile_expected_dispatches, int)
        and isinstance(profile_observed_dispatches, int)
        and profile_expected_dispatches > 0
        and profile_observed_dispatches == profile_expected_dispatches
    ):
        return None
    evidence = {
        "preflight": {
            "path": str(QWEN30_NATIVE_PREFLIGHT),
            "sha256": _file_sha256(QWEN30_NATIVE_PREFLIGHT),
            "schema": preflight.get("schema"),
            "status": preflight.get("status"),
        },
        "direct_full_token": {
            "path": str(QWEN30_NATIVE_FULL_TOKEN),
            "sha256": _file_sha256(QWEN30_NATIVE_FULL_TOKEN),
            "schema": full_token.get("schema"),
            "status": full_token.get("status"),
        },
        "source_user_prompt_a": {
            "path": str(QWEN30_NATIVE_PROMPT_A),
            "sha256": _file_sha256(QWEN30_NATIVE_PROMPT_A),
            "schema": prompt_a.get("schema"),
            "status": prompt_a.get("status"),
            "full_model_forward_count": prompt_counts[0],
        },
        "source_user_prompt_b": {
            "path": str(QWEN30_NATIVE_PROMPT_B),
            "sha256": _file_sha256(QWEN30_NATIVE_PROMPT_B),
            "schema": prompt_b.get("schema"),
            "status": prompt_b.get("status"),
            "full_model_forward_count": prompt_counts[1],
        },
        "complete_gpu_profile": {
            "path": str(QWEN30_NATIVE_PROFILE_TOKEN),
            "sha256": _file_sha256(QWEN30_NATIVE_PROFILE_TOKEN),
            "schema": profile_token.get("schema"),
            "status": profile_token.get("status"),
            "coverage": (
                f"{profile_observed_dispatches}_of_{profile_expected_dispatches}"
                "_gpu_timed_dispatch_samples"
            ),
            "expected_complete_token_dispatch_samples": profile_expected_dispatches,
            "gpu_timing_sample_count": profile_observed_dispatches,
        },
    }
    payload = {
        "schema": PHYSICAL_RUNTIME_SCHEMA,
        "status": PHYSICAL_RUNTIME_STATUS,
        "recorded_at": _utc_now(),
        "binding": {
            "model_id": "Qwen3-Coder-30B-A3B-Instruct",
            "source_content_identity_sha256": source_identity["content_identity_sha256"],
            "source_revalidation_seal_sha256": revalidation["seal_sha256"],
            "complete_artifact_admission_seal_sha256": admission["seal_sha256"],
            "complete_manifest_seal_sha256": binding["manifest_seal_sha256"],
            "runtime_executable_sha256": full_token["runtime_executable_sha256"],
        },
        "runtime": {
            "native_exact_decoder": True,
            "full_token_execution": True,
            "all_layers_executed": True,
            "all_weight_tensors_bound": True,
            "tokenizer_bound": True,
            "prompt_template_bound": True,
            "distinct_source_template_prompt_continuations_observed": True,
            "model_alone": True,
            "no_fallback": True,
            "raw_bf16_teacher_not_runtime_participant": True,
            "measured_token_count": measured_token_count,
            "timing_scope": "complete_model_token_loop",
            "gate_up_swiglu_kernel": gate_up_swiglu_kernel,
            "custom_kernel_used": gate_up_swiglu_kernel
            == QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_KERNEL,
        },
        "evidence": evidence,
        "claim_boundary": {
            "native_direct_packed_metal_runtime_gate_only": True,
            "does_not_claim_coherence": True,
            "does_not_claim_prompt_quality_or_capability": True,
            "does_not_claim_hcli": True,
            "does_not_claim_clean_tps_or_tg": True,
            "does_not_claim_tournament_qualification": True,
            "diagnostic_profile_is_not_a_clean_tps_measurement": True,
        },
    }
    existing = _sealed_document(
        QWEN30_EXACT_FULL_TOKEN_RUNTIME_RECEIPT,
        label=str(QWEN30_EXACT_FULL_TOKEN_RUNTIME_RECEIPT),
    )
    if isinstance(existing, Mapping) and (
        existing.get("schema") == PHYSICAL_RUNTIME_SCHEMA
        and existing.get("status") == PHYSICAL_RUNTIME_STATUS
        and existing.get("binding") == payload["binding"]
        and existing.get("evidence") == payload["evidence"]
    ):
        return existing
    # A runtime receipt is a mutable canonical pointer.  A correction to
    # evidence metadata must preserve the older, sealed document rather than
    # overwrite it in place.  This is deliberately not an executable
    # supersession: both documents bind the same live binary and raw runtime
    # artifacts; the archive records the historical malformed description.
    if (
        isinstance(existing, Mapping)
        and existing.get("schema") == PHYSICAL_RUNTIME_SCHEMA
        and existing.get("status") == PHYSICAL_RUNTIME_STATUS
        and _is_sha256(existing.get("seal_sha256"))
    ):
        history_path = QWEN30_EXACT_FULL_TOKEN_RUNTIME_HISTORY / (
            "QWEN30_EXACT_FULL_TOKEN_RUNTIME_RECEIPT_"
            f"{existing['seal_sha256']}.json"
        )
        if history_path.exists():
            archived = _sealed_document(history_path, label=str(history_path))
            if not isinstance(archived, Mapping) or archived.get("seal_sha256") != existing.get("seal_sha256"):
                raise Qwen30BootstrapError(
                    "existing runtime receipt history does not preserve the prior canonical receipt"
                )
        else:
            _atomic_json(history_path, existing)
            archived = _sealed_document(history_path, label=str(history_path))
            if not isinstance(archived, Mapping) or archived.get("seal_sha256") != existing.get("seal_sha256"):
                raise Qwen30BootstrapError(
                    "failed to preserve prior canonical runtime receipt before evidence correction"
                )
    sealed = seal(payload)
    _atomic_json(QWEN30_EXACT_FULL_TOKEN_RUNTIME_RECEIPT, sealed)
    return sealed


def _sealed_stage_failure(stage: str, binding: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return a terminal failure for this exact scientific input/binary.

    A failed native stage is a negative result, not permission for the watcher
    to spend the same machine time on the same command indefinitely.  A new
    admitted artifact/source binding, or a rebuilt executable, deliberately
    reopens the experiment.
    """

    prior = _read_json(QWEN30_NATIVE_LAST_PROCESS)
    if not isinstance(prior, Mapping) or prior.get("stage") != stage:
        return None
    outcome = str(prior.get("outcome") or "")
    if outcome.startswith("EARNED_STAGE_RESULT"):
        return None
    observed_binding = prior.get("binding")
    if not isinstance(observed_binding, Mapping):
        return None
    if (
        observed_binding.get("manifest_seal_sha256") != binding.get("manifest_seal_sha256")
        or observed_binding.get("source_revision") != binding.get("source_revision")
        or observed_binding.get("source_audit_seal_sha256") != binding.get("source_audit_seal_sha256")
    ):
        return None
    try:
        current_executable_sha256 = _file_sha256(QWEN30_NATIVE_RUNTIME_EXECUTABLE)
    except Qwen30BootstrapError:
        # The normal executable-not-present branch below will explain this.
        return None
    if prior.get("runtime_executable_sha256") != current_executable_sha256:
        return None
    return dict(prior)


def _simdgroup_component_sources() -> dict[str, str]:
    """Return the exact source hashes covered by the candidate parity gate."""

    paths = {
        "shader_qwen_binary_metal": REPO_ROOT / "crates/hawking-core/shaders/qwen_binary.metal",
        "qwen30_runtime_rust": REPO_ROOT / "crates/hawking-core/src/model/qwen30_complete_runtime.rs",
        "metal_registry_rust": REPO_ROOT / "crates/hawking-core/src/metal/mod.rs",
    }
    return {label: _file_sha256(path) for label, path in paths.items()}


def _current_qwen30_exact_runtime_binding() -> dict[str, Any] | None:
    """Return only the live, sealed corrected runtime binding.

    A SIMD component test is permitted to propose an all-layer candidate only
    when it is tied to the exact native runtime currently accepted by the
    physical gate.  A source-hash-only component receipt is not enough after a
    runtime reissue, because a stale production executable could otherwise be
    paired with a freshly compiled test binary.
    """

    receipt = _sealed_document(
        QWEN30_EXACT_FULL_TOKEN_RUNTIME_RECEIPT,
        label=str(QWEN30_EXACT_FULL_TOKEN_RUNTIME_RECEIPT),
    )
    if not isinstance(receipt, Mapping) or receipt.get("status") != PHYSICAL_RUNTIME_STATUS:
        return None
    binding = receipt.get("binding")
    if not isinstance(binding, Mapping) or not _is_sha256(binding.get("runtime_executable_sha256")):
        return None
    try:
        current_sha = _file_sha256(QWEN30_NATIVE_RUNTIME_EXECUTABLE)
    except Qwen30BootstrapError:
        return None
    if binding.get("runtime_executable_sha256") != current_sha:
        return None
    return {
        "canonical_runtime_receipt_path": str(QWEN30_EXACT_FULL_TOKEN_RUNTIME_RECEIPT),
        "canonical_runtime_receipt_seal_sha256": receipt.get("seal_sha256"),
        "runtime_executable_path": str(QWEN30_NATIVE_RUNTIME_EXECUTABLE),
        "runtime_executable_sha256": current_sha,
        "model_id": binding.get("model_id"),
        "complete_manifest_seal_sha256": binding.get("complete_manifest_seal_sha256"),
    }


def _current_paired_scalar_order_cpu_gate(
    binding: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Return only the exact CPU discriminator bound to the live scalar control.

    This is deliberately a scheduling input, not a model/runtime/kernel
    selection decision.  It makes the next guarded device experiment visible
    to the detached watcher without letting a historical CPU diagnostic reopen
    HCLI, change the scalar runtime, or certify a Metal result.
    """

    receipt = _sealed_document(
        QWEN30_GATEUP_PAIRED_SCALAR_ORDER_CPU_PARITY,
        label=str(QWEN30_GATEUP_PAIRED_SCALAR_ORDER_CPU_PARITY),
    )
    canonical_runtime = _sealed_document(
        QWEN30_EXACT_FULL_TOKEN_RUNTIME_RECEIPT,
        label=str(QWEN30_EXACT_FULL_TOKEN_RUNTIME_RECEIPT),
    )
    if not isinstance(receipt, Mapping) or not isinstance(canonical_runtime, Mapping):
        return None
    if not (
        receipt.get("schema")
        == "hawking.ascension.qwen30_direct_packed_gate_up_precision_order_discriminator.v1"
        and receipt.get("status")
        == "EARNED_CPU_DIRECT_PACKED_GATE_UP_ORDER_PRECISION_DISCRIMINATOR"
        and receipt.get("outcome")
        == "PRECISION_CONTRACTION_DIFFERENCE_OBSERVED_PAIRED_SCALAR_ORDER_CPU_EXACT"
        and canonical_runtime.get("schema") == PHYSICAL_RUNTIME_SCHEMA
        and canonical_runtime.get("status") == PHYSICAL_RUNTIME_STATUS
    ):
        return None
    observed = receipt.get("binding")
    observed_runtime = observed.get("runtime") if isinstance(observed, Mapping) else None
    observations = receipt.get("observations")
    canonical_binding = canonical_runtime.get("binding")
    if not (
        isinstance(observed, Mapping)
        and isinstance(observed_runtime, Mapping)
        and isinstance(observations, Mapping)
        and isinstance(canonical_binding, Mapping)
    ):
        return None
    try:
        current_executable_sha256 = _file_sha256(QWEN30_NATIVE_RUNTIME_EXECUTABLE)
    except Qwen30BootstrapError:
        return None
    exact = (
        observed.get("manifest_seal_sha256") == binding.get("manifest_seal_sha256")
        and observed.get("source_audit_seal_sha256")
        == binding.get("source_audit_seal_sha256")
        and observed.get("source_revision") == binding.get("source_revision")
        and observed_runtime.get("path") == str(QWEN30_EXACT_FULL_TOKEN_RUNTIME_RECEIPT)
        and observed_runtime.get("schema") == PHYSICAL_RUNTIME_SCHEMA
        and observed_runtime.get("status") == PHYSICAL_RUNTIME_STATUS
        and observed_runtime.get("seal_sha256") == canonical_runtime.get("seal_sha256")
        and observed_runtime.get("runtime_executable_sha256") == current_executable_sha256
        and canonical_binding.get("runtime_executable_sha256") == current_executable_sha256
        and observations.get(
            "scalar_control_vs_paired_scalar_order_nonfused_difference_observed"
        )
        is False
    )
    return dict(receipt) if exact else None


def _paired_scalar_order_compile_refusal(
    binding: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Recognize the sealed initial MSL syntax refusal without promoting it.

    The first scalar-order candidate never reached a device dispatch because
    this compiler rejects the C++ ``precise`` type qualifier.  It is negative
    science, not a numerical/template decision, so the watcher must keep the
    scalar/HCLI boundary closed until a separately named successor settles.
    """

    receipt = _sealed_document(
        QWEN30_GATEUP_PAIRED_SCALAR_ORDER_TEMPLATE_PARITY,
        label=str(QWEN30_GATEUP_PAIRED_SCALAR_ORDER_TEMPLATE_PARITY),
    )
    scalar_runtime = _current_qwen30_exact_runtime_binding()
    if not isinstance(receipt, Mapping) or not isinstance(scalar_runtime, Mapping):
        return None
    observed = receipt.get("binding")
    results = receipt.get("candidate_results")
    facts = receipt.get("all_layer_device_parity_and_exact_completion_parity")
    failures = receipt.get("failures")
    if not (
        receipt.get("schema")
        == "hawking.ascension.qwen30_paired_scalar_order_gate_up_template_parity.v1"
        and receipt.get("status")
        == "REJECTED_QWEN30_DIRECT_PACKED_GATE_UP_PAIRED_SCALAR_ORDER_ALL_LAYER_TEMPLATE_PARITY"
        and isinstance(observed, Mapping)
        and isinstance(results, Mapping)
        and isinstance(facts, Mapping)
        and isinstance(failures, list)
        and observed.get("complete_manifest_seal_sha256") == binding.get("manifest_seal_sha256")
        and observed.get("scalar_runtime_receipt_path")
        == scalar_runtime.get("canonical_runtime_receipt_path")
        and observed.get("scalar_runtime_receipt_seal_sha256")
        == scalar_runtime.get("canonical_runtime_receipt_seal_sha256")
        and observed.get("scalar_runtime_executable_sha256")
        == scalar_runtime.get("runtime_executable_sha256")
        and results.get("prompt_a_sha256") is None
        and results.get("prompt_b_sha256") is None
        and not facts
        and any(value == "candidate prompt A returned 2" for value in failures)
    ):
        return None
    return dict(receipt)


def _simdgroup_component_parity() -> dict[str, Any] | None:
    receipt = _sealed_document(
        QWEN30_SIMDGROUP_COMPONENT_PARITY,
        label=str(QWEN30_SIMDGROUP_COMPONENT_PARITY),
    )
    if not isinstance(receipt, Mapping):
        return None
    if receipt.get("schema") != "hawking.ascension.qwen30_binary_simdgroup_component_parity.v1":
        return None
    if receipt.get("status") != "EARNED_QWEN30_PACKED_BINARY_SIMDGROUP_COMPONENT_PARITY":
        return None
    if receipt.get("returncode") != 0:
        return None
    runtime_binding = _current_qwen30_exact_runtime_binding()
    observed_binding = receipt.get("runtime_binding")
    if not isinstance(runtime_binding, Mapping) or observed_binding != runtime_binding:
        return None
    try:
        source_hashes = _simdgroup_component_sources()
    except Qwen30BootstrapError:
        return None
    if receipt.get("candidate_source_sha256") != source_hashes:
        return None
    return dict(receipt)


def run_simdgroup_component_parity() -> None:
    """Run the bounded Metal-vs-CPU gate for the opt-in packed GEMV candidate.

    It intentionally tests a synthetic fixed packed tensor, not a model token.
    CPU is allowed solely as the component oracle. A passing receipt therefore
    permits an *experiment* on the native complete graph; it does not qualify
    the candidate's all-layer numerics, generation, TPS, or HCLI behavior.
    """

    runtime_binding = _current_qwen30_exact_runtime_binding()
    if not isinstance(runtime_binding, Mapping):
        raise Qwen30BootstrapError(
            "SIMD component parity requires a current sealed Qwen30 exact runtime binding"
        )
    stdout_path = RUNTIME_ROOT / "QWEN30_PACKED_BINARY_SIMDGROUP_COMPONENT_PARITY.stdout.log"
    stderr_path = RUNTIME_ROOT / "QWEN30_PACKED_BINARY_SIMDGROUP_COMPONENT_PARITY.stderr.log"
    command = [
        "cargo",
        "test",
        "-p",
        "hawking-core",
        "binary_simdgroup_candidate_matches_scalar_and_packed_cpu_oracle",
        "--lib",
    ]
    environment = os.environ.copy()
    environment["CARGO_TARGET_DIR"] = str(REPO_ROOT / "workspace/ops/build/rust")
    lease_error: str | None = None
    try:
        with _qwen30_native_gpu_quiet_lease(
            stage="qwen30_simdgroup_component_parity_current_runtime"
        ):
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=300,
                env=environment,
                check=False,
            )
        stdout = completed.stdout
        stderr = completed.stderr
        returncode: int | None = completed.returncode
        timeout_error = None
    except subprocess.TimeoutExpired as exc:
        stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
        returncode = None
        timeout_error = "candidate component parity test exceeded 300 seconds"
    except Qwen30BootstrapError as exc:
        stdout = ""
        stderr = ""
        returncode = None
        timeout_error = None
        lease_error = str(exc)
    _atomic_text(stdout_path, stdout)
    _atomic_text(stderr_path, stderr)
    source_hashes: dict[str, str] | None
    try:
        source_hashes = _simdgroup_component_sources()
    except Qwen30BootstrapError:
        source_hashes = None
    expected_test = "binary_simdgroup_candidate_matches_scalar_and_packed_cpu_oracle"
    passed = (
        timeout_error is None
        and returncode == 0
        and expected_test in stdout
        and "test result: ok" in stdout
        and source_hashes is not None
        and lease_error is None
    )
    payload = {
            "schema": "hawking.ascension.qwen30_binary_simdgroup_component_parity.v1",
            "recorded_at": _utc_now(),
            "status": (
                "EARNED_QWEN30_PACKED_BINARY_SIMDGROUP_COMPONENT_PARITY"
                if passed
                else "QWEN30_PACKED_BINARY_SIMDGROUP_COMPONENT_PARITY_FAILED"
            ),
            "command": command,
            "returncode": returncode,
            "timeout_error": timeout_error,
            "lease_error": lease_error,
            "expected_test": expected_test,
            "candidate_source_sha256": source_hashes,
            "runtime_binding": dict(runtime_binding),
            "cargo_target_dir": environment["CARGO_TARGET_DIR"],
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "stdout_sha256": _sha256(stdout.encode("utf-8")),
            "stderr_sha256": _sha256(stderr.encode("utf-8")),
            "claim_boundary": {
                "actual_metal_candidate_compared_with_scalar_metal_and_cpu_packed_oracle": True,
                "component_receipt_is_bound_to_the_current_corrected_scalar_runtime": True,
                "synthetic_component_only_not_a_qwen30_model_token": True,
                "does_not_claim_generation_hcli_clean_tps_tg_capability_or_tournament_qualification": True,
            },
        }
    _atomic_json(QWEN30_SIMDGROUP_COMPONENT_PARITY, seal(payload))


@contextlib.contextmanager
def _qwen30_native_gpu_quiet_lease(*, stage: str) -> Iterator[None]:
    """Use the shared family GPU lock for one bounded native parity run.

    This is a development/parity lease only.  It deliberately does not claim
    the quiet conditions required by a clean qualifying TPS benchmark, but it
    prevents Q30/Q80 component workers from overlapping this all-layer
    comparison.  A busy lock fails immediately rather than creating a hidden
    queue behind another physical experiment.
    """

    QWEN_FAMILY_GPU_LEASE_ROOT.mkdir(parents=True, exist_ok=True)
    with QWEN_FAMILY_GPU_LEASE_LOCK.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise Qwen30BootstrapError(
                "shared Qwen GPU lease is busy; template parity was not started"
            ) from exc
        acquired = _utc_now()
        _atomic_json(
            QWEN_FAMILY_GPU_LEASE_STATUS,
            {
                "schema": "hawking.ascension.gpu_lease.v1",
                "status": "ACTIVE_EXCLUSIVE_GPU_LEASE",
                "worker": "qwen30-native-runtime",
                "pid": os.getpid(),
                "ppid": os.getppid(),
                "stage": stage,
                "acquired_at": acquired,
                "claim_boundary": "development all-layer parity lease only; not a clean qualifying benchmark receipt",
            },
        )
        try:
            yield
        finally:
            _atomic_json(
                QWEN_FAMILY_GPU_LEASE_STATUS,
                {
                    "schema": "hawking.ascension.gpu_lease.v1",
                    "status": "RELEASED",
                    "worker": "qwen30-native-runtime",
                    "pid": os.getpid(),
                    "ppid": os.getppid(),
                    "stage": stage,
                    "acquired_at": acquired,
                    "released_at": _utc_now(),
                    "claim_boundary": "development all-layer parity lease only; not a clean qualifying benchmark receipt",
                },
            )
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def run_route_major_input_offset_metal_regression() -> None:
    """Run the focused native-Metal regression for the repaired MoE offset.

    This is deliberately narrower than a model token: it proves that a direct
    packed binary matvec honors an offset into a route-major activation buffer.
    It therefore protects the exact wiring fix without presenting a component
    check as generation, a complete runtime, HCLI, or a performance result.
    """

    if not QWEN30_NATIVE_RUNTIME_EXECUTABLE.is_file():
        raise Qwen30BootstrapError(
            "cannot run route-major Metal regression before the production runtime exists"
        )
    try:
        binding: Mapping[str, Any] | None = _native_runtime_binding()
    except Qwen30BootstrapError:
        # The kernel unit test remains meaningful even if an independently
        # maintained admission document is temporarily unavailable. Record
        # that absence instead of borrowing a stale artifact binding.
        binding = None
    target_dir = REPO_ROOT / "workspace/ops/build/rust"
    command = [
        "cargo",
        "test",
        "-p",
        "hawking-core",
        "direct_packed_matvec_honors_route_major_input_buffer_offset",
        "--lib",
    ]
    environment = os.environ.copy()
    environment["CARGO_TARGET_DIR"] = str(target_dir)
    started_at = _utc_now()
    completed: subprocess.CompletedProcess[str] | None = None
    failure: str | None = None
    try:
        with _qwen30_native_gpu_quiet_lease(
            stage="qwen30_route_major_input_offset_metal_regression"
        ):
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=600,
                env=environment,
                check=False,
            )
    except (OSError, subprocess.TimeoutExpired, Qwen30BootstrapError) as exc:
        failure = str(exc)
    passed = completed is not None and completed.returncode == 0 and failure is None
    stdout = completed.stdout if completed is not None else ""
    stderr = completed.stderr if completed is not None else ""
    payload = {
        "schema": "hawking.ascension.qwen30_route_major_input_offset_metal_regression.v1",
        "status": (
            "EARNED_DIRECT_PACKED_ROUTE_MAJOR_INPUT_OFFSET_METAL_REGRESSION"
            if passed
            else "FAILED_DIRECT_PACKED_ROUTE_MAJOR_INPUT_OFFSET_METAL_REGRESSION"
        ),
        "recorded_at": _utc_now(),
        "started_at": started_at,
        "finished_at": _utc_now(),
        "binding": {
            "model_id": binding.get("model_id") if isinstance(binding, Mapping) else None,
            "complete_manifest_seal_sha256": (
                binding.get("manifest_seal_sha256") if isinstance(binding, Mapping) else None
            ),
            "source_revision": binding.get("source_revision") if isinstance(binding, Mapping) else None,
            "runtime_executable_path": str(QWEN30_NATIVE_RUNTIME_EXECUTABLE),
            "runtime_executable_sha256": _file_sha256(QWEN30_NATIVE_RUNTIME_EXECUTABLE),
            "runtime_source_path": str(
                REPO_ROOT / "crates/hawking-core/src/model/qwen30_complete_runtime.rs"
            ),
            "runtime_source_sha256": _file_sha256(
                REPO_ROOT / "crates/hawking-core/src/model/qwen30_complete_runtime.rs"
            ),
        },
        "test": {
            "command": command,
            "cargo_target_dir": str(target_dir),
            "test_name": "direct_packed_matvec_honors_route_major_input_buffer_offset",
            "native_metal_required": True,
            "route_major_test_vectors": {
                "first_slice": [1.0, 2.0, 3.0, 4.0],
                "selected_second_slice": [10.0, 20.0, 30.0, 40.0],
                "expected_direct_packed_dot": -20.0,
                "former_offset_zero_dot": -2.0,
            },
            "returncode": completed.returncode if completed is not None else None,
            "stdout_sha256": _sha256(stdout.encode("utf-8")),
            "stderr_sha256": _sha256(stderr.encode("utf-8")),
            "stderr_tail": stderr[-4096:],
            "failure": failure,
        },
        "gpu_lease_status_path": str(QWEN_FAMILY_GPU_LEASE_STATUS),
        "claim_boundary": {
            "proves_only_the_direct_packed_route_major_input_offset_wiring": True,
            "not_a_complete_48_layer_token_or_generation_receipt": True,
            "not_a_runtime_hcli_tps_tg_capability_or_tournament_receipt": True,
            "does_not_measure_or_claim_model_quality": True,
        },
    }
    sealed = seal(payload)
    _atomic_json(QWEN30_ROUTE_MAJOR_INPUT_OFFSET_METAL_REGRESSION, sealed)
    _status(
        RUNTIME_ROOT / "QWEN30_COMPLETE_RUNTIME_STATUS.json",
        lane="B_QWEN30_COMPLETE_RUNTIME",
        phase=(
            "QWEN30_ROUTE_MAJOR_INPUT_OFFSET_METAL_REGRESSION_EARNED"
            if passed
            else "QWEN30_ROUTE_MAJOR_INPUT_OFFSET_METAL_REGRESSION_FAILED"
        ),
        regression_receipt_path=str(QWEN30_ROUTE_MAJOR_INPUT_OFFSET_METAL_REGRESSION),
        regression_receipt_seal_sha256=sealed.get("seal_sha256"),
        runtime_executable_sha256=payload["binding"]["runtime_executable_sha256"],
        claim_boundary={
            "focused_wiring_regression_only_not_a_reearned_native_runtime": True,
            "fresh_scalar_preflight_full_token_template_and_profile_stages_remain_required": True,
        },
    )
    if not passed:
        raise Qwen30BootstrapError(
            "route-major direct-packed Metal regression failed; see "
            f"{QWEN30_ROUTE_MAJOR_INPUT_OFFSET_METAL_REGRESSION}"
        )


def _completion_ids(document: Mapping[str, Any]) -> list[int] | None:
    execution = document.get("execution")
    values = execution.get("completion_token_ids") if isinstance(execution, Mapping) else None
    if not isinstance(values, list) or not values or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values
    ):
        return None
    return list(values)


def _prompt_ids(document: Mapping[str, Any]) -> list[int] | None:
    execution = document.get("execution")
    values = execution.get("prompt_token_ids") if isinstance(execution, Mapping) else None
    if not isinstance(values, list) or not values or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values
    ):
        return None
    return list(values)


def _template_generation_parity(
    control: Mapping[str, Any], candidate: Mapping[str, Any]
) -> tuple[bool, dict[str, Any]]:
    """Compare exact source-template token paths, never approximate text."""

    control_execution = control.get("execution")
    candidate_execution = candidate.get("execution")
    if not isinstance(control_execution, Mapping) or not isinstance(candidate_execution, Mapping):
        return False, {"reason": "missing execution object"}
    control_prompt = _prompt_ids(control)
    candidate_prompt = _prompt_ids(candidate)
    control_completion = _completion_ids(control)
    candidate_completion = _completion_ids(candidate)
    required_candidate_facts = all(
        candidate_execution.get(field) is True
        for field in (
            "all_48_layers_executed_for_each_forward",
            "final_norm_lm_head_device_argmax_executed",
            "autoregressive_feedback_executed",
        )
    ) and _source_user_template_was_applied(candidate)
    exact = (
        required_candidate_facts
        and control_prompt is not None
        and candidate_prompt == control_prompt
        and control_completion is not None
        and candidate_completion == control_completion
        and candidate_execution.get("full_model_forward_count")
        == control_execution.get("full_model_forward_count")
        and candidate_execution.get("completion_feedback_full_forwards")
        == control_execution.get("completion_feedback_full_forwards")
    )
    return exact, {
        "control_prompt_token_ids": control_prompt,
        "candidate_prompt_token_ids": candidate_prompt,
        "control_completion_token_ids": control_completion,
        "candidate_completion_token_ids": candidate_completion,
        "control_full_model_forward_count": control_execution.get("full_model_forward_count"),
        "candidate_full_model_forward_count": candidate_execution.get("full_model_forward_count"),
        "control_completion_feedback_full_forwards": control_execution.get(
            "completion_feedback_full_forwards"
        ),
        "candidate_completion_feedback_full_forwards": candidate_execution.get(
            "completion_feedback_full_forwards"
        ),
        "candidate_required_native_template_facts": required_candidate_facts,
    }


def _run_bounded_native_template_stage(stage: str, binding: Mapping[str, Any]) -> dict[str, Any]:
    """Run one candidate prompt stage, preserving stdout/stderr and exact JSON."""

    result_path, stdout_path, stderr_path = _native_stage_paths(stage)
    command = _native_stage_command(stage, binding)
    started = _utc_now()
    record = {
        "schema": "hawking.ascension.qwen30_simdgroup_template_parity_process.v1",
        "phase": "RUNNING",
        "stage": stage,
        "started_at": started,
        "command": command,
        "runtime_executable_sha256": _file_sha256(QWEN30_NATIVE_RUNTIME_EXECUTABLE),
        "binding": dict(binding),
        "result_path": str(result_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }
    _atomic_json(QWEN30_SIMDGROUP_TEMPLATE_PARITY_ACTIVE, record)
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        stdout, stderr, returncode, timeout_error = (
            completed.stdout,
            completed.stderr,
            completed.returncode,
            None,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        returncode = None
        timeout_error = "bounded candidate source-template stage exceeded 900 seconds"
    _atomic_text(stdout_path, stdout)
    _atomic_text(stderr_path, stderr)
    terminal = {
        **record,
        "phase": "EXITED",
        "finished_at": _utc_now(),
        "returncode": returncode,
        "timeout_error": timeout_error,
        "stdout_sha256": _sha256(stdout.encode("utf-8")),
        "stderr_sha256": _sha256(stderr.encode("utf-8")),
    }
    if timeout_error is not None or returncode != 0:
        terminal["outcome"] = "CANDIDATE_TEMPLATE_STAGE_FAILED"
        _atomic_json(QWEN30_SIMDGROUP_TEMPLATE_PARITY_LAST, terminal)
        _atomic_json(QWEN30_SIMDGROUP_TEMPLATE_PARITY_ACTIVE, {**terminal, "phase": "TERMINAL"})
        raise Qwen30BootstrapError(
            f"{stage} did not complete successfully (returncode={returncode}, timeout={timeout_error})"
        )
    try:
        document = _parse_native_stdout(stdout_path)
    except Qwen30BootstrapError:
        terminal["outcome"] = "CANDIDATE_TEMPLATE_STAGE_INVALID_STDOUT"
        _atomic_json(QWEN30_SIMDGROUP_TEMPLATE_PARITY_LAST, terminal)
        _atomic_json(QWEN30_SIMDGROUP_TEMPLATE_PARITY_ACTIVE, {**terminal, "phase": "TERMINAL"})
        raise
    if not _native_result_matches_document(stage, document, binding):
        terminal["outcome"] = "CANDIDATE_TEMPLATE_STAGE_INVALID_RESULT"
        _atomic_json(QWEN30_SIMDGROUP_TEMPLATE_PARITY_LAST, terminal)
        _atomic_json(QWEN30_SIMDGROUP_TEMPLATE_PARITY_ACTIVE, {**terminal, "phase": "TERMINAL"})
        raise Qwen30BootstrapError(f"{stage} result failed its current artifact/template/native binding")
    _atomic_json(result_path, document)
    terminal["outcome"] = "EARNED_CANDIDATE_TEMPLATE_STAGE_RESULT_WRITTEN"
    terminal["result_sha256"] = _file_sha256(result_path)
    _atomic_json(QWEN30_SIMDGROUP_TEMPLATE_PARITY_LAST, terminal)
    _atomic_json(QWEN30_SIMDGROUP_TEMPLATE_PARITY_ACTIVE, {**terminal, "phase": "TERMINAL"})
    return document


def _archive_stale_simdgroup_template_parity_decision() -> Mapping[str, Any] | None:
    """Preserve an older SIMD A/B decision before its working paths are reused.

    The route-major correction produced a new scalar executable and new prompt
    controls.  Candidate A/B result paths are deliberately reused for the
    next bounded experiment, so a receipt tied to the revoked executable must
    be copied into an immutable, seal-addressed history first.  This avoids a
    historical rejection silently pointing at freshly overwritten evidence.
    """

    prior = _sealed_document(
        QWEN30_SIMDGROUP_TEMPLATE_PARITY,
        label=str(QWEN30_SIMDGROUP_TEMPLATE_PARITY),
    )
    if not isinstance(prior, Mapping):
        return None
    prior_seal = prior.get("seal_sha256")
    if not _is_sha256(prior_seal):
        raise Qwen30BootstrapError("prior SIMD template decision has no valid seal")
    receipt_history_path = (
        QWEN30_SIMDGROUP_TEMPLATE_PARITY_HISTORY
        / f"QWEN30_PACKED_BINARY_SIMDGROUP_TEMPLATE_PARITY_{prior_seal}.json"
    )
    if not receipt_history_path.exists():
        _atomic_json(receipt_history_path, prior)
    candidate_history: dict[str, str] = {}
    for label, source_path in {
        "prompt_a": QWEN30_NATIVE_SIMDGROUP_CANDIDATE_PROMPT_A,
        "prompt_b": QWEN30_NATIVE_SIMDGROUP_CANDIDATE_PROMPT_B,
    }.items():
        candidate = _read_json(source_path)
        if not isinstance(candidate, Mapping):
            continue
        archived = (
            QWEN30_SIMDGROUP_TEMPLATE_PARITY_HISTORY
            / f"{source_path.stem}_{prior_seal}.json"
        )
        if not archived.exists():
            _atomic_json(archived, candidate)
        candidate_history[label] = str(archived)
    manifest_path = (
        QWEN30_SIMDGROUP_TEMPLATE_PARITY_HISTORY
        / f"QWEN30_PACKED_BINARY_SIMDGROUP_TEMPLATE_PARITY_HISTORY_{prior_seal}.json"
    )
    if not manifest_path.exists():
        _atomic_json(
            manifest_path,
            seal(
                {
                    "schema": "hawking.ascension.qwen30_packed_binary_simdgroup_template_parity_history.v1",
                    "recorded_at": _utc_now(),
                    "historical_receipt_path": str(receipt_history_path),
                    "historical_receipt_seal_sha256": prior_seal,
                    "historical_candidate_result_paths": candidate_history,
                    "reason": "current corrected scalar executable/template controls require a new binding-distinct SIMD A/B decision",
                    "claim_boundary": {
                        "history_preserves_prior_negative_science_only": True,
                        "does_not_authorize_reuse_of_old_runtime_or_candidate_outputs": True,
                    },
                }
            ),
        )
    return {
        "history_manifest_path": str(manifest_path),
        "prior_receipt_path": str(receipt_history_path),
        "prior_receipt_seal_sha256": prior_seal,
        "prior_candidate_result_paths": candidate_history,
    }


def run_simdgroup_template_parity() -> None:
    """Execute the two exact source-template controls before serving SIMD work."""

    binding = _native_runtime_binding()
    runtime_binding = _current_qwen30_exact_runtime_binding()
    if not isinstance(runtime_binding, Mapping):
        raise Qwen30BootstrapError(
            "SIMD template parity requires the current sealed scalar runtime receipt"
        )
    if _current_simdgroup_template_parity_decision(binding) is not None:
        raise Qwen30BootstrapError(
            "current SIMD template parity decision is already sealed; refusing to overwrite it"
        )
    control_a = _native_result_matches("prompt-a", binding)
    control_b = _native_result_matches("prompt-b", binding)
    component = _simdgroup_component_parity()
    candidate_token = _native_result_matches("simdgroup-candidate-token", binding)
    control_token = _native_result_matches("full-token", binding)
    prerequisites = {
        "scalar_source_template_prompt_a": control_a is not None,
        "scalar_source_template_prompt_b": control_b is not None,
        "current_component_parity": component is not None,
        "current_all_layer_bos_candidate_matches_control": _simdgroup_candidate_matches_control(
            control_token, candidate_token
        ),
    }
    if not all(prerequisites.values()):
        raise Qwen30BootstrapError(f"SIMD template parity prerequisites missing: {prerequisites}")
    try:
        source_hashes = _simdgroup_component_sources()
    except Qwen30BootstrapError as exc:
        raise Qwen30BootstrapError(f"cannot bind SIMD candidate sources: {exc}") from exc
    superseded_history = _archive_stale_simdgroup_template_parity_decision()
    failures: list[str] = []
    candidate_a: Mapping[str, Any] | None = None
    candidate_b: Mapping[str, Any] | None = None
    details: dict[str, Any] = {}
    with _qwen30_native_gpu_quiet_lease(stage="qwen30_simdgroup_template_ab_parity"):
        try:
            candidate_a = _run_bounded_native_template_stage(
                "simdgroup-candidate-prompt-a", binding
            )
            match_a, detail_a = _template_generation_parity(control_a, candidate_a)
            details["prompt_a"] = detail_a
            if not match_a:
                failures.append("prompt A exact token/template parity differs from scalar control")
            candidate_b = _run_bounded_native_template_stage(
                "simdgroup-candidate-prompt-b", binding
            )
            match_b, detail_b = _template_generation_parity(control_b, candidate_b)
            details["prompt_b"] = detail_b
            if not match_b:
                failures.append("prompt B exact token/template parity differs from scalar control")
        except Qwen30BootstrapError as exc:
            failures.append(str(exc))
    passed = not failures and candidate_a is not None and candidate_b is not None
    payload = {
        "schema": "hawking.ascension.qwen30_packed_binary_simdgroup_template_parity.v1",
        "status": (
            "EARNED_QWEN30_PACKED_BINARY_SIMDGROUP_TEMPLATE_PARITY"
            if passed
            else "REJECTED_QWEN30_PACKED_BINARY_SIMDGROUP_TEMPLATE_PARITY"
        ),
        "recorded_at": _utc_now(),
        "binding": {
            "model_id": "Qwen3-Coder-30B-A3B-Instruct",
            "complete_manifest_seal_sha256": binding.get("manifest_seal_sha256"),
            "source_revision": binding.get("source_revision"),
            "runtime_executable_sha256": _file_sha256(QWEN30_NATIVE_RUNTIME_EXECUTABLE),
            "canonical_runtime_receipt_path": runtime_binding[
                "canonical_runtime_receipt_path"
            ],
            "canonical_runtime_receipt_seal_sha256": runtime_binding[
                "canonical_runtime_receipt_seal_sha256"
            ],
            "candidate_source_sha256": source_hashes,
        },
        "supersedes_stale_candidate_decision": superseded_history,
        "prerequisites": prerequisites,
        "scalar_controls": {
            "prompt_a_path": str(QWEN30_NATIVE_PROMPT_A),
            "prompt_a_sha256": _file_sha256(QWEN30_NATIVE_PROMPT_A),
            "prompt_b_path": str(QWEN30_NATIVE_PROMPT_B),
            "prompt_b_sha256": _file_sha256(QWEN30_NATIVE_PROMPT_B),
        },
        "candidate_controls": {
            "prompt_a_path": str(QWEN30_NATIVE_SIMDGROUP_CANDIDATE_PROMPT_A),
            "prompt_a_sha256": (
                _file_sha256(QWEN30_NATIVE_SIMDGROUP_CANDIDATE_PROMPT_A)
                if candidate_a is not None
                else None
            ),
            "prompt_b_path": str(QWEN30_NATIVE_SIMDGROUP_CANDIDATE_PROMPT_B),
            "prompt_b_sha256": (
                _file_sha256(QWEN30_NATIVE_SIMDGROUP_CANDIDATE_PROMPT_B)
                if candidate_b is not None
                else None
            ),
        },
        "exact_token_parity": details,
        "failures": failures,
        "gpu_lease_status_path": str(QWEN_FAMILY_GPU_LEASE_STATUS),
        "claim_boundary": {
            "two_exact_source_template_prompt_token_paths_compared": True,
            "current_corrected_scalar_runtime_binding_is_distinct_from_archived_prior_decision": True,
            "candidate_is_not_served_unless_this_receipt_is_earned": True,
            "does_not_claim_coherence_hcli_clean_tps_tg_capability_or_tournament": True,
        },
    }
    _atomic_json(QWEN30_SIMDGROUP_TEMPLATE_PARITY, seal(payload))


def _current_simdgroup_template_parity_decision(
    binding: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Return a sealed current decision, whether the SIMD trial passed or failed.

    The scalar server must not be launched in the gap between a BOS-only
    candidate check and the two actual source-template controls.  A rejected
    template result is a valid *decision* for keeping scalar serving; an old
    receipt or an absent decision is not.
    """

    receipt = _sealed_document(
        QWEN30_SIMDGROUP_TEMPLATE_PARITY,
        label=str(QWEN30_SIMDGROUP_TEMPLATE_PARITY),
    )
    if not isinstance(receipt, Mapping) or receipt.get("schema") != (
        "hawking.ascension.qwen30_packed_binary_simdgroup_template_parity.v1"
    ):
        return None
    if receipt.get("status") not in {
        "EARNED_QWEN30_PACKED_BINARY_SIMDGROUP_TEMPLATE_PARITY",
        "REJECTED_QWEN30_PACKED_BINARY_SIMDGROUP_TEMPLATE_PARITY",
    }:
        return None
    observed = receipt.get("binding")
    current_runtime = _current_qwen30_exact_runtime_binding()
    if not isinstance(observed, Mapping) or not isinstance(current_runtime, Mapping):
        return None
    try:
        source_hashes = _simdgroup_component_sources()
    except Qwen30BootstrapError:
        return None
    controls = receipt.get("scalar_controls")
    if not isinstance(controls, Mapping):
        return None
    if not (
        observed.get("complete_manifest_seal_sha256") == binding.get("manifest_seal_sha256")
        and observed.get("source_revision") == binding.get("source_revision")
        and observed.get("runtime_executable_sha256")
        == current_runtime.get("runtime_executable_sha256")
        and observed.get("canonical_runtime_receipt_path")
        == current_runtime.get("canonical_runtime_receipt_path")
        and observed.get("canonical_runtime_receipt_seal_sha256")
        == current_runtime.get("canonical_runtime_receipt_seal_sha256")
        and observed.get("candidate_source_sha256") == source_hashes
        and controls.get("prompt_a_sha256") == _file_sha256(QWEN30_NATIVE_PROMPT_A)
        and controls.get("prompt_b_sha256") == _file_sha256(QWEN30_NATIVE_PROMPT_B)
    ):
        return None
    return dict(receipt)


def _gateup_fused_candidate_sources() -> dict[str, str]:
    """Hash every source file which selects or implements this narrow trial."""

    paths = {
        "fused_gate_up_shader": REPO_ROOT
        / "crates/hawking-core/shaders/qwen_direct_packed_gate_up_swiglu_fused.metal",
        "qwen30_complete_runtime": REPO_ROOT
        / "crates/hawking-core/src/model/qwen30_complete_runtime.rs",
        "metal_registry": REPO_ROOT / "crates/hawking-core/src/metal/mod.rs",
        "native_runtime_entrypoint": REPO_ROOT
        / "crates/hawking-core/examples/ascension_qwen30_complete_native_runtime.rs",
        "packed_matvec_control_shader": REPO_ROOT / "crates/hawking-core/shaders/qwen_binary.metal",
        "runtime_lane": REPO_ROOT / "lab/operators/ascension_qwen30_bootstrap_lanes.py",
    }
    return {label: _file_sha256(path) for label, path in paths.items()}


def _gateup_fused_component_matches_binding(
    binding: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Accept only the source-bound component result; it remains component-only."""

    receipt = _read_json(QWEN30_GATEUP_FUSED_COMPONENT_RAW)
    if not isinstance(receipt, Mapping):
        return None
    if (
        receipt.get("schema")
        != "hawking.ascension.qwen30_direct_packed_gate_up_swiglu_fused_component.v1"
        or receipt.get("status")
        != "PASS_DIRECT_PACKED_QWEN30_GATE_UP_SWIGLU_FUSED_COMPONENT_BENCHMARK_NOT_MODEL_TPS"
    ):
        return None
    observed = receipt.get("binding")
    candidate = receipt.get("candidate")
    parity = receipt.get("parity")
    if not isinstance(observed, Mapping) or not isinstance(candidate, Mapping) or not isinstance(parity, Mapping):
        return None
    if (
        observed.get("manifest_seal_sha256") != binding.get("manifest_seal_sha256")
        or observed.get("source_audit_seal_sha256") != binding.get("source_audit_seal_sha256")
        or observed.get("source_revision") != binding.get("source_revision")
        or candidate.get("id") != "qwen30-direct-packed-all-row-gate-up-swiglu-fused"
        or parity.get("all_within_tolerance") is not True
        or parity.get("fused_vs_baseline_swiglu_max_abs_error") != 0.0
    ):
        return None
    return dict(receipt)


def _build_gateup_fused_candidate_runtime() -> dict[str, Any]:
    """Build an isolated candidate executable without replacing the watcher control."""

    target_dir = QWEN30_GATEUP_FUSED_CANDIDATE_EXECUTABLE.parents[2]
    stdout_path = RUNTIME_ROOT / "QWEN30_DIRECT_PACKED_GATE_UP_SWIGLU_FUSED_CANDIDATE_BUILD.stdout.log"
    stderr_path = RUNTIME_ROOT / "QWEN30_DIRECT_PACKED_GATE_UP_SWIGLU_FUSED_CANDIDATE_BUILD.stderr.log"
    command = [
        "cargo",
        "build",
        "-p",
        "hawking-core",
        "--example",
        "ascension_qwen30_complete_native_runtime",
    ]
    environment = os.environ.copy()
    environment["CARGO_TARGET_DIR"] = str(target_dir)
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
            env=environment,
        )
        stdout, stderr, returncode, timeout_error = (
            completed.stdout,
            completed.stderr,
            completed.returncode,
            None,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        returncode = None
        timeout_error = "isolated fused candidate build exceeded 900 seconds"
    _atomic_text(stdout_path, stdout)
    _atomic_text(stderr_path, stderr)
    built = (
        timeout_error is None
        and returncode == 0
        and QWEN30_GATEUP_FUSED_CANDIDATE_EXECUTABLE.is_file()
    )
    result = {
        "command": command,
        "cargo_target_dir": str(target_dir),
        "returncode": returncode,
        "timeout_error": timeout_error,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "stdout_sha256": _sha256(stdout.encode("utf-8")),
        "stderr_sha256": _sha256(stderr.encode("utf-8")),
        "candidate_executable_path": str(QWEN30_GATEUP_FUSED_CANDIDATE_EXECUTABLE),
        "candidate_executable_sha256": (
            _file_sha256(QWEN30_GATEUP_FUSED_CANDIDATE_EXECUTABLE) if built else None
        ),
        "built": built,
        "claim_boundary": "isolated candidate build only; it does not alter the detached scalar runtime or HTTP adapter",
    }
    if not built:
        raise Qwen30BootstrapError(
            "isolated fused candidate runtime build did not produce its executable"
        )
    return result


def _gateup_fused_candidate_command(
    binding: Mapping[str, Any], *, prompt: str
) -> list[str]:
    return [
        str(QWEN30_GATEUP_FUSED_CANDIDATE_EXECUTABLE),
        "--manifest",
        str(binding["manifest_path"]),
        "--expected-manifest-seal-sha256",
        str(binding["manifest_seal_sha256"]),
        "--expected-source-audit-seal-sha256",
        str(binding["source_audit_seal_sha256"]),
        "--expected-source-revision",
        str(binding["source_revision"]),
        "--mode",
        "generate-greedy",
        "--prompt",
        prompt,
        "--prompt-template",
        "source-user-chat",
        "--packed-matvec-kernel",
        "control",
        "--gate-up-swiglu-kernel",
        "fused-candidate-device-parity",
        "--max-new-tokens",
        "2",
        "--max-seq-len",
        "256",
    ]


def _gateup_fused_candidate_result_matches(
    document: Mapping[str, Any], binding: Mapping[str, Any], *, executable_sha256: str
) -> tuple[bool, dict[str, Any]]:
    """Validate the candidate's full graph and all selected-route device parity."""

    runtime_binding = document.get("runtime_binding")
    execution = document.get("execution")
    if not isinstance(runtime_binding, Mapping) or not isinstance(execution, Mapping):
        return False, {"reason": "missing runtime_binding or execution object"}
    parity = execution.get("gate_up_swiglu_device_control_parity")
    if not isinstance(parity, Mapping):
        return False, {"reason": "missing fused route-major device parity object"}
    full_forwards = execution.get("full_model_forward_count")
    expected_layers = full_forwards * 48 if isinstance(full_forwards, int) and full_forwards > 0 else None
    expected_routes = expected_layers * 8 if isinstance(expected_layers, int) else None
    expected_values = expected_routes * 768 if isinstance(expected_routes, int) else None
    max_error = parity.get("max_abs_error")
    tolerance = parity.get("tolerance_max_abs")
    exact = (
        document.get("schema") == "hawking.ascension.qwen30_complete_native_runtime_result.v1"
        and document.get("status")
        == "EARNED_QWEN30_DIRECT_PACKED_NATIVE_GREEDY_AUTOREGRESSIVE_EXECUTED_UNQUALIFIED"
        and document.get("runtime_executable_sha256") == executable_sha256
        and runtime_binding.get("manifest_seal_sha256") == binding.get("manifest_seal_sha256")
        and runtime_binding.get("source_revision") == binding.get("source_revision")
        and runtime_binding.get("packed_matvec_kernel")
        == "scalar_one_thread_per_row_control"
        and runtime_binding.get("gate_up_swiglu_kernel")
        == "fused_direct_packed_gate_up_swiglu_candidate_with_device_control_parity"
        and execution.get("all_48_layers_executed_for_each_forward") is True
        and execution.get("final_norm_lm_head_device_argmax_executed") is True
        and execution.get("autoregressive_feedback_executed") is True
        and _source_user_template_was_applied(document)
        and parity.get("enabled") is True
        and parity.get("valid") is True
        and parity.get("all_selected_route_major_activations_compared_on_device") is True
        and parity.get("full_model_forwards_without_device_parity") == 0
        and parity.get("full_model_forwards_compared") == full_forwards
        and parity.get("layers_compared") == expected_layers
        and parity.get("routed_experts_compared") == expected_routes
        and parity.get("activation_values_compared") == expected_values
        and isinstance(max_error, (int, float))
        and not isinstance(max_error, bool)
        and isinstance(tolerance, (int, float))
        and not isinstance(tolerance, bool)
        and float(max_error) <= float(tolerance)
    )
    return exact, {
        "candidate_runtime_executable_sha256": document.get("runtime_executable_sha256"),
        "expected_candidate_runtime_executable_sha256": executable_sha256,
        "runtime_binding": dict(runtime_binding),
        "full_model_forward_count": full_forwards,
        "route_major_device_parity": dict(parity),
        "all_required_native_facts": exact,
    }


def _run_gateup_fused_candidate_prompt(
    *, label: str, prompt: str, binding: Mapping[str, Any], executable_sha256: str
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    result_path = (
        QWEN30_NATIVE_GATEUP_FUSED_CANDIDATE_PROMPT_A
        if label == "A"
        else QWEN30_NATIVE_GATEUP_FUSED_CANDIDATE_PROMPT_B
    )
    stdout_path = RUNTIME_ROOT / f"{result_path.stem}.stdout.log"
    stderr_path = RUNTIME_ROOT / f"{result_path.stem}.stderr.log"
    command = _gateup_fused_candidate_command(binding, prompt=prompt)
    active = {
        "schema": "hawking.ascension.qwen30_gate_up_swiglu_fused_template_parity_process.v1",
        "phase": "RUNNING",
        "label": label,
        "started_at": _utc_now(),
        "orchestrator_pid": os.getpid(),
        "orchestrator_ppid": os.getppid(),
        "command": command,
        "candidate_executable_sha256": executable_sha256,
        "binding": dict(binding),
        "result_path": str(result_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }
    try:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        active["child_pid"] = process.pid
        _atomic_json(QWEN30_GATEUP_FUSED_TEMPLATE_PARITY_ACTIVE, active)
        stdout, stderr = process.communicate(timeout=1800)
        returncode, timeout_error = process.returncode, None
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        process.terminate()
        try:
            extra_stdout, extra_stderr = process.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            extra_stdout, extra_stderr = process.communicate()
        stdout += extra_stdout or ""
        stderr += extra_stderr or ""
        returncode = None
        timeout_error = "fused candidate source-template prompt exceeded 1800 seconds"
    _atomic_text(stdout_path, stdout)
    _atomic_text(stderr_path, stderr)
    terminal = {
        **active,
        "phase": "EXITED",
        "finished_at": _utc_now(),
        "returncode": returncode,
        "timeout_error": timeout_error,
        "stdout_sha256": _sha256(stdout.encode("utf-8")),
        "stderr_sha256": _sha256(stderr.encode("utf-8")),
    }
    if timeout_error is not None or returncode != 0:
        terminal["outcome"] = "FUSED_CANDIDATE_TEMPLATE_STAGE_FAILED"
        _atomic_json(QWEN30_GATEUP_FUSED_TEMPLATE_PARITY_LAST, terminal)
        _atomic_json(QWEN30_GATEUP_FUSED_TEMPLATE_PARITY_ACTIVE, {**terminal, "phase": "TERMINAL"})
        raise Qwen30BootstrapError(
            f"fused candidate prompt {label} failed (returncode={returncode}, timeout={timeout_error})"
        )
    try:
        document = _parse_native_stdout(stdout_path)
    except Qwen30BootstrapError:
        terminal["outcome"] = "FUSED_CANDIDATE_TEMPLATE_STAGE_INVALID_STDOUT"
        _atomic_json(QWEN30_GATEUP_FUSED_TEMPLATE_PARITY_LAST, terminal)
        _atomic_json(QWEN30_GATEUP_FUSED_TEMPLATE_PARITY_ACTIVE, {**terminal, "phase": "TERMINAL"})
        raise
    matches, facts = _gateup_fused_candidate_result_matches(
        document, binding, executable_sha256=executable_sha256
    )
    terminal["native_result_matches"] = matches
    terminal["native_result_facts"] = facts
    if not matches:
        terminal["outcome"] = "FUSED_CANDIDATE_TEMPLATE_STAGE_INVALID_RESULT"
        _atomic_json(QWEN30_GATEUP_FUSED_TEMPLATE_PARITY_LAST, terminal)
        _atomic_json(QWEN30_GATEUP_FUSED_TEMPLATE_PARITY_ACTIVE, {**terminal, "phase": "TERMINAL"})
        raise Qwen30BootstrapError(f"fused candidate prompt {label} failed current native route-parity binding")
    _atomic_json(result_path, document)
    terminal["outcome"] = "EARNED_FUSED_CANDIDATE_TEMPLATE_STAGE_RESULT_WRITTEN"
    terminal["result_sha256"] = _file_sha256(result_path)
    _atomic_json(QWEN30_GATEUP_FUSED_TEMPLATE_PARITY_LAST, terminal)
    _atomic_json(QWEN30_GATEUP_FUSED_TEMPLATE_PARITY_ACTIVE, {**terminal, "phase": "TERMINAL"})
    return document, terminal


def run_gateup_fused_template_parity() -> None:
    """Run a bounded all-layer native parity gate for the fused MoE candidate.

    The candidate executable is isolated from the detached scalar runtime and
    server.  It must match the immutable scalar source-template controls and
    compare its device-produced route-major activations with its retained
    direct-packed control at every selected expert before it can be proposed
    for integration.  This function never switches the serving kernel.
    """

    binding = _native_runtime_binding()
    scalar_runtime_binding = _current_qwen30_exact_runtime_binding()
    if not isinstance(scalar_runtime_binding, Mapping):
        raise Qwen30BootstrapError(
            "fused gate/up parity requires the current sealed scalar runtime receipt"
        )
    component = _gateup_fused_component_matches_binding(binding)
    control_a = _native_result_matches("prompt-a", binding)
    control_b = _native_result_matches("prompt-b", binding)
    prerequisites = {
        "current_source_bound_component_receipt": component is not None,
        "immutable_scalar_source_template_prompt_a": control_a is not None,
        "immutable_scalar_source_template_prompt_b": control_b is not None,
        "serving_scalar_control_not_replaced": True,
    }
    build: Mapping[str, Any] | None = None
    failures: list[str] = []
    candidate_a: Mapping[str, Any] | None = None
    candidate_b: Mapping[str, Any] | None = None
    candidate_facts: dict[str, Any] = {}
    try:
        source_hashes = _gateup_fused_candidate_sources()
    except Qwen30BootstrapError as exc:
        source_hashes = None
        failures.append(f"candidate source hash failed: {exc}")
    if all(prerequisites.values()) and source_hashes is not None:
        try:
            build = _build_gateup_fused_candidate_runtime()
        except Qwen30BootstrapError as exc:
            failures.append(str(exc))
    else:
        failures.append(f"fused candidate prerequisites missing: {prerequisites}")
    if build is not None:
        executable_sha256 = build.get("candidate_executable_sha256")
        if not _is_sha256(executable_sha256):
            failures.append("isolated fused candidate executable hash is unavailable")
        else:
            # The shared lease is deliberately acquired only after CPU-only
            # build/preflight work is complete.  A busy Q80 owner is an
            # explicit deferral, not a queue behind an unknown workload.
            try:
                with _qwen30_native_gpu_quiet_lease(
                    stage="qwen30_gateup_fused_all_layer_source_template_parity"
                ):
                    candidate_a, process_a = _run_gateup_fused_candidate_prompt(
                        label="A",
                        prompt="Reply with the single word native.",
                        binding=binding,
                        executable_sha256=executable_sha256,
                    )
                    candidate_facts["prompt_a_process"] = process_a
                    match_a, detail_a = _template_generation_parity(control_a, candidate_a)
                    candidate_facts["prompt_a_exact_token_parity"] = detail_a
                    if not match_a:
                        failures.append("prompt A exact completion/token path differs from scalar control")
                    candidate_b, process_b = _run_gateup_fused_candidate_prompt(
                        label="B",
                        prompt="Write a one-line Python function named add.",
                        binding=binding,
                        executable_sha256=executable_sha256,
                    )
                    candidate_facts["prompt_b_process"] = process_b
                    match_b, detail_b = _template_generation_parity(control_b, candidate_b)
                    candidate_facts["prompt_b_exact_token_parity"] = detail_b
                    if not match_b:
                        failures.append("prompt B exact completion/token path differs from scalar control")
            except Qwen30BootstrapError as exc:
                failures.append(str(exc))
    passed = (
        not failures
        and candidate_a is not None
        and candidate_b is not None
        and build is not None
    )
    payload = {
        "schema": "hawking.ascension.qwen30_direct_packed_gate_up_swiglu_fused_template_parity.v1",
        "status": (
            "EARNED_QWEN30_DIRECT_PACKED_GATE_UP_SWIGLU_FUSED_ALL_LAYER_TEMPLATE_PARITY"
            if passed
            else "REJECTED_QWEN30_DIRECT_PACKED_GATE_UP_SWIGLU_FUSED_ALL_LAYER_TEMPLATE_PARITY"
        ),
        "recorded_at": _utc_now(),
        "binding": {
            "model_id": "Qwen3-Coder-30B-A3B-Instruct",
            "complete_manifest_seal_sha256": binding.get("manifest_seal_sha256"),
            "source_revision": binding.get("source_revision"),
            "scalar_runtime_receipt_path": scalar_runtime_binding[
                "canonical_runtime_receipt_path"
            ],
            "scalar_runtime_receipt_seal_sha256": scalar_runtime_binding[
                "canonical_runtime_receipt_seal_sha256"
            ],
            "scalar_runtime_executable_sha256": scalar_runtime_binding[
                "runtime_executable_sha256"
            ],
            "candidate_runtime_executable_path": (
                build.get("candidate_executable_path") if build is not None else str(QWEN30_GATEUP_FUSED_CANDIDATE_EXECUTABLE)
            ),
            "candidate_runtime_executable_sha256": (
                build.get("candidate_executable_sha256") if build is not None else None
            ),
            "candidate_source_sha256": source_hashes,
        },
        "component_proposal": {
            "path": str(QWEN30_GATEUP_FUSED_COMPONENT_RAW),
            "sha256": (_file_sha256(QWEN30_GATEUP_FUSED_COMPONENT_RAW) if component is not None else None),
            "status": component.get("status") if component is not None else None,
            "component_only_not_model_tps": True,
        },
        "build": dict(build) if build is not None else None,
        "prerequisites": prerequisites,
        "scalar_controls": {
            "prompt_a_path": str(QWEN30_NATIVE_PROMPT_A),
            "prompt_a_sha256": _file_sha256(QWEN30_NATIVE_PROMPT_A) if control_a is not None else None,
            "prompt_b_path": str(QWEN30_NATIVE_PROMPT_B),
            "prompt_b_sha256": _file_sha256(QWEN30_NATIVE_PROMPT_B) if control_b is not None else None,
        },
        "candidate_results": {
            "prompt_a_path": str(QWEN30_NATIVE_GATEUP_FUSED_CANDIDATE_PROMPT_A),
            "prompt_a_sha256": (_file_sha256(QWEN30_NATIVE_GATEUP_FUSED_CANDIDATE_PROMPT_A) if candidate_a is not None else None),
            "prompt_b_path": str(QWEN30_NATIVE_GATEUP_FUSED_CANDIDATE_PROMPT_B),
            "prompt_b_sha256": (_file_sha256(QWEN30_NATIVE_GATEUP_FUSED_CANDIDATE_PROMPT_B) if candidate_b is not None else None),
        },
        "all_layer_device_parity_and_exact_completion_parity": candidate_facts,
        "failures": failures,
        "gpu_lease_status_path": str(QWEN_FAMILY_GPU_LEASE_STATUS),
        "claim_boundary": {
            "candidate_is_not_selected_for_http_server_or_detached_scalar_runtime": True,
            "candidate_uses_direct_admitted_packed_weights_and_native_metal_only": True,
            "device_control_comparison_covers_each_selected_expert_route_for_each_executed_layer": True,
            "does_not_claim_coherence_hcli_clean_tps_tg_capability_or_tournament": True,
            "fresh_complete_token_profile_and_clean_hcli_tps_gate_remain_required": True,
        },
    }
    _atomic_json(QWEN30_GATEUP_FUSED_TEMPLATE_PARITY, seal(payload))


def _current_gateup_fused_template_decision(
    binding: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Return a current *executed* all-layer fused-candidate decision.

    A rejected parity result is useful negative science and can settle the
    candidate decision, but a build/launch failure is not a parity experiment.
    In particular, it must not unblock the scalar HTTP adapter merely because
    it happened to be serialized with a ``REJECTED_*`` status.  Require the
    candidate binary plus both source-template result documents before a
    rejection is allowed to count as an observed decision.
    """

    receipt = _sealed_document(
        QWEN30_GATEUP_FUSED_TEMPLATE_PARITY,
        label=str(QWEN30_GATEUP_FUSED_TEMPLATE_PARITY),
    )
    if not isinstance(receipt, Mapping) or receipt.get("schema") != (
        "hawking.ascension.qwen30_direct_packed_gate_up_swiglu_fused_template_parity.v1"
    ):
        return None
    if receipt.get("status") not in {
        "EARNED_QWEN30_DIRECT_PACKED_GATE_UP_SWIGLU_FUSED_ALL_LAYER_TEMPLATE_PARITY",
        "REJECTED_QWEN30_DIRECT_PACKED_GATE_UP_SWIGLU_FUSED_ALL_LAYER_TEMPLATE_PARITY",
    }:
        return None
    observed = receipt.get("binding")
    scalar_runtime = _current_qwen30_exact_runtime_binding()
    controls = receipt.get("scalar_controls")
    build = receipt.get("build")
    candidate_results = receipt.get("candidate_results")
    candidate_facts = receipt.get("all_layer_device_parity_and_exact_completion_parity")
    if (
        not isinstance(observed, Mapping)
        or not isinstance(scalar_runtime, Mapping)
        or not isinstance(controls, Mapping)
        or not isinstance(build, Mapping)
        or not isinstance(candidate_results, Mapping)
        or not isinstance(candidate_facts, Mapping)
    ):
        return None
    try:
        source_hashes = _gateup_fused_candidate_sources()
    except Qwen30BootstrapError:
        return None
    if not (
        observed.get("complete_manifest_seal_sha256") == binding.get("manifest_seal_sha256")
        and observed.get("source_revision") == binding.get("source_revision")
        and observed.get("scalar_runtime_receipt_path")
        == scalar_runtime.get("canonical_runtime_receipt_path")
        and observed.get("scalar_runtime_receipt_seal_sha256")
        == scalar_runtime.get("canonical_runtime_receipt_seal_sha256")
        and observed.get("scalar_runtime_executable_sha256")
        == scalar_runtime.get("runtime_executable_sha256")
        and observed.get("candidate_source_sha256") == source_hashes
        and controls.get("prompt_a_sha256") == _file_sha256(QWEN30_NATIVE_PROMPT_A)
        and controls.get("prompt_b_sha256") == _file_sha256(QWEN30_NATIVE_PROMPT_B)
    ):
        return None
    candidate_executable_path = build.get("candidate_executable_path")
    candidate_executable_sha256 = build.get("candidate_executable_sha256")
    if not (
        build.get("built") is True
        and isinstance(candidate_executable_path, str)
        and candidate_executable_path
        == observed.get("candidate_runtime_executable_path")
        and _is_sha256(candidate_executable_sha256)
        and candidate_executable_sha256
        == observed.get("candidate_runtime_executable_sha256")
    ):
        return None
    expected_candidate_results = {
        "prompt_a": QWEN30_NATIVE_GATEUP_FUSED_CANDIDATE_PROMPT_A,
        "prompt_b": QWEN30_NATIVE_GATEUP_FUSED_CANDIDATE_PROMPT_B,
    }
    for label, result_path in expected_candidate_results.items():
        path_key = f"{label}_path"
        sha_key = f"{label}_sha256"
        recorded_path = candidate_results.get(path_key)
        recorded_sha256 = candidate_results.get(sha_key)
        process = candidate_facts.get(f"{label}_process")
        if not (
            recorded_path == str(result_path)
            and _is_sha256(recorded_sha256)
            and result_path.is_file()
            and _file_sha256(result_path) == recorded_sha256
            and isinstance(process, Mapping)
            and process.get("outcome")
            == "EARNED_FUSED_CANDIDATE_TEMPLATE_STAGE_RESULT_WRITTEN"
            and process.get("result_path") == str(result_path)
            and process.get("result_sha256") == recorded_sha256
            and process.get("candidate_executable_sha256")
            == candidate_executable_sha256
        ):
            return None
    return dict(receipt)


def revoke_qwen30_route_offset_defect() -> None:
    """Perform the one-way Q30 route-major MoE runtime revocation.

    This command is deliberately explicit rather than folded into a normal
    watcher cycle: it archives a previously sealed PASS, replaces its canonical
    authority with a sealed non-PASS record, and terminates only the matching
    loopback server before any corrected binary can be built or relaunched.
    """

    revocation = _revoke_qwen30_route_offset_runtime_receipt()
    raw_control_history = _archive_route_major_defect_observations(revocation)
    shutdown = _shutdown_qwen30_native_http_adapter_for_route_offset_revocation(revocation)
    _status(
        RUNTIME_ROOT / "QWEN30_COMPLETE_RUNTIME_STATUS.json",
        lane="B_QWEN30_COMPLETE_RUNTIME",
        phase="QWEN30_EXACT_RUNTIME_REVOKED_ROUTE_MAJOR_MOE_INPUT_OFFSET_DEFECT",
        runtime_supersession_path=str(QWEN30_EXACT_FULL_TOKEN_RUNTIME_SUPERSESSION),
        runtime_supersession_seal_sha256=revocation.get("seal_sha256"),
        historical_pass_archive_path=revocation.get("historical_pass_archive_path"),
        historical_raw_control_archive_path=str(QWEN30_ROUTE_MAJOR_DEFECT_HISTORY_MANIFEST),
        historical_raw_control_archive_seal_sha256=raw_control_history.get("seal_sha256"),
        server_shutdown=shutdown,
        required_before_reissue=revocation.get("required_before_reissue"),
        claim_boundary={
            "old_runtime_hcli_transport_and_any_tps_implications_are_rejected": True,
            "corrected_scalar_evidence_must_be_earned_from_a_new_executable": True,
        },
    )


def _direct_sampled_token(result: Mapping[str, Any]) -> int | None:
    execution = result.get("execution")
    step = execution.get("step") if isinstance(execution, Mapping) else None
    token = step.get("sampled_token_id") if isinstance(step, Mapping) else None
    return token if isinstance(token, int) and token >= 0 else None


def _simdgroup_candidate_matches_control(
    control: Mapping[str, Any] | None,
    candidate: Mapping[str, Any] | None,
) -> bool:
    if not isinstance(control, Mapping) or not isinstance(candidate, Mapping):
        return False
    control_token = _direct_sampled_token(control)
    candidate_token = _direct_sampled_token(candidate)
    if control_token is None or candidate_token is None or candidate_token != control_token:
        return False
    execution = candidate.get("execution")
    if not isinstance(execution, Mapping):
        return False
    return (
        execution.get("all_48_layers_executed") is True
        and execution.get("final_norm_lm_head_device_argmax_executed") is True
    )


def _poll_native_stage() -> dict[str, Any] | None:
    """Return running/terminal state; never infer success from a PID alone."""

    global _ACTIVE_NATIVE_PROCESS
    record = _read_json(QWEN30_NATIVE_ACTIVE)
    if record is None or record.get("phase") != "RUNNING":
        return None
    pid = record.get("pid")
    if not isinstance(pid, int):
        return {"state": "INVALID_ACTIVE_RECORD", "record": record}
    if _ACTIVE_NATIVE_PROCESS is not None and _ACTIVE_NATIVE_PROCESS.pid == pid:
        returncode = _ACTIVE_NATIVE_PROCESS.poll()
        if returncode is None:
            return {"state": "RUNNING", "record": record}
        _ACTIVE_NATIVE_PROCESS = None
        return {"state": "TERMINAL", "process": _settle_native_process(record, returncode)}
    if _pid_is_alive(pid):
        return {"state": "INHERITED_RUNNING", "record": record}
    # The watcher was restarted after a child exited. The exit code cannot be
    # reconstructed safely, but a schema/binding-checked JSON result still can.
    return {"state": "TERMINAL", "process": _settle_native_process(record, None)}


def _native_http_adapter_command(binding: Mapping[str, Any]) -> list[str]:
    kernel = _native_http_adapter_kernel_contract()
    return [
        str(QWEN30_NATIVE_HTTP_SERVER),
        "--manifest",
        str(binding["manifest_path"]),
        "--expected-manifest-seal-sha256",
        str(binding["manifest_seal_sha256"]),
        "--expected-source-audit-seal-sha256",
        str(binding["source_audit_seal_sha256"]),
        "--expected-source-revision",
        str(binding["source_revision"]),
        "--bind",
        QWEN30_NATIVE_HTTP_BIND,
        "--max-seq-len",
        "256",
        "--max-output-tokens",
        "16",
        "--gate-up-swiglu-kernel",
        str(kernel["gate_up_swiglu_kernel_cli"]),
    ]


def _native_http_adapter_health() -> dict[str, Any] | None:
    """Ask the real loopback adapter whether its loaded runtime is ready."""

    url = f"http://{QWEN30_NATIVE_HTTP_BIND}/healthz"
    try:
        with urllib.request.urlopen(url, timeout=2.0) as response:
            raw = response.read(16 * 1024)
            if response.status != 200:
                return None
    except (OSError, urllib.error.URLError, urllib.error.HTTPError):
        return None
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(document, Mapping):
        return None
    if not (
        document.get("ready") is True
        and document.get("provider") == "qwen30-direct-packed-native-metal"
        and document.get("model_alone") is True
        and document.get("fallback_count") == 0
    ):
        return None
    return dict(document)


def _native_http_adapter_context() -> dict[str, Any] | None:
    """Read the adapter's actual context surface without treating it as HCLI.

    The direct endpoint can expose a live artifact/template binding before any
    client-level HCLI receipt exists.  Keeping this narrow makes the transport
    smoke useful provenance while preserving the later HCLI/capability gates.
    """

    url = f"http://{QWEN30_NATIVE_HTTP_BIND}/v1/hawking/context"
    try:
        with urllib.request.urlopen(url, timeout=2.0) as response:
            raw = response.read(16 * 1024)
            if response.status != 200:
                return None
    except (OSError, urllib.error.URLError, urllib.error.HTTPError):
        return None
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return dict(document) if isinstance(document, Mapping) else None


def _write_native_http_transport_smoke(
    adapter_binding: Mapping[str, Any],
    *,
    health: Mapping[str, Any],
    context: Mapping[str, Any],
    active: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Seal only a real live health/context transport observation.

    This receipt deliberately cannot satisfy the physical measured-HCLI gate:
    it contains no model prompt, session, quality, clean-rate, capability, or
    manager-operation assertion.  It gives later transport/HCLI work an exact
    server process/binary/artifact baseline without promoting a listener into a
    manager model.
    """

    expected_context = (
        context.get("model_id") == "Qwen3-Coder-30B-A3B-Instruct"
        and context.get("arch") == "Qwen3MoeForCausalLM"
        and context.get("artifact_seal_sha256") == adapter_binding.get("manifest_seal_sha256")
        and context.get("model_alone") is True
        and context.get("fallback_count") == 0
        and context.get("capability_status")
        == "UNQUALIFIED_DIRECT_PACKED_NATIVE_RUNTIME_ONLY"
        and isinstance(context.get("source_chat_template_sha256"), str)
        and isinstance(context.get("tokenizer_config_sha256"), str)
        and context.get("kernel_id") == adapter_binding.get("kernel_id")
        and context.get("custom_kernel_used")
        is adapter_binding.get("custom_kernel_used")
    )
    expected_health = (
        health.get("ready") is True
        and health.get("provider") == "qwen30-direct-packed-native-metal"
        and health.get("model_alone") is True
        and health.get("fallback_count") == 0
        and health.get("kernel_id") == adapter_binding.get("kernel_id")
        and health.get("custom_kernel_used")
        is adapter_binding.get("custom_kernel_used")
    )
    server_binary_sha256 = active.get("server_binary_sha256")
    if (
        not expected_context
        or not expected_health
        or not _is_sha256(server_binary_sha256)
        or server_binary_sha256 != adapter_binding.get("server_binary_sha256")
        or not _native_http_adapter_binding_matches(active, adapter_binding)
    ):
        return None
    payload = {
        "schema": "hawking.ascension.qwen30_native_http_transport_smoke.v1",
        "status": "PASS_DIRECT_PACKED_NATIVE_HTTP_HEALTH_CONTEXT_TRANSPORT_UNQUALIFIED",
        "recorded_at": _utc_now(),
        "binding": {
            **dict(adapter_binding),
            "model_id": "Qwen3-Coder-30B-A3B-Instruct",
            "complete_manifest_seal_sha256": adapter_binding.get("manifest_seal_sha256"),
            "runtime_server_binary_sha256": server_binary_sha256,
            "runtime_server_binary_path": active.get("server_binary_path"),
            "loopback_endpoint": f"http://{QWEN30_NATIVE_HTTP_BIND}",
            "health_path": "/healthz",
            "context_path": "/v1/hawking/context",
        },
        "server_process": {
            "pid": active.get("pid"),
            "started_at": active.get("started_at"),
            "active_record_path": str(QWEN30_NATIVE_HTTP_ACTIVE),
        },
        "observed_health_response": dict(health),
        "observed_context_response": dict(context),
        "observed_health_response_sha256": _sha256(
            json.dumps(dict(health), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ),
        "observed_context_response_sha256": _sha256(
            json.dumps(dict(context), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ),
        "claim_boundary": {
            "actual_live_loopback_health_and_context_observed": True,
            "does_not_claim_prompt_generation": True,
            "does_not_claim_hcli": True,
            "does_not_claim_coherence_capability_clean_tps_tg_or_tournament": True,
            "does_not_claim_session_or_manager_operations": True,
        },
    }
    existing = _sealed_document(
        QWEN30_NATIVE_HTTP_TRANSPORT_SMOKE,
        label=str(QWEN30_NATIVE_HTTP_TRANSPORT_SMOKE),
    )
    if isinstance(existing, Mapping) and (
        existing.get("schema") == payload["schema"]
        and existing.get("status") == payload["status"]
        and existing.get("binding") == payload["binding"]
        and existing.get("observed_health_response_sha256")
        == payload["observed_health_response_sha256"]
        and existing.get("observed_context_response_sha256")
        == payload["observed_context_response_sha256"]
    ):
        return dict(existing)
    if isinstance(existing, Mapping) and isinstance(existing.get("seal_sha256"), str):
        # A rebuilt server is a materially different transport observation.
        # Retain the prior bounded live control instead of silently rewriting
        # history when the current receipt is refreshed for its new binary.
        historical = QWEN30_NATIVE_HTTP_TRANSPORT_SMOKE_HISTORY / (
            f"QWEN30_NATIVE_HTTP_TRANSPORT_SMOKE_{existing['seal_sha256']}.json"
        )
        if not historical.exists():
            _atomic_json(historical, existing)
    sealed = seal(payload)
    _atomic_json(QWEN30_NATIVE_HTTP_TRANSPORT_SMOKE, sealed)
    return sealed


def _http_post_sse(
    path: str,
    payload: Mapping[str, Any],
    *,
    headers: Mapping[str, str] = (),
) -> tuple[int, dict[str, str], bytes]:
    """Make one bounded actual loopback SSE request to the native adapter."""

    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        f"http://{QWEN30_NATIVE_HTTP_BIND}{path}",
        data=encoded,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "text/event-stream", **dict(headers)},
    )
    try:
        with urllib.request.urlopen(request, timeout=180.0) as response:
            body = response.read(256 * 1024 + 1)
            if len(body) > 256 * 1024:
                raise Qwen30BootstrapError("bounded native HTTP SSE response exceeded 256 KiB")
            return (
                int(response.status),
                {key.lower(): value for key, value in response.headers.items()},
                body,
            )
    except urllib.error.HTTPError as exc:
        body = exc.read(256 * 1024 + 1)
        return (
            int(exc.code),
            {key.lower(): value for key, value in exc.headers.items()},
            body[: 256 * 1024],
        )
    except (OSError, urllib.error.URLError) as exc:
        raise Qwen30BootstrapError(f"native HTTP SSE request to {path} failed: {exc}") from exc


def _parse_sse_events(raw: bytes, *, label: str) -> list[Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Qwen30BootstrapError(f"{label} SSE body is not UTF-8: {exc}") from exc
    events: list[Any] = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        lines = [line for line in block.splitlines() if line.startswith("data: ")]
        if len(lines) != 1:
            raise Qwen30BootstrapError(f"{label} SSE event has an unexpected framing shape")
        payload = lines[0].removeprefix("data: ")
        if payload == "[DONE]":
            events.append("[DONE]")
            continue
        try:
            event = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise Qwen30BootstrapError(f"{label} SSE event is not JSON: {exc}") from exc
        if not isinstance(event, Mapping):
            raise Qwen30BootstrapError(f"{label} SSE event JSON root is not an object")
        events.append(dict(event))
    if not events or events[-1] != "[DONE]":
        raise Qwen30BootstrapError(f"{label} SSE response has no terminal [DONE] event")
    return events


def _positive_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    observed = float(value)
    return observed if math.isfinite(observed) and observed > 0 else None


def _native_sse_transport_observation(raw: bytes) -> dict[str, Any]:
    events = _parse_sse_events(raw, label="native")
    token_events = [event for event in events if isinstance(event, Mapping) and isinstance(event.get("text"), str)]
    stats_events = [event.get("stats") for event in events if isinstance(event, Mapping) and isinstance(event.get("stats"), Mapping)]
    if not token_events or len(stats_events) != 1:
        raise Qwen30BootstrapError("native SSE did not contain one or more text events and exactly one stats event")
    stats = dict(stats_events[0])
    for field in ("full_token_execution", "all_layers_executed", "native_direct_packed_metal", "model_alone"):
        if stats.get(field) is not True:
            raise Qwen30BootstrapError(f"native SSE stats did not prove {field}")
    if stats.get("fallback_count") != 0:
        raise Qwen30BootstrapError("native SSE stats reported a fallback")
    if not isinstance(stats.get("completed_decode_forwards"), int) or stats["completed_decode_forwards"] <= 0:
        raise Qwen30BootstrapError("native SSE stats lack positive completed_decode_forwards")
    if _positive_float(stats.get("decode_ms")) is None or _positive_float(stats.get("dec_tps")) is None:
        raise Qwen30BootstrapError("native SSE stats lack positive diagnostic decode timing")
    completion = "".join(str(event["text"]) for event in token_events)
    if not completion:
        raise Qwen30BootstrapError("native SSE emitted no completion text")
    return {
        "completion_text_sha256": _sha256(completion.encode("utf-8")),
        "completion_bytes": len(completion.encode("utf-8")),
        "raw_sse_sha256": _sha256(raw),
        "stats": stats,
    }


def _chat_sse_transport_observation(
    raw: bytes,
    headers: Mapping[str, str],
    *,
    session_id: str,
    expected_kernel_id: str,
    expected_custom_kernel_used: bool,
) -> dict[str, Any]:
    if headers.get("x-hawking-session-id") != session_id:
        raise Qwen30BootstrapError("OpenAI chat SSE response did not echo X-Hawking-Session-Id")
    events = _parse_sse_events(raw, label="OpenAI chat")
    token_events: list[Mapping[str, Any]] = []
    telemetry: Mapping[str, Any] | None = None
    terminal: Mapping[str, Any] | None = None
    for event in events:
        if not isinstance(event, Mapping):
            continue
        choices = event.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
            delta = choices[0].get("delta")
            if isinstance(delta, Mapping) and isinstance(delta.get("content"), str):
                token_events.append(event)
        observed_telemetry = event.get("hawking_manager_operations")
        if isinstance(observed_telemetry, Mapping):
            telemetry = observed_telemetry
        if isinstance(event.get("hawking_direct_packed_stats"), Mapping):
            terminal = event
    if not token_events or telemetry is None or terminal is None:
        raise Qwen30BootstrapError("OpenAI chat SSE lacks content, telemetry, or terminal direct-packed stats")
    completion = "".join(
        str(_mapping(_mapping(event.get("choices", [{}])[0]).get("delta")).get("content", ""))
        for event in token_events
    )
    if not completion:
        raise Qwen30BootstrapError("OpenAI chat SSE emitted no assistant content")
    if telemetry.get("session_id") != session_id or telemetry.get("session_header") != "X-Hawking-Session-Id":
        raise Qwen30BootstrapError("OpenAI chat telemetry does not bind the requested session header")
    if telemetry.get("gravity_artifact_id") != "Qwen30-Gravity-Manager-Artifact":
        raise Qwen30BootstrapError("OpenAI chat telemetry has an unexpected Gravity artifact id")
    if telemetry.get("no_fallback") is not True or telemetry.get("native_direct_packed_metal") is not True:
        raise Qwen30BootstrapError("OpenAI chat telemetry did not bind direct packed no-fallback execution")
    if (
        telemetry.get("kernel_id") != expected_kernel_id
        or telemetry.get("custom_kernel_used") is not expected_custom_kernel_used
    ):
        raise Qwen30BootstrapError("OpenAI chat telemetry did not bind the active production kernel")
    if telemetry.get("session_state_supported") is not False or telemetry.get("context_reused") is not False:
        raise Qwen30BootstrapError("bounded chat adapter unexpectedly claimed durable session/KV semantics")
    controls = telemetry.get("manager_operations_controls")
    if not isinstance(controls, Mapping) or controls.get("available") is not False:
        raise Qwen30BootstrapError("bounded chat adapter did not honestly mark manager controls unavailable")
    stats = terminal.get("hawking_direct_packed_stats")
    assert isinstance(stats, Mapping)
    if (
        stats.get("kernel_id") != expected_kernel_id
        or stats.get("custom_kernel_used") is not expected_custom_kernel_used
    ):
        raise Qwen30BootstrapError("OpenAI chat terminal stats did not bind the active production kernel")
    native = _native_sse_transport_observation(
        b"".join(
            b"data: " + json.dumps(
                {"text": _mapping(_mapping(event.get("choices", [{}])[0]).get("delta")).get("content")},
                separators=(",", ":"),
            ).encode("utf-8") + b"\n\n"
            for event in token_events
        )
        + b"data: "
        + json.dumps({"stats": dict(stats)}, separators=(",", ":")).encode("utf-8")
        + b"\n\n"
        + b"data: [DONE]\n\n"
    )
    return {
        "session_id_sha256": _sha256(session_id.encode("utf-8")),
        "completion_text_sha256": _sha256(completion.encode("utf-8")),
        "completion_bytes": len(completion.encode("utf-8")),
        "raw_sse_sha256": _sha256(raw),
        "response_header_session_id_sha256": _sha256(headers["x-hawking-session-id"].encode("utf-8")),
        "telemetry": dict(telemetry),
        "direct_packed_stats": native["stats"],
    }


def run_native_http_chat_smoke() -> dict[str, Any]:
    """Run two fixed production-bound chat SSE controls, unscored by design."""

    binding = _native_runtime_binding()
    runtime_receipt = _sealed_document(
        QWEN30_EXACT_FULL_TOKEN_RUNTIME_RECEIPT,
        label=str(QWEN30_EXACT_FULL_TOKEN_RUNTIME_RECEIPT),
    )
    runtime_binding = _native_http_adapter_runtime_binding(binding, runtime_receipt)
    deployment = _paired_scalar_order_production_http_adapter_deployment(
        binding, runtime_receipt
    )
    transport_receipt = _sealed_document(
        QWEN30_NATIVE_HTTP_TRANSPORT_SMOKE,
        label=str(QWEN30_NATIVE_HTTP_TRANSPORT_SMOKE),
    )
    active = _read_json(QWEN30_NATIVE_HTTP_ACTIVE)
    if not isinstance(runtime_binding, Mapping) or not runtime_binding.get("production_no_parity"):
        raise Qwen30BootstrapError(
            "production chat smoke requires the current exact no-parity runtime receipt"
        )
    if not isinstance(deployment, Mapping):
        raise Qwen30BootstrapError(
            "production chat smoke requires a current sealed HTTP adapter deployment"
        )
    deployment_binding = deployment.get("binding")
    assert isinstance(deployment_binding, Mapping)
    adapter_binding = {
        **dict(runtime_binding),
        "production_http_adapter_deployment_path": str(
            QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_HTTP_ADAPTER_DEPLOYMENT
        ),
        "production_http_adapter_deployment_seal_sha256": deployment.get("seal_sha256"),
        "server_binary_path": str(QWEN30_NATIVE_HTTP_SERVER),
        "server_binary_sha256": deployment_binding.get("active_server_binary_sha256"),
    }
    health = _native_http_adapter_health()
    context = _native_http_adapter_context()
    if not isinstance(transport_receipt, Mapping):
        raise Qwen30BootstrapError("sealed live health/context transport receipt is required before chat smoke")
    if not isinstance(active, Mapping) or not _native_http_adapter_binding_matches(active, adapter_binding):
        raise Qwen30BootstrapError(
            "current native HTTP adapter active record does not bind the current production runtime/server"
        )
    transport_binding = transport_receipt.get("binding")
    if not isinstance(transport_binding, Mapping) or any(
        transport_binding.get(field) != adapter_binding.get(field)
        for field in adapter_binding
    ) or (
        transport_binding.get("runtime_server_binary_sha256")
        != active.get("server_binary_sha256")
    ):
        raise Qwen30BootstrapError(
            "health/context smoke does not bind the current production native HTTP adapter"
        )
    if health is None or context is None:
        raise Qwen30BootstrapError("native HTTP adapter is not presently health/context ready")
    if (
        health.get("ready") is not True
        or health.get("kernel_id") != adapter_binding.get("kernel_id")
        or health.get("custom_kernel_used") is not adapter_binding.get("custom_kernel_used")
        or context.get("hcli_complete_token_telemetry_available") is not True
        or context.get("artifact_seal_sha256") != binding.get("manifest_seal_sha256")
        or context.get("kernel_id") != adapter_binding.get("kernel_id")
        or context.get("custom_kernel_used") is not adapter_binding.get("custom_kernel_used")
    ):
        raise Qwen30BootstrapError("native HTTP adapter health/context did not bind production transport facts")
    prompts = (
        ("A", "Reply with the single word native.", "qwen30-chat-smoke-a"),
        ("B", "Write a one-line Python function named add.", "qwen30-chat-smoke-b"),
    )
    observations: list[dict[str, Any]] = []
    # These requests execute complete direct-packed forwards.  Hold the shared
    # Qwen lease across both controls so a Q80 stage cannot overlap this actual
    # transport evidence.  This is a smoke lease, never a clean TPS lease.
    with _qwen30_native_gpu_quiet_lease(
        stage="qwen30_paired_scalar_order_production_native_http_chat_sse_smoke"
    ):
        for label, prompt, session_id in prompts:
            status, headers, raw = _http_post_sse(
                "/v1/chat/completions",
                {
                    "model": "Qwen3-Coder-30B-A3B-Instruct",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "stream": True,
                    "max_tokens": 2,
                },
                headers={"X-Hawking-Session-Id": session_id},
            )
            if status != 200 or "text/event-stream" not in headers.get("content-type", ""):
                raise Qwen30BootstrapError(
                    f"OpenAI chat transport control {label} returned HTTP {status}"
                )
            observations.append(
                {
                    "label": label,
                    "source_user_prompt_sha256": _sha256(prompt.encode("utf-8")),
                    "request_max_tokens": 2,
                    **_chat_sse_transport_observation(
                        raw,
                        headers,
                        session_id=session_id,
                        expected_kernel_id=str(adapter_binding["kernel_id"]),
                        expected_custom_kernel_used=bool(
                            adapter_binding["custom_kernel_used"]
                        ),
                    ),
                }
            )
    if observations[0]["completion_text_sha256"] == observations[1]["completion_text_sha256"]:
        raise Qwen30BootstrapError(
            "two distinct source-template chat controls produced identical output bytes"
        )
    payload = {
        "schema": "hawking.ascension.qwen30_direct_packed_hcli_transport_smoke.v1",
        "status": "EARNED_DIRECT_PACKED_NATIVE_CHAT_SSE_TRANSPORT_HCLI_UNQUALIFIED",
        "recorded_at": _utc_now(),
        "binding": {
            **adapter_binding,
            "model_id": "Qwen3-Coder-30B-A3B-Instruct",
            "source_content_identity_sha256": runtime_receipt.get("binding", {}).get(
                "source_content_identity_sha256"
            ),
            "source_revalidation_seal_sha256": runtime_receipt.get("binding", {}).get(
                "source_revalidation_seal_sha256"
            ),
            "complete_artifact_admission_seal_sha256": binding.get(
                "admission_receipt_seal_sha256"
            ),
            "complete_manifest_seal_sha256": binding.get("manifest_seal_sha256"),
            "endpoint": f"http://{QWEN30_NATIVE_HTTP_BIND}",
            "chat_path": "/v1/chat/completions",
            "session_header": "X-Hawking-Session-Id",
        },
        "health_context_transport_receipt": {
            "path": str(QWEN30_NATIVE_HTTP_TRANSPORT_SMOKE),
            "seal_sha256": transport_receipt.get("seal_sha256"),
        },
        "measurement": {
            "uses_exact_native_runtime": True,
            "model_alone": True,
            "no_fallback": True,
            "prompt_dependent_transport_generation": True,
            "measured_request_count": len(observations),
            "completed_generated_tokens": sum(
                int(row["direct_packed_stats"].get("completion_tokens", 0))
                for row in observations
            ),
            "openai_chat_sse_framing_verified": True,
            "source_template_bound_one_user_message_only": True,
            "exclusive_qwen_gpu_lease_held_for_complete_forward_requests": True,
            "observations": observations,
            "manager_operations_state": "NOT_YET_RUN_UNEARNED",
            "durable_session_kv_state": "NOT_IMPLEMENTED_UNEARNED",
            "coherence": "UNSCORED_NOT_A_CAPABILITY_EVALUATION",
            "clean_tps": "NOT_MEASURED",
        },
        "claim_boundary": {
            "actual_loopback_openai_chat_sse_transport_observed": True,
            "does_not_claim_hcli_pass": True,
            "does_not_claim_coherence_or_capability": True,
            "does_not_claim_durable_sessions_kv_or_manager_operations": True,
            "does_not_claim_clean_tps_tg_or_tournament_qualification": True,
            "diagnostic_decode_fields_are_not_clean_tps": True,
            "shared_qwen_gpu_lease_applied_to_the_actual_complete_forward_requests": True,
        },
    }
    existing = _sealed_document(
        QWEN30_NATIVE_HTTP_CHAT_SMOKE,
        label=str(QWEN30_NATIVE_HTTP_CHAT_SMOKE),
    )
    if isinstance(existing, Mapping) and existing.get("binding") != payload["binding"]:
        _archive_native_http_receipt(
            QWEN30_NATIVE_HTTP_CHAT_SMOKE,
            QWEN30_NATIVE_HTTP_CHAT_SMOKE_HISTORY,
        )
    sealed = seal(payload)
    _atomic_json(QWEN30_NATIVE_HTTP_CHAT_SMOKE, sealed)
    _write_native_http_adapter_status(
        "NATIVE_HTTP_ADAPTER_PRODUCTION_CHAT_SSE_TRANSPORT_EARNED_HCLI_UNQUALIFIED",
        binding=adapter_binding,
        pid=active.get("pid"),
        endpoint_url=f"http://{QWEN30_NATIVE_HTTP_BIND}",
        transport_smoke_receipt_path=str(QWEN30_NATIVE_HTTP_TRANSPORT_SMOKE),
        transport_smoke_receipt_seal_sha256=transport_receipt.get("seal_sha256"),
        chat_sse_smoke_receipt_path=str(QWEN30_NATIVE_HTTP_CHAT_SMOKE),
        chat_sse_smoke_receipt_seal_sha256=sealed.get("seal_sha256"),
    )
    return sealed


def _native_http_adapter_binding_matches(record: Mapping[str, Any], binding: Mapping[str, Any]) -> bool:
    observed = record.get("binding")
    return isinstance(observed, Mapping) and dict(observed) == dict(binding)


def _write_native_http_adapter_status(
    phase: str,
    *,
    binding: Mapping[str, Any],
    **fields: Any,
) -> dict[str, Any]:
    document = {
        "schema": "hawking.ascension.qwen30_native_http_adapter.v1",
        "recorded_at": _utc_now(),
        "phase": phase,
        "bind": QWEN30_NATIVE_HTTP_BIND,
        "binding": dict(binding),
        **fields,
        "claim_boundary": {
            "actual_adapter_process_is_required_before_any_endpoint_is_named_ready": True,
            "adapter_readiness_is_not_hcli_capability_clean_tps_tg_or_tournament_qualification": True,
            "raw_bf16_source_is_not_a_runtime_participant": True,
        },
    }
    _atomic_json(QWEN30_NATIVE_HTTP_STATUS, document)
    return document


def _reconcile_paired_scalar_order_production_native_http_adapter(
    binding: Mapping[str, Any],
    *,
    exact_runtime_receipt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Reconcile only the explicitly deployed production no-parity adapter."""

    global _ACTIVE_NATIVE_HTTP_PROCESS
    runtime_binding = _native_http_adapter_runtime_binding(binding, exact_runtime_receipt)
    if not isinstance(runtime_binding, Mapping) or not runtime_binding.get("production_no_parity"):
        return _write_native_http_adapter_status(
            "NATIVE_HTTP_ADAPTER_PRODUCTION_RUNTIME_BINDING_INVALID_HCLI_CLOSED",
            binding=dict(binding),
            canonical_runtime_receipt_path=str(QWEN30_EXACT_FULL_TOKEN_RUNTIME_RECEIPT),
        )
    deployment = _paired_scalar_order_production_http_adapter_deployment(
        binding, exact_runtime_receipt
    )
    if not isinstance(deployment, Mapping):
        return _write_native_http_adapter_status(
            "NATIVE_HTTP_ADAPTER_PRODUCTION_DEPLOYMENT_REQUIRED_HCLI_CLOSED",
            binding=runtime_binding,
            deployment_path=str(QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_HTTP_ADAPTER_DEPLOYMENT),
            required_binary_source=str(QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_HTTP_SERVER_SOURCE),
        )
    deployment_binding = deployment.get("binding")
    assert isinstance(deployment_binding, Mapping)
    adapter_binding = {
        **dict(runtime_binding),
        "production_http_adapter_deployment_path": str(
            QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_HTTP_ADAPTER_DEPLOYMENT
        ),
        "production_http_adapter_deployment_seal_sha256": deployment.get("seal_sha256"),
        "server_binary_path": str(QWEN30_NATIVE_HTTP_SERVER),
        "server_binary_sha256": deployment_binding.get("active_server_binary_sha256"),
    }
    try:
        current_server_sha256 = _file_sha256(QWEN30_NATIVE_HTTP_SERVER)
    except Qwen30BootstrapError as exc:
        return _write_native_http_adapter_status(
            "NATIVE_HTTP_ADAPTER_PRODUCTION_BINARY_UNHASHABLE_HCLI_CLOSED",
            binding=adapter_binding,
            error=str(exc),
        )
    if current_server_sha256 != adapter_binding.get("server_binary_sha256"):
        return _write_native_http_adapter_status(
            "NATIVE_HTTP_ADAPTER_PRODUCTION_BINARY_BINDING_MISMATCH_HCLI_CLOSED",
            binding=adapter_binding,
            observed_server_binary_sha256=current_server_sha256,
        )

    active = _read_json(QWEN30_NATIVE_HTTP_ACTIVE)
    if isinstance(active, Mapping) and active.get("phase") == "RUNNING":
        pid = active.get("pid")
        if not _native_http_adapter_binding_matches(active, adapter_binding):
            # Terminate only an owned process using this exact local adapter
            # executable path.  A malformed/foreign record remains fail-closed
            # rather than granting this lane authority over an unknown PID.
            if (
                isinstance(pid, int)
                and _pid_is_alive(pid)
                and active.get("server_binary_path") == str(QWEN30_NATIVE_HTTP_SERVER)
            ):
                try:
                    os.kill(pid, signal.SIGTERM)
                except OSError as exc:
                    return _write_native_http_adapter_status(
                        "NATIVE_HTTP_ADAPTER_STALE_PROCESS_TERM_FAILED_HCLI_CLOSED",
                        binding=adapter_binding,
                        stale_active_record=dict(active),
                        error=str(exc),
                    )
                deadline = time.monotonic() + 10.0
                while _pid_is_alive(pid) and time.monotonic() < deadline:
                    time.sleep(0.1)
                if _pid_is_alive(pid):
                    return _write_native_http_adapter_status(
                        "NATIVE_HTTP_ADAPTER_STALE_PROCESS_DID_NOT_EXIT_HCLI_CLOSED",
                        binding=adapter_binding,
                        stale_active_record=dict(active),
                    )
                terminal = {
                    **dict(active),
                    "phase": "TERMINAL",
                    "finished_at": _utc_now(),
                    "outcome": "TERMINATED_STALE_OR_WRONG_RUNTIME_HTTP_BINDING",
                }
                _atomic_json(QWEN30_NATIVE_HTTP_ACTIVE, terminal)
                _atomic_json(QWEN30_NATIVE_HTTP_LAST_PROCESS, terminal)
                active = terminal
            else:
                return _write_native_http_adapter_status(
                    "NATIVE_HTTP_ADAPTER_STALE_OR_UNOWNED_ACTIVE_RECORD_HCLI_CLOSED",
                    binding=adapter_binding,
                    stale_active_record=dict(active),
                )
        elif isinstance(pid, int) and _pid_is_alive(pid):
            health = _native_http_adapter_health()
            context = _native_http_adapter_context() if health is not None else None
            transport_smoke = (
                _write_native_http_transport_smoke(
                    adapter_binding,
                    health=health,
                    context=context,
                    active=active,
                )
                if health is not None and context is not None
                else None
            )
            return _write_native_http_adapter_status(
                "NATIVE_HTTP_ADAPTER_SERVING_UNQUALIFIED"
                if transport_smoke is not None
                else "NATIVE_HTTP_ADAPTER_LOADING_OR_METADATA_UNVERIFIED_HCLI_CLOSED",
                binding=adapter_binding,
                pid=pid,
                process_state="OWNED"
                if _ACTIVE_NATIVE_HTTP_PROCESS and _ACTIVE_NATIVE_HTTP_PROCESS.pid == pid
                else "INHERITED",
                server_binary_path=str(QWEN30_NATIVE_HTTP_SERVER),
                server_binary_sha256=current_server_sha256,
                active_record_path=str(QWEN30_NATIVE_HTTP_ACTIVE),
                health=health,
                context=context,
                transport_smoke_receipt_path=(
                    str(QWEN30_NATIVE_HTTP_TRANSPORT_SMOKE)
                    if transport_smoke is not None
                    else None
                ),
                transport_smoke_receipt_seal_sha256=(
                    transport_smoke.get("seal_sha256")
                    if transport_smoke is not None
                    else None
                ),
                endpoint_url=(
                    f"http://{QWEN30_NATIVE_HTTP_BIND}"
                    if transport_smoke is not None
                    else None
                ),
            )
        else:
            terminal = {
                **dict(active),
                "phase": "TERMINAL",
                "finished_at": _utc_now(),
                "outcome": "NATIVE_HTTP_ADAPTER_EXITED_OR_UNREACHABLE",
            }
            _atomic_json(QWEN30_NATIVE_HTTP_ACTIVE, terminal)
            _atomic_json(QWEN30_NATIVE_HTTP_LAST_PROCESS, terminal)

    command = _native_http_adapter_command(binding)
    stdout_path = RUNTIME_ROOT / "QWEN30_NATIVE_HTTP_ADAPTER.stdout.log"
    stderr_path = RUNTIME_ROOT / "QWEN30_NATIVE_HTTP_ADAPTER.stderr.log"
    try:
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
    except OSError as exc:
        return _write_native_http_adapter_status(
            "NATIVE_HTTP_ADAPTER_PRODUCTION_LAUNCH_FAILED_HCLI_CLOSED",
            binding=adapter_binding,
            error=str(exc),
            command=command,
        )
    _ACTIVE_NATIVE_HTTP_PROCESS = process
    record = {
        "schema": "hawking.ascension.qwen30_native_http_adapter_process.v1",
        "phase": "RUNNING",
        "pid": process.pid,
        "started_at": _utc_now(),
        "binding": adapter_binding,
        "server_binary_path": str(QWEN30_NATIVE_HTTP_SERVER),
        "server_binary_sha256": current_server_sha256,
        "command": command,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "claim_boundary": {
            "spawn_is_not_endpoint_readiness_hcli_or_tps_pass": True,
            "direct_packed_native_metal_server_only": True,
            "production_no_parity_kernel_is_explicit": True,
        },
    }
    _atomic_json(QWEN30_NATIVE_HTTP_ACTIVE, record)
    return _write_native_http_adapter_status(
        "NATIVE_HTTP_ADAPTER_PRODUCTION_LAUNCHED_AWAITING_REAL_HEALTH",
        binding=adapter_binding,
        pid=process.pid,
        server_binary_path=str(QWEN30_NATIVE_HTTP_SERVER),
        server_binary_sha256=current_server_sha256,
        active_record_path=str(QWEN30_NATIVE_HTTP_ACTIVE),
    )


def _reconcile_native_http_adapter(
    binding: Mapping[str, Any],
    *,
    exact_runtime_receipt: Mapping[str, Any] | None,
    simdgroup_candidate_matches_control: bool,
    simdgroup_template_decision: Mapping[str, Any] | None,
    gateup_fused_decision: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Start/verify the actual local HCLI adapter after kernel decisions settle.

    The endpoint is intentionally delayed until the current direct runtime is
    sealed, the SIMD candidate has its BOS control check, and both source-
    template kernel decisions are current and settled. A rejected candidate is
    negative science: it is never selected or served. Once both decisions are
    sealed, the endpoint may resume only with the retained scalar control; it
    cannot start during an unresolved candidate trial.
    """

    if _paired_scalar_order_production_deployment() is not None:
        return _reconcile_paired_scalar_order_production_native_http_adapter(
            binding,
            exact_runtime_receipt=exact_runtime_receipt,
        )

    global _ACTIVE_NATIVE_HTTP_PROCESS
    simdgroup_template_passed = bool(
        isinstance(simdgroup_template_decision, Mapping)
        and simdgroup_template_decision.get("status")
        == "EARNED_QWEN30_PACKED_BINARY_SIMDGROUP_TEMPLATE_PARITY"
    )
    gateup_fused_template_passed = bool(
        isinstance(gateup_fused_decision, Mapping)
        and gateup_fused_decision.get("status")
        == "EARNED_QWEN30_DIRECT_PACKED_GATE_UP_SWIGLU_FUSED_ALL_LAYER_TEMPLATE_PARITY"
    )
    settled_simdgroup_template_decision = bool(
        isinstance(simdgroup_template_decision, Mapping)
        and simdgroup_template_decision.get("status")
        in {
            "EARNED_QWEN30_PACKED_BINARY_SIMDGROUP_TEMPLATE_PARITY",
            "REJECTED_QWEN30_PACKED_BINARY_SIMDGROUP_TEMPLATE_PARITY",
        }
    )
    settled_gateup_fused_template_decision = bool(
        isinstance(gateup_fused_decision, Mapping)
        and gateup_fused_decision.get("status")
        in {
            "EARNED_QWEN30_DIRECT_PACKED_GATE_UP_SWIGLU_FUSED_ALL_LAYER_TEMPLATE_PARITY",
            "REJECTED_QWEN30_DIRECT_PACKED_GATE_UP_SWIGLU_FUSED_ALL_LAYER_TEMPLATE_PARITY",
        }
    )
    eligible = (
        exact_runtime_receipt is not None
        and simdgroup_candidate_matches_control
        and settled_simdgroup_template_decision
        and settled_gateup_fused_template_decision
    )
    if not eligible:
        # A prior watcher revision could have launched the scalar loopback
        # adapter after a BOS-only candidate check.  Do not leave that
        # listener alive while the source-template kernel decisions remain
        # unresolved.  This is a bounded, binding-checked shutdown rather
        # than treating the server as an ordinary crash: once both decisions
        # are sealed, the same immutable server binary may be launched again.
        predecision_shutdown: dict[str, Any] | None = None
        active = _read_json(QWEN30_NATIVE_HTTP_ACTIVE)
        if (
            isinstance(active, Mapping)
            and active.get("phase") == "RUNNING"
            and _native_http_adapter_binding_matches(active, binding)
        ):
            pid = active.get("pid")
            shutdown_action = "SERVER_ALREADY_NOT_ALIVE"
            if isinstance(pid, int) and _pid_is_alive(pid):
                try:
                    os.kill(pid, signal.SIGTERM)
                except OSError as exc:
                    shutdown_action = "TERM_FAILED"
                    predecision_shutdown = {
                        "pid": pid,
                        "action": shutdown_action,
                        "error": str(exc),
                    }
                else:
                    deadline = time.monotonic() + 10.0
                    while _pid_is_alive(pid) and time.monotonic() < deadline:
                        time.sleep(0.1)
                    shutdown_action = (
                        "TERM_CONFIRMED"
                        if not _pid_is_alive(pid)
                        else "TERM_SENT_PROCESS_STILL_ALIVE"
                    )
            if predecision_shutdown is None:
                predecision_shutdown = {"pid": pid, "action": shutdown_action}
            terminal = {
                **dict(active),
                "phase": "TERMINAL",
                "finished_at": _utc_now(),
                "outcome": "TERMINATED_PREDECISION_KERNEL_DECISION_GATING",
                "shutdown": predecision_shutdown,
            }
            _atomic_json(QWEN30_NATIVE_HTTP_ACTIVE, terminal)
            _atomic_json(QWEN30_NATIVE_HTTP_LAST_PROCESS, terminal)
        return _write_native_http_adapter_status(
            "WAITING_FOR_CURRENT_EXACT_RUNTIME_AND_SETTLED_KERNEL_DECISIONS",
            binding=binding,
            exact_runtime_receipt_present=exact_runtime_receipt is not None,
            simdgroup_candidate_matches_scalar_control=simdgroup_candidate_matches_control,
            simdgroup_template_decision_status=(
                simdgroup_template_decision.get("status")
                if isinstance(simdgroup_template_decision, Mapping)
                else None
            ),
            gateup_fused_decision_status=(
                gateup_fused_decision.get("status")
                if isinstance(gateup_fused_decision, Mapping)
                else None
            ),
            simdgroup_template_parity_passed=simdgroup_template_passed,
            gateup_fused_template_parity_passed=gateup_fused_template_passed,
            simdgroup_template_decision_settled=settled_simdgroup_template_decision,
            gateup_fused_template_decision_settled=settled_gateup_fused_template_decision,
            predecision_adapter_shutdown=predecision_shutdown,
        )
    if not QWEN30_NATIVE_HTTP_SERVER.is_file():
        return _write_native_http_adapter_status(
            "NATIVE_HTTP_ADAPTER_BINARY_BUILD_PENDING",
            binding=binding,
            server_binary_path=str(QWEN30_NATIVE_HTTP_SERVER),
        )
    try:
        server_binary_sha256 = _file_sha256(QWEN30_NATIVE_HTTP_SERVER)
    except Qwen30BootstrapError as exc:
        return _write_native_http_adapter_status(
            "NATIVE_HTTP_ADAPTER_BINARY_UNHASHABLE",
            binding=binding,
            error=str(exc),
        )
    active = _read_json(QWEN30_NATIVE_HTTP_ACTIVE)
    if isinstance(active, Mapping) and active.get("phase") == "RUNNING" and _native_http_adapter_binding_matches(active, binding):
        pid = active.get("pid")
        if isinstance(pid, int) and _pid_is_alive(pid):
            health = _native_http_adapter_health()
            context = _native_http_adapter_context() if health is not None else None
            transport_smoke = (
                _write_native_http_transport_smoke(
                    binding,
                    health=health,
                    context=context,
                    active=active,
                )
                if health is not None and context is not None
                else None
            )
            return _write_native_http_adapter_status(
                "NATIVE_HTTP_ADAPTER_SERVING_UNQUALIFIED"
                if health is not None
                else "NATIVE_HTTP_ADAPTER_LOADING_DIRECT_PACKED_MODEL",
                binding=binding,
                pid=pid,
                process_state="OWNED" if _ACTIVE_NATIVE_HTTP_PROCESS and _ACTIVE_NATIVE_HTTP_PROCESS.pid == pid else "INHERITED",
                server_binary_path=str(QWEN30_NATIVE_HTTP_SERVER),
                server_binary_sha256=server_binary_sha256,
                active_record_path=str(QWEN30_NATIVE_HTTP_ACTIVE),
                health=health,
                context=context,
                transport_smoke_receipt_path=(
                    str(QWEN30_NATIVE_HTTP_TRANSPORT_SMOKE) if transport_smoke is not None else None
                ),
                transport_smoke_receipt_seal_sha256=(
                    transport_smoke.get("seal_sha256") if transport_smoke is not None else None
                ),
                endpoint_url=(f"http://{QWEN30_NATIVE_HTTP_BIND}" if health is not None else None),
            )
        terminal = {
            **dict(active),
            "phase": "TERMINAL",
            "finished_at": _utc_now(),
            "outcome": "NATIVE_HTTP_ADAPTER_EXITED_OR_UNREACHABLE",
        }
        _atomic_json(QWEN30_NATIVE_HTTP_ACTIVE, terminal)
        _atomic_json(QWEN30_NATIVE_HTTP_LAST_PROCESS, terminal)
        active = terminal
    prior = _read_json(QWEN30_NATIVE_HTTP_LAST_PROCESS)
    if isinstance(prior, Mapping) and (
        prior.get("outcome") == "NATIVE_HTTP_ADAPTER_EXITED_OR_UNREACHABLE"
        and _native_http_adapter_binding_matches(prior, binding)
        and prior.get("server_binary_sha256") == server_binary_sha256
    ):
        return _write_native_http_adapter_status(
            "NATIVE_HTTP_ADAPTER_TERMINAL_AWAITING_MATERIAL_SERVER_OR_BINDING_CHANGE",
            binding=binding,
            prior=prior,
        )
    command = _native_http_adapter_command(binding)
    stdout_path = RUNTIME_ROOT / "QWEN30_NATIVE_HTTP_ADAPTER.stdout.log"
    stderr_path = RUNTIME_ROOT / "QWEN30_NATIVE_HTTP_ADAPTER.stderr.log"
    try:
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
    except OSError as exc:
        return _write_native_http_adapter_status(
            "NATIVE_HTTP_ADAPTER_LAUNCH_FAILED",
            binding=binding,
            error=str(exc),
            command=command,
        )
    _ACTIVE_NATIVE_HTTP_PROCESS = process
    record = {
        "schema": "hawking.ascension.qwen30_native_http_adapter_process.v1",
        "phase": "RUNNING",
        "pid": process.pid,
        "started_at": _utc_now(),
        "binding": dict(binding),
        "server_binary_path": str(QWEN30_NATIVE_HTTP_SERVER),
        "server_binary_sha256": server_binary_sha256,
        "command": command,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "claim_boundary": {
            "spawn_is_not_endpoint_readiness_or_hcli_pass": True,
            "direct_packed_native_metal_server_only": True,
        },
    }
    _atomic_json(QWEN30_NATIVE_HTTP_ACTIVE, record)
    return _write_native_http_adapter_status(
        "NATIVE_HTTP_ADAPTER_LAUNCHED_AWAITING_REAL_HEALTH",
        binding=binding,
        pid=process.pid,
        server_binary_path=str(QWEN30_NATIVE_HTTP_SERVER),
        server_binary_sha256=server_binary_sha256,
        active_record_path=str(QWEN30_NATIVE_HTTP_ACTIVE),
    )


def _write_hcli_handoff(
    binding: Mapping[str, Any],
    full_token: Mapping[str, Any] | None,
    native_http_adapter: Mapping[str, Any] | None,
) -> None:
    full_token_earned = full_token is not None
    runtime_receipt = _sealed_document(
        QWEN30_EXACT_FULL_TOKEN_RUNTIME_RECEIPT,
        label=str(QWEN30_EXACT_FULL_TOKEN_RUNTIME_RECEIPT),
    )
    runtime_binding = _native_http_adapter_runtime_binding(binding, runtime_receipt)
    deployment = _paired_scalar_order_production_http_adapter_deployment(
        binding, runtime_receipt
    )
    adapter_binding: dict[str, Any] | None = None
    if isinstance(runtime_binding, Mapping) and isinstance(deployment, Mapping):
        deployment_binding = deployment.get("binding")
        if isinstance(deployment_binding, Mapping):
            adapter_binding = {
                **dict(runtime_binding),
                "production_http_adapter_deployment_path": str(
                    QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_HTTP_ADAPTER_DEPLOYMENT
                ),
                "production_http_adapter_deployment_seal_sha256": deployment.get("seal_sha256"),
                "server_binary_path": str(QWEN30_NATIVE_HTTP_SERVER),
                "server_binary_sha256": deployment_binding.get("active_server_binary_sha256"),
            }
    adapter_serving = (
        isinstance(native_http_adapter, Mapping)
        and native_http_adapter.get("phase") == "NATIVE_HTTP_ADAPTER_SERVING_UNQUALIFIED"
        and isinstance(adapter_binding, Mapping)
        and native_http_adapter.get("binding") == adapter_binding
    )
    chat_smoke = _sealed_document(
        QWEN30_NATIVE_HTTP_CHAT_SMOKE,
        label=str(QWEN30_NATIVE_HTTP_CHAT_SMOKE),
    )
    chat_smoke_earned = isinstance(chat_smoke, Mapping) and (
        chat_smoke.get("schema") == "hawking.ascension.qwen30_direct_packed_hcli_transport_smoke.v1"
        and chat_smoke.get("status")
        == "EARNED_DIRECT_PACKED_NATIVE_CHAT_SSE_TRANSPORT_HCLI_UNQUALIFIED"
        and isinstance(chat_smoke.get("binding"), Mapping)
        and isinstance(adapter_binding, Mapping)
        and all(
            chat_smoke["binding"].get(field) == adapter_binding.get(field)
            for field in adapter_binding
        )
    )
    _atomic_json(
        QWEN30_HCLI_HANDOFF,
        {
            "schema": "hawking.ascension.qwen30_native_runtime_hcli_handoff.v1",
            "recorded_at": _utc_now(),
            "status": (
                "DIRECT_NATIVE_HTTP_CHAT_SSE_TRANSPORT_EARNED_HCLI_UNQUALIFIED"
                if chat_smoke_earned and adapter_serving
                else "DIRECT_NATIVE_HTTP_ADAPTER_READY_HCLI_EVALUATION_NOT_RUN"
                if adapter_serving
                else "DIRECT_NATIVE_FULL_TOKEN_EARNED_HCLI_HTTP_ADAPTER_NOT_IMPLEMENTED"
                if full_token_earned
                else "WAITING_FOR_DIRECT_NATIVE_FULL_TOKEN_BEFORE_HCLI_HTTP_ADAPTER"
            ),
            "binding": dict(adapter_binding) if isinstance(adapter_binding, Mapping) else dict(binding),
            "direct_runtime_evidence": str(QWEN30_NATIVE_FULL_TOKEN) if full_token_earned else None,
            "native_http_adapter_status_path": str(QWEN30_NATIVE_HTTP_STATUS),
            "native_http_adapter_endpoint_url": (
                native_http_adapter.get("endpoint_url") if adapter_serving else None
            ),
            "unqualified_chat_sse_transport_receipt_path": str(QWEN30_NATIVE_HTTP_CHAT_SMOKE),
            "unqualified_chat_sse_transport_receipt_seal_sha256": (
                chat_smoke.get("seal_sha256") if chat_smoke_earned else None
            ),
            "required_endpoint_contract": {
                "local_only": True,
                "health": "GET /healthz",
                "context": "GET /v1/hawking/context",
                "generation": "POST /v1/hawking/generate with native SSE token stream",
                "no_bf16_shadow_or_mps_production_provider": True,
                "no_endpoint_url_is_claimed_or_configured_yet": not adapter_serving,
            },
            "hcli_client_contract_after_adapter": [
                "hcli run --prompt TEXT --model-url http://127.0.0.1:PORT --max-output-tokens N --json",
                "hcli bench --prompt TEXT --model-url http://127.0.0.1:PORT --warmup N --runs N --max-output-tokens N",
            ],
            "remaining_before_hcli_pass": [
                "source-template-bound prompt quality evaluation (coherence remains unscored)",
                "HCLI client chat/session/restart/tool receipts on that actual endpoint",
                "durable Context/KV session semantics and Agent-OS operation probes",
            ],
            "claim_boundary": {
                "this_is_a_generated_handoff_contract_not_an_hcli_pass": True,
                "not_clean_tps_tg_capability_or_tournament_qualification": True,
            },
        },
    )


def run_runtime_cycle() -> None:
    status_path = RUNTIME_ROOT / "QWEN30_COMPLETE_RUNTIME_STATUS.json"
    source_audit = _read_json(QWEN30_ROOT / "QWEN30_SOURCE_BODY_AUDIT_CANDIDATE.json")
    frontier = _read_json(QWEN30_ROOT / "QWEN30_GRAVITY_FRONTIER.json")
    weight_map = _weight_index()
    source = _source_summary(weight_map)
    complete_candidate = _complete_binary_candidate()
    try:
        binding = _native_runtime_binding()
    except Qwen30BootstrapError as exc:
        _status(
            status_path,
            lane="B_QWEN30_COMPLETE_RUNTIME",
            phase="WAITING_FOR_PROTECTED_COMPLETE_ARTIFACT_ADMISSION_BINDING",
            source=source,
            physical_complete_binary_candidate=complete_candidate,
            native_runtime_executable=str(QWEN30_NATIVE_RUNTIME_EXECUTABLE),
            admission_error=str(exc),
        )
        return

    production_deployment = _paired_scalar_order_production_deployment()
    if _paired_scalar_order_production_requested() and production_deployment is None:
        _status(
            status_path,
            lane="B_QWEN30_COMPLETE_RUNTIME",
            phase="QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_DEPLOYMENT_INVALID_HCLI_CLOSED",
            deployment_path=str(QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_DEPLOYMENT),
            current_runtime_executable_sha256=(
                _file_sha256(QWEN30_NATIVE_RUNTIME_EXECUTABLE)
                if QWEN30_NATIVE_RUNTIME_EXECUTABLE.is_file()
                else None
            ),
            hcli_server_state="CLOSED",
        )
        return

    preflight = _native_result_matches("preflight", binding)
    full_token = _native_result_matches("full-token", binding)
    prompt_a = _native_result_matches("prompt-a", binding)
    prompt_b = _native_result_matches("prompt-b", binding)
    profile_token = _native_result_matches("profile-token", binding)
    partial_profile_negative = (
        _preserve_incomplete_profile_negative(binding) if profile_token is None else None
    )
    simdgroup_component_parity = _simdgroup_component_parity()
    simdgroup_candidate_token = _native_result_matches("simdgroup-candidate-token", binding)
    simdgroup_candidate_matches_control = _simdgroup_candidate_matches_control(
        full_token, simdgroup_candidate_token
    )
    exact_runtime_receipt = _build_exact_full_token_runtime_receipt(
        binding,
        preflight=preflight,
        full_token=full_token,
        prompt_a=prompt_a,
        prompt_b=prompt_b,
        profile_token=profile_token,
    )
    simdgroup_template_decision = _current_simdgroup_template_parity_decision(binding)
    gateup_fused_decision = _current_gateup_fused_template_decision(binding)
    paired_scalar_order_cpu_gate = _current_paired_scalar_order_cpu_gate(binding)
    paired_scalar_order_compile_refusal = _paired_scalar_order_compile_refusal(binding)
    native_http_adapter = _reconcile_native_http_adapter(
        binding,
        exact_runtime_receipt=exact_runtime_receipt,
        simdgroup_candidate_matches_control=simdgroup_candidate_matches_control,
        simdgroup_template_decision=simdgroup_template_decision,
        gateup_fused_decision=gateup_fused_decision,
    )
    report = {
        "schema": "hawking.ascension.qwen30_complete_gravity_assembly.v1",
        "status": "DIRECT_PACKED_NATIVE_RUNTIME_EXECUTABLE_BOUND",
        "recorded_at": _utc_now(),
        "source": source,
        "source_body_audit_seal_sha256": source_audit.get("seal_sha256") if source_audit else None,
        "gravity_frontier_seal_sha256": frontier.get("seal_sha256") if frontier else None,
        "physical_complete_binary_candidate": complete_candidate,
        "native_complete_binary_reader": _complete_binary_reader(),
        "native_runtime": {
            "executable_path": str(QWEN30_NATIVE_RUNTIME_EXECUTABLE),
            "executable_present": QWEN30_NATIVE_RUNTIME_EXECUTABLE.is_file(),
            "binding": binding,
            "preflight_result_path": str(QWEN30_NATIVE_PREFLIGHT),
            "full_token_result_path": str(QWEN30_NATIVE_FULL_TOKEN),
            "prompt_a_result_path": str(QWEN30_NATIVE_PROMPT_A),
            "prompt_b_result_path": str(QWEN30_NATIVE_PROMPT_B),
            "kernel_profile_result_path": str(QWEN30_NATIVE_PROFILE_TOKEN),
            "partial_profile_negative_path": str(QWEN30_NATIVE_PROFILE_PARTIAL_NEGATIVE),
            "simdgroup_component_parity_path": str(QWEN30_SIMDGROUP_COMPONENT_PARITY),
            "simdgroup_candidate_full_token_path": str(QWEN30_NATIVE_SIMDGROUP_CANDIDATE_TOKEN),
            "physical_exact_full_token_runtime_receipt_path": str(
                QWEN30_EXACT_FULL_TOKEN_RUNTIME_RECEIPT
            ),
            "preflight_earned": preflight is not None,
            "full_token_earned": full_token is not None,
            "prompt_a_executed": prompt_a is not None,
            "prompt_b_executed": prompt_b is not None,
            "diagnostic_gpu_profile_executed": profile_token is not None,
            "incomplete_gpu_profile_preserved_and_reopened": partial_profile_negative is not None,
            "simdgroup_component_parity_earned": simdgroup_component_parity is not None,
            "simdgroup_candidate_token_executed": simdgroup_candidate_token is not None,
            "simdgroup_candidate_matches_scalar_control_argmax": simdgroup_candidate_matches_control,
            "simdgroup_template_parity_decision_path": str(QWEN30_SIMDGROUP_TEMPLATE_PARITY),
            "simdgroup_template_parity_decision_status": (
                simdgroup_template_decision.get("status")
                if isinstance(simdgroup_template_decision, Mapping)
                else None
            ),
            "gateup_fused_template_parity_decision_path": str(
                QWEN30_GATEUP_FUSED_TEMPLATE_PARITY
            ),
            "gateup_fused_template_parity_decision_status": (
                gateup_fused_decision.get("status")
                if isinstance(gateup_fused_decision, Mapping)
                else None
            ),
            "physical_exact_full_token_runtime_gate_passed": exact_runtime_receipt is not None,
            "native_http_adapter_status_path": str(QWEN30_NATIVE_HTTP_STATUS),
            "native_http_adapter_phase": native_http_adapter.get("phase"),
        },
        "full_artifact_contract": {
            "all_tensors_required": True,
            "complete_physical_bpw_required_at_most": 1.5,
            "forbids_component_only_bpw": True,
            "requires_native_packed_metal_execution": True,
            "requires_complete_admission_verified_immutable_payload_catalog": True,
            "forbids_per_token_payload_sha256_rescan": True,
            "requires_full_artifact_revalidation_on_process_restart": True,
            "requires_exact_48_layer_token": True,
            "requires_hcli_then_tg3": True,
        },
        "next_automatic_native_stage": (
            "preflight" if preflight is None else "full-token" if full_token is None
            else "prompt-a" if prompt_a is None else "prompt-b" if prompt_b is None
            else "profile-token" if profile_token is None
            else "simdgroup-component-parity" if simdgroup_component_parity is None
            else "simdgroup-candidate-token" if simdgroup_candidate_token is None
            else "simdgroup-template-parity"
            if simdgroup_candidate_matches_control and simdgroup_template_decision is None
            else "gateup-fused-template-parity"
            if simdgroup_candidate_matches_control and gateup_fused_decision is None
            else "hcli-adapter" if simdgroup_candidate_matches_control else "candidate-rejected"
        ),
        "claim_boundary": {
            "direct_runtime_stage_receipts_are_not_capability_hcli_tps_tg_or_tournament_receipts": True,
            "does_not_load_raw_model_body_into_runtime": True,
        },
    }
    _atomic_json(RUNTIME_ROOT / "QWEN30_COMPLETE_GRAVITY_ASSEMBLY_CANDIDATE.json", report)
    _write_hcli_handoff(binding, full_token, native_http_adapter)

    active = _poll_native_stage()
    if active is not None and active.get("state") in {"RUNNING", "INHERITED_RUNNING"}:
        record = active.get("record") if isinstance(active.get("record"), Mapping) else {}
        _status(
            status_path,
            lane="B_QWEN30_COMPLETE_RUNTIME",
            phase="NATIVE_DIRECT_PACKED_STAGE_RUNNING",
            stage=record.get("stage"),
            native_process_pid=record.get("pid"),
            started_at=record.get("started_at"),
            active_process_state=active.get("state"),
            source=source,
            physical_complete_binary_candidate=complete_candidate,
            current_artifact=str(RUNTIME_ROOT / "QWEN30_COMPLETE_GRAVITY_ASSEMBLY_CANDIDATE.json"),
            hcli_handoff_path=str(QWEN30_HCLI_HANDOFF),
        )
        return
    if active is not None:
        process = active.get("process") if isinstance(active.get("process"), Mapping) else active
        if str(process.get("outcome") if isinstance(process, Mapping) else "").startswith("EARNED_STAGE_RESULT"):
            # Re-read only an exact schema/binding checked receipt and advance
            # immediately. A completed preflight must not sit idle for another
            # watcher interval before the bounded full-token attempt begins.
            preflight = _native_result_matches("preflight", binding)
            full_token = _native_result_matches("full-token", binding)
            prompt_a = _native_result_matches("prompt-a", binding)
            prompt_b = _native_result_matches("prompt-b", binding)
            profile_token = _native_result_matches("profile-token", binding)
            simdgroup_component_parity = _simdgroup_component_parity()
            simdgroup_candidate_token = _native_result_matches("simdgroup-candidate-token", binding)
            simdgroup_candidate_matches_control = _simdgroup_candidate_matches_control(
                full_token, simdgroup_candidate_token
            )
            exact_runtime_receipt = _build_exact_full_token_runtime_receipt(
                binding,
                preflight=preflight,
                full_token=full_token,
                prompt_a=prompt_a,
                prompt_b=prompt_b,
                profile_token=profile_token,
            )
            simdgroup_template_decision = _current_simdgroup_template_parity_decision(binding)
            gateup_fused_decision = _current_gateup_fused_template_decision(binding)
            native_http_adapter = _reconcile_native_http_adapter(
                binding,
                exact_runtime_receipt=exact_runtime_receipt,
                simdgroup_candidate_matches_control=simdgroup_candidate_matches_control,
                simdgroup_template_decision=simdgroup_template_decision,
                gateup_fused_decision=gateup_fused_decision,
            )
            _write_hcli_handoff(binding, full_token, native_http_adapter)
        else:
            _status(
                status_path,
                lane="B_QWEN30_COMPLETE_RUNTIME",
                phase="NATIVE_DIRECT_PACKED_STAGE_TERMINAL_RECORDED",
                native_process=process,
                source=source,
                physical_complete_binary_candidate=complete_candidate,
                current_artifact=str(RUNTIME_ROOT / "QWEN30_COMPLETE_GRAVITY_ASSEMBLY_CANDIDATE.json"),
                hcli_handoff_path=str(QWEN30_HCLI_HANDOFF),
            )
            return

    if not QWEN30_NATIVE_RUNTIME_EXECUTABLE.is_file():
        _status(
            status_path,
            lane="B_QWEN30_COMPLETE_RUNTIME",
            phase="NATIVE_RUNTIME_EXECUTABLE_BUILD_PENDING",
            executable_path=str(QWEN30_NATIVE_RUNTIME_EXECUTABLE),
            source=source,
            physical_complete_binary_candidate=complete_candidate,
            hcli_handoff_path=str(QWEN30_HCLI_HANDOFF),
        )
        return

    if production_deployment is not None and profile_token is not None:
        adapter_phase = (
            native_http_adapter.get("phase")
            if isinstance(native_http_adapter, Mapping)
            else None
        )
        adapter_serving = adapter_phase == "NATIVE_HTTP_ADAPTER_SERVING_UNQUALIFIED"
        adapter_launching = adapter_phase in {
            "NATIVE_HTTP_ADAPTER_PRODUCTION_LAUNCHED_AWAITING_REAL_HEALTH",
            "NATIVE_HTTP_ADAPTER_LOADING_OR_METADATA_UNVERIFIED_HCLI_CLOSED",
        }
        _status(
            status_path,
            lane="B_QWEN30_COMPLETE_RUNTIME",
            phase=(
                "QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_RUNTIME_REQUALIFIED_HTTP_TRANSPORT_SERVING_HCLI_UNQUALIFIED"
                if adapter_serving
                else "QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_HTTP_ADAPTER_LOADING_HCLI_CLOSED"
                if adapter_launching
                else "QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_RUNTIME_REQUALIFIED_HCLI_CLOSED"
            ),
            deployment_path=str(QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_DEPLOYMENT),
            deployment_seal_sha256=production_deployment.get("seal_sha256"),
            production_gate_up_swiglu_kernel=QWEN30_PAIRED_SCALAR_ORDER_PRODUCTION_KERNEL,
            runtime_receipt_path=str(QWEN30_EXACT_FULL_TOKEN_RUNTIME_RECEIPT),
            runtime_receipt_seal_sha256=(
                exact_runtime_receipt.get("seal_sha256")
                if isinstance(exact_runtime_receipt, Mapping)
                else None
            ),
            completed_stages=["preflight", "full-token", "prompt-a", "prompt-b", "profile-token"],
            native_http_adapter_phase=adapter_phase,
            next_required_before_adapter=(
                ["actual HCLI/coherence and clean performance gates"]
                if adapter_serving
                else [
                    "review fresh host-stage profile",
                    "explicit adapter authorization and rebind",
                    "actual HCLI/coherence and clean performance gates",
                ]
            ),
            hcli_server_state=("SERVING_UNQUALIFIED" if adapter_serving else "CLOSED"),
            source=source,
            physical_complete_binary_candidate=complete_candidate,
            hcli_handoff_path=str(QWEN30_HCLI_HANDOFF),
        )
        return

    if (
        profile_token is not None
        and paired_scalar_order_compile_refusal is not None
        and not QWEN30_GATEUP_PAIRED_SCALAR_ORDER_PRAGMA_SUCCESSOR_TEMPLATE_PARITY.exists()
    ):
        _status(
            status_path,
            lane="B_QWEN30_COMPLETE_RUNTIME",
            phase="QWEN30_PAIRED_SCALAR_ORDER_COMPILE_REFUSAL_AWAITING_SOURCE_REVISION_BOUND_SUCCESSOR",
            initial_compile_refusal_path=str(
                QWEN30_GATEUP_PAIRED_SCALAR_ORDER_TEMPLATE_PARITY
            ),
            initial_compile_refusal_seal_sha256=paired_scalar_order_compile_refusal.get(
                "seal_sha256"
            ),
            initial_compile_refusal_reason=(
                "unsupported precise float qualifier; no device/template result exists"
            ),
            successor_template_parity_path=str(
                QWEN30_GATEUP_PAIRED_SCALAR_ORDER_PRAGMA_SUCCESSOR_TEMPLATE_PARITY
            ),
            current_scalar_selection="direct_packed_scalar_control",
            hcli_server_state="CLOSED",
            source=source,
            physical_complete_binary_candidate=complete_candidate,
            hcli_handoff_path=str(QWEN30_HCLI_HANDOFF),
        )
        return
    if (
        profile_token is not None
        and paired_scalar_order_cpu_gate is not None
        and not QWEN30_GATEUP_PAIRED_SCALAR_ORDER_TEMPLATE_PARITY.exists()
    ):
        _status(
            status_path,
            lane="B_QWEN30_COMPLETE_RUNTIME",
            phase="WAITING_FOR_QWEN80_DIRECT_STAGE_RELEASE",
            paired_scalar_order_cpu_gate_path=str(
                QWEN30_GATEUP_PAIRED_SCALAR_ORDER_CPU_PARITY
            ),
            paired_scalar_order_cpu_gate_seal_sha256=paired_scalar_order_cpu_gate.get(
                "seal_sha256"
            ),
            paired_scalar_order_cpu_gate_outcome=paired_scalar_order_cpu_gate.get(
                "outcome"
            ),
            next_guarded_gate="CURRENT_BINDING_DEVICE_AND_SOURCE_TEMPLATE_A_B_PARITY",
            current_scalar_selection="direct_packed_scalar_control",
            hcli_server_state="CLOSED",
            source=source,
            physical_complete_binary_candidate=complete_candidate,
            hcli_handoff_path=str(QWEN30_HCLI_HANDOFF),
        )
        return
    if profile_token is not None and simdgroup_component_parity is None:
        _status(
            status_path,
            lane="B_QWEN30_COMPLETE_RUNTIME",
            phase="WAITING_FOR_QWEN30_SIMDGROUP_COMPONENT_PARITY_BEFORE_CANDIDATE",
            control_profile_path=str(QWEN30_NATIVE_PROFILE_TOKEN),
            required_component_parity_path=str(QWEN30_SIMDGROUP_COMPONENT_PARITY),
            source=source,
            physical_complete_binary_candidate=complete_candidate,
            hcli_handoff_path=str(QWEN30_HCLI_HANDOFF),
        )
        return
    if simdgroup_candidate_token is not None and not simdgroup_candidate_matches_control:
        _status(
            status_path,
            lane="B_QWEN30_COMPLETE_RUNTIME",
            phase="QWEN30_SIMDGROUP_CANDIDATE_REJECTED_ARGMAX_DIFFERS_FROM_SCALAR_CONTROL",
            scalar_control_token=_direct_sampled_token(full_token) if full_token else None,
            candidate_token=_direct_sampled_token(simdgroup_candidate_token),
            candidate_result_path=str(QWEN30_NATIVE_SIMDGROUP_CANDIDATE_TOKEN),
            source=source,
            physical_complete_binary_candidate=complete_candidate,
            hcli_handoff_path=str(QWEN30_HCLI_HANDOFF),
        )
        return
    stage = (
        "preflight" if preflight is None else "full-token" if full_token is None
        else "prompt-a" if prompt_a is None else "prompt-b" if prompt_b is None
        else "profile-token" if profile_token is None
        else "simdgroup-candidate-token" if simdgroup_candidate_token is None else None
    )
    if stage is None:
        _status(
            status_path,
            lane="B_QWEN30_COMPLETE_RUNTIME",
            phase=(
                "DIRECT_PACKED_NATIVE_PROMPT_STAGES_EARNED_AWAITING_KERNEL_DECISIONS"
                if simdgroup_template_decision is None or gateup_fused_decision is None
                else "DIRECT_PACKED_NATIVE_PROMPT_STAGES_EARNED_AWAITING_HCLI_ADAPTER"
            ),
            source=source,
            physical_complete_binary_candidate=complete_candidate,
            preflight_path=str(QWEN30_NATIVE_PREFLIGHT),
            full_token_path=str(QWEN30_NATIVE_FULL_TOKEN),
            prompt_a_path=str(QWEN30_NATIVE_PROMPT_A),
            prompt_b_path=str(QWEN30_NATIVE_PROMPT_B),
            diagnostic_kernel_profile_path=str(QWEN30_NATIVE_PROFILE_TOKEN),
            simdgroup_component_parity_path=str(QWEN30_SIMDGROUP_COMPONENT_PARITY),
            simdgroup_candidate_token_path=str(QWEN30_NATIVE_SIMDGROUP_CANDIDATE_TOKEN),
            simdgroup_candidate_matches_scalar_control_argmax=simdgroup_candidate_matches_control,
            simdgroup_template_parity_decision_path=str(QWEN30_SIMDGROUP_TEMPLATE_PARITY),
            simdgroup_template_parity_decision_status=(
                simdgroup_template_decision.get("status")
                if isinstance(simdgroup_template_decision, Mapping)
                else None
            ),
            gateup_fused_template_parity_decision_path=str(
                QWEN30_GATEUP_FUSED_TEMPLATE_PARITY
            ),
            gateup_fused_template_parity_decision_status=(
                gateup_fused_decision.get("status")
                if isinstance(gateup_fused_decision, Mapping)
                else None
            ),
            hcli_handoff_path=str(QWEN30_HCLI_HANDOFF),
        )
        return
    sealed_failure = _sealed_stage_failure(stage, binding)
    if sealed_failure is not None:
        _status(
            status_path,
            lane="B_QWEN30_COMPLETE_RUNTIME",
            phase="NATIVE_DIRECT_PACKED_STAGE_FAILED_AWAITING_MATERIAL_RUNTIME_CHANGE",
            failed_stage=stage,
            sealed_failure=sealed_failure,
            retry_reopens_only_when=[
                "admitted artifact/source binding changes",
                "native runtime executable sha256 changes after a concrete operator fix",
            ],
            source=source,
            physical_complete_binary_candidate=complete_candidate,
            hcli_handoff_path=str(QWEN30_HCLI_HANDOFF),
        )
        return
    try:
        record = _launch_native_stage(stage, binding)
    except Qwen30BootstrapError as exc:
        _status(
            status_path,
            lane="B_QWEN30_COMPLETE_RUNTIME",
            phase="NATIVE_DIRECT_PACKED_STAGE_LAUNCH_FAILED",
            requested_stage=stage,
            launch_error=str(exc),
            source=source,
            physical_complete_binary_candidate=complete_candidate,
            hcli_handoff_path=str(QWEN30_HCLI_HANDOFF),
        )
        return
    _status(
        status_path,
        lane="B_QWEN30_COMPLETE_RUNTIME",
        phase="NATIVE_DIRECT_PACKED_STAGE_LAUNCHED",
        requested_stage=stage,
        native_process_pid=record.get("pid"),
        native_stdout_path=record.get("stdout_path"),
        native_stderr_path=record.get("stderr_path"),
        source=source,
        physical_complete_binary_candidate=complete_candidate,
        current_artifact=str(RUNTIME_ROOT / "QWEN30_COMPLETE_GRAVITY_ASSEMBLY_CANDIDATE.json"),
        hcli_handoff_path=str(QWEN30_HCLI_HANDOFF),
    )


def run_tg3_cycle() -> None:
    status_path = TG3_ROOT / "QWEN30_TG3_ASCENT_STATUS.json"
    _status(status_path, lane="C_QWEN30_METAL_TG3", phase="COMPLETE_TOKEN_GATE_AUDIT_RUNNING")
    runtime = _read_json(RUNTIME_ROOT / "QWEN30_COMPLETE_GRAVITY_ASSEMBLY_CANDIDATE.json")
    native_active = _read_json(QWEN30_NATIVE_ACTIVE)
    native_last = _read_json(QWEN30_NATIVE_LAST_PROCESS)
    try:
        binding = _native_runtime_binding()
        native_preflight = _native_result_matches("preflight", binding)
        native_full_token = _native_result_matches("full-token", binding)
    except Qwen30BootstrapError:
        binding = None
        native_preflight = None
        native_full_token = None
    router_probe = _read_json(PHYSICAL_ROOT / "kernel" / "QWEN_DUAL_ROUTE_METAL_COMPONENT_PROBE.json")
    direct_router = None
    if router_probe:
        direct_router = next(
            (row for row in router_probe.get("lanes", []) if row.get("model_id") == "Qwen3-Coder-30B-A3B-Instruct"),
            None,
        )
    report = {
        "schema": "hawking.ascension.qwen30_tg3_ascent.v1",
        "status": (
            "FULL_TOKEN_TG3_BLOCKED_ON_CAPABILITY_HCLI_AND_CLEAN_TPS_AFTER_NATIVE_FULL_TOKEN"
            if native_full_token is not None
            else "FULL_TOKEN_TG3_BLOCKED_ON_DIRECT_NATIVE_EXECUTION_IN_PROGRESS"
            if isinstance(native_active, Mapping) and native_active.get("phase") == "RUNNING"
            else "FULL_TOKEN_TG3_BLOCKED_AWAITING_DIRECT_NATIVE_EXECUTION"
        ),
        "recorded_at": _utc_now(),
        "complete_artifact_assembly_input": runtime,
        "native_runtime_stage": {
            "binding": binding,
            "preflight_path": str(QWEN30_NATIVE_PREFLIGHT),
            "preflight_earned": native_preflight is not None,
            "full_token_path": str(QWEN30_NATIVE_FULL_TOKEN),
            "full_token_earned": native_full_token is not None,
            "active_process": native_active,
            "last_process": native_last,
            "hcli_handoff_path": str(QWEN30_HCLI_HANDOFF),
        },
        "physical_complete_binary_candidate": _complete_binary_candidate(),
        "direct_metal_router_component": direct_router,
        "direct_metal_gqa_attention_component": _qwen30_gqa_component(),
        "required_clean_benchmark": {
            "run_model_alone": True,
            "base_true_tps_minimum": 333.0,
            "no_fallback": True,
            "complete_token_timing": True,
            "hcli_prompt_dependent_generation": True,
        },
        "blocking_native_operators": [
            "none_pending_full-token-result" if native_full_token is not None else "direct-packed full-token process/result",
            "prompt-dependent native generation and reserved-tail sampler handling",
            "direct-packed native HTTP/HCLI adapter",
            "clean sustained complete-token TPS profiler and TG10/TG3 gates",
        ],
        "claim_boundary": {
            "router_component_timing_is_not_model_tps": True,
            "no_100_tps_or_tg3_receipt_emitted": True,
        },
    }
    _atomic_json(TG3_ROOT / "QWEN30_TG3_ASCENT_CANDIDATE.json", report)
    _status(
        status_path,
        lane="C_QWEN30_METAL_TG3",
        phase="WAITING_FOR_NATIVE_COMPLETE_TOKEN_RUNTIME",
        current_artifact=str(TG3_ROOT / "QWEN30_TG3_ASCENT_CANDIDATE.json"),
        direct_router_component_available=direct_router is not None,
        direct_attention_component_available=_qwen30_gqa_component().get("status") == "BYTE_HASH_BOUND_EXACT_GQA_COMPONENT",
    )


def watch(kind: str, *, idle_seconds: float) -> int:
    if idle_seconds <= 0:
        raise Qwen30BootstrapError("idle_seconds must be positive")
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
                # A real native stage is deliberately polled much more often
                # than the quiescent planning lane. That keeps a detached
                # all-layer execution's heartbeat moving and starts the next
                # bounded stage soon after its checked receipt lands.
                active = _read_json(QWEN30_NATIVE_ACTIVE) if kind == "runtime" else None
                adapter_active = _read_json(QWEN30_NATIVE_HTTP_ACTIVE) if kind == "runtime" else None
                busy = (active and active.get("phase") == "RUNNING") or (
                    adapter_active and adapter_active.get("phase") == "RUNNING"
                )
                interval = min(idle_seconds, 10.0) if busy else idle_seconds
                time.sleep(interval)
    finally:
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGINT, old_int)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for kind in (
        "runtime",
        "tg3",
        "kernel-parity",
        "kernel-template-parity",
        "gateup-fused-template-parity",
        "route-major-input-offset-regression",
        "revoke-route-offset-defect",
        "transition-validated-payload-cache",
        "transition-paired-scalar-order-production",
        "deploy-production-http-adapter",
        "begin-validated-payload-requalification",
        "http-smoke",
    ):
        once = commands.add_parser(kind, help=f"run one Qwen30 {kind} bootstrap cycle")
        once.add_argument("--watch", action="store_true")
        once.add_argument("--idle-seconds", type=float, default=300.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.watch:
        if args.command in {
            "kernel-parity",
            "kernel-template-parity",
            "gateup-fused-template-parity",
            "route-major-input-offset-regression",
            "revoke-route-offset-defect",
            "transition-validated-payload-cache",
            "transition-paired-scalar-order-production",
            "deploy-production-http-adapter",
            "begin-validated-payload-requalification",
            "http-smoke",
        }:
            raise Qwen30BootstrapError(
                f"{args.command} is a bounded one-shot gate and cannot watch"
            )
        return watch(args.command, idle_seconds=args.idle_seconds)
    if args.command == "runtime":
        run_runtime_cycle()
    elif args.command == "tg3":
        run_tg3_cycle()
    elif args.command == "kernel-parity":
        run_simdgroup_component_parity()
    elif args.command == "kernel-template-parity":
        run_simdgroup_template_parity()
    elif args.command == "gateup-fused-template-parity":
        run_gateup_fused_template_parity()
    elif args.command == "route-major-input-offset-regression":
        run_route_major_input_offset_metal_regression()
    elif args.command == "revoke-route-offset-defect":
        revoke_qwen30_route_offset_defect()
    elif args.command == "transition-validated-payload-cache":
        transition_qwen30_runtime_to_validated_payload_catalog()
    elif args.command == "transition-paired-scalar-order-production":
        transition_qwen30_runtime_to_paired_scalar_order_production()
    elif args.command == "deploy-production-http-adapter":
        deploy_qwen30_paired_scalar_order_production_http_adapter()
        run_runtime_cycle()
    elif args.command == "begin-validated-payload-requalification":
        begin_qwen30_validated_payload_catalog_requalification()
    else:
        run_native_http_chat_smoke()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
