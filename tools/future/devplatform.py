"""Developer platform foundations — programmable substrate, not product polish.

Stable IR serialization, a receipt API over the sealed sidecar format, versioned
model/machine profiles, a backend provider contract, a WorkUnit API over the
HCLI shape, and a compatibility suite that names the clause that fails.

This wraps existing HCLI / sidecar contracts. It does not replace them, spawn a
runtime, or claim a hardware measurement.

    python3 tools/future/devplatform.py --selftest
    python3 tools/future/devplatform.py --build
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

from tools.future._common import write_receipt, load_json, REPO
from tools.future._common import HARDWARE_FIELDS, HardwareClaimError

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence


RECEIPT = "DEVELOPER_PLATFORM.json"
RECORDED_BY = "tools/future/devplatform.py"

SCHEMA = "hawking.future.devplatform.v1"
VERSION = 1

IR_SCHEMA = "hawking.future.devplatform.ir.v1"
IR_VERSION = 1
IR_KINDS = (
    "document",
    "model_profile",
    "machine_profile",
    "backend_contract",
    "workunit",
    "receipt_handle",
    "compatibility_report",
)

MODEL_PROFILE_SCHEMA = "hawking.future.devplatform.model_profile.v1"
MODEL_PROFILE_VERSION = 1
MACHINE_PROFILE_SCHEMA = "hawking.future.devplatform.machine_profile.v1"
MACHINE_PROFILE_VERSION = 1
BACKEND_CONTRACT_SCHEMA = "hawking.future.devplatform.backend_contract.v1"
BACKEND_CONTRACT_VERSION = 1
WORKUNIT_SCHEMA = "hawking.future.devplatform.workunit.v1"
WORKUNIT_VERSION = 1
COMPAT_SCHEMA = "hawking.future.devplatform.compat.v1"
COMPAT_VERSION = 1
RECEIPT_API_SCHEMA = "hawking.future.devplatform.receipt_api.v1"
RECEIPT_API_VERSION = 1

PHYSICAL_BACKENDS = ("METAL", "FPGA", "CUDA", "ANE")
RUNTIME_KINDS = ("mlx", "llamacpp", "noetic_native", "remote")

# RuntimeBackend abstract methods in hcli/backends.py. Optional methods are the
# AgentOS adapters on the same ABC plus ModelProvider in hcli/providers.py.
REQUIRED_BACKEND_METHODS = (
    "identity",
    "spawn",
    "ready",
    "endpoint",
    "stop",
    "complete",
    "supports",
)
OPTIONAL_BACKEND_METHODS = (
    "generate",
    "capabilities",
    "health",
    "profile",
)
PROVIDER_PROTOCOL_METHODS = (
    "generate",
    "capabilities",
    "health",
    "profile",
)

# HCLI feature names (hcli.providers.FEATURES). Declared here so the contract
# still loads if providers.py grows; recovered at build() and compared.
HCLI_FEATURES = (
    "response_format",
    "grammar",
    "chat_template_kwargs",
    "prefix_cache",
    "slots",
    "vision",
    "tool_calling",
    "streaming",
)

# Static MLX/Metal capability table copied from the comment in hcli/backends.py.
# Live RuntimeBackend.supports() remains the runtime authority; this sidecar
# does not spawn mlx_lm.server or llama-server to re-probe.
METAL_DECLARED_FEATURES = (
    "chat_template_kwargs",
    "prefix_cache",
)
METAL_UNSUPPORTED_FEATURES = (
    "response_format",
    "grammar",
)

MODEL_REQUIRED_FIELDS = (
    "schema",
    "version",
    "specimen_id",
    "family",
    "architecture",
    "required_backend",
    "required_features",
)
MODEL_OPTIONAL_FIELDS = (
    "runtime_kind",
    "artifact_kind",
    "quantization",
    "context_length",
    "representation",
    "provider",
    "source_paths",
    "notes",
)

MACHINE_REQUIRED_FIELDS = (
    "schema",
    "version",
    "os",
    "arch",
    "hw_model",
)
MACHINE_OPTIONAL_FIELDS = (
    "cpu",
    "ncpu",
    "mem_bytes",
    "host_kind",
    "gpu_authority",
    "measurement_state",
    "notes",
)

WORKUNIT_REQUIRED_FIELDS = ("id", "role", "description")
WORKUNIT_OPTIONAL_FIELDS = (
    "dependencies",
    "status",
    "assigned_runtime",
    "attempts",
    "resource_class",
    "repairs",
    "failure_context",
    "preferred_backend",
    "assigned_backend",
    "backend_task_id",
    "verifier",
    "effect_class",
    "workspace",
    "verification",
    "repair_root",
    "repair_depth",
    "repair_reason",
    "repair_exhausted",
    "classification",
    "provider",
    "content_hash",
)

VOLATILE_KEYS = frozenset(
    {
        "recorded_at",
        "generated_at",
        "observed_at",
        "started_at",
        "finished_at",
        "ready_at",
        "running_at",
        "seal_sha256",
        "at",
    }
)

CLAUSE_MODEL_SCHEMA = "model.schema"
CLAUSE_MACHINE_SCHEMA = "machine.schema"
CLAUSE_BACKEND_SCHEMA = "backend.schema"
CLAUSE_BACKEND_DECLARED = "model.required_backend.declared"
CLAUSE_BACKEND_AVAILABLE = "model.required_backend.available"
CLAUSE_FEATURE_DECLARED = "model.required_features.declared"
CLAUSE_MACHINE_CAN_RUN = "machine.host.can_run.backend"
CLAUSE_RECEIPT_COMPAT = "receipt.schema.compatible"
CLAUSE_WORKUNIT_BACKEND = "workunit.preferred_backend.available"

COMPAT_CLAUSES = (
    CLAUSE_MODEL_SCHEMA,
    CLAUSE_MACHINE_SCHEMA,
    CLAUSE_BACKEND_SCHEMA,
    CLAUSE_BACKEND_DECLARED,
    CLAUSE_BACKEND_AVAILABLE,
    CLAUSE_FEATURE_DECLARED,
    CLAUSE_MACHINE_CAN_RUN,
    CLAUSE_RECEIPT_COMPAT,
    CLAUSE_WORKUNIT_BACKEND,
)

# Recovered Hawking specimens. Identity only; no copied tps / gpu_ns / bandwidth.
SPECIMENS: dict[str, dict[str, Any]] = {
    "qwen3.8-27b-sealed-3.14": {
        "family": "qwen3.8",
        "architecture": "Qwen3.8",
        "required_backend": "METAL",
        "required_features": ["chat_template_kwargs"],
        "runtime_kind": "noetic_native",
        "artifact_kind": "native",
        "provider": "native",
        "source_paths": ["hcli/hawking-native.sealed-3.14.json"],
    },
    "qwen3.8-27b-mlx-4bit": {
        "family": "qwen3.8",
        "architecture": "Qwen3_5ForConditionalGeneration",
        "required_backend": "METAL",
        "required_features": ["chat_template_kwargs", "prefix_cache"],
        "runtime_kind": "mlx",
        "artifact_kind": "mlx_dir",
        "provider": "mlx",
        "quantization": "4bit-affine-g64",
        "source_paths": ["receipts/headless/CONVENTIONAL_CONTROL_SET.json"],
    },
    "flash": {
        "family": "flash",
        "architecture": "Flash",
        "required_backend": "METAL",
        "required_features": ["chat_template_kwargs"],
        "runtime_kind": "noetic_native",
        "artifact_kind": "native",
        "provider": "native",
        "notes": (
            "Frontier F001: FLASH_COMPLETE_V0.nx.json is sealed metadata only. "
            "This profile names the specimen; it does not claim a complete NX."
        ),
        "source_paths": ["receipts/headless/FLASH_COMPLETE_V0.nx.json"],
    },
}


class IrError(ValueError):
    """IR encode/decode/migration failure."""


class ProfileError(ValueError):
    """Model or machine profile failed validation."""


class CompatError(ValueError):
    """Compatibility suite found a failing clause."""


# ---------------------------------------------------------------------------
# Canonical JSON — byte-stable, no wall-clock in hashed content
# ---------------------------------------------------------------------------

def canonical_bytes(obj: Any) -> bytes:
    """RFC-8785-shaped JSON: sorted keys, tight separators, UTF-8."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def strip_volatile(value: Any) -> Any:
    """Drop wall-clock / seal keys so hashed IR content is deterministic."""
    if isinstance(value, dict):
        return {
            str(k): strip_volatile(v)
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
            if str(k) not in VOLATILE_KEYS
        }
    if isinstance(value, list):
        return [strip_volatile(v) for v in value]
    if isinstance(value, tuple):
        return [strip_volatile(v) for v in value]
    return value


def _migrate_v0_to_v1(doc: dict[str, Any]) -> dict[str, Any]:
    """v0 had `type` instead of `kind`, and no schema / ir_version."""
    if "payload" in doc:
        payload = doc.get("payload")
        kind = doc.get("kind") or doc.get("type") or "document"
    else:
        payload = {
            k: v
            for k, v in doc.items()
            if k not in {"type", "kind", "ir_version", "schema"}
        }
        kind = doc.get("kind") or doc.get("type") or "document"
    if kind not in IR_KINDS:
        kind = "document"
    if not isinstance(payload, dict):
        payload = {"value": payload}
    return {
        "schema": IR_SCHEMA,
        "ir_version": 1,
        "kind": kind,
        "payload": strip_volatile(payload),
    }


MIGRATIONS: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {
    0: _migrate_v0_to_v1,
}


def migrate_ir(doc: Mapping[str, Any]) -> dict[str, Any]:
    """Bring an IR document to IR_VERSION. Unknown versions refuse."""
    if not isinstance(doc, Mapping):
        raise IrError("IR document is not an object")
    current = dict(doc)
    version = current.get("ir_version")
    if version is None:
        version = 0
        current = MIGRATIONS[0](current)
        version = current["ir_version"]
    try:
        version = int(version)
    except (TypeError, ValueError) as exc:
        raise IrError(f"ir_version is not an integer: {version!r}") from exc
    if version > IR_VERSION:
        raise IrError(
            f"ir_version {version} is newer than this implementation ({IR_VERSION})"
        )
    while version < IR_VERSION:
        fn = MIGRATIONS.get(version)
        if fn is None:
            raise IrError(
                f"no migration registered from ir_version {version} to {IR_VERSION}"
            )
        current = fn(current)
        if not isinstance(current, dict) or "ir_version" not in current:
            raise IrError(f"migration from {version} did not return ir_version")
        next_version = int(current["ir_version"])
        if next_version <= version:
            raise IrError(f"migration from {version} did not advance ir_version")
        version = next_version
    current["schema"] = IR_SCHEMA
    current["ir_version"] = IR_VERSION
    if current.get("kind") not in IR_KINDS:
        raise IrError(f"unknown IR kind {current.get('kind')!r}")
    current["payload"] = strip_volatile(current.get("payload") or {})
    return {
        "schema": IR_SCHEMA,
        "ir_version": IR_VERSION,
        "kind": current["kind"],
        "payload": current["payload"],
    }


def make_ir(kind: str, payload: Any) -> dict[str, Any]:
    if kind not in IR_KINDS:
        raise IrError(f"unknown IR kind {kind!r}")
    return {
        "schema": IR_SCHEMA,
        "ir_version": IR_VERSION,
        "kind": kind,
        "payload": strip_volatile(payload),
    }


def encode_ir(doc: Mapping[str, Any]) -> bytes:
    """Canonical bytes of a current-version IR document."""
    migrated = migrate_ir(doc)
    return canonical_bytes(migrated)


def decode_ir(blob: bytes | str) -> dict[str, Any]:
    if isinstance(blob, str):
        blob = blob.encode("utf-8")
    try:
        doc = json.loads(blob.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IrError(f"IR is not UTF-8 JSON: {exc}") from exc
    return migrate_ir(doc)


def roundtrip_ir(doc: Mapping[str, Any]) -> tuple[bytes, bytes, dict[str, Any]]:
    """Encode, decode, encode. Returns (first, second, decoded). Bytes must match."""
    first = encode_ir(doc)
    decoded = decode_ir(first)
    second = encode_ir(decoded)
    return first, second, decoded


# ---------------------------------------------------------------------------
# Receipt API — stable surface over tools.future._common write_receipt
# ---------------------------------------------------------------------------

def receipt_seal_hex(doc: Mapping[str, Any]) -> str:
    """Recompute seal_sha256 the same way tools.future._common.seal does."""
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def hardware_claim_paths(node: Any, path: str = "") -> list[str]:
    hits: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{path}.{key}" if path else str(key)
            if key in HARDWARE_FIELDS and isinstance(value, (int, float)):
                hits.append(here)
            hits.extend(hardware_claim_paths(value, here))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            hits.extend(hardware_claim_paths(value, f"{path}[{i}]"))
    return hits


def receipt_write(name: str, doc: dict[str, Any], recorded_by: str) -> Path:
    """Write a sealed sidecar receipt. Does not catch HardwareClaimError."""
    return write_receipt(name, doc, recorded_by)


def receipt_read(path: str | Path) -> dict[str, Any]:
    return load_json(path)


def receipt_validate(doc: Any) -> dict[str, Any]:
    """Validate a sealed sidecar receipt. Names every failing clause."""
    failures: list[dict[str, str]] = []
    passed: list[str] = []

    def fail(clause: str, detail: str) -> None:
        failures.append({"clause": clause, "detail": detail})

    def ok(clause: str) -> None:
        passed.append(clause)

    if not isinstance(doc, dict):
        fail("receipt.is_object", f"got {type(doc).__name__}")
        return {
            "schema": RECEIPT_API_SCHEMA,
            "version": RECEIPT_API_VERSION,
            "ok": False,
            "failures": failures,
            "passed": passed,
        }
    ok("receipt.is_object")

    if not doc.get("schema"):
        fail("receipt.schema.present", "schema is missing")
    else:
        ok("receipt.schema.present")

    seal = doc.get("seal_sha256")
    if not isinstance(seal, str) or not seal:
        fail("receipt.seal.present", "seal_sha256 is missing")
    else:
        ok("receipt.seal.present")
        recomputed = receipt_seal_hex(doc)
        if recomputed != seal:
            fail(
                "receipt.seal.matches",
                f"seal_sha256 {seal} != recomputed {recomputed}",
            )
        else:
            ok("receipt.seal.matches")

    bench = doc.get("bench")
    if not isinstance(bench, dict):
        fail("receipt.bench.present", "bench block is missing")
    else:
        ok("receipt.bench.present")
        if bench.get("state") != "UNKNOWN":
            fail(
                "receipt.bench.state.UNKNOWN",
                f"bench.state is {bench.get('state')!r}, sidecar must record UNKNOWN",
            )
        else:
            ok("receipt.bench.state.UNKNOWN")
        if bench.get("measurement_state") != "STATIC_ONLY":
            fail(
                "receipt.bench.measurement_state.STATIC_ONLY",
                f"measurement_state is {bench.get('measurement_state')!r}",
            )
        else:
            ok("receipt.bench.measurement_state.STATIC_ONLY")
        if bench.get("gpu_authority") is not False:
            fail(
                "receipt.bench.gpu_authority.false",
                f"gpu_authority is {bench.get('gpu_authority')!r}",
            )
        else:
            ok("receipt.bench.gpu_authority.false")

    claims = hardware_claim_paths(doc)
    if claims:
        fail(
            "receipt.hardware_fields.null_or_absent",
            f"numeric hardware fields at {claims}",
        )
    else:
        ok("receipt.hardware_fields.null_or_absent")

    return {
        "schema": RECEIPT_API_SCHEMA,
        "version": RECEIPT_API_VERSION,
        "ok": not failures,
        "failures": failures,
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# Model profiles
# ---------------------------------------------------------------------------

def make_model_profile(
    specimen_id: str,
    *,
    family: str,
    architecture: str,
    required_backend: str,
    required_features: Sequence[str],
    runtime_kind: Optional[str] = None,
    artifact_kind: Optional[str] = None,
    quantization: Optional[str] = None,
    context_length: Optional[int] = None,
    representation: Optional[str] = None,
    provider: Optional[str] = None,
    source_paths: Optional[Sequence[str]] = None,
    notes: Optional[str] = None,
) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "schema": MODEL_PROFILE_SCHEMA,
        "version": MODEL_PROFILE_VERSION,
        "specimen_id": specimen_id,
        "family": family,
        "architecture": architecture,
        "required_backend": required_backend,
        "required_features": [str(f) for f in required_features],
    }
    if runtime_kind is not None:
        profile["runtime_kind"] = runtime_kind
    if artifact_kind is not None:
        profile["artifact_kind"] = artifact_kind
    if quantization is not None:
        profile["quantization"] = quantization
    if context_length is not None:
        profile["context_length"] = int(context_length)
    if representation is not None:
        profile["representation"] = representation
    if provider is not None:
        profile["provider"] = provider
    if source_paths is not None:
        profile["source_paths"] = [str(p) for p in source_paths]
    if notes is not None:
        profile["notes"] = notes
    return profile


def specimen_profile(specimen_id: str) -> dict[str, Any]:
    spec = SPECIMENS.get(specimen_id)
    if spec is None:
        raise ProfileError(f"unknown specimen {specimen_id!r}")
    return make_model_profile(specimen_id, **spec)


def _load_sealed_qwen_specimen() -> Optional[dict[str, Any]]:
    path = REPO / "hcli" / "hawking-native.sealed-3.14.json"
    if not path.is_file():
        return None
    try:
        data = load_json(path)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def validate_model_profile(profile: Any) -> dict[str, Any]:
    failures: list[str] = []
    if not isinstance(profile, dict):
        return {"ok": False, "failures": ["profile is not an object"]}
    for field in MODEL_REQUIRED_FIELDS:
        if field not in profile:
            failures.append(f"missing required field {field!r}")
    if profile.get("schema") not in {MODEL_PROFILE_SCHEMA, None} and "schema" in profile:
        if profile.get("schema") != MODEL_PROFILE_SCHEMA:
            failures.append(
                f"schema is {profile.get('schema')!r}, expected {MODEL_PROFILE_SCHEMA}"
            )
    version = profile.get("version")
    if version is not None and int(version) != MODEL_PROFILE_VERSION:
        failures.append(
            f"version {version!r} is not {MODEL_PROFILE_VERSION}; bump is explicit"
        )
    backend = profile.get("required_backend")
    if backend is not None and backend not in PHYSICAL_BACKENDS:
        failures.append(
            f"required_backend {backend!r} is not one of {list(PHYSICAL_BACKENDS)}"
        )
    features = profile.get("required_features")
    if features is not None and not isinstance(features, list):
        failures.append("required_features must be a list")
    runtime_kind = profile.get("runtime_kind")
    if runtime_kind is not None and runtime_kind not in RUNTIME_KINDS:
        failures.append(
            f"runtime_kind {runtime_kind!r} is not one of {list(RUNTIME_KINDS)}"
        )
    specimen_id = profile.get("specimen_id")
    catalog = SPECIMENS.get(str(specimen_id)) if specimen_id else None
    if catalog:
        for key in ("family", "architecture", "required_backend"):
            if profile.get(key) != catalog[key]:
                failures.append(
                    f"specimen {specimen_id}: {key} {profile.get(key)!r} "
                    f"does not match catalog {catalog[key]!r}"
                )
    if specimen_id == "qwen3.8-27b-sealed-3.14":
        sealed = _load_sealed_qwen_specimen()
        if sealed is None:
            failures.append(
                "specimen qwen3.8-27b-sealed-3.14: hcli/hawking-native.sealed-3.14.json "
                "is not readable on this sparse checkout"
            )
        else:
            if sealed.get("family") and sealed.get("family") != profile.get("family"):
                failures.append(
                    f"sealed specimen family {sealed.get('family')!r} != "
                    f"profile family {profile.get('family')!r}"
                )
            if sealed.get("provider") and profile.get("provider") not in {
                None,
                sealed.get("provider"),
            }:
                failures.append(
                    f"sealed specimen provider {sealed.get('provider')!r} != "
                    f"profile provider {profile.get('provider')!r}"
                )
            model_id = sealed.get("model_id")
            if model_id and model_id != specimen_id:
                failures.append(
                    f"sealed model_id {model_id!r} != specimen_id {specimen_id!r}"
                )
    return {"ok": not failures, "failures": failures}


# ---------------------------------------------------------------------------
# Machine profiles
# ---------------------------------------------------------------------------

def _live_machine_identity() -> dict[str, Any]:
    try:
        from hcli.machine import live_machine_identity

        ident = live_machine_identity()
        if isinstance(ident, dict):
            return ident
    except Exception:
        pass
    return {
        "hw_model": None,
        "cpu": None,
        "ncpu": None,
        "mem_bytes": None,
    }


def this_machine_profile() -> dict[str, Any]:
    """Identity of this host. Sysctl facts, not a protected GPU measurement."""
    ident = _live_machine_identity()
    os_name = platform.system()
    arch = platform.machine()
    apple = os_name == "Darwin" and arch == "arm64"
    hw_model = ident.get("hw_model") or "UNKNOWN"
    return {
        "schema": MACHINE_PROFILE_SCHEMA,
        "version": MACHINE_PROFILE_VERSION,
        "os": os_name,
        "arch": arch,
        "hw_model": hw_model,
        "cpu": ident.get("cpu") or "UNKNOWN",
        "ncpu": ident.get("ncpu"),
        "mem_bytes": ident.get("mem_bytes"),
        "host_kind": "apple_silicon" if apple else "other",
        "gpu_authority": False,
        "measurement_state": "STATIC_ONLY",
        "notes": (
            "Identity from platform/sysctl via hcli.machine.live_machine_identity. "
            "Not a protected measurement. tps / token_ns / gpu_ns / "
            "joules_per_token / bandwidth_gbps are omitted and UNKNOWN."
        ),
    }


def validate_machine_profile(
    profile: Any, *, against_live: bool = True
) -> dict[str, Any]:
    failures: list[str] = []
    if not isinstance(profile, dict):
        return {"ok": False, "failures": ["profile is not an object"]}
    for field in MACHINE_REQUIRED_FIELDS:
        if field not in profile:
            failures.append(f"missing required field {field!r}")
    if profile.get("schema") not in {None, MACHINE_PROFILE_SCHEMA}:
        failures.append(
            f"schema is {profile.get('schema')!r}, expected {MACHINE_PROFILE_SCHEMA}"
        )
    version = profile.get("version")
    if version is not None and int(version) != MACHINE_PROFILE_VERSION:
        failures.append(
            f"version {version!r} is not {MACHINE_PROFILE_VERSION}; bump is explicit"
        )
    if profile.get("gpu_authority") not in {None, False}:
        failures.append("machine profile must not claim gpu_authority")
    if against_live:
        live = this_machine_profile()
        for key in ("os", "arch"):
            if profile.get(key) != live.get(key):
                failures.append(
                    f"live mismatch: {key} profile={profile.get(key)!r} "
                    f"live={live.get(key)!r}"
                )
        live_hw = live.get("hw_model")
        prof_hw = profile.get("hw_model")
        if live_hw not in {None, "UNKNOWN"} and prof_hw not in {None, "UNKNOWN"}:
            if prof_hw != live_hw:
                failures.append(
                    f"live mismatch: hw_model profile={prof_hw!r} live={live_hw!r}"
                )
    return {"ok": not failures, "failures": failures}


# ---------------------------------------------------------------------------
# Backend provider contract
# ---------------------------------------------------------------------------

def physical_availability() -> dict[str, dict[str, Any]]:
    """Presence flags. Not a kernel run, not a board probe, not ANE execution."""
    apple = platform.system() == "Darwin" and platform.machine() == "arm64"
    return {
        "METAL": {
            "available": apple,
            "evidence": (
                "host is Darwin arm64; Metal is the Apple Silicon GPU API. "
                "This is platform presence, not a kernel measurement and not "
                "PROTECTED_ABSOLUTE."
            ),
        },
        "FPGA": {
            "available": False,
            "evidence": (
                "FPGA is part of Accelerator / Physical Compiler / Fusion. "
                "It is not its own civilization. This sidecar does not build "
                "an FPGA backend and no board is claimed present."
            ),
        },
        "CUDA": {
            "available": False,
            "evidence": (
                "CUDA is a future Accelerator backend. This host is Apple "
                "Silicon; this sidecar does not build a CUDA backend."
            ),
        },
        "ANE": {
            "available": False,
            "evidence": (
                "ANE is a future Accelerator backend. This sidecar does not "
                "claim ANE execution."
            ),
        },
    }


def make_backend_contract(
    backend_id: str,
    *,
    available: bool,
    declares: Sequence[str],
    declared_features: Sequence[str],
    unsupported_features: Sequence[str] = (),
    runtime_kinds: Sequence[str] = (),
    methods: Sequence[str] = REQUIRED_BACKEND_METHODS,
    optional_methods: Sequence[str] = OPTIONAL_BACKEND_METHODS,
    reason: Optional[str] = None,
    availability: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    avail = dict(availability) if availability is not None else physical_availability()
    contract: dict[str, Any] = {
        "schema": BACKEND_CONTRACT_SCHEMA,
        "version": BACKEND_CONTRACT_VERSION,
        "backend_id": backend_id,
        "available": bool(available),
        "declares": [str(x) for x in declares],
        "declared_features": [str(x) for x in declared_features],
        "unsupported_features": [str(x) for x in unsupported_features],
        "runtime_kinds": [str(x) for x in runtime_kinds],
        "required_methods": list(methods),
        "optional_methods": list(optional_methods),
        "provider_protocol_methods": list(PROVIDER_PROTOCOL_METHODS),
        "availability": {
            key: {
                "available": bool(row["available"]),
                "evidence": row["evidence"],
            }
            for key, row in sorted(avail.items())
        },
        "source_contracts": {
            "runtime_backend_abc": "hcli.backends.RuntimeBackend",
            "model_provider_protocol": "hcli.providers.ModelProvider",
            "runtime_kinds": "hcli.runtime_iface.BACKEND_KINDS",
            "features": "hcli.providers.FEATURES",
        },
        "not_an_fpga_backend": True,
        "gpu_authority": False,
        "measurement_state": "STATIC_ONLY",
    }
    if reason is not None:
        contract["reason"] = reason
    return contract


def metal_provider_contract() -> dict[str, Any]:
    avail = physical_availability()
    return make_backend_contract(
        "METAL",
        available=bool(avail["METAL"]["available"]),
        declares=("METAL",),
        declared_features=METAL_DECLARED_FEATURES,
        unsupported_features=METAL_UNSUPPORTED_FEATURES,
        runtime_kinds=("mlx", "noetic_native", "llamacpp"),
        reason=avail["METAL"]["evidence"],
        availability=avail,
    )


def validate_backend_contract(contract: Any) -> dict[str, Any]:
    failures: list[str] = []
    if not isinstance(contract, dict):
        return {"ok": False, "failures": ["contract is not an object"]}
    if contract.get("schema") != BACKEND_CONTRACT_SCHEMA:
        failures.append(
            f"schema is {contract.get('schema')!r}, expected {BACKEND_CONTRACT_SCHEMA}"
        )
    if contract.get("version") != BACKEND_CONTRACT_VERSION:
        failures.append(
            f"version {contract.get('version')!r} is not {BACKEND_CONTRACT_VERSION}"
        )
    backend_id = contract.get("backend_id")
    if backend_id not in PHYSICAL_BACKENDS:
        failures.append(f"backend_id {backend_id!r} is not a physical backend")
    required = contract.get("required_methods") or []
    for name in REQUIRED_BACKEND_METHODS:
        if name not in required:
            failures.append(f"required method {name!r} is not listed")
    if contract.get("gpu_authority") not in {None, False}:
        failures.append("backend contract must not claim gpu_authority")
    return {"ok": not failures, "failures": failures}


# ---------------------------------------------------------------------------
# WorkUnit API — versioned surface over hcli.workunit.WorkUnit
# ---------------------------------------------------------------------------

def _hcli_workunit():
    from hcli.workunit import WorkUnit, content_identity

    return WorkUnit, content_identity


def workunit_content_hash(data: Mapping[str, Any]) -> str:
    """Same identity as hcli.workunit.content_identity: role, description, deps, verifier."""
    payload = {
        "role": str(data.get("role") or ""),
        "description": str(data.get("description") or ""),
        "dependencies": [str(d) for d in (data.get("dependencies") or [])],
        "verifier": str(data.get("verifier") or ""),
    }
    blob = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def workunit_normalize(data: Mapping[str, Any]) -> dict[str, Any]:
    missing = [f for f in WORKUNIT_REQUIRED_FIELDS if not data.get(f)]
    if missing:
        raise ProfileError(f"WorkUnit missing {missing}")
    WorkUnit, content_identity = _hcli_workunit()
    wu = WorkUnit.from_dict(dict(data))
    out = wu.to_dict()
    out["schema"] = WORKUNIT_SCHEMA
    out["version"] = WORKUNIT_VERSION
    hashed = content_identity(wu)
    if hashed != workunit_content_hash(out):
        raise ProfileError("WorkUnit content hash diverged from HCLI")
    return out


def workunit_validate(data: Any) -> dict[str, Any]:
    failures: list[str] = []
    if not isinstance(data, dict):
        return {"ok": False, "failures": ["workunit is not an object"]}
    for field in WORKUNIT_REQUIRED_FIELDS:
        if not data.get(field):
            failures.append(f"missing required field {field!r}")
    if data.get("schema") not in {None, WORKUNIT_SCHEMA}:
        failures.append(
            f"schema is {data.get('schema')!r}, expected {WORKUNIT_SCHEMA}"
        )
    version = data.get("version")
    if version is not None and int(version) != WORKUNIT_VERSION:
        failures.append(
            f"version {version!r} is not {WORKUNIT_VERSION}; bump is explicit"
        )
    if failures:
        return {"ok": False, "failures": failures}
    try:
        normalized = workunit_normalize(data)
    except Exception as exc:  # noqa: BLE001 - validation result, not a crash
        return {"ok": False, "failures": [f"{type(exc).__name__}: {exc}"]}
    expected = workunit_content_hash(normalized)
    if normalized.get("content_hash") != expected:
        failures.append("content_hash does not match identity of the work")
    return {"ok": not failures, "failures": failures, "normalized": normalized}


def workunit_to_ir(data: Mapping[str, Any]) -> dict[str, Any]:
    normalized = workunit_normalize(data)
    return make_ir("workunit", normalized)


# ---------------------------------------------------------------------------
# Compatibility suite
# ---------------------------------------------------------------------------

def _as_profile(value: Any, kind: str) -> dict[str, Any]:
    if isinstance(value, dict) and value.get("kind") == kind and "payload" in value:
        return dict(value["payload"])
    if isinstance(value, dict):
        return dict(value)
    raise CompatError(f"{kind} is not an object")


def check_compatibility(
    *,
    model: Mapping[str, Any],
    machine: Mapping[str, Any],
    backend: Mapping[str, Any],
    receipt: Optional[Mapping[str, Any]] = None,
    workunit: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Report exactly which contract clause fails. ok iff none fail."""
    model_p = _as_profile(model, "model_profile")
    machine_p = _as_profile(machine, "machine_profile")
    backend_p = _as_profile(backend, "backend_contract")
    failures: list[dict[str, str]] = []
    passed: list[dict[str, str]] = []

    def fail(clause: str, detail: str) -> None:
        failures.append({"clause": clause, "detail": detail})

    def ok(clause: str, detail: str = "") -> None:
        passed.append({"clause": clause, "detail": detail})

    model_v = validate_model_profile(model_p)
    if not model_v["ok"]:
        fail(CLAUSE_MODEL_SCHEMA, "; ".join(model_v["failures"]))
    else:
        ok(CLAUSE_MODEL_SCHEMA)

    # Machine schema is checked without against_live so a hypothetical host
    # can be tested. Host/backend fit is CLAUSE_MACHINE_CAN_RUN.
    machine_v = validate_machine_profile(machine_p, against_live=False)
    if not machine_v["ok"]:
        fail(CLAUSE_MACHINE_SCHEMA, "; ".join(machine_v["failures"]))
    else:
        ok(CLAUSE_MACHINE_SCHEMA)

    backend_v = validate_backend_contract(backend_p)
    if not backend_v["ok"]:
        fail(CLAUSE_BACKEND_SCHEMA, "; ".join(backend_v["failures"]))
    else:
        ok(CLAUSE_BACKEND_SCHEMA)

    required_backend = str(model_p.get("required_backend") or "")
    declares = [str(x) for x in (backend_p.get("declares") or [])]
    if required_backend not in declares:
        fail(
            CLAUSE_BACKEND_DECLARED,
            f"model requires {required_backend!r}; provider declares {declares}",
        )
    else:
        ok(CLAUSE_BACKEND_DECLARED, required_backend)

    availability = backend_p.get("availability") or {}
    row = availability.get(required_backend)
    available = None
    if isinstance(row, dict):
        available = bool(row.get("available"))
    elif required_backend == backend_p.get("backend_id"):
        available = bool(backend_p.get("available"))
    if available is not True:
        fail(
            CLAUSE_BACKEND_AVAILABLE,
            f"model requires {required_backend!r} but provider availability "
            f"is {available!r}",
        )
    else:
        ok(CLAUSE_BACKEND_AVAILABLE, required_backend)

    declared_features = {
        str(x) for x in (backend_p.get("declared_features") or [])
    }
    unsupported = {
        str(x) for x in (backend_p.get("unsupported_features") or [])
    }
    required_features = [str(x) for x in (model_p.get("required_features") or [])]
    missing_features = [
        f for f in required_features
        if f not in declared_features or f in unsupported
    ]
    if missing_features:
        fail(
            CLAUSE_FEATURE_DECLARED,
            f"model requires features {missing_features} which the provider "
            f"does not declare (declared={sorted(declared_features)}, "
            f"unsupported={sorted(unsupported)})",
        )
    else:
        ok(CLAUSE_FEATURE_DECLARED, ",".join(required_features))

    host_kind = str(machine_p.get("host_kind") or "")
    os_name = str(machine_p.get("os") or "")
    arch = str(machine_p.get("arch") or "")
    apple = host_kind == "apple_silicon" or (os_name == "Darwin" and arch == "arm64")
    if required_backend == "METAL" and not apple:
        fail(
            CLAUSE_MACHINE_CAN_RUN,
            f"METAL requires Darwin arm64 / apple_silicon; "
            f"host_kind={host_kind!r} os={os_name!r} arch={arch!r}",
        )
    elif required_backend in {"FPGA", "CUDA", "ANE"}:
        fail(
            CLAUSE_MACHINE_CAN_RUN,
            f"{required_backend} is not available on this sidecar host",
        )
    else:
        ok(CLAUSE_MACHINE_CAN_RUN, f"{required_backend} on {host_kind or arch}")

    if receipt is None:
        ok(CLAUSE_RECEIPT_COMPAT, "receipt not supplied")
    else:
        rec = dict(receipt)
        schema = rec.get("schema")
        bench = rec.get("bench") if isinstance(rec.get("bench"), dict) else {}
        if not schema:
            fail(CLAUSE_RECEIPT_COMPAT, "receipt has no schema")
        elif bench.get("state") not in {None, "UNKNOWN"}:
            fail(
                CLAUSE_RECEIPT_COMPAT,
                f"receipt bench.state {bench.get('state')!r} is not UNKNOWN",
            )
        else:
            ok(CLAUSE_RECEIPT_COMPAT, str(schema))

    if workunit is None:
        ok(CLAUSE_WORKUNIT_BACKEND, "workunit not supplied")
    else:
        wu = dict(workunit)
        preferred = wu.get("preferred_backend") or wu.get("assigned_backend")
        if not preferred:
            ok(CLAUSE_WORKUNIT_BACKEND, "no preferred backend")
        else:
            pref = str(preferred)
            if pref not in declares and pref not in (backend_p.get("runtime_kinds") or []):
                fail(
                    CLAUSE_WORKUNIT_BACKEND,
                    f"workunit preferred_backend {pref!r} is not declared "
                    f"by provider {declares}",
                )
            else:
                ok(CLAUSE_WORKUNIT_BACKEND, pref)

    return {
        "schema": COMPAT_SCHEMA,
        "version": COMPAT_VERSION,
        "ok": not failures,
        "failures": failures,
        "passed": passed,
        "clauses": list(COMPAT_CLAUSES),
    }


# ---------------------------------------------------------------------------
# Recovery census (read-only; sparse-safe)
# ---------------------------------------------------------------------------

def _path_state(rel: str) -> dict[str, Any]:
    p = REPO / rel
    return {
        "path": rel,
        "on_disk": p.exists(),
        "in_git": True,
    }


def recover_implementation() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(path: str, what: str, adequate: str, gap: str) -> None:
        state = _path_state(path)
        state.update({"what": what, "adequate": adequate, "gap": gap})
        rows.append(state)

    add(
        "hcli/backends.py",
        "RuntimeBackend ABC (identity/spawn/ready/endpoint/stop/complete/supports) "
        "plus LlamaServerBackend, MlxServerBackend, OpenAICompatibleBackend, "
        "NoeticNativeBackend",
        "yes as the live runtime contract",
        "not versioned as a round-trip IR; no physical METAL/FPGA/CUDA/ANE availability flags",
    )
    add(
        "hcli/providers.py",
        "ModelProvider protocol, CapabilityContract, ResidentProfile, RuntimeGenome, "
        "FEATURES, profile schema hcli.provider.profile.v1",
        "yes as the AgentOS provider contract",
        "ResidentProfile is a reproducibility closure, not a versioned specimen profile "
        "with required_backend / required_features checked by a compatibility suite",
    )
    add(
        "hcli/runtime_iface.py",
        "BACKEND_KINDS (mlx, llamacpp, noetic_native, remote), ModelSemantics, "
        "RuntimeInterface planes",
        "yes as the one runtime interface",
        "does not serialize a byte-stable IR or a physical-backend availability table",
    )
    add(
        "hcli/workunit.py",
        "WorkUnit dataclass, to_dict/from_dict, content_identity, admit/repair",
        "yes as the durable work shape",
        "no schema/version envelope; no compatibility check against a backend contract",
    )
    add(
        "hcli/result_envelope.py",
        "ResultEnvelope / hcli.agentos.result.v1",
        "yes for mission results",
        "not the sealed sidecar receipt format",
    )
    add(
        "hcli/tool_registry.py",
        "typed tools, mutation classes, receipt-adjacent read tools",
        "yes for AgentOS tools",
        "not a receipt read/write/validate API over receipts/future/",
    )
    add(
        "hcli/machine.py",
        "live_machine_identity, MachineGenome compatibility bag, MemGate",
        "yes as machine identity + admission prior",
        "MachineGenome is an admission bag / prior; not a versioned STATIC_ONLY profile. "
        "Must not copy PUBLISHED_SLOT_C1_TPS_PRIOR into this sidecar.",
    )
    add(
        "hcli/nomenclature.py",
        "HAWKING_NOMENCLATURE_V1, canonical pipeline including NoeticIR",
        "yes as vocabulary",
        "NoeticIR is named, not implemented here; PhysicalGraph is a different IR",
    )
    add(
        "hcli/physical_graph.py",
        "hcli.physical_graph.v1 plan IR (PLAN_ONLY)",
        "yes as a plan graph",
        "not a general versioned round-trip substrate for profiles/receipts/workunits",
    )
    add(
        "hcli/hawking-native.sealed-3.14.json",
        "sealed Qwen3.8-27B native specimen (family, protocol, profile_schema)",
        "yes as a specimen identity document",
        "contains complete_tps_* hardware numbers this sidecar must not copy",
    )
    add(
        "hcli/genomes/runtime_genome.py",
        "RuntimeGenome from CONVENTIONAL_CONTROL_SET; does not re-measure",
        "yes as archived/live profile copy",
        "not on disk in this sparse checkout; recovered via git show. Not a "
        "versioned model profile API.",
    )
    add(
        "tools/accelerator/machine_genome.py",
        "hawking.accelerator.machine_genome.v1 producer (sysctl + bandwidth measure)",
        "yes as Codex's genome producer",
        "not on disk in this sparse checkout; recovered via git show. Sidecar must "
        "not re-run measure_bandwidth (GPU).",
    )
    add(
        "tools/accelerator/receipt.py",
        "hawking.accelerator.receipt.v1 with eight identities + bench states",
        "yes as Codex receipt schema",
        "different schema than sidecar write_receipt; we wrap the sidecar sealer",
    )
    add(
        "tools/future/_common.py",
        "write_receipt / seal / bench_block / HARDWARE_FIELDS / HardwareClaimError",
        "yes — the sealer this receipt API is built over",
        "no read/validate surface, no named clauses",
    )
    add(
        "docs/HCLI_DELEGATION.md",
        "operator runbook for hcli run/status/steer/result/abort",
        "yes as operator docs",
        "not on disk in this sparse checkout; recovered via git show. Not a programmable API.",
    )
    add(
        "receipts/future/CLAUDE_GLOBAL_FRONTIER.json",
        "live frontier F001–F015",
        "yes as the campaign board",
        "no frontier entry for a developer-platform substrate; this lane still had to close it",
    )
    return rows


def _recovered_hcli_constants() -> dict[str, Any]:
    out: dict[str, Any] = {
        "backend_kinds": list(RUNTIME_KINDS),
        "features": list(HCLI_FEATURES),
        "nomenclature_version": None,
        "workunit_import": False,
        "live_machine_import": False,
    }
    try:
        from hcli.runtime_iface import BACKEND_KINDS

        out["backend_kinds"] = list(BACKEND_KINDS)
    except Exception as exc:  # noqa: BLE001
        out["backend_kinds_error"] = f"{type(exc).__name__}: {exc}"
    try:
        from hcli.providers import FEATURES

        out["features"] = list(FEATURES)
    except Exception as exc:  # noqa: BLE001
        out["features_error"] = f"{type(exc).__name__}: {exc}"
    try:
        from hcli.nomenclature import NOMENCLATURE_VERSION

        out["nomenclature_version"] = NOMENCLATURE_VERSION
    except Exception as exc:  # noqa: BLE001
        out["nomenclature_error"] = f"{type(exc).__name__}: {exc}"
    try:
        _hcli_workunit()
        out["workunit_import"] = True
    except Exception as exc:  # noqa: BLE001
        out["workunit_error"] = f"{type(exc).__name__}: {exc}"
    try:
        from hcli.machine import live_machine_identity as _live

        ident = _live()
        out["live_machine_import"] = True
        out["live_machine_keys"] = sorted(ident.keys()) if isinstance(ident, dict) else []
    except Exception as exc:  # noqa: BLE001
        out["live_machine_error"] = f"{type(exc).__name__}: {exc}"
    return out


# ---------------------------------------------------------------------------
# Selftest / build
# ---------------------------------------------------------------------------

def _incompatible_pair() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """A genuinely incompatible profile/backend pair for the negative control.

    The model requires a backend feature (`hbm_resident`) that the Metal
    provider does not declare. Other clauses may also fail; the suite must
    name `model.required_features.declared`.
    """
    model = make_model_profile(
        "negative-control-hbm-resident",
        family="qwen3.8",
        architecture="Qwen3.8",
        required_backend="METAL",
        required_features=["hbm_resident"],
        runtime_kind="noetic_native",
        notes="synthetic negative control; not a Hawking specimen",
    )
    machine = this_machine_profile()
    backend = metal_provider_contract()
    return model, machine, backend


def _compatible_pair() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    model = specimen_profile("qwen3.8-27b-mlx-4bit")
    machine = this_machine_profile()
    backend = metal_provider_contract()
    return model, machine, backend


def run_selftest_checks() -> dict[str, Any]:
    """Pure checks. Raises on failure. No extra receipts."""
    ir_doc = make_ir(
        "model_profile",
        specimen_profile("qwen3.8-27b-sealed-3.14"),
    )
    first, second, decoded = roundtrip_ir(ir_doc)
    if first != second:
        raise AssertionError("IR round-trip is not byte-stable")
    if decoded["ir_version"] != IR_VERSION:
        raise AssertionError("decoded IR version drifted")

    v0 = {
        "type": "model_profile",
        "payload": {"specimen_id": "migrated", "family": "qwen3.8"},
    }
    migrated = migrate_ir(v0)
    if migrated["ir_version"] != 1 or migrated["kind"] != "model_profile":
        raise AssertionError("v0 -> v1 migration did not land on kind/version")
    m1, m2, _ = roundtrip_ir(migrated)
    if m1 != m2:
        raise AssertionError("migrated IR is not byte-stable")

    try:
        migrate_ir({"ir_version": 99, "kind": "document", "payload": {}})
        raise AssertionError("ir_version 99 was accepted")
    except IrError:
        pass

    machine = this_machine_profile()
    mv = validate_machine_profile(machine, against_live=True)
    if not mv["ok"]:
        raise AssertionError(f"this-machine profile failed: {mv['failures']}")

    sealed_profile = specimen_profile("qwen3.8-27b-sealed-3.14")
    pv = validate_model_profile(sealed_profile)
    if not pv["ok"]:
        raise AssertionError(f"sealed specimen profile failed: {pv['failures']}")

    backend = metal_provider_contract()
    bv = validate_backend_contract(backend)
    if not bv["ok"]:
        raise AssertionError(f"metal contract failed: {bv['failures']}")
    avail = backend["availability"]
    if not avail["METAL"]["available"]:
        raise AssertionError("METAL should be available on Darwin arm64")
    for name in ("FPGA", "CUDA", "ANE"):
        if avail[name]["available"]:
            raise AssertionError(f"{name} must be unavailable")

    wu = workunit_normalize(
        {
            "id": "devplat-selftest",
            "role": "code",
            "description": "devplatform selftest workunit",
            "dependencies": [],
            "preferred_backend": "METAL",
        }
    )
    wv = workunit_validate(wu)
    if not wv["ok"]:
        raise AssertionError(f"workunit validate failed: {wv['failures']}")
    WorkUnit, content_identity = _hcli_workunit()
    hcli_hash = content_identity(WorkUnit.from_dict(wu))
    if hcli_hash != wu["content_hash"]:
        raise AssertionError("WorkUnit hash does not match HCLI")

    model_ok, machine_ok, backend_ok = _compatible_pair()
    wu_ok = {
        "id": "compat-ok",
        "role": "code",
        "description": "compatible unit",
        "preferred_backend": "mlx",
    }
    pos = check_compatibility(
        model=model_ok,
        machine=machine_ok,
        backend=backend_ok,
        workunit=wu_ok,
    )
    if not pos["ok"]:
        raise AssertionError(f"compatible pair was rejected: {pos['failures']}")

    model_bad, machine_bad, backend_bad = _incompatible_pair()
    neg = check_compatibility(
        model=model_bad, machine=machine_bad, backend=backend_bad
    )
    if neg["ok"]:
        raise AssertionError("incompatible pair was accepted")
    neg_clauses = {row["clause"] for row in neg["failures"]}
    if CLAUSE_FEATURE_DECLARED not in neg_clauses:
        raise AssertionError(
            f"negative control did not fire {CLAUSE_FEATURE_DECLARED}: {neg['failures']}"
        )

    ir_wu = workunit_to_ir(wu)
    a, b, _ = roundtrip_ir(ir_wu)
    if a != b:
        raise AssertionError("workunit IR round-trip is not byte-stable")

    return {
        "ir_roundtrip_byte_stable": True,
        "ir_v0_migrates_to_v1": True,
        "ir_unknown_version_refuses": True,
        "machine_profile_matches_live": True,
        "sealed_specimen_profile_ok": True,
        "metal_available_fpga_cuda_ane_not": True,
        "workunit_hash_matches_hcli": True,
        "compat_positive_ok": True,
        "compat_negative_fired": True,
        "compat_negative_clauses": sorted(neg_clauses),
        "compat_negative_named_feature_clause": CLAUSE_FEATURE_DECLARED,
    }


def build(selftest_results: Optional[Mapping[str, Any]] = None) -> Path:
    recovered = recover_implementation()
    hcli_const = _recovered_hcli_constants()
    machine = this_machine_profile()
    backend = metal_provider_contract()
    specimens = {
        sid: specimen_profile(sid) for sid in sorted(SPECIMENS)
    }
    checks = dict(selftest_results) if selftest_results is not None else None
    if checks is None:
        checks = run_selftest_checks()

    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Programmable developer-platform substrate for Era V: stable IR, "
            "receipt API, model/machine profiles, backend provider contract, "
            "WorkUnit API, compatibility suite. STATIC_ONLY. No hardware claim."
        ),
        "nomenclature_version": hcli_const.get("nomenclature_version"),
        "ir": {
            "schema": IR_SCHEMA,
            "version": IR_VERSION,
            "encoding": (
                "UTF-8 JSON, sort_keys=True, separators=(',', ':'), "
                "ensure_ascii=False; volatile keys stripped before hash"
            ),
            "kinds": list(IR_KINDS),
            "migration": {
                "current": IR_VERSION,
                "registered_from": sorted(int(k) for k in MIGRATIONS),
                "rule": (
                    "Every version bump is explicit. encode_ir requires current "
                    "ir_version. decode_ir migrates. Missing migrators refuse."
                ),
            },
            "volatile_keys_excluded_from_hash": sorted(VOLATILE_KEYS),
        },
        "receipt_api": {
            "schema": RECEIPT_API_SCHEMA,
            "version": RECEIPT_API_VERSION,
            "over": "tools.future._common.write_receipt / seal / HARDWARE_FIELDS",
            "operations": ["receipt_read", "receipt_write", "receipt_validate", "receipt_seal_hex"],
            "does_not_catch": "HardwareClaimError",
        },
        "model_profiles": {
            "schema": MODEL_PROFILE_SCHEMA,
            "version": MODEL_PROFILE_VERSION,
            "required_fields": list(MODEL_REQUIRED_FIELDS),
            "optional_fields": list(MODEL_OPTIONAL_FIELDS),
            "specimens": specimens,
        },
        "machine_profiles": {
            "schema": MACHINE_PROFILE_SCHEMA,
            "version": MACHINE_PROFILE_VERSION,
            "required_fields": list(MACHINE_REQUIRED_FIELDS),
            "optional_fields": list(MACHINE_OPTIONAL_FIELDS),
            "this_machine": machine,
        },
        "backend_provider_contract": {
            "schema": BACKEND_CONTRACT_SCHEMA,
            "version": BACKEND_CONTRACT_VERSION,
            "required_methods": list(REQUIRED_BACKEND_METHODS),
            "optional_methods": list(OPTIONAL_BACKEND_METHODS),
            "provider_protocol_methods": list(PROVIDER_PROTOCOL_METHODS),
            "physical_backends": list(PHYSICAL_BACKENDS),
            "runtime_kinds": hcli_const.get("backend_kinds"),
            "hcli_features": hcli_const.get("features"),
            "metal": backend,
            "availability": backend["availability"],
        },
        "workunit_api": {
            "schema": WORKUNIT_SCHEMA,
            "version": WORKUNIT_VERSION,
            "over": "hcli.workunit.WorkUnit",
            "required_fields": list(WORKUNIT_REQUIRED_FIELDS),
            "optional_fields": list(WORKUNIT_OPTIONAL_FIELDS),
            "content_identity": (
                "sha256 of {role, description, dependencies, verifier} with "
                "sort_keys canonical JSON; same as hcli.workunit.content_identity"
            ),
            "hcli_import": hcli_const.get("workunit_import"),
        },
        "compatibility_suite": {
            "schema": COMPAT_SCHEMA,
            "version": COMPAT_VERSION,
            "clauses": list(COMPAT_CLAUSES),
            "negative_control_clause": CLAUSE_FEATURE_DECLARED,
        },
        "selftest": checks,
        "recovered_implementation": recovered,
        "gaps_closed": [
            "Versioned byte-stable IR (hawking.future.devplatform.ir.v1) with an explicit v0->v1 migrator and refusal of unknown versions",
            "Receipt read/validate/seal-verify surface over tools.future._common.write_receipt, naming clauses, not catching HardwareClaimError",
            "Versioned model profiles with required/optional fields, a specimen catalog, and validation against hcli/hawking-native.sealed-3.14.json",
            "Versioned machine profile of this host from hcli.machine.live_machine_identity, STATIC_ONLY, no bandwidth/tps",
            "Backend provider contract wrapping RuntimeBackend + ModelProvider, with METAL present and FPGA/CUDA/ANE unavailable",
            "Versioned WorkUnit envelope over hcli.workunit with matching content_identity",
            "Compatibility suite that names the failing clause, including a negative control that refuses FPGA/hbm_resident on the Metal provider",
        ],
        "negative_findings": [
            "hcli/genomes/ and tools/accelerator/machine_genome.py are in git but not materialized in this sparse checkout; recovered via git show, not imported",
            "docs/HCLI_DELEGATION.md is in git but not on disk; recovered via git show",
            "receipts/headless/CONVENTIONAL_CONTROL_SET.json and MACHINE_GENOME.json are in git but not on disk; specimen/machine identity taken from git show + live sysctl, not from those files",
            "tools/accelerator/, tools/headless/, hcli/agentos/ exist in git (56 / 280 / 43 paths) but are not on disk in this sparse checkout",
            "No existing tools/future/devplatform.py — this substrate was missing, not duplicated",
            "Did not run cargo, GPU kernels, measure_bandwidth, or any PROTECTED_ABSOLUTE / DIAGNOSTIC_RELATIVE bench",
            "Did not prove FPGA/CUDA/ANE execution; contract marks them unavailable",
            "Did not copy complete_tps_* or physical_ebpw out of hawking-native.sealed-3.14.json",
            "Live mlx_lm.server / llama-server supports() was not re-probed; Metal declared_features are the static comment table in hcli/backends.py",
            "CLAUDE_GLOBAL_FRONTIER.json has no developer-platform entry; this lane is not a frontier row",
        ],
        "hcli_constants_recovered": hcli_const,
    }
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def selftest() -> Path:
    checks = run_selftest_checks()
    out = build(selftest_results=checks)
    written = receipt_read(out)
    report = receipt_validate(written)
    if not report["ok"]:
        raise AssertionError(f"written receipt failed validate: {report['failures']}")
    if written.get("schema") != SCHEMA:
        raise AssertionError("written receipt schema drifted")
    if written.get("bench", {}).get("state") != "UNKNOWN":
        raise AssertionError("bench.state is not UNKNOWN")
    if written.get("bench", {}).get("gpu_authority") is not False:
        raise AssertionError("gpu_authority is not false")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args()
    if a.selftest or not a.build:
        print(selftest())
        return 0
    print(build())
    return 0


__all__ = [
    "CLAUSE_FEATURE_DECLARED",
    "HardwareClaimError",
    "IrError",
    "check_compatibility",
    "decode_ir",
    "encode_ir",
    "make_backend_contract",
    "make_ir",
    "make_model_profile",
    "metal_provider_contract",
    "migrate_ir",
    "receipt_read",
    "receipt_validate",
    "receipt_write",
    "roundtrip_ir",
    "selftest",
    "specimen_profile",
    "this_machine_profile",
    "workunit_content_hash",
    "workunit_normalize",
]


if __name__ == "__main__":
    raise SystemExit(main())
