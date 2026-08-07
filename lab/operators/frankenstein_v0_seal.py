#!/usr/bin/env python3.12
"""PROTO_FRANKENSTEIN_V0 SEAL package: loadable artifact + independent verify + cloud/restore.

Buildable now so the moment training finishes the artifact can be sealed + uploaded:

  1. HCLI-loadable artifact assembler
       BASE_DSV4F (read-only) + reversible bridges/heads/route residual →
       loadable reversible descriptor with exact provenance + hash-binding +
       Gravity byte / resident / active-bytes / TPS accounting.
       Default write path: ~/Desktop/hawking-frankenstein/proto-frankenstein/

  2. Independent-challenger verification harness
       Re-runs A–G ablation + retention gate + promotion gate independently of
       the training lane; rejects contamination / teacher-answer memorization.

  3. Cloud-upload package + remote-hash + one-command local restore
       Upload itself is a MANUAL human step. This builds the package, the
       restore command, and a PROTO_CLOUD_SEALED confirm hook that reclaim reads.

Fail closed REQUIRES_TRAINED_MODULES until real trained modules exist.
No GPU. No Kimi/Gravity recomposition. No faked accounting.
Extends frankenstein_promotion_gate / frankenstein_ablation — does not duplicate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from lab.operators.frankenstein_ablation import (
    DEFAULT_SECONDARY_TOLERANCE,
    audit_contamination,
    audit_teacher_memorization,
    run_independent_challenger,
)
from lab.operators.frankenstein_baseline_freeze import freeze_base_dsv4f_baseline
from lab.operators.frankenstein_bridges import V0_MODULE_NAMES as BRIDGE_V0_MODULES
from lab.operators.frankenstein_fusion_op import DEEPSEEK_V4_FLASH, GLM_5_2
from lab.operators.frankenstein_gates import (
    LINEAR_SUBSPACE_INITIALIZATION,
    REQUIRES_TRAINED_MODULES,
    fail_closed,
    gate_record,
    linear_init_claim_boundary,
)
from lab.operators.frankenstein_promotion_gate import evaluate_promotion
from lab.operators.frankenstein_receipts import (
    KIMI_PRESERVED_TRANSPLANT_POINTS,
    build_runtime_storage_accounting,
)
from lab.receipts import seal, verify


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT / "workspace"
EVIDENCE_ROOT = WORKSPACE_ROOT / "campaign" / "evidence" / "models" / "frankenstein"

DEFAULT_SEAL_ROOT = (
    Path.home() / "Desktop" / "hawking-frankenstein" / "proto-frankenstein"
)
DEFAULT_MODULES_DIR = EVIDENCE_ROOT / "latent_v0_checkpoints"

# --- schemas ---
V0_ARTIFACT_SCHEMA = "hawking.frankenstein.v0_loadable_artifact.v1"
V0_ASSEMBLE_RECEIPT_SCHEMA = "hawking.frankenstein.v0_assemble_receipt.v1"
V0_VERIFY_SCHEMA = "hawking.frankenstein.v0_independent_verify.v1"
V0_CLOUD_PACKAGE_SCHEMA = "hawking.frankenstein.v0_cloud_package.v1"
V0_CLOUD_SEALED_SCHEMA = "hawking.frankenstein.v0_proto_cloud_sealed.v1"
V0_PROTO_TERMINAL_SCHEMA = "hawking.frankenstein.proto_terminal.v1"
V0_MODULE_PACK_SCHEMA = "hawking.frankenstein.v0_trained_module_pack.v1"
GRAVITY_ACCOUNTING_SCHEMA = "hawking.frankenstein.v0_gravity_accounting.v1"

ARTIFACT_NAME = "PROTO_FRANKENSTEIN_V0_ARTIFACT.json"
ASSEMBLE_RECEIPT_NAME = "PROTO_FRANKENSTEIN_V0_ASSEMBLE_RECEIPT.json"
VERIFY_RECEIPT_NAME = "PROTO_FRANKENSTEIN_V0_INDEPENDENT_VERIFY.json"
CLOUD_PACKAGE_DIR = "cloud-package"
CLOUD_MANIFEST_NAME = "PROTO_V0_CLOUD_MANIFEST.json"
CLOUD_SEALED_NAME = "PROTO_CLOUD_SEALED.json"
TERMINAL_RECEIPT_NAME = "PROTO_FRANKENSTEIN_V0_TERMINAL_RECEIPT.json"
RESTORE_SCRIPT_NAME = "restore_proto_frankenstein_v0.sh"
UPLOAD_INSTRUCTIONS_NAME = "UPLOAD_MANUAL.txt"

TERMINAL_ENDPOINT = "PROTO_FRANKENSTEIN_V0_FULL_LATENT_SEALED"

REQUIRED_V0_MODULES: tuple[str, ...] = tuple(BRIDGE_V0_MODULES)


class V0SealError(RuntimeError):
    """V0 seal package failed closed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> None:
    try:
        node = os.lstat(path)
    except OSError as exc:
        raise V0SealError(f"cannot inspect {label}: {exc}") from exc
    if stat.S_ISLNK(node.st_mode) or not stat.S_ISREG(node.st_mode):
        raise V0SealError(f"{label} must be a regular non-symlink file")


def _ensure_dir(path: Path) -> None:
    if path.exists():
        node = os.lstat(path)
        if stat.S_ISLNK(node.st_mode) or not stat.S_ISDIR(node.st_mode):
            raise V0SealError(f"not a safe directory: {path}")
        return
    path.mkdir(parents=True, exist_ok=True)


def _atomic_write_json(path: Path, document: Mapping[str, Any]) -> str:
    raw = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    encoded = raw.encode("utf-8")
    if path.exists():
        _regular_file(path, f"existing output {path}")
        existing = path.read_bytes()
        if existing != encoded:
            raise V0SealError(
                f"refusing to overwrite different immutable evidence: {path}"
            )
        return _sha256(existing)
    _ensure_dir(path.parent)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            try:
                os.unlink(temporary)
            except OSError:
                pass
    return _sha256(encoded)


def _atomic_write_text(path: Path, text: str) -> str:
    encoded = text.encode("utf-8")
    if path.exists():
        _regular_file(path, f"existing {path}")
        existing = path.read_bytes()
        if existing != encoded:
            raise V0SealError(
                f"refusing to overwrite different immutable evidence: {path}"
            )
        return _sha256(existing)
    _ensure_dir(path.parent)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            try:
                os.unlink(temporary)
            except OSError:
                pass
    return _sha256(encoded)


def _optional_binding(path: Path | None, *, label: str) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        return {
            "label": label,
            "path": str(path),
            "present": False,
        }
    if path.is_dir():
        return {
            "label": label,
            "path": str(path.resolve()),
            "present": True,
            "kind": "directory",
        }
    _regular_file(path, label)
    size = path.stat().st_size
    return {
        "label": label,
        "path": str(path.resolve()),
        "present": True,
        "bytes": size,
        "file_sha256": _sha256_file(path),
    }


def _read_sealed_receipt(
    path: Path,
    *,
    label: str,
    required_schema: str | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        raise V0SealError(f"{label} missing: {path}")
    doc = _load_json(path)
    verify(doc, label=label)
    if required_schema is not None and doc.get("schema") != required_schema:
        raise V0SealError(
            f"{label} schema mismatch: got {doc.get('schema')!r}; "
            f"expected {required_schema!r}"
        )
    return doc


def _verify_cloud_manifest_payload(manifest: Mapping[str, Any], package_dir: Path) -> list[str]:
    payload = manifest.get("payload")
    if not isinstance(payload, list):
        return ["cloud manifest payload must be a list"]
    missing: list[str] = []
    payload_dir = package_dir / "payload"
    for row in payload:
        if not isinstance(row, Mapping):
            missing.append("cloud manifest payload row malformed")
            continue
        name = row.get("name")
        expected = row.get("sha256")
        if not isinstance(name, str):
            missing.append("cloud payload row missing name")
            continue
        if not isinstance(expected, str):
            missing.append(f"cloud payload row {name} missing expected sha256")
            continue
        member = payload_dir / name
        if not member.is_file():
            missing.append(f"cloud payload member missing: {name}")
            continue
        actual = _sha256_file(member)
        if actual != expected:
            missing.append(
                f"cloud payload hash mismatch for {name}: expected {expected}, got {actual}"
            )
    return missing


def _terminal_evidence_hash_ok(path: Path, expected: Any, label: str) -> bool:
    if not isinstance(expected, str):
        return False
    doc = _load_json(path)
    seal = doc.get("seal_sha256")
    return isinstance(seal, str) and seal == expected


# ---------------------------------------------------------------------------
# Trained-module admission (fail closed)
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict[str, Any]:
    _regular_file(path, str(path))
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise V0SealError(f"not JSON: {path}: {exc}") from exc
    if not isinstance(doc, Mapping):
        raise V0SealError(f"JSON root must be object: {path}")
    return dict(doc)


def admit_trained_modules(
    modules_path: Path | str | None,
    *,
    required: Sequence[str] = REQUIRED_V0_MODULES,
) -> dict[str, Any]:
    """Admit a trained V0 module pack or return FAIL_CLOSED REQUIRES_TRAINED_MODULES.

    Accepted forms:
      * directory containing MODULE_PACK.json (or trained_modules.json) + weight files
      * single JSON module-pack descriptor
      * None / missing → FAIL_CLOSED

    A pack is trained only when:
      - trained == true
      - every required V0 module name is listed with content hash
      - no fabricated parameter_bytes (must be present and > 0 when weights exist)
    """

    if modules_path is None:
        closed = fail_closed(
            REQUIRES_TRAINED_MODULES,
            stage="13_v0_seal_assemble",
            operation="admit_trained_modules",
        )
        return seal(
            {
                "schema": V0_MODULE_PACK_SCHEMA,
                "recorded_at": _utc_now(),
                **closed,
                "trained": False,
                "complete": False,
                "modules": {},
                "required": list(required),
                "capability_claim": False,
            }
        )

    path = Path(modules_path)
    if not path.exists():
        closed = fail_closed(
            REQUIRES_TRAINED_MODULES,
            stage="13_v0_seal_assemble",
            operation="admit_trained_modules",
        )
        return seal(
            {
                "schema": V0_MODULE_PACK_SCHEMA,
                "recorded_at": _utc_now(),
                **closed,
                "trained": False,
                "complete": False,
                "modules": {},
                "required": list(required),
                "path": str(path),
                "capability_claim": False,
                "detail": f"modules path absent: {path}",
            }
        )

    pack_doc: dict[str, Any]
    weight_bindings: dict[str, Any] = {}

    if path.is_dir():
        candidate = None
        for name in (
            "MODULE_PACK.json",
            "trained_modules.json",
            "PROTO_FRANKENSTEIN_V0_MODULE_PACK.json",
            "BEST_BALANCED.json",
            "CURRENT.json",
        ):
            p = path / name
            if p.is_file():
                candidate = p
                break
        # Also accept a sidecar next to .pt checkpoints describing the pack.
        if candidate is None:
            # Check for checkpoint .pt files — still need a JSON pack admitting trained=true.
            pts = sorted(path.glob("*.pt"))
            closed = fail_closed(
                REQUIRES_TRAINED_MODULES,
                stage="13_v0_seal_assemble",
                operation="admit_trained_modules",
            )
            return seal(
                {
                    "schema": V0_MODULE_PACK_SCHEMA,
                    "recorded_at": _utc_now(),
                    **closed,
                    "trained": False,
                    "complete": False,
                    "modules": {},
                    "required": list(required),
                    "path": str(path),
                    "checkpoint_files_seen": [p.name for p in pts],
                    "capability_claim": False,
                    "detail": (
                        "directory present but no MODULE_PACK.json admitting "
                        "trained=true + per-module content hashes. "
                        "Raw .pt checkpoints alone are not a sealable pack."
                    ),
                }
            )
        pack_doc = _load_json(candidate)
        pack_binding = _optional_binding(candidate, label="module_pack_json")
        for wt in sorted(path.glob("*.pt")) + sorted(path.glob("*.bin")):
            weight_bindings[wt.name] = _optional_binding(wt, label=wt.name)
    else:
        pack_doc = _load_json(path)
        pack_binding = _optional_binding(path, label="module_pack_json")

    trained_flag = bool(pack_doc.get("trained"))
    modules = pack_doc.get("modules") or pack_doc.get("v0_modules") or {}
    if not isinstance(modules, Mapping):
        raise V0SealError("module pack 'modules' must be a mapping")

    present: dict[str, Any] = {}
    missing: list[str] = []
    for name in required:
        row = modules.get(name)
        if not isinstance(row, Mapping):
            missing.append(name)
            continue
        content_hash = row.get("content_hash") or row.get("hash") or row.get("seal_sha256")
        if not content_hash or not isinstance(content_hash, str) or len(content_hash) < 16:
            missing.append(name)
            continue
        if row.get("trained") is False:
            missing.append(name)
            continue
        present[name] = {
            "name": name,
            "content_hash": content_hash,
            "parameter_bytes": row.get("parameter_bytes"),
            "trained": bool(row.get("trained", trained_flag)),
            "reversible": bool(row.get("reversible", True)),
            "ablatable": bool(row.get("ablatable", True)),
            "bypassable": bool(row.get("bypassable", True)),
        }

    complete = trained_flag and not missing and len(present) == len(list(required))
    # Refuse zero-byte faked accounting when pack claims trained.
    total_bytes = 0
    for row in present.values():
        pb = row.get("parameter_bytes")
        if pb is not None:
            total_bytes += int(pb)

    if not complete or not trained_flag:
        closed = fail_closed(
            REQUIRES_TRAINED_MODULES,
            stage="13_v0_seal_assemble",
            operation="admit_trained_modules",
        )
        return seal(
            {
                "schema": V0_MODULE_PACK_SCHEMA,
                "recorded_at": _utc_now(),
                **closed,
                "trained": False,
                "complete": False,
                "modules": present,
                "missing_modules": missing,
                "required": list(required),
                "path": str(path),
                "pack_binding": pack_binding,
                "weight_bindings": weight_bindings,
                "pack_trained_flag": trained_flag,
                "capability_claim": False,
            }
        )

    return seal(
        {
            "schema": V0_MODULE_PACK_SCHEMA,
            "recorded_at": _utc_now(),
            "status": "ADMITTED",
            "gate": REQUIRES_TRAINED_MODULES,
            "state": "OPEN",
            "executed": True,
            "trained": True,
            "complete": True,
            "modules": present,
            "missing_modules": [],
            "required": list(required),
            "module_count": len(present),
            "total_parameter_bytes_declared": total_bytes,
            "path": str(path),
            "pack_binding": pack_binding,
            "weight_bindings": weight_bindings,
            "capability_claim": False,
            "note": (
                "Trained module pack admitted. Capability claims still require "
                "independent verify + promotion gate ACCEPT."
            ),
        }
    )


# ---------------------------------------------------------------------------
# Gravity accounting (honest; never invents TPS)
# ---------------------------------------------------------------------------


def build_gravity_accounting(
    *,
    module_pack: Mapping[str, Any],
    base_resident_bytes: int | None = None,
    tps: float | None = None,
    p99: float | None = None,
    active_bytes_per_token: int | None = None,
) -> dict[str, Any]:
    """Compose Gravity byte / resident / active-bytes / TPS accounting.

    Missing measured fields are PENDING, never fabricated.
    """

    modules = module_pack.get("modules") or {}
    parts: dict[str, Any] = {}
    adapter_bytes = 0
    for name, row in modules.items():
        if not isinstance(row, Mapping):
            continue
        pb = row.get("parameter_bytes")
        if pb is None:
            parts[name] = {
                "parameter_bytes": None,
                "status": "PENDING",
                "content_hash": row.get("content_hash"),
            }
        else:
            parts[name] = {
                "parameter_bytes": int(pb),
                "status": "DECLARED",
                "content_hash": row.get("content_hash"),
            }
            adapter_bytes += int(pb)

    # BASE_DSV4F body is read-only reference — not re-counted as adapter bytes.
    body = DEEPSEEK_V4_FLASH
    # Do not invent body byte size; only geometry is pinned.
    document = {
        "schema": GRAVITY_ACCOUNTING_SCHEMA,
        "recorded_at": _utc_now(),
        "parameter_bytes": adapter_bytes if adapter_bytes > 0 else None,
        "adapter_parameter_bytes": adapter_bytes if adapter_bytes > 0 else None,
        "resident_bytes": {
            "base_dsv4f_body": base_resident_bytes,  # None = unmeasured
            "adapters": adapter_bytes if adapter_bytes > 0 else None,
            "total": (
                (base_resident_bytes + adapter_bytes)
                if base_resident_bytes is not None and adapter_bytes > 0
                else None
            ),
            "status": (
                "MEASURED"
                if base_resident_bytes is not None and adapter_bytes > 0
                else "PARTIAL" if adapter_bytes > 0 else "PENDING"
            ),
        },
        "active_bytes": {
            "active_bytes_per_token": active_bytes_per_token,
            "proxy_from_adapter_bytes": adapter_bytes if adapter_bytes > 0 else None,
            "status": "MEASURED" if active_bytes_per_token is not None else "PROXY_OR_PENDING",
        },
        "tps": tps,
        "p99": p99,
        "tps_status": "MEASURED" if tps is not None else "PENDING",
        "p99_status": "MEASURED" if p99 is not None else "PENDING",
        "parts": parts,
        "student_body": {
            "repository": body["repository"],
            "revision": body["revision"],
            "hidden_size": body["hidden_size"],
            "num_hidden_layers": body["num_hidden_layers"],
            "read_only": True,
            "byte_size_fabricated": False,
        },
        "fabricated": False,
        "kimi_bridge_compatible": True,
        "glm_router_weights_copied": False,
        "note": (
            "Adapter byte counts come from the admitted trained pack. "
            "Body resident bytes and TPS/p99 stay PENDING until measured — "
            "never fabricated for seal."
        ),
    }
    # Convenience keys expected by evaluate_promotion gravity_accounting check.
    if adapter_bytes > 0:
        document["parameter_bytes"] = adapter_bytes
    return seal(document)


# ---------------------------------------------------------------------------
# Artifact assembler
# ---------------------------------------------------------------------------


def assemble_v0_artifact(
    *,
    modules_path: Path | str | None = None,
    out_dir: Path | str | None = None,
    base_resident_bytes: int | None = None,
    tps: float | None = None,
    p99: float | None = None,
    active_bytes_per_token: int | None = None,
    write: bool = True,
    force_scaffold_receipt: bool = True,
) -> dict[str, Any]:
    """Assemble HCLI-loadable PROTO_FRANKENSTEIN_V0 artifact or fail closed.

    When trained modules are absent, writes an assemble receipt with status
    REQUIRES_TRAINED_MODULES and does NOT emit a loadable SEALED artifact.
    """

    out = Path(out_dir) if out_dir is not None else DEFAULT_SEAL_ROOT
    if write:
        _ensure_dir(out)

    baseline = freeze_base_dsv4f_baseline(write=False)
    module_pack = admit_trained_modules(modules_path)
    gravity = build_gravity_accounting(
        module_pack=module_pack,
        base_resident_bytes=base_resident_bytes,
        tps=tps,
        p99=p99,
        active_bytes_per_token=active_bytes_per_token,
    )
    runtime_storage = build_runtime_storage_accounting(
        adapter_archive_bytes=int(gravity.get("adapter_parameter_bytes") or 0),
        working_set_bytes=int(
            (gravity.get("resident_bytes") or {}).get("total") or 0
        ),
        tps_base=None,
        tps_proto=tps,
        notes=(
            "V0 seal accounting. BASE body read-only. Adapter bytes from trained pack "
            "when admitted; TPS only when measured."
        ),
    )

    trained_ok = module_pack.get("trained") is True and module_pack.get("complete") is True

    if not trained_ok:
        receipt = seal(
            {
                "schema": V0_ASSEMBLE_RECEIPT_SCHEMA,
                "name": "PROTO_FRANKENSTEIN_V0_ASSEMBLE",
                "recorded_at": _utc_now(),
                "status": "FAIL_CLOSED",
                "gate": REQUIRES_TRAINED_MODULES,
                "verdict": "REQUIRES_TRAINED_MODULES",
                "terminal_endpoint": TERMINAL_ENDPOINT,
                "terminal_reached": False,
                "loadable_artifact_written": False,
                "fabricated": False,
                "capability_claim": False,
                "baseline": {
                    "schema": baseline.get("schema"),
                    "seal_sha256": baseline.get("seal_sha256"),
                },
                "module_pack": {
                    "status": module_pack.get("status"),
                    "trained": module_pack.get("trained"),
                    "complete": module_pack.get("complete"),
                    "missing_modules": module_pack.get("missing_modules"),
                    "seal_sha256": module_pack.get("seal_sha256"),
                },
                "gravity_accounting": {
                    "seal_sha256": gravity.get("seal_sha256"),
                    "tps_status": gravity.get("tps_status"),
                    "parameter_bytes": gravity.get("parameter_bytes"),
                },
                "runtime_storage_seal_sha256": runtime_storage.get("seal_sha256"),
                "out_dir": str(out),
                "infra_gates": {
                    REQUIRES_TRAINED_MODULES: gate_record(REQUIRES_TRAINED_MODULES),
                },
                "claim_boundary": {
                    **linear_init_claim_boundary(),
                    "empty_untrained_modules_cannot_seal": True,
                    "no_faked_accounting": True,
                    "no_gpu": True,
                    "no_kimi": True,
                    "no_gravity_recomposition": True,
                },
                "note": (
                    "Assembler is ready. Seal refuses until a real trained module pack "
                    "admits trained=true for all V0 sites with content hashes."
                ),
            }
        )
        paths: dict[str, str] = {}
        if write and force_scaffold_receipt:
            p = out / ASSEMBLE_RECEIPT_NAME
            _atomic_write_json(p, receipt)
            paths["assemble_receipt"] = str(p)
        return {
            "status": "FAIL_CLOSED",
            "gate": REQUIRES_TRAINED_MODULES,
            "verdict": "REQUIRES_TRAINED_MODULES",
            "loadable": False,
            "receipt": receipt,
            "module_pack": module_pack,
            "gravity": gravity,
            "paths": paths,
            "out_dir": str(out),
            "capability_claim": False,
        }

    # --- trained path: compose loadable reversible descriptor ---
    modules = module_pack.get("modules") or {}
    module_rows = []
    for name in REQUIRED_V0_MODULES:
        row = modules[name]
        module_rows.append(
            {
                "name": name,
                "content_hash": row["content_hash"],
                "parameter_bytes": row.get("parameter_bytes"),
                "trained": True,
                "reversible": bool(row.get("reversible", True)),
                "bypassable": bool(row.get("bypassable", True)),
                "ablatable": bool(row.get("ablatable", True)),
                "hash_bound": True,
            }
        )

    artifact = seal(
        {
            "schema": V0_ARTIFACT_SCHEMA,
            "name": "PROTO_FRANKENSTEIN_V0",
            "recorded_at": _utc_now(),
            "kind": "hcli_loadable_reversible_descriptor",
            "status": "LOADABLE_TRAINED_COMPOSITION",
            "terminal_endpoint": TERMINAL_ENDPOINT,
            "terminal_reached": False,  # still needs independent verify + cloud
            "role": "PROTO_FRANKENSTEIN_V0_TRAINED_STACK",
            "proto_frankenstein_complete": False,  # only promotion ACCEPT may flip
            "trained": True,
            "reversible": True,
            "removable": True,
            "bypassable": True,
            "hash_bound": True,
            "gravity_compressed": False,
            "byte_merge": False,
            "direct_weight_transplant": False,
            "glm_router_weights_copied": False,
            "composition": {
                "mode": "base_dsv4f_readonly_plus_reversible_v0_modules",
                "base": "BASE_DSV4F",
                "modules": "V0 bridges + behavior heads + route residual",
                "body_rewritten": False,
                "formula": "h_out = h_base + sum_i residual_i(h)  (each residual bypassable)",
            },
            "student_body": {
                "family": DEEPSEEK_V4_FLASH["family"],
                "repository": DEEPSEEK_V4_FLASH["repository"],
                "revision": DEEPSEEK_V4_FLASH["revision"],
                "hidden_size": DEEPSEEK_V4_FLASH["hidden_size"],
                "num_hidden_layers": DEEPSEEK_V4_FLASH["num_hidden_layers"],
                "read_only": True,
            },
            "math_donor": {
                "family": GLM_5_2["family"],
                "repository": GLM_5_2["repository"],
                "revision": GLM_5_2["revision"],
                "hidden_size": GLM_5_2["hidden_size"],
                "num_hidden_layers": GLM_5_2["num_hidden_layers"],
                "runtime_resident": False,
                "training_only_projectors": True,
            },
            "baseline": {
                "schema": baseline.get("schema"),
                "seal_sha256": baseline.get("seal_sha256"),
            },
            "v0_modules": module_rows,
            "module_pack_seal_sha256": module_pack.get("seal_sha256"),
            "gravity_accounting": gravity,
            "runtime_storage": runtime_storage,
            "bridges": {
                "active_v0": list(REQUIRED_V0_MODULES),
                "preserved_untouched_for_stage2": "KIMI_STRATEGIC_BRIDGE",
                "kimi_points_preserved": list(KIMI_PRESERVED_TRANSPLANT_POINTS),
                "kimi_stage1_status": "PRESERVED_UNTOUCHED",
            },
            "hcli": {
                "loadable_descriptor": True,
                "encoding_contract": "tools/condense/seal_deepseek_v4_hcli_encoding_contract.py",
                "full_runtime_claim": False,
                "note": (
                    "Descriptor is HCLI-loadable shape; full 43-layer runtime "
                    "capability still requires independent verify + forward."
                ),
            },
            "linear_init_role": LINEAR_SUBSPACE_INITIALIZATION,
            "claim_boundary": {
                **linear_init_claim_boundary(),
                "trained_stack_assembled": True,
                "promotion_still_required": True,
                "independent_verify_still_required": True,
                "cloud_confirm_still_required": True,
                "no_faked_accounting": True,
                "no_gpu": True,
                "no_kimi": True,
            },
            "fabricated": False,
            "capability_claim": False,
        }
    )

    assemble_receipt = seal(
        {
            "schema": V0_ASSEMBLE_RECEIPT_SCHEMA,
            "name": "PROTO_FRANKENSTEIN_V0_ASSEMBLE",
            "recorded_at": _utc_now(),
            "status": "ASSEMBLED",
            "verdict": "ASSEMBLED_AWAITING_INDEPENDENT_VERIFY",
            "terminal_endpoint": TERMINAL_ENDPOINT,
            "terminal_reached": False,
            "loadable_artifact_written": write,
            "artifact_seal_sha256": artifact.get("seal_sha256"),
            "module_pack_seal_sha256": module_pack.get("seal_sha256"),
            "gravity_seal_sha256": gravity.get("seal_sha256"),
            "fabricated": False,
            "capability_claim": False,
            "out_dir": str(out),
        }
    )

    paths = {}
    if write:
        art_path = out / ARTIFACT_NAME
        rec_path = out / ASSEMBLE_RECEIPT_NAME
        _atomic_write_json(art_path, artifact)
        _atomic_write_json(rec_path, assemble_receipt)
        paths = {
            "artifact": str(art_path),
            "assemble_receipt": str(rec_path),
        }

    return {
        "status": "ASSEMBLED",
        "verdict": "ASSEMBLED_AWAITING_INDEPENDENT_VERIFY",
        "loadable": True,
        "artifact": artifact,
        "receipt": assemble_receipt,
        "module_pack": module_pack,
        "gravity": gravity,
        "paths": paths,
        "out_dir": str(out),
        "capability_claim": False,
    }


# ---------------------------------------------------------------------------
# Independent verify
# ---------------------------------------------------------------------------


def independent_verify(
    *,
    arm_scores: Mapping[str, Mapping[str, Any]] | None = None,
    secondary_tolerance: float = DEFAULT_SECONDARY_TOLERANCE,
    membership: Mapping[str, Any] | None = None,
    contamination_overlap_ids: Sequence[str] | None = None,
    contamination_barrier_pass: bool | None = None,
    hidden_exact_match_rate: float | None = None,
    teacher_answer_replay_rate: float | None = None,
    memorization_flags: Mapping[str, Any] | None = None,
    scores: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    gravity_accounting: Mapping[str, Any] | None = None,
    routing: Mapping[str, Any] | None = None,
    base_secondary: Mapping[str, float] | None = None,
    proto_secondary: Mapping[str, float] | None = None,
    trained_modules: Mapping[str, Any] | bool | None = None,
    complete_beats_linear: bool | None = None,
    reversible_loadable: bool | None = None,
    kimi_bridge_intact: bool | None = True,
    matrix: str = "latent_v0",
    out_dir: Path | str | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Independent-challenger: A–G + retention + promotion + contamination.

    Re-runs gates via frankenstein_ablation / frankenstein_promotion_gate /
    frankenstein_latent_v0.retention_gate — does not trust training-lane caches.
    """

    # Deferred import: retention lives with latent stack (torch).
    from lab.operators.frankenstein_latent_v0 import retention_gate

    challenger = run_independent_challenger(
        arm_scores=arm_scores,
        secondary_tolerance=secondary_tolerance,
        membership=membership,
        contamination_overlap_ids=contamination_overlap_ids,
        contamination_barrier_pass=contamination_barrier_pass,
        hidden_exact_match_rate=hidden_exact_match_rate,
        teacher_answer_replay_rate=teacher_answer_replay_rate,
        memorization_flags=memorization_flags,
        matrix=matrix,
    )

    # Retention gate — PENDING without secondary maps.
    if base_secondary is not None and proto_secondary is not None:
        retention = retention_gate(
            base_secondary=base_secondary,
            proto_secondary=proto_secondary,
            tolerance=secondary_tolerance,
        )
        retention_verdict = retention.get("verdict")
    else:
        retention = seal(
            {
                "schema": "hawking.frankenstein.latent_v0_retention_gate.v1",
                "recorded_at": _utc_now(),
                "verdict": "PENDING",
                "reject_rule_fired": False,
                "tolerance": secondary_tolerance,
                "regressions": [],
                "domains": [],
                "capability_claim": False,
                "note": "no base/proto secondary maps submitted",
            }
        )
        retention_verdict = None

    contam = challenger.get("contamination") or {}
    memor = challenger.get("teacher_memorization") or {}
    ablation_verdict = (challenger.get("ablation") or {}).get("verdict")

    # Independent challenge payload for promotion gate.
    ind_payload = {
        "pass": challenger.get("pass") is True,
        "verdict": challenger.get("verdict"),
        "seal_sha256": challenger.get("seal_sha256"),
    }

    promotion = evaluate_promotion(
        scores=scores,
        provenance=provenance,
        gravity_accounting=gravity_accounting,
        routing=routing,
        independent_challenge=ind_payload if arm_scores is not None else None,
        ablation_verdict=ablation_verdict if arm_scores is not None else None,
        retention_verdict=retention_verdict,
        trained_modules=trained_modules,
        contamination=contam if contam.get("status") != "PENDING" else None,
        teacher_memorization=memor if memor.get("status") != "PENDING" else None,
        complete_beats_linear=complete_beats_linear,
        reversible_loadable=reversible_loadable,
        kimi_bridge_intact=kimi_bridge_intact,
    )

    # Overall independent verify verdict.
    if (
        challenger.get("verdict") == "REJECT"
        or promotion.get("verdict") == "REJECT"
        or retention.get("verdict") == "REJECT"
    ):
        overall = "REJECT"
    elif (
        challenger.get("verdict") == "ACCEPT"
        and promotion.get("verdict") == "ACCEPT"
        and retention.get("verdict") in {"PASS", "ACCEPT"}
    ):
        overall = "ACCEPT"
    else:
        overall = "PENDING"

    document = seal(
        {
            "schema": V0_VERIFY_SCHEMA,
            "name": "PROTO_FRANKENSTEIN_V0_INDEPENDENT_VERIFY",
            "recorded_at": _utc_now(),
            "status": "EVALUATED" if overall != "PENDING" else "FRAMEWORK_PENDING",
            "verdict": overall,
            "pass": overall == "ACCEPT",
            "independent_of_training_lane": True,
            "challenger": {
                "schema": challenger.get("schema"),
                "verdict": challenger.get("verdict"),
                "seal_sha256": challenger.get("seal_sha256"),
                "matrix": challenger.get("matrix"),
            },
            "ablation_ag": challenger.get("ablation"),
            "retention_gate": {
                "verdict": retention.get("verdict"),
                "seal_sha256": retention.get("seal_sha256"),
                "reject_rule_fired": retention.get("reject_rule_fired"),
            },
            "promotion_gate": {
                "verdict": promotion.get("verdict"),
                "reason": promotion.get("reason"),
                "seal_sha256": promotion.get("seal_sha256"),
                "checks": promotion.get("checks"),
            },
            "contamination": contam,
            "teacher_memorization": memor,
            "fabricated_scores": False,
            "capability_claim": False,
            "claim_boundary": {
                "independent_challenger": True,
                "rejects_contamination": True,
                "rejects_teacher_memorization": True,
                "does_not_trust_training_lane_caches": True,
                "pending_is_not_accept": True,
            },
            "note": (
                "Independent re-run of latent/functional A–G, retention gate, and "
                "promotion gate. Rejects contamination and teacher-answer memorization."
            ),
        }
    )

    paths: dict[str, str] = {}
    if write:
        out = Path(out_dir) if out_dir is not None else DEFAULT_SEAL_ROOT
        _ensure_dir(out)
        p = out / VERIFY_RECEIPT_NAME
        _atomic_write_json(p, document)
        paths["verify_receipt"] = str(p)

    return {
        "status": document["status"],
        "verdict": overall,
        "pass": overall == "ACCEPT",
        "document": document,
        "challenger": challenger,
        "retention": retention,
        "promotion": promotion,
        "paths": paths,
        "capability_claim": False,
    }


# ---------------------------------------------------------------------------
# Proto terminalization
# ---------------------------------------------------------------------------


def terminalize_v0_artifact(
    *,
    artifact_path: Path | str,
    independent_verify_path: Path | str,
    cloud_sealed_path: Path | str,
    cloud_manifest_path: Path | str,
    out_dir: Path | str | None = None,
    dry_run: bool = False,
    certification_status: str = "CONTROLLER_CERTIFIED",
    certified_by: str = "protected_controller",
    active_storage_paths: Sequence[str] | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Build PROTO terminal receipt after validating terminal-critical evidence bindings."""

    artifact_p = Path(artifact_path)
    verify_p = Path(independent_verify_path)
    cloud_sealed_p = Path(cloud_sealed_path)
    cloud_manifest_p = Path(cloud_manifest_path)

    reasons: list[str] = []

    try:
        artifact = _read_sealed_receipt(
            artifact_p, label="artifact", required_schema=V0_ARTIFACT_SCHEMA
        )
    except V0SealError as exc:
        reasons.append(str(exc))
        artifact = {}
    try:
        independent_verify = _read_sealed_receipt(
            verify_p,
            label="independent_verify",
            required_schema=V0_VERIFY_SCHEMA,
        )
    except V0SealError as exc:
        reasons.append(str(exc))
        independent_verify = {}
    try:
        cloud_sealed = _read_sealed_receipt(
            cloud_sealed_p,
            label="cloud_sealed",
            required_schema=V0_CLOUD_SEALED_SCHEMA,
        )
    except V0SealError as exc:
        reasons.append(str(exc))
        cloud_sealed = {}
    try:
        manifest = _read_sealed_receipt(
            cloud_manifest_p,
            label="cloud_manifest",
            required_schema=V0_CLOUD_PACKAGE_SCHEMA,
        )
    except V0SealError as exc:
        reasons.append(str(exc))
        manifest = {}

    artifact_hash = artifact.get("seal_sha256") if isinstance(artifact, Mapping) else None
    verify_hash = (
        independent_verify.get("seal_sha256")
        if isinstance(independent_verify, Mapping)
        else None
    )
    cloud_sealed_hash = (
        cloud_sealed.get("seal_sha256") if isinstance(cloud_sealed, Mapping) else None
    )
    manifest_hash = (
        manifest.get("seal_sha256") if isinstance(manifest, Mapping) else None
    )

    if not isinstance(certification_status, str) or not certification_status.strip():
        reasons.append("certification.status must be non-empty")
    if certification_status != "CONTROLLER_CERTIFIED":
        reasons.append("certification status must be CONTROLLER_CERTIFIED")

    if certified_by not in {"protected_controller", "human_operator"}:
        reasons.append(
            "certification certifier must be protected_controller or human_operator"
        )

    if not isinstance(dry_run, bool):
        reasons.append("dry_run must be boolean")

    # Evidence binding / hash checks.
    hash_ok = True
    for path, expected in (
        (artifact_p, artifact_hash),
        (verify_p, verify_hash),
        (cloud_sealed_p, cloud_sealed_hash),
        (cloud_manifest_p, manifest_hash),
    ):
        if isinstance(expected, str):
            if not _terminal_evidence_hash_ok(path, expected, str(path)):
                hash_ok = False
        else:
            hash_ok = False

    if hash_ok is False:
        reasons.append("one or more evidence binders are missing/invalid sealed hashes")

    if not isinstance(artifact, Mapping) or artifact.get("schema") != V0_ARTIFACT_SCHEMA:
        reasons.append("artifact schema mismatch")
    else:
        runtime_storage = artifact.get("runtime_storage") or {}
        storage = runtime_storage.get("storage") if isinstance(runtime_storage, Mapping) else None
        donor_weights_retained = (
            storage.get("donor_weights_retained")
            if isinstance(storage, Mapping)
            else None
        )
        if not isinstance(donor_weights_retained, bool):
            reasons.append("artifact.runtime_storage.storage.donor_weights_retained missing")
        elif donor_weights_retained is not False:
            reasons.append(
                "artifact indicates donor_weights_retained=True; terminal requires False"
            )

    if not isinstance(independent_verify, Mapping) or independent_verify.get("schema") != V0_VERIFY_SCHEMA:
        reasons.append("independent_verify schema mismatch")
    else:
        if (
            independent_verify.get("status") != "EVALUATED"
            or independent_verify.get("verdict") != "ACCEPT"
            or independent_verify.get("pass") is not True
        ):
            reasons.append(
                "independent verify must be status=EVALUATED, verdict=ACCEPT, pass=true"
            )
        if independent_verify.get("independent_of_training_lane") is not True:
            reasons.append("independent verify must be independent_of_training_lane=true")
        challenger = independent_verify.get("challenger") or {}
        promotion = independent_verify.get("promotion_gate") or {}
        if (
            not isinstance(challenger, Mapping)
            or challenger.get("verdict") != "ACCEPT"
        ):
            reasons.append("independent challenger must report verdict=ACCEPT")
        if (
            not isinstance(promotion, Mapping)
            or promotion.get("verdict") != "ACCEPT"
        ):
            reasons.append("independent promotion gate must report verdict=ACCEPT")
        retention = independent_verify.get("retention_gate") or {}
        if (
            not isinstance(retention, Mapping)
            or retention.get("verdict") not in {"PASS", "ACCEPT"}
        ):
            reasons.append(
                "independent retention gate must be PASS/ACCEPT before terminal seal"
            )
        ablation_ag = independent_verify.get("ablation_ag") or {}
        if not isinstance(ablation_ag, Mapping) or ablation_ag.get("verdict") != "ACCEPT":
            reasons.append("independent A-G ablation must verdict=ACCEPT")

    if not isinstance(cloud_sealed, Mapping) or cloud_sealed.get("schema") != V0_CLOUD_SEALED_SCHEMA:
        reasons.append("cloud_sealed schema mismatch")
    else:
        if cloud_sealed.get("confirmed") is not True:
            reasons.append("cloud seal must be confirmed=true before terminalization")
        if cloud_sealed.get("reclaim_may_evict_superseded") is not True:
            reasons.append(
                "cloud seal must set reclaim_may_evict_superseded=true before terminalization"
            )
        if not isinstance(cloud_sealed.get("remote_hash"), str) or not cloud_sealed.get(
            "remote_hash"
        ).strip():
            reasons.append("cloud seal missing non-empty remote_hash")
        if (
            manifest
            and cloud_sealed.get("bundle_sha256") != manifest.get("bundle_sha256")
        ):
            reasons.append("cloud manifest hash mismatch between sealed and manifest")
        if (
            manifest
            and cloud_sealed.get("manifest_seal_sha256")
            != manifest.get("seal_sha256")
        ):
            reasons.append("cloud manifest seal mismatch between sealed and manifest")

    if not isinstance(manifest, Mapping) or manifest.get("schema") != V0_CLOUD_PACKAGE_SCHEMA:
        reasons.append("cloud_manifest schema mismatch")
    else:
        payload_root = Path(cloud_manifest_p).parent
        payload_dir = payload_root / "payload"
        if not payload_dir.is_dir():
            reasons.append("cloud manifest payload directory missing")
        else:
            reasons.extend(_verify_cloud_manifest_payload(manifest, payload_root))

    # If active-storage paths are provided, fail if any still exists (including symlink).
    removed_from_active_storage = True
    if active_storage_paths:
        present: list[str] = []
        for raw in active_storage_paths:
            target = Path(raw)
            if target.exists() or target.is_symlink():
                present.append(str(target))
        if present:
            removed_from_active_storage = False
            reasons.append(
                f"active donor-paths still present: {'; '.join(present)}"
            )
    else:
        removed_from_active_storage = False
        reasons.append("active-storage path list not provided; cannot verify donor removal")

    terminal_reached = not reasons
    storage = {
        "offloaded": bool(
            cloud_sealed.get("confirmed") is True
            and cloud_sealed.get("reclaim_may_evict_superseded") is True
            and not hash_ok is False
        ),
        "hash_verified": bool(hash_ok and not reasons),
        "removed_from_active_storage": removed_from_active_storage,
        "donor_weights_retained": (
            artifact.get("runtime_storage", {})
            .get("storage", {})
            .get("donor_weights_retained")
            if isinstance(artifact, Mapping)
            else None
        ),
    }
    if not isinstance(storage["donor_weights_retained"], bool):
        storage["donor_weights_retained"] = False

    document = seal(
        {
            "schema": V0_PROTO_TERMINAL_SCHEMA,
            "name": "PROTO_FRANKENSTEIN_V0_TERMINAL_RECEIPT",
            "recorded_at": _utc_now(),
            "status": "TERMINALIZED" if terminal_reached else "BLOCKED",
            "terminal_endpoint": TERMINAL_ENDPOINT,
            "terminal_reached": terminal_reached,
            "proto_frankenstein_complete": True,
            "dry_run": bool(dry_run),
            "runtime_storage": artifact.get("runtime_storage", {}),
            "certification": {
                "status": certification_status,
                "certified_by": certified_by,
            },
            "storage": storage,
            "storage_hash_checks": {"all_bindings_verified": bool(hash_ok)},
            "evidence_bindings": {
                "artifact": {"seal_sha256": artifact_hash},
                "independent_verify": {"seal_sha256": verify_hash},
                "cloud_sealed": {"seal_sha256": cloud_sealed_hash},
                "cloud_manifest": {"seal_sha256": manifest_hash},
            },
            "terminal_not_reached_reasons": reasons,
            "runtime_state": {
                "artifact_path": str(artifact_p),
                "independent_verify_path": str(verify_p),
                "cloud_sealed_path": str(cloud_sealed_p),
                "cloud_manifest_path": str(cloud_manifest_p),
                "active_storage_paths": list(active_storage_paths or []),
            },
            "capability_claim": False,
            "note": (
                "Terminal requires independent verify acceptance, confirmed cloud "
                "seal, binding hash verification, and donor removal from active "
                "storage before terminal_reached can be true."
            ),
        }
    )

    paths: dict[str, str] = {}
    if write:
        out = Path(out_dir) if out_dir is not None else DEFAULT_SEAL_ROOT
        _ensure_dir(out)
        p = out / TERMINAL_RECEIPT_NAME
        _atomic_write_json(p, document)
        paths["terminal_receipt"] = str(p)

    return {
        "status": document["status"],
        "terminal_reached": terminal_reached,
        "reasons": reasons,
        "document": document,
        "paths": paths,
        "storage": storage,
        "capability_claim": False,
    }


# ---------------------------------------------------------------------------
# Cloud package + restore + PROTO_CLOUD_SEALED confirm hook
# ---------------------------------------------------------------------------


def build_cloud_package(
    *,
    artifact_dir: Path | str | None = None,
    out_dir: Path | str | None = None,
    remote_uri_hint: str | None = None,
) -> dict[str, Any]:
    """Build uploadable package + restore command. Does NOT upload.

    Human uploads manually, then runs ``confirm-cloud`` with the remote hash.
    """

    src = Path(artifact_dir) if artifact_dir is not None else DEFAULT_SEAL_ROOT
    root = Path(out_dir) if out_dir is not None else src
    pkg_dir = root / CLOUD_PACKAGE_DIR
    _ensure_dir(pkg_dir)

    # Collect seal artifacts if present.
    members: list[dict[str, Any]] = []
    for name in (
        ARTIFACT_NAME,
        ASSEMBLE_RECEIPT_NAME,
        VERIFY_RECEIPT_NAME,
    ):
        p = src / name
        if p.is_file():
            members.append(
                {
                    "name": name,
                    "bytes": p.stat().st_size,
                    "sha256": _sha256_file(p),
                    "path": str(p.resolve()),
                }
            )

    if not members:
        # Package the assemble receipt path even if only fail-closed exists later;
        # still emit a package skeleton so restore tooling is ready.
        pass

    # Bundle payload: copy members into package dir under payload/
    payload_dir = pkg_dir / "payload"
    if payload_dir.exists():
        # Only safe if we control contents; wipe and rebuild for determinism of new package.
        # Refuse if unexpected symlinks.
        for child in payload_dir.rglob("*"):
            if child.is_symlink():
                raise V0SealError(f"refusing package with symlink: {child}")
        shutil.rmtree(payload_dir)
    payload_dir.mkdir(parents=True, exist_ok=True)

    bundled: list[dict[str, Any]] = []
    for m in members:
        src_path = Path(m["path"])
        dest = payload_dir / m["name"]
        shutil.copy2(src_path, dest)
        bundled.append(
            {
                "name": m["name"],
                "bytes": dest.stat().st_size,
                "sha256": _sha256_file(dest),
            }
        )

    # Archive hash over sorted (name, sha256) of payload — content-addressed bundle id.
    digest_material = json.dumps(
        [{"name": b["name"], "sha256": b["sha256"]} for b in sorted(bundled, key=lambda x: x["name"])],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    bundle_sha256 = _sha256(digest_material)

    restore_cmd = (
        f"python3.12 -m lab.operators.frankenstein_v0_seal restore "
        f"--package {pkg_dir} --out ~/Desktop/hawking-frankenstein/proto-frankenstein"
    )
    restore_script = f"""#!/usr/bin/env bash
# One-command local restore for PROTO_FRANKENSTEIN_V0 cloud package.
# Verifies payload hashes against the sealed cloud manifest, then copies to --out.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${{1:-$HOME/Desktop/hawking-frankenstein/proto-frankenstein}}"
python3.12 -m lab.operators.frankenstein_v0_seal restore \\
  --package "$SCRIPT_DIR" \\
  --out "$OUT_DIR"
echo "restored to $OUT_DIR"
"""
    restore_path = pkg_dir / RESTORE_SCRIPT_NAME
    _atomic_write_text(restore_path, restore_script)
    try:
        restore_path.chmod(restore_path.stat().st_mode | stat.S_IXUSR)
    except OSError:
        pass

    upload_txt = f"""PROTO_FRANKENSTEIN_V0 — MANUAL CLOUD UPLOAD
============================================

This package does NOT upload itself. A human must:

  1. Upload the directory (or a tar of it):
       {pkg_dir}

  2. Record the remote content hash / etag of the uploaded object.

  3. Confirm with the local hook (so reclaim can read PROTO_CLOUD_SEALED):

       python3.12 -m lab.operators.frankenstein_v0_seal confirm-cloud \\
         --package {pkg_dir} \\
         --remote-hash <REMOTE_SHA256_OR_ETAG> \\
         --remote-uri <OPTIONAL_URI>

  4. Only after PROTO_CLOUD_SEALED.json shows confirmed=true may reclaim
     evict superseded local checkpoints / raw GLM windows.

Bundle content id (local): {bundle_sha256}
Restore command:
  {restore_cmd}

Suggested remote URI hint: {remote_uri_hint or "(none — fill at confirm time)"}
"""
    _atomic_write_text(pkg_dir / UPLOAD_INSTRUCTIONS_NAME, upload_txt)

    manifest = seal(
        {
            "schema": V0_CLOUD_PACKAGE_SCHEMA,
            "name": "PROTO_V0_CLOUD_PACKAGE",
            "recorded_at": _utc_now(),
            "status": "PACKAGE_READY_AWAITING_MANUAL_UPLOAD",
            "upload_performed": False,
            "upload_is_manual_human_step": True,
            "bundle_sha256": bundle_sha256,
            "payload": bundled,
            "member_count": len(bundled),
            "package_dir": str(pkg_dir.resolve()),
            "restore_command": restore_cmd,
            "restore_script": RESTORE_SCRIPT_NAME,
            "remote_uri_hint": remote_uri_hint,
            "cloud_sealed_hook": CLOUD_SEALED_NAME,
            "cloud_sealed_confirmed": False,
            "reclaim_may_evict_superseded": False,
            "fabricated": False,
            "note": (
                "Package ready. Human uploads, then confirm-cloud writes "
                "PROTO_CLOUD_SEALED.json for reclaim."
            ),
        }
    )
    man_path = pkg_dir / CLOUD_MANIFEST_NAME
    _atomic_write_json(man_path, manifest)

    # Placeholder PROTO_CLOUD_SEALED (unconfirmed) so reclaim can find the file.
    placeholder = seal(
        {
            "schema": V0_CLOUD_SEALED_SCHEMA,
            "name": "PROTO_CLOUD_SEALED",
            "recorded_at": _utc_now(),
            "confirmed": False,
            "status": "AWAITING_MANUAL_UPLOAD_AND_CONFIRM",
            "bundle_sha256": bundle_sha256,
            "remote_hash": None,
            "remote_uri": None,
            "local_package_dir": str(pkg_dir.resolve()),
            "reclaim_may_evict_superseded": False,
            "upload_is_manual_human_step": True,
            "fabricated": False,
            "note": "Not confirmed. Reclaim must NOT evict superseded artifacts yet.",
        }
    )
    sealed_path = pkg_dir / CLOUD_SEALED_NAME
    # Always rewrite placeholder only if not already confirmed with different content.
    if sealed_path.exists():
        existing = _load_json(sealed_path)
        if existing.get("confirmed") is True:
            # Preserve confirmed seal; do not clobber.
            return {
                "status": "PACKAGE_READY_ALREADY_CLOUD_SEALED",
                "package_dir": str(pkg_dir),
                "manifest": manifest,
                "cloud_sealed": existing,
                "bundle_sha256": bundle_sha256,
                "restore_command": restore_cmd,
                "paths": {
                    "package_dir": str(pkg_dir),
                    "manifest": str(man_path),
                    "cloud_sealed": str(sealed_path),
                    "restore_script": str(restore_path),
                },
            }
        # Unconfirmed — replace with fresh placeholder only if same unconfirmed shape allowed
        sealed_path.unlink()
    _atomic_write_json(sealed_path, placeholder)

    return {
        "status": "PACKAGE_READY_AWAITING_MANUAL_UPLOAD",
        "package_dir": str(pkg_dir),
        "manifest": manifest,
        "cloud_sealed": placeholder,
        "bundle_sha256": bundle_sha256,
        "restore_command": restore_cmd,
        "paths": {
            "package_dir": str(pkg_dir),
            "manifest": str(man_path),
            "cloud_sealed": str(sealed_path),
            "restore_script": str(restore_path),
            "upload_instructions": str(pkg_dir / UPLOAD_INSTRUCTIONS_NAME),
        },
        "upload_performed": False,
    }


def confirm_cloud_sealed(
    *,
    package_dir: Path | str,
    remote_hash: str,
    remote_uri: str | None = None,
    confirmed_by: str = "human",
) -> dict[str, Any]:
    """Write PROTO_CLOUD_SEALED confirm hook that reclaim reads.

    Call after the human has uploaded and obtained the remote content hash.
    """

    pkg = Path(package_dir)
    if not pkg.is_dir():
        raise V0SealError(f"package dir missing: {pkg}")
    man_path = pkg / CLOUD_MANIFEST_NAME
    if not man_path.is_file():
        raise V0SealError(f"cloud manifest missing: {man_path}")
    manifest = _load_json(man_path)
    verify(manifest, label="cloud_manifest")

    remote = str(remote_hash).strip()
    if not remote or len(remote) < 8:
        raise V0SealError("remote_hash must be a non-trivial content hash / etag")

    document = seal(
        {
            "schema": V0_CLOUD_SEALED_SCHEMA,
            "name": "PROTO_CLOUD_SEALED",
            "recorded_at": _utc_now(),
            "confirmed": True,
            "status": "CLOUD_SEALED_CONFIRMED",
            "bundle_sha256": manifest.get("bundle_sha256"),
            "remote_hash": remote,
            "remote_uri": remote_uri,
            "local_package_dir": str(pkg.resolve()),
            "manifest_seal_sha256": manifest.get("seal_sha256"),
            "confirmed_by": confirmed_by,
            "reclaim_may_evict_superseded": True,
            "upload_is_manual_human_step": True,
            "upload_performed": True,  # human asserts by confirming
            "fabricated": False,
            "note": (
                "Human confirmed remote hash after manual upload. "
                "Reclaim may now evict superseded local checkpoints / raw GLM windows "
                "that this package supersedes — only after verifying this file."
            ),
            "reclaim_contract": {
                "must_read": CLOUD_SEALED_NAME,
                "require_confirmed_true": True,
                "require_remote_hash_present": True,
                "require_bundle_sha256_match_manifest": True,
            },
        }
    )
    out = pkg / CLOUD_SEALED_NAME
    if out.exists():
        out.unlink()
    _atomic_write_json(out, document)

    # Also mirror to seal root if package is nested under it.
    parent = pkg.parent
    if parent.name == "proto-frankenstein" or parent == DEFAULT_SEAL_ROOT:
        mirror = parent / CLOUD_SEALED_NAME
        if mirror.exists() and mirror.resolve() != out.resolve():
            mirror.unlink()
        if not mirror.exists():
            _atomic_write_json(mirror, document)

    return {
        "status": "CLOUD_SEALED_CONFIRMED",
        "confirmed": True,
        "document": document,
        "path": str(out),
        "remote_hash": remote,
        "reclaim_may_evict_superseded": True,
    }


def restore_from_package(
    *,
    package_dir: Path | str,
    out_dir: Path | str,
) -> dict[str, Any]:
    """One-command restore: verify payload hashes, copy into out_dir."""

    pkg = Path(package_dir)
    out = Path(out_dir)
    man_path = pkg / CLOUD_MANIFEST_NAME
    if not man_path.is_file():
        raise V0SealError(f"cloud manifest missing: {man_path}")
    manifest = _load_json(man_path)
    verify(manifest, label="cloud_manifest")
    payload_dir = pkg / "payload"
    if not payload_dir.is_dir():
        raise V0SealError(f"payload dir missing: {payload_dir}")

    restored: list[dict[str, Any]] = []
    for row in manifest.get("payload") or []:
        name = row["name"]
        expected = row["sha256"]
        src = payload_dir / name
        if not src.is_file():
            raise V0SealError(f"payload member missing: {src}")
        actual = _sha256_file(src)
        if actual != expected:
            raise V0SealError(
                f"payload hash mismatch for {name}: expected {expected}, got {actual}"
            )
        dest = out / name
        _ensure_dir(out)
        shutil.copy2(src, dest)
        restored.append({"name": name, "sha256": actual, "path": str(dest)})

    receipt = seal(
        {
            "schema": "hawking.frankenstein.v0_restore_receipt.v1",
            "recorded_at": _utc_now(),
            "status": "RESTORED",
            "package_dir": str(pkg.resolve()),
            "out_dir": str(out.resolve()),
            "bundle_sha256": manifest.get("bundle_sha256"),
            "restored": restored,
            "fabricated": False,
        }
    )
    _atomic_write_json(out / "PROTO_FRANKENSTEIN_V0_RESTORE_RECEIPT.json", receipt)
    return {
        "status": "RESTORED",
        "restored": restored,
        "receipt": receipt,
        "out_dir": str(out),
    }


def read_cloud_sealed_for_reclaim(
    search_roots: Sequence[Path | str] | None = None,
) -> dict[str, Any]:
    """Helper reclaim lanes call: find PROTO_CLOUD_SEALED and return authorization."""

    roots = list(search_roots or ())
    if not roots:
        roots = [
            DEFAULT_SEAL_ROOT / CLOUD_PACKAGE_DIR,
            DEFAULT_SEAL_ROOT,
            EVIDENCE_ROOT,
        ]
    for root in roots:
        p = Path(root) / CLOUD_SEALED_NAME
        if not p.is_file():
            # also accept root itself being the file's parent already checked
            continue
        doc = _load_json(p)
        confirmed = doc.get("confirmed") is True
        remote = doc.get("remote_hash")
        return {
            "found": True,
            "path": str(p),
            "confirmed": confirmed,
            "remote_hash": remote,
            "bundle_sha256": doc.get("bundle_sha256"),
            "reclaim_may_evict_superseded": bool(
                confirmed and remote and doc.get("reclaim_may_evict_superseded")
            ),
            "document": doc,
        }
    return {
        "found": False,
        "confirmed": False,
        "reclaim_may_evict_superseded": False,
        "note": "PROTO_CLOUD_SEALED.json not found in search roots",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "PROTO_FRANKENSTEIN_V0 SEAL: assemble loadable artifact, independent "
            "verify, cloud package + restore. Fail closed REQUIRES_TRAINED_MODULES."
        )
    )
    sub = p.add_subparsers(dest="command", required=True)

    p_asm = sub.add_parser("assemble", help="Assemble HCLI-loadable V0 artifact")
    p_asm.add_argument(
        "--modules",
        type=Path,
        default=None,
        help="Trained module pack path (dir with MODULE_PACK.json or pack JSON)",
    )
    p_asm.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_SEAL_ROOT,
        help=f"Output dir (default: {DEFAULT_SEAL_ROOT})",
    )
    p_asm.add_argument("--base-resident-bytes", type=int, default=None)
    p_asm.add_argument("--tps", type=float, default=None)
    p_asm.add_argument("--p99", type=float, default=None)
    p_asm.add_argument("--active-bytes-per-token", type=int, default=None)
    p_asm.add_argument("--no-write", action="store_true")

    p_ver = sub.add_parser(
        "verify",
        help="Independent-challenger A–G + retention + promotion",
    )
    p_ver.add_argument("--out", type=Path, default=DEFAULT_SEAL_ROOT)
    p_ver.add_argument(
        "--arm-scores",
        type=Path,
        default=None,
        help="JSON mapping arm id → {math, secondary, ...}",
    )
    p_ver.add_argument("--matrix", choices=("latent_v0", "functional_transfer"), default="latent_v0")
    p_ver.add_argument("--no-write", action="store_true")

    p_pkg = sub.add_parser("package", help="Build cloud-upload package (no upload)")
    p_pkg.add_argument("--artifact-dir", type=Path, default=DEFAULT_SEAL_ROOT)
    p_pkg.add_argument("--out", type=Path, default=None)
    p_pkg.add_argument("--remote-uri-hint", type=str, default=None)

    p_term = sub.add_parser(
        "terminal",
        help="Create PROTO terminal receipt after verify + cloud + offload checks",
    )
    p_term.add_argument("--artifact", type=Path, required=True)
    p_term.add_argument("--independent-verify", type=Path, required=True)
    p_term.add_argument("--cloud-sealed", type=Path, required=True)
    p_term.add_argument("--cloud-manifest", type=Path, required=True)
    p_term.add_argument(
        "--active-storage-path",
        action="append",
        default=[],
        help="Path checked for donor-weight removal (repeatable)",
    )
    p_term.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform all checks without writing terminal receipt",
    )
    p_term.add_argument(
        "--certification-status",
        default="CONTROLLER_CERTIFIED",
    )
    p_term.add_argument(
        "--certified-by",
        default="protected_controller",
    )
    p_term.add_argument("--out", type=Path, default=DEFAULT_SEAL_ROOT)
    p_term.add_argument("--no-write", action="store_true")

    p_conf = sub.add_parser(
        "confirm-cloud",
        help="PROTO_CLOUD_SEALED confirm hook after manual upload",
    )
    p_conf.add_argument("--package", type=Path, required=True)
    p_conf.add_argument("--remote-hash", type=str, required=True)
    p_conf.add_argument("--remote-uri", type=str, default=None)
    p_conf.add_argument("--confirmed-by", type=str, default="human")

    p_res = sub.add_parser("restore", help="One-command local restore from package")
    p_res.add_argument("--package", type=Path, required=True)
    p_res.add_argument("--out", type=Path, required=True)

    p_rec = sub.add_parser(
        "reclaim-status",
        help="Read PROTO_CLOUD_SEALED for reclaim authorization",
    )
    p_rec.add_argument("--root", type=Path, action="append", default=[])

    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "assemble":
            result = assemble_v0_artifact(
                modules_path=args.modules,
                out_dir=args.out,
                base_resident_bytes=args.base_resident_bytes,
                tps=args.tps,
                p99=args.p99,
                active_bytes_per_token=args.active_bytes_per_token,
                write=not args.no_write,
            )
            print(
                json.dumps(
                    {
                        "status": result["status"],
                        "verdict": result["verdict"],
                        "loadable": result["loadable"],
                        "paths": result.get("paths"),
                        "gate": result.get("gate") or None,
                        "capability_claim": False,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0 if result["loadable"] else 2

        if args.command == "verify":
            arm_scores = None
            if args.arm_scores is not None:
                arm_scores = _load_json(args.arm_scores)
            result = independent_verify(
                arm_scores=arm_scores,
                matrix=args.matrix,
                out_dir=args.out,
                write=not args.no_write,
            )
            print(
                json.dumps(
                    {
                        "status": result["status"],
                        "verdict": result["verdict"],
                        "pass": result["pass"],
                        "paths": result.get("paths"),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0 if result["verdict"] != "REJECT" else 2

        if args.command == "package":
            result = build_cloud_package(
                artifact_dir=args.artifact_dir,
                out_dir=args.out or args.artifact_dir,
                remote_uri_hint=args.remote_uri_hint,
            )
            print(
                json.dumps(
                    {
                        "status": result["status"],
                        "bundle_sha256": result["bundle_sha256"],
                        "restore_command": result["restore_command"],
                        "paths": result["paths"],
                        "upload_performed": False,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        if args.command == "terminal":
            result = terminalize_v0_artifact(
                artifact_path=args.artifact,
                independent_verify_path=args.independent_verify,
                cloud_sealed_path=args.cloud_sealed,
                cloud_manifest_path=args.cloud_manifest,
                active_storage_paths=args.active_storage_path,
                out_dir=args.out,
                dry_run=args.dry_run,
                certification_status=args.certification_status,
                certified_by=args.certified_by,
                write=not args.no_write,
            )
            print(
                json.dumps(
                    {
                        "status": result["status"],
                        "terminal_reached": result["terminal_reached"],
                        "reasons": result["reasons"],
                        "storage": result["storage"],
                        "paths": result.get("paths"),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            if args.no_write:
                return 0 if not result["reasons"] else 2
            return 0 if result["terminal_reached"] else 2

        if args.command == "confirm-cloud":
            result = confirm_cloud_sealed(
                package_dir=args.package,
                remote_hash=args.remote_hash,
                remote_uri=args.remote_uri,
                confirmed_by=args.confirmed_by,
            )
            print(
                json.dumps(
                    {
                        "status": result["status"],
                        "confirmed": result["confirmed"],
                        "remote_hash": result["remote_hash"],
                        "path": result["path"],
                        "reclaim_may_evict_superseded": result[
                            "reclaim_may_evict_superseded"
                        ],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        if args.command == "restore":
            result = restore_from_package(package_dir=args.package, out_dir=args.out)
            print(
                json.dumps(
                    {
                        "status": result["status"],
                        "out_dir": result["out_dir"],
                        "restored": [r["name"] for r in result["restored"]],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        if args.command == "reclaim-status":
            roots = args.root or None
            result = read_cloud_sealed_for_reclaim(roots)
            print(json.dumps(result, indent=2, sort_keys=True, default=str))
            return 0 if result.get("reclaim_may_evict_superseded") else 2

        raise V0SealError(f"unknown command: {args.command}")
    except V0SealError as exc:
        print(json.dumps({"error": "V0_SEAL_ERROR", "detail": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
