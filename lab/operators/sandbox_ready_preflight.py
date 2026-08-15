#!/usr/bin/env python3.12
"""Fail-closed post-Proto preflight for the HCLI execution sandbox.

This module is deliberately an admission verifier, not a campaign launcher.  It
does not fetch Qwen weights, load a model, delete Proto data, create worktrees,
or certify a terminal state.  It joins the evidence that must already exist:

* controller-certified terminal Proto receipt;
* loadable/reversible Proto artifact, independent ACCEPT, and cloud seal;
* removal of configured Proto donor paths from the active local envelope;
* isolated executor/reviewer directory and policy scaffolding;
* disk and 5 GiB process-tree resource reservations.

Missing, malformed, unsealed, or merely candidate evidence is a blocker.
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

from lab.execution_sandbox import (
    ExecutionSandboxPolicy,
    SandboxAction,
    SandboxPrincipal,
)
from lab.receipts import SealIntegrityError, seal, verify


SCHEMA = "hawking.ascension.sandbox_ready_preflight.v1"
CONFIG_SCHEMA = "hawking.ascension.sandbox_ready_config.v1"
TERMINAL_ENDPOINT = "PROTO_FRANKENSTEIN_V0_FULL_LATENT_SEALED"
PROTO_ARTIFACT_SCHEMA = "hawking.frankenstein.v0_loadable_artifact.v1"
PROTO_VERIFY_SCHEMA = "hawking.frankenstein.v0_independent_verify.v1"
PROTO_CLOUD_SCHEMA = "hawking.frankenstein.v0_proto_cloud_sealed.v1"
PROTO_CLOUD_MANIFEST_SCHEMA = "hawking.frankenstein.v0_cloud_package.v1"
READY_STATUS = "SANDBOX_FOUNDATION_PREFLIGHT_READY"
BLOCKED_STATUS = "BLOCKED"
GIB = 1024**3
MAX_PROCESS_TREE_RSS_BYTES = 5 * GIB

REQUIRED_V0_MODULES: tuple[str, ...] = (
    "GLM_EARLY_CONTEXT_BRIDGE",
    "GLM_METHOD_BRIDGE",
    "GLM_DECOMPOSITION_BRIDGE",
    "GLM_PRE_ROUTER_BRIDGE",
    "GLM_POST_MOE_BRIDGE",
    "GLM_FORMALIZATION_BRIDGE",
    "GLM_REPAIR_BRIDGE",
    "GLM_LATE_CONSOLIDATION_BRIDGE",
    "GLM_VALUE_HEAD",
    "GLM_METHOD_CONDITIONED_ROUTE_RESIDUAL",
)


class SandboxReadyInputError(ValueError):
    """Configuration or evidence is malformed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _is_regular_file(path: Path) -> bool:
    try:
        node = os.lstat(path)
    except OSError:
        return False
    return stat.S_ISREG(node.st_mode) and not stat.S_ISLNK(node.st_mode)


def _is_safe_directory(path: Path) -> bool:
    try:
        node = os.lstat(path)
    except OSError:
        return False
    return stat.S_ISDIR(node.st_mode) and not stat.S_ISLNK(node.st_mode)


def _load_json(path: Path, *, require_seal: bool = True) -> dict[str, Any]:
    if not _is_regular_file(path):
        raise SandboxReadyInputError(f"missing or unsafe regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SandboxReadyInputError(f"unreadable JSON {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise SandboxReadyInputError(f"JSON root must be an object: {path}")
    document = dict(value)
    if require_seal:
        try:
            verify(document, label=str(path))
        except SealIntegrityError as exc:
            raise SandboxReadyInputError(str(exc)) from exc
    return document


def _gate(name: str, passed: bool, reasons: Sequence[str], **evidence: Any) -> dict[str, Any]:
    return {
        "name": name,
        "status": "PASS" if passed else "BLOCKED",
        "passed": passed,
        "reasons": list(dict.fromkeys(str(reason) for reason in reasons if reason)),
        "evidence": evidence,
    }


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _binding_seal(terminal: Mapping[str, Any], name: str) -> str | None:
    bindings = _mapping(terminal.get("evidence_bindings"))
    binding = _mapping(bindings.get(name))
    value = binding.get("seal_sha256")
    return value if isinstance(value, str) else None


def _check_terminal(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    path = Path(str(config.get("terminal_receipt", ""))).expanduser()
    reasons: list[str] = []
    try:
        document = _load_json(path)
    except SandboxReadyInputError as exc:
        return _gate("proto_terminal", False, [str(exc)], path=str(path)), None

    endpoint = document.get("terminal_endpoint", document.get("endpoint"))
    if endpoint != TERMINAL_ENDPOINT:
        reasons.append(f"terminal endpoint must be {TERMINAL_ENDPOINT}")
    if document.get("terminal_reached") is not True:
        reasons.append("terminal_reached must be true")
    if document.get("proto_frankenstein_complete") is not True:
        reasons.append("proto_frankenstein_complete must be true")
    if document.get("dry_run") is not False:
        reasons.append("dry_run must be explicitly false")

    certification = _mapping(document.get("certification"))
    if certification.get("status") != "CONTROLLER_CERTIFIED":
        reasons.append("terminal receipt requires CONTROLLER_CERTIFIED certification")
    if certification.get("certified_by") not in {"protected_controller", "human_operator"}:
        reasons.append("terminal certifier must be protected_controller or human_operator")

    storage = _mapping(document.get("storage"))
    required_true = ("offloaded", "hash_verified", "removed_from_active_storage")
    for key in required_true:
        if storage.get(key) is not True:
            reasons.append(f"storage.{key} must be true")
    if storage.get("donor_weights_retained") is not False:
        reasons.append("storage.donor_weights_retained must be false")

    bindings = _mapping(document.get("evidence_bindings"))
    for name in ("artifact", "independent_verify", "cloud_sealed", "cloud_manifest"):
        binding = _mapping(bindings.get(name))
        if not isinstance(binding.get("seal_sha256"), str):
            reasons.append(f"evidence_bindings.{name}.seal_sha256 is required")

    return _gate(
        "proto_terminal",
        not reasons,
        reasons,
        path=str(path),
        seal_sha256=document.get("seal_sha256"),
        endpoint=endpoint,
    ), document


def _check_artifact(config: Mapping[str, Any], terminal: Mapping[str, Any] | None) -> dict[str, Any]:
    path = Path(str(config.get("artifact", ""))).expanduser()
    reasons: list[str] = []
    try:
        document = _load_json(path)
    except SandboxReadyInputError as exc:
        return _gate("proto_artifact", False, [str(exc)], path=str(path))

    if document.get("schema") != PROTO_ARTIFACT_SCHEMA:
        reasons.append(f"artifact schema must be {PROTO_ARTIFACT_SCHEMA}")
    for key in ("trained", "reversible", "bypassable", "hash_bound"):
        if document.get(key) is not True:
            reasons.append(f"artifact.{key} must be true")
    hcli = _mapping(document.get("hcli"))
    if hcli.get("loadable_descriptor") is not True:
        reasons.append("artifact must be HCLI-loadable")
    endpoint = document.get("terminal_endpoint", document.get("endpoint"))
    if endpoint != TERMINAL_ENDPOINT:
        reasons.append("artifact terminal endpoint mismatch")

    modules = document.get("v0_modules")
    names = {
        row.get("name")
        for row in modules
        if isinstance(row, Mapping) and isinstance(row.get("name"), str)
    } if isinstance(modules, list) else set()
    missing = sorted(set(REQUIRED_V0_MODULES) - names)
    if missing:
        reasons.append(f"artifact missing required V0 modules: {missing}")

    runtime_storage = _mapping(document.get("runtime_storage"))
    storage = _mapping(runtime_storage.get("storage"))
    if storage.get("donor_weights_retained") is not False:
        reasons.append("artifact runtime storage must not retain donor weights")

    expected = _binding_seal(terminal or {}, "artifact")
    if expected is None:
        reasons.append("terminal receipt does not bind artifact seal")
    elif expected != document.get("seal_sha256"):
        reasons.append("terminal artifact binding does not match artifact seal")

    return _gate(
        "proto_artifact",
        not reasons,
        reasons,
        path=str(path),
        seal_sha256=document.get("seal_sha256"),
        module_count=len(names),
    )


def _check_independent_verify(
    config: Mapping[str, Any], terminal: Mapping[str, Any] | None
) -> dict[str, Any]:
    path = Path(str(config.get("independent_verify", ""))).expanduser()
    reasons: list[str] = []
    try:
        document = _load_json(path)
    except SandboxReadyInputError as exc:
        return _gate("proto_independent_verify", False, [str(exc)], path=str(path))

    if document.get("schema") != PROTO_VERIFY_SCHEMA:
        reasons.append(f"verify schema must be {PROTO_VERIFY_SCHEMA}")
    if document.get("verdict") != "ACCEPT" or document.get("pass") is not True:
        reasons.append("independent Proto verifier must ACCEPT")
    if document.get("independent_of_training_lane") is not True:
        reasons.append("verification must be independent of training lane")
    if _mapping(document.get("challenger")).get("verdict") != "ACCEPT":
        reasons.append("independent challenger must ACCEPT")
    if _mapping(document.get("promotion_gate")).get("verdict") != "ACCEPT":
        reasons.append("promotion gate must ACCEPT")
    if _mapping(document.get("retention_gate")).get("verdict") not in {"PASS", "ACCEPT"}:
        reasons.append("retention gate must PASS")
    ablation = _mapping(document.get("ablation_ag"))
    if ablation.get("verdict") != "ACCEPT":
        reasons.append("complete A-G ablation must ACCEPT")

    expected = _binding_seal(terminal or {}, "independent_verify")
    if expected is None:
        reasons.append("terminal receipt does not bind independent verifier seal")
    elif expected != document.get("seal_sha256"):
        reasons.append("terminal independent-verify binding mismatch")

    return _gate(
        "proto_independent_verify",
        not reasons,
        reasons,
        path=str(path),
        seal_sha256=document.get("seal_sha256"),
    )


def _check_cloud(config: Mapping[str, Any], terminal: Mapping[str, Any] | None) -> dict[str, Any]:
    seal_path = Path(str(config.get("cloud_sealed", ""))).expanduser()
    manifest_path = Path(str(config.get("cloud_manifest", ""))).expanduser()
    restore_path = Path(str(config.get("restore_script", ""))).expanduser()
    reasons: list[str] = []
    try:
        cloud = _load_json(seal_path)
        manifest = _load_json(manifest_path)
    except SandboxReadyInputError as exc:
        return _gate(
            "proto_cloud_offload",
            False,
            [str(exc)],
            cloud_sealed=str(seal_path),
            cloud_manifest=str(manifest_path),
        )

    if cloud.get("schema") != PROTO_CLOUD_SCHEMA:
        reasons.append(f"cloud seal schema must be {PROTO_CLOUD_SCHEMA}")
    if manifest.get("schema") != PROTO_CLOUD_MANIFEST_SCHEMA:
        reasons.append(f"cloud manifest schema must be {PROTO_CLOUD_MANIFEST_SCHEMA}")
    for key in ("confirmed", "upload_performed", "reclaim_may_evict_superseded"):
        if cloud.get(key) is not True:
            reasons.append(f"cloud seal {key} must be true")
    remote_hash = cloud.get("remote_hash")
    if not isinstance(remote_hash, str) or len(remote_hash.strip()) < 8:
        reasons.append("cloud seal requires a non-trivial remote_hash")
    if cloud.get("bundle_sha256") != manifest.get("bundle_sha256"):
        reasons.append("cloud bundle hash does not match manifest")
    if cloud.get("manifest_seal_sha256") != manifest.get("seal_sha256"):
        reasons.append("cloud seal does not bind manifest seal")

    expected_cloud = _binding_seal(terminal or {}, "cloud_sealed")
    expected_manifest = _binding_seal(terminal or {}, "cloud_manifest")
    if expected_cloud != cloud.get("seal_sha256"):
        reasons.append("terminal cloud-seal binding mismatch")
    if expected_manifest != manifest.get("seal_sha256"):
        reasons.append("terminal cloud-manifest binding mismatch")

    package_dir = manifest_path.parent
    payload_dir = package_dir / "payload"
    payload = manifest.get("payload")
    if not isinstance(payload, list) or not payload:
        reasons.append("cloud manifest payload must contain sealed Proto members")
    else:
        for row in payload:
            if not isinstance(row, Mapping) or not isinstance(row.get("name"), str):
                reasons.append("cloud manifest payload row malformed")
                continue
            member = payload_dir / row["name"]
            if not _is_regular_file(member):
                reasons.append(f"cloud payload member missing: {row['name']}")
                continue
            if row.get("sha256") != _sha256_file(member):
                reasons.append(f"cloud payload hash mismatch: {row['name']}")

    if not _is_regular_file(restore_path):
        reasons.append("one-command restore script missing or unsafe")
    else:
        mode = os.lstat(restore_path).st_mode
        if not (mode & stat.S_IXUSR):
            reasons.append("one-command restore script is not owner-executable")

    return _gate(
        "proto_cloud_offload",
        not reasons,
        reasons,
        cloud_sealed=str(seal_path),
        cloud_manifest=str(manifest_path),
        restore_script=str(restore_path),
        bundle_sha256=manifest.get("bundle_sha256"),
        remote_hash=remote_hash,
    )


def _check_active_envelope(config: Mapping[str, Any]) -> dict[str, Any]:
    raw = config.get("active_storage_paths_must_be_absent")
    reasons: list[str] = []
    if not isinstance(raw, list) or not raw:
        return _gate(
            "proto_removed_from_active_envelope",
            False,
            ["active_storage_paths_must_be_absent must be a non-empty list"],
        )
    present: list[str] = []
    for value in raw:
        path = Path(str(value)).expanduser()
        if path.exists() or path.is_symlink():
            present.append(str(path))
    if present:
        reasons.append("configured Proto donor/body paths still exist in active storage")
    return _gate(
        "proto_removed_from_active_envelope",
        not reasons,
        reasons,
        checked_paths=[str(Path(str(v)).expanduser()) for v in raw],
        present_paths=present,
    )


def _within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except (OSError, ValueError):
        return False


def _check_sandbox(config: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(str(config.get("root", ""))).expanduser()
    executor = Path(str(config.get("executor_worktree_root", ""))).expanduser()
    reviewer = Path(str(config.get("reviewer_readonly_root", ""))).expanduser()
    receipts = Path(str(config.get("receipts_root", ""))).expanduser()
    logs = Path(str(config.get("logs_root", ""))).expanduser()
    reasons: list[str] = []

    for label, path in (
        ("root", root),
        ("executor_worktree_root", executor),
        ("reviewer_readonly_root", reviewer),
        ("receipts_root", receipts),
        ("logs_root", logs),
    ):
        if not _is_safe_directory(path):
            reasons.append(f"sandbox {label} missing or unsafe: {path}")
        elif label != "root" and not _within(path, root):
            reasons.append(f"sandbox {label} escapes sandbox root")

    reviewer_enforcement = config.get("reviewer_enforcement")
    reviewer_fs_readonly = False
    if _is_safe_directory(reviewer):
        mode = stat.S_IMODE(os.lstat(reviewer).st_mode)
        reviewer_fs_readonly = not bool(mode & 0o222)
        if reviewer_enforcement == "filesystem_readonly" and not reviewer_fs_readonly:
            reasons.append("reviewer root must be filesystem read-only (no write bits)")
    if reviewer_enforcement not in {"filesystem_readonly", "policy_readonly_scaffold"}:
        reasons.append(
            "reviewer_enforcement must be filesystem_readonly or policy_readonly_scaffold"
        )

    selectors = config.get("allowed_test_selectors")
    if not isinstance(selectors, list) or not selectors or not all(
        isinstance(item, str) and item for item in selectors
    ):
        reasons.append("allowed_test_selectors must be a non-empty string list")
        selectors = []
    downloads = config.get("approved_download_ids")
    if downloads != []:
        reasons.append("approved_download_ids must remain empty before controller admission")

    if not reasons:
        policy = ExecutionSandboxPolicy(
            owned_worktree_roots=(str(executor),),
            sandbox_root=str(root),
            allowed_test_selectors=frozenset(selectors),
            approved_download_ids=frozenset(downloads),
        )
        model = SandboxPrincipal.SANDBOX_MODEL
        allowed = policy.authorize(
            model,
            SandboxAction.EDIT_OWNED_WORKTREE,
            target=executor / "candidate.py",
        )
        outside = policy.authorize(
            model,
            SandboxAction.EDIT_OWNED_WORKTREE,
            target=root.parent / "escape.py",
        )
        credential = policy.authorize(
            model,
            SandboxAction.READ_SOURCE,
            target=executor / ".env",
        )
        merge = policy.authorize(model, SandboxAction.MERGE_SELF)
        unknown = policy.authorize(model, "unknown_future_effect")
        if not allowed.allowed:
            reasons.append("sandbox policy refused owned executor edit")
        for label, decision in (
            ("outside edit", outside),
            ("credential read", credential),
            ("self merge", merge),
            ("unknown action", unknown),
        ):
            if decision.allowed:
                reasons.append(f"sandbox policy did not fail closed for {label}")

    return _gate(
        "sandbox_isolation_policy",
        not reasons,
        reasons,
        root=str(root),
        executor_worktree_root=str(executor),
        reviewer_readonly_root=str(reviewer),
        reviewer_enforcement=reviewer_enforcement,
        reviewer_filesystem_readonly=reviewer_fs_readonly,
        approved_download_ids=downloads,
        allowed_test_selectors=selectors,
    )


def _check_authority(config: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    required_files = config.get("required_files")
    if not isinstance(required_files, list) or not required_files:
        return _gate("protected_authority", False, ["authority required_files missing"])
    checked: list[str] = []
    for value in required_files:
        path = Path(str(value)).expanduser()
        checked.append(str(path))
        if not _is_regular_file(path):
            reasons.append(f"protected authority file missing or unsafe: {path}")
    return _gate("protected_authority", not reasons, reasons, checked_files=checked)


def _check_resources(config: Mapping[str, Any], sandbox: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    disk_path = Path(str(config.get("disk_path", ""))).expanduser()
    if not disk_path.exists():
        return _gate("resource_reservation", False, [f"disk_path missing: {disk_path}"])
    free = shutil.disk_usage(disk_path).free
    floor = config.get("minimum_free_disk_bytes")
    qwen30 = config.get("qwen30_body_reservation_bytes")
    pack = config.get("qwen30_pack_working_reservation_bytes")
    rss = config.get("process_tree_rss_cap_bytes")
    swap_growth = config.get("swap_growth_allowed")
    for label, value in (
        ("minimum_free_disk_bytes", floor),
        ("qwen30_body_reservation_bytes", qwen30),
        ("qwen30_pack_working_reservation_bytes", pack),
    ):
        if not isinstance(value, int) or value <= 0:
            reasons.append(f"{label} must be a positive integer")
    if floor != 25 * GIB:
        reasons.append("minimum free-disk floor must be exactly 25 GiB")
    if rss != MAX_PROCESS_TREE_RSS_BYTES:
        reasons.append("process-tree RSS cap must be exactly 5 GiB")
    if swap_growth is not False:
        reasons.append("swap_growth_allowed must be false")

    required = sum(value for value in (floor, qwen30, pack) if isinstance(value, int))
    if free < required:
        reasons.append("free disk cannot preserve floor after 30B body + pack reservation")

    root = Path(str(sandbox.get("root", ""))).expanduser()
    try:
        same_device = os.stat(root).st_dev == os.stat(disk_path).st_dev
    except OSError:
        same_device = False
    if not same_device:
        reasons.append("sandbox root and reservation disk are not on the same filesystem")

    return _gate(
        "resource_reservation",
        not reasons,
        reasons,
        disk_path=str(disk_path),
        free_bytes=free,
        required_free_bytes=required,
        post_reservation_free_bytes=free - required,
        minimum_free_disk_bytes=floor,
        qwen30_body_reservation_bytes=qwen30,
        qwen30_pack_working_reservation_bytes=pack,
        process_tree_rss_cap_bytes=rss,
        swap_growth_allowed=swap_growth,
    )


def evaluate_sandbox_ready(config: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate post-Proto sandbox-foundation readiness without side effects."""
    if not isinstance(config, Mapping):
        raise SandboxReadyInputError("config must be a mapping")
    if config.get("schema") != CONFIG_SCHEMA:
        raise SandboxReadyInputError(f"config schema must be {CONFIG_SCHEMA}")

    proto = _mapping(config.get("proto"))
    sandbox = _mapping(config.get("sandbox"))
    authority = _mapping(config.get("authority"))
    resources = _mapping(config.get("resources"))

    terminal_gate, terminal = _check_terminal(proto)
    gates = {
        "proto_terminal": terminal_gate,
        "proto_artifact": _check_artifact(proto, terminal),
        "proto_independent_verify": _check_independent_verify(proto, terminal),
        "proto_cloud_offload": _check_cloud(proto, terminal),
        "proto_removed_from_active_envelope": _check_active_envelope(proto),
        "sandbox_isolation_policy": _check_sandbox(sandbox),
        "protected_authority": _check_authority(authority),
        "resource_reservation": _check_resources(resources, sandbox),
    }
    blockers = [
        f"{name}: {reason}"
        for name, gate in gates.items()
        if not gate["passed"]
        for reason in (gate["reasons"] or ["gate blocked"])
    ]
    proto_gates = (
        "proto_terminal",
        "proto_artifact",
        "proto_independent_verify",
        "proto_cloud_offload",
        "proto_removed_from_active_envelope",
    )
    scaffold_gates = (
        "sandbox_isolation_policy",
        "protected_authority",
        "resource_reservation",
    )
    proto_ready = all(gates[name]["passed"] for name in proto_gates)
    scaffold_ready = all(gates[name]["passed"] for name in scaffold_gates)
    ready = proto_ready and scaffold_ready

    return seal(
        {
            "schema": SCHEMA,
            "recorded_at": _utc_now(),
            "status": READY_STATUS if ready else BLOCKED_STATUS,
            "sandbox_foundation_preflight_ready": ready,
            "option_c_live_claim": False,
            "qwen30_code_only_overlap_permitted": scaffold_ready,
            "qwen30_body_admission_candidate": ready,
            "qwen30_body_download_started": False,
            "proto_terminal_claimed": proto_ready,
            "gates": gates,
            "blockers": blockers,
            "claim_boundary": {
                "preflight_not_terminal_certification": True,
                "preflight_not_qwen_download_authorization": True,
                "option_c_live_still_requires_promoted_30b_and_80b": True,
                "sandbox_models_have_report_only_authority": True,
                "no_network_calls": True,
                "no_model_loads": True,
                "no_deletions": True,
            },
        }
    )


def _atomic_write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="sandbox-ready config JSON")
    parser.add_argument("--receipt", help="optional atomic output receipt path")
    parser.add_argument(
        "--allow-blocked",
        action="store_true",
        help="return zero for inspection even when readiness is blocked",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = _load_json(Path(args.config).expanduser(), require_seal=False)
        result = evaluate_sandbox_ready(config)
    except SandboxReadyInputError as exc:
        result = seal(
            {
                "schema": SCHEMA,
                "recorded_at": _utc_now(),
                "status": BLOCKED_STATUS,
                "sandbox_foundation_preflight_ready": False,
                "blockers": [str(exc)],
                "claim_boundary": {
                    "preflight_not_terminal_certification": True,
                    "preflight_not_qwen_download_authorization": True,
                },
            }
        )
    if args.receipt:
        _atomic_write_json(Path(args.receipt).expanduser(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("sandbox_foundation_preflight_ready") or args.allow_blocked else 2


if __name__ == "__main__":
    raise SystemExit(main())
