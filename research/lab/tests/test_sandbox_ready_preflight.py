"""Fail-closed tests for the post-Proto sandbox-ready verifier."""
from __future__ import annotations

import json
import os
from pathlib import Path

from lab.operators.sandbox_ready_preflight import (
    BLOCKED_STATUS,
    CONFIG_SCHEMA,
    GIB,
    MAX_PROCESS_TREE_RSS_BYTES,
    PROTO_ARTIFACT_SCHEMA,
    PROTO_CLOUD_MANIFEST_SCHEMA,
    PROTO_CLOUD_SCHEMA,
    PROTO_VERIFY_SCHEMA,
    READY_STATUS,
    REQUIRED_V0_MODULES,
    TERMINAL_ENDPOINT,
    evaluate_sandbox_ready,
)
from lab.receipts import seal, verify


def _write(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def _fixture(tmp_path: Path) -> dict:
    sandbox = tmp_path / "sandbox-ready"
    executor = sandbox / "worktrees" / "executor"
    reviewer = sandbox / "worktrees" / "reviewer-readonly"
    receipts = sandbox / "receipts"
    logs = sandbox / "logs"
    for path in (executor, reviewer, receipts, logs):
        path.mkdir(parents=True, exist_ok=True)
    reviewer.chmod(0o555)

    proto = tmp_path / "proto"
    artifact_path = proto / "PROTO_FRANKENSTEIN_V0_ARTIFACT.json"
    verify_path = proto / "PROTO_FRANKENSTEIN_V0_INDEPENDENT_VERIFY.json"
    package = proto / "cloud-package"
    payload = package / "payload"
    payload.mkdir(parents=True)

    artifact = seal(
        {
            "schema": PROTO_ARTIFACT_SCHEMA,
            "terminal_endpoint": TERMINAL_ENDPOINT,
            "trained": True,
            "reversible": True,
            "bypassable": True,
            "hash_bound": True,
            "hcli": {"loadable_descriptor": True},
            "v0_modules": [{"name": name} for name in REQUIRED_V0_MODULES],
            "runtime_storage": {"storage": {"donor_weights_retained": False}},
        }
    )
    independent = seal(
        {
            "schema": PROTO_VERIFY_SCHEMA,
            "verdict": "ACCEPT",
            "pass": True,
            "independent_of_training_lane": True,
            "challenger": {"verdict": "ACCEPT"},
            "promotion_gate": {"verdict": "ACCEPT"},
            "retention_gate": {"verdict": "PASS"},
            "ablation_ag": {"verdict": "ACCEPT"},
        }
    )
    _write(artifact_path, artifact)
    _write(verify_path, independent)

    payload_artifact = payload / artifact_path.name
    payload_verify = payload / verify_path.name
    _write(payload_artifact, artifact)
    _write(payload_verify, independent)

    import hashlib

    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = seal(
        {
            "schema": PROTO_CLOUD_MANIFEST_SCHEMA,
            "bundle_sha256": "a" * 64,
            "payload": [
                {"name": payload_artifact.name, "sha256": sha(payload_artifact)},
                {"name": payload_verify.name, "sha256": sha(payload_verify)},
            ],
        }
    )
    manifest_path = package / "PROTO_V0_CLOUD_MANIFEST.json"
    _write(manifest_path, manifest)
    cloud = seal(
        {
            "schema": PROTO_CLOUD_SCHEMA,
            "confirmed": True,
            "upload_performed": True,
            "reclaim_may_evict_superseded": True,
            "remote_hash": "remote-deadbeef",
            "bundle_sha256": manifest["bundle_sha256"],
            "manifest_seal_sha256": manifest["seal_sha256"],
        }
    )
    cloud_path = package / "PROTO_CLOUD_SEALED.json"
    _write(cloud_path, cloud)
    restore = package / "restore_proto_frankenstein_v0.sh"
    restore.write_text("#!/bin/sh\nexit 0\n")
    restore.chmod(0o700)

    terminal = seal(
        {
            "schema": "hawking.frankenstein.proto_terminal.v1",
            "terminal_endpoint": TERMINAL_ENDPOINT,
            "terminal_reached": True,
            "proto_frankenstein_complete": True,
            "dry_run": False,
            "certification": {
                "status": "CONTROLLER_CERTIFIED",
                "certified_by": "protected_controller",
            },
            "storage": {
                "offloaded": True,
                "hash_verified": True,
                "removed_from_active_storage": True,
                "donor_weights_retained": False,
            },
            "evidence_bindings": {
                "artifact": {"seal_sha256": artifact["seal_sha256"]},
                "independent_verify": {"seal_sha256": independent["seal_sha256"]},
                "cloud_sealed": {"seal_sha256": cloud["seal_sha256"]},
                "cloud_manifest": {"seal_sha256": manifest["seal_sha256"]},
            },
        }
    )
    terminal_path = proto / "PROTO_FRANKENSTEIN_V0_TERMINAL_RECEIPT.json"
    _write(terminal_path, terminal)

    authority_a = tmp_path / "execution_sandbox.py"
    authority_b = tmp_path / "verification_authority.py"
    authority_a.write_text("policy\n")
    authority_b.write_text("authority\n")
    absent_active_path = tmp_path / "already-evicted-proto-body"

    return {
        "schema": CONFIG_SCHEMA,
        "proto": {
            "terminal_receipt": str(terminal_path),
            "artifact": str(artifact_path),
            "independent_verify": str(verify_path),
            "cloud_sealed": str(cloud_path),
            "cloud_manifest": str(manifest_path),
            "restore_script": str(restore),
            "active_storage_paths_must_be_absent": [str(absent_active_path)],
        },
        "sandbox": {
            "root": str(sandbox),
            "executor_worktree_root": str(executor),
            "reviewer_readonly_root": str(reviewer),
            "reviewer_enforcement": "filesystem_readonly",
            "receipts_root": str(receipts),
            "logs_root": str(logs),
            "allowed_test_selectors": ["unit", "sandbox"],
            "approved_download_ids": [],
        },
        "authority": {"required_files": [str(authority_a), str(authority_b)]},
        "resources": {
            "disk_path": str(tmp_path),
            "minimum_free_disk_bytes": 25 * GIB,
            "qwen30_body_reservation_bytes": 1,
            "qwen30_pack_working_reservation_bytes": 1,
            "process_tree_rss_cap_bytes": MAX_PROCESS_TREE_RSS_BYTES,
            "swap_growth_allowed": False,
        },
    }


def test_complete_evidence_passes_and_is_sealed(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    result = evaluate_sandbox_ready(config)
    verify(result)
    assert result["status"] == READY_STATUS
    assert result["sandbox_foundation_preflight_ready"] is True
    assert result["qwen30_body_admission_candidate"] is True
    assert result["qwen30_body_download_started"] is False
    assert result["option_c_live_claim"] is False
    assert not result["blockers"]


def test_missing_terminal_receipt_blocks_but_code_scaffold_can_continue(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    Path(config["proto"]["terminal_receipt"]).unlink()
    result = evaluate_sandbox_ready(config)
    assert result["status"] == BLOCKED_STATUS
    assert result["sandbox_foundation_preflight_ready"] is False
    assert result["qwen30_body_admission_candidate"] is False
    assert result["qwen30_code_only_overlap_permitted"] is True
    assert result["gates"]["proto_terminal"]["passed"] is False


def test_candidate_or_dry_run_terminal_never_passes(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    path = Path(config["proto"]["terminal_receipt"])
    terminal = json.loads(path.read_text())
    terminal["terminal_reached"] = False
    terminal["dry_run"] = True
    terminal["certification"]["status"] = "CANDIDATE"
    _write(path, seal(terminal))
    result = evaluate_sandbox_ready(config)
    assert result["gates"]["proto_terminal"]["passed"] is False
    joined = " ".join(result["gates"]["proto_terminal"]["reasons"])
    assert "terminal_reached" in joined
    assert "dry_run" in joined
    assert "CONTROLLER_CERTIFIED" in joined


def test_tampered_artifact_blocks_even_when_terminal_claims_complete(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    path = Path(config["proto"]["artifact"])
    artifact = json.loads(path.read_text())
    artifact["trained"] = False
    path.write_text(json.dumps(artifact))  # stale seal on purpose
    result = evaluate_sandbox_ready(config)
    assert result["gates"]["proto_artifact"]["passed"] is False
    assert any("seal mismatch" in reason for reason in result["gates"]["proto_artifact"]["reasons"])


def test_present_active_proto_path_blocks_offload_gate(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    active = Path(config["proto"]["active_storage_paths_must_be_absent"][0])
    active.mkdir()
    result = evaluate_sandbox_ready(config)
    gate = result["gates"]["proto_removed_from_active_envelope"]
    assert gate["passed"] is False
    assert str(active) in gate["evidence"]["present_paths"]


def test_reviewer_directory_must_be_read_only(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    reviewer = Path(config["sandbox"]["reviewer_readonly_root"])
    reviewer.chmod(0o755)
    try:
        result = evaluate_sandbox_ready(config)
        assert result["gates"]["sandbox_isolation_policy"]["passed"] is False
        assert any("read-only" in reason for reason in result["gates"]["sandbox_isolation_policy"]["reasons"])
    finally:
        reviewer.chmod(0o755)


def test_resource_policy_requires_5g_cap_and_no_swap_growth(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    config["resources"]["process_tree_rss_cap_bytes"] = 6 * GIB
    config["resources"]["swap_growth_allowed"] = True
    result = evaluate_sandbox_ready(config)
    gate = result["gates"]["resource_reservation"]
    assert gate["passed"] is False
    assert any("5 GiB" in reason for reason in gate["reasons"])
    assert any("swap_growth_allowed" in reason for reason in gate["reasons"])


def test_no_download_is_preapproved(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    config["sandbox"]["approved_download_ids"] = ["qwen30-body"]
    result = evaluate_sandbox_ready(config)
    assert result["gates"]["sandbox_isolation_policy"]["passed"] is False
    assert result["qwen30_body_download_started"] is False
