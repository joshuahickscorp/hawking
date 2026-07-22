#!/usr/bin/env python3.12
"""Fake-only adversarial tests for exact Kimi-K2.6 Phase-2 source release."""
from __future__ import annotations

import copy
import hashlib
import os
import pathlib
import sys
from typing import Any, Mapping, Sequence

import pytest


CONDENSE = pathlib.Path(__file__).resolve().parents[1]
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import kimi_k26_phase2_release as release  # noqa: E402

phase1 = release.phase1


def _git_sha(raw: bytes) -> str:
    return hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()  # noqa: S324


class FakeProbe:
    def __init__(self, blocked: str | None = None) -> None:
        self.blocked = blocked
        self.calls = 0

    def inspect(
        self,
        layout: phase1.SessionLayout,
        *,
        release_roots: Sequence[pathlib.Path],
        queue_roots: Sequence[pathlib.Path],
        lease_paths: Sequence[pathlib.Path],
        owned_lease_paths: Sequence[pathlib.Path],
        mop_root: pathlib.Path,
        shared_xet: pathlib.Path,
        repo_root: pathlib.Path,
    ) -> Mapping[str, Any]:
        del layout, release_roots, queue_roots, owned_lease_paths, repo_root
        self.calls += 1
        checks: dict[str, dict[str, Any]] = {
            "readers": {"status": "PASS", "matches": [], "failures": []},
            "processes": {"status": "PASS", "matches": [], "failures": []},
            "launchd": {
                "status": "PASS",
                "matching_configuration": False,
                "output_sha256": hashlib.sha256(b"fake-launchd").hexdigest(),
                "failures": [],
            },
            "queues": {"status": "PASS", "pending_entries": [], "failures": []},
            "leases": {
                "status": "PASS",
                "checked_paths": [os.fspath(path) for path in lease_paths],
                "conflicts": [],
                "failures": [],
            },
            "git_push": {
                "status": "PASS",
                "head": "a" * 40,
                "branch": "codex/fake",
                "upstream": "origin/codex/fake",
                "remote_head": "a" * 40,
                "worktree_clean": True,
                "head_pushed_exactly": True,
                "failures": [],
            },
            "mop": {
                "status": "PASS",
                "boundary": release._directory_boundary(mop_root, label="fake MOP"),
            },
            "shared_xet": {
                "status": "PASS",
                "boundary": release._directory_boundary(
                    shared_xet, label="fake shared Xet"
                ),
            },
        }
        if self.blocked is not None:
            checks[self.blocked]["status"] = "BLOCKED"
        return checks


def _mkdir(path: pathlib.Path) -> None:
    path.mkdir(parents=True, mode=0o700, exist_ok=True)
    path.chmod(0o700)


def _fake_world(tmp_path: pathlib.Path) -> dict[str, Any]:
    parent = tmp_path / "sessions"
    session = parent / "phase2-test"
    _mkdir(parent)
    layout = phase1.layout_for(session, parent=parent)
    for path in (
        layout.session,
        layout.hub,
        layout.xet,
        layout.build,
        layout.tmp,
        layout.hf_home,
        layout.recovery,
        layout.evidence,
        layout.model_cache,
        layout.model_cache / "snapshots",
        layout.snapshot,
        layout.blobs,
        layout.capsule,
    ):
        _mkdir(path)
    mop = tmp_path / "mop"
    shared = tmp_path / "shared-xet"
    repo = tmp_path / "repo"
    for path in (mop, shared, repo):
        _mkdir(path)
    (mop / "preserve").write_bytes(b"MOP must remain")
    (shared / "preserve").write_bytes(b"shared Xet must remain")
    (layout.capsule / "payload.bin").write_bytes(b"rollback payload")
    (layout.recovery / "record.json").write_bytes(b"recovery")
    (layout.evidence / "audit-history.json").write_bytes(b"evidence")

    rows: list[dict[str, Any]] = []
    weight_paths: list[pathlib.Path] = []
    metadata_paths: list[pathlib.Path] = []
    weight_blobs: list[pathlib.Path] = []
    metadata_blobs: list[pathlib.Path] = []
    for index in range(1, 65):
        relative = f"model-{index:05d}-of-000064.safetensors"
        raw = f"weight-{index:02d}".encode()
        digest = hashlib.sha256(raw).hexdigest()
        blob = layout.blobs / digest
        blob.write_bytes(raw)
        link = layout.snapshot / relative
        link.symlink_to(os.path.relpath(blob, link.parent))
        rows.append(
            {
                "blob_id": _git_sha(raw),
                "path": relative,
                "sha256": digest,
                "size": len(raw),
            }
        )
        weight_paths.append(link)
        weight_blobs.append(blob)
    for index in range(32):
        relative = f"metadata/meta-{index:02d}.json"
        raw = f"metadata-{index:02d}".encode()
        digest = hashlib.sha256(raw).hexdigest()
        blob = layout.blobs / digest
        blob.write_bytes(raw)
        link = layout.snapshot / relative
        _mkdir(link.parent)
        link.symlink_to(os.path.relpath(blob, link.parent))
        rows.append(
            {
                "blob_id": _git_sha(raw),
                "path": relative,
                "sha256": digest,
                "size": len(raw),
            }
        )
        metadata_paths.append(link)
        metadata_blobs.append(blob)
    rows.sort(key=lambda row: row["path"])
    manifest = phase1.seal_document(
        {
            "schema": "hawking.kimi_k26.official_manifest.v1",
            "repo": phase1.KIMI_REPO,
            "sha": phase1.KIMI_REVISION,
            "file_count": 96,
            "weight_shards": 64,
            "total_bytes": sum(row["size"] for row in rows),
            "weight_bytes": sum(
                row["size"] for row in rows if release.WEIGHT_RE.fullmatch(row["path"])
            ),
            "files": rows,
        }
    )
    source = phase1.seal_document(
        {
            "schema": "hawking.kimi_k26.release_cycle.source_verification.v1",
            "status": "PASS_EXACT_IMMUTABLE_SOURCE",
            "repo": phase1.KIMI_REPO,
            "revision": phase1.KIMI_REVISION,
            "manifest_seal_sha256": manifest["seal_sha256"],
            "file_count": 96,
            "weight_shards": 64,
            "logical_bytes": manifest["total_bytes"],
            "weight_bytes": manifest["weight_bytes"],
        }
    )
    capsule = phase1.seal_document(
        {
            "schema": "hawking.kimi_k26.release_cycle.rollback_capsule.v1",
            "status": "PASS_EXACT_PAYLOAD_RESULT_CAPTURE",
            "session": os.fspath(layout.session),
            "payload": {"sha256": hashlib.sha256(b"rollback payload").hexdigest()},
            "mop_touched": False,
        }
    )
    _mkdir(layout.xet / "staging" / "nested")
    (layout.xet / "root-object").write_bytes(b"xet-root")
    (layout.xet / "staging" / "nested" / "chunk").write_bytes(b"xet-chunk")
    queue_roots = (layout.session / "queue", layout.session / "outbox")
    lease_paths = (
        layout.evidence / release.RELEASE_LEASE_NAME,
        layout.evidence / release.DOWNLOAD_LEASE_NAME,
    )
    return {
        "layout": layout,
        "manifest": manifest,
        "source": source,
        "capsule": capsule,
        "mop": mop,
        "shared": shared,
        "repo": repo,
        "weight_paths": weight_paths,
        "weight_blobs": weight_blobs,
        "metadata_paths": metadata_paths,
        "metadata_blobs": metadata_blobs,
        "queue_roots": queue_roots,
        "lease_paths": lease_paths,
    }


def _bundle(world: Mapping[str, Any], probe: FakeProbe | None = None) -> dict[str, Any]:
    return release.build_release_bundle(
        world["layout"],
        world["manifest"],
        world["source"],
        world["capsule"],
        probe=probe or FakeProbe(),
        queue_roots=world["queue_roots"],
        lease_paths=world["lease_paths"],
        mop_root=world["mop"],
        shared_xet=world["shared"],
        repo_root=world["repo"],
    )


def _execute(
    world: Mapping[str, Any],
    bundle: Mapping[str, Any],
    token: str,
    *,
    probe: release.AuditProbe | None = None,
    capsule_verifier: release.Verifier | None = None,
    fault_after: int | None = None,
    fault_after_unlink_before_commit: int | None = None,
) -> dict[str, Any]:
    return release.execute_release(
        world["layout"],
        bundle,
        confirmation_token=token,
        manifest=world["manifest"],
        source_verifier=lambda _layout: copy.deepcopy(world["source"]),
        capsule_verifier=(
            capsule_verifier
            or (lambda _layout: copy.deepcopy(world["capsule"]))
        ),
        probe=probe or FakeProbe(),
        queue_roots=world["queue_roots"],
        lease_paths=world["lease_paths"],
        mop_root=world["mop"],
        shared_xet=world["shared"],
        repo_root=world["repo"],
        fault_after=fault_after,
        fault_after_unlink_before_commit=fault_after_unlink_before_commit,
    )


def test_inventory_is_exact_64_weights_32_metadata_and_xet(tmp_path: pathlib.Path) -> None:
    world = _fake_world(tmp_path)
    bundle = _bundle(world)
    inventory = release.verify_inventory(bundle["inventory"])
    assert inventory["weight_symlink_count"] == 64
    assert inventory["weight_blob_count"] == 64
    assert inventory["metadata_symlink_count_retained"] == 32
    assert inventory["metadata_blob_count_retained"] == 32
    assert inventory["xet_leaf_count"] == 2
    assert inventory["xet_directory_count"] == 2
    assert inventory["authoritative_delete_views"] == 1
    assert inventory["globs_used"] is False
    assert inventory["recursive_delete_used"] is False
    assert bundle["status"] == "PASS_CONFIRMATION_REQUIRED"
    assert bundle["confirmation_token"] == release.derive_confirmation_token(bundle["audit"])


def test_incomplete_transfer_blob_is_never_a_weight_target_and_blocks(
    tmp_path: pathlib.Path,
) -> None:
    world = _fake_world(tmp_path)
    partial = world["layout"].blobs / (world["weight_blobs"][0].name + ".incomplete")
    partial.write_bytes(b"nonresumable transfer residue")
    with pytest.raises(
        release.Phase2ReleaseError, match="incomplete_transfer_artifacts"
    ):
        _bundle(world)
    assert partial.read_bytes() == b"nonresumable transfer residue"
    assert all(path.is_symlink() for path in world["weight_paths"])
    assert all(path.exists() for path in world["weight_blobs"])


@pytest.mark.parametrize("blocked", sorted(release._CHECK_NAMES))
def test_every_required_audit_blocker_prevents_confirmation(
    tmp_path: pathlib.Path, blocked: str
) -> None:
    world = _fake_world(tmp_path)
    bundle = _bundle(world, FakeProbe(blocked))
    assert bundle["status"] == "BLOCKED"
    assert blocked.upper() in bundle["audit"]["blockers"]
    assert bundle["confirmation_token"] is None
    with pytest.raises(release.Phase2ReleaseError, match="blocked audit"):
        release.derive_confirmation_token(bundle["audit"])
    assert all(path.exists() for path in world["weight_paths"])


def test_wrong_confirmation_token_writes_and_deletes_nothing(tmp_path: pathlib.Path) -> None:
    world = _fake_world(tmp_path)
    bundle = _bundle(world)
    with pytest.raises(release.Phase2ReleaseError, match="confirmation token"):
        _execute(world, bundle, "CONFIRM-NOT-THE-AUDIT")
    assert all(path.is_symlink() for path in world["weight_paths"])
    assert all(path.exists() for path in world["weight_blobs"])
    assert (world["layout"].xet / "root-object").exists()


def test_exact_release_deletes_only_weights_and_xet_and_seals_receipt(
    tmp_path: pathlib.Path,
) -> None:
    world = _fake_world(tmp_path)
    bundle = _bundle(world)
    token = release.derive_confirmation_token(bundle["audit"])
    receipt = _execute(world, bundle, token)
    verified = release.verify_receipt(receipt, bundle)
    assert verified["weight_symlink_count_deleted"] == 64
    assert verified["weight_blob_count_deleted"] == 64
    assert verified["xet_leaf_count_deleted"] == 2
    assert verified["xet_directory_count_deleted"] == 2
    assert all(not path.exists() for path in world["weight_paths"])
    assert all(not path.exists() for path in world["weight_blobs"])
    assert all(path.is_symlink() for path in world["metadata_paths"])
    assert all(path.read_bytes().startswith(b"metadata-") for path in world["metadata_blobs"])
    assert list(world["layout"].xet.iterdir()) == []
    assert (world["layout"].capsule / "payload.bin").read_bytes() == b"rollback payload"
    assert (world["layout"].recovery / "record.json").read_bytes() == b"recovery"
    assert (world["layout"].evidence / "audit-history.json").read_bytes() == b"evidence"
    assert (world["mop"] / "preserve").read_bytes() == b"MOP must remain"
    assert (world["shared"] / "preserve").read_bytes() == b"shared Xet must remain"
    assert verified["free_bytes_delta"] == (
        verified["free_bytes_after"] - verified["free_bytes_before"]
    )
    journal_path = pathlib.Path(verified["journal_path"])
    receipt_path = pathlib.Path(verified["receipt_path"])
    assert journal_path.is_file() and receipt_path.is_file()
    assert release._read_journal(journal_path)[-1]["event"] == "TERMINAL"
    reconciled = release.reconcile_release(
        world["layout"],
        bundle,
        confirmation_token=token,
        capsule_verifier=lambda _layout: copy.deepcopy(world["capsule"]),
        lease_paths=world["lease_paths"],
    )
    assert reconciled["seal_sha256"] == verified["seal_sha256"]


def test_blob_substitution_after_audit_blocks_before_any_unlink(tmp_path: pathlib.Path) -> None:
    world = _fake_world(tmp_path)
    bundle = _bundle(world)
    world["weight_blobs"][17].write_bytes(b"same-ish but substituted")
    token = release.derive_confirmation_token(bundle["audit"])
    with pytest.raises(release.Phase2ReleaseError, match="size|SHA-256|changed"):
        _execute(world, bundle, token)
    assert all(path.is_symlink() for path in world["weight_paths"])
    assert all(path.exists() for path in world["weight_blobs"])


def test_probe_time_same_size_retained_metadata_mutation_blocks_zero_delete(
    tmp_path: pathlib.Path,
) -> None:
    world = _fake_world(tmp_path)
    bundle = _bundle(world)

    class MutatingProbe(FakeProbe):
        def inspect(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]:
            checks = super().inspect(*args, **kwargs)
            target = world["metadata_blobs"][7]
            target.write_bytes(b"X" * target.stat().st_size)
            return checks

    token = release.derive_confirmation_token(bundle["audit"])
    with pytest.raises(
        release.Phase2ReleaseError, match="after the slow live probes|SHA-256"
    ):
        _execute(world, bundle, token, probe=MutatingProbe())
    assert all(path.is_symlink() for path in world["weight_paths"])
    assert all(path.exists() for path in world["weight_blobs"])
    journal, receipt = release._attempt_artifacts(world["layout"], bundle["seal_sha256"])
    assert not journal.exists()
    assert not receipt.exists()


def test_capsule_mutation_during_final_inventory_blocks_before_journal_start(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = _fake_world(tmp_path)
    bundle = _bundle(world)
    token = release.derive_confirmation_token(bundle["audit"])
    capsule_state = {"value": copy.deepcopy(world["capsule"])}
    original_inventory = release.build_exact_inventory
    calls = 0

    def mutating_inventory(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        result = original_inventory(*args, **kwargs)
        if calls == 2:
            changed = copy.deepcopy(capsule_state["value"])
            changed.pop("seal_sha256")
            changed["payload"]["sha256"] = "f" * 64
            capsule_state["value"] = phase1.seal_document(changed)
        return result

    monkeypatch.setattr(release, "build_exact_inventory", mutating_inventory)
    with pytest.raises(
        release.Phase2ReleaseError,
        match="rollback capsule changed during the slow final inventory",
    ):
        _execute(
            world,
            bundle,
            token,
            capsule_verifier=lambda _layout: copy.deepcopy(capsule_state["value"]),
        )
    assert calls == 2
    assert all(path.is_symlink() for path in world["weight_paths"])
    assert all(path.exists() for path in world["weight_blobs"])
    journal, receipt = release._attempt_artifacts(world["layout"], bundle["seal_sha256"])
    assert not journal.exists()
    assert not receipt.exists()


def test_fault_after_n_unlinks_has_durable_partial_receipt_and_reconciles(
    tmp_path: pathlib.Path,
) -> None:
    world = _fake_world(tmp_path)
    bundle = _bundle(world)
    token = release.derive_confirmation_token(bundle["audit"])
    with pytest.raises(release.Phase2PartialReleaseError) as caught:
        _execute(world, bundle, token, fault_after=5)
    partial = release.verify_receipt(caught.value.receipt, bundle)
    assert partial["terminal_status"] == "PARTIAL_FAILURE"
    assert partial["completed_count"] == 5
    assert partial["next_row"] == bundle["inventory"]["delete_entries"][5]
    assert partial["unattempted_rows"] == bundle["inventory"]["delete_entries"][5:]
    assert partial["preserved_node_status"]["status"] == "PASS"
    journal_path = pathlib.Path(partial["journal_path"])
    receipt_path = pathlib.Path(partial["receipt_path"])
    assert journal_path.is_file() and receipt_path.is_file()
    assert journal_path.stat().st_mode & 0o777 == 0o600
    assert receipt_path.stat().st_mode & 0o777 == 0o600
    records = release._read_journal(journal_path)
    assert [row["event"] for row in records] == [
        "START", *(["PREPARE", "COMMIT"] * 5), "TERMINAL"
    ]
    assert all(not path.exists() for path in world["weight_paths"][:5])
    assert all(path.is_symlink() for path in world["weight_paths"][5:])
    assert all(path.exists() for path in world["weight_blobs"])

    reconciled = release.reconcile_release(
        world["layout"],
        bundle,
        confirmation_token=token,
        capsule_verifier=lambda _layout: copy.deepcopy(world["capsule"]),
        lease_paths=world["lease_paths"],
    )
    assert reconciled["seal_sha256"] == partial["seal_sha256"]
    with pytest.raises(release.Phase2PartialReleaseError):
        _execute(world, bundle, token)
    assert all(path.is_symlink() for path in world["weight_paths"][5:])


def test_hard_crash_after_unlink_before_commit_reconciles_truthfully(
    tmp_path: pathlib.Path,
) -> None:
    world = _fake_world(tmp_path)
    bundle = _bundle(world)
    token = release.derive_confirmation_token(bundle["audit"])
    with pytest.raises(release._SimulatedHardCrash):
        _execute(world, bundle, token, fault_after_unlink_before_commit=0)

    journal_path, receipt_path = release._attempt_artifacts(
        world["layout"], bundle["seal_sha256"]
    )
    assert [row["event"] for row in release._read_journal(journal_path)] == [
        "START", "PREPARE"
    ]
    assert not receipt_path.exists()
    assert not world["weight_paths"][0].exists()
    assert world["weight_paths"][1].is_symlink()

    reconciled = release.reconcile_release(
        world["layout"],
        bundle,
        confirmation_token=token,
        capsule_verifier=lambda _layout: copy.deepcopy(world["capsule"]),
        lease_paths=world["lease_paths"],
    )
    verified = release.verify_receipt(reconciled, bundle)
    assert verified["terminal_status"] == "PARTIAL_FAILURE"
    assert verified["completed_count"] == 1
    assert verified["deletion_performed"] is True
    assert verified["completed_rows"] == bundle["inventory"]["delete_entries"][:1]
    assert verified["next_row"] == bundle["inventory"]["delete_entries"][1]
    assert verified["intent_outcomes"] == [
        {
            "delete_index": 0,
            "row": bundle["inventory"]["delete_entries"][0],
            "outcome": "RECONCILED_ABSENT_AFTER_DURABLE_PREPARE",
            "certainty": "PATH_ABSENCE_CONFIRMED_ORIGINAL_INODE_UNLINK_INFERRED",
            "journal_record_seal_sha256": verified["intent_outcomes"][0][
                "journal_record_seal_sha256"
            ],
        }
    ]
    assert [row["event"] for row in release._read_journal(journal_path)] == [
        "START", "PREPARE", "COMMIT", "TERMINAL"
    ]
    assert world["weight_paths"][1].is_symlink()
    second = release.reconcile_release(
        world["layout"],
        bundle,
        confirmation_token=token,
        capsule_verifier=lambda _layout: copy.deepcopy(world["capsule"]),
        lease_paths=world["lease_paths"],
    )
    assert second["seal_sha256"] == verified["seal_sha256"]
    assert world["weight_paths"][1].is_symlink()
    assert all(path.exists() for path in world["weight_blobs"])


def test_symlink_swap_never_touches_external_victim(tmp_path: pathlib.Path) -> None:
    world = _fake_world(tmp_path)
    bundle = _bundle(world)
    victim = tmp_path / "victim"
    victim.write_bytes(b"never touch")
    blob = world["weight_blobs"][5]
    blob.unlink()
    blob.symlink_to(victim)
    token = release.derive_confirmation_token(bundle["audit"])
    with pytest.raises(release.Phase2ReleaseError, match="not a symlink|not.*regular|type/size"):
        _execute(world, bundle, token)
    assert victim.read_bytes() == b"never touch"
    assert all(path.is_symlink() for path in world["weight_paths"])


def test_held_download_lease_blocks_before_revalidation_or_delete(tmp_path: pathlib.Path) -> None:
    world = _fake_world(tmp_path)
    bundle = _bundle(world)
    lease = world["layout"].evidence / release.DOWNLOAD_LEASE_NAME
    descriptor = os.open(lease, os.O_RDWR | os.O_CREAT, 0o600)
    os.chmod(lease, 0o600)
    import fcntl

    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(release.Phase2ReleaseError, match="lease is already held"):
            _execute(world, bundle, release.derive_confirmation_token(bundle["audit"]))
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
    assert all(path.is_symlink() for path in world["weight_paths"])


def test_resealed_cross_bundle_substitution_is_rejected(tmp_path: pathlib.Path) -> None:
    first = _fake_world(tmp_path / "one")
    second = _fake_world(tmp_path / "two")
    left = _bundle(first)
    right = _bundle(second)
    body = copy.deepcopy(left)
    body.pop("seal_sha256")
    body["audit"] = right["audit"]
    body["audit_seal_sha256"] = right["audit_seal_sha256"]
    body["confirmation_token"] = right["confirmation_token"]
    substituted = phase1.seal_document(body)
    with pytest.raises(release.Phase2ReleaseError, match="binding|belongs"):
        release.verify_bundle(substituted)


def test_module_contains_no_glob_or_recursive_delete_primitive() -> None:
    source = pathlib.Path(release.__file__).read_text()
    assert ".glob(" not in source
    assert ".rglob(" not in source
    assert "rmtree" not in source
    assert "rm -rf" not in source
    assert "os.unlink(" in source
    assert "os.rmdir(" in source
