from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import pathlib
import stat
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[3]
CONDENSE = ROOT / "tools" / "condense"
sys.path.insert(0, str(CONDENSE))
MODULE_PATH = CONDENSE / "appendix_physical_release_packet.py"
SPEC = importlib.util.spec_from_file_location("appendix_physical_release_packet", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_json(path: pathlib.Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _reference(path: pathlib.Path, base: pathlib.Path, **extra: object) -> dict:
    raw = path.read_bytes()
    return {
        "path": path.relative_to(base).as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        **extra,
    }


def _boundary() -> tuple[dict, dict]:
    observed = 123456789
    owner_snapshot = {
        "schema": "hawking.appendix_owner_snapshot.v1",
        "exclusive_shared_heavy_lease_held": True,
        "lock_path": "/tmp/studio_heavy.lock",
        "lock_device": 1,
        "lock_inode": 2,
        "owners": [],
        "observed_at_unix_ns": observed,
    }
    owner_sha = MODULE.canonical_sha256(owner_snapshot)
    resource = {"ok": True, "pressure_level": 1, "swap_used_mb": 0.0}
    observation = MODULE._stamp({
        "schema": MODULE.BOUNDARY_OBSERVATION_SCHEMA,
        "observer_state_sha256": "a" * 64,
        "final_packet_file_sha256": "b" * 64,
        "final_packet_canonical_sha256": "c" * 64,
        "all_recorded_hashes_verified": True,
        "verified_reference_count": 9,
        "verified_references_sha256": "d" * 64,
        "owner_snapshot": owner_snapshot,
        "owner_snapshot_sha256": owner_sha,
        "resource_snapshot": resource,
        "ram_swap_guard_healthy": True,
        "observed_at_unix_ns": observed,
        "default_mutation_requested": False,
    }, "observation_sha256")
    attestation = MODULE._stamp({
        "schema": MODULE.evidence_gate.RELEASE_BOUNDARY_SCHEMA,
        "final_interpretation_ready": True,
        "final_packet_sha256": "b" * 64,
        "observer_state_sha256": "a" * 64,
        "all_recorded_hashes_verified": True,
        "active_heavy_owner_count": 0,
        "owner_snapshot_sha256": owner_sha,
        "ram_swap_guard_healthy": True,
        "observed_at_unix_ns": observed,
    }, "attestation_sha256")
    return observation, attestation


def _final_fixture(root: pathlib.Path) -> tuple[pathlib.Path, dict]:
    workspace = root / "workspace"
    post = workspace / "post"
    frozen_root = post / "final_inputs"
    report_root = workspace
    post.mkdir(parents=True)
    frozen_root.mkdir()
    (post / "observation.json").write_text("observation\n", encoding="utf-8")

    plan = {"plan_sha256": "1" * 64}
    campaign = {"campaign_sha256": "2" * 64}
    report_index = {"index_sha256": "3" * 64}
    frozen_paths = {}
    for name, value in (
        ("campaign_plan", plan), ("campaign", campaign), ("report_index", report_index),
    ):
        path = frozen_root / f"{name}.json"
        _write_json(path, value)
        frozen_paths[name] = path

    reports = {}
    checkpoint_paths = {}
    for group in ("sub-120B", "120B"):
        report_path = workspace / "reports" / group / "report.json"
        _write_json(report_path, {"group": group, "complete": True})
        report_ref = _reference(report_path, report_root, complete=True, kind="evidence")
        reports[group] = report_ref
        checkpoint = MODULE._stamp({
            "schema": "hawking.doctor_v5_ultra_report_checkpoint.v1",
            "verified": True,
            "source_deletion_permitted": False,
            "report_artifact": {
                "path": report_ref["path"], "sha256": report_ref["sha256"],
                "bytes": report_ref["bytes"],
            },
        }, "checkpoint_sha256")
        checkpoint_path = workspace / "reports" / group / "checkpoint.json"
        _write_json(checkpoint_path, checkpoint)
        checkpoint_paths[group] = checkpoint_path

    final_packet = MODULE._stamp({
        "schema": MODULE.FINAL_PACKET_SCHEMA,
        "ready": True,
        "plan_sha256": plan["plan_sha256"],
        "campaign_sha256": campaign["campaign_sha256"],
        "source_observation": _reference(post / "observation.json", post),
        "artifact_path_bases": {
            "source_observation": str(post),
            "frozen_inputs": str(post),
            "reports_and_checkpoints": str(report_root),
        },
        "frozen_inputs": {
            name: _reference(path, post) for name, path in frozen_paths.items()
        },
        "reports": reports,
        "accepted_report_checkpoint_groups": ["120B", "sub-120B"],
        "accepted_report_checkpoints": {
            group: _reference(
                path, report_root,
                checkpoint_sha256=json.loads(path.read_text())["checkpoint_sha256"],
            )
            for group, path in checkpoint_paths.items()
        },
        "source_deletion_permitted": False,
    }, "packet_sha256")
    final_path = post / "final_interpretation_packet.json"
    _write_json(final_path, final_packet)
    observer = MODULE._stamp({
        "schema": MODULE.OBSERVER_SCHEMA,
        "final_interpretation_ready": True,
        "final_interpretation_packet": _reference(final_path, post),
        "source_deletion_permitted": False,
    }, "state_sha256")
    return post, observer


def _source_manifest(
    observation: dict | None = None, boundary: dict | None = None,
) -> dict:
    if observation is None or boundary is None:
        observation, boundary = _boundary()
    return MODULE._stamp({
        "schema": MODULE.evidence_gate.SOURCE_MANIFEST_SCHEMA,
        "source_base_commit": "a" * 40,
        "source_base_commit_role": "repository-base-only-not-byte-authority",
        "scope": "isolated-exact-critical-source-capsule",
        "release_boundary_attestation_sha256": boundary["attestation_sha256"],
        "release_boundary_observation_sha256": observation["observation_sha256"],
        "required_paths_sha256": MODULE.canonical_sha256(
            sorted(MODULE.evidence_gate.REQUIRED_SOURCE_PATHS)
        ),
        "entry_count": len(MODULE.evidence_gate.REQUIRED_SOURCE_PATHS),
        "symlink_count": 0,
        "entries": (entries := [
            {"path": path, "sha256": f"{index + 1:064x}", "size_bytes": 1}
            for index, path in enumerate(sorted(MODULE.evidence_gate.REQUIRED_SOURCE_PATHS))
        ]),
        "capsule_sha256": MODULE.canonical_sha256(entries),
    }, "manifest_sha256")


def _toolchain() -> tuple[dict, str]:
    host = "arm64-apple-darwin"
    cargo_version = "cargo 1.90.0\nrelease: 1.90.0\nhost: arm64-apple-darwin\n"
    rustc_version = "rustc 1.90.0\nbinary: rustc\nhost: arm64-apple-darwin\n"
    selection_environment = {
        "CARGO_HOME": "/home/test/.cargo", "HOME": "/home/test",
        "LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin",
        "RUSTUP_HOME": "/home/test/.rustup", "TZ": "UTC",
    }
    version_environment = {
        "LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin", "TZ": "UTC",
    }

    def row(name: str, version: str, digest: str) -> dict:
        path = f"/toolchain/bin/{name}"
        return {
            "invocation_path": path,
            "resolved_binary": {"path": path, "sha256": digest * 64, "size_bytes": 1},
            "version_verbose": version,
            "version_verbose_sha256": MODULE.canonical_sha256(version),
            "selection": {
                "mode": "direct", "discovered_invocation_path": path,
                "discovered_resolved_binary": {
                    "path": path, "sha256": digest * 64, "size_bytes": 1,
                },
                "selection_environment": selection_environment,
                "selector_argv_sha256": None,
                "selected_invocation_path": path,
                "version_probe_environment": version_environment,
                "version_probe_argv_sha256": MODULE.canonical_sha256([
                    path, "-Vv" if name == "cargo" else "-vV",
                ]),
            },
        }

    return {
        "cargo": row("cargo", cargo_version, "6"),
        "rustc": row("rustc", rustc_version, "7"),
    }, host


def _release_receipt(source: dict, boundary: dict | None = None) -> dict:
    if boundary is None:
        _observation, boundary = _boundary()
    cargo = next(row for row in source["entries"] if row["path"] == "Cargo.lock")
    toolchain, host = _toolchain()
    target_directory = MODULE._unique_target_dir(boundary, source)
    device = {
        "path": str((target_directory / host / "release" / "hawking-tq-device-probe").resolve()),
        "sha256": "c" * 64, "size_bytes": 1,
    }
    spec = {
        "path": str((target_directory / host / "release" / "hawking-tq-spec-probe").resolve()),
        "sha256": "d" * 64, "size_bytes": 1,
    }
    artifacts = {
        "hawking-tq-device-probe": {
            "target_name": "hawking-tq-device-probe", "target_kind": ["bin"],
            "fresh": False, "executable": device,
        },
        "hawking-tq-spec-probe": {
            "target_name": "hawking-tq-spec-probe", "target_kind": ["bin"],
            "fresh": False, "executable": spec,
        },
    }
    closures = {}
    for name, probe_path, executable in (
        ("hawking-tq-device-probe", MODULE.ROOT / "crates/hawking/src/tq_device_probe.rs", device),
        ("hawking-tq-spec-probe", MODULE.ROOT / "crates/hawking/src/tq_spec_probe.rs", spec),
    ):
        entries = [{
            "path": str(probe_path.resolve()), "sha256": "7" * 64, "size_bytes": 1,
        }]
        closures[name] = {
            "dep_info": {
                "path": str(pathlib.Path(executable["path"]).with_suffix(".d")),
                "sha256": "8" * 64, "size_bytes": 1,
            },
            "entry_count": 1, "entries": entries,
            "closure_sha256": MODULE.canonical_sha256(entries),
        }
    environment = MODULE._build_environment(toolchain, target_directory)
    path_directories = [
        {
            "path": raw, "device": 1, "inode": index + 1, "mode": 0o755,
            "executable_entries": [],
            "executable_entries_sha256": MODULE.canonical_sha256([]),
        }
        for index, raw in enumerate(environment["PATH"].split(":"))
    ]
    path_snapshot = {
        "value": environment["PATH"], "directories": path_directories,
        "snapshot_sha256": MODULE.canonical_sha256(path_directories),
    }
    configuration_entries: list[dict] = []
    context = MODULE._stamp({
        "schema": MODULE.BUILD_CONTEXT_SCHEMA,
        "environment_keys": sorted(environment),
        "environment_sha256": MODULE.canonical_sha256(environment),
        "path_snapshot": path_snapshot,
        "configuration_snapshot": {
            "entries": configuration_entries,
            "snapshot_sha256": MODULE.canonical_sha256(configuration_entries),
        },
        "cargo_home": {
            "path": environment["CARGO_HOME"], "device": 1, "inode": 20, "mode": 0o700,
        },
        "isolated_directories": {
            name: {"path": environment[name], "device": 1, "inode": 30 + index, "mode": 0o700}
            for index, name in enumerate(("HOME", "RUSTUP_HOME", "TMPDIR"))
        },
        "toolchain_selection_sha256": MODULE.canonical_sha256({
            name: toolchain[name]["selection"] for name in ("cargo", "rustc")
        }),
        "ambient_environment_inherited": False,
    }, "context_sha256")
    return MODULE._stamp({
        "schema": MODULE.evidence_gate.RELEASE_BUILD_SCHEMA,
        "source_base_commit": source["source_base_commit"],
        "source_base_commit_role": "repository-base-only-not-byte-authority",
        "source_manifest_sha256": source["manifest_sha256"],
        "source_authority_capsule_sha256": source["capsule_sha256"],
        "cargo_lock_sha256": cargo["sha256"],
        "release_boundary_attestation_sha256": boundary["attestation_sha256"],
        "build_argv_sha256": MODULE.canonical_sha256(
            MODULE._exact_build_argv(
                toolchain["cargo"]["invocation_path"], host, target_directory,
            )
        ),
        "profile": "release", "features": ["tq"], "target_host": host,
        "target_directory": str(target_directory),
        "toolchain": toolchain,
        "build_environment": environment,
        "build_execution_context": context,
        "compiler_artifacts": artifacts,
        "compiled_source_closures": closures,
        "success": True,
        "built_at_unix_ns": boundary["observed_at_unix_ns"],
        "build_log": {"path": "/tmp/build.log", "sha256": "b" * 64, "size_bytes": 1},
        "probes": {"device": device, "spec": spec},
        "runtime_defaults_changed": False,
    }, "receipt_sha256")


def test_final_packet_verifier_checks_every_recorded_file_and_rejects_tamper(
    tmp_path: pathlib.Path,
) -> None:
    post, observer = _final_fixture(tmp_path)
    errors, verified = MODULE.verify_final_packet_references(
        observer, post_root=post, workspace_root=tmp_path / "workspace",
    )
    assert errors == []
    assert verified is not None and verified["verified_reference_count"] == 9
    report = tmp_path / "workspace" / "reports" / "120B" / "report.json"
    report.write_text("tampered\n", encoding="utf-8")
    errors, verified = MODULE.verify_final_packet_references(
        observer, post_root=post, workspace_root=tmp_path / "workspace",
    )
    assert verified is None
    assert any("byte identity mismatch" in error for error in errors)


def test_final_packet_verifier_rejects_symlinked_reference(tmp_path: pathlib.Path) -> None:
    post, observer = _final_fixture(tmp_path)
    packet_path = post / observer["final_interpretation_packet"]["path"]
    real = post / "real-final.json"
    packet_path.rename(real)
    packet_path.symlink_to(real)
    raw = real.read_bytes()
    observer["final_interpretation_packet"]["sha256"] = hashlib.sha256(raw).hexdigest()
    observer["final_interpretation_packet"]["bytes"] = len(raw)
    observer = MODULE._stamp(observer, "state_sha256")
    errors, verified = MODULE.verify_final_packet_references(
        observer, post_root=post, workspace_root=tmp_path / "workspace",
    )
    assert verified is None
    assert any("symlink" in error for error in errors)


def test_boundary_verifier_detects_forged_owner_snapshot_and_stale_observer() -> None:
    observation, attestation = _boundary()
    assert MODULE.validate_release_boundary_attestation(
        attestation, observation=observation,
    ) == []
    forged = copy.deepcopy(observation)
    forged["owner_snapshot"]["owners"] = [{"pid": 7}]
    forged = MODULE._stamp(forged, "observation_sha256")
    errors = MODULE.validate_release_boundary_attestation(attestation, observation=forged)
    assert any("owner-free" in error for error in errors)
    current = MODULE._stamp({
        "schema": MODULE.OBSERVER_SCHEMA,
        "final_interpretation_ready": True,
        "final_interpretation_packet": {"path": "x", "sha256": "e" * 64, "bytes": 1},
        "source_deletion_permitted": False,
    }, "state_sha256")
    errors = MODULE.validate_release_boundary_attestation(
        attestation, observation=observation, observer=current,
    )
    assert any("stale" in error for error in errors)


def test_source_manifest_is_exact_and_build_receipt_rejects_argv_or_lock_cross_credit() -> None:
    source = _source_manifest()
    assert MODULE.validate_clean_source_manifest(source) == []
    source["entries"].pop()
    source["entry_count"] -= 1
    source["capsule_sha256"] = MODULE.canonical_sha256(source["entries"])
    source = MODULE._stamp(source, "manifest_sha256")
    assert any("exactly" in error for error in MODULE.validate_clean_source_manifest(source))

    complete = _source_manifest()
    receipt = _release_receipt(complete)
    assert MODULE.validate_release_build_receipt(receipt, source_manifest=complete) == []
    receipt["build_argv_sha256"] = "e" * 64
    receipt["cargo_lock_sha256"] = "f" * 64
    receipt = MODULE._stamp(receipt, "receipt_sha256")
    errors = MODULE.validate_release_build_receipt(receipt, source_manifest=complete)
    assert any("pinned toolchain/host" in error for error in errors)
    assert any("Cargo.lock" in error for error in errors)


def test_source_capsule_ignores_unrelated_git_dirt_but_requires_two_stable_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(MODULE, "_git", lambda *argv: "a" * 40 + "\n")

    def identity(path: pathlib.Path, **_kwargs: object) -> dict:
        relative = path.relative_to(MODULE.ROOT).as_posix()
        digest = hashlib.sha256(relative.encode()).hexdigest()
        return {"path": str(path.resolve()), "sha256": digest, "size_bytes": 1}

    monkeypatch.setattr(MODULE, "_stable_file_identity", identity)
    monkeypatch.setattr(
        MODULE.physical_counter_attestation, "file_identity",
        lambda path: {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.relative_to(MODULE.ROOT).as_posix().encode()).hexdigest(),
            "size_bytes": 1,
        },
    )
    observation, boundary = _boundary()
    capsule = MODULE.build_clean_source_manifest(
        release_boundary=boundary, boundary_observation=observation,
    )
    assert capsule["source_base_commit_role"] == "repository-base-only-not-byte-authority"
    assert capsule["entry_count"] == len(MODULE.evidence_gate.REQUIRED_SOURCE_PATHS)

    calls: dict[str, int] = {}

    def unstable(path: pathlib.Path, **_kwargs: object) -> dict:
        key = path.relative_to(MODULE.ROOT).as_posix()
        calls[key] = calls.get(key, 0) + 1
        suffix = "changed" if calls[key] > 1 and key == sorted(calls)[0] else key
        return {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(suffix.encode()).hexdigest(),
            "size_bytes": 1,
        }

    monkeypatch.setattr(MODULE, "_stable_file_identity", unstable)
    with pytest.raises(MODULE.EvidenceError, match="two-pass"):
        MODULE.build_clean_source_manifest(
            release_boundary=boundary, boundary_observation=observation,
        )


def test_release_build_is_bound_to_exact_boundary_and_capsule_authority() -> None:
    source = _source_manifest()
    _observation, boundary = _boundary()
    receipt = _release_receipt(source, boundary)
    assert MODULE.validate_release_build_receipt(
        receipt, source_manifest=source, release_boundary=boundary,
    ) == []
    other = copy.deepcopy(boundary)
    other["observed_at_unix_ns"] += 1
    other = MODULE._stamp(other, "attestation_sha256")
    errors = MODULE.validate_release_build_receipt(
        receipt, source_manifest=source, release_boundary=other,
    )
    assert any("aggregate release boundary" in error for error in errors)
    assert any("predates" in error for error in errors)


def test_release_build_rejects_env_substitution_and_stale_default_probe_credit() -> None:
    source = _source_manifest()
    receipt = _release_receipt(source)
    receipt["build_environment"]["CARGO_TARGET_DIR"] = "/tmp/substituted-target"
    receipt["toolchain"]["cargo"]["invocation_path"] = "/tmp/substituted-cargo"
    receipt["probes"]["device"] = {
        "path": str((MODULE.ROOT / "target" / "release" / "hawking-tq-device-probe").resolve()),
        "sha256": "e" * 64, "size_bytes": 1,
    }
    receipt = MODULE._stamp(receipt, "receipt_sha256")
    errors = MODULE.validate_release_build_receipt(receipt, source_manifest=source)
    assert any("deterministic" in error or "target-isolated" in error for error in errors)
    assert any("pinned toolchain/host" in error for error in errors)
    assert any("not the executable emitted by Cargo" in error for error in errors)


def test_release_build_rejects_cargo_fresh_cache_credit() -> None:
    source = _source_manifest()
    receipt = _release_receipt(source)
    receipt["compiler_artifacts"]["hawking-tq-device-probe"]["fresh"] = True
    receipt = MODULE._stamp(receipt, "receipt_sha256")
    errors = MODULE.validate_release_build_receipt(receipt, source_manifest=source)
    assert any("newly built" in error for error in errors)


def test_release_build_rejects_existing_boundary_target_before_cargo(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation, boundary = _boundary()
    source = _source_manifest(observation, boundary)
    workspace = tmp_path / "workspace"
    target = workspace / "target" / "appendix-release" / "already-present"
    target.mkdir(parents=True)
    toolchain, host = _toolchain()
    monkeypatch.setattr(MODULE, "build_clean_source_manifest", lambda **_kwargs: source)
    monkeypatch.setattr(MODULE, "_tool_binding", lambda name: toolchain[name])
    monkeypatch.setattr(MODULE, "_verify_toolchain_live", lambda _value: None)
    monkeypatch.setattr(MODULE, "_rustc_host", lambda _value: host)
    monkeypatch.setattr(MODULE, "_unique_target_dir", lambda *_args: target)
    monkeypatch.setattr(MODULE, "ROOT", workspace)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("Cargo ran despite an existing unique target")

    monkeypatch.setattr(MODULE.subprocess, "run", forbidden)
    admission = MODULE.ReleaseAdmission(
        lease=None,
        observer={"state_sha256": "a" * 64},
        boundary_observation=observation,
        boundary_attestation=boundary,
    )
    with pytest.raises(MODULE.EvidenceError, match="cached target credit is forbidden"):
        MODULE.run_release_build(
            admission, source_manifest=source, build_log=tmp_path / "build.log",
        )


def test_source_and_compiler_closures_reject_missing_dependencies() -> None:
    assert "tools/condense/ram_scheduler.py" in MODULE.evidence_gate.REQUIRED_SOURCE_PATHS
    source = _source_manifest()
    source["entries"] = [
        row for row in source["entries"]
        if row["path"] != "tools/condense/ram_scheduler.py"
    ]
    source["entry_count"] = len(source["entries"])
    source["capsule_sha256"] = MODULE.canonical_sha256(source["entries"])
    source = MODULE._stamp(source, "manifest_sha256")
    assert any("exactly" in error for error in MODULE.validate_clean_source_manifest(source))

    complete = _source_manifest()
    receipt = _release_receipt(complete)
    closure = receipt["compiled_source_closures"]["hawking-tq-device-probe"]
    closure["entries"] = []
    closure["entry_count"] = 0
    closure["closure_sha256"] = MODULE.canonical_sha256([])
    receipt = MODULE._stamp(receipt, "receipt_sha256")
    errors = MODULE.validate_release_build_receipt(receipt, source_manifest=complete)
    assert any("compiled source closure is empty" in error for error in errors)
    assert any("omits its direct probe source" in error for error in errors)


def test_corpus_receipt_is_owner_free_boundary_bound_and_detects_added_file(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "failed.partial").write_bytes(b"negative")
    (tmp_path / "result.json").write_text("{}\n", encoding="utf-8")
    index = MODULE.appendix_corpus.build_index(
        tmp_path, active_owners=[], source_base_commit="a" * 40,
    )
    observation, attestation = _boundary()
    monkeypatch.setattr(MODULE.spec_reentry_scaffold, "active_heavy_owners", lambda: [])
    pre_receipt, _pre_attestation = MODULE.build_corpus_verification(
        index, boundary_attestation=attestation, boundary_observation=observation,
        verified_at_unix_ns=98, verification_phase="pre_release_build",
    )
    receipt, corpus_attestation = MODULE.build_corpus_verification(
        index, boundary_attestation=attestation, boundary_observation=observation,
        verified_at_unix_ns=99, verification_phase="post_release_build",
        parent_verification_receipt=pre_receipt,
    )
    assert MODULE.validate_corpus_verification(
        corpus_attestation, receipt=receipt, index=index,
        boundary_attestation=attestation, boundary_observation=observation,
        parent_verification_receipt=pre_receipt,
    ) == []
    changed_boundary = copy.deepcopy(attestation)
    changed_boundary["observed_at_unix_ns"] += 1
    changed_boundary = MODULE._stamp(changed_boundary, "attestation_sha256")
    errors = MODULE.validate_corpus_verification(
        corpus_attestation, receipt=receipt, index=index,
        boundary_attestation=changed_boundary, boundary_observation=observation,
    )
    assert any("exact release boundary" in error or "observation field" in error for error in errors)
    (tmp_path / "late.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(MODULE.EvidenceError, match="added_files"):
        MODULE.build_corpus_verification(
            index, boundary_attestation=attestation, boundary_observation=observation,
            verified_at_unix_ns=100, verification_phase="post_release_build",
            parent_verification_receipt=pre_receipt,
        )


def test_evidence_manifest_rejects_duplicate_bytes_even_at_distinct_paths(
    tmp_path: pathlib.Path,
) -> None:
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    first.write_text('{"same":true}\n', encoding="utf-8")
    second.write_bytes(first.read_bytes())
    with pytest.raises(MODULE.EvidenceError, match="reuses"):
        MODULE.build_evidence_manifest("device", [first, second])


def test_assembler_is_deterministic_and_never_writes_through_a_red_gate(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    device_paths = []
    for name in ("z", "a"):
        path = tmp_path / f"device-{name}.json"
        _write_json(path, {"cell_id": name})
        device_paths.append(path)
    spec_paths = []
    for runtime in ("computed", "stored"):
        path = tmp_path / f"spec-{runtime}.json"
        _write_json(path, {"runtime_path": runtime})
        spec_paths.append(path)
    device_manifest = MODULE.build_evidence_manifest("device", device_paths)
    spec_manifest = MODULE.build_evidence_manifest("spec", spec_paths)
    observation, attestation = _boundary()
    monkeypatch.setattr(MODULE, "validate_release_boundary_attestation", lambda *a, **k: [])
    monkeypatch.setattr(MODULE, "validate_corpus_verification", lambda *a, **k: [])
    monkeypatch.setattr(MODULE, "validate_clean_source_manifest", lambda *a, **k: [])
    monkeypatch.setattr(MODULE, "validate_release_build_receipt", lambda *a, **k: [])
    monkeypatch.setattr(MODULE.evidence_gate, "validate_gate", lambda *a, **k: [])
    kwargs = dict(
        boundary_attestation=attestation, boundary_observation=observation,
        corpus_index={"index_sha256": "1" * 64},
        corpus_attestation={"attestation_sha256": "2" * 64},
        corpus_prebuild_receipt={"verification_receipt_sha256": "0" * 64},
        corpus_receipt={"verification_receipt_sha256": "3" * 64},
        source_manifest={"manifest_sha256": "4" * 64},
        release_build={"receipt_sha256": "5" * 64},
        device_manifest=device_manifest, spec_manifest=spec_manifest,
    )
    packet1, receipt1 = MODULE.assemble_physical_packet(**kwargs)
    packet2, receipt2 = MODULE.assemble_physical_packet(**kwargs)
    assert packet1 == packet2 and receipt1 == receipt2
    assert [row["cell_id"] for row in packet1["device_evidence"]] == ["a", "z"]
    assert [row["runtime_path"] for row in packet1["spec_evidence"]] == ["stored", "computed"]
    monkeypatch.setattr(MODULE.evidence_gate, "validate_gate", lambda *a, **k: ["cross-credit"])
    with pytest.raises(MODULE.EvidenceError, match="cross-credit"):
        MODULE.assemble_physical_packet(**kwargs)


def test_actual_command_fails_before_any_builder_when_release_is_closed(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(MODULE, "current_status", lambda **_kwargs: {
        "prelease_admission_ready": False, "blockers": ["Doctor final is false"],
    })
    called = False

    def forbidden() -> dict:
        nonlocal called
        called = True
        raise AssertionError("builder ran behind closed boundary")

    monkeypatch.setattr(MODULE, "build_clean_source_manifest", forbidden)
    rc = MODULE.main(["source-manifest", "--output", str(tmp_path / "manifest.json")])
    assert rc == 75
    assert called is False
    assert not (tmp_path / "manifest.json").exists()


def test_immutable_group_never_replaces_or_partially_installs(tmp_path: pathlib.Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    second.write_text('{"prior":true}\n', encoding="utf-8")
    with pytest.raises(MODULE.EvidenceError, match="replace"):
        MODULE._atomic_json_group(((first, {"new": 1}), (second, {"different": 2})))
    assert not first.exists()
    assert second.read_text(encoding="utf-8") == '{"prior":true}\n'


def test_immutable_group_seals_permissions_and_rejects_writable_existing_bytes(
    tmp_path: pathlib.Path,
) -> None:
    sealed = tmp_path / "sealed.json"
    MODULE._atomic_bytes_group(((sealed, b"sealed\n"),))
    assert sealed.read_bytes() == b"sealed\n"
    assert stat.S_IMODE(sealed.stat().st_mode) == 0o400

    writable = tmp_path / "writable.json"
    writable.write_bytes(b"same\n")
    writable.chmod(0o600)
    with pytest.raises(MODULE.EvidenceError, match="remains writable"):
        MODULE._atomic_bytes_group(((writable, b"same\n"),))
    assert writable.read_bytes() == b"same\n"
    assert stat.S_IMODE(writable.stat().st_mode) == 0o600


def test_immutable_group_rolls_back_on_short_write(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "receipt.json"
    original_write = MODULE.os.write
    calls = 0

    def short_write(descriptor: int, raw: object) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            return 0
        return original_write(descriptor, raw)

    monkeypatch.setattr(MODULE.os, "write", short_write)
    with pytest.raises(OSError, match="short write"):
        MODULE._atomic_bytes_group(((output, b"sealed\n"),))
    assert not output.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_immutable_group_rejects_symlink_and_parent_replacement_without_redirect(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    attacker = tmp_path / "attacker"
    attacker.mkdir()
    symlink_parent = tmp_path / "symlink-parent"
    symlink_parent.symlink_to(attacker, target_is_directory=True)
    with pytest.raises((OSError, MODULE.EvidenceError)):
        MODULE._atomic_bytes_group(((symlink_parent / "escaped.json", b"no\n"),))
    assert not (attacker / "escaped.json").exists()

    victim = tmp_path / "victim"
    victim.mkdir()
    moved = tmp_path / "victim-retained"
    original_link = MODULE.os.link
    replaced = False

    def replace_parent_after_link(*args: object, **kwargs: object) -> None:
        nonlocal replaced
        original_link(*args, **kwargs)
        if not replaced:
            replaced = True
            victim.rename(moved)
            victim.symlink_to(attacker, target_is_directory=True)

    monkeypatch.setattr(MODULE.os, "link", replace_parent_after_link)
    with pytest.raises(MODULE.EvidenceError, match="parent changed|parent was replaced"):
        MODULE._atomic_bytes_group(((victim / "receipt.json", b"sealed\n"),))
    assert not (attacker / "receipt.json").exists()
    assert not (moved / "receipt.json").exists()


def test_release_build_environment_ignores_ambient_variables_and_context_detects_drift(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    toolchain, _host = _toolchain()
    cargo_home = tmp_path / "cargo-home"
    cargo_home.mkdir()
    target = tmp_path / "target" / "appendix-release" / ("f" * 64)
    monkeypatch.setenv("RUSTFLAGS", "--cfg ambient_attack")
    monkeypatch.setenv("CARGO_BUILD_TARGET", "attacker-target")
    monkeypatch.setenv("SECRET_SHOULD_NOT_LEAK", "secret")
    first = MODULE._build_environment(
        toolchain, target, cargo_home=cargo_home, lease_fd=19,
    )
    monkeypatch.setenv("RUSTFLAGS", "--cfg second_attack")
    monkeypatch.setenv("PATH", "/attacker/bin")
    second = MODULE._build_environment(
        toolchain, target, cargo_home=cargo_home, lease_fd=19,
    )
    assert first == second
    assert set(first) == MODULE.DETERMINISTIC_BUILD_ENVIRONMENT_KEYS
    assert "RUSTFLAGS" not in first and "SECRET_SHOULD_NOT_LEAK" not in first

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    executable = bin_dir / "helper"
    executable.write_text("helper\n", encoding="utf-8")
    executable.chmod(0o700)
    for name in ("HOME", "RUSTUP_HOME", "TMPDIR"):
        pathlib.Path(first[name]).mkdir(parents=True, exist_ok=True)
    first["PATH"] = str(bin_dir)
    context = MODULE._build_execution_context(first, toolchain=toolchain)
    assert MODULE._validate_build_execution_context_live(
        context, environment=first, toolchain=toolchain,
    ) == []
    (cargo_home / "config.toml").write_text("[build]\nincremental = true\n", encoding="utf-8")
    errors = MODULE._validate_build_execution_context_live(
        context, environment=first, toolchain=toolchain,
    )
    assert any("configuration changed" in error for error in errors)


def test_final_assembly_cas_rejects_stale_corpus_and_final_references(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    result = corpus_root / "result.json"
    result.write_text('{"status":"complete"}\n', encoding="utf-8")
    observation, boundary = _boundary()
    monkeypatch.setattr(MODULE, "validate_release_boundary_attestation", lambda *a, **k: [])
    monkeypatch.setattr(MODULE.spec_reentry_scaffold, "active_heavy_owners", lambda: [])
    verified = {
        "final_packet_file_sha256": observation["final_packet_file_sha256"],
        "final_packet": {"packet_sha256": observation["final_packet_canonical_sha256"]},
        "verified_reference_count": observation["verified_reference_count"],
        "verified_references_sha256": observation["verified_references_sha256"],
    }
    monkeypatch.setattr(MODULE, "verify_final_packet_references", lambda _observer: ([], verified))
    index = MODULE.appendix_corpus.build_index(
        corpus_root, active_owners=[], source_base_commit="a" * 40,
    )
    pre, _ = MODULE.build_corpus_verification(
        index, boundary_attestation=boundary, boundary_observation=observation,
        verified_at_unix_ns=100, verification_phase="pre_release_build",
    )
    post, attestation = MODULE.build_corpus_verification(
        index, boundary_attestation=boundary, boundary_observation=observation,
        verified_at_unix_ns=101, verification_phase="post_release_build",
        parent_verification_receipt=pre,
    )
    kwargs = dict(
        current_observer={"state_sha256": "a" * 64},
        boundary_attestation=boundary, boundary_observation=observation,
        corpus_index=index, corpus_prebuild_receipt=pre,
        corpus_receipt=post, corpus_attestation=attestation,
    )
    MODULE._revalidate_final_assembly_cas(**kwargs)
    result.write_text('{"status":"changed"}\n', encoding="utf-8")
    with pytest.raises(MODULE.EvidenceError, match="corpus CAS failed|stale"):
        MODULE._revalidate_final_assembly_cas(**kwargs)
    result.write_text('{"status":"complete"}\n', encoding="utf-8")
    stale_verified = copy.deepcopy(verified)
    stale_verified["verified_references_sha256"] = "e" * 64
    monkeypatch.setattr(
        MODULE, "verify_final_packet_references", lambda _observer: ([], stale_verified),
    )
    with pytest.raises(MODULE.EvidenceError, match="changed verified_references_sha256"):
        MODULE._revalidate_final_assembly_cas(**kwargs)


def test_prepare_release_uses_one_boundary_and_ordered_atomic_parent_chain(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    (corpus_root / "result.json").write_text(
        '{"status":"complete"}\n', encoding="utf-8",
    )
    report_root = tmp_path / "reports" / "physical_release"
    output_dir = report_root / "prepared" / "generation"
    monkeypatch.setattr(MODULE, "REPORT_ROOT", report_root)
    monkeypatch.setattr(MODULE.spec_reentry_scaffold, "active_heavy_owners", lambda: [])
    monkeypatch.setattr(
        MODULE, "_recheck_under_lease",
        lambda expected: ({"state_sha256": expected}, [], {"ok": True}),
    )
    observation, boundary = _boundary()
    source = _source_manifest(observation, boundary)
    events: list[str] = []
    monkeypatch.setattr(
        MODULE, "build_clean_source_manifest",
        lambda **_kwargs: events.append("source") or source,
    )
    original_index = MODULE.appendix_corpus.build_index

    def build_index(*args: object, **kwargs: object) -> dict:
        events.append("index")
        return original_index(*args, **kwargs)

    monkeypatch.setattr(MODULE.appendix_corpus, "build_index", build_index)
    original_verification = MODULE.build_corpus_verification

    def verify(*args: object, **kwargs: object) -> tuple[dict, dict]:
        events.append(str(kwargs["verification_phase"]))
        return original_verification(*args, **kwargs)

    monkeypatch.setattr(MODULE, "build_corpus_verification", verify)

    def release_build(
        _admission: object, *, source_manifest: dict,
        build_log: pathlib.Path, write_log: bool,
    ) -> tuple[dict, str]:
        assert source_manifest is source and write_log is False
        events.append("build")
        log = "synthetic fresh build\n"
        receipt = _release_receipt(source, boundary)
        receipt["build_log"] = MODULE._byte_binding(build_log, log.encode())
        return MODULE._stamp(receipt, "receipt_sha256"), log

    monkeypatch.setattr(MODULE, "run_release_build", release_build)
    monkeypatch.setattr(
        MODULE, "_revalidate_final_assembly_cas",
        lambda **_kwargs: events.append("final_cas"),
    )
    monkeypatch.setattr(
        MODULE, "_revalidate_build_outputs_before_seal",
        lambda *_args, **_kwargs: events.append("build_cas"),
    )
    original_atomic = MODULE._atomic_bytes_group

    def atomic(rows: object) -> None:
        events.append("atomic")
        original_atomic(rows)

    monkeypatch.setattr(MODULE, "_atomic_bytes_group", atomic)
    admission = MODULE.ReleaseAdmission(
        lease=None,
        observer={"state_sha256": "a" * 64},
        boundary_observation=observation,
        boundary_attestation=boundary,
    )
    prepared = MODULE.prepare_release(
        admission, corpus_root=corpus_root, output_dir=output_dir,
    )
    assert events == [
        "source", "index", "pre_release_build", "build",
        "post_release_build", "final_cas", "build_cas", "atomic",
    ]
    pre = json.loads((output_dir / "corpus_prebuild_verification_receipt.json").read_text())
    post = json.loads((output_dir / "corpus_postbuild_verification_receipt.json").read_text())
    capsule = json.loads((output_dir / "critical_source_capsule.json").read_text())
    build = json.loads((output_dir / "release_build_receipt.json").read_text())
    assert {
        capsule["release_boundary_attestation_sha256"],
        build["release_boundary_attestation_sha256"],
        pre["release_boundary_attestation_sha256"],
        post["release_boundary_attestation_sha256"],
    } == {boundary["attestation_sha256"]}
    assert post["parent_verification_receipt_sha256"] == pre["verification_receipt_sha256"]
    assert prepared["phase_order"][-1] == "atomic_seal"
