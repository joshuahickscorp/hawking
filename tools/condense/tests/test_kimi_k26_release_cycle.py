#!/usr/bin/env python3.12
from __future__ import annotations

import copy
import hashlib
import json
import os
import pathlib
import stat
import subprocess
import sys
import zipfile

import pytest

CONDENSE = pathlib.Path(__file__).resolve().parents[1]
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import kimi_k26_release_cycle as cycle  # noqa: E402


def _private_parent(tmp_path: pathlib.Path) -> pathlib.Path:
    parent = tmp_path / "release-sessions"
    parent.mkdir(mode=0o700)
    parent.chmod(0o700)
    return parent


def _layout(tmp_path: pathlib.Path, session_id: str = "test-session") -> cycle.SessionLayout:
    parent = _private_parent(tmp_path)
    value = cycle.init_session(session_id, parent=parent)
    assert value["status"] == "PRIVATE_SESSION_CREATED_NO_LIVE_ACTION"
    return cycle.layout_for(parent / session_id, parent=parent)


def _write_private(path: pathlib.Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    for parent in [path.parent, *path.parent.parents]:
        if parent == path.anchor:
            break
        if parent.exists() and "release-sessions" in parent.parts:
            parent.chmod(0o700)
    path.write_bytes(raw)
    path.chmod(0o600)


def _git_blob(raw: bytes) -> str:
    return hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()  # noqa: S324


def _small_source(
    monkeypatch: pytest.MonkeyPatch,
    layout: cycle.SessionLayout,
) -> tuple[pathlib.Path, dict[str, pathlib.Path]]:
    config = b'{"model":"test"}\n'
    shard = b"exact-packed-weight-shard"
    rows = [
        {
            "blob_id": _git_blob(config),
            "path": "config.json",
            "sha256": None,
            "size": len(config),
        },
        {
            "blob_id": "1" * 40,
            "path": "model-00001-of-000001.safetensors",
            "sha256": hashlib.sha256(shard).hexdigest(),
            "size": len(shard),
        },
    ]
    unsigned = {
        "file_count": 2,
        "files": rows,
        "largest_shard": len(shard),
        "last_modified": "2026-05-19T09:01:54.000Z",
        "library_name": "transformers",
        "license_api": "other",
        "pipeline_tag": "image-text-to-text",
        "repo": cycle.KIMI_REPO,
        "resolved_at": "2026-07-21T05:52:56Z",
        "schema": "hawking.kimi_k26.official_manifest.v1",
        "sha": cycle.KIMI_REVISION,
        "total_bytes": len(config) + len(shard),
        "weight_bytes": len(shard),
        "weight_shards": 1,
    }
    manifest = cycle.seal_document(unsigned)
    monkeypatch.setattr(cycle, "KIMI_FILE_COUNT", 2)
    monkeypatch.setattr(cycle, "KIMI_WEIGHT_SHARDS", 1)
    monkeypatch.setattr(cycle, "KIMI_TOTAL_BYTES", len(config) + len(shard))
    monkeypatch.setattr(cycle, "KIMI_WEIGHT_BYTES", len(shard))
    monkeypatch.setattr(cycle, "KIMI_LARGEST_SHARD_BYTES", len(shard))
    monkeypatch.setattr(cycle, "KIMI_MANIFEST_SEAL_SHA256", manifest["seal_sha256"])
    manifest_path = layout.build / "test-manifest.json"
    _write_private(
        manifest_path,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
    )

    layout.blobs.mkdir(parents=True, mode=0o700)
    layout.snapshot.mkdir(parents=True, mode=0o700)
    blob_paths: dict[str, pathlib.Path] = {}
    for row, raw in zip(rows, (config, shard), strict=True):
        content_id = row["sha256"] or row["blob_id"]
        target = layout.blobs / content_id
        _write_private(target, raw)
        link = layout.snapshot / row["path"]
        link.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        link.symlink_to(os.path.relpath(target, link.parent))
        blob_paths[row["path"]] = target
    return manifest_path, blob_paths


def _fake_runner(
    *, lsof_output: str = "", ps_output: str = "1 0 /sbin/launchd\n"
):
    def run(argv: tuple[str, ...] | list[str]) -> subprocess.CompletedProcess[str]:
        if argv[0] == "/usr/sbin/lsof":
            return subprocess.CompletedProcess(argv, 0 if lsof_output else 1, lsof_output, "")
        if argv[0] == "/bin/ps":
            return subprocess.CompletedProcess(argv, 0, ps_output, "")
        raise AssertionError(argv)

    return run


def test_init_session_is_private_and_has_no_live_capability(tmp_path: pathlib.Path) -> None:
    parent = _private_parent(tmp_path)
    value = cycle.init_session("phase1-001", parent=parent)
    assert cycle.verify_sealed_document(value)["download_executed"] is False
    assert value["delete_capability_present"] is False
    layout = cycle.layout_for(parent / "phase1-001", parent=parent)
    for path in (
        parent,
        layout.session,
        layout.hub,
        layout.xet,
        layout.build,
        layout.tmp,
        layout.hf_home,
        layout.recovery,
        layout.evidence,
    ):
        assert stat.S_IMODE(path.stat().st_mode) == 0o700
        assert path.stat().st_uid == os.getuid()


@pytest.mark.parametrize("session_id", ["../escape", ".", "UPPER", "a/b", ""])
def test_init_session_rejects_unsafe_ids(
    tmp_path: pathlib.Path, session_id: str
) -> None:
    with pytest.raises(cycle.ReleaseCycleError):
        cycle.init_session(session_id, parent=_private_parent(tmp_path))


def test_layout_rejects_symlinked_component(tmp_path: pathlib.Path) -> None:
    layout = _layout(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    layout.xet.rmdir()
    layout.xet.symlink_to(outside, target_is_directory=True)
    with pytest.raises((cycle.ReleaseCycleError, OSError)):
        cycle.validate_layout(layout)


@pytest.mark.parametrize("scratch_name", ["tmp", "hf_home"])
def test_layout_rejects_missing_scratch_root(
    tmp_path: pathlib.Path, scratch_name: str
) -> None:
    layout = _layout(tmp_path)
    getattr(layout, scratch_name).rmdir()
    with pytest.raises((cycle.ReleaseCycleError, OSError)):
        cycle.validate_layout(layout)


@pytest.mark.parametrize("scratch_name", ["tmp", "hf_home"])
def test_layout_rejects_symlinked_scratch_root(
    tmp_path: pathlib.Path, scratch_name: str
) -> None:
    layout = _layout(tmp_path)
    scratch = getattr(layout, scratch_name)
    scratch.rmdir()
    outside = tmp_path / f"outside-{scratch_name}"
    outside.mkdir()
    scratch.symlink_to(outside, target_is_directory=True)
    with pytest.raises((cycle.ReleaseCycleError, OSError)):
        cycle.validate_layout(layout)


@pytest.mark.parametrize("scratch_name", ["tmp", "hf_home"])
def test_layout_rejects_nonprivate_scratch_mode(
    tmp_path: pathlib.Path, scratch_name: str
) -> None:
    layout = _layout(tmp_path)
    getattr(layout, scratch_name).chmod(0o755)
    with pytest.raises(cycle.ReleaseCycleError, match="expected 700"):
        cycle.validate_layout(layout)


def test_download_plan_is_exact_dedicated_and_inert(tmp_path: pathlib.Path) -> None:
    layout = _layout(tmp_path)
    first = cycle.build_download_plan(layout)
    second = cycle.build_download_plan(layout)
    assert first == second
    assert first["status"] == "PLANNED_NOT_EXECUTED"
    assert first["command_argv"] == [
        str(cycle.HF_CLI),
        "download",
        cycle.KIMI_REPO,
        "--revision",
        cycle.KIMI_REVISION,
        "--repo-type",
        "model",
        "--cache-dir",
        str(layout.hub),
        "--max-workers",
        "8",
    ]
    assert first["environment"]["HF_HUB_CACHE"] == str(layout.hub)
    assert first["environment"]["HF_XET_CACHE"] == str(layout.xet)
    assert first["environment"]["HF_XET_CACHE"] != str(cycle.SHARED_HF_XET_ROOT)
    assert first["environment"]["HF_XET_HIGH_PERFORMANCE"] == "1"
    assert first["environment"]["HF_XET_CHUNK_CACHE_SIZE_BYTES"] == "0"
    assert first["environment"]["HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS"] == "8"
    assert "HF_TOKEN" not in first["environment"]
    assert first["transfer_profile"] == {
        "hardware_target": "OBSERVED_10_GBIT_ETHERNET_LINK",
        "end_to_end_throughput_status": "NOT_YET_MEASURED",
        "maximum_file_download_workers": 8,
        "xet_high_performance": True,
        "xet_chunk_cache_size_bytes": 0,
        "xet_data_max_concurrent_file_downloads": 8,
        "reason": (
            "eight simultaneous approximately 9 GB shard downloads plus Xet internal "
            "range concurrency, without a second chunk-cache copy"
        ),
        "live_measurement_and_ramp_authority": "LIVE_SUPERVISOR_ONLY",
        "phase1_claims_saturation": False,
    }
    assert first["environment"]["TMPDIR"] == str(layout.tmp)
    assert first["environment"]["TMP"] == str(layout.tmp)
    assert first["environment"]["TEMP"] == str(layout.tmp)
    assert first["environment"]["HF_HOME"] == str(layout.hf_home)
    assert first["environment"]["PYTHONPYCACHEPREFIX"] == str(
        layout.tmp / "pycache"
    )
    assert first["temporary_directory_policy"][
        "fallback_to_system_or_shared_tmp_forbidden"
    ] is True
    assert first["temporary_directory_policy"][
        "directory_precreated_uid_owned_mode_0700"
    ] is True
    assert first["one_copy_law"]["source_payload_location"] == str(layout.hub)
    assert first["one_copy_law"]["local_dir_copy_forbidden"] is True
    assert first["one_copy_law"]["xet_chunk_cache_disabled"] is True
    assert first["one_copy_law"]["shared_xet_forbidden"] == str(
        cycle.SHARED_HF_XET_ROOT
    )
    assert first["one_copy_law"]["build_directory_may_contain_source_copy"] is False
    assert first["one_copy_law"]["recovery_directory_may_contain_source_copy"] is False
    assert first["network_accessed"] is False
    assert first["executor_present_in_this_phase"] is False
    runtime = first["transfer_runtime"]
    assert runtime["cli"]["sha256"] == cycle.HF_CLI_SHA256
    assert runtime["cli"]["exact_shebang"] == cycle.HF_CLI_SHEBANG.rstrip("\n")
    assert runtime["resolved_interpreter"]["path"] == str(cycle.TRANSFER_INTERPRETER)
    assert runtime["resolved_interpreter"]["sha256"] == cycle.TRANSFER_INTERPRETER_SHA256
    assert runtime["distributions"]["huggingface_hub"]["version"] == "1.24.0"
    assert runtime["distributions"]["hf_xet"]["version"] == "1.5.2"
    assert cycle.verify_transfer_runtime_binding(runtime) == runtime
    profiles = {row["profile_id"]: row for row in first["restart_profiles"]}
    assert profiles["PRIMARY_8"]["command_argv"][-1] == "8"
    assert profiles["CONDITIONAL_RESTART_16"]["command_argv"][-1] == "16"
    assert profiles["CONDITIONAL_RESTART_16"]["environment"][
        "HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS"
    ] == "16"
    assert profiles["CONDITIONAL_RESTART_16"][
        "prior_transfer_process_must_be_fully_exited"
    ] is True
    assert profiles["CONDITIONAL_RESTART_16"]["concurrent_with_primary_forbidden"] is True
    assert cycle.verify_download_plan(first, layout) == first


def test_resealed_transfer_pin_substitution_is_rejected(tmp_path: pathlib.Path) -> None:
    layout = _layout(tmp_path)
    plan = cycle.build_download_plan(layout)
    plan["environment"]["HF_XET_HIGH_PERFORMANCE"] = "0"
    plan = cycle.seal_document(plan)
    with pytest.raises(cycle.ReleaseCycleError, match="exact deterministic"):
        cycle.verify_download_plan(plan, layout)


def _replace_nested(value: dict, path: tuple[object, ...], replacement: object) -> None:
    cursor: object = value
    for component in path[:-1]:
        cursor = cursor[component]  # type: ignore[index]
    cursor[path[-1]] = replacement  # type: ignore[index]


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("cli", "logical_bytes"), 1),
        (("cli", "sha256"), "0" * 64),
        (("cli", "mode"), "0700"),
        (("cli", "uid"), -1),
        (("cli", "hard_links"), 2),
        (("cli", "exact_shebang"), "#!/substituted/python"),
        (("interpreter_chain", 0, "target"), "other-python"),
        (("interpreter_chain", 1, "target"), "/tmp/python"),
        (("resolved_interpreter", "path"), "/tmp/python"),
        (("resolved_interpreter", "logical_bytes"), 1),
        (("resolved_interpreter", "sha256"), "1" * 64),
        (("resolved_interpreter", "mode"), "0777"),
        (("resolved_interpreter", "uid"), os.getuid()),
        (("resolved_interpreter", "hard_links"), 2),
        (("distributions", "huggingface_hub", "version"), "1.24.1"),
        (("distributions", "huggingface_hub", "record_sha256"), "2" * 64),
        (("distributions", "hf_xet", "version"), "1.5.3"),
        (("distributions", "hf_xet", "record_sha256"), "3" * 64),
    ],
)
def test_resealed_runtime_binding_substitutions_are_rejected(
    path: tuple[object, ...], replacement: object
) -> None:
    runtime = cycle.build_transfer_runtime_binding()
    changed = copy.deepcopy(runtime)
    _replace_nested(changed, path, replacement)
    changed = cycle.seal_document(changed)
    with pytest.raises(cycle.ReleaseCycleError, match="exact deterministic runtime"):
        cycle.verify_transfer_runtime_binding(changed)


def test_resealed_native_xet_artifact_substitution_is_rejected() -> None:
    runtime = cycle.build_transfer_runtime_binding()
    changed = copy.deepcopy(runtime)
    native = next(
        row
        for row in changed["relevant_dist_info_and_native_artifacts"]
        if row["relative_path"] == "hf_xet/hf_xet.abi3.so"
    )
    native["sha256"] = "4" * 64
    changed = cycle.seal_document(changed)
    with pytest.raises(cycle.ReleaseCycleError, match="exact deterministic runtime"):
        cycle.verify_transfer_runtime_binding(changed)


@pytest.mark.parametrize("mutation", ["symlink", "hardlink", "wrong_mode"])
def test_runtime_builder_rejects_unsafe_hf_launcher_node(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    raw = cycle.HF_CLI.read_bytes()
    launcher = tmp_path / "hf"
    if mutation == "symlink":
        target = tmp_path / "target"
        _write_private(target, raw)
        launcher.symlink_to(target)
    else:
        _write_private(launcher, raw)
        if mutation == "hardlink":
            os.link(launcher, tmp_path / "alias")
        else:
            launcher.chmod(0o700)
    monkeypatch.setattr(cycle, "HF_CLI", launcher)
    with pytest.raises(cycle.ReleaseCycleError, match="regular|hard-link|mode"):
        cycle.build_transfer_runtime_binding()


def test_real_frozen_manifest_and_archive_verify() -> None:
    manifest = cycle.verify_manifest()
    archive = cycle.verify_recovery_archive()
    assert manifest["manifest_seal_sha256"] == cycle.KIMI_MANIFEST_SEAL_SHA256
    assert archive["archive_sha256"] == cycle.ACCEPTED_ARCHIVE_SHA256
    assert archive["archive_bytes"] == cycle.ACCEPTED_ARCHIVE_BYTES
    assert archive["credential_entry_present"] is False


def test_strict_json_rejects_duplicate_keys_and_nan() -> None:
    with pytest.raises(cycle.ReleaseCycleError, match="duplicate"):
        cycle.strict_json_bytes(b'{"x":1,"x":2}', label="duplicate")
    with pytest.raises(cycle.ReleaseCycleError, match="non-finite"):
        cycle.strict_json_bytes(b'{"x":NaN}', label="nan")


def test_manifest_rejects_resealed_substitution(tmp_path: pathlib.Path) -> None:
    original = json.loads(cycle.OFFICIAL_MANIFEST.read_text())
    original["total_bytes"] -= 1
    substituted = cycle.seal_document(original)
    path = tmp_path / "manifest.json"
    _write_private(path, cycle.canonical_json(substituted))
    with pytest.raises(cycle.ReleaseCycleError, match="frozen"):
        cycle.verify_manifest(path)


def test_old_898063_archive_is_explicitly_rejected(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "old.zip"
    _write_private(path, b"x" * cycle.REJECTED_OLD_ARCHIVE_BYTES)
    with pytest.raises(cycle.ReleaseCycleError, match="898,063"):
        cycle.verify_recovery_archive(path)


def test_archive_hash_substitution_is_rejected(tmp_path: pathlib.Path) -> None:
    raw = bytearray(cycle.SANITIZED_ARCHIVE.read_bytes())
    raw[len(raw) // 2] ^= 1
    path = tmp_path / "substituted.zip"
    _write_private(path, bytes(raw))
    with pytest.raises(cycle.ReleaseCycleError, match="f77fecfa"):
        cycle.verify_recovery_archive(path)


@pytest.mark.parametrize(
    "name", ["../escape.json", "/absolute.json", "a/../../escape", "a\\b.json"]
)
def test_zip_slip_names_are_rejected(name: str) -> None:
    with pytest.raises(cycle.ReleaseCycleError):
        cycle._safe_zip_name(name)


def test_zip_symlink_member_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    target = cycle.io.BytesIO()
    with zipfile.ZipFile(target, "w") as archive:
        safe = zipfile.ZipInfo("safe.json")
        safe.external_attr = (stat.S_IFREG | 0o600) << 16
        archive.writestr(safe, b"{}")
        link = zipfile.ZipInfo("link.json")
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(link, b"../outside")
    monkeypatch.setattr(cycle, "ACCEPTED_ARCHIVE_ENTRIES", 2)
    monkeypatch.setattr(
        cycle, "ACCEPTED_ARCHIVE_UNCOMPRESSED_BYTES", len(b"{}") + len(b"../outside")
    )
    monkeypatch.setattr(
        cycle,
        "RECOVERY_ENTRY_ALLOWLIST",
        {"safe.json": (2, hashlib.sha256(b"{}").hexdigest())},
    )
    with pytest.raises(cycle.ReleaseCycleError, match="non-regular"):
        cycle._verify_archive_raw(target.getvalue())


def test_recovery_extracts_only_explicit_text_allowlist(tmp_path: pathlib.Path) -> None:
    layout = _layout(tmp_path)
    entries = {
        cycle.GRAVITY_FINAL_RELATIVE,
        cycle.BEST_RESULT_RELATIVE,
        cycle.TEACHER_CAPTURE_RECORD_RELATIVE,
    }
    value = cycle.extract_sanitized_recovery(layout, entries=entries)
    assert value["status"] == "PASS_ALLOWLISTED_TEXT_ONLY"
    assert {row["relative_path"] for row in value["extracted"]} == entries
    assert value["binary_payload_extracted_from_archive"] is False
    for name in entries:
        output = layout.recovery / name
        assert output.is_file()
        assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert not (layout.recovery / "reference_run/probe_coding.json").exists()


def test_recovery_extractor_refuses_overwrite_and_unlisted_entry(
    tmp_path: pathlib.Path,
) -> None:
    layout = _layout(tmp_path)
    cycle.extract_sanitized_recovery(
        layout, entries={cycle.TEACHER_CAPTURE_RECORD_RELATIVE}
    )
    with pytest.raises(cycle.ReleaseCycleError, match="overwrite"):
        cycle.extract_sanitized_recovery(
            layout, entries={cycle.TEACHER_CAPTURE_RECORD_RELATIVE}
        )
    with pytest.raises(cycle.ReleaseCycleError, match="allowlist"):
        cycle.extract_sanitized_recovery(layout, entries={"../escape"})


def test_source_verifier_accepts_exact_dedicated_symlink_snapshot(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = _layout(tmp_path)
    manifest, _ = _small_source(monkeypatch, layout)
    value = cycle.verify_source(layout, manifest_path=manifest)
    assert value["status"] == "PASS_EXACT_IMMUTABLE_SOURCE"
    assert value["file_count"] == 2
    assert value["logical_bytes"] == cycle.KIMI_TOTAL_BYTES
    assert value["shared_xet_used"] is False


def test_source_verifier_rejects_symlink_escape(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = _layout(tmp_path)
    manifest, _ = _small_source(monkeypatch, layout)
    link = layout.snapshot / "config.json"
    link.unlink()
    link.symlink_to(tmp_path / "outside")
    with pytest.raises(cycle.ReleaseCycleError, match="absolute|escapes"):
        cycle.verify_source(layout, manifest_path=manifest)


def test_source_verifier_rejects_hardlinked_blob(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = _layout(tmp_path)
    manifest, blobs = _small_source(monkeypatch, layout)
    os.link(blobs["config.json"], layout.build / "external-alias")
    with pytest.raises(cycle.ReleaseCycleError, match="hard-link"):
        cycle.verify_source(layout, manifest_path=manifest)


def test_source_verifier_rejects_blob_identity_substitution(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = _layout(tmp_path)
    manifest, blobs = _small_source(monkeypatch, layout)
    link = layout.snapshot / "config.json"
    link.unlink()
    link.symlink_to(os.path.relpath(blobs["model-00001-of-000001.safetensors"], link.parent))
    with pytest.raises(cycle.ReleaseCycleError, match="substituted"):
        cycle.verify_source(layout, manifest_path=manifest)


def test_inventory_deduplicates_hardlinks_and_blocks_release(
    tmp_path: pathlib.Path,
) -> None:
    layout = _layout(tmp_path)
    first = layout.hub / "a.bin"
    _write_private(first, b"shared inode")
    os.link(first, layout.hub / "b.bin")
    value = cycle.build_source_release_inventory(layout)
    regular = [row for row in value["inventory_rows"] if row["type"] == "regular"]
    assert len(regular) == 1
    assert len(regular[0]["aliases"]) == 2
    assert value["entry_count"] > value["unique_physical_object_count"]
    assert value["logical_bytes_deduplicated"] == sum(
        row["logical_bytes"] for row in value["inventory_rows"]
    )
    assert value["allocated_bytes_deduplicated"] == sum(
        row["allocated_bytes"] for row in value["inventory_rows"]
    )
    assert value["status"] == "BLOCKED"
    assert any("HARDLINK" in item for item in value["violations"])


def test_inventory_blocks_mop_symlink_and_path_overlap(tmp_path: pathlib.Path) -> None:
    layout = _layout(tmp_path)
    mop = tmp_path / "mop"
    mop.mkdir()
    protected = mop / "do-not-touch"
    protected.write_bytes(b"MOP")
    (layout.hub / "mop-link").symlink_to(protected)
    value = cycle.build_inventory(
        (layout.hub, layout.xet), session_root=layout.session, mop_root=mop
    )
    assert value["status"] == "BLOCKED"
    assert any("MOP_SYMLINK_TARGET" in item for item in value["violations"])
    with pytest.raises(cycle.ReleaseCycleError, match="overlaps MOP"):
        cycle.build_inventory((mop,), session_root=tmp_path, mop_root=mop)


def test_inventory_seal_and_allocated_sum_detect_substitution(
    tmp_path: pathlib.Path,
) -> None:
    layout = _layout(tmp_path)
    _write_private(layout.hub / "body.bin", b"body")
    value = cycle.build_source_release_inventory(layout)
    assert cycle.verify_inventory(value)["status"] == "PASS"
    changed = json.loads(json.dumps(value))
    changed["allocated_bytes_deduplicated"] += 512
    with pytest.raises(cycle.ReleaseCycleError, match="seal"):
        cycle.verify_inventory(changed)


def test_pre_release_audit_passes_only_clean_inventory_readers_processes_and_queues(
    tmp_path: pathlib.Path,
) -> None:
    layout = _layout(tmp_path)
    _write_private(layout.hub / "body.bin", b"body")
    inventory = cycle.build_source_release_inventory(layout)
    value = cycle.build_pre_release_audit(
        inventory, queue_roots=(layout.session / "queue",), runner=_fake_runner()
    )
    assert value["status"] == "PASS"
    assert value["exact_release_allocated_bytes"] == inventory[
        "allocated_bytes_deduplicated"
    ]
    assert value["deletion_authorized"] is False
    assert value["deletion_performed"] is False


def test_pre_release_audit_blocks_lsof_process_and_queue(
    tmp_path: pathlib.Path,
) -> None:
    layout = _layout(tmp_path)
    _write_private(layout.hub / "body.bin", b"body")
    queue = layout.session / "queue"
    queue.mkdir(mode=0o700)
    _write_private(queue / "pending.json", b"{}")
    inventory = cycle.build_source_release_inventory(layout)
    lsof = "COMMAND PID USER FD TYPE DEVICE SIZE/OFF NODE NAME\npython 42 u 3r REG 1,1 4 7 body\n"
    ps = f"42 1 hf download {layout.hub}\n"
    value = cycle.build_pre_release_audit(
        inventory,
        queue_roots=(queue,),
        runner=_fake_runner(lsof_output=lsof, ps_output=ps),
    )
    assert value["status"] == "BLOCKED"
    assert set(value["blockers"]) == {
        "MATCHING_PROCESS_PRESENT",
        "OPEN_FILE_READERS_PRESENT",
        "QUEUE_OR_OUTBOX_NOT_EMPTY",
    }


def _synthetic_capsule(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[cycle.SessionLayout, pathlib.Path, pathlib.Path]:
    layout = _layout(tmp_path)
    layout.capsule.mkdir(mode=0o700)
    payload_raw = b"payload"
    capture_raw = b"capture"
    payload = layout.capsule / cycle.BEST_PAYLOAD_BASENAME
    capture = layout.capsule / cycle.TEACHER_CAPTURE_BASENAME
    _write_private(payload, payload_raw)
    _write_private(capture, capture_raw)
    monkeypatch.setattr(cycle, "BEST_PAYLOAD_BYTES", len(payload_raw))
    monkeypatch.setattr(cycle, "BEST_PAYLOAD_SHA256", hashlib.sha256(payload_raw).hexdigest())
    monkeypatch.setattr(cycle, "TEACHER_CAPTURE_BYTES", len(capture_raw))
    monkeypatch.setattr(
        cycle, "TEACHER_CAPTURE_SHA256", hashlib.sha256(capture_raw).hexdigest()
    )
    result = cycle.seal_document(
        {
            "schema": "hawking.kimi_k26.f1_hidden_doctor_result.v1",
            "status": "PASS",
            "candidate": "P1",
            "source": {"repo": cycle.KIMI_REPO, "revision": cycle.KIMI_REVISION},
            "payload": {
                "bytes": len(payload_raw),
                "sha256": hashlib.sha256(payload_raw).hexdigest(),
                "base_component_bytes": 4_022_298,
                "doctor_component_bytes": 974_848,
                "header_overhead_bytes": 4_669,
            },
        }
    )
    record = cycle.seal_document(
        {
            "schema": "hawking.kimi_k26.f1_teacher_capture.v1",
            "status": "PASS",
            "revision": cycle.KIMI_REVISION,
            "capture_bytes": len(capture_raw),
            "capture_sha256": hashlib.sha256(capture_raw).hexdigest(),
        }
    )
    final = cycle.seal_document(
        {
            "schema": "hawking.kimi_k26.gravity_final.v1",
            "status": "CLOSED",
            "terminal_outcome": "OUTCOME_C",
            "best_deployable_candidate": {
                "candidate": "P1_DUAL_PATH_RECOVERY_R16X2",
                "complete_physical_bytes": len(payload_raw),
                "payload_sha256": hashlib.sha256(payload_raw).hexdigest(),
            },
        }
    )
    monkeypatch.setattr(cycle, "BEST_RESULT_SEAL_SHA256", result["seal_sha256"])
    monkeypatch.setattr(
        cycle, "TEACHER_CAPTURE_RECORD_SEAL_SHA256", record["seal_sha256"]
    )
    monkeypatch.setattr(cycle, "GRAVITY_FINAL_SEAL_SHA256", final["seal_sha256"])
    for relative, value in (
        (cycle.BEST_RESULT_RELATIVE, result),
        (cycle.TEACHER_CAPTURE_RECORD_RELATIVE, record),
        (cycle.GRAVITY_FINAL_RELATIVE, final),
    ):
        _write_private(
            layout.recovery / relative,
            (json.dumps(value, sort_keys=True) + "\n").encode(),
        )
    return layout, payload, capture


def test_exact_payload_result_capture_verifier(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, payload, capture = _synthetic_capsule(tmp_path, monkeypatch)
    value = cycle.verify_payload_result_capture(layout)
    assert value["status"] == "PASS_EXACT_PAYLOAD_RESULT_CAPTURE"
    assert value["payload"]["path"] == str(payload)
    assert value["capture"]["path"] == str(capture)
    assert value["mop_touched"] is False


def test_payload_hash_substitution_and_hardlink_are_rejected(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, payload, _ = _synthetic_capsule(tmp_path, monkeypatch)
    payload.write_bytes(b"changed")
    with pytest.raises(cycle.ReleaseCycleError, match="hash changed"):
        cycle.verify_payload_result_capture(layout)
    payload.write_bytes(b"payload")
    os.link(payload, layout.build / "payload-alias")
    with pytest.raises(cycle.ReleaseCycleError, match="hard-link"):
        cycle.verify_payload_result_capture(layout)


def test_phase1_preflight_is_deterministic_and_never_calls_subprocess(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = _layout(tmp_path)

    def forbidden(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("phase-1 preflight attempted a subprocess")

    monkeypatch.setattr(subprocess, "run", forbidden)
    first = cycle.build_preflight(layout)
    second = cycle.build_preflight(layout)
    assert first == second
    assert first["status"] == "PASS_PHASE1_NO_LIVE_ACTION"
    assert first["network_accessed"] is False
    assert first["download_executed"] is False
    assert first["delete_capability_present"] is False


def test_cli_exposes_no_download_delete_or_release_execution_command() -> None:
    action = next(action for action in cycle._parser()._actions if action.dest == "command")
    assert set(action.choices) == {
        "init-session",
        "preflight",
        "plan-download",
        "verify-source",
        "verify-recovery",
    }
    for forbidden in {"download", "delete", "release", "execute", "evict"}:
        assert forbidden not in action.choices
