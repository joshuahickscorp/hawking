from __future__ import annotations

import hashlib
import inspect
import os
from pathlib import Path

import pytest

from tools.condense.glm52_common import seal
from tools.condense.glm52_grounding import (
    ABSENCE_OBSERVATION_SCHEMA,
    FILE_OBSERVATION_SCHEMA,
    RESOURCE_SAMPLE_SCHEMA,
    DiskSample,
    GroundingError,
    MemorySample,
    ProducerAuthenticator,
    ResourceFloorError,
    ResourceReservePolicy,
    TrustedFilesystemObserver,
    _sample_resources_with_providers,
    parse_darwin_memory,
    parse_linux_meminfo,
    sample_resources,
    verify_authenticated_observation,
)


AUTH = ProducerAuthenticator(b"grounding-test-producer-key-32-bytes-minimum!!")
OTHER_AUTH = ProducerAuthenticator(b"another-grounding-producer-key-32-bytes!!")
FIXED_TIME = "2026-07-21T15:16:17.123456Z"


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _observer(root: Path, **kwargs: object) -> TrustedFilesystemObserver:
    return TrustedFilesystemObserver(
        root,
        root_id="artifact-root:test",
        authenticator=AUTH,
        clock=lambda: FIXED_TIME,
        chunk_bytes=3,
        **kwargs,
    )


def test_regular_file_observation_is_grounded_authenticated_and_deterministic(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    nested = root / "nested"
    nested.mkdir(parents=True)
    payload = b"streamed-grounding-payload"
    target = nested / "artifact.bin"
    target.write_bytes(payload)
    observer = _observer(root)

    first = observer.observe_regular_file(
        "nested/artifact.bin",
        expected_size_bytes=len(payload),
        expected_sha256=_sha(payload),
    )
    second = observer.observe_regular_file(
        "nested/artifact.bin",
        expected_size_bytes=len(payload),
        expected_sha256=_sha(payload),
    )

    assert first == second
    assert first["schema"] == FILE_OBSERVATION_SCHEMA
    assert first["status"] == "PASS"
    assert first["relative_path"] == "nested/artifact.bin"
    assert first["logical_bytes"] == len(payload)
    assert first["allocated_bytes"] == target.stat().st_blocks * 512
    assert first["device"] == target.stat().st_dev
    assert first["inode"] == target.stat().st_ino
    assert first["observed_sha256"] == _sha(payload)
    assert first["expected_sha256"] == _sha(payload)
    assert first["producer_key_identity_sha256"] == AUTH.key_identity_sha256
    assert verify_authenticated_observation(first, AUTH) == first


@pytest.mark.parametrize(
    "relative_path",
    [
        "/etc/passwd",
        "../outside",
        "nested/../../outside",
        "nested/../artifact",
        "nested//artifact",
        "./artifact",
        "artifact/.",
        "artifact/",
    ],
)
def test_file_observation_rejects_traversal_and_non_normal_paths(
    tmp_path: Path, relative_path: str
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(GroundingError):
        _observer(root).observe_regular_file(
            relative_path,
            expected_size_bytes=0,
            expected_sha256=_sha(b""),
        )


def test_file_observation_rejects_symlink_leaf_and_symlink_directory(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    (root / "leaf-link").symlink_to(outside)
    (root / "directory-link").symlink_to(tmp_path, target_is_directory=True)

    observer = _observer(root)
    with pytest.raises(GroundingError, match="symlink"):
        observer.observe_regular_file(
            "leaf-link",
            expected_size_bytes=7,
            expected_sha256=_sha(b"outside"),
        )
    with pytest.raises(GroundingError, match="symlink"):
        observer.observe_regular_file(
            "directory-link/outside.bin",
            expected_size_bytes=7,
            expected_sha256=_sha(b"outside"),
        )


def test_file_observation_rejects_symlink_root(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "root-link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(GroundingError, match="root"):
        _observer(link).observe_absence("missing")


def test_symlink_ancestor_of_trusted_root_is_rejected_for_all_observations(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real-parent"
    root = real_parent / "root"
    root.mkdir(parents=True)
    (root / "artifact").write_bytes(b"trusted")
    alias_parent = tmp_path / "alias-parent"
    alias_parent.symlink_to(real_parent, target_is_directory=True)
    aliased_root = alias_parent / "root"

    with pytest.raises(GroundingError, match="symlink trusted root component"):
        _observer(aliased_root).observe_regular_file(
            "artifact",
            expected_size_bytes=7,
            expected_sha256=_sha(b"trusted"),
        )
    with pytest.raises(GroundingError, match="symlink trusted root component"):
        sample_resources(
            aliased_root,
            root_id="workspace:test",
            policy=ResourceReservePolicy(emergency_floor_bytes=0),
            authenticator=AUTH,
        )


def test_file_observation_rejects_hardlink_even_when_target_is_contained(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    original = root / "original.bin"
    alias = root / "alias.bin"
    payload = b"one inode, two names"
    original.write_bytes(payload)
    os.link(original, alias)

    with pytest.raises(GroundingError, match="hard link"):
        _observer(root).observe_regular_file(
            "alias.bin",
            expected_size_bytes=len(payload),
            expected_sha256=_sha(payload),
        )


def test_strict_expected_size_and_hash_are_required(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "artifact").write_bytes(b"abc")
    observer = _observer(root)

    with pytest.raises(GroundingError, match="size"):
        observer.observe_regular_file(
            "artifact", expected_size_bytes=True, expected_sha256=_sha(b"abc")
        )
    with pytest.raises(GroundingError, match="lowercase"):
        observer.observe_regular_file(
            "artifact", expected_size_bytes=3, expected_sha256=_sha(b"abc").upper()
        )
    with pytest.raises(GroundingError, match="size mismatch"):
        observer.observe_regular_file(
            "artifact", expected_size_bytes=4, expected_sha256=_sha(b"abc")
        )


def test_same_size_wrong_hash_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "artifact").write_bytes(b"actual")
    with pytest.raises(GroundingError, match="SHA-256 mismatch"):
        _observer(root).observe_regular_file(
            "artifact",
            expected_size_bytes=6,
            expected_sha256=_sha(b"forged"),
        )


def test_same_inode_toctou_mutation_after_stream_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    target = root / "artifact"
    target.write_bytes(b"trusted")

    class MutatingObserver(TrustedFilesystemObserver):
        def _after_stream_before_post_fstat(self, relative_path: str, fd: int) -> None:
            assert relative_path == "artifact"
            target.write_bytes(b"mutated")

    observer = MutatingObserver(
        root,
        root_id="artifact-root:test",
        authenticator=AUTH,
        clock=lambda: FIXED_TIME,
        chunk_bytes=2,
    )
    with pytest.raises(GroundingError, match="changed while it was being read"):
        observer.observe_regular_file(
            "artifact",
            expected_size_bytes=7,
            expected_sha256=_sha(b"trusted"),
        )


def test_name_to_inode_swap_after_fstat_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    target = root / "artifact"
    replacement = root / "replacement"
    payload = b"same-content"
    target.write_bytes(payload)
    replacement.write_bytes(payload)
    original_inode = target.stat().st_ino
    replacement_inode = replacement.stat().st_ino
    assert original_inode != replacement_inode

    class SwappingObserver(TrustedFilesystemObserver):
        def _after_post_fstat_before_path_recheck(
            self, relative_path: str, fd: int
        ) -> None:
            assert relative_path == "artifact"
            os.replace(replacement, target)

    observer = SwappingObserver(
        root,
        root_id="artifact-root:test",
        authenticator=AUTH,
        clock=lambda: FIXED_TIME,
        chunk_bytes=2,
    )
    with pytest.raises(GroundingError, match="different metadata"):
        observer.observe_regular_file(
            "artifact",
            expected_size_bytes=len(payload),
            expected_sha256=_sha(payload),
        )


def test_receipt_tampering_rejected_even_with_new_plain_seal(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    payload = b"authenticated"
    (root / "artifact").write_bytes(payload)
    receipt = _observer(root).observe_regular_file(
        "artifact",
        expected_size_bytes=len(payload),
        expected_sha256=_sha(payload),
    )
    tampered = dict(receipt)
    tampered["logical_bytes"] += 1
    tampered = seal(tampered)

    with pytest.raises(GroundingError, match="HMAC"):
        verify_authenticated_observation(tampered, AUTH)
    with pytest.raises(GroundingError, match="identity"):
        verify_authenticated_observation(receipt, OTHER_AUTH)


def test_authenticated_observation_freshness_is_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    receipt = _observer(root).observe_absence("missing")
    assert verify_authenticated_observation(
        receipt,
        AUTH,
        now="2026-07-21T15:16:18.123456Z",
        max_age_seconds=1,
    ) == receipt
    with pytest.raises(GroundingError, match="stale"):
        verify_authenticated_observation(
            receipt,
            AUTH,
            now="2026-07-21T15:16:18.123457Z",
            max_age_seconds=1,
        )


def test_absence_observation_handles_missing_leaf_and_ancestor(tmp_path: Path) -> None:
    root = tmp_path / "root"
    (root / "existing").mkdir(parents=True)
    observer = _observer(root)

    leaf = observer.observe_absence("existing/missing.bin")
    ancestor = observer.observe_absence("missing-tree/child.bin")

    assert leaf["schema"] == ABSENCE_OBSERVATION_SCHEMA
    assert leaf["absent"] is True
    assert leaf["first_missing_component"] == "existing/missing.bin"
    assert leaf["existing_parent"] == "existing"
    assert ancestor["first_missing_component"] == "missing-tree"
    assert ancestor["existing_parent"] == "."
    verify_authenticated_observation(leaf, AUTH)
    verify_authenticated_observation(ancestor, AUTH)


def test_absence_rejects_existing_path_and_symlink(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "exists").write_bytes(b"x")
    (root / "link").symlink_to(tmp_path, target_is_directory=True)
    observer = _observer(root)

    with pytest.raises(GroundingError, match="path exists"):
        observer.observe_absence("exists")
    with pytest.raises(GroundingError, match="symlink"):
        observer.observe_absence("link/missing")


def test_absence_creation_race_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    target = root / "missing"

    class RacingObserver(TrustedFilesystemObserver):
        def _after_absence_probe_before_recheck(
            self, relative_path: str, first_missing_component: str
        ) -> None:
            assert first_missing_component == "missing"
            target.write_bytes(b"appeared")

    observer = RacingObserver(
        root,
        root_id="artifact-root:test",
        authenticator=AUTH,
        clock=lambda: FIXED_TIME,
    )
    with pytest.raises(GroundingError, match="appeared"):
        observer.observe_absence("missing")


def test_linux_meminfo_parser_uses_available_ram_and_swap_delta() -> None:
    sample = parse_linux_meminfo(
        """MemTotal:       16777216 kB
MemFree:         1000000 kB
MemAvailable:    8388608 kB
SwapTotal:       2097152 kB
SwapFree:        1572864 kB
"""
    )
    assert sample == MemorySample(
        total_ram_bytes=16 * 1024**3,
        available_ram_bytes=8 * 1024**3,
        total_swap_bytes=2 * 1024**3,
        used_swap_bytes=512 * 1024**2,
        source="linux:/proc/meminfo",
    )


def test_darwin_resource_parser_uses_vm_pages_and_swapusage() -> None:
    sample = parse_darwin_memory(
        hw_memsize=str(32 * 1024**3),
        vm_stat="""Mach Virtual Memory Statistics: (page size of 16384 bytes.)
Pages free:                               1000.
Pages active:                            9000.
Pages inactive:                          2000.
Pages speculative:                        500.
""",
        swapusage="total = 4.00G  used = 1.50G  free = 2.50G  (encrypted)",
    )
    assert sample.total_ram_bytes == 32 * 1024**3
    assert sample.available_ram_bytes == 3500 * 16384
    assert sample.total_swap_bytes == 4 * 1024**3
    assert sample.used_swap_bytes == 3 * 1024**3 // 2
    assert sample.source == "darwin:sysctl+vm_stat"


def _disk(free: int) -> DiskSample:
    return DiskSample(
        total_bytes=100_000,
        free_bytes=free,
        used_bytes=100_000 - free,
        device=77,
    )


def _memory(*, available: int = 30_000, used_swap: int = 1_000) -> MemorySample:
    return MemorySample(
        total_ram_bytes=50_000,
        available_ram_bytes=available,
        total_swap_bytes=10_000,
        used_swap_bytes=used_swap,
        source="test:live-sample",
    )


def _policy() -> ResourceReservePolicy:
    return ResourceReservePolicy(
        emergency_floor_bytes=10_000,
        largest_atomic_source_write_bytes=20_000,
        largest_compact_shard_write_bytes=15_000,
        next_checkpoint_write_bytes=5_000,
        xet_reconstruction_scratch_bytes=18_000,
        two_largest_official_source_shards_bytes=12_000,
        projected_remaining_compact_bytes=4_000,
        projected_teacher_evidence_bytes=3_000,
        active_scratch_bytes=2_000,
        current_best_artifact_bytes=1_000,
        rollback_capsule_bytes=500,
        minimum_available_ram_bytes=20_000,
        maximum_swap_used_bytes=2_000,
    )


def test_live_resource_sample_enforces_operational_reserve_and_is_authenticated(
    tmp_path: Path,
) -> None:
    policy = _policy()
    receipt = _sample_resources_with_providers(
        tmp_path,
        root_id="workspace:test",
        policy=policy,
        authenticator=AUTH,
        clock=lambda: FIXED_TIME,
        platform_name="linux",
        disk_sampler=lambda fd: _disk(40_000),
        memory_sampler=lambda: _memory(),
        enforce=True,
    )

    assert receipt["schema"] == RESOURCE_SAMPLE_SCHEMA
    assert receipt["status"] == "PASS"
    assert receipt["operational_reserve_floor_bytes"] == 20_000
    assert receipt["additional_reserved_bytes"] == 10_500
    assert receipt["required_free_disk_bytes"] == 30_500
    assert receipt["usable_raw_window_bytes"] == 9_500
    assert receipt["disk_operational_reserve_ok"] is True
    assert receipt["available_ram_floor_ok"] is True
    assert receipt["swap_usage_ceiling_ok"] is True
    assert receipt["refusal_reasons"] == []
    verify_authenticated_observation(receipt, AUTH)


@pytest.mark.parametrize(
    ("disk_free", "available_ram", "used_swap", "reason"),
    [
        (30_499, 30_000, 1_000, "disk_operational_reserve"),
        (40_000, 19_999, 1_000, "available_ram_floor"),
        (40_000, 30_000, 2_001, "swap_usage_ceiling"),
    ],
)
def test_resource_floor_violation_refuses_with_authenticated_evidence(
    tmp_path: Path,
    disk_free: int,
    available_ram: int,
    used_swap: int,
    reason: str,
) -> None:
    with pytest.raises(ResourceFloorError) as raised:
        _sample_resources_with_providers(
            tmp_path,
            root_id="workspace:test",
            policy=_policy(),
            authenticator=AUTH,
            clock=lambda: FIXED_TIME,
            platform_name="darwin",
            disk_sampler=lambda fd: _disk(disk_free),
            memory_sampler=lambda: _memory(
                available=available_ram, used_swap=used_swap
            ),
            enforce=True,
        )

    refusal = raised.value.receipt
    assert refusal["status"] == "REFUSED"
    assert reason in refusal["refusal_reasons"]
    assert verify_authenticated_observation(refusal, AUTH) == refusal


def test_resource_refusal_can_be_returned_without_bypassing_status(
    tmp_path: Path,
) -> None:
    receipt = _sample_resources_with_providers(
        tmp_path,
        root_id="workspace:test",
        policy=_policy(),
        authenticator=AUTH,
        clock=lambda: FIXED_TIME,
        platform_name="linux",
        disk_sampler=lambda fd: _disk(1),
        memory_sampler=lambda: _memory(),
        enforce=False,
    )
    assert receipt["status"] == "REFUSED"
    assert receipt["disk_operational_reserve_ok"] is False
    verify_authenticated_observation(receipt, AUTH)


def test_resource_policy_rejects_bool_and_negative_values() -> None:
    with pytest.raises(GroundingError):
        ResourceReservePolicy(emergency_floor_bytes=True)
    with pytest.raises(GroundingError):
        ResourceReservePolicy(active_scratch_bytes=-1)


def test_public_resource_api_has_no_fact_or_freshness_injection_surface(
    tmp_path: Path,
) -> None:
    parameters = set(inspect.signature(sample_resources).parameters)
    assert parameters == {"root", "root_id", "policy", "authenticator"}

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        sample_resources(
            tmp_path,
            root_id="workspace:test",
            policy=ResourceReservePolicy(emergency_floor_bytes=0),
            authenticator=AUTH,
            disk_sampler=lambda fd: _disk(100_000),  # type: ignore[call-arg]
        )


def test_public_resource_api_samples_live_os_facts(tmp_path: Path) -> None:
    receipt = sample_resources(
        tmp_path,
        root_id="workspace:live-test",
        policy=ResourceReservePolicy(emergency_floor_bytes=0),
        authenticator=AUTH,
    )
    assert receipt["status"] == "PASS"
    assert receipt["platform"] in {"darwin", "linux"}
    assert receipt["memory_sample_source"] in {
        "darwin:sysctl+vm_stat",
        "linux:/proc/meminfo",
    }
    assert receipt["sampled_at"] != FIXED_TIME
    verify_authenticated_observation(
        receipt,
        AUTH,
        max_age_seconds=5,
    )
