#!/usr/bin/env python3.12
"""Offline adversarial tests for the GLM-5.2 live Xet executor foundation."""
from __future__ import annotations

import hashlib
import io
import json
import os
import pathlib
import signal
import subprocess
import sys
from typing import Any, Mapping

import pytest


CONDENSE = pathlib.Path(__file__).resolve().parents[1]
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import glm52_xet_live as live  # noqa: E402
from glm52_common import Glm52Error, canonical, seal, verify_sealed  # noqa: E402


TRIAL_IDS = [
    "DEFAULT_UNSET",
    "FILES_08", "FILES_16", "FILES_24", "FILES_32", "FILES_48",
    "FIXED_16", "FIXED_32", "FIXED_64", "HIGH_PERFORMANCE",
    "CACHE_1G_COLD", "CACHE_1G_REPLAY",
]
COUNTS = [8, 8, 16, 24, 32, 48, 8, 8, 8, 8, 8, 8]


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _fake_plan() -> dict[str, Any]:
    ranges = []
    for index in range(48):
        identity = {
            "schema": "hawking.glm52.xet_body_range_identity.v1",
            "path": f"model-{index + 1:05d}-of-00282.safetensors",
            "xet_hash": _sha(f"xet-{index}"),
            "lfs_sha256": _sha(f"lfs-{index}"),
            "start": 16,
            "end": 32,
            "length": 16,
        }
        ranges.append({
            "range_id_sha256": hashlib.sha256(canonical(identity)).hexdigest(),
            "path": identity["path"],
            "xet_hash": identity["xet_hash"],
            "lfs_sha256": identity["lfs_sha256"],
            "file_bytes": 64,
            "start": 16,
            "end": 32,
            "length": 16,
        })
    matrix = []
    for ordinal, (trial_id, count) in enumerate(zip(TRIAL_IDS, COUNTS)):
        environment: dict[str, str] = {}
        if trial_id.startswith("FILES_"):
            environment["HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS"] = str(
                int(trial_id.removeprefix("FILES_"))
            )
        if trial_id.startswith("FIXED_"):
            environment["HF_XET_FIXED_DOWNLOAD_CONCURRENCY"] = trial_id.removeprefix("FIXED_")
        if trial_id == "HIGH_PERFORMANCE":
            environment["HF_XET_HIGH_PERFORMANCE"] = "1"
        if trial_id.startswith("CACHE_"):
            environment["HF_XET_CHUNK_CACHE_SIZE_BYTES"] = str(1024**3)
        ids = [row["range_id_sha256"] for row in ranges[:count]]
        matrix.append({
            "ordinal": ordinal,
            "trial_id": trial_id,
            "kind": "BOUNDED_XET_BODY_RANGE",
            "caller_concurrent_shard_streams": count,
            "configured_file_download_limit": "TEST",
            "transfer_profile": "TEST",
            "high_performance": trial_id == "HIGH_PERFORMANCE",
            "chunk_cache_policy": "TEST",
            "environment": environment,
            "range_count": count,
            "ordered_range_ids": ids,
            "ordered_range_ids_sha256": hashlib.sha256(canonical(ids)).hexdigest(),
            "planned_payload_bytes": count * 16,
            "selectable_after_schedule_refreeze": True,
            "diagnostic_only_reason": None,
        })
    bounded = sum(row["planned_payload_bytes"] for row in matrix)
    largest_bytes = 64
    largest_hash = _sha("largest-full-body")
    return seal({
        "schema": live.autotune.PLAN_SCHEMA,
        "status": "PASS_OFFLINE_PLAN_BODY_NOT_READ",
        "repo": live.REPO_ID,
        "revision": live.REVISION,
        "inputs": [
            {
                "path": path,
                "schema": "test",
                "status": "PASS",
                "seal_sha256": _sha(path),
            }
            for path in (
                "GLM52_OFFICIAL_MANIFEST.json",
                "GLM52_SOURCE_FORMAT_LEDGER.json",
                "GLM52_SHARD_DEPENDENCY_GRAPH.json",
                "GLM52_SOURCE_ADMISSION.json",
            )
        ],
        "toolchain_binding": {"schema": "test", "sha256": _sha("toolchain")},
        "resource_reserve_policy": live._expected_resource_policy_binding(),
        "range_strategy": {"body_ranges": ranges, "range_bytes": 16},
        "trial_matrix": matrix,
        "largest_shard_validation": {
            "path": "model-00200-of-00282.safetensors",
            "bytes": largest_bytes,
            "xet_hash": _sha("largest-xet"),
            "lfs_sha256": largest_hash,
            "passes": ["acquisition pass", "steady pass"],
        },
        "network_budget": {
            "bounded_range_payload_bytes": bounded,
            "largest_shard_validation_bytes": largest_bytes * 2,
            "planned_maximum_bytes": bounded + largest_bytes * 2,
            "hard_cap_bytes": 100_000,
        },
    })


@pytest.fixture
def plan(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    value = _fake_plan()

    def accept_plan(
        candidate: Mapping[str, Any],
        *,
        root: pathlib.Path = live.REPO_ROOT,
        rebuild: bool = True,
    ) -> dict[str, Any]:
        del root, rebuild
        verify_sealed(dict(candidate))
        return dict(candidate)

    monkeypatch.setattr(live, "validate_live_plan", accept_plan)
    return value


def _spec(
    plan: Mapping[str, Any],
    trial_id: str = "DEFAULT_UNSET",
    *,
    capability_seal: str = "a" * 64,
    consumed: int = 0,
    cap: int = 10_000,
) -> dict[str, Any]:
    return live.build_trial_spec(
        plan,
        trial_id,
        capability_seal_sha256=capability_seal,
        campaign_consumed_bytes=consumed,
        trial_network_cap_bytes=cap,
        timeout_seconds=10,
        sample_interval_seconds=0.1,
        rebuild_plan=False,
    )


def _capability(spec: Mapping[str, Any], *, expires: int = 10**30) -> dict[str, Any]:
    return seal({
        "schema": live.CAPABILITY_SCHEMA,
        "status": "AUTHORIZED",
        "repo": live.REPO_ID,
        "revision": live.REVISION,
        "plan_seal_sha256": spec["plan_seal_sha256"],
        "trial_id": spec["trial"]["trial_id"],
        "allowed_kind": spec["trial"]["kind"],
        "max_network_bytes": spec["network_budget"]["trial_network_cap_bytes"],
        "controller": {
            "controller_epoch": "epoch-1",
            "checkpoint_seal_sha256": "b" * 64,
            "lease_identity_sha256": "c" * 64,
            "telegram_receipt_seal_sha256": "d" * 64,
        },
        "expires_unix_ns": expires,
        "credentials_serialized": False,
    })


class _Verifier:
    def __init__(self, accepted: bool = True) -> None:
        self.accepted = accepted
        self.calls = 0

    def verify_live_capability(self, capability: Mapping[str, Any], **kwargs: Any) -> bool:
        assert capability["credentials_serialized"] is False
        assert kwargs["spec"]["credentials_serialized"] is False
        self.calls += 1
        return self.accepted


def test_plan_verification_is_delegated_with_rebuild(monkeypatch: pytest.MonkeyPatch) -> None:
    value = _fake_plan()
    calls: list[tuple[pathlib.Path, bool]] = []

    def verify(candidate: Mapping[str, Any], *, root: pathlib.Path, rebuild: bool) -> dict[str, Any]:
        calls.append((root, rebuild))
        return dict(candidate)

    monkeypatch.setattr(live.autotune, "verify_plan", verify)
    assert live.validate_live_plan(value, root=pathlib.Path("/tmp/test"), rebuild=True) == value
    assert calls == [(pathlib.Path("/tmp/test"), True)]


def test_trial_spec_is_exact_sealed_and_projects_ordered_ranges(plan: dict[str, Any]) -> None:
    spec = _spec(plan, "FILES_16")
    verify_sealed(spec)
    assert set(spec) == live.SPEC_FIELDS
    assert spec["trial"] == plan["trial_matrix"][2]
    assert [item["range_id_sha256"] for item in spec["targets"]] == spec["trial"][
        "ordered_range_ids"
    ]
    assert sum(item["length"] for item in spec["targets"]) == spec["network_budget"][
        "planned_payload_bytes"
    ]
    assert spec["credentials_serialized"] is False
    assert spec["public_hub_auth"]["refresh_headers_serialized"] is False
    assert live.validate_trial_spec(spec, plan) == spec


def test_spec_and_result_reject_resource_policy_substitution(plan: dict[str, Any]) -> None:
    spec = _spec(plan)
    assert spec["resource_policy"] == live._expected_resource_policy_binding()
    substituted = json.loads(json.dumps(spec))
    substituted.pop("seal_sha256")
    substituted["resource_policy"]["minimum_available_ram_bytes"] -= 1
    with pytest.raises(Glm52Error, match="resource policy binding"):
        live.validate_trial_spec(seal(substituted), plan)

    result = _result_for_spec(
        spec,
        elapsed=0.1,
        actual_network=spec["trial"]["planned_payload_bytes"],
    )
    forged = json.loads(json.dumps(result))
    forged.pop("seal_sha256")
    forged["resource_policy"]["binding"]["maximum_swap_used_bytes"] -= 1
    with pytest.raises(Glm52Error, match="reserve policy"):
        live.validate_trial_result(seal(forged), plan=plan)


def test_spec_budget_and_environment_are_fail_closed(plan: dict[str, Any]) -> None:
    with pytest.raises(Glm52Error, match="does not fit the trial"):
        _spec(plan, "FILES_48", cap=100)
    with pytest.raises(Glm52Error, match="remaining campaign"):
        _spec(plan, consumed=99_999, cap=2)
    corrupt = json.loads(json.dumps(plan))
    corrupt["trial_matrix"][0]["environment"] = {"HF_TOKEN": "secret"}
    corrupt = seal(corrupt)
    with pytest.raises(Glm52Error, match="unapproved environment"):
        _spec(corrupt)


def test_capability_requires_independent_live_verifier_and_expiry(plan: dict[str, Any]) -> None:
    provisional = _spec(plan)
    capability = _capability(provisional)
    spec = _spec(plan, capability_seal=capability["seal_sha256"])
    verifier = _Verifier()
    assert live.validate_live_capability(
        capability, plan=plan, spec=spec, verifier=verifier, now_unix_ns=1
    ) == capability
    assert verifier.calls == 1
    with pytest.raises(Glm52Error, match="verifier is required"):
        live.validate_live_capability(
            capability, plan=plan, spec=spec, verifier=None, now_unix_ns=1
        )
    with pytest.raises(Glm52Error, match="refused"):
        live.validate_live_capability(
            capability, plan=plan, spec=spec, verifier=_Verifier(False), now_unix_ns=1
        )
    expired = _capability(provisional, expires=1)
    expired_spec = _spec(plan, capability_seal=expired["seal_sha256"])
    with pytest.raises(Glm52Error, match="expired"):
        live.validate_live_capability(
            expired, plan=plan, spec=expired_spec, verifier=_Verifier(), now_unix_ns=2
        )


def test_execute_refuses_without_explicit_environment_before_spawn(plan: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    spec = _spec(plan)
    capability = _capability(spec)
    monkeypatch.delenv(live.EXECUTE_ENV, raising=False)
    called = False

    def forbidden(*args: Any, **kwargs: Any) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    with pytest.raises(Glm52Error, match=live.EXECUTE_ENV):
        live.execute_trial(
            plan,
            spec,
            capability,
            capability_verifier=_Verifier(),
            resource_sampler=None,
            network_counter=None,
        )
    assert called is False


def test_child_environment_removes_inherited_xet_and_secret_execution_flags(plan: dict[str, Any]) -> None:
    spec = _spec(plan, "FIXED_32")
    env = live.child_environment(spec, {
        "PATH": "/bin",
        "HF_XET_FIXED_DOWNLOAD_CONCURRENCY": "999",
        "HF_XET_CLIENT_RETRY_MAX_ATTEMPTS": "99",
        live.EXECUTE_ENV: "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "0",
    })
    xet = {key: value for key, value in env.items() if key.startswith("HF_XET_")}
    assert xet == {
        "HF_XET_FIXED_DOWNLOAD_CONCURRENCY": "32",
        "HF_XET_LOG_DEST": "stderr",
        "HF_XET_LOG_FORMAT": "json",
    }
    assert live.EXECUTE_ENV not in env
    assert env[live.CHILD_ENV] == "1"
    assert env[live.SPEC_SEAL_ENV] == spec["seal_sha256"]
    assert env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"


def test_public_hub_group_uses_token_false_refresh_without_serializing_values() -> None:
    class Session:
        kwargs: dict[str, Any]

        def new_download_stream_group(self, **kwargs: Any) -> object:
            self.kwargs = kwargs
            return object()

    session = Session()
    _group, evidence = live._public_hub_stream_group(session)
    assert "token_refresh_url" in session.kwargs
    assert "/xet-read-token/" in session.kwargs["token_refresh_url"]
    assert all(key.lower() != "authorization" for key in session.kwargs["token_refresh_headers"])
    assert evidence["authorization_header_present"] is False
    assert evidence["header_values_serialized"] is False
    rendered = json.dumps(evidence)
    assert session.kwargs["token_refresh_url"] not in rendered
    assert all(value not in rendered for value in session.kwargs["token_refresh_headers"].values())


def test_streams_exact_ranges_to_sha256_in_memory_without_network(
    plan: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec(plan, "DEFAULT_UNSET")
    contents = {
        target["xet_hash"]: bytes([index]) * target["file_bytes"]
        for index, target in enumerate(spec["targets"], start=1)
    }

    class Stream:
        def __init__(self, data: bytes) -> None:
            self.data = data
            self.cancelled = False

        def __iter__(self):
            yield self.data[:5]
            yield self.data[5:]

        def cancel(self) -> None:
            self.cancelled = True

    class Group:
        def download_stream(self, info: Any, *, start: int, end: int) -> Stream:
            return Stream(contents[info.hash][start:end])

    class Session:
        aborted = False

        def sigint_abort(self) -> None:
            self.aborted = True

    monkeypatch.setattr(live, "_runtime_config", lambda: {
        "versions": live.PINNED_VERSIONS,
        "effective_xet_config": {
            **{key: 1 for key in live.RUNTIME_CONFIG_FIELDS},
            "log.dest": "stderr",
            "log.format": "json",
        },
    })
    ticks = iter(range(1, 10_000))
    result = live.stream_spec_in_memory(
        spec,
        session_factory=Session,
        group_factory=lambda _session: (Group(), {
            "mode": "TEST_NO_NETWORK",
            "authorization_header_present": False,
            "header_values_serialized": False,
            "xet_access_token_serialized": False,
        }),
        clock_ns=lambda: next(ticks) * 1_000_000_000,
    )
    verify_sealed(result)
    assert result["payload_bytes"] == spec["trial"]["planned_payload_bytes"]
    assert result["python_body_file_writes"] == 0
    for target, row in zip(spec["targets"], result["range_results"]):
        expected = contents[target["xet_hash"]][target["start"]:target["end"]]
        assert row["bytes"] == len(expected)
        assert row["sha256"] == hashlib.sha256(expected).hexdigest()


def test_xet_log_parser_preserves_missing_metrics_as_null() -> None:
    missing = live.parse_xet_json_logs([
        json.dumps({"level": "info", "message": "download finished"}) + "\n",
        "not-json\n",
    ])
    assert missing["retry_rate"] is None
    assert missing["maximum_reconstruction_latency_seconds"] is None
    assert missing["maximum_temporary_amplification_ratio"] is None
    assert missing["missing_metrics_are_zero"] is False
    assert missing["unparseable_lines"] == 1

    explicit = live.parse_xet_json_logs([
        json.dumps({
            "retry_rate": 0.125,
            "reconstruction_latency_ms": 250,
            "temporary_amplification_ratio": 1.3,
            "configuration": {"downloads": 16},
        }) + "\n",
    ])
    assert explicit["retry_rate"] == 0.125
    assert explicit["maximum_reconstruction_latency_seconds"] == 0.25
    assert explicit["maximum_temporary_amplification_ratio"] == 1.3
    assert explicit["configuration_related_json_events"] == 1
    assert explicit["raw_logs_serialized"] is False


def test_darwin_parsers_are_strict_and_do_not_default_to_zero() -> None:
    assert live.parse_swapusage(
        "total = 2048.00M  used = 325.75M  free = 1722.25M  (encrypted)"
    ) == int(325.75 * 1024**2)
    vm = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free: 10.
Pages inactive: 20.
Pages speculative: 3.
Swapouts: 7.
"""
    assert live.parse_vm_stat(vm) == (33 * 16384, 7)
    assert live.parse_thermal_warning("Note: No thermal warning level has been recorded") is False
    assert live.parse_thermal_warning("Thermal Warning Level: 1") is True
    assert live.parse_ps_cpu_rss(" 25.5 1024\n") == (25.5, 1024**2)
    assert live.parse_iostat_total_bytes("disk0\n KB/t xfrs MB\n 10 2 3.5\n") == int(3.5 * 1024**2)
    netstat = """Name Mtu Network Address Ipkts Ierrs Ibytes Opkts Oerrs Obytes Coll
lo0 16384 <Link#1> 10 0 100 10 0 100 0
en0 1500 <Link#2> aa:bb 20 0 200 30 0 300 0
bridge0* 1500 <Link#3> aa:cc 40 0 400 50 0 500 0
"""
    assert live.parse_netstat_link_bytes(netstat) == 500
    assert live.parse_du_allocated_bytes("123\t/tmp/xet\n") == 123 * 1024
    with pytest.raises(Glm52Error):
        live.parse_swapusage("unavailable")
    with pytest.raises(Glm52Error):
        live.parse_vm_stat("page size of 4096 bytes")
    with pytest.raises(Glm52Error):
        live.parse_thermal_warning("")


def test_resource_sample_rejects_missing_or_unknown_fields() -> None:
    sample = _resource_sample(123)
    assert live._validate_resource_sample(sample, pid=123) == sample
    missing = dict(sample)
    missing.pop("swapouts")
    with pytest.raises(Glm52Error, match="incomplete"):
        live._validate_resource_sample(missing, pid=123)
    unknown = {**sample, "made_up_zero": 0}
    with pytest.raises(Glm52Error, match="unknown"):
        live._validate_resource_sample(unknown, pid=123)


class _TimeoutProcess:
    pid = 777

    def __init__(self) -> None:
        self.waits = 0

    def poll(self) -> None:
        return None

    def wait(self, timeout: float) -> int:
        del timeout
        self.waits += 1
        if self.waits == 1:
            raise subprocess.TimeoutExpired("child", 1)
        return -signal.SIGKILL


def test_child_cancellation_escalates_from_sigint_to_sigkill(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "killpg", lambda pid, sig: sent.append((pid, sig)))
    process = _TimeoutProcess()
    live._cancel_child(process, grace_seconds=0.01)  # type: ignore[arg-type]
    assert sent == [(777, signal.SIGINT), (777, signal.SIGKILL)]


def _resource_sample(pid: int, **updates: Any) -> dict[str, Any]:
    value = {
        "schema": live.RESOURCE_SAMPLE_SCHEMA,
        "sampled_monotonic_ns": 1,
        "pid": pid,
        "swap_used_bytes": 100,
        "swapouts": 7,
        "thermal_warning": False,
        "free_disk_bytes": live.autotune.REQUIRED_FREE_DISK_BYTES + 10_000_000,
        "available_ram_bytes": live.autotune.MINIMUM_AVAILABLE_RAM_BYTES + 10_000_000,
        "cpu_percent": 20.0,
        "process_rss_bytes": 1_000,
        "disk_write_bytes_per_second": 100.0,
        "materialized_raw_allocated_bytes": 1_000,
        "reconstruction_latency_seconds": 0.1,
        "retry_rate": 0.01,
        "temporary_amplification_ratio": 1.1,
    }
    value.update(updates)
    return value


@pytest.mark.parametrize(
    ("updates", "reason"),
    [
        ({"thermal_warning": True}, "THERMAL_WARNING"),
        ({"free_disk_bytes": live.autotune.REQUIRED_FREE_DISK_BYTES - 1},
         "DISK_FLOOR_RISK"),
        ({"available_ram_bytes": live.autotune.MINIMUM_AVAILABLE_RAM_BYTES - 1},
         "RAM_FLOOR_RISK"),
        ({"swap_used_bytes": live.autotune.MAXIMUM_SWAP_USED_BYTES + 1},
         "ABSOLUTE_SWAP_CEILING"),
        ({"swap_used_bytes": 101}, "SWAP_GROWTH"),
        ({"swapouts": 8}, "NEW_SWAPOUTS"),
        ({"materialized_raw_allocated_bytes":
          live.autotune.MAXIMUM_MATERIALIZED_RAW_ALLOCATED_BYTES + 1},
         "MATERIALIZED_RAW_ALLOCATION_CEILING"),
        ({"materialized_raw_allocated_bytes": 10_001_001},
         "LIVE_ALLOCATION_GROWTH_CEILING"),
    ],
)
def test_live_resource_policy_enforcement_is_fail_closed(
    updates: dict[str, Any],
    reason: str,
) -> None:
    baseline = _resource_sample(123)
    sample = _resource_sample(123, **updates)
    with pytest.raises(Glm52Error, match=reason):
        live._enforce_resource_policy_sample(
            sample,
            baseline=baseline,
            policy=live._expected_resource_policy_binding(),
        )


def _log_evidence() -> dict[str, Any]:
    return live.parse_xet_json_logs([
        json.dumps({
            "retry_rate": 0.01,
            "reconstruction_latency_seconds": 0.1,
            "temporary_amplification_ratio": 1.1,
        }) + "\n"
    ])


def _child_for_spec(
    spec: Mapping[str, Any],
    *,
    elapsed: float,
    largest_digest: str | None = None,
) -> dict[str, Any]:
    rows = []
    for target in spec["targets"]:
        rows.append({
            "range_id_sha256": target["range_id_sha256"],
            "path": target["path"],
            "start": target["start"],
            "end": target["end"],
            "bytes": target["length"],
            "sha256": largest_digest or _sha(target["range_id_sha256"]),
            "elapsed_seconds": elapsed,
        })
    effective = {key: 1 for key in live.RUNTIME_CONFIG_FIELDS}
    effective["client.ac_initial_download_concurrency"] = 4
    effective["log.dest"] = "stderr"
    effective["log.format"] = "json"
    return seal({
        "schema": live.CHILD_RESULT_SCHEMA,
        "status": "PASS_STREAMED_IN_MEMORY",
        "spec_seal_sha256": spec["seal_sha256"],
        "plan_seal_sha256": spec["plan_seal_sha256"],
        "trial_id": spec["trial"]["trial_id"],
        "kind": spec["trial"]["kind"],
        "pid": 123,
        "runtime": {
            "versions": live.PINNED_VERSIONS,
            "effective_xet_config": effective,
        },
        "public_hub_auth": {
            "mode": "TEST_PUBLIC",
            "authorization_header_present": False,
            "header_values_serialized": False,
            "xet_access_token_serialized": False,
        },
        "range_results": rows,
        "payload_bytes": sum(row["bytes"] for row in rows),
        "started_monotonic_ns": 1,
        "finished_monotonic_ns": 2,
        "elapsed_seconds": elapsed,
        "python_body_file_writes": 0,
        "error": None,
    })


class _WritePipe:
    def __init__(self) -> None:
        self.writes: list[str] = []

    def write(self, value: str) -> int:
        self.writes.append(value)
        return len(value)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


class _ProtocolProcess:
    pid = 123

    def __init__(self, spec: Mapping[str, Any], child: Mapping[str, Any]) -> None:
        self.stdin = _WritePipe()
        self.stdout = io.StringIO("".join((
            json.dumps({
                "protocol": live.CHILD_PROTOCOL,
                "status": "READY",
                "spec_seal_sha256": spec["seal_sha256"],
            }) + "\n",
            json.dumps({
                "protocol": live.CHILD_PROTOCOL,
                "status": "RESULT",
                "result": child,
            }) + "\n",
        )))
        self.stderr = io.StringIO(json.dumps({
            "retry_rate": 0.01,
            "reconstruction_latency_seconds": 0.01,
            "temporary_amplification_ratio": 1.01,
        }) + "\n")
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float) -> int:
        del timeout
        self.returncode = 0
        return 0


class _Samples:
    def __init__(self, samples: list[dict[str, Any]]) -> None:
        self.samples = iter(samples)
        self.calls = 0

    def sample(self, pid: int) -> Mapping[str, Any]:
        assert pid == 123
        self.calls += 1
        return next(self.samples)


class _Network:
    def __init__(self, counters: list[int]) -> None:
        self.counters = iter(counters)
        self.calls = 0

    def snapshot(self) -> int:
        self.calls += 1
        return next(self.counters)

    def evidence(self) -> Mapping[str, Any]:
        return {
            "schema": live.NETWORK_COUNTER_SCHEMA,
            "method": "TEST_FAKE_NO_NETWORK",
            "scope": "TEST",
            "counts_unrelated_host_traffic": True,
            "monotonicity_required": True,
            "credentials_serialized": False,
        }


def _authorized_spec(plan: Mapping[str, Any], *, cap: int = 10_000):
    provisional = _spec(plan, cap=cap)
    capability = _capability(provisional)
    return (
        _spec(plan, capability_seal=capability["seal_sha256"], cap=cap),
        capability,
    )


def test_subsecond_protocol_gets_three_real_samples_without_body_or_network(
    plan: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, capability = _authorized_spec(plan)
    child = _child_for_spec(spec, elapsed=0.02)
    process = _ProtocolProcess(spec, child)
    sampler = _Samples([
        _resource_sample(123, sampled_monotonic_ns=index)
        for index in (1, 2, 3)
    ])
    actual = spec["trial"]["planned_payload_bytes"]
    counter = _Network([1_000, 1_010, 1_000 + actual, 1_000 + actual])
    monkeypatch.setenv(live.EXECUTE_ENV, "1")
    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: process)

    result = live.execute_trial(
        plan,
        spec,
        capability,
        capability_verifier=_Verifier(),
        resource_sampler=sampler,
        network_counter=counter,
    )
    assert result["status"] == "PASS_COMPLETE_MEASURED"
    assert result["resource_policy"]["sample_count"] == 3
    assert len(result["resource_observations"]["samples"]) == 1
    assert sampler.calls == 3
    assert counter.calls == 4
    assert result["network_accounting"]["post_child_exit_counter_included"] is True
    assert result["network_accounting"][
        "cap_crossing_cancellation_then_counter_required"
    ] is True
    assert result["body_persistence"]["python_body_file_writes"] == 0


def test_cap_crossing_cancels_then_accounts_post_cancel_counter(
    plan: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, capability = _authorized_spec(plan, cap=1_000)
    process = _ProtocolProcess(spec, _child_for_spec(spec, elapsed=0.02))
    sampler = _Samples([_resource_sample(123), _resource_sample(123)])
    counter = _Network([1_000, 2_001, 2_010])
    cancellations: list[int] = []
    monkeypatch.setenv(live.EXECUTE_ENV, "1")
    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(
        live,
        "_cancel_child",
        lambda candidate: cancellations.append(candidate.pid),
    )
    with pytest.raises(Glm52Error, match="cancellation-accounted delta=1010"):
        live.execute_trial(
            plan,
            spec,
            capability,
            capability_verifier=_Verifier(),
            resource_sampler=sampler,
            network_counter=counter,
        )
    assert cancellations == [123, 123]
    assert counter.calls == 3
    assert not any('"command":"ACK"' in item for item in process.stdin.writes)


def test_pre_go_thermal_gate_cancels_before_go(
    plan: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, capability = _authorized_spec(plan)
    process = _ProtocolProcess(spec, _child_for_spec(spec, elapsed=0.02))
    sampler = _Samples([_resource_sample(123, thermal_warning=True)])
    counter = _Network([1_000])
    cancellations: list[int] = []
    monkeypatch.setenv(live.EXECUTE_ENV, "1")
    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(
        live,
        "_cancel_child",
        lambda candidate: cancellations.append(candidate.pid),
    )
    with pytest.raises(Glm52Error, match="THERMAL_WARNING"):
        live.execute_trial(
            plan,
            spec,
            capability,
            capability_verifier=_Verifier(),
            resource_sampler=sampler,
            network_counter=counter,
        )
    assert cancellations == [123]
    assert counter.calls == 0
    assert not any('"command":"GO"' in item for item in process.stdin.writes)


def _result_for_spec(
    spec: Mapping[str, Any],
    *,
    elapsed: float,
    actual_network: int,
    largest_digest: str | None = None,
) -> dict[str, Any]:
    child = _child_for_spec(spec, elapsed=elapsed, largest_digest=largest_digest)
    samples = [
        _resource_sample(123, sampled_monotonic_ns=index)
        for index in (1, 2, 3)
    ]
    result = live._build_trial_result(
        spec,
        child,
        raw_samples=samples,
        network_samples=[1_000, 1_000 + actual_network // 2, 1_000 + actual_network],
        network_evidence={
            "schema": live.NETWORK_COUNTER_SCHEMA,
            "method": "TEST",
            "scope": "TEST",
            "counts_unrelated_host_traffic": True,
            "monotonicity_required": True,
            "credentials_serialized": False,
        },
        log_evidence=_log_evidence(),
    )
    return result


def test_result_marks_missing_retry_metric_incomplete_not_zero(plan: dict[str, Any]) -> None:
    spec = _spec(plan)
    child = _child_for_spec(spec, elapsed=1.0)
    samples = [_resource_sample(123) for _ in range(3)]
    for sample in samples:
        sample.pop("retry_rate")
    result = live._build_trial_result(
        spec,
        child,
        raw_samples=samples,
        network_samples=[10, 20, 30],
        network_evidence={"method": "TEST"},
        log_evidence=live.parse_xet_json_logs([]),
    )
    assert result["status"] == "INCOMPLETE_REQUIRED_XET_METRICS"
    assert result["resource_observations"]["missing_metrics"] == ["retry_rate"]
    assert result["resource_observations"]["before"]["retry_rate"] is None


def test_overall_assembler_requires_all_12_and_two_full_largest_hashes(plan: dict[str, Any]) -> None:
    results = []
    consumed = 0
    for ordinal, trial_id in enumerate(TRIAL_IDS):
        provisional = _spec(plan, trial_id, consumed=consumed)
        capability = _capability(provisional)
        spec = _spec(
            plan,
            trial_id,
            capability_seal=capability["seal_sha256"],
            consumed=consumed,
        )
        actual = spec["trial"]["planned_payload_bytes"]
        result = _result_for_spec(spec, elapsed=1.0 + ordinal, actual_network=actual)
        assert live.validate_trial_result(result, plan=plan) == result
        results.append(result)
        consumed += actual

    validations = []
    largest = plan["largest_shard_validation"]
    for lane in ("acquisition", "steady"):
        preliminary = live.build_largest_validation_spec(
            plan,
            lane=lane,
            selected_trial_id="DEFAULT_UNSET",
            capability_seal_sha256="a" * 64,
            campaign_consumed_bytes=consumed,
            trial_network_cap_bytes=10_000,
            rebuild_plan=False,
        )
        capability = _capability(preliminary)
        spec = live.build_largest_validation_spec(
            plan,
            lane=lane,
            selected_trial_id="DEFAULT_UNSET",
            capability_seal_sha256=capability["seal_sha256"],
            campaign_consumed_bytes=consumed,
            trial_network_cap_bytes=10_000,
            rebuild_plan=False,
        )
        result = _result_for_spec(
            spec,
            elapsed=2.0,
            actual_network=largest["bytes"],
            largest_digest=largest["lfs_sha256"],
        )
        evidence = live.build_largest_validation_evidence(
            plan,
            result,
            lane=lane,
            selected_trial_id="DEFAULT_UNSET",
        )
        validations.append(evidence)
        consumed += largest["bytes"]

    overall = live.assemble_autotune_result(
        plan,
        results,
        validations,
        required_free_bytes=live.autotune.REQUIRED_FREE_DISK_BYTES,
        required_available_ram_bytes=live.autotune.MINIMUM_AVAILABLE_RAM_BYTES,
        rebuild_plan=False,
    )
    verify_sealed(overall)
    assert overall["status"] == "PASS_LIVE_XET_AUTOTUNE_COMPLETE_SCHEDULE_REFREEZE_REQUIRED"
    assert overall["coverage"]["trial_ids_in_plan_order"] == TRIAL_IDS
    assert overall["coverage"]["required_file_settings_measured"] == [8, 16, 24, 32, 48]
    assert overall["coverage"]["repeated_range_sha256_consistent"] is True
    assert overall["selections"]["acquisition"]["trial_id"] == "DEFAULT_UNSET"
    assert overall["selections"]["steady"]["trial_id"] == "DEFAULT_UNSET"
    assert overall["selected_profile"] == {
        "acquisition": overall["selections"]["acquisition"],
        "steady": overall["selections"]["steady"],
    }
    assert live.validate_autotune_result(overall, plan=plan) == overall
    assert len(overall["largest_shard_validations"]) == 2
    assert overall["network_budget"]["planned_total_payload_bytes"] == plan["network_budget"][
        "planned_maximum_bytes"
    ]
    assert overall["claim_boundary"]["streaming_schedule_refreeze_required"] is True
    assert overall["claim_boundary"]["full_model_downloaded"] is False

    divergent_profile = json.loads(json.dumps(overall))
    divergent_profile["selected_profile"]["steady"] = divergent_profile["selected_profile"][
        "acquisition"
    ]
    divergent_profile = seal(divergent_profile)
    with pytest.raises(Glm52Error, match="selected_profile"):
        live.validate_autotune_result(divergent_profile, plan=plan)

    missing_setting = json.loads(json.dumps(overall))
    missing_setting["coverage"]["required_file_settings_measured"] = [8, 16, 24, 32]
    missing_setting = seal(missing_setting)
    with pytest.raises(Glm52Error, match="coverage"):
        live.validate_autotune_result(missing_setting, plan=plan)

    wrong_full_hash = json.loads(json.dumps(overall))
    wrong_full_hash["largest_shard_validations"][1]["observed_sha256"] = "f" * 64
    wrong_full_hash = seal(wrong_full_hash)
    with pytest.raises(Glm52Error, match="largest-shard"):
        live.validate_autotune_result(wrong_full_hash, plan=plan)

    with pytest.raises(Glm52Error, match="exactly 12"):
        live.assemble_autotune_result(
            plan,
            results[:-1],
            validations,
            required_free_bytes=live.autotune.REQUIRED_FREE_DISK_BYTES,
            required_available_ram_bytes=live.autotune.MINIMUM_AVAILABLE_RAM_BYTES,
            rebuild_plan=False,
        )
    with pytest.raises(Glm52Error, match="two full"):
        live.assemble_autotune_result(
            plan,
            results,
            validations[:1],
            required_free_bytes=live.autotune.REQUIRED_FREE_DISK_BYTES,
            required_available_ram_bytes=live.autotune.MINIMUM_AVAILABLE_RAM_BYTES,
            rebuild_plan=False,
        )
    with pytest.raises(Glm52Error, match="caller resource floors differ"):
        live.assemble_autotune_result(
            plan,
            results,
            validations,
            required_free_bytes=live.autotune.REQUIRED_FREE_DISK_BYTES - 1,
            required_available_ram_bytes=live.autotune.MINIMUM_AVAILABLE_RAM_BYTES,
            rebuild_plan=False,
        )


def test_module_has_no_destination_download_or_destructive_gc_surface() -> None:
    source = (CONDENSE / "glm52_xet_live.py").read_text(encoding="utf-8")
    forbidden = (
        "start_download_file(",
        "download_files(",
        "hf_hub_download(",
        "snapshot_download(",
        "shutil.rmtree(",
        ".unlink(",
        "os.remove(",
    )
    assert all(token not in source for token in forbidden)
    assert ".download_stream(" in source
    assert "XetSession" in source
