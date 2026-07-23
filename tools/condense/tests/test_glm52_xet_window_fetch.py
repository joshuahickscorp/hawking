#!/usr/bin/env python3.12
"""Zero-network adversarial tests for selected-profile Xet window materialization."""
from __future__ import annotations

import copy
import hashlib
import os
import pathlib
import sys
import threading
import time
from typing import Any, Iterable, Mapping

import pytest


TESTS = pathlib.Path(__file__).resolve().parent
CONDENSE = TESTS.parent
REPO_ROOT = CONDENSE.parents[1]
for path in (str(CONDENSE), str(TESTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

import glm52_grounding as grounding  # noqa: E402
import glm52_schedule_freeze as freezer  # noqa: E402
import glm52_state as state  # noqa: E402
import glm52_xet_window_fetch as fetch  # noqa: E402
import test_glm52_schedule_freeze as schedule_helpers  # noqa: E402
from glm52_common import seal, utc_now, verify_sealed  # noqa: E402


PAYLOAD = b"selected-profile-xet-window-materialization-test-payload!".ljust(64, b".")
GROUNDING_KEY = b"glm52-window-fetch-grounding-test-key-material!!"


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class _CapabilityVerifier:
    def __init__(self, accepted: bool = True) -> None:
        self.accepted = accepted
        self.calls = 0

    def verify_window_fetch_capability(
        self,
        intent: Mapping[str, Any],
        *,
        frozen_schedule: Mapping[str, Any],
        resource_policy: Mapping[str, Any],
    ) -> bool:
        self.calls += 1
        assert intent["frozen_schedule_seal_sha256"] == frozen_schedule["seal_sha256"]
        assert intent["resource_policy_seal_sha256"] == resource_policy["seal_sha256"]
        return self.accepted


class _FakeStreamProvider:
    def __init__(self, payloads: Mapping[str, bytes], *, chunk_bytes: int = 11) -> None:
        self.payloads = dict(payloads)
        self.chunk_bytes = chunk_bytes
        self.abort_calls = 0
        self.opened: list[str] = []
        self.active = 0
        self.maximum_active = 0
        self._lock = threading.Lock()

    def open_stream(self, target: Mapping[str, Any]) -> Iterable[bytes]:
        path = str(target["path"])
        self.opened.append(path)
        payload = self.payloads[path]

        def chunks() -> Iterable[bytes]:
            with self._lock:
                self.active += 1
                self.maximum_active = max(self.maximum_active, self.active)
            try:
                for offset in range(0, len(payload), self.chunk_bytes):
                    yield payload[offset:offset + self.chunk_bytes]
            finally:
                with self._lock:
                    self.active -= 1

        return chunks()

    def abort(self) -> None:
        self.abort_calls += 1


class _FakeResourceSampler:
    def __init__(
        self,
        *,
        free_bytes: int = 600_000_000_000,
        inline_free: list[int] | None = None,
    ) -> None:
        self.free_bytes_value = free_bytes
        self.inline_free = list(inline_free or [])
        self.authenticated_calls = 0
        self.inline_calls = 0
        self.live_memory_calls = 0

    def authenticated_sample(
        self,
        root: str | os.PathLike[str],
        *,
        root_id: str,
        policy: grounding.ResourceReservePolicy,
        authenticator: grounding.ProducerAuthenticator,
    ) -> Mapping[str, Any]:
        self.authenticated_calls += 1
        free = self.free_bytes_value

        def disk(root_fd: int) -> grounding.DiskSample:
            return grounding.DiskSample(
                total_bytes=1_000_000_000_000,
                free_bytes=free,
                used_bytes=1_000_000_000_000 - free,
                device=int(os.fstat(root_fd).st_dev),
            )

        return grounding._sample_resources_with_providers(
            root,
            root_id=root_id,
            policy=policy,
            authenticator=authenticator,
            clock=utc_now,
            platform_name="darwin",
            disk_sampler=disk,
            memory_sampler=lambda: grounding.MemorySample(
                total_ram_bytes=96 * 1024**3,
                available_ram_bytes=64 * 1024**3,
                total_swap_bytes=16 * 1024**3,
                used_swap_bytes=0,
                source="offline-fake-memory",
            ),
            enforce=True,
        )

    def free_disk_bytes(self, root_fd: int) -> int:
        del root_fd
        self.inline_calls += 1
        if self.inline_free:
            return self.inline_free.pop(0)
        return self.free_bytes_value

    def allocation_unit_bytes(self, root_fd: int) -> int:
        del root_fd
        return 4096

    def live_memory_sample(self) -> Mapping[str, Any]:
        self.live_memory_calls += 1
        return _live_sample()


def _live_sample(
    *,
    available_ram_bytes: int = 64 * 1024**3,
    swap_used_bytes: int = 0,
    swapouts: int = 0,
) -> dict[str, Any]:
    return {
        "schema": fetch.LIVE_MEMORY_SAMPLE_SCHEMA,
        "sampled_monotonic_ns": time.monotonic_ns(),
        "available_ram_bytes": available_ram_bytes,
        "swap_used_bytes": swap_used_bytes,
        "swapouts": swapouts,
        "source": "offline-fake-live-memory",
    }


class _SequencedMemorySampler(_FakeResourceSampler):
    def __init__(self, samples: Iterable[Mapping[str, Any]]) -> None:
        super().__init__()
        self.samples = [dict(sample) for sample in samples]
        if not self.samples:
            raise ValueError("at least one live sample is required")
        self._last = dict(self.samples[-1])
        self._memory_lock = threading.Lock()

    def live_memory_sample(self) -> Mapping[str, Any]:
        with self._memory_lock:
            self.live_memory_calls += 1
            raw = self.samples.pop(0) if self.samples else dict(self._last)
        raw["sampled_monotonic_ns"] = time.monotonic_ns()
        return raw


class _AbortBlockingProvider(_FakeStreamProvider):
    """Open one stream that cannot finish until the monitor aborts it."""

    def __init__(self, payloads: Mapping[str, bytes]) -> None:
        super().__init__(payloads)
        self.started = threading.Event()
        self.aborted = threading.Event()

    def open_stream(self, target: Mapping[str, Any]) -> Iterable[bytes]:
        path = str(target["path"])
        self.opened.append(path)

        def blocked() -> Iterable[bytes]:
            with self._lock:
                self.active += 1
                self.maximum_active = max(self.maximum_active, self.active)
            self.started.set()
            try:
                if not self.aborted.wait(2):
                    raise AssertionError("continuous resource monitor did not abort stream")
                if False:  # pragma: no cover - makes this a zero-byte generator
                    yield b""
            finally:
                with self._lock:
                    self.active -= 1

        return blocked()

    def abort(self) -> None:
        self.abort_calls += 1
        self.aborted.set()


def _unit_memory_monitor() -> fetch._ContinuousMemoryMonitor:
    return fetch._ContinuousMemoryMonitor(
        sampler=_FakeResourceSampler(),
        policy=grounding.ResourceReservePolicy(
            minimum_available_ram_bytes=16 * 1024**3,
            maximum_swap_used_bytes=8 * 1024**3,
        ),
        abort_event=threading.Event(),
        provider=_FakeStreamProvider({}),
        fetch_intent_seal_sha256="a" * 64,
        resource_policy_seal_sha256="b" * 64,
        resource_before_seal_sha256="c" * 64,
        source_root_identity={
            "normalized_path_sha256": "d" * 64,
            "device": 1,
            "inode": 2,
        },
    )


@pytest.fixture
def bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> dict[str, Any]:
    # The schedule-freeze helper intentionally exercises a reduced fake plan
    # predating the live executor's exact resource-policy binding.  Bring the
    # private fixture forward locally so this adapter's tests exercise the
    # current authenticated plan/result contract without weakening a validator.
    plan_body = copy.deepcopy(schedule_helpers._fake_plan())
    plan_body.pop("seal_sha256")
    plan_body["resource_reserve_policy"] = (
        freezer.xet_live._expected_resource_policy_binding()
    )
    plan = seal(plan_body)

    def accept_plan(
        candidate: Mapping[str, Any],
        *,
        root: pathlib.Path = freezer.REPO_ROOT,
        rebuild: bool = True,
    ) -> dict[str, Any]:
        del root, rebuild
        return dict(verify_sealed(dict(candidate), label="offline fake Xet plan"))

    monkeypatch.setattr(freezer.xet_live, "validate_live_plan", accept_plan)
    preliminary = schedule_helpers._preliminary()
    contract = schedule_helpers._contract(plan, preliminary)
    contract_body = copy.deepcopy(contract)
    contract_body.pop("seal_sha256")
    contract_body["source"]["shards"][0]["lfs_sha256"] = _sha_bytes(PAYLOAD)
    contract = seal(contract_body)
    auth = schedule_helpers._auth()
    raw_body = copy.deepcopy(schedule_helpers._raw_result(plan))
    raw_body.pop("seal_sha256")
    raw_body["bindings"]["resource_reserve_policy"] = copy.deepcopy(
        plan["resource_reserve_policy"]
    )
    raw = seal(raw_body)
    attested = freezer.attest_xet_autotune_result(
        raw,
        plan,
        contract,
        auth=auth,
        controller_anchor_sha256=hashlib.sha256(b"schedule-anchor").hexdigest(),
        rebuild_plan=False,
    )
    schedule = freezer.freeze_schedule(
        preliminary,
        plan,
        attested,
        contract,
        auth=auth,
        rebuild_plan=False,
    )
    resource_policy = state.read_sealed_json(
        REPO_ROOT / "GLM52_RESOURCE_RESERVE_POLICY.json"
    )
    artifacts = fetch.FetchArtifacts(
        expected_contract=contract,
        preliminary_schedule=preliminary,
        xet_autotune_plan=plan,
        producer_attested_xet_result=attested,
        frozen_schedule=schedule,
        resource_policy=resource_policy,
    )
    source_root = tmp_path / "source"
    source_root.mkdir()
    intent = fetch.build_window_fetch_intent(
        artifacts,
        schedule_index=0,
        lane="acquisition",
        source_root=source_root,
        controller_anchor_sha256=hashlib.sha256(b"fetch-anchor").hexdigest(),
        authority_nonce_sha256=hashlib.sha256(b"fetch-nonce").hexdigest(),
        auth=auth,
        rebuild_plan=False,
    )
    return {
        "plan": plan,
        "contract": contract,
        "auth": auth,
        "artifacts": artifacts,
        "source_root": source_root,
        "intent": intent,
        "grounding_auth": grounding.ProducerAuthenticator(GROUNDING_KEY),
        "path": schedule_helpers.SHARD,
    }


def _materialize(
    bundle: Mapping[str, Any],
    *,
    provider: _FakeStreamProvider,
    sampler: _FakeResourceSampler | None = None,
    verifier: _CapabilityVerifier | None = None,
    intent: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return fetch.materialize_window(
        bundle["artifacts"],
        bundle["intent"] if intent is None else intent,
        source_root=bundle["source_root"],
        auth=bundle["auth"],
        grounding_authenticator=bundle["grounding_auth"],
        capability_verifier=verifier or _CapabilityVerifier(),
        stream_provider=provider,
        resource_sampler=sampler or _FakeResourceSampler(),
        rebuild_plan=False,
    )


def test_fake_stream_materializes_exact_scheduled_shard_and_authenticates_receipt(
    bundle: Mapping[str, Any],
) -> None:
    provider = _FakeStreamProvider({bundle["path"]: PAYLOAD})
    sampler = _FakeResourceSampler()
    verifier = _CapabilityVerifier()
    receipt = _materialize(
        bundle,
        provider=provider,
        sampler=sampler,
        verifier=verifier,
    )
    published = bundle["source_root"] / bundle["path"]
    assert published.read_bytes() == PAYLOAD
    assert stat_mode(published) == 0o444
    assert published.stat().st_nlink == 1
    assert provider.opened == [bundle["path"]]
    assert provider.maximum_active == 1
    assert verifier.calls == 1
    assert sampler.authenticated_calls == 2
    assert receipt["status"] == "PASS_EXPLICIT_ADAPTER_MATERIALIZATION"
    assert receipt["streamed_body_bytes"] == len(PAYLOAD)
    assert receipt["published_paths"] == [bundle["path"]]
    assert receipt["retained_partial_paths"] == []
    assert receipt["worker_dispatch_enabled"] is False
    monitor = fetch._producer_verify(
        receipt["continuous_resource_monitor"],
        auth=bundle["auth"],
        label="test continuous resource monitor",
    )
    assert monitor["status"] == "PASS_CONTINUOUS_RAM_SWAP_GUARD"
    assert monitor["policy"] == {
        "minimum_available_ram_bytes": 16 * 1024**3,
        "maximum_swap_used_bytes": 8 * 1024**3,
        "maximum_swap_growth_bytes": 0,
        "maximum_new_swapouts": 0,
    }
    assert monitor["bindings"]["fetch_intent_seal_sha256"] == (
        bundle["intent"]["seal_sha256"]
    )
    assert monitor["bindings"]["resource_policy_seal_sha256"] == (
        bundle["artifacts"].resource_policy["seal_sha256"]
    )
    assert monitor["bindings"]["source_root"] == bundle["intent"]["source_root"]
    assert monitor["sample_attempt_count"] >= 2
    assert monitor["failure_reasons"] == []
    assert monitor["monitor_thread_quiesced"] is True
    assert monitor["stream_workers_quiesced"] is True
    assert monitor["final_sample_after_stream_quiescence"] is True
    assert monitor["provider_abort_requested"] is False
    assert monitor["coverage"] == {
        "started_before_stream_workers": True,
        "covered_entire_stream_worker_lifetime": True,
        "covered_authenticated_prepublish_sample": True,
        "covered_durable_publish_boundary": True,
        "ended_after_publish_boundary": True,
        "unobserved_intervals_claimed_safe": False,
    }
    assert monitor["final_sample_after_publish_boundary"] is True
    verified = fetch._producer_verify(
        receipt,
        auth=bundle["auth"],
        label="test fetch receipt",
    )
    assert verified == receipt
    assert not any("glm52-partial" in path.name for path in bundle["source_root"].iterdir())


def stat_mode(path: pathlib.Path) -> int:
    import stat
    return stat.S_IMODE(path.stat().st_mode)


@pytest.mark.parametrize(
    "payload, message",
    [
        (b"wrong".ljust(64, b"!"), "LFS SHA-256 mismatch"),
        (PAYLOAD + b"overflow", "exceeded the exact shard size"),
    ],
)
def test_hash_or_size_failure_retains_partial_and_publishes_nothing(
    bundle: Mapping[str, Any], payload: bytes, message: str
) -> None:
    provider = _FakeStreamProvider({bundle["path"]: payload})
    with pytest.raises(fetch.WindowFetchError, match=message) as captured:
        _materialize(bundle, provider=provider)
    destination = bundle["source_root"] / bundle["path"]
    assert not destination.exists()
    partials = list(bundle["source_root"].glob(".*glm52-partial-*"))
    assert len(partials) == 1
    assert captured.value.retained_partials
    assert captured.value.published_paths == ()
    assert partials[0].read_bytes()


def test_static_destination_symlink_and_partial_symlink_never_touch_victim(
    bundle: Mapping[str, Any],
) -> None:
    victim = bundle["source_root"].parent / "victim"
    victim_bytes = b"do-not-touch"
    victim.write_bytes(victim_bytes)
    destination = bundle["source_root"] / bundle["path"]
    destination.symlink_to(victim)
    provider = _FakeStreamProvider({bundle["path"]: PAYLOAD})
    with pytest.raises(fetch.WindowFetchError, match="overwrite is forbidden"):
        _materialize(bundle, provider=provider)
    assert victim.read_bytes() == victim_bytes
    assert provider.opened == []

    destination.unlink()
    partial_name = fetch._partial_name(
        destination.name,
        bundle["intent"]["seal_sha256"],
    )
    partial = bundle["source_root"] / partial_name
    partial.symlink_to(victim)
    with pytest.raises(fetch.WindowFetchError, match="overwrite is forbidden"):
        _materialize(bundle, provider=provider)
    assert victim.read_bytes() == victim_bytes
    assert partial.is_symlink()


def test_signed_intent_target_substitution_and_refused_capability_write_nothing(
    bundle: Mapping[str, Any],
) -> None:
    tampered_body = {
        key: copy.deepcopy(value)
        for key, value in bundle["intent"].items()
        if key not in {"seal_sha256", "producer_hmac_sha256"}
    }
    tampered_body["targets"][0]["path"] = "unscheduled.safetensors"
    tampered = state.seal_producer_authenticated_evidence(
        tampered_body,
        auth=bundle["auth"],
    )
    provider = _FakeStreamProvider({bundle["path"]: PAYLOAD})
    with pytest.raises(fetch.WindowFetchError, match="scheduled authoritative view"):
        _materialize(bundle, provider=provider, intent=tampered)
    assert list(bundle["source_root"].iterdir()) == []

    verifier = _CapabilityVerifier(False)
    with pytest.raises(fetch.WindowFetchError, match="refused intent"):
        _materialize(bundle, provider=provider, verifier=verifier)
    assert verifier.calls == 1
    assert provider.opened == []
    assert list(bundle["source_root"].iterdir()) == []


def test_inline_disk_floor_refuses_before_first_body_write_and_retains_empty_partial(
    bundle: Mapping[str, Any],
) -> None:
    floor = state.OFFICIAL_RESOURCE_POLICY_REQUIRED_FREE_DISK_BYTES
    sampler = _FakeResourceSampler(inline_free=[floor, floor])
    provider = _FakeStreamProvider({bundle["path"]: PAYLOAD})
    with pytest.raises(fetch.WindowFetchError, match="disk-floor reservation") as captured:
        _materialize(bundle, provider=provider, sampler=sampler)
    assert not (bundle["source_root"] / bundle["path"]).exists()
    partials = list(bundle["source_root"].glob(".*glm52-partial-*"))
    assert len(partials) == 1
    assert partials[0].read_bytes() == b""
    assert captured.value.retained_partials


def test_injected_verifier_and_provider_cannot_mutate_authoritative_view(
    bundle: Mapping[str, Any],
) -> None:
    class MutatingVerifier:
        def verify_window_fetch_capability(
            self,
            intent: Mapping[str, Any],
            *,
            frozen_schedule: Mapping[str, Any],
            resource_policy: Mapping[str, Any],
        ) -> bool:
            assert isinstance(intent, dict)
            intent["maximum_streamed_body_bytes"] = 1
            assert isinstance(frozen_schedule, dict)
            frozen_schedule.clear()
            assert isinstance(resource_policy, dict)
            resource_policy.clear()
            return True

    class MutatingProvider(_FakeStreamProvider):
        def open_stream(self, target: Mapping[str, Any]) -> Iterable[bytes]:
            assert isinstance(target, dict)
            original = str(target["path"])
            target["path"] = "unscheduled.safetensors"
            target["logical_bytes"] = 1
            self.opened.append(original)
            return (PAYLOAD[offset:offset + self.chunk_bytes]
                    for offset in range(0, len(PAYLOAD), self.chunk_bytes))

    provider = MutatingProvider({bundle["path"]: PAYLOAD})
    receipt = _materialize(
        bundle,
        provider=provider,
        verifier=MutatingVerifier(),
    )
    assert receipt["streamed_body_bytes"] == len(PAYLOAD)
    assert receipt["targets"][0]["path"] == bundle["path"]
    assert (bundle["source_root"] / bundle["path"]).read_bytes() == PAYLOAD


def test_import_and_intent_build_are_body_and_stream_free(bundle: Mapping[str, Any]) -> None:
    # The fixture built and authenticated the intent before any StreamProvider existed.
    assert bundle["intent"]["worker_dispatch_enabled"] is False
    assert bundle["intent"]["maximum_streamed_body_bytes"] == len(PAYLOAD)
    assert list(bundle["source_root"].iterdir()) == []


@pytest.mark.parametrize(
    "violating_sample,required_reason",
    [
        (
            _live_sample(available_ram_bytes=16 * 1024**3 - 1),
            "AVAILABLE_RAM_FLOOR",
        ),
        (
            _live_sample(swap_used_bytes=8 * 1024**3 + 1),
            "ABSOLUTE_SWAP_CEILING",
        ),
        (_live_sample(swap_used_bytes=1), "SWAP_GROWTH"),
        (_live_sample(swapouts=1), "NEW_SWAPOUTS"),
    ],
    ids=("ram-floor", "absolute-swap", "swap-growth", "new-swapout"),
)
def test_continuous_memory_threshold_aborts_blocked_stream_and_never_publishes(
    bundle: Mapping[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    violating_sample: Mapping[str, Any],
    required_reason: str,
) -> None:
    monkeypatch.setattr(fetch, "LIVE_MEMORY_MONITOR_INTERVAL_SECONDS", 0.005)
    sampler = _SequencedMemorySampler([_live_sample(), violating_sample])
    provider = _AbortBlockingProvider({bundle["path"]: PAYLOAD})
    with pytest.raises(fetch.WindowFetchError, match=required_reason) as captured:
        _materialize(bundle, provider=provider, sampler=sampler)

    assert provider.started.is_set()
    assert provider.abort_calls >= 1
    assert provider.active == 0
    assert not (bundle["source_root"] / bundle["path"]).exists()
    partials = list(bundle["source_root"].glob(".*glm52-partial-*"))
    assert len(partials) == 1
    assert partials[0].read_bytes() == b""
    assert captured.value.published_paths == ()
    assert captured.value.retained_partials
    monitor = fetch._producer_verify(
        captured.value.resource_monitor_evidence,
        auth=bundle["auth"],
        label="test aborted continuous resource monitor",
    )
    assert monitor["status"] == "ABORTED_CONTINUOUS_RAM_SWAP_GUARD"
    assert required_reason in monitor["failure_reasons"]
    assert monitor["provider_abort_requested"] is True
    assert monitor["monitor_thread_quiesced"] is True
    assert monitor["stream_workers_quiesced"] is True
    assert monitor["no_publish_after_observed_resource_violation"] is True
    assert monitor["trigger_sample"] is not None
    assert monitor["periodic_sample_count"] >= 1


def test_monitor_sampling_failure_is_fail_closed_while_stream_is_active(
    bundle: Mapping[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fetch, "LIVE_MEMORY_MONITOR_INTERVAL_SECONDS", 0.005)

    class FailingSampler(_FakeResourceSampler):
        def live_memory_sample(self) -> Mapping[str, Any]:
            self.live_memory_calls += 1
            if self.live_memory_calls > 1:
                raise OSError("offline injected counter failure")
            return _live_sample()

    provider = _AbortBlockingProvider({bundle["path"]: PAYLOAD})
    with pytest.raises(
        fetch.WindowFetchError, match="RESOURCE_MONITOR_SAMPLE_FAILED"
    ) as captured:
        _materialize(bundle, provider=provider, sampler=FailingSampler())
    monitor = captured.value.resource_monitor_evidence
    assert monitor["failure_reasons"] == ["RESOURCE_MONITOR_SAMPLE_FAILED"]
    assert monitor["sample_error_sha256"] is not None
    assert monitor["provider_abort_requested"] is True
    assert provider.active == 0
    assert not (bundle["source_root"] / bundle["path"]).exists()


def test_midstream_ram_violation_retains_written_partial_but_never_publishes(
    bundle: Mapping[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fetch, "LIVE_MEMORY_MONITOR_INTERVAL_SECONDS", 0.001)
    first_chunk_consumed = threading.Event()

    class MidstreamSampler(_FakeResourceSampler):
        def live_memory_sample(self) -> Mapping[str, Any]:
            self.live_memory_calls += 1
            if self.live_memory_calls > 1:
                if not first_chunk_consumed.wait(2):
                    raise AssertionError("first stream chunk was not consumed")
                return _live_sample(available_ram_bytes=16 * 1024**3 - 1)
            return _live_sample()

    class PartialThenBlockProvider(_AbortBlockingProvider):
        def open_stream(self, target: Mapping[str, Any]) -> Iterable[bytes]:
            path = str(target["path"])
            self.opened.append(path)

            def chunks() -> Iterable[bytes]:
                with self._lock:
                    self.active += 1
                    self.maximum_active = max(self.maximum_active, self.active)
                self.started.set()
                try:
                    yield self.payloads[path][:11]
                    first_chunk_consumed.set()
                    if not self.aborted.wait(2):
                        raise AssertionError("monitor did not abort partial stream")
                finally:
                    with self._lock:
                        self.active -= 1

            return chunks()

    provider = PartialThenBlockProvider({bundle["path"]: PAYLOAD})
    with pytest.raises(fetch.WindowFetchError, match="AVAILABLE_RAM_FLOOR") as captured:
        _materialize(bundle, provider=provider, sampler=MidstreamSampler())
    assert provider.active == 0
    assert provider.abort_calls >= 1
    assert not (bundle["source_root"] / bundle["path"]).exists()
    partials = list(bundle["source_root"].glob(".*glm52-partial-*"))
    assert len(partials) == 1
    assert partials[0].read_bytes() == PAYLOAD[:11]
    assert captured.value.published_paths == ()
    assert captured.value.resource_monitor_evidence[
        "no_publish_after_observed_resource_violation"
    ] is True


def test_sample_timestamp_freshness_boundaries_are_exact() -> None:
    maximum_age_ns = int(fetch.LIVE_MEMORY_MONITOR_MAX_BLIND_SECONDS * 1e9)
    tolerance = fetch.LIVE_MEMORY_SAMPLE_FUTURE_TOLERANCE_NS
    attempt = 10_000_000_000
    observed = attempt + 10_000

    assert fetch._sample_timestamp_reasons(
        attempt,
        attempt_started_ns=attempt,
        observed_ns=observed,
    ) == []
    assert fetch._sample_timestamp_reasons(
        observed + tolerance,
        attempt_started_ns=attempt,
        observed_ns=observed,
    ) == []
    assert fetch._sample_timestamp_reasons(
        observed + tolerance + 1,
        attempt_started_ns=attempt,
        observed_ns=observed,
    ) == ["SAMPLE_TIMESTAMP_IN_FUTURE"]
    assert fetch._sample_timestamp_reasons(
        attempt - 1,
        attempt_started_ns=attempt,
        observed_ns=observed,
    ) == ["SAMPLE_TIMESTAMP_BEFORE_ATTEMPT"]

    old_sample = 20_000_000_000
    assert fetch._sample_timestamp_reasons(
        old_sample,
        attempt_started_ns=old_sample,
        observed_ns=old_sample + maximum_age_ns,
    ) == []
    assert fetch._sample_timestamp_reasons(
        old_sample,
        attempt_started_ns=old_sample + 1,
        observed_ns=old_sample + maximum_age_ns + 1,
    ) == ["SAMPLE_TIMESTAMP_BEFORE_ATTEMPT", "SAMPLE_TIMESTAMP_TOO_OLD"]


def test_monitor_rejects_clock_regression_and_exact_gap_latency_overruns() -> None:
    maximum_blind_ns = int(fetch.LIVE_MEMORY_MONITOR_MAX_BLIND_SECONDS * 1e9)
    base = 30_000_000_000
    monitor = _unit_memory_monitor()
    assert monitor._record_sample(
        {**_live_sample(), "sampled_monotonic_ns": base},
        phase="baseline",
        attempt_started_ns=base,
        observed_ns=base,
    ) == []
    reasons = monitor._record_sample(
        {**_live_sample(), "sampled_monotonic_ns": base},
        phase="periodic",
        attempt_started_ns=base,
        observed_ns=base,
    )
    assert reasons == ["SAMPLE_CLOCK_NOT_MONOTONIC"]

    exact = _unit_memory_monitor()
    assert exact._record_sample(
        {**_live_sample(), "sampled_monotonic_ns": base},
        phase="baseline",
        attempt_started_ns=base,
        observed_ns=base,
    ) == []
    assert exact._record_sample(
        {
            **_live_sample(),
            "sampled_monotonic_ns": base + maximum_blind_ns,
        },
        phase="periodic",
        attempt_started_ns=base + maximum_blind_ns,
        observed_ns=base + maximum_blind_ns,
    ) == []

    gap = _unit_memory_monitor()
    assert gap._record_sample(
        {**_live_sample(), "sampled_monotonic_ns": base},
        phase="baseline",
        attempt_started_ns=base,
        observed_ns=base,
    ) == []
    gap_reasons = gap._record_sample(
        {
            **_live_sample(),
            "sampled_monotonic_ns": base + maximum_blind_ns + 1,
        },
        phase="periodic",
        attempt_started_ns=base + maximum_blind_ns + 1,
        observed_ns=base + maximum_blind_ns + 1,
    )
    assert gap_reasons == ["MONITOR_OBSERVATION_GAP"]

    latency = _unit_memory_monitor()
    latency_reasons = latency._record_sample(
        {
            **_live_sample(),
            "sampled_monotonic_ns": base + maximum_blind_ns + 1,
        },
        phase="baseline",
        attempt_started_ns=base,
        observed_ns=base + maximum_blind_ns + 1,
    )
    assert latency_reasons == ["MONITOR_SAMPLE_LATENCY"]


def test_stale_baseline_timestamp_fails_before_any_stream_and_never_publishes(
    bundle: Mapping[str, Any]
) -> None:
    class StaleBaselineSampler(_FakeResourceSampler):
        def live_memory_sample(self) -> Mapping[str, Any]:
            self.live_memory_calls += 1
            return {**_live_sample(), "sampled_monotonic_ns": 1}

    provider = _AbortBlockingProvider({bundle["path"]: PAYLOAD})
    with pytest.raises(
        fetch.WindowFetchError, match="SAMPLE_TIMESTAMP_BEFORE_ATTEMPT"
    ) as captured:
        _materialize(bundle, provider=provider, sampler=StaleBaselineSampler())
    assert provider.started.is_set() is False
    assert provider.active == 0
    assert not (bundle["source_root"] / bundle["path"]).exists()
    monitor = captured.value.resource_monitor_evidence
    assert "SAMPLE_TIMESTAMP_BEFORE_ATTEMPT" in monitor["failure_reasons"]
    assert "SAMPLE_TIMESTAMP_TOO_OLD" in monitor["failure_reasons"]
    assert monitor["stream_workers_quiesced"] is True
    assert monitor["monitor_thread_quiesced"] is True
    assert monitor["final_sample_after_stream_quiescence"] is False
    assert monitor["no_publish_after_observed_resource_violation"] is True


@pytest.mark.parametrize(
    "timestamp_mode,required_reason",
    [
        ("stale", "SAMPLE_TIMESTAMP_BEFORE_ATTEMPT"),
        ("future", "SAMPLE_TIMESTAMP_IN_FUTURE"),
        ("regression", "SAMPLE_CLOCK_NOT_MONOTONIC"),
    ],
)
def test_runtime_timestamp_attack_aborts_active_stream_and_never_publishes(
    bundle: Mapping[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    timestamp_mode: str,
    required_reason: str,
) -> None:
    monkeypatch.setattr(fetch, "LIVE_MEMORY_MONITOR_INTERVAL_SECONDS", 0.001)
    provider = _AbortBlockingProvider({bundle["path"]: PAYLOAD})

    class TimestampAttackSampler(_FakeResourceSampler):
        baseline_timestamp = 0

        def live_memory_sample(self) -> Mapping[str, Any]:
            self.live_memory_calls += 1
            if self.live_memory_calls == 1:
                self.baseline_timestamp = time.monotonic_ns()
                return {
                    **_live_sample(),
                    "sampled_monotonic_ns": self.baseline_timestamp,
                }
            if not provider.started.wait(2):
                raise AssertionError("stream did not become active")
            if timestamp_mode == "future":
                timestamp = (
                    time.monotonic_ns()
                    + fetch.LIVE_MEMORY_SAMPLE_FUTURE_TOLERANCE_NS
                    + 10_000_000
                )
            elif timestamp_mode == "regression":
                timestamp = self.baseline_timestamp
            else:
                timestamp = self.baseline_timestamp + 1
            return {**_live_sample(), "sampled_monotonic_ns": timestamp}

    with pytest.raises(fetch.WindowFetchError, match=required_reason) as captured:
        _materialize(
            bundle,
            provider=provider,
            sampler=TimestampAttackSampler(),
        )
    assert provider.started.is_set()
    assert provider.abort_calls >= 1
    assert provider.active == 0
    assert not (bundle["source_root"] / bundle["path"]).exists()
    monitor = captured.value.resource_monitor_evidence
    assert required_reason in monitor["failure_reasons"]
    assert monitor["provider_abort_requested"] is True
    assert monitor["stream_workers_quiesced"] is True
    assert monitor["monitor_thread_quiesced"] is True
    assert monitor["no_publish_after_observed_resource_violation"] is True


def test_final_sampler_failure_never_publishes_or_claims_a_valid_final_sample(
    bundle: Mapping[str, Any]
) -> None:
    class FinalFailureSampler(_FakeResourceSampler):
        def live_memory_sample(self) -> Mapping[str, Any]:
            self.live_memory_calls += 1
            if self.live_memory_calls == 2:
                raise OSError("offline injected final-sample failure")
            return _live_sample()

    provider = _FakeStreamProvider({bundle["path"]: PAYLOAD})
    with pytest.raises(
        fetch.WindowFetchError, match="RESOURCE_MONITOR_SAMPLE_FAILED"
    ) as captured:
        _materialize(bundle, provider=provider, sampler=FinalFailureSampler())
    assert provider.active == 0
    assert not (bundle["source_root"] / bundle["path"]).exists()
    partials = list(bundle["source_root"].glob(".*glm52-partial-*"))
    assert len(partials) == 1
    assert partials[0].read_bytes() == PAYLOAD
    monitor = captured.value.resource_monitor_evidence
    assert monitor["status"] == "ABORTED_CONTINUOUS_RAM_SWAP_GUARD"
    assert monitor["sample_attempt_count"] == 2
    assert monitor["valid_sample_count"] == 1
    assert monitor["failure_reasons"] == ["RESOURCE_MONITOR_SAMPLE_FAILED"]
    assert monitor["final_sample_after_stream_quiescence"] is False
    assert monitor["stream_workers_quiesced"] is True
    assert monitor["monitor_thread_quiesced"] is True
    assert monitor["no_publish_after_observed_resource_violation"] is True


def test_poststream_prepublish_swapout_is_observed_before_any_visible_artifact(
    bundle: Mapping[str, Any]
) -> None:
    class PrepublishViolationSampler(_FakeResourceSampler):
        def live_memory_sample(self) -> Mapping[str, Any]:
            self.live_memory_calls += 1
            if self.live_memory_calls == 2:
                return _live_sample(swapouts=1)
            return _live_sample()

    provider = _FakeStreamProvider({bundle["path"]: PAYLOAD})
    with pytest.raises(fetch.WindowFetchError, match="NEW_SWAPOUTS") as captured:
        _materialize(
            bundle,
            provider=provider,
            sampler=PrepublishViolationSampler(),
        )
    assert provider.active == 0
    assert not (bundle["source_root"] / bundle["path"]).exists()
    partials = list(bundle["source_root"].glob(".*glm52-partial-*"))
    assert len(partials) == 1
    assert partials[0].read_bytes() == PAYLOAD
    monitor = captured.value.resource_monitor_evidence
    assert monitor["failure_reasons"] == ["NEW_SWAPOUTS"]
    assert monitor["stream_workers_quiesced"] is True
    assert monitor["final_sample_after_stream_quiescence"] is False
    assert monitor["final_sample_after_publish_boundary"] is False
    assert monitor["published_paths_at_monitor_evidence_seal"] == []
    assert monitor["no_publish_after_observed_resource_violation"] is True


def test_visibility_gate_rechecks_swap_immediately_before_atomic_rename(
    bundle: Mapping[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fetch, "LIVE_MEMORY_MONITOR_INTERVAL_SECONDS", 10.0)

    class VisibilityGateViolationSampler(_FakeResourceSampler):
        def live_memory_sample(self) -> Mapping[str, Any]:
            self.live_memory_calls += 1
            if self.live_memory_calls == 4:
                return _live_sample(swap_used_bytes=1)
            return _live_sample()

    provider = _FakeStreamProvider({bundle["path"]: PAYLOAD})
    with pytest.raises(fetch.WindowFetchError, match="SWAP_GROWTH") as captured:
        _materialize(
            bundle,
            provider=provider,
            sampler=VisibilityGateViolationSampler(),
        )
    assert provider.active == 0
    assert not (bundle["source_root"] / bundle["path"]).exists()
    partials = list(bundle["source_root"].glob(".*glm52-partial-*"))
    assert len(partials) == 1
    assert partials[0].read_bytes() == PAYLOAD
    monitor = captured.value.resource_monitor_evidence
    assert monitor["failure_reasons"] == ["SWAP_GROWTH"]
    assert monitor["final_sample_after_stream_quiescence"] is True
    assert monitor["final_sample_after_publish_boundary"] is False
    assert monitor["published_paths_at_monitor_evidence_seal"] == []
    assert monitor["no_publish_after_observed_resource_violation"] is True


def test_slow_cooperative_abort_records_quiescence_deadline_before_safe_return(
    bundle: Mapping[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fetch, "LIVE_MEMORY_MONITOR_INTERVAL_SECONDS", 0.001)
    monkeypatch.setattr(fetch, "STREAM_QUIESCENCE_DEADLINE_SECONDS", 0.01)
    provider_started = threading.Event()

    class ViolationSampler(_FakeResourceSampler):
        def live_memory_sample(self) -> Mapping[str, Any]:
            self.live_memory_calls += 1
            if self.live_memory_calls > 1:
                if not provider_started.wait(2):
                    raise AssertionError("provider did not start")
                return _live_sample(available_ram_bytes=16 * 1024**3 - 1)
            return _live_sample()

    class SlowAbortProvider(_FakeStreamProvider):
        def open_stream(self, target: Mapping[str, Any]) -> Iterable[bytes]:
            path = str(target["path"])
            self.opened.append(path)

            def chunks() -> Iterable[bytes]:
                with self._lock:
                    self.active += 1
                    self.maximum_active = max(self.maximum_active, self.active)
                provider_started.set()
                try:
                    # Deliberately ignore cooperative abort long enough for the
                    # bounded quiescence detector, then quiesce so the adapter
                    # can return without closing a descriptor under this writer.
                    time.sleep(0.05)
                    if False:  # pragma: no cover - zero-byte stream
                        yield b""
                finally:
                    with self._lock:
                        self.active -= 1

            return chunks()

    provider = SlowAbortProvider({bundle["path"]: PAYLOAD})
    with pytest.raises(
        fetch.WindowFetchError, match="STREAM_WORKER_QUIESCENCE_DEADLINE"
    ) as captured:
        _materialize(bundle, provider=provider, sampler=ViolationSampler())
    assert provider.abort_calls >= 1
    assert provider.active == 0
    assert not (bundle["source_root"] / bundle["path"]).exists()
    monitor = captured.value.resource_monitor_evidence
    assert monitor["stream_quiescence_deadline_crossed"] is True
    assert "STREAM_WORKER_QUIESCENCE_DEADLINE" in monitor["failure_reasons"]
    assert monitor["stream_workers_quiesced"] is True
    assert monitor["stream_termination_model"][
        "inprocess_native_thread_hard_termination_available"
    ] is False
    assert monitor["stream_termination_model"][
        "permanent_hang_requires_subprocess_isolation"
    ] is True


def test_inflight_periodic_sample_cannot_race_clean_stream_exit(
    bundle: Mapping[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fetch, "LIVE_MEMORY_MONITOR_INTERVAL_SECONDS", 0.001)
    periodic_entered = threading.Event()
    release_periodic = threading.Event()
    provider_completed = threading.Event()

    class RacingSampler(_FakeResourceSampler):
        def live_memory_sample(self) -> Mapping[str, Any]:
            self.live_memory_calls += 1
            if self.live_memory_calls == 2:
                periodic_entered.set()
                if not release_periodic.wait(2):
                    raise AssertionError("test did not release in-flight periodic sample")
                return _live_sample()
            return _live_sample()

    class CoordinatedCleanProvider(_FakeStreamProvider):
        def open_stream(self, target: Mapping[str, Any]) -> Iterable[bytes]:
            path = str(target["path"])
            self.opened.append(path)

            def chunks() -> Iterable[bytes]:
                with self._lock:
                    self.active += 1
                    self.maximum_active = max(self.maximum_active, self.active)
                try:
                    if not periodic_entered.wait(2):
                        raise AssertionError("periodic sampler did not enter")
                    yield self.payloads[path]
                finally:
                    with self._lock:
                        self.active -= 1
                    provider_completed.set()

            return chunks()

    sampler = RacingSampler()
    provider = CoordinatedCleanProvider({bundle["path"]: PAYLOAD})
    result: dict[str, Any] = {}

    def run() -> None:
        try:
            result["receipt"] = _materialize(
                bundle, provider=provider, sampler=sampler
            )
        except BaseException as exc:  # surfaced in the asserting test thread
            result["error"] = exc

    thread = threading.Thread(target=run, name="glm52-clean-exit-race-test")
    thread.start()
    assert periodic_entered.wait(2)
    assert provider_completed.wait(2)
    deadline = time.monotonic() + 2
    while provider.active and time.monotonic() < deadline:
        time.sleep(0.001)
    assert provider.active == 0
    # The materializer may be waiting for the same serialized sampler at its
    # prepublish gate.  Releasing a green in-flight sample must not race the
    # clean stream exit or duplicate an abort.
    time.sleep(0.02)
    release_periodic.set()
    thread.join(2)
    assert not thread.is_alive()
    assert "error" not in result
    assert provider.abort_calls == 0
    monitor = result["receipt"]["continuous_resource_monitor"]
    assert monitor["status"] == "PASS_CONTINUOUS_RAM_SWAP_GUARD"
    assert monitor["periodic_sample_count"] == 1
    assert monitor["sample_attempt_count"] >= 5
    assert monitor["final_sample_after_stream_quiescence"] is True
    assert monitor["final_sample_after_publish_boundary"] is True
    assert (bundle["source_root"] / bundle["path"]).read_bytes() == PAYLOAD
