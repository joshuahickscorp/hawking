#!/usr/bin/env python3.12
"""Zero-network adversarial tests for selected-profile Xet window materialization."""
from __future__ import annotations

import copy
import hashlib
import os
import pathlib
import sys
import threading
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
