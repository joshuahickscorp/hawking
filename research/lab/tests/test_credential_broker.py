"""Offline contract tests for the reusable credential-broker scaffold.

No network, no Hub, no live Qwen acquisition. Exercises types, floor proofs,
preflight gates, secret isolation, and lifecycle transitions that generalise
tonight's DeepSeek / GLM / Kimi discipline.
"""
from __future__ import annotations

import pytest

from lab.operators.credential_broker import (
    AcquisitionPreflight,
    ArchitectureClassification,
    CredentialBroker,
    CredentialBrokerError,
    FloorViolation,
    ImmutableRevision,
    LifecycleError,
    LifecyclePhase,
    OfficialSource,
    ScientificPurpose,
    SourceLifecycle,
    StorageForecast,
    assert_disk_floor,
    validate_preflight,
)
from lab.operators.credential_broker.preflight import prove_floor_for_preflight
from lab.operators.credential_broker.types import (
    FileEntry,
    GravityPlanSummary,
    HashInventory,
    RangeRequest,
    RuntimeMemoryForecast,
    TypeError_,
)


REVISION = "a" * 40
SHA = "b" * 64


def _revision() -> ImmutableRevision:
    return ImmutableRevision(commit=REVISION, requested="main")


def _inventory() -> HashInventory:
    return HashInventory(
        files=(
            FileEntry(path="config.json", bytes=100, sha256=SHA, kind="control"),
            FileEntry(
                path="model-00001-of-00002.safetensors",
                bytes=1024,
                lfs_sha256=SHA,
                kind="weight",
            ),
        ),
        revision=_revision(),
    )


def _storage() -> StorageForecast:
    return StorageForecast(
        peak_source_bytes=1024,
        peak_intermediate_bytes=512,
        sealed_artifact_bytes=256,
        protected_floor_bytes=15 * 1024**3,
        max_inflight_bytes=1024,
    )


def test_immutable_revision_requires_40_char_commit() -> None:
    with pytest.raises(TypeError_):
        ImmutableRevision(commit="main")


def test_hash_inventory_rejects_duplicates() -> None:
    with pytest.raises(TypeError_):
        HashInventory(
            files=(
                FileEntry(path="a.safetensors", bytes=1, kind="weight"),
                FileEntry(path="a.safetensors", bytes=2, kind="weight"),
            ),
            revision=_revision(),
        )


def test_range_request_expected_bytes() -> None:
    r = RangeRequest(range_id="r0", path="x.safetensors", start=0, end=8, kind="header")
    assert r.expected_bytes == 8


def test_disk_floor_pass_and_fail() -> None:
    floor = 15 * 1024**3
    proof = assert_disk_floor(
        "/tmp",
        protected_floor_bytes=floor,
        additional_bytes=100,
        stage="preflight",
        free_bytes_provider=lambda _p: floor + 200,
    )
    assert proof.status == "PASS"
    assert proof.remaining_bytes == floor + 100

    with pytest.raises(FloorViolation):
        assert_disk_floor(
            "/tmp",
            protected_floor_bytes=floor,
            additional_bytes=100,
            stage="stream",
            free_bytes_provider=lambda _p: floor + 50,
        )


def test_disk_floor_cannot_be_lowered_below_minimum() -> None:
    with pytest.raises(FloorViolation):
        assert_disk_floor(
            "/tmp",
            protected_floor_bytes=1,
            additional_bytes=0,
            stage="bad",
            free_bytes_provider=lambda _p: 10**18,
        )


def test_public_environment_refuses_ambient_token(monkeypatch: pytest.MonkeyPatch) -> None:
    broker = CredentialBroker()
    monkeypatch.setenv("HF_TOKEN", "hf_should_not_leak")
    with pytest.raises(CredentialBrokerError):
        broker.apply_public_environment()


def test_public_session_never_exports_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    broker = CredentialBroker()
    handle = broker.mint_public_session(repository="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    assert handle.allows_gated is False
    assert handle.as_dict()["token_material_present_on_handle"] is False
    assert broker.authorization_header(handle) is None


def test_gated_session_keeps_token_inside_broker_only() -> None:
    broker = CredentialBroker()
    handle = broker.mint_gated_session(
        token="hf_secret_value",
        repository="some/gated-model",
    )
    assert "hf_secret_value" not in str(handle.as_dict())
    headers = broker.authorization_header(handle)
    assert headers is not None and headers["Authorization"].startswith("Bearer ")
    broker.revoke(handle)
    with pytest.raises(CredentialBrokerError):
        broker.authorization_header(handle)


def test_receipt_payload_cannot_embed_token() -> None:
    broker = CredentialBroker()
    with pytest.raises(CredentialBrokerError):
        broker.assert_no_token_in_mapping({"hf_token": "hf_abc"}, label="receipt")
    with pytest.raises(CredentialBrokerError):
        broker.assert_no_token_in_mapping({"note": "hf_abc123leak"}, label="receipt")


def test_preflight_requires_retention_paths_and_matching_revision() -> None:
    floor = prove_floor_for_preflight(
        "/tmp",
        _storage(),
        free_bytes_provider=lambda _p: 100 * 1024**3,
    )
    source = OfficialSource(
        repository="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        revision=_revision(),
        license_id="apache-2.0",
    )
    arch = ArchitectureClassification(
        model_type="qwen3_moe",
        architectures=("Qwen3MoeForCausalLM",),
        hidden_size=2048,
        num_hidden_layers=48,
    )
    with pytest.raises(Exception):
        AcquisitionPreflight(
            source=source,
            inventory=_inventory(),
            architecture=arch,
            storage=_storage(),
            runtime_memory=RuntimeMemoryForecast(
                peak_resident_bytes=8 * 1024**3, working_set_bytes=4 * 1024**3
            ),
            gravity_plan=GravityPlanSummary(
                plan_id="qwen30b-g0",
                transform_family="stream_pack",
                target_artifact_schema="hawking.gravity.artifact.v1",
            ),
            purpose=ScientificPurpose(
                purpose_id="bootstrap-30b",
                statement="executor model for HCLI self-optimisation",
                programme="bootstrap_qwen_30b",
                success_metric="parity ladder P0-P13",
            ),
            floor_proof=floor,
            source_retention_paths=(),  # refused
        )


def test_preflight_happy_path() -> None:
    storage = _storage()
    floor = prove_floor_for_preflight(
        "/tmp",
        storage,
        free_bytes_provider=lambda _p: 100 * 1024**3,
    )
    doc = AcquisitionPreflight(
        source=OfficialSource(
            repository="Qwen/Qwen3-Coder-30B-A3B-Instruct",
            revision=_revision(),
            license_id="apache-2.0",
            license_file_sha256=SHA,
        ),
        inventory=_inventory(),
        architecture=ArchitectureClassification(
            model_type="qwen3_moe",
            architectures=("Qwen3MoeForCausalLM",),
            hidden_size=2048,
        ),
        storage=storage,
        runtime_memory=RuntimeMemoryForecast(
            peak_resident_bytes=8 * 1024**3, working_set_bytes=4 * 1024**3
        ),
        gravity_plan=GravityPlanSummary(
            plan_id="qwen30b-g0",
            transform_family="stream_pack",
            target_artifact_schema="hawking.gravity.artifact.v1",
        ),
        purpose=ScientificPurpose(
            purpose_id="bootstrap-30b",
            statement="executor model for HCLI self-optimisation",
            programme="bootstrap_qwen_30b",
            success_metric="parity ladder P0-P13",
        ),
        floor_proof=floor,
        source_retention_paths=("/tmp/hawking-broker-cache",),
    )
    sealed = validate_preflight(doc)
    assert sealed["schema"] == "hawking.ascension.acquisition_preflight.v1"
    assert sealed["claims"]["models_never_receive_hf_token"] is True
    assert sealed["lifecycle"][-1] == "evict_source"


def test_lifecycle_stream_verify_transform_seal_evict() -> None:
    life = SourceLifecycle(
        task_id="window-0",
        repository="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        revision_commit=REVISION,
    )
    life.seal_preflight(SHA)
    life.begin_stream(expected_bytes=1024)
    assert life.source_resident is True
    life.complete_verify(observed_bytes=1024, content_sha256=SHA)
    life.begin_transform()
    life.complete_transform(artifact_ids=["artifact-0.gravity"])
    life.seal(SHA)
    life.evict_source(bytes_reclaimed=1024)
    assert life.phase is LifecyclePhase.EVICTED
    assert life.source_resident is False
    snap = life.snapshot()
    assert snap["lifecycle_law"] == [
        "stream",
        "verify",
        "gravity_transform",
        "seal",
        "evict_source",
    ]


def test_lifecycle_refuses_evict_before_seal() -> None:
    life = SourceLifecycle(
        task_id="window-1",
        repository="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        revision_commit=REVISION,
    )
    life.seal_preflight(SHA)
    life.begin_stream(expected_bytes=10)
    life.complete_verify(observed_bytes=10, content_sha256=SHA)
    with pytest.raises(LifecycleError):
        life.evict_source(bytes_reclaimed=10)


def test_lifecycle_refuses_transform_before_verify() -> None:
    life = SourceLifecycle(
        task_id="window-2",
        repository="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        revision_commit=REVISION,
    )
    life.seal_preflight(SHA)
    life.begin_stream(expected_bytes=10)
    with pytest.raises(LifecycleError):
        life.begin_transform()


def test_storage_forecast_forbids_full_source_accumulation() -> None:
    with pytest.raises(TypeError_):
        StorageForecast(
            peak_source_bytes=1,
            peak_intermediate_bytes=1,
            sealed_artifact_bytes=1,
            protected_floor_bytes=15 * 1024**3,
            max_inflight_bytes=1,
            no_full_source_accumulation=False,
        )
