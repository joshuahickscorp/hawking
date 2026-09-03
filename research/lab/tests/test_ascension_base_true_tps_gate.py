"""Focused fail-closed tests for the real-HCLI BASE_TRUE_TPS sidecar."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from lab.operators import ascension_physical_gatekeeper as gatekeeper
from lab.operators.ascension_base_true_tps_gate import (
    BaseTrueTpsGate,
    BenchSample,
    PrerequisiteEvidence,
    PROMPT_PROBES,
    TG10_TPS,
    TG3_TPS,
    _accepts_probe,
)
from lab.receipts import seal, verify


def _evidence(tmp_path: Path, *, custom_kernel: bool = True) -> PrerequisiteEvidence:
    spec = gatekeeper.MODEL_SPECS[0]
    source = gatekeeper.SourceBinding(
        content_identity_sha256="a" * 64,
        identity_seal_sha256="b" * 64,
        revalidation_seal_sha256="c" * 64,
        source_dir=tmp_path,
        weight_shard_count=spec.shard_count,
        control_file_count=1,
    )
    artifact = gatekeeper.Check(
        requirement="artifact",
        passed=True,
        path=tmp_path / "artifact.json",
        seal_sha256="d" * 64,
        reasons=[],
        details={"physical_bpw": 1.2},
        document={},
        document_sha256="e" * 64,
    )
    runtime = gatekeeper.Check(
        requirement="runtime",
        passed=True,
        path=tmp_path / "runtime.json",
        seal_sha256="f" * 64,
        reasons=[],
        details={},
        document={
            "binding": {
                "complete_manifest_seal_sha256": "1" * 64,
                "complete_artifact_admission_seal_sha256": "d" * 64,
            }
        },
        document_sha256="2" * 64,
    )
    return PrerequisiteEvidence(
        spec=spec,
        source=source,
        artifact=artifact,
        runtime=runtime,
        endpoint_url="http://127.0.0.1:18430",
        endpoint_context={
            "kernel_id": "qwen30-packed-simdgroup-full-token",
            "custom_kernel_used": custom_kernel,
        },
        endpoint_health={"ready": True, "provider": "qwen30-native-metal"},
        endpoint_status={"server_binary_sha256": "3" * 64, "pid": 42},
        hcli_binary_sha256="4" * 64,
        quiet_conditions={"passed": True},
    )


def _samples(tps: float, *, count: int = 24) -> list[BenchSample]:
    # Two full generated-token forwards per HCLI request gives 48 total, the
    # minimum sustained sample population for this strict sidecar.
    result: list[BenchSample] = []
    for ordinal in range(count):
        result.append(
            BenchSample(
                prompt_id=PROMPT_PROBES[ordinal % len(PROMPT_PROBES)].identifier,
                prompt_sha256=PROMPT_PROBES[ordinal % len(PROMPT_PROBES)].prompt_sha256,
                request_ordinal=ordinal + 1,
                decode_ms=2_000.0 / tps,
                completed_decode_forwards=2,
                output_tokens=2,
                base_true_tps=tps,
                hcli_bench_receipt_path=f"/receipts/{ordinal}.json",
                hcli_bench_receipt_seal_sha256=f"{ordinal:064x}",
            )
        )
    return result


def _wire_runner(runner: BaseTrueTpsGate, *, tps: float) -> None:
    probes = [
        {
            "probe_id": probe.identifier,
            "prompt_sha256": probe.prompt_sha256,
            "accepted": True,
            "runtime_decode_ms": 10.0,
            "runtime_completed_decode_forwards": 1,
        }
        for probe in PROMPT_PROBES
    ]
    runner._probe_generation = lambda _evidence: (probes, [])  # type: ignore[method-assign]
    runner._run_benchmarks = lambda _evidence: (  # type: ignore[method-assign]
        _samples(tps),
        [
            {
                "prompt_id": probe.identifier,
                "prompt_sha256": probe.prompt_sha256,
                "receipt_path": f"/receipts/{probe.identifier}.json",
                "receipt_seal_sha256": "9" * 64,
            }
            for probe in PROMPT_PROBES
        ],
        [],
    )


def test_probe_acceptance_needs_prompt_specific_coherent_output_and_rejects_collapse() -> None:
    assert _accepts_probe(PROMPT_PROBES[0], "HAWKING")
    assert _accepts_probe(PROMPT_PROBES[1], '{"status":"ok"}')
    assert _accepts_probe(PROMPT_PROBES[2], "def add(a, b):\n    return a + b")
    assert not _accepts_probe(PROMPT_PROBES[0], "aaaaaaaaaaaaaaaa")
    assert not _accepts_probe(PROMPT_PROBES[1], "not json")
    assert not _accepts_probe(PROMPT_PROBES[2], "def add(a, b):\n    return a - b")


def test_tg10_is_not_written_below_the_true_threshold(tmp_path: Path) -> None:
    runner = BaseTrueTpsGate(physical_root=tmp_path / "physical", hcli_binary=tmp_path / "hcli")
    _wire_runner(runner, tps=TG10_TPS - 0.01)

    status = runner._advance_measurement(_evidence(tmp_path))

    paths = runner.paths(gatekeeper.MODEL_SPECS[0])
    assert status["phase"] == "HCLI_PASS_BASE_TRUE_TPS_BELOW_TG10"
    assert paths["hcli"].is_file()
    assert not paths["kernel"].exists()
    assert not paths["tg10"].exists()
    assert not paths["tg3"].exists()


def test_tg10_seals_only_after_complete_hcli_sustained_measurement(tmp_path: Path) -> None:
    runner = BaseTrueTpsGate(physical_root=tmp_path / "physical", hcli_binary=tmp_path / "hcli")
    _wire_runner(runner, tps=TG10_TPS + 1.0)

    status = runner._advance_measurement(_evidence(tmp_path))

    paths = runner.paths(gatekeeper.MODEL_SPECS[0])
    assert status["phase"] == "TG10_OPERATIONAL_EARNED_CONTINUING_TO_TG3"
    assert "TG10" in status["earned_rungs"]
    assert paths["kernel"].is_file()
    tg10 = verify(__import__("json").loads(paths["tg10"].read_text()), label="TG10")
    assert tg10["median_base_true_tps"] >= TG10_TPS
    assert tg10["complete_native_model"] is True
    assert tg10["real_metal"] is True
    assert tg10["autoregressive_generation"] is True
    assert tg10["hcli_pass"] is True
    assert tg10["fallback_count"] == 0
    assert not paths["tg3"].exists()


def test_tg3_never_appears_before_333_and_is_sealed_when_earned(tmp_path: Path) -> None:
    runner = BaseTrueTpsGate(physical_root=tmp_path / "physical", hcli_binary=tmp_path / "hcli")
    _wire_runner(runner, tps=TG3_TPS + 2.0)

    status = runner._advance_measurement(_evidence(tmp_path))

    paths = runner.paths(gatekeeper.MODEL_SPECS[0])
    assert status["phase"] == "TG3_QUALIFIED"
    assert "TG3" in status["earned_rungs"]
    tg3 = verify(__import__("json").loads(paths["tg3"].read_text()), label="TG3")
    assert tg3["schema"] == "hawking.ascension.physical_tg3_qualification.v1"
    assert tg3["status"] == "PASS_TG3_FULL_MODEL_QUALIFICATION"
    assert tg3["measurement"]["base_true_tokens_per_second"] >= TG3_TPS


def test_hcli_measurement_does_not_promote_without_actual_custom_kernel_provenance(tmp_path: Path) -> None:
    runner = BaseTrueTpsGate(physical_root=tmp_path / "physical", hcli_binary=tmp_path / "hcli")
    _wire_runner(runner, tps=TG3_TPS + 5.0)

    status = runner._advance_measurement(_evidence(tmp_path, custom_kernel=False))

    paths = runner.paths(gatekeeper.MODEL_SPECS[0])
    assert status["phase"] == "HCLI_PASS_TPS_MEASURED_CUSTOM_KERNEL_PROVENANCE_BLOCKED"
    assert paths["hcli"].is_file()
    assert not paths["kernel"].exists()
    assert not paths["tg10"].exists()
    assert not paths["tg3"].exists()


def test_incomplete_measurement_cannot_be_hcli_eligible(tmp_path: Path) -> None:
    runner = BaseTrueTpsGate(physical_root=tmp_path / "physical", hcli_binary=tmp_path / "hcli")
    evidence = _evidence(tmp_path)
    measurement = runner._measurement(evidence, _samples(500.0, count=1), [])

    reasons = runner._measurement_valid_for_hcli(measurement)

    assert any("too few HCLI requests" in reason for reason in reasons)
    assert any("too few complete token forwards" in reason for reason in reasons)


def test_qwen80_current_pointer_is_reported_as_current_admission_not_historical_stale(
    tmp_path: Path, monkeypatch
) -> None:
    """The status layer follows the versioned pointer without weakening gates."""

    runner = BaseTrueTpsGate(physical_root=tmp_path / "physical", hcli_binary=tmp_path / "hcli")
    spec = gatekeeper.MODEL_SPECS[1]
    selected = gatekeeper.LoadedReceipt(
        path=tmp_path / "physical" / "qwen80" / "complete-gravity" / "complete-admission" / "receipts" / "current.json",
        present=True,
        sealed=True,
        document={
            "schema": gatekeeper.ARTIFACT_ADMISSION_SCHEMA,
            "status": gatekeeper.ARTIFACT_ADMISSION_STATUS,
            "seal_sha256": "9" * 64,
        },
        seal_sha256="9" * 64,
        document_sha256="8" * 64,
        errors=[],
    )

    def fake_select(_spec, _paths):
        return selected, [], {
            "admission_selection": "CURRENT_POINTER",
            "current_pointer_path": "/physical/qwen80/complete-gravity/QWEN80_COMPLETE_BINARY_GRAVITY_ADMISSION_CURRENT.json",
            "current_pointer_seal_sha256": "7" * 64,
            "selected_admission_receipt_path": str(selected.path),
            "selected_manifest_seal_sha256": "6" * 64,
        }

    monkeypatch.setattr(gatekeeper, "_select_current_artifact_admission", fake_select)

    summary = runner._current_admission_summary(spec)

    assert summary is not None
    assert summary["admission_selection"] == "CURRENT_POINTER"
    assert summary["selected_admission_receipt_seal_sha256"] == "9" * 64
    assert summary["historical_fixed_receipt_is_not_used"] is True


def test_inflight_measurement_refuses_a_runtime_revoked_after_its_start(tmp_path: Path) -> None:
    runner = BaseTrueTpsGate(physical_root=tmp_path / "physical", hcli_binary=tmp_path / "hcli")
    spec = gatekeeper.MODEL_SPECS[0]
    paths = gatekeeper._paths(runner.root, spec)
    runtime = seal(
        {
            "schema": gatekeeper.RUNTIME_SCHEMA,
            "status": "PASS_EXACT_NATIVE_FULL_TOKEN_RUNTIME",
            "binding": {
                "model_id": spec.model_id,
                "runtime_executable_sha256": "a" * 64,
            },
            "runtime": {},
        }
    )
    paths["runtime"].parent.mkdir(parents=True, exist_ok=True)
    paths["runtime"].write_text(json.dumps(runtime, sort_keys=True), encoding="utf-8")
    raw = paths["runtime"].read_bytes()
    archive = (
        paths["runtime"].parent
        / "runtime-receipt-history"
        / f"{spec.prefix}_EXACT_FULL_TOKEN_RUNTIME_RECEIPT_{runtime['seal_sha256']}.json"
    )
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(raw)
    archive_sha = hashlib.sha256(raw).hexdigest()
    supersession = seal(
        {
            "schema": gatekeeper.RUNTIME_SUPERSESSION_SCHEMA,
            "status": "REVOKED_TEST_RUNTIME_DEFECT",
            "recorded_at": "2026-08-08T00:00:00Z",
            "binding": {
                "model_id": spec.model_id,
                "canonical_runtime_receipt_path": str(paths["runtime"]),
                "superseded_runtime_receipt_seal_sha256": runtime["seal_sha256"],
                "defective_runtime_executable_sha256": "a" * 64,
                "archived_runtime_receipt_path": str(archive),
                "archived_runtime_receipt_document_sha256": archive_sha,
            },
            "revoked_runtime": {
                "canonical_receipt_path": str(paths["runtime"]),
                "canonical_receipt_seal_sha256": runtime["seal_sha256"],
                "complete_manifest_seal_sha256": "b" * 64,
                "model_id": spec.model_id,
                "runtime_executable_sha256": "a" * 64,
            },
            "historical_pass_archive_path": str(archive),
            "historical_pass_archive_sha256": archive_sha,
            "defect": {"class": "TEST"},
            "invalidates": {
                "canonical_native_runtime_pass": True,
                "all_old_full_token_prompt_and_profile_controls_bound_to_runtime_sha": True,
                "native_http_adapter_and_transport_handoff_bound_to_runtime_sha": True,
                "any_hcli_tps_tg_capability_or_tournament_consumer_of_that_sha": True,
            },
            "required_before_reissue": ["new executable"],
            "consumer_contract": {
                "fail_closed_if_canonical_status_is_not_pass": True,
                "fail_closed_if_this_supersession_revokes_the_bound_receipt_seal_or_runtime_executable_sha256": True,
                "historical_archive_is_for_negative_science_only_not_a_gate_authority": True,
            },
            "claim_boundary": {"revocation": True},
        }
    )
    paths["runtime_supersession"].write_text(
        json.dumps(supersession, sort_keys=True), encoding="utf-8"
    )
    evidence = _evidence(tmp_path)

    reasons = runner._runtime_evidence_still_current(evidence)

    assert any("CURRENT_RUNTIME_REVOKED" in reason for reason in reasons)
