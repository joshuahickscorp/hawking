from __future__ import annotations

import json
from pathlib import Path

import pytest

from lab.operators import ascension_paired_authority_materialization_refusal as refusal
from lab.operators.ascension_manager_tournament_readiness_report import (
    AUTHORITY_SPECS,
    ReadinessPaths,
)
from lab.operators.ascension_manager_tournament_readiness_report import (
    STATUS_PREPARED as READINESS_PREPARED_STATUS,
)
from lab.receipts import seal, verify


def _write(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def _paths(tmp_path: Path) -> ReadinessPaths:
    return ReadinessPaths.from_roots(
        repository_root=tmp_path / "repo",
        lifecycle_root=tmp_path / "lifecycle",
        physical_root=tmp_path / "physical",
        authority_root=tmp_path / "lifecycle" / "paired-authorities",
    )


def _refused_readiness(
    paths: ReadinessPaths, *, claimed_ready: bool
) -> dict[str, object]:
    assert claimed_ready is False
    sources = {
        spec.key: {
            "source_path": str(paths.repository_root / spec.source_relative_path),
            "source_sha256": f"{ordinal:064x}",
            "expected_schema": spec.expected_schema,
            "expected_status": spec.expected_status,
            "prepared_definition_verified": True,
        }
        for ordinal, spec in enumerate(AUTHORITY_SPECS, start=1)
    }
    materializations = {
        spec.key: {"prepared": False, "issues": ["missing"]} for spec in AUTHORITY_SPECS
    }
    return seal(
        {
            "schema": "hawking.ascension.manager_tournament_readiness_report.v1",
            "status": "REFUSED_MANAGER_TOURNAMENT_READINESS_CONJUNCTION_INCOMPLETE_OR_UNTRUSTED",
            "missing_prerequisites": [
                "both_exact_tg10_operational_receipts",
                "prepared_paired_and_protected_contract_identities",
            ],
            "blockers": [
                "qwen30 operational ascent lacks a sealed TG10 receipt identity"
            ],
            "prepared_authority_source_identities": sources,
            "prepared_authority_materializations": materializations,
        }
    )


def _write_observed_records(paths: ReadinessPaths) -> None:
    _write(
        paths.operational_ascent,
        seal(
            {
                "schema": "hawking.ascension.physical_operational_ascent.v1",
                "status": "WAITING_FOR_BOTH_VALID_TG10_OPERATIONAL_RECEIPTS",
                "both_valid_tg10_receipts": False,
                "evidence": {
                    "models": {
                        "qwen30": {"tg10_receipt_seal_sha256": None},
                        "qwen80": {"tg10_receipt_seal_sha256": None},
                    }
                },
            }
        ),
    )
    _write(
        paths.physical_root
        / "qwen30"
        / "complete-runtime"
        / "QWEN30_EXACT_FULL_TOKEN_RUNTIME_RECEIPT.json",
        seal(
            {
                "schema": "hawking.ascension.physical_exact_full_token_runtime.v1",
                "status": "PASS_EXACT_NATIVE_FULL_TOKEN_RUNTIME",
            }
        ),
    )
    _write(
        paths.physical_root
        / "qwen30"
        / "complete-runtime"
        / "QWEN30_NATIVE_HTTP_ADAPTER_STATUS.json",
        {
            "schema": "hawking.ascension.qwen30_native_http_adapter.v1",
            "phase": "UNQUALIFIED",
        },
    )
    _write(
        paths.physical_root
        / "qwen30"
        / "tps-gate"
        / "QWEN30_BASE_TRUE_TPS_GATE_STATUS.json",
        {
            "schema": "hawking.ascension.physical_base_true_tps_gate_status.v1",
            "phase": "BLOCKED",
        },
    )
    _write(
        paths.physical_root / "qwen30" / "tg3" / "QWEN30_TG3_ASCENT_STATUS.json",
        {"schema": "hawking.ascension.qwen30_tg3_ascent.v1", "status": "BLOCKED"},
    )
    _write(
        paths.physical_root
        / "qwen80"
        / "complete-runtime"
        / "QWEN80_COMPLETE_DECODER_READINESS_AFTER_SOURCE_TOKEN_L0_20260809T063300Z.json",
        {
            "schema": "hawking.ascension.qwen80_complete_decoder_readiness_result.v1",
            "status": "INCOMPLETE",
            "complete_decoder_readiness_earned": False,
        },
    )
    _write(
        paths.physical_root
        / "qwen80"
        / "tps-gate"
        / "QWEN80_BASE_TRUE_TPS_GATE_STATUS.json",
        {
            "schema": "hawking.ascension.physical_base_true_tps_gate_status.v1",
            "phase": "WAITING",
        },
    )
    _write(
        paths.physical_root / "qwen80" / "tg3" / "QWEN80_TG3_ASCENT_STATUS.json",
        {"schema": "hawking.ascension.qwen80_bootstrap_lanes.v1", "phase": "WAITING"},
    )


def test_builds_sealed_refusal_without_creating_any_prepared_authority(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    _write_observed_records(paths)

    report = refusal.build_materialization_refusal(
        paths,
        recorded_at="2026-08-09T12:00:00Z",
        readiness_builder=_refused_readiness,
    )

    assert report["schema"] == refusal.SCHEMA
    assert report["status"] == refusal.STATUS
    assert report["prepared"] is False
    assert report["authority_chain_materialized"] is False
    assert report["expected_authority_documents_written"] == []
    assert report["operational_tg10_snapshot"]["both_valid_tg10_receipts"] is False
    assert (
        report["operational_tg10_snapshot"]["qwen30_tg10_receipt_seal_sha256"] is None
    )
    assert (
        report["operational_tg10_snapshot"]["qwen80_tg10_receipt_seal_sha256"] is None
    )
    assert len(report["missing_sealed_model_authorities"]) == 6
    assert all(
        value is False
        for value in report["readiness_report_delta"][
            "prepared_authorities_after"
        ].values()
    )
    assert report["execution_boundary"]["model_weights_loaded"] is False
    assert report["execution_boundary"]["metal_device_or_dispatch_performed"] is False
    verify(report)

    output = refusal.write_new_refusal(paths, report)
    assert output == paths.authority_root / refusal.OUTPUT_FILENAME
    verify(json.loads(output.read_text(encoding="utf-8")))
    assert not any(paths.authority_document(spec).exists() for spec in AUTHORITY_SPECS)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        refusal.write_new_refusal(paths, report)


def test_unsealed_or_unrelated_observations_never_become_topology_authority(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    _write_observed_records(paths)

    report = refusal.build_materialization_refusal(
        paths,
        recorded_at="2026-08-09T12:00:00Z",
        readiness_builder=_refused_readiness,
    )

    adapter = report["fixed_observed_records"]["qwen30_native_http_adapter"]
    decoder = report["fixed_observed_records"]["qwen80_complete_decoder_readiness"]
    assert adapter["sealed"] is False
    assert decoder["sealed"] is False
    assert all(
        "no matching sealed one-body/many-logical-session topology authority"
        in row["reason"]
        for row in report["missing_sealed_model_authorities"]
    )
    assert (
        report["required_one_body_topology"]["qwen30"]["topology_assertion"][
            "resident_model_processes"
        ]
        == 1
    )
    assert (
        report["required_one_body_topology"]["qwen80"]["topology_assertion"][
            "logical_session_policy"
        ]
        == "many_logical_sessions"
    )


def test_fails_closed_if_an_injected_snapshot_claims_readiness(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_observed_records(paths)

    def ready_snapshot(_: ReadinessPaths, *, claimed_ready: bool) -> dict[str, object]:
        assert claimed_ready is False
        return seal(
            {
                "schema": "hawking.ascension.manager_tournament_readiness_report.v1",
                "status": READINESS_PREPARED_STATUS,
                "missing_prerequisites": [],
                "blockers": [],
                "prepared_authority_source_identities": {},
                "prepared_authority_materializations": {},
            }
        )

    with pytest.raises(
        refusal.PairedAuthorityMaterializationError, match="cannot materialize"
    ):
        refusal.build_materialization_refusal(paths, readiness_builder=ready_snapshot)
    assert not (paths.authority_root / refusal.OUTPUT_FILENAME).exists()
