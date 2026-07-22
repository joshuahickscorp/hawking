from __future__ import annotations

import copy
import hashlib
import inspect
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from tools.condense.glm52_common import seal
from tools.condense import glm52_terminal_proofs as proofs


REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def current_proofs() -> dict[str, dict[str, Any]]:
    return proofs.derive_all_ready_stop_proofs(REPO_ROOT)


def _copy_bound_inputs(
    tmp_path: Path, proof: dict[str, Any]
) -> Path:
    for group in ("artifact_bindings", "document_bindings"):
        for relative in proof[group]:
            source = REPO_ROOT / relative
            target = tmp_path / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    return tmp_path


def _mutate_sealed(path: Path, mutation: Any) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    mutation(value)
    path.write_text(
        json.dumps(seal(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_all_and_only_nine_current_derivations_validate_read_only(
    current_proofs: dict[str, dict[str, Any]],
) -> None:
    assert tuple(current_proofs) == proofs.READY_STOP_CONDITIONS
    assert len(current_proofs) == 9
    validated = proofs.validate_all_ready_stop_proofs(REPO_ROOT, current_proofs)
    assert validated == current_proofs
    assert sum(value["status"] == "PASS" for value in current_proofs.values()) == 8
    assert current_proofs["kimi_raw_source_safely_released"]["status"] == "BLOCKED"


def test_proofs_bind_actual_validator_module_bytes(
    current_proofs: dict[str, dict[str, Any]],
) -> None:
    module_path = Path(proofs.__file__)
    raw = module_path.read_bytes()
    expected = {
        "path": "tools/condense/glm52_terminal_proofs.py",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }
    assert proofs.validator_source_binding() == expected
    assert {value["validator_source"]["sha256"] for value in current_proofs.values()} == {
        expected["sha256"]
    }


def test_semantic_derivations_are_not_production_receipts(
    current_proofs: dict[str, dict[str, Any]],
) -> None:
    for value in current_proofs.values():
        assert "seal_sha256" not in value
        assert "producer_hmac_sha256" not in value
        assert "expected_contract_sha256" not in value


def test_generic_test_true_object_is_rejected() -> None:
    with pytest.raises(proofs.TerminalProofError, match="fields"):
        proofs.validate_stop_proof(
            REPO_ROOT,
            "adapter_twin_green",
            {"test": True},
        )


def test_unknown_or_not_ready_stop_is_rejected() -> None:
    with pytest.raises(proofs.TerminalProofError, match="not offline-evidence-ready"):
        proofs.derive_stop_proof(REPO_ROOT, "capability_result_sealed")


def test_one_stops_proof_cannot_be_swapped_into_another(
    current_proofs: dict[str, dict[str, Any]],
) -> None:
    with pytest.raises(proofs.TerminalProofError, match="condition"):
        proofs.validate_stop_proof(
            REPO_ROOT,
            "corpus_integrity_green",
            current_proofs["adapter_twin_green"],
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema", "hawking.generic.test.v1", "schema"),
        ("campaign_id", "different-campaign", "campaign"),
        ("source_repo", "community/quantized-glm", "repository"),
        ("source_revision", "0" * 40, "revision"),
    ],
)
def test_wrong_proof_identity_is_rejected(
    current_proofs: dict[str, dict[str, Any]],
    field: str,
    value: str,
    message: str,
) -> None:
    changed = copy.deepcopy(current_proofs["bf16_source_manifest_complete"])
    changed[field] = value
    with pytest.raises(proofs.TerminalProofError, match=message):
        proofs.validate_stop_proof(
            REPO_ROOT, "bf16_source_manifest_complete", changed
        )


def test_status_or_blocker_shape_cannot_be_forged(
    current_proofs: dict[str, dict[str, Any]],
) -> None:
    changed = copy.deepcopy(current_proofs["adapter_twin_green"])
    changed["status"] = "BLOCKED"
    with pytest.raises(proofs.TerminalProofError, match="status/proven"):
        proofs.validate_stop_proof(REPO_ROOT, "adapter_twin_green", changed)

    changed = copy.deepcopy(current_proofs["kimi_raw_source_safely_released"])
    changed["blockers"] = []
    with pytest.raises(proofs.TerminalProofError, match="blocker"):
        proofs.validate_stop_proof(
            REPO_ROOT, "kimi_raw_source_safely_released", changed
        )


def test_validator_source_drift_is_rejected(
    current_proofs: dict[str, dict[str, Any]],
) -> None:
    changed = copy.deepcopy(current_proofs["external_baseline_matrix_complete"])
    changed["validator_source"]["sha256"] = "0" * 64
    with pytest.raises(proofs.TerminalProofError, match="validator-source bytes"):
        proofs.validate_stop_proof(
            REPO_ROOT, "external_baseline_matrix_complete", changed
        )


def test_wrong_bound_artifact_seal_is_rejected(
    current_proofs: dict[str, dict[str, Any]],
) -> None:
    changed = copy.deepcopy(current_proofs["exact_logical_weight_ledger_sealed"])
    changed["artifact_bindings"]["GLM52_LOGICAL_WEIGHT_LEDGER.json"][
        "seal_sha256"
    ] = "f" * 64
    with pytest.raises(proofs.TerminalProofError, match="grounded derivation"):
        proofs.validate_stop_proof(
            REPO_ROOT, "exact_logical_weight_ledger_sealed", changed
        )


def test_wrong_authoritative_total_is_rejected(
    current_proofs: dict[str, dict[str, Any]],
) -> None:
    changed = copy.deepcopy(current_proofs["exact_logical_weight_ledger_sealed"])
    changed["facts"]["logical_weight_denominator"] += 1
    with pytest.raises(proofs.TerminalProofError, match="grounded derivation"):
        proofs.validate_stop_proof(
            REPO_ROOT, "exact_logical_weight_ledger_sealed", changed
        )


def test_manifest_completeness_cannot_be_promoted_to_body_verification(
    current_proofs: dict[str, dict[str, Any]],
) -> None:
    proof = current_proofs["bf16_source_manifest_complete"]
    assert proof["facts"]["official_files"] == 295
    assert proof["facts"]["weight_shards"] == 282
    assert proof["facts"]["body_shards_fetched"] == 0
    assert proof["facts"]["body_shards_sha256_verified"] == 0
    assert proof["facts"]["body_verification_complete"] is False
    assert proof["scope"]["manifest_completeness_is_not_body_verification"] is True

    changed = copy.deepcopy(proof)
    changed["facts"]["body_shards_fetched"] = 282
    changed["facts"]["body_shards_sha256_verified"] = 282
    changed["facts"]["body_verification_complete"] = True
    with pytest.raises(proofs.TerminalProofError, match="grounded derivation"):
        proofs.validate_stop_proof(
            REPO_ROOT, "bf16_source_manifest_complete", changed
        )


def test_adapter_and_corpus_boundaries_do_not_overclaim(
    current_proofs: dict[str, dict[str, Any]],
) -> None:
    adapter = current_proofs["adapter_twin_green"]
    assert adapter["facts"]["body_backed_parent_parity"] is False
    assert adapter["facts"]["bf16_reference_forward_validated"] is False
    assert adapter["facts"]["capability_claimed"] is False
    corpus = current_proofs["corpus_integrity_green"]
    assert corpus["facts"]["model_payload_downloaded"] is False
    assert corpus["facts"]["network_access_used"] is False
    assert corpus["facts"]["capability_claimed"] is False
    assert corpus["facts"]["withheld_context_rungs"] == ["256K", "1M"]


def test_kimi_partial_evidence_and_rollback_exceptions_remain_visible(
    current_proofs: dict[str, dict[str, Any]],
) -> None:
    final = current_proofs["kimi_final_evidence_verified"]
    assert final["facts"]["terminal_outcome"] == "OUTCOME_C"
    assert final["facts"]["partial_prerequisite_evidence_preserved"] is True
    release = current_proofs["kimi_raw_source_safely_released"]
    assert release["proven"] is False
    assert release["facts"]["rollback_payload_preserved"] is False
    assert release["facts"]["rollback_runtime_preserved"] is False
    assert release["facts"]["rollback_capsule_status"] == (
        "DEGRADED_BY_PRIOR_BROAD_CLEANSE"
    )
    assert release["facts"]["credential_rotation_pending"] is True
    assert release["facts"]["total_cleanup_delta_bytes"] == 597_515_915_264
    assert len(release["blockers"]) == 4
    assert len(release["scope"]["remediation_required"]) == 5


@pytest.mark.parametrize(
    ("condition", "relative", "mutation"),
    [
        (
            "official_glm52_immutable_revision_sealed",
            "GLM52_SOURCE_ADMISSION.json",
            lambda value: value.__setitem__("revision", "0" * 40),
        ),
        (
            "bf16_source_manifest_complete",
            "GLM52_OFFICIAL_MANIFEST.json",
            lambda value: value.__setitem__("source_logical_bytes", 1),
        ),
        (
            "adapter_twin_green",
            "GLM52_ADAPTER_TWIN.json",
            lambda value: value.__setitem__("schema", "hawking.generic.v1"),
        ),
        (
            "corpus_integrity_green",
            "GLM52_CORPUS_INTEGRITY.json",
            lambda value: value.__setitem__("status", "PASS_CAPABILITY"),
        ),
    ],
)
def test_resealed_wrong_revision_total_schema_or_status_is_rejected(
    tmp_path: Path,
    current_proofs: dict[str, dict[str, Any]],
    condition: str,
    relative: str,
    mutation: Any,
) -> None:
    root = _copy_bound_inputs(tmp_path, current_proofs[condition])
    _mutate_sealed(root / relative, mutation)
    with pytest.raises(proofs.TerminalProofError, match="frozen seal"):
        proofs.derive_stop_proof(root, condition)


def test_bad_artifact_self_seal_is_rejected(
    tmp_path: Path,
    current_proofs: dict[str, dict[str, Any]],
) -> None:
    condition = "exact_logical_weight_ledger_sealed"
    root = _copy_bound_inputs(tmp_path, current_proofs[condition])
    path = root / "GLM52_LOGICAL_WEIGHT_LEDGER.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["logical_weight_denominator"] += 1
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(proofs.TerminalProofError, match="seal mismatch"):
        proofs.derive_stop_proof(root, condition)


def test_resealed_false_body_claim_in_manifest_is_rejected(
    tmp_path: Path,
    current_proofs: dict[str, dict[str, Any]],
) -> None:
    condition = "bf16_source_manifest_complete"
    root = _copy_bound_inputs(tmp_path, current_proofs[condition])

    def mutation(value: dict[str, Any]) -> None:
        row = next(item for item in value["files"] if item["is_weight"])
        row["download_state"] = "PRESENT"
        row["verification_state"] = "FULL_FILE_VERIFIED"
        row["local"]["state"] = "PRESENT_VERIFIED"

    _mutate_sealed(root / "GLM52_OFFICIAL_MANIFEST.json", mutation)
    with pytest.raises(proofs.TerminalProofError, match="frozen seal"):
        proofs.derive_stop_proof(root, condition)


def test_module_has_no_write_receipt_or_external_execution_api() -> None:
    public_functions = {
        name
        for name, value in vars(proofs).items()
        if inspect.isfunction(value)
        and value.__module__ == proofs.__name__
        and not name.startswith("_")
    }
    assert public_functions == {
        "validator_source_binding",
        "derive_stop_proof",
        "derive_all_ready_stop_proofs",
        "validate_stop_proof",
        "validate_all_ready_stop_proofs",
    }
    source = Path(proofs.__file__).read_text(encoding="utf-8")
    for forbidden in ("atomic_json(", "unlink(", "remove(", "subprocess.", "socket."):
        assert forbidden not in source


def test_reader_rejects_symlink_ancestor_root_and_hardlinked_evidence(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(proofs.TerminalProofError, match="root component"):
        proofs._Reader(alias)

    evidence = real / "evidence.json"
    evidence.write_bytes(b"{}")
    (real / "evidence-link.json").hardlink_to(evidence)
    reader = proofs._Reader(real)
    with pytest.raises(proofs.TerminalProofError, match="multiple hard links"):
        reader.raw("evidence.json")


def test_reader_rejects_root_replacement_after_initialization(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    reader = proofs._Reader(root)
    moved = tmp_path / "old-root"
    root.rename(moved)
    root.mkdir()
    (root / "evidence.json").write_bytes(b"{}")
    with pytest.raises(proofs.TerminalProofError, match="root identity changed"):
        reader.raw("evidence.json")
