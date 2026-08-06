"""Contract tests for the canonical Gravity/Condense terminology layer."""
from __future__ import annotations

from copy import deepcopy
import json

import pytest

from lab.operators.condense_controller import (
    CONDENSE_SCHEMA,
    GRAVITY_SCHEMA,
    CondenseController,
    CondenseTask,
    GravityController,
    GravityTask,
)
from lab.receipts import Receipt, read_any_receipt, seal, verify
from lab.science_registry import (
    CONDENSE_OPERATOR_REGISTRY_SCHEMA,
    GRAVITY_OPERATOR_REGISTRY_SCHEMA,
    DEFAULT_REGISTRY,
)
from lab.semantic_taxonomy import (
    ARTIFACT_PROVENANCE_SCHEMA,
    CANONICAL_IDENTITY,
    CONDENSE_OPERATION,
    LEGACY_CONDENSE_MAPPING_ID,
    SEMANTIC_TAG_SCHEMA,
    SemanticTaxonomyError,
    normalize_semantic_tags,
    semantic_tags,
    taxonomy_manifest,
)
from lab.spec import SCHEMA, SpecError, validate_spec


def _spec(*, schema: str = SCHEMA, semantic_tags: dict | None = None) -> dict:
    value = {
        "schema": schema,
        "campaign_id": "semantic-test",
        "phases": ["precheck"],
        "steps": [],
        "reproduction": "true",
    }
    if semantic_tags is not None:
        value["semantic_tags"] = semantic_tags
    return value


def test_taxonomy_makes_gravity_public_and_condense_an_operation() -> None:
    taxonomy = taxonomy_manifest()
    assert taxonomy["schema"] == "hawking.semantic_taxonomy.v1"
    assert taxonomy["canonical_identity"]["id"] == CANONICAL_IDENTITY
    assert taxonomy["canonical_identity"]["public"] is True
    public_cli = taxonomy["canonical_identity"]["public_cli"]
    assert public_cli["command"] == "hawking gravity serve --artifact <PATH>"
    assert public_cli["status_command"] == "hawking gravity --json"
    assert public_cli["environment_variable_scope"] == (
        "legacy hawking serve --gravity selector only"
    )
    assert public_cli["command_exists"] is True
    assert public_cli["compatibility_commands"][0]["command"] == (
        "hawking serve --gravity <PATH>"
    )
    condense = taxonomy["operations"][CONDENSE_OPERATION]
    assert condense["engine_operation"] is True
    assert condense["canonical_identity"] == CANONICAL_IDENTITY
    assert condense["output_identity"] == CANONICAL_IDENTITY
    assert taxonomy["tag_contract"]["historical_records_are_rewritten"] is False
    mapping = taxonomy["tag_contract"]["field_tag_mapping"]
    assert mapping["id"] == LEGACY_CONDENSE_MAPPING_ID
    assert mapping["version"] == 1
    assert {entry["unit"] for entry in mapping["entries"]} >= {
        "schema_identifier",
        "identity_identifier",
        "operation_identifier",
        "reference_count",
    }
    legacy_schema = taxonomy["compatibility"]["legacy_schema_prefixes"][0]
    assert legacy_schema["deprecated"] is True
    assert legacy_schema["superseded_by"] == CANONICAL_IDENTITY
    assert legacy_schema["field_tag_mapping_id"] == LEGACY_CONDENSE_MAPPING_ID
    # Press is a legacy spelling of the Condense route, never a competing
    # operation or public identity.
    assert set(taxonomy["operations"]) == {CONDENSE_OPERATION}
    press = next(
        entry
        for entry in taxonomy["compatibility"]["legacy_command_aliases"]
        if entry["command"] == "hawking press"
    )
    assert press["maps_to_engine_operation"] == CONDENSE_OPERATION
    assert press["not_an_operation"] is True


def test_legacy_condense_spec_is_explicitly_resolved_without_mutating_input() -> None:
    raw = _spec(schema="hawking.condense.experiment_spec.v1")
    original = deepcopy(raw)

    spec = validate_spec(raw)

    assert raw == original
    assert spec.schema == SCHEMA
    assert spec.semantic_tags["schema"] == SEMANTIC_TAG_SCHEMA
    assert spec.semantic_tags["canonical_identity"] == CANONICAL_IDENTITY
    assert spec.semantic_tags["operation"] == CONDENSE_OPERATION
    assert spec.semantic_tags["source_tag_status"] == "derived_from_legacy_schema"
    assert spec.semantic_tags["legacy_alias"]["raw_schema"] == raw["schema"]
    assert spec.semantic_tags["legacy_alias"]["deprecated"] is True
    assert spec.semantic_tags["legacy_alias"]["superseded_by"] == CANONICAL_IDENTITY
    assert spec.semantic_tags["legacy_alias"]["field_tag_mapping_id"] == (
        LEGACY_CONDENSE_MAPPING_ID
    )
    assert spec.to_dict()["semantic_tags"]["legacy_alias"]["historical_record_rewritten"] is False


@pytest.mark.parametrize(
    "semantic_tags",
    [
        {
            "schema": SEMANTIC_TAG_SCHEMA,
            "canonical_identity": "condense",
            "operation": CONDENSE_OPERATION,
            "artifact_kind": "experiment_spec",
        },
        {
            "schema": SEMANTIC_TAG_SCHEMA,
            "canonical_identity": CANONICAL_IDENTITY,
            "operation": "gravity",
            "artifact_kind": "experiment_spec",
        },
    ],
)
def test_conflicting_identity_tags_fail_closed(semantic_tags: dict) -> None:
    with pytest.raises(SpecError, match="semantic_tags"):
        validate_spec(_spec(semantic_tags=semantic_tags))


def test_new_typed_receipts_are_sealed_with_gravity_identity_and_condense_operation() -> None:
    document = Receipt(
        campaign_id="taxonomy",
        verdict="PASS",
        artifacts=("/unresolved/model.gravity", "sha256:" + "a" * 64),
    ).to_dict()

    verify(document, label="taxonomy receipt")
    tags = document["semantic_tags"]
    assert tags["canonical_identity"] == CANONICAL_IDENTITY
    assert tags["operation"] == CONDENSE_OPERATION
    assert tags["artifact_kind"] == "lab_receipt"
    provenance = tags["artifact_provenance"]
    assert provenance == {
        "schema": ARTIFACT_PROVENANCE_SCHEMA,
        "assertion": "references_declared_not_resolved",
        "reference_count": 2,
        "locator_kinds": ["content_digest", "filesystem_path"],
    }
    assert "exists" not in provenance
    assert "available" not in provenance
    assert Receipt.from_dict(document).semantic_tags == tags


def test_normalizing_historical_receipt_does_not_rewrite_its_file(tmp_path) -> None:
    raw = {
        "schema": "hawking.condense.campaign_receipt.v1",
        "campaign_id": "old",
        "status": "retired",
        "seal_sha256": "historic-placeholder",
    }
    path = tmp_path / "old.json"
    path.write_text(json.dumps(raw, sort_keys=True), encoding="utf-8")
    before = path.read_bytes()

    normalized = read_any_receipt(path)

    assert path.read_bytes() == before
    assert normalized["raw_schema"] == raw["schema"]
    assert normalized["semantic_tags"]["canonical_identity"] == CANONICAL_IDENTITY
    assert normalized["semantic_tags"]["operation"] == CONDENSE_OPERATION
    assert normalized["semantic_tags"]["source_tag_status"] == "historical_unlabeled"
    assert normalized["semantic_tags"]["legacy_alias"]["raw_schema"] == raw["schema"]
    assert normalized["semantic_tags"]["legacy_alias"]["deprecated"] is True
    assert normalized["semantic_tags"]["legacy_alias"]["superseded_by"] == CANONICAL_IDENTITY


def test_normalizing_sealed_legacy_receipt_never_reseals_the_historical_bytes(tmp_path) -> None:
    raw = seal(
        {
            "schema": "hawking.condense.campaign_receipt.v1",
            "campaign_id": "sealed-old",
            "status": "retired",
            "artifacts": ["historic.gravity"],
        }
    )
    path = tmp_path / "sealed-old.json"
    path.write_text(json.dumps(raw, sort_keys=True), encoding="utf-8")
    before = path.read_bytes()

    normalized = read_any_receipt(path)

    assert path.read_bytes() == before
    verify(raw, label="sealed legacy source")
    assert normalized["seal_sha256"] == raw["seal_sha256"]
    assert normalized["semantic_tags"]["legacy_alias"]["deprecated"] is True
    assert normalized["semantic_tags"]["artifact_provenance"]["assertion"] == (
        "references_declared_not_resolved"
    )


def test_reading_an_old_sealed_receipt_only_derives_an_in_memory_tag() -> None:
    raw = seal(
        {
            "schema": "hawking.lab.receipt.v1",
            "campaign_id": "old-sealed",
            "verdict": "PASS",
            "inputs": {},
            "method": {},
            "measurement": {},
            "commit": "old",
            "phase": "complete",
            "status": "historical",
            "at": "2026-01-01T00:00:00Z",
            "reproduction": "historical",
            "artifacts": [],
            "summary": {},
        }
    )
    original = deepcopy(raw)

    restored = Receipt.from_dict(raw)

    assert raw == original
    verify(raw, label="original historical receipt")
    assert restored.semantic_tags["canonical_identity"] == CANONICAL_IDENTITY
    assert restored.semantic_tags["source_tag_status"] == "historical_unlabeled"


def test_non_inventive_artifact_provenance_only_summarizes_declared_references() -> None:
    tags = semantic_tags(
        operation=CONDENSE_OPERATION,
        artifact_kind="fixture",
        artifact_references=("relative.fixture", "https://example.invalid/model.gravity"),
    )
    provenance = tags["artifact_provenance"]
    assert provenance["assertion"] == "references_declared_not_resolved"
    assert provenance["reference_count"] == 2
    assert provenance["locator_kinds"] == ["opaque_reference", "uri"]

    with pytest.raises(SemanticTaxonomyError, match="availability-claim"):
        normalize_semantic_tags(
            {
                "schema": SEMANTIC_TAG_SCHEMA,
                "canonical_identity": CANONICAL_IDENTITY,
                "operation": CONDENSE_OPERATION,
                "artifact_kind": "fixture",
                "artifact_provenance": {
                    "schema": ARTIFACT_PROVENANCE_SCHEMA,
                    "assertion": "no_artifact_presence_asserted",
                    "reference_count": 0,
                    "locator_kinds": [],
                    "available": True,
                },
            },
            operation=CONDENSE_OPERATION,
            artifact_kind="fixture",
        )


def test_specs_only_mark_receipt_and_fixture_values_as_unresolved_references() -> None:
    raw = _spec()
    raw["receipt"] = "workspace/evidence/receipt.json"
    raw["fixture"] = "fixture.json"

    spec = validate_spec(raw)

    assert spec.semantic_tags["artifact_provenance"] == {
        "schema": ARTIFACT_PROVENANCE_SCHEMA,
        "assertion": "references_declared_not_resolved",
        "reference_count": 2,
        "locator_kinds": ["filesystem_path", "opaque_reference"],
    }


def test_gravity_controller_is_canonical_and_condense_names_remain_aliases() -> None:
    assert CondenseController is GravityController
    assert CondenseTask is GravityTask
    assert GRAVITY_SCHEMA == "hawking.gravity.rotation_controller.v1"
    assert CONDENSE_SCHEMA == "hawking.condense.rotation_controller.v1"

    controller = GravityController(
        [GravityTask("one", source_bytes=8, metadata_bytes=1, artifact_bytes={"gravity": 4})],
        byte_budget_bytes=16,
        heavy_lease_token="lease",
    )
    snapshot = controller.snapshot()
    assert snapshot["schema"] == GRAVITY_SCHEMA
    assert snapshot["semantic_tags"]["canonical_identity"] == CANONICAL_IDENTITY
    assert snapshot["semantic_tags"]["operation"] == CONDENSE_OPERATION
    assert snapshot["legacy_schema_compatibility"]["deprecated"] is True
    assert snapshot["legacy_schema_compatibility"]["superseded_by"] == CANONICAL_IDENTITY


def test_live_operator_registry_uses_gravity_schema_with_a_condense_alias() -> None:
    summary = DEFAULT_REGISTRY.summary()
    assert summary["schema"] == GRAVITY_OPERATOR_REGISTRY_SCHEMA
    assert summary["legacy_schema_aliases"] == [CONDENSE_OPERATOR_REGISTRY_SCHEMA]
    assert summary["semantic_tags"]["canonical_identity"] == CANONICAL_IDENTITY
    assert summary["semantic_tags"]["operation"] == CONDENSE_OPERATION
    assert summary["legacy_schema_compatibility"] == [
        {
            "raw_schema": CONDENSE_OPERATOR_REGISTRY_SCHEMA,
            "status": "legacy_operation_namespace",
            "deprecated": True,
            "superseded_by": CANONICAL_IDENTITY,
            "canonical_identity": CANONICAL_IDENTITY,
            "operation": CONDENSE_OPERATION,
            "field_tag_mapping_id": LEGACY_CONDENSE_MAPPING_ID,
            "historical_record_rewritten": False,
        }
    ]
