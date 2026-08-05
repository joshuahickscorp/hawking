"""Canonical Gravity/Condense semantic taxonomy.

This module is deliberately data-led: the JSON manifest is the shared
machine-readable contract for the Python lab and HCLI.  It labels new output;
it never edits historical artifacts or changes their sealed bytes.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


TAXONOMY_PATH = Path(__file__).with_name("semantic_taxonomy.json")
TAXONOMY_SCHEMA = "hawking.semantic_taxonomy.v1"
SEMANTIC_TAG_SCHEMA = "hawking.semantic_tags.v1"
ARTIFACT_PROVENANCE_SCHEMA = "hawking.artifact_provenance.v1"
CANONICAL_IDENTITY = "gravity"
CONDENSE_OPERATION = "condense"
LEGACY_CONDENSE_MAPPING_ID = "hawking.gravity.condense_compatibility.v1"
_SOURCE_TAG_STATUSES = frozenset(
    {
        "declared",
        "generated_current",
        "derived_from_legacy_schema",
        "historical_unlabeled",
    }
)
_ARTIFACT_PROVENANCE_ASSERTIONS = frozenset(
    {
        "no_artifact_presence_asserted",
        "references_declared_not_resolved",
    }
)
_ARTIFACT_LOCATOR_KINDS = frozenset(
    {
        "opaque_reference",
        "filesystem_path",
        "uri",
        "content_digest",
    }
)
_SHA256_REFERENCE = re.compile(r"^sha256:[0-9a-fA-F]{64}$")


class SemanticTaxonomyError(ValueError):
    """A semantic tag conflicts with the canonical Gravity taxonomy."""


def taxonomy_manifest() -> dict[str, Any]:
    """Return a copy of the canonical, machine-readable taxonomy manifest."""
    try:
        raw = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - deployment failure
        raise SemanticTaxonomyError(f"cannot load semantic taxonomy: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("schema") != TAXONOMY_SCHEMA:
        raise SemanticTaxonomyError("semantic taxonomy has an unsupported schema")
    identity = raw.get("canonical_identity")
    operations = raw.get("operations")
    claim_limits = raw.get("capability_claim_limits")
    tag_contract = raw.get("tag_contract")
    provenance_contract = raw.get("artifact_provenance_contract")
    compatibility = raw.get("compatibility")
    if (
        not isinstance(identity, Mapping)
        or identity.get("id") != CANONICAL_IDENTITY
        or not isinstance(identity.get("public_cli"), Mapping)
        or not isinstance(identity["public_cli"].get("command"), str)
        or not identity["public_cli"]["command"]
        or not isinstance(identity["public_cli"].get("command_exists"), bool)
        or not isinstance(operations, Mapping)
        or not isinstance(operations.get(CONDENSE_OPERATION), Mapping)
        or operations[CONDENSE_OPERATION].get("canonical_identity") != CANONICAL_IDENTITY
        or operations[CONDENSE_OPERATION].get("output_identity") != CANONICAL_IDENTITY
        or operations[CONDENSE_OPERATION].get("engine_operation") is not True
        or not isinstance(claim_limits, Mapping)
        or claim_limits.get("hcli_executes_optimization") is not False
        or claim_limits.get("v4_runtime_claim") is not False
        or not isinstance(tag_contract, Mapping)
        or tag_contract.get("schema") != SEMANTIC_TAG_SCHEMA
        or tag_contract.get("artifact_provenance_key") != "artifact_provenance"
        or not isinstance(tag_contract.get("field_tag_mapping"), Mapping)
        or tag_contract["field_tag_mapping"].get("id") != LEGACY_CONDENSE_MAPPING_ID
        or tag_contract["field_tag_mapping"].get("version") != 1
        or not isinstance(tag_contract["field_tag_mapping"].get("entries"), list)
        or not isinstance(provenance_contract, Mapping)
        or provenance_contract.get("schema") != ARTIFACT_PROVENANCE_SCHEMA
        or not isinstance(provenance_contract.get("assertion_values"), list)
        or not _ARTIFACT_PROVENANCE_ASSERTIONS.issubset(
            set(provenance_contract["assertion_values"])
        )
        or not isinstance(provenance_contract.get("locator_kind_values"), list)
        or not _ARTIFACT_LOCATOR_KINDS.issubset(
            set(provenance_contract["locator_kind_values"])
        )
        or not isinstance(compatibility, Mapping)
        or not isinstance(compatibility.get("legacy_schema_prefixes"), list)
        or not isinstance(compatibility.get("legacy_command_aliases"), list)
    ):
        raise SemanticTaxonomyError("semantic taxonomy does not bind Condense to Gravity")
    legacy_entries = compatibility["legacy_schema_prefixes"]
    if not any(
        isinstance(entry, Mapping)
        and entry.get("prefix") == "hawking.condense."
        and entry.get("deprecated") is True
        and entry.get("superseded_by") == CANONICAL_IDENTITY
        and entry.get("field_tag_mapping_id") == LEGACY_CONDENSE_MAPPING_ID
        and entry.get("operation") == CONDENSE_OPERATION
        for entry in legacy_entries
    ):
        raise SemanticTaxonomyError("legacy Condense schemas must advertise their Gravity successor")
    mapping_entries = tag_contract["field_tag_mapping"]["entries"]
    if not all(
        isinstance(entry, Mapping)
        and all(
            isinstance(entry.get(field), str) and entry[field]
            for field in ("source_field", "target_field", "transform", "unit")
        )
        for entry in mapping_entries
    ):
        raise SemanticTaxonomyError("field/tag mapping entries must name source, target, transform, and unit")
    required_mapping_targets = {
        "semantic_tags.legacy_alias.raw_schema",
        "semantic_tags.canonical_identity",
        "semantic_tags.operation",
        "semantic_tags.legacy_alias.deprecated",
        "semantic_tags.legacy_alias.superseded_by",
        "semantic_tags.artifact_provenance.reference_count",
    }
    if not required_mapping_targets.issubset(
        {entry["target_field"] for entry in mapping_entries}
    ):
        raise SemanticTaxonomyError("field/tag mapping omits a required Gravity compatibility target")
    if not any(
        isinstance(entry, Mapping)
        and entry.get("command") == "hawking press"
        and entry.get("maps_to_engine_operation") == CONDENSE_OPERATION
        and entry.get("not_an_operation") is True
        for entry in compatibility["legacy_command_aliases"]
    ):
        raise SemanticTaxonomyError("Press must remain a legacy command alias, not an operation")
    return deepcopy(raw)


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SemanticTaxonomyError(f"{label} must be a non-empty string")
    return value


def legacy_alias_for_schema(schema: str | None) -> dict[str, Any] | None:
    """Resolve a legacy namespace without changing its original schema value."""
    if not schema:
        return None
    raw_schema = _string(schema, "schema")
    for entry in taxonomy_manifest()["compatibility"]["legacy_schema_prefixes"]:
        if raw_schema.startswith(entry["prefix"]):
            return {
                "raw_schema": raw_schema,
                "status": entry["status"],
                "deprecated": entry["deprecated"],
                "superseded_by": entry["superseded_by"],
                "canonical_identity": entry["canonical_identity"],
                "operation": entry["operation"],
                "field_tag_mapping_id": entry["field_tag_mapping_id"],
                "historical_record_rewritten": False,
            }
    return None


def _normalized_declared_references(
    references: Sequence[object] | None,
) -> tuple[str, ...]:
    """Normalize supplied artifact references without resolving any of them.

    This deliberately treats references as opaque document text.  It does not
    stat paths, open files, make a network request, or infer current artifact
    availability.  A non-empty reference list therefore becomes a claim about
    the source document only, not about the referenced artifact.
    """
    if references is None:
        return ()
    if isinstance(references, (str, bytes)) or not isinstance(references, Sequence):
        raise SemanticTaxonomyError("artifact references must be a sequence of non-empty strings")
    normalized: list[str] = []
    for index, reference in enumerate(references):
        if not isinstance(reference, str) or not reference.strip():
            raise SemanticTaxonomyError(
                f"artifact reference at index {index} must be a non-empty string"
            )
        normalized.append(reference)
    return tuple(normalized)


def _locator_kind(reference: str) -> str:
    """Classify a reference's syntax only; this says nothing about existence."""
    if _SHA256_REFERENCE.fullmatch(reference):
        return "content_digest"
    if "://" in reference:
        return "uri"
    if reference.startswith(("/", "./", "../", "~")) or "/" in reference:
        return "filesystem_path"
    return "opaque_reference"


def artifact_provenance(
    *,
    declared_references: Sequence[object] | None = None,
) -> dict[str, Any]:
    """Summarize artifact references without asserting an artifact exists.

    The returned structure is safe for plans, receipts, and status output.  It
    communicates only whether the input record named references; no filesystem
    probe, download, or model/artifact availability inference occurs here.
    """
    references = _normalized_declared_references(declared_references)
    if not references:
        return {
            "schema": ARTIFACT_PROVENANCE_SCHEMA,
            "assertion": "no_artifact_presence_asserted",
            "reference_count": 0,
            "locator_kinds": [],
        }
    return {
        "schema": ARTIFACT_PROVENANCE_SCHEMA,
        "assertion": "references_declared_not_resolved",
        "reference_count": len(references),
        "locator_kinds": sorted({_locator_kind(reference) for reference in references}),
    }


def normalize_artifact_provenance(
    value: Mapping[str, Any] | None,
    *,
    declared_references: Sequence[object] | None = None,
) -> dict[str, Any]:
    """Validate non-assertive provenance, or derive it from declared references.

    Only non-inventive assertions are in this vocabulary.  In particular,
    callers cannot smuggle an ``exists``/``available`` claim through taxonomy
    metadata: evidence-bearing artifact checks belong to their own sealed,
    runtime-specific contracts.
    """
    expected = artifact_provenance(declared_references=declared_references)
    if value is None:
        return expected
    if not isinstance(value, Mapping):
        raise SemanticTaxonomyError("artifact_provenance must be an object")
    allowed = {"schema", "assertion", "reference_count", "locator_kinds"}
    unknown = set(value) - allowed
    if unknown:
        raise SemanticTaxonomyError(
            f"artifact_provenance has unknown or availability-claim fields: {sorted(unknown)}"
        )
    if value.get("schema") != ARTIFACT_PROVENANCE_SCHEMA:
        raise SemanticTaxonomyError(
            f"artifact_provenance.schema must be {ARTIFACT_PROVENANCE_SCHEMA!r}"
        )
    assertion = value.get("assertion")
    if assertion not in _ARTIFACT_PROVENANCE_ASSERTIONS:
        raise SemanticTaxonomyError("artifact_provenance.assertion is unsupported")
    reference_count = value.get("reference_count")
    if isinstance(reference_count, bool) or not isinstance(reference_count, int) or reference_count < 0:
        raise SemanticTaxonomyError("artifact_provenance.reference_count must be a non-negative integer")
    locator_kinds = value.get("locator_kinds")
    if not isinstance(locator_kinds, list) or any(
        not isinstance(kind, str) or kind not in _ARTIFACT_LOCATOR_KINDS
        for kind in locator_kinds
    ):
        raise SemanticTaxonomyError(
            "artifact_provenance.locator_kinds must be a list of supported locator kinds"
        )
    if locator_kinds != sorted(set(locator_kinds)):
        raise SemanticTaxonomyError(
            "artifact_provenance.locator_kinds must be sorted and duplicate-free"
        )
    if assertion == "no_artifact_presence_asserted" and (
        reference_count != 0 or locator_kinds
    ):
        raise SemanticTaxonomyError(
            "no_artifact_presence_asserted requires zero references and no locator kinds"
        )
    if assertion == "references_declared_not_resolved" and (
        reference_count <= 0 or not locator_kinds
    ):
        raise SemanticTaxonomyError(
            "references_declared_not_resolved requires declared references and locator kinds"
        )
    normalized = {
        "schema": ARTIFACT_PROVENANCE_SCHEMA,
        "assertion": assertion,
        "reference_count": reference_count,
        "locator_kinds": list(locator_kinds),
    }
    if declared_references is not None and normalized != expected:
        raise SemanticTaxonomyError(
            "artifact_provenance conflicts with the record's declared artifact references"
        )
    return normalized


def semantic_tags(
    *,
    operation: str | None = None,
    artifact_kind: str | None = None,
    raw_schema: str | None = None,
    source_tag_status: str | None = None,
    artifact_references: Sequence[object] | None = None,
) -> dict[str, Any]:
    """Create canonical tags for newly emitted or normalized metadata.

    ``operation`` is intentionally optional: a Gravity artifact need not claim
    that it was Condensed.  A legacy Condense schema is represented as a
    compatibility alias only; the source record remains untouched.
    """
    if operation is not None and operation != CONDENSE_OPERATION:
        raise SemanticTaxonomyError(
            f"operation {operation!r} conflicts with supported operation {CONDENSE_OPERATION!r}"
        )
    if artifact_kind is not None:
        artifact_kind = _string(artifact_kind, "artifact_kind")
    legacy = legacy_alias_for_schema(raw_schema)
    status = source_tag_status or (
        "derived_from_legacy_schema" if legacy is not None else "generated_current"
    )
    if status not in _SOURCE_TAG_STATUSES:
        raise SemanticTaxonomyError(f"unsupported source_tag_status {status!r}")
    result: dict[str, Any] = {
        "schema": SEMANTIC_TAG_SCHEMA,
        "canonical_identity": CANONICAL_IDENTITY,
        "source_tag_status": status,
        "artifact_provenance": artifact_provenance(
            declared_references=artifact_references,
        ),
    }
    if operation is not None:
        result["operation"] = operation
    if artifact_kind is not None:
        result["artifact_kind"] = artifact_kind
    if legacy is not None:
        result["legacy_alias"] = legacy
    return result


def normalize_semantic_tags(
    value: Mapping[str, Any] | None,
    *,
    operation: str | None = None,
    artifact_kind: str | None = None,
    raw_schema: str | None = None,
    missing_status: str | None = None,
    artifact_references: Sequence[object] | None = None,
) -> dict[str, Any]:
    """Validate declared tags or synthesize explicit tags for unlabeled input.

    This normalizes an in-memory representation only.  Callers must explicitly
    write a new document if they want the tags persisted.
    """
    expected = semantic_tags(
        operation=operation,
        artifact_kind=artifact_kind,
        raw_schema=raw_schema,
        source_tag_status=missing_status,
        artifact_references=artifact_references,
    )
    if value is None:
        return expected
    if not isinstance(value, Mapping):
        raise SemanticTaxonomyError("semantic_tags must be an object")
    allowed = {
        "schema",
        "canonical_identity",
        "operation",
        "artifact_kind",
        "artifact_provenance",
        "source_tag_status",
        "legacy_alias",
    }
    unknown = set(value) - allowed
    if unknown:
        raise SemanticTaxonomyError(f"semantic_tags has unknown fields: {sorted(unknown)}")
    if value.get("schema") != SEMANTIC_TAG_SCHEMA:
        raise SemanticTaxonomyError(
            f"semantic_tags.schema must be {SEMANTIC_TAG_SCHEMA!r}"
        )
    if value.get("canonical_identity") != CANONICAL_IDENTITY:
        raise SemanticTaxonomyError(
            "semantic_tags.canonical_identity must be 'gravity'; Condense is an operation"
        )
    actual_operation = value.get("operation")
    if actual_operation is not None and actual_operation != CONDENSE_OPERATION:
        raise SemanticTaxonomyError(
            "semantic_tags.operation must be 'condense' when supplied"
        )
    if operation is not None and actual_operation != operation:
        raise SemanticTaxonomyError(
            f"semantic_tags.operation must be {operation!r} for this record"
        )
    actual_kind = value.get("artifact_kind")
    if actual_kind is not None:
        _string(actual_kind, "semantic_tags.artifact_kind")
    if artifact_kind is not None and actual_kind != artifact_kind:
        raise SemanticTaxonomyError(
            f"semantic_tags.artifact_kind must be {artifact_kind!r} for this record"
        )
    try:
        provenance = normalize_artifact_provenance(
            value.get("artifact_provenance"),
            declared_references=artifact_references,
        )
    except SemanticTaxonomyError as exc:
        raise SemanticTaxonomyError(f"semantic_tags.artifact_provenance: {exc}") from exc
    status = value.get("source_tag_status", "declared")
    if not isinstance(status, str) or status not in _SOURCE_TAG_STATUSES:
        raise SemanticTaxonomyError(
            "semantic_tags.source_tag_status must name a supported source status"
        )
    declared_legacy = value.get("legacy_alias")
    if declared_legacy is not None and not isinstance(declared_legacy, Mapping):
        raise SemanticTaxonomyError("semantic_tags.legacy_alias must be an object")
    if declared_legacy is not None and declared_legacy != expected.get("legacy_alias"):
        raise SemanticTaxonomyError("semantic_tags.legacy_alias conflicts with the raw schema")
    result = dict(expected)
    result["source_tag_status"] = status
    result["artifact_provenance"] = provenance
    return result
