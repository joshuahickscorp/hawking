#!/usr/bin/env python3
"""Query Hawking's canonical Gravity method registry by model capabilities.

The registry remains the sole owner of method descriptions and evidence.  This
reader only performs deterministic tag matching; it does not promote a method
or turn a scoped Flash result into a cross-model conclusion.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from fnmatch import fnmatchcase
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.odyssey.modellake_lineage import (  # noqa: E402
    SELECTED_TENSOR_HEADER_MAX_NAMES,
    SELECTED_TENSOR_HEADER_MAX_SHARDS,
    SELECTED_TENSOR_HEADERS_SCHEMA,
    validate_selected_tensor_header_witness,
)

REGISTRY = ROOT / "tools/foundry/GRAVITY_METHOD_REGISTRY.json"
SCAR_ATLAS = ROOT / "tools/foundry/NEGATIVE_TRANSFER_ATLAS.json"
OBSERVATION_SCHEMA = "hawking.gravity.observed_traits.v1"
EXECUTION_SCHEMA = "hawking.gravity.method_execution.v2"
REUSE_SCHEMA = EXECUTION_SCHEMA  # compatibility name; new receipts are v2.
PASSPORT_CATALOG_SCHEMA = "hawking.gravity.organ_passport_catalog.v1"
PASSPORT_SCHEMA = "hawking.gravity.organ_passport.v1"
STATIC_TRAITS_SCHEMA = "hawking.gravity.static_organ_traits.v1"
PASSPORT_ASSESSMENT_SCHEMA = "hawking.gravity.organ_passport_assessment.v1"
E13_PREFLIGHT_REL = Path(
    "receipts/headless/FLASH_E13_VOCABULARY_PRODUCER_FRONTIER_HAWKING_V3_20260914.json"
)
E13_FIRST_REFUSAL_REL = Path(
    "receipts/headless/FLASH_E13_WIKINEWS_FIXED_CORPUS_REFUSAL_20260912.json"
)
E13_UNIVERSE_REL = Path(
    "receipts/headless/FLASH_E13_LEXICAL_UNIVERSE_CENSUS_20260912.json"
)
E13_REFUSAL_REL = Path(
    "receipts/headless/FLASH_E13_WIKIMEDIA_BROADER_FIXED_CORPUS_REFUSAL_20260912.json"
)
E13_WITHHELD_STATE = (
    "WITHHELD_BROADER_WIKIMEDIA_CONTROL_UNDERPOWERED__DESIGN_REVIEW_REQUIRED"
)
E06_PREFILTER_REL = Path("receipts/headless/FLASH_E06_CONDITIONAL_INFORMATION_PREFILTER_BOUND_20260912.json")
E06_SEALED_STATE = "SEALED_FIXED_ROW_NORM_CONDITIONAL_INFORMATION_PREFILTER_NEGATIVE"
E02_REPLICATION_REL = Path(
    "receipts/headless/FLASH_EXPERT_LAWFUL_ATOM_SHARING_TOP8_REPLICATION_SEALED_20260912.json"
)
E02_SEALED_STATE = "SEALED_FIXED_TOP8_PAIR_REPLICATION_NEGATIVE__ORIGINAL_PAIR_BOUNDED_SIGNAL_ONLY"
SEALED_JSON_ADAPTER_TIMEOUT_SECONDS = 120.0
SCOPED_SCAR_FIELDS = frozenset(
    {"id", "scope", "evidence", "evidence_sha256", "verdict", "does_not_rule_out", "reopen_if"}
)
SCOPED_SCAR_REQUIRED_SCOPE_KEYS = frozenset(
    {
        "source_revision",
        "organ_scope",
        "representation_family",
        "tested_contract",
        "rate_scope",
        "control_id",
        "control_sha256",
    }
)


@dataclass(frozen=True)
class FileIdentity:
    """A regular-file identity captured without executing its contents."""

    path: str
    sha256: str
    bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "bytes": self.bytes}


@dataclass(frozen=True)
class MethodExecutionAdapter:
    """One admitted automatic adapter, parsed before any child is spawned."""

    id: str
    architecture_tags: frozenset[str]
    required_inputs: tuple[str, ...]
    required_input_paths: tuple[Path, ...]
    runner_path: Path
    runner_kind: str
    timeout_seconds: float
    verifier_path: Path
    verifier_kind: str
    verifier_symbol: str
    argv_template: tuple[str, ...]
    verification_spec: dict[str, Any]


def _canonical_json_sha256(document: Any) -> str:
    return hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _load_sealed_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read {label}: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"{label} must be a JSON object")
    unsealed = dict(document)
    observed_seal = unsealed.pop("seal_sha256", None)
    if observed_seal != _canonical_json_sha256(unsealed):
        raise ValueError(f"{label} seal mismatch")
    return document, observed_seal


def _display_path(root: Path, path: Path) -> str:
    """Use repository-relative names when possible, absolute names otherwise."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _resolve_execution_path(path: Path, root: Path) -> Path:
    """Give the parent and child process one unambiguous output location.

    Resolve the parent, but retain the requested final leaf so a pre-existing
    destination symlink is observed and refused rather than followed.
    """
    requested = Path(path)
    candidate = requested if requested.is_absolute() else root / requested
    return candidate.parent.resolve() / candidate.name


def load_registry(path: Path = REGISTRY) -> dict[str, Any]:
    doc = json.loads(path.read_text())
    if doc.get("schema") != "hawking.foundry.gravity_method_registry.v2":
        raise ValueError(f"unsupported registry schema: {doc.get('schema')!r}")
    methods = doc.get("methods")
    if not isinstance(methods, list) or not methods:
        raise ValueError("canonical registry has no queryable methods")
    owners = doc.get("canonical_owners")
    if not isinstance(owners, dict) or not owners:
        raise ValueError("canonical registry has no ownership map")
    validate_passports(doc)
    return doc


def _string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{label} must be a list of non-empty strings")
    return list(value)


def _relative_regular_file(raw_path: Any, label: str) -> None:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label} must be a non-empty repository-relative path")
    path = Path(raw_path)
    if path.is_absolute() or ".." in path.parts or not (ROOT / path).is_file():
        raise ValueError(f"{label} is not an existing regular repository file: {raw_path!r}")


def _validate_predicate(predicate: Any, label: str) -> None:
    """Validate the deliberately small, static-only recognition predicate AST."""
    if not isinstance(predicate, dict) or not predicate:
        raise ValueError(f"{label} must be a non-empty predicate object")
    boolean_keys = {"all", "any", "not"}
    present_boolean = boolean_keys & set(predicate)
    if present_boolean:
        if set(predicate) != present_boolean:
            raise ValueError(f"{label} cannot mix boolean and leaf predicate fields")
        for key in present_boolean:
            children = predicate[key]
            if not isinstance(children, list) or not children:
                raise ValueError(f"{label}.{key} must be a non-empty predicate list")
            for index, child in enumerate(children):
                _validate_predicate(child, f"{label}.{key}[{index}]")
        return

    kind = predicate.get("kind")
    if kind in {"config_equals", "config_contains", "identity_equals"}:
        if set(predicate) != {"kind", "path", "value"}:
            raise ValueError(f"{label} {kind} predicate has an unsupported schema")
        if not isinstance(predicate["path"], str) or not predicate["path"]:
            raise ValueError(f"{label} {kind}.path must be a non-empty string")
        return
    if kind in {"static_feature", "architecture_tag", "tensor_glob", "recognized_organ"}:
        if set(predicate) != {"kind", "value"}:
            raise ValueError(f"{label} {kind} predicate has an unsupported schema")
        if not isinstance(predicate["value"], str) or not predicate["value"]:
            raise ValueError(f"{label} {kind}.value must be a non-empty string")
        return
    if kind == "tensor_header_exact":
        if set(predicate) != {"kind", "name", "dtype", "shape"}:
            raise ValueError(f"{label} tensor_header_exact predicate has an unsupported schema")
        if not isinstance(predicate["name"], str) or not predicate["name"]:
            raise ValueError(f"{label} tensor_header_exact.name must be a non-empty string")
        if not isinstance(predicate["dtype"], str) or not predicate["dtype"]:
            raise ValueError(f"{label} tensor_header_exact.dtype must be a non-empty string")
        shape = predicate["shape"]
        if (
            not isinstance(shape, list)
            or any(isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0 for dimension in shape)
        ):
            raise ValueError(f"{label} tensor_header_exact.shape must be non-negative integer dimensions")
        return
    raise ValueError(f"{label} has an unrecognized predicate kind: {kind!r}")


def _law_ids(document: dict[str, Any]) -> set[str]:
    result: set[str] = set()
    for method in document["methods"]:
        for law in method.get("laws") or []:
            if isinstance(law, dict) and isinstance(law.get("id"), str):
                result.add(law["id"])
    return result


def validate_passports(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate the registry-owned passport catalog before it can guide a query.

    Passports are compact links to source facts, controls, methods, and native
    owners.  They are intentionally not an execution engine or a second
    accounting system: ``reuse_method`` below remains the only automatic
    mutation/execution path.
    """
    catalog = document.get("organ_passports")
    if not isinstance(catalog, dict) or catalog.get("schema") != PASSPORT_CATALOG_SCHEMA:
        raise ValueError("canonical registry has no supported organ-passport catalog")
    if not isinstance(catalog.get("policy"), dict):
        raise ValueError("organ-passport catalog has no policy")
    entries = catalog.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("organ-passport catalog has no entries")

    method_ids = {str(method["id"]) for method in document["methods"]}
    law_ids = _law_ids(document)
    atlas = json.loads(SCAR_ATLAS.read_text())
    if atlas.get("schema") != "hawking.foundry.negative_transfer_atlas.v1":
        raise ValueError("unsupported negative-transfer atlas schema")
    atlas_ids = set((atlas.get("entries") or {}).keys())
    seen: set[str] = set()
    required_fields = {
        "id",
        "schema",
        "semantic_role",
        "recognition",
        "trait_projection",
        "tensor_contract",
        "state_contract",
        "source_accounting",
        "invariants",
        "approximation_contract",
        "method_ids",
        "law_ids",
        "scar_ids",
        "scoped_scars",
        "reference_controls",
        "reproduction_bundle",
        "native_primitives",
        "bottlenecks",
        "evidence_refs",
        "claim_boundary",
        "reopen_if",
    }
    for index, passport in enumerate(entries):
        label = f"organ_passports.entries[{index}]"
        if not isinstance(passport, dict) or set(passport) != required_fields:
            actual = sorted(passport) if isinstance(passport, dict) else type(passport).__name__
            raise ValueError(f"{label} has an unsupported schema: {actual}")
        passport_id = passport.get("id")
        if not isinstance(passport_id, str) or not passport_id or passport_id in seen:
            raise ValueError(f"{label} id must be a unique non-empty string")
        seen.add(passport_id)
        if passport.get("schema") != PASSPORT_SCHEMA:
            raise ValueError(f"{label} has an unsupported passport schema")
        for field in ("semantic_role", "claim_boundary"):
            if not isinstance(passport.get(field), str) or not passport[field]:
                raise ValueError(f"{label}.{field} must be a non-empty string")
        _validate_predicate(passport["recognition"], f"{label}.recognition")

        projection = passport["trait_projection"]
        if not isinstance(projection, dict) or set(projection) != {
            "architecture_tags", "static_features", "facts"
        }:
            raise ValueError(f"{label}.trait_projection has an unsupported schema")
        _string_list(projection["architecture_tags"], f"{label}.trait_projection.architecture_tags")
        _string_list(projection["static_features"], f"{label}.trait_projection.static_features")
        facts = projection["facts"]
        if not isinstance(facts, list) or not facts:
            raise ValueError(f"{label}.trait_projection.facts must be non-empty")
        for fact_index, fact in enumerate(facts):
            fact_label = f"{label}.trait_projection.facts[{fact_index}]"
            if not isinstance(fact, dict) or set(fact) != {"id", "status", "value", "evidence"}:
                raise ValueError(f"{fact_label} has an unsupported schema")
            if not isinstance(fact["id"], str) or not fact["id"]:
                raise ValueError(f"{fact_label}.id must be non-empty")
            if fact["status"] not in {"OBSERVED_STATIC", "UNKNOWN"}:
                raise ValueError(f"{fact_label}.status is unsupported")
            if fact["status"] == "UNKNOWN" and fact["value"] is not None:
                raise ValueError(f"{fact_label} must use null for an unknown value")
            if fact["status"] == "OBSERVED_STATIC":
                _relative_regular_file(fact["evidence"], f"{fact_label}.evidence")
            elif fact["evidence"] is not None:
                raise ValueError(f"{fact_label} must not cite evidence for an unknown value")

        tensor_contract = passport["tensor_contract"]
        if not isinstance(tensor_contract, dict) or set(tensor_contract) != {
            "selectors", "shape_contract", "producer_consumer"
        }:
            raise ValueError(f"{label}.tensor_contract has an unsupported schema")
        _string_list(tensor_contract["selectors"], f"{label}.tensor_contract.selectors")
        for field in ("shape_contract", "producer_consumer"):
            if not isinstance(tensor_contract[field], str) or not tensor_contract[field]:
                raise ValueError(f"{label}.tensor_contract.{field} must be a non-empty string")
        for field in ("state_contract", "approximation_contract"):
            if not isinstance(passport[field], dict) or set(passport[field]) != {"status", "contract" if field == "state_contract" else "exact"}:
                raise ValueError(f"{label}.{field} has an unsupported schema")
        accounting = passport["source_accounting"]
        if not isinstance(accounting, dict) or not {
            "status", "active_bytes_per_token", "compute_per_token", "active_measurement_status"
        }.issubset(accounting):
            raise ValueError(f"{label}.source_accounting must expose source and active accounting status")
        active_status = accounting["active_measurement_status"]
        if active_status == "UNMEASURED":
            if accounting["active_bytes_per_token"] is not None or accounting["compute_per_token"] is not None:
                raise ValueError(f"{label}.source_accounting must not invent active bytes or compute")
        elif active_status == "OBSERVED_NATIVE":
            if any(
                isinstance(accounting[field], bool) or not isinstance(accounting[field], (int, float))
                for field in ("active_bytes_per_token", "compute_per_token")
            ):
                raise ValueError(f"{label}.source_accounting observed active metrics must be numeric")
            _relative_regular_file(accounting.get("active_evidence"), f"{label}.source_accounting.active_evidence")
        else:
            raise ValueError(f"{label}.source_accounting has an unadmitted active measurement status")
        for field in ("invariants", "bottlenecks", "evidence_refs", "reopen_if"):
            _string_list(passport[field], f"{label}.{field}")
        for evidence_index, evidence in enumerate(passport["evidence_refs"]):
            _relative_regular_file(evidence, f"{label}.evidence_refs[{evidence_index}]")
        for field, permitted in (("method_ids", method_ids), ("law_ids", law_ids)):
            values = _string_list(passport[field], f"{label}.{field}")
            missing = set(values) - permitted
            if missing:
                raise ValueError(f"{label}.{field} references unknown IDs: {sorted(missing)}")

        scars = passport["scoped_scars"]
        if not isinstance(scars, list):
            raise ValueError(f"{label}.scoped_scars must be a list")
        scoped_ids: set[str] = set()
        for scar_index, scar in enumerate(scars):
            scar_label = f"{label}.scoped_scars[{scar_index}]"
            if not isinstance(scar, dict) or set(scar) != SCOPED_SCAR_FIELDS:
                raise ValueError(f"{scar_label} has an unsupported schema")
            if not isinstance(scar["id"], str) or not scar["id"] or scar["id"] in scoped_ids:
                raise ValueError(f"{scar_label}.id must be a unique non-empty string")
            scoped_ids.add(scar["id"])
            if not isinstance(scar["scope"], dict) or not scar["scope"] or not all(
                isinstance(key, str) and key and isinstance(value, str) and value
                for key, value in scar["scope"].items()
            ):
                raise ValueError(f"{scar_label}.scope must be a non-empty string map")
            missing_scope = SCOPED_SCAR_REQUIRED_SCOPE_KEYS - set(scar["scope"])
            if missing_scope:
                raise ValueError(
                    f"{scar_label}.scope omits required exact-scope keys: {sorted(missing_scope)}"
                )
            _relative_regular_file(scar["evidence"], f"{scar_label}.evidence")
            evidence_sha256 = scar["evidence_sha256"]
            if (
                not isinstance(evidence_sha256, str)
                or len(evidence_sha256) != 64
                or any(character not in "0123456789abcdef" for character in evidence_sha256)
            ):
                raise ValueError(f"{scar_label}.evidence_sha256 must be a lowercase SHA-256")
            if evidence_sha256 != _sha256(ROOT / Path(scar["evidence"])):
                raise ValueError(f"{scar_label}.evidence_sha256 does not bind its evidence file")
            for field in ("verdict", "does_not_rule_out", "reopen_if"):
                if not isinstance(scar[field], str) or not scar[field]:
                    raise ValueError(f"{scar_label}.{field} must be a non-empty string")
        scar_ids = _string_list(passport["scar_ids"], f"{label}.scar_ids")
        missing_scars = set(scar_ids) - (scoped_ids | atlas_ids)
        if missing_scars:
            raise ValueError(f"{label}.scar_ids references unknown IDs: {sorted(missing_scars)}")

        controls = passport["reference_controls"]
        if not isinstance(controls, list) or not controls:
            raise ValueError(f"{label}.reference_controls must be non-empty")
        control_ids: set[str] = set()
        for control_index, control in enumerate(controls):
            control_label = f"{label}.reference_controls[{control_index}]"
            if not isinstance(control, dict) or set(control) != {"id", "evidence", "scope"}:
                raise ValueError(f"{control_label} has an unsupported schema")
            if not isinstance(control["id"], str) or not control["id"] or control["id"] in control_ids:
                raise ValueError(f"{control_label}.id must be unique and non-empty")
            control_ids.add(control["id"])
            _relative_regular_file(control["evidence"], f"{control_label}.evidence")
            if not isinstance(control["scope"], str) or not control["scope"]:
                raise ValueError(f"{control_label}.scope must be a non-empty string")

        bundle = passport["reproduction_bundle"]
        if not isinstance(bundle, dict) or set(bundle) != {"entrypoint", "receipt", "scope"}:
            raise ValueError(f"{label}.reproduction_bundle has an unsupported schema")
        _relative_regular_file(bundle["entrypoint"], f"{label}.reproduction_bundle.entrypoint")
        _relative_regular_file(bundle["receipt"], f"{label}.reproduction_bundle.receipt")
        if not isinstance(bundle["scope"], str) or not bundle["scope"]:
            raise ValueError(f"{label}.reproduction_bundle.scope must be a non-empty string")

        primitives = passport["native_primitives"]
        if not isinstance(primitives, list) or not primitives:
            raise ValueError(f"{label}.native_primitives must be non-empty")
        for primitive_index, primitive in enumerate(primitives):
            primitive_label = f"{label}.native_primitives[{primitive_index}]"
            if not isinstance(primitive, dict) or set(primitive) != {"file", "symbol", "scope"}:
                raise ValueError(f"{primitive_label} has an unsupported schema")
            _relative_regular_file(primitive["file"], f"{primitive_label}.file")
            if not all(isinstance(primitive[field], str) and primitive[field] for field in ("symbol", "scope")):
                raise ValueError(f"{primitive_label} must name a symbol and scope")
    return entries


def _value_at_path(document: dict[str, Any], path: str) -> Any:
    current: Any = document
    for component in path.split("."):
        if not isinstance(current, dict) or component not in current:
            return None
        current = current[component]
    return current


def _validate_selected_tensor_headers(
    tensor_headers: Any,
    *,
    tensor_names: list[str],
    identity: dict[str, Any],
) -> None:
    """Bind a shared validated witness to this trait projection's identity.

    ModelLake owns the witness schema/caps/descriptor rules. Foundry owns the
    additional relation between that validated witness and a passport's static
    source identity.
    """
    selected = validate_selected_tensor_header_witness(
        tensor_headers, allowed_tensor_names=tensor_names
    )
    expected_manifest = _canonical_json_sha256(selected)
    if identity.get("tensor_header_manifest_sha256") != expected_manifest:
        raise ValueError("selected tensor headers do not match their static identity")
    if identity.get("tensor_header_manifest_identity_algorithm") != "canonical-json-utf8-sha256.v1":
        raise ValueError("selected tensor headers have an unsupported identity algorithm")
    for field, value in (
        ("tensor_header_index_name", selected["index_name"]),
        ("tensor_header_index_sha256", selected["index_sha256"]),
        ("tensor_header_source_classification", selected["source_classification"]),
    ):
        if identity.get(field) != value:
            raise ValueError(f"selected tensor headers do not match static identity {field}")


def _validate_static_traits(static_traits: Any) -> dict[str, Any]:
    if not isinstance(static_traits, dict) or static_traits.get("schema") != STATIC_TRAITS_SCHEMA:
        raise ValueError("unsupported static organ-traits schema")
    identity = static_traits.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("static organ traits have no identity")
    for field in ("artifact_id", "revision", "config_sha256", "tensor_manifest_sha256"):
        if not isinstance(identity.get(field), str) or not identity[field]:
            raise ValueError(f"static organ traits identity lacks {field}")
    if not isinstance(static_traits.get("config"), dict):
        raise ValueError("static organ traits have no configuration object")
    tensor_names = _string_list(static_traits.get("tensor_names"), "static organ traits tensor_names")
    if len(set(tensor_names)) != len(tensor_names):
        raise ValueError("static organ traits tensor_names must not duplicate a header name")
    _string_list(static_traits.get("architecture_tags"), "static organ traits architecture_tags")
    _string_list(static_traits.get("features"), "static organ traits features")
    organs = static_traits.get("organs")
    if not isinstance(organs, list) or not all(isinstance(organ, dict) for organ in organs):
        raise ValueError("static organ traits organs must be an object list")
    if static_traits.get("static_only") is not True or static_traits.get("loaded_weights") is not False:
        raise ValueError("static organ traits must remain header/config-only")
    expected_config_hash = hashlib.sha256(
        json.dumps(
            static_traits["config"], sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()
    expected_tensor_manifest_hash = hashlib.sha256(
        "\n".join(sorted(static_traits["tensor_names"])).encode("utf-8")
    ).hexdigest()
    if identity["config_sha256"] != expected_config_hash:
        raise ValueError("static organ-traits config identity does not match its configuration")
    if identity["tensor_manifest_sha256"] != expected_tensor_manifest_hash:
        raise ValueError("static organ-traits tensor identity does not match its header names")
    tensor_headers = static_traits.get("selected_tensor_headers")
    if tensor_headers is None:
        if (
            "tensor_header_manifest_sha256" in identity
            or "tensor_header_manifest_identity_algorithm" in identity
            or "tensor_header_index_name" in identity
            or "tensor_header_index_sha256" in identity
            or "tensor_header_source_classification" in identity
        ):
            raise ValueError("static organ traits claim selected headers without a witness")
    else:
        _validate_selected_tensor_headers(
            tensor_headers, tensor_names=tensor_names, identity=identity
        )
    return static_traits


def _predicate_matches(predicate: dict[str, Any], static_traits: dict[str, Any]) -> bool:
    if "all" in predicate:
        return all(_predicate_matches(child, static_traits) for child in predicate["all"])
    if "any" in predicate:
        return any(_predicate_matches(child, static_traits) for child in predicate["any"])
    if "not" in predicate:
        return not any(_predicate_matches(child, static_traits) for child in predicate["not"])
    kind = predicate["kind"]
    if kind == "config_equals":
        return _value_at_path(static_traits["config"], predicate["path"]) == predicate["value"]
    if kind == "config_contains":
        value = _value_at_path(static_traits["config"], predicate["path"])
        return isinstance(value, list) and predicate["value"] in value
    if kind == "identity_equals":
        return _value_at_path(static_traits["identity"], predicate["path"]) == predicate["value"]
    if kind == "static_feature":
        return predicate["value"] in static_traits["features"]
    if kind == "architecture_tag":
        return predicate["value"] in static_traits["architecture_tags"]
    if kind == "tensor_glob":
        return any(fnmatchcase(name, predicate["value"]) for name in static_traits["tensor_names"])
    if kind == "tensor_header_exact":
        witness = static_traits.get("selected_tensor_headers") or {}
        return any(
            descriptor["name"] == predicate["name"]
            and descriptor["dtype"] == predicate["dtype"]
            and descriptor["shape"] == predicate["shape"]
            for descriptor in witness.get("headers", [])
        )
    if kind == "recognized_organ":
        return any(organ.get("organ") == predicate["value"] for organ in static_traits["organs"])
    raise ValueError(f"validated predicate reached unsupported kind {kind!r}")


def _identity_predicates(predicate: dict[str, Any]) -> list[dict[str, Any]]:
    if "all" in predicate or "any" in predicate or "not" in predicate:
        result: list[dict[str, Any]] = []
        for children in (predicate.get("all", []), predicate.get("any", []), predicate.get("not", [])):
            for child in children:
                result.extend(_identity_predicates(child))
        return result
    return [predicate] if predicate.get("kind") == "identity_equals" else []


def _scoped_passport_scars(passport: dict[str, Any], scope: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        scar for scar in passport["scoped_scars"]
        if all(scope.get(key) == value for key, value in scar["scope"].items())
    ]


def assess_passports(
    observation: dict[str, Any],
    static_traits: dict[str, Any],
    *,
    evidence_handles: set[str] | None = None,
    candidate_scope: dict[str, str] | None = None,
    path: Path = REGISTRY,
) -> dict[str, Any]:
    """Project static traits through one passport owner, then reuse method selection.

    Dynamic source/output/state evidence is never synthesized from config or
    header names.  A recognized passport without its named controls is held,
    and only ``reuse_method`` may subsequently execute an admitted adapter.
    """
    static_traits = _validate_static_traits(static_traits)
    decisions = assess_methods(observation, path)
    registry_document = load_registry(path)
    passports = validate_passports(registry_document)
    handles = set(evidence_handles or set())
    if not all(isinstance(handle, str) and handle for handle in handles):
        raise ValueError("evidence handles must be non-empty strings")
    if candidate_scope is not None and not all(
        isinstance(key, str) and key and isinstance(value, str) and value
        for key, value in candidate_scope.items()
    ):
        raise ValueError("candidate scope must be a string map")
    if candidate_scope and candidate_scope.get("source_revision") not in {
        None, static_traits["identity"]["revision"]
    }:
        raise ValueError("candidate scope cannot override the static source revision")
    scope = dict(candidate_scope or {})
    scope["source_revision"] = static_traits["identity"]["revision"]
    decisions_by_id = {decision["method"]["id"]: decision for decision in decisions}
    rows: list[dict[str, Any]] = []
    for passport in passports:
        recognized = _predicate_matches(passport["recognition"], static_traits)
        selected_header_present = static_traits.get("selected_tensor_headers") is not None
        identity_matches = all(
            _predicate_matches(predicate, static_traits)
            for predicate in _identity_predicates(passport["recognition"])
            # A header-bound identity is meaningful only once the optional
            # target-only witness exists. Without it, the exact-shape
            # predicate remains a static mismatch rather than pretending the
            # absent witness identifies a different source.
            if selected_header_present
            or not str(predicate.get("path", "")).startswith("tensor_header_")
        )
        static_tags = set(passport["trait_projection"]["architecture_tags"])
        static_features = set(passport["trait_projection"]["static_features"])
        projected = static_tags.issubset(static_traits["architecture_tags"]) and static_features.issubset(
            static_traits["features"]
        )
        method_decisions = [decisions_by_id[method_id] for method_id in passport["method_ids"]]
        missing_controls = sorted(
            control["id"] for control in passport["reference_controls"] if control["id"] not in handles
        )
        matched_scars = _scoped_passport_scars(passport, scope)
        if not identity_matches:
            status = "WITHHELD_SOURCE_IDENTITY_MISMATCH"
        elif not recognized:
            status = "WITHHELD_STATIC_TRAIT_MISMATCH"
        elif not projected:
            status = "WITHHELD_STATIC_TRAIT_MISMATCH"
        elif matched_scars and missing_controls:
            # The historical negative is useful, but a caller without the
            # passport's named controls must not have its own candidate
            # control silently treated as verified.
            status = "SCAR_RECORDED__WITHHELD_MISSING_CONTROL"
        elif matched_scars:
            status = "SCAR_SKIP"
        elif missing_controls:
            status = "WITHHELD_MISSING_CONTROL"
        elif any(decision["applicable"] for decision in method_decisions):
            status = "APPLY"
        else:
            status = "WITHHELD_MISSING_CONTROL"
        rows.append(
            {
                "id": passport["id"],
                "recognized": recognized,
                "source_identity_verified": identity_matches,
                "static_projection_verified": projected,
                "status": status,
                "method_assessment": [
                    {
                        "id": decision["method"]["id"],
                        "applicable": decision["applicable"],
                        "reasons": decision["reasons"],
                    }
                    for decision in method_decisions
                ],
                "missing_reference_controls": missing_controls,
                "matched_scars": [
                    {
                        "id": scar["id"],
                        "verdict": scar["verdict"],
                        "does_not_rule_out": scar["does_not_rule_out"],
                        "reopen_if": scar["reopen_if"],
                    }
                    for scar in matched_scars
                ],
                "claim_boundary": passport["claim_boundary"],
            }
        )
    return {
        "schema": PASSPORT_ASSESSMENT_SCHEMA,
        "static_identity": dict(static_traits["identity"]),
        "static_only": True,
        "execution_authority": "reuse_method only after an admitted method assessment",
        "passport_assessments": rows,
    }


def validate_execution_frontier(
    path: Path = REGISTRY,
    *,
    e13_preflight_path: Path | None = None,
    e13_refusal_path: Path | None = None,
    e13_universe_path: Path | None = None,
    e06_prefilter_path: Path | None = None,
) -> dict[str, Any]:
    """Fail closed when the canonical frontier contradicts sealed E13 controls.

    This reads only the registry and the existing sealed vocabulary preflight;
    it never opens a specimen or source payload. The dated execution-frontier
    checkpoint remains historical evidence, while this function validates the
    live registry that selects current work.
    """
    document = load_registry(path)
    frontier = document.get("execution_frontier")
    if not isinstance(frontier, dict):
        raise ValueError("canonical registry has no execution frontier")
    e13_entries: list[tuple[str, dict[str, Any]]] = []
    for section, entries in frontier.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, dict) and entry.get("id") == "E13":
                e13_entries.append((section, entry))
    if len(e13_entries) != 1:
        raise ValueError("canonical execution frontier must contain exactly one E13 entry")
    section, entry = e13_entries[0]
    if section != "withheld_without_native_bank":
        raise ValueError("E13 must remain withheld, not ready, until its named controls exist")
    if entry.get("state") != E13_WITHHELD_STATE:
        raise ValueError("E13 registry state does not match the required withheld control state")
    evidence = entry.get("evidence")
    if (
        not isinstance(evidence, list)
        or str(E13_PREFLIGHT_REL) not in evidence
        or str(E13_FIRST_REFUSAL_REL) not in evidence
        or str(E13_UNIVERSE_REL) not in evidence
        or str(E13_REFUSAL_REL) not in evidence
    ):
        raise ValueError(
            "E13 registry must cite its sealed preflight, both corpus refusals, and universe census"
        )

    preflight = e13_preflight_path or ROOT / E13_PREFLIGHT_REL
    try:
        from hawking.agentos.flash_vocabulary_prefilter import (
            verify_flash_vocabulary_prefilter_receipt_path,
        )

        receipt = verify_flash_vocabulary_prefilter_receipt_path(preflight)
    except (ImportError, OSError, ValueError) as exc:
        raise ValueError(f"cannot independently verify E13 vocabulary preflight: {exc}") from exc
    if receipt.get("status") != "WITHHELD_MISSING_VOCABULARY_CONTROL_PLAN":
        raise ValueError("sealed E13 vocabulary preflight no longer supports its no-plan boundary")

    first_refusal_path = ROOT / E13_FIRST_REFUSAL_REL
    first_refusal, first_refusal_seal = _load_sealed_json(
        first_refusal_path, "E13 first fixed-corpus refusal"
    )
    first_failing = (first_refusal.get("execution") or {}).get(
        "first_underpowered_stratum"
    ) or {}
    if (
        first_refusal.get("status") != "WITHHELD_FIXED_WIKINEWS_CONTROL_UNDERPOWERED"
        or first_failing.get("id") != "rare-han-short"
        or first_failing.get("eligible_rows") != 166
        or first_failing.get("required_rows") != 512
    ):
        raise ValueError("sealed E13 first refusal no longer supports its boundary")

    universe_path = e13_universe_path or ROOT / E13_UNIVERSE_REL
    universe, universe_seal = _load_sealed_json(universe_path, "E13 lexical universe census")
    cells = {
        cell.get("id"): cell
        for cell in universe.get("cells", [])
        if isinstance(cell, dict)
    }
    if (
        universe.get("status") != "STRUCTURALLY_CAPABLE__CORPUS_FREQUENCY_UNPROVEN"
        or universe.get("all_script_byte_cells_structurally_possible") is not True
        or (cells.get("han-short") or {}).get("eligible_rows") != 5426
        or (universe.get("runtime_boundary") or {}).get("corpus_opened") is not False
    ):
        raise ValueError("sealed E13 universe census no longer supports its structural boundary")

    refusal_path = e13_refusal_path or ROOT / E13_REFUSAL_REL
    refusal, observed_refusal_seal = _load_sealed_json(
        refusal_path, "E13 broader fixed-corpus refusal"
    )
    failing = (refusal.get("execution") or {}).get("first_underpowered_stratum") or {}
    boundary = refusal.get("runtime_boundary") or {}
    if (
        refusal.get("schema") != "hawking.flash_e13.fixed_corpus_refusal.v1"
        or refusal.get("status") != E13_WITHHELD_STATE
        or failing.get("id") != "rare-latin-short"
        or failing.get("eligible_rows") != 330
        or failing.get("required_rows") != 512
        or (refusal.get("execution") or {}).get("evidence_directory_committed") is not False
        or boundary.get("tensor_payload_bytes_read") != 0
        or boundary.get("row_extraction_performed") is not False
        or boundary.get("model_loaded") is not False
    ):
        raise ValueError("sealed E13 refusal no longer supports the fixed-corpus boundary")

    e06_entries: list[tuple[str, dict[str, Any]]] = []
    for frontier_section, entries in frontier.items():
        if not isinstance(entries, list):
            continue
        for frontier_entry in entries:
            if isinstance(frontier_entry, dict) and frontier_entry.get("id") == "E06":
                e06_entries.append((frontier_section, frontier_entry))
    if len(e06_entries) != 1:
        raise ValueError("canonical execution frontier must contain exactly one E06 entry")
    e06_section, e06_entry = e06_entries[0]
    if e06_section != "sealed_now" or e06_entry.get("state") != E06_SEALED_STATE:
        raise ValueError("E06 must remain sealed at its measured prefilter scope")
    e06_evidence = e06_entry.get("evidence")
    if not isinstance(e06_evidence, list) or str(E06_PREFILTER_REL) not in e06_evidence:
        raise ValueError("E06 registry must cite its sealed conditional-information prefilter")
    e06_prefilter = e06_prefilter_path or ROOT / E06_PREFILTER_REL
    try:
        from tools.odyssey.flash_e06_conditional_information_prefilter import (
            verify_prefilter_receipt,
        )

        e06_receipt = verify_prefilter_receipt(
            json.loads(e06_prefilter.read_text(encoding="utf-8"))
        )
    except (ImportError, OSError, ValueError) as exc:
        raise ValueError(f"cannot independently verify E06 prefilter receipt: {exc}") from exc
    if e06_receipt.get("status") != (
        "E06_PREFILTER_STOP_AUXILIARY_COST_OR_NULL_CONTROL_NOT_CLEARED"
    ) or (e06_receipt.get("escalation_signal") or {}).get("candidate_for_separate_review") is not False:
        raise ValueError("sealed E06 prefilter no longer supports its scoped negative disposition")

    e02_entries: list[tuple[str, dict[str, Any]]] = []
    for frontier_section, entries in frontier.items():
        if not isinstance(entries, list):
            continue
        for frontier_entry in entries:
            if isinstance(frontier_entry, dict) and frontier_entry.get("id") == "E02":
                e02_entries.append((frontier_section, frontier_entry))
    if len(e02_entries) != 1:
        raise ValueError("canonical execution frontier must contain exactly one E02 entry")
    e02_section, e02_entry = e02_entries[0]
    if e02_section != "sealed_now" or e02_entry.get("state") != E02_SEALED_STATE:
        raise ValueError("E02 fixed top-eight replication must remain sealed at its measured scope")
    e02_evidence = e02_entry.get("evidence")
    if not isinstance(e02_evidence, list) or str(E02_REPLICATION_REL) not in e02_evidence:
        raise ValueError("E02 registry must cite its sealed top-eight replication")
    e02_replication = ROOT / E02_REPLICATION_REL
    try:
        from tools.odyssey.expert_atom_sharing_screen import verify_replication_receipt

        e02_receipt = verify_replication_receipt(
            json.loads(e02_replication.read_text(encoding="utf-8"))
        )
    except (ImportError, OSError, ValueError) as exc:
        raise ValueError(f"cannot independently verify E02 replication receipt: {exc}") from exc
    replication = e02_receipt.get("replication") or {}
    if (
        replication.get("status") != "NOT_REPLICATED_ACROSS_PREDECLARED_TOP8_ROUTE_BANK"
        or replication.get("replicated") is not False
    ):
        raise ValueError("sealed E02 receipt no longer supports its scoped replication negative")
    vocabulary_geometry = (receipt.get("source_identity") or {}).get("vocabulary_geometry") or {}
    e06_information = e06_receipt.get("conditional_information") or {}
    e06_accounting = e06_receipt.get("projected_complete_code_accounting") or {}
    e06_signal = e06_receipt.get("escalation_signal") or {}
    return {
        "registry": str(path),
        "e13_section": section,
        "e13_state": entry["state"],
        "e13_preflight": str(preflight),
        "e13_preflight_seal_sha256": receipt["seal_sha256"],
        "e13_refusal": str(refusal_path),
        "e13_refusal_seal_sha256": observed_refusal_seal,
        "e13_first_refusal": str(first_refusal_path),
        "e13_first_refusal_seal_sha256": first_refusal_seal,
        "e13_lexical_universe": str(universe_path),
        "e13_lexical_universe_seal_sha256": universe_seal,
        "e13_han_short_structural_rows": cells["han-short"]["eligible_rows"],
        "e13_first_underpowered_stratum": failing["id"],
        "e13_first_underpowered_eligible_rows": failing["eligible_rows"],
        "e13_first_underpowered_required_rows": failing["required_rows"],
        "e13_tensor_row_count": vocabulary_geometry.get("tensor_row_count"),
        "e13_tokenizer_addressable_count": vocabulary_geometry.get("addressable_token_count"),
        "e13_lexical_eligible_count": vocabulary_geometry.get(
            "e13_eligible_model_vocab_token_count"
        ),
        "e13_tensor_only_tail_count": vocabulary_geometry.get(
            "tensor_rows_without_tokenizer_surface"
        ),
        "e13_row_extraction_performed": receipt.get("row_extraction_performed"),
        "e13_tensor_payload_bytes_read": receipt.get("tensor_payload_bytes_read"),
        "e06_section": e06_section,
        "e06_state": e06_entry["state"],
        "e06_prefilter": str(e06_prefilter),
        "e06_prefilter_seal_sha256": e06_receipt["seal_sha256"],
        "e06_conditional_saving_percent": e06_information.get(
            "conditional_code_saving_percent_of_global"
        ),
        "e06_conditional_net_saving_bits_after_auxiliary_cost": e06_accounting.get(
            "conditional_net_saving_bits_vs_global"
        ),
        "e06_candidate_for_separate_review": e06_signal.get(
            "candidate_for_separate_review"
        ),
        "e02_section": e02_section,
        "e02_state": e02_entry["state"],
        "e02_replication": str(e02_replication),
        "e02_replication_seal_sha256": e02_receipt["seal_sha256"],
        "e02_original_calibration_pair_reproduced": replication.get(
            "original_calibration_pair_reproduced"
        ),
        "e02_independent_replication_pair_count": replication.get(
            "independent_replication_pair_count"
        ),
        "e02_independent_survivor_count": replication.get(
            "independent_survivor_count"
        ),
        "e02_replicated": replication.get("replicated"),
        "valid": True,
    }


def select_methods(tags: set[str], path: Path = REGISTRY) -> list[dict[str, Any]]:
    """Return methods whose required architecture tags are a subset of *tags*."""
    selected = []
    for method in load_registry(path)["methods"]:
        required = set(method["applicability"]["architecture_tags"])
        if required.issubset(tags):
            selected.append(method)
    return selected


def assess_methods(observation: dict[str, Any], path: Path = REGISTRY) -> list[dict[str, Any]]:
    """Apply tags, required features, exclusions, and objective without guessing."""
    if observation.get("schema") != OBSERVATION_SCHEMA:
        raise ValueError(f"unsupported observation schema: {observation.get('schema')!r}")
    tags = set(observation.get("architecture_tags") or [])
    features = set(observation.get("features") or [])
    if not all(isinstance(value, str) and value for value in tags | features):
        raise ValueError("observation tags and features must be non-empty strings")
    objective = observation.get("objective_family")
    decisions = []
    for method in load_registry(path)["methods"]:
        applicability = method["applicability"]
        required_tags = set(applicability["architecture_tags"]) - {"any_model"}
        required_features = set(applicability["required_features"])
        excluded_present = set(applicability["excluded_features"]) & features
        missing_tags = required_tags - tags
        missing_features = required_features - features
        reasons = []
        if objective and method.get("family") != objective:
            reasons.append(f"objective_family={objective!r} does not match {method.get('family')!r}")
        if missing_tags:
            reasons.append("missing architecture tags: " + ", ".join(sorted(missing_tags)))
        if missing_features:
            reasons.append("missing required features: " + ", ".join(sorted(missing_features)))
        if excluded_present:
            reasons.append("excluded features present: " + ", ".join(sorted(excluded_present)))
        decisions.append({
            "method": method,
            "applicable": not reasons,
            "reasons": reasons,
        })
    return decisions


def matching_scars(observation: dict[str, Any], path: Path = SCAR_ATLAS) -> list[dict[str, Any]]:
    """Return named negative-transfer evidence explicitly observed by the caller."""
    atlas = json.loads(path.read_text())
    if atlas.get("schema") != "hawking.foundry.negative_transfer_atlas.v1":
        raise ValueError("unsupported negative-transfer atlas schema")
    entries = atlas.get("entries") or {}
    result = []
    for scar_id in observation.get("candidate_levers") or []:
        if scar_id in entries:
            result.append({"id": scar_id, **entries[scar_id]})
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path, *, display_path: str | None = None) -> FileIdentity:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular file: {path}")
    return FileIdentity(
        path=display_path or str(path),
        sha256=_sha256(path),
        bytes=path.stat().st_size,
    )


def _root_relative_file(root: Path, raw_path: Any, field: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"adapter {field} must be a non-empty relative path")
    declared = Path(raw_path)
    if declared.is_absolute():
        raise ValueError(f"adapter {field} must be relative to the repository root")
    candidate = (root / declared).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"adapter {field} escapes the repository root") from exc
    return candidate


def _parse_execution_adapter(adapter: dict[str, Any], root: Path) -> MethodExecutionAdapter:
    """Parse a closed adapter family; registry prose is never executable argv."""
    if not isinstance(adapter, dict):
        raise ValueError("automatic adapter is not an object")
    adapter_id = adapter["id"]
    if not isinstance(adapter_id, str) or not adapter_id:
        raise ValueError("automatic adapter has no stable id")
    tags = adapter["architecture_tags"]
    if not isinstance(tags, list) or not all(isinstance(tag, str) and tag for tag in tags):
        raise ValueError("automatic adapter architecture_tags must be non-empty strings")
    required_inputs = adapter["required_inputs"]
    if not isinstance(required_inputs, list) or not required_inputs:
        raise ValueError("automatic adapter required_inputs must be a non-empty list")
    if not all(isinstance(path, str) and path and not Path(path).is_absolute() for path in required_inputs):
        raise ValueError("automatic adapter required_inputs must be relative non-empty paths")
    input_paths = tuple(
        _root_relative_file(root, raw_path, f"required_inputs[{index}]")
        for index, raw_path in enumerate(required_inputs)
    )

    nr_keys = {"id", "architecture_tags", "required_inputs", "runner", "verifier"}
    if set(adapter) == nr_keys:
        runner = adapter["runner"]
        verifier = adapter["verifier"]
        if not isinstance(runner, dict) or not isinstance(verifier, dict):
            raise ValueError("automatic NR adapter runner and verifier must be typed objects")
        if set(runner) != {"kind", "path", "timeout_seconds"}:
            raise ValueError("automatic NR adapter runner has an unsupported schema")
        if set(verifier) != {"kind", "path", "symbol"}:
            raise ValueError("automatic NR adapter verifier has an unsupported schema")
        if runner.get("kind") != "flash_complete_nr.v1":
            raise ValueError("automatic NR adapter runner kind is not supported")
        if verifier.get("kind") != "nr_container_validate.v1":
            raise ValueError("automatic NR adapter verifier kind is not supported")
        if runner.get("path") != "tools/flash_complete_nr.py":
            raise ValueError("flash_complete_nr.v1 must use tools/flash_complete_nr.py")
        if verifier.get("path") != "tools/nr_container.py" or verifier.get("symbol") != "validate":
            raise ValueError("nr_container_validate.v1 must use tools/nr_container.py:validate")
        timeout_seconds = runner["timeout_seconds"]
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
            raise ValueError("automatic NR adapter timeout_seconds must be numeric")
        if not 0 < float(timeout_seconds) <= 3600:
            raise ValueError("automatic NR adapter timeout_seconds must be in (0, 3600]")
        return MethodExecutionAdapter(
            id=adapter_id,
            architecture_tags=frozenset(tags),
            required_inputs=tuple(required_inputs),
            required_input_paths=input_paths,
            runner_path=_root_relative_file(root, runner["path"], "runner.path"),
            runner_kind=runner["kind"],
            timeout_seconds=float(timeout_seconds),
            verifier_path=_root_relative_file(root, verifier["path"], "verifier.path"),
            verifier_kind=verifier["kind"],
            verifier_symbol=verifier["symbol"],
            argv_template=(),
            verification_spec={},
        )

    sealed_keys = {
        "id", "architecture_tags", "required_inputs", "runner", "argv", "verifier", "verification"
    }
    if set(adapter) != sealed_keys:
        raise ValueError(
            "automatic adapter has an unsupported schema: "
            f"fields={sorted(adapter)}"
        )
    runner = adapter["runner"]
    verifier = adapter["verifier"]
    template = adapter["argv"]
    specification = adapter["verification"]
    if not isinstance(runner, str) or not runner:
        raise ValueError("sealed-json adapter runner must be a non-empty relative path")
    if not isinstance(verifier, str) or not verifier:
        raise ValueError("sealed-json adapter verifier must be a non-empty label")
    if not isinstance(template, list) or not template or not all(
        isinstance(value, str) and value for value in template
    ):
        raise ValueError("sealed-json adapter argv must be a non-empty string list")
    allowed_placeholders = {"{artifact_path}", "{target_ebpw}", "{generation}"}
    if any(
        ("{" in value or "}" in value) and value not in allowed_placeholders
        for value in template
    ):
        raise ValueError("sealed-json adapter argv has an unsupported placeholder")
    if template.count("{artifact_path}") != 1:
        raise ValueError("sealed-json adapter argv must bind exactly one staging artifact path")
    if not isinstance(specification, dict) or set(specification) != {"kind", "schema", "status"}:
        raise ValueError("sealed-json adapter verification has an unsupported schema")
    expected_status = specification.get("status")
    statuses = [expected_status] if isinstance(expected_status, str) else expected_status
    if (
        specification.get("kind") != "sealed_json"
        or not isinstance(specification.get("schema"), str)
        or not specification["schema"]
        or not isinstance(statuses, list)
        or not statuses
        or not all(isinstance(value, str) and value for value in statuses)
    ):
        raise ValueError("sealed-json adapter verification is not a closed schema/status contract")
    verifier_spec = dict(specification)
    verifier_spec["label"] = verifier
    return MethodExecutionAdapter(
        id=adapter_id,
        architecture_tags=frozenset(tags),
        required_inputs=tuple(required_inputs),
        required_input_paths=input_paths,
        runner_path=_root_relative_file(root, runner, "runner"),
        runner_kind="sealed_json_runner.v1",
        timeout_seconds=SEALED_JSON_ADAPTER_TIMEOUT_SECONDS,
        verifier_path=Path(__file__).resolve(),
        verifier_kind="sealed_json.v1",
        verifier_symbol="_verify_sealed_json_artifact",
        argv_template=tuple(template),
        verification_spec=verifier_spec,
    )


def _tail(value: str | bytes | None, limit: int = 4000) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode(errors="replace")
    return value[-limit:]


def _load_nr_container_verifier(path: Path, symbol: str):
    spec = importlib.util.spec_from_file_location("hawking_gravity_nr_container", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load verifier module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    validate = getattr(module, symbol, None)
    if not callable(validate):
        raise RuntimeError(f"verifier module {path} has no callable {symbol}")
    seal_payload = getattr(module, "seal_payload_sha256_v1", None)
    if not callable(seal_payload):
        raise RuntimeError(f"verifier module {path} has no callable seal_payload_sha256_v1")
    return validate, seal_payload


def _verify_staged_output(
    adapter: MethodExecutionAdapter,
    staging_path: Path,
) -> tuple[bool, dict[str, Any]]:
    """Run the verifier bound by the parsed adapter, never by registry label."""
    implementation = _file_identity(
        adapter.verifier_path, display_path=_display_path(ROOT, adapter.verifier_path)
    )
    verification: dict[str, Any] = {
        "kind": adapter.verifier_kind,
        "implementation": implementation.as_dict(),
        "symbol": adapter.verifier_symbol,
        "executed": False,
        "scope": (
            "NR portability/container validation and legacy payload-seal integrity only; "
            "this is not a capability verifier"
        ),
    }
    try:
        artifact = json.loads(staging_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        verification.update({"status": "OUTPUT_NOT_VALID_JSON", "error": str(exc)})
        return False, verification
    if not isinstance(artifact, dict):
        verification.update({"status": "OUTPUT_NOT_OBJECT"})
        return False, verification
    verification["artifact_status"] = artifact.get("status")
    if adapter.verifier_kind == "sealed_json.v1":
        try:
            verified, sealed_report = _verify_sealed_json_artifact(artifact, adapter)
        except Exception as exc:
            verification.update({"status": "VERIFIER_ERROR", "executed": True, "error": str(exc)})
            return False, verification
        verification.update(sealed_report)
        verification.update({
            "status": "PASS" if verified else "FAIL",
            "executed": True,
            "scope": (
                "sealed source-bound receipt schema, declared status, and payload seal only; "
                "this is not a capability verifier"
            ),
        })
        return verified, verification
    if adapter.verifier_kind != "nr_container_validate.v1":
        verification.update({"status": "VERIFIER_KIND_NOT_ADMITTED"})
        return False, verification
    try:
        validate, seal_payload = _load_nr_container_verifier(
            adapter.verifier_path, adapter.verifier_symbol
        )
    except Exception as exc:  # loading is distinct from running the verifier symbol
        verification.update({"status": "VERIFIER_LOAD_ERROR", "error": str(exc)})
        return False, verification
    try:
        valid, problems = validate(artifact)
    except Exception as exc:  # verifier failure is an observed failed outcome
        verification.update({"status": "VERIFIER_ERROR", "executed": True, "error": str(exc)})
        return False, verification
    seal = artifact.get("seal_sha256")
    unsealed = dict(artifact)
    unsealed.pop("seal_sha256", None)
    try:
        expected_seal = seal_payload(unsealed)
    except Exception as exc:
        verification.update({"status": "SEAL_CHECK_ERROR", "executed": True, "error": str(exc)})
        return False, verification
    verified = bool(valid) and seal == expected_seal
    verification.update(
        {
            "status": "PASS" if verified else "FAIL",
            "executed": True,
            "valid": bool(valid),
            "problems": problems,
            "seal_matches": seal == expected_seal,
            "seal_algorithm": "python-json-sort-keys-default-separators-sha256.v1",
        }
    )
    return verified, verification


def _path_state(path: Path, *, display_path: str) -> dict[str, Any]:
    """Describe a destination without following a symlink or accepting a directory."""
    try:
        if path.is_symlink():
            return {"state": "SYMLINK", "path": display_path}
        if not path.exists():
            return {"state": "ABSENT", "path": display_path}
        if not path.is_file():
            return {"state": "NOT_REGULAR", "path": display_path}
        return {
            "state": "REGULAR",
            "path": display_path,
            "identity": _file_identity(path, display_path=display_path).as_dict(),
        }
    except OSError as exc:
        return {"state": "UNREADABLE", "path": display_path, "error": str(exc)}


def _same_file_content_identity(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Compare content identities while allowing promotion to change the path."""
    return left.get("sha256") == right.get("sha256") and left.get("bytes") == right.get("bytes")


def _remove_staging_file(path: Path) -> dict[str, str]:
    """Remove only the attempt-scoped staging leaf; never touch a final artifact."""
    try:
        if path.is_symlink() or path.is_file():
            path.unlink()
            return {"status": "REMOVED"}
        if path.exists():
            return {"status": "NOT_REMOVED_NONREGULAR"}
        return {"status": "NOT_PRESENT"}
    except OSError as exc:
        return {"status": "REMOVE_FAILED", "error": str(exc)}


def _write_receipt(path: Path, document: dict[str, Any]) -> dict[str, Any]:
    """Create a receipt atomically without ever replacing an existing one.

    ``Path.replace`` gives atomic visibility, but its overwrite behavior is
    wrong for historical evidence.  A same-directory hard link gives the
    needed no-replace create: it either installs the complete temporary file
    at the final leaf or refuses because another writer already owns it.
    """
    destination_display_path = str(path)
    result = {
        "mode": "ATOMIC_EXCLUSIVE_CREATE_NO_REPLACE",
        "written": False,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return {
            **result,
            "status": "NOT_WRITTEN_IO_ERROR",
            "error": str(exc),
            "destination_state": _path_state(path, display_path=destination_display_path),
        }

    temporary = path.with_name(f".{path.name}.gravity-{uuid4().hex}.receipt-staging")
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except OSError as exc:
        return {
            **result,
            "status": "NOT_WRITTEN_STAGING_CREATE_ERROR",
            "error": str(exc),
            "destination_state": _path_state(path, display_path=destination_display_path),
        }

    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(document, indent=2, sort_keys=True) + "\n")
        try:
            os.link(temporary, path)
        except FileExistsError:
            return {
                **result,
                "status": "NOT_WRITTEN_DESTINATION_OCCUPIED",
                "destination_state": _path_state(
                    path, display_path=destination_display_path
                ),
            }
        except OSError as exc:
            return {
                **result,
                "status": "NOT_WRITTEN_IO_ERROR",
                "error": str(exc),
                "destination_state": _path_state(
                    path, display_path=destination_display_path
                ),
            }
        return {
            "mode": "ATOMIC_EXCLUSIVE_CREATE_NO_REPLACE",
            "status": "WRITTEN_ATOMICALLY_NO_REPLACE",
            "written": True,
        }
    except OSError as exc:
        return {
            **result,
            "status": "NOT_WRITTEN_IO_ERROR",
            "error": str(exc),
            "destination_state": _path_state(path, display_path=destination_display_path),
        }
    finally:
        _remove_staging_file(temporary)


def _publish_staging_artifact(staging_path: Path, destination: Path) -> dict[str, Any]:
    """Publish one verified staging file without replacing a concurrent final."""
    try:
        os.link(staging_path, destination)
    except FileExistsError:
        return {"status": "NOT_PROMOTED_DESTINATION_OCCUPIED", "published": False}
    except OSError as exc:
        return {
            "status": "NOT_PROMOTED_IO_ERROR",
            "published": False,
            "error": str(exc),
        }
    return {"status": "PROMOTED_ATOMICALLY_NO_REPLACE", "published": True}


def _adapter_command(
    adapter: dict[str, Any], invocation: dict[str, Any], artifact_path: Path
) -> list[str]:
    """Build a declared adapter command without introducing a shell surface.

    Older adapters use the closed-NR accounting convention.  New reusable
    discriminators declare a tiny argv template in the registry so the generic
    method path can invoke and verify them without duplicating a bespoke
    launcher for each specimen.
    """
    target = str(invocation.get("target_ebpw") or "0.5")
    generation = str(invocation.get("generation") or "automatic-gravity-reuse-v1")
    runner = str(ROOT / adapter["runner"])
    template = adapter.get("argv")
    if template is None:
        return [
            sys.executable,
            runner,
            "--target-ebpw",
            target,
            "--generation",
            generation,
            "--out",
            str(artifact_path),
        ]
    if not isinstance(template, list) or not all(isinstance(value, str) for value in template):
        raise ValueError(f"adapter {adapter.get('id')!r} has an invalid argv template")
    substitutions = {
        "{artifact_path}": str(artifact_path),
        "{target_ebpw}": target,
        "{generation}": generation,
    }
    return [
        sys.executable,
        runner,
        *[substitutions.get(value, value) for value in template],
    ]


def _build_execution_command(
    adapter: MethodExecutionAdapter,
    invocation: Any,
    staging_path: Path,
) -> list[str]:
    """Render only a parsed adapter's closed argv contract into a child argv."""
    if not isinstance(invocation, dict):
        raise ValueError("observation invocation must be an object")

    def argument(name: str, default: str) -> str:
        value = invocation.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise ValueError(f"invocation {name!r} must be a scalar")
        rendered = str(value)
        if not rendered:
            raise ValueError(f"invocation {name!r} must not be empty")
        return rendered

    target = argument("target_ebpw", "0.5")
    generation = argument("generation", "automatic-gravity-reuse-v1")
    if adapter.runner_kind == "flash_complete_nr.v1":
        return [
            sys.executable,
            str(adapter.runner_path),
            "--target-ebpw", target,
            "--generation", generation,
            "--out", str(staging_path),
        ]
    if adapter.runner_kind == "sealed_json_runner.v1":
        substitutions = {
            "{artifact_path}": str(staging_path),
            "{target_ebpw}": target,
            "{generation}": generation,
        }
        return [
            sys.executable,
            str(adapter.runner_path),
            *[substitutions[value] if value in substitutions else value
              for value in adapter.argv_template],
        ]
    raise ValueError(f"automatic adapter runner kind is not admitted: {adapter.runner_kind}")


def _verify_sealed_json_artifact(
    artifact: dict[str, Any], adapter: MethodExecutionAdapter | dict[str, Any]
) -> tuple[bool, dict[str, Any]]:
    """Verify a self-sealed source-bound scientific receipt declaratively."""
    if isinstance(adapter, MethodExecutionAdapter):
        specification = adapter.verification_spec
        verifier_label = str(specification.get("label") or adapter.verifier_symbol)
    else:
        specification = adapter.get("verification")
        verifier_label = str(adapter.get("verifier") or "sealed JSON receipt")
    if not isinstance(specification, dict) or specification.get("kind") != "sealed_json":
        adapter_id = adapter.id if isinstance(adapter, MethodExecutionAdapter) else adapter.get("id")
        raise ValueError(f"adapter {adapter_id!r} has no sealed-json verifier")
    expected_schema = specification.get("schema")
    expected_status = specification.get("status")
    statuses = {expected_status} if isinstance(expected_status, str) else set(expected_status or [])
    seal = artifact.get("seal_sha256")
    unsealed = dict(artifact)
    unsealed.pop("seal_sha256", None)
    schema_matches = artifact.get("schema") == expected_schema
    status_matches = artifact.get("status") in statuses
    seal_candidates = {
        "python-json-sort-keys-default-separators-sha256.v1": hashlib.sha256(
            json.dumps(unsealed, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        # Source-bound Rust and newer Flash screen receipts use compact
        # serde-compatible JSON. This is a fixed historical verifier family,
        # not a caller-selected fallback or an arbitrary receipt algorithm.
        "python-json-sort-keys-compact-utf8-sha256.v1": hashlib.sha256(
            json.dumps(unsealed, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    matching_algorithms = [
        algorithm for algorithm, expected in seal_candidates.items() if seal == expected
    ]
    seal_matches = bool(matching_algorithms)
    verification = {
        "verifier": verifier_label,
        "schema_matches": schema_matches,
        "status_matches": status_matches,
        "seal_matches": seal_matches,
        "seal_algorithm": matching_algorithms[0] if len(matching_algorithms) == 1 else None,
    }
    return schema_matches and status_matches and seal_matches, verification


def reuse_method(
    observation: dict[str, Any],
    *,
    receipt_path: Path,
    artifact_path: Path,
    registry_path: Path = REGISTRY,
) -> dict[str, Any]:
    """Execute the sole admitted automatic method through a bounded contract.

    Public method discovery remains a separate, literal tag query.  This path
    consumes a richer observation, applies exclusions, binds the one typed
    adapter, and only publishes a fresh artifact after its declared verifier
    has run successfully.
    """
    root = ROOT.resolve()
    requested_receipt = Path(receipt_path)
    resolved_receipt = _resolve_execution_path(requested_receipt, root)
    requested_artifact = Path(artifact_path)
    resolved_artifact = _resolve_execution_path(requested_artifact, root)
    resolved_registry = _resolve_execution_path(Path(registry_path), root)

    decisions = assess_methods(observation, resolved_registry)
    applicable = [decision["method"] for decision in decisions if decision["applicable"]]
    tags = set(observation.get("architecture_tags") or [])
    runnable: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for method in applicable:
        for adapter in (method.get("automation") or {}).get("adapters", []):
            if isinstance(adapter, dict) and set(adapter.get("architecture_tags") or []).issubset(tags):
                runnable.append((method, adapter))
    base = {
        "schema": EXECUTION_SCHEMA,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "observation": observation,
        "observation_identity": {
            "algorithm": "canonical-json-sha256.v1",
            "sha256": _canonical_json_sha256(observation),
            "artifact_id": observation.get("artifact_id"),
            "architecture_tags": sorted(tags),
        },
        "method_assessment": [
            {
                "id": decision["method"]["id"],
                "applicable": decision["applicable"],
                "reasons": decision["reasons"],
            }
            for decision in decisions
        ],
        "scars": matching_scars(observation),
        "receipt_destination": {
            "requested_path": str(requested_receipt),
            "resolved_path": str(resolved_receipt),
        },
        "artifact_destination": {
            "requested_path": str(requested_artifact),
            "resolved_path": str(resolved_artifact),
        },
    }

    receipt_pre_run_state = _path_state(
        resolved_receipt, display_path=str(resolved_receipt)
    )
    artifact_pre_run_state = _path_state(
        resolved_artifact, display_path=str(resolved_artifact)
    )
    base["receipt_destination"]["pre_run_state"] = receipt_pre_run_state
    base["artifact_destination"]["pre_run_state"] = artifact_pre_run_state

    def non_written_result(
        status: str,
        *,
        refusal: str,
        invoked: bool,
        receipt_state: dict[str, Any],
        receipt_write_status: str,
        **updates: Any,
    ) -> dict[str, Any]:
        """Return a destination refusal without seizing another attempt's receipt."""
        base.update(updates)
        base.update(
            {
                "status": status,
                "refusal": refusal,
                "invoked": invoked,
                "receipt_write": {
                    "mode": "ATOMIC_EXCLUSIVE_CREATE_NO_REPLACE",
                    "status": receipt_write_status,
                    "written": False,
                    "destination_state": receipt_state,
                },
            }
        )
        return base

    def finish(status: str, **updates: Any) -> dict[str, Any]:
        base.update(updates)
        base["status"] = status
        # This exact value is also persisted in a successful receipt.  A
        # failed exclusive create has no receipt to amend, so the in-memory
        # result below is deliberately explicit about the non-write.
        base["receipt_write"] = {
            "mode": "ATOMIC_EXCLUSIVE_CREATE_NO_REPLACE",
            "status": "WRITTEN_ATOMICALLY_NO_REPLACE",
            "written": True,
        }
        receipt_write = _write_receipt(resolved_receipt, base)
        if receipt_write["written"]:
            return base
        intended_status = base["status"]
        base["intended_status"] = intended_status
        base["status"] = (
            "RECEIPT_WRITE_REFUSED_DESTINATION_OCCUPIED_NO_WRITE"
            if receipt_write["status"] == "NOT_WRITTEN_DESTINATION_OCCUPIED"
            else "RECEIPT_WRITE_FAILED_NO_WRITE"
        )
        base["receipt_write"] = receipt_write
        return base

    if resolved_receipt == resolved_artifact:
        return non_written_result(
            "REFUSED_OUTPUT_DESTINATION_COLLISION_NO_WRITE",
            refusal="evidence receipt and final artifact resolve to the same path",
            invoked=False,
            receipt_state=receipt_pre_run_state,
            receipt_write_status="NOT_WRITTEN_DESTINATION_COLLISION",
            artifact_promotion={
                "status": "NOT_ATTEMPTED_DESTINATION_COLLISION",
                "published": False,
            },
        )

    occupied_destinations = [
        ("evidence receipt", receipt_pre_run_state),
        ("final artifact", artifact_pre_run_state),
    ]
    occupied_destinations = [
        (label, state)
        for label, state in occupied_destinations
        if state["state"] != "ABSENT"
    ]
    if occupied_destinations:
        occupied_text = ", ".join(
            f"{label} ({state['state']})" for label, state in occupied_destinations
        )
        return non_written_result(
            "REFUSED_OUTPUT_DESTINATION_OCCUPIED_NO_WRITE",
            refusal=(
                "both final artifact and evidence receipt destinations must be absent; "
                f"occupied: {occupied_text}"
            ),
            invoked=False,
            receipt_state=receipt_pre_run_state,
            receipt_write_status="NOT_WRITTEN_DESTINATION_PRECHECK_FAILED",
            artifact_promotion={
                "status": "NOT_ATTEMPTED_DESTINATION_OCCUPIED",
                "published": False,
            },
        )

    if len(runnable) != 1:
        return finish(
            "REFUSED",
            refusal=(
                "no applicable method has a compatible automatic adapter"
                if not runnable else
                "more than one automatic adapter matched; selection is ambiguous"
            ),
            invoked=False,
        )

    method, raw_adapter = runnable[0]
    base.update({
        "selected_method": method["id"],
        "selected_adapter": raw_adapter.get("id"),
        "laws": method.get("laws") or [],
        "claim_boundary": method["claim_boundary"],
    })
    try:
        adapter = _parse_execution_adapter(raw_adapter, root)
    except ValueError as exc:
        return finish(
            "REFUSED",
            refusal=f"automatic adapter contract is not admitted: {exc}",
            invoked=False,
        )

    try:
        registry_identity = _file_identity(
            resolved_registry, display_path=_display_path(root, resolved_registry)
        )
        runner_identity = _file_identity(
            adapter.runner_path, display_path=_display_path(root, adapter.runner_path)
        )
        verifier_identity = _file_identity(
            adapter.verifier_path, display_path=_display_path(root, adapter.verifier_path)
        )
    except ValueError as exc:
        return finish(
            "REFUSED",
            refusal=f"automatic adapter source identity is unavailable: {exc}",
            invoked=False,
        )

    missing: list[str] = []
    input_identities: list[dict[str, Any]] = []
    for declared_path, input_path in zip(adapter.required_inputs, adapter.required_input_paths):
        try:
            input_identities.append(
                _file_identity(input_path, display_path=declared_path).as_dict()
            )
        except ValueError:
            missing.append(declared_path)
    if missing:
        return finish(
            "REFUSED",
            refusal="cheap falsifier found missing or non-regular inputs: " + ", ".join(missing),
            cheap_falsifier={
                "passed": False,
                "required_inputs": list(adapter.required_inputs),
                "available_input_identities": input_identities,
            },
            invoked=False,
        )

    base["cheap_falsifier"] = {
        "passed": True,
        "required_inputs": list(adapter.required_inputs),
        "input_identities": input_identities,
    }
    base["execution_contract"] = {
        "registry": registry_identity.as_dict(),
        "runner": {
            "kind": adapter.runner_kind,
            "implementation": runner_identity.as_dict(),
            "timeout_seconds": adapter.timeout_seconds,
        },
        "verifier": {
            "kind": adapter.verifier_kind,
            "implementation": verifier_identity.as_dict(),
            "symbol": adapter.verifier_symbol,
        },
    }

    receipt_pre_spawn_state = _path_state(
        resolved_receipt, display_path=str(resolved_receipt)
    )
    destination_pre_state = _path_state(
        resolved_artifact, display_path=str(resolved_artifact)
    )
    base["receipt_destination"]["pre_spawn_state"] = receipt_pre_spawn_state
    base["artifact_destination"]["pre_spawn_state"] = destination_pre_state
    occupied_before_spawn = [
        ("evidence receipt", receipt_pre_spawn_state),
        ("final artifact", destination_pre_state),
    ]
    occupied_before_spawn = [
        (label, state)
        for label, state in occupied_before_spawn
        if state["state"] != "ABSENT"
    ]
    if occupied_before_spawn:
        occupied_text = ", ".join(
            f"{label} ({state['state']})" for label, state in occupied_before_spawn
        )
        return non_written_result(
            "REFUSED_OUTPUT_DESTINATION_OCCUPIED_NO_WRITE",
            refusal=(
                "both final artifact and evidence receipt destinations must remain absent "
                f"until adapter spawn; occupied: {occupied_text}"
            ),
            invoked=False,
            receipt_state=receipt_pre_spawn_state,
            receipt_write_status="NOT_WRITTEN_DESTINATION_PRESPAWN_CHECK_FAILED",
            artifact_promotion={
                "status": "NOT_ATTEMPTED_DESTINATION_OCCUPIED",
                "published": False,
            },
        )

    try:
        resolved_artifact.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return finish(
            "REFUSED",
            refusal=f"could not create artifact destination directory: {exc}",
            invoked=False,
        )

    attempt_id = uuid4().hex
    staging_path = resolved_artifact.with_name(
        f".{resolved_artifact.name}.gravity-{attempt_id}.staging"
    )
    staging_display_path = str(staging_path)
    staging_pre_state = _path_state(staging_path, display_path=staging_display_path)
    base["attempt"] = {"id": attempt_id}
    base["staging"] = {
        "path": staging_display_path,
        "pre_run_state": staging_pre_state,
    }
    if staging_pre_state["state"] != "ABSENT":
        return finish(
            "REFUSED",
            refusal="attempt staging path was unexpectedly occupied",
            invoked=False,
        )

    invocation = observation.get("invocation") or {}
    try:
        command = _build_execution_command(adapter, invocation, staging_path)
    except ValueError as exc:
        return finish(
            "REFUSED",
            refusal=f"automatic adapter invocation is not admitted: {exc}",
            invoked=False,
        )
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started = time.monotonic()
    base["invocation"] = {
        "argv": command,
        "cwd": str(root),
        "started_at": started_at,
        "timeout_seconds": adapter.timeout_seconds,
    }

    def execution_details(**updates: Any) -> dict[str, Any]:
        details = dict(base["invocation"])
        details["duration_ms"] = round((time.monotonic() - started) * 1000, 3)
        details.update(updates)
        return details

    try:
        completed = subprocess.run(
            command,
            cwd=root,
            text=True,
            capture_output=True,
            timeout=adapter.timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        base["staging"]["cleanup"] = _remove_staging_file(staging_path)
        return finish(
            "EXECUTION_TIMED_OUT",
            invoked=True,
            invocation=execution_details(
                outcome="TIMED_OUT",
                stdout_tail=_tail(exc.output),
                stderr_tail=_tail(exc.stderr),
            ),
            artifact_destination={
                **base["artifact_destination"],
                "post_run_state": _path_state(
                    resolved_artifact, display_path=str(resolved_artifact)
                ),
            },
        )
    except OSError as exc:
        base["staging"]["cleanup"] = _remove_staging_file(staging_path)
        return finish(
            "EXECUTION_FAILED",
            invoked=True,
            invocation=execution_details(outcome="SPAWN_FAILED", error=str(exc)),
            artifact_destination={
                **base["artifact_destination"],
                "post_run_state": _path_state(
                    resolved_artifact, display_path=str(resolved_artifact)
                ),
            },
        )
    if completed.returncode != 0:
        base["staging"]["cleanup"] = _remove_staging_file(staging_path)
        return finish(
            "EXECUTION_FAILED",
            invoked=True,
            invocation=execution_details(
                outcome="RETURNED_NONZERO",
                returncode=completed.returncode,
                stdout_tail=_tail(getattr(completed, "stdout", None)),
                stderr_tail=_tail(getattr(completed, "stderr", None)),
            ),
            artifact_destination={
                **base["artifact_destination"],
                "post_run_state": _path_state(
                    resolved_artifact, display_path=str(resolved_artifact)
                ),
            },
        )

    stage_post_run = _path_state(staging_path, display_path=staging_display_path)
    base["staging"]["post_run_state"] = stage_post_run
    if stage_post_run["state"] != "REGULAR":
        base["staging"]["cleanup"] = _remove_staging_file(staging_path)
        return finish(
            "OUTPUT_MISSING",
            invoked=True,
            invocation=execution_details(
                outcome="RETURNED_ZERO_NO_REGULAR_STAGING_OUTPUT",
                returncode=completed.returncode,
                stdout_tail=_tail(getattr(completed, "stdout", None)),
                stderr_tail=_tail(getattr(completed, "stderr", None)),
            ),
            artifact_destination={
                **base["artifact_destination"],
                "post_run_state": _path_state(
                    resolved_artifact, display_path=str(resolved_artifact)
                ),
            },
        )

    staged_identity = stage_post_run["identity"]
    verified, verification = _verify_staged_output(adapter, staging_path)
    base["verification"] = verification
    if not verified:
        base["staging"]["cleanup"] = _remove_staging_file(staging_path)
        return finish(
            "VERIFICATION_FAILED",
            invoked=True,
            invocation=execution_details(
                outcome="RETURNED_ZERO_VERIFICATION_FAILED",
                returncode=completed.returncode,
                stdout_tail=_tail(getattr(completed, "stdout", None)),
                stderr_tail=_tail(getattr(completed, "stderr", None)),
            ),
            artifact_destination={
                **base["artifact_destination"],
                "post_verification_state": _path_state(
                    resolved_artifact, display_path=str(resolved_artifact)
                ),
            },
        )

    stage_before_promotion = _path_state(staging_path, display_path=staging_display_path)
    base["staging"]["pre_promotion_state"] = stage_before_promotion
    if (
        stage_before_promotion["state"] != "REGULAR"
        or stage_before_promotion["identity"] != staged_identity
    ):
        base["staging"]["cleanup"] = _remove_staging_file(staging_path)
        return finish(
            "VERIFICATION_FAILED",
            invoked=True,
            invocation=execution_details(
                outcome="STAGING_OUTPUT_CHANGED_AFTER_VERIFICATION",
                returncode=completed.returncode,
            ),
            artifact_destination={
                **base["artifact_destination"],
                "post_verification_state": _path_state(
                    resolved_artifact, display_path=str(resolved_artifact)
                ),
            },
        )

    receipt_before_promotion = _path_state(
        resolved_receipt, display_path=str(resolved_receipt)
    )
    destination_before_promotion = _path_state(
        resolved_artifact, display_path=str(resolved_artifact)
    )
    base["receipt_destination"]["pre_promotion_state"] = receipt_before_promotion
    base["artifact_destination"]["pre_promotion_state"] = destination_before_promotion
    if (
        receipt_before_promotion != receipt_pre_spawn_state
        or destination_before_promotion != destination_pre_state
    ):
        base["staging"]["cleanup"] = _remove_staging_file(staging_path)
        return non_written_result(
            "PROMOTION_REFUSED_DESTINATION_CHANGED_NO_WRITE",
            refusal=(
                "a final artifact or evidence receipt destination changed during execution; "
                "the verified staging output was not promoted"
            ),
            invoked=True,
            receipt_state=receipt_before_promotion,
            receipt_write_status="NOT_WRITTEN_DESTINATION_CHANGED_DURING_EXECUTION",
            invocation=execution_details(
                outcome="DESTINATION_CHANGED_DURING_EXECUTION_NO_REPLACE",
                returncode=completed.returncode,
            ),
            artifact_destination={
                **base["artifact_destination"],
                "post_run_state": destination_before_promotion,
            },
            artifact_promotion={
                "status": "NOT_PROMOTED_DESTINATION_CHANGED_DURING_EXECUTION",
                "published": False,
            },
        )

    promotion = _publish_staging_artifact(staging_path, resolved_artifact)
    if not promotion["published"]:
        base["staging"]["cleanup"] = _remove_staging_file(staging_path)
        occupied = promotion["status"] == "NOT_PROMOTED_DESTINATION_OCCUPIED"
        return non_written_result(
            (
                "PROMOTION_REFUSED_DESTINATION_OCCUPIED_NO_WRITE"
                if occupied
                else "PROMOTION_FAILED_NO_WRITE"
            ),
            refusal=(
                "atomic no-replace promotion found an occupied final artifact destination"
                if occupied
                else "atomic no-replace promotion could not publish the verified staging output"
            ),
            invoked=True,
            receipt_state=_path_state(resolved_receipt, display_path=str(resolved_receipt)),
            receipt_write_status=(
                "NOT_WRITTEN_DESTINATION_OCCUPIED_DURING_PROMOTION"
                if occupied
                else "NOT_WRITTEN_PROMOTION_IO_ERROR"
            ),
            invocation=execution_details(
                outcome=(
                    "ATOMIC_NO_REPLACE_DESTINATION_OCCUPIED"
                    if occupied
                    else "ATOMIC_NO_REPLACE_PROMOTION_FAILED"
                ),
                returncode=completed.returncode,
                **({"error": promotion["error"]} if "error" in promotion else {}),
            ),
            artifact_destination={
                **base["artifact_destination"],
                "post_promotion_state": _path_state(
                    resolved_artifact, display_path=str(resolved_artifact)
                ),
            },
            artifact_promotion=promotion,
        )

    published_state = _path_state(resolved_artifact, display_path=str(resolved_artifact))
    base["staging"]["cleanup"] = _remove_staging_file(staging_path)
    base["artifact_destination"]["post_promotion_state"] = published_state
    if (
        published_state["state"] != "REGULAR"
        or not _same_file_content_identity(published_state["identity"], staged_identity)
    ):
        return finish(
            "PROMOTION_FAILED",
            invoked=True,
            invocation=execution_details(
                outcome="PUBLISHED_IDENTITY_DOES_NOT_MATCH_VERIFIED_STAGING_OUTPUT",
                returncode=completed.returncode,
            ),
        )

    return finish(
        "EXECUTED_VERIFIED",
        invoked=True,
        invocation=execution_details(
            outcome="EXECUTED_VERIFIED_AND_PROMOTED",
            returncode=completed.returncode,
            stdout_tail=_tail(getattr(completed, "stdout", None)),
            stderr_tail=_tail(getattr(completed, "stderr", None)),
        ),
        promotion={"status": "PROMOTED_ATOMICALLY", "source_identity": staged_identity},
        new_evidence={
            "path": str(resolved_artifact),
            "identity": published_state["identity"],
            "status": verification.get("artifact_status"),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tags",
        required=False,
        help="comma-separated architecture/capability tags, e.g. moe,routed_experts",
    )
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--json", action="store_true", help="emit machine-readable selected entries")
    parser.add_argument(
        "--owners",
        action="store_true",
        help="emit the canonical implementation-owner map instead of selecting methods",
    )
    parser.add_argument(
        "--validate-execution-frontier",
        action="store_true",
        help="verify current E13 withholding against its sealed preflight without reading source payloads",
    )
    parser.add_argument("--reuse-observation", type=Path,
                        help="select and execute a compatible method from an observed-traits JSON")
    parser.add_argument("--out", type=Path,
                        help="method-execution evidence receipt (required with --reuse-observation)")
    parser.add_argument("--artifact-out", type=Path,
                        help="invoked method output (required with --reuse-observation)")
    args = parser.parse_args()
    if args.validate_execution_frontier:
        if args.tags or args.owners or args.reuse_observation or args.out or args.artifact_out:
            parser.error("--validate-execution-frontier cannot be combined with selection or reuse arguments")
        result = validate_execution_frontier(args.registry)
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print(
                f"E13\t{result['e13_state']}\t"
                f"sealed preflight {result['e13_preflight_seal_sha256']}"
            )
        return 0
    if args.reuse_observation:
        if not args.out or not args.artifact_out:
            parser.error("--out and --artifact-out are required with --reuse-observation")
        observation = json.loads(args.reuse_observation.read_text())
        result = reuse_method(
            observation,
            receipt_path=args.out,
            artifact_path=args.artifact_out,
            registry_path=args.registry,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] in {
            "EXECUTED_VERIFIED",
            "REFUSED",
            "REFUSED_OUTPUT_DESTINATION_COLLISION_NO_WRITE",
            "REFUSED_OUTPUT_DESTINATION_OCCUPIED_NO_WRITE",
        } else 1
    if not args.owners and not args.tags:
        parser.error("--tags is required unless --owners is selected")
    if args.owners:
        owners = load_registry(args.registry)["canonical_owners"]
        if args.json:
            print(json.dumps({"canonical_owners": owners}, indent=2, sort_keys=True))
        else:
            for scope, owner in owners.items():
                print(f"{scope}\t{owner}")
        return 0
    tags = {tag.strip() for tag in args.tags.split(",") if tag.strip()}
    selected = select_methods(tags, args.registry)
    if args.json:
        print(json.dumps({"tags": sorted(tags), "methods": selected}, indent=2, sort_keys=True))
        return 0
    for method in selected:
        print(f"{method['id']}\t{method['status']}\t{method['title']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
