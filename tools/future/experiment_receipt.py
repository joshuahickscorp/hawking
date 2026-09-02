"""Shared ArtifactIdentity + ExperimentReceipt emission contract.

Hawking has many receipt shapes and no shared identity. Path is a location;
identity is a fact (kind, producer, input seals, machine, commit, content
seal). ExperimentReceipt is the roadmap ResultEnvelope (H-ROADMAP §1.2)
plus evidence_tier and a falsifier. Producers keep their existing fields
and attach this envelope; historical receipts are not rewritten.

    python3 -m pytest tools/future/test_experiment_receipt.py -q -o addopts=""
"""
from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(
    0,
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))),
)

import hashlib
import json
import platform
from typing import Any, Iterable, Mapping

from tools.future._common import git

IDENTITY_SCHEMA = "hawking.artifact.identity.v1"
IDENTITY_VERSION = 1
RECEIPT_SCHEMA = "hawking.experiment.receipt.v1"
RECEIPT_VERSION = 1
RECORDED_BY = "tools/future/experiment_receipt.py"

# Roadmap §1.2 verdicts, plus UNVERIFIED from hcli.result_envelope.
VERDICTS = (
    "ACCEPT",
    "REJECT",
    "INCONCLUSIVE",
    "BLOCKED",
    "UNVERIFIED",
)

# Same ladder science_corpus already uses. Never merge tiers.
EVIDENCE_TIERS = (
    "STATIC",
    "FUNCTIONAL_SIM",
    "COST_MODEL",
    "CYCLE_APPROX",
    "HARDWARE_MEASURED",
)

# ResultEnvelope fields from H-ROADMAP.md §1.2. Keep this list aligned
# with the roadmap, not a competing envelope.
ROADMAP_ENVELOPE_FIELDS = (
    "verdict",
    "claim",
    "scope",
    "facts",
    "hypotheses",
    "evidence",
    "artifacts",
    "hashes",
    "tests",
    "controls",
    "negative_controls",
    "failures",
    "resource_usage",
    "qualification",
    "contamination",
    "uncertainty",
    "next_actions",
    "receipts",
)

# Extra emission fields the contract requires on top of the roadmap envelope.
CONTRACT_FIELDS = (
    "schema",
    "version",
    "evidence_tier",
    "falsifier",
    "identity",
)

# hcli.result_envelope aliases. Canonical names stay the roadmap's.
ALIAS_FIELDS = (
    "verified_facts",  # facts
    "resource_use",  # resource_usage
    "next_action",  # next_actions[0]
    "receipt_paths",  # receipts
)

# Bookkeeping / location. Never part of the content seal or identity key.
# This repo has been burned by code keying on basename.
_NOT_CONTENT = frozenset(
    {
        "seal_sha256",
        "artifact_identity",
        "experiment_receipt",
        "bench",
        "head",
        "branch",
        "path",
        "location",
        "filename",
        "rel",
        "receipt_path",
        "receipt_paths",
    }
)

# Identity key is these facts only. location/path is recorded separately.
_IDENTITY_KEY_FIELDS = (
    "kind",
    "producer",
    "inputs",
    "machine",
    "commit",
    "content_sha256",
)

NAMED_PRODUCERS: tuple[str, ...] = (
    "tools/future/ba_delta_ab.py",
    "tools/future/lpc_baselines.py",
    "tools/future/capability_eval.py",
)


class ContractError(ValueError):
    """An identity or experiment receipt failed the shared contract."""


def canonical_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def content_seal(payload: Any) -> str:
    """Hash of artifact content with location and bookkeeping stripped."""
    if isinstance(payload, Mapping):
        body = {k: v for k, v in payload.items() if k not in _NOT_CONTENT}
        return sha256_text(canonical_dumps(body))
    return sha256_text(canonical_dumps(payload))


def canonicalize_inputs(inputs: Iterable[Any] | None) -> list[dict[str, str]]:
    """Input seals only. role + content_sha256. Paths are dropped."""
    rows: list[dict[str, str]] = []
    for item in inputs or ():
        if isinstance(item, Mapping):
            rows.append(
                {
                    "role": str(item.get("role") or ""),
                    "content_sha256": str(item.get("content_sha256") or ""),
                }
            )
        else:
            rows.append({"role": str(item), "content_sha256": ""})
    rows.sort(key=lambda r: (r["role"], r["content_sha256"]))
    return rows


def input_ref(role: str, payload: Any) -> dict[str, str]:
    return {"role": role, "content_sha256": content_seal(payload)}


def current_commit() -> str:
    return git("rev-parse", "HEAD") or "UNKNOWN"


def current_machine() -> str:
    return f"{platform.system()}-{platform.machine()}"


def identity_material(
    *,
    kind: str,
    producer: str,
    inputs: Iterable[Any] | None,
    machine: str,
    commit: str,
    content_sha256: str,
    location: str | None = None,
) -> dict[str, Any]:
    """Facts that form the identity key.

    location is accepted so callers can pass it, and is deliberately
    omitted from the returned material. Identity is not a file path.

    MUTATION_CHECK: adding ``"location": location`` here must make
    test_artifact_identity_stable_under_path_change_different_under_content
    FAIL. Restore after the check.
    """
    del location  # location is a fact about where a copy sits, not identity
    return {
        "kind": str(kind),
        "producer": str(producer),
        "inputs": canonicalize_inputs(inputs),
        "machine": str(machine),
        "commit": str(commit),
        "content_sha256": str(content_sha256),
    }


def identity_key(material: Mapping[str, Any]) -> str:
    payload = {k: material[k] for k in _IDENTITY_KEY_FIELDS}
    return sha256_text(canonical_dumps(payload))


def artifact_identity(
    *,
    kind: str,
    producer: str,
    content: Any,
    inputs: Iterable[Any] | None = None,
    machine: str | None = None,
    commit: str | None = None,
    location: str | None = None,
) -> dict[str, Any]:
    """Stable identity for a produced artifact. Path is location, not identity."""
    if not kind:
        raise ContractError("ArtifactIdentity.kind is empty")
    if not producer:
        raise ContractError("ArtifactIdentity.producer is empty")
    seal = content_seal(content)
    material = identity_material(
        kind=kind,
        producer=producer,
        inputs=inputs,
        machine=machine if machine is not None else current_machine(),
        commit=commit if commit is not None else current_commit(),
        content_sha256=seal,
        location=location,
    )
    ident = {
        "schema": IDENTITY_SCHEMA,
        "version": IDENTITY_VERSION,
        **material,
        "identity_key": identity_key(material),
        "location": location,
    }
    validate_artifact_identity(ident)
    return ident


def validate_artifact_identity(ident: Mapping[str, Any]) -> None:
    if ident.get("schema") != IDENTITY_SCHEMA:
        raise ContractError(f"identity schema {ident.get('schema')!r}")
    missing = [k for k in _IDENTITY_KEY_FIELDS if k not in ident]
    if missing:
        raise ContractError(f"ArtifactIdentity missing {missing}")
    if not ident.get("identity_key"):
        raise ContractError("ArtifactIdentity.identity_key is empty")
    expected = identity_key(ident)
    if ident["identity_key"] != expected:
        raise ContractError("ArtifactIdentity.identity_key does not match sealed facts")
    # Path must not be able to impersonate identity: recomputing the key
    # with location stuffed in must not be how identity_key was made.
    if "location" in ident and ident["location"] is not None:
        stuffed = dict(ident)
        stuffed["location"] = ident["location"]
        # identity_key() ignores unknown fields because it picks KEY_FIELDS.
        # Explicit: location is not a key field.
        if "location" in _IDENTITY_KEY_FIELDS:
            raise ContractError("location leaked into identity key fields")


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return [value]


def experiment_receipt(
    *,
    claim: str,
    verdict: str,
    evidence_tier: str,
    falsifier: str,
    identity: Mapping[str, Any],
    scope: Any = None,
    facts: Iterable[Any] | None = None,
    hypotheses: Iterable[Any] | None = None,
    evidence: Iterable[Any] | None = None,
    artifacts: Iterable[Any] | None = None,
    hashes: Iterable[Any] | None = None,
    tests: Any = None,
    controls: Iterable[Any] | None = None,
    negative_controls: Iterable[Any] | None = None,
    failures: Iterable[Any] | None = None,
    resource_usage: Mapping[str, Any] | None = None,
    qualification: Any = None,
    contamination: Iterable[Any] | None = None,
    uncertainty: Any = None,
    next_actions: Iterable[Any] | None = None,
    receipts: Iterable[Any] | None = None,
) -> dict[str, Any]:
    """Roadmap ResultEnvelope + evidence_tier + falsifier + ArtifactIdentity."""
    if verdict not in VERDICTS:
        raise ContractError(f"unknown verdict {verdict!r}")
    if evidence_tier not in EVIDENCE_TIERS:
        raise ContractError(f"unknown evidence_tier {evidence_tier!r}")
    if not claim:
        raise ContractError("claim is empty")
    if not falsifier:
        raise ContractError("falsifier is empty")
    validate_artifact_identity(identity)

    facts_list = _as_list(facts)
    usage = dict(resource_usage or {})
    actions = _as_list(next_actions)
    receipt_list = _as_list(receipts)
    ident = dict(identity)
    hash_list = _as_list(hashes) or [
        {"kind": "content_sha256", "value": ident.get("content_sha256")},
        {"kind": "identity_key", "value": ident.get("identity_key")},
    ]
    artifact_list = _as_list(artifacts)
    if not artifact_list:
        artifact_list = [ident]

    env = {
        "schema": RECEIPT_SCHEMA,
        "version": RECEIPT_VERSION,
        "verdict": verdict,
        "claim": claim,
        "scope": scope,
        "facts": facts_list,
        "verified_facts": facts_list,
        "hypotheses": _as_list(hypotheses),
        "evidence": _as_list(evidence),
        "artifacts": artifact_list,
        "hashes": hash_list,
        "tests": tests if tests is not None else [],
        "controls": _as_list(controls),
        "negative_controls": _as_list(negative_controls),
        "failures": _as_list(failures),
        "resource_usage": usage,
        "resource_use": usage,
        "qualification": qualification,
        "contamination": _as_list(contamination),
        "uncertainty": _as_list(uncertainty) if not isinstance(uncertainty, str) else [uncertainty],
        "next_actions": actions,
        "next_action": actions[0] if actions else None,
        "receipts": receipt_list,
        "receipt_paths": receipt_list,
        "evidence_tier": evidence_tier,
        "falsifier": falsifier,
        "identity": ident,
    }
    validate_experiment_receipt(env)
    return env


def validate_experiment_receipt(env: Mapping[str, Any]) -> None:
    if env.get("schema") != RECEIPT_SCHEMA:
        raise ContractError(f"receipt schema {env.get('schema')!r}")
    missing = [k for k in ROADMAP_ENVELOPE_FIELDS if k not in env]
    missing.extend(k for k in CONTRACT_FIELDS if k not in env)
    if missing:
        raise ContractError(f"ExperimentReceipt missing {missing}")
    if env.get("verdict") not in VERDICTS:
        raise ContractError(f"unknown verdict {env.get('verdict')!r}")
    if env.get("evidence_tier") not in EVIDENCE_TIERS:
        raise ContractError(f"unknown evidence_tier {env.get('evidence_tier')!r}")
    if not env.get("claim"):
        raise ContractError("claim is empty")
    if not env.get("falsifier"):
        raise ContractError("falsifier is empty")
    ident = env.get("identity")
    if not isinstance(ident, Mapping):
        raise ContractError("identity is missing")
    validate_artifact_identity(ident)
    # Aliases must not drift from canonical names.
    if env.get("verified_facts") != env.get("facts"):
        raise ContractError("verified_facts alias drifted from facts")
    if env.get("resource_use") != env.get("resource_usage"):
        raise ContractError("resource_use alias drifted from resource_usage")


def attach(
    doc: Mapping[str, Any],
    *,
    claim: str,
    verdict: str,
    evidence_tier: str,
    falsifier: str,
    producer: str | None = None,
    kind: str = "experiment_receipt",
    location: str | None = None,
    inputs: Iterable[Any] | None = None,
    machine: str | None = None,
    commit: str | None = None,
    **envelope: Any,
) -> dict[str, Any]:
    """Add ArtifactIdentity + ExperimentReceipt without dropping producer fields."""
    out = dict(doc)
    recorded_by = producer or str(out.get("recorded_by") or RECORDED_BY)
    ident = artifact_identity(
        kind=kind,
        producer=recorded_by,
        content=out,
        inputs=inputs,
        machine=machine,
        commit=commit,
        location=location,
    )
    env = experiment_receipt(
        claim=claim,
        verdict=verdict,
        evidence_tier=evidence_tier,
        falsifier=falsifier,
        identity=ident,
        **envelope,
    )
    out["artifact_identity"] = ident
    out["experiment_receipt"] = env
    return out


def extract_receipt(doc: Mapping[str, Any]) -> dict[str, Any]:
    env = doc.get("experiment_receipt")
    if isinstance(env, Mapping) and env.get("schema") == RECEIPT_SCHEMA:
        validate_experiment_receipt(env)
        return dict(env)
    if doc.get("schema") == RECEIPT_SCHEMA:
        validate_experiment_receipt(doc)
        return dict(doc)
    raise ContractError("document does not carry hawking.experiment.receipt.v1")


def emit_named_producers() -> dict[str, dict[str, Any]]:
    """Call the three real producers. Each must emit a valid ExperimentReceipt."""
    from tools.future import ba_delta_ab
    from tools.future import capability_eval
    from tools.future import lpc_baselines

    emitted: dict[str, dict[str, Any]] = {}

    ba_doc = ba_delta_ab.build()
    emitted["tools/future/ba_delta_ab.py"] = extract_receipt(ba_doc)

    lpc_path = lpc_baselines.build()
    lpc_doc = json.loads(lpc_path.read_text())
    emitted["tools/future/lpc_baselines.py"] = extract_receipt(lpc_doc)

    ce_path = capability_eval.build()
    ce_doc = json.loads(ce_path.read_text())
    emitted["tools/future/capability_eval.py"] = extract_receipt(ce_doc)

    missing = [p for p in NAMED_PRODUCERS if p not in emitted]
    if missing:
        raise ContractError(f"named producers did not emit: {missing}")
    for name, env in emitted.items():
        validate_experiment_receipt(env)
        if env["identity"]["producer"] != name:
            raise ContractError(
                f"{name} identity.producer is {env['identity']['producer']!r}"
            )
    return emitted
