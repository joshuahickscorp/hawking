#!/usr/bin/env python3
"""Query Hawking's canonical Gravity method registry by model capabilities.

The registry remains the sole owner of method descriptions and evidence.  This
reader only performs deterministic tag matching; it does not promote a method
or turn a scoped Flash result into a cross-model conclusion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "tools/foundry/GRAVITY_METHOD_REGISTRY.json"
SCAR_ATLAS = ROOT / "tools/foundry/NEGATIVE_TRANSFER_ATLAS.json"
OBSERVATION_SCHEMA = "hawking.gravity.observed_traits.v1"
REUSE_SCHEMA = "hawking.gravity.method_reuse.v1"


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
    return doc


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


def _write_receipt(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def reuse_method(
    observation: dict[str, Any],
    *,
    receipt_path: Path,
    artifact_path: Path,
    registry_path: Path = REGISTRY,
) -> dict[str, Any]:
    """Select, falsify, invoke, verify, and receipt one applicable method."""
    decisions = assess_methods(observation, registry_path)
    applicable = [decision["method"] for decision in decisions if decision["applicable"]]
    tags = set(observation.get("architecture_tags") or [])
    runnable: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for method in applicable:
        for adapter in (method.get("automation") or {}).get("adapters", []):
            if set(adapter.get("architecture_tags") or []).issubset(tags):
                runnable.append((method, adapter))
    base = {
        "schema": REUSE_SCHEMA,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "observation": observation,
        "method_assessment": [
            {
                "id": decision["method"]["id"],
                "applicable": decision["applicable"],
                "reasons": decision["reasons"],
            }
            for decision in decisions
        ],
        "scars": matching_scars(observation),
    }
    if len(runnable) != 1:
        base.update({
            "status": "REFUSED",
            "refusal": (
                "no applicable method has a compatible automatic adapter"
                if not runnable else
                "more than one automatic adapter matched; selection is ambiguous"
            ),
            "invoked": False,
        })
        _write_receipt(receipt_path, base)
        return base

    method, adapter = runnable[0]
    missing = [path for path in adapter["required_inputs"] if not (ROOT / path).is_file()]
    if missing:
        base.update({
            "status": "REFUSED",
            "refusal": "cheap falsifier found missing inputs: " + ", ".join(missing),
            "selected_method": method["id"],
            "selected_adapter": adapter["id"],
            "invoked": False,
        })
        _write_receipt(receipt_path, base)
        return base

    invocation = observation.get("invocation") or {}
    target = str(invocation.get("target_ebpw") or "0.5")
    generation = str(invocation.get("generation") or "automatic-gravity-reuse-v1")
    command = [
        sys.executable,
        str(ROOT / adapter["runner"]),
        "--target-ebpw", target,
        "--generation", generation,
        "--out", str(artifact_path),
    ]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if completed.returncode != 0:
        base.update({
            "status": "EXECUTION_FAILED",
            "selected_method": method["id"],
            "selected_adapter": adapter["id"],
            "laws": method.get("laws") or [],
            "cheap_falsifier": {"passed": True, "required_inputs": adapter["required_inputs"]},
            "invoked": True,
            "returncode": completed.returncode,
            "stderr": completed.stderr[-4000:],
        })
        _write_receipt(receipt_path, base)
        return base

    artifact = json.loads(artifact_path.read_text())
    sys.path.insert(0, str(ROOT / "tools"))
    from nr_container import validate
    valid, problems = validate(artifact)
    seal = artifact.get("seal_sha256")
    unsealed = dict(artifact)
    unsealed.pop("seal_sha256", None)
    expected_seal = hashlib.sha256(json.dumps(unsealed, sort_keys=True).encode()).hexdigest()
    verified = valid and seal == expected_seal
    base.update({
        "status": "EXECUTED_VERIFIED" if verified else "VERIFICATION_FAILED",
        "selected_method": method["id"],
        "selected_adapter": adapter["id"],
        "laws": method.get("laws") or [],
        "cheap_falsifier": {"passed": True, "required_inputs": adapter["required_inputs"]},
        "invoked": True,
        "invocation": {"argv": command[1:], "returncode": completed.returncode},
        "verification": {
            "verifier": adapter["verifier"],
            "valid": valid,
            "problems": problems,
            "seal_matches": seal == expected_seal,
        },
        "new_evidence": {
            "path": str(artifact_path),
            "sha256": _sha256(artifact_path),
            "status": artifact.get("status"),
        },
        "claim_boundary": method["claim_boundary"],
    })
    _write_receipt(receipt_path, base)
    return base


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
    parser.add_argument("--reuse-observation", type=Path,
                        help="select and execute a compatible method from an observed-traits JSON")
    parser.add_argument("--out", type=Path,
                        help="method-reuse evidence receipt (required with --reuse-observation)")
    parser.add_argument("--artifact-out", type=Path,
                        help="invoked method output (required with --reuse-observation)")
    args = parser.parse_args()
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
        return 0 if result["status"] in {"EXECUTED_VERIFIED", "REFUSED"} else 1
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
