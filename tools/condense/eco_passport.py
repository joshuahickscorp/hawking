#!/usr/bin/env python3.12
"""Hawking Passport: the one identity/receipt graph for the ecosystem frontier.

The directive requires "one identity/receipt graph across artifact, Doctor treatment,
physical bytes, capability contract, Context Horizon, session state, device profile,
and client compatibility." This module is that graph.

A `Passport` binds eight facets into a single content-addressed identity and self-seals
into `passport_sha256` using the campaign's exact canonical hashing form (so an imported
cell's real seals validate here). Passports form a prefix/branch DAG (constitution 6.9):

    parent passport identity
      + token/event/treatment delta
      + model/tokenizer/profile identity
      + position policy
      + KV/state codec
      = child passport identity

Claim separation (constitution 3) is enforced structurally: every facet declares the
claim layer it may support, and `verify_passport` refuses a physical-BPW claim that
smuggles runtime/context/agent bytes into the standalone-artifact facet.

Default-off: minting or verifying a Passport has no runtime side effect. It is a
descriptor, not an activation.
"""
from __future__ import annotations

import os
import sys
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from eco_common import (  # noqa: E402
    EcoError, SCHEMA_PASSPORT, SCHEMA_IDENTITY_EDGE,
    hash_value, seal_field, sealed, is_sha256, now_iso,
)

# The eight identity dimensions, in canonical order, each pinned to the claim layer
# it may support (constitution section 3). "environment" / "interop" are neither a
# model claim nor an agent claim; they scope the deployment.
DIMENSIONS: tuple[str, ...] = (
    "artifact",
    "doctor_treatment",
    "physical_bytes",
    "capability_contract",
    "context_horizon",
    "session_state",
    "device_profile",
    "client_compat",
)

CLAIM_LAYERS: dict[str, str] = {
    "artifact": "standalone",
    "doctor_treatment": "standalone",       # embedded treatment counts in physical BPW
    "physical_bytes": "standalone",
    "capability_contract": "standalone",
    "context_horizon": "context_system",
    "session_state": "agent_system",
    "device_profile": "environment",
    "client_compat": "interop",
}

# Byte roles that MUST be excluded from the standalone physical-bytes facet. Runtime
# indexes, KV pages, session logs, and workspace caches are context/agent-system bytes;
# folding them into physical_bytes would launder a failed standalone claim.
NON_STANDALONE_BYTE_ROLES: frozenset[str] = frozenset({
    "kv_cache", "runtime_index", "workspace_cache", "session_state",
    "retrieval_index", "external_memory", "ssd_state_plane",
})


def make_facet(dimension: str, value: dict[str, Any]) -> dict[str, Any]:
    """A single identity facet: {dimension, claim_layer, value, facet_sha256}."""
    if dimension not in CLAIM_LAYERS:
        raise EcoError(f"unknown identity dimension: {dimension}")
    if not isinstance(value, dict):
        raise EcoError(f"facet value for {dimension} must be an object")
    facet = {
        "dimension": dimension,
        "claim_layer": CLAIM_LAYERS[dimension],
        "value": value,
        "facet_sha256": hash_value(value),
    }
    return facet


def _validate_physical_bytes_facet(value: dict[str, Any]) -> list[str]:
    """The physical-bytes facet must account only for standalone artifact bytes,
    with all corrections/codebooks/exceptions/protected-islands/routing counted, and
    no runtime bytes smuggled in (constitution 4.1 / 3 / hard prohibitions)."""
    errs: list[str] = []
    breakdown = value.get("byte_breakdown")
    if not isinstance(breakdown, dict):
        errs.append("physical_bytes.byte_breakdown missing")
        return errs
    for role in breakdown:
        if role in NON_STANDALONE_BYTE_ROLES:
            errs.append(f"physical_bytes must not include runtime role '{role}'")
    total = value.get("all_in_model_payload_bytes")
    if not isinstance(total, (int, float)) or total <= 0:
        errs.append("physical_bytes.all_in_model_payload_bytes must be positive")
    else:
        summed = sum(v for v in breakdown.values() if isinstance(v, (int, float)))
        # The breakdown must not exceed the declared all-in total (it may equal it).
        if summed > total + 1:  # 1-byte rounding slack
            errs.append("physical_bytes.byte_breakdown exceeds all-in total")
    bpw = value.get("all_in_model_payload_bpw")
    if not isinstance(bpw, (int, float)) or bpw <= 0:
        errs.append("physical_bytes.all_in_model_payload_bpw must be positive")
    return errs


def mint_passport(
    facets: dict[str, dict[str, Any]],
    *,
    parent_label: str,
    rate_id: str,
    branch: str,
    bindings: dict[str, Any] | None = None,
    profile: str = "EXTREME_LOCAL",
) -> dict[str, Any]:
    """Bind the eight facets into one sealed Passport.

    `facets` maps dimension -> facet value dict (raw values, not wrapped facets).
    `bindings` carries immutable campaign-evidence references (plan_sha256,
    cell_identity_sha256, result_sha256, ...) that this identity descends from.
    """
    missing = [d for d in DIMENSIONS if d not in facets]
    if missing:
        raise EcoError(f"passport missing dimensions: {', '.join(missing)}")
    wrapped = {d: make_facet(d, facets[d]) for d in DIMENSIONS}
    phys_errs = _validate_physical_bytes_facet(facets["physical_bytes"])
    if phys_errs:
        raise EcoError("; ".join(phys_errs))
    bindings = dict(bindings or {})
    for k, v in bindings.items():
        if k.endswith("_sha256") and v is not None and not is_sha256(v):
            raise EcoError(f"binding {k} is not a sha256")
    passport = {
        "schema": SCHEMA_PASSPORT,
        "profile": profile,
        "parent_label": parent_label,
        "rate_id": str(rate_id),
        "branch": branch,
        "facets": wrapped,
        "bindings": bindings,
        "claim_boundary": {d: CLAIM_LAYERS[d] for d in DIMENSIONS},
        "created_at": now_iso(),
    }
    return seal_field(passport, "passport_sha256")


def verify_passport(passport: dict[str, Any]) -> tuple[bool, list[str]]:
    """Structurally verify a Passport. Returns (ok, reasons_if_not)."""
    reasons: list[str] = []
    if not isinstance(passport, dict):
        return False, ["passport is not an object"]
    if passport.get("schema") != SCHEMA_PASSPORT:
        reasons.append("wrong schema")
    facets = passport.get("facets")
    if not isinstance(facets, dict):
        return False, reasons + ["facets missing"]
    for dim in DIMENSIONS:
        f = facets.get(dim)
        if not isinstance(f, dict):
            reasons.append(f"facet {dim} missing")
            continue
        if f.get("claim_layer") != CLAIM_LAYERS[dim]:
            reasons.append(f"facet {dim} claim_layer tampered")
        if f.get("facet_sha256") != hash_value(f.get("value")):
            reasons.append(f"facet {dim} content hash mismatch")
    phys = facets.get("physical_bytes", {})
    if isinstance(phys, dict):
        reasons.extend(_validate_physical_bytes_facet(phys.get("value", {})))
    if not sealed(passport, "passport_sha256"):
        reasons.append("passport self-seal invalid")
    for k, v in (passport.get("bindings") or {}).items():
        if k.endswith("_sha256") and v is not None and not is_sha256(v):
            reasons.append(f"binding {k} malformed")
    return (not reasons), reasons


def identity_edge(
    parent: dict[str, Any],
    *,
    delta_kind: str,
    delta: dict[str, Any],
    model_identity: str,
    position_policy: str,
    kv_state_codec: str,
) -> dict[str, Any]:
    """Derive a child identity from a parent Passport plus a delta (constitution 6.9).

    The child identity is content-addressed over (parent identity, delta, model/
    tokenizer/profile identity, position policy, KV/state codec) so exact prefix reuse,
    copy-on-write branches, session forks, and rollback all address the same node.
    """
    ok, why = verify_passport(parent)
    if not ok:
        raise EcoError(f"parent passport invalid: {'; '.join(why)}")
    edge = {
        "schema": SCHEMA_IDENTITY_EDGE,
        "parent_passport_sha256": parent["passport_sha256"],
        "delta_kind": delta_kind,
        "delta": delta,
        "model_identity": model_identity,
        "position_policy": position_policy,
        "kv_state_codec": kv_state_codec,
        "created_at": now_iso(),
    }
    edge["child_identity_sha256"] = hash_value({k: v for k, v in edge.items() if k != "created_at"})
    return edge


def selftest() -> dict[str, Any]:
    """Offline self-check used by the CLI and tests."""
    phys = {
        "all_in_model_payload_bpw": 2.34,
        "all_in_model_payload_bytes": 4_200_000_000,
        "byte_breakdown": {
            "packed_2d_tensor_bytes": 4_000_000_000,
            "lossless_non_2d_passthrough_bytes": 150_000_000,
            "doctor_correction_bytes": 40_000_000,
            "codebook_bytes": 10_000_000,
        },
    }
    facets = {
        "artifact": {"family": "qwen2.5-dense", "label": "14B", "source_sha256": "0" * 64},
        "doctor_treatment": {"branch": "doctor_static", "treatment_bytes": 40_000_000},
        "physical_bytes": phys,
        "capability_contract": {"ppl_rel_delta_max": 0.08, "capability_abs_delta_min": -0.05},
        "context_horizon": {"nominal": 32768, "kv_precision": "int4", "layer": "context_system"},
        "session_state": {"continuum": "event_sourced", "layer": "agent_system"},
        "device_profile": {"name": "Studio-M3Ultra-96", "weight_budget_gb": 78.0},
        "client_compat": {"openai_chat": True, "mcp": True, "hide": True},
    }
    passport = mint_passport(facets, parent_label="14B", rate_id="2", branch="doctor_static")
    ok, why = verify_passport(passport)
    if not ok:
        raise EcoError(f"selftest passport failed: {why}")
    edge = identity_edge(
        passport, delta_kind="context_fork", delta={"tokens": 4096},
        model_identity="qwen2.5-14b", position_policy="yarn", kv_state_codec="int4",
    )
    # Tamper detection.
    tampered = dict(passport)
    tampered_facets = {k: dict(v) for k, v in passport["facets"].items()}
    tampered_facets["physical_bytes"] = dict(tampered_facets["physical_bytes"])
    tampered_facets["physical_bytes"]["value"] = dict(tampered_facets["physical_bytes"]["value"])
    tampered_facets["physical_bytes"]["value"]["all_in_model_payload_bpw"] = 0.5
    tampered["facets"] = tampered_facets
    ok2, _ = verify_passport(tampered)
    if ok2:
        raise EcoError("selftest failed: tamper not detected")
    return {
        "ok": True,
        "passport_sha256": passport["passport_sha256"],
        "child_identity_sha256": edge["child_identity_sha256"],
        "tamper_detected": True,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(selftest(), indent=2, sort_keys=True))
