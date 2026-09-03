"""Non-promoting Formalizer and Repair controllers plus Q0--Q6 contract loading.

These are deliberately proposal controllers.  A formalizer may add a new Lean
proof-state, and a repair loop may record compiler-feedback candidates, but
neither can mark fidelity, proof validity, Tribunal admission, launch, or
capability as passed.  Those outcomes remain with independent verifiers and
the existing lattice/limit gates.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ramanujan.layout import CONTRACTS_ROOT, REPO_ROOT, resolve_ramanujan_path
from ramanujan.roles import capability_for
from ramanujan.search import repair_from_error
from ramanujan.stores import StoreRefused, Stores


ROOT = REPO_ROOT
CONTRACTS_PATH = CONTRACTS_ROOT / "RAMANUJAN_Q0_Q6_CONTRACTS.json"
QUALIFICATION_IDS = ("Q0", "Q1", "Q2", "Q3", "Q4", "Q5", "Q6")


class ControllerRefused(RuntimeError):
    """A controller received an invalid state or attempted to certify itself."""


def _bound_path(repo_root: Path, relative: str) -> Path:
    """Resolve a historical receipt binding only for the real repository."""
    if repo_root.resolve() != REPO_ROOT:
        return repo_root / relative
    if relative.startswith("ramanujan/"):
        return resolve_ramanujan_path(repo_root / relative)
    if relative.startswith("odyssey/"):
        from lab.layout import odyssey_path

        return odyssey_path(*Path(relative).parts[1:])
    return repo_root / relative


def verify_q0_evidence_bundle(path: Path, repo_root: Path) -> dict[str, Any]:
    """Verify the sealed Q0 bundle and every bound leaf byte."""
    if repo_root.resolve() == REPO_ROOT:
        path = resolve_ramanujan_path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ControllerRefused(f"Q0 evidence bundle is unreadable: {exc}") from exc
    if value.get("schema") != "hawking.ramanujan.q0_evidence_bundle.v1":
        raise ControllerRefused("Q0 evidence bundle schema mismatch")
    seal_value = value.get("seal_sha256")
    unsigned = {key: item for key, item in value.items() if key != "seal_sha256"}
    observed_seal = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    if seal_value != observed_seal:
        raise ControllerRefused("Q0 evidence bundle seal mismatch")
    if value.get("production_authority") is not False or value.get("research_authority") is not False:
        raise ControllerRefused("Q0 evidence bundle illegally grants authority")
    image = value.get("image")
    if (
        not isinstance(image, dict)
        or image.get("id") != "sha256:21114fb4b7066b5a7c535d36685211147a920233fc7544a922846056c8ec03ad"
        or image.get("size_bytes") != 4_644_372_611
    ):
        raise ControllerRefused("Q0 evidence bundle image identity or size differs from the live lock")
    leaves = value.get("leaf_sha256")
    if not isinstance(leaves, dict) or not leaves:
        raise ControllerRefused("Q0 evidence bundle has no leaf chain")
    for relative, expected in leaves.items():
        if not isinstance(relative, str) or not isinstance(expected, str) or len(expected) != 64:
            raise ControllerRefused("Q0 leaf binding is malformed")
        leaf = _bound_path(repo_root, relative)
        if leaf.is_symlink() or not leaf.is_file():
            raise ControllerRefused(f"Q0 evidence leaf is absent or unsafe: {relative}")
        if hashlib.sha256(leaf.read_bytes()).hexdigest() != expected:
            raise ControllerRefused(f"Q0 evidence leaf changed: {relative}")
    return value


def qualification_contracts(path: Path = CONTRACTS_PATH) -> dict[str, dict[str, Any]]:
    """Load the single qualification contract and reject partial/duplicate tables."""
    if not path.is_absolute():
        path = resolve_ramanujan_path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ControllerRefused(f"qualification contracts are unreadable: {exc}") from exc
    if raw.get("schema") != "hawking.ramanujan.qualification_contracts.v1":
        raise ControllerRefused("qualification contract schema mismatch")
    recorded_seal = raw.get("seal_sha256")
    unsigned = {key: value for key, value in raw.items() if key != "seal_sha256"}
    observed_seal = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    if recorded_seal != observed_seal:
        raise ControllerRefused("qualification contract seal mismatch")
    bindings = raw.get("receipt_bindings")
    if not isinstance(bindings, dict) or not bindings:
        raise ControllerRefused("qualification contracts lack immutable receipt bindings")
    repo_root = REPO_ROOT
    for relative, expected in bindings.items():
        if not isinstance(relative, str) or not isinstance(expected, str) or len(expected) != 64:
            raise ControllerRefused("qualification receipt binding is malformed")
        receipt_path = _bound_path(repo_root, relative)
        try:
            observed = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        except OSError as exc:
            raise ControllerRefused(f"qualification receipt is unavailable: {relative}: {exc}") from exc
        if observed != expected:
            raise ControllerRefused(f"qualification receipt provenance changed: {relative}")
    bundle_relative = "ramanujan/RAMANUJAN_Q0_EVIDENCE_BUNDLE.json"
    if bundle_relative not in bindings:
        raise ControllerRefused("qualification contracts do not bind the Q0 leaf-evidence bundle")
    verify_q0_evidence_bundle(_bound_path(repo_root, bundle_relative), repo_root)
    rows = raw.get("contracts")
    if not isinstance(rows, list):
        raise ControllerRefused("qualification contracts must be a list")
    by_id: dict[str, dict[str, Any]] = {}
    required = {"id", "name", "purpose", "required_evidence", "admission", "status", "receipt"}
    optional = {"supporting_receipts"}
    for row in rows:
        if not isinstance(row, dict) or not required.issubset(row) or set(row) - required - optional:
            raise ControllerRefused("each qualification contract must have the complete fixed schema")
        qid = row.get("id")
        if not isinstance(qid, str) or qid in by_id:
            raise ControllerRefused("qualification ids must be unique strings")
        if not all(isinstance(row.get(key), str) and row[key].strip() for key in required - {"required_evidence"}):
            raise ControllerRefused(f"qualification {qid!r} has an empty required field")
        evidence = row.get("required_evidence")
        if not isinstance(evidence, list) or not evidence or not all(isinstance(item, str) and item for item in evidence):
            raise ControllerRefused(f"qualification {qid!r} must name required evidence")
        supporting = row.get("supporting_receipts")
        if supporting is not None and (
            not isinstance(supporting, list)
            or not supporting
            or not all(isinstance(item, str) and item.strip() for item in supporting)
        ):
            raise ControllerRefused(f"qualification {qid!r} has invalid supporting receipts")
        by_id[qid] = dict(row)
    if tuple(by_id) != QUALIFICATION_IDS:
        raise ControllerRefused(f"qualification ids must be exactly {QUALIFICATION_IDS}")
    return by_id


class FormalizerController:
    """Write a new formalization proposal without assigning a fidelity tier."""

    role_id = "formalizer"

    def propose(
        self,
        stores: Stores,
        *,
        claim_id: str,
        proof_state_id: str,
        lean: str,
        informal_binding: str,
    ) -> dict[str, Any]:
        cap = capability_for(self.role_id)
        cap.require("read_claim")
        cap.require("write_lean")
        claim = stores.claims.get(claim_id)
        if claim is None:
            raise ControllerRefused(f"cannot formalize absent claim {claim_id!r}")
        if claim.in_graveyard:
            raise ControllerRefused(f"cannot formalize buried claim {claim_id!r}")
        if not isinstance(lean, str) or not lean.strip():
            raise ControllerRefused("Lean proposal must be non-empty")
        if informal_binding != claim.statement:
            raise ControllerRefused("informal binding must equal the stored claim statement")
        try:
            entry = stores.add_proof_state(
                proof_state_id, claim_id, lean, actor=self.role_id,
                controller="FormalizerController",
                informal_binding_sha256=hashlib.sha256(informal_binding.encode()).hexdigest(),
            )
        except StoreRefused as exc:
            raise ControllerRefused(str(exc)) from exc
        return {
            "status": "PROPOSED_PENDING_INDEPENDENT_FIDELITY",
            "qualification": "Q4",
            "claim_id": claim_id,
            "proof_state_id": entry.id,
            "actor": self.role_id,
            "tier_before": int(claim.tier),
            "tier_after": int(claim.tier),
            "promotion": "REFUSED_TO_CONTROLLER",
        }


class RepairController:
    """Record compiler-feedback repair candidates without overwriting Lean state."""

    role_id = "prover"

    def propose(
        self,
        stores: Stores,
        *,
        proof_state_id: str,
        compiler_error: str,
        candidate: str | None = None,
    ) -> dict[str, Any]:
        cap = capability_for(self.role_id)
        cap.require("read_lean")
        cap.require("run_lean")
        source = stores.proof_states.get(proof_state_id)
        if source is None:
            raise ControllerRefused(f"cannot repair absent proof state {proof_state_id!r}")
        if not isinstance(compiler_error, str) or not compiler_error.strip():
            raise ControllerRefused("compiler feedback must be non-empty")
        proposed = candidate if candidate is not None else repair_from_error(source.lean, compiler_error)
        if proposed is not None and (not isinstance(proposed, str) or not proposed.strip()):
            raise ControllerRefused("repair candidate must be non-empty when supplied")
        payload = {
            "controller": "RepairController",
            "source_proof_state": proof_state_id,
            "claim_id": source.claim_id,
            "compiler_error_sha256": hashlib.sha256(compiler_error.encode()).hexdigest(),
            "candidate_sha256": None if proposed is None else hashlib.sha256(proposed.encode()).hexdigest(),
            "candidate_available": proposed is not None,
            "requires": ["pinned compiler check", "new formalizer write", "independent verifier"],
        }
        row = stores.ledger.append("proof_attempt", payload, actor=self.role_id)
        return {
            "status": "NO_REPAIR_RULE_MATCHED" if proposed is None else "PROPOSED_PENDING_COMPILER",
            "qualification": "Q5",
            "source_proof_state": proof_state_id,
            "candidate": proposed,
            "ledger_seq": row.seq,
            "source_is_unchanged": stores.proof_states[proof_state_id].lean == source.lean,
            "promotion": "REFUSED_TO_CONTROLLER",
        }
