#!/usr/bin/env python3
"""Materialize the paired-cognition sealed authority chain and prove TG10 refusal.

CPU/file-only. Creates sealed inputs, runs the five example contracts in dependency
order, and runs the TG10 development activation state machine against the true
current operational status (both TG10 operational passes FALSE).

Identity convention used by every validator in this chain:
  - document_sha256  = json_sha of the full sealed document (including seal_sha256)
  - document_seal_sha256 / seal_sha256 = json_sha of the document with seal_sha256 removed
  - json_sha = SHA-256 of compact UTF-8 JSON with sorted object keys (serde_json Map)
  - raw file-byte sha256 is NOT used by these contracts
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[6]
if not (ROOT / "Cargo.toml").exists():
    # worktree root is six levels up from this file:
    # paired-authorities / lifecycle / ascension-sandbox / records / campaign / workspace / ROOT
    ROOT = Path(__file__).resolve()
    for parent in ROOT.parents:
        if (parent / "Cargo.toml").exists() and (parent / "crates" / "hawking-core").exists():
            ROOT = parent
            break

OUT_DIR = ROOT / "workspace/campaign/records/ascension-sandbox/lifecycle/paired-authorities"
INPUT_DIR = OUT_DIR / "inputs"
OPS_STATUS_CANONICAL_PATH = (
    "/Users/scammermike/Downloads/hawking/workspace/campaign/records/"
    "ascension-sandbox/physical/lifecycle/ASCENSION_OPERATIONAL_ASCENT_STATUS.json"
)
# Prefer the live canonical document; fall back to the worktree copy.
OPS_STATUS_FILE_CANDIDATES = [
    Path(OPS_STATUS_CANONICAL_PATH),
    ROOT
    / "workspace/campaign/records/ascension-sandbox/physical/lifecycle/"
    / "ASCENSION_OPERATIONAL_ASCENT_STATUS.json",
]

QWEN30 = "qwen30"
QWEN80 = "qwen80"
TG10_BASE_TRUE_TPS = 100

PRIMARY_ACTIONS = [
    "submit_proposal",
    "protected_review",
    "falsify",
    "test_in_isolated_worktree",
    "accept_verified_proposal_for_own_champion",
    "mutate_own_champion",
    "promote_own_champion",
]
WORK_ACTIONS = [
    "submit_proposal",
    "protected_review",
    "falsify",
    "test_in_isolated_worktree",
]
ACCEPTANCE_EVIDENCE = [
    "sealed_proposal",
    "protected_adversarial_review",
    "independent_falsification",
    "independent_verifier_receipt",
]
TELEMETRY_FIELDS = [
    "eligible_sessions",
    "queued_request_count",
    "scheduled_turn_count",
    "oldest_wait_dispatches",
    "consecutive_dispatch_count",
]


def sha256_json(obj) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    return hashlib.sha256(raw).hexdigest()


def seal_value(obj: dict) -> dict:
    unsigned = {k: v for k, v in obj.items() if k != "seal_sha256"}
    sealed = dict(unsigned)
    sealed["seal_sha256"] = sha256_json(unsigned)
    return sealed


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def binding_for(path: Path) -> dict:
    """SealedDocumentBinding: document_sha256 is json_sha of the whole sealed Value."""
    document = load_json(path)
    if "seal_sha256" not in document:
        raise SystemExit(f"missing seal_sha256 in {path}")
    # Re-verify seal using the chain convention.
    unsigned = {k: v for k, v in document.items() if k != "seal_sha256"}
    expected_seal = sha256_json(unsigned)
    if document["seal_sha256"] != expected_seal:
        raise SystemExit(
            f"seal mismatch in {path}: recorded={document['seal_sha256']} expected={expected_seal}"
        )
    return {
        "path": str(path.resolve()),
        "document_sha256": sha256_json(document),
        "document": document,
    }


def authority_boundary() -> dict:
    return {
        "new_physical_model_processes_authorized": 0,
        "server_starts_authorized": 0,
        "port_binds_authorized": 0,
        "gpu_leases_authorized": 0,
        "tournament_state_mutations_authorized": 0,
        "paired_world_activation_authorized": False,
    }


def execution_boundary() -> dict:
    return {
        "live_artifact_scan_performed": False,
        "model_weights_loaded": False,
        "metal_device_or_dispatch_performed": False,
        "gpu_lease_or_registry_mutated": False,
        "model_or_decoder_token_executed": False,
        "logical_session_created": False,
        "runtime_watcher_or_server_started": False,
        "port_bound_or_listener_created": False,
        "hcli_executed": False,
        "tps_or_tg_measured": False,
        "tournament_state_mutated": False,
    }


def cross_lane_policy() -> dict:
    return {
        "allow_cross_lane_mission_reads": False,
        "allow_cross_lane_experiment_reads": False,
        "allow_cross_lane_receipt_reads": False,
        "allow_cross_lane_worktree_reads": False,
        "allow_cross_lane_session_reads": False,
        "allow_cross_lane_frontier_reads": False,
        "allow_cross_lane_patch_reads": False,
        "allow_cross_lane_score_reads": False,
        "verified_generic_knowledge_plane_publication_only": True,
    }


def mission_authority() -> dict:
    return {
        "primary_candidate_mutation_authority": True,
        "primary_candidate_promotion_authority": True,
        "helper_may_inspect_and_critique": True,
        "helper_may_propose_or_test_in_private_worktree": True,
        "helper_may_mutate_primary_champion": False,
        "helper_may_promote_primary_champion": False,
        "opposite_lane_may_mutate_primary_champion": False,
        "primary_or_helper_may_self_score": False,
        "protected_adversarial_review_required_before_promotion": True,
        "independent_verifier_required_before_promotion": True,
        "hard_gate_conjunction_required_before_activation": True,
        "post_tg3_freeze_required_before_final_evaluation": True,
        "solo_and_symmetric_orchestrator_evaluations_required": True,
    }


def private_namespaces(lane: str) -> dict:
    return {
        "mission": f"sealed://{lane}/mission",
        "experiments": f"sealed://{lane}/experiments",
        "receipts": f"sealed://{lane}/receipts",
        "worktree": f"sealed://{lane}/worktree",
        "sessions": f"sealed://{lane}/sessions",
        "frontier": f"sealed://{lane}/frontier",
        "patches": f"sealed://{lane}/patches",
        "scores": f"sealed://{lane}/scores",
        "protected_adversarial_reviews": f"sealed://{lane}/protected-adversarial-reviews",
        "independent_verification": f"sealed://{lane}/independent-verification",
        "hard_gate_conjunction": f"sealed://{lane}/hard-gate-conjunction",
        "post_tg3_freeze": f"sealed://{lane}/post-tg3-freeze",
        "solo_manager_evaluation": f"sealed://{lane}/solo-manager-evaluation",
        "symmetric_orchestrator_evaluation": f"sealed://{lane}/symmetric-orchestrator-evaluation",
    }


def lane_def(primary: str, helper: str) -> dict:
    return {
        "lane_id": f"{primary}-candidate-world",
        "primary_model_key": primary,
        "helper_model_key": helper,
        "private_namespaces": private_namespaces(f"{primary}-lane"),
        "mission_authority": mission_authority(),
    }


def endpoint(model_key: str) -> dict:
    if model_key == QWEN30:
        return {"host": "127.0.0.1", "port": 18430}
    if model_key == QWEN80:
        return {"host": "127.0.0.1", "port": 18480}
    raise ValueError(model_key)


def topology(model_key: str) -> dict:
    return {
        "resident_model_processes": 1,
        "immutable_weight_copies": 1,
        "logical_session_policy": "many_logical_sessions",
        "endpoint": endpoint(model_key),
    }


def sealed_fixture_document(model_key: str, role: str) -> dict:
    return seal_value(
        {
            "schema": f"hawking.ascension.{model_key}.{role}.fixture.v1",
            "status": f"PREPARED_{role}_FIXTURE",
            "model_key": model_key,
        }
    )


def contract_binding(model_key: str, role: str) -> dict:
    document = sealed_fixture_document(model_key, role)
    return {
        "path": f"/sealed/{model_key}/{role}.json",
        "document_sha256": sha256_json(document),
        "document": document,
        "topology_assertion": topology(model_key),
    }


def tg10_not_earned() -> dict:
    return {
        "required_base_true_tps": TG10_BASE_TRUE_TPS,
        "operational_pass": False,
        "coherent_hcli_pass": False,
        "complete_token_path_measured": False,
        "fallback_count": 0,
        "median_base_true_tps": None,
        "receipt_seal_sha256": None,
    }


def model_authority(model_key: str) -> dict:
    return {
        "model_key": model_key,
        "activation": {
            "contract": contract_binding(model_key, "activation"),
            "tg10": tg10_not_earned(),
        },
        "memory": contract_binding(model_key, "memory"),
        "session": contract_binding(model_key, "session"),
    }


def actor_ids(primary: str, helper: str) -> tuple[str, str, str]:
    return (
        f"{primary}-primary-in-{primary}-lane",
        f"{helper}-helper-in-{primary}-lane",
        f"{helper}-opponent-reviewer-in-{primary}-lane",
    )


def action_policy(primary: str, helper: str) -> dict:
    primary_actor, helper_actor, opponent_actor = actor_ids(primary, helper)
    return {
        "lane_id": f"{primary}-candidate-world",
        "primary_model_key": primary,
        "helper_model_key": helper,
        "opponent_model_key": helper,
        "primary_actor_id": primary_actor,
        "helper_actor_id": helper_actor,
        "opponent_actor_id": opponent_actor,
        "primary_allowed_actions": list(PRIMARY_ACTIONS),
        "helper_allowed_actions": list(WORK_ACTIONS),
        "opponent_allowed_actions": list(WORK_ACTIONS),
        "primary_champion_namespace": f"sealed://{primary}-lane/patches",
        "primary_acceptance_request_namespace": f"sealed://{primary}-lane/primary-acceptance-request",
        "helper_isolated_worktree_namespace": f"sealed://{primary}-lane/worktree",
        "opponent_isolated_worktree_namespace": f"sealed://{primary}-lane/opponent-isolated-worktree",
        "proposal_namespace": f"sealed://{primary}-lane/proposals",
        "protected_review_namespace": f"sealed://{primary}-lane/protected-reviews",
        "falsification_namespace": f"sealed://{primary}-lane/falsifications",
        "required_primary_acceptance_evidence": list(ACCEPTANCE_EVIDENCE),
    }


def mutation_action_policy_slim(primary: str, helper: str) -> dict:
    full = action_policy(primary, helper)
    return {
        k: full[k]
        for k in [
            "lane_id",
            "primary_model_key",
            "helper_model_key",
            "opponent_model_key",
            "primary_actor_id",
            "helper_actor_id",
            "opponent_actor_id",
            "primary_allowed_actions",
            "helper_allowed_actions",
            "opponent_allowed_actions",
        ]
    }


def run_example(example: str, input_path: Path, out_path: Path) -> None:
    if out_path.exists():
        out_path.unlink()
    cmd = [
        "cargo",
        "run",
        "-q",
        "-p",
        "hawking-core",
        "--example",
        example,
        "--",
        "--input",
        str(input_path.resolve()),
        "--out",
        str(out_path.resolve()),
    ]
    print(f"+ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise SystemExit(f"{example} failed with exit {result.returncode}")
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if not out_path.is_file():
        raise SystemExit(f"{example} did not create {out_path}")


def build_lane_input() -> dict:
    return seal_value(
        {
            "schema": "hawking.ascension.paired_cognition_lane_authority_input.v1",
            "activation_requested": False,
            # Honest §9 missions: Q80 primary + Q30 helper, Q30 primary + Q80 helper.
            "model_authorities": [model_authority(QWEN30), model_authority(QWEN80)],
            "lanes": [lane_def(QWEN80, QWEN30), lane_def(QWEN30, QWEN80)],
            "cross_lane_read_policy": cross_lane_policy(),
            "authority_boundary": authority_boundary(),
            "execution_boundary": execution_boundary(),
        }
    )


def build_mutation_input(lane_path: Path) -> dict:
    lane = binding_for(lane_path)
    return seal_value(
        {
            "schema": "hawking.ascension.paired_cognition_mutation_authority_input.v1",
            "lane_authority": lane,
            "lane_action_policies": [
                action_policy(QWEN80, QWEN30),
                action_policy(QWEN30, QWEN80),
            ],
            "protected_record_policy": {
                "evidence_receipts_immutable": True,
                "mission_records_immutable": True,
                "tournament_receipts_immutable": True,
                "all_roles_may_rewrite_evidence_receipts": False,
                "all_roles_may_rewrite_mission_records": False,
                "all_roles_may_rewrite_tournament_receipts": False,
                "all_roles_may_delete_protected_records": False,
                "cross_lane_private_record_release_allowed": False,
                "protected_tournament_receipt_namespace": "sealed://tournament/protected-receipts",
            },
            "final_selection_reservation": {
                "post_tg3_freeze_required": True,
                "protected_final_selection_required": True,
                "solo_manager_evaluation_required": True,
                "symmetric_orchestrator_evaluation_required": True,
                "manager_selection_authorized_by_this_contract": False,
                "tournament_selection_authorized_by_this_contract": False,
                "final_selection_authorized_by_this_contract": False,
                "post_tg3_freeze_namespace": "sealed://tournament/post-tg3-freeze",
                "protected_final_selection_namespace": "sealed://tournament/protected-final-selection",
            },
            "authority_boundary": authority_boundary(),
            "execution_boundary": execution_boundary(),
        }
    )


def release_identity(payload, provenance, redaction, verifier, publication) -> dict:
    payload_sha = sha256_json(payload)
    unsigned = {
        "schema": "hawking.ascension.paired_cognition_knowledge_plane_immutable_release_identity.v1",
        "release_id": "",
        "immutable": True,
        "revision": 0,
        "supersedes_release_id": None,
        "public_payload_sha256": payload_sha,
        "source_commitment_sha256": provenance["source_commitment_sha256"],
        "provenance_receipt_sha256": provenance["provenance_receipt_sha256"],
        "redaction_receipt_sha256": redaction["redaction_receipt_sha256"],
        "independent_verifier_receipt_sha256": verifier["verifier_receipt_sha256"],
        "publisher_receipt_sha256": publication["publisher_receipt_sha256"],
        "publisher_actor_id": publication["publisher_actor_id"],
        "verifier_actor_id": verifier["verifier_actor_id"],
    }
    preimage = {k: v for k, v in unsigned.items() if k not in ("release_id", "seal_sha256")}
    unsigned["release_id"] = f"kp-{sha256_json(preimage)}"
    return seal_value(unsigned)


def build_knowledge_input(lane_path: Path, mutation_path: Path) -> dict:
    lane = binding_for(lane_path)
    mutation = binding_for(mutation_path)
    payload = {
        "classification": "generic_mechanism_science",
        "topics": ["mechanism", "numerical_stability"],
        "generic_summary": "A fixed reduction order makes a numerical mechanism reproducible.",
        "generic_claim": "A documented reduction order preserves repeatable arithmetic behavior.",
        "public_evidence_abstract": (
            "Independent arithmetic checks agree on the stated mechanism under a fixed input fixture."
        ),
    }
    provenance = {
        "origin_lane_id": "qwen80-candidate-world",
        "claim_author_actor_id": "qwen80-primary-in-qwen80-lane",
        "source_commitment_sha256": "a" * 64,
        "provenance_receipt_sha256": "b" * 64,
        "source_scope": "sealed_lane_evidence_redacted_to_generic_claim",
        "private_evidence_disclosed": False,
        "cross_lane_private_read_used": False,
    }
    redaction = {
        "redactor_actor_id": "knowledge-plane-redaction-reviewer",
        "redaction_receipt_sha256": "c" * 64,
        "redaction_complete": True,
        "candidate_strategy_removed": True,
        "frontier_removed": True,
        "private_score_removed": True,
        "current_patch_removed": True,
        "current_hidden_task_removed": True,
        "private_namespace_identifiers_removed": True,
        "raw_experiment_history_removed": True,
        "private_receipt_content_removed": True,
        "redactor_independent_from_claim_author": True,
    }
    verifier = {
        "verifier_actor_id": "knowledge-plane-independent-verifier",
        "verifier_receipt_sha256": "d" * 64,
        "approval": "approved_generic_only",
        "verified_generic_scope": True,
        "reviewed_redaction": True,
        "verified_no_lane_private_leak": True,
        "verifier_independent_from_claim_author": True,
        "verifier_independent_from_publisher": True,
    }
    publication = {
        "publisher_actor_id": "knowledge-plane-release-steward",
        "publisher_receipt_sha256": "e" * 64,
        "publisher_independent_from_claim_author": True,
        "publisher_independent_from_verifier": True,
        "append_only_identity_registration_requested": True,
    }
    identity = release_identity(payload, provenance, redaction, verifier, publication)
    return seal_value(
        {
            "schema": "hawking.ascension.paired_cognition_knowledge_plane_release_input.v1",
            "lane_authority": lane,
            "mutation_authority": mutation,
            "knowledge_plane_policy": {
                "knowledge_plane_namespace": "sealed://knowledge-plane/generic-mechanism-science",
                "release_registry_namespace": "sealed://knowledge-plane/append-only-release-registry",
                "generic_mechanism_science_only": True,
                "append_only_release_identities": True,
                "release_identity_mutation_authorized": False,
                "external_publication_authorized": False,
                "lane_private_record_access_authorized": False,
                "candidate_world_activation_authorized": False,
            },
            "discovery_release_requests": [
                {
                    "release_identity": identity,
                    "public_payload": payload,
                    "provenance": provenance,
                    "redaction": redaction,
                    "independent_verification": verifier,
                    "publication": publication,
                }
            ],
            "authority_boundary": authority_boundary(),
            "execution_boundary": execution_boundary(),
        }
    )


def body_id(model_key: str) -> str:
    return f"{model_key}-one-resident-body"


def role_assignment(primary: str, role: str) -> dict:
    helper = QWEN80 if primary == QWEN30 else QWEN30
    primary_actor, helper_actor, _ = actor_ids(primary, helper)
    if role == "primary":
        actor = primary_actor
        physical = primary
    else:
        actor = helper_actor
        physical = helper
    session_namespace = f"sealed://{primary}-lane/sessions/logical/{actor}"
    return {
        "lane_id": f"{primary}-candidate-world",
        "role": role,
        "logical_actor_id": actor,
        "physical_model_key": physical,
        "physical_body_id": body_id(physical),
        "session_namespace": session_namespace,
        "queue_namespace": f"{session_namespace}/queue",
        "scheduling_weight": 1,
        "max_queued_requests": 8,
        "max_inflight_turns": 1,
        "private_lane_visibility_only": True,
    }


def telemetry(model_key: str) -> dict:
    return {
        "physical_model_key": model_key,
        "physical_body_id": body_id(model_key),
        "telemetry_namespace": f"sealed://paired-scheduler/{model_key}/fairness-telemetry",
        "scheduling_discipline": "round_robin_equal_weight",
        "max_inflight_turns_per_body": 1,
        "max_consecutive_dispatches_to_same_session": 1,
        "fairness_lag_bound_dispatches": 1,
        "required_counter_fields": list(TELEMETRY_FIELDS),
        "telemetry_contains_request_content": False,
        "telemetry_contains_generated_tokens": False,
        "telemetry_contains_lane_private_scores": False,
        "telemetry_contains_private_namespace_identifiers": False,
    }


def build_scheduler_input(lane_path: Path, mutation_path: Path, knowledge_path: Path) -> dict:
    return seal_value(
        {
            "schema": "hawking.ascension.paired_cognition_role_resource_scheduler_input.v1",
            "lane_authority": binding_for(lane_path),
            "mutation_authority": binding_for(mutation_path),
            "knowledge_authority": binding_for(knowledge_path),
            "body_resource_budgets": [
                {
                    "model_key": QWEN30,
                    "physical_body_id": body_id(QWEN30),
                    "endpoint": endpoint(QWEN30),
                    "resident_model_processes": 1,
                    "immutable_weight_copies": 1,
                    "max_logical_sessions": 2,
                    "max_inflight_turns": 1,
                },
                {
                    "model_key": QWEN80,
                    "physical_body_id": body_id(QWEN80),
                    "endpoint": endpoint(QWEN80),
                    "resident_model_processes": 1,
                    "immutable_weight_copies": 1,
                    "max_logical_sessions": 2,
                    "max_inflight_turns": 1,
                },
            ],
            "role_session_assignments": [
                role_assignment(QWEN30, "primary"),
                role_assignment(QWEN30, "helper"),
                role_assignment(QWEN80, "primary"),
                role_assignment(QWEN80, "helper"),
            ],
            "queue_fairness_telemetry_policies": [telemetry(QWEN30), telemetry(QWEN80)],
            "paired_development_activation_requested": False,
            "final_mode_reservation": {
                "qwen30_tg3_receipt": None,
                "qwen80_tg3_receipt": None,
                "post_tg3_freeze_seal_sha256": None,
                "solo_manager_evaluation_requested": False,
                "symmetric_orchestrator_evaluation_requested": False,
                "winner_selection_requested": False,
                "final_mode_authorized_by_this_contract": False,
                "winner_selection_authorized_by_this_contract": False,
            },
            "authority_boundary": authority_boundary(),
            "execution_boundary": execution_boundary(),
        }
    )


def load_operational_status() -> dict:
    for path in OPS_STATUS_FILE_CANDIDATES:
        if path.is_file():
            document = load_json(path)
            # Path binding must be the canonical Downloads path even when the
            # bytes are loaded from a worktree copy.
            return {
                "path": OPS_STATUS_CANONICAL_PATH,
                "document_sha256": sha256_json(document),
                "document": document,
                "source_file": str(path),
            }
    raise SystemExit("ASCENSION_OPERATIONAL_ASCENT_STATUS.json not found")


def build_tg10_input(
    lane_path: Path,
    mutation_path: Path,
    knowledge_path: Path,
    scheduler_path: Path,
) -> dict:
    ops = load_operational_status()
    source_file = ops.pop("source_file")
    print(f"operational ascent loaded from {source_file}", flush=True)
    doc = ops["document"]
    if doc.get("both_valid_tg10_receipts") is not False:
        raise SystemExit(
            "refusing to materialize: operational status claims both_valid_tg10_receipts "
            f"= {doc.get('both_valid_tg10_receipts')}; expected FALSE"
        )
    models = doc.get("evidence", {}).get("models", {})
    for model_key in (QWEN30, QWEN80):
        model = models.get(model_key) or {}
        if model.get("tg10_receipt_seal_sha256") is not None:
            raise SystemExit(
                f"refusing to fabricate: {model_key} already has tg10_receipt_seal_sha256"
            )
    return seal_value(
        {
            "schema": "hawking.ascension.paired_cognition_tg10_development_activation_input.v1",
            "lane_authority": binding_for(lane_path),
            "mutation_authority": binding_for(mutation_path),
            "knowledge_authority": binding_for(knowledge_path),
            "scheduler_authority": binding_for(scheduler_path),
            "operational_ascent_status": {
                "path": ops["path"],
                "document_sha256": ops["document_sha256"],
                "document": ops["document"],
            },
            "qwen30_tg10_receipt": None,
            "qwen80_tg10_receipt": None,
            "paired_development_activation_requested": False,
            "authority_boundary": authority_boundary(),
            "execution_boundary": execution_boundary(),
        }
    )


def summarize_sealed(path: Path) -> dict:
    doc = load_json(path)
    return {
        "path": str(path.resolve()),
        "schema": doc.get("schema"),
        "status": doc.get("status"),
        "prepared": doc.get("prepared"),
        "document_sha256": sha256_json(doc),
        "document_seal_sha256": doc.get("seal_sha256"),
        "paired_development_active": doc.get("paired_development_active"),
        "paired_candidate_worlds_active": doc.get("paired_candidate_worlds_active"),
        "both_exact_fresh_tg10_operational_receipts_present": doc.get(
            "both_exact_fresh_tg10_operational_receipts_present"
        ),
        "state_blockers": doc.get("state_blockers"),
        "paired_development_activation_authorized_by_this_contract": doc.get(
            "paired_development_activation_authorized_by_this_contract"
        ),
    }


def main() -> int:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lane_in = INPUT_DIR / "PAIRED_COGNITION_LANE_INPUT.json"
    lane_out = OUT_DIR / "PAIRED_COGNITION_LANE_AUTHORITY.json"
    mutation_in = INPUT_DIR / "PAIRED_COGNITION_MUTATION_INPUT.json"
    mutation_out = OUT_DIR / "PAIRED_COGNITION_MUTATION_AUTHORITY.json"
    knowledge_in = INPUT_DIR / "PAIRED_COGNITION_KNOWLEDGE_INPUT.json"
    knowledge_out = OUT_DIR / "PAIRED_COGNITION_KNOWLEDGE_AUTHORITY.json"
    scheduler_in = INPUT_DIR / "PAIRED_COGNITION_SCHEDULER_INPUT.json"
    scheduler_out = OUT_DIR / "PAIRED_COGNITION_ROLE_RESOURCE_SCHEDULER_AUTHORITY.json"
    tg10_in = INPUT_DIR / "PAIRED_COGNITION_TG10_ACTIVATION_INPUT.json"
    tg10_out = OUT_DIR / "PAIRED_COGNITION_TG10_DEVELOPMENT_ACTIVATION.json"

    # Fresh materialization: remove prior sealed outputs if present.
    for path in (lane_out, mutation_out, knowledge_out, scheduler_out, tg10_out):
        if path.exists():
            path.unlink()

    write_json(lane_in, build_lane_input())
    run_example(
        "ascension_paired_cognition_lane_authority_contract",
        lane_in,
        lane_out,
    )

    write_json(mutation_in, build_mutation_input(lane_out))
    run_example(
        "ascension_paired_cognition_mutation_authority_contract",
        mutation_in,
        mutation_out,
    )

    write_json(knowledge_in, build_knowledge_input(lane_out, mutation_out))
    run_example(
        "ascension_paired_cognition_knowledge_plane_release_gate",
        knowledge_in,
        knowledge_out,
    )

    write_json(scheduler_in, build_scheduler_input(lane_out, mutation_out, knowledge_out))
    run_example(
        "ascension_paired_cognition_role_resource_scheduler_authority",
        scheduler_in,
        scheduler_out,
    )

    write_json(
        tg10_in,
        build_tg10_input(lane_out, mutation_out, knowledge_out, scheduler_out),
    )
    run_example(
        "ascension_paired_cognition_tg10_development_activation_state_machine",
        tg10_in,
        tg10_out,
    )

    chain = {
        "lane": summarize_sealed(lane_out),
        "mutation": summarize_sealed(mutation_out),
        "knowledge": summarize_sealed(knowledge_out),
        "scheduler": summarize_sealed(scheduler_out),
        "tg10_development_activation": summarize_sealed(tg10_out),
    }
    proof = summarize_sealed(tg10_out)
    if proof["status"] == "MANAGER_ASCENT_TOURNAMENT_ACTIVE":
        raise SystemExit("FATAL: state machine emitted MANAGER_ASCENT_TOURNAMENT_ACTIVE")
    if proof.get("paired_development_active"):
        raise SystemExit("FATAL: paired_development_active is true")
    if proof.get("paired_development_activation_authorized_by_this_contract"):
        raise SystemExit("FATAL: activation authorized by this contract")
    if "REFUSED" not in (proof.get("status") or ""):
        raise SystemExit(f"expected REFUSED status, got {proof.get('status')}")

    missing_before_real_activation = [
        "QWEN30_TG10_OPERATIONAL_PASS exact sealed receipt (coherent HCLI, complete-token path, fallback_count=0, median BASE_TRUE_TPS >= 100)",
        "QWEN80_TG10_OPERATIONAL_PASS exact sealed receipt (same requirements)",
        "canonical operational ascent both_valid_tg10_receipts=true with matching evidence_fingerprint",
        "lane/scheduler TG10 readiness must rebind the exact receipt seals for both models",
        "a later protected controller (outside this CPU-only state machine) to perform actual activation; this contract only prepares or refuses",
        "TG3 qualifications, post-TG3 freeze, protected corpus/scorecard/selection contracts remain missing for any manager tournament",
    ]
    summary = {
        "authority_chain_materialized": True,
        "identity_convention": {
            "document_sha256": "json_sha of full sealed document including seal_sha256",
            "document_seal_sha256": "json_sha of document with seal_sha256 removed (equals seal_sha256 field)",
            "raw_file_bytes_sha256": "not used by these five contracts",
        },
        "chain": chain,
        "refusal_proof": {
            "output_path": proof["path"],
            "status": proof["status"],
            "prepared": proof["prepared"],
            "both_exact_fresh_tg10_operational_receipts_present": proof[
                "both_exact_fresh_tg10_operational_receipts_present"
            ],
            "paired_development_active": proof["paired_development_active"],
            "paired_development_activation_authorized_by_this_contract": proof[
                "paired_development_activation_authorized_by_this_contract"
            ],
            "state_blockers": proof["state_blockers"],
            "manager_ascent_tournament_active": False,
            "named_unmet_condition": [
                b
                for b in (proof["state_blockers"] or [])
                if "tg10" in b.lower() or "receipt" in b.lower() or "both" in b.lower()
            ],
        },
        "inputs_still_missing_before_real_activation": missing_before_real_activation,
        "claim_boundary": [
            "No TG10 pass was fabricated for either candidate.",
            "No document asserts a measured BASE_TRUE_TPS for either candidate.",
            "No server, lease, Metal, model load, or tournament state mutation occurred.",
            "MANAGER_ASCENT_TOURNAMENT_ACTIVE remains unreachable while both operational passes are false.",
        ],
    }
    summary_path = OUT_DIR / "PAIRED_AUTHORITY_CHAIN_MATERIALIZATION_PROOF.json"
    write_json(summary_path, seal_value(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nWrote proof summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
