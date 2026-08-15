"""Enforcement tests for lab.execution_sandbox (bible §21)."""
from __future__ import annotations

import pytest

from lab.execution_sandbox import (
    MODEL_ALLOW_LIST,
    MODEL_DENY_LIST,
    EffectReceipt,
    ExecutionSandboxPolicy,
    PathClass,
    SandboxAction,
    SandboxPolicyError,
    SandboxPrincipal,
    assert_model_cannot,
    classify_path,
    default_model_policy,
)


WORKTREE = "/tmp/hcli-sandbox-wt-demo"


@pytest.fixture
def policy() -> ExecutionSandboxPolicy:
    return default_model_policy(owned_worktree=WORKTREE, sandbox_root="/tmp")


# ---------------------------------------------------------------------------
# Deny-list: every §21 forbidden action is actually rejected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("action", sorted(MODEL_DENY_LIST, key=lambda a: a.value))
def test_deny_list_actions_rejected_for_sandbox_model(action: SandboxAction) -> None:
    assert_model_cannot(action)
    policy = ExecutionSandboxPolicy()
    decision = policy.authorize(SandboxPrincipal.SANDBOX_MODEL, action)
    assert decision.allowed is False
    assert "may not" in decision.reason or "deny" in decision.reason.lower()
    with pytest.raises(SandboxPolicyError) as excinfo:
        policy.require(SandboxPrincipal.SANDBOX_MODEL, action)
    assert excinfo.value.decision.action == action
    assert excinfo.value.decision.allowed is False


def test_sign_own_results_hard_reject(policy: ExecutionSandboxPolicy) -> None:
    with pytest.raises(SandboxPolicyError, match="sign_own_results"):
        policy.require(
            SandboxPrincipal.SANDBOX_MODEL,
            SandboxAction.SIGN_OWN_RESULTS,
            target=f"{WORKTREE}/results.json",
        )


def test_merge_self_hard_reject(policy: ExecutionSandboxPolicy) -> None:
    with pytest.raises(SandboxPolicyError, match="merge_self"):
        policy.require(SandboxPrincipal.SANDBOX_MODEL, SandboxAction.MERGE_SELF)


def test_delete_stable_artifacts_hard_reject(policy: ExecutionSandboxPolicy) -> None:
    with pytest.raises(SandboxPolicyError, match="delete_stable"):
        policy.require(
            SandboxPrincipal.SANDBOX_MODEL,
            SandboxAction.DELETE_STABLE_ARTIFACTS,
            target="/repo/stable_artifacts/promoted.bin",
        )


def test_modify_held_out_prompts_hard_reject(policy: ExecutionSandboxPolicy) -> None:
    with pytest.raises(SandboxPolicyError, match="held_out"):
        policy.require(
            SandboxPrincipal.SANDBOX_MODEL,
            SandboxAction.MODIFY_HELD_OUT_PROMPTS,
            target="/protected/held_out/prompts.jsonl",
        )


def test_modify_promotion_thresholds_hard_reject(policy: ExecutionSandboxPolicy) -> None:
    with pytest.raises(SandboxPolicyError, match="promotion_threshold"):
        policy.require(
            SandboxPrincipal.SANDBOX_MODEL,
            SandboxAction.MODIFY_PROMOTION_THRESHOLDS,
        )


def test_modify_protected_oracle_hard_reject(policy: ExecutionSandboxPolicy) -> None:
    with pytest.raises(SandboxPolicyError, match="protected_oracle"):
        policy.require(
            SandboxPrincipal.SANDBOX_MODEL,
            SandboxAction.MODIFY_PROTECTED_ORACLE,
        )


def test_read_credentials_hard_reject(policy: ExecutionSandboxPolicy) -> None:
    with pytest.raises(SandboxPolicyError, match="credentials"):
        policy.require(
            SandboxPrincipal.SANDBOX_MODEL,
            SandboxAction.READ_OR_PRINT_CREDENTIALS,
            target="/Users/x/.ssh/id_ed25519",
        )


# ---------------------------------------------------------------------------
# Path classification maps protected paths onto deny semantics
# ---------------------------------------------------------------------------


def test_classify_credential_paths() -> None:
    assert classify_path("/home/u/.ssh/id_ed25519") is PathClass.CREDENTIAL
    assert classify_path("/app/.env") is PathClass.CREDENTIAL
    assert classify_path("/secrets/api_token") is PathClass.CREDENTIAL


def test_classify_held_out_and_oracle() -> None:
    assert (
        classify_path("/workspace/benchmarks/held_out/math.jsonl")
        is PathClass.HELD_OUT_PROMPTS
    )
    assert (
        classify_path("/workspace/protected_oracle/numeric_parity/v2.1.json")
        is PathClass.PROTECTED_ORACLE
    )
    assert (
        classify_path("/workspace/config/promotion_thresholds.json")
        is PathClass.PROMOTION_THRESHOLDS
    )


def test_edit_owned_worktree_allowed(policy: ExecutionSandboxPolicy) -> None:
    d = policy.require(
        SandboxPrincipal.SANDBOX_MODEL,
        SandboxAction.EDIT_OWNED_WORKTREE,
        target=f"{WORKTREE}/lab/execution_sandbox.py",
    )
    assert d.allowed is True
    assert d.path_class is PathClass.OWNED_WORKTREE


def test_edit_outside_owned_worktree_rejected(policy: ExecutionSandboxPolicy) -> None:
    with pytest.raises(SandboxPolicyError, match="outside sandbox root|owned worktree"):
        policy.require(
            SandboxPrincipal.SANDBOX_MODEL,
            SandboxAction.EDIT_OWNED_WORKTREE,
            target="/other/repo/file.py",
        )


def test_edit_held_out_path_rejected_even_if_under_worktree_name() -> None:
    # Worktree that contains a held_out segment still classifies as held-out.
    wt = "/tmp/wt"
    policy = default_model_policy(owned_worktree=wt)
    with pytest.raises(SandboxPolicyError, match="held_out"):
        policy.require(
            SandboxPrincipal.SANDBOX_MODEL,
            SandboxAction.EDIT_OWNED_WORKTREE,
            target=f"{wt}/datasets/held_out/prompts.jsonl",
        )


def test_read_source_blocks_credentials(policy: ExecutionSandboxPolicy) -> None:
    with pytest.raises(SandboxPolicyError, match="credentials"):
        policy.require(
            SandboxPrincipal.SANDBOX_MODEL,
            SandboxAction.READ_SOURCE,
            target=f"{WORKTREE}/.env",
        )


def test_read_source_blocks_held_out(policy: ExecutionSandboxPolicy) -> None:
    with pytest.raises(SandboxPolicyError, match="held-out"):
        policy.require(
            SandboxPrincipal.SANDBOX_MODEL,
            SandboxAction.READ_SOURCE,
            target="/repo/hidden_eval/suite.json",
        )


# ---------------------------------------------------------------------------
# Allow-list actions succeed for models
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "action",
    [
        SandboxAction.COMPILE,
        SandboxAction.INSPECT_PUBLIC_PROFILE,
        SandboxAction.REQUEST_PROTECTED_BENCHMARK,
        SandboxAction.REQUEST_ROLLBACK,
        SandboxAction.EMIT_EFFECT_RECEIPT,
    ],
)
def test_allow_list_actions_permitted(
    policy: ExecutionSandboxPolicy, action: SandboxAction
) -> None:
    d = policy.require(SandboxPrincipal.SANDBOX_MODEL, action)
    assert d.allowed is True


def test_allow_list_and_deny_list_partition() -> None:
    # Every enum member is either allow or deny (closed vocabulary).
    all_actions = set(SandboxAction)
    assert MODEL_ALLOW_LIST | MODEL_DENY_LIST == all_actions
    assert MODEL_ALLOW_LIST.isdisjoint(MODEL_DENY_LIST)


def test_run_allowed_tests_selector_enforced() -> None:
    policy = ExecutionSandboxPolicy(allowed_test_selectors=frozenset({"unit", "sandbox"}))
    policy.require(
        SandboxPrincipal.SANDBOX_MODEL,
        SandboxAction.RUN_ALLOWED_TESTS,
        context={"test_selector": "unit"},
    )
    with pytest.raises(SandboxPolicyError, match="test selector"):
        policy.require(
            SandboxPrincipal.SANDBOX_MODEL,
            SandboxAction.RUN_ALLOWED_TESTS,
            context={"test_selector": "live_gpu"},
        )


def test_approved_download_enforced() -> None:
    policy = ExecutionSandboxPolicy(
        approved_download_ids=frozenset({"corpus-v1"}),
    )
    policy.require(
        SandboxPrincipal.SANDBOX_MODEL,
        SandboxAction.REQUEST_APPROVED_DOWNLOAD,
        context={"download_id": "corpus-v1"},
    )
    with pytest.raises(SandboxPolicyError, match="download_id"):
        policy.require(
            SandboxPrincipal.SANDBOX_MODEL,
            SandboxAction.REQUEST_APPROVED_DOWNLOAD,
            context={"download_id": "mystery-weights"},
        )


# ---------------------------------------------------------------------------
# Controllers are not bound by the model deny-list
# ---------------------------------------------------------------------------


def test_protected_controller_may_sign_and_delete() -> None:
    policy = ExecutionSandboxPolicy()
    for action in (
        SandboxAction.SIGN_OWN_RESULTS,  # controller signs results (not "own" in model sense)
        SandboxAction.DELETE_STABLE_ARTIFACTS,
        SandboxAction.MODIFY_PROMOTION_THRESHOLDS,
    ):
        d = policy.require(SandboxPrincipal.PROTECTED_CONTROLLER, action)
        assert d.allowed is True


def test_human_operator_may_modify_oracle() -> None:
    policy = ExecutionSandboxPolicy()
    d = policy.require(
        SandboxPrincipal.HUMAN_OPERATOR,
        SandboxAction.MODIFY_PROTECTED_ORACLE,
        target="/protected/oracle",
    )
    assert d.allowed is True


# ---------------------------------------------------------------------------
# Effect receipts record denials without raising
# ---------------------------------------------------------------------------


def test_effect_receipt_records_denial(policy: ExecutionSandboxPolicy) -> None:
    receipt = policy.effect_receipt(
        SandboxPrincipal.SANDBOX_MODEL,
        SandboxAction.MERGE_SELF,
    )
    assert isinstance(receipt, EffectReceipt)
    assert receipt.allowed is False
    payload = receipt.to_dict()
    assert payload["schema"].startswith("hawking.lab.sandbox_effect_receipt")
    assert payload["action"] == "merge_self"
