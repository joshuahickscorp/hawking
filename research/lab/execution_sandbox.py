#!/usr/bin/env python3.12
"""Execution sandbox permission boundary (Ascension Bible §21).

Standing subsystem that generalizes the session's grok-orchestration and
frankenstein promotion-gate discipline: models may operate inside an owned
worktree with bounded writes; they may not touch protected oracles, held-out
prompts, promotion thresholds, merge/sign their own results, delete stable
artifacts, or read credentials.

This module is the *enforceable policy structure* — not documentation.
Callers invoke ``ExecutionSandboxPolicy.require`` / ``authorize`` before
side-effecting actions. Denied actions raise ``SandboxPolicyError``.

OS-level confinement (Seatbelt / hide-kernel security_sandbox) is orthogonal
and may wrap the same action set later; this policy is the in-process gate
that every lab operator and orchestrator can call today without OS privileges.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


SCHEMA = "hawking.lab.execution_sandbox_policy.v1"
EFFECT_RECEIPT_SCHEMA = "hawking.lab.sandbox_effect_receipt.v1"


class SandboxPrincipal(str, Enum):
    """Who is requesting the action."""

    SANDBOX_MODEL = "sandbox_model"
    PROTECTED_CONTROLLER = "protected_controller"
    HUMAN_OPERATOR = "human_operator"


class SandboxAction(str, Enum):
    """Bible §21 allow-list and deny-list as a closed action vocabulary."""

    # --- Models MAY (§21 allow-list + sandbox capabilities) ---
    READ_SOURCE = "read_source"
    INSPECT_PUBLIC_PROFILE = "inspect_public_profile"
    EDIT_OWNED_WORKTREE = "edit_owned_worktree"
    COMPILE = "compile"
    RUN_ALLOWED_TESTS = "run_allowed_tests"
    REQUEST_PROTECTED_BENCHMARK = "request_protected_benchmark"
    REQUEST_APPROVED_DOWNLOAD = "request_approved_download"
    GENERATE_ARTIFACT = "generate_artifact"
    EMIT_EFFECT_RECEIPT = "emit_effect_receipt"
    REQUEST_ROLLBACK = "request_rollback"

    # --- Models MAY NOT (§21 deny-list) ---
    MODIFY_PROTECTED_ORACLE = "modify_protected_oracle"
    MODIFY_HELD_OUT_PROMPTS = "modify_held_out_prompts"
    MODIFY_PROMOTION_THRESHOLDS = "modify_promotion_thresholds"
    MERGE_SELF = "merge_self"
    SIGN_OWN_RESULTS = "sign_own_results"
    DELETE_STABLE_ARTIFACTS = "delete_stable_artifacts"
    READ_OR_PRINT_CREDENTIALS = "read_or_print_credentials"


# Explicit allow-list for sandbox models. Unknown actions fail closed.
MODEL_ALLOW_LIST: frozenset[SandboxAction] = frozenset(
    {
        SandboxAction.READ_SOURCE,
        SandboxAction.INSPECT_PUBLIC_PROFILE,
        SandboxAction.EDIT_OWNED_WORKTREE,
        SandboxAction.COMPILE,
        SandboxAction.RUN_ALLOWED_TESTS,
        SandboxAction.REQUEST_PROTECTED_BENCHMARK,
        SandboxAction.REQUEST_APPROVED_DOWNLOAD,
        SandboxAction.GENERATE_ARTIFACT,
        SandboxAction.EMIT_EFFECT_RECEIPT,
        SandboxAction.REQUEST_ROLLBACK,
    }
)

MODEL_DENY_LIST: frozenset[SandboxAction] = frozenset(
    {
        SandboxAction.MODIFY_PROTECTED_ORACLE,
        SandboxAction.MODIFY_HELD_OUT_PROMPTS,
        SandboxAction.MODIFY_PROMOTION_THRESHOLDS,
        SandboxAction.MERGE_SELF,
        SandboxAction.SIGN_OWN_RESULTS,
        SandboxAction.DELETE_STABLE_ARTIFACTS,
        SandboxAction.READ_OR_PRINT_CREDENTIALS,
    }
)

# Path-class markers used when classifying filesystem targets.
_CREDENTIAL_NAME_MARKERS: tuple[str, ...] = (
    ".env",
    ".pem",
    ".key",
    "credentials",
    "secrets",
    "id_rsa",
    "id_ed25519",
    ".netrc",
    "token",
    "api_key",
)

_CREDENTIAL_DIR_MARKERS: tuple[str, ...] = (
    ".ssh",
    ".aws",
    ".gnupg",
    "secrets",
)

_HELD_OUT_MARKERS: tuple[str, ...] = (
    "held_out",
    "held-out",
    "hidden_eval",
    "hidden-eval",
    "hidden_prompts",
    "heldout",
)

_ORACLE_MARKERS: tuple[str, ...] = (
    "protected_oracle",
    "oracle_protected",
    "numeric_parity",
    "numeric-parity",
)

_THRESHOLD_MARKERS: tuple[str, ...] = (
    "promotion_threshold",
    "promotion_thresholds",
    "frozen_targets",
    "promotion_gate_targets",
)

_STABLE_ARTIFACT_MARKERS: tuple[str, ...] = (
    "stable_artifact",
    "stable_artifacts",
    "promoted_artifact",
    "sealed_stable",
)


class SandboxPolicyError(PermissionError):
    """Raised when a sandbox principal attempts a denied action."""

    def __init__(self, decision: "PolicyDecision") -> None:
        self.decision = decision
        super().__init__(decision.reason)


class PathClass(str, Enum):
    ORDINARY = "ordinary"
    OWNED_WORKTREE = "owned_worktree"
    PROTECTED_ORACLE = "protected_oracle"
    HELD_OUT_PROMPTS = "held_out_prompts"
    PROMOTION_THRESHOLDS = "promotion_thresholds"
    STABLE_ARTIFACT = "stable_artifact"
    CREDENTIAL = "credential"
    OUTSIDE_SANDBOX = "outside_sandbox"


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    principal: SandboxPrincipal
    action: SandboxAction
    reason: str
    path_class: PathClass | None = None
    target: str | None = None
    schema: str = SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "allowed": self.allowed,
            "principal": self.principal.value,
            "action": self.action.value,
            "reason": self.reason,
            "path_class": None if self.path_class is None else self.path_class.value,
            "target": self.target,
        }


@dataclass(frozen=True)
class EffectReceipt:
    """Record of an authorized (or denied) sandbox effect attempt."""

    action: SandboxAction
    principal: SandboxPrincipal
    allowed: bool
    target: str | None
    detail: Mapping[str, Any] = field(default_factory=dict)
    schema: str = EFFECT_RECEIPT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "action": self.action.value,
            "principal": self.principal.value,
            "allowed": self.allowed,
            "target": self.target,
            "detail": dict(self.detail),
        }


def _normalize_action(action: SandboxAction | str) -> SandboxAction:
    if isinstance(action, SandboxAction):
        return action
    try:
        return SandboxAction(str(action))
    except ValueError as exc:
        raise ValueError(f"unknown sandbox action {action!r}") from exc


def _path_parts(path: str | Path) -> tuple[str, ...]:
    # PurePosixPath keeps classification stable across platforms for policy tests.
    p = PurePosixPath(str(path).replace("\\", "/"))
    return tuple(part for part in p.parts if part not in {"/", "."})


def classify_path(
    path: str | Path | None,
    *,
    owned_worktree_roots: Sequence[str | Path] = (),
    sandbox_root: str | Path | None = None,
) -> PathClass:
    """Classify a filesystem target for policy checks."""

    if path is None:
        return PathClass.ORDINARY

    text = str(path).replace("\\", "/")
    lower = text.lower()
    name = PurePosixPath(text).name.lower()
    parts_lower = tuple(p.lower() for p in _path_parts(text))

    # Credentials first — never readable by sandbox models.
    if any(marker in name for marker in _CREDENTIAL_NAME_MARKERS):
        return PathClass.CREDENTIAL
    if any(marker in parts_lower for marker in _CREDENTIAL_DIR_MARKERS):
        return PathClass.CREDENTIAL
    if "/.ssh/" in f"/{lower}/" or "/.aws/" in f"/{lower}/":
        return PathClass.CREDENTIAL

    if any(m in lower for m in _HELD_OUT_MARKERS) or any(
        m in parts_lower for m in _HELD_OUT_MARKERS
    ):
        return PathClass.HELD_OUT_PROMPTS

    if any(m in lower for m in _ORACLE_MARKERS) or any(
        m in parts_lower for m in _ORACLE_MARKERS
    ):
        return PathClass.PROTECTED_ORACLE

    if any(m in lower for m in _THRESHOLD_MARKERS) or any(
        m in parts_lower for m in _THRESHOLD_MARKERS
    ):
        return PathClass.PROMOTION_THRESHOLDS

    if any(m in lower for m in _STABLE_ARTIFACT_MARKERS) or any(
        m in parts_lower for m in _STABLE_ARTIFACT_MARKERS
    ):
        return PathClass.STABLE_ARTIFACT

    # Owned worktree membership.
    resolved_text = text
    for root in owned_worktree_roots:
        root_text = str(root).replace("\\", "/").rstrip("/")
        if resolved_text == root_text or resolved_text.startswith(root_text + "/"):
            return PathClass.OWNED_WORKTREE

    if sandbox_root is not None:
        root_text = str(sandbox_root).replace("\\", "/").rstrip("/")
        if not (
            resolved_text == root_text or resolved_text.startswith(root_text + "/")
        ):
            # Absolute paths outside sandbox_root are outside; relative paths
            # without an owned-worktree match are ordinary (caller-relative).
            if PurePosixPath(text).is_absolute():
                return PathClass.OUTSIDE_SANDBOX

    return PathClass.ORDINARY


@dataclass
class ExecutionSandboxPolicy:
    """Deny-by-default policy for sandbox-model principals.

    Protected controller and human operator principals are not subject to the
    model allow/deny lists (they own Numeric Parity, promotion, signing, and
    deletion authority — bible §22). Path classification still labels targets
    for audit receipts.
    """

    owned_worktree_roots: tuple[str, ...] = ()
    sandbox_root: str | None = None
    allowed_test_selectors: frozenset[str] = frozenset()
    approved_download_ids: frozenset[str] = frozenset()

    def with_owned_worktree(self, root: str | Path) -> "ExecutionSandboxPolicy":
        roots = self.owned_worktree_roots + (str(root),)
        return ExecutionSandboxPolicy(
            owned_worktree_roots=roots,
            sandbox_root=self.sandbox_root,
            allowed_test_selectors=self.allowed_test_selectors,
            approved_download_ids=self.approved_download_ids,
        )

    def authorize(
        self,
        principal: SandboxPrincipal | str,
        action: SandboxAction | str,
        *,
        target: str | Path | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> PolicyDecision:
        principal_e = (
            principal
            if isinstance(principal, SandboxPrincipal)
            else SandboxPrincipal(str(principal))
        )
        try:
            action_e = _normalize_action(action)
        except ValueError:
            return PolicyDecision(
                allowed=False,
                principal=principal_e,
                # Stable placeholder so callers can still serialize decisions;
                # the reason carries the original unknown name.
                action=SandboxAction.READ_SOURCE,
                reason=f"unknown sandbox action {action!r}; deny-by-default",
                target=None if target is None else str(target),
            )

        path_class = classify_path(
            target,
            owned_worktree_roots=self.owned_worktree_roots,
            sandbox_root=self.sandbox_root,
        )
        target_s = None if target is None else str(target)
        ctx = dict(context or {})

        # Controllers / humans are the authority; still audit path class.
        if principal_e in {
            SandboxPrincipal.PROTECTED_CONTROLLER,
            SandboxPrincipal.HUMAN_OPERATOR,
        }:
            return PolicyDecision(
                allowed=True,
                principal=principal_e,
                action=action_e,
                reason=f"{principal_e.value} is verification authority; action permitted",
                path_class=path_class,
                target=target_s,
            )

        # --- Sandbox model path ---
        if action_e in MODEL_DENY_LIST:
            return PolicyDecision(
                allowed=False,
                principal=principal_e,
                action=action_e,
                reason=(
                    f"sandbox model may not perform {action_e.value} "
                    f"(bible §21 deny-list)"
                ),
                path_class=path_class,
                target=target_s,
            )

        if action_e not in MODEL_ALLOW_LIST:
            return PolicyDecision(
                allowed=False,
                principal=principal_e,
                action=action_e,
                reason=f"action {action_e.value} not on sandbox model allow-list; deny-by-default",
                path_class=path_class,
                target=target_s,
            )

        # Path-sensitive refinements of allowed actions.
        if action_e == SandboxAction.EDIT_OWNED_WORKTREE:
            if path_class == PathClass.CREDENTIAL:
                return PolicyDecision(
                    allowed=False,
                    principal=principal_e,
                    action=action_e,
                    reason="sandbox model may not edit credential paths",
                    path_class=path_class,
                    target=target_s,
                )
            if path_class in {
                PathClass.PROTECTED_ORACLE,
                PathClass.HELD_OUT_PROMPTS,
                PathClass.PROMOTION_THRESHOLDS,
                PathClass.STABLE_ARTIFACT,
            }:
                return PolicyDecision(
                    allowed=False,
                    principal=principal_e,
                    action=action_e,
                    reason=(
                        f"edit denied: target classified as {path_class.value} "
                        f"(maps to §21 deny-list)"
                    ),
                    path_class=path_class,
                    target=target_s,
                )
            if path_class == PathClass.OUTSIDE_SANDBOX:
                return PolicyDecision(
                    allowed=False,
                    principal=principal_e,
                    action=action_e,
                    reason="edit denied: target is outside sandbox root",
                    path_class=path_class,
                    target=target_s,
                )
            if path_class != PathClass.OWNED_WORKTREE:
                return PolicyDecision(
                    allowed=False,
                    principal=principal_e,
                    action=action_e,
                    reason=(
                        "edit denied: target is not under an owned worktree root; "
                        "bounded writes only"
                    ),
                    path_class=path_class,
                    target=target_s,
                )

        if action_e == SandboxAction.READ_SOURCE:
            if path_class == PathClass.CREDENTIAL:
                return PolicyDecision(
                    allowed=False,
                    principal=principal_e,
                    action=action_e,
                    reason="sandbox model may not read or print credentials",
                    path_class=path_class,
                    target=target_s,
                )
            # Reading held-out *prompts content* is a different action class;
            # path markers for held-out still block generic read when the path
            # is clearly held-out material (prevents prompt leakage).
            if path_class == PathClass.HELD_OUT_PROMPTS:
                return PolicyDecision(
                    allowed=False,
                    principal=principal_e,
                    action=action_e,
                    reason="sandbox model may not read held-out prompts",
                    path_class=path_class,
                    target=target_s,
                )

        if action_e == SandboxAction.GENERATE_ARTIFACT:
            if path_class in {
                PathClass.PROTECTED_ORACLE,
                PathClass.HELD_OUT_PROMPTS,
                PathClass.PROMOTION_THRESHOLDS,
                PathClass.STABLE_ARTIFACT,
                PathClass.CREDENTIAL,
            }:
                return PolicyDecision(
                    allowed=False,
                    principal=principal_e,
                    action=action_e,
                    reason=f"artifact write denied on {path_class.value}",
                    path_class=path_class,
                    target=target_s,
                )
            if target_s is not None and path_class not in {
                PathClass.OWNED_WORKTREE,
                PathClass.ORDINARY,
            }:
                return PolicyDecision(
                    allowed=False,
                    principal=principal_e,
                    action=action_e,
                    reason=f"artifact generation denied for path class {path_class.value}",
                    path_class=path_class,
                    target=target_s,
                )

        if action_e == SandboxAction.RUN_ALLOWED_TESTS:
            selector = str(ctx.get("test_selector") or "")
            if self.allowed_test_selectors and selector not in self.allowed_test_selectors:
                return PolicyDecision(
                    allowed=False,
                    principal=principal_e,
                    action=action_e,
                    reason=(
                        f"test selector {selector!r} not in allowed_test_selectors"
                    ),
                    path_class=path_class,
                    target=target_s,
                )

        if action_e == SandboxAction.REQUEST_APPROVED_DOWNLOAD:
            download_id = str(ctx.get("download_id") or "")
            if self.approved_download_ids and download_id not in self.approved_download_ids:
                return PolicyDecision(
                    allowed=False,
                    principal=principal_e,
                    action=action_e,
                    reason=f"download_id {download_id!r} is not pre-approved",
                    path_class=path_class,
                    target=target_s,
                )

        # REQUEST_PROTECTED_BENCHMARK / REQUEST_ROLLBACK / COMPILE /
        # INSPECT_PUBLIC_PROFILE / EMIT_EFFECT_RECEIPT: allow-list is sufficient.
        return PolicyDecision(
            allowed=True,
            principal=principal_e,
            action=action_e,
            reason=f"sandbox model may perform {action_e.value}",
            path_class=path_class,
            target=target_s,
        )

    def require(
        self,
        principal: SandboxPrincipal | str,
        action: SandboxAction | str,
        *,
        target: str | Path | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> PolicyDecision:
        """Authorize or raise ``SandboxPolicyError``."""

        decision = self.authorize(
            principal, action, target=target, context=context
        )
        if not decision.allowed:
            raise SandboxPolicyError(decision)
        return decision

    def effect_receipt(
        self,
        principal: SandboxPrincipal | str,
        action: SandboxAction | str,
        *,
        target: str | Path | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> EffectReceipt:
        """Authorize attempt and return an effect receipt (allowed or denied)."""

        decision = self.authorize(
            principal, action, target=target, context=context
        )
        return EffectReceipt(
            action=decision.action,
            principal=decision.principal,
            allowed=decision.allowed,
            target=decision.target,
            detail={
                "reason": decision.reason,
                "path_class": (
                    None
                    if decision.path_class is None
                    else decision.path_class.value
                ),
                "context": dict(context or {}),
            },
        )


def default_model_policy(
    *,
    owned_worktree: str | Path | None = None,
    sandbox_root: str | Path | None = None,
) -> ExecutionSandboxPolicy:
    """Factory for a typical single-worktree sandbox model session."""

    policy = ExecutionSandboxPolicy(
        sandbox_root=None if sandbox_root is None else str(sandbox_root),
    )
    if owned_worktree is not None:
        policy = policy.with_owned_worktree(owned_worktree)
    return policy


def assert_model_cannot(action: SandboxAction | str) -> None:
    """Unit-test helper: prove a bare model principal is denied ``action``."""

    policy = ExecutionSandboxPolicy()
    decision = policy.authorize(SandboxPrincipal.SANDBOX_MODEL, action)
    if decision.allowed:
        raise AssertionError(
            f"expected deny for sandbox_model action={action!r}, got allow: {decision.reason}"
        )
